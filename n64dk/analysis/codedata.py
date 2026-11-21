"""Code or data?  An evidence-scoring classifier for one address interval.

Discovery has to decide, for bytes nobody catalogued (and for catalogued rows it doubts), whether
they are instructions.  Words alone cannot settle it: float arrays decode as ``lui`` runs, jump
tables as ``lb`` from ``$zero``, display lists as ``swc1``/``ldc1``, and compressed bytes contain
anything, including a convincing prologue.  So :func:`classify` collects evidence for both sides,
each item with a weight and a sentence, and answers ``code``, ``data`` or ``ambiguous``.

Code evidence
    * references: ``jal`` target (3, strong), ``j`` tail target (2), target of a resolved switch
      table (3, strong), entry of a function-pointer table or a data word pointing at the start
      (1.5), address taken by a ``lui``/``addiu`` pair (1);
    * control flow from the start closes: it reaches ``jr $ra`` or a tail jump, never runs off the
      end, never branches out, never executes an undefined word (2);
    * stack discipline: ``addiu sp,sp,-N`` released by ``addiu sp,sp,+N`` (1.5);
    * saved-register consistency: every callee-saved register written is saved and restored (1).

Data evidence
    * a resolved jump table lies in the interval (4, strong); loads of data widths into it (2:
      ``lh``/``lb``/``lwc1``/``ldc1``...), word loads (1), stores into it (1.5);
    * undefined opcodes reached from the start (3, strong); writes to ``$zero`` (1 or 2); a branch
      into a delay slot (1.5); most of the interval not reached from its start (3, strong);
    * shapes: NUL-terminated ASCII/Shift-JIS strings over most of the bytes (3, strong), a display
      list ending in ``G_ENDDL`` (3, strong), plausible floats sharing exponents (2, or 1 without
      a shared exponent), an array of aligned addresses into the image (2), a fixed-stride record
      layout (1).

Decision: strong evidence on both sides is ``ambiguous``.  ``code`` needs at least one reference
(call, tail jump, table or pointer: shapes and control flow alone never make code), a score of at
least 3, no strong data evidence and a score at least twice the data score plus one; ``data`` is
the mirror image without the reference requirement.  Everything else is ``ambiguous``, and
nothing here promotes an ambiguous interval: callers report it.

When the interval's control flow closes before its end, the unreached remainder is described
(``Verdict.extent``, notes) but only counts against code when it dominates the interval.
"""
from __future__ import annotations

import bisect
import struct
from collections import Counter
from dataclasses import dataclass, field

from .. import mips
from . import flow as flowmod
from .datamap import string_at

CODE, DATA, AMBIGUOUS = "code", "data", "ambiguous"
STRONG = 3.0
MAX_BYTES = 0x4000
_DATA_LOADS = {"lb", "lbu", "lh", "lhu", "lwc1", "ldc1", "ld", "lwl", "lwr", "ldl", "ldr"}
_KSEG0 = (0x80000000, 0x80800000)


def hx(value: int) -> str:
    return f"0x{value & 0xFFFFFFFF:08X}"


@dataclass(frozen=True)
class Evidence:
    side: str               # "code" or "data"
    weight: float
    what: str
    reference: bool = False   # a reference from elsewhere (as opposed to the bytes' own shape)


@dataclass
class Verdict:
    lo: int
    hi: int
    verdict: str
    code_score: float
    data_score: float
    code: list[Evidence]
    data: list[Evidence]
    notes: list[str] = field(default_factory=list)
    extent: tuple[int, int] | None = None     # the part reached from the start when control flow closes

    def as_dict(self) -> dict:
        return {"lo": hx(self.lo), "hi": hx(self.hi), "verdict": self.verdict,
                "code_score": round(self.code_score, 2), "data_score": round(self.data_score, 2),
                "code": [f"{e.weight:g} {e.what}" for e in self.code],
                "data": [f"{e.weight:g} {e.what}" for e in self.data], "notes": self.notes,
                "extent": [hx(self.extent[0]), hx(self.extent[1])] if self.extent else None}


class _Index:
    """Sorted addresses for range queries."""

    def __init__(self, mapping: dict):
        self.mapping = mapping
        self.keys = sorted(mapping)

    def within(self, lo: int, hi: int) -> list[int]:
        return self.keys[bisect.bisect_left(self.keys, lo):bisect.bisect_left(self.keys, hi)]


@dataclass
class Context:
    """References from outside the interval.  Build it from a catalogue with :meth:`from_image`
    or fill the fields by hand (tests)."""
    call_sites: dict[int, list[int]] = field(default_factory=dict)     # jal target -> sites
    jump_sites: dict[int, list[int]] = field(default_factory=dict)     # j target -> sites
    switch_targets: set[int] = field(default_factory=set)
    switch_tables: dict[int, list[int]] = field(default_factory=dict)  # jr site -> targets (for flow)
    pointer_targets: dict[int, int] = field(default_factory=dict)      # address -> data words/table entries
    address_refs: dict[int, int] = field(default_factory=dict)         # address -> lui/addiu pairs
    loads: dict[int, set[str]] = field(default_factory=dict)           # address -> load opcodes
    stores: dict[int, set[str]] = field(default_factory=dict)
    tables: list[tuple[int, int, int]] = field(default_factory=list)   # (table, bytes, jr site)
    image: tuple[int, int] | None = None                               # [lo, hi) of the image

    def __post_init__(self):
        self._index = {}

    def within(self, name: str, lo: int, hi: int) -> list[int]:
        index = self._index.get(name)
        if index is None or index.mapping is not getattr(self, name):
            index = self._index[name] = _Index(getattr(self, name))
        return index.within(lo, hi)

    @classmethod
    def from_image(cls, data: bytes, delta: int, rows, indirect=None) -> "Context":
        """References found in the catalogued functions of a linear image (``rows`` = (name,
        vram, size)), data words outside them that point into the image, and, given an
        :class:`n64dk.analysis.indirect.ImageResult`, its switch tables and pointer-table entries."""
        ctx = cls(image=(delta, delta + len(data)))
        spans = sorted((v, v + s) for _n, v, s in rows if s > 0 and 0 <= v - delta and v - delta + s <= len(data))
        for lo, hi in spans:
            code = data[lo - delta:hi - delta]
            for ref in mips.scan_refs(code, lo):
                if ref.kind == "call":
                    ctx.call_sites.setdefault(ref.address, []).append(ref.site)
                elif ref.kind == "jump":
                    ctx.jump_sites.setdefault(ref.address, []).append(ref.site)
                elif ref.kind == "addr":
                    ctx.address_refs[ref.address] = ctx.address_refs.get(ref.address, 0) + 1
                elif ref.kind == "load":
                    ctx.loads.setdefault(ref.address, set()).add(ref.opcode)
                elif ref.kind == "store":
                    ctx.stores.setdefault(ref.address, set()).add(ref.opcode)
        firsts = [lo for lo, _hi in spans]
        top = delta + len(data)
        for offset in range(0, len(data) - 3, 4):
            word = struct.unpack_from(">I", data, offset)[0]
            if word & 3 or not delta <= word < top:
                continue
            here = delta + offset
            i = bisect.bisect_right(firsts, here) - 1
            if i >= 0 and here < spans[i][1]:
                continue                                    # inside catalogued code: an instruction
            ctx.pointer_targets[word] = ctx.pointer_targets.get(word, 0) + 1
        if indirect is not None:
            for site in indirect.sites:
                if site.kind == "switch" and site.status == "definite":
                    ctx.switch_targets.update(site.targets)
                    ctx.tables.append((site.table, 4 * site.count, site.site))
                elif site.status in ("definite", "finite") and site.kind != "switch-unbounded":
                    for target in site.targets:
                        ctx.pointer_targets[target] = ctx.pointer_targets.get(target, 0) + 1
            for result in indirect.functions.values():
                ctx.switch_tables.update(result.tables)
        return ctx


# --------------------------------------------------------------------------- shape tests

def float_shape(words: list[int]):
    """(plausible share, largest same-exponent group) when at least 75% of at least four non-zero
    words are floats of magnitude 2^-15..2^16, else None."""
    nonzero = [w for w in words if w]
    if len(nonzero) < 4:
        return None
    plausible = [w for w in nonzero if 0x70 <= (w >> 23) & 0xFF <= 0x8F]
    share = len(plausible) / len(nonzero)
    if share < 0.75:
        return None
    group = Counter((w >> 23) & 0xFF for w in plausible).most_common(1)[0][1]
    return share, group


def pointer_shape(words: list[int], image) -> tuple[int, int] | None:
    lo, hi = image or _KSEG0
    nonzero = [w for w in words if w]
    pointers = [w for w in nonzero if not w & 3 and lo <= w < hi]
    if len(nonzero) >= 3 and len(pointers) >= 0.9 * len(nonzero):
        return len(pointers), len(nonzero)
    return None


def string_share(raw: bytes) -> float:
    body = raw.rstrip(b"\0")
    if len(body) < 8:
        return 0.0
    covered, p, end = 0, 0, len(raw)
    while p < len(body):
        found = string_at(raw, p, end)
        if found:
            covered += found[0] - p
            p = found[0]
        else:
            p += 1
    return min(1.0, covered / len(body))


def display_list(raw: bytes, address: int):
    """(family, commands) when the bytes are an 8-aligned GBI display list ending in G_ENDDL with
    at least 90% known opcodes over at least three commands."""
    if address & 7:
        return None
    from ..assets import gbi
    for family in ("f3dex2", "f3dex", "f3d"):
        table, end_word = gbi.OPCODES[family], gbi.ENDDL_WORDS[family]
        known = total = 0
        ended = False
        for i in range(0, len(raw) - 7, 8):
            w0, w1 = struct.unpack_from(">II", raw, i)
            total += 1
            if w0 == end_word and w1 == 0:
                known += 1
                ended = True
                break
            if (w0 >> 24) in table:
                known += 1
        if ended and total >= 3 and known >= 0.9 * total:
            return family, total
    return None


def stride_shape(words: list[int]):
    """(record words, constant columns, records) for a fixed-stride layout: at least four records,
    at least half of the columns identical in every record, one of them non-zero, and at least
    one column that varies."""
    for period in (2, 3, 4, 5, 6, 8):
        records = len(words) // period
        if records < 4:
            continue
        constant = [c for c in range(period) if len({words[r * period + c] for r in range(records)}) == 1]
        if len(constant) >= max(1, period // 2) and len(constant) < period \
                and any(words[c] for c in constant):
            return period, len(constant), records
    return None


# --------------------------------------------------------------------------- classification

def decide(code: list[Evidence], data: list[Evidence]) -> tuple[str, float, float]:
    cs, ds = sum(e.weight for e in code), sum(e.weight for e in data)
    strong_code = any(e.weight >= STRONG for e in code)
    strong_data = any(e.weight >= STRONG for e in data)
    if strong_code and strong_data:
        return AMBIGUOUS, cs, ds
    if any(e.reference for e in code) and cs >= 3 and not strong_data and cs >= 2 * ds + 1:
        return CODE, cs, ds
    if ds >= 3 and not strong_code and ds >= 2 * cs + 1:
        return DATA, cs, ds
    return AMBIGUOUS, cs, ds


def classify(data: bytes, delta: int, lo: int, hi: int, ctx: Context | None = None,
             max_bytes: int = MAX_BYTES) -> Verdict:
    """Classify [lo, hi) of a linear image (file offset = address - delta)."""
    ctx = ctx or Context()
    notes: list[str] = []
    offset = lo - delta
    size = min(hi - lo, max_bytes)
    if hi - lo > max_bytes:
        notes.append(f"only the first {max_bytes:#x} of {hi - lo:#x} bytes examined")
    raw = data[max(0, offset):max(0, offset) + size] if offset >= 0 else b""
    raw = raw[:len(raw) & ~3]
    words = [w for (w,) in struct.iter_unpack(">I", raw)]
    code_ev: list[Evidence] = []
    data_ev: list[Evidence] = []
    if not words:
        return Verdict(lo, hi, AMBIGUOUS, 0.0, 0.0, [], [], notes + ["no bytes"])

    def code(weight, what, reference=False):
        code_ev.append(Evidence("code", weight, what, reference))

    def dat(weight, what, reference=False):
        data_ev.append(Evidence("data", weight, what, reference))

    # references from elsewhere ---------------------------------------------------------------
    calls = [s for s in ctx.call_sites.get(lo, ()) if not lo <= s < hi]
    if calls:
        code(3.0, f"called by {len(calls)} jal site(s), e.g. {hx(calls[0])}", True)
    jumps = [s for s in ctx.jump_sites.get(lo, ()) if not lo <= s < hi]
    if jumps:
        code(2.0, f"tail-jump target of {len(jumps)} j site(s), e.g. {hx(jumps[0])}", True)
    cases = ctx.within("switch_targets", lo, hi)
    if cases:
        code(3.0, f"{len(cases)} resolved jump-table target(s) inside, e.g. {hx(min(cases))}", True)
    if ctx.pointer_targets.get(lo):
        code(1.5, f"pointed at by {ctx.pointer_targets[lo]} data word(s) or pointer-table entries", True)
    if ctx.address_refs.get(lo):
        code(1.0, f"address taken by {ctx.address_refs[lo]} lui/addiu pair(s)", True)

    # control flow from the start -------------------------------------------------------------
    tables = {s: t for s, t in ctx.switch_tables.items() if lo <= s < lo + len(raw)}
    fl = flowmod.analyse(raw, lo, tables)
    reached = sorted(i for b in fl.blocks if b.number in fl.reachable for i in range(b.first, b.last + 1))
    reached_set = set(reached)
    kinds = Counter(kind for _s, kind, _t in fl.exits)
    invalid_reached = [fl.insns[i].vram for i in reached if fl.insns[i].op == ".word"]
    closed = bool(kinds.get("return") or kinds.get("jump-outside")) and not kinds.get("falls-off-end") \
        and not kinds.get("branch-outside") and not invalid_reached
    unresolved_jr = bool(kinds.get("computed-jump"))
    if closed:
        code(2.0, "control flow from the start closes (" + ", ".join(sorted(kinds)) + ")")
        notes_end = lo + 4 * (max(reached) + 1)
        unreached = [w for i, w in enumerate(words) if i not in reached_set and i < len(fl.insns)]
        if any(unreached):
            tail = [w for w in words[max(reached) + 1:]]
            notes.append(f"{sum(1 for w in unreached if w)} non-zero word(s) not reached from the start")
            shape_notes = _shapes(tail, raw[4 * (max(reached) + 1):], notes_end, ctx)
            notes += [f"after the reached code: {e.what}" for e in shape_notes]
        extent = (lo, notes_end)
    else:
        extent = None
    frame = fl.frame
    anomalies = " ".join(frame.anomalies)
    if frame.size:
        released = any(fl.insns[i].word == (0x27BD0000 | frame.size) for i in reached)
        if released and "released by" not in anomalies and "without allocation" not in anomalies:
            code(1.5, f"stack frame of {frame.size:#x} bytes allocated and released")
    if frame.saved and closed and set(frame.saved) <= frame.restored \
            and "without being saved" not in anomalies and "never saved" not in anomalies:
        code(1.0, "saved registers (" + ", ".join(flowmod.reg_name(r) for r in sorted(frame.saved)) + ") restored")

    # impossible control flow -----------------------------------------------------------------
    if invalid_reached:
        dat(3.0, f"{len(invalid_reached)} undefined opcode(s) reached from the start, first at {hx(invalid_reached[0])}")
    zero_writes = odd = 0
    window = lo & 0xFF800000
    far = []
    for i in reached:
        insn = fl.insns[i]
        if not insn.word or insn.op == ".word":
            continue
        ins = mips.decode(insn.word, insn.vram)
        if ins.outputsToGprZero() and not ins.isNop():
            zero_writes += 1
        if not ins.isImplemented() or ins.notEmittedByCompilers():
            odd += 1
        if insn.kind == "call" and insn.target is not None and not window <= insn.target < window + 0x800000:
            far.append(insn.target)
    if zero_writes:
        dat(2.0 if zero_writes >= 2 else 1.0, f"{zero_writes} instruction(s) writing $zero")
    if odd >= 2:
        dat(1.0, f"{odd} instruction(s) compilers do not emit")
    if far:
        dat(1.0, f"jal to {hx(far[0])}, outside the image's 8 MiB window")
    if any("inside a delay slot" in a for a in fl.anomalies):
        dat(1.5, "a branch targets a delay slot")
    nonzero_unreached = sum(1 for i, w in enumerate(words) if w and i not in reached_set)
    nonzero_reached = sum(1 for i in reached if words[i])
    if not unresolved_jr and nonzero_unreached >= 4 and nonzero_unreached >= 2 * max(1, nonzero_reached):
        dat(3.0, f"most of the interval is not reached from its start ({nonzero_unreached} of "
                 f"{nonzero_unreached + nonzero_reached} non-zero words)")
        invalid_rest = sum(1 for i, insn in enumerate(fl.insns) if i not in reached_set and insn.op == ".word")
        if invalid_rest:
            dat(1.5, f"{invalid_rest} undefined opcode(s) in the unreached part")

    # references as data ----------------------------------------------------------------------
    for table, size_, site in ctx.tables:
        if table < hi and lo < table + size_:
            dat(4.0, f"holds the jump table at {hx(table)} ({size_ // 4} entries) of the switch at {hx(site)}", True)
            break
    loads: set[str] = set()
    tail_loads: set[str] = set()
    for address in ctx.within("loads", lo, hi):
        if not closed or (address - lo) // 4 in reached_set:
            loads |= ctx.loads[address]
        else:
            tail_loads |= ctx.loads[address]
    if tail_loads:
        notes.append("the unreached part is loaded by " + ", ".join(sorted(tail_loads)))
    if loads & _DATA_LOADS:
        dat(2.0, "loaded as data (" + ", ".join(sorted(loads)) + ")", True)
    elif loads:
        dat(1.0, "loaded by " + ", ".join(sorted(loads)), True)
    written = ctx.within("stores", lo, hi)
    if written:
        dat(1.5, f"written by stores at {len(written)} address(es)", True)

    # shapes (only scored when control flow does not close) -----------------------------------
    if not closed:
        data_ev.extend(_shapes(words, raw, lo, ctx))
    verdict, cs, ds = decide(code_ev, data_ev)
    return Verdict(lo, hi, verdict, cs, ds, code_ev, data_ev, notes, extent)


def _shapes(words: list[int], raw: bytes, address: int, ctx: Context) -> list[Evidence]:
    out = []
    share = string_share(raw)
    if share >= 0.6:
        out.append(Evidence("data", 3.0, f"NUL-terminated strings cover {share:.0%} of the bytes"))
    dl = display_list(raw, address)
    if dl:
        out.append(Evidence("data", 3.0, f"{dl[0]} display list of {dl[1]} commands ending in G_ENDDL"))
    floats = float_shape(words)
    if floats:
        plausible, group = floats
        nonzero = sum(1 for w in words if w)
        if group >= 3 and group >= 0.3 * nonzero:
            out.append(Evidence("data", 2.0, f"{plausible:.0%} plausible floats, {group} sharing one exponent"))
        else:
            out.append(Evidence("data", 1.0, f"{plausible:.0%} plausible floats"))
    pointers = pointer_shape(words, ctx.image)
    if pointers:
        out.append(Evidence("data", 2.0, f"{pointers[0]} of {pointers[1]} non-zero words are aligned addresses "
                                         "inside the image (pointer array)"))
    stride = stride_shape(words)
    if stride:
        out.append(Evidence("data", 1.0, f"stride {4 * stride[0]}: {stride[1]} of {stride[0]} words identical "
                                         f"across {stride[2]} records"))
    return out


def summarise(verdicts: list[Verdict]) -> dict:
    return dict(Counter(v.verdict for v in verdicts))
