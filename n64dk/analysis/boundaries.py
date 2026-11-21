"""Boundary solver: turn boundary evidence into reversible catalogue corrections.

A function catalogue (``[(name, vram, size)]`` in one linear image's address space) is checked
row by row against independent evidence gathered with the existing analyses:

* call targets (``jal``) from this image and any other image whose code may call into it;
* address-taken ``lui``/``addiu`` pairs and data pointers, excluding jump-table slots
  (``fragments.audit``);
* ``flow.entry_evidence``: caller temporaries read at entry, saved registers used without a
  save, restores without saves, stack release without allocation, control running off the end,
  branches leaving the range, ``j`` outside;
* jump tables (``flow.find_jump_tables``), trailing padding (``flow.nop_roles``), words that do
  not decode (embedded data) and hand-written instructions (``causes.target_causes``);
* manifest-verified intervals, which are ground truth and are never altered.

The result is a list of **operations**.  Every operation is self-contained JSON: the rows it
removes and the rows it inserts (so it can be applied, skipped or inverted without the
evidence), the bytes it releases to a non-function class, a confidence, the evidence as
concrete facts with addresses, and what it affects.  ``apply`` executes operations at or above a
confidence threshold; ``invert`` produces the exact inverse.

Kinds: ``merge``, ``split``, ``extend``, ``shrink``, ``retype`` (``data``, ``jump-table``,
``padding``, ``handwritten``, ``nonstandard-abi``) and ``add``.  ``retype handwritten`` and
``retype nonstandard-abi`` are annotations (the row stays a function) and change no row.

Nothing here knows a game, a compiler version or a project layout.
"""
from __future__ import annotations

import bisect
import json
import re
import struct
from pathlib import Path

from . import flow as flowmod
from . import fragments as fragmod
from ..match import causes as causemod

Row = tuple[str, int, int]

# entry_evidence problems that are about the START of a range (the range is entered from inside
# another body) versus its END (the range stops before the real end of the body).
_START_PROBLEMS = ("before writing", "at entry without saving", "never saved", "without allocation",
                   "without being saved", "never saves $ra")
_END_PROBLEMS = ("runs off the end", "leave the range")
_STRONG_INTERIOR = ("branch-target@", "previous-body-branches-past-start", "reads-$")
_KIND_ORDER = ("merge", "split", "extend", "shrink", "retype", "add")


def hx(value: int) -> str:
    return f"0x{value:08X}"


def _row_json(row: Row) -> dict:
    return {"name": row[0], "vram": hx(row[1]), "size": hx(row[2])}


def _row_tuple(item: dict) -> Row:
    return (item["name"], int(item["vram"], 16), int(item["size"], 16))


# ---------------------------------------------------------------------------------------------
# evidence


class Image:
    """One linear image: file offset = vram - vram_delta."""

    def __init__(self, data: bytes, vram_delta: int):
        self.data, self.delta = data, vram_delta

    @property
    def lo(self) -> int:
        return self.delta

    @property
    def hi(self) -> int:
        return self.delta + len(self.data)

    def word(self, vram: int) -> int | None:
        off = vram - self.delta
        return struct.unpack_from(">I", self.data, off)[0] if 0 <= off <= len(self.data) - 4 else None

    def bytes(self, vram: int, size: int) -> bytes:
        off = vram - self.delta
        return self.data[off:off + size]

    def flow(self, vram: int, size: int) -> flowmod.Flow:
        return flowmod.analyse_in_image(self.data, self.delta, vram, size)


def call_sites(images: list[Image], lo: int, hi: int) -> dict[int, list[int]]:
    """``jal`` target -> sites, for targets in [lo, hi), from every image."""
    out: dict[int, list[int]] = {}
    for image in images:
        data = image.data
        for off, (word,) in enumerate(struct.iter_unpack(">I", data[:len(data) & ~3])):
            if word >> 26 != 3:
                continue
            site = image.delta + 4 * off
            target = ((site + 4) & 0xF0000000) | ((word & 0x03FFFFFF) << 2)
            if lo <= target < hi:
                out.setdefault(target, []).append(site)
    return out


def _table_slots(image: Image, rows: list[Row]) -> tuple[set[int], dict[int, list[int]]]:
    """Addresses of jump-table slot words and {jr site: targets} for every catalogued row."""
    slots: set[int] = set()
    tables: dict[int, list[int]] = {}
    for _name, vram, size in rows:
        code = image.bytes(vram, size)
        found = flowmod.find_jump_tables(code, vram, image.word)
        for site, targets in found.items():
            tables[site] = targets
            packed = b"".join(struct.pack(">I", t) for t in targets[:4])
            at = image.data.find(packed)
            while at >= 0 and at % 4:
                at = image.data.find(packed, at + 1)
            if at >= 0:
                base = image.delta + at
                slots.update(base + 4 * k for k in range(len(targets)))
    return slots, tables


def _terminated(fl: flowmod.Flow) -> tuple[bool, int]:
    """(ends in an unconditional terminator + delay slot, bytes of trailing zero padding)."""
    insns = fl.insns
    end = len(insns)
    while end and insns[end - 1].word == 0:
        end -= 1
    pad = 4 * (len(insns) - end)
    # the last zero may be the terminator's delay slot
    for back in (2, 1):
        k = end - back
        if 0 <= k < len(insns):
            insn = insns[k]
            if insn.kind in ("return", "jump", "jr") or (insn.kind == "branch" and insn.word >> 16 == 0x1000) \
                    or insn.op == "eret":
                slot_pad = 4 if back == 1 else 0          # delay slot was counted as padding
                return True, max(0, pad - slot_pad)
    return False, pad


def _closed(fl: flowmod.Flow) -> list[str]:
    """What keeps a range from being a closed body (empty when it is closed)."""
    open_ = []
    kinds = {}
    for site, kind, target in fl.exits:
        kinds.setdefault(kind, []).append((site, target))
    for site, target in kinds.get("branch-outside", []):
        open_.append(f"branch at {hx(site)} leaves the range to {hx(target)}")
    for site, _t in kinds.get("falls-off-end", []):
        open_.append(f"control runs off the end after {hx(site)}")
    if not _terminated(fl)[0] and not kinds.get("falls-off-end"):
        open_.append("does not end in a terminator")
    return open_


def _open_targets(fl: flowmod.Flow) -> tuple[set[int], bool]:
    targets = {t for _s, kind, t in fl.exits if kind == "branch-outside" and t is not None}
    return targets, any(kind == "falls-off-end" for _s, kind, _t in fl.exits)


class Facts:
    """Evidence for one catalogue row."""

    def __init__(self, row: Row, image: Image, calls: list[int], start: fragmod.Start | None):
        self.row = row
        self.name, self.vram, self.size = row
        self.end = self.vram + self.size
        self.calls = calls
        self.taken = [e for e in (start.entry_evidence if start else []) if not e.startswith(("jal@", "first-start"))]
        self.interior = list(start.interior_evidence) if start else []
        self.flow = image.flow(self.vram, self.size)
        self.ev = flowmod.entry_evidence(self.flow)
        self.start_problems = [p for p in self.ev["problems"] if any(k in p for k in _START_PROBLEMS)]
        self.end_problems = [p for p in self.ev["problems"] if any(k in p for k in _END_PROBLEMS)]
        self.terminated, self.padding = _terminated(self.flow)
        self.invalid = [i.vram for i in self.flow.insns if i.op == ".word" and i.word]
        self.zero = all(i.word == 0 for i in self.flow.insns)
        self.noncode: str | None = None          # padding | jump-table | data, set by the solver
        self.has_frame = bool(self.flow.insns) and self.flow.insns[0].word >> 16 == 0x27BD \
            and bool(self.flow.insns[0].word & 0x8000)

    @property
    def entry(self) -> bool:
        return bool(self.calls or self.taken)

    @property
    def strong_interior(self) -> list[str]:
        out = [e for e in self.interior if e.startswith(_STRONG_INTERIOR)]
        out += [p for p in self.start_problems if "before writing" in p or "never saved" in p
                or "without allocation" in p]
        return out

    @property
    def fragment(self) -> bool:
        if self.entry or self.noncode:
            return False
        if self.strong_interior:
            return True
        # weak evidence only (previous entry falls through): a row with its own frame is a function
        return bool(self.interior or self.start_problems) and not self.has_frame

    def why_entry(self) -> list[str]:
        out = [f"jal from {hx(s)}" for s in self.calls[:4]]
        if len(self.calls) > 4:
            out.append(f"... {len(self.calls)} call sites in all")
        return out + self.taken[:3]


# ---------------------------------------------------------------------------------------------
# solver


class Verified:
    """Manifest-verified intervals: (start, end, label)."""

    def __init__(self, intervals=()):
        self.items = sorted((int(s), int(e), str(label)) for s, e, label in intervals)
        self.starts = [s for s, _e, _l in self.items]

    def overlapping(self, start: int, end: int) -> list[tuple[int, int, str]]:
        out = []
        k = bisect.bisect_right(self.starts, start)
        for s, e, label in self.items[max(0, k - 64):]:
            if s >= end:
                break
            if e > start and s < end:
                out.append((s, e, label))
        return out

    def exact(self, start: int, end: int) -> bool:
        return any(s == start and e == end for s, e, _l in self.overlapping(start, end))


def call_stub(image: "Image", vram: int, size: int) -> tuple[int, int] | None:
    """(bytes of the stub, address of the function after it) when the range begins with ``jal f`` and its
    delay slot, then zero words, then code that opens a frame and saves $ra within its first instructions."""
    words = [image.word(vram + 4 * k) for k in range(min(size // 4, 16))]
    if len(words) < 8 or None in words or words[0] >> 26 != 3 or words[1] == 0:
        return None
    k = 2
    while k < len(words) and words[k] == 0:
        k += 1
    if k == 2 or k >= len(words) - 2:
        return None
    body = words[k:k + 8]
    frame = [i for i, w in enumerate(body) if w >> 16 == 0x27BD and w & 0x8000]
    saves = [i for i, w in enumerate(body) if w >> 16 == 0xAFBF]
    if not frame or not saves or saves[0] < frame[0] or any(w >> 26 in (2, 3) or w == 0x03E00008 for w in body[:saves[0]]):
        return None
    return 8, vram + 4 * k


_RA_RETURN, _SP_FROM_FP = 0x03E00008, (0x03C0E821, 0x03C0E825, 0x03C0E82D)     # jr $ra; move $sp, $fp


def unreachable_epilogue(owner_words: list[int], piece_words: list[int]) -> str | None:
    """Why ``piece_words`` are the epilogue of the function ``owner_words`` that nothing reaches, or None.

    An unoptimising compiler emits a function's epilogue even when control never gets there (the body ends
    in an endless loop).  A catalogue cut at the last jump then lists the epilogue as a function of its own.
    It is that epilogue, and no function, when all of this holds: the owner opens a frame and never returns;
    the piece does nothing but restore registers from the stack, release a frame and return; the frame it
    releases is the owner's; and every register it restores was saved at that offset by the owner."""
    if not owner_words or owner_words[0] >> 16 != 0x27BD or not owner_words[0] & 0x8000 \
            or _RA_RETURN in owner_words:
        return None
    frame = 0x10000 - (owner_words[0] & 0xFFFF)
    saved = {(w & 0xFFFF): (w >> 16) & 31 for w in owner_words
             if w >> 26 in (0x2B, 0x3F) and (w >> 21) & 31 == 29}         # sw/sd reg, offset($sp)
    restored, released, returned = [], False, False
    for w in piece_words:
        if w == 0 or w in _SP_FROM_FP:
            continue
        if w >> 26 in (0x23, 0x37) and (w >> 21) & 31 == 29:              # lw/ld reg, offset($sp)
            if saved.get(w & 0xFFFF) != (w >> 16) & 31:
                return None
            restored.append((w >> 16) & 31)
        elif w >> 16 == 0x27BD and not w & 0x8000:
            if (w & 0xFFFF) != frame or released:
                return None
            released = True
        elif w == _RA_RETURN:
            returned = True
        else:
            return None
    if not (restored and released and returned):
        return None
    return (f"the epilogue of a function that never returns: it releases the same {frame}-byte frame and restores "
            f"{len(restored)} register(s), each saved at that offset by the function's prologue")


def _op(kind: str, confidence: float, evidence: list[str], remove=(), insert=(), released=(), **extra) -> dict:
    op = {"kind": kind, "confidence": round(confidence, 2), "status": "proposed",
          "remove": [_row_json(r) for r in remove], "insert": [_row_json(r) for r in insert],
          "released": [{"vram": hx(v), "size": hx(s), "class": c} for v, s, c in released],
          "evidence": evidence}
    op.update(extra)
    return op


def solve(image: Image, catalogue: list[Row], also: list[Image] = (), verified: Verified | None = None,
          keep_padding: bool = False, name_format: str = "func_{vram:08X}", name_shift: int = 0,
          extent: tuple[int, int] | None = None, stats: dict | None = None, max_gap: int = 0x400) -> list[dict]:
    """Propose boundary operations for ``catalogue`` (already in ``image``'s address space).

    ``keep_padding``: the project's convention keeps alignment padding inside the preceding
    function (no shrink).  ``name_format``/``name_shift`` name new rows (``name_shift`` converts
    an image address to the catalogue's address space for the name)."""
    verified = verified or Verified()
    rows = sorted({r for r in catalogue if r[2] > 0 and image.lo <= r[1] and r[1] + r[2] <= image.hi},
                  key=lambda r: r[1])
    if not rows:
        return []
    lo, hi = extent or (rows[0][1], max(r[1] + r[2] for r in rows))
    stats = stats if stats is not None else {}
    stats.setdefault("padding_kept_by_verified_unit", 0)
    stats.setdefault("call_sites_outside_code", 0)
    # call sites in this image count only inside catalogued rows (a whole ROM holds data and
    # compressed segments whose words can look like ``jal``); other images are trusted whole
    code_starts = [r[1] for r in rows]

    def in_code(site: int) -> bool:
        k = bisect.bisect_right(code_starts, site) - 1
        return k >= 0 and rows[k][1] <= site < rows[k][1] + rows[k][2]
    calls = {}
    for target, sites in call_sites([image], lo, hi).items():
        kept = [s for s in sites if in_code(s)]
        stats["call_sites_outside_code"] += len(sites) - len(kept)
        if kept:
            calls[target] = kept
    for target, sites in call_sites(list(also), lo, hi).items():
        calls.setdefault(target, []).extend(sites)
    slots, _tables = _table_slots(image, rows)
    audit_items = {}
    audited = fragmod.audit(image.data, image.delta, rows, [(i.data, i.delta) for i in also], slots)
    for item in audited["starts"]:
        start = fragmod.Start(item["name"], int(item["vram"], 16), item["size"], item["entry_evidence"],
                              item["interior_evidence"])
        audit_items[start.vram] = start
    # audit truncates entry evidence to 4 items; that is enough to know there is some
    facts = [Facts(r, image, calls.get(r[1], []), audit_items.get(r[1])) for r in rows]
    by_start = {f.vram: f for f in facts}
    starts = [f.vram for f in facts]
    name = lambda v: name_format.format(vram=v + name_shift)   # noqa: E731
    ops: list[dict] = []
    consumed: set[int] = set()          # row starts already used by an operation

    call_targets = sorted(calls)

    def bridge(end: int, start: int) -> bool:
        """Is [end, start) empty, or uncatalogued code that nothing calls (so the chain stays
        contiguous through it)?"""
        if start == end:
            return True
        if start < end or start - end > max_gap:
            return False
        k = bisect.bisect_left(call_targets, end)
        if k < len(call_targets) and call_targets[k] < start:
            return False
        words = [image.word(v) for v in range(end, start, 4)]
        if all(w == 0 for w in words):
            return False
        return all(i.op != ".word" for i in flowmod.decode(image.bytes(end, start - end), end))

    def row_after(end: int) -> Facts | None:
        return by_start.get(end)

    for f in facts:
        if f.zero:
            f.noncode = "padding"
        elif not f.calls and f.size >= 8 and all(
                (w := image.word(v)) is not None and not w & 3 and lo <= w < hi for v in range(f.vram, f.end, 4)):
            f.noncode = "jump-table"
        elif not f.entry and f.invalid and (len(f.invalid) * 4 >= f.size // 4 or f.invalid[0] == f.vram):
            f.noncode = "data"

    # --- merges: a run of fragments joins the nearest preceding real entry while contiguous
    k = 0
    while k < len(facts):
        owner = facts[k]
        if owner.fragment:
            k += 1
            continue
        chain, end, j = [], owner.end, k + 1
        while j < len(facts) and facts[j].fragment and bridge(end, facts[j].vram):
            chain.append(facts[j])
            end = facts[j].end
            j += 1
        if not chain:
            k += 1
            continue
        if len(chain) == 1 and chain[0].size > 8 and not _closed(owner.flow):
            # One word that nothing executes in front of a function: the body before it is closed, nothing
            # refers to the word, and what follows it opens a frame and is a closed body with complete entry
            # evidence.  The word is not code (padding of another kind, a constant); the function starts after it.
            piece = chain[0]
            rest = image.flow(piece.vram + 4, piece.size - 4)
            second = rest.insns[0].word if rest.insns else 0
            if second >> 16 == 0x27BD and second & 0x8000 and not _closed(rest) \
                    and flowmod.entry_evidence(rest)["complete"]:
                why = ("one word in front of a function that nothing executes: the body before it is closed, nothing "
                       "calls or points to it, and the frame is opened by the word after it")
                ops.append(_op("shrink", 0.9, [f"{piece.name} at {hx(piece.vram)}: " + "; ".join(
                    (piece.strong_interior or piece.start_problems)[:2]), why], [piece.row],
                    [(name(piece.vram + 4), piece.vram + 4, piece.size - 4)], [(piece.vram, 4, "data")], certain=why))
                consumed.add(piece.vram)
                k = j
                continue
        merged = image.flow(owner.vram, end - owner.vram)
        open_ = _closed(merged)
        extended = []
        while open_ and j < len(facts) and not facts[j].entry and bridge(end, facts[j].vram) and len(extended) < 16:
            targets, falls = _open_targets(merged)
            if not falls and not any(facts[j].vram <= t < facts[j].end for t in targets):
                break
            extended.append(facts[j])
            end = facts[j].end
            j += 1
            merged = image.flow(owner.vram, end - owner.vram)
            open_ = _closed(merged)
        absorbed = chain + extended
        evidence = [f"{owner.name} at {hx(owner.vram)} is an entry: " + (", ".join(owner.why_entry()) or
                                                                       "no interior evidence (not a fragment)")]
        weak = False
        for f in chain:
            strong = f.strong_interior
            weak |= not strong
            evidence.append(f"{f.name} at {hx(f.vram)}: no call or pointer evidence; "
                            + "; ".join((strong or f.interior or f.start_problems)[:3]))
        for f in extended:
            evidence.append(f"{f.name} at {hx(f.vram)} absorbed to close the body (no entry evidence)")
        bridged = (end - owner.vram) - owner.size - sum(f.size for f in absorbed)
        if bridged:
            evidence.append(f"bridges {bridged} uncatalogued byte(s) of decodable code that no call targets")
        merged_ev = flowmod.entry_evidence(merged)
        if open_:
            confidence = 0.5
        elif weak:
            confidence = 0.75
        else:
            confidence = 0.9
        if merged_ev["complete"]:
            evidence.append(f"merged body {hx(owner.vram)}..{hx(end)} is closed and entry_evidence is complete")
        else:
            confidence = min(confidence, 0.6)
            evidence.append("merged body still has entry problems: " + "; ".join(merged_ev["problems"][:2]))
        new = (owner.name, owner.vram, end - owner.vram)
        certain = None
        if len(absorbed) == 1 and not bridged and not open_ and merged_ev["complete"]:
            certain = unreachable_epilogue([i.word for i in owner.flow.insns], [i.word for i in absorbed[0].flow.insns])
        if certain is None and not open_ and merged_ev["complete"] and not weak and not extended:
            # The first part is not a function by itself: its control runs into the parts after it (a branch
            # into them, or off its end), nothing calls or points to those parts, each of them could not be
            # entered as a function (it restores what it never saved, or reads what only the code before it
            # set), and the whole is a closed body.  One function, cut where a catalogue saw a return.  Code
            # between the parts that no row lists and nothing calls belongs to it as well.
            alone = _closed(owner.flow)
            targets, falls = _open_targets(owner.flow)
            if alone and (falls or targets) and all(owner.end <= t < end for t in targets):
                certain = ("the first part is not a closed body (its control runs into the next part), no part "
                           "after it can be entered as a function, and the whole is a closed body")
        if certain:
            evidence.append(certain)
        ops.append(_op("merge", 0.99 if certain else confidence, evidence, [owner.row, *(f.row for f in absorbed)],
                       [new], absorbed=[f.name for f in absorbed], open=open_,
                       closed=not open_ and merged_ev["complete"], **({"certain": certain} if certain else {})))
        consumed.update(f.vram for f in [owner, *absorbed])
        k = j

    # --- per-row operations
    for f in facts:
        if f.vram in consumed:
            continue
        stub = call_stub(image, f.vram, f.size)
        if stub is not None:
            first, rest = stub
            why = ("a call and its delay slot, padding, then a body that opens a frame and saves $ra: a compiler "
                   "never calls before it saves $ra in a function that saves it, so the call is a routine of its "
                   "own that does not return (hand-written) and the function starts after the padding")
            ops.append(_op("split", 0.95, [f"{f.name}: {why}"], [f.row],
                           [(f.name, f.vram, first), (name(rest), rest, f.end - rest)],
                           [(f.vram + first, rest - f.vram - first, "padding")], at=[hx(rest)], open=[], certain=why))
            consumed.add(f.vram)
            continue
        # entire row is not code
        if f.noncode == "padding":
            ops.append(_op("retype", 0.95, [f"{f.name}: all {f.size} bytes are zero"], [f.row], [],
                           [(f.vram, f.size, "padding")], to="padding"))
            continue
        if f.noncode == "jump-table":
            in_slots = any(v in slots for v in range(f.vram, f.end, 4))
            ops.append(_op("retype", 0.95 if in_slots else 0.85,
                           [f"{f.name}: every word is an aligned code address inside the catalogue span"
                            + ("; a jump table loads from it" if in_slots else "")],
                           [f.row], [], [(f.vram, f.size, "jump-table")], to="jump-table"))
            continue
        if f.noncode == "data":
            ops.append(_op("retype", 0.85, [f"{f.name}: {len(f.invalid)} of {f.size // 4} words do not decode "
                                            f"(first at {hx(f.invalid[0])}); no call or pointer evidence"],
                           [f.row], [], [(f.vram, f.size, "data")], to="data"))
            continue
        # interior call targets: split
        interior = sorted(t for t in calls if f.vram < t < f.end and not t & 3)
        cuts = []
        for t in interior:
            prev = image.flow(f.vram if not cuts else cuts[-1], t - (f.vram if not cuts else cuts[-1]))
            term, _pad = _terminated(prev)
            if term:
                cuts.append(t)
        if cuts:
            bounds = [f.vram, *cuts, f.end]
            parts = [(f.name if a == f.vram else name(a), a, b - a) for a, b in zip(bounds, bounds[1:])]
            open_ = []
            for a, b in zip(bounds, bounds[1:]):
                open_ += _closed(image.flow(a, b - a))
            evidence = [f"jal into the interior at {hx(t)} from {', '.join(hx(s) for s in calls[t][:3])}"
                        for t in cuts]
            evidence.append("the part before each cut ends in a terminator and delay slot")
            ops.append(_op("split", 0.9 if not open_ else 0.6, evidence, [f.row], parts, at=[hx(c) for c in cuts],
                           open=open_))
            consumed.add(f.vram)
            continue
        if f.calls and any("before writing" in p or "at entry without saving" in p for p in f.start_problems):
            ops.append(_op("retype", 0.9, [f"{f.name} is called ({', '.join(f.why_entry()[:2])}) but "
                                           + "; ".join(f.start_problems[:2])],
                           to="nonstandard-abi", flags=["needs-tu"], target=_row_json(f.row)))
        hand = [c for c in causemod.target_causes(f.flow) if c.id == "assembly.handwritten" and c.confidence >= 0.85]
        if hand:
            ops.append(_op("retype", hand[0].confidence, [f"{f.name}: " + "; ".join(hand[0].evidence)], to="handwritten",
                           target=_row_json(f.row)))
        # end of the row
        nxt = row_after(f.end)
        gap_end = next((s for s in starts if s >= f.end), hi)
        if any("runs off the end" in p for p in f.end_problems) and f.flow.insns:
            last = f.flow.insns[-1]
            if last.control and gap_end > f.end and nxt is None:
                ops.append(_op("extend", 0.9, [f"{f.name}: last instruction {last.text!r} at {hx(last.vram)} "
                                               f"has its delay slot at {hx(f.end)}, outside the row; the bytes "
                                               "after it are not catalogued"],
                               [f.row], [(f.name, f.vram, f.size + 4)]))
                continue
            if gap_end > f.end and nxt is None:
                for stop in range(f.end + 8, gap_end + 4, 4):
                    trial = image.flow(f.vram, stop - f.vram)
                    if _terminated(trial)[0] and not _closed(trial):
                        ops.append(_op("extend", 0.8, [f"{f.name}: control runs off the end at {hx(f.end)}; the "
                                                       f"uncatalogued bytes up to {hx(stop)} close the body"],
                                       [f.row], [(f.name, f.vram, stop - f.vram)]))
                        break
                continue
        if f.terminated and f.padding and not keep_padding and \
                any(s <= f.end - f.padding and f.end <= e for s, e, _l in verified.overlapping(f.end - f.padding, f.end)):
            stats["padding_kept_by_verified_unit"] += 1      # the verified unit's text includes it
        elif f.terminated and f.padding and not keep_padding:
            body = f.size - f.padding
            ops.append(_op("shrink", 0.95, [f"{f.name}: {f.padding} byte(s) of zero words after the final "
                                            f"terminator at {hx(f.vram + body - 8)} (nop_roles: padding)"],
                           [f.row], [(f.name, f.vram, body)], [(f.vram + body, f.padding, "padding")]))
            continue
        if f.invalid and f.terminated:
            # data after the final terminator: everything from the first invalid word that follows it
            last_term = max((i.index for i in f.flow.insns if i.kind in ("return", "jump", "jr")), default=-1)
            tail = [v for v in f.invalid if v > f.vram + 4 * (last_term + 1)]
            if tail and last_term >= 0:
                body = 4 * (last_term + 2)
                if body < f.size:
                    ops.append(_op("shrink", 0.8, [f"{f.name}: word(s) after the final terminator at "
                                                   f"{hx(f.vram + 4 * last_term)} do not decode (first {hx(tail[0])})"],
                                   [f.row], [(f.name, f.vram, body)], [(f.vram + body, f.size - body, "data")]))

    # --- missing entries: jal targets with no containing row
    for target in sorted(calls):
        if target & 3 or target in by_start:
            continue
        k = bisect.bisect_right(starts, target) - 1
        if k >= 0 and facts[k].vram <= target < facts[k].end:
            continue
        stop = min(starts[k + 1] if k + 1 < len(starts) else hi, target + 0x10000)
        if stop - target < 8:
            continue
        trial = image.flow(target, stop - target)
        term, pad = _terminated(trial)
        size = stop - target - (pad if not keep_padding else 0)
        if size < 8:
            continue
        trial = image.flow(target, size)
        ev = flowmod.entry_evidence(trial)
        good = term and not _closed(trial) and not [i for i in trial.insns if i.op == ".word" and i.word]
        released = [(target + size, stop - target - size, "padding")] if size < stop - target else []
        ops.append(_op("add", 0.85 if good and ev["complete"] else 0.6,
                       [f"jal target {hx(target)} from {', '.join(hx(s) for s in calls[target][:3])} has no row",
                        "closed body up to the next row" if good else "body is not closed: " + "; ".join(_closed(trial)[:2])],
                       [], [(name(target), target, size)], released))

    # --- blocking against verified intervals and affects
    for n, op in enumerate(ops, 1):
        op["id"] = f"op{n:04d}"
        touched = [_row_tuple(r) for r in op["remove"] + op["insert"]]
        if "target" in op:
            touched.append(_row_tuple(op["target"]))
        spans = [(v, v + s) for _n, v, s in touched] + \
                [(int(r["vram"], 16), int(r["vram"], 16) + int(r["size"], 16)) for r in op["released"]]
        hits = []
        for a, b in spans:
            for s, e, label in verified.overlapping(a, b):
                if (s, e, label) not in hits:
                    hits.append((s, e, label))
        # an unchanged row equal to a verified interval is not an alteration
        op["affects"] = {
            "functions": len(op["insert"]) - len(op["remove"]),
            "code_bytes": sum(int(r["size"], 16) for r in op["insert"]) - sum(int(r["size"], 16) for r in op["remove"]),
            "manifest_rows": [f"{label} {hx(s)}+{hx(e - s)}" for s, e, label in hits],
        }
        if hits:
            op["status"] = "blocked"
            op["blocked_by"] = op["affects"]["manifest_rows"][:4]
    return ops


# ---------------------------------------------------------------------------------------------
# apply / invert


def apply(catalogue: list[Row], operations: list[dict], min_confidence: float = 0.8):
    """Returns (new_catalogue, applied, skipped).  An operation is skipped when it is blocked,
    below ``min_confidence``, when a row it removes is no longer present exactly, or when a
    row it inserts would overlap a row that stays."""
    rows = set(catalogue)
    applied, skipped = [], []
    for op in operations:
        reason = None
        remove = [_row_tuple(r) for r in op.get("remove", [])]
        insert = [_row_tuple(r) for r in op.get("insert", [])]
        if op.get("status") == "blocked":
            reason = "blocked by verified interval"
        elif op.get("confidence", 0) < min_confidence:
            reason = f"confidence {op.get('confidence')} < {min_confidence}"
        elif any(r not in rows for r in remove):
            reason = "stale: a row to remove is not in the catalogue"
        else:
            rest = rows - set(remove)
            for _n, v, s in insert:
                if any(v < rv + rs and rv < v + s for _rn, rv, rs in rest):
                    reason = f"conflict: inserted row at {hx(v)} overlaps a remaining row"
                    break
        if reason:
            skipped.append({**op, "skip_reason": reason})
            continue
        rows = (rows - set(remove)) | set(insert)
        applied.append(op)
    return sorted(rows, key=lambda r: (r[1], r[0])), applied, skipped


def apply_certain(catalogue: list[Row], operations: list[dict]):
    """``apply`` restricted to the operations the solver marked ``certain``: the ones whose evidence leaves no
    other reading of the bytes (see ``unreachable_epilogue`` and the stray leading word).  Returns
    (new_catalogue, applied).  Everything else stays a proposal for a reader."""
    rows, applied, _skipped = apply(catalogue, [op for op in operations if op.get("certain")], 0.0)
    return rows, applied


def invert(operations: list[dict]) -> list[dict]:
    """The exact inverse: reversed order, rows removed and inserted swapped, released bytes
    reclaimed.  Inverse operations are unconditional (confidence 1, never blocked)."""
    out = []
    for op in reversed(operations):
        inv = {k: v for k, v in op.items() if k not in ("remove", "insert", "released", "status",
                                                      "blocked_by", "skip_reason")}
        inv.update({"id": f"{op.get('id', 'op')}-inverse", "inverse_of": op.get("id"), "confidence": 1.0,
                    "status": "proposed", "remove": op.get("insert", []), "insert": op.get("remove", []),
                    "reclaimed": op.get("released", []), "released": []})
        if "affects" in op:
            inv["affects"] = {**op["affects"], "functions": -op["affects"]["functions"],
                              "code_bytes": -op["affects"]["code_bytes"]}
        out.append(inv)
    return out


def summary(catalogue: list[Row], operations: list[dict], new_catalogue: list[Row] | None = None,
            min_confidence: float = 0.8) -> dict:
    by_kind: dict[str, int] = {}
    for op in operations:
        if op.get("confidence", 0) >= min_confidence:
            key = op["kind"] + (f":{op['to']}" if op.get("to") else "") + \
                (" (blocked)" if op.get("status") == "blocked" else "")
            by_kind[key] = by_kind.get(key, 0) + 1
    out = {"rows_before": len(catalogue), "code_bytes_before": sum(s for _n, _v, s in catalogue),
           "operations": len(operations), f"at_or_above_{min_confidence}": dict(sorted(by_kind.items())),
           "blocked": sum(1 for op in operations if op.get("status") == "blocked")}
    if new_catalogue is not None:
        out["rows_after"] = len(new_catalogue)
        out["code_bytes_after"] = sum(s for _n, _v, s in new_catalogue)
    return out


# ---------------------------------------------------------------------------------------------
# catalogue files

_TOML_LINE = re.compile(r'^(\s*)\{\s*name\s*=\s*"([^"]+)"\s*,\s*vram\s*=\s*(0x[0-9A-Fa-f]+)\s*,\s*size\s*=\s*'
                        r'(0x[0-9A-Fa-f]+)\s*\}\s*,?(.*)$')
_SYM_LINE = re.compile(r'^\s*(\w+)\s*=\s*(0x[0-9A-Fa-f]+)\s*;(.*type:func.*)$')
_SIZE_ATTR = re.compile(r'size:(0x[0-9A-Fa-f]+)')
_HEX = re.compile(r'^(0x)?[0-9A-Fa-f]+$')


def _tsv_columns(lines: list[str]) -> dict[str, int] | None:
    for line in lines:
        cells = [c.strip().upper() for c in line.lstrip("#").strip().split("\t")]
        if len(cells) < 3:
            cells = line.lstrip("#").upper().split()
        if {"VRAM", "SIZE", "NAME"} <= set(cells):
            return {c: i for i, c in enumerate(cells)}
    return None


def catalogue_format(path) -> str:
    text = Path(path).read_text()
    lines = text.splitlines()
    if any(_TOML_LINE.match(l) for l in lines):
        return "toml"
    if re.search(r"(?m)^\s*functions\s*=\s*\[\s*\]", text) or re.search(r"(?m)^\s*functions\s*=\s*\[\s*$", text) \
            and not re.search(r"(?m)^\s*\{", text):
        return "toml"                # a TOML function table that lists no function: an empty catalogue
    if any(_SYM_LINE.match(l) for l in lines):
        return "symbol_addrs"
    if _tsv_columns(lines[:20]):
        return "tsv"
    raise ValueError(f"{path}: not a recognised catalogue (TOML function table, symbol_addrs.txt or a TSV with "
                     "VRAM, SIZE and NAME columns)")


def _parse_lines(path) -> tuple[str, list[str], list[tuple[int, Row]], dict]:
    fmt = catalogue_format(path)
    lines = Path(path).read_text().splitlines()
    entries, meta = [], {}
    if fmt == "tsv":
        cols = _tsv_columns(lines[:20])
        meta["columns"] = cols
    for k, line in enumerate(lines):
        if fmt == "toml":
            m = _TOML_LINE.match(line)
            if m:
                entries.append((k, (m.group(2), int(m.group(3), 16), int(m.group(4), 16))))
        elif fmt == "symbol_addrs":
            m = _SYM_LINE.match(line)
            if m:
                size = _SIZE_ATTR.search(m.group(3))
                entries.append((k, (m.group(1), int(m.group(2), 16), int(size.group(1), 16) if size else 0)))
        else:
            if line.startswith("#") or not line.strip():
                continue
            cells = line.split("\t") if "\t" in line else line.split()
            try:
                row = (cells[meta["columns"]["NAME"]], int(cells[meta["columns"]["VRAM"]], 16),
                       int(cells[meta["columns"]["SIZE"]], 16))
            except (IndexError, ValueError):
                continue
            entries.append((k, row))
    return fmt, lines, entries, meta


def read_catalogue(path) -> list[Row]:
    """(name, vram, size) sorted by address, from a TOML function table, a ``symbol_addrs.txt`` or
    a TSV with VRAM, SIZE and NAME columns."""
    _fmt, _lines, entries, _meta = _parse_lines(path)
    return sorted({row for _k, row in entries}, key=lambda r: (r[1], r[0]))


def write_catalogue(path, rows: list[Row], like) -> Path:
    """Write ``rows`` to the NEW file ``path`` in the format of ``like`` (the original catalogue).
    Unchanged rows keep their original line (comments included); new rows are written in the
    original style next to the original row they replace."""
    path, like = Path(path), Path(like)
    if path.resolve() == like.resolve():
        raise ValueError("refusing to overwrite the input catalogue; give a new path")
    fmt, lines, entries, meta = _parse_lines(like)
    original = {row: k for k, row in entries}
    entry_lines = sorted(k for k, _r in entries)
    order = sorted(entries, key=lambda e: e[1][1])
    anchors = [row[1] for _k, row in order]
    lower = bool(re.search(r"vram\s*=\s*0x[0-9a-f]*[a-f]", "\n".join(lines))) if fmt == "toml" else False
    sample = lines[entry_lines[0]] if entry_lines else ""

    def render(row: Row) -> str:
        n, v, s = row
        if fmt == "toml":
            indent = _TOML_LINE.match(sample).group(1) if sample and _TOML_LINE.match(sample) else "    "
            vh = f"{v:08x}" if lower else f"{v:08X}"
            sh = f"{s:x}" if lower else f"{s:X}"
            return f'{indent}{{ name = "{n}", vram = 0x{vh}, size = 0x{sh} }},'
        if fmt == "symbol_addrs":
            return f"{n} = 0x{v:08X}; // type:func size:0x{s:X}"
        cols = meta["columns"]
        cells = ["-"] * (max(cols.values()) + 1)
        ref = order[max(0, bisect.bisect_right(anchors, v) - 1)][1] if order else None
        cells[cols["NAME"]], cells[cols["VRAM"]], cells[cols["SIZE"]] = n, f"0x{v:08X}", f"0x{s:X}"
        for key in cols:
            if key.startswith("ROM") and ref is not None:
                ref_line = lines[original[ref]]
                ref_cells = ref_line.split("\t") if "\t" in ref_line else ref_line.split()
                try:
                    rom = int(ref_cells[cols[key]], 16) + (v - ref[1])
                    cells[cols[key]] = f"0x{rom:X}"
                except (IndexError, ValueError):
                    pass
            elif key not in ("NAME", "VRAM", "SIZE"):
                cells[cols[key]] = "boundary-solve"
        return "\t".join(cells)

    placed: dict[int, list[str]] = {}
    for row in sorted(rows, key=lambda r: (r[1], r[0])):
        if row in original:
            slot = original[row]
            text = lines[slot]
        else:
            k = bisect.bisect_right(anchors, row[1]) - 1
            slot = order[max(k, 0)][0] if order else (len(lines) - 1)
            text = render(row)
        placed.setdefault(slot, []).append((row[1], text))
    out = []
    entry_set = set(entry_lines)
    for k, line in enumerate(lines):
        if k not in entry_set:
            out.append(line)
        for _v, text in sorted(placed.get(k, []), key=lambda t: t[0]):
            out.append(text)
    if not entry_lines:
        out += [t for k in sorted(placed) for _v, t in placed[k]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    return path


def dump_operations(path, operations: list[dict], **header) -> None:
    Path(path).write_text(json.dumps({**header, "operations": operations}, indent=1) + "\n")


def load_operations(path) -> tuple[list[dict], dict]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return data, {}
    return data["operations"], {k: v for k, v in data.items() if k != "operations"}


def shift_rows(rows: list[Row], shift: int) -> list[Row]:
    return [(n, v + shift, s) for n, v, s in rows]


def shift_operations(operations: list[dict], shift: int) -> list[dict]:
    """The same operations with every address moved by ``shift`` (image <-> catalogue space)."""
    def mv(item):
        return {**item, "vram": hx(int(item["vram"], 16) + shift)}
    out = []
    for op in operations:
        new = dict(op)
        for key in ("remove", "insert", "released", "reclaimed"):
            if key in op:
                new[key] = [mv(r) for r in op[key]]
        if "target" in op:
            new["target"] = mv(op["target"])
        if "at" in op:
            new["at"] = [hx(int(a, 16) + shift) for a in op["at"]]
        out.append(new)
    return out
