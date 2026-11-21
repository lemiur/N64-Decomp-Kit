"""Data, ``.rodata`` and BSS solver (report only): partition one image's non-code bytes into data objects.

For one linear image (``data`` with file offset = address - ``delta``, catalogue ``rows`` of
``(name, vram, size)``) the bytes no catalogued function covers are cut into objects at addresses
the kit has evidence for, and each object gets a kind, its evidence, the functions that reference
it and, when exactly one function does, an owner.  Evidence sources (each can be switched off):

``hilo``
    ``lui``/``%lo`` pairs in catalogued functions (:func:`n64dk.mips.scan_refs`): loads, stores and
    address-taken uses.  The access width is a lower bound on the object's size; ``lwc1``/``ldc1``
    make ``f32``/``f64`` literals, and two ``lwc1`` of the halves of one 8-aligned double into an
    odd/even register pair (the o32 ``-mfp32`` form) make one ``f64``.
``gprel``
    loads, stores and ``addiu`` relative to ``$gp`` (small data), with ``_gp`` either given or set
    in this image's own code (:func:`n64dk.analysis.gp.scan`, all setting sites agreeing).
``jtbl``
    switch tables from :mod:`n64dk.analysis.indirect` (:func:`tables_from_indirect`): a definite,
    bounded table's size is its entry count; an unbounded one's is a lower bound.  Without them, a
    run of words at a referenced start that all point into the referencing function is a
    ``jtbl-shape`` (its run is clipped at the next evidenced start).
``pointer``
    aligned words outside code whose value is an address in the image's data or BSS (the ROM-side
    analogue of an ``R_MIPS_32`` relocation target): the target is an object start referenced from
    data, and runs of such words are pointer arrays.
``string``
    a NUL-terminated ASCII or Shift-JIS string at an evidenced start that is only address-taken or
    read bytewise (:func:`n64dk.analysis.datamap.string_at`); its length is the evidence span.
``bss``
    references past the last initialised byte: into ``bss`` when the caller gives it (the boot
    segment's clear bounds, :mod:`n64dk.boot`), else into the gap after the image that references
    reach without a hole larger than ``bss_gap`` (an inference, recorded as such).

Every row is a hypothesis.  ``size`` is the extent to the next evidenced start (or the region end):
the object cannot be larger unless that start is an interior reference (``&s.field``, which
compilers emit as ``%lo(sym+k)``), and it may be smaller (padding, unreferenced neighbours).
``min_size`` is what the evidence itself covers.  Byte ownership is summarised as **owned** (inside
exactly one row's evidence span), **conflicting** (one row's evidence runs past the next evidenced
start, or into code or past the region) and **unowned** (split into ``adjacent``: in a row's extent
beyond its evidence; ``unclaimed``: before the region's first evidenced start; and the zero bytes
among both).  Nothing here is verified, nothing grants credit, and there is no write path into a
catalogue, a manifest or the byte ledger.
"""
from __future__ import annotations

import bisect
import struct
from collections import Counter
from dataclasses import dataclass, field, replace

from .. import mips
from . import gp as gpmod
from .datamap import string_at

MASK = 0xFFFFFFFF
GP = 28
SOURCES = ("hilo", "gprel", "jtbl", "pointer", "string", "bss")
BSS_GAP = 0x4000                  # largest hole between BSS references when the BSS end is inferred
BSS_LIMIT = 0x400000              # never infer more BSS than this past the image
CLAIM_NOTE = ("hypothesis: objects are cut at evidenced starts; size is the extent to the next start "
              "(an upper bound unless that start is an interior reference); min_size is what the evidence "
              "covers; nothing here is verified or grants credit")

_ACCESS = {
    "lwc1": "f32", "swc1": "f32", "ldc1": "f64", "sdc1": "f64",
    "lw": "word", "sw": "word", "lwu": "word", "ll": "word", "sc": "word",
    "lwl": "byte", "lwr": "byte", "swl": "byte", "swr": "byte",
    "lh": "half", "lhu": "half", "sh": "half", "lb": "byte", "lbu": "byte", "sb": "byte",
    "ld": "dword", "sd": "dword", "ldl": "byte", "ldr": "byte", "sdl": "byte", "sdr": "byte",
}
_WIDTH = {"f32": 4, "f64": 8, "word": 4, "half": 2, "byte": 1, "dword": 8, "addr": 1}
# I-type opcodes relative to $gp: opcode -> (access, kind)
_GP_OPS = {9: ("addr", "addr"), 32: ("byte", "load"), 33: ("half", "load"), 35: ("word", "load"),
           36: ("byte", "load"), 37: ("half", "load"), 39: ("word", "load"), 40: ("byte", "store"),
           41: ("half", "store"), 43: ("word", "store"), 49: ("f32", "load"), 53: ("f64", "load"),
           55: ("dword", "load"), 57: ("f32", "store"), 61: ("f64", "store"), 63: ("dword", "store")}
_RODATA_KINDS = {"jtbl", "jtbl-shape", "f32", "f64", "string", "sjis"}


def hx(value: int) -> str:
    return f"0x{value & MASK:08X}"


def _simm(word: int) -> int:
    imm = word & 0xFFFF
    return imm - 0x10000 if imm & 0x8000 else imm


@dataclass(frozen=True)
class Claim:
    """One piece of evidence that an object starts at ``address``."""
    address: int
    source: str                    # hilo, hilo-pair, gprel, jtbl, pointer
    access: str                    # f32 f64 word half byte dword addr table pointer
    kind: str                      # load store addr table pointer
    width: int                     # bytes the evidence covers from address (a lower bound unless exact)
    function: int | None = None    # referencing function (code evidence only)
    site: int | None = None        # instruction, or data word holding the pointer
    exact: bool = False            # width is the whole object (a bounded switch table)
    reg: int = -1                  # rt/ft of the instruction (pairs the halves of an o32 double)


@dataclass(frozen=True)
class Table:
    """A jump table found elsewhere (:func:`tables_from_indirect`)."""
    address: int
    size: int
    function: int
    exact: bool
    basis: str


@dataclass
class Row:
    start: int
    size: int                      # extent to the next evidenced start or the region end
    min_size: int                  # bytes the evidence covers from start
    kind: str
    region: str                    # data or bss
    evidence: tuple[str, ...]
    functions: tuple[int, ...]
    data_refs: int                 # pointer words in data that hold this address
    written: bool
    section_hint: str              # .data, .rodata?, .bss or -
    conflicts: list[str] = field(default_factory=list)

    @property
    def owner_function(self) -> int | None:
        return self.functions[0] if len(self.functions) == 1 else None

    def as_dict(self) -> dict:
        return {"start": hx(self.start), "size": hex(self.size), "min_size": hex(self.min_size), "kind": self.kind,
                "region": self.region, "evidence": list(self.evidence),
                "owner_function": hx(self.owner_function) if self.owner_function is not None else None,
                "functions": len(self.functions), "data_refs": self.data_refs, "written": self.written,
                "section_hint": self.section_hint, "conflicts": list(self.conflicts)}


@dataclass
class Evidence:
    """Everything :func:`collect` found in one image; :func:`partition` turns it into rows."""
    data: bytes
    delta: int
    extent: tuple[int, int]
    regions: list[tuple[int, int]]          # initialised non-code ranges
    code: list[tuple[int, int]]             # merged catalogued code spans inside the extent
    functions: dict[int, int]               # function start -> end
    bss: tuple[int, int] | None
    bss_basis: str
    gp: dict
    claims: list[Claim]
    pointer_words: set[int]                 # data-word addresses holding an address in the image or its BSS


@dataclass
class Solution:
    extent: tuple[int, int]
    bss: tuple[int, int] | None
    bss_basis: str
    regions: list[tuple[int, int]]
    rows: list[Row]
    conflicts: list[dict]
    code_overlap: list[dict]
    external: dict
    gp: dict
    sources: tuple[str, ...]
    summary: dict

    def as_dict(self, limit: int = 50) -> dict:
        return {"claim": CLAIM_NOTE, "extent": [hx(a) for a in self.extent],
                "bss": [hx(a) for a in self.bss] if self.bss else None, "bss_basis": self.bss_basis,
                "sources": list(self.sources), "gp": self.gp, "summary": self.summary,
                "conflicts": self.conflicts[:limit], "code_overlap": self.code_overlap[:limit],
                "external_references": self.external}

    def to_tsv(self) -> str:
        lines = [f"# {CLAIM_NOTE}",
                 "start\tsize\tmin_size\tkind\tregion\tsection_hint\towner_function\tfunctions\tdata_refs\tevidence\tconflicts"]
        for r in self.rows:
            owner = hx(r.owner_function) if r.owner_function is not None else ("shared" if r.functions else "-")
            lines.append("\t".join([hx(r.start), f"0x{r.size:X}", f"0x{r.min_size:X}", r.kind, r.region, r.section_hint,
                                    owner, str(len(r.functions)), str(r.data_refs), ",".join(r.evidence) or "-",
                                    ";".join(r.conflicts) or "-"]))
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- intervals

def _union(spans) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in sorted(s for s in spans if s[1] > s[0]):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _subtract(spans, minus) -> list[tuple[int, int]]:
    out, minus = [], _union(minus)
    firsts = [m[0] for m in minus]
    for a, b in _union(spans):
        cur = a
        k = max(bisect.bisect_right(firsts, a) - 1, 0)
        for x, y in minus[k:]:
            if x >= b:
                break
            if y <= cur:
                continue
            if x > cur:
                out.append((cur, x))
            cur = max(cur, y)
        if cur < b:
            out.append((cur, b))
    return out


def _measure(spans) -> int:
    return sum(b - a for a, b in _union(spans))


def _find(spans: list[tuple[int, int]], address: int) -> int:
    """Index of the span holding address, or -1 (spans sorted and disjoint)."""
    i = bisect.bisect_right(spans, (address, MASK + 1)) - 1
    return i if i >= 0 and spans[i][0] <= address < spans[i][1] else -1


# --------------------------------------------------------------------------- evidence

def code_rows(data: bytes, delta: int, rows, extent: tuple[int, int]) -> list[tuple[int, int]]:
    """``(vram, size)`` of catalogued functions that start inside the extent and lie in the data."""
    lo, hi = extent
    return sorted({(v, s) for _n, v, s in rows
                   if s > 0 and lo <= v < hi and 0 <= v - delta and v - delta + s <= len(data)})


def hilo_claims(data: bytes, delta: int, functions: list[tuple[int, int]]) -> list[Claim]:
    out = []
    for vram, size in functions:
        code = data[vram - delta:vram - delta + size]
        for ref in mips.scan_refs(code, vram):
            if ref.kind not in ("load", "store", "addr"):
                continue
            word = struct.unpack_from(">I", data, ref.site - delta)[0]
            reg = (word >> 16) & 31
            if ref.kind == "addr" and reg == GP:
                continue                                    # _gp itself being set, not an object
            access = "addr" if ref.kind == "addr" else _ACCESS.get(ref.opcode, "word")
            out.append(Claim(ref.address, "hilo", access, ref.kind, _WIDTH[access], vram, ref.site, reg=reg))
    return out


def find_gp(data: bytes, delta: int, rows) -> dict:
    """``_gp`` from this image's own setting sites (all agreeing), else a reason."""
    found = gpmod.scan(data, delta, rows)
    values = set(found["sites"].values())
    record = {"value": None, "basis": "", "sites": len(found["sites"]), "uses": found["uses"]}
    if len(values) == 1:
        value = values.pop()
        if value and gpmod.plausible(value):
            record["value"] = value
            record["basis"] = f"set in this image at {record['sites']} site(s)"
            return record
        record["basis"] = f"set to {hx(value)}: cleared or outside RDRAM, no small data"
    elif values:
        record["basis"] = "setting sites disagree: " + ", ".join(hx(v) for v in sorted(values))
    else:
        record["basis"] = "not set in this image" + (" although it is used: pass gp=" if found["uses"] else "")
    return record


def gprel_claims(data: bytes, delta: int, functions: list[tuple[int, int]], gp: int) -> list[Claim]:
    out = []
    for vram, size in functions:
        start = vram - delta
        for offset in range(start, start + size - 3, 4):
            word = struct.unpack_from(">I", data, offset)[0]
            op, rs, rt = word >> 26, (word >> 21) & 31, (word >> 16) & 31
            if rs != GP or op not in _GP_OPS or (op == 9 and rt == GP):
                continue
            access, kind = _GP_OPS[op]
            out.append(Claim((gp + _simm(word)) & MASK, "gprel", access, kind, _WIDTH[access], vram,
                             vram + offset - start, reg=rt))
    return out


def tables_from_indirect(result) -> list[Table]:
    """Switch tables of an :class:`n64dk.analysis.indirect.ImageResult`: definite ones with an exact
    size, unbounded ones with a lower bound.  Pointer tables are not used (their table address is a
    structure base, not necessarily an object start)."""
    out = []
    for site in result.sites:
        if site.table is None or site.count <= 0:
            continue
        if site.kind == "switch" and site.status == "definite":
            out.append(Table(site.table, 4 * site.count, site.function, True, "switch-definite"))
        elif site.kind == "switch-unbounded":
            out.append(Table(site.table, 4 * site.count, site.function, False, "switch-unbounded"))
    return out


def infer_bss(claims: list[Claim], start: int, gap: int = BSS_GAP, limit: int = BSS_LIMIT) -> tuple[int, int] | None:
    """[start, end) reached by code references at or past ``start`` with no hole over ``gap``."""
    end = None
    for claim in sorted((c for c in claims if start <= c.address < start + limit), key=lambda c: c.address):
        if claim.address - (end if end is not None else start) > gap:
            break
        end = max(end or start, claim.address + max(claim.width, 1))
    return (start, (end + 15) & ~15) if end is not None else None


def collect(data: bytes, delta: int, rows, *, extent: tuple[int, int] | None = None,
            bss: tuple[int, int] | None = None, gp: int | None = None, tables=(),
            bss_gap: int = BSS_GAP) -> Evidence:
    """Gather every source's claims once (``partition`` may then drop sources, e.g. for ablations).

    ``extent``: the initialised image in address space (default: the whole file); ``bss``: the BSS
    bounds when known (else inferred); ``gp``: ``_gp`` when known (else looked for in this image);
    ``tables``: :class:`Table` rows (e.g. :func:`tables_from_indirect`)."""
    lo, hi = extent or (delta, delta + len(data))
    lo, hi = max(lo, delta), min(hi, delta + len(data))
    functions = code_rows(data, delta, rows, (lo, hi))
    code = _union((v, min(v + s, hi)) for v, s in functions)
    regions = _subtract([(lo, hi)], code)
    claims = hilo_claims(data, delta, functions)
    if gp is None:
        gp_record = find_gp(data, delta, [("", v, s) for v, s in functions])
    else:
        gp_record = {"value": gp, "basis": "given", "sites": None, "uses": None}
    if gp_record["value"] is not None:
        claims += gprel_claims(data, delta, functions, gp_record["value"])
    gp_record = {**gp_record, "value": hx(gp_record["value"]) if gp_record["value"] is not None else None}
    for t in tables:
        claims.append(Claim(t.address, "jtbl", "table", "table", t.size, t.function, None, t.exact))
    if bss is not None:
        basis = "given"
    else:
        bss = infer_bss([c for c in claims if c.source in ("hilo", "gprel")], hi, bss_gap)
        basis = (f"inferred: code references past the image with no hole over 0x{bss_gap:X} bytes"
                 if bss else "none: no code reference past the image")
    top = bss[1] if bss else hi
    pointer_words: set[int] = set()
    for a, b in regions:
        for p in range((a + 3) & ~3, b - 3, 4):
            word = struct.unpack_from(">I", data, p - delta)[0]
            if lo <= word < top:
                pointer_words.add(p)
                if _find(code, word) < 0:
                    claims.append(Claim(word, "pointer", "pointer", "pointer", 1, None, p))
    return Evidence(data, delta, (lo, hi), regions, code, {v: v + s for v, s in functions}, bss, basis,
                    gp_record, claims, pointer_words)


# --------------------------------------------------------------------------- rows

def _pair_doubles(by_addr: dict[int, list[Claim]]) -> None:
    """``lwc1 $f(n+1), X; lwc1 $fn, X+4`` (n even, X 8-aligned, one function): one f64 at X."""
    for x in sorted(a for a in by_addr if a % 8 == 0):
        upper = by_addr.get(x + 4)
        if not upper:
            continue
        firsts = {(c.function, c.reg) for c in by_addr[x] if c.access == "f32" and c.kind == "load"}
        moved = [c for c in upper if c.access == "f32" and c.kind == "load" and c.reg % 2 == 0
                 and (c.function, c.reg + 1) in firsts]
        if not moved:
            continue
        partners = {(m.function, m.reg + 1) for m in moved}
        by_addr[x] = [replace(c, source="hilo-pair", access="f64", width=8)
                      if c.access == "f32" and c.kind == "load" and (c.function, c.reg) in partners else c
                      for c in by_addr[x]]
        by_addr[x] += [replace(m, address=x, source="hilo-pair", access="f64", width=8) for m in moved]
        rest = [c for c in upper if c not in moved]
        if rest:
            by_addr[x + 4] = rest
        else:
            del by_addr[x + 4]


def _run(ev: Evidence, start: int, end: int, test) -> int:
    """Bytes of consecutive aligned words from start (< end) that pass ``test(address, word)``."""
    if start % 4:
        return 0
    p = start
    while p + 4 <= end and test(p, struct.unpack_from(">I", ev.data, p - ev.delta)[0]):
        p += 4
    return p - start


def _describe(ev: Evidence, start: int, end: int, region_end: int, region: str, claims: list[Claim],
              strings: bool = True) -> Row:
    code_claims = [c for c in claims if c.function is not None]
    functions = tuple(sorted({c.function for c in code_claims}))
    accesses = {c.access for c in claims if c.source != "pointer"}
    tags = Counter(f"{c.source}:{c.access}" for c in claims)
    evidence = [f"{t}*{n}" if n > 1 else t for t, n in sorted(tags.items())]
    written = any(c.kind == "store" for c in claims)
    width = max(c.width for c in claims)
    data_refs = sum(1 for c in claims if c.source == "pointer")
    kind = None
    if region == "bss":
        kind = "bss"
    elif any(c.source == "jtbl" for c in claims):
        kind = "jtbl"
        width = max(c.width for c in claims if c.source == "jtbl")
        if any(c.exact for c in claims if c.source == "jtbl"):
            evidence.append("exact-count")
    if kind is None and strings and accesses <= {"addr", "byte"}:
        # a start code takes the address of may hold a short string ("%d\n"); a start only data points
        # at needs the usual four characters
        shortest = 1 if any(c.access == "addr" for c in code_claims) else 4
        found = string_at(ev.data, start - ev.delta, region_end - ev.delta, shortest)
        if found:
            kind = found[1]
            width = found[0] - (start - ev.delta)
            evidence.append(f"shape:{kind}")
    loads = accesses - {"addr"}
    if kind is None and loads == {"f32"}:
        kind = "f32"
    elif kind is None and loads == {"f64"}:
        kind = "f64"
    if kind is None and code_claims and all(c.access in ("word", "addr") for c in code_claims):
        spans = [(f, ev.functions[f]) for f in functions if f in ev.functions]
        run = _run(ev, start, end, lambda _p, w: any(a <= w < b for a, b in spans))
        if run >= 8:
            kind, width = "jtbl-shape", max(width, run)
            evidence.append("shape:jtbl-run")
    if kind is None:
        run = _run(ev, start, end, lambda p, _w: p in ev.pointer_words)
        if run >= 8:
            kind, width = "pointers", max(width, run)
            evidence.append("shape:pointer-run")
    if kind is None:
        if loads:
            kind = "object"
        elif accesses:
            kind = "address-taken"
        else:
            kind = "pointer-target"
    hint = (".bss" if region == "bss" else ".data" if written
            else ".rodata?" if kind in _RODATA_KINDS else "-")
    return Row(start, end - start, width, kind, region, tuple(evidence), functions, data_refs, written, hint)


def partition(ev: Evidence, sources=SOURCES) -> Solution:
    """Rows, conflicts and the byte-ownership summary from the claims of the chosen sources."""
    sources = tuple(s for s in SOURCES if s in sources)
    allowed = set(sources) & {"hilo", "gprel", "jtbl", "pointer"}
    if "hilo" in allowed:
        allowed.add("hilo-pair")
    regions = ev.regions
    bss = ev.bss if "bss" in sources else None
    by_addr: dict[int, list[Claim]] = {}
    code_overlap, external = [], Counter()
    for c in ev.claims:
        if c.source not in allowed:
            continue
        if _find(regions, c.address) >= 0 or (bss and bss[0] <= c.address < bss[1]):
            by_addr.setdefault(c.address, []).append(c)
        elif _find(ev.code, c.address) >= 0:
            if c.kind != "addr":                     # an address-taken code address is a function pointer
                code_overlap.append({"address": hx(c.address), "source": c.source, "access": c.access,
                                     "function": hx(c.function) if c.function is not None else None})
        else:
            external[c.source] += 1
    _pair_doubles(by_addr)
    rows: list[Row] = []
    conflicts: list[dict] = []
    groups = [(a, b, "data") for a, b in regions] + ([(bss[0], bss[1], "bss")] if bss else [])
    for a, b, region in groups:
        starts = sorted(x for x in by_addr if a <= x < b)
        for i, s in enumerate(starts):
            e = starts[i + 1] if i + 1 < len(starts) else b
            row = _describe(ev, s, e, b, region, by_addr[s], "string" in sources)
            reach = s + row.min_size
            if reach > e:
                cut = min(reach, b)
                if cut > e:
                    what = {"jtbl": "table-overlap", "string": "string-overlap", "sjis": "string-overlap"}.get(
                        row.kind, "width-overlap")
                    conflicts.append({"type": what, "start": hx(e), "end": hx(cut), "row": hx(s),
                                      "next": [hx(x) for x in starts[i + 1:] if x < cut][:4]})
                    row.conflicts.append(f"{what} {hx(e)}..{hx(cut)}")
                if reach > b:
                    conflicts.append({"type": "region-crossing", "start": hx(b), "end": hx(reach), "row": hx(s),
                                      "next": []})
                    row.conflicts.append(f"region-crossing {hx(b)}..{hx(reach)}")
            rows.append(row)
    summary = _summary(ev, rows, regions, bss, conflicts, code_overlap, external)
    return Solution(ev.extent, bss, ev.bss_basis if bss else "not used", regions, rows, conflicts, code_overlap,
                    dict(external), ev.gp, sources, summary)


def solve(data: bytes, delta: int, rows, *, sources=SOURCES, **kw) -> Solution:
    """:func:`collect` then :func:`partition`."""
    return partition(collect(data, delta, rows, **kw), sources)


# --------------------------------------------------------------------------- summary and scoring

def _summary(ev: Evidence, rows: list[Row], regions, bss, conflicts, code_overlap, external) -> dict:
    data_rows = [r for r in rows if r.region == "data"]
    total = sum(b - a for a, b in regions)
    ends = [b for _a, b in regions]

    def clip(r: Row) -> tuple[int, int]:
        b = ends[_find(regions, r.start)]
        return r.start, min(r.start + r.min_size, r.start + r.size, b)

    conflict_spans = _union((int(c["start"], 16), int(c["end"], 16)) for c in conflicts
                            if c["type"] != "region-crossing" and _find(regions, int(c["start"], 16)) >= 0)
    evidenced = _union(clip(r) for r in data_rows)
    owned = _subtract(evidenced, conflict_spans)
    unowned = _subtract(regions, evidenced + conflict_spans)
    firsts = {}
    for r in data_rows:
        k = _find(regions, r.start)
        firsts[k] = min(firsts.get(k, r.start), r.start)
    adjacent = unclaimed = zeros = 0
    for a, b in unowned:
        k = _find(regions, a)
        if k in firsts and a >= firsts[k]:
            adjacent += b - a
        else:
            unclaimed += b - a
        zeros += ev.data[a - ev.delta:b - ev.delta].count(0)
    bss_rows = [r for r in rows if r.region == "bss"]
    kinds = Counter(r.kind for r in rows)
    out = {
        "non_code_bytes": total,
        "owned": _measure(owned),
        "conflicting": _measure(conflict_spans),
        "unowned": _measure(unowned),
        "unowned_adjacent": adjacent,
        "unowned_unclaimed": unclaimed,
        "unowned_zero_bytes": zeros,
        "rows": len(data_rows),
        "bss_rows": len(bss_rows),
        "bss_bytes": (bss[1] - bss[0]) if bss else 0,
        "bss_evidenced": _measure((r.start, min(r.start + r.min_size, r.start + r.size)) for r in bss_rows),
        "kinds": dict(sorted(kinds.items())),
        "single_owner_rows": sum(1 for r in rows if r.owner_function is not None),
        "shared_rows": sum(1 for r in rows if len(r.functions) > 1),
        "data_only_rows": sum(1 for r in rows if not r.functions and r.data_refs),
        "conflicts": dict(Counter(c["type"] for c in conflicts)),
        "code_overlap": len(code_overlap),
        "external_references": sum(external.values()),
        "exact": False,
    }
    return out


def compare(rows: list[Row], truth, lo: int | None = None, hi: int | None = None) -> dict:
    """Score rows against reference objects ``(start, size or None, label)`` inside [lo, hi).

    Start precision and recall; for matched starts, whether the row's size equals the reference's,
    whether the extent bounds it (``size >= truth``), and evidence overstatements (``min_size >
    truth``: the evidence claimed more bytes than the object has)."""
    def inside(a):
        return (lo is None or a >= lo) and (hi is None or a < hi)

    ref: dict[int, tuple] = {}
    for start, size, label in truth:
        if inside(start):
            ref.setdefault(start, (size, label))
    pred = {r.start: r for r in rows if inside(r.start)}
    hit = sorted(set(pred) & set(ref))
    sized = [a for a in hit if ref[a][0]]
    over = [a for a in sized if pred[a].min_size > ref[a][0]]
    kinds: dict[str, dict] = {}
    for a, r in pred.items():
        k = kinds.setdefault(r.kind, {"predicted": 0, "true_start": 0})
        k["predicted"] += 1
        k["true_start"] += a in ref
    labels: dict[str, dict] = {}
    for a, (_size, label) in ref.items():
        k = labels.setdefault(label, {"truth": 0, "found": 0})
        k["truth"] += 1
        k["found"] += a in pred
    owners = sum(1 for a in hit if pred[a].owner_function is not None)
    return {
        "predicted": len(pred), "truth": len(ref), "true_starts": len(hit),
        "precision": round(len(hit) / len(pred), 4) if pred else None,
        "recall": round(len(hit) / len(ref), 4) if ref else None,
        "sized_matches": len(sized),
        "size_equal": sum(1 for a in sized if pred[a].size == ref[a][0]),
        "size_upper_bound_holds": sum(1 for a in sized if pred[a].size >= ref[a][0]),
        "min_size_overstated": len(over),
        "overstated_examples": [hx(a) for a in over[:20]],
        "matched_with_single_owner": owners,
        "by_kind": dict(sorted(kinds.items())),
        "by_truth_label": dict(sorted(labels.items())),
        "false_start_examples": [hx(a) for a in sorted(set(pred) - set(ref))[:20]],
        "missed_examples": [hx(a) for a in sorted(set(ref) - set(pred))[:20]],
    }
