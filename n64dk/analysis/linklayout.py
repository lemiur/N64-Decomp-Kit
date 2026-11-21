"""Build graph and link-order reconstruction (an inference: it grants no match credit).

A ROM was linked from objects in one order.  GNU ld (and the other linkers of the N64 toolchains)
lays out every output section by walking the input objects in that order: the ``.text`` of each
object in turn, then, in the next output section, their ``.data`` in the same order, and so on.
Each input section starts at its own alignment and the bytes before it are fill.  So the object
order is visible once in every output section an object contributes to, and the order of two
units' ``.text`` must equal the order of their ``.data``, ``.rodata`` and ``.bss`` wherever there is
evidence for both.

This module models that and solves for the order:

* **Build graph.**  Segments (ROM placement, RAM placement, codec, overlay group), units (kind
  ``c``, ``asm``, ``rsp``, ``asset`` or ``raw``) with the object each becomes and, for library code,
  its archive and member; pieces (one unit's contribution to one section: ``.text`` ``.data``
  ``.rodata`` ``.sdata`` ``.lit4`` ``.lit8`` ``.ctors`` ``.sbss`` ``.bss``), placed at an address or
  only known to exist with a size; linker symbols (segment bounds, ``_gp``, ``_end``).
* **Constraints, each with provenance and a confidence.**  ``before`` (partial order),
  ``adjacent`` (no other unit that contributes to that section lies between the two) and
  ``group`` (archive members lie together: ld takes an archive's extracted members in at the
  archive's place in the input list).  A constraint across segments is invalid (each segment is
  linked in its own order); one between mutually exclusive overlays is a contradiction.  Both
  are reported, never applied.
* **Derivation.**  The placed pieces of one section, in address order, give ``before`` between
  consecutive units.  Exact pieces whose gap is nothing or alignment fill give ``adjacent`` and a
  minimum alignment for the later piece; any other gap is a hole (zero fill, or unattributed
  content, which becomes a ``raw`` unit).  Pieces that overlap are reported.
* **Solver.**  A cycle is broken by setting aside its weakest step; the cycle and every
  constraint on it are reported.  Adjacency, grouping and size constraints are checked while the
  orders that remain are enumerated, all of them up to ``limit``; when none satisfies them all,
  the weakest is set aside and reported the same way.  Every retained ordering is kept: a
  position range per unit, the pairs whose order differs between orderings and a confidence for
  each step of the representative order.
* **Outputs.**  The graph as JSON, an object manifest and a GNU ld script for the representative
  order, with the unresolved alternatives and the contradictions as comments.

Evidence extractors turn what the kit already has into pieces and constraints: catalogue units
(:mod:`n64dk.analysis.units`), the data addresses each unit's code references (``%hi``/``%lo``
pairs, :func:`n64dk.mips.scan_refs`; an address only one unit references is taken as that unit's
data), SDK archive membership (the signature database records the library object of each
signature), source names that assertion strings leave (:mod:`n64dk.analysis.devrecords`) and
``_gp`` (:mod:`n64dk.analysis.gp`).

Nothing here verifies bytes.  The layout is a hypothesis: it never grants match credit, never
overrides byte evidence and never promotes a row.
"""
from __future__ import annotations

import bisect
import csv
import heapq
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SECTIONS = (".text", ".data", ".rodata", ".sdata", ".lit4", ".lit8", ".ctors", ".sbss", ".bss")
NOLOAD = frozenset((".sbss", ".bss"))
UNIT_KINDS = ("c", "asm", "rsp", "asset", "raw")
CONSTRAINT_KINDS = ("before", "adjacent", "group")
# How far an address that only one unit's code references says the address is that unit's data:
# read-only data (literals, jump tables, strings) is almost always private to its unit; writable
# data and especially .bss are often another unit's globals.
REFERENCE_CONFIDENCE = {".rodata": 0.8, ".lit4": 0.8, ".lit8": 0.8, ".data": 0.6, ".sdata": 0.6,
                        ".ctors": 0.6, ".bss": 0.5, ".sbss": 0.5}
ARCHIVE_CONFIDENCE = 0.6
CREDIT = "none: an inferred build graph and link order; it verifies no bytes and grants no match credit"
DEFAULT_LIMIT, DEFAULT_BUDGET = 256, 500_000
RAM_END = 0x80800000


def hx(value: int | None) -> str | None:
    return None if value is None else f"0x{value:08X}"


def align_up(value: int, alignment: int) -> int:
    alignment = max(1, alignment)
    return (value + alignment - 1) // alignment * alignment


def ident(text: str) -> str:
    """A linker-script identifier made from a name."""
    out = re.sub(r"[^A-Za-z0-9_]", "_", text)
    return out if out and not out[0].isdigit() else "_" + out


def _comment(text: str) -> str:
    return " ".join(str(text).replace("*/", "* /").split())


def _int(value) -> int:
    return value if isinstance(value, int) else int(str(value), 0)


# ---------------------------------------------------------------------------------------------
# graph


@dataclass
class Piece:
    """One unit's contribution to one section.  ``start`` None: the unit has this section, its
    place is unknown.  ``exact`` False: start and size are a lower bound of the real extent."""
    unit: str
    section: str
    start: int | None = None
    size: int | None = None
    align: int | None = None
    exact: bool = True
    confidence: float = 1.0
    provenance: str = ""

    @property
    def placed(self) -> bool:
        return self.start is not None and self.size is not None

    @property
    def end(self) -> int | None:
        return self.start + self.size if self.placed else None

    def as_dict(self) -> dict:
        return {"unit": self.unit, "section": self.section, "start": hx(self.start),
                "size": None if self.size is None else f"0x{self.size:X}", "align": self.align,
                "exact": self.exact, "confidence": self.confidence, "provenance": self.provenance}


@dataclass
class Unit:
    name: str
    segment: str
    kind: str = "c"
    archive: str | None = None
    member: str | None = None
    derived: bool = False
    names: list = field(default_factory=list)      # name evidence: {"name", "source", "confidence"}

    def object_path(self, build: str = "build") -> str:
        if self.archive and self.member:
            return f"*{self.archive}.a:{self.member}"
        return f"{build}/{self.segment}/{self.name}.o"

    def as_dict(self) -> dict:
        return {"name": self.name, "segment": self.segment, "kind": self.kind, "archive": self.archive,
                "member": self.member, "derived": self.derived, "object": self.object_path(),
                "names": self.names}


@dataclass
class Segment:
    name: str
    vram: int | None = None
    rom: int | None = None
    exclusive: str | None = None     # overlay group: segments of one group share RAM and exclude each other
    codec: str | None = None
    sections: dict = field(default_factory=dict)   # section -> (vram start, vram end)

    def as_dict(self) -> dict:
        return {"name": self.name, "vram": hx(self.vram), "rom": None if self.rom is None else f"0x{self.rom:X}",
                "exclusive": self.exclusive, "codec": self.codec,
                "sections": {s: [hx(a), hx(b)] for s, (a, b) in self.sections.items()}}


@dataclass
class Constraint:
    id: int
    kind: str                        # before | adjacent | group
    units: tuple
    segment: str | None
    confidence: float
    provenance: str
    section: str | None = None
    origin: str = "given"            # given | derived
    status: str = "active"           # active | rejected | invalid
    reason: str = ""

    def as_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "units": list(self.units), "segment": self.segment,
                "section": self.section, "confidence": round(self.confidence, 3), "provenance": self.provenance,
                "origin": self.origin, "status": self.status, "reason": self.reason}


class LinkGraph:
    """Segments, units, pieces, constraints and linker symbols of one program."""

    def __init__(self):
        self.segments: dict[str, Segment] = {}
        self.units: dict[str, Unit] = {}
        self.pieces: list[Piece] = []
        self.constraints: list[Constraint] = []
        self.symbols: dict[str, dict] = {}
        self.holes: list[dict] = []
        self.alignments: list[dict] = []
        self.problems: list[dict] = []     # contradictions in the evidence itself
        self.hints: list[dict] = []        # evidence that the unit partition is wrong (split/merge)
        self.evidence: list[dict] = []     # what each extractor found
        self._next_id = 0
        self._derived = False

    # building ---------------------------------------------------------------------------------

    def add_segment(self, name: str, vram: int | None = None, rom: int | None = None, exclusive: str | None = None,
                    codec: str | None = None, sections: dict | None = None) -> Segment:
        if name in self.segments:
            raise ValueError(f"segment {name!r} already exists")
        segment = Segment(name, vram, rom, exclusive, codec, dict(sections or {}))
        self.segments[name] = segment
        return segment

    def add_unit(self, name: str, segment: str, kind: str = "c", archive: str | None = None,
                 member: str | None = None, derived: bool = False) -> Unit:
        if segment not in self.segments:
            raise ValueError(f"unknown segment {segment!r}")
        if kind not in UNIT_KINDS:
            raise ValueError(f"unit kind {kind!r} is not one of {', '.join(UNIT_KINDS)}")
        if name in self.units:
            raise ValueError(f"unit {name!r} already exists")
        unit = Unit(name, segment, kind, archive, member, derived)
        self.units[name] = unit
        self._derived = False
        return unit

    def add_piece(self, unit: str, section: str, start: int | None = None, size: int | None = None,
                  align: int | None = None, exact: bool = True, confidence: float = 1.0, provenance: str = "") -> Piece:
        if unit not in self.units:
            raise ValueError(f"unknown unit {unit!r}")
        if section not in SECTIONS:
            raise ValueError(f"unknown section {section!r}")
        piece = Piece(unit, section, start, size, align, exact, float(confidence), provenance)
        self.pieces.append(piece)
        self._derived = False
        return piece

    def add_constraint(self, kind: str, units, confidence: float, provenance: str, section: str | None = None,
                       origin: str = "given") -> Constraint:
        if kind not in CONSTRAINT_KINDS:
            raise ValueError(f"constraint kind {kind!r} is not one of {', '.join(CONSTRAINT_KINDS)}")
        units = tuple(units)
        missing = [u for u in units if u not in self.units]
        if missing:
            raise ValueError(f"unknown unit(s) {', '.join(missing)}")
        if kind in ("before", "adjacent") and (len(units) != 2 or units[0] == units[1]):
            raise ValueError(f"{kind} needs two different units")
        if kind == "adjacent" and section is None:
            raise ValueError("adjacent is relative to one section: give the section")
        segments = sorted({self.units[u].segment for u in units})
        constraint = Constraint(self._next_id, kind, units, segments[0] if len(segments) == 1 else None,
                                float(confidence), provenance, section, origin)
        self._next_id += 1
        if constraint.segment is None:
            groups = {self.segments[s].exclusive for s in segments}
            exclusive = len(groups) == 1 and None not in groups
            constraint.status = "invalid"
            constraint.reason = ("units in mutually exclusive overlays share no link order" if exclusive else
                                 "units in different segments: each segment is linked in its own order")
            self.problems.append({"kind": "exclusive-overlays" if exclusive else "cross-segment", "origin": origin,
                                  "segments": segments, "constraint": constraint.as_dict(),
                                  "reason": f"{constraint.reason} ({', '.join(units)}: {provenance})"})
        self.constraints.append(constraint)
        return constraint

    def before(self, a: str, b: str, confidence: float, provenance: str, section: str | None = None) -> Constraint:
        return self.add_constraint("before", (a, b), confidence, provenance, section)

    def adjacent(self, a: str, b: str, section: str, confidence: float, provenance: str) -> Constraint:
        return self.add_constraint("adjacent", (a, b), confidence, provenance, section)

    def group(self, members, confidence: float, provenance: str) -> Constraint:
        return self.add_constraint("group", members, confidence, provenance)

    def set_symbol(self, name: str, value: int | None, provenance: str) -> None:
        self.symbols[name] = {"value": value, "provenance": provenance}

    # derivation -------------------------------------------------------------------------------

    def _reset_derived(self) -> None:
        derived = {name for name, unit in self.units.items() if unit.derived}
        for name in derived:
            del self.units[name]
        self.pieces = [p for p in self.pieces if p.unit not in derived]
        self.constraints = [c for c in self.constraints if c.origin != "derived"]
        self.problems = [p for p in self.problems if p.get("origin") != "derived"]
        self.holes, self.alignments = [], []

    def derive(self, read=None, max_align: int = 16, materialize: bool = True) -> None:
        """Constraints from placed pieces.  ``read(segment, vram, size) -> bytes | None`` gives the
        bytes of a gap (None: unknown).  ``max_align``: the largest input-section alignment a gap
        may be explained by.  ``materialize``: unattributed content between exact pieces becomes a
        ``raw`` unit.  Runs once until the graph changes."""
        if self._derived:
            return
        self._reset_derived()
        by_key = defaultdict(list)
        for piece in self.pieces:
            if piece.placed:
                by_key[(self.units[piece.unit].segment, piece.section)].append(piece)
        raw = []
        for (segment, section), pieces in sorted(by_key.items()):
            pieces.sort(key=lambda p: (p.start, p.end, p.unit))
            runs = []
            for piece in pieces:
                if not runs or runs[-1] != piece.unit:
                    runs.append(piece.unit)
            for unit, count in sorted(Counter(runs).items()):
                if count > 1:
                    self.problems.append({"kind": "split-piece", "origin": "derived", "segment": segment, "section": section,
                                          "units": [unit], "reason": f"{unit} has {section} bytes on both sides of another "
                                          f"unit's: the unit partition or an attribution is wrong"})
            self._edges_between(segment, section, pieces, read, max_align, raw, materialize)
            self._section_ends(segment, section, pieces, read, max_align)
        for segment, section, p, q, confidence in raw:
            name = f"raw_{ident(segment)}_{p.end:08X}"
            if name in self.units:
                name = f"{name}_{section.strip('.')}"
            self.units[name] = Unit(name, segment, "raw", derived=True)
            size = q.start - p.end
            self.pieces.append(Piece(name, section, p.end, size, None, True, confidence,
                                     f"0x{size:X} unattributed non-zero bytes between {p.unit} and {q.unit} in {section}"))
            why = f"{section}: unattributed bytes 0x{p.end:08X}-0x{q.start:08X} fill the gap between {p.unit} and {q.unit}"
            self.add_constraint("adjacent", (p.unit, name), confidence, why, section, origin="derived")
            self.add_constraint("adjacent", (name, q.unit), confidence, why, section, origin="derived")
        self._derived = True

    def _edges_between(self, segment, section, pieces, read, max_align, raw, materialize: bool = True) -> None:
        for p, q in zip(pieces, pieces[1:]):
            if p.unit == q.unit:
                continue
            confidence = min(p.confidence, q.confidence)
            where = (f"{section}: {p.unit} 0x{p.start:08X}-0x{p.end:08X} then {q.unit} "
                     f"0x{q.start:08X}-0x{q.end:08X}")
            if q.start < p.end:
                self.problems.append({"kind": "overlap" if p.exact and q.exact else "interleave", "origin": "derived",
                                      "segment": segment, "section": section, "units": [p.unit, q.unit],
                                      "reason": f"pieces overlap ({where}): no order of the two explains them"})
                continue
            self.add_constraint("before", (p.unit, q.unit), confidence,
                                f"{where}; input sections lie in object order", section, origin="derived")
            if not (p.exact and q.exact):
                continue
            data = read(segment, p.end, q.start - p.end) if read is not None and q.start > p.end else None
            gap = infer_gap(p.end, q.start, q.align, max_align, data)
            if gap["kind"] in ("abutting", "padding"):
                fill = "nothing between" if gap["kind"] == "abutting" else \
                    f"0x{gap['gap']:X} bytes of fill to a {gap['alignment_min']}-byte boundary"
                self.add_constraint("adjacent", (p.unit, q.unit), confidence, f"{where}: {fill}", section, origin="derived")
                if gap["kind"] == "padding":
                    self.alignments.append({"unit": q.unit, "section": section, "start": hx(q.start),
                                            "alignment_min": gap["alignment_min"], "alignments": gap["alignments"],
                                            "fill": gap["gap"], "zero": gap["zero"]})
            else:
                self.holes.append({"segment": segment, "section": section, "start": hx(p.end), "size": gap["gap"],
                                   "kind": gap["kind"], "after": p.unit, "before": q.unit})
                if gap["kind"] == "content" and materialize:
                    raw.append((segment, section, p, q, confidence))

    def _section_ends(self, segment, section, pieces, read, max_align) -> None:
        bounds = self.segments[segment].sections.get(section)
        exact = [p for p in pieces if p.exact]
        if not bounds or not exact:
            return
        lo, hi = bounds
        first, last = exact[0], max(exact, key=lambda p: p.end)
        if first.start > lo:
            data = read(segment, lo, first.start - lo) if read is not None else None
            gap = infer_gap(lo, first.start, first.align, max_align, data)
            if gap["kind"] not in ("abutting", "padding"):
                self.holes.append({"segment": segment, "section": section, "start": hx(lo), "size": gap["gap"],
                                   "kind": gap["kind"], "after": None, "before": first.unit})
        if last.end < hi:
            data = read(segment, last.end, hi - last.end) if read is not None else None
            zero = None if data is None else not any(data)
            if hi - last.end >= max_align or zero is False:
                self.holes.append({"segment": segment, "section": section, "start": hx(last.end), "size": hi - last.end,
                                   "kind": "content" if zero is False else "zero-fill" if zero else "unexplained",
                                   "after": last.unit, "before": None})

    # export -----------------------------------------------------------------------------------

    def as_dict(self) -> dict:
        pieces = defaultdict(list)
        for piece in self.pieces:
            pieces[piece.unit].append(piece.as_dict())
        return {"model": "build-graph-v1", "credit": CREDIT,
                "segments": [s.as_dict() for s in self.segments.values()],
                "units": [dict(u.as_dict(), pieces=pieces.get(u.name, [])) for u in self.units.values()],
                "constraints": [c.as_dict() for c in self.constraints],
                "symbols": {k: {"value": hx(v["value"]), "provenance": v["provenance"]} for k, v in self.symbols.items()},
                "holes": self.holes, "alignments": self.alignments, "problems": self.problems, "hints": self.hints,
                "evidence": self.evidence}


def infer_gap(end: int, start: int, align: int | None = None, max_align: int = 16, data: bytes | None = None) -> dict:
    """What lies between one input section's end and the next one's start.  ``abutting``: nothing;
    ``padding``: the bytes reach the next power-of-two boundary (``alignments``: every alignment up
    to ``max_align``, or the known ``align``, that explains them; ``alignment_min`` the smallest);
    ``zero-fill``: zero bytes no alignment explains; ``content``: non-zero bytes; ``unexplained``:
    bytes unknown and no alignment explains them; ``overlap``: the next starts before this ends."""
    gap = start - end
    if gap < 0:
        return {"kind": "overlap", "gap": gap, "alignments": [], "alignment_min": None, "zero": None}
    if gap == 0:
        return {"kind": "abutting", "gap": 0, "alignments": [], "alignment_min": None, "zero": None}
    candidates = [align] if align else [1 << k for k in range(max_align.bit_length()) if (1 << k) <= max_align]
    fitting = [a for a in candidates if start % a == 0 and align_up(end, a) == start]
    zero = None if data is None else not any(data)
    if fitting and zero is not False:
        return {"kind": "padding", "gap": gap, "alignments": fitting, "alignment_min": min(fitting), "zero": zero}
    kind = "content" if zero is False else "zero-fill" if zero else "unexplained"
    return {"kind": kind, "gap": gap, "alignments": [], "alignment_min": None, "zero": zero}


# ---------------------------------------------------------------------------------------------
# solver


def _unit_key(graph: LinkGraph):
    text = {}
    for piece in graph.pieces:
        if piece.section == ".text" and piece.placed:
            text[piece.unit] = min(text.get(piece.unit, piece.start), piece.start)
    return lambda unit: (0, text[unit], unit) if unit in text else (1, 0, unit)


def _segment_order(graph: LinkGraph) -> list[Segment]:
    return sorted(graph.segments.values(), key=lambda s: (s.rom is None, s.rom or 0, s.vram or 0, s.name))


def _find_cycle(nodes, succ) -> list | None:
    color = dict.fromkeys(nodes, 0)
    for root in nodes:
        if color[root]:
            continue
        color[root] = 1
        path, stack = [root], [(root, iter(succ.get(root, ())))]
        while stack:
            node, successors = stack[-1]
            following = next(successors, None)
            if following is None:
                stack.pop()
                path.pop()
                color[node] = 2
                continue
            if color.get(following, 0) == 1:
                return path[path.index(following):]
            if color.get(following, 0) == 0:
                color[following] = 1
                path.append(following)
                stack.append((following, iter(succ.get(following, ()))))
    return None


def _topological(nodes, edges, key) -> list:
    succ, indeg = defaultdict(list), dict.fromkeys(nodes, 0)
    for a, b in edges:
        succ[a].append(b)
        indeg[b] += 1
    heap = [(key(n), n) for n in nodes if indeg[n] == 0]
    heapq.heapify(heap)
    out = []
    while heap:
        _key, node = heapq.heappop(heap)
        out.append(node)
        for following in succ[node]:
            indeg[following] -= 1
            if indeg[following] == 0:
                heapq.heappush(heap, (key(following), following))
    return out


def _enumerate(nodes, edges, adjacency, groups, fits, key, limit, budget):
    """Linear extensions of ``edges`` over ``nodes`` that keep every adjacency ((a, b, blockers):
    no unit of ``blockers`` other than b between a and b) and every group (members contiguous) and pass
    ``fits(order)``; depth first, candidates in ``key`` order, at most ``limit`` orderings and
    ``budget`` steps.  Returns (orderings, truncated, steps)."""
    if not nodes:
        return [()], False, 0
    succ, indeg = {n: [] for n in nodes}, dict.fromkeys(nodes, 0)
    for a, b in edges:
        succ[a].append(b)
        indeg[b] += 1
    adj_after = defaultdict(list)
    for a, b, blockers in adjacency:
        adj_after[a].append((b, blockers))
    group_of = defaultdict(list)
    for index, members in enumerate(groups):
        for member in members:
            group_of[member].append(index)
    count = [0] * len(groups)
    open_groups: set = set()
    open_adj: list = []
    placed: set = set()
    available = {n for n in nodes if indeg[n] == 0}
    order: list = []
    solutions: list = []
    steps, truncated = 0, False

    def allowed(node) -> bool:
        for index in open_groups:
            if node not in groups[index]:
                return False
        for b, blockers in open_adj:
            if node != b and node in blockers:
                return False
        return True

    def apply(node):
        added, opened, closed, removed = [], [], [], []
        order.append(node)
        placed.add(node)
        available.discard(node)
        for following in succ[node]:
            indeg[following] -= 1
            if indeg[following] == 0:
                available.add(following)
                added.append(following)
        for index in group_of[node]:
            count[index] += 1
            if count[index] == 1:
                open_groups.add(index)
                opened.append(index)
            if count[index] == len(groups[index]):
                open_groups.discard(index)
                closed.append(index)
        keep = []
        for entry in open_adj:
            (removed if entry[0] == node else keep).append(entry)
        open_adj[:] = keep
        new = 0
        for b, blockers in adj_after[node]:
            if b not in placed:
                open_adj.append((b, blockers))
                new += 1
        return node, added, opened, closed, removed, new

    def revert(state) -> None:
        node, added, opened, closed, removed, new = state
        if new:
            del open_adj[len(open_adj) - new:]
        open_adj.extend(removed)
        for index in group_of[node]:
            count[index] -= 1
        for index in closed:
            open_groups.add(index)
        for index in opened:
            open_groups.discard(index)
        for following in succ[node]:
            indeg[following] += 1
        for following in added:
            available.discard(following)
        available.add(node)
        placed.discard(node)
        order.pop()

    frames = [[sorted(available, key=key), 0, None]]
    while frames:
        frame = frames[-1]
        if frame[2] is not None:
            revert(frame[2])
            frame[2] = None
        if frame[1] >= len(frame[0]):
            frames.pop()
            continue
        node = frame[0][frame[1]]
        frame[1] += 1
        steps += 1
        if steps > budget:
            truncated = True
            break
        if not allowed(node):
            continue
        frame[2] = apply(node)
        if len(order) == len(nodes):
            if fits is None or fits(order):
                solutions.append(tuple(order))
                if len(solutions) >= limit:
                    truncated = True
                    break
            continue
        frames.append([sorted(available, key=key), 0, None])
    return solutions, truncated, steps


def _fit_check(graph: LinkGraph, segment: Segment, members: set):
    """``fits(order)`` for the pieces known only by size: in every section they must fit, with
    their alignment, between the placed pieces the order puts around them (and inside the
    section's bounds when known).  (None, None) when no such piece exists."""
    wanted = defaultdict(lambda: defaultdict(list))
    placed = defaultdict(dict)
    confidence = 1.0
    for piece in graph.pieces:
        if piece.unit not in members:
            continue
        if piece.placed and piece.exact:
            start, end = placed[piece.section].get(piece.unit, (piece.start, piece.end))
            placed[piece.section][piece.unit] = (min(start, piece.start), max(end, piece.end))
        elif piece.start is None and piece.size:
            wanted[piece.section][piece.unit].append((piece.size, piece.align or 1))
            confidence = min(confidence, piece.confidence)
    if not wanted:
        return None, None
    bounds = dict(segment.sections)

    def fits(order) -> bool:
        for section, need in wanted.items():
            lo, hi = bounds.get(section, (None, None))
            cursor, pending = lo, False
            here = placed[section]
            for unit in order:
                if unit in here:
                    start, end = here[unit]
                    if pending and cursor is not None and cursor > start:
                        return False
                    cursor, pending = end, False
                if unit in need and cursor is not None:
                    for size, alignment in need[unit]:
                        cursor = align_up(cursor, alignment) + size
                    pending = True
            if pending and cursor is not None and hi is not None and cursor > hi:
                return False
        return True
    return fits, confidence


def _solve_segment(graph: LinkGraph, segment: Segment, limit: int, budget: int, key, contradictions: list) -> dict:
    nodes = sorted((name for name, unit in graph.units.items() if unit.segment == segment.name), key=key)
    members = set(nodes)
    active = [c for c in graph.constraints if c.segment == segment.name and c.status == "active"]
    edges = defaultdict(list)
    for c in active:
        if c.kind in ("before", "adjacent"):
            edges[c.units].append(c)
    while True:
        succ = defaultdict(list)
        for a, b in edges:
            succ[a].append(b)
        for a in succ:
            succ[a].sort(key=key)
        cycle = _find_cycle(nodes, succ)
        if cycle is None:
            break
        steps = list(zip(cycle, cycle[1:] + cycle[:1]))
        strength = {s: max(c.confidence for c in edges[s]) for s in steps}
        weakest = min(steps, key=lambda s: (strength[s], -max(c.id for c in edges[s])))
        involved = [c for s in steps for c in edges[s]]
        rejected = edges.pop(weakest)
        for c in rejected:
            c.status = "rejected"
            c.reason = (f"set aside to break a cycle: {weakest[0]} -> {weakest[1]} is its weakest step "
                        f"({strength[weakest]:.2f})")
        contradictions.append({"kind": "cycle", "segment": segment.name, "cycle": list(cycle),
                               "constraints": [c.as_dict() for c in involved], "rejected": [c.id for c in rejected],
                               "reason": f"the constraints order {' -> '.join(cycle)} -> {cycle[0]} in a circle; "
                                         f"{weakest[0]} -> {weakest[1]} (confidence {strength[weakest]:.2f}) set aside"})
    contributors = defaultdict(set)
    for piece in graph.pieces:
        if piece.unit in members and (piece.size is None or piece.size > 0):
            contributors[piece.section].add(piece.unit)
    fits, fit_confidence = _fit_check(graph, segment, members)
    while True:
        adjacency = [c for c in active if c.kind == "adjacent" and c.status == "active"]
        groups = [c for c in active if c.kind == "group" and c.status == "active"]
        adj = [(c.units[0], c.units[1], contributors[c.section]) for c in adjacency]
        grp = [g for g in (frozenset(c.units) & members for c in groups) if len(g) > 1]
        solutions, truncated, steps = _enumerate(nodes, list(edges), adj, grp, fits, key, limit, budget)
        if solutions or truncated:
            break
        candidates = [((c.confidence, -c.id), c) for c in adjacency + groups]
        if fits is not None:
            candidates.append(((fit_confidence, 1), None))
        if not candidates:
            break
        _rank, weakest = min(candidates, key=lambda item: item[0])
        if weakest is None:
            fits = None
            contradictions.append({"kind": "unsatisfiable", "segment": segment.name, "rejected": ["sizes"],
                                   "reason": "no ordering lets the pieces known only by size fit between the placed "
                                             "pieces: the size check is set aside"})
            continue
        weakest.status = "rejected"
        weakest.reason = "set aside: no ordering satisfies every adjacency, grouping and size constraint together"
        implied = edges.get(weakest.units) if weakest.kind == "adjacent" else None
        if implied is not None and weakest in implied:
            implied.remove(weakest)
            if not implied:
                del edges[weakest.units]
        contradictions.append({"kind": "unsatisfiable", "segment": segment.name,
                               "constraints": [c.as_dict() for c in adjacency + groups], "rejected": [weakest.id],
                               "reason": f"no ordering satisfies every adjacency, grouping and size constraint; the "
                                         f"weakest, {weakest.kind} {', '.join(weakest.units)} "
                                         f"({weakest.confidence:.2f}), set aside"})
    representative = list(solutions[0]) if solutions else _topological(nodes, list(edges), key)
    maps = [{u: i for i, u in enumerate(solution)} for solution in solutions]
    free = {}
    if maps:
        for unit in nodes:
            seen = [m[unit] for m in maps]
            if min(seen) != max(seen):
                free[unit] = [min(seen), max(seen)]
    ambiguous = []
    for unit in sorted(free, key=key):
        for other in nodes:
            if other == unit or (other in free and key(other) < key(unit)):
                continue
            first = sum(1 for m in maps if m[unit] < m[other])
            if 0 < first < len(maps):
                ambiguous.append({"a": unit, "b": other, "a_first": first, "of": len(maps)})
    steps_out = []
    for a, b in zip(representative, representative[1:]):
        direct = [c for c in edges.get((a, b), []) if c.status == "active"]
        evidence = max((c.confidence for c in direct), default=0.0)
        share_before = sum(1 for m in maps if m[a] < m[b]) / len(maps) if maps else None
        share_next = sum(1 for m in maps if m[b] == m[a] + 1) / len(maps) if maps else None
        confidence = evidence if evidence > 0 else (0.5 * share_before if share_before is not None else 0.0)
        steps_out.append({"from": a, "to": b, "confidence": round(confidence, 3), "evidence": round(evidence, 3),
                          "forced": bool(maps) and not truncated and share_before == 1.0,
                          "share_before": None if share_before is None else round(share_before, 3),
                          "share_next": None if share_next is None else round(share_next, 3),
                          "constraints": [c.id for c in direct], "provenance": [c.provenance for c in direct[:3]]})
    return {"segment": segment.name, "units": len(nodes), "order": representative, "orderings": len(solutions),
            "complete": bool(solutions) and not truncated, "truncated": truncated, "search_steps": steps,
            "positions": free, "ambiguous_pairs": ambiguous[:500], "ambiguous_pair_count": len(ambiguous),
            "edges": steps_out}


def solve(graph: LinkGraph, limit: int = DEFAULT_LIMIT, budget: int = DEFAULT_BUDGET, read=None) -> dict:
    """Solve every segment's link order.  ``limit``: orderings kept per segment; ``budget``:
    search steps per enumeration.  Contradictions (evidence that overlaps, cycles, unsatisfiable
    soft constraints, constraints across segments or between exclusive overlays) are listed, each
    with the constraints involved and what was set aside; nothing is dropped silently."""
    graph.derive(read)
    for c in graph.constraints:
        if c.status == "rejected":
            c.status, c.reason = "active", ""
    contradictions = [dict(problem) for problem in graph.problems]
    key = _unit_key(graph)
    segments = {}
    for segment in _segment_order(graph):
        segments[segment.name] = _solve_segment(graph, segment, limit, budget, key, contradictions)
    orderings, complete = 1, True
    for result in segments.values():
        orderings *= max(1, result["orderings"])
        complete = complete and result["complete"]
    return {"model": "link-layout-v1", "credit": CREDIT, "consistent": not contradictions, "complete": complete,
            "orderings": orderings, "segments": segments, "contradictions": contradictions,
            "constraints": dict(Counter(f"{c.kind}:{c.status}" for c in graph.constraints))}


# ---------------------------------------------------------------------------------------------
# outputs


def _section_lists(graph: LinkGraph, segment: Segment) -> tuple[list[str], list[str]]:
    present = set(segment.sections) | {p.section for p in graph.pieces if graph.units[p.unit].segment == segment.name}
    present |= {".text", ".data", ".rodata", ".bss"}

    def rank(section):
        known = segment.sections.get(section)
        return (0, known[0], SECTIONS.index(section)) if known else (1, 0, SECTIONS.index(section))
    ordered = sorted(present, key=rank)
    return [s for s in ordered if s not in NOLOAD], [s for s in ordered if s in NOLOAD]


def linker_script(graph: LinkGraph, solution: dict, build: str = "build") -> str:
    """A GNU ld script for the representative order of every segment; alternatives, unordered
    steps and contradictions as comments."""
    out = ["/* Generated by `n64dk link-layout`: the link order inferred from the evidence in the build",
           " * graph.  A hypothesis, not a verified layout: it verifies no bytes and grants no match credit.",
           f" * Orderings retained: {solution['orderings']}"
           f"{'' if solution['complete'] else ' (at least: the enumeration was truncated)'}; "
           f"contradictions: {len(solution['contradictions'])}. */"]
    for item in solution["contradictions"]:
        out.append(f"/* CONTRADICTION ({item['kind']}): {_comment(item.get('reason', ''))} */")
    out += ["", "OUTPUT_ARCH (mips)", "", "SECTIONS", "{", "    __romPos = 0;"]
    gp = graph.symbols.get("_gp")
    if gp and gp.get("value") is not None:
        out.append(f"    _gp = 0x{gp['value']:08X};  /* {_comment(gp.get('provenance', ''))} */")
    elif gp:
        out.append(f"    /* _gp: not established ({_comment(gp.get('provenance', ''))}) */")
    exclusive = defaultdict(list)
    for segment in graph.segments.values():
        if segment.exclusive:
            exclusive[segment.exclusive].append(segment.name)
    for segment in _segment_order(graph):
        result = solution["segments"].get(segment.name)
        if result is None:
            continue
        order, name = result["order"], ident(segment.name)
        free = result.get("positions", {})
        weak = {(e["from"], e["to"]) for e in result.get("edges", []) if not e["evidence"]}
        out.append("")
        out.append(f"    /* segment {segment.name}: {len(order)} units, {result['orderings']} retained ordering(s)"
                   f"{', enumeration truncated' if result['truncated'] else ''} */")
        if segment.exclusive:
            others = [s for s in exclusive[segment.exclusive] if s != segment.name]
            out.append(f"    /* overlay group {segment.exclusive}: mutually exclusive with {', '.join(others)} (same RAM) */")
        if segment.codec:
            out.append(f"    /* stored {segment.codec}-compressed in ROM: this output section is the decompressed payload */")
        if segment.rom is not None:
            out.append(f"    __romPos = 0x{segment.rom:X};")
        out.append(f"    {name}_ROM_START = __romPos;")
        vma = f"0x{segment.vram:08X} " if segment.vram is not None else ""
        out.append(f"    .{name} {vma}: AT({name}_ROM_START)")
        out.append("    {")
        loaded, noload = _section_lists(graph, segment)
        for section in loaded:
            tag = ident(section.strip(".").upper())
            out.append(f"        {name}_{tag}_START = .;")
            for index, unit in enumerate(order):
                if section == loaded[0]:
                    if unit in free:
                        lo, hi = free[unit]
                        out.append(f"        /* ALTERNATIVE: {unit} is at position {index} here; the retained "
                                   f"orderings put it anywhere from position {lo} to {hi} */")
                    if index and (order[index - 1], unit) in weak:
                        out.append(f"        /* UNORDERED: no constraint places {order[index - 1]} directly before {unit} */")
                out.append(f"        {graph.units[unit].object_path(build)}({section});")
            out.append(f"        {name}_{tag}_END = .;")
        out.append("    }")
        out.append(f"    __romPos += SIZEOF(.{name});")
        out.append(f"    {name}_ROM_END = __romPos;")
        if noload:
            out.append(f"    .{name}.noload (NOLOAD) :")
            out.append("    {")
            for section in noload:
                tag = ident(section.strip(".").upper())
                out.append(f"        {name}_{tag}_START = .;")
                for unit in order:
                    out.append(f"        {graph.units[unit].object_path(build)}({section});")
                out.append(f"        {name}_{tag}_END = .;")
            for unit in order:
                out.append(f"        {graph.units[unit].object_path(build)}(COMMON);")
            out.append("    }")
    end = graph.symbols.get("_end")
    if end and end.get("value") is not None:
        out.append(f"    _end = 0x{end['value']:08X};  /* {_comment(end.get('provenance', ''))} */")
    else:
        why = f" ({_comment(end.get('provenance', ''))})" if end else ""
        out.append(f"    _end = .;  /* end of the last segment above{why} */")
    out += ["", "    /DISCARD/ :", "    {",
            "        *(.MIPS.abiflags) *(.MIPS.options) *(.reginfo) *(.pdr) *(.mdebug) *(.mdebug.*)",
            "        *(.gnu.attributes) *(.comment) *(.note) *(.note.*)", "    }", "}", ""]
    return "\n".join(out)


MANIFEST_COLUMNS = ("segment", "position", "positions", "unit", "kind", "object", "archive", "member",
                    "text_start", "text_size", "sections", "names")


def manifest(graph: LinkGraph, solution: dict, build: str = "build") -> list[dict]:
    """One row per unit in solved order: the object it becomes and what is known of it."""
    pieces = defaultdict(list)
    for piece in graph.pieces:
        pieces[piece.unit].append(piece)
    rows = []
    for segment, result in solution["segments"].items():
        free = result.get("positions", {})
        for position, name in enumerate(result["order"]):
            unit = graph.units[name]
            text = [p for p in pieces[name] if p.section == ".text" and p.placed]
            rows.append({"segment": segment, "position": position,
                         "positions": f"{free[name][0]}-{free[name][1]}" if name in free else str(position),
                         "unit": name, "kind": unit.kind, "object": unit.object_path(build),
                         "archive": unit.archive or "", "member": unit.member or "",
                         "text_start": hx(min(p.start for p in text)) if text else "",
                         "text_size": f"0x{sum(p.size for p in text):X}" if text else "",
                         "sections": ",".join(sorted({p.section for p in pieces[name]}, key=SECTIONS.index)),
                         "names": ";".join(sorted({str(n.get("name")) for n in unit.names}))})
    return rows


def write_manifest(rows: list[dict], path) -> Path:
    path = Path(path)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(MANIFEST_COLUMNS)
        for row in rows:
            writer.writerow([row[c] for c in MANIFEST_COLUMNS])
    return path


def summary(graph: LinkGraph, solution: dict) -> dict:
    return {"model": solution["model"], "credit": CREDIT, "segments": len(graph.segments), "units": len(graph.units),
            "unit_kinds": dict(Counter(u.kind for u in graph.units.values())),
            "pieces": len(graph.pieces), "pieces_placed": sum(1 for p in graph.pieces if p.placed),
            "constraints": solution["constraints"],
            "holes": dict(Counter(h["kind"] for h in graph.holes)),
            "hints": dict(Counter(h["kind"] for h in graph.hints)),
            "contradictions": dict(Counter(c["kind"] for c in solution["contradictions"])),
            "orderings": solution["orderings"], "complete": solution["complete"],
            "symbols": {k: hx(v["value"]) for k, v in graph.symbols.items() if k in ("_gp", "_end")},
            "per_segment": {name: {"units": r["units"], "orderings": r["orderings"], "complete": r["complete"],
                                   "ambiguous_pairs": r["ambiguous_pair_count"],
                                   "unevidenced_steps": sum(1 for e in r["edges"] if not e["evidence"])}
                            for name, r in solution["segments"].items()}}


# ---------------------------------------------------------------------------------------------
# evidence extractors


def text_units(graph: LinkGraph, segment: str, partition, kinds: dict | None = None) -> dict[str, list]:
    """Units from a catalogue partition (``analysis.units.units``: objects with ``start``,
    ``end``, ``functions`` and ``closed``), each with an exact ``.text`` piece.  ``kinds``:
    vram -> catalogue tag; a unit whose functions are mostly ``asm`` is an ``asm`` unit.
    Returns unit name -> its catalogue rows."""
    out = {}
    for item in partition:
        name = f"{ident(segment)}_{item.start:08X}"
        tags = Counter((kinds or {}).get(vram, "c") for _name, vram, _size in item.functions)
        kind = "asm" if tags and tags.most_common(1)[0][0] == "asm" else "c"
        graph.add_unit(name, segment, kind)
        end = "its end is confirmed by alignment padding" if getattr(item, "closed", False) else \
            "its end is not confirmed by padding"
        graph.add_piece(name, ".text", item.start, item.end - item.start, None, True, 1.0,
                        f"catalogue: {len(item.functions)} function(s) 0x{item.start:08X}-0x{item.end:08X} taken as "
                        f"one object (analysis.units); {end}")
        out[name] = list(item.functions)
    return out


def code_references(image: bytes, delta: int, functions: dict[str, list], lo: int, hi: int) -> dict[str, list]:
    """Per unit, the (address, kind) of every ``%hi``/``%lo`` data reference its functions make
    into [lo, hi) (kind: load, store or addr)."""
    from .. import mips
    out = {}
    for unit, rows in functions.items():
        found = []
        for _name, vram, size in rows:
            offset = vram - delta
            if offset < 0 or offset + size > len(image):
                continue
            for ref in mips.scan_refs(image[offset:offset + size], vram, call_targets=False):
                if ref.kind in ("load", "store", "addr") and lo <= ref.address < hi:
                    found.append((ref.address, ref.kind))
        out[unit] = found
    return out


def private_addresses(references: dict[str, list]) -> tuple[dict[str, list[int]], dict]:
    """Addresses only one unit references, per unit (sorted), and counts."""
    users = defaultdict(set)
    for unit, items in references.items():
        for address, _kind in items:
            users[address].add(unit)
    private = defaultdict(set)
    for address, who in users.items():
        if len(who) == 1:
            private[next(iter(who))].add(address)
    return ({unit: sorted(found) for unit, found in private.items()},
            {"addresses": len(users), "shared": sum(1 for who in users.values() if len(who) > 1)})


def reference_pieces(graph: LinkGraph, sections: dict, private: dict[str, list[int]]) -> int:
    """A lower-bound piece per unit and data section: from the first to the last address in the
    section that only that unit's code references."""
    added = 0
    for section, (lo, hi) in sorted(sections.items()):
        if section == ".text" or section not in SECTIONS:
            continue
        confidence = REFERENCE_CONFIDENCE.get(section, 0.5)
        for unit, addresses in sorted(private.items()):
            if unit not in graph.units:
                continue
            inside = addresses[bisect.bisect_left(addresses, lo):bisect.bisect_left(addresses, hi)]
            if not inside:
                continue
            graph.add_piece(unit, section, inside[0], inside[-1] - inside[0] + 1, None, False, confidence,
                            f"{len(inside)} address(es) in {section} only {unit}'s code references (%hi/%lo), "
                            f"0x{inside[0]:08X}-0x{inside[-1]:08X}: a lower bound of its {section}")
            added += 1
    return added


def _misplaced(ranks: list[int]) -> int:
    """Items outside a longest increasing subsequence."""
    tails: list[int] = []
    for rank in ranks:
        index = bisect.bisect_left(tails, rank)
        if index == len(tails):
            tails.append(rank)
        else:
            tails[index] = rank
    return len(ranks) - len(tails)


def _split_score(rank: dict, addresses: dict, lo: int, split: int, hi: int) -> int:
    score = 0
    for part_lo, part_hi in ((lo, split), (split, hi)):
        intervals = []
        for unit, found in addresses.items():
            i, j = bisect.bisect_left(found, part_lo), bisect.bisect_left(found, part_hi)
            if i < j:
                intervals.append((found[i], found[j - 1], rank[unit]))
        intervals.sort()
        reach = None
        for first, last, _rank in intervals:
            if reach is not None and first <= reach:
                score += 1
            reach = last if reach is None else max(reach, last)
        score += _misplaced([r for _first, _last, r in intervals])
    return score


def infer_data_split(order: list[str], addresses: dict[str, list[int]], lo: int, hi: int,
                     stores=frozenset(), max_candidates: int = 256) -> dict:
    """Where an image's initialised data splits into two output sections, when nothing else says.

    Uses the link-order fact itself: in each output section the units' data lies in their text
    order.  Every 16-byte boundary at or below a referenced address is a candidate; the score is
    the number of units whose private-reference interval overlaps an earlier one plus the units
    out of text order, summed over both parts.  The lowest score wins; no split wins ties.  The
    part with the larger share of stores is ``.data``, the other ``.rodata``."""
    rank = {unit: i for i, unit in enumerate(order)}
    addresses = {u: a for u, a in addresses.items() if u in rank and a}
    candidates = sorted({a & ~0xF for found in addresses.values() for a in found if lo < (a & ~0xF) < hi})
    if len(candidates) > max_candidates:
        step = len(candidates) / max_candidates
        candidates = sorted({candidates[int(i * step)] for i in range(max_candidates)})
    unsplit = _split_score(rank, addresses, lo, hi, hi)
    best, best_score = lo, unsplit
    for split in candidates:
        score = _split_score(rank, addresses, lo, split, hi)
        if score < best_score:
            best, best_score = split, score
    if best == lo:
        return {"split": None, "sections": {".data": (lo, hi)}, "score": unsplit, "unsplit_score": unsplit,
                "candidates": len(candidates), "reason": "no split point puts more units in text order: one section"}

    def store_share(a: int, z: int) -> float:
        inside = [x for found in addresses.values() for x in found if a <= x < z]
        return sum(1 for x in inside if x in stores) / len(inside) if inside else 0.0
    first, second = store_share(lo, best), store_share(best, hi)
    names = (".rodata", ".data") if second > first else (".data", ".rodata")
    return {"split": best, "sections": {names[0]: (lo, best), names[1]: (best, hi)}, "score": best_score,
            "unsplit_score": unsplit, "candidates": len(candidates), "store_share": [round(first, 3), round(second, 3)],
            "reason": f"split at 0x{best:08X}: {best_score} unit(s) out of order or overlapping against {unsplit} "
                      f"without a split; stores name the parts"}


def sdk_membership(names: dict[int, str], entries_by_name: dict[str, list[dict]]) -> dict[int, dict]:
    """Library member and archive of SDK-named functions: the signature database records the
    object each signature came from (``obj``) and the builds that contain it (``sources`` tags
    ``library:version:archive``).  Only names whose signatures agree on one object are used."""
    out = {}
    for vram, name in names.items():
        entries = entries_by_name.get(name) or []
        objects = {e.get("obj") for e in entries if e.get("obj")}
        archives = Counter(tag.split(":")[-1] for e in entries for tag in e.get("sources", []) if tag.count(":") >= 2)
        if len(objects) != 1 or not archives:
            continue
        obj = next(iter(objects))
        member = re.sub(r"\.(c|s|S|cc|cpp)$", "", obj) + ".o"
        archive = archives.most_common(1)[0][0]
        out[vram] = {"name": name, "member": member, "archive": archive,
                     "provenance": f"SDK signature database: {name} is in {obj} of {archive}"}
    return out


def apply_membership(graph: LinkGraph, functions: dict[str, list], members: dict[int, dict]) -> dict:
    """Archive and member per unit from :func:`sdk_membership`; one ``group`` constraint per
    archive and segment; hints where the partition disagrees (a unit holding several members, a
    member spread over several units)."""
    stats = {"units_with_member": 0, "units_spanning_members": 0, "groups": 0}
    by_archive = defaultdict(list)
    holders = defaultdict(list)
    for unit_name, rows in functions.items():
        found = [members[vram] for _name, vram, _size in rows if vram in members]
        if not found:
            continue
        unit = graph.units[unit_name]
        distinct = sorted({(m["archive"], m["member"]) for m in found})
        archives = {archive for archive, _member in distinct}
        if len(distinct) == 1:
            unit.archive, unit.member = distinct[0]
            stats["units_with_member"] += 1
            holders[distinct[0]].append(unit_name)
        else:
            stats["units_spanning_members"] += 1
            if len(archives) == 1:
                unit.archive = next(iter(archives))
            graph.hints.append({"kind": "split", "unit": unit_name, "members": [m for _a, m in distinct],
                                "reason": f"{unit_name} holds functions of {len(distinct)} library members: "
                                          f"probably {len(distinct)} objects"})
        for archive, member in distinct:
            unit.names.append({"name": member, "source": f"sdk:{archive}", "confidence": "medium"})
        if unit.archive:
            by_archive[(unit.segment, unit.archive)].append(unit_name)
    for (archive, member), units in sorted(holders.items()):
        if len(units) > 1:
            graph.hints.append({"kind": "merge", "units": units, "member": member,
                                "reason": f"{len(units)} units carry {archive}:{member}: probably one object"})
    for (segment, archive), units in sorted(by_archive.items()):
        if len(units) > 1:
            graph.group(units, ARCHIVE_CONFIDENCE,
                        f"SDK signature database: {len(units)} units are members of {archive}; ld takes an "
                        f"archive's extracted members in at the archive's place in the input list")
            stats["groups"] += 1
    return stats


def apply_source_names(graph: LinkGraph, functions: dict[str, list], attributions: list[dict]) -> int:
    """Unit names from ``devrecords.attribute_functions`` (assertion strings name their own file);
    a unit whose functions name several files gets a split hint."""
    by_vram = {_int(a["vram"]): a["source_names"] for a in attributions}
    named = 0
    for unit_name, rows in functions.items():
        names = sorted({n for _name, vram, _size in rows for n in by_vram.get(vram, [])})
        if not names:
            continue
        named += 1
        for name in names:
            graph.units[unit_name].names.append({"name": name, "source": "assertion string (__FILE__)",
                                                 "confidence": "medium" if len(names) == 1 else "low"})
        if len(names) > 1:
            graph.hints.append({"kind": "split", "unit": unit_name, "members": names,
                                "reason": f"{unit_name} names {len(names)} source files: probably several objects"})
    return named


_TAG = re.compile(r"vram\s*=\s*0x([0-9A-Fa-f]+).*#\s*([A-Za-z_]+)")
_SYMBOL = re.compile(r"^\s*([A-Za-z_.$][\w.$]*)\s*=\s*0x([0-9A-Fa-f]+)\s*;")


def catalogue_tags(path) -> dict[int, str]:
    """vram -> the kind comment a ``decompile-init`` catalogue writes after each row (asm, c, ...)."""
    out = {}
    try:
        text = Path(path).read_text()
    except OSError:
        return out
    for line in text.splitlines():
        match = _TAG.search(line)
        if match:
            out[int(match.group(1), 16)] = match.group(2).lower()
    return out


def read_symbol_file(path) -> dict[int, str]:
    out = {}
    try:
        text = Path(path).read_text()
    except OSError:
        return out
    for line in text.splitlines():
        match = _SYMBOL.match(line)
        if match:
            out[int(match.group(2), 16)] = match.group(1)
    return out


def _read_json(path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def _exclusive_groups(spans: dict[str, tuple[int, int]]) -> dict[str, str]:
    """Segments whose RAM ranges overlap load over each other: one overlay group each."""
    parent = {name: name for name in spans}

    def root(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name
    names = sorted(spans)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if spans[a][0] < spans[b][1] and spans[b][0] < spans[a][1]:
                parent[root(a)] = root(b)
    groups = defaultdict(list)
    for name in names:
        groups[root(name)].append(name)
    out = {}
    for members in groups.values():
        if len(members) > 1:
            label = f"overlay_{min(spans[m][0] for m in members):08X}"
            for member in members:
                out[member] = label
    return out


def from_project(project, images=None, sdk: bool = True, source_names: bool = True, max_align: int = 16) -> LinkGraph:
    """The build graph of a ``decompile-init`` project: one segment per configured image with a
    catalogue; units from the catalogue partition; data pieces from code references (sections
    from ``survey/layout.json`` for the main image, an inferred split otherwise); SDK membership;
    source names; ``_gp``; segment symbols.  Derived constraints are added before returning."""
    from ..match import images as imagemod
    from . import units as unitsmod
    project = Path(project)
    registry = imagemod.load_registry(project)
    names = list(images) if images else sorted(registry, key=lambda n: (n != "main", n))
    layout = _read_json(project / "survey" / "layout.json")
    main_layout = layout.get("main") or {}
    loaded = {}
    graph = LinkGraph()
    for name in names:
        try:
            spec = imagemod.resolve(project, name)
            rows = [row for row in spec.rows() if row[2] >= 4]
            if not rows:
                graph.evidence.append({"image": name, "skipped": "no catalogue rows"})
                continue
            loaded[name] = (spec, spec.load(), spec.delta, sorted(rows, key=lambda r: r[1]))
        except (OSError, ValueError, LookupError) as error:    # an image that cannot load gives no evidence
            graph.evidence.append({"image": name, "skipped": f"{type(error).__name__}: {error}"})
    geometry = {}
    for name, (spec, data, delta, rows) in loaded.items():
        text = (rows[0][1], max(v + s for _n, v, s in rows))
        sections, rom, codec, source = {}, None, None, "catalogue span"
        if name == "main" and main_layout.get("sections"):
            for item in main_layout["sections"]:
                if item.get("section") in SECTIONS and item.get("vram") and item.get("vram_end"):
                    sections[item["section"]] = (_int(item["vram"]), _int(item["vram_end"]))
            rom = _int(main_layout["rom"]) if main_layout.get("rom") else None
            source = "survey layout"
        for item in layout.get("code_segments", []):
            if item.get("name") == name:
                rom = _int(item["rom"]) if item.get("rom") else None
                codec = item.get("codec")
        sections.setdefault(".text", text)
        end = max(b for _a, b in sections.values()) if source == "survey layout" else delta + len(data)
        geometry[name] = {"sections": sections, "rom": rom, "codec": codec, "source": source,
                          "image_end": delta + len(data), "span": (sections[".text"][0], end)}
    exclusive = _exclusive_groups({n: g["span"] for n, g in geometry.items()})
    entries_by_name = defaultdict(list)
    sdk_names = read_symbol_file(project / "survey" / "sdk_symbols.txt") if sdk else {}
    if sdk_names:
        from ..sdk.db import Database
        for entry in Database.load().entries.values():
            entries_by_name[entry["name"]].append(entry)
    for name, (spec, data, delta, rows) in loaded.items():
        geo = geometry[name]
        segment = graph.add_segment(name, geo["sections"][".text"][0], geo["rom"], exclusive.get(name), geo["codec"])
        tags = {v - spec.catalogue_shift: t for v, t in catalogue_tags(spec.catalogue).items()} if spec.catalogue else {}
        functions = text_units(graph, name, unitsmod.units(data, delta, rows), tags)
        record = {"image": name, "units": len(functions), "sections_from": geo["source"]}
        sections = {".text": geo["sections"][".text"]}
        if geo["source"] == "survey layout":
            data_sections = {s: r for s, r in geo["sections"].items() if s != ".text"}
            region = None
        elif name != "main":
            text_end, image_end = geo["sections"][".text"][1], geo["image_end"]
            above = [g["span"][0] for n, g in geometry.items() if n != name and g["span"][0] >= image_end]
            region = (text_end, image_end)
            data_sections = {".bss": (image_end, min(above + [RAM_END]))}
        else:
            data_sections, region = {}, None
            record["data"] = "skipped: the main image has no section layout"
        ranges = list(data_sections.values()) + ([region] if region else [])
        if ranges:
            lo, hi = min(a for a, _b in ranges), max(b for _a, b in ranges)
            references = code_references(data, delta, functions, lo, hi)
            private, counts = private_addresses(references)
            record.update(counts)
            if region and region[1] > region[0]:
                stores = {a for items in references.values() for a, kind in items if kind == "store"}
                inside = {u: [a for a in found if region[0] <= a < region[1]] for u, found in private.items()}
                split = infer_data_split(list(functions), inside, region[0], region[1], stores)
                data_sections.update(split["sections"])
                record["data_split"] = {"split": hx(split["split"]), "score": split["score"],
                                        "unsplit_score": split["unsplit_score"], "reason": split["reason"]}
            record["reference_pieces"] = reference_pieces(graph, data_sections, private)
        sections.update(data_sections)
        segment.sections = sections
        for section, (lo, hi) in sections.items():
            tag = ident(section.strip(".").upper())
            how = "survey layout" if geo["source"] == "survey layout" or section == ".text" else "inferred"
            graph.set_symbol(f"{ident(name)}_{tag}_START", lo, f"{section} of {name} ({how})")
            graph.set_symbol(f"{ident(name)}_{tag}_END", hi, f"{section} of {name} ({how})")
        if sdk_names:
            starts = {v for _n, v, _s in rows}
            members = sdk_membership({v: n for v, n in sdk_names.items() if v in starts}, entries_by_name)
            record["sdk"] = apply_membership(graph, functions, members)
        if source_names:
            from . import devrecords
            found = devrecords.source_names(data)
            attributions = devrecords.attribute_functions(data, delta, rows, found, delta) if found else []
            record["source_named_units"] = apply_source_names(graph, functions, attributions)
        graph.evidence.append(record)
    if loaded:
        from . import gp as gpmod
        found = gpmod.find({n: (d, de, r) for n, (_s, d, de, r) in loaded.items()})
        value = _int(found["value"]) if found.get("value") and found.get("confidence") == "high" else None
        graph.set_symbol("_gp", value, found.get("reason") or "not established")
        plain = [n for n in loaded if n not in exclusive]
        if plain and all(geometry[n]["source"] == "survey layout" and ".bss" in geometry[n]["sections"] for n in plain):
            graph.set_symbol("_end", max(geometry[n]["sections"][".bss"][1] for n in plain), "end of .bss (survey layout)")
        else:
            unknown = [n for n in plain if geometry[n]["source"] != "survey layout"]
            graph.set_symbol("_end", None, f"not established: the .bss end of {', '.join(unknown) or 'a segment'} is not known")

    def read(segment, vram, size):
        item = loaded.get(segment)
        if item is None or size > 0x100000:
            return None
        _spec, data, delta, _rows = item
        offset = vram - delta
        if offset < 0 or offset + size > len(data):
            return None
        return data[offset:offset + size]
    graph.derive(read, max_align)
    return graph
