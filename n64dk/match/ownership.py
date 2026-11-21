"""Byte-ownership ledger derived from the authoritative interval inventory.

Every catalogue section is partitioned into non-overlapping classified spans.  The ledger does
not turn claims into proof: verified C, assembly, structural exclusions, manifest claims,
conflicts and unknown bytes remain distinct.  This makes numerator and denominator changes
auditable instead of presenting boundary corrections as decompilation progress.

The second half of the module is the shared ownership model of physical ROM, decoded image ranges
and runtime allocations (:class:`OwnershipModel`) and the separate provenance of a declared and an
inferred ``_gp`` (:func:`gp_provenance`), used by decompile-run's opt-in data stage
(:mod:`n64dk.match.datastage`).
"""
from __future__ import annotations

import bisect
from collections import Counter, defaultdict
from dataclasses import dataclass, field


STATE_CLASS = {
    "manifest_exact_unit": "verified_c",
    "manifest_exact_member": "verified_c",
    "manifest_nonexact": "claimed_c",
    "assembly_unit": "assembly",
    "assembly_member": "assembly",
    "fragment": "fragment",
    "padding": "padding",
    "data": "data",
    "duplicate_owner": "conflict",
    "boundary_conflict": "conflict",
    "scratch_candidate": "unowned_code",
    "tracked_source_unverified": "unowned_code",
    "unowned": "unowned_code",
}
AUTHORITATIVE = {"verified_c", "assembly", "fragment", "padding", "data"}


def build(inventory: dict, image: str | None = None) -> dict:
    rows = [row for row in inventory.get("functions", []) if image is None or row["image"] == image]
    sections = {}
    for row in rows:
        key = (row["image"], int(row["section_vram"], 0), int(row["section_size"]))
        sections.setdefault(key, []).append(row)
    ledgers, all_segments, problems = [], [], []
    for (image_name, start, size), members in sorted(sections.items()):
        end = start + size; cursor = start; segments = []
        ordered = sorted(members, key=lambda row: (int(row["vram"], 0), int(row["size"]), row["name"]))
        for row in ordered:
            left = int(row["vram"], 0); right = left + int(row["size"])
            if left < cursor:
                problems.append({"image": image_name, "kind": "overlap", "at": f"0x{left:X}",
                                 "function": row["name"]})
            if left > cursor:
                segments.append({"start": cursor, "end": left, "size": left - cursor,
                                 "class": "unknown", "owner": None, "evidence": "catalogue gap"})
            cls = STATE_CLASS.get(row.get("ownership_state"), "unowned_code")
            segments.append({"start": left, "end": right, "size": right - left, "class": cls,
                             "owner": row.get("owner") or row["name"], "function": row["name"],
                             "evidence": row.get("reason"), "inventory_id": row.get("id")})
            cursor = max(cursor, right)
        if cursor < end:
            segments.append({"start": cursor, "end": end, "size": end - cursor,
                             "class": "unknown", "owner": None, "evidence": "catalogue gap"})
        totals = Counter()
        for segment in segments: totals[segment["class"]] += segment["size"]
        authoritative = sum(value for key, value in totals.items() if key in AUTHORITATIVE)
        unresolved = size - authoritative
        ledgers.append({"image": image_name, "section_vram": f"0x{start:X}", "size": size,
                        "classified_bytes": size, "authoritative_bytes": authoritative,
                        "unresolved_bytes": unresolved, "totals": dict(totals), "segments": segments})
        all_segments.extend((image_name, segment) for segment in segments)
    totals = Counter()
    for _image, segment in all_segments: totals[segment["class"]] += segment["size"]
    section_bytes = sum(section["size"] for section in ledgers)
    authoritative_bytes = sum(section["authoritative_bytes"] for section in ledgers)
    return {"model": "byte-ownership-v1", "images": len({key[0] for key in sections}),
            "sections": len(ledgers), "section_bytes": section_bytes,
            "classified_bytes": sum(section["classified_bytes"] for section in ledgers),
            "authoritative_bytes": authoritative_bytes,
            "unresolved_bytes": section_bytes - authoritative_bytes,
            "totals": dict(totals), "problems": problems,
            "complete_partition": all(section["classified_bytes"] == section["size"] for section in ledgers),
            "authoritative_complete": not problems and authoritative_bytes == section_bytes,
            "ledgers": ledgers}


def progress(inventory: dict, ledger: dict) -> dict:
    summary = inventory.get("summary", {})
    verified_functions = sum(row.get("ownership_state") in {"manifest_exact_unit", "manifest_exact_member"}
                             for row in inventory.get("functions", []))
    claimed_functions = sum(row.get("ownership_state") == "manifest_nonexact"
                            for row in inventory.get("functions", []))
    return {"functions": {"catalogue": summary.get("functions"),
                          "declared": summary.get("declared_functions"),
                          "verified_c": verified_functions, "claimed_c": claimed_functions,
                          "denominator_delta": summary.get("denominator_delta")},
            "bytes": {"catalogue_sections": ledger["section_bytes"],
                      "verified_c": ledger["totals"].get("verified_c", 0),
                      "claimed_c": ledger["totals"].get("claimed_c", 0),
                      "assembly": ledger["totals"].get("assembly", 0),
                      "structural": sum(ledger["totals"].get(key, 0)
                                        for key in ("fragment", "padding", "data")),
                      "unresolved": ledger["unresolved_bytes"]},
            "invariants": {"one_class_per_byte": ledger["complete_partition"] and not ledger["problems"],
                           "all_bytes_authoritatively_owned": ledger["authoritative_complete"]}}


# =============================================================================================
# The shared ownership model (decompile-run's data stage; closeout directive section 2)
#
# Three address spaces, never mixed:
#
# * ``rom``      physical ROM offsets;
# * ``image``    one decoded image's link (VRAM) range, per image: overlays that share VRAM are
#                separate images, so their intervals never collide;
# * ``runtime``  allocations an image makes at run time (.bss, .sbss, common symbols): they own
#                no ROM byte and no image byte, and the model refuses to give them any.
#
# An interval is ``start, size, space, kind, owner`` with its evidence.  Its status is computed,
# never given: ``settled`` (strictly verified or declared bytes), ``candidate`` (a hypothesis),
# ``ambiguous`` (another owner's candidate covers the same bytes, or the same object has several
# placements: several layouts explain the bytes and none is preferred) or ``conflict`` (it
# contradicts a settled interval, or two settled intervals overlap, or it is misaligned).
# Nothing here grants credit: ``settled`` records who owns bytes, not that they were decompiled.
# =============================================================================================

SPACES = ("rom", "image", "runtime")
KINDS = ("code", "data", "rodata", "bss", "sbss", "jtbl", "literal-pool", "common", "padding")
RUNTIME_KINDS = frozenset({"bss", "sbss", "common"})
LOADED_DATA_KINDS = frozenset({"data", "rodata", "jtbl", "literal-pool"})
STATUSES = ("settled", "candidate", "ambiguous", "conflict")
# evidence that may settle a placement: the strict verifier compared the bytes, or the project
# (a symbol file, a manifest, a linker map where permitted) declares them
SETTLING = frozenset({"strict", "declared"})
# evidence independent of the unit that needs the placement (data stage)
INDEPENDENT = frozenset({"declared", "alias-declared", "reloc", "data-pointer", "jtbl", "loader",
                         "adjacent-settled"})
ALIGNMENTS = (1, 2, 4, 8, 16)
SECTION_KIND = {".text": "code", ".data": "data", ".sdata": "data", ".rodata": "rodata", ".rdata": "rodata",
                ".lit4": "literal-pool", ".lit8": "literal-pool", ".bss": "bss", ".sbss": "sbss",
                "COMMON": "common", ".scommon": "common"}


def hx(value: int) -> str:
    return f"0x{value:08X}"


def align_up(value: int, align: int) -> int:
    return value if align <= 1 else (value + align - 1) // align * align


def section_kind(name: str) -> str:
    """The interval kind of an object-file section (``.rodata.str1.4`` is ``rodata``)."""
    if name in SECTION_KIND:
        return SECTION_KIND[name]
    for prefix in (".rodata", ".rdata", ".sdata", ".sbss", ".data", ".bss", ".lit"):
        if name.startswith(prefix):
            return SECTION_KIND.get(prefix, "literal-pool")
    return "data"


@dataclass(frozen=True)
class Evidence:
    kind: str           # strict, declared, alias-declared, reloc, data-pointer, jtbl, loader, adjacent-settled,
                        # unit-derived, single-reference, catalogue, inferred, ...
    detail: str = ""
    origin: str = ""    # where it came from: function:0x..., data:0x..., a unit, a file

    @property
    def independent(self) -> bool:
        return self.kind in INDEPENDENT or self.kind in SETTLING

    def as_dict(self) -> dict:
        return {"kind": self.kind, "detail": self.detail, "origin": self.origin}


@dataclass(eq=False)
class Interval:
    start: int
    size: int                       # 0: only the start is known (a placement with no known extent)
    space: str
    kind: str
    owner: str
    image: str = ""                 # empty for the rom space only
    obj: str = ""                   # the logical object (a unit's section): its placements are alternatives
    align: int = 1
    settled: bool = False           # as given: strict or declared evidence
    evidence: list = field(default_factory=list)
    status: str = "candidate"       # computed by OwnershipModel.classify
    notes: list = field(default_factory=list)

    @property
    def end(self) -> int:
        return self.start + self.size

    @property
    def bucket(self) -> tuple[str, str]:
        return self.space, self.image

    def covers(self, address: int) -> bool:
        return self.start <= address < self.end if self.size else address == self.start

    def ref(self) -> dict:
        return {"space": self.space, "image": self.image, "start": hx(self.start), "size": self.size,
                "kind": self.kind, "owner": self.owner, **({"object": self.obj} if self.obj else {})}

    def as_dict(self) -> dict:
        return {**self.ref(), "align": self.align, "status": self.status,
                "evidence": [e.as_dict() for e in self.evidence], **({"notes": self.notes} if self.notes else {})}


def _overlap(a: Interval, b: Interval) -> bool:
    if a.size and b.size:
        return a.start < b.end and b.start < a.end
    if not a.size and not b.size:
        return a.start == b.start
    point, span = (a, b) if not a.size else (b, a)
    return span.covers(point.start)


class OwnershipModel:
    """Intervals in three address spaces with explicit ambiguity.  ``add`` records claims;
    ``classify`` computes every status; ``who_owns`` answers for one address."""

    def __init__(self):
        self.intervals: list[Interval] = []
        self.conflicts: list[dict] = []
        self.ambiguities: list[dict] = []
        self.runtime_extent: dict[str, dict] = {}       # image -> its runtime allocation range, with basis
        self._rom_of: dict = {}                         # image -> address -> rom offset | None
        self._keys: dict = {}
        self._objects: dict = defaultdict(list)         # (owner, obj) -> intervals
        self._owned: dict = defaultdict(list)           # owner -> intervals
        self._sorted: dict = {}                         # bucket -> (starts, intervals, largest size)
        self._ends: dict = defaultdict(list)            # (bucket, end) -> settled intervals
        self._starts: dict = defaultdict(list)          # (bucket, start) -> settled intervals
        self._dirty = True

    # --- building
    def map_image(self, image: str, rom_of) -> None:
        """``rom_of(address)``: the ROM offset of an image address (an int), or anything else when the
        address is not a plain copy of ROM bytes (a compressed segment, no segment)."""
        self._rom_of[image] = rom_of

    def set_runtime_extent(self, image: str, start: int, end: int, basis: str) -> None:
        self.runtime_extent[image] = {"start": hx(start), "end": hx(end), "basis": basis}

    def add(self, start: int, size: int, space: str, kind: str, owner: str, *, image: str = "", obj: str = "",
            evidence=(), settled: bool = False, align: int = 1, note: str = "") -> Interval:
        if space not in SPACES:
            raise ValueError(f"space {space!r}: one of {', '.join(SPACES)}")
        if kind not in KINDS:
            raise ValueError(f"kind {kind!r}: one of {', '.join(KINDS)}")
        if (space == "rom") == bool(image):
            raise ValueError("a rom interval names no image; an image or runtime interval names its image")
        if kind in RUNTIME_KINDS and space != "runtime":
            raise ValueError(f"{kind} is a runtime allocation: it owns no ROM or image bytes")
        if space == "runtime" and kind not in RUNTIME_KINDS:
            raise ValueError(f"{kind} in the runtime space: only {', '.join(sorted(RUNTIME_KINDS))} are allocations")
        if start < 0 or size < 0:
            raise ValueError("negative start or size")
        if align < 1 or align & (align - 1):
            raise ValueError(f"alignment {align} is not a power of two")
        evidence = [e if isinstance(e, Evidence) else Evidence(*e) for e in evidence]
        if settled and not any(e.kind in SETTLING for e in evidence):
            raise ValueError("a settled interval needs strict or declared evidence")
        self._dirty = True
        key = (space, image, start, size, kind, owner, obj)
        found = self._keys.get(key)
        if found is not None:
            found.evidence.extend(e for e in evidence if e not in found.evidence)
            found.settled = found.settled or settled
            found.align = max(found.align, align)
            if note and note not in found.notes:
                found.notes.append(note)
            return found
        item = Interval(start, size, space, kind, owner, image, obj, align, settled, evidence,
                        notes=[note] if note else [])
        self.intervals.append(item)
        self._keys[key] = item
        self._owned[owner].append(item)
        if obj:
            self._objects[(owner, obj)].append(item)
        return item

    # --- statuses
    def _mark(self, item: Interval, status: str) -> None:
        if item.status == "conflict":
            return
        if status == "conflict" or item.status != "ambiguous":
            item.status = status

    def classify(self) -> "OwnershipModel":
        self.conflicts, self.ambiguities = [], []
        self._sorted, self._ends, self._starts = {}, defaultdict(list), defaultdict(list)
        for item in self.intervals:
            item.status = "settled" if item.settled else "candidate"
        for item in self.intervals:
            if item.start % item.align:
                self._mark(item, "conflict")
                self.conflicts.append({"type": "misaligned", "interval": item.ref(),
                                       "message": f"{hx(item.start)} is not {item.align}-aligned"})
        buckets: dict = defaultdict(list)
        for item in self.intervals:
            buckets[item.bucket].append(item)
        for bucket, items in buckets.items():
            items.sort(key=lambda i: (i.start, i.size))
            active: list[Interval] = []
            for item in items:
                active = [a for a in active if a.end > item.start or a.start == item.start]
                for other in active:
                    if other.owner != item.owner and _overlap(other, item):
                        self._pair(other, item)
                active.append(item)
            self._sorted[bucket] = ([i.start for i in items], items, max((i.size for i in items), default=0))
        for (owner, obj), items in self._objects.items():
            places = {(i.bucket, i.start) for i in items}
            if len(places) < 2:
                continue
            settled_places = {(i.bucket, i.start) for i in items if i.settled}
            for item in items:
                if item.settled:
                    if len(settled_places) > 1:      # two verified placements of one object
                        self._mark(item, "conflict")
                elif settled_places:                 # another placement of the object is settled
                    self._mark(item, "conflict")
                else:                                # several layouts, none preferred
                    self._mark(item, "ambiguous")
            entry = {"owner": owner, "object": obj,
                     "places": sorted(f"{space}:{image}:{hx(start)}" for (space, image), start in places)}
            if settled_places:
                self.conflicts.append({"type": "settled-alternatives" if len(settled_places) > 1
                                       else "contradicts-settled", **entry})
            else:
                self.ambiguities.append({"type": "alternative-layouts", **entry})
        for item in self.intervals:
            if item.status == "settled" and item.size:
                self._ends[(item.bucket, item.end)].append(item)
                self._starts[(item.bucket, item.start)].append(item)
        self._dirty = False
        return self

    def _pair(self, a: Interval, b: Interval) -> None:
        if a.settled and b.settled:
            self._mark(a, "conflict")
            self._mark(b, "conflict")
            self.conflicts.append({"type": "settled-overlap", "intervals": [a.ref(), b.ref()]})
        elif a.settled or b.settled:
            settled, other = (a, b) if a.settled else (b, a)
            self._mark(other, "conflict")
            self.conflicts.append({"type": "contradicts-settled", "interval": other.ref(), "settled": settled.ref()})
        else:
            self._mark(a, "ambiguous")
            self._mark(b, "ambiguous")
            self.ambiguities.append({"type": "overlap", "intervals": [a.ref(), b.ref()]})

    def _ready(self) -> None:
        if self._dirty:
            self.classify()

    # --- questions
    def _covering(self, space: str, image: str, address: int) -> list[Interval]:
        self._ready()
        starts, items, largest = self._sorted.get((space, image), ([], [], 0))
        hi = bisect.bisect_right(starts, address)
        lo = bisect.bisect_left(starts, address - largest)
        return [i for i in items[lo:hi] if i.covers(address)]

    def rom_span(self, item: Interval) -> tuple[int, int] | None:
        """The ROM bytes an interval owns: None for runtime allocations (they own none) and for
        image ranges that are not a plain copy of ROM bytes."""
        if item.space == "rom":
            return item.start, item.end
        if item.space != "image":
            return None
        rom_of = self._rom_of.get(item.image)
        first = rom_of(item.start) if rom_of else None
        if not isinstance(first, int) or isinstance(first, bool):
            return None
        if item.size and rom_of(item.end - 1) != first + item.size - 1:
            return None
        return first, first + item.size

    def who_owns(self, address: int, space: str = "image", image: str = "") -> dict:
        """Every interval that covers ``address`` and a status: ``unowned``, ``settled``,
        ``candidate``, ``ambiguous`` (several owners or layouts, none preferred) or ``conflict``."""
        if space not in SPACES:
            raise ValueError(f"space {space!r}: one of {', '.join(SPACES)}")
        hits = self._covering(space, "" if space == "rom" else image, address)
        if space == "rom":
            for item in self.intervals:
                span = self.rom_span(item) if item.space == "image" else None
                if span and (span[0] <= address < span[1] if item.size else address == span[0]):
                    hits.append(item)
        owners = sorted({i.owner for i in hits})
        if not hits:
            status = "unowned"
        elif any(i.status == "conflict" for i in hits):
            status = "conflict"
        elif len(owners) > 1 or any(i.status == "ambiguous" for i in hits):
            status = "ambiguous"
        elif any(i.status == "settled" for i in hits):
            status = "settled"
        else:
            status = "candidate"
        return {"address": hx(address), "space": space, "image": image if space != "rom" else "", "status": status,
                "owners": owners, "intervals": [i.as_dict() for i in sorted(hits, key=lambda i: (i.start, i.size))]}

    def placements(self, owner: str, obj: str) -> list[Interval]:
        """Every placement recorded for one logical object (a unit's section)."""
        self._ready()
        return list(self._objects.get((owner, obj), ()))

    def owned_by(self, owner: str) -> list[Interval]:
        return list(self._owned.get(owner, ()))

    def settled_neighbours(self, image: str, start: int, end: int, kinds=LOADED_DATA_KINDS, zero=None,
                           exclude: str = "") -> list[Evidence]:
        """Settled intervals of ``kinds`` (another owner's) that end where ``[start, end)`` begins, or
        begin where it ends, at an alignment and over zero bytes only (``zero(lo, hi)``): the
        archive order of the neighbours' own verified bytes, independent of the unit in between."""
        self._ready()
        bucket, out = ("image", image), []
        for gap in range(0, 16):
            for item in self._ends.get((bucket, start - gap), ()):
                if item.kind in kinds and item.owner != exclude \
                        and any(align_up(item.end, a) == start for a in ALIGNMENTS) \
                        and (zero is None or zero(item.end, start)):
                    out.append(Evidence("adjacent-settled", f"follows {item.owner}'s {item.kind} ending at "
                                                            f"{hx(item.end)}", item.owner))
            for item in self._starts.get((bucket, end + gap), ()):
                if item.kind in kinds and item.owner != exclude \
                        and any(align_up(end, a) == item.start and item.start % a == 0 for a in ALIGNMENTS) \
                        and (zero is None or zero(end, item.start)):
                    out.append(Evidence("adjacent-settled", f"precedes {item.owner}'s {item.kind} starting at "
                                                            f"{hx(item.start)}", item.owner))
        return out

    def summary(self) -> dict:
        self._ready()
        by_status = Counter(i.status for i in self.intervals)
        by_kind = Counter(f"{i.space}:{i.kind}" for i in self.intervals)
        interval_bytes: dict = defaultdict(Counter)
        for item in self.intervals:
            interval_bytes[item.space][item.status] += item.size
        settled_rom = sum((span[1] - span[0]) for i in self.intervals if i.status == "settled"
                          for span in [self.rom_span(i)] if span)
        return {"intervals": len(self.intervals), "by_status": dict(by_status), "by_kind": dict(sorted(by_kind.items())),
                "interval_bytes": {space: dict(c) for space, c in sorted(interval_bytes.items())},
                "settled_rom_bytes": settled_rom,
                # runtime allocations own no ROM byte: an invariant, reported so a reader can check it
                "runtime_intervals_with_rom_bytes": sum(1 for i in self.intervals
                                                        if i.space == "runtime" and self.rom_span(i)),
                "conflicts": len(self.conflicts), "ambiguities": len(self.ambiguities)}

    def to_dict(self, limit: int | None = None) -> dict:
        self._ready()
        items = sorted(self.intervals, key=lambda i: (SPACES.index(i.space), i.image, i.start, i.size, i.owner))
        return {"model": "ownership-v2", "spaces": list(SPACES), "summary": self.summary(),
                "runtime_extent": self.runtime_extent,
                "conflicts": self.conflicts[:limit], "ambiguities": self.ambiguities[:limit],
                "intervals": [i.as_dict() for i in items]}


# --------------------------------------------------------------------------------- _gp provenance

def gp_provenance(declared: int | None, inferred: dict | None, declared_source: str = "project symbols") -> dict:
    """``_gp`` as the project declares it and as the code that sets ``$gp`` implies it
    (:func:`n64dk.analysis.gp.find`), kept apart.  ``value`` is what evidence may use: the declared
    value when there is one and nothing contradicts it, the inferred value only when it is the sole
    one and of high confidence.  A disagreement is ``conflict`` with no value: it is reported, never
    resolved by preferring one side."""
    inferred = inferred or None
    record = {"declared": {"value": hx(declared), "source": declared_source} if declared is not None else None,
              "inferred": None, "status": "none", "value": None, "basis": None, "conflict": None, "sites": {}}
    value = None
    if inferred is not None:
        raw = inferred.get("value")
        value = int(str(raw), 16) if raw else None
        record["inferred"] = {"value": raw, "confidence": inferred.get("confidence"),
                              "reason": inferred.get("reason"), "sites": sorted(inferred.get("sites") or {})}
        for site in inferred.get("sites") or {}:
            image, _sep, address = str(site).rpartition(":")
            try:
                record["sites"].setdefault(image, []).append(int(address, 16))
            except ValueError:
                continue
    if declared is not None and value is not None and declared != value:
        record.update(status="conflict", conflict=f"the project declares _gp = {hx(declared)}; the code that sets "
                                                  f"$gp implies {hx(value)}: neither is used as evidence")
    elif declared is not None:
        record.update(status="agree" if value is not None else "declared-only", value=declared, basis="declared")
    elif value is not None and (inferred or {}).get("confidence") == "high":
        record.update(status="inferred-only", value=value, basis="inferred")
    elif value is not None:
        record["status"] = "inferred-low"
    return record


def gp_independent_of(provenance: dict, image: str, spans) -> bool:
    """Whether ``_gp`` may serve as evidence for a unit whose code is ``spans`` ((start, size)): a
    declared value always; an inferred one only when no site that sets ``$gp`` lies in the unit
    (otherwise the unit's own instructions would establish the base its comparison needs)."""
    if provenance.get("value") is None:
        return False
    if provenance.get("basis") == "declared":
        return True
    sites = (provenance.get("sites") or {}).get(image, ())
    return not any(start <= site < start + size for site in sites for start, size in spans)
