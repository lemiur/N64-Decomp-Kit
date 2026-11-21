"""decompile-run's opt-in ``data`` stage: the data references that keep units waiting, grouped and
placed from evidence independent of each waiting unit.

A unit (a resolver case, a donor transplant row) whose code is byte-identical but whose strict
comparison still needed an address read from its own instructions waits (case ``assumed``; donor
``identical-awaiting-symbols``).  That address is the question, never the answer: a retail
instruction under comparison cannot also establish the address that makes the comparison exact.
For one image this stage

1. builds the shared ownership model (:class:`n64dk.match.ownership.OwnershipModel`): the code of
   every unit and catalogued function; the sections strictly verified units placed (settled); the
   sections waiting units would place (candidates, alternatives of one object kept apart); objects
   one function alone references (:mod:`n64dk.analysis.datasolve`, candidates); the image's
   runtime allocation range (BSS owns no ROM byte);
2. turns every waiting unit's data needs into requests and groups them by shared symbol (per image)
   or by section and address (per image): one group per question.  Calls are not data needs;
3. answers each group only from evidence independent of each unit in it: the project's explicit
   symbols (the name, or another name, at the address); ``%hi``/``%lo`` and ``$gp``-relative
   references from functions outside the unit (``$gp``-relative only with a ``_gp`` the unit's own
   code does not establish, :func:`n64dk.match.ownership.gp_provenance`); data words outside the
   unit's own bytes that hold the address; switch tables of other functions; loader records
   (segment starts); and, for a section, settled neighbours that end where it begins or begin where
   it ends, at an alignment, over zero bytes (archive order).  Section constraints (the region, code,
   past the image, inside a definite switch table) and data widths (alignment) filter;
4. lists every waiting unit with what it waits for, and marks for a strict batch replay those whose
   every data need is established and that the resolver can re-verify with a symbol file.

Group statuses: ``established``, ``awaiting-evidence``, ``ambiguous`` (the unit's placement overlaps
another owner's candidate, or several placements of one object are recorded, or the address lies in
catalogued code that is not settled), ``conflict`` (contradicts a declaration, a settled interval or
a section constraint; or the units read different addresses for one symbol), ``outside-image``
(another image holds the address; several do when overlays overlap) and ``code-reference`` (the
address of a function, taken rather than called: not data).  Nothing here is verified and
nothing grants credit: an established placement only lets the strict verifier compare a unit with
that address declared, and the verifier's verdict is the only result.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import ownership as own
from .ownership import Evidence, hx

# n64dk.match.transplant's verdicts (the stage reads the donor report; it builds nothing)
DONOR_EXACT = "exact-with-transferred-symbols"
DONOR_ASSUMED = "identical-awaiting-symbols"
DATA_STAGE_MARK = "// data-stage:"
CLAIM_NOTE = ("placements are hypotheses until the strict verifier compares a unit with them declared; an "
              "address read from a unit's own instructions is the question, never the evidence; nothing here "
              "grants credit")
_OBJECT_KIND = {"jtbl": "jtbl", "jtbl-shape": "jtbl", "f32": "literal-pool", "f64": "literal-pool",
                "string": "rodata", "sjis": "rodata"}
_DONOR_FIELDS = ("target_vram", "size", "verdict", "donor", "donor_unit", "functions", "awaiting")


def _int(value) -> int:
    return value if isinstance(value, int) else int(str(value), 0)


def _inside(spans, lo: int, hi: int) -> bool:
    """``[lo, hi)`` lies inside one of ``spans`` (sorted, disjoint ``(start, end)`` pairs)."""
    i = bisect.bisect_right(spans, (lo, float("inf"))) - 1
    return i >= 0 and spans[i][0] <= lo and hi <= spans[i][1]


def case_unit(image: str, vram: int) -> str:
    return f"case:{image}:{hx(vram)}"


def donor_unit(image: str, vram: int) -> str:
    return f"donor:{image}:{hx(vram)}"


# --------------------------------------------------------------------------------- needs

@dataclass(frozen=True)
class Need:
    """One address a waiting unit's strict comparison read from its own instructions."""
    unit: str
    origin: str                 # case | donor
    image: str
    ref: str                    # symbol | section
    name: str
    address: int                # the question, never evidence
    size: int = 0               # a section's size in the unit's object
    spans: tuple = ()           # the unit's code: (start, size)
    call: bool = False
    label: str = ""             # the case's function or the donor unit

    @property
    def key(self) -> tuple:
        if self.ref == "section":
            return self.image, "section", self.name, self.address
        return self.image, "symbol", self.name


def _needs(unit: str, origin: str, image: str, strict, awaiting, spans, label: str) -> list[Need]:
    strict = strict or {}
    sizes = {row["section"]: int(row.get("size") or 0) for row in strict.get("sections") or [] if row.get("section")}
    items = strict.get("assumptions") or [
        {"kind": "section" if str(name).startswith(".") else "symbol", "name": name, "address": address}
        for name, address in (awaiting or {}).items()]
    out = []
    for item in items:
        try:
            address = _int(item.get("address"))
        except (TypeError, ValueError):
            continue
        ref = "section" if item.get("kind") == "section" else "symbol"
        name = str(item.get("name"))
        out.append(Need(unit, origin, image, ref, name, address, sizes.get(name, 0) if ref == "section" else 0,
                        tuple(spans), bool(item.get("call")), label))
    return out


def needs_from_case(case) -> list[Need]:
    if case.status != "assumed":
        return []
    return _needs(case_unit(case.image, case.vram), "case", case.image, case.strict, None,
                  [(case.vram, case.size)], case.name)


def needs_from_donor(row: dict, image: str) -> list[Need]:
    if row.get("verdict") != DONOR_ASSUMED:
        return []
    vram = _int(row["target_vram"])
    return _needs(donor_unit(image, vram), "donor", image, row.get("strict"), row.get("awaiting"),
                  [(vram, int(row.get("size") or 0))], f"{row.get('donor')}:{row.get('donor_unit')}")


@dataclass
class Group:
    key: tuple
    needs: list = field(default_factory=list)
    status: str = "awaiting-evidence"
    reasons: list = field(default_factory=list)
    support: dict = field(default_factory=dict)          # unit -> [Evidence]
    own_references: dict = field(default_factory=dict)   # unit -> references only its own bytes make
    region: str = ""
    homes: list = field(default_factory=list)
    basis: str = ""
    mutual: bool = False        # supported only by instructions of other units waiting on the same question

    @property
    def image(self) -> str:
        return self.key[0]

    @property
    def ref(self) -> str:
        return self.key[1]

    @property
    def name(self) -> str:
        return self.key[2]

    @property
    def addresses(self) -> list[int]:
        return sorted({n.address for n in self.needs})

    @property
    def units(self) -> list[str]:
        return sorted({n.unit for n in self.needs})

    def as_dict(self) -> dict:
        return {"image": self.image, "ref": self.ref, "name": self.name,
                "addresses": [hx(a) for a in self.addresses], "status": self.status, "region": self.region,
                "size": max((n.size for n in self.needs), default=0), "units": self.units,
                "reasons": self.reasons, "basis": self.basis, "mutual": self.mutual,
                "support": {unit: [e.as_dict() for e in found] for unit, found in sorted(self.support.items())},
                "own_references": dict(sorted(self.own_references.items())),
                **({"homes": self.homes} if self.homes else {})}


def group_needs(needs) -> list[Group]:
    """One group per question: a symbol per image, or a section at one address per image (a
    section is the unit's own; two units placing theirs at one address are two claims on the same
    bytes, which the ownership model reports)."""
    groups: dict[tuple, Group] = {}
    for need in needs:
        if not need.call:
            groups.setdefault(need.key, Group(need.key)).needs.append(need)
    return [groups[key] for key in sorted(groups, key=lambda k: tuple(str(part) for part in k))]


# --------------------------------------------------------------------------------- the model

def _sections(model: own.OwnershipModel, image: str, unit: str, strict, exact: bool) -> None:
    """The data sections one unit's strict comparison placed: settled when the unit is exact and
    the base was not read from its own instructions, otherwise candidates of that unit's object."""
    for row in (strict or {}).get("sections") or []:
        name, base = row.get("section"), row.get("base")
        if not name or not base:
            continue
        kind = own.section_kind(name)
        if kind in own.RUNTIME_KINDS or kind == "code":
            continue
        start = _int(base)
        if row.get("status") == "equal":
            size = int(row.get("compared") or row.get("size") or 0)
            if exact and not row.get("base_assumed"):
                model.add(start, size, "image", kind, unit, image=image, obj=name, settled=True,
                          evidence=[Evidence("strict", f"{name}: {size} bytes equal at {hx(start)}", unit)])
            else:
                model.add(start, size, "image", kind, unit, image=image, obj=name,
                          evidence=[Evidence("unit-derived", f"{name} equal at the base its own instructions imply",
                                             unit)])
        elif row.get("status") == "unverified" and row.get("first_diff"):
            size = _int(row["first_diff"])
            model.add(start, size, "image", kind, unit, image=image, obj=name, note="partial",
                      evidence=[Evidence("unit-derived", f"{name}: equal for {size} bytes, zero padding after where "
                                                         "the image differs", unit)])


def build_model(image: str, *, rows, cases=(), donor_rows=(), manifest_rows=(), objects=(), bss=None,
                bss_basis: str = "", rom_of=None) -> own.OwnershipModel:
    """The image's ownership model.  ``manifest_rows``: ``(start, size, source, cls)``; ``objects``:
    :class:`n64dk.analysis.datasolve.Row` (only those one function alone references are used)."""
    model = own.OwnershipModel()
    if rom_of is not None:
        model.map_image(image, rom_of)
    if bss:
        model.set_runtime_extent(image, bss[0], bss[1], bss_basis)
    units: list[tuple[int, int, str]] = []
    for row in donor_rows:
        if row.get("verdict") not in (DONOR_EXACT, DONOR_ASSUMED):
            continue
        vram, size = _int(row["target_vram"]), int(row.get("size") or 0)
        unit, exact = donor_unit(image, vram), row.get("verdict") == DONOR_EXACT
        label = f"{row.get('donor')}:{row.get('donor_unit')}"
        if size:
            model.add(vram, size, "image", "code", unit, image=image, obj=".text", settled=exact,
                      evidence=[Evidence("strict", f"donor unit verified strictly: {label}", unit) if exact else
                                Evidence("unit-derived", f"donor unit byte-identical, awaiting: {label}", unit)])
            units.append((vram, vram + size, unit))
        _sections(model, image, unit, row.get("strict"), exact)
    taken = sorted(units)
    for case in cases:
        i = bisect.bisect_right(taken, (case.vram, float("inf"), "")) - 1
        if i >= 0 and taken[i][0] <= case.vram < taken[i][1]:
            continue                                         # a donor unit covers it: the unit is the owner
        unit, exact = case_unit(case.image, case.vram), case.status == "exact"
        kind = "strict" if exact else "unit-derived" if case.status == "assumed" else "catalogue"
        if case.size:
            model.add(case.vram, case.size, "image", "code", unit, image=image, obj=".text", settled=exact,
                      evidence=[Evidence(kind, f"case {case.name}: {case.status}", unit)])
            units.append((case.vram, case.vram + case.size, unit))
        if case.status in ("exact", "assumed"):
            _sections(model, image, unit, case.strict, exact)
    for start, size, source, cls in manifest_rows:
        if size <= 0:
            continue
        unit = f"manifest:{source}"
        model.add(start, size, "image", "data" if cls == "data" else "code", unit, image=image, settled=True,
                  evidence=[Evidence("declared", "project manifest row", source)])
        units.append((start, start + size, unit))
    units.sort()
    starts = [u[0] for u in units]

    def owner_of(address: int) -> str | None:
        i = bisect.bisect_right(starts, address) - 1
        for j in range(i, max(-1, i - 8), -1):              # units barely overlap: a short walk back
            if units[j][0] <= address < units[j][1]:
                return units[j][2]
        return None
    names = {}
    for name, vram, size in rows:
        names[vram] = name
        if size > 0 and owner_of(vram) is None:
            model.add(vram, size, "image", "code", f"function:{name}", image=image,
                      evidence=[Evidence("catalogue", name, "catalogue")])
    for row in objects:
        fn = row.owner_function
        if fn is None or row.min_size <= 0:
            continue
        owner = owner_of(fn) or f"function:{names.get(fn, hx(fn))}"
        found = Evidence("single-reference", f"only the function at {hx(fn)} references it ({row.kind})",
                         f"function:{hx(fn)}")
        if row.region == "bss":
            kind = "sbss" if any(str(e).startswith("gprel") for e in row.evidence) else "bss"
            model.add(row.start, row.min_size, "runtime", kind, owner, image=image, evidence=[found])
        else:
            kind = _OBJECT_KIND.get(row.kind, "data")
            model.add(row.start, row.min_size, "image", kind, owner, image=image, evidence=[found])
    return model.classify()


# --------------------------------------------------------------------------------- placement

@dataclass
class Context:
    image: str
    claims: dict                                    # address -> [datasolve.Claim]
    regions: list                                   # initialised non-code ranges, sorted
    code: list                                      # catalogued code spans, sorted
    bss: tuple | None = None
    declared: dict = field(default_factory=dict)    # explicit project symbols: name -> address
    provenance: dict = field(default_factory=dict)  # ownership.gp_provenance
    loader_starts: frozenset = frozenset()
    homes: dict = field(default_factory=dict)       # every other image -> (start, end)
    tables: list = field(default_factory=list)      # definite switch tables: (start, end)
    zero: object = None                             # (lo, hi) -> the image's bytes there are all zero

    def __post_init__(self):
        self.declared_at: dict[int, set] = defaultdict(set)
        for name, address in self.declared.items():
            self.declared_at[address].add(name)

    def region_of(self, address: int) -> str:
        if _inside(self.regions, address, address + 1):
            return "data"
        if self.bss and self.bss[0] <= address < self.bss[1]:
            return "bss"
        if _inside(self.code, address, address + 1):
            return "code"
        return "outside"


def place(group: Group, ctx: Context, model: own.OwnershipModel) -> Group:
    """Answer one group from independent evidence (see the module docstring)."""
    addresses = group.addresses
    if len(addresses) > 1:
        group.status = "conflict"
        group.reasons.append("the waiting units read different addresses for this symbol: "
                             + ", ".join(hx(a) for a in addresses))
        return group
    x = addresses[0]
    section = group.ref == "section"
    size = max((n.size for n in group.needs), default=0) if section else 0
    group.region = ctx.region_of(x)
    conflicts: list[str] = []
    ambiguous: list[str] = []
    declared = None if section else ctx.declared.get(group.name)
    if declared is not None and declared != x:
        conflicts.append(f"the project declares {group.name} at {hx(declared)}")
    if group.region == "outside" and not conflicts:
        group.homes = sorted(name for name, (lo, hi) in ctx.homes.items() if lo <= x < hi)
        group.status = "outside-image"
        group.reasons.append(f"outside this image; its evidence is in {', '.join(group.homes)}" if group.homes
                             else "outside this image and every other known image")
        if len(group.homes) > 1:
            group.reasons.append("several images map this address (overlapping overlays): which one holds it "
                                 "is ambiguous")
        return group
    if group.region == "code":
        here = [i for i in model.who_owns(x, "image", ctx.image)["intervals"] if i["kind"] == "code"]
        if not section and not conflicts and any(i["start"] == hx(x) for i in here):
            group.status = "code-reference"
            group.reasons.append("the address of a function (taken, not called): this stage places data only")
            return group
        settled_code = any(i["status"] == "settled" for i in here)
        if settled_code:
            conflicts.append("inside strictly verified or declared code")
        else:
            ambiguous.append("inside catalogued code that is not verified: code or data")
    if section:
        runtime = own.section_kind(group.name) in own.RUNTIME_KINDS
        wanted = "bss" if runtime else "data"
        if group.region in ("data", "bss") and group.region != wanted:
            conflicts.append(f"{group.name} would lie in " + ("initialised bytes" if runtime else
                                                             "the runtime allocation range, which has no ROM bytes"))
        elif size and group.region == wanted and not _inside([ctx.bss] if runtime else ctx.regions, x, x + size):
            conflicts.append(f"{group.name} ({size} bytes) would run past its region, into code or past the image")
        for need in group.needs:
            for item in model.placements(need.unit, group.name):
                if item.start != x:
                    continue
                if item.status == "conflict":
                    conflicts.append(f"{need.unit}: its {group.name} contradicts a settled interval")
                elif item.status == "ambiguous":
                    ambiguous.append(f"{need.unit}: another owner's candidate, or another placement of its "
                                     f"{group.name}, explains the same bytes")
    else:
        for lo, hi in ctx.tables:
            if lo < x < hi:
                conflicts.append(f"inside the definite switch table at {hx(lo)}")
    skipped: Counter = Counter()
    supporters: set = set()
    for need in group.needs:
        mine = list(need.spans) + [(i.start, i.size) for i in model.owned_by(need.unit) if i.size]

        def owned(address, mine=mine) -> bool:
            return address is not None and any(s <= address < s + n for s, n in mine)
        support, own_refs = [], 0
        if declared is not None and declared == x:
            support.append(Evidence("declared", f"{group.name} = {hx(x)}", "project symbols"))
        if not section:
            for name in sorted(ctx.declared_at.get(x, set()) - {group.name, "_gp"}):
                support.append(Evidence("alias-declared", f"{name} = {hx(x)}", "project symbols"))
        for claim in ctx.claims.get(x, ()):
            if claim.source == "pointer":
                if owned(claim.site):
                    own_refs += 1
                else:
                    support.append(Evidence("data-pointer", f"the word at {hx(claim.site)} holds {hx(x)}",
                                            f"data:{hx(claim.site)}"))
                continue
            if claim.function is None or owned(claim.function) or owned(claim.site):
                own_refs += 1
                continue
            if claim.source == "gprel" and not own.gp_independent_of(ctx.provenance, ctx.image, need.spans):
                skipped["gp-relative reference without a usable _gp"] += 1
                continue
            if claim.source != "jtbl" and claim.width in (2, 4, 8) and x % claim.width:     # an access width
                skipped["misaligned for its access width"] += 1
                continue
            supporters.add(claim.function)
            where = f" at {hx(claim.site)}" if claim.site is not None else ""
            support.append(Evidence("jtbl" if claim.source == "jtbl" else "reloc",
                                    f"{claim.source} {claim.kind} ({claim.access}, {claim.width} bytes){where}",
                                    f"function:{hx(claim.function)}"))
        if x in ctx.loader_starts:
            support.append(Evidence("loader", f"a segment starts at {hx(x)}", "image registry"))
        if section and size:
            support += model.settled_neighbours(ctx.image, x, x + size, zero=ctx.zero, exclude=need.unit)
        group.support[need.unit] = list(dict.fromkeys(support))
        if own_refs:
            group.own_references[need.unit] = own_refs
    group.reasons += [f"{count} reference(s) not used: {why}" for why, count in sorted(skipped.items())]
    if conflicts:
        group.status = "conflict"
        group.reasons += conflicts
    elif ambiguous:
        group.status = "ambiguous"
        group.reasons += ambiguous
    elif all(group.support.get(need.unit) for need in group.needs):
        group.status = "established"
        kinds = Counter(e.kind for found in group.support.values() for e in found)
        group.basis = ", ".join(f"{kind} x{count}" for kind, count in sorted(kinds.items()))
        only_code = all(e.kind in ("reloc", "jtbl") for found in group.support.values() for e in found)
        group.mutual = only_code and bool(supporters) and all(
            any(s <= fn < s + n for need in group.needs for s, n in need.spans) for fn in supporters)
        if group.mutual:
            group.reasons.append("supported only by the instructions of other units waiting on the same question")
    else:
        missing = [need.unit for need in group.needs if not group.support.get(need.unit)]
        group.reasons.append(f"no evidence independent of {len(missing)} unit(s)"
                             + (" (only their own instructions refer to the address)"
                                if any(group.own_references.get(u) for u in missing) else ""))
    return group


def replay_list(needs, groups, functions: dict | None = None) -> list[dict]:
    """Every waiting unit with what it waits for; ``replay``: ``resolve`` (every data need is
    established and the resolver can re-verify the case with a symbol file), ``blocked`` (every
    data need is established, but no replay path exists for it yet: ``requirement`` says what is
    missing) or ``waiting``."""
    by_key = {g.key: g for g in groups}
    by_unit: dict[str, list[Need]] = defaultdict(list)
    for need in needs:
        by_unit[need.unit].append(need)
    out = []
    for unit, items in sorted(by_unit.items()):
        data = [n for n in items if not n.call]
        if not data:
            continue
        first = items[0]
        found = [by_key[n.key] for n in data]
        entry = {"unit": unit, "origin": first.origin, "label": first.label, "image": first.image,
                 "code": [[hx(s), n] for s, n in first.spans], "functions": (functions or {}).get(unit, 1),
                 "needs": len(data), "established": sum(g.status == "established" for g in found),
                 "calls": sum(1 for n in items if n.call),
                 "blocking": sorted({f"{g.ref} {g.name}: {g.status}" for g in found if g.status != "established"})}
        if entry["established"] < len(data):
            entry["replay"] = "waiting"
        elif entry["calls"]:
            entry.update(replay="blocked", requirement="the unit also waits for call targets, which this stage "
                                                       "does not place")
        elif any(n.ref == "section" for n in data):
            entry.update(replay="blocked", requirement="a declared section base for the unit: the resolver and the "
                                                       "donor stage read section bases from manifests only")
        elif first.origin == "donor":
            entry.update(replay="blocked", requirement="donor re-verification with declared data symbols "
                                                       "(transplant.run takes none)")
        else:
            entry["replay"] = "resolve"
        if entry["established"] == len(data):
            entry["declare"] = {n.name: hx(n.address) for n in data if n.ref == "symbol"}
            entry["sections"] = {n.name: hx(n.address) for n in data if n.ref == "section"}
            entry["why"] = {n.name: by_key[n.key].basis for n in data}
            entry["mutual"] = any(by_key[n.key].mutual for n in data)
            entry["digest"] = hashlib.sha256(json.dumps([entry["declare"], entry["sections"]],
                                                        sort_keys=True).encode()).hexdigest()[:16]
        out.append(entry)
    return out


# --------------------------------------------------------------------------------- the stage

def run_image(*, image: str, data: bytes, delta: int, rows, cases=(), donor_rows=(), manifest_rows=(),
              declared: dict | None = None, provenance: dict | None = None, loader_starts=(), homes=None,
              rom_of=None, extent=None, bss=None, tables=(), evidence=None) -> dict:
    """The data stage of one image, as a JSON-ready document (``write`` stores it).  ``evidence``:
    a :class:`n64dk.analysis.datasolve.Evidence` (collected here when None)."""
    from ..analysis import datasolve
    provenance = provenance or own.gp_provenance(None, None)
    gp_ok = provenance.get("value") is not None
    if evidence is None:
        evidence = datasolve.collect(data, delta, rows, extent=extent, bss=bss,
                                     gp=provenance["value"] if gp_ok else None, tables=tables)
    solution = datasolve.partition(evidence, tuple(s for s in datasolve.SOURCES if gp_ok or s != "gprel"))
    claims: dict[int, list] = defaultdict(list)
    for claim in evidence.claims:
        if claim.source != "gprel" or gp_ok:
            claims[claim.address].append(claim)
    definite = sorted((c.address, c.address + c.width) for c in evidence.claims if c.source == "jtbl" and c.exact)
    model = build_model(image, rows=rows, cases=cases, donor_rows=donor_rows, manifest_rows=manifest_rows,
                        objects=solution.rows, bss=evidence.bss, bss_basis=evidence.bss_basis, rom_of=rom_of)

    def zero(lo: int, hi: int) -> bool:
        a, b = lo - delta, hi - delta
        return 0 <= a <= b <= len(data) and not any(data[a:b])
    ctx = Context(image, claims, list(evidence.regions), list(evidence.code), evidence.bss, dict(declared or {}),
                  provenance, frozenset(loader_starts), dict(homes or {}), definite, zero)
    needs = [n for case in cases for n in needs_from_case(case)]
    needs += [n for row in donor_rows for n in needs_from_donor(row, image)]
    groups = [place(group, ctx, model) for group in group_needs(needs)]
    functions = {donor_unit(image, _int(r["target_vram"])): len(r.get("functions") or ()) or 1
                 for r in donor_rows if r.get("verdict") == DONOR_ASSUMED}
    units = replay_list(needs, groups, functions)
    ready = [u for u in units if u["established"] == u["needs"]]
    summary = {"needs": len(needs), "call_needs": sum(1 for n in needs if n.call), "groups": len(groups),
               "groups_by_status": dict(Counter(g.status for g in groups)),
               "groups_mutual_only": sum(1 for g in groups if g.status == "established" and g.mutual),
               "units": len(units), "units_by_replay": dict(Counter(u["replay"] for u in units)),
               "units_fully_established": len(ready),
               "functions_waiting": {origin: sum(u["functions"] for u in units if u["origin"] == origin)
                                     for origin in ("case", "donor")},
               "functions_fully_established": {origin: sum(u["functions"] for u in ready if u["origin"] == origin)
                                               for origin in ("case", "donor")},
               "gp": provenance.get("status"), "ownership": model.summary()}
    return {"model": "data-ownership-v1", "image": image, "claim": CLAIM_NOTE, "gp": provenance,
            "evidence": {"extent": [hx(a) for a in evidence.extent], "regions": len(evidence.regions),
                         "bss": [hx(a) for a in evidence.bss] if evidence.bss else None,
                         "bss_basis": evidence.bss_basis,
                         "claims": dict(Counter(c.source for c in evidence.claims)),
                         "definite_tables": len(definite),
                         "gp_relative": "used" if gp_ok else f"not used: _gp {provenance.get('status')}"},
            "summary": summary, "groups": [g.as_dict() for g in groups], "units": units,
            "ownership": model.to_dict()}


def _atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def write(folder: Path, document: dict) -> dict:
    """``<folder>/data_ownership.json`` (the document) and ``<folder>/data_replay.json`` (the units);
    returns the stage's checkpoint record."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path, replay = folder / "data_ownership.json", folder / "data_replay.json"
    _atomic(path, json.dumps(document, indent=1, sort_keys=True, default=str) + "\n")
    _atomic(replay, json.dumps(document["units"], indent=1, sort_keys=True) + "\n")
    return {"summary": document["summary"], "file": str(path), "replay_file": str(replay),
            "replay_units": sum(1 for u in document["units"] if u.get("replay") == "resolve")}


def load_units(state: Path, image: str, donor_report: Path) -> tuple[list, list[dict]]:
    """The image's cases (read-only) and the donor stage's exact and waiting rows (the fields this
    stage reads only: the donor report also holds near hits and notes)."""
    cases: list = []
    if Path(state).is_dir():
        from . import resolve as resolvemod
        store = resolvemod.CaseStore(state, readonly=True)
        try:
            cases = [case for case in store.all() if case.image == image]
        finally:
            store.close()
    rows: list[dict] = []
    if Path(donor_report).is_file():
        for row in json.loads(Path(donor_report).read_text()).get("rows") or []:
            if row.get("verdict") in (DONOR_EXACT, DONOR_ASSUMED):
                strict = row.get("strict") or {}
                rows.append({**{k: row.get(k) for k in _DONOR_FIELDS},
                             "strict": {"assumptions": strict.get("assumptions") or [],
                                        "sections": strict.get("sections") or []}})
    return cases, rows


def units_digest(cases, donor_rows) -> str:
    """What the stage's result depends on in the case store and the donor report."""
    digest = hashlib.sha256()
    for case in sorted(cases, key=lambda c: (c.image, c.vram)):
        digest.update(json.dumps([case.image, case.vram, case.size, case.status, (case.strict or {}).get("assumptions"),
                                  (case.strict or {}).get("sections")], sort_keys=True, default=str).encode())
    for row in donor_rows:
        digest.update(json.dumps(row, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def merge_declared(folder: Path, lines: dict[str, tuple[int, str]]) -> dict:
    """Add established placements to ``<folder>/declared_symbols.txt``, the file the export, the
    receipts and the library already carry beside a run's exact cases.  Lines are only ever added
    (an exact case verified with one must verify again); a name the file already declares at another
    address is a conflict, reported and not written."""
    path = Path(folder) / "declared_symbols.txt"
    existing = path.read_text().splitlines() if path.is_file() else []
    have: dict[str, int] = {}
    for line in existing:
        name, sep, rest = line.partition(" = ")
        if sep:
            try:
                have[name.strip()] = int(rest.split(";")[0].strip(), 0)
            except ValueError:
                continue
    added, ok, conflicts = [], [], []
    for name, (address, why) in sorted(lines.items(), key=lambda kv: (kv[1][0], kv[0])):
        if name in have and have[name] != address:
            conflicts.append(f"{name}: {path.name} declares {hx(have[name])}; the data stage established {hx(address)}")
            continue
        ok.append(name)
        if name not in have:
            have[name] = address
            added.append(f"{name} = {hx(address)}; {DATA_STAGE_MARK} {why}")
    if added:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic(path, "".join(line + "\n" for line in existing + added))
    return {"file": str(path), "declared": ok, "added": len(added), "conflicts": conflicts}
