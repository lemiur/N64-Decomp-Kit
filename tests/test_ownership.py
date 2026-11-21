import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from n64dk.match import datastage
from n64dk.match import ownership as own
from n64dk.match.ownership import Evidence, OwnershipModel, build, gp_independent_of, gp_provenance, progress


class OwnershipTest(unittest.TestCase):
    def inventory(self):
        base = {"image": "main", "section_vram": "0x80000000", "section_size": 0x30,
                "section_rom": "0x1000", "state": "remaining", "reason": "fixture", "id": "x"}
        rows = [
            {**base, "name": "a", "vram": "0x80000000", "size": 0x10,
             "ownership_state": "manifest_exact_unit", "owner": "src/a.c"},
            {**base, "name": "b", "vram": "0x80000010", "size": 0x08,
             "ownership_state": "manifest_nonexact", "owner": "src/b.c"},
            {**base, "name": "c", "vram": "0x80000020", "size": 0x10,
             "ownership_state": "assembly_unit", "owner": "asm/c.s"},
        ]
        return {"summary": {"functions": 3, "declared_functions": 4, "denominator_delta": -1},
                "functions": rows}

    def test_partition_preserves_gaps_and_claims(self):
        result = build(self.inventory())
        self.assertTrue(result["complete_partition"])
        self.assertFalse(result["authoritative_complete"])
        self.assertEqual(result["totals"], {"verified_c": 16, "claimed_c": 8,
                                            "unknown": 8, "assembly": 16})
        self.assertEqual(result["unresolved_bytes"], 16)

    def test_progress_separates_denominator_and_claims(self):
        inventory = self.inventory(); result = progress(inventory, build(inventory))
        self.assertEqual(result["functions"]["verified_c"], 1)
        self.assertEqual(result["functions"]["claimed_c"], 1)
        self.assertEqual(result["functions"]["denominator_delta"], -1)
        self.assertTrue(result["invariants"]["one_class_per_byte"])


# --------------------------------------------------------------------------- the shared model

STRICT = Evidence("strict", "fixture: compared equal", "unit")
GUESS = Evidence("unit-derived", "fixture", "unit")


class ModelTest(unittest.TestCase):
    def test_runtime_allocations_own_no_rom_bytes(self):
        model = OwnershipModel()
        model.map_image("main", lambda address: address - 0x80000000 + 0x1000)
        data = model.add(0x80000100, 0x10, "image", "rodata", "A", image="main", settled=True, evidence=[STRICT])
        bss = model.add(0x80000100, 0x40, "runtime", "bss", "B", image="main", evidence=[GUESS])
        self.assertEqual(model.rom_span(data), (0x1100, 0x1110))
        self.assertIsNone(model.rom_span(bss))
        for space, kind in (("image", "bss"), ("image", "sbss"), ("rom", "common"), ("runtime", "data")):
            with self.assertRaises(ValueError):
                model.add(0x80000000, 4, space, kind, "C", image="" if space == "rom" else "main")
        with self.assertRaises(ValueError):
            model.add(0x1000, 4, "rom", "data", "C", image="main")          # a rom interval names no image
        rom = model.who_owns(0x1104, "rom")
        self.assertEqual((rom["status"], rom["owners"]), ("settled", ["A"]))
        self.assertEqual(model.who_owns(0x80000120, "runtime", "main")["owners"], ["B"])
        # the same address in another space is another question: the BSS does not contest the rodata
        self.assertEqual(model.who_owns(0x80000104, "image", "main")["status"], "settled")
        summary = model.summary()
        self.assertEqual((summary["settled_rom_bytes"], summary["runtime_intervals_with_rom_bytes"]), (0x10, 0))

    def test_candidates_of_two_owners_over_the_same_bytes_stay_ambiguous(self):
        model = OwnershipModel()
        a = model.add(0x80000000, 0x10, "image", "rodata", "A", image="main", evidence=[GUESS])
        b = model.add(0x80000008, 0x10, "image", "rodata", "B", image="main", evidence=[GUESS])
        model.classify()
        self.assertEqual((a.status, b.status), ("ambiguous", "ambiguous"))
        answer = model.who_owns(0x8000000C, "image", "main")
        self.assertEqual((answer["status"], answer["owners"]), ("ambiguous", ["A", "B"]))
        self.assertEqual(model.ambiguities[0]["type"], "overlap")
        self.assertEqual(model.who_owns(0x80000020, "image", "main")["status"], "unowned")

    def test_a_candidate_over_settled_bytes_is_a_conflict(self):
        model = OwnershipModel()
        settled = model.add(0x80000000, 0x10, "image", "data", "A", image="main", settled=True, evidence=[STRICT])
        other = model.add(0x8000000C, 0x8, "image", "jtbl", "B", image="main", evidence=[GUESS])
        model.classify()
        self.assertEqual((settled.status, other.status), ("settled", "conflict"))
        self.assertEqual(model.conflicts[0]["type"], "contradicts-settled")
        self.assertEqual(model.who_owns(0x8000000C, "image", "main")["status"], "conflict")
        self.assertEqual(model.who_owns(0x80000004, "image", "main")["status"], "settled")
        both = OwnershipModel()
        x = both.add(0x80000000, 8, "image", "data", "A", image="main", settled=True, evidence=[STRICT])
        y = both.add(0x80000004, 8, "image", "data", "B", image="main", settled=True,
                     evidence=[Evidence("declared", "manifest", "m.tsv")])
        both.classify()
        self.assertEqual((x.status, y.status, both.conflicts[0]["type"]), ("conflict", "conflict", "settled-overlap"))

    def test_alternative_placements_of_one_object(self):
        model = OwnershipModel()
        first = model.add(0x80000000, 8, "image", "rodata", "U", image="main", obj=".rodata", evidence=[GUESS])
        second = model.add(0x80000100, 8, "image", "rodata", "U", image="main", obj=".rodata", evidence=[GUESS])
        model.classify()
        self.assertEqual((first.status, second.status), ("ambiguous", "ambiguous"))
        self.assertEqual(model.ambiguities[0]["type"], "alternative-layouts")
        self.assertEqual(len(model.placements("U", ".rodata")), 2)
        model.add(0x80000200, 8, "image", "rodata", "U", image="main", obj=".rodata", settled=True, evidence=[STRICT])
        model.classify()
        self.assertEqual((first.status, second.status), ("conflict", "conflict"))
        self.assertEqual([i.status for i in model.placements("U", ".rodata")][-1], "settled")

    def test_settling_needs_strict_or_declared_evidence(self):
        model = OwnershipModel()
        with self.assertRaises(ValueError):
            model.add(0x80000000, 4, "image", "data", "A", image="main", settled=True,
                      evidence=[Evidence("reloc", "another function", "function:0x80001000")])
        model.add(0x80000000, 4, "image", "data", "A", image="main", evidence=[GUESS])
        again = model.add(0x80000000, 4, "image", "data", "A", image="main", evidence=[("reloc", "x", "f")])
        self.assertEqual((len(model.intervals), [e.kind for e in again.evidence]), (1, ["unit-derived", "reloc"]))
        self.assertTrue(again.evidence[1].independent)
        self.assertFalse(again.evidence[0].independent)

    def test_alignment(self):
        model = OwnershipModel()
        bad = model.add(0x80000004, 8, "image", "literal-pool", "A", image="main", align=8, evidence=[GUESS])
        model.classify()
        self.assertEqual((bad.status, model.conflicts[0]["type"]), ("conflict", "misaligned"))
        with self.assertRaises(ValueError):
            model.add(0x80000000, 8, "image", "data", "A", image="main", align=3)
        self.assertEqual((own.align_up(0x8000010C, 8), own.align_up(0x8000010C, 1)), (0x80000110, 0x8000010C))

    def test_overlays_that_share_vram_do_not_collide(self):
        model = OwnershipModel()
        one = model.add(0x80400000, 0x100, "image", "code", "ovl1:f", image="ovl1", evidence=[GUESS])
        two = model.add(0x80400000, 0x100, "image", "code", "ovl2:g", image="ovl2", evidence=[GUESS])
        model.classify()
        self.assertEqual((one.status, two.status), ("candidate", "candidate"))
        self.assertEqual(model.who_owns(0x80400010, "image", "ovl2")["owners"], ["ovl2:g"])

    def test_settled_neighbours_in_archive_order(self):
        model = OwnershipModel()
        model.add(0x80000100, 0xC, "image", "rodata", "A", image="main", settled=True, evidence=[STRICT])
        model.add(0x80000120, 0x10, "image", "rodata", "B", image="main", settled=True, evidence=[STRICT])
        found = model.settled_neighbours("main", 0x80000110, 0x8000011C, zero=lambda lo, hi: True)
        self.assertEqual(sorted(e.origin for e in found), ["A", "B"])
        self.assertTrue(all(e.kind == "adjacent-settled" for e in found))
        self.assertEqual(model.settled_neighbours("main", 0x80000110, 0x8000011C, zero=lambda lo, hi: False), [])
        self.assertEqual([e.origin for e in model.settled_neighbours("main", 0x80000110, 0x8000011C, exclude="A",
                                                                     zero=lambda lo, hi: True)], ["B"])
        # 0x80000114 is no alignment of A's end (0x8000010C): only B, at 16, still bounds it
        self.assertEqual([e.origin for e in model.settled_neighbours("main", 0x80000114, 0x80000118,
                                                                     zero=lambda lo, hi: True)], ["B"])


class GpProvenanceTest(unittest.TestCase):
    INFERRED = {"value": "0x80100000", "confidence": "high", "reason": "set at 2 site(s)",
                "sites": {"main:0x80000410": "0x80100000", "main:0x80000810": "0x80100000"}}

    def test_declared_and_inferred_are_kept_apart(self):
        found = gp_provenance(0x80100000, self.INFERRED)
        self.assertEqual((found["status"], found["value"], found["basis"]), ("agree", 0x80100000, "declared"))
        self.assertEqual(found["inferred"]["value"], "0x80100000")
        self.assertEqual(found["sites"], {"main": [0x80000410, 0x80000810]})
        self.assertEqual(gp_provenance(0x80100000, None)["status"], "declared-only")
        alone = gp_provenance(None, self.INFERRED)
        self.assertEqual((alone["status"], alone["value"], alone["basis"]), ("inferred-only", 0x80100000, "inferred"))
        low = gp_provenance(None, {**self.INFERRED, "confidence": "low"})
        self.assertEqual((low["status"], low["value"]), ("inferred-low", None))
        self.assertEqual(gp_provenance(None, None)["status"], "none")

    def test_a_disagreement_is_a_conflict_and_no_value(self):
        found = gp_provenance(0x80200000, self.INFERRED)
        self.assertEqual((found["status"], found["value"]), ("conflict", None))
        self.assertIn("0x80200000", found["conflict"])
        self.assertIn("0x80100000", found["conflict"])
        self.assertFalse(gp_independent_of(found, "main", [(0x80001000, 0x20)]))

    def test_an_inferred_gp_is_not_evidence_for_the_unit_that_sets_it(self):
        alone = gp_provenance(None, self.INFERRED)
        self.assertFalse(gp_independent_of(alone, "main", [(0x80000400, 0x20)]))
        self.assertTrue(gp_independent_of(alone, "main", [(0x80001000, 0x20)]))
        self.assertTrue(gp_independent_of(alone, "ovl", [(0x80000400, 0x20)]))
        declared = gp_provenance(0x80100000, None)
        self.assertTrue(gp_independent_of(declared, "main", [(0x80000400, 0x20)]))


# --------------------------------------------------------------------------- grouping and placement

def claim(address, function, site=None, source="hilo", width=4, access="word", kind="load"):
    return SimpleNamespace(address=address, source=source, access=access, kind=kind, width=width,
                           function=function, site=site if site is not None else (function or 0) + 4, exact=False)


def need(unit, name, address, spans, ref="symbol", size=0, image="main", origin="case", call=False):
    return datastage.Need(unit, origin, image, ref, name, address, size, tuple(spans), call, unit)


def context(claims=(), **kw):
    by_address = {}
    for c in claims:
        by_address.setdefault(c.address, []).append(c)
    base = {"regions": [(0x80001000, 0x80002000)], "code": [(0x80000000, 0x80001000)],
            "bss": (0x80002000, 0x80003000), "zero": lambda lo, hi: True}
    base.update(kw)
    return datastage.Context("main", by_address, **base)


def placed(needs, ctx, model=None):
    model = model or OwnershipModel().classify()
    return [datastage.place(g, ctx, model) for g in datastage.group_needs(needs)]


class GroupingTest(unittest.TestCase):
    F, G, H = 0x80000000, 0x80000100, 0x80000200
    X = 0x80001010

    def test_needs_are_grouped_by_symbol_section_and_image(self):
        needs = [need("u1", "D_80001010", self.X, [(self.F, 0x40)]),
                 need("u2", "D_80001010", self.X, [(self.G, 0x40)]),
                 need("u3", "D_80001010", self.X, [(self.G, 0x40)], image="ovl"),
                 need("u1", ".rodata", 0x80001100, [(self.F, 0x40)], ref="section", size=0x10),
                 need("u2", ".rodata", 0x80001120, [(self.G, 0x40)], ref="section", size=0x10),
                 need("u2", "func_80000200", self.H, [(self.G, 0x40)], call=True)]
        groups = datastage.group_needs(needs)
        self.assertEqual(sorted((g.image, g.ref, g.name, len(g.needs)) for g in groups),
                         [("main", "section", ".rodata", 1), ("main", "section", ".rodata", 1),
                          ("main", "symbol", "D_80001010", 2), ("ovl", "symbol", "D_80001010", 1)])

    def test_a_units_own_instructions_never_establish_its_address(self):
        ctx = context([claim(self.X, self.F)])
        (group,) = placed([need("u1", "D_80001010", self.X, [(self.F, 0x40)])], ctx)
        self.assertEqual(group.status, "awaiting-evidence")
        self.assertEqual(group.own_references, {"u1": 1})
        self.assertEqual(group.support, {"u1": []})
        # a data word inside the unit's own bytes is the unit's too
        model = OwnershipModel()
        model.add(0x80001080, 0x10, "image", "data", "u1", image="main", obj=".data", evidence=[GUESS])
        model.classify()
        ctx = context([claim(self.X, None, site=0x80001084, source="pointer", width=1, access="pointer",
                             kind="pointer")])
        (group,) = placed([need("u1", "D_80001010", self.X, [(self.F, 0x40)])], ctx, model)
        self.assertEqual((group.status, group.own_references), ("awaiting-evidence", {"u1": 1}))

    def test_independent_references_establish_and_mutual_support_is_flagged(self):
        ctx = context([claim(self.X, self.F), claim(self.X, self.G), claim(self.X, None, site=0x80001800,
                                                                            source="pointer", width=1)])
        (group,) = placed([need("u1", "D_80001010", self.X, [(self.F, 0x40)])], ctx)
        self.assertEqual(group.status, "established")
        self.assertEqual(sorted(e.kind for e in group.support["u1"]), ["data-pointer", "reloc"])
        self.assertFalse(group.mutual)
        ctx = context([claim(self.X, self.F), claim(self.X, self.G)])
        (group,) = placed([need("u1", "D_80001010", self.X, [(self.F, 0x40)]),
                           need("u2", "D_80001010", self.X, [(self.G, 0x40)])], ctx)
        self.assertEqual((group.status, group.mutual), ("established", True))
        self.assertEqual(group.support["u1"][0].origin, f"function:0x{self.G:08X}")

    def test_declarations(self):
        ctx = context(declared={"D_80001010": 0x80001020})
        (group,) = placed([need("u1", "D_80001010", self.X, [(self.F, 0x40)])], ctx)
        self.assertEqual(group.status, "conflict")
        ctx = context(declared={"gScore": self.X, "_gp": self.X})
        (group,) = placed([need("u1", "D_80001010", self.X, [(self.F, 0x40)])], ctx)
        self.assertEqual((group.status, [e.kind for e in group.support["u1"]]), ("established", ["alias-declared"]))
        ctx = context(declared={"D_80001010": self.X})
        (group,) = placed([need("u1", "D_80001010", self.X, [(self.F, 0x40)])], ctx)
        self.assertEqual([e.kind for e in group.support["u1"]], ["declared"])

    def test_constraints(self):
        ctx = context([claim(self.X + 2, self.G)])
        groups = placed([need("u1", "a", 0x80000010, [(self.F, 0x40)]),                       # catalogued code
                         need("u1", "b", self.X + 2, [(self.F, 0x40)]),                        # misaligned word
                         need("u1", ".rodata", 0x80001FF8, [(self.F, 0x40)], ref="section", size=0x10),
                         need("u1", ".bss", 0x80001000, [(self.F, 0x40)], ref="section", size=0x10),
                         need("u1", "c", 0x80001100, [(self.F, 0x40)]),
                         need("u2", "c", 0x80001104, [(self.G, 0x40)])], ctx)
        status = {(g.ref, g.name): g for g in groups}
        self.assertEqual(status[("symbol", "a")].status, "ambiguous")             # code or data: not settled code
        self.assertEqual(status[("symbol", "b")].status, "awaiting-evidence")
        self.assertIn("misaligned", " ".join(status[("symbol", "b")].reasons))
        self.assertEqual(status[("section", ".rodata")].status, "conflict")       # runs past the region
        self.assertEqual(status[("section", ".bss")].status, "conflict")          # BSS in initialised bytes
        self.assertEqual(status[("symbol", "c")].status, "conflict")              # two addresses for one symbol
        tables = context([claim(0x80001008, self.G)], tables=[(0x80001000, 0x80001020)])
        (inside,) = placed([need("u1", "d", 0x80001008, [(self.F, 0x40)])], tables)
        self.assertEqual(inside.status, "conflict")

    def test_settled_code_and_function_addresses(self):
        model = OwnershipModel()
        model.add(self.F, 0x40, "image", "code", "case:main:f", image="main", settled=True, evidence=[STRICT])
        model.classify()
        groups = placed([need("u2", "f", self.F, [(self.G, 0x40)]), need("u2", "inside", self.F + 8, [(self.G, 0x40)])],
                        context(), model)
        self.assertEqual({g.name: g.status for g in groups}, {"f": "code-reference", "inside": "conflict"})

    def test_outside_the_image_and_overlapping_overlays(self):
        ctx = context(homes={"ovl1": (0x80400000, 0x80410000), "ovl2": (0x80400000, 0x80420000),
                             "boot": (0x80000400, 0x80000800)})
        (group,) = placed([need("u1", "D_80400010", 0x80400010, [(self.F, 0x40)])], ctx)
        self.assertEqual((group.status, group.homes), ("outside-image", ["ovl1", "ovl2"]))
        self.assertIn("ambiguous", " ".join(group.reasons))

    def test_gp_relative_support_needs_a_gp_independent_of_the_unit(self):
        inferred = gp_provenance(None, {"value": "0x80009000", "confidence": "high", "reason": "",
                                        "sites": {"main:0x80000110": "0x80009000"}})
        gprel = [claim(self.X, self.H, source="gprel")]
        (group,) = placed([need("u1", "D_80001010", self.X, [(self.G, 0x40)])], context(gprel, provenance=inferred))
        self.assertEqual(group.status, "awaiting-evidence")                       # u1 sets $gp itself
        self.assertIn("usable _gp", " ".join(group.reasons))
        (group,) = placed([need("u1", "D_80001010", self.X, [(self.F, 0x40)])], context(gprel, provenance=inferred))
        self.assertEqual(group.status, "established")
        conflict = gp_provenance(0x80008000, {"value": "0x80009000", "confidence": "high", "sites": {}})
        (group,) = placed([need("u1", "D_80001010", self.X, [(self.F, 0x40)])], context(gprel, provenance=conflict))
        self.assertEqual(group.status, "awaiting-evidence")

    def test_a_section_between_settled_neighbours(self):
        model = OwnershipModel()
        model.add(0x80001000, 0xC, "image", "rodata", "A", image="main", settled=True, evidence=[STRICT])
        model.add(0x80001020, 0x10, "image", "rodata", "C", image="main", settled=True, evidence=[STRICT])
        model.add(0x80001010, 0x10, "image", "rodata", "B", image="main", obj=".rodata", evidence=[GUESS])
        model.classify()
        needs = [need("B", ".rodata", 0x80001010, [(self.F, 0x40)], ref="section", size=0x10, origin="donor")]
        (group,) = placed(needs, context(), model)
        self.assertEqual(group.status, "established")
        self.assertEqual(sorted(e.origin for e in group.support["B"]), ["A", "C"])
        (entry,) = datastage.replay_list(needs, [group])
        self.assertEqual((entry["replay"], entry["sections"]), ("blocked", {".rodata": "0x80001010"}))
        self.assertIn("section base", entry["requirement"])
        model.add(0x80001018, 0x8, "image", "jtbl", "D", image="main", evidence=[GUESS])   # another owner's claim
        model.classify()
        (group,) = placed(needs, context(), model)
        self.assertEqual(group.status, "ambiguous")

    def test_replay_list(self):
        needs = [need("u1", "D_80001010", self.X, [(self.F, 0x40)]),
                 need("u2", "D_80001010", self.X, [(self.G, 0x40)]),
                 need("u2", "func_80000200", self.H, [(self.G, 0x40)], call=True),
                 need("u3", "D_80001010", self.X, [(self.H, 0x40)], origin="donor"),
                 need("u4", "D_80001100", 0x80001100, [(self.H + 0x40, 0x40)])]
        ctx = context([claim(self.X, 0x80000300)])
        units = {u["unit"]: u for u in datastage.replay_list(needs, placed(needs, ctx))}
        self.assertEqual(units["u1"]["replay"], "resolve")
        self.assertEqual(units["u1"]["declare"], {"D_80001010": "0x80001010"})
        self.assertEqual(units["u2"]["replay"], "blocked")                       # it also waits for a call
        self.assertEqual(units["u3"]["replay"], "blocked")                       # no donor replay path
        self.assertEqual((units["u4"]["replay"], units["u4"]["established"]), ("waiting", 0))


# --------------------------------------------------------------------------- the stage on a tiny image

BASE = 0x80001000
AT, V0, V1 = 1, 2, 3
JR_RA = 0x03E00008


def _i(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def _words(*ws):
    ws = list(ws) + [0] * (8 - len(ws))
    return b"".join(struct.pack(">I", w) for w in ws)


def _assumed(name, vram, symbol, address):
    return SimpleNamespace(image="main", name=name, vram=vram, size=0x20, status="assumed",
                           strict={"verdict": "assumed", "sections": [], "assumptions": [
                               {"kind": "symbol", "name": symbol, "address": f"0x{address:08X}", "call": False}]})


class StageTest(unittest.TestCase):
    def image(self):
        def lui(rt, imm): return _i(15, 0, rt, imm)
        def lw(rt, imm, rs): return _i(35, rs, rt, imm)
        f = _words(lui(AT, 0x8000), lw(V0, 0x1060, AT), JR_RA)          # needs D_80001060
        g = _words(lui(AT, 0x8000), lw(V1, 0x1060, AT), JR_RA)          # another function reads it too
        h = _words(lui(AT, 0x8000), lw(V0, 0x1070, AT), JR_RA)          # needs D_80001070, alone
        return f + g + h + bytes(0x40)

    def test_one_image_end_to_end(self):
        rows = [("f", BASE, 0x20), ("g", BASE + 0x20, 0x20), ("h", BASE + 0x40, 0x20)]
        cases = [_assumed("f", BASE, "D_80001060", 0x80001060), _assumed("h", BASE + 0x40, "D_80001070", 0x80001070)]
        document = datastage.run_image(image="main", data=self.image(), delta=BASE, rows=rows, cases=cases,
                                       rom_of=lambda address: address - BASE + 0x1000)
        groups = {g["name"]: g for g in document["groups"]}
        self.assertEqual(groups["D_80001060"]["status"], "established")
        self.assertEqual(groups["D_80001060"]["support"]["case:main:0x80001000"][0]["origin"], "function:0x80001020")
        self.assertEqual(groups["D_80001060"]["own_references"], {"case:main:0x80001000": 1})
        self.assertEqual(groups["D_80001070"]["status"], "awaiting-evidence")
        units = {u["unit"]: u for u in document["units"]}
        self.assertEqual(units["case:main:0x80001000"]["replay"], "resolve")
        self.assertEqual(units["case:main:0x80001040"]["replay"], "waiting")
        self.assertEqual(document["summary"]["units_by_replay"], {"resolve": 1, "waiting": 1})
        self.assertEqual(document["gp"]["status"], "none")
        self.assertIn("never the evidence", document["claim"])
        # D_80001070 is read by h alone: an object h's unit may own, a candidate in the model
        owner = [i for i in document["ownership"]["intervals"] if i["start"] == "0x80001070"]
        self.assertEqual([(i["owner"], i["status"]) for i in owner], [("case:main:0x80001040", "candidate")])
        self.assertNotIn("exact", json.dumps(document["summary"]))


class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="n64dk-datastage-test-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_merge_declared_only_adds_and_reports_conflicts(self):
        (self.root / "declared_symbols.txt").write_text("D_80000010 = 0x80000010;\n")
        merged = datastage.merge_declared(self.root, {"D_80000010": (0x80000020, "x"), "D_80000030": (0x80000030, "y"),
                                                      "D_80000040": (0x80000040, "z")})
        self.assertEqual((merged["added"], len(merged["conflicts"])), (2, 1))
        text = (self.root / "declared_symbols.txt").read_text()
        self.assertTrue(text.startswith("D_80000010 = 0x80000010;\n"))
        self.assertIn(f"D_80000030 = 0x80000030; {datastage.DATA_STAGE_MARK} y\n", text)
        self.assertEqual(datastage.merge_declared(self.root, {"D_80000030": (0x80000030, "y")})["added"], 0)

    def test_replay_rebuilds_each_established_case_once(self):
        import argparse
        from n64dk.commands import run as runmod
        entries = [{"unit": "case:main:0x80001000", "origin": "case", "label": "f", "replay": "resolve",
                    "declare": {"D_80001060": "0x80001060"}, "why": {"D_80001060": "reloc x1"}, "digest": "d1"},
                   {"unit": "case:main:0x80001040", "origin": "case", "label": "h", "replay": "waiting"}]
        (self.root / "data_replay.json").write_text(json.dumps(entries))
        seen = []

        def fake_resolve_hard(options):
            seen.append((options.only_statuses, options.only_names, options.refine, options.symbols,
                         options.write_scope))
            return {"statuses": {"exact": 1}}

        def options(**overrides):
            return argparse.Namespace(symbols=["symbol_addrs.txt"], **overrides)
        result = runmod.replay_data(fake_resolve_hard, options, self.root, {})
        self.assertEqual((result["units"], result["promoted"]), (1, 1))
        self.assertEqual(seen[0][:3], (("assumed",), ["f"], None))
        self.assertEqual(seen[0][3], ["symbol_addrs.txt", str(self.root / "declared_symbols.txt")])
        self.assertIs(seen[0][4], False)                     # the sweep's scope file is not rewritten
        self.assertIn(datastage.DATA_STAGE_MARK, (self.root / "declared_symbols.txt").read_text())
        again = runmod.replay_data(fake_resolve_hard, options, self.root, {})
        self.assertEqual((len(seen), again["already_replayed"], again["units"]), (1, 1, 0))

    def test_declare_assumed_keeps_the_data_stage_lines(self):
        import argparse
        from n64dk.commands import run as runmod
        from n64dk.match import resolve as resolvemod
        store = resolvemod.CaseStore(self.root / "state")
        store.put(resolvemod.Case(resolvemod.case_id("main", 0x80001000, 0x20), "main", "f", 0x80001000, 0x20,
                                  "digest", status="assumed", strict={"verdict": "assumed", "assumptions": [
                                      {"kind": "symbol", "name": "D_80046354", "address": "0x80046354"}]}))
        store.close()
        folder = self.root / "main"
        folder.mkdir()
        kept = f"D_80001060 = 0x80001060; {datastage.DATA_STAGE_MARK} reloc x1"
        (folder / "declared_symbols.txt").write_text("D_80000001 = 0x80000001;\n" + kept + "\n")
        runmod.declare_assumed(lambda options: {"statuses": {}},
                               lambda **o: argparse.Namespace(symbols=[], **o), self.root / "state", "main", folder, {})
        self.assertEqual((folder / "declared_symbols.txt").read_text(), "D_80046354 = 0x80046354;\n" + kept + "\n")

    def test_load_units_keeps_only_what_the_stage_reads(self):
        report = {"rows": [{"target_vram": "0x80002000", "size": 0x40, "verdict": datastage.DONOR_ASSUMED,
                            "donor": "d", "donor_unit": "src/a.c", "functions": ["a", "b"],
                            "awaiting": {".rodata": "0x80003000"}, "notes": ["long"],
                            "strict": {"assumptions": [{"kind": "section", "name": ".rodata", "address": "0x80003000"}],
                                       "sections": [{"section": ".rodata", "size": 16, "status": "equal",
                                                     "base": "0x80003000", "base_assumed": True}], "notes": ["x"]}},
                           {"target_vram": "0x80004000", "size": 4, "verdict": "mismatch"}]}
        (self.root / "report.json").write_text(json.dumps(report))
        cases, rows = datastage.load_units(self.root / "no-state", "main", self.root / "report.json")
        self.assertEqual((cases, len(rows)), ([], 1))
        self.assertNotIn("notes", rows[0])
        (found,) = datastage.needs_from_donor(rows[0], "main")
        self.assertEqual((found.ref, found.name, found.address, found.size, found.unit),
                         ("section", ".rodata", 0x80003000, 16, "donor:main:0x80002000"))
        model = datastage.build_model("main", rows=[("a", 0x80002000, 0x20), ("b", 0x80002020, 0x20)],
                                      donor_rows=rows)
        self.assertEqual(sorted((i.kind, i.status) for i in model.owned_by("donor:main:0x80002000")),
                         [("code", "candidate"), ("rodata", "candidate")])
        self.assertEqual(model.who_owns(0x80002020, "image", "main")["owners"], ["donor:main:0x80002000"])

    def test_transplant_verdict_names_agree(self):
        from n64dk.match import transplant
        self.assertEqual((datastage.DONOR_EXACT, datastage.DONOR_ASSUMED),
                         (transplant.EXACT_VERDICT, transplant.ASSUMED_VERDICT))


if __name__ == "__main__": unittest.main()
