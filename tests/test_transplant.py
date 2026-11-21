"""Donor transplant: masking, scan, symbol transfer, unit planning (synthetic) and a real
end-to-end transplant that skips without the toolchain or the game images."""
from __future__ import annotations

import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from n64dk.match import transplant as tp

NOP, JR_RA = 0x00000000, 0x03E00008


def jal(target: int) -> int:
    return 0x0C000000 | ((target >> 2) & 0x3FFFFFF)


def code(*words: int) -> bytes:
    return struct.pack(f">{len(words)}I", *words)


def donor_fn(lui: int, lo: int, callee: int, field: int = 0x10, extra: tuple[int, ...] = ()) -> bytes:
    return code(0x27BDFFE8,              # addiu sp, sp, -0x18      (kept)
                0xAFBF0014,              # sw ra, 0x14(sp)          (kept)
                0x3C010000 | lui,        # lui at, %hi(sym)         (masked)
                0x8C220000 | lo,         # lw v0, %lo(sym)(at)      (masked)
                jal(callee),             # jal callee               (masked)
                0x8C880000 | field,      # lw t0, field(a0)         (kept)
                *extra,
                0x8FBF0014,              # lw ra, 0x14(sp)
                JR_RA,
                0x27BD0018)              # addiu sp, sp, 0x18


DONOR_NAMES = {0x80123456: "gValue", 0x80101000: "callee", 0x80101100: "other"}


def name_at(address):
    return (DONOR_NAMES[address], 0) if address in DONOR_NAMES else (None, 0)


def make_unit(functions: list[tuple[str, bytes]], vram: int, donor: str = "D", source: str = "src/a.c") -> dict:
    records, offset = [], 0
    blob = b"".join(c for _n, c in functions)
    for k, (name, c) in enumerate(functions):
        records.append(tp.function_record(c, vram + offset, name, offset, len(c), k, name_at, (vram, vram + len(blob))))
        offset += len(c)
    masked, _ = tp.mask(blob, vram)
    return {"donor": donor, "source": source, "manifest": "m.tsv", "line": 1, "pipeline": "ido5.3",
            "flags_text": "", "flags": ["-O2"], "normalizer_args": [], "section_bases": {}, "symbols": {},
            "vram": vram, "size": len(blob), "core_size": len(blob), "unit_masked": tp.signature(masked),
            "functions": records}


def image_with(vram_delta: int, placed: dict[int, bytes], size: int = 0x4000, base: int = 0x80200000) -> bytes:
    data = bytearray(size)
    for vram, c in placed.items():
        off = vram - vram_delta
        data[off:off + len(c)] = c
    return bytes(data)


class MaskTest(unittest.TestCase):
    def test_masks_relocations_keeps_ordinary_immediates(self):
        c = donor_fn(0x8012, 0x3456, 0x80101000)
        masked, sites = tp.mask(c, 0x80100000)
        raw = tp.words_of(c)
        self.assertEqual(masked[0], raw[0])                 # stack adjust kept
        self.assertEqual(masked[1], raw[1])                 # stack offset kept
        self.assertEqual(masked[2], 0x3C010000)             # lui immediate masked
        self.assertEqual(masked[3], 0x8C220000)             # %lo masked
        self.assertEqual(masked[4], 0x0C000000)             # jal target masked
        self.assertEqual(masked[5], raw[5])                 # struct field offset kept
        kinds = {i: (k, a) for i, k, a in sites}
        self.assertEqual(kinds[2], ("hi", 0x80123456))
        self.assertEqual(kinds[3], ("lo", 0x80123456))
        self.assertEqual(kinds[4], ("jal", 0x80101000))

    def test_lui_ori_constant_keeps_ori(self):
        c = code(0x3C043F80, 0x34840001, JR_RA, NOP)        # lui a0,0x3f80; ori a0,a0,1
        masked, sites = tp.mask(c, 0x80100000)
        self.assertEqual(masked[0], 0x3C040000)
        self.assertEqual(masked[1], 0x34840001)
        self.assertEqual(sites[0][1], "lui")                # unpaired: masked, never transferred

    def test_trailing_padding_is_not_part_of_the_signature(self):
        c = donor_fn(0x8012, 0x3456, 0x80101000)
        self.assertEqual(tp.strip_padding(c + bytes(8)), c)


class ScanTransferTest(unittest.TestCase):
    delta = 0x80200000 - 0x1000

    def index(self, *units):
        return tp.Index({str(k): u for k, u in enumerate(units)}, {"D": {"root": "/x", "image": "/x", "delta": 0, "include": []}})

    def test_exact_hit_and_symbol_transfer(self):
        unit = make_unit([("donor_f", donor_fn(0x8012, 0x3456, 0x80101000))], 0x80100000)
        target = donor_fn(0x8022, 0x3466, 0x80205000)
        image = image_with(self.delta, {0x80200100: target})
        hits = tp.scan(image, self.delta, [("t_f", 0x80200100, len(target))], self.index(unit))
        self.assertEqual([h.kind for h in hits], ["exact"])
        result = tp.transfer(hits)
        by_name = {v["donor_symbol"]: v["target_address"] for v in result["symbols"].values()}
        self.assertEqual(by_name, {"gValue": 0x80223466, "callee": 0x80205000})
        self.assertIn("gValue = 0x80223466;", tp.symbol_file(result))
        self.assertIn("callee = 0x80205000; // type:func", tp.symbol_file(result))

    def test_near_hit_is_reported_not_exact(self):
        unit = make_unit([("donor_f", donor_fn(0x8012, 0x3456, 0x80101000))], 0x80100000)
        target = donor_fn(0x8022, 0x3466, 0x80205000, field=0x14)    # one ordinary immediate differs
        image = image_with(self.delta, {0x80200100: target})
        hits = tp.scan(image, self.delta, [("t_f", 0x80200100, len(target))], self.index(unit))
        self.assertEqual([(h.kind, h.diffs) for h in hits], [("near", [5])])
        self.assertFalse(hits[0].lost_lo)                              # a0 is no lui's register
        jobs, _partial = tp.plan(hits, image, self.delta)
        self.assertEqual(jobs, [])                                      # never transplanted

    def test_a_near_hit_where_only_a_lost_lo_half_differs_is_verified(self):
        # (the scan sets lost_lo when every differing word is a load/store/addiu immediate on a
        # register some lui of the function sets; the near test above, on a pointer argument, is not)
        unit = make_unit([("donor_f", donor_fn(0x8012, 0x3456, 0x80101000))], 0x80100000)
        target = donor_fn(0x8012, 0x3456, 0x80101000)
        image = image_with(self.delta, {0x80200100: target})
        near = tp.Hit("t_f", 0x80200100, len(target), unit, unit["functions"][0], "near", [3], status="near",
                      lost_lo=True)
        jobs, _partial = tp.plan([near], image, self.delta)
        self.assertEqual([job.vram for job in jobs], [0x80200100])
        self.assertEqual(near.status, "candidate")
        ordinary = tp.Hit("t_f", 0x80200100, len(target), unit, unit["functions"][0], "near", [5], status="near")
        self.assertEqual(tp.plan([ordinary], image, self.delta)[0], [])

    def test_contradiction_marks_hits_for_review(self):
        a = make_unit([("fa", donor_fn(0x8012, 0x3456, 0x80101000))], 0x80100000, source="a.c")
        b = make_unit([("fb", donor_fn(0x8012, 0x3456, 0x80101100, extra=(0x00851021,)))], 0x80100800, source="b.c")
        ta = donor_fn(0x8022, 0x3466, 0x80205000)
        tb = donor_fn(0x8022, 0x3470, 0x80205100, extra=(0x00851021,))  # gValue elsewhere
        image = image_with(self.delta, {0x80200100: ta, 0x80200200: tb})
        hits = tp.scan(image, self.delta, [("ta", 0x80200100, len(ta)), ("tb", 0x80200200, len(tb))],
                       self.index(a, b))
        result = tp.transfer(hits)
        self.assertEqual(len(result["contradictions"]), 1)
        self.assertEqual({h.status for h in hits}, {"needs-review"})
        self.assertNotIn(("D", 0x80123456), result["symbols"])

    def test_generic_shape_does_not_vote(self):
        unit = make_unit([("getter", code(0x3C028012, 0x8C423456, JR_RA, NOP))], 0x80100000)
        t1, t2 = code(0x3C028022, 0x8C420010, JR_RA, NOP), code(0x3C028022, 0x8C420020, JR_RA, NOP)
        image = image_with(self.delta, {0x80200100: t1, 0x80200200: t2})
        hits = tp.scan(image, self.delta, [("g1", 0x80200100, 16), ("g2", 0x80200200, 16)], self.index(unit))
        result = tp.transfer(hits)
        self.assertTrue(all(h.template for h in hits))
        self.assertEqual(result["contradictions"], [])
        self.assertEqual(result["symbols"], {})

    def test_two_donor_symbols_on_one_address_are_suspicious(self):
        a = make_unit([("fa", donor_fn(0x8012, 0x3456, 0x80101000))], 0x80100000, source="a.c")
        b = make_unit([("fb", donor_fn(0x8012, 0x3456, 0x80101100, extra=(0x00851021,)))], 0x80100800, source="b.c")
        ta = donor_fn(0x8022, 0x3466, 0x80205000)
        tb = donor_fn(0x8022, 0x3466, 0x80205000, extra=(0x00851021,))  # callee and other -> one address
        image = image_with(self.delta, {0x80200100: ta, 0x80200200: tb})
        hits = tp.scan(image, self.delta, [("ta", 0x80200100, len(ta)), ("tb", 0x80200200, len(tb))],
                       self.index(a, b))
        result = tp.transfer(hits)
        self.assertEqual(len(result["suspicious"]), 1)
        self.assertEqual(result["suspicious"][0]["target_address"], "0x80205000")


class UnitPlanTest(unittest.TestCase):
    delta = 0x80200000 - 0x1000

    def setUp(self):
        self.f1 = donor_fn(0x8012, 0x3456, 0x80101000)
        self.f2 = donor_fn(0x8012, 0x3456, 0x80101100, extra=(0x00851021,))
        self.unit = make_unit([("f1", self.f1), ("f2", self.f2)], 0x80100000)
        self.index = tp.Index({"u": self.unit}, {})

    def test_whole_unit_when_all_functions_are_contiguous(self):
        t1 = donor_fn(0x8022, 0x3466, 0x80205000)
        t2 = donor_fn(0x8022, 0x3466, 0x80205100, extra=(0x00851021,))
        image = image_with(self.delta, {0x80200100: t1 + t2})
        funcs = [("t1", 0x80200100, len(t1)), ("t2", 0x80200100 + len(t1), len(t2))]
        hits = tp.scan(image, self.delta, funcs, self.index)
        jobs, partial = tp.plan(hits, image, self.delta, {f[1] for f in funcs})
        self.assertEqual(len(jobs), 1)
        self.assertEqual((jobs[0].vram, jobs[0].size), (0x80200100, len(t1) + len(t2)))
        self.assertEqual(sorted(h.name for h in jobs[0].hits), ["t1", "t2"])
        self.assertEqual(partial, [])

    def test_partial_unit_lists_missing_neighbours(self):
        t1 = donor_fn(0x8022, 0x3466, 0x80205000)
        image = image_with(self.delta, {0x80200100: t1 + code(*([0x24020001] * 12))})
        funcs = [("t1", 0x80200100, len(t1))]
        hits = tp.scan(image, self.delta, funcs, self.index)
        jobs, partial = tp.plan(hits, image, self.delta, {0x80200100})
        self.assertEqual(jobs, [])
        self.assertEqual(hits[0].status, "partial-unit")
        self.assertEqual([m["donor_function"] for m in partial[0]["missing_neighbours"]], ["f2"])

    def test_a_unit_whose_function_differs_only_in_a_lost_lo_half_is_verified(self):
        # the second function reads through a register its lui sets further down (a loop): the
        # pair tracker does not see the pair, the %lo half stays unmasked and differs
        def looped(lo, hi):
            return code(0x8D020000 | lo,         # lw v0, %lo(sym)(t0)    (not masked: no lui before it)
                        0x3C080000 | hi,         # lui t0, %hi(sym)
                        0x00851021, 0x00851021, 0x00851021,
                        JR_RA, 0x27BD0018)
        unit = make_unit([("f1", self.f1), ("f2", looped(0x3456, 0x8012))], 0x80100000)
        index = tp.Index({"u": unit}, {})
        t1 = donor_fn(0x8022, 0x3466, 0x80205000)
        t2 = looped(0x3466, 0x8022)
        image = image_with(self.delta, {0x80200100: t1 + t2})
        funcs = [("t1", 0x80200100, len(t1)), ("t2", 0x80200100 + len(t1), len(t2))]
        hits = tp.scan(image, self.delta, funcs, index)
        self.assertEqual([(h.name, h.kind, h.lost_lo) for h in hits], [("t1", "exact", False), ("t2", "near", True)])
        jobs, partial = tp.plan(hits, image, self.delta, {f[1] for f in funcs})
        self.assertEqual([(job.vram, job.size, job.whole) for job in jobs], [(0x80200100, len(t1) + len(t2), False)])
        self.assertEqual(partial, [])
        self.assertEqual(tp.unit_definitions(jobs, {}), {})                 # not found whole: defines nothing
        # an ordinary immediate that differs (a field offset on a pointer argument) is another function
        other = code(0x8C820014, 0x3C088022, 0x00851021, 0x00851021, 0x00851021, JR_RA, 0x27BD0018)
        donor_other = make_unit([("f1", self.f1), ("f2", code(0x8C820010, 0x3C088012, 0x00851021, 0x00851021,
                                                             0x00851021, JR_RA, 0x27BD0018))], 0x80100000)
        image = image_with(self.delta, {0x80200100: t1 + other})
        hits = tp.scan(image, self.delta, funcs, tp.Index({"u": donor_other}, {}))
        jobs, partial = tp.plan(hits, image, self.delta, {f[1] for f in funcs})
        self.assertEqual(jobs, [])
        self.assertEqual([m["donor_function"] for m in partial[0]["missing_neighbours"]], ["f2"])

    def test_unit_must_start_at_a_target_function_start(self):
        t2 = donor_fn(0x8022, 0x3466, 0x80205100, extra=(0x00851021,))
        image = image_with(self.delta, {0x80200100: t2})
        hits = tp.scan(image, self.delta, [("t2", 0x80200100, len(t2))], self.index)
        jobs, _ = tp.plan(hits, image, self.delta, {0x80200100})
        self.assertEqual(jobs, [])


class AmbiguousTargetTest(unittest.TestCase):
    """One target function that is several functions of one donor at once (two getters, the same
    routine for two devices) is no witness for the address at the sites where they differ."""
    delta = 0x80200000 - 0x1000

    def setUp(self):
        # the same shape twice in the donor: another variable, the same callee
        self.a = make_unit([("get_a", donor_fn(0x8012, 0x3456, 0x80101000))], 0x80100000, source="src/a.c")
        self.b = make_unit([("get_b", donor_fn(0x8012, 0x3460, 0x80101000))], 0x80100400, source="src/b.c")
        self.user = make_unit([("user", donor_fn(0x8012, 0x3460, 0x80101100, extra=(0x00851021,)))], 0x80100800,
                              source="src/user.c")
        self.index = tp.Index({"a": self.a, "b": self.b, "u": self.user},
                              {"D": {"root": "/x", "image": "/x", "delta": 0, "include": []}})

    def test_the_differing_site_does_not_vote_and_the_shared_one_does(self):
        getter = donor_fn(0x8022, 0x3466, 0x80205000)                       # stands once in the target
        user = donor_fn(0x8022, 0x3470, 0x80205100, extra=(0x00851021,))    # the only real user of 0x80223470
        image = image_with(self.delta, {0x80200100: getter, 0x80200200: user})
        hits = tp.scan(image, self.delta, [("t_get", 0x80200100, len(getter)), ("t_user", 0x80200200, len(user))],
                       self.index)
        self.assertEqual(sorted(h.function["name"] for h in hits if h.name == "t_get"), ["get_a", "get_b"])
        ambiguous = tp.ambiguous_sites(hits)
        offsets = {offset for _hit, offset in ambiguous}
        self.assertEqual(offsets, {8, 12})                                  # the lui/lw pair; not the jal
        result = tp.transfer(hits)
        self.assertEqual(result["contradictions"], [])
        placed = {v["donor_address"]: (v["target_address"], v["functions"]) for v in result["symbols"].values()}
        # get_b's reading of the getter would have put 0x80123460 at 0x80223466 against the user's 0x80223470
        self.assertEqual(placed[0x80123460], (0x80223470, ["t_user"]))
        self.assertNotIn(0x80123456, placed)
        self.assertEqual(placed[0x80101000], (0x80205000, ["t_get"]))       # the callee both readings share
        self.assertEqual({h.status for h in hits}, {"candidate"})

    def test_one_reading_only_still_votes(self):
        getter = donor_fn(0x8022, 0x3466, 0x80205000)
        image = image_with(self.delta, {0x80200100: getter})
        index = tp.Index({"a": self.a}, {"D": {"root": "/x", "image": "/x", "delta": 0, "include": []}})
        hits = tp.scan(image, self.delta, [("t_get", 0x80200100, len(getter))], index)
        self.assertEqual(tp.ambiguous_sites(hits), set())
        self.assertEqual({v["target_address"] for v in tp.transfer(hits)["symbols"].values()},
                         {0x80223466, 0x80205000})


class UnitDefinitionTest(unittest.TestCase):
    """A unit found whole places each of its functions, also one the catalogue gives no hit for."""
    delta = 0x80200000 - 0x1000

    def test_a_function_the_catalogue_cut_short_is_defined_by_its_unit(self):
        f1 = donor_fn(0x8012, 0x3456, 0x80101000)
        f2 = donor_fn(0x8012, 0x3456, 0x80101100, extra=(0x00851021,))
        unit = make_unit([("f1", f1), ("f2", f2)], 0x80100000)
        index = tp.Index({"u": unit}, {})
        t1 = donor_fn(0x8022, 0x3466, 0x80205000)
        t2 = donor_fn(0x8022, 0x3466, 0x80205100, extra=(0x00851021,))
        image = image_with(self.delta, {0x80200100: t1 + t2})
        # the catalogue cuts the second function's last two instructions off as a function of its own
        funcs = [("t1", 0x80200100, len(t1)), ("t2", 0x80200100 + len(t1), len(t2) - 8),
                 ("stub", 0x80200100 + len(t1) + len(t2) - 8, 8)]
        hits = tp.scan(image, self.delta, funcs, index)
        self.assertEqual([h.name for h in hits], ["t1"])
        found = tp.transfer(hits)["definitions"]
        self.assertNotIn(("D", 0x80100000 + len(f1)), found)
        jobs, _partial = tp.plan(hits, image, self.delta, {f[1] for f in funcs})
        self.assertEqual([job.whole for job in jobs], [True])
        found = tp.unit_definitions(jobs, found)
        self.assertEqual(found[("D", 0x80100000 + len(f1))], {0x80200100 + len(t1)})
        self.assertEqual(found[("D", 0x80100000)], {0x80200100})
        # a caller's own instruction is then confirmed by the unit around the callee
        self.assertEqual(tp.classify_pin("D", 0x80100000 + len(f1), 0x80200100 + len(t1), {}, found, {"caller"},
                                         span=(0x80200400, 0x80200440)), ("declare", "definition"))

    def test_a_single_function_job_adds_nothing(self):
        unit = make_unit([("f", donor_fn(0x8012, 0x3456, 0x80101000))], 0x80100000)
        job = tp.Job([], unit, 0x80200100, unit["core_size"])
        self.assertEqual(tp.unit_definitions([job], {}), {})


class VerifiedDefinitionTest(unittest.TestCase):
    """A unit verified exact places its functions for the units that wait for one of them."""

    def setUp(self):
        self.f1 = donor_fn(0x8012, 0x3456, 0x80101000)
        self.f2 = donor_fn(0x8012, 0x3456, 0x80101100, extra=(0x00851021,))
        self.unit = make_unit([("f1", self.f1), ("f2", self.f2)], 0x80100000)

    def row(self, verdict, **more):
        return {"verdict": verdict, "target_vram": 0x80200100, "_unit": self.unit, **more}

    def test_an_exact_unit_defines_each_of_its_functions_once(self):
        found = {}
        self.assertEqual(tp.verified_definitions([self.row(tp.EXACT_VERDICT)], found), 2)
        self.assertEqual(found, {("D", 0x80100000): {0x80200100}, ("D", 0x80100000 + len(self.f1)): {0x80200100 + len(self.f1)}})
        self.assertEqual(tp.verified_definitions([self.row(tp.EXACT_VERDICT)], found), 0)      # nothing new
        # the caller that waited (its own instruction was the only vote) is now settled
        table = {("D", 0x80100000 + len(self.f1)): {"target_address": 0x80200100 + len(self.f1), "hits": 1,
                                                     "functions": ["caller"], "supporters": [("caller", 0x80200400, 0x40)]}}
        self.assertEqual(tp.classify_pin("D", 0x80100000 + len(self.f1), 0x80200100 + len(self.f1), table, found,
                                         {"caller"}, span=(0x80200400, 0x80200440)), ("declare", "definition"))

    def test_waiting_refused_and_common_shape_units_define_nothing(self):
        found = {}
        rows = [self.row(tp.ASSUMED_VERDICT), self.row("mismatch"), self.row(tp.EXACT_VERDICT, template=True),
                {"verdict": tp.EXACT_VERDICT, "target_vram": 0x80200100}]                       # no unit kept
        self.assertEqual(tp.verified_definitions(rows, found), 0)
        self.assertEqual(found, {})

    def test_a_function_also_found_elsewhere_stays_unsettled(self):
        found = {("D", 0x80100000): {0x80300000}}                                               # the scan's hit
        tp.verified_definitions([self.row(tp.EXACT_VERDICT)], found)
        self.assertEqual(tp.classify_pin("D", 0x80100000, 0x80200100, {}, found, {"caller"},
                                         span=(0x80200400, 0x80200440))[0], "unchecked")


class CrossDonorDefinitionTest(unittest.TestCase):
    """What a verified unit defines settles the units of other donors too, and its global data
    settles the units that refer to it (measured on a title: ``bcopy`` verified from one donor's
    source, awaited by a unit of another; a queue in one unit's ``.bss``, awaited by its user)."""

    def setUp(self):
        self.f1 = donor_fn(0x8012, 0x3456, 0x80101000)
        self.unit = make_unit([("bcopy", self.f1)], 0x80100000)
        self.row = {"verdict": tp.EXACT_VERDICT, "target_vram": 0x80200100, "_unit": self.unit,
                    "_bases": {".bss": 0x80300000}, "_defines": [("gQueue", 0x80300008, 0x80150008, ".bss")]}
        self.definitions, self.verified, self.defined_as = {}, {}, {}
        tp.verified_definitions([self.row], self.definitions, self.verified, self.defined_as)

    def pin(self, donor, daddr, taddr, name, same_code=None, span=(0x80200400, 0x80200440), definitions=None):
        return tp.classify_pin(donor, daddr, taddr, {}, self.definitions if definitions is None else definitions,
                               {"caller"}, span=span, name=name, verified=self.verified, same_code=same_code)

    def test_a_verified_unit_defines_its_global_data_for_its_own_donor(self):
        self.assertEqual(self.definitions[("D", 0x80150008)], {0x80300008})
        self.assertIn("gQueue is defined there by the verified unit", self.defined_as[("D", 0x80150008, 0x80300008)])
        self.assertEqual(self.pin("D", 0x80150008, 0x80300008, "gQueue"), ("declare", "definition"))

    def test_another_donors_unit_is_settled_by_the_same_name(self):
        decision, origin = self.pin("E", 0x80400000, 0x80200100, "bcopy")
        self.assertEqual(decision, "declare")
        self.assertIn("defines bcopy there (the same name)", origin)
        self.assertEqual(self.pin("E", 0x80400000, 0x80200104, "bcopy")[0], "unchecked")        # not where it starts
        self.assertEqual(self.pin("E", 0x80400000, 0x80200100, "memcpy")[0], "unchecked")       # another name, no code

    def test_a_made_up_name_proves_nothing_across_donors(self):
        unit = make_unit([("func_80101000", self.f1)], 0x80100000)
        verified = {}
        tp.verified_definitions([{**self.row, "_unit": unit, "_defines": []}], {}, verified, {})
        self.assertEqual(tp.classify_pin("E", 0x80400000, 0x80200100, {}, {}, {"caller"}, span=(0x80200400, 0x80200440),
                                         name="func_80101000", verified=verified)[0], "unchecked")

    def test_another_name_is_settled_when_this_donors_code_is_the_same(self):
        seen = []

        def same(daddr, taddr, size):
            seen.append((daddr, taddr, size))
            return True
        decision, origin = self.pin("E", 0x80400000, 0x80200100, "copy_bytes", same_code=same)
        self.assertEqual((decision, seen), ("declare", [(0x80400000, 0x80200100, len(self.f1))]))
        self.assertIn("this donor's code for copy_bytes is the same", origin)
        self.assertEqual(self.pin("E", 0x80400000, 0x80200100, "copy_bytes", same_code=lambda *a: False)[0], "unchecked")

    def test_never_from_inside_the_unit_or_when_the_function_stands_elsewhere_too(self):
        self.assertEqual(self.pin("E", 0x80400000, 0x80200100, "bcopy", span=(0x80200000, 0x80200200))[0], "unchecked")
        elsewhere = {("E", 0x80400000): {0x80200100, 0x80250000}}
        self.assertEqual(self.pin("E", 0x80400000, 0x80200100, "bcopy", definitions=elsewhere)[0], "unchecked")

    def test_a_common_shape_unit_defines_nothing_for_anyone(self):
        verified = {}
        tp.verified_definitions([{**self.row, "template": True}], {}, verified, {})
        self.assertEqual(verified, {})


class DataContentPinTest(unittest.TestCase):
    """A named datum of the donor whose bytes stand unchanged at the pin, and a fixed address, settle a pin
    the unit's own instruction was the only vote for (measured: the video mode tables, ``osResetType``)."""
    TABLE = {("D", 0x80150000): {"target_address": 0x80300000, "hits": 1, "functions": ["caller"],
                                 "supporters": [("caller", 0x80200400, 0x40)]}}

    def pin(self, daddr, taddr, same_data, table=None, span=(0x80200400, 0x80200440)):
        return tp.classify_pin("D", daddr, taddr, self.TABLE if table is None else table, {}, {"caller"},
                               span=span, name="osViModeNtscLan1", verified={}, same_data=same_data)

    def test_equal_bytes_declare_the_pin(self):
        decision, origin = self.pin(0x80150000, 0x80300000, lambda d, t: 64)
        self.assertEqual(decision, "declare")
        self.assertIn("64 bytes of osViModeNtscLan1 stand at that address", origin)
        self.assertEqual(self.pin(0x80150000, 0x80300000, lambda d, t: 0), ("self", ""))
        self.assertEqual(self.pin(0x80160000, 0x80310000, lambda d, t: 32, table={})[0], "declare")

    def test_a_fixed_address_is_declared_even_when_only_the_unit_votes(self):
        table = {("D", 0x8000030C): {"target_address": 0x8000030C, "hits": 1, "functions": ["caller"],
                                     "supporters": [("caller", 0x80200400, 0x40)]}}
        self.assertEqual(self.pin(0x8000030C, 0x8000030C, lambda d, t: 0, table=table), ("declare", "fixed-address"))

    def test_never_inside_the_units_own_bytes(self):
        self.assertEqual(self.pin(0x80150000, 0x80200420, lambda d, t: 64, table={})[0], "unchecked")


class CataloguedFunctionPinTest(unittest.TestCase):
    """A function whose address a unit takes, held by no donor unit: settled when this donor's code for the
    name is the catalogued function at the pin and no other."""

    def pin(self, same_function):
        return tp.classify_pin("D", 0x80150000, 0x80300000, {}, {}, {"caller"}, span=(0x80200400, 0x80200440),
                               name="__osExceptionPreamble", verified={}, same_function=same_function)

    def test_the_same_code_at_the_catalogued_function_declares_it(self):
        decision, origin = self.pin(lambda d, t: 16)
        self.assertEqual(decision, "declare")
        self.assertIn("16-byte function catalogued there", origin)
        self.assertEqual(self.pin(lambda d, t: 0)[0], "unchecked")


class CopiesTest(unittest.TestCase):
    """A routine a library holds twice (with other data) is paired with the donor's copies by their order."""

    def setUp(self):
        import tempfile, types
        self.tmp = tempfile.TemporaryDirectory()
        twin = lambda hi, lo: bytes.fromhex(f"3c02{hi:04x}" f"8c42{lo:04x}" "03e00008" "00000000")
        other = bytes.fromhex("00851021" "03e00008" "00000000" "00000000")
        image = twin(0x8010, 0x0010) + other + twin(0x8010, 0x0020) + other
        path = Path(self.tmp.name) / "donor.bin"
        path.write_bytes(image)
        self.index = types.SimpleNamespace(donors={"D": {"image": str(path), "delta": 0x80100000}})
        self.second = {"vram": 0x80100020, "size": 16}                   # the donor function: its second copy
        self.hit = lambda vram: types.SimpleNamespace(kind="exact", function=self.second, vram=vram, donor="D",
                                                      placed=False, notes=[])

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_copies_of_a_donor_function_are_found_by_its_code(self):
        self.assertEqual(tp.donor_copies(self.index.donors["D"], 0x80100020, 16), [0x80100000, 0x80100020])

    def test_the_second_copy_there_is_the_second_copy_here(self):
        first, second = self.hit(0x80200100), self.hit(0x80200500)
        kept, fixed = tp.pair_copies([first, second], self.index)
        self.assertEqual((fixed, [h.vram for h in kept]), (1, [0x80200500]))
        self.assertTrue(second.placed)

    def test_other_counts_decide_nothing(self):
        hits = [self.hit(0x80200100), self.hit(0x80200500), self.hit(0x80200900)]
        kept, fixed = tp.pair_copies(hits, self.index)
        self.assertEqual((fixed, len(kept)), (0, 3))
        self.assertFalse(any(h.placed for h in hits))


class SameFunctionCodeTest(unittest.TestCase):
    def test_equal_code_with_other_addresses_is_the_same_function(self):
        donor = donor_fn(0x8012, 0x3456, 0x80101000)
        target = donor_fn(0x8022, 0x1234, 0x80205000)
        self.assertTrue(tp.same_function_code(donor, 0x80100000, target, 0x80200100))
        self.assertFalse(tp.same_function_code(donor, 0x80100000, target + bytes(4), 0x80200100))    # another length
        other = donor_fn(0x8022, 0x1234, 0x80205000, extra=(0x00851021,))
        self.assertFalse(tp.same_function_code(donor + bytes(4), 0x80100000, other, 0x80200100))    # another word

    def test_a_fixed_address_must_be_the_same_address(self):
        status = bytes.fromhex("3c02a404" "8c420010" "03e00008" "00000000")      # lui/lw 0xA4040010 (SP_STATUS)
        other = bytes.fromhex("3c02a480" "8c420018" "03e00008" "00000000")       # lui/lw 0xA4800018 (SI_STATUS)
        self.assertTrue(tp.same_function_code(status, 0x80100000, status, 0x80200000))
        self.assertFalse(tp.same_function_code(status, 0x80100000, other, 0x80200000))


class DataDefinesTest(unittest.TestCase):
    def test_global_symbols_of_placed_sections_only(self):
        import types
        section = lambda name: types.SimpleNamespace(name=name)
        symbol = lambda name, bind, sec, value: types.SimpleNamespace(name=name, bind=bind, section=sec, value=value)
        elf = types.SimpleNamespace(sections=[section(""), section(".text"), section(".bss"), section(".data")],
                                    symbols=[symbol("f", 1, 1, 0), symbol("gQueue", 1, 2, 8), symbol("local", 0, 2, 0),
                                             symbol("gTable", 1, 3, 4), symbol("ext", 1, 0, 0)])
        found = tp.data_defines(elf, {".bss": 0x80300000}, {".bss": 0x80150000})
        self.assertEqual(found, [("gQueue", 0x80300008, 0x80150008, ".bss")])            # .data has no base: nothing
        self.assertEqual(tp.data_defines(elf, {".data": 0x80400000}, {}), [("gTable", 0x80400004, None, ".data")])


class PlacesTest(unittest.TestCase):
    """Units of different sizes beginning at one address are different places."""

    def test_a_stub_unit_does_not_stand_in_for_the_unit_that_begins_with_it(self):
        stub = make_unit([("noop", code(JR_RA))], 0x80100000, donor="A", source="src/noop.c")
        whole = make_unit([("head", code(JR_RA, NOP)), ("f", donor_fn(0x8012, 0x3456, 0x80101000))], 0x80100400,
                          donor="B", source="src/unit.c")
        same = make_unit([("head", code(JR_RA, NOP)), ("f", donor_fn(0x8012, 0x3456, 0x80101000))], 0x80100400,
                         donor="C", source="src/unit.c")
        jobs = [tp.Job([], stub, 0x80200100, stub["core_size"]), tp.Job([], whole, 0x80200100, whole["core_size"]),
                tp.Job([], same, 0x80200100, same["core_size"]), tp.Job([], stub, 0x80200300, stub["core_size"])]
        groups = tp.places(jobs)
        self.assertEqual(sorted(groups), [(0x80200100, 4), (0x80200100, whole["core_size"]), (0x80200300, 4)])
        self.assertEqual([job.unit["donor"] for job in groups[(0x80200100, whole["core_size"])]], ["B", "C"])
        # both verified exact: the larger row stays, the stub inside it is superseded
        rows = [{"verdict": tp.EXACT_VERDICT, "target_vram": 0x80200100, "size": 4, "notes": []},
                {"verdict": tp.EXACT_VERDICT, "target_vram": 0x80200100, "size": whole["core_size"], "notes": []}]
        self.assertEqual([row["verdict"] for row in tp.select_rows(rows)], ["superseded", tp.EXACT_VERDICT])


class SectionChainTest(unittest.TestCase):
    """``.data`` and ``.bss`` sections of different units lying end to end settle each other's address."""

    class _Section:
        def __init__(self, size, align=16):
            self.size, self.align = size, align

    class _Elf:
        def __init__(self, **sections):
            self.sections_ = {"." + name: section for name, section in sections.items()}

        def section(self, name):
            return self.sections_.get(name)

    def unit(self, text, *, bss=None, data=None, equal=(".data",)):
        # bss / data: (address the unit's code points to, size of the compiled section)
        sections, bases = {}, {}
        for name, item in (("bss", bss), ("data", data)):
            if item is not None:
                sections[name] = self._Section(item[1], item[2] if len(item) > 2 else 16)
                bases["." + name] = item[0]
        return {"key": ("D", f"u{text:x}.c", text), "text": (text, 0x40), "elf": self._Elf(**sections),
                "bases": bases, "equal": set(equal)}

    def chains(self, units, rom=bytes(0x200)):
        return [[(piece["name"], piece["address"]) for piece in chain]
                for chain in tp.section_chains(tp.section_pieces(units), rom, lambda vram: vram - 0x80300000)]

    def test_a_segment_that_does_not_start_on_the_largest_alignment(self):
        # every section 16-aligned within its segment, the segment itself placed at 8 modulo 16
        a = self.unit(0x80200000, bss=(0x80400008, 0x20))
        b = self.unit(0x80200100, bss=(0x80400028, 0x74))              # ends at …9C: the next starts at …A8
        c = self.unit(0x80200200, bss=(0x804000A8, 0x10))
        self.assertEqual(self.chains([c, a, b]), [[(".bss", 0x80400008), (".bss", 0x80400028), (".bss", 0x804000A8)]])
        d = self.unit(0x80200300, bss=(0x804000C0, 0x10))              # at 0 modulo 16: not laid out with them
        self.assertEqual(len(self.chains([a, b, c, d])), 1)
        self.assertNotIn((".bss", 0x804000C0), self.chains([a, b, c, d])[0])
        rom = bytearray(0x200)
        rom[0x108:0x110] = b"\x80\x30\x01\x08" * 2
        rom[0x118:0x120] = b"\x12\x34\x56\x78" * 2
        e = self.unit(0x80200400, data=(0x80300108, 0x10))
        f = self.unit(0x80200500, data=(0x80300118, 0x10))
        self.assertEqual(self.chains([e, f], bytes(rom)), [[(".data", 0x80300108), (".data", 0x80300118)]])

    def test_bss_abutting_at_the_compiled_sizes_in_the_order_of_the_code(self):
        a = self.unit(0x80200000, bss=(0x80400000, 0x20))
        b = self.unit(0x80200100, bss=(0x80400020, 0x74))              # 0x74 bytes: the next starts at 0x…A0
        c = self.unit(0x80200200, bss=(0x804000A0, 0x10))
        self.assertEqual(self.chains([c, a, b]), [[(".bss", 0x80400000), (".bss", 0x80400020), (".bss", 0x804000A0)]])
        # one unit alone, or a hole between two, is no chain
        self.assertEqual(self.chains([a]), [])
        self.assertEqual(self.chains([a, c]), [])
        # the same addresses with the units' code in the other order: a coincidence, not a layout
        late = self.unit(0x80200300, bss=(0x80400000, 0x20))
        self.assertEqual(self.chains([late, b]), [])
        # code that overlaps the previous unit's is not after it
        inside = self.unit(0x80200020, bss=(0x80400020, 0x74))
        self.assertEqual(self.chains([a, inside]), [])
        # two units cannot own one address: neither claim is a piece
        twin = self.unit(0x80200400, bss=(0x80400020, 0x10))
        self.assertEqual(self.chains([a, b, twin]), [])

    def test_data_needs_equal_contents_zero_gaps_and_the_read_only_thresholds(self):
        rom = bytearray(0x200)
        rom[0x100:0x108] = b"\x80\x40\x10\x50\x80\x40\x10\xD0"            # eight non-zero bytes in the first piece
        a = self.unit(0x80200000, data=(0x80300100, 0x10, 8))
        b = self.unit(0x80200100, data=(0x80300110, 0x10, 8))
        self.assertEqual(self.chains([a, b], bytes(rom)), [[(".data", 0x80300100), (".data", 0x80300110)]])
        # contents the comparison did not find equal: no piece
        differs = self.unit(0x80200100, data=(0x80300110, 0x10, 8), equal=())
        self.assertEqual(self.chains([a, differs], bytes(rom)), [])
        # all but zeros over the chain confirms nothing (SETTLE_MIN_NONZERO)
        self.assertEqual(self.chains([a, b], bytes(0x200)), [])
        # a gap that holds something is another unit's data
        far = self.unit(0x80200100, data=(0x80300118, 0x10, 8))
        self.assertEqual(self.chains([a, far], bytes(rom)), [])

    def test_settlement_records_a_base_and_promotes_nothing(self):
        def row(vram, base, size, verdict):
            return {"verdict": verdict, "donor": "D", "donor_unit": f"u{vram:x}.c", "target_vram": vram, "size": 0x40,
                    "notes": [], "strict": {"sections": []}, "_object": self._Elf(bss=self._Section(size)),
                    "_elf_spec": (None, None, {".bss": base}, None)}
        rows = [row(0x80200000, 0x80400000, 0x20, tp.ASSUMED_VERDICT), row(0x80200100, 0x80400020, 0x40, tp.EXACT_VERDICT),
                row(0x80200200, 0x80400100, 0x10, tp.ASSUMED_VERDICT)]          # the third abuts nothing
        target = SimpleNamespace(rom=bytes(16), vram_to_rom=lambda vram: vram - 0x80300000)
        settled = {}
        recorded = tp.settle_sections(rows, target, settled)
        self.assertEqual(sorted(key[2] for key in recorded), [0x80200000, 0x80200100])
        item = settled[("D", "u80200000.c", 0x80200000)][".bss"]
        self.assertEqual((item["base"], item["size"]), (0x80400000, 0x20))
        self.assertTrue(item["origin"].startswith(tp.CHAIN_ORIGIN + ": "))
        self.assertEqual([r["verdict"] for r in rows], [tp.ASSUMED_VERDICT, tp.EXACT_VERDICT, tp.ASSUMED_VERDICT])
        self.assertEqual(tp.settle_sections(rows, target, settled), [])            # nothing new the second time
        # a pair the caller banned (a chain that did not hold once rows were selected) is not settled
        banned = {(("D", "u80200000.c", 0x80200000), ".bss")}
        self.assertNotIn(("D", "u80200000.c", 0x80200000), {k for k in tp.settle_sections(rows, target, {}, banned)})

    def test_an_exact_row_resting_on_a_chain_that_no_longer_holds_is_reported(self):
        def row(vram, base, size, origin=None):
            return {"verdict": tp.EXACT_VERDICT, "donor": "D", "donor_unit": f"u{vram:x}.c", "target_vram": vram,
                    "size": 0x40, "notes": [], "strict": {"sections": []}, "_bases": {".bss": base},
                    "pin_origin": {"section:.bss": origin} if origin else {},
                    "_object": self._Elf(bss=self._Section(size)), "_elf_spec": (None, None, {".bss": base}, None)}
        target = SimpleNamespace(rom=bytes(16), vram_to_rom=lambda vram: vram - 0x80300000)
        origin = tp.chain_origin({"kind": "bss", "name": ".bss"})
        held = [row(0x80200000, 0x80400000, 0x20, origin), row(0x80200100, 0x80400020, 0x40)]
        listing, broken = tp.chain_report(held, target)
        self.assertEqual(broken, set())
        self.assertEqual([(u["address"], u["rests_on_it"]) for u in listing[0]["units"]],
                         [("0x80400000", True), ("0x80400020", False)])
        alone = [row(0x80200000, 0x80400000, 0x20, origin)]                         # its neighbour is gone
        self.assertEqual(tp.chain_report(alone, target)[1], {(("D", "u80200000.c", 0x80200000), ".bss")})


class FunctionsInsideTest(unittest.TestCase):
    """Every catalogued function wholly inside a unit's compared bytes is reproduced by the unit."""
    delta = 0x80200000 - 0x1000

    def test_parts_of_a_cut_function_count_and_a_function_running_on_does_not(self):
        body = code(*([0x24020001] * 12))                                   # the unit: 0x30 bytes
        image = image_with(self.delta, {0x80200100: body + bytes(8) + code(0x24020002, JR_RA)})
        functions = [("head", 0x80200100, 0x10), ("cut", 0x80200110, 0x18),     # inside
                     ("tail", 0x80200128, 0x10),                             # ends in the unit's zero padding
                     ("next", 0x80200138, 0x8),                              # after the unit
                     ("before", 0x802000F0, 0x10)]
        self.assertEqual(tp.functions_inside(0x80200100, 0x30, functions, image, self.delta),
                         [0x80200100, 0x80200110, 0x80200128])
        # a catalogued function that starts inside and holds other code past the unit's end
        merged = [("head", 0x80200100, 0x10), ("merged", 0x80200110, 0x30)]
        self.assertEqual(tp.functions_inside(0x80200100, 0x30, merged, image, self.delta), [0x80200100])


class IndexPersistenceTest(unittest.TestCase):
    def test_round_trip(self):
        unit = make_unit([("f", donor_fn(0x8012, 0x3456, 0x80101000))], 0x80100000)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            tp.Index({"k": unit}, {"D": {}}).save(path)
            again = tp.Index.load(path)
        self.assertEqual(list(again.by_sig), list(tp.Index({"k": unit}).by_sig))

    def test_donor_toml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "donors.toml"
            path.write_text('[donor.a]\nroot = "/r"\nimage = "rom.z64"\nvram_delta = "0x7FFFF400"\n'
                            'manifests = ["m.tsv=ido5.3", { path = "k.tsv", pipeline = "kmc+as" }]\n')
            (donor,) = tp.load_donors(path)
        self.assertEqual(donor.image, Path("/r/rom.z64"))
        self.assertEqual(donor.vram_delta, 0x7FFFF400)
        self.assertEqual([(m.path, m.pipeline) for m in donor.manifests], [("m.tsv", "ido5.3"), ("k.tsv", "kmc+as")])


GAMES = Path(os.environ.get("N64DK_FIXTURE_ROOT", "/nonexistent"))
TOOLCHAIN = Path(os.environ.get("N64DK_TOOLCHAIN", "/nonexistent"))
BTX, NAGANO = GAMES / "BattleTanx", GAMES / "nagano"


@unittest.skipUnless((TOOLCHAIN / "ido5.3").is_dir(), "IDO toolchain not configured (N64DK_TOOLCHAIN)")
@unittest.skipUnless((BTX / "baseroms/us/baserom.z64").is_file() and (NAGANO / "baseroms/us/baserom.z64").is_file(),
                     "BattleTanx / nagano images not present")
class RealTransplantTest(unittest.TestCase):
    def test_battletanx_libultra_into_nagano(self):
        from n64dk.match.engine import Target
        from n64dk.rom import Rom
        donor = tp.DonorSpec("BattleTanx", BTX, BTX / "baseroms/us/baserom.z64",
                             [tp.Manifest("config/us/matching_units.tsv", "ido5.3")],
                             catalogue=[BTX / "config/us/recomp_function_boundaries.toml"],
                             include=[BTX / "include"])
        with tempfile.TemporaryDirectory() as tmp:
            index, info = tp.build_index([donor], Path(tmp) / "index.json")
            self.assertGreater(info["indexed"], 100)
            _again, info2 = tp.build_index([donor], Path(tmp) / "index.json")
            self.assertEqual(info2["indexed"], 0)                      # incremental
            target = Target.from_rom(Rom.load(NAGANO / "baseroms/us/baserom.z64"))
            functions = tp.load_functions([NAGANO / "analysis/survey/functions.toml"], target.delta)
            near = [f for f in functions if 0x8004E000 <= f[1] < 0x80050000]
            report = tp.run(None, target, near, index, out_dir=Path(tmp) / "out", jobs=2,
                            toolchain_root=TOOLCHAIN)
            self.assertGreaterEqual(report["verdicts"].get(tp.EXACT_VERDICT, 0), 3, report["verdicts"])
            self.assertTrue(any(p.name.startswith("proposed_BattleTanx") for p in (Path(tmp) / "out").iterdir()))


if __name__ == "__main__":
    unittest.main()


class CommonShapeTest(unittest.TestCase):
    def test_a_unit_is_a_common_shape_only_when_every_function_of_it_is(self):
        from types import SimpleNamespace as Hit
        self.assertTrue(tp.common_shape([Hit(template=True)]))
        self.assertTrue(tp.common_shape([Hit(template=True), Hit(template=True)]))
        # one getter among functions that stand nowhere else: the unit's place is not in doubt
        self.assertFalse(tp.common_shape([Hit(template=True), Hit(template=False), Hit(template=False)]))
        self.assertFalse(tp.common_shape([Hit(template=False)]))
        self.assertFalse(tp.common_shape([]))


class SourceOrderTest(unittest.TestCase):
    """A shape found at several places is placed by the functions written around it."""

    def hit(self, function, unit_source, vram):
        unit = {"donor": "T", "source": unit_source, "functions": [function]}
        site = {"donor_symbol": "s", "donor_address": 1, "target_address": 2, "addend": 0, "kind": "lo", "index": 0}
        function.setdefault("vram", 0x80100000)
        return tp.Hit(function["name"], vram, 8, unit, function, "exact", sites=[site])

    def test_a_repeated_shape_is_placed_next_to_its_uniquely_found_neighbour(self):
        first, getter, other_getter = {"name": "first"}, {"name": "get"}, {"name": "get2"}
        order = {("T", "u/first.c"): ("u.c", 0), ("T", "u/get.c"): ("u.c", 1), ("T", "v/get2.c"): ("v.c", 0)}
        functions = [("a", 0x100, 8), ("b", 0x108, 8), ("c", 0x200, 8)]
        hits = [self.hit(first, "u/first.c", 0x100),
                self.hit(getter, "u/get.c", 0x108), self.hit(getter, "u/get.c", 0x200),        # same shape twice
                self.hit(other_getter, "v/get2.c", 0x108), self.hit(other_getter, "v/get2.c", 0x200)]
        self.assertEqual(tp.corroborate(hits, functions, order), 1)
        self.assertEqual([(h.function["name"], h.vram) for h in hits if h.placed], [("get", 0x108)])
        tp.transfer(hits)
        self.assertEqual([(h.function["name"], h.vram, h.template) for h in hits[1:]],
                         [("get", 0x108, False), ("get", 0x200, True), ("get2", 0x108, True), ("get2", 0x200, True)])

    def test_nothing_is_placed_without_an_order(self):
        getter = {"name": "get"}
        hits = [self.hit(getter, "u/get.c", 0x108), self.hit(getter, "u/get.c", 0x200)]
        self.assertEqual(tp.corroborate(hits, [("b", 0x108, 8), ("c", 0x200, 8)], {}), 0)


class AlternativesTest(unittest.TestCase):
    def job(self, donor, source, functions=1, masked="m", flags=("-O2",), pipeline="kmc+as"):
        unit = {"donor": donor, "source": source, "functions": [{}] * functions, "unit_masked": masked,
                "flags": list(flags), "pipeline": pipeline}
        return tp.Job([], unit, 0x80001000, 0x40)

    def test_library_units_never_come_before_project_and_sdk_units(self):
        jobs = [self.job("t0123456789abcdef_main", "main/func_80001000.c"),
                self.job("sdk_2_0J_libgultra_rom", "src/os/createthread.c", functions=3),
                self.job("Rugrats64", "src/runtime/create_thread.c")]
        ordered = sorted(jobs, key=tp.alternative_order)
        self.assertEqual([job.unit["donor"] for job in ordered],
                         ["Rugrats64", "sdk_2_0J_libgultra_rom", "t0123456789abcdef_main"])

    def test_the_same_build_from_several_library_versions_is_tried_once(self):
        jobs = [self.job(f"sdk_2_0{v}_libgultra_rom", "/x/ultralib/src/os/createthread.c",
                         flags=("-O3", f"-DBUILD_VERSION=VERSION_{v}")) for v in "FHIJKL"]
        jobs.append(self.job("sdk_2_0J_libgultra_rom", "/x/ultralib/src/os/createthread.c", masked="other",
                             flags=("-O3", "-DBUILD_VERSION=VERSION_J")))
        jobs.append(self.job("sdk_2_0J_libultra_rom", "/x/ultralib/src/os/createthread.c", pipeline="ido5.3",
                             flags=("-O1",)))
        kept = tp.distinct_alternatives(jobs)
        self.assertEqual(len(kept), 3)                               # one per distinct code, compiler and flags
        self.assertGreaterEqual(tp.MAX_ALTERNATIVES, 8)

    def test_different_sources_with_the_same_code_are_both_kept(self):
        jobs = [self.job("Rugrats64", "src/runtime/create_thread.c"), self.job("sdk_2_0J_libgultra_rom", "src/os/createthread.c")]
        self.assertEqual(len(tp.distinct_alternatives(jobs)), 2)


class AssumedRowsTest(unittest.TestCase):
    def test_exact_rows_are_chosen_before_rows_awaiting_symbols(self):
        rows = [{"verdict": tp.ASSUMED_VERDICT, "target_vram": 0x80001000, "size": 0x40, "notes": []},
                {"verdict": tp.EXACT_VERDICT, "target_vram": 0x80001020, "size": 0x20, "notes": []},
                {"verdict": tp.ASSUMED_VERDICT, "target_vram": 0x80002000, "size": 0x20, "notes": []}]
        chosen = tp.select_rows(rows)
        self.assertEqual([row["verdict"] for row in chosen], ["superseded", tp.EXACT_VERDICT, tp.ASSUMED_VERDICT])

    def test_transfer_reports_where_donor_functions_were_found(self):
        callee = code(0x03E00008, 0x24020007)
        caller = donor_fn(0x8030, 0x1234, 0x80200100)
        unit = make_unit([("callee", callee)], 0x80200100, source="src/b.c")
        index = tp.Index({"u": unit}, {"D": {"root": ".", "include": [], "delta": 0, "image": ""}})
        image = image_with(0x80000000, {0x80001000: callee})
        hits = tp.scan(image, 0x80000000, [("func_80001000", 0x80001000, len(callee))], index)
        found = tp.transfer(hits)["definitions"]
        self.assertEqual(found.get(("D", 0x80200100)), {0x80001000})
        del caller


class RodataSettlementTest(unittest.TestCase):
    """Rows that wait only for their read-only section's address are settled by chains of units
    lying end to end; a lone small constant confirms nothing."""

    class _Target:
        def __init__(self, rom):
            self.rom = rom

        @staticmethod
        def vram_to_rom(address):
            return address - 0x80000000

    def _piece(self, address, size, align=8):
        return {"address": address, "size": size, "align": align, "name": ".rodata", "row": {}}

    def test_chains_follow_alignment_and_zero_gaps(self):
        from n64dk.match import transplant
        rom = bytearray(0x100)
        rom[0x14:0x18] = b"\x01\x02\x03\x04"                 # non-zero gap byte region
        pieces = [self._piece(0x80000000, 4), self._piece(0x80000008, 8), self._piece(0x80000010, 4),
                  self._piece(0x80000018, 4)]
        chains = transplant.rodata_chains(pieces, bytes(rom), self._Target.vram_to_rom)
        # 0x00..0x04 then (aligned to 8) 0x08..0x10 then 0x10..0x14; 0x18 is aligned too but the
        # gap 0x14..0x18 is not zero
        self.assertEqual([[p["address"] - 0x80000000 for p in chain] for chain in chains], [[0, 8, 0x10], [0x18]])

    def test_a_lone_float_is_not_promoted_and_a_chain_is(self):
        from unittest import mock
        from n64dk.match import transplant
        from n64dk.match.verify import Verdict
        rom = bytearray(0x100)
        rom[0:4] = b"\x45\x1e\xc0\x00"
        rom[8:16] = b"\x3f\x80\x00\x01\x40\x20\x10\x08"             # 9 non-zero bytes in all
        def row(base, size, waiting=True):
            return {"verdict": transplant.ASSUMED_VERDICT if waiting else transplant.EXACT_VERDICT,
                    "awaiting": {".rodata": f"0x{base:08X}"} if waiting else None, "target_vram": base, "size": 8,
                    "strict": {"sections": [{"section": ".rodata", "base": f"0x{base:08X}", "size": size,
                                             "compared": size, "status": "equal"}]},
                    "_object": object(), "_elf_spec": (None, {}, {}, None)}
        alone = [row(0x80000000, 4)]
        found = mock.patch.object(transplant.cmp, "find_section",
                                  lambda elf, name: type("S", (), {"align": 8, "name": name})())
        exact = mock.patch("n64dk.match.verify.strict_compare",
                           lambda *a, **k: Verdict("exact", True, sections=[]))
        with found, exact:
            report = transplant.settle_rodata(alone, self._Target(bytes(rom)), set())
            self.assertEqual(report["promoted"], 0)
            self.assertEqual(alone[0]["verdict"], transplant.ASSUMED_VERDICT)
            both = [row(0x80000000, 4), row(0x80000008, 8, waiting=False)]
            report = transplant.settle_rodata(both, self._Target(bytes(rom)), set())
        self.assertEqual((report["chains"], report["promoted"]), (1, 1))
        self.assertEqual(both[0]["verdict"], transplant.EXACT_VERDICT)
        self.assertIn("end to end", both[0]["corroborated"][".rodata"])

    def test_the_strict_verdict_still_decides(self):
        from unittest import mock
        from n64dk.match import transplant
        from n64dk.match.verify import Verdict
        rom = bytearray(0x100)
        rom[0:8] = b"\x45\x1e\xc0\x00\x41\x00\x00\x00"
        rom[8:16] = b"\x3f\x80\x00\x01\x40\x20\x10\x08"
        rows = []
        for base in (0x80000000, 0x80000008):
            rows.append({"verdict": transplant.ASSUMED_VERDICT, "awaiting": {".rodata": f"0x{base:08X}"},
                         "target_vram": base, "size": 8, "_object": object(), "_elf_spec": (None, {}, {}, None),
                         "strict": {"sections": [{"section": ".rodata", "base": f"0x{base:08X}", "size": 8,
                                                  "compared": 8, "status": "equal"}]}})
        with mock.patch.object(transplant.cmp, "find_section",
                               lambda elf, name: type("S", (), {"align": 8, "name": name})()), \
                mock.patch("n64dk.match.verify.strict_compare",
                           lambda *a, **k: Verdict("mismatch", False, sections=[])):
            report = transplant.settle_rodata(rows, self._Target(bytes(rom)), set())
        self.assertEqual((report["chains"], report["rows_rechecked"], report["promoted"]), (1, 2, 0))
        self.assertTrue(all(r["verdict"] == transplant.ASSUMED_VERDICT for r in rows))


class RodataThresholdTest(unittest.TestCase):
    def test_adjacent_pieces_with_few_nonzero_bytes_confirm_nothing(self):
        from unittest import mock
        from n64dk.match import transplant
        rom = bytearray(0x40)
        rom[0:4] = b"\x45\x1e\xc0\x00"
        rom[8:16] = b"\x3f\x80\x00\x00\x40\x00\x00\x00"          # 6 non-zero bytes over 16
        rows = [{"verdict": transplant.ASSUMED_VERDICT, "awaiting": {".rodata": f"0x{base:08X}"}, "target_vram": base,
                 "size": 8, "_object": object(), "_elf_spec": (None, {}, {}, None),
                 "strict": {"sections": [{"section": ".rodata", "base": f"0x{base:08X}", "size": size,
                                          "compared": size, "status": "equal"}]}}
                for base, size in ((0x80000000, 4), (0x80000008, 8))]
        target = RodataSettlementTest._Target(bytes(rom))
        with mock.patch.object(transplant.cmp, "find_section",
                               lambda elf, name: type("S", (), {"align": 8, "name": name})()):
            report = transplant.settle_rodata(rows, target, set())
        self.assertEqual((report["chains"], report["promoted"]), (0, 0))


class PartialRodataTest(unittest.TestCase):
    """A split unit's read-only section padded with zeros where the original unit had more data is
    settled only up to where the next unit's own code puts its data."""

    def _section(self, data, align=16):
        return type("S", (), {"align": align, "name": ".rodata", "data": data})()

    def _rows(self, first_state, second_state):
        from n64dk.match import transplant
        def row(base, state):
            return {"verdict": transplant.ASSUMED_VERDICT, "awaiting": {".rodata": f"0x{base:08X}"},
                    "target_vram": base, "size": 8, "_object": object(), "_elf_spec": (None, {}, {}, None),
                    "strict": {"sections": [dict(state, section=".rodata", base=f"0x{base:08X}")]}}
        return [row(0x80000000, first_state), row(0x80000008, second_state)]

    def test_padded_piece_owns_up_to_the_next_unit(self):
        from unittest import mock
        from n64dk.match import transplant
        from n64dk.match.verify import Verdict
        rom = bytearray(0x40)
        rom[0:16] = b"\x3f\x80\x00\x01\x40\x20\x10\x08" + b"\x41\x00\x00\x00\x42\x00\x00\x00"
        first = bytes(rom[0:8]) + bytes(8)                   # 8 bytes of data, 8 of padding
        sections = {"a": self._section(first), "b": self._section(bytes(rom[8:16]), align=8)}
        rows = self._rows({"size": 16, "compared": 16, "status": "unverified", "first_diff": "0x8"},
                          {"size": 8, "compared": 8, "status": "equal"})
        rows[0]["_object"], rows[1]["_object"] = "a", "b"
        calls = []

        def strict(elf, *args):
            calls.append((elf, args[5], args[6]))
            return Verdict("exact", True, sections=[])
        with mock.patch.object(transplant.cmp, "find_section", lambda elf, name: sections[elf]), \
                mock.patch("n64dk.match.verify.strict_compare", strict):
            report = transplant.settle_rodata(rows, RodataSettlementTest._Target(bytes(rom)), set())
        self.assertEqual((report["partial_pieces"], report["chains"], report["promoted"], report["promoted_partial"]),
                         (1, 1, 2, 1))
        self.assertEqual(calls[0], ("a", {".rodata": 0x80000000}, {".rodata": 8}))
        self.assertIn("owns 8 bytes", rows[0]["corroborated"][".rodata"])

    def test_padding_at_the_end_of_a_chain_is_not_forgiven(self):
        from unittest import mock
        from n64dk.match import transplant
        rom = bytearray(0x40)
        rom[0:16] = b"\x3f\x80\x00\x01\x40\x20\x10\x08" + b"\x41\x00\x00\x00\x42\x00\x00\x00"
        sections = {"a": self._section(bytes(rom[0:8]), align=8),
                    "b": self._section(bytes(rom[8:12]) + bytes(12))}
        rows = self._rows({"size": 8, "compared": 8, "status": "equal"},
                          {"size": 16, "compared": 16, "status": "unverified", "first_diff": "0x4"})
        rows[0]["_object"], rows[1]["_object"] = "a", "b"
        with mock.patch.object(transplant.cmp, "find_section", lambda elf, name: sections[elf]), \
                mock.patch("n64dk.match.verify.strict_compare", lambda *a: self.fail("nothing to verify")):
            report = transplant.settle_rodata(rows, RodataSettlementTest._Target(bytes(rom)), set())
        self.assertEqual(report["promoted"], 0)

    def test_nonzero_tail_is_not_padding(self):
        from n64dk.match import transplant
        state = {"size": 16, "compared": 16, "status": "unverified", "first_diff": "0x8"}
        self.assertIsNone(transplant._partial_piece(state, self._section(bytes(8) + b"\x01" + bytes(7))))
        self.assertIsNotNone(transplant._partial_piece(state, self._section(b"\x01" + bytes(15))))


class PinOriginTest(unittest.TestCase):
    """A unit may not be proven with an address that only its own retail instructions support."""
    delta = 0x80200000 - 0x1000

    def test_transfer_records_the_supporting_functions(self):
        unit = make_unit([("donor_f", donor_fn(0x8012, 0x3456, 0x80101000))], 0x80100000)
        target = donor_fn(0x8022, 0x3466, 0x80205000)
        image = image_with(self.delta, {0x80200100: target})
        index = tp.Index({"0": unit}, {"D": {"root": "/x", "image": "/x", "delta": 0, "include": []}})
        hits = tp.scan(image, self.delta, [("t_f", 0x80200100, len(target))], index)
        result = tp.transfer(hits)
        self.assertEqual({tuple(v["functions"]) for v in result["symbols"].values()}, {("t_f",)})

    def test_a_pin_supported_only_by_the_unit_itself_is_never_declared(self):
        table = {("D", 0x80123456): {"target_address": 0x80223466, "hits": 1, "functions": ["t_f"]}}
        self.assertEqual(tp.classify_pin("D", 0x80123456, 0x80223466, table, {}, {"t_f"}), ("self", ""))

    def test_another_function_makes_the_pin_independent(self):
        table = {("D", 0x80123456): {"target_address": 0x80223466, "hits": 2, "functions": ["t_f", "t_g"]}}
        decision, origin = tp.classify_pin("D", 0x80123456, 0x80223466, table, {}, {"t_f"})
        self.assertEqual(decision, "declare")
        self.assertIn("1 other function(s), e.g. t_g", origin)

    def test_a_self_supported_call_is_settled_by_the_callee_found_at_its_address(self):
        # the unit is the only voter for the callee's address (nobody else calls it), and the
        # donor function of that name stands, identical, at exactly that address and nowhere else
        table = {("D", 0x80101000): {"target_address": 0x80205000, "hits": 1, "functions": ["t_f"],
                                     "supporters": [("t_f", 0x80200100, 0x40)]}}
        unit = (0x80200100, 0x80200140)
        unique = {("D", 0x80101000): {0x80205000}}
        self.assertEqual(tp.classify_pin("D", 0x80101000, 0x80205000, table, unique, {"t_f"}, span=unit),
                         ("declare", "definition"))
        self.assertEqual(tp.classify_pin("D", 0x80101000, 0x80205000, table, unique, {"t_f"}),
                         ("declare", "definition"))
        # no such function there: still only the unit's own word for it
        self.assertEqual(tp.classify_pin("D", 0x80101000, 0x80205000, table, {}, {"t_f"}, span=unit), ("self", ""))
        # the function stands at several places: the unit's instruction would pick one
        several = {("D", 0x80101000): {0x80205000, 0x80206000}}
        self.assertEqual(tp.classify_pin("D", 0x80101000, 0x80205000, table, several, {"t_f"}, span=unit)[0],
                         "unchecked")
        # the definition found is the unit's own code (a call inside the unit)
        own = {("D", 0x80101000): {0x80200120}}
        table_own = {("D", 0x80101000): {"target_address": 0x80200120, "hits": 1, "functions": ["t_f"],
                                         "supporters": [("t_f", 0x80200100, 0x40)]}}
        self.assertEqual(tp.classify_pin("D", 0x80101000, 0x80200120, table_own, own, {"t_f"}, span=unit),
                         ("self", ""))

    def test_definition_fixed_conflict_and_unchecked(self):
        self.assertEqual(tp.classify_pin("D", 0x80101000, 0x80205000, {}, {("D", 0x80101000): {0x80205000}}, set()),
                         ("declare", "definition"))
        self.assertEqual(tp.classify_pin("D", 0xA4600000, 0xA4600000, {}, {}, set()), ("declare", "fixed-address"))
        table = {("D", 0x80123456): {"target_address": 0x80223000, "hits": 3, "functions": ["a", "b", "c"]}}
        self.assertEqual(tp.classify_pin("D", 0x80123456, 0x80223466, table, {}, set())[0], "conflict")
        self.assertEqual(tp.classify_pin("D", 0x80123456, 0x80223466, {}, {}, set()), ("unchecked", ""))
        self.assertEqual(tp.classify_pin("D", None, 0x80223466, table, {}, set()), ("unchecked", ""))
