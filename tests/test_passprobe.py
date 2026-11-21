"""Pass signatures and causal attribution on hand-built instruction words (no toolchain)."""
import unittest

from n64dk.match import passprobe as pp


def r(fn, rs=0, rt=0, rd=0, sa=0):
    return (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | fn


def i(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


NOP = 0
JR_RA = r(8, 31)


def ADDU(rd, rs, rt):
    return r(0x21, rs, rt, rd)


def SLL(rd, rt, sa):
    return r(0, 0, rt, rd, sa)


def MULT(rs, rt):
    return r(0x18, rs, rt)


def MFLO(rd):
    return r(0x12, 0, 0, rd)


def LW(rt, offset, base):
    return i(35, base, rt, offset)


def SW(rt, offset, base):
    return i(43, base, rt, offset)


def ADDIU(rt, rs, imm):
    return i(9, rs, rt, imm)


def LUI(rt, imm):
    return i(15, 0, rt, imm)


def BEQ(rs, rt, offset):
    return i(4, rs, rt, offset)


def BNE(rs, rt, offset):
    return i(5, rs, rt, offset)


def BEQL(rs, rt, offset):
    return i(20, rs, rt, offset)


def JAL(target):
    return (3 << 26) | target


SCHED_T = [LW(2, 0, 4), LW(3, 4, 4), ADDU(2, 2, 3), JR_RA, NOP]
SCHED_C = [LW(3, 4, 4), LW(2, 0, 4), ADDU(2, 2, 3), JR_RA, NOP]
PERM_T = [ADDU(8, 4, 5), ADDU(9, 4, 6), SW(8, 0, 7), JR_RA, SW(9, 4, 7)]
PERM_C = [ADDU(9, 4, 5), ADDU(8, 4, 6), SW(9, 0, 7), JR_RA, SW(8, 4, 7)]
SPILL_T = [ADDIU(29, 29, -32), SW(31, 28, 29), JAL(0x40), NOP, LW(31, 28, 29), JR_RA, ADDIU(29, 29, 32)]
SPILL_C = [ADDIU(29, 29, -32), SW(31, 28, 29), SW(4, 24, 29), JAL(0x40), NOP, LW(4, 24, 29), LW(31, 28, 29), JR_RA,
           ADDIU(29, 29, 32)]
SLOT_T = [ADDIU(3, 0, 5), BEQ(4, 0, 2), NOP, ADDU(2, 3, 4), JR_RA, ADDU(2, 3, 0)]
SLOT_C = [BEQ(4, 0, 2), ADDIU(3, 0, 5), ADDU(2, 3, 4), JR_RA, ADDU(2, 3, 0)]
LIKELY_T = [BEQL(4, 5, 2), ADDIU(2, 0, 1), ADDIU(2, 0, 0), JR_RA, NOP]
LIKELY_C = [BEQ(4, 5, 2), NOP, ADDIU(2, 0, 0), JR_RA, NOP]
FP_T = [ADDIU(29, 29, -24), SW(31, 20, 29), JAL(0x40), NOP, LW(31, 20, 29), JR_RA, ADDIU(29, 29, 24)]
FP_C = [ADDIU(29, 29, -24), SW(31, 20, 29), SW(30, 16, 29), ADDU(30, 29, 0), JAL(0x40), NOP, ADDU(29, 30, 0),
        LW(31, 20, 29), LW(30, 16, 29), JR_RA, ADDIU(29, 29, 24)]
AT_T = [LUI(1, 0x8001), ADDU(1, 1, 4), LW(2, 0x100, 1), JR_RA, NOP]          # indexed load through $at
AT_C = [LUI(2, 0x8001), ADDU(2, 2, 4), LW(2, 0x100, 2), JR_RA, NOP]          # the same through $v0
HAZARD_T = [LW(2, 0, 4), NOP, ADDU(2, 2, 5), JR_RA, NOP]
HAZARD_C = [LW(2, 0, 4), ADDU(2, 2, 5), JR_RA, NOP]
MUL_T = [ADDIU(8, 0, 12), MULT(4, 8), MFLO(2), JR_RA, NOP]                   # a * 12 with mult
MUL_C = [SLL(2, 4, 1), ADDU(2, 2, 4), SLL(2, 2, 2), JR_RA, NOP]               # ((a << 1) + a) << 2
LOOP_T = [LW(8, 0, 4), ADDIU(5, 5, -1), ADDU(2, 2, 8), BNE(5, 0, -4), ADDIU(4, 4, 4), JR_RA, NOP]   # pointer walk
LOOP_C = [SLL(9, 6, 2), ADDU(9, 9, 4), LW(8, 0, 9), ADDIU(6, 6, 1), ADDU(2, 2, 8), BNE(6, 5, -6), NOP, JR_RA,
          NOP]                                                                                    # scaled index
CSE_T = [LW(2, 8, 4), LW(3, 8, 4), ADDU(2, 2, 3), JR_RA, NOP]                 # p[2] loaded twice
CSE_C = [LW(2, 8, 4), ADDU(2, 2, 2), JR_RA, NOP]
LEVEL_T = [ADDU(2, 4, 5), JR_RA, NOP]
LEVEL_C = [ADDIU(29, 29, -8), SW(4, 0, 29), SW(5, 4, 29), LW(2, 0, 29), LW(3, 4, 29), ADDU(2, 2, 3),
           ADDIU(29, 29, 8), JR_RA, NOP]
SIGN_T = [0x84820004, 0x03E00008, 0x00000000]                                 # lh (as tests/test_causes.py)
SIGN_C = [0x94820004, 0x03E00008, 0x00000000]                                 # lhu


def ranked(target, candidate, **kw):
    kw.setdefault("wrap", False)
    return pp.attribute(target, candidate, **kw)


def top(target, candidate, **kw):
    return next(row for row in ranked(target, candidate, **kw) if not row["ruled_out"])


def by_component(rows):
    return {row["component"]: row for row in rows}


class DecodeTest(unittest.TestCase):
    def test_names_and_fields(self):
        self.assertEqual(pp.decode(0).name, "nop")
        add = pp.decode(ADDU(2, 4, 5))
        self.assertEqual((add.name, add.operands), ("addu", (("r", 4), ("r", 5), ("r", 2))))
        self.assertEqual(add.writes, frozenset({("r", 2)}))
        load = pp.decode(LW(2, -4, 29))
        self.assertEqual((load.name, load.base, load.imm), ("lw", 29, -4))
        self.assertTrue(pp.decode(BEQL(4, 5, 2)).likely)
        self.assertTrue(pp.decode(JR_RA).control)
        self.assertEqual(pp.decode(SLL(2, 4, 3)).shamt, 3)

    def test_float_names(self):
        mul_s = (17 << 26) | (16 << 21) | (4 << 16) | (2 << 11) | (0 << 6) | 2
        cvt_d_s = (17 << 26) | (16 << 21) | (2 << 11) | (0 << 6) | 33
        cvt_s_w = (17 << 26) | (20 << 21) | (2 << 11) | (0 << 6) | 32
        bc1fl = (17 << 26) | (8 << 21) | (2 << 16) | 3
        self.assertEqual([pp.decode(w).name for w in (mul_s, cvt_d_s, cvt_s_w, bc1fl)],
                         ["mul.s", "cvt.d.s", "cvt.s.w", "bc1fl"])
        self.assertTrue(pp.decode(bc1fl).likely)

    def test_skeleton_ignores_registers_only(self):
        self.assertEqual(pp.skeleton(ADDU(2, 4, 5)), pp.skeleton(ADDU(3, 6, 7)))
        self.assertEqual(pp.skeleton(LW(2, 8, 29)), pp.skeleton(LW(3, 8, 4)))
        self.assertNotEqual(pp.skeleton(LW(2, 8, 4)), pp.skeleton(LW(2, 12, 4)))
        self.assertNotEqual(pp.skeleton(SLL(2, 4, 1)), pp.skeleton(SLL(2, 4, 2)))


class SignatureTest(unittest.TestCase):
    def test_identical_code_has_no_component(self):
        self.assertEqual(pp.attribute(SCHED_T, list(SCHED_T), wrap=False), [])

    def test_reordering_is_scheduling(self):
        rows = ranked(SCHED_T, SCHED_C, family="gcc")
        self.assertEqual(rows[0]["component"], "scheduling")
        self.assertGreaterEqual(rows[0]["confidence"], 0.9)
        self.assertIn("-fno-schedule-insns", rows[0]["next_experiment"])
        self.assertTrue(by_component(rows)["register-allocation"]["ruled_out"])

    def test_permuted_registers_are_allocation(self):
        rows = ranked(PERM_T, PERM_C)
        self.assertEqual(rows[0]["component"], "register-allocation")
        self.assertIn("consistent exchange", rows[0]["evidence"][0])
        self.assertIn("$t1->$t0", rows[0]["evidence"][0])
        self.assertTrue(by_component(rows)["scheduling"]["ruled_out"])
        self.assertTrue(by_component(rows)["delay-slot"]["ruled_out"])

    def test_extra_stack_traffic_is_spill(self):
        row = top(SPILL_T, SPILL_C, family="gcc")
        self.assertEqual(row["component"], "spill-reload")
        self.assertIn("stack", row["dimensions"])
        self.assertNotIn("structural", [r["component"] for r in ranked(SPILL_T, SPILL_C)])

    def test_slot_fill_is_delay_slot(self):
        rows = ranked(SLOT_T, SLOT_C, family="gcc")
        first = next(row for row in rows if not row["ruled_out"])
        self.assertEqual(first["component"], "delay-slot")
        self.assertIn("before the branch", first["evidence"][0])
        self.assertTrue(by_component(rows)["hazard-nops"]["ruled_out"])
        self.assertIn("-fno-delayed-branch", first["next_experiment"])

    def test_likely_branch(self):
        rows = ranked(LIKELY_T, LIKELY_C, family="ido")
        first = next(row for row in rows if not row["ruled_out"])
        self.assertEqual(first["component"], "branch-likely")
        self.assertIn("-mips1", first["next_experiment"])
        self.assertNotIn("structural", by_component(rows))
        self.assertNotIn("delay-slot", by_component(rows))

    def test_frame_pointer(self):
        row = top(FP_T, FP_C, family="gcc")
        self.assertEqual(row["component"], "frame-pointer")
        self.assertIn("-fomit-frame-pointer", row["next_experiment"])

    def test_at_forms_are_the_assembler(self):
        rows = ranked(AT_T, AT_C, family="gcc")
        first = next(row for row in rows if not row["ruled_out"])
        self.assertEqual(first["component"], "assembler")
        self.assertIn("$at", first["dimensions"])
        allocation = by_component(rows)["register-allocation"]
        self.assertTrue(any("$at" in text for text in allocation["contra"]))

    def test_only_nops_is_hazard(self):
        row = top(HAZARD_T, HAZARD_C)
        self.assertEqual(row["component"], "hazard-nops")
        self.assertTrue(any("follow loads" in text for text in row["evidence"]))

    def test_multiplication_by_shifts(self):
        row = top(MUL_T, MUL_C)
        self.assertEqual(row["component"], "constant-multiplication")
        self.assertIn("retail multiplies", row["evidence"][0])

    def test_index_against_pointer_walk(self):
        row = top(LOOP_T, LOOP_C, family="gcc")
        self.assertEqual(row["component"], "strength-reduction")
        self.assertIn("-fno-strength-reduce", row["next_experiment"])
        self.assertNotIn("structural", by_component(ranked(LOOP_T, LOOP_C)))

    def test_repeated_load_is_cse(self):
        row = top(CSE_T, CSE_C)
        self.assertEqual(row["component"], "cse-reuse")
        self.assertIn("2 load(s) in retail", row["evidence"][0])

    def test_stack_resident_code_is_level(self):
        row = top(LEVEL_T, LEVEL_C, family="gcc")
        self.assertEqual(row["component"], "level")
        self.assertIn("unoptimised", row["evidence"][0])

    def test_ruled_out_rows_come_last_with_a_reason(self):
        rows = ranked(PERM_T, PERM_C)
        seen_ruled_out = False
        for row in rows:
            if row["ruled_out"]:
                seen_ruled_out = True
                self.assertEqual(row["confidence"], 0.0)
                self.assertTrue(row["contra"])
            else:
                self.assertFalse(seen_ruled_out)
        for row in rows:
            self.assertEqual(set(row), {"component", "pass", "confidence", "evidence", "contra", "dimensions",
                                        "ruled_out", "next_experiment"})


class WrappedTest(unittest.TestCase):
    def test_existing_signedness_diagnosis_is_reused(self):
        rows = pp.attribute(SIGN_T, SIGN_C)
        self.assertEqual(rows[0]["component"], "source-types")
        self.assertFalse(rows[0]["pass"])
        self.assertTrue(any("causes.py type.signedness" in text for text in rows[0]["evidence"]))
        structural = by_component(rows).get("structural")
        if structural is not None:
            self.assertTrue(any("source-types" in text for text in structural["contra"]))

    def test_wrapped_reports_every_diagnosis(self):
        found = pp.wrapped(SIGN_T, SIGN_C)
        self.assertIn("source-types", [component for component, _confidence, _text in found])


class LearnedTest(unittest.TestCase):
    RESULTS = {"builds": [
        {"lane": "kmc+as", "family": "gcc", "role": "base", "ok": True,
         "functions": {"f": [f"{w:08X}" for w in SCHED_T], "g": [f"{w:08X}" for w in PERM_T]}},
        {"lane": "kmc+as", "family": "gcc", "role": "sched1", "component": "scheduling",
         "change": "add -fno-schedule-insns", "ok": True,
         "functions": {"f": [f"{w:08X}" for w in SCHED_C], "g": [f"{w:08X}" for w in PERM_T]}},
        {"lane": "kmc+as", "family": "gcc", "role": "no-branch-likely", "component": "branch-likely",
         "change": "add -mno-branch-likely", "ok": False, "error": "cc1: Invalid option", "functions": {}},
    ]}

    def test_learn_measures_each_pair(self):
        learned = pp.learn(self.RESULTS)
        entry = learned["lanes"]["kmc+as"]["scheduling"]
        self.assertTrue(entry["supported"])
        self.assertEqual((entry["functions"], entry["effect"], entry["hits"]), (2, 1, 1))
        self.assertIn("order", entry["dimensions"])
        self.assertEqual(entry["top"], {"scheduling": 1})
        self.assertFalse(learned["lanes"]["kmc+as"]["branch-likely"]["supported"])
        self.assertEqual(learned["families"]["gcc"]["scheduling"]["effect"], 1)
        self.assertEqual(learned["checks"][0]["expected"], "scheduling")
        self.assertIn("| kmc+as | scheduling |", pp.summary_markdown(learned))

    def test_learned_profile_weighs_attribution(self):
        learned = pp.learn(self.RESULTS)
        plain = by_component(ranked(SCHED_T, SCHED_C, family="gcc"))["scheduling"]
        weighed = by_component(ranked(SCHED_T, SCHED_C, lane="kmc+as", learned=learned))["scheduling"]
        self.assertTrue(any("pass probes" in text for text in weighed["evidence"]))
        self.assertGreaterEqual(weighed["confidence"], plain["confidence"])
        likely = by_component(ranked(LIKELY_T, LIKELY_C, lane="kmc+as", learned=learned))["branch-likely"]
        self.assertTrue(any("not accepted" in text for text in likely["contra"]))

    def test_a_pass_without_effect_is_weakened(self):
        learned = {"families": {"gcc": {"scheduling": {"supported": True, "functions": 10, "effect": 0,
                                                       "flags": ["add -fno-schedule-insns"], "pairs": ["sched1"],
                                                       "dimensions": {}}}}}
        plain = by_component(ranked(SCHED_T, SCHED_C, family="gcc"))["scheduling"]
        weak = by_component(ranked(SCHED_T, SCHED_C, family="gcc", learned=learned))["scheduling"]
        self.assertLess(weak["confidence"], plain["confidence"])
        self.assertTrue(any("changed none" in text for text in weak["contra"]))


class PairsAndTruthTest(unittest.TestCase):
    def test_apply(self):
        pairs = {(p.family, p.name): p for p in pp.PASS_PAIRS}
        base = ["-O2", "-G0", "-mips3", "-mgp32", "-mfp32"]
        self.assertEqual(pairs["gcc", "level-O1"].apply(base), ["-O1", "-G0", "-mips3", "-mgp32", "-mfp32"])
        self.assertEqual(pairs["gcc", "sched1"].apply(base), base + ["-fno-schedule-insns"])
        self.assertIsNone(pairs["gcc", "sched1"].apply(base + ["-fno-schedule-insns"]))
        self.assertEqual(pairs["gcc", "isa-mips1"].apply(base)[2], "-mips1")
        ido = ["-O2", "-mips2", "-G", "0", "-non_shared", "-Wab,-r4300_mul"]
        self.assertNotIn("-Wab,-r4300_mul", pairs["ido", "r4300-mul"].apply(ido))
        self.assertIn("-Wab,-r4300_mul", pairs["ido", "r4300-mul"].apply(["-O2", "-mips2"]))
        self.assertIsNone(pairs["ido", "level-g"].apply(["-g", "-mips2"]))
        self.assertEqual({p.family for p in pp.pairs_for("ido")}, {"ido"})
        self.assertTrue(all(p.component in pp.COMPONENTS for p in pp.PASS_PAIRS))

    def test_truth_of_recipe_differences(self):
        def labels(a, b):
            return pp.truth(pp.recipe_delta(a, b))
        self.assertEqual(labels("kmc+as -O2 -G0 -mips3", "kmc+as -O1 -G0 -mips3")["labels"], ["level"])
        isa = labels("ido5.3 -O2 -mips2 -G 0", "ido5.3 -O2 -mips1 -G 0")
        self.assertEqual(isa["labels"], ["isa"])
        self.assertEqual(isa["accepts"], ["branch-likely", "hazard-nops"])
        self.assertEqual(labels("kmc+as -O2", "kmc+atload -O2")["accepts"], ["assembler"])
        self.assertEqual(labels("kmc+as -O2", "kmc+as -O2 -fno-schedule-insns")["labels"], ["scheduling"])
        self.assertEqual(labels("ido5.3 -O2 -G 0", "ido5.3 -O2 -G 8")["labels"], ["small-data"])
        mixed = labels("ido5.3 -O2 -mips2", "ido7.1 -O1 -mips2")
        self.assertFalse(mixed["single"])
        self.assertEqual(mixed["labels"], ["compiler", "level"])

    def test_lane_family(self):
        self.assertEqual(pp.lane_family("kmc+asn64 -O2"), "gcc")
        self.assertEqual(pp.lane_family("custom:mipspro744"), "ido")
        self.assertEqual(pp.lane_family("custom:egcs112 -O2"), "gcc")
        self.assertIsNone(pp.lane_family(None))


if __name__ == "__main__":
    unittest.main()
