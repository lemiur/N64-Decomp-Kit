import json
import unittest
from unittest import mock

from n64dk.match import tv

BASE = 0x80001000
TARGET = 0x80001234
ZERO, V0, V1, A0, A1, A2 = 0, 2, 3, 4, 5, 6
T0, T1, T2, T3, T6, T7, S0, S1, SP, RA = 8, 9, 10, 11, 14, 15, 16, 17, 29, 31


def R(fn, rs=0, rt=0, rd=0, sa=0):
    return (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | fn


def I(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def J(op, target):
    return (op << 26) | ((target >> 2) & 0x3FFFFFF)


NOP, JR_RA = 0, R(8, rs=RA)


def leaf(insn):
    """``jr ra`` with ``insn`` in its delay slot."""
    return [JR_RA, insn]


def ret12(*prefix):
    """``prefix``, then return 1 on the fall-through path and 2 on the taken path (target +3)."""
    return [*prefix, JR_RA, I(9, ZERO, V0, 1), JR_RA, I(9, ZERO, V0, 2)]


ADD1 = leaf(I(9, A0, V0, 1))                                   # v0 = a0 + 1
ADD2 = leaf(I(9, A0, V0, 2))
SLL3 = leaf(R(0, rt=A0, rd=V0, sa=3))                          # v0 = a0 << 3
MUL8 = [I(9, ZERO, T0, 8), R(24, rs=A0, rt=T0), R(18, rd=V0), JR_RA, NOP]   # mult a0, 8; mflo
ADD_CHAIN8 = [R(33, A0, A0, V0), R(33, V0, V0, V0), JR_RA, R(33, V0, V0, V0)]
SUBU = leaf(R(35, A0, A1, V0))                                 # v0 = a0 - a1
NEG_ADD = [R(35, ZERO, A1, T0), JR_RA, R(33, A0, T0, V0)]      # v0 = a0 + (0 - a1)
RENAME_A = [R(33, A0, A1, T0), JR_RA, R(37, T0, ZERO, V0)]
RENAME_B = [R(33, A1, A0, T3), JR_RA, R(37, T3, ZERO, V0)]
STORES_AB = [I(43, A0, A1, 0), I(43, A0, A2, 4), JR_RA, NOP]
STORES_BA = [I(43, A0, A2, 4), I(43, A0, A1, 0), JR_RA, NOP]
SLT_BNE = ret12(R(42, A0, ZERO, T0), I(5, T0, ZERO, 3), NOP)  # slt t0,a0,zero; bnez t0
BLTZ = ret12(I(1, A0, 0, 3), NOP)
SLTI_BEQZ = ret12(I(10, A0, T0, 1), I(4, T0, ZERO, 3), NOP)  # slti t0,a0,1; beqz t0
BGTZ = ret12(I(7, A0, 0, 3), NOP)
FILLED = [I(4, A0, ZERO, 2), I(9, ZERO, V0, 7), I(9, ZERO, V0, 1), JR_RA, NOP]
UNFILLED = [I(9, ZERO, V0, 7), I(4, A0, ZERO, 2), NOP, I(9, ZERO, V0, 1), JR_RA, NOP]
SLT, SLTU = leaf(R(42, A0, A1, V0)), leaf(R(43, A0, A1, V0))
SRL4, SRA4 = leaf(R(2, rt=A0, rd=V0, sa=4)), leaf(R(3, rt=A0, rd=V0, sa=4))
SRA31, SRA30 = leaf(R(3, rt=A0, rd=V0, sa=31)), leaf(R(3, rt=A0, rd=V0, sa=30))
SLLV = leaf(R(4, rs=A1, rt=A0, rd=V0))                         # v0 = a0 << (a1 & 31)


def masked_sllv(mask):
    return [I(12, A1, T0, mask), JR_RA, R(4, rs=T0, rt=A0, rd=V0)]


ADDU, ADD = leaf(R(33, A0, A1, V0)), leaf(R(32, A0, A1, V0))
STORE_THEN_LOAD = [I(43, A0, A2, 0), I(35, A1, V0, 0), JR_RA, NOP]   # *a0 = a2; return *a1
LOAD_THEN_STORE = [I(35, A1, V0, 0), I(43, A0, A2, 0), JR_RA, NOP]
MMIO_LOAD = [I(15, 0, T0, 0xA460), I(35, T0, V0, 0x10), JR_RA, NOP]  # lw v0, PI_STATUS
MMIO_STORE = [I(15, 0, T0, 0xA460), I(43, T0, A0, 0x10), JR_RA, NOP]
DIV = [R(26, A0, A1), R(18, rd=V0), JR_RA, NOP]                # div a0,a1; mflo v0
GUARDED_DIV = [I(4, A1, ZERO, 4), NOP, R(26, A0, A1), JR_RA, R(18, rd=V0),   # a1 ? a0/a1 : 0
               JR_RA, R(37, rd=V0)]
SLOT_AFTER = ret12(I(4, A0, ZERO, 3), I(9, A0, A0, 1))        # beq reads a0 before the slot's a0+1
SLOT_BEFORE = ret12(I(9, A0, A0, 1), I(4, A0, ZERO, 3), NOP)
BREAK, RET = [R(13), JR_RA, NOP], [JR_RA, NOP]
ESCAPE = [I(9, SP, SP, -24), I(43, SP, RA, 20), J(3, TARGET), I(9, SP, A0, 16),   # f(&local)
          I(35, SP, RA, 20), JR_RA, I(9, SP, SP, 24)]


def summing(acc=V0, init=0):
    """v0 = a0 + (a0-1) + ... + 1 (0 when a0 <= 0), accumulated in ``acc`` from ``init``."""
    return [I(9, ZERO, acc, init) if init else R(37, rd=acc),
            I(6, A0, 0, 5), NOP,                               # blez a0, end
            R(33, acc, A0, acc), I(9, A0, A0, -1),             # loop: acc += a0; a0 -= 1
            I(7, A0, 0, -3), NOP,                              # bgtz a0, loop
            JR_RA, NOP if acc == V0 else R(37, acc, ZERO, V0)]


def original(const=3, slt=10, branch=4):                       # as tests/test_ir.py
    return [I(9, SP, SP, -24), I(43, SP, RA, 20), I(43, SP, S0, 16), R(37, A0, ZERO, S0),
            I(35, A0, T6, 4), I(9, T6, T7, const), I(43, S0, ZERO, 12), I(43, S0, T7, 8),
            J(3, TARGET), I(9, T6, A0, 1), I(slt, V0, T0, 10), I(branch, T0, ZERO, 2), NOP,
            I(43, S0, V0, 0), I(35, SP, RA, 20), I(35, SP, S0, 16), JR_RA, I(9, SP, SP, 24)]


def renamed():
    """``original`` with other registers, another frame, swapped stores and a hoisted load."""
    return [I(9, SP, SP, -32), I(43, SP, S1, 24), I(43, SP, RA, 28), I(35, A0, T1, 4),
            R(37, A0, ZERO, S1), I(9, T1, T2, 3), I(43, S1, T2, 8), I(43, S1, ZERO, 12),
            J(3, TARGET), I(9, T1, A0, 1), I(10, V0, T3, 10), I(4, T3, ZERO, 2), NOP,
            I(43, S1, V0, 0), I(35, SP, RA, 28), I(35, SP, S1, 24), JR_RA, I(9, SP, SP, 32)]


def check(a, b, limits=None, vram_b=BASE):
    return tv.validate(a, BASE, b, vram_b, limits)


def s32(text):
    v = int(text, 16)
    return v - (1 << 32) if v & 0x80000000 else v


class EquivalentTest(unittest.TestCase):
    def assertEquivalent(self, a, b, limits=None):
        r = check(a, b, limits)
        self.assertEqual(r.outcome, "equivalent", (r.reason, r.counterexample, r.paths))
        self.assertIsNone(r.counterexample)
        self.assertTrue(any("evidence only" in x for x in r.assumptions))
        self.assertFalse(r.as_dict()["proof"])
        return r

    def test_register_renaming(self):
        self.assertEquivalent(RENAME_A, RENAME_B)

    def test_reordered_independent_stores(self):
        r = self.assertEquivalent(STORES_AB, STORES_BA)
        self.assertGreater(r.paths["domain"], 0)                 # misaligned a0: outside A's domain

    def test_multiply_by_8_is_shift_by_3(self):
        self.assertEquivalent(SLL3, MUL8)
        self.assertEquivalent(SLL3, ADD_CHAIN8)

    def test_subtraction_is_addition_of_the_negation(self):
        self.assertEquivalent(SUBU, NEG_ADD)

    def test_set_and_branch_forms(self):
        self.assertEquivalent(SLT_BNE, BLTZ)
        self.assertEquivalent(SLTI_BEQZ, BGTZ)

    def test_independent_instruction_moved_into_the_delay_slot(self):
        self.assertEquivalent(FILLED, UNFILLED)

    def test_shift_amount_is_masked_as_mips_does(self):
        self.assertEquivalent(SLLV, masked_sllv(31))
        self.assertEquivalent(SLLV, masked_sllv(63))

    def test_reallocated_function_with_a_call(self):
        r = self.assertEquivalent(original(), renamed())
        self.assertGreater(r.paths["equal"], 0)

    def test_relocated_copy(self):
        r = check(original(), original(), vram_b=0x80D00000)
        self.assertEqual(r.outcome, "equivalent", r.reason)

    def test_a_loop_cut_by_the_unroll_bound_is_never_equivalent(self):
        # round 12 soundness alarm: agreement up to the bound hid oracle-confirmed differences
        r = check(summing(), summing(acc=T0))
        self.assertEqual(r.outcome, "timeout", r.reason)
        self.assertIn("loop bound", r.reason)
        self.assertTrue(any("unrolled to 4" in x for x in r.assumptions), r.assumptions)
        self.assertGreater(r.paths["a_truncated"], 0)


class CounterexampleTest(unittest.TestCase):
    def assertWitness(self, a, b, component="v0"):
        r = check(a, b)
        self.assertEqual(r.outcome, "counterexample", (r.reason, r.paths))
        ce = r.counterexample
        self.assertIn(component, ce["differs"])
        self.assertNotEqual(ce["a"][component], ce["b"][component])
        return ce

    def test_constant_off_by_one(self):
        ce = self.assertWitness(ADD1, ADD2)
        self.assertEqual(int(ce["b"]["v0"], 16) - int(ce["a"]["v0"], 16), 1)

    def test_slt_against_sltu_needs_a_negative_input(self):
        ce = self.assertWitness(SLT, SLTU)
        x, y = s32(ce["inputs"]["arg0"]), s32(ce["inputs"]["arg1"])
        self.assertNotEqual(x < y, (x & 0xFFFFFFFF) < (y & 0xFFFFFFFF))
        self.assertTrue(x < 0 or y < 0)

    def test_logical_against_arithmetic_shift(self):
        ce = self.assertWitness(SRL4, SRA4)
        self.assertLess(s32(ce["inputs"]["arg0"]), 0)

    def test_shift_boundaries(self):
        self.assertWitness(SRA31, SRA30)
        ce = self.assertWitness(SLLV, masked_sllv(15))
        self.assertTrue(int(ce["inputs"]["arg1"], 16) & 16)

    def test_add_traps_on_signed_overflow_where_addu_wraps(self):
        ce = self.assertWitness(ADDU, ADD, "fault")
        self.assertEqual(ce["b"]["fault"], ["integer-overflow"])
        x, y = s32(ce["inputs"]["arg0"]), s32(ce["inputs"]["arg1"])
        self.assertFalse(-2 ** 31 <= x + y < 2 ** 31)
        self.assertWitness(ADD, ADDU, "fault")                  # an overflow trap is behaviour, not domain

    def test_branch_reading_the_register_its_delay_slot_writes(self):
        ce = self.assertWitness(SLOT_AFTER, SLOT_BEFORE)
        self.assertIn(ce["inputs"]["arg0"], ("0x0", "0xFFFFFFFF"))

    def test_mutated_function_with_a_call(self):
        self.assertWitness(original(), original(const=4), "call0.memory")
        self.assertWitness(original(), original(slt=11), "memory")   # slti -> sltiu

    def test_a_defined_argument_difference_is_not_hidden_by_clobbered_ones(self):
        # round 12 soundness alarm: a2/a3 clobbered by the first call (undefined in both) hid a1
        def two_calls(k):
            return [I(9, SP, SP, -24), I(43, SP, RA, 20), J(3, TARGET), NOP, I(9, ZERO, A1, k), J(3, TARGET),
                    NOP, I(35, SP, RA, 20), JR_RA, I(9, SP, SP, 24)]
        ce = self.assertWitness(two_calls(4), two_calls(5), "call1.a1")
        self.assertEqual((ce["a"]["call1.a1"], ce["b"]["call1.a1"]), ("0x4", "0x5"))

    def test_loop_with_wrong_initial_value(self):
        self.assertWitness(summing(), summing(init=1))

    def test_break_is_a_fault(self):
        self.assertWitness(RET, BREAK, "fault")


class AliasTest(unittest.TestCase):
    def test_store_then_load_differs_when_the_pointers_are_equal(self):
        r = check(STORE_THEN_LOAD, LOAD_THEN_STORE)
        self.assertEqual(r.outcome, "counterexample", r.reason)
        ce = r.counterexample
        self.assertEqual(ce["inputs"]["arg0"], ce["inputs"]["arg1"])
        self.assertEqual(ce["a"]["v0"], ce["inputs"]["arg2"])
        self.assertGreater(r.paths["case_splits"]["alias"], 0)

    def test_equivalent_only_under_a_no_alias_assumption(self):
        r = check(STORE_THEN_LOAD, LOAD_THEN_STORE, {"noalias": True})
        self.assertEqual(r.outcome, "equivalent", r.reason)
        self.assertTrue(any(x.startswith("no-alias assumed") for x in r.assumptions), r.assumptions)


class UnsupportedTest(unittest.TestCase):
    def test_mmio_is_never_reordered_or_merged(self):
        for words in (MMIO_LOAD, MMIO_STORE):
            r = check(words, words)
            self.assertEqual(r.outcome, "unsupported")
            self.assertIn("MMIO", r.reason)

    def test_escaping_stack_address(self):
        r = check(ESCAPE, ESCAPE)
        self.assertEqual(r.outcome, "unsupported")
        self.assertIn("stack address", r.reason)


class UndefinedTest(unittest.TestCase):
    def test_division_by_zero_is_undefined_dependent(self):
        r = check(DIV, GUARDED_DIV)
        self.assertEqual(r.outcome, "equivalent", (r.reason, r.counterexample))
        self.assertGreater(r.paths["undefined_dependent"], 0)
        self.assertTrue(any("undefined-dependent" in x for x in r.assumptions))
        self.assertGreater(r.paths["case_splits"]["div"], 0)


class SelfCheckTest(unittest.TestCase):
    FIXTURES = {
        "add1": ADD1, "sll3": SLL3, "mul8": MUL8, "chain": ADD_CHAIN8, "subu": SUBU, "neg": NEG_ADD,
        "rename": RENAME_A, "stores": STORES_AB, "slt_bne": SLT_BNE, "bltz": BLTZ, "slti": SLTI_BEQZ,
        "bgtz": BGTZ, "filled": FILLED, "unfilled": UNFILLED, "slt": SLT, "sltu": SLTU, "srl": SRL4,
        "sra": SRA4, "sra31": SRA31, "sllv": SLLV, "sllv15": masked_sllv(15), "addu": ADDU, "add": ADD,
        "store_load": STORE_THEN_LOAD, "load_store": LOAD_THEN_STORE, "div": DIV, "guarded": GUARDED_DIV,
        "slot_after": SLOT_AFTER, "slot_before": SLOT_BEFORE, "break": BREAK, "loop": summing(),
        "original": original(), "renamed": renamed(),
    }

    def test_every_fixture_is_equivalent_to_itself(self):
        for name, words in self.FIXTURES.items():
            with self.subTest(name):
                r = check(words, words)
                if r.paths.get("a_truncated"):              # a loop cut by the unroll bound is never
                    self.assertEqual(r.outcome, "timeout", r.reason)    # proven, not even against itself
                else:
                    self.assertEqual(r.outcome, "equivalent", (r.reason, r.counterexample))

    def test_a_model_that_cannot_separate_paths_is_reported(self):
        with mock.patch.object(tv._Facts, "add", lambda self, atom, pol: True):
            r = check(SLT_BNE, SLT_BNE)
        self.assertEqual(r.outcome, "inconsistent-model")
        self.assertIn("self-check", r.reason)


class BudgetTest(unittest.TestCase):
    def test_path_budget_gives_timeout(self):
        r = check(summing(), summing(), {"max_paths": 3})
        self.assertEqual(r.outcome, "timeout")
        self.assertIn("budget", r.reason)

    def test_unknown_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            check(ADD1, ADD1, {"unrol": 2})

    def test_outcomes_are_the_documented_five(self):
        self.assertEqual(tv.OUTCOMES, ("equivalent", "counterexample", "unsupported", "timeout",
                                       "inconsistent-model"))


class ApiTest(unittest.TestCase):
    def test_counterexample_record_is_json(self):
        rec = json.loads(tv.counterexample_record(check(ADD1, ADD2)))
        self.assertEqual(rec["schema"], "n64dk.tv/1")
        self.assertEqual(rec["outcome"], "counterexample")
        self.assertFalse(rec["proof"])
        self.assertIn("arg0", rec["counterexample"]["inputs"])
        self.assertTrue(rec["counterexample"]["path_a"] is not None)

    def test_validate_rewrite_uses_the_injected_compiler(self):
        sources = {"x + 1": ADD1, "1 + x": leaf(I(9, A0, V0, 1)), "x + 2": ADD2}
        compile_fn = lambda src: (sources[src], BASE)
        self.assertEqual(tv.validate_rewrite("x + 1", "1 + x", compile_fn).outcome, "equivalent")
        self.assertEqual(tv.validate_rewrite("x + 1", "x + 2", compile_fn).outcome, "counterexample")
        r = tv.validate_rewrite("x + 1", "x +", compile_fn)
        self.assertEqual(r.outcome, "unsupported")
        self.assertIn("compile failed", r.reason)


class TermTest(unittest.TestCase):
    x, y = tv.S("x"), tv.S("y")

    def test_linear_normal_form(self):
        x, y, c = self.x, self.y, tv.C
        self.assertEqual(tv._add(32, x, y), tv._add(32, y, x))
        self.assertEqual(tv._shift("shl", 32, x, c(32, 3)), tv._mul(32, x, c(32, 8)))
        self.assertEqual(tv._sub(32, x, y), tv._add(32, x, tv._sub(32, c(32, 0), y)))
        self.assertEqual(tv._add(32, tv._add(32, x, c(32, 4)), c(32, 8)), tv._add(32, x, c(32, 12)))
        self.assertEqual(tv._sub(32, x, x), c(32, 0))

    def test_bitwise_and_comparisons(self):
        x, y, c = self.x, self.y, tv.C
        self.assertEqual(tv._bit("xor", 32, [x, x]), c(32, 0))
        self.assertEqual(tv._bit("and", 32, [tv._add(32, x, c(32, 4)), c(32, 3)]), tv._bit("and", 32, [x, c(32, 3)]))
        self.assertEqual(tv._bit("and", 32, [tv._bit("and", 32, [x, c(32, 63)]), c(32, 31)]),
                         tv._bit("and", 32, [x, c(32, 31)]))
        self.assertEqual(tv._eq(tv._sub(32, x, y), c(32, 0)), tv._eq(tv._sub(32, y, x), c(32, 0)))
        self.assertEqual(tv._eq(tv._add(32, x, c(32, 1)), c(32, 0)), ("eq", 1, x, c(32, 0xFFFFFFFF)))
        self.assertEqual(tv._slt(c(32, 0), x), tv._not(tv._slt(x, c(32, 1))))
        self.assertEqual(tv._ult(x, c(32, 1)), tv._eq(x, c(32, 0)))

    def test_facts_decide_from_intervals(self):
        x, c = self.x, tv.C
        f = tv._Facts()
        self.assertTrue(f.add(tv._eq(x, c(32, 5)), True))
        self.assertFalse(f.decide(tv._slt(x, c(32, 3))))
        self.assertFalse(f.add(tv._eq(x, c(32, 6)), True))
        g = tv._Facts()
        self.assertTrue(g.add(tv._slt(x, c(32, 1)), True))          # x <= 0
        self.assertIsNone(g.decide(tv._eq(x, c(32, 0))))
        self.assertFalse(g.decide(tv._eq(x, c(32, 4))))
        self.assertTrue(g.decide(tv._slt(x, c(32, 5))))
        self.assertFalse(g.copy().add(tv._not(tv._slt(x, c(32, 1))), True))
        self.assertTrue(g.add(tv._eq(x, c(32, 0)), False))          # x < 0 now
        self.assertTrue(g.decide(tv._slt(x, c(32, 0))) is None)     # intervals ignore one hole
        self.assertFalse(g.copy().add(tv._eq(x, c(32, 0)), True))


if __name__ == "__main__":
    unittest.main()
