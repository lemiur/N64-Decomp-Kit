"""Differential behaviour oracle (round 12): agreement is evidence, never proof; fails closed."""
import struct
import unittest

from n64dk import emu
from n64dk.match import oracle

JR_RA, NOP = 0x03E00008, 0


def r_type(rs, rt, rd, fn):
    return (rs << 21) | (rt << 16) | (rd << 11) | fn


def i_type(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def code(*words):
    return struct.pack(f">{len(words)}I", *words)


VRAM = 0x80001000
ADD = code(r_type(4, 5, 2, 0x21), JR_RA, NOP)                    # v0 = a0 + a1
ADD_SWAPPED = code(r_type(5, 4, 2, 0x21), JR_RA, NOP)            # v0 = a1 + a0
SUB = code(r_type(4, 5, 2, 0x23), JR_RA, NOP)                    # v0 = a0 - a1
STORE = code(i_type(43, 4, 5, 4), JR_RA, NOP)                    # *(a0 + 4) = a1
STORE_8 = code(i_type(43, 4, 5, 8), JR_RA, NOP)                  # *(a0 + 8) = a1
JAL = (3 << 26) | ((0x80002000 >> 2) & 0x3FFFFFF)


def call_with(value):                                            # f(value), saving $ra in a frame
    return code(i_type(9, 29, 29, -24), i_type(43, 29, 31, 20), i_type(9, 0, 4, value), JAL, NOP,
                i_type(35, 29, 31, 20), JR_RA, i_type(9, 29, 29, 24))


CALL, CALL_8 = call_with(7), call_with(8)
LOOP = code(0x1000FFFF, NOP)                                     # b . (never returns)
SAVES_S0 = code(i_type(9, 0, 16, 1), JR_RA, NOP)                 # clobbers callee-saved s0

ELSEWHERE = 0x80D00000
J_F = (2 << 26) | ((0x80002000 >> 2) & 0x3FFFFFF)                # j f (the stubbed callee)


def j(address):
    return (2 << 26) | ((address >> 2) & 0x3FFFFFF)


def jal(address):
    return (3 << 26) | ((address >> 2) & 0x3FFFFFF)


# v0 = f() + a0, with a0 kept across the call: in callee-saved s0 (saved and restored), in caller-saved
# a0 itself (wrong: the callee may clobber it), or in the frame (a different frame, same behaviour)
PROLOGUE = (i_type(9, 29, 29, -32), i_type(43, 29, 31, 28), i_type(43, 29, 16, 24))
EPILOGUE = (i_type(35, 29, 31, 28), i_type(35, 29, 16, 24), JR_RA, i_type(9, 29, 29, 32))
KEEP_S0 = code(*PROLOGUE, r_type(4, 0, 16, 0x21), JAL, NOP, r_type(2, 16, 2, 0x21), *EPILOGUE)
KEEP_A0 = code(*PROLOGUE, JAL, NOP, r_type(2, 4, 2, 0x21), *EPILOGUE)
KEEP_FRAME = code(i_type(9, 29, 29, -40), i_type(43, 29, 31, 28), i_type(43, 29, 16, 24), i_type(43, 29, 4, 32),
                  JAL, NOP, i_type(35, 29, 4, 32), r_type(2, 4, 2, 0x21),
                  i_type(35, 29, 31, 28), i_type(35, 29, 16, 24), JR_RA, i_type(9, 29, 29, 40))


def stray(to):                                                   # j to+8; nop; v0 = a0 + a1; return
    return code(j(to + 8), NOP, r_type(4, 5, 2, 0x21), JR_RA, NOP)


def recursive(entry):                                            # f(n) = (n & 3) ? f((n & 3) - 1) + 1 : 0
    return code(i_type(9, 29, 29, -24), i_type(43, 29, 31, 20), i_type(12, 4, 4, 3), i_type(4, 4, 0, 5),
                r_type(0, 0, 2, 0x25), jal(entry), i_type(9, 4, 4, -1), i_type(9, 2, 2, 1), NOP,
                i_type(35, 29, 31, 20), JR_RA, i_type(9, 29, 29, 24))


NOTHING = code(JR_RA, NOP)
ZERO = code(r_type(0, 0, 2, 0x25), JR_RA, NOP)                   # v0 = 0
STORE_FAR = code(i_type(15, 0, 8, 0x8100), i_type(43, 8, 5, 0), JR_RA, NOP)   # *(0x81000000) = a1: above RAM
LOAD_FAR = code(i_type(15, 0, 8, 0x8100), i_type(35, 8, 2, 0), JR_RA, NOP)    # v0 = *(0x81000000)
HIGH_BYTE = code(r_type(0, 5, 8, 0x02) | (24 << 6), i_type(40, 4, 8, 0xFF), JR_RA, NOP)   # a0[0xFF] = a1 >> 24
SPILL = code(i_type(43, 4, 5, 0xFF), NOP, JR_RA, NOP)            # misaligned word at a0 + 0xFF (spills a line)


def load_global(offset):                                         # v0 = *(0x80400000 + offset): zero-filled memory
    return code(i_type(15, 0, 8, 0x8040), i_type(35, 8, 2, offset), JR_RA, NOP)


def tail_call(last):                                             # f(7); return f(last), the second as `j f`
    return code(i_type(9, 29, 29, -24), i_type(43, 29, 31, 20), JAL, i_type(9, 0, 4, 7),
                i_type(35, 29, 31, 20), i_type(9, 29, 29, 24), J_F, i_type(9, 0, 4, last))


def slti(imm):                                                   # v0 = a0 < imm
    return code(i_type(10, 4, 2, imm), JR_RA, NOP)


class OracleTest(unittest.TestCase):
    def check(self, target, candidate, expected):
        verdict = oracle.compare(b"", target, VRAM, candidate, per_tier=12, coverage_rounds=8)
        self.assertEqual(verdict.verdict, expected, verdict.as_dict())
        self.assertFalse(verdict.as_dict()["proof"])
        return verdict

    def test_equivalent_code_is_not_distinguished(self):
        verdict = self.check(ADD, ADD_SWAPPED, "no-difference-found")
        self.assertEqual(verdict.coverage, 1.0)

    def test_result_register_difference(self):
        verdict = self.check(ADD, SUB, "differs")
        self.assertIsNotNone(verdict.counterexample)

    def test_memory_effect_difference(self):
        self.check(STORE, STORE, "no-difference-found")
        self.check(STORE, STORE_8, "differs")

    def test_call_arguments_are_observed(self):
        self.check(CALL, CALL, "no-difference-found")
        self.check(CALL, CALL_8, "differs")

    def test_callee_saved_registers_are_observed(self):
        self.check(ADD, SAVES_S0, "differs")

    def test_fails_closed_on_budget(self):
        verdict = oracle.compare(b"", LOOP, VRAM, LOOP, per_tier=4, coverage_rounds=2, max_steps=500)
        self.assertEqual(verdict.verdict, "unknown")

    def test_candidate_elsewhere(self):
        verdict = oracle.compare(b"", ADD, VRAM, ADD_SWAPPED, candidate_vram=0x80D00000, per_tier=8, coverage_rounds=4)
        self.assertEqual(verdict.verdict, "no-difference-found")

    def test_mutants(self):
        found = oracle.mutants(STORE + ADD, limit=8)
        self.assertTrue(found)
        for label, mutated in found:
            self.assertEqual(len(mutated), len(STORE + ADD))
            self.assertNotEqual(mutated, STORE + ADD, label)

    # ---- round-12 second review: O5, O6, O8, O9, O10, O12 and the input distribution ----
    def test_caller_saved_registers_do_not_survive_a_call(self):
        self.check(KEEP_S0, KEEP_A0, "differs")                  # passed only because the stub kept $a0
        self.check(KEEP_S0, KEEP_FRAME, "no-difference-found")   # memory across the call is fine

    def elsewhere(self, target, candidate):
        return oracle.compare(b"", target, VRAM, candidate, candidate_vram=ELSEWHERE, per_tier=12, coverage_rounds=8)

    def test_relocated_candidate_cannot_run_the_target(self):
        verdict = self.elsewhere(stray(VRAM), stray(VRAM))        # its `j` lands in the target's code
        self.assertEqual(verdict.verdict, "unknown", verdict.as_dict())
        self.assertGreater(verdict.candidate_unfinished, 0)
        verdict = self.elsewhere(stray(VRAM), stray(ELSEWHERE))   # linked where it runs
        self.assertEqual(verdict.verdict, "no-difference-found", verdict.as_dict())

    def test_relocated_recursion_through_the_symbol(self):
        verdict = self.elsewhere(recursive(VRAM), recursive(VRAM))
        self.assertEqual(verdict.verdict, "no-difference-found", verdict.as_dict())

    def test_accesses_outside_ram_fail_closed(self):
        self.check(NOTHING, STORE_FAR, "unknown")                # the store was dropped: agreed
        self.check(ZERO, LOAD_FAR, "unknown")                    # open bus read 0: agreed
        verdict = self.check(LOAD_FAR, LOAD_FAR, "unknown")
        self.assertEqual(verdict.inconclusive, 0)                # unknown, not "outside the domain"

    def test_misaligned_accesses_are_address_errors(self):
        self.check(HIGH_BYTE, SPILL, "differs")                  # its spill went unlogged
        self.check(load_global(0), load_global(1), "differs")    # both read zeros
        self.assertTrue(oracle.BareMachine(None).strict_alignment)
        lw = i_type(35, 4, 2, 1)
        self.assertNotIn("address error", emu._translate(VRAM, lw))   # boot emulation is unchanged
        self.assertIn("address error", emu._translate(VRAM, lw, False, True))

    def test_tail_call_to_a_stub_returns(self):
        self.check(tail_call(8), tail_call(8), "no-difference-found")
        self.check(tail_call(8), tail_call(9), "differs")

    def test_tlb_mapped_call_targets_leave_the_image_alone(self):
        runner = oracle.Runner(b"", VRAM, ADD, VRAM, {0x00400000})
        try:
            self.assertEqual(runner.base[0x400000:0x400004], bytes(4))
        finally:
            runner.release()

    def test_pointer_arguments(self):
        self.assertEqual(oracle._pointer_args(code(r_type(4, 0, 16, 0x21), i_type(35, 16, 2, 0), JR_RA, NOP)), {0})
        self.assertEqual(oracle._pointer_args(code(i_type(9, 4, 16, 8), i_type(35, 16, 2, 0), JR_RA, NOP)), {0})
        # jal's target bits are not a destination register (here they read as $a1)
        self.assertEqual(oracle._pointer_args(code(jal(0x80140000), i_type(35, 5, 2, 0), JR_RA, NOP)), {1})

    def test_boundary_tier_reaches_the_targets_immediates(self):
        self.assertTrue(any(i.args[0] == 1000 for i in oracle._inputs("boundary", 12, 0, slti(1000))))
        self.check(slti(1000), slti(1001), "differs")            # differs only at a0 == 1000

    def test_structured_arena(self):
        palette = oracle._palette(slti(1000))
        arena = oracle._arena(5, palette)
        self.assertEqual(len(arena), oracle.ARENA_SIZE)
        self.assertEqual(arena, oracle._arena.__wrapped__(5, palette))
        heads = [struct.unpack_from(">I", arena, b)[0] for b in range(0, oracle.ARENA_SIZE, oracle.BLOCK)]
        self.assertTrue(any(oracle.ARENA <= w < oracle.ARENA + oracle.ARENA_SIZE for w in heads))
        self.assertTrue(set(heads) & set(palette))

    def test_equivalent_mutant_sites_are_not_offered(self):
        self.assertEqual(oracle.mutants(code(r_type(4, 0, 2, 0x21), JR_RA, NOP)), [])   # move v0, a0
        self.assertEqual(oracle.mutants(code(i_type(9, 4, 0, 5), JR_RA, NOP)), [])       # addiu $zero, a0, 5
        self.assertEqual(oracle.mutants(code(i_type(4, 4, 5, 1), NOP, JR_RA, NOP)), [])  # beq a0, a1, .+8


if __name__ == "__main__":
    unittest.main()
