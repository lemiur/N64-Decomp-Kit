import unittest

from n64dk.match import ir

BASE = 0x80001000
TARGET = 0x80001234
ZERO, AT, V0, A0, A1 = 0, 1, 2, 4, 5
T0, T1, T2, T3, T6, T7, S0, S1, SP, RA = 8, 9, 10, 11, 14, 15, 16, 17, 29, 31


def R(fn, rs=0, rt=0, rd=0, sa=0):
    return (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | fn


def I(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def J(op, target):
    return (op << 26) | ((target >> 2) & 0x3FFFFFF)


def F(fmt, ft, fs, fd, fn):
    return (17 << 26) | (fmt << 21) | (ft << 16) | (fs << 11) | (fd << 6) | fn


NOP, JR_RA = 0, R(8, rs=RA)


def original(const=3, slt=10, branch=4):
    return [
        I(9, SP, SP, -24),        # addiu sp, sp, -24
        I(43, SP, RA, 20),        # sw    ra, 20(sp)
        I(43, SP, S0, 16),        # sw    s0, 16(sp)
        R(37, A0, ZERO, S0),      # move  s0, a0
        I(35, A0, T6, 4),         # lw    t6, 4(a0)
        I(9, T6, T7, const),      # addiu t7, t6, const
        I(43, S0, ZERO, 12),      # sw    zero, 12(s0)
        I(43, S0, T7, 8),         # sw    t7, 8(s0)
        J(3, TARGET),             # jal   TARGET
        I(9, T6, A0, 1),          #  addiu a0, t6, 1          (delay slot: an argument)
        I(slt, V0, T0, 10),       # slti  t0, v0, 10
        I(branch, T0, ZERO, 2),   # beqz  t0, epilogue
        NOP,
        I(43, S0, V0, 0),         # sw    v0, 0(s0)
        I(35, SP, RA, 20),        # lw    ra, 20(sp)
        I(35, SP, S0, 16),        # lw    s0, 16(sp)
        JR_RA,
        I(9, SP, SP, 24),         #  addiu sp, sp, 24
    ]


def renamed():
    """``original`` with other registers, another frame, swapped saves and stores, a hoisted load."""
    return [
        I(9, SP, SP, -32),
        I(43, SP, S1, 24),        # sw    s1, 24(sp)
        I(43, SP, RA, 28),        # sw    ra, 28(sp)
        I(35, A0, T1, 4),         # lw    t1, 4(a0)           (above the move)
        R(37, A0, ZERO, S1),      # move  s1, a0
        I(9, T1, T2, 3),          # addiu t2, t1, 3
        I(43, S1, T2, 8),         # sw    t2, 8(s1)           (stores swapped)
        I(43, S1, ZERO, 12),      # sw    zero, 12(s1)
        J(3, TARGET),
        I(9, T1, A0, 1),
        I(10, V0, T3, 10),        # slti  t3, v0, 10
        I(4, T3, ZERO, 2),
        NOP,
        I(43, S1, V0, 0),
        I(35, SP, RA, 28),
        I(35, SP, S1, 24),
        JR_RA,
        I(9, SP, SP, 32),
    ]


def fp(words, **options):
    return ir.fingerprint(ir.lift(words, BASE), **options)


def only(fn, kind):
    found = [s for _b, s in fn.statements(kind)]
    assert len(found) == 1, found
    return found[0]


class AllocationIndependenceTest(unittest.TestCase):
    def test_renamed_and_rescheduled_function_has_the_same_fingerprint(self):
        a, b = fp(original()), fp(renamed())
        self.assertEqual(a.digest, b.digest, ir.compare(a, b)["differing_effects"])
        self.assertEqual(ir.compare(a, b)["verdict"], "same-semantics")
        self.assertEqual(ir.similarity(a, b), 1.0)
        self.assertNotEqual(a.profile["frame"], b.profile["frame"])       # allocation stays out
        self.assertTrue(a.profile["promoted"] and b.profile["promoted"])

    def test_effects_are_semantic(self):
        effects = fp(original()).effects
        self.assertIn("store32(add(#8,arg0),add(#3,load32s(add(#4,arg0))))", effects)
        self.assertIn("store32(add(#c,arg0),#0)", effects)
        self.assertIn("call(#80001234)", effects)
        self.assertIn("call.arg0(#80001234):add(#1,load32s(add(#4,arg0)))", effects)
        self.assertIn("branch(lt(result.v0(#80001234),#a))", effects)
        self.assertIn("store32(arg0,result.v0(#80001234))", effects)
        self.assertFalse([k for k in effects if "stack" in k or k.endswith(":in")])     # saves are gone

    def test_value_numbers_ignore_registers(self):
        a, b = ir.lift(original(), BASE), ir.lift(renamed(), BASE)
        self.assertEqual(set(ir.value_numbers(a).values()), set(ir.value_numbers(b).values()))

    def test_text_shows_statements(self):
        text = ir.lift(original(), BASE).text()
        self.assertIn("call(#80001234, args=[a0=v", text)
        self.assertIn("branch(lt(", text)
        self.assertIn("store32(", text)


class SemanticDifferenceTest(unittest.TestCase):
    def check_different(self, words):
        a, b = fp(original()), fp(words)
        result = ir.compare(a, b)
        self.assertNotEqual(a.digest, b.digest)
        self.assertEqual(result["verdict"], "different-semantics")
        self.assertTrue(0.5 < result["score"] < 1.0, result)
        return result

    def test_changed_constant(self):
        result = self.check_different(original(const=5))
        self.assertIn("store32(add(#8,arg0),add(#5,load32s(add(#4,arg0))))", result["differing_effects"])

    def test_changed_comparison(self):
        result = self.check_different(original(slt=11))                  # sltiu: unsigned compare
        self.assertIn("branch(ltu(result.v0(#80001234),#a))", result["differing_effects"])

    def test_inverted_branch(self):
        result = self.check_different(original(branch=5))                # bnez: same test, other target
        self.assertEqual(result["differing_effects"], [])
        self.assertLess(result["parts"]["cfg"], 1.0)


class DelaySlotTest(unittest.TestCase):
    def test_branch_reads_its_operands_before_the_delay_slot(self):
        words = [I(4, A0, ZERO, 2),          # beq   a0, zero, L
                 I(9, A0, A0, 1),            #  addiu a0, a0, 1        (runs on both paths)
                 I(43, A1, A0, 0),           # sw    a0, 0(a1)
                 JR_RA,                      # L: jr ra
                 I(43, A1, A0, 4)]           #  sw    a0, 4(a1)
        fn = ir.lift(words, BASE)
        branch = only(fn, "branch")
        self.assertEqual(ir.shape(fn, branch.args[0]), "eq(#0,arg0)")
        self.assertEqual([s.kind for s in fn.blocks[0].stmts], ["branch"])
        stores = [s for _b, s in fn.statements("store")]
        self.assertEqual([ir.shape(fn, s.args[1]) for s in stores], ["add(#1,arg0)"] * 2)
        incremented = fn.find(stores[0].args[1])
        self.assertEqual(fn.find(stores[1].args[1]), incremented)
        self.assertEqual(fn.values[incremented].block, 0)                 # lifted with the branch block
        users = fn.def_use()[incremented]
        self.assertEqual(len([u for u in users if u[0] == "stmt"]), 2)

    def likely(self, op):
        return [I(9, ZERO, V0, 1),           # li    v0, 1
                I(op, A0, ZERO, 2),          # beql  a0, zero, L
                I(9, ZERO, V0, 5),           #  li    v0, 5
                I(43, A1, ZERO, 0),          # sw    zero, 0(a1)
                JR_RA,                       # L: jr ra
                NOP]

    def test_branch_likely_slot_runs_on_the_taken_edge_only(self):
        fn = ir.lift(self.likely(20), BASE)
        returned = fn.value(only(fn, "ret").args[0])
        self.assertEqual(returned.op, "phi")
        self.assertEqual(sorted(fn.value(a).attr for a in returned.args), [1, 5])
        slot = [b for b in fn.blocks if b.start == -1]
        self.assertEqual(len(slot), 1)
        self.assertEqual(slot[0].stmts[-1].kind, "goto")

    def test_ordinary_branch_slot_runs_on_both_edges(self):
        fn = ir.lift(self.likely(4), BASE)
        returned = fn.value(only(fn, "ret").args[0])
        self.assertEqual((returned.op, returned.attr), ("const", 5))
        self.assertFalse([b for b in fn.blocks if b.start == -1])

    def test_float_compare_and_bc1f(self):
        words = [I(49, A0, 4, 0),            # lwc1  f4, 0(a0)
                 I(49, A0, 6, 4),            # lwc1  f6, 4(a0)
                 F(16, 6, 4, 0, 0x3C),       # c.lt.s f4, f6
                 (17 << 26) | (8 << 21) | 2,  # bc1f  L
                 F(16, 6, 4, 0, 0),          #  add.s f0, f4, f6
                 F(16, 6, 4, 0, 1),          # sub.s f0, f4, f6
                 JR_RA,                      # L: jr ra
                 NOP]
        fn = ir.lift(words, BASE)
        branch = only(fn, "branch")
        self.assertEqual(ir.shape(fn, branch.args[0]), "c.lt.s(load32f(arg0),load32f(add(#4,arg0)))")
        not_taken = fn.blocks[branch.targets[0]]                          # bc1f: the swapped edge
        self.assertEqual(not_taken.start, 5)
        returned = fn.value(only(fn, "ret").args[1])
        self.assertEqual(returned.op, "phi")
        self.assertEqual(sorted(fn.value(a).op for a in returned.args), ["add.s", "sub.s"])

    def test_address_pairs_fold(self):
        words = [I(15, ZERO, AT, 0x8001), I(35, AT, T0, 0x2345),          # lw t0, %lo(0x80012345)(at)
                 I(15, ZERO, T1, 0x8001), I(9, T1, T1, -16),              # t1 = 0x8000FFF0
                 I(43, T1, T0, 0), JR_RA, NOP]
        self.assertIn("store32(#8000fff0,load32s(#80012345))", fp(words).effects)
        self.assertIn("store32(#addr,load32s(#addr))", fp(words, symbolic_addresses=True).effects)

    def test_escaping_stack_keeps_memory_but_drops_saves(self):
        words = [I(9, SP, SP, -24), I(43, SP, RA, 20), J(3, TARGET),
                 I(9, SP, A0, 16),           #  addiu a0, sp, 16      (address of a local escapes)
                 I(35, SP, RA, 20), JR_RA, I(9, SP, SP, 24)]
        result = fp(words)
        self.assertFalse(result.profile["promoted"])
        self.assertEqual(result.profile["saves"], 1)
        self.assertFalse([k for k in result.effects if k.startswith("store")])
        self.assertIn("call.arg0(#80001234):stack", result.effects)


class FailClosedTest(unittest.TestCase):
    def test_unsupported_instructions_are_explicit(self):
        for word in (0x40086000,                 # mfc0 t0, $12
                     I(47, A0, 0x19, 0),         # cache
                     F(20, 6, 4, 0, 0)):         # add.w: not a valid COP1 operation
            with self.subTest(word=hex(word)):
                fn = ir.lift([word, JR_RA, R(37, T0, ZERO, V0)], BASE)
                unknown = only(fn, "unknown")
                self.assertEqual(unknown.attr[0], word)
                self.assertIn("unknown(", fn.text())
                result = ir.fingerprint(fn)
                self.assertEqual(result.unknown, 1)
                self.assertFalse(result.complete)
                self.assertTrue([k for k in result.effects if k.startswith("unknown(")])
                self.assertEqual(ir.compare(result, result)["verdict"], "incomplete")

    def test_falling_off_the_end_is_unknown(self):
        fn = ir.lift([I(9, ZERO, V0, 1), I(9, V0, V0, 1)], BASE)
        self.assertEqual(only(fn, "unknown").attr[2], "falls off the end")

    def test_unreachable_padding_is_ignored(self):
        result = fp([JR_RA, NOP, NOP, NOP])
        self.assertTrue(result.complete)
        self.assertEqual(result.cfg, (("ret", ()),))


if __name__ == "__main__":
    unittest.main()
