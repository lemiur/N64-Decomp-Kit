"""Indirect control flow (n64dk.analysis.indirect) on hand-assembled big-endian MIPS words."""
import struct
import unittest

from n64dk.analysis import indirect

BASE = 0x80001000
AT, V0, V1, A0, T6, T7, T9, SP, RA = 1, 2, 3, 4, 14, 15, 25, 29, 31
F1, F2 = 0x80004000, 0x80004100


def I(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def R(rs, rt, rd, sa, funct):
    return (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | funct


def lui(rt, imm): return I(15, 0, rt, imm)
def addiu(rt, rs, imm): return I(9, rs, rt, imm)
def sltiu(rt, rs, imm): return I(11, rs, rt, imm)
def andi(rt, rs, imm): return I(12, rs, rt, imm)
def lw(rt, off, base): return I(35, base, rt, off)
def sw(rt, off, base): return I(43, base, rt, off)
def beq(rs, rt, off): return I(4, rs, rt, off)
def bne(rs, rt, off): return I(5, rs, rt, off)
def sll(rd, rt, sa): return R(0, rt, rd, sa, 0)
def addu(rd, rs, rt): return R(rs, rt, rd, 0, 0x21)
def move(rd, rs): return R(rs, 0, rd, 0, 0x25)
def jr(rs): return R(rs, 0, 0, 0, 8)
def jalr(rs): return R(rs, 0, RA, 0, 9)
def jal(target): return (3 << 26) | ((target >> 2) & 0x3FFFFFF)


NOP = 0
JR_RA = jr(RA)
FRAME_IN = [addiu(SP, SP, -0x18), sw(RA, 0x14, SP)]
FRAME_OUT = [lw(RA, 0x14, SP), JR_RA, addiu(SP, SP, 0x18)]


def code(words):
    return b"".join(struct.pack(">I", w) for w in words)


def reader(*pieces):
    """read_word over (address, [words]) pieces."""
    memory = {}
    for address, words in pieces:
        for k, word in enumerate(words):
            memory[address + 4 * k] = word
    return memory.get


def image(pieces, lo, hi):
    out = bytearray(hi - lo)
    for address, words in pieces:
        out[address - lo:address - lo + 4 * len(words)] = code(words)
    return bytes(out)


# IDO: sltiu at,a0,3; beqz at,default; sll t6,a0,2 (slot); lui at; addu at,at,t6; lw t6,0x2000(at); jr t6
IDO_SWITCH = [sltiu(AT, A0, 3), beq(AT, 0, 10), sll(T6, A0, 2), lui(AT, 0x8000), addu(AT, AT, T6),
              lw(T6, 0x2000, AT), jr(T6), NOP,
              JR_RA, addiu(V0, 0, 1), JR_RA, addiu(V0, 0, 2), JR_RA, move(V0, 0)]
IDO_TABLE = [BASE + 0x20, BASE + 0x28, BASE + 0x30, BASE + 0x20]       # a fourth word the bound excludes

# KMC/GCC: addiu v0,a0,-2; sltiu v1,v0,2; beqz v1,default; sll v0,v0,2; lui v1; addiu v1,0x2000;
# addu v0,v0,v1; lw v0,0(v0); jr v0
GCC_SWITCH = [addiu(V0, A0, -2), sltiu(V1, V0, 2), beq(V1, 0, 11), sll(V0, V0, 2), lui(V1, 0x8000),
              addiu(V1, V1, 0x2000), addu(V0, V0, V1), lw(V0, 0, V0), jr(V0), NOP,
              JR_RA, addiu(V0, 0, 1), JR_RA, addiu(V0, 0, 2), JR_RA, NOP]
GCC_TABLE = [BASE + 0x28, BASE + 0x30, BASE + 0x38]


class SwitchTest(unittest.TestCase):
    def test_ido_switch_bounded_by_sltiu_in_the_delay_slot(self):
        read = reader((BASE, IDO_SWITCH), (0x80002000, IDO_TABLE))
        result = indirect.analyse_function(code(IDO_SWITCH), BASE, read)
        [site] = result.sites
        self.assertEqual((site.site, site.kind, site.status, site.call), (BASE + 0x18, "switch", "definite", False))
        self.assertEqual((site.table, site.count, site.stride), (0x80002000, 3, 4))
        self.assertEqual(site.targets, (BASE + 0x20, BASE + 0x28, BASE + 0x30))
        self.assertEqual(result.owned, [(0x80002000, 12)])
        self.assertEqual(result.tables, {BASE + 0x18: [BASE + 0x20, BASE + 0x28, BASE + 0x30]})

    def test_gcc_switch_with_lower_bound_and_addiu_table(self):
        read = reader((BASE, GCC_SWITCH), (0x80002000, GCC_TABLE))
        [site] = indirect.analyse_function(code(GCC_SWITCH), BASE, read).sites
        self.assertEqual((site.kind, site.status, site.table, site.count), ("switch", "definite", 0x80002000, 2))
        self.assertEqual(site.targets, (BASE + 0x28, BASE + 0x30))

    def test_bound_on_the_taken_edge_of_bnez(self):
        words = [sltiu(AT, A0, 2), bne(AT, 0, 3), sll(T6, A0, 2), JR_RA, move(V0, 0),
                 lui(AT, 0x8000), addu(AT, AT, T6), lw(T6, 0x2000, AT), jr(T6), NOP,
                 JR_RA, addiu(V0, 0, 1), JR_RA, addiu(V0, 0, 2)]
        read = reader((BASE, words), (0x80002000, [BASE + 0x28, BASE + 0x30, BASE]))
        [site] = indirect.analyse_function(code(words), BASE, read).sites
        self.assertEqual((site.kind, site.status, site.count), ("switch", "definite", 2))

    def test_unbounded_table_is_only_a_candidate_set(self):
        words = [sll(T6, A0, 2), lui(AT, 0x8000), addu(AT, AT, T6), lw(T6, 0x2000, AT), jr(T6), NOP,
                 JR_RA, NOP, JR_RA, NOP]
        read = reader((BASE, words), (0x80002000, [BASE + 0x18, BASE + 0x20, 0]))
        result = indirect.analyse_function(code(words), BASE, read)
        [site] = result.sites
        self.assertEqual((site.kind, site.status, site.count), ("switch-unbounded", "finite", 2))
        self.assertEqual(result.owned, [])                 # only a bounded switch owns its table

    def test_table_outside_the_function_is_not_a_switch(self):
        read = reader((BASE, IDO_SWITCH), (0x80002000, [F1, F2, 0x80004201]))
        [site] = indirect.analyse_function(code(IDO_SWITCH), BASE, read, starts={BASE, F1, F2}).sites
        self.assertEqual((site.kind, site.status), ("pointer-table", "finite"))
        self.assertEqual((site.targets, site.rejected), ((F1, F2), (0x80004201,)))


class PointerTest(unittest.TestCase):
    def test_masked_index_into_a_function_pointer_table(self):
        words = FRAME_IN + [andi(T6, A0, 1), sll(T6, T6, 2), lui(T7, 0x8000), addu(T7, T7, T6),
                            lw(T9, 0x3000, T7), jalr(T9), NOP] + FRAME_OUT
        read = reader((BASE, words), (0x80003000, [F1, F2, 0x12345678]))
        [site] = indirect.analyse_function(code(words), BASE, read, starts={BASE, F1, F2}).sites
        self.assertTrue(site.call)
        self.assertEqual((site.kind, site.status, site.targets, site.count), ("pointer-table", "finite", (F1, F2), 2))

    def test_structure_stride_with_pointer_field(self):
        words = [sll(T6, A0, 3), lui(T7, 0x8000), addu(T7, T7, T6), lw(T9, 0x3004, T7), jr(T9), NOP]
        table = [0x11111111, F1, 0x22222222, F2, 0x33333333, 0]
        read = reader((BASE, words), (0x80003000, table), (F1, [JR_RA, NOP]), (F2, [JR_RA, NOP]))
        [site] = indirect.analyse_function(code(words), BASE, read, starts={BASE, F1, F2}).sites
        self.assertEqual((site.kind, site.status, site.stride, site.targets), ("pointer-table", "finite", 8, (F1, F2)))

    def test_la_then_jalr_is_definite_and_a_bad_constant_impossible(self):
        good = [lui(T9, 0x8000), addiu(T9, T9, 0x4000), jalr(T9), NOP, JR_RA, NOP]
        bad = [lui(T9, 0x8000), addiu(T9, T9, 0x4002), jalr(T9), NOP, JR_RA, NOP]
        read = reader((F1, [JR_RA, NOP]))
        [site] = indirect.analyse_function(code(good), BASE, read, starts={F1}).sites
        self.assertEqual((site.kind, site.status, site.targets), ("constant", "definite", (F1,)))
        [site] = indirect.analyse_function(code(bad), BASE, read, starts={F1}).sites
        self.assertEqual((site.status, site.rejected), ("impossible", (0x80004002,)))

    def test_pointer_spilled_to_the_stack_and_reloaded(self):
        words = [lui(T6, 0x8000), addiu(T6, T6, 0x4000), sw(T6, 0x10, SP), lw(T9, 0x10, SP), jalr(T9), NOP,
                 JR_RA, NOP]
        [site] = indirect.analyse_function(code(words), BASE, reader(), starts={F1}).sites
        self.assertEqual((site.status, site.targets), ("definite", (F1,)))

    def test_argument_is_unresolved_with_provenance(self):
        [site] = indirect.analyse_function(code([jr(A0), NOP]), BASE, reader()).sites
        self.assertEqual(site.status, "unresolved")
        self.assertIn("$a0", " ".join(site.provenance))


class ImageTest(unittest.TestCase):
    SETTER, CALLER, OPEN = 0x80001000, 0x80001100, 0x80001200
    GLOBAL = 0x80005000

    def pieces(self, open_store=False):
        setter = [lui(T6, 0x8000), addiu(T6, T6, 0x4000), lui(AT, 0x8000), sw(T6, 0x5000, AT), JR_RA, NOP]
        caller = FRAME_IN + [lui(T9, 0x8000), lw(T9, 0x5000, T9), jalr(T9), NOP] + FRAME_OUT + [NOP]
        pieces = [(self.SETTER, setter), (self.CALLER, caller), (F1, [JR_RA, NOP])]
        rows = [("setter", self.SETTER, 4 * len(setter)), ("caller", self.CALLER, 4 * len(caller)),
                ("target", F1, 8)]
        if open_store:
            other = [lui(AT, 0x8000), sw(A0, 0x5000, AT), JR_RA, NOP]
            pieces.append((self.OPEN, other))
            rows.append(("other", self.OPEN, 4 * len(other)))
        return pieces, rows

    def run_image(self, open_store=False):
        pieces, rows = self.pieces(open_store)
        data = image(pieces, 0x80001000, 0x80005010)
        return indirect.analyse_image(data, 0x80001000, rows)

    def test_pointer_stored_by_one_function_called_by_another(self):
        result = self.run_image()
        [site] = [s for s in result.sites if s.function == self.CALLER]
        self.assertEqual((site.kind, site.status, site.targets, site.table),
                         ("stored-pointer", "finite", (F1,), self.GLOBAL))

    def test_unknown_store_leaves_it_unresolved(self):
        result = self.run_image(open_store=True)
        [site] = [s for s in result.sites if s.function == self.CALLER]
        self.assertEqual((site.status, site.targets), ("unresolved", (F1,)))
        self.assertIn("open", " ".join(site.provenance))

    def test_summary_counts(self):
        summary = self.run_image().summary()
        self.assertEqual((summary["functions"], summary["jalr_sites"], summary["by_status"]["finite"]), (3, 1, 1))
        self.assertEqual(summary["no_return"], 0)


class NoReturnTest(unittest.TestCase):
    PANIC, WRAPPER, GUARDED = 0x80001000, 0x80001100, 0x80001200

    def test_infinite_loop_and_callers_that_always_reach_it(self):
        panic = [beq(0, 0, -1), NOP]                                   # b . (spins forever)
        wrapper = FRAME_IN + [jal(self.PANIC), NOP] + FRAME_OUT        # calls it unconditionally
        guarded = [beq(A0, 0, 3), NOP, jal(self.PANIC), NOP, JR_RA, NOP]
        pieces = [(self.PANIC, panic), (self.WRAPPER, wrapper), (self.GUARDED, guarded)]
        rows = [("panic", self.PANIC, 8), ("wrapper", self.WRAPPER, 4 * len(wrapper)),
                ("guarded", self.GUARDED, 4 * len(guarded))]
        result = indirect.analyse_image(image(pieces, 0x80001000, 0x80001300), 0x80001000, rows)
        self.assertEqual(result.no_return, {self.PANIC, self.WRAPPER})
        fake = {(f["site"], f["callee"], f["inside"]) for f in result.fake}
        self.assertIn((indirect.hx(self.WRAPPER + 8), indirect.hx(self.PANIC), True), fake)
        self.assertIn((indirect.hx(self.GUARDED + 8), indirect.hx(self.PANIC), True), fake)

    def test_seeds_and_unknown_callees(self):
        shape = indirect.Shape(0x100, 0x110, [(((0x104, 0x9000), ), ("return", None), ())])
        self.assertEqual(indirect.no_return_functions({0x100: shape}), set())          # unknown callee returns
        self.assertEqual(indirect.no_return_functions({0x100: shape, 0x9000: indirect.Shape(0x9000, 0x9008, [
            ((), None, (0,))])}), {0x100, 0x9000})                                      # self loop never returns
        self.assertEqual(indirect.no_return_functions({0x100: shape}, seeds={0x100}), {0x100})


if __name__ == "__main__":
    unittest.main()
