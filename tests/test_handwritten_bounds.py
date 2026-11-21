"""Hand-written code told apart by named structural reasons, and the function boundaries such code
needs (the Sesame Street cases: a handler reached only by a branch, a delay slot taken for a
start, a routine that never returns behind an ``eret``, a codec routine reached by ``bal`` behind
an 8-byte table and called once from past the text end, an endless loop's unreachable epilogue).
Every image is synthetic: built word by word here."""
import struct
import unittest

from n64dk.analysis import functions as fn
from n64dk.rom import Rom

ZERO, AT, V0, V1, A0, A1, T0, T1, T2, T3, T5 = 0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 13
S0, K0, K1, SP, FP, RA = 16, 26, 27, 29, 30, 31
NOP, JR_RA = 0, 0x03E00008
BASE = 0x80000400


def R(rs, rt, rd, sa, funct):
    return (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | funct


def I(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def addiu(rt, rs, imm):
    return I(9, rs, rt, imm)


def lw(rt, off, base):
    return I(0x23, base, rt, off)


def sw(rt, off, base):
    return I(0x2B, base, rt, off)


def lh(rt, off, base):
    return I(0x21, base, rt, off)


def jal(target):
    return (3 << 26) | ((target >> 2) & 0x3FFFFFF)


def j(target):
    return (2 << 26) | ((target >> 2) & 0x3FFFFFF)


def br(op, rs, rt, here, target):
    return I(op, rs, rt, (target - here - 4) >> 2)


def bal(here, target):
    return 0x04110000 | (((target - here - 4) >> 2) & 0xFFFF)


def mtc0(rt, rd):
    return 0x40800000 | (rt << 16) | (rd << 11)


def move(rd, rs):
    return R(rs, ZERO, rd, 0, 0x21)


def lui(rt, imm):
    return I(0x0F, 0, rt, imm)


def image(words, vram=BASE):
    return fn.Image(b"".join(w.to_bytes(4, "big") for w in words), vram)


def classified(words):
    img = image(words + [0, 0])
    f = fn.Function(BASE, BASE + 4 * len(words), BASE + 4 * len(words))
    fn.classify(img, f)
    return f


def rom_of(words_at: dict[int, list[int]]) -> Rom:
    data = bytearray(0x101000)
    struct.pack_into(">I", data, 0, 0x80371240)
    struct.pack_into(">I", data, 8, BASE)
    for off, words in words_at.items():
        for i, w in enumerate(words):
            struct.pack_into(">I", data, 0x1000 + off - 0x400 + 4 * i, w)
    return Rom(bytes(data))


class HandwrittenReasonTest(unittest.TestCase):
    def test_compiler_output_has_no_reason(self):
        f = classified([R(A0, A1, V0, 0, 0x21), JR_RA, NOP])
        self.assertEqual((f.kind, f.reasons), ("stub", []))
        framed = classified([addiu(SP, SP, -24), sw(RA, 20, SP), jal(BASE + 0x100), NOP,
                             lw(RA, 20, SP), JR_RA, addiu(SP, SP, 24)])
        self.assertEqual((framed.kind, framed.reasons), ("c", []))

    def test_privileged_instructions(self):
        f = classified([mtc0(A0, 12), JR_RA, NOP])
        self.assertEqual(f.kind, "asm")
        self.assertIn("privileged", f.reasons)

    def test_kernel_registers(self):
        f = classified([lw(K1, 0x118, K0), JR_RA, NOP])
        self.assertEqual(f.kind, "asm")
        self.assertIn("kernel_register", f.reasons)

    def test_callee_saved_register_clobbered_without_a_frame(self):
        f = classified([addiu(S0, A0, 1), JR_RA, move(V0, S0)])
        self.assertEqual(f.kind, "asm")
        self.assertEqual(f.reasons, ["callee_saved_clobbered"])

    def test_return_address_kept_in_another_register(self):
        f = classified([move(T5, RA), jal(BASE + 0x100), NOP, R(T5, 0, 0, 0, 8), NOP])
        self.assertEqual(f.kind, "asm")
        self.assertIn("ra_in_register", f.reasons)

    def test_pc_read_and_data_relative_to_ra(self):
        f = classified([bal(BASE, BASE + 8), NOP, addiu(T3, RA, 0x190), lh(T2, 0, T3), JR_RA, NOP])
        self.assertEqual(f.kind, "asm")
        self.assertIn("pc_read", f.reasons)
        self.assertIn("ra_relative", f.reasons)

    def test_a_branch_out_of_a_frameless_body(self):
        f = classified([br(4, A0, ZERO, BASE, BASE + 0x40), NOP, JR_RA, NOP])
        self.assertEqual(f.kind, "asm")
        self.assertEqual(f.reasons, ["branch_out"])

    def test_a_frameless_jump_into_another_routine(self):
        f = classified([br(5, A0, ZERO, BASE, BASE + 0x10), NOP, j(BASE + 0x100), NOP, j(BASE + 0x200), NOP])
        self.assertIn("tail_jump", f.reasons)

    def test_a_call_with_no_frame_and_ra_kept_nowhere(self):
        f = classified([move(A0, ZERO), jal(BASE + 0x100), NOP])
        self.assertEqual((f.kind, f.reasons), ("asm", ["call_without_saving_ra"]))

    def test_a_frame_released_before_the_body_ends(self):
        f = classified([addiu(SP, SP, -8), 0xF7AC0000, lw(A0, 0, SP), lw(A1, 4, SP), addiu(SP, SP, 8),
                        move(V1, A0), R(0, A0, A0, 20, 2), addiu(V0, A0, 1), addiu(V0, V0, 2),
                        JR_RA, NOP])                                         # sdc1 $f12, 0($sp)
        self.assertEqual(f.reasons, ["frame_released_early"])
        epilogue = classified([addiu(SP, SP, -24), sw(RA, 20, SP), jal(BASE + 0x100), NOP, lw(RA, 20, SP),
                               addiu(SP, SP, 24), JR_RA, NOP])                # IDO -O0: pop right before jr ra
        self.assertEqual(epilogue.reasons, [])

    def test_reasons_are_named_in_the_catalogue(self):
        self.assertTrue(set(fn.HANDWRITTEN_REASONS) >= {"privileged", "kernel_register", "callee_saved_clobbered",
                                                         "ra_in_register", "branch_out", "branched_into", "pc_read",
                                                         "ra_relative"})


class HandwrittenBoundaryTest(unittest.TestCase):
    """exceptasm.s-like code: an entry calls a dispatcher (eret) whose handler is reached only by
    a branch; a routine that never returns sits behind the eret and is reached by an address pair."""

    @classmethod
    def setUpClass(cls):
        X = BASE + 0x40              # dispatcher
        M = BASE + 0x70              # called by the dispatcher: lies between it and the handler
        H = BASE + 0x80              # handler, entered only by the branch at X+4
        C = BASE + 0xC0              # never returns: address taken by the entry
        D = BASE + 0x100             # called by C
        cls.X, cls.H, cls.C, cls.D = X, H, C, D
        cls.cat = fn.discover(rom_of({
            0x400: [lui(A0, 0x8000), addiu(A0, A0, C & 0xFFFF), sw(A0, 0, SP), jal(X), NOP, JR_RA, NOP],
            X - BASE + 0x400: [lw(T0, 0x18, K0), br(5, T0, ZERO, X + 4, H), NOP, jal(M), NOP, lw(K1, 0x118, K0),
                               mtc0(K1, 12), 0x42000018],                         # eret
            M - BASE + 0x400: [JR_RA, NOP],
            H - BASE + 0x400: [lui(AT, 0x3000), R(T0, AT, T1, 0, 0x24), R(0, T1, T1, 28, 2), addiu(T2, ZERO, 1),
                               br(5, T1, T2, H + 0x10, X + 0x14), NOP,              # bne back: the nop is a slot
                               addiu(T1, ZERO, 1), sw(T1, 0x18, K0), br(4, ZERO, ZERO, H + 0x20, X + 0x14),
                               sw(K1, 0x118, K0)],
            C - BASE + 0x400: [move(A0, ZERO), jal(D), NOP, NOP, NOP, NOP, NOP, NOP],
            D - BASE + 0x400: [addiu(SP, SP, -24), sw(RA, 20, SP), lw(RA, 20, SP), JR_RA, addiu(SP, SP, 24)],
        }))
        cls.by = {f.start: f for f in cls.cat.functions}

    def test_a_branch_target_behind_a_return_is_code_not_data(self):
        self.assertIn(self.H, self.by)
        self.assertFalse(any(a <= self.H < b for a, b in self.cat.data_regions))
        self.assertEqual(self.by[self.H].body_end, self.H + 0x28)
        self.assertEqual(self.by[self.H].kind, "asm")

    def test_a_delay_slot_is_never_a_start(self):
        self.assertNotIn(self.H + 0x14, self.by)

    def test_a_routine_that_never_returns_behind_an_eret_is_code(self):
        self.assertIn(self.C, self.by)
        self.assertFalse(any(a <= self.C < b for a, b in self.cat.data_regions))

    def test_trailing_alignment_is_not_body(self):
        f = self.by[self.C]
        self.assertEqual((f.body_end, f.end), (self.C + 0xC, self.D))
        self.assertIn("pad:0x34", fn.to_toml(self.cat))

    def test_cross_branches_are_one_unit_finding(self):
        units = [f for f in self.cat.findings if f.kind == "shared_unit"]
        self.assertTrue(any(f.function == self.H and str(f"{self.X:08X}") in f.detail for f in units))
        self.assertIn("branched_into", self.by[self.X].reasons)


class CodecTailTest(unittest.TestCase):
    """A routine past the text end, called once and framed, reaches by ``bal`` a routine behind an
    8-byte constant table that decodes as a jal; the table is data, both routines are found."""

    def test(self):
        S, T, TABLE, L, P = BASE + 0x20, BASE + 0x40, BASE + 0x50, BASE + 0x58, BASE + 0xA4
        cat = fn.discover(rom_of({
            0x400: [JR_RA, NOP],
            S - BASE + 0x400: [addiu(SP, SP, -24), sw(RA, 20, SP), jal(P), NOP, lw(RA, 20, SP), JR_RA,
                               addiu(SP, SP, 24)],
            T - BASE + 0x400: [R(A0, A1, V0, 0, 0x21), JR_RA, NOP],
            TABLE - BASE + 0x400: [0x0CCD2CCD, 0x53337FFF],
            L - BASE + 0x400: [lh(T2, 0, T1)] + [addiu(T1, T1, 2)] * 16 + [JR_RA, NOP],   # longer than a sweep window
            P - BASE + 0x400: [addiu(SP, SP, -0x44), sw(RA, 0, SP), addiu(T1, A0, 0), bal(P + 0xC, L), NOP,
                               0x04100000 | (((TABLE - P - 0x18) >> 2) & 0xFFFF), NOP,   # bltzal $zero: PC read
                               addiu(T3, RA, 0x18), R(T3, 0, RA, 0, 9), NOP,              # jalr to P+0x34
                               lw(RA, 0, SP), JR_RA, addiu(SP, SP, 0x44),
                               JR_RA, NOP],                                                 # reached only so
        }))
        starts = {f.start for f in cat.functions}
        self.assertLessEqual({S, T, L, P}, starts)
        self.assertNotIn(TABLE, starts)
        self.assertEqual(cat.text_end, P + 0x3C)        # no sweep past code one call vouches for
        self.assertTrue(any(f.address == TABLE for f in cat.findings if f.kind in ("stray_words", "in_text_data")))
        self.assertIn("pc_read", next(f for f in cat.functions if f.start == P).reasons)


class EpilogueFragmentTest(unittest.TestCase):
    """GCC -O0 ends a function with an endless loop by ``j .`` and then its epilogue, which nothing
    reaches: the epilogue belongs to the function, not to a function of its own."""

    def test(self):
        F, G = BASE + 0x20, BASE + 0x60
        cat = fn.discover(rom_of({
            0x400: [jal(F), NOP, jal(G), NOP, JR_RA, NOP],
            F - BASE + 0x400: [addiu(SP, SP, -32), sw(RA, 28, SP), sw(FP, 24, SP), move(FP, SP),
                               jal(G), NOP, j(F + 0x18), NOP,
                               move(SP, FP), lw(RA, 28, SP), lw(FP, 24, SP), JR_RA, addiu(SP, SP, 32)],
            G - BASE + 0x400: [JR_RA, NOP],
        }))
        by = {f.start: f for f in cat.functions}
        self.assertNotIn(F + 0x20, by)
        self.assertEqual(by[F].body_end, F + 0x34)


if __name__ == "__main__":
    unittest.main()
