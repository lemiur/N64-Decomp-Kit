"""EGCS 1.1.2 ``-O2 -mips3`` switch dispatch and blind function discovery.

With 64-bit registers EGCS adds the scaled index to the table base with ``daddu`` (the other
lanes use ``addu``):

    sltiu at, a0, N ; beqz at, default ; sll t0, a0, 2 ; lui v0, %hi(T)
    daddu v0, v0, t0 ; lw v0, %lo(T)(v0) ; jr v0 ; nop

Unrecognised, the table words became data-pointer seeds; every case label right after a
``jr ra`` and its delay slot is a split point, so each case was accepted as a function start,
the switch body was cut at the first one and the gap sweep then added the remaining labels
(case 0, default).  On the synthetic corpus that was 127 false starts over 14 switches.
Without the fix ``EgcsSwitchTest`` finds starts 0x440, 0x448, 0x450 and 0x45C as well.
"""
import struct
import unittest

from n64dk.analysis import functions as fn
from n64dk.rom import Rom

ZERO, AT, V0, V1, A0, A1, T0, SP, RA = 0, 1, 2, 3, 4, 5, 8, 29, 31
ADDU, DADDU = 0x21, 0x2D


def R(rs, rt, rd, sa, funct):
    return (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | funct


def I(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def jal(target):
    return (3 << 26) | ((target >> 2) & 0x3FFFFFF)


def addiu(rt, rs, imm):
    return I(9, rs, rt, imm)


def lw(rt, off, base):
    return I(0x23, base, rt, off)


def sw(rt, off, base):
    return I(0x2B, base, rt, off)


def jr(rs):
    return R(rs, 0, 0, 0, 8)


NOP = 0
SWITCH, CASES, DEFAULT, AFTER, TABLE = 0x80000420, (0x80000440, 0x80000448, 0x80000450), 0x8000045C, 0x80000464, 0x80000500


def program(add: int) -> dict[int, list[int]]:
    """Entry calls a three-case switch dispatched through ``add`` (``addu`` or ``daddu``).  The
    function right behind the switch starts after a ``jr ra`` and its delay slot exactly like
    the case labels do, and is reached only through a data word: the true-start control."""
    return {
        0x400: [jal(SWITCH), NOP, jr(RA), NOP],
        0x420: [I(0x0B, A0, AT, 3),                       # sltiu at, a0, 3
                I(4, AT, ZERO, (DEFAULT - 0x80000428) // 4),  # beqz at, default
                R(0, A0, T0, 2, 0),                       # sll t0, a0, 2
                I(0x0F, 0, V0, 0x8000),                   # lui v0, %hi(table)
                R(V0, T0, V0, 0, add),                    # (d)addu v0, v0, t0
                lw(V0, TABLE & 0xFFFF, V0),               # lw v0, %lo(table)(v0)
                jr(V0), NOP,
                jr(RA), I(0x0E, A0, V0, 0x76),            # case 0: xori v0, a0, 0x76
                jr(RA), addiu(V0, A0, 0x5E),              # case 1
                addiu(V0, ZERO, 0x7D), jr(RA), R(V0, A0, V0, 0, 0x23),   # case 2: subu v0, v0, a0
                jr(RA), addiu(V0, ZERO, -1),              # default
                addiu(SP, SP, -0x18), sw(RA, 0x14, SP),   # 0x464: framed function, pointer-only
                lw(RA, 0x14, SP), jr(RA), addiu(SP, SP, 0x18)],
        0x500: list(CASES),
        0x600: [AFTER],
    }


def rom_of(words_at: dict[int, list[int]], entry: int = 0x80000400) -> Rom:
    data = bytearray(0x101000)
    struct.pack_into(">I", data, 0, 0x80371240)
    struct.pack_into(">I", data, 8, entry)
    for off, words in words_at.items():
        for i, w in enumerate(words):
            struct.pack_into(">I", data, 0x1000 + off - 0x400 + 4 * i, w)
    return Rom(bytes(data))


class EgcsSwitchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = fn.discover(rom_of(program(DADDU)))

    def test_case_labels_are_not_function_starts(self):
        starts = [f.start for f in self.cat.functions]
        self.assertEqual(starts, [0x80000400, SWITCH, AFTER])
        for label in CASES + (DEFAULT,):
            self.assertNotIn(label, starts)

    def test_the_daddu_table_is_recognised(self):
        [table] = self.cat.jump_tables
        self.assertEqual((table.address, table.count, table.function, table.bounded, table.targets),
                         (TABLE, 3, SWITCH, True, list(CASES)))

    def test_the_switch_keeps_its_cases_and_default(self):
        by = {f.start: f for f in self.cat.functions}
        self.assertEqual(by[SWITCH].body_end, AFTER)

    def test_true_start_after_a_return_is_kept(self):
        by = {f.start: f for f in self.cat.functions}
        self.assertIn(AFTER, by)
        self.assertEqual(by[AFTER].body_end, AFTER + 0x14)


class AdduControlTest(unittest.TestCase):
    """The established ``addu`` dispatch (KMC, GCC 2.7.2/2.8.1) gives the same catalogue."""

    def test_same_catalogue_as_daddu(self):
        egcs = fn.discover(rom_of(program(DADDU)))
        gcc = fn.discover(rom_of(program(ADDU)))
        self.assertEqual([(f.start, f.body_end) for f in gcc.functions],
                         [(f.start, f.body_end) for f in egcs.functions])
        self.assertEqual([(t.address, t.count, t.targets) for t in gcc.jump_tables],
                         [(t.address, t.count, t.targets) for t in egcs.jump_tables])


class DadduWithoutConstantTest(unittest.TestCase):
    """A ``daddu`` of two run-time registers before ``jr`` (a pointer fetched from a computed
    address) is not a table dispatch."""

    def test_no_table(self):
        vram = 0x80001000
        words = [R(A0, A1, V1, 0, DADDU),                 # daddu v1, a0, a1
                 lw(V0, 0, V1), jr(V0), NOP]
        data = struct.pack(f">{len(words)}I", *words) + struct.pack(">I", CASES[0]) * 4
        an = fn.Analyser(fn.Image(data, vram), vram)
        self.assertIsNone(an.jump_table(vram, vram + 8))


class DaddiuBaseTest(unittest.TestCase):
    """``la`` with 64-bit registers is ``lui; daddiu``: the table base it builds is a table base
    (flow.find_jump_tables already folds ``daddiu``; the discovery analyser must too)."""

    def test_lui_daddiu_daddu_lw_jr_is_a_table(self):
        vram, table = 0x80001000, 0x80001020
        words = [I(0x0F, 0, V0, table >> 16),                # lui v0, %hi(table)
                 I(25, V0, V0, table & 0xFFFF),              # daddiu v0, v0, %lo(table)
                 R(V0, A0, V0, 0, DADDU),                    # daddu v0, v0, a0
                 lw(V0, 0, V0), jr(V0), NOP,                 # 0x10: jr v0
                 jr(RA), NOP,                                # 0x18: the case
                 vram + 0x18, vram + 0x18]                   # 0x20: the table
        data = struct.pack(f">{len(words)}I", *words)
        an = fn.Analyser(fn.Image(data, vram), vram)
        found = an.jump_table(vram, vram + 0x10)
        self.assertIsNotNone(found)
        self.assertEqual((found.address, found.targets), (table, [vram + 0x18, vram + 0x18]))


if __name__ == "__main__":
    unittest.main()
