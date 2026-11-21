"""The six ways a masked comparison accepted wrong sources in the Sesame work, against the kit's one
strict comparison (``verify.strict_compare``; resolve acceptance, strict export, receipt issue and
replay and donor acceptance all call it).  Each case builds a correct object (the "truth", whose
relocated image is the target) and a wrong candidate that differs only where a masked compare looks
away; the candidate must never be ``exact``, the truth must be.  Synthetic objects only.
"""
import struct
import unittest

from n64dk.elf import R_MIPS_26, R_MIPS_32, R_MIPS_HI16, R_MIPS_LO16
from n64dk.match import verify
from n64dk.match.symbols import SymbolMap

try:
    from .test_verify_adversarial import BASE, GLOBAL, JR_RA, LOCAL, LUI, LW, LWC1, NOP, TEXT, build, i_type, words
except ImportError:      # run as a top-level module (unittest discover -s tests)
    from test_verify_adversarial import BASE, GLOBAL, JR_RA, LOCAL, LUI, LW, LWC1, NOP, TEXT, build, i_type, words

J, JAL, SB, ORI, ADDIU = 2, 3, 40, 13, 9
A1, T0, AT, F0 = 5, 8, 1, 0
RODATA = 0x80001000
GVAR = 0x80002000


def jtype(op, target):
    return (op << 26) | ((target >> 2) & 0x3FFFFFF)


def strict(elf, rom, symbols=None, bases=None, size=None, starts=None):
    size = elf.section(".text").size if size is None else size
    return verify.strict_compare(elf, bytes(rom), lambda v: v - BASE, TEXT, size,
                                 SymbolMap(symbols or {}, infer_names=False), bases, None, starts)


def image(elf, bases, symbols=None, size=0x4000):
    rom = bytearray(size)
    for name, base in bases.items():
        section = elf.section(name)
        if section is not None and section.size:
            data = elf.relocate(name, bases, symbols or {})
            rom[base - BASE:base - BASE + len(data)] = data
    return rom


class Case1InternalJump(unittest.TestCase):
    def obj(self, label):
        body = words(jtype(J, label), NOP, NOP, JR_RA, NOP, JR_RA, NOP, NOP)
        return build([(".text", body)], relocations={".text": [(0, R_MIPS_26, ".text")]})

    def test_wrong_internal_jump_is_rejected(self):
        rom = image(self.obj(0x14), {".text": TEXT})
        self.assertTrue(strict(self.obj(0x14), rom).exact)
        self.assertEqual(strict(self.obj(0x0C), rom).verdict, "mismatch")

    def test_wrong_local_call_is_rejected(self):
        def obj(value):
            body = words(jtype(JAL, 0), NOP, JR_RA, NOP, JR_RA, NOP, JR_RA, NOP)
            return build([(".text", body)], [("helper", value, LOCAL, 2, ".text")],
                         {".text": [(0, R_MIPS_26, "helper")]})
        rom = image(obj(0x10), {".text": TEXT})
        self.assertTrue(strict(obj(0x10), rom).exact)
        self.assertEqual(strict(obj(0x18), rom).verdict, "mismatch")


def global_store(offset, extra=0):
    """lui a1,%hi(gvar); lw a1,%lo(gvar)(a1); sb zero,offset(a1): the offset is not relocated."""
    body = words(i_type(LUI, 0, A1, 0), i_type(LW, A1, A1, extra), i_type(SB, A1, 0, offset), JR_RA, NOP)
    return build([(".text", body)], [("gvar", 0, GLOBAL, 0, 0)],
                 {".text": [(0, R_MIPS_HI16, "gvar"), (4, R_MIPS_LO16, "gvar")]})


class Case2OffsetOnLoadedRegister(unittest.TestCase):
    def test_store_offset_after_symbol_load_is_compared(self):
        rom = image(global_store(1), {".text": TEXT}, {"gvar": GVAR})
        self.assertTrue(strict(global_store(1), rom, {"gvar": GVAR}).exact)
        self.assertEqual(strict(global_store(2), rom, {"gvar": GVAR}).verdict, "mismatch")
        # with the symbol undeclared (taken from the retail words) the offset still decides
        self.assertEqual(strict(global_store(2), rom).verdict, "mismatch")


class Case3UnrelocatedConstants(unittest.TestCase):
    def obj(self, hi, lo, op):
        return build([(".text", words(i_type(LUI, 0, T0, hi), i_type(op, T0, T0, lo), JR_RA, NOP))])

    def test_hardware_address_constants_are_compared(self):
        for op in (ORI, ADDIU):
            rom = image(self.obj(0xA460, 0x10, op), {".text": TEXT})
            self.assertTrue(strict(self.obj(0xA460, 0x10, op), rom).exact)
            self.assertEqual(strict(self.obj(0xA460, 0x14, op), rom).verdict, "mismatch")
            self.assertEqual(strict(self.obj(0xA450, 0x10, op), rom).verdict, "mismatch")


def named_load(name, addend=0):
    body = words(i_type(LUI, 0, A1, 0), i_type(LW, A1, A1, addend), JR_RA, NOP)
    return build([(".text", body)], [(name, 0, GLOBAL, 0, 0)],
                 {".text": [(0, R_MIPS_HI16, name), (4, R_MIPS_LO16, name)]})


class Case4SymbolAddresses(unittest.TestCase):
    def test_wrong_addend_on_declared_symbol(self):
        rom = image(named_load("gvar"), {".text": TEXT}, {"gvar": GVAR})
        self.assertTrue(strict(named_load("gvar"), rom, {"gvar": GVAR}).exact)
        self.assertEqual(strict(named_load("gvar", 4), rom, {"gvar": GVAR}).verdict, "mismatch")

    def test_address_bearing_name_must_resolve_to_its_address(self):
        rom = image(named_load("x"), {".text": TEXT}, {"x": GVAR + 4})
        # the candidate says D_80002000 but the retail word reads 0x80002004: a wrong symbol
        verdict = strict(named_load("D_80002000"), rom)
        self.assertEqual(verdict.verdict, "mismatch")
        self.assertTrue(any("D_80002000" in c for c in verdict.conflicts))
        # declared at the wrong address in a symbol file: same contradiction
        verdict = strict(named_load("D_80002000"), rom, {"D_80002000": GVAR + 4})
        self.assertEqual(verdict.verdict, "mismatch")
        # symbol+addend that lands on the name's address is consistent (an assumption, never proof)
        rom = image(named_load("x", 4), {".text": TEXT}, {"x": GVAR})
        self.assertEqual(strict(named_load("D_80002000", 4), rom).verdict, "assumed")

    def test_address_bearing_function_call(self):
        def call(name):
            return build([(".text", words(jtype(JAL, 0), NOP, JR_RA, NOP))], [(name, 0, GLOBAL, 2, 0)],
                         {".text": [(0, R_MIPS_26, name)]})
        rom = image(call("f"), {".text": TEXT}, {"f": 0x80000800})
        self.assertEqual(strict(call("func_80000900"), rom).verdict, "mismatch")
        self.assertEqual(strict(call("func_80000800"), rom).verdict, "assumed")

    def test_one_name_one_address_across_units(self):
        rom_a = image(named_load("x"), {".text": TEXT}, {"x": GVAR})
        rom_b = image(named_load("x"), {".text": TEXT}, {"x": GVAR + 8})
        rows = [{"unit": "a", "verdict": strict(named_load("gvar"), rom_a)},
                {"unit": "b", "verdict": strict(named_load("gvar"), rom_b)}]
        self.assertIn("gvar", verify.reconcile_assumptions(rows)["contradictions"])


def float_unit(value, offset=0):
    rodata = struct.pack(">f", value) + bytes(4)
    body = words(i_type(LUI, 0, AT, 0), i_type(LWC1, AT, F0, offset), JR_RA, NOP)
    return build([(".text", body), (".rodata", rodata)], [],
                 {".text": [(0, R_MIPS_HI16, ".rodata"), (4, R_MIPS_LO16, ".rodata")]})


class Case5OwnedConstants(unittest.TestCase):
    def test_float_literal_sign_is_compared(self):
        bases = {".text": TEXT, ".rodata": RODATA}
        rom = image(float_unit(-1.0), bases)
        self.assertTrue(strict(float_unit(-1.0), rom, bases={".rodata": RODATA}).exact)
        self.assertEqual(strict(float_unit(1.0), rom, bases={".rodata": RODATA}).verdict, "mismatch")
        # base taken from the retail words: the literal is still compared there
        self.assertEqual(strict(float_unit(1.0), rom).verdict, "mismatch")

    def test_switch_table_entries_are_compared(self):
        def unit(label):
            body = words(i_type(LUI, 0, AT, 0), i_type(LW, AT, AT, 0), JR_RA, NOP, JR_RA, NOP, JR_RA, NOP)
            table = struct.pack(">II", 0x10, label)
            return build([(".text", body), (".rodata", table)], [],
                         {".text": [(0, R_MIPS_HI16, ".rodata"), (4, R_MIPS_LO16, ".rodata")],
                          ".rodata": [(0, R_MIPS_32, ".text"), (4, R_MIPS_32, ".text")]})
        bases = {".text": TEXT, ".rodata": RODATA}
        rom = image(unit(0x18), bases)
        self.assertTrue(strict(unit(0x18), rom, bases={".rodata": RODATA}).exact)
        self.assertEqual(strict(unit(0x10), rom, bases={".rodata": RODATA}).verdict, "mismatch")
        self.assertEqual(strict(unit(0x10), rom).verdict, "mismatch")


class Case6Size(unittest.TestCase):
    def test_code_past_the_owned_size_is_rejected(self):
        truth = build([(".text", words(JR_RA, NOP))])
        rom = image(truth, {".text": TEXT})
        rom[TEXT - BASE + 8:TEXT - BASE + 12] = words(JR_RA)          # the next function
        longer = build([(".text", words(JR_RA, NOP, JR_RA, NOP))])
        self.assertTrue(strict(truth, rom, size=8).exact)
        self.assertEqual(strict(longer, rom, size=8).verdict, "mismatch")

    def test_alignment_words_are_not_code(self):
        # a 0x10-byte routine whose catalogue says 0x20: zero words after it are alignment in the image,
        # but a candidate shorter than the owned size proves nothing about them
        rom = image(build([(".text", words(NOP, NOP, JR_RA, NOP))]), {".text": TEXT})
        short = build([(".text", words(NOP, NOP, JR_RA, NOP))])
        self.assertTrue(strict(short, rom, size=0x10).exact)
        self.assertNotEqual(strict(short, rom, size=0x20).verdict, "exact")
        # trailing zero padding of the object past the owned size is padding, non-zero is not
        padded = build([(".text", words(NOP, NOP, JR_RA, NOP, 0, 0, 0, 0))])
        self.assertTrue(strict(padded, rom, size=0x10).exact)


if __name__ == "__main__":
    unittest.main()
