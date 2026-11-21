"""Seeded property tests of the strict verifier and relocation handling (round 12, item R).

Each property is checked over generated inputs (fixed seeds, so a failure is reproducible: the
failing case's seed and index are in the subTest message; a minimised failing case is to be added
to ``tests/test_verify_adversarial.py`` as a regression test).

Properties:
1. Changing any owned byte of the target (code or data) never leaves a strict ``exact``.
2. Changing a byte the unit does not own never changes an ``exact`` verdict.
3. An undeclared or wrongly declared external address never yields ``exact``; the right one does.
4. A wrong ``_gp`` never yields ``exact`` for GP-relative code; the right one does.
5. The relocation-masked hash is invariant under relinking (lineage), and changes with any
   non-relocated field.
"""
import random
import struct
import unittest

from n64dk.analysis import lineage
from tests.test_verify_adversarial import (BASE, GLOBAL, LUI, LW, NOTYPE, PLAIN, R_MIPS_GPREL16, R_MIPS_HI16,
                                           R_MIPS_LO16, TEXT, build, i_type, image, strict, words, JR_RA, NOP)

CASES = 120
DATA = 0x80001000


def random_code(rng: random.Random, n: int) -> bytes:
    # arbitrary words: the comparison of unrelocated code is byte-exact whatever they decode to
    return words(*[rng.getrandbits(32) for _ in range(n)], JR_RA, NOP)


def random_data(rng: random.Random, n: int) -> bytes:
    body = bytearray(rng.getrandbits(8) for _ in range(n))
    body[-1] |= 1                                        # no trailing zeros (a separate rule)
    return bytes(body)


class OwnedBytesProperty(unittest.TestCase):
    def test_any_owned_byte_change_breaks_exact_and_unowned_changes_do_not(self):
        rng = random.Random(1201)
        for case in range(CASES):
            code = random_code(rng, rng.randint(2, 24))
            data = random_data(rng, 4 * rng.randint(1, 16))
            elf = build([(".text", code), (".data", data)])
            bases = {".text": TEXT, ".data": DATA}
            rom = image(elf, bases)
            with self.subTest(seed=1201, case=case):
                self.assertEqual(strict(elf, rom, bases={".data": DATA}).verdict, "exact")
                where = rng.choice([(TEXT, len(code)), (DATA, len(data))])
                offset = where[0] - BASE + rng.randrange(where[1])
                changed = bytearray(rom)
                changed[offset] ^= 1 << rng.randrange(8)
                self.assertNotEqual(strict(elf, changed, bases={".data": DATA}).verdict, "exact")
                outside = bytearray(rom)
                spot = rng.randrange(0x2000, len(rom))
                outside[spot] ^= 0xFF
                self.assertEqual(strict(elf, outside, bases={".data": DATA}).verdict, "exact")


class DeclaredAddressProperty(unittest.TestCase):
    def test_only_the_declared_right_address_is_exact(self):
        rng = random.Random(1202)
        code = words(i_type(LUI, 0, 2, 0), i_type(LW, 2, 2, 0), JR_RA, NOP)
        elf = build([(".text", code)], symbols=[("gTarget", 0, GLOBAL, NOTYPE, 0)],
                    relocations={".text": [(0, R_MIPS_HI16, "gTarget"), (4, R_MIPS_LO16, "gTarget")]})
        for case in range(CASES):
            address = 0x80100000 + 4 * rng.randrange(0x40000)
            rom = image(elf, {".text": TEXT}, {"gTarget": address})
            with self.subTest(seed=1202, case=case, address=hex(address)):
                self.assertEqual(strict(elf, rom, symbols={"gTarget": address}).verdict, "exact")
                self.assertNotEqual(strict(elf, rom).verdict, "exact")                     # undeclared
                wrong = address + 4 * rng.randint(1, 0x1000) * rng.choice((-1, 1))
                self.assertNotEqual(strict(elf, rom, symbols={"gTarget": wrong}).verdict, "exact")


class GpProperty(unittest.TestCase):
    def test_only_the_right_gp_is_exact(self):
        rng = random.Random(1203)
        code = words(i_type(LW, 28, 2, 0), JR_RA, NOP)
        elf = build([(".text", code)], symbols=[("gSmall", 0, GLOBAL, NOTYPE, 0)],
                    relocations={".text": [(0, R_MIPS_GPREL16, "gSmall")]})
        for case in range(CASES):
            gp = 0x80080000 + 16 * rng.randrange(0x1000)
            small = gp + 4 * rng.randint(-0x1F00, 0x1F00)
            symbols = {"_gp": gp, "gSmall": small}
            rom = image(elf, {".text": TEXT}, symbols)
            with self.subTest(seed=1203, case=case, gp=hex(gp), small=hex(small)):
                self.assertEqual(strict(elf, rom, symbols=symbols).verdict, "exact")
                shift = 4 * rng.randint(1, 0x40) * rng.choice((-1, 1))
                self.assertNotEqual(strict(elf, rom, symbols={"_gp": gp + shift, "gSmall": small}).verdict, "exact")


class MaskedHashProperty(unittest.TestCase):
    def test_relinking_keeps_the_masked_hash_and_edits_change_it(self):
        rng = random.Random(1204)
        for case in range(CASES):
            def body(hi, lo, target, constant):
                return words(0x3C080000 | hi, 0x8D090000 | lo, 0x0C000000 | target, 0x25290000 | constant,
                             JR_RA, NOP, 0x27BDFFE8, 0xAFBF0014)
            constant = rng.getrandbits(15)
            a = body(rng.getrandbits(16), rng.getrandbits(16), rng.getrandbits(26), constant)
            b = body(rng.getrandbits(16), rng.getrandbits(16), rng.getrandbits(26), constant)
            c = body(rng.getrandbits(16), rng.getrandbits(16), rng.getrandbits(26), constant ^ 1)
            with self.subTest(seed=1204, case=case):
                ha, hb, hc = (lineage.hash_function("t", "f", 0, x).masked for x in (a, b, c))
                self.assertEqual(ha, hb)
                self.assertNotEqual(ha, hc)


if __name__ == "__main__":
    unittest.main()
