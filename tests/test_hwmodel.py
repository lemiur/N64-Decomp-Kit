"""Hardware address model (n64dk.analysis.hwmodel) on hand-built addresses and MIPS words."""
import struct
import unittest

from n64dk.analysis import hwmodel

BASE = 0x80001000
A0, A1, V0, T6, T7, T8, RA = 4, 5, 2, 14, 15, 24, 31


def I(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def R(rs, rt, rd, sa, funct):
    return (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | funct


def code(words):
    return b"".join(struct.pack(">I", w) for w in words)


def lui(rt, imm): return I(15, 0, rt, imm)
def lw(rt, off, base): return I(35, base, rt, off)
def sw(rt, off, base): return I(43, base, rt, off)
def andi(rt, rs, imm): return I(12, rs, rt, imm)
def sll(rd, rt, sa): return R(0, rt, rd, sa, 0)
def addu(rd, rs, rt): return R(rs, rt, rd, 0, 0x21)


JR_RA = R(RA, 0, 0, 0, 8)


class AddressTest(unittest.TestCase):
    def test_segments_and_registers(self):
        vi = hwmodel.classify(0xA4400004)
        self.assertEqual((vi.segment, vi.device, vi.register, vi.physical, vi.cached),
                         ("kseg1", "vi", "VI_ORIGIN", 0x04400004, False))
        ram = hwmodel.classify(0x80001000)
        self.assertEqual((ram.segment, ram.device, ram.kind, ram.cached), ("kseg0", "rdram", "memory", True))
        self.assertEqual(hwmodel.classify(0xB0000000).device, "cart-rom")
        self.assertEqual(hwmodel.classify(0xBFC007C0).device, "pif-ram")
        self.assertEqual(hwmodel.classify(0xA4000000).device, "sp-dmem")

    def test_physical_alias_and_translated_segments(self):
        self.assertEqual(hwmodel.physical(0x80400000), hwmodel.physical(0xA0400000))
        self.assertEqual(hwmodel.classify(0x04400004).device, "tlb-mapped")      # KUSEG is translated
        self.assertEqual(hwmodel.classify_physical(0x04400004)[0], "vi")
        self.assertIsNone(hwmodel.physical(0x00001000))

    def test_side_effects_and_volatility(self):
        self.assertTrue(hwmodel.is_side_effecting(0xA4400004, store=True))
        self.assertFalse(hwmodel.is_side_effecting(0xA4600010, store=False))   # PI_STATUS read
        self.assertTrue(hwmodel.is_side_effecting(0xA404001C, store=False))    # SP_SEMAPHORE read
        self.assertFalse(hwmodel.is_side_effecting(0x80001000, store=True))
        self.assertTrue(hwmodel.is_volatile(0xA0001000))                       # uncached RDRAM
        self.assertFalse(hwmodel.is_volatile(0x80001000))
        self.assertTrue(hwmodel.is_volatile(0xA4600010))


class AccessTest(unittest.TestCase):
    def test_exact_register_accesses(self):
        words = [lui(T6, 0xA460), lw(T7, 0x10, T6), lui(T8, 0xA440), sw(A0, 4, T8), JR_RA, 0]
        found = hwmodel.mmio_accesses(code(words), BASE)
        self.assertEqual([(a.register, a.store, a.side_effecting, a.form) for a in found],
                         [("PI_STATUS", False, False, "exact"), ("VI_ORIGIN", True, True, "exact")])
        self.assertEqual(found[0].site, BASE + 4)
        self.assertIn("8DCF0010", found[0].evidence)
        self.assertEqual(hwmodel.roles(found), {"storage-dma": 1, "video": 1})

    def test_bounded_and_computed_addresses_keep_evidence(self):
        bounded = [andi(T7, A0, 3), sll(T7, T7, 2), lui(T6, 0xA450), addu(T6, T6, T7), lw(V0, 0, T6), JR_RA, 0]
        [access] = hwmodel.mmio_accesses(code(bounded), BASE)
        self.assertEqual((access.form, access.device, len(access.addresses)), ("set", "ai", 4))
        computed = [lui(T6, 0xA410), sll(T7, A0, 2), addu(T6, T6, T7), sw(A1, 0, T6), JR_RA, 0]
        [access] = hwmodel.mmio_accesses(code(computed), BASE)
        self.assertEqual((access.form, access.device, access.addresses, access.store), ("computed", "dpc", (), True))
        self.assertIn("0xa4100000", access.evidence)
        self.assertTrue(access.side_effecting)

    def test_ordinary_memory_is_not_reported(self):
        words = [lui(T6, 0x8001), lw(T7, 0x10, T6), lw(V0, 4, A0), JR_RA, 0]
        self.assertEqual(hwmodel.mmio_accesses(code(words), BASE), [])


if __name__ == "__main__":
    unittest.main()
