"""Generic loader/codec understanding: static loop classes, PI DMA roles, dynamic capture receipts."""
import hashlib
import struct
import tomllib
import unittest
from pathlib import Path

from n64dk.analysis import loaders_generic as lg
from n64dk.match import images

ZERO, AT, V0, A0, A1, A2, A3 = 0, 1, 2, 4, 5, 6, 7
T0, T1, T2, T3, T4, T5, T6, T7, T8, T9 = 8, 9, 10, 11, 12, 13, 14, 15, 24, 25
SP, RA = 29, 31


def R(fn, rs=0, rt=0, rd=0, sa=0):
    return (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | fn


def I(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def addu(rd, rs, rt): return R(0x21, rs, rt, rd)
def subu(rd, rs, rt): return R(0x23, rs, rt, rd)
def or_(rd, rs, rt): return R(0x25, rs, rt, rd)
def sltu(rd, rs, rt): return R(0x2B, rs, rt, rd)
def sll(rd, rt, sa): return R(0x00, 0, rt, rd, sa)
def srl(rd, rt, sa): return R(0x02, 0, rt, rd, sa)
def jr(rs): return R(0x08, rs)
def addiu(rt, rs, imm): return I(9, rs, rt, imm)
def andi(rt, rs, imm): return I(12, rs, rt, imm)
def ori(rt, rs, imm): return I(13, rs, rt, imm)
def lui(rt, imm): return I(15, 0, rt, imm)
def lbu(rt, off, base): return I(36, base, rt, off)
def lw(rt, off, base): return I(35, base, rt, off)
def sb(rt, off, base): return I(40, base, rt, off)
def sw(rt, off, base): return I(43, base, rt, off)
def jal(addr): return (3 << 26) | ((addr >> 2) & 0x3FFFFFF)


NOP, JR_RA = 0, jr(RA)


def br(op, rs, rt, label):
    return lambda pc, labels: I(op, rs, rt, (labels[label] - pc - 4) >> 2)


def assemble(vram, items):
    """``items``: words, ``("label", name)`` or callables ``f(pc, labels)`` resolved in a second pass."""
    labels, pc = {}, vram
    for item in items:
        if isinstance(item, tuple):
            labels[item[1]] = pc
        else:
            pc += 4
    words, pc = [], vram
    for item in items:
        if isinstance(item, tuple):
            continue
        words.append(item(pc, labels) if callable(item) else item)
        pc += 4
    return struct.pack(f">{len(words)}I", *words), labels


MEMCPY = [("label", "loop"), lw(T0, 0, A1), addiu(A1, A1, 4), sw(T0, 0, A0), addiu(A2, A2, -4),
          br(5, A2, ZERO, "loop"), addiu(A0, A0, 4), JR_RA, NOP]
CHECKSUM = [or_(V0, ZERO, ZERO), ("label", "loop"), lw(T0, 0, A0), addiu(A0, A0, 4), br(5, A0, A1, "loop"),
            addu(V0, V0, T0), JR_RA, NOP]
FIXUP = [("label", "loop"), lw(T0, 0, A0), addiu(A0, A0, 4), addu(T1, A2, T0), lw(T2, 0, T1), addu(T2, T2, A2),
         sw(T2, 0, T1), br(5, A0, A1, "loop"), NOP, JR_RA, NOP]
# An LZ decoder: a code byte whose bits (shifted out of a register) choose literal or back reference;
# a reference is 16 bits: 4-bit length - 3, 12-bit distance - 1 (window 0x1000).
LZ = [("label", "outer"), lbu(T0, 0, A0), addiu(A0, A0, 1), addiu(T1, ZERO, 8),
      ("label", "bitloop"), andi(T2, T0, 0x80), br(4, T2, ZERO, "backref"), sll(T0, T0, 1),
      lbu(T3, 0, A0), addiu(A0, A0, 1), sb(T3, 0, A1), br(4, ZERO, ZERO, "next"), addiu(A1, A1, 1),
      ("label", "backref"), lbu(T4, 0, A0), lbu(T5, 1, A0), addiu(A0, A0, 2), sll(T6, T4, 8), or_(T6, T6, T5),
      andi(T7, T6, 0x0FFF), srl(T8, T6, 12), addiu(T8, T8, 3), subu(T9, A1, T7), addiu(T9, T9, -1),
      ("label", "copy"), lbu(V0, 0, T9), addiu(T9, T9, 1), sb(V0, 0, A1), addiu(T8, T8, -1),
      br(5, T8, ZERO, "copy"), addiu(A1, A1, 1),
      ("label", "next"), addiu(T1, T1, -1), br(5, T1, ZERO, "bitloop"), NOP,
      sltu(AT, A1, A2), br(5, AT, ZERO, "outer"), NOP, JR_RA, NOP]
FILL = [("label", "loop"), sw(ZERO, 0, A0), addiu(A0, A0, 4), br(5, A0, A1, "loop"), NOP, JR_RA, NOP]

PI_FUNC = 0x80003000
# pi_start(cart offset a0, dram a1, size a2): an osPiRawStartDma-like routine (spills, a call that
# translates the RAM address, PI register stores)
PI_START = [addiu(SP, SP, -24), sw(RA, 20, SP), sw(A0, 24, SP), sw(A2, 32, SP),
            jal(0x80004000), or_(A0, A1, ZERO),
            lui(T0, 0xA460), sw(V0, 0, T0),
            lw(T1, 24, SP), lui(AT, 0x1000), or_(T1, T1, AT), sw(T1, 4, T0),
            lw(T2, 32, SP), addiu(T2, T2, -1), sw(T2, 12, T0),
            lw(RA, 20, SP), JR_RA, addiu(SP, SP, 24)]
CALLER = [lui(A0, 0x0001), ori(A0, A0, 0x2340), lui(A1, 0x8020), jal(PI_FUNC), addiu(A2, ZERO, 0x1000),
          JR_RA, NOP]


def summary(items, vram=0x80001000):
    code, _labels = assemble(vram, items)
    return lg.summarise(code, vram)


class StaticTests(unittest.TestCase):
    def test_memcpy_is_copy(self):
        s = summary(MEMCPY)
        self.assertEqual(s["class"], "copy")
        self.assertEqual(len(s["loops"]), 1)
        self.assertEqual((s["loops"][0]["width"], s["loops"][0]["stride"]), (4, 4))
        self.assertFalse(s["lz"])
        self.assertNotIn("pi_dma", s)

    def test_checksum(self):
        s = summary(CHECKSUM)
        self.assertEqual(s["class"], "checksum")
        self.assertEqual(s["loops"][0]["accumulator"], V0)

    def test_relocation_pass_is_fixup(self):
        self.assertEqual(summary(FIXUP)["class"], "fixup")

    def test_fill(self):
        self.assertEqual(summary(FILL)["class"], "fill")

    def test_lz_decoder_is_transform_with_window(self):
        s = summary(LZ, 0x80002000)
        self.assertEqual(s["class"], "transform")
        self.assertTrue(s["lz"])
        self.assertEqual(s["window"], 0x1000)
        classes = [loop["class"] for loop in s["loops"]]
        self.assertEqual(len(classes), 3)                       # copy loop, bit loop, token loop
        self.assertIn("copy", classes)                         # the back-reference copy on its own
        self.assertTrue(any(loop["bit_buffer"] for loop in s["loops"]))

    def test_straight_line_has_no_class(self):
        s = summary([addu(V0, A0, A1), JR_RA, NOP])
        self.assertIsNone(s["class"])
        self.assertEqual(s["loops"], [])

    def test_pi_dma_roles_and_call_site_range(self):
        code, _ = assemble(PI_FUNC, PI_START)
        pi = lg.summarise(code, PI_FUNC)["pi_dma"]
        self.assertEqual(pi["direction"], "cart-to-ram")
        self.assertEqual(pi["roles"]["dram"]["arg"], "a1")       # through the translating call
        self.assertEqual(pi["roles"]["cart"]["arg"], "a0")       # through a stack spill
        self.assertEqual(pi["roles"]["len"], {"arg": "a2", "addend": -1})
        self.assertNotIn("range", pi)
        caller, _ = assemble(0x80005000, CALLER)
        calls = lg.call_constants(lg.decode_range(caller, 0x80005000))
        self.assertEqual(calls, [(0x8000500C, PI_FUNC, (0x12340, 0x80200000, 0x1000, None))])
        ranges = lg.dma_ranges({PI_FUNC: pi}, calls)
        self.assertEqual(len(ranges), 1)
        self.assertEqual((ranges[0]["rom"], ranges[0]["rom_end"], ranges[0]["ram"], ranges[0]["size"]),
                         ("0x00012340", "0x00013340", "0x80200000", 0x1000))

    def test_constant_pi_dma_gives_a_range(self):
        code, _ = assemble(0x80001000, [lui(T0, 0xA460), lui(T1, 0x0010), sw(T1, 0, T0), lui(T1, 0x1000),
                                        ori(T1, T1, 0x2000), sw(T1, 4, T0), addiu(T1, ZERO, 0xFF),
                                        sw(T1, 12, T0), JR_RA, NOP])
        pi = lg.summarise(code, 0x80001000)["pi_dma"]
        self.assertEqual(pi["range"], {"rom": "0x00002000", "rom_end": "0x00002100", "ram": "0x80100000",
                                       "size": 0x100})


ROM = bytes(0x80 | (i & 0x7F) for i in range(0x400))       # synthetic cartridge bytes, never zero
CODE_AT = 0x80001000
COPY = assemble(CODE_AT, MEMCPY)[0]
DMA = assemble(CODE_AT, [lui(T0, 0xA460), sw(A1, 0, T0), sw(A0, 4, T0), addiu(T1, A2, -1), sw(T1, 12, T0),
                         JR_RA, NOP])[0]


def machine(code):
    m = lg.bare_machine(ROM, ram_size=0x40000)    # stack window 0x8002FF00..0x8003FF10
    p = CODE_AT & 0x1FFFFFFF
    m.ram[p:p + len(code)] = code
    return m


class CaptureTests(unittest.TestCase):
    def test_copy_from_cartridge(self):
        r = lg.capture(machine(COPY), CODE_AT, (0x80010000, 0xB0000100, 0x40))
        self.assertEqual(r["status"], "ok", r["reason"])
        self.assertEqual(r["instructions"], 16 * 6 + 2)
        self.assertEqual(r["ram_range"], ["0x80010000", "0x80010040"])
        self.assertEqual(r["output_sha256"], hashlib.sha256(ROM[0x100:0x140]).hexdigest())
        self.assertEqual(r["rom_ranges"], [{"rom": "0x00000000", "rom_end": "0x00000400", "via": "cart-read",
                                            "granule": 4096}])
        self.assertEqual(r["ram_written"], [{"ram": "0x80010000", "end": "0x80010040", "taint": "rom-derived"}])
        self.assertFalse(r["strict"])
        # deterministic: the same machine state gives the same receipt and cache key
        again = lg.capture(machine(COPY), CODE_AT, (0x80010000, 0xB0000100, 0x40))
        self.assertEqual(again, r)
        other = lg.capture(machine(COPY), CODE_AT, (0x80010000, 0xB0000100, 0x44))
        self.assertNotEqual(other["cache_key"], r["cache_key"])

    def test_pi_dma_is_recorded_and_tainted(self):
        m = machine(DMA)
        r = lg.capture(m, CODE_AT, (0x10000200, 0x00010000, 0x80))
        self.assertEqual(r["status"], "ok", r["reason"])
        self.assertEqual(r["rom_ranges"][0], {"rom": "0x00000200", "rom_end": "0x00000280", "via": "pi-dma",
                                              "ram": "0x80010000"})
        self.assertEqual(r["ram_written"], [{"ram": "0x80010000", "end": "0x80010080", "taint": "rom-dma"}])
        self.assertEqual(lg.capture_output(m, r), ROM[0x200:0x280])
        # the same routine, statically: the PI roles of its arguments
        roles = lg.summarise(DMA, CODE_AT)["pi_dma"]["roles"]
        self.assertEqual((roles["dram"]["arg"], roles["cart"]["arg"], roles["len"]["addend"]), ("a1", "a0", -1))

    def test_fails_closed_on_the_instruction_limit(self):
        r = lg.capture(machine(COPY), CODE_AT, (0x80010000, 0xB0000100, 0x40), lg.Limits(max_instructions=50))
        self.assertEqual(r["status"], "failed")
        self.assertIn("step budget", r["reason"])
        self.assertIsNone(r["output_sha256"])
        self.assertIsNone(r["ram_range"])
        with self.assertRaises(ValueError):
            lg.register_image(r, "decoded")
        with self.assertRaises(ValueError):
            lg.capture_output(machine(COPY), r)

    def test_fails_closed_on_open_bus_and_on_no_output(self):
        r = lg.capture(machine(COPY), CODE_AT, (0x80010000, 0xB0100000, 0x10))   # past the cartridge
        self.assertEqual(r["status"], "failed")
        self.assertIn("open bus", r["reason"])
        r = lg.capture(machine(COPY), CODE_AT, (0x80010000, 0x80018000, 0x10))   # RAM to RAM: no ROM read
        self.assertEqual(r["status"], "failed")
        self.assertEqual(r["reason"], "no ROM-derived output")

    def test_output_limit(self):
        r = lg.capture(machine(COPY), CODE_AT, (0x80010000, 0xB0000100, 0x40), lg.Limits(max_output=0x20))
        self.assertEqual(r["status"], "failed")
        self.assertIn("max_output", r["reason"])

    def test_register_image_proposal(self):
        r = lg.capture(machine(COPY), CODE_AT, (0x80010000, 0xB0000100, 0x40))
        p = lg.register_image(r, "decoded_main")
        self.assertFalse(p["applied"])
        self.assertFalse(p["strict"])
        self.assertEqual(p["provenance"]["cache_key"], r["cache_key"])
        table = tomllib.loads(p["toml"])["images"]["decoded_main"]
        self.assertEqual(table["vram_delta"], 0x80010000)
        self.assertEqual(table["extent"], [0x80010000, 0x80010040])
        transform = table["raw_segments"][0][3]
        self.assertTrue(transform.startswith("capture:"))
        # the registry never maps decoded bytes back to ROM as a copy
        spec = images.ImageSpec("decoded_main", Path("decoded/decoded_main.bin"), table["vram_delta"],
                                raw_segments=[tuple(table["raw_segments"][0])])
        rom, why = spec.rom_of(0x10)
        self.assertIsNone(rom)
        self.assertIn("not representable", why)
        with self.assertRaises(ValueError):
            lg.register_image(r, "bad name")


if __name__ == "__main__":
    unittest.main()
