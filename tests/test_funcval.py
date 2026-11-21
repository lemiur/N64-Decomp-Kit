"""Deterministic functional validation on the built-in interpreter: milestones, checkpoints, stop
classes, entropy, and the first divergence between two synthetic images (no real ROM)."""
import struct
import unittest

from n64dk.match import funcval
from n64dk.rom import Rom

ZERO, V0, T0, T1, T2, T3 = 0, 2, 8, 9, 10, 11


def I(op, rs, rt, imm): return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)
def lui(rt, imm): return I(15, 0, rt, imm)
def ori(rt, rs, imm): return I(13, rs, rt, imm)
def addiu(rt, rs, imm): return I(9, rs, rt, imm)
def sw(rt, off, base): return I(43, base, rt, off)
def lw(rt, off, base): return I(35, base, rt, off)
def j(addr): return (2 << 26) | ((addr >> 2) & 0x3FFFFFF)


IDLE = I(4, 0, 0, -1)                                    # b .
SYSCALL = 0x0000000C
GAME = 0x80180000
# entry: DMA ROM 0x2000..0x2100 to 0x80180000 (index 7), write VI_STATUS (index 10), jump there
BOOT = [lui(T0, 0xA460), lui(T1, 0x0018), sw(T1, 0, T0), lui(T1, 0x1000), ori(T1, T1, 0x2000), sw(T1, 4, T0),
        addiu(T1, ZERO, 0xFF), sw(T1, 0xC, T0), lui(T2, 0xA440), addiu(T3, ZERO, 3), sw(T3, 0, T2), j(GAME), 0]


def rom(entry_code, game_code=(addiu(V0, ZERO, 7), IDLE, 0)):
    data = bytearray(0x101000 + 0x1000)
    struct.pack_into(">III", data, 0, 0x80371240, 0, 0x80000400)
    struct.pack_into(f">{len(entry_code)}I", data, 0x1000, *entry_code)
    struct.pack_into(f">{len(game_code)}I", data, 0x2000, *game_code)
    return Rom(bytes(data))


CONFIG = funcval.Config(max_instructions=1000, ram_size=0x200000, main=False, watch=(("game", GAME),))


class MilestoneTests(unittest.TestCase):
    def test_boot_milestones_and_stop(self):
        t = funcval.BuiltinBackend().run(rom(BOOT), CONFIG)
        self.assertEqual([(m.name, m.step) for m in t.milestones],
                         [("boot", 0), ("entry", 0), ("first_dma", 7), ("first_vi_write", 10), ("watch:game", 13),
                          ("handoff", 13), ("end", 14)])
        boot, entry, dma = t.milestones[:3]
        self.assertEqual(boot.rdram, entry.rdram)
        self.assertNotEqual(dma.rdram, boot.rdram)            # the DMA wrote RAM
        self.assertIsNone(dma.pc)                             # inside an instruction
        self.assertEqual(t.milestone("handoff").pc, GAME)
        self.assertEqual(t.stop["class"], "idle")
        self.assertEqual((t.events["dmas"], t.events["dma_bytes"], t.events["vi_writes"]), (1, 0x100, 1))
        self.assertEqual(t.events["first_dmas"][0]["rom"], "0x00002000")
        d = t.as_dict()
        self.assertFalse(d["strict"])
        self.assertNotIn("pages", d)                          # only hashes leave the module

    def test_deterministic_and_keyed(self):
        a = funcval.BuiltinBackend().run(rom(BOOT), CONFIG).as_dict()
        b = funcval.BuiltinBackend().run(rom(BOOT), CONFIG).as_dict()
        self.assertEqual(a, b)
        c = funcval.BuiltinBackend().run(rom(BOOT), funcval.Config(max_instructions=999, ram_size=0x200000,
                                                                   main=False)).as_dict()
        self.assertNotEqual(c["cache_key"], a["cache_key"])

    def test_checkpoint_from_a_fresh_run(self):
        cfg = funcval.Config(max_instructions=1000, ram_size=0x200000, main=False, checkpoints=(3,))
        t = funcval.BuiltinBackend().run(rom(BOOT), cfg)
        cp = t.milestone("instructions:3")
        self.assertEqual((cp.step, cp.pc), (3, 0x8000040C))
        self.assertEqual(cp.rdram, t.milestone("boot").rdram)   # no RAM store in the first three
        self.assertNotEqual(cp.cpu, t.milestone("boot").cpu)

    def test_entropy_seed_changes_ram_not_the_image(self):
        zero = funcval.BuiltinBackend().run(rom(BOOT), CONFIG)
        cfg = funcval.Config(max_instructions=1000, ram_size=0x200000, main=False, entropy_seed=7)
        seeded = funcval.BuiltinBackend().run(rom(BOOT), cfg)
        self.assertNotEqual(seeded.milestone("boot").rdram, zero.milestone("boot").rdram)
        self.assertEqual([m.name for m in seeded.milestones][:4], ["boot", "entry", "first_dma", "first_vi_write"])
        self.assertEqual(seeded.as_dict(), funcval.BuiltinBackend().run(rom(BOOT), cfg).as_dict())

    def test_crash_and_exception_classes(self):
        crash = funcval.BuiltinBackend().run(rom([lui(T0, 0x0040), lw(T1, 0, T0), 0]), CONFIG)
        self.assertEqual(crash.stop["class"], "crash")
        self.assertEqual(crash.stop["steps"], 1)
        exc = funcval.BuiltinBackend().run(rom([addiu(T0, ZERO, 1), SYSCALL, 0]), CONFIG)
        self.assertEqual(exc.stop["class"], "exception")
        budget = funcval.BuiltinBackend().run(rom(BOOT), funcval.Config(max_instructions=5, ram_size=0x200000,
                                                                        main=False))
        self.assertEqual(budget.stop["class"], "budget")

    def test_backends_listing(self):
        listed = funcval.backends()
        self.assertEqual(listed[0], {"name": "builtin", "available": True, "installed": True})
        self.assertEqual({b["name"] for b in listed[1:]}, {"ares", "mupen64plus", "simple64"})
        self.assertTrue(all(b["available"] is False for b in listed[1:]))    # detection only


class DivergenceTests(unittest.TestCase):
    def test_same_image_does_not_diverge(self):
        out = funcval.first_divergence(rom(BOOT), rom(BOOT), CONFIG)
        self.assertIsNone(out["first_milestone"])
        self.assertIsNone(out["first_instruction"])
        self.assertEqual(out["image_pages_differing"], 0)

    def test_first_divergence_is_bisected_to_the_instruction(self):
        changed = rom(BOOT, (addiu(V0, ZERO, 8), IDLE, 0))
        out = funcval.first_divergence(rom(BOOT), changed, CONFIG)
        # ROM 0x2000 is inside the first MiB, so the images already differ in RAM at boot
        self.assertEqual(out["image_pages_differing"], 1)
        self.assertEqual(out["image_pages_first"], ["0x80001000"])
        self.assertEqual(out["first_milestone"]["a"]["name"], "boot")
        self.assertEqual(out["first_cpu_milestone"]["a"]["name"], "end")
        first = out["first_instruction"]
        self.assertEqual((first["instructions"], first["diverging_index"], first["diverging_pc"]),
                         (14, 13, "0x80180000"))
        self.assertTrue(first["exact"])

    def test_bisection_budget_fails_closed(self):
        changed = rom(BOOT, (addiu(V0, ZERO, 8), IDLE, 0))
        out = funcval.first_divergence(rom(BOOT), changed, CONFIG, bisect_budget=30)
        self.assertFalse(out["first_instruction"]["exact"])
        self.assertIsNone(out["first_instruction"]["instructions"])


if __name__ == "__main__":
    unittest.main()
