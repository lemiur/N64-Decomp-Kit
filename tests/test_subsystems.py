"""Subsystem hypotheses (n64dk.analysis.subsystems) on a hand-assembled two-cluster image."""
import struct
import unittest

from n64dk.analysis import runtime, subsystems

DELTA = 0x80001000
A0, A1, T6, SP, RA = 4, 5, 14, 29, 31
A1F, A2F, AISET, R1F, R2F = 0x80001000, 0x80001100, 0x80001200, 0x80001300, 0x80001400


def I(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def jal(target):
    return (3 << 26) | ((target >> 2) & 0x3FFFFFF)


JR_RA = (RA << 21) | 8


def calls(target):
    return [I(9, SP, SP, -0x18), I(43, SP, RA, 0x14), jal(target), 0, I(35, SP, RA, 0x14), JR_RA, I(9, SP, SP, 0x18)]


def model():
    pieces = [(A1F, calls(A2F)), (A2F, calls(AISET)), (AISET, [JR_RA, 0]), (R1F, calls(R2F)),
              # lui t6,0xA410; sw a0,0(t6) (DPC_START); sw a1,4(t6) (DPC_END)
              (R2F, [I(15, 0, T6, 0xA410), I(43, T6, A0, 0), I(43, T6, A1, 4), JR_RA, 0])]
    data = bytearray(0x1000)
    rows = []
    for address, words in pieces:
        data[address - DELTA:address - DELTA + 4 * len(words)] = b"".join(struct.pack(">I", w) for w in words)
        rows.append((f"f_{address:08X}", address, 4 * len(words)))
    return runtime.analyse_images([("main", bytes(data), DELTA, rows)], {AISET: "osAiSetNextBuffer"})


class SubsystemTest(unittest.TestCase):
    def test_two_clusters_get_role_hypotheses(self):
        found = subsystems.recover(model())
        by_member = {f: s for s in found for f in s.functions}
        audio, renderer = by_member[A1F], by_member[R1F]
        self.assertIsNot(audio, renderer)
        self.assertEqual(set(audio.functions), {A1F, A2F, AISET})
        self.assertEqual(set(renderer.functions), {R1F, R2F})
        self.assertEqual((audio.label, renderer.label), ("audio", "renderer"))
        self.assertTrue(audio.hypothesis and renderer.hypothesis)
        self.assertGreater(audio.confidence, 0)
        self.assertTrue(any("osAiSetNextBuffer" in e for e in audio.evidence))
        self.assertTrue(any("renderer hardware" in e for e in renderer.evidence))
        summary = subsystems.summary(found)
        self.assertEqual(summary["functions_by_label"], {"audio": 3, "renderer": 2})

    def test_label_propagation_is_deterministic(self):
        adj = {1: {2: 1.0}, 2: {1: 1.0, 3: 1.0}, 3: {2: 1.0}, 4: {5: 1.0}, 5: {4: 1.0}, 6: {}}
        labels = subsystems.communities(adj)
        self.assertEqual(labels, {1: 2, 2: 2, 3: 2, 4: 5, 5: 5, 6: 6})

    def test_name_roles(self):
        self.assertEqual(subsystems.name_role("osAiSetNextBuffer"), "audio")
        self.assertEqual(subsystems.name_role("osViSwapBuffer"), "video")
        self.assertEqual(subsystems.name_role("osCreateThread"), "os-runtime")
        self.assertIsNone(subsystems.name_role("func_80001234"))


if __name__ == "__main__":
    unittest.main()
