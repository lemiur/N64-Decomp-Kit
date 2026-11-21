"""Generated-code detection: template families with a generator hypothesis that regenerates every
member, the hold-out check, switches and fixed-stride data tables (hand-built words only)."""
import struct
import unittest

from n64dk.analysis import generated

ZERO, V0, A0, A1, T0, AT, SP, RA = 0, 2, 4, 5, 8, 1, 29, 31


def I(op, rs, rt, imm): return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)
def lui(rt, imm): return I(15, 0, rt, imm)
def addiu(rt, rs, imm): return I(9, rs, rt, imm)
def lw(rt, off, base): return I(35, base, rt, off)
def sw(rt, off, base): return I(43, base, rt, off)
def jal(addr): return (3 << 26) | ((addr >> 2) & 0x3FFFFFF)
def code(words): return struct.pack(f">{len(words)}I", *words)


JR_RA = 0x03E00008
HI = lambda p: ((p + 0x8000) >> 16) & 0xFFFF                     # noqa: E731
LO = lambda p: p & 0xFFFF                                        # noqa: E731
ADDRESSES = [0x80127FF0, 0x80128010, 0x80130020, 0x8013FFF8]     # lo halves on both sides of 0x8000


def wrapper(p):
    """f(void) { g(&table_entry_p); } -- the same wrapper generated per table entry."""
    return [addiu(SP, SP, -24), sw(RA, 20, SP), lui(A0, HI(p)), jal(0x80300000), addiu(A0, A0, LO(p)),
            lw(RA, 20, SP), JR_RA, addiu(SP, SP, 24)]


def mover(off, off2):
    """*(a1 + off2) = *(a0 + off), the field offsets generated from one table."""
    return [lw(V0, off, A0), 0, sw(V0, off2, A1), JR_RA, 0]


def functions(bodies, base=0x80100000, pad=0):
    out, at = [], base
    for i, words in enumerate(bodies):
        blob = code(words + [0] * pad)
        out.append((f"f{i}", at, blob))
        at += len(blob)
    return out


class TemplateTests(unittest.TestCase):
    def test_wrappers_share_one_address_parameter(self):
        found = generated.clusters(functions([wrapper(p) for p in ADDRESSES]))
        self.assertEqual(len(found), 1)
        c = found[0]
        self.assertEqual((c.kind, len(c.members), c.words), ("wrapper", 4, 8))
        self.assertEqual(c.hypothesis.params, [{"name": "p0", "kind": "address", "bits": 32}])
        self.assertEqual(c.table, [[p] for p in ADDRESSES])
        self.assertEqual((c.template_exact, c.holdout_exact), (4, 4))
        # regenerating from template + parameters gives each member's words
        for p in ADDRESSES:
            self.assertEqual(generated.instantiate(c.hypothesis, [p]), wrapper(p))
        # and a new table entry the generator would have produced
        self.assertEqual(generated.instantiate(c.hypothesis, [0x80200004]), wrapper(0x80200004))

    def test_constant_offset_relation_and_holdout(self):
        members = [mover(0x10, 0x18), mover(0x20, 0x28), mover(0x34, 0x3C)]
        c = generated.clusters(functions(members))[0]
        self.assertEqual(len(c.hypothesis.params), 1)                    # off2 = off + 8
        self.assertEqual(c.hypothesis.slots[1]["source"], {"kind": "delta", "param": 0, "delta": 8})
        self.assertEqual((c.template_exact, c.holdout_exact), (3, 3))
        # a member that breaks the relation: the fit keeps it exact, the hold-out finds it
        c = generated.clusters(functions(members + [mover(0x40, 0x4C)]))[0]
        self.assertEqual(len(c.hypothesis.params), 2)
        self.assertEqual((c.template_exact, c.holdout_exact), (4, 3))

    def test_duplicates_padding_and_thresholds(self):
        same = [mover(0x10, 0x18)] * 3
        c = generated.clusters(functions(same, pad=3))[0]
        # pads: the delay-slot nop plus the three padding words
        self.assertEqual((c.kind, len(c.hypothesis.params), c.pads), ("duplicate", 0, [4, 4, 4]))
        self.assertEqual(c.holdout_exact, 3)
        self.assertEqual(generated.clusters(functions(same[:2])), [])              # too few members
        self.assertEqual(generated.clusters(functions([[JR_RA, 0]] * 5)), [])     # too short
        mixed = [mover(0x10, 0x18), mover(0x20, 0x28), [lw(V0, 0, A0), 0, lw(V0, 4, A0), JR_RA, 0]]
        self.assertEqual(generated.clusters(functions(mixed)), [])                # different shapes

    def test_large_switch(self):
        words = [I(11, A0, AT, 100), I(4, AT, ZERO, 6), 0x00044080, lui(AT, 0x8030), 0x00280821,
                 lw(T0, 0x100, AT), 0x01000008, 0]
        found = generated.switches(code(words), 0x80100000)
        self.assertEqual(found, [{"function": "0x80100000", "at": "0x80100000", "cases": 100}])
        self.assertEqual(generated.switches(code(words), 0x80100000, min_cases=128), [])


class TableTests(unittest.TestCase):
    def test_fixed_stride_table_with_generated_columns(self):
        base = 0x80200000
        records = [[0x80200100 + 0x20 * k, k, 0x3F800000] for k in range(12)]     # pointer, index, 1.0f
        data = struct.pack(">36I", *[w for r in records for w in r])
        found = generated.record_tables(data, base, image=(0x80200000, 0x80300000))
        self.assertEqual(len(found), 1)
        t = found[0]
        self.assertEqual((t["at"], t["stride"], t["records"], t["bytes"]), ("0x80200000", 12, 12, 144))
        self.assertEqual(t["columns"], ["ap", "ap", "const"])
        self.assertEqual(t["column_classes"], ["p", "i", "f"])
        self.assertTrue(t["template_exact"])
        self.assertTrue(t["holdout_exact"])

    def test_generate_rows(self):
        rules = [{"kind": "ap", "start": 10, "step": 2}, {"kind": "free"}, {"kind": "const", "value": 7}]
        self.assertEqual(generated.generate_rows(rules, [[100], [200]], 2), [10, 100, 7, 12, 200, 7])

    def test_noise_is_not_a_table(self):
        data = struct.pack(">32I", *[(i * 0x9E3779B1) & 0xFFFFFFFF for i in range(1, 33)])
        self.assertEqual(generated.record_tables(data, 0x80200000), [])


if __name__ == "__main__":
    unittest.main()
