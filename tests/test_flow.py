import struct
import unittest

from n64dk.analysis import flow

BASE = 0x80001000


def code(words):
    return b"".join(struct.pack(">I", w) for w in words)


# addiu sp,-0x20; sw ra; sw s0; move s0,a0; jal; addiu a0,s0,4; beqz v0,+2; nop; lw v0,0(s0);
# lw ra; lw s0; jr ra; addiu sp,+0x20; padding
FULL = [0x27BDFFE0, 0xAFBF001C, 0xAFB00018, 0x00808025, 0x0C000800, 0x26040004, 0x10400002, 0x00000000,
        0x8E020000, 0x8FBF001C, 0x8FB00018, 0x03E00008, 0x27BD0020, 0x00000000, 0x00000000, 0x00000000]
# lhu v0,0x60(s0); addu v1,v0,t0; sh v1,0x24(s0); lw ra,0x1c(sp); jr ra; addiu sp,+0x20
FRAGMENT = [0x96020060, 0x00481821, 0xA6030024, 0x8FBF001C, 0x03E00008, 0x27BD0020]
# beql a0,zero,L; addiu t0,zero,5 (annulled); addiu t0,zero,7; L: jr ra; addu v0,t0,zero
LIKELY = [0x50800002, 0x24080005, 0x24080007, 0x03E00008, 0x01001021]
# move v0,zero; L: lw a0,4(a0); bnez a0,L; addiu v0,v0,1; jr ra; nop
LOOP = [0x00001025, 0x8C840004, 0x1480FFFE, 0x24420001, 0x03E00008, 0x00000000]
# addiu sp,-0x18; sw ra; bnez a0,+0x40 (outside); nop; jal; nop   -- runs off the end
SPLIT = [0x27BDFFE8, 0xAFBF0014, 0x1480000F, 0x00000000, 0x0C000800, 0x00000000]
# sltiu at,a0,3; beqz at,default; sll t0,a0,2; lui at; addu at,at,t0; lw t0,0x100(at); jr t0; nop;
# case0: jr ra; li v0,1  case1: jr ra; li v0,2  default: jr ra; move v0,zero
SWITCH = [0x2C810003, 0x1020000A, 0x00044080, 0x3C018001, 0x00280821, 0x8C280100, 0x01000008, 0x00000000,
          0x03E00008, 0x24020001, 0x03E00008, 0x24020002, 0x03E00008, 0x00001025]
GETTER = [0x03E00008, 0x8C820008]
# lw t0,0(a0); nop; addu v0,t0,a1; jr ra; nop
MIPS1 = [0x8C880000, 0x00000000, 0x01051021, 0x03E00008, 0x00000000]


class FlowTest(unittest.TestCase):
    def test_complete_function(self):
        f = flow.analyse(code(FULL), BASE)
        evidence = flow.entry_evidence(f)
        self.assertTrue(evidence["complete"], evidence)
        self.assertEqual((f.frame.size, sorted(f.frame.saved), sorted(f.frame.restored)), (0x20, [16, 31], [16, 31]))
        self.assertEqual(f.calls, [(BASE + 0x10, 0x80002000)])
        self.assertEqual(f.live_in, frozenset({4, 16, 29, 31}))       # a0, plus s0/ra read only to save them
        self.assertEqual(evidence["arguments_read"], 1)
        self.assertEqual([kind for _s, kind, _t in f.exits], ["return"])

    def test_fragment_reads_caller_context(self):
        evidence = flow.entry_evidence(flow.analyse(code(FRAGMENT), BASE))
        self.assertFalse(evidence["complete"])
        text = " ".join(evidence["problems"])
        self.assertIn("$t0", text)                 # caller temporary
        self.assertIn("$s0", text)                 # saved register used without a save
        self.assertIn("never saved", text)         # $ra restored but not saved here
        self.assertIn("without allocation", text)

    def test_branch_likely_slot_only_defines_on_the_taken_edge(self):
        f = flow.analyse(code(LIKELY), BASE)
        self.assertNotIn(8, f.live_in)             # t0 is written on both paths before the read
        self.assertTrue(flow.entry_evidence(f)["complete"])

    def test_loop_and_shape(self):
        f = flow.analyse(code(LOOP), BASE)
        self.assertEqual(len(f.loops), 1)
        self.assertEqual(max(b.loop_depth for b in f.blocks), 1)
        self.assertIn("linked-list-walk", [s["shape"] for s in flow.shapes(f)])

    def test_branch_outside_and_fall_off_end(self):
        evidence = flow.entry_evidence(flow.analyse(code(SPLIT), BASE))
        self.assertFalse(evidence["complete"])
        self.assertEqual(evidence["exits"], {"branch-outside": 1, "falls-off-end": 1})

    def test_trailing_padding_is_not_a_fall_through(self):
        f = flow.analyse(code(GETTER + [0, 0]), BASE)
        self.assertEqual([kind for _s, kind, _t in f.exits], ["return"])
        self.assertEqual([role for _a, role in flow.nop_roles(f)], ["padding", "padding"])
        self.assertEqual(flow.shapes(f)[0]["shape"], "getter")

    def test_switch_needs_its_jump_table(self):
        blind = flow.analyse(code(SWITCH), BASE)
        self.assertIn("computed-jump", [kind for _s, kind, _t in blind.exits])
        table = {BASE + 0x18: [BASE + 0x20, BASE + 0x28, BASE + 0x30]}
        seen = flow.analyse(code(SWITCH), BASE, table)
        self.assertEqual(sorted(kind for _s, kind, _t in seen.exits), ["return"] * 3)
        self.assertEqual(len(seen.reachable), len(seen.blocks))
        self.assertEqual(flow.shapes(seen)[0]["shape"], "switch-dispatcher")

    def test_nop_roles(self):
        roles = dict(flow.nop_roles(flow.analyse(code(FULL), BASE)))
        self.assertEqual(roles[BASE + 0x1C], "delay-slot:branch")
        self.assertEqual(roles[BASE + 0x34], "padding")
        roles = [role for _a, role in flow.nop_roles(flow.analyse(code(MIPS1), BASE))]
        self.assertEqual(roles, ["load-delay", "delay-slot:return"])


class DadduJumpTableTest(unittest.TestCase):
    """EGCS -O2 -mips3 dispatches through ``daddu``: the table must be found as for ``addu``."""

    def table_function(self, add_funct: int):
        base = 0x80001000
        # sll v0,a0,2 ; lui v1,%hi(T) ; (d)addu v0,v0,v1 ; lw v0,%lo(T)(v0) ; jr v0 ; nop ; 4 cases: jr ra ; nop
        table = 0x80002000
        hi, lo = (table + 0x8000) >> 16, table & 0xFFFF
        words = [0x00041080, 0x3C030000 | hi, 0x00431000 | add_funct, 0x8C420000 | lo, 0x00400008, 0]
        cases = [base + 4 * len(words) + 8 * k for k in range(4)]
        words += [0x03E00008, 0] * 4
        memory = {table + 4 * k: target for k, target in enumerate(cases)}
        return code(words), base, memory.get, cases

    def test_daddu_dispatch_is_recognised_like_addu(self):
        found = {}
        for funct in (0x21, 0x2D):
            blob, base, read, cases = self.table_function(funct)
            found[funct] = flow.find_jump_tables(blob, base, read)
            self.assertEqual(found[funct], {base + 16: cases})
        self.assertEqual(found[0x21], found[0x2D])

    def test_daddu_of_two_runtime_registers_is_not_a_table(self):
        # daddu v0,a0,a1 ; lw v0,0(v0) ; jr v0 : no lui, so no table
        blob = code([0x0085102D, 0x8C420000, 0x00400008, 0])
        self.assertEqual(flow.find_jump_tables(blob, 0x80001000, lambda a: 0x80001000), {})


if __name__ == "__main__":
    unittest.main()
