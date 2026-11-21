"""Cross-function type constraints (n64dk.analysis.typeflow) on hand-assembled big-endian MIPS."""
import struct
import unittest

from n64dk.analysis.typeflow import declarations, infer

BASE = 0x80001000
ZERO, AT, V0, V1, A0, A1, T6, T7, T8, T9, S0, GP, SP, RA = 0, 1, 2, 3, 4, 5, 14, 15, 24, 25, 16, 28, 29, 31
NOP = 0
JR_RA = 0x03E00008


def i_type(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def r_type(rs, rt, rd, sa, funct):
    return (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | funct


def hi(address):
    return ((address + 0x8000) >> 16) & 0xFFFF


def lo(address):
    return address & 0xFFFF


def LUI(rt, imm): return i_type(15, 0, rt, imm)
def ADDIU(rt, rs, imm): return i_type(9, rs, rt, imm)
def ANDI(rt, rs, imm): return i_type(12, rs, rt, imm)
def LB(rt, off, base): return i_type(32, base, rt, off)
def LBU(rt, off, base): return i_type(36, base, rt, off)
def LH(rt, off, base): return i_type(33, base, rt, off)
def LHU(rt, off, base): return i_type(37, base, rt, off)
def LW(rt, off, base): return i_type(35, base, rt, off)
def LWC1(ft, off, base): return i_type(49, base, ft, off)
def LDC1(ft, off, base): return i_type(53, base, ft, off)
def SH(rt, off, base): return i_type(41, base, rt, off)
def SW(rt, off, base): return i_type(43, base, rt, off)
def BEQ(rs, rt, off): return i_type(4, rs, rt, off)
def JAL(target): return (3 << 26) | ((target >> 2) & 0x3FFFFFF)
def MOVE(rd, rs): return r_type(rs, ZERO, rd, 0, 0x25)
def ADDU(rd, rs, rt): return r_type(rs, rt, rd, 0, 0x21)
def SLL(rd, rt, sa): return r_type(0, rt, rd, sa, 0)
def SRA(rd, rt, sa): return r_type(0, rt, rd, sa, 3)


def analyse(*functions, **options):
    """``functions``: (name, vram, words); code lives at file offset ``vram - BASE``."""
    image, catalogue = bytearray(), []
    for name, vram, words in functions:
        offset = vram - BASE
        image.extend(bytes(max(0, offset + 4 * len(words) - len(image))))
        struct.pack_into(f">{len(words)}I", image, offset, *words)
        catalogue.append((name, vram, 4 * len(words)))
    return infer(bytes(image), BASE, catalogue, **options)


def frame_call(target, *before):
    """A non-leaf wrapper: prologue, ``before`` (last word fills the delay slot), jal, epilogue."""
    *setup, delay = before or (NOP,)
    return [ADDIU(SP, SP, -0x18), SW(RA, 0x14, SP), *setup, JAL(target), delay,
            LW(RA, 0x14, SP), JR_RA, ADDIU(SP, SP, 0x18)]


class WidthAndSignednessTest(unittest.TestCase):
    S16, U8, F32, F64, STORED = 0x80109000, 0x80109010, 0x80109020, 0x80109028, 0x80109030

    def report(self, **options):
        words = [LUI(AT, hi(self.S16)), LH(V0, lo(self.S16), AT),
                 LUI(AT, hi(self.U8)), LBU(V1, lo(self.U8), AT),
                 LUI(AT, hi(self.F32)), LWC1(4, lo(self.F32), AT),
                 LUI(AT, hi(self.F64)), LDC1(6, lo(self.F64), AT),
                 LUI(AT, hi(self.STORED)), SH(ZERO, lo(self.STORED), AT),
                 JR_RA, NOP]
        return analyse(("reader", BASE, words), **options)

    def test_opcode_fixes_width_signedness_and_float(self):
        found = self.report().globals
        self.assertEqual((found[self.S16].type, found[self.S16].width, found[self.S16].signed), ("s16", 2, True))
        self.assertEqual((found[self.U8].type, found[self.U8].signed), ("u8", False))
        self.assertEqual(found[self.F32].type, "f32")
        self.assertEqual(found[self.F64].type, "f64")
        self.assertEqual((found[self.STORED].width, found[self.STORED].signed), (2, None))   # a store proves no sign
        self.assertFalse(any(info.conflicts for info in found.values()))

    def test_every_constraint_records_its_instruction(self):
        evidence = self.report().globals[self.S16]["evidence"]
        self.assertEqual(len(evidence), 1)
        self.assertEqual((evidence[0].function, evidence[0].site, evidence[0].op), ("reader", BASE + 4, "lh"))
        self.assertEqual(evidence[0].via, ())

    def test_declarations_use_names_and_flag_unproven_signedness(self):
        text = declarations(self.report(names={self.S16: "gTimer"}))
        self.assertIn("extern s16 gTimer;", text)
        self.assertIn("extern u8 D_80109010;", text)
        self.assertIn("extern f32 D_80109020;", text)
        self.assertIn("extern f64 D_80109028;", text)
        self.assertIn("extern u16 D_80109030;   /* signedness unproven */", text)

    def test_gp_relative_access_needs_the_gp_value(self):
        words = [LW(V0, -0x7FF0, GP), JR_RA, NOP]
        self.assertEqual(analyse(("small", BASE, words)).globals, {})
        found = analyse(("small", BASE, words), gp=0x80108000).globals
        self.assertEqual(found[0x80100010].type, "s32")


class StructureTest(unittest.TestCase):
    OBJECT = 0x80100100

    def test_offsets_through_a_global_base_form_a_struct(self):
        g = self.OBJECT
        words = [LUI(T6, hi(g)), ADDIU(T6, T6, lo(g)), LW(T7, 0, T6), LH(T8, 4, T6), LBU(T9, 6, T6),
                 SW(ZERO, 8, T6), JR_RA, NOP]
        report = analyse(("touch", BASE, words))
        info = report.globals[g]
        self.assertEqual(info.type, "struct D_80100100_s")
        self.assertEqual({o: f.type for o, f in info.fields.items()}, {0: "s32", 4: "s16", 6: "u8", 8: "s32"})
        self.assertEqual(info.size, 12)
        text = declarations(report)
        self.assertIn("struct D_80100100_s {", text)
        self.assertIn("/* 0x04 */ s16 unk_4;", text)
        self.assertIn("u8 pad_7[0x1];", text)
        self.assertIn("extern struct D_80100100_s D_80100100;", text)

    def test_argument_dereferences_describe_the_pointee(self):
        report = analyse(("getter", BASE, [LH(V0, 0x10, A0), JR_RA, LBU(V1, 0x12, A0)]))
        arg = report.arguments[(BASE, 4)]
        self.assertTrue(arg.pointer)
        self.assertEqual(arg.type, "struct getter_arg0_s *")
        self.assertEqual({o: f.type for o, f in arg.fields.items()}, {0x10: "s16", 0x12: "u8"})

    def test_register_index_makes_an_array_and_function_addresses_are_not_data(self):
        table = 0x80100600
        words = [LUI(AT, hi(table)), ADDU(AT, AT, T6), LH(V0, lo(table), AT),
                 LUI(A0, hi(BASE)), ADDIU(A0, A0, lo(BASE)), JR_RA, NOP]
        report = analyse(("index", BASE, words))
        self.assertEqual(set(report.globals), {table})
        self.assertEqual(report.globals[table].type, "s16[]")
        self.assertIn("extern s16 D_80100600[];", declarations(report))

    def test_a_stored_global_address_makes_a_pointer(self):
        target, holder = 0x80100700, 0x80100710
        words = [LUI(T6, hi(target)), ADDIU(T6, T6, lo(target)), LUI(AT, hi(holder)), SW(T6, lo(holder), AT),
                 JR_RA, NOP]
        report = analyse(("link", BASE, words))
        self.assertEqual(report.globals[holder].type, "void *")
        self.assertIn("extern void *D_80100710;", declarations(report))


class PropagationTest(unittest.TestCase):
    CALLER, CALLEE, CALLER2 = BASE, BASE + 0x100, BASE + 0x200
    OBJECT, POINTER = 0x80100200, 0x80100400

    def report(self):
        callee = [LW(V0, 4, A0), LH(V1, 8, A0), JR_RA, NOP]
        caller = frame_call(self.CALLEE, LUI(A0, hi(self.OBJECT)), ADDIU(A0, A0, lo(self.OBJECT)))
        caller2 = frame_call(self.CALLEE, LUI(T6, hi(self.POINTER)), LW(A0, lo(self.POINTER), T6))
        return analyse(("caller", self.CALLER, caller), ("callee", self.CALLEE, callee),
                       ("caller2", self.CALLER2, caller2))

    def test_callee_fields_land_on_the_global_the_caller_passes(self):
        report = self.report()
        info = report.globals[self.OBJECT]
        self.assertEqual(info.type, "struct D_80100200_s")
        self.assertEqual({o: f.type for o, f in info.fields.items()}, {4: "s32", 8: "s16"})
        evidence = info.fields[8].evidence[0]
        self.assertEqual((evidence.function, evidence.op), ("callee", "lh"))
        self.assertEqual(evidence.site, self.CALLEE + 4)
        # the jal is the fourth word of the caller (prologue, prologue, lui, jal)
        self.assertEqual(evidence.via, ("caller@0x8000100C passes &D_80100200 as $a0 to callee",))
        self.assertIn(" <- caller@0x8000100C", str(evidence))

    def test_argument_report_keeps_both_sides(self):
        arg = self.report().arguments[(self.CALLEE, 4)]
        self.assertTrue(arg.pointer)
        self.assertEqual(sorted(arg.fields), [4, 8])
        self.assertIn("caller", {e.function for e in arg.evidence})      # the address passed
        self.assertIn("callee", {e.function for e in arg.evidence})      # the dereferences

    def test_a_pointer_global_receives_the_pointee_layout(self):
        report = self.report()
        info = report.globals[self.POINTER]
        self.assertTrue(info.pointer)
        self.assertEqual(info.type, "struct D_80100400_0_s *")
        self.assertEqual({o: f.type for o, f in info.fields.items()}, {4: "s32", 8: "s16"})
        text = declarations(report)
        self.assertIn("struct D_80100400_0_s {", text)
        self.assertIn("u8 pad_0[0x4];", text)
        self.assertIn("extern struct D_80100400_0_s *D_80100400;", text)

    def test_callers_do_not_exchange_fields(self):
        report = self.report()
        self.assertNotIn(0, report.globals[self.OBJECT].fields)     # caller2's lw of its pointer stays its own

    def test_fixed_point_through_a_pass_through_function(self):
        top, mid, leaf, obj = BASE, BASE + 0x100, BASE + 0x200, 0x80100300
        report = analyse(
            ("top", top, frame_call(mid, LUI(A0, hi(obj)), ADDIU(A0, A0, lo(obj)), ADDIU(A0, A0, 0x10))),
            ("mid", mid, frame_call(leaf)),
            ("leaf", leaf, [LHU(V0, 4, A0), JR_RA, NOP]))
        field = report.globals[obj].fields[0x14]
        self.assertEqual((field.type, field.signed), ("u16", False))
        evidence = field.evidence[0]
        self.assertEqual(evidence.function, "leaf")
        self.assertEqual(len(evidence.via), 2)
        self.assertTrue(evidence.via[0].startswith("mid@"))
        self.assertTrue(evidence.via[1].startswith("top@"))
        self.assertIn("&D_80100300+0x10", evidence.via[1])
        self.assertTrue(report.arguments[(mid, 4)].pointer)                  # pointer use flowed back

    def test_recursion_terminates(self):
        rec = analyse(("rec", BASE, [LW(V0, 0, A0), JAL(BASE), ADDIU(A0, A0, 4), JR_RA, NOP]))
        self.assertTrue(any("recursive" in note for note in rec.notes))
        f, g = BASE, BASE + 0x100
        mutual = analyse(("f", f, [LW(V0, 0, A0), JAL(g), ADDIU(A0, A0, 4), JR_RA, NOP]),
                         ("g", g, [JAL(f), ADDIU(A0, A0, 4), JR_RA, NOP]))
        self.assertGreater(mutual.truncated, 0)
        self.assertEqual(mutual.globals, {})


class ConflictTest(unittest.TestCase):
    WIDE, SIGN, CLEAN = 0x80100800, 0x80100810, 0x80100820

    def test_conflicts_are_kept_with_both_sides(self):
        report = analyse(
            ("f1", BASE, [LUI(AT, hi(self.WIDE)), LB(V0, lo(self.WIDE), AT), JR_RA, NOP]),
            ("f2", BASE + 0x100, [LUI(AT, hi(self.WIDE)), LW(V0, lo(self.WIDE), AT), JR_RA, NOP]),
            ("f3", BASE + 0x200, [LUI(AT, hi(self.SIGN)), LBU(V0, lo(self.SIGN), AT),
                                  LB(V1, lo(self.SIGN), AT), LH(T6, lo(self.CLEAN), AT), JR_RA, NOP]))
        wide = report.globals[self.WIDE]
        self.assertEqual([c.kind for c in wide.conflicts], ["width"])
        self.assertEqual({e.function for e in wide.conflicts[0].evidence}, {"f1", "f2"})
        self.assertEqual(wide.type, "u8|s32")
        sign = report.globals[self.SIGN]
        self.assertEqual([c.kind for c in sign.conflicts], ["signedness"])
        self.assertIsNone(sign.signed)
        text = declarations(report)
        self.assertNotIn("D_80100800", text)
        self.assertNotIn("D_80100810", text)
        self.assertIn("extern s16 D_80100820;", text)

    def test_a_conflict_through_a_call_names_both_functions(self):
        g, caller, callee = 0x80100900, BASE, BASE + 0x100
        words = [LUI(AT, hi(g)), LW(T6, lo(g), AT), LUI(A0, hi(g)), JAL(callee), ADDIU(A0, A0, lo(g)), JR_RA, NOP]
        report = analyse(("caller", caller, words), ("callee", callee, [LH(V0, 0, A0), JR_RA, NOP]))
        conflict = report.globals[g].conflicts[0]
        self.assertEqual(conflict.kind, "width")
        by_function = {e.function: e for e in conflict.evidence}
        self.assertEqual(set(by_function), {"caller", "callee"})
        self.assertEqual(by_function["caller"].via, ())
        self.assertEqual(len(by_function["callee"].via), 1)
        self.assertNotIn("D_80100900", declarations(report))

    def test_pointer_used_as_a_byte_is_a_conflict(self):
        g = 0x80100A00
        words = [LUI(AT, hi(g)), LW(T6, lo(g), AT), LH(V0, 2, T6), LBU(V1, lo(g), AT), JR_RA, NOP]
        info = analyse(("mixed", BASE, words)).globals[g]
        self.assertIn("pointer", [c.kind for c in info.conflicts])


class ArgumentNarrowingTest(unittest.TestCase):
    def test_masks_and_shift_pairs_type_parameters(self):
        words = [ANDI(A0, A0, 0xFF), SLL(A1, A1, 16), SRA(A1, A1, 16), JR_RA, NOP]
        report = analyse(("narrow", BASE, words))
        self.assertEqual(report.arguments[(BASE, 4)].type, "u8")
        self.assertEqual(report.arguments[(BASE, 5)].type, "s16")
        self.assertEqual(report.arguments[(BASE, 5)].evidence[0].op, "sll+sra")


class ControlFlowTest(unittest.TestCase):
    def test_a_register_with_two_definitions_is_dropped_at_a_join(self):
        first, second = 0x80100500, 0x80200500
        words = [LUI(T6, hi(first)), BEQ(A1, ZERO, 2), NOP, LUI(T6, hi(second)),
                 LW(V0, lo(first), T6), JR_RA, NOP]
        self.assertEqual(analyse(("join", BASE, words)).globals, {})

    def test_a_register_written_once_survives_a_join(self):
        words = [MOVE(S0, A0), BEQ(A1, ZERO, 2), NOP, NOP, LH(V0, 4, S0), JR_RA, NOP]
        arg = analyse(("saved", BASE, words)).arguments[(BASE, 4)]
        self.assertEqual(arg.fields[4].type, "s16")

    def test_a_spilled_argument_survives_a_join(self):
        words = [ADDIU(SP, SP, -0x20), SW(A0, 0x20, SP), BEQ(A1, ZERO, 2), NOP, NOP,
                 LW(T6, 0x20, SP), LBU(V0, 3, T6), JR_RA, ADDIU(SP, SP, 0x20)]
        arg = analyse(("spill", BASE, words)).arguments[(BASE, 4)]
        self.assertTrue(arg.pointer)
        self.assertEqual(arg.fields[3].type, "u8")

    def test_functions_outside_the_image_are_noted(self):
        report = infer(b"\0" * 8, BASE, [("far", BASE + 0x100, 8)])
        self.assertEqual(report.globals, {})
        self.assertTrue(any("far" in note for note in report.notes))


if __name__ == "__main__":
    unittest.main()
