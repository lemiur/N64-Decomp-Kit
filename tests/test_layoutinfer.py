"""n64dk.match.layoutinfer: array and structure-field evidence from hand-assembled MIPS words.

Stdlib only: no decoder library, no toolchain, no ROM.  Facts of other functions are duck-typed
namespaces shaped like ``n64dk.analysis.values`` facts.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from n64dk.match import layoutinfer as li

ZERO, V0, V1, A0, A1, T0, T1, T2, T3, T4, T6, T7, T8, S0, SP, RA = \
    0, 2, 3, 4, 5, 8, 9, 10, 11, 12, 14, 15, 24, 16, 29, 31
VRAM = 0x80001000


def i_type(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def r_type(rs, rt, rd, sa, funct):
    return (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | funct


def lw(rt, off, base): return i_type(35, base, rt, off)
def lh(rt, off, base): return i_type(33, base, rt, off)
def lbu(rt, off, base): return i_type(36, base, rt, off)
def sw(rt, off, base): return i_type(43, base, rt, off)
def addiu(rt, rs, imm): return i_type(9, rs, rt, imm)
def addu(rd, rs, rt): return r_type(rs, rt, rd, 0, 0x21)
def sll(rd, rt, sa): return r_type(0, rt, rd, sa, 0)
def multu(rs, rt): return r_type(rs, rt, 0, 0, 0x19)
def mflo(rd): return r_type(0, 0, rd, 0, 0x12)
def bne(rs, rt, at, to): return i_type(5, rs, rt, to - (at + 1))      # instruction indexes
def jal(target): return (3 << 26) | ((target >> 2) & 0x3FFFFFF)


JR_RA, NOP = r_type(RA, 0, 0, 0, 8), 0


def code(words):
    return b"".join(w.to_bytes(4, "big") for w in words)


def propose(words, **kwargs):
    return li.propose("f", code(words), VRAM, **kwargs)


def by_label(variants):
    return {v.label: v for v in variants}


class ScanTest(unittest.TestCase):
    def test_fields_off_one_argument_register(self):
        variants, evidence = propose([lw(V0, 0, A0), lh(V1, 4, A0), lbu(T0, 6, A0), addu(V0, V0, V1),
                                      JR_RA, addu(V0, V0, T0)])
        self.assertEqual(sorted(a.offset for a in evidence.objects[0].fields), [0, 4, 6])
        struct = by_label(variants)["struct"]
        self.assertIn("s32 unk_0;", struct.structs)
        self.assertIn("s16 unk_4;", struct.structs)
        self.assertIn("u8 unk_6;", struct.structs)
        self.assertEqual(struct.prototype, "s32 f(struct LI_f_a0 *);")
        self.assertNotIn("array", by_label(variants))

    def test_pointer_stepped_in_a_loop_is_an_array_of_its_stride(self):
        words = [addu(T0, A0, ZERO),            # 0: move t0, a0
                 lw(T1, 4, T0),                 # 1: loop: t1 = t0->4
                 addu(V0, V0, T1),              # 2
                 bne(T0, A1, 3, 1),             # 3: back to 1
                 addiu(T0, T0, 12),             # 4: delay slot: t0 += 12
                 JR_RA, NOP]
        variants, evidence = propose(words)
        self.assertEqual(evidence.loops, 1)
        strided = evidence.objects[0].strided
        self.assertEqual([(s.stride, s.element_offset, s.how) for s in strided], [(12, 4, "loop")])
        array = by_label(variants)["array"]
        self.assertIn("struct LI_f_a0_arr_eC {", array.structs)
        self.assertIn("char pad_0[0x4];", array.structs)
        self.assertIn("s32 unk_4;", array.structs)
        self.assertIn("char pad_8[0x4];", array.structs)          # the element is exactly 12 bytes
        self.assertIn("struct LI_f_a0_arr_eC *", array.prototype)
        self.assertNotIn("struct", by_label(variants))            # an array from 0 is not a structure

    def test_shifted_index_gives_the_stride(self):
        _variants, evidence = propose([sll(T6, A1, 3), addu(T7, A0, T6), lh(V0, 2, T7), JR_RA, NOP])
        self.assertEqual([(s.stride, s.element_offset, s.width) for s in evidence.objects[0].strided],
                         [(8, 2, 2)])
        self.assertNotIn(1, evidence.objects)                     # the index is not an object

    def test_multiplied_index_gives_the_stride(self):
        _variants, evidence = propose([addiu(T6, ZERO, 12), multu(A1, T6), mflo(T7), addu(T8, A0, T7),
                                       lw(V0, 8, T8), JR_RA, NOP])
        self.assertEqual([(s.stride, s.element_offset) for s in evidence.objects[0].strided], [(12, 8)])

    def test_shift_and_add_multiplication(self):
        # i * 12 == ((i << 1) + i) << 2
        _variants, evidence = propose([sll(T6, A1, 1), addu(T6, T6, A1), sll(T6, T6, 2), addu(T7, A0, T6),
                                       lw(V0, 0, T7), JR_RA, NOP])
        self.assertEqual([s.stride for s in evidence.objects[0].strided], [12])

    def test_loop_counter_scaled_inside_the_loop(self):
        words = [addu(T0, ZERO, ZERO),          # 0: i = 0
                 sll(T1, T0, 3),                # 1: loop
                 addu(T2, A0, T1),              # 2
                 sw(ZERO, 4, T2),               # 3: p[i].unk_4 = 0
                 addiu(T0, T0, 1),              # 4
                 bne(T0, A1, 5, 1),             # 5
                 NOP, JR_RA, NOP]
        variants, evidence = propose(words)
        self.assertEqual([(s.stride, s.element_offset, s.store) for s in evidence.objects[0].strided],
                         [(8, 4, True)])
        self.assertIn("s32 unk_4;", by_label(variants)["array"].structs)
        self.assertTrue(by_label(variants)["array"].prototype.startswith("void f("))

    def test_two_arguments_added_are_not_an_index(self):
        variants, evidence = propose([addu(T0, A0, A1), lw(V0, 0, T0), JR_RA, NOP])
        self.assertEqual(evidence.objects, {})
        self.assertEqual(variants, [])

    def test_call_delay_slot_and_saved_register(self):
        words = [addu(S0, A0, ZERO), lw(V0, 0, S0), lh(V1, 4, S0),
                 jal(0x80002000), addiu(A0, S0, 0x10),      # the argument is set in the delay slot
                 lw(T0, 8, S0),                             # s0 survives the call
                 JR_RA, NOP]
        _variants, evidence = propose(words)
        obj = evidence.objects[0]
        self.assertEqual(obj.passed, [(0x80002000, 0, 0x10, 0)])
        self.assertEqual(sorted(a.offset for a in obj.fields), [0, 4, 8])

    def test_local_jump_is_not_a_call(self):
        words = [lw(V0, 0, A0), (2 << 26) | (((VRAM + 16) >> 2) & 0x3FFFFFF), NOP, NOP, lh(V1, 4, A0),
                 JR_RA, NOP]
        _variants, evidence = propose(words)
        self.assertEqual(evidence.objects[0].passed, [])
        self.assertEqual(sorted(a.offset for a in evidence.objects[0].fields), [0, 4])


class VariantTest(unittest.TestCase):
    def test_unrolled_loop_offers_the_smaller_element(self):
        words = [addu(T0, A0, ZERO), lw(T1, 0, T0), lw(T2, 4, T0), lw(T3, 8, T0), lw(T4, 12, T0),
                 addiu(T0, T0, 16), bne(T0, A1, 6, 1), NOP, JR_RA, NOP]
        variants, _evidence = propose(words)
        found = by_label(variants)
        self.assertIn("s32 unk_C;", found["array"].structs)
        self.assertEqual(found["alt"].prototype, "void f(s32 *);")

    def test_two_widths_at_one_offset_offer_both(self):
        words = [lh(V0, 4, A0), lw(V1, 4, A0), lw(T0, 4, A0), sw(ZERO, 0, A0), JR_RA, NOP]
        variants, _evidence = propose(words)
        found = by_label(variants)
        self.assertIn("s32 unk_4;", found["struct"].structs)
        self.assertIn("s16 unk_4;", found["alt"].structs)

    def test_misaligned_offsets_are_not_fields(self):
        variants, _evidence = propose([lw(V0, 2, A0), lw(V1, 8, A0), lh(T0, 12, A0), JR_RA, NOP])
        struct = by_label(variants)["struct"].structs
        self.assertNotIn("unk_2;", struct)
        self.assertIn("s32 unk_8;", struct)
        self.assertIn("s16 unk_C;", struct)

    def test_cross_function_fields_from_callee_and_caller(self):
        callee, caller = 0x80002000, 0x80003000
        facts = {
            callee: NS(accesses=[NS(base=("param", 0), offset=2, width=2, signed=False, floating=False,
                                    store=False, site=callee + 4)], calls=[]),
            caller: NS(accesses=[NS(base=("param", 0), offset=0x20, width=4, signed=True, floating=False,
                                    store=False, site=caller + 8)],
                       calls=[NS(target=VRAM, args={4: (("param", 0), 8)})]),
        }
        words = [lw(V0, 0, A0), lh(V1, 4, A0), jal(callee), addiu(A0, A0, 0x10), JR_RA, NOP]
        variants, _evidence = propose(words, facts=facts)
        found = by_label(variants)
        self.assertNotIn("unk_12", found["struct"].structs)
        cross = found["cross"].structs
        self.assertIn("u16 unk_12;", cross)                        # callee's 2 + passed 0x10
        self.assertIn("s32 unk_18;", cross)                        # caller's 0x20 - passed 8
        self.assertIn("s16 unk_4;", cross)

    def test_cap_and_order(self):
        words = [addu(T0, A0, ZERO), lw(T1, 0, T0), lw(T2, 4, T0), lw(T3, 8, T0), lw(T4, 12, T0),
                 addiu(T0, T0, 16), bne(T0, A1, 6, 1), NOP, JR_RA, NOP]
        self.assertEqual([v.label for v in propose(words, limit=1)[0]], ["array"])
        self.assertLessEqual(len(propose(words)[0]), li.MAX_VARIANTS)

    def test_large_functions_are_left_alone(self):
        words = [lw(V0, 0, A0), lh(V1, 4, A0)] * 0x81 + [JR_RA, NOP]
        self.assertGreater(4 * len(words), li.MAX_SIZE)
        self.assertEqual(propose(words)[0], [])

    def test_recovered_prototype_keeps_return_and_float_arguments(self):
        typing = li.Typing("struct X *", [])
        self.assertEqual(li.prototype("f", {0: typing, 1: typing}, "u8 f(void *, f32);", True),
                         "u8 f(struct X *, f32);")
        self.assertEqual(li.prototype("f", {2: typing}, None, False), "void f(s32, s32, struct X *);")
        self.assertEqual(li.prototype("f", {0: typing}, "s32 g(void *);", False), "void f(struct X *);")

    def test_element_must_be_exactly_its_stride(self):
        members = [li.Member(0, 8, "f64")]
        self.assertIsNone(li.struct_text("E", members, size=12))     # a double forces 8-byte alignment
        self.assertIsNotNone(li.struct_text("E", members, size=16))


class HookTest(unittest.TestCase):
    WORDS = [lw(V0, 0, A0), lh(V1, 4, A0), lbu(T0, 6, A0), addu(V0, V0, V1), JR_RA, addu(V0, V0, T0)]

    def test_off_by_default(self):
        self.assertFalse(li.enabled({}))
        self.assertFalse(li.enabled({li.ENV: "0"}))
        self.assertTrue(li.enabled({li.ENV: "1"}))
        self.assertEqual(li.cap({li.ENV_MAX: "1"}), 1)
        self.assertEqual(li.cap({li.ENV_MAX: "9"}), li.MAX_VARIANTS)
        self.assertEqual(li.cap({}), li.MAX_VARIANTS)

    def test_draft_modes_write_contexts(self):
        with tempfile.TemporaryDirectory() as tmp:
            modes = li.draft_modes("f", code(self.WORDS), VRAM, Path(tmp) / "context", environ={})
            self.assertEqual([m[0] for m in modes], ["layout:struct"])
            label, path, declarations, extra = modes[0]
            context = path.read_text()
            self.assertIn("typedef signed short s16;", context)
            self.assertIn("s32 f(struct LI_f_a0 *);", context)
            self.assertNotIn("typedef", declarations)            # the draft's prelude has them
            self.assertIn("struct LI_f_a0 {", declarations)
            self.assertIsNone(extra)

    def test_draft_modes_use_the_focused_context(self):
        rendered = NS(prototypes={VRAM: "s32 f(void *);"},
                      focused=lambda facts, vram, include_self, typedefs:
                      ("TYPEDEFS\n" if typedefs else "") + "s32 callee(s32);\n")
        facts = {VRAM: NS(accesses=[], calls=[])}
        with tempfile.TemporaryDirectory() as tmp:
            modes = li.draft_modes("f", code(self.WORDS), VRAM, Path(tmp), rendered, facts, environ={})
            context = modes[0][1].read_text()
            self.assertTrue(context.startswith("TYPEDEFS\ns32 callee(s32);\n"))
            self.assertTrue(modes[0][2].startswith("s32 callee(s32);\n"))

    def test_nothing_to_offer_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "context"
            self.assertEqual(li.draft_modes("f", code([JR_RA, NOP]), VRAM, folder, environ={}), [])
            self.assertFalse(folder.exists())


if __name__ == "__main__":
    unittest.main()
