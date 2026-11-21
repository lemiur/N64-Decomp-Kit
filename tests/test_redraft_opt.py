"""Draft generators for optimised builds: a decompiler-style draft of a known source must come
back, through one of the generator's rewrites, to the bytes the source compiles to.

The compilers are needed for the round trips (skipped without ``N64DK_TOOLCHAIN``); the
declining cases need nothing."""
from __future__ import annotations

import struct
import unittest

from n64dk.match import autodraft, redraft, redraft_elements, redraft_gcc_opt, redraft_ido_opt
from tests.test_redraft_debug import VRAM, _compiler

GCC = ("kmc+as", ["-O2", "-G0", "-mips3", "-mgp32", "-mfp32"], "kmc")
IDO = ("ido5.3", ["-O2", "-mips2", "-non_shared", "-G", "0", "-Xcpluscomm", "-Wab,-r4300_mul"], "ido53")

# (what the programmer wrote, what a decompiler renders): the draft differs from the truth in
# exactly the way the rule under test repairs
CASES = {
    "bytes": ("extern s32 D_80100000[];\nvoid f(s32 i) {\n    D_80100000[i * 2] = 0;\n}\n",
              "extern M2C_UNK D_80100000;\nvoid f(s32 arg0) {\n    *(&D_80100000 + (arg0 * 8)) = 0;\n}\n"),
    "passed": ("void func_80002000(s32, s32);\nvoid f(s32 a) {\n    func_80002000(a, 0);\n}\n",
               "M2C_UNK func_80002000();\nvoid f(void) {\n    func_80002000(0);\n}\n"),
}


def words(*values: int) -> bytes:
    return b"".join(struct.pack(">I", value) for value in values)


class DecliningTest(unittest.TestCase):
    """Nothing here may raise, and code that is not the generator's situation gets no rewrite."""

    DRAFT = autodraft.PRELUDE + "\nvoid f(void) {\n}\n"
    # addiu sp,-24; sw ra; sw fp; addu fp,sp,zero ...: an unoptimised GCC frame
    DEBUG_GCC = words(0x27BDFFE8, 0xAFBF0014, 0xAFBE0010, 0x03A0F021, 0x03C0E821, 0x8FBF0014, 0x8FBE0010,
                      0x03E00008, 0x27BD0018)

    def test_empty_ragged_and_garbage_input(self):
        for module in (redraft_gcc_opt, redraft_ido_opt):
            for code in (b"", b"\x00\x01\x02", bytes(64), b"\xff" * 32):
                rows = module.variants(self.DRAFT, "f", code=code, vram=VRAM, pipeline="kmc+as", flags=("-O2",))
                self.assertEqual(rows, [], module.__name__)
            self.assertEqual(module.variants("not C at all {{{", "f", code=self.DEBUG_GCC, vram=VRAM,
                                             pipeline="kmc+as", flags=("-O2",)), [])

    def test_the_gcc_generator_leaves_ido_and_unoptimised_code_alone(self):
        rows = redraft_gcc_opt.variants(self.DRAFT, "f", code=self.DEBUG_GCC, vram=VRAM, pipeline="kmc+as",
                                        flags=("-O0",))
        self.assertEqual(rows, [])
        rows = redraft_gcc_opt.variants(self.DRAFT, "f", code=words(0x03E00008, 0x00000000), vram=VRAM,
                                        pipeline="ido5.3", flags=("-O2",), family="ido")
        self.assertEqual(rows, [])

    def test_a_rewrite_is_never_the_draft_itself_and_the_number_is_bounded(self):
        for module in (redraft_gcc_opt, redraft_ido_opt):
            rows = module.variants(self.DRAFT, "f", code=words(0x03E00008, 0x00000000), vram=VRAM,
                                   pipeline="kmc+as" if module is redraft_gcc_opt else "ido5.3", flags=("-O2",))
            self.assertLessEqual(len(rows), 6)
            self.assertNotIn(self.DRAFT, [text for _label, text in rows])


class AddressedGlobalTest(unittest.TestCase):
    """A scalar global the target reaches through its address in a register is offered as element 0 of an
    array (measured on a title; an unlinked object has no addresses, so the code is written out here)."""
    # lui a2,0x8010; addiu a2,a2,0; lw v0,0(a2); beq v0,a0,+3; nop; sw a0,0(a2); jr ra; nop
    CODE = words(0x3C068010, 0x24C60000, 0x8CC20000, 0x10440002, 0x00000000, 0xACC40000, 0x03E00008, 0x00000000)
    DRAFT = autodraft.PRELUDE + ("\nextern s32 D_80100000;\nextern s32 D_80100100;\n\nvoid f(s32 arg0) {\n"
                                 "    if (D_80100000 != arg0) {\n        D_80100000 = arg0;\n    }\n}\n")

    def rows(self, draft):
        return redraft_gcc_opt.variants(draft, "f", code=self.CODE, vram=VRAM, pipeline="kmc+as", flags=("-O2",))

    def test_the_global_is_offered_as_an_array(self):
        texts = [text for label, text in self.rows(self.DRAFT) if "addressed" in label]
        self.assertTrue(texts)
        self.assertIn("extern s32 D_80100000[];", texts[0])
        self.assertIn("D_80100000[0] != arg0", texts[0])
        self.assertIn("D_80100000[0] = arg0", texts[0])
        self.assertNotIn("extern s32 D_80100000;", texts[0])
        self.assertIn("extern s32 D_80100100;", texts[0])                 # not addressed by the target: untouched

    def test_a_global_the_draft_only_takes_the_address_of_is_left_alone(self):
        draft = self.DRAFT.replace("D_80100000 != arg0", "arg0 != 3").replace("D_80100000 = arg0;", "func_80002000(&D_80100000);")
        self.assertFalse([label for label, _text in self.rows(draft) if "addressed" in label])

    def test_a_global_read_directly_and_passed_by_address_changes_in_both_places(self):
        draft = self.DRAFT.replace("D_80100000 = arg0;", "func_80002000(&D_80100000);")
        texts = [text for label, text in self.rows(draft) if "addressed" in label]
        self.assertTrue(texts)
        body = " ".join(texts[0][texts[0].index("void f("):].split())
        self.assertIn("if (D_80100000[0] != arg0) { func_80002000(D_80100000); }", body)


class PositionalParametersTest(unittest.TestCase):
    def test_parameters_a_decompiler_left_out_are_put_back(self):
        fixed = redraft_gcc_opt.positional_parameters("void f(s32 arg2, s32 *arg3) {\n    *arg3 = arg2;\n}\n", "f")
        self.assertEqual(fixed, "void f(s32 arg0, s32 arg1, s32 arg2, s32 *arg3) {\n    *arg3 = arg2;\n}\n")
        fixed = redraft_gcc_opt.positional_parameters("s32 f(s32 arg0, s32 arg5) {\n    return arg5;\n}\n", "f")
        self.assertIn("(s32 arg0, s32 arg1, s32 arg2, s32 arg3, s32 arg4, s32 arg5)", fixed)

    def test_complete_named_or_empty_lists_are_left_alone(self):
        for text in ("void f(s32 arg0, s32 arg1) {\n}\n", "void f(void) {\n}\n", "void f(s32 count, s32 arg2) {\n}\n"):
            self.assertIsNone(redraft_gcc_opt.positional_parameters(text, "f"))

    def test_it_is_offered_first_for_optimised_gcc(self):
        # move v0,a2 ; jr ra ; nop  (optimised: no frame)
        code = words(0x00C01021, 0x03E00008, 0x00000000)
        draft = autodraft.PRELUDE + "\ns32 f(s32 arg2) {\n    return arg2;\n}\n"
        rows = redraft_gcc_opt.variants(draft, "f", code=code, vram=VRAM, pipeline="kmc+as", flags=("-O2",))
        self.assertEqual(rows[0][0], "parameters")
        self.assertIn("s32 f(s32 arg0, s32 arg1, s32 arg2)", rows[0][1])


class SignTestTest(unittest.TestCase):
    # sll v0,a0,24 ; bgez v0,+2 ; nop ; addiu v0,zero,1 ; jr ra ; nop
    CODE = words(0x00041600, 0x04410002, 0x00000000, 0x24020001, 0x03E00008, 0x00000000)
    DRAFT = autodraft.PRELUDE + "\ns32 f(s32 arg0) {\n    if (arg0 & 0x80) {\n        return 1;\n    }\n    return 0;\n}\n"

    def test_a_mask_of_the_top_bit_becomes_a_signed_comparison_where_the_target_tests_the_sign(self):
        rows = redraft_gcc_opt.variants(self.DRAFT, "f", code=self.CODE, vram=VRAM, pipeline="kmc+as", flags=("-O2",))
        texts = [text for label, text in rows if "signs" in label]
        self.assertTrue(texts)
        self.assertIn("if (((s8) arg0) < 0)", texts[0])

    def test_without_the_shift_form_the_mask_stays(self):
        code = words(0x30820080, 0x10400002, 0x00000000, 0x24020001, 0x03E00008, 0x00000000)    # andi; beqz
        rows = redraft_gcc_opt.variants(self.DRAFT, "f", code=code, vram=VRAM, pipeline="kmc+as", flags=("-O2",))
        self.assertFalse([label for label, _text in rows if "signs" in label])


class SharedStepTest(unittest.TestCase):
    CODE = words(0x00C01021, 0x03E00008, 0x00000000)                       # optimised: no frame
    DRAFT = autodraft.PRELUDE + ("\nM2C_UNK func_80002000();\ns32 f(s32 arg0, s32 arg1) {\n    s32 i;\n\n    i = 0;\n"
                                 "    do {\n        if (!(arg1 & 0xF)) {\n            i += 1;\n        } else {\n"
                                 "            func_80002000(1);\n            i += 1;\n            func_80002000(2);\n"
                                 "        }\n    } while (i < arg0);\n    return i;\n}\n")

    def rows(self, draft):
        return redraft_gcc_opt.variants(draft, "f", code=self.CODE, vram=VRAM, pipeline="kmc+as", flags=("-O2",))

    def test_a_step_copied_into_both_branches_stands_once_after_the_if(self):
        texts = [text for label, text in self.rows(self.DRAFT) if "step" in label]
        self.assertTrue(texts)
        body = " ".join(texts[0][texts[0].index("s32 f("):].split())
        self.assertIn("if (arg1 & 0xF) { func_80002000(1); func_80002000(2); } i += 1;", body)

    def test_branches_that_do_not_share_one_step_are_left_alone(self):
        draft = self.DRAFT.replace("            i += 1;\n            func_80002000(2);", "            func_80002000(2);")
        self.assertFalse([label for label, _text in self.rows(draft) if "step" in label])


class ReuseAndStrideTest(unittest.TestCase):
    # andi $v0, $v0, 0xFFFF; andi $v0, $v0, 0xFF; jr $ra; nop: the target does narrow values, so the
    # temporaries' own types stand (the ``ints`` rule would make them words otherwise)
    CODE = words(0x3042FFFF, 0x304200FF, 0x03E00008, 0x00000000)
    DRAFT = autodraft.PRELUDE + ("\nextern u8 D_80080D78[];\nextern u16 D_80080CFC[];\nM2C_UNK func_80002000();\n"
                                 "void f(s32 arg0) {\n    u8 temp_s0_5;\n    u16 temp_s0_9;\n\n"
                                 "    temp_s0_5 = *(D_80080D78 + (arg0 * 0x28));\n    func_80002000(temp_s0_5);\n"
                                 "    temp_s0_9 = D_80080CFC[arg0];\n    func_80002000(temp_s0_9);\n}\n")

    def body(self, draft):
        rows = redraft_gcc_opt.variants(draft, "f", code=self.CODE, vram=VRAM, pipeline="kmc+as", flags=("-O2",))
        texts = [text for label, text in rows if "reuse" in label or "stride" in label]
        return " ".join(texts[0].split()) if texts else ""

    def test_two_temporaries_of_one_saved_register_become_one_variable_of_the_wider_type(self):
        body = self.body(self.DRAFT)
        self.assertIn("u16 temp_s0_5;", body)
        self.assertNotIn("temp_s0_9", body)

    def test_scaled_access_to_a_scalar_array_becomes_an_array_of_structures(self):
        body = self.body(self.DRAFT)
        self.assertIn("struct n64dk_D_80080D78 { u8 v; char pad[39]; };", body)
        self.assertIn("temp_s0_5 = D_80080D78[arg0].v;", body)

    def test_an_array_also_used_another_way_keeps_its_declaration(self):
        body = self.body(self.DRAFT.replace("func_80002000(temp_s0_5);", "func_80002000(temp_s0_5 + D_80080D78[1]);"))
        self.assertNotIn("n64dk_D_80080D78", body)

    def test_temporaries_whose_uses_overlap_stay_apart(self):
        draft = self.DRAFT.replace("func_80002000(temp_s0_9);", "func_80002000(temp_s0_9 + temp_s0_5);")
        self.assertIn("temp_s0_9", self.body(draft))


class GroupedGlobalsTest(unittest.TestCase):
    # lui $s1, 0x8008; addiu $s1, $s1, 0x7FA4; jr $ra; nop: the target forms the base address
    CODE = words(0x3C118008, 0x26317FA4, 0x03E00008, 0x00000000)
    DRAFT = autodraft.PRELUDE + ("\nextern s32 D_80087FA4;\nextern s32 D_80087FB4;\nextern s16 D_80087FDC;\n"
                                 "extern s32 D_80090000;\n"
                                 "void f(void) {\n    if (D_80087FA4 != 8) {\n        D_80087FB4 = D_80090000;\n"
                                 "        D_80087FDC = 0;\n    }\n}\n")

    def body(self, draft):
        rows = redraft_gcc_opt.variants(draft, "f", code=self.CODE, vram=VRAM, pipeline="kmc+as", flags=("-O2",))
        texts = [text for label, text in rows if "grouped" in label]
        return " ".join(texts[0].split()) if texts else ""

    def test_globals_above_a_formed_address_become_members_of_one_structure(self):
        body = self.body(self.DRAFT)
        self.assertIn("struct n64dk_D_80087FA4 { s32 f_0; char pad_4[12]; s32 f_10; char pad_14[36]; s16 f_38; };", body)
        self.assertIn("extern struct n64dk_D_80087FA4 D_80087FA4;", body)
        self.assertIn("if (D_80087FA4.f_0 != 8) { D_80087FA4.f_10 = D_80090000; D_80087FA4.f_38 = 0; }", body)
        self.assertNotIn("extern s32 D_80087FB4;", body)

    def test_nothing_is_grouped_when_the_target_forms_no_such_address(self):
        code = words(0x03E00008, 0x00000000)
        rows = redraft_gcc_opt.variants(self.DRAFT, "f", code=code, vram=VRAM, pipeline="kmc+as", flags=("-O2",))
        self.assertFalse([label for label, _text in rows if "grouped" in label])

    def test_a_member_that_would_overlap_its_neighbour_leaves_the_draft_alone(self):
        draft = self.DRAFT.replace("extern s32 D_80087FB4;", "extern s32 D_80087FA6;").replace("D_80087FB4", "D_80087FA6")
        self.assertEqual(self.body(draft), "")


class InnerAndTailsTest(unittest.TestCase):
    # addiu $s1, $s2, 0x74; jr $ra; nop: a saved register pointing into the parameter's object
    CODE = words(0x26510074, 0x03E00008, 0x00000000)
    DRAFT = autodraft.PRELUDE + ("\nM2C_UNK func_80002000();\ns32 func_80003000();\n"
                                 "void f(void *arg0, s32 arg1) {\n    s32 var_v0_5;\n\n"
                                 "    M2C_FIELD(arg0, s32 *, 0x78) = 1;\n"
                                 "    if (arg1 == 2) {\n        var_v0_5 = func_80003000(1);\n    } else {\n"
                                 "        var_v0_5 = func_80003000(0);\n    }\n"
                                 "    func_80002000(var_v0_5, M2C_FIELD(arg0, s32 *, 0xB0));\n"
                                 "    M2C_FIELD(arg0, s32 *, 8) = 0;\n}\n")

    def body(self, draft, code=None):
        rows = redraft_gcc_opt.variants(draft, "f", code=code or self.CODE, vram=VRAM, pipeline="kmc+as", flags=("-O2",))
        texts = [text for label, text in rows if "inner" in label or "tails" in label]
        return " ".join(texts[0][texts[0].index("void f("):].split()) if texts else ""

    def test_members_past_the_kept_offset_go_through_a_pointer_to_it(self):
        body = self.body(self.DRAFT)
        self.assertIn("s8 *n64dk_inner_74 = ((s8 *) arg0) + 0x74;", body)
        self.assertIn("M2C_FIELD(n64dk_inner_74, s32 *, 0x4) = 1;", body)
        self.assertIn("M2C_FIELD(n64dk_inner_74, s32 *, 0x3c)", body)
        self.assertIn("M2C_FIELD(arg0, s32 *, 8) = 0;", body)

    def test_a_use_fed_by_both_branches_is_written_in_each(self):
        body = self.body(self.DRAFT)
        self.assertIn("if (arg1 == 2) { func_80002000(func_80003000(1), M2C_FIELD(n64dk_inner_74, s32 *, 0x3c)); } else { "
                      "func_80002000(func_80003000(0), M2C_FIELD(n64dk_inner_74, s32 *, 0x3c)); }", body)

    def test_a_temporary_read_again_later_is_not_folded(self):
        draft = self.DRAFT.replace("M2C_FIELD(arg0, s32 *, 8) = 0;", "M2C_FIELD(arg0, s32 *, 8) = var_v0_5;")
        self.assertIn("var_v0_5 = func_80003000(1);", self.body(draft))

    def test_without_the_kept_pointer_the_accesses_stay(self):
        body = self.body(self.DRAFT, code=words(0x03E00008, 0x00000000))
        self.assertNotIn("n64dk_inner", body)


class TailChainTest(unittest.TestCase):
    CODE = words(0x00C01021, 0x03E00008, 0x00000000)
    DRAFT = autodraft.PRELUDE + ("\nM2C_UNK func_80002000();\n"
                                 "void f(s32 arg0, s32 arg1) {\n    s32 var_a1_7;\n\n    var_a1_7 = 10;\n"
                                 "    if (arg0 != 1) {\n        if (arg1 == 2) {\n            var_a1_7 = 20;\n"
                                 "        } else if (arg1 == 3) {\n            var_a1_7 = 30;\n        }\n    }\n"
                                 "    func_80002000(0, var_a1_7);\n}\n")

    def body(self, draft):
        rows = redraft_gcc_opt.variants(draft, "f", code=self.CODE, vram=VRAM, pipeline="kmc+as", flags=("-O2",))
        texts = [text for label, text in rows if label.split("+") == ["tails"]]
        return " ".join(texts[0][texts[0].index("void f("):].split()) if texts else ""

    def test_the_use_is_written_on_every_path_as_an_else_if_chain(self):
        self.assertIn("if (arg0 == 1) { func_80002000(0, 10); } else { if (arg1 == 2) { func_80002000(0, 20); } else { "
                      "if (arg1 == 3) { func_80002000(0, 30); } else { func_80002000(0, 10); } } }", self.body(self.DRAFT))

    def test_a_temporary_tested_by_a_condition_is_left_alone(self):
        self.assertEqual(self.body(self.DRAFT.replace("arg1 == 3", "var_a1_7 == 3")), "")

    def test_a_single_setting_is_not_a_join(self):
        draft = self.DRAFT.replace("            var_a1_7 = 20;\n", "").replace("            var_a1_7 = 30;\n", "")
        self.assertEqual(self.body(draft), "")


class AccessWidthsTest(unittest.TestCase):
    # lhu $v0, 0x48($a0); sb $v0, 0($a1); lh $v1, 8($a0); lhu $v1, 8($a0); jr $ra; nop
    CODE = words(0x94820048, 0xA0A20000, 0x84830008, 0x94830008, 0x03E00008, 0x00000000)
    DRAFT = autodraft.PRELUDE + ("\nvoid f(void *arg0, void *arg1) {\n    s32 temp_v0_3;\n\n"
                                 "    temp_v0_3 = M2C_FIELD(arg0, s16 *, 0x48);\n"
                                 "    *((s32 *) arg1) = temp_v0_3;\n"
                                 "    M2C_FIELD(arg1, s32 *, 4) = M2C_FIELD(arg0, s16 *, 8);\n}\n")

    def body(self, draft):
        rows = redraft_gcc_opt.variants(draft, "f", code=self.CODE, vram=VRAM, pipeline="kmc+as", flags=("-O2",))
        texts = [text for label, text in rows if label.split("+") == ["widths"]]
        return " ".join(texts[0][texts[0].index("void f("):].split()) if texts else ""

    def test_a_read_takes_the_extension_the_target_loads_with(self):
        self.assertIn("temp_v0_3 = M2C_FIELD(arg0, u16 *, 0x48);", self.body(self.DRAFT))

    def test_a_plain_store_takes_the_width_the_target_stores(self):
        self.assertIn("*((s8 *) arg1) = temp_v0_3;", self.body(self.DRAFT))

    def test_an_offset_the_target_reads_both_ways_keeps_its_type(self):
        self.assertIn("M2C_FIELD(arg0, s16 *, 8)", self.body(self.DRAFT))

    def test_an_access_through_a_global_address_keeps_its_type(self):
        draft = self.DRAFT.replace("M2C_FIELD(arg1, s32 *, 4) = M2C_FIELD(arg0, s16 *, 8);",
                                   "M2C_FIELD(arg1, s32 *, 4) = *((s32 *) (((s8 *) (&D_80100000)) + (temp_v0_3 * 4)));"
                                   ).replace("void f(", "extern s32 D_80100000;\nvoid f(")
        self.assertIn("*((s32 *) (((s8 *) (&D_80100000)) + (temp_v0_3 * 4)))", self.body(draft))

    def test_an_offset_the_target_never_touches_keeps_its_type(self):
        self.assertIn("M2C_FIELD(arg1, s32 *, 4) =", self.body(self.DRAFT))


class SwappedReturnsTest(unittest.TestCase):
    CODE = words(0x00C01021, 0x03E00008, 0x00000000)
    DRAFT = autodraft.PRELUDE + ("\nM2C_UNK func_80002000();\ns32 f(s32 arg0) {\n    func_80002000(0);\n"
                                 "    if (arg0 >= 0) {\n        func_80002000(1);\n        return 0;\n    }\n"
                                 "    func_80002000(2);\n    return 1;\n}\n")

    def body(self, draft):
        rows = redraft_gcc_opt.variants(draft, "f", code=self.CODE, vram=VRAM, pipeline="kmc+as", flags=("-O2",))
        texts = [text for label, text in rows if label.split("+") == ["returns"]]
        return " ".join(texts[0][texts[0].index("s32 f("):].split()) if texts else ""

    def test_the_returning_block_and_the_rest_change_places(self):
        self.assertIn("func_80002000(0); if (arg0 < 0) { func_80002000(2); return 1; } func_80002000(1); return 0; }",
                      self.body(self.DRAFT))

    def test_a_block_that_does_not_return_is_left_alone(self):
        self.assertEqual(self.body(self.DRAFT.replace("        return 0;\n", "")), "")


class ConstantLastTest(unittest.TestCase):
    CODE = words(0x00C01021, 0x03E00008, 0x00000000)
    DRAFT = autodraft.PRELUDE + "\ns32 f(s32 arg0, s32 arg1) {\n    return arg0 + 0xB + (arg1 >> 5);\n}\n"

    def body(self, draft):
        rows = redraft_gcc_opt.variants(draft, "f", code=self.CODE, vram=VRAM, pipeline="kmc+as", flags=("-O2",))
        texts = [text for label, text in rows if label.split("+") == ["assoc"]]
        return " ".join(texts[0][texts[0].index("s32 f("):].split()) if texts else ""

    def test_a_constant_in_the_middle_of_a_sum_joins_the_last_term(self):
        self.assertIn("return arg0 + ((arg1 >> 5) + 0xB);", self.body(self.DRAFT))

    def test_a_sum_that_ends_in_its_constant_is_left_alone(self):
        self.assertEqual(self.body(self.DRAFT.replace("arg0 + 0xB + (arg1 >> 5)", "arg0 + (arg1 >> 5) + 0xB")), "")


class SplitShiftsTest(unittest.TestCase):
    CODE = words(0x00C01021, 0x03E00008, 0x00000000)
    DRAFT = autodraft.PRELUDE + ("\nM2C_UNK func_80002000();\nvoid f(s32 arg0) {\n    s32 var_s0_8;\n\n"
                                 "    var_s0_8 = ((s32) (arg0 << 0x10)) >> 0xD;\n    func_80002000(var_s0_8);\n"
                                 "    if (arg0 != 0) {\n        var_s0_8 = ((s32) (arg0 << 0x10)) >> 0xD;\n"
                                 "        func_80002000(var_s0_8);\n    }\n}\n")

    def body(self, draft):
        rows = redraft_gcc_opt.variants(draft, "f", code=self.CODE, vram=VRAM, pipeline="kmc+as", flags=("-O2",))
        texts = [text for label, text in rows if label.split("+") == ["shifts"]]
        return " ".join(texts[0][texts[0].index("void f("):].split()) if texts else ""

    def test_only_the_repeated_computation_is_split(self):
        body = self.body(self.DRAFT)
        self.assertIn("var_s0_8 = ((s32) (arg0 << 0x10)) >> 0xD; func_80002000(var_s0_8);", body)
        self.assertIn("if (arg0 != 0) { var_s0_8 = arg0 << 0x10; var_s0_8 >>= 0xD; func_80002000(var_s0_8); }", body)

    def test_a_single_computation_is_left_alone(self):
        self.assertEqual(self.body(self.DRAFT.replace("        var_s0_8 = ((s32) (arg0 << 0x10)) >> 0xD;\n", "")), "")


class ScanningLoopsTest(unittest.TestCase):
    CODE = words(0x00C01021, 0x03E00008, 0x00000000)
    DRAFT = autodraft.PRELUDE + ("\nM2C_UNK func_80002000();\nvoid f(u8 *arg0) {\n    u8 *var_s0_6;\n\n"
                                 "    var_s0_6 = arg0;\n    do {\n        var_s0_6 += 1;\n    } while (*var_s0_6 != 0);\n"
                                 "    func_80002000(var_s0_6 - 1);\n}\n")

    def body(self, draft):
        rows = redraft_gcc_opt.variants(draft, "f", code=self.CODE, vram=VRAM, pipeline="kmc+as", flags=("-O2",))
        texts = [text for label, text in rows if label.split("+") == ["scan"]]
        return " ".join(texts[0][texts[0].index("void f("):].split()) if texts else ""

    def test_a_step_then_test_scan_becomes_test_and_step(self):
        self.assertIn("var_s0_6 = arg0; while ((*(var_s0_6++)) != 0) { } func_80002000(var_s0_6 - 1);", self.body(self.DRAFT))

    def test_a_loop_that_does_more_than_step_is_left_alone(self):
        draft = self.DRAFT.replace("        var_s0_6 += 1;\n", "        var_s0_6 += 1;\n        func_80002000(0);\n")
        self.assertEqual(self.body(draft), "")


class BlockCopiesTest(unittest.TestCase):
    CODE = words(0x00C01021, 0x03E00008, 0x00000000)
    DRAFT = autodraft.PRELUDE + ("\nstruct Buf { u8 unk_0; s8 unk_1; s8 unk_2; };\nextern struct Buf D_800E7A5C;\n"
                                 "extern s8 D_80002044;\nextern s8 D_80002045;\nextern s8 D_80002050;\n"
                                 "void f(void) {\n    D_800E7A5C.unk_0 = (u8) D_80002044;\n"
                                 "    D_800E7A5C.unk_1 = (s8) D_80002045;\n}\n")

    def body(self, draft):
        rows = redraft_gcc_opt.variants(draft, "f", code=self.CODE, vram=VRAM, pipeline="kmc+as", flags=("-O2",))
        texts = [text for label, text in rows if label.split("+") == ["block"]]
        return " ".join(texts[0].split()) if texts else ""

    def test_bytes_assigned_one_by_one_become_one_copy(self):
        body = self.body(self.DRAFT)
        self.assertIn("struct n64dk_bytes2 { char c[2]; };", body)
        self.assertIn("void f(void) { *((struct n64dk_bytes2 *) (&D_800E7A5C)) = *((struct n64dk_bytes2 *) (&D_80002044)); }", body)

    def test_bytes_that_are_not_neighbours_stay_assignments(self):
        self.assertEqual(self.body(self.DRAFT.replace("(s8) D_80002045", "(s8) D_80002050")), "")


class EndlessLoopsTest(unittest.TestCase):
    CODE = words(0x00C01021, 0x03E00008, 0x00000000)
    DRAFT = autodraft.PRELUDE + ("\nextern s32 D_800E7A58;\nextern s16 D_800881A0[];\nvoid f(void) {\n    D_800E7A58 = 0;\n"
                                 "loop_8:\n    if (D_800881A0[D_800E7A58] == -1) {\n        D_800E7A58 += 1;\n"
                                 "        goto loop_8;\n    }\n}\n")

    def body(self, draft):
        rows = redraft_gcc_opt.variants(draft, "f", code=self.CODE, vram=VRAM, pipeline="kmc+as", flags=("-O2",))
        texts = [text for label, text in rows if label.split("+") == ["forever"]]
        return " ".join(texts[0][texts[0].index("void f("):].split()) if texts else ""

    def test_a_test_and_goto_back_becomes_an_endless_loop_with_a_break(self):
        self.assertIn("D_800E7A58 = 0; while (1) { if (D_800881A0[D_800E7A58] != (-1)) { break; } D_800E7A58 += 1; } }",
                      self.body(self.DRAFT))

    def test_a_label_reached_from_elsewhere_is_left_alone(self):
        draft = self.DRAFT.replace("    D_800E7A58 = 0;\n", "    if (D_800E7A58 == 3) {\n        goto loop_8;\n    }\n")
        self.assertEqual(self.body(draft), "")


class ComparisonSidesTest(unittest.TestCase):
    # lbu $v1, 2($a0); lbu $v0, 3($a0); sltu $v0, $v0, $v1; jr $ra; nop
    CODE = words(0x90830002, 0x90820003, 0x0043102B, 0x03E00008, 0x00000000)
    DRAFT = autodraft.PRELUDE + ("\ns32 f(void *arg0) {\n    return M2C_FIELD(arg0, u8 *, 3) < M2C_FIELD(arg0, u8 *, 2);\n}\n")

    def body(self, draft, code=None):
        rows = redraft_gcc_opt.variants(draft, "f", code=code or self.CODE, vram=VRAM, pipeline="kmc+as", flags=("-O2",))
        texts = [text for label, text in rows if label.split("+") == ["sides"]]
        return " ".join(texts[0][texts[0].index("s32 f("):].split()) if texts else ""

    def test_a_comparison_is_turned_to_the_order_the_target_loads_in(self):
        self.assertIn("return M2C_FIELD(arg0, u8 *, 2) > M2C_FIELD(arg0, u8 *, 3);", self.body(self.DRAFT))

    def test_a_comparison_already_in_load_order_is_left_alone(self):
        draft = self.DRAFT.replace("M2C_FIELD(arg0, u8 *, 3) < M2C_FIELD(arg0, u8 *, 2)", "M2C_FIELD(arg0, u8 *, 2) > M2C_FIELD(arg0, u8 *, 3)")
        self.assertEqual(self.body(draft), "")

    def test_loads_seen_in_both_orders_decide_nothing_for_one_comparison(self):
        code = words(0x90830002, 0x90820003, 0x90820003, 0x90830002, 0x03E00008, 0x00000000)
        self.assertEqual(self.body(self.DRAFT, code=code), "")

    def test_two_comparisons_take_the_order_of_their_own_load_pairs(self):
        # lbu 2, lbu 3, sltu; lbu 3, lbu 2, sltu: the first compares 2 with 3, the second 3 with 2
        code = words(0x90830002, 0x90820003, 0x0043102B, 0x90830003, 0x90820002, 0x0043102B, 0x03E00008, 0x00000000)
        draft = self.DRAFT.replace("    return M2C_FIELD(arg0, u8 *, 3) < M2C_FIELD(arg0, u8 *, 2);",
                                   "    s32 n;\n\n    n = M2C_FIELD(arg0, u8 *, 3) < M2C_FIELD(arg0, u8 *, 2);\n"
                                   "    n += M2C_FIELD(arg0, u8 *, 2) < M2C_FIELD(arg0, u8 *, 3);\n    return n;")
        body = self.body(draft, code=code)
        self.assertIn("n = M2C_FIELD(arg0, u8 *, 2) > M2C_FIELD(arg0, u8 *, 3);", body)
        self.assertIn("n += M2C_FIELD(arg0, u8 *, 3) > M2C_FIELD(arg0, u8 *, 2);", body)


class WidenedLocalsTest(unittest.TestCase):
    DRAFT = autodraft.PRELUDE + ("\nM2C_UNK func_80002000();\nvoid f(s8 arg0) {\n    s8 var_s3_27;\n    u16 temp_v0_3;\n\n"
                                 "    var_s3_27 = arg0;\n    temp_v0_3 = arg0;\n    func_80002000(var_s3_27, temp_v0_3);\n}\n")

    def body(self, code):
        rows = redraft_gcc_opt.variants(self.DRAFT, "f", code=code, vram=VRAM, pipeline="kmc+as", flags=("-O2",))
        texts = [text for label, text in rows if label.split("+") == ["ints"]]
        return " ".join(texts[0][texts[0].index("void f("):].split()) if texts else ""

    def test_locals_of_a_width_the_target_never_narrows_to_become_words(self):
        body = self.body(words(0x00C01021, 0x03E00008, 0x00000000))
        self.assertIn("void f(s8 arg0) { s32 var_s3_27; s32 temp_v0_3;", body)      # the parameter keeps its type

    def test_a_width_the_target_does_narrow_to_is_kept(self):
        # sll $v0, $a0, 24; sra $v0, $v0, 24; jr $ra; nop: signed bytes are made in a register
        body = self.body(words(0x00041600, 0x00021603, 0x03E00008, 0x00000000))
        self.assertIn("s8 var_s3_27; s32 temp_v0_3;", body)


class MemberGlobalsTest(unittest.TestCase):
    CODE = words(0x00C01021, 0x03E00008, 0x00000000)
    DRAFT = autodraft.PRELUDE + ("\nextern u8 D_800C4EEB;\nextern s32 D_80088090;\nM2C_UNK func_80002000();\n"
                                 "void f(void *arg0) {\n    M2C_FIELD(arg0, s32 *, 0x2C) = 0;\n"
                                 "    func_80002000(D_800C4EEB, &D_80088090);\n}\n")

    def body(self, draft):
        rows = redraft_gcc_opt.variants(draft, "f", code=self.CODE, vram=VRAM, pipeline="kmc+as", flags=("-O2",))
        texts = [text for label, text in rows if label.split("+") == ["members"]]
        return " ".join(texts[0].split()) if texts else ""

    def test_a_scalar_global_used_directly_becomes_a_one_member_structure(self):
        body = self.body(self.DRAFT)
        self.assertIn("extern struct { u8 v; } D_800C4EEB;", body)
        self.assertIn("func_80002000(D_800C4EEB.v, &D_80088090);", body)
        self.assertIn("extern s32 D_80088090;", body)       # its address is what is used: unchanged

    def test_a_function_using_no_scalar_global_gets_no_variant(self):
        self.assertEqual(self.body(self.DRAFT.replace("D_800C4EEB, ", "0, ")), "")


class SwitchFromZeroTest(unittest.TestCase):
    DRAFT = autodraft.PRELUDE + ("\nM2C_UNK func_80002000();\nvoid f(u32 arg0) {\n    switch (arg0) {\n    case 4:\n"
                                 "        func_80002000(1);\n        break;\n    case 5:\n    case 14:\n"
                                 "        func_80002000(2);\n        break;\n    }\n}\n")

    def body(self, code):
        rows = redraft_gcc_opt.variants(self.DRAFT, "f", code=code, vram=VRAM, pipeline="kmc+as", flags=("-O2",))
        texts = [text for label, text in rows if label.split("+") == ["cases"]]
        return " ".join(texts[0][texts[0].index("void f("):].split()) if texts else ""

    def test_a_table_indexed_from_zero_gets_a_case_zero(self):
        # sltiu $v0, $a0, 0xF; jr $ra; nop: the range test of cases 0..14, nothing subtracted
        self.assertIn("switch (arg0) { case 0: break; case 4:", self.body(words(0x2C82000F, 0x03E00008, 0x00000000)))

    def test_a_table_indexed_from_its_lowest_case_is_left_alone(self):
        # addiu $v1, $a0, -4; sltiu $v0, $v1, 0xB; jr $ra; nop
        self.assertEqual(self.body(words(0x2483FFFC, 0x2C62000B, 0x03E00008, 0x00000000)), "")


class NarrowedResultsTest(unittest.TestCase):
    DRAFT = autodraft.PRELUDE + ("\ns16 func_800A1D20();                                /* extern */\nM2C_UNK func_80002000();\n"
                                 "void f(void) {\n    s16 temp_s4_18;\n\n    temp_s4_18 = func_800A1D20();\n"
                                 "    func_80002000(temp_s4_18);\n}\n")

    def text(self, code):
        rows = redraft_gcc_opt.variants(self.DRAFT, "f", code=code, vram=VRAM, pipeline="kmc+as", flags=("-O2",))
        texts = [text for label, text in rows if "narrowed" in label.split("+")]
        return " ".join(texts[0].split()) if texts else ""

    def test_a_result_narrowed_right_after_the_call_is_cast_there_and_kept_as_a_word(self):
        # jal; nop; sll $v0, $v0, 16; sra $s4, $v0, 16; jr $ra; nop
        text = self.text(words(0x0C000800, 0x00000000, 0x00021400, 0x0002A403, 0x03E00008, 0x00000000))
        self.assertIn("s32 func_800A1D20();", text)
        self.assertIn("s32 temp_s4_18; temp_s4_18 = (s16) func_800A1D20();", text)

    def test_without_the_narrowing_after_a_call_nothing_changes(self):
        self.assertEqual(self.text(words(0x0C000800, 0x00000000, 0x03E00008, 0x00000000)), "")


class CaseChainsTest(unittest.TestCase):
    CODE = words(0x00C01021, 0x03E00008, 0x00000000)
    DRAFT = autodraft.PRELUDE + ("\nM2C_UNK func_80002000();\nvoid f(s32 arg0) {\n    switch (arg0) {\n    case 0:\n"
                                 "        func_80002000(1);\n        return;\n    case 1:\n        func_80002000(2);\n"
                                 "        break;\n    }\n}\n")

    def body(self, draft):
        rows = redraft_gcc_opt.variants(draft, "f", code=self.CODE, vram=VRAM, pipeline="kmc+as", flags=("-O2",))
        texts = [text for label, text in rows if "chain" in label.split("+")]
        return " ".join(texts[0][texts[0].index("void f("):].split()) if texts else ""

    def test_a_small_switch_becomes_a_chain_in_case_order(self):
        self.assertIn("if (arg0 == 0) { func_80002000(1); } else if (arg0 == 1) { func_80002000(2); }", self.body(self.DRAFT))

    def test_a_switch_with_fall_through_is_left_alone(self):
        self.assertEqual(self.body(self.DRAFT.replace("        return;\n", "")), "")


class LiteralAndJoinedTest(unittest.TestCase):
    # lui $at, 0x8008; sw $v0, 0x7784($at); jr $ra; nop: the target stores to the global once
    CODE = words(0x3C018008, 0xAC227784, 0x03E00008, 0x00000000)
    DRAFT = autodraft.PRELUDE + ("\nextern M2C_UNK D_80002D48;\nextern s32 D_80087784;\nM2C_UNK func_80002000();\n"
                                 "void f(s32 arg0) {\n    s32 pad[2];\n    M2C_UNK sp10;\n    M2C_UNK sp14;\n"
                                 "    M2C_UNK sp18;\n    M2C_UNK sp1C;\n\n"
                                 "    sp10 = M2C_UNALIGNED32(M2C_FIELD(&D_80002D48, M2C_UNK *, 0));\n"
                                 "    sp14 = M2C_UNALIGNED32(M2C_FIELD(&D_80002D48, M2C_UNK *, 4));\n"
                                 "    sp18 = M2C_UNALIGNED32(M2C_FIELD(&D_80002D48, M2C_UNK *, 8));\n"
                                 "    sp1C = M2C_UNALIGNED32(M2C_FIELD(&D_80002D48, M2C_UNK *, 0xC));\n"
                                 "    if (arg0 == 1) {\n        D_80087784 = 5;\n        func_80002000(&sp10);\n"
                                 "    } else {\n        func_80002000(0);\n        D_80087784 = 6;\n    }\n}\n")

    def body(self, draft, wanted, code=None):
        rows = redraft_gcc_opt.variants(draft, "f", code=code or self.CODE, vram=VRAM, pipeline="kmc+as", flags=("-O2",))
        texts = [text for label, text in rows if set(label.split("+")) == set(wanted)]
        return " ".join(texts[0][texts[0].index("void f("):].split()) if texts else ""

    def test_word_copies_from_a_constant_become_one_initialised_array(self):
        body = self.body(self.DRAFT, ["literal"])
        self.assertIn("struct n64dk_text16 sp10 = D_80002D48;", body)
        self.assertNotIn("M2C_UNALIGNED32", body)
        self.assertNotIn("sp14", body)
        self.assertNotIn("pad[", body)                      # the padding stood in for the words the array now holds

    def test_a_later_word_used_by_itself_keeps_the_copies(self):
        draft = self.DRAFT.replace("func_80002000(0);", "func_80002000(sp18);")
        self.assertEqual(self.body(draft, ["literal"]), "")

    def test_a_store_made_in_both_branches_moves_to_their_ends(self):
        body = self.body(self.DRAFT, ["joined"])
        self.assertIn("if (arg0 == 1) { func_80002000(&sp10); D_80087784 = 5; } else { func_80002000(0); D_80087784 = 6; }", body)

    def test_two_stores_in_the_target_leave_the_branches_alone(self):
        code = words(0x3C018008, 0xAC227784, 0x3C018008, 0xAC237784, 0x03E00008, 0x00000000)
        self.assertEqual(self.body(self.DRAFT, ["joined"], code=code), "")


class ElementFormTest(unittest.TestCase):
    HEAD = "#ifndef N64DK_M2C_PRELUDE\n#endif\n"

    def body(self, text: str) -> str:
        return redraft_elements.rewrite(self.HEAD + text)[0][len(self.HEAD):]

    def test_multipliers_and_offsets_are_divided_by_the_element_size(self):
        out = self.body("extern M2C_UNK D_80100000;\nextern u16 D_80100100[];\nvoid f(s32 i, s32 *p) {\n"
                        "    *(&D_80100000 + (i * 8)) = 0;\n    x = *((i * 0xA) + D_80100100);\n    g(p + 0x10);\n"
                        "    y = *(&D_80100000 + (i * 4));\n}\n")
        self.assertIn("*(&D_80100000 + (i * 2)) = 0;", out)
        self.assertIn("x = *((i * 5) + D_80100100);", out)
        self.assertIn("g(p + 4);", out)
        self.assertIn("y = *(&D_80100000 + i);", out)

    def test_values_members_bytes_and_unknown_sizes_are_left_alone(self):
        text = ("extern u8 D_80100200[];\nvoid f(s32 *p, Thing *q, s32 i) {\n    i = *p + 4;\n    i = p[2] + 4;\n"
                "    g(D_80100200 + 4, q + 8, p->x + 4);\n}\n")
        self.assertEqual(redraft_elements.rewrite(self.HEAD + text), (self.HEAD + text, 0))

    def test_a_sum_that_does_not_divide_stops_the_whole_rewrite(self):
        text = "extern s32 D_80100000;\nvoid f(s32 i) {\n    *(&D_80100000 + (i * 8)) = *(&D_80100000 + (i * 6));\n}\n"
        self.assertEqual(redraft_elements.rewrite(self.HEAD + text)[1], 0)

    def test_offered_for_every_compiler(self):
        source = self.HEAD + "extern s32 D_80100000;\nvoid f(s32 i) {\n    *(&D_80100000 + (i * 8)) = 0;\n}\n"
        for pipeline in ("ido5.3", "kmc+as", "sn"):
            self.assertEqual([label for label, _t in redraft_elements.variants(source, "f", pipeline=pipeline)],
                             ["elements"])
        self.assertEqual(redraft_elements.variants(self.HEAD + "void f(void) {\n}\n", "f", pipeline="kmc+as"), [])


class RoundTripBase:
    module, recipe = None, ("", [], "")
    cases: tuple[str, ...] = ()

    @classmethod
    def setUpClass(cls):
        pipeline, flags, key = cls.recipe
        found, reason = _compiler(pipeline, flags, key)
        if found is None:
            raise unittest.SkipTest(reason)
        cls.compile = staticmethod(found)

    def round_trip(self, case: str) -> list[str]:
        truth, draft = CASES[case]
        target = self.compile(autodraft.PRELUDE + "\n" + truth)
        pipeline, flags, _key = self.recipe
        source = autodraft.PRELUDE + "\n" + draft
        self.assertNotEqual(self._built(source), target, "the draft must not already match")
        rows = self.module.variants(source, "f", code=target, vram=VRAM, pipeline=pipeline, flags=tuple(flags))
        return [label for label, text in rows if self._built(text) == target]

    def _built(self, text: str):
        try:
            return self.compile(text)
        except Exception:                                    # noqa: BLE001 - a rewrite that does not build is not a match
            return None


class GccOptRoundTrip(RoundTripBase, unittest.TestCase):
    module, recipe = redraft_gcc_opt, GCC

    def test_pointer_arithmetic_written_in_bytes(self):
        self.assertTrue(self.round_trip("bytes"))

    def test_an_argument_passed_through_untouched(self):
        self.assertTrue(self.round_trip("passed"))


class IdoOptRoundTrip(RoundTripBase, unittest.TestCase):
    module, recipe = redraft_ido_opt, IDO

    def test_pointer_arithmetic_written_in_bytes_needs_the_element_form(self):
        # The byte cast is right in meaning and changes the temporary IDO picks, so the
        # optimised-build generator alone does not bring this one back; the element form does.
        self.assertEqual(self.round_trip("bytes"), [])
        truth, draft = CASES["bytes"]
        target = self.compile(autodraft.PRELUDE + "\n" + truth)
        rows = redraft.variants(autodraft.PRELUDE + "\n" + draft, "f", code=target, vram=VRAM, pipeline=IDO[0],
                                flags=tuple(IDO[1]))
        self.assertIn("elements", [label for label, text in rows if self._built(text) == target])

    def test_an_argument_passed_through_untouched(self):
        self.assertTrue(self.round_trip("passed"))


if __name__ == "__main__":
    unittest.main()
