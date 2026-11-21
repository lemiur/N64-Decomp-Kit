import struct
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from n64dk.match import autodraft, m2c_bridge
from n64dk.commands.accelerate import _draft_mapping


def asm(*words):
    return b"".join(struct.pack(">I", w) for w in words)


class EmitTest(unittest.TestCase):
    def test_address_pair_is_symbolized_and_constant_pair_is_not(self):
        code = asm(0x3C028009, 0x8C4287D0,      # lui v0,0x8009 ; lw v0,-0x7830(v0)  -> address
                   0x3C03AAAA, 0x3463AAAB,      # lui v1,0xAAAA ; ori v1,v1,0xAAAB   -> constant
                   0x3C043F80, 0x24840000,      # lui a0,0x3F80 ; addiu a0,a0,0 -> constant
                   0x03E00008, 0x00000000)
        text = m2c_bridge.emit_asm("f", code, 0x80001000, macros=False)
        self.assertIn("%hi(D_800887D0)", text)
        self.assertIn("%lo(D_800887D0)($v0)", text)
        self.assertNotIn("AAAAAAAB", text)
        self.assertNotIn("D_3F800000", text)
        self.assertIn("0xAAAB", text)

    def test_indexed_pair_survives_addu(self):
        code = asm(0x3C018009, 0x00220821, 0x8C2287D0, 0x03E00008, 0x00000000)   # lui at; addu at,at,v0; lw
        self.assertIn("%lo(D_800887D0)($at)", m2c_bridge.emit_asm("f", code, 0x80001000, macros=False))

    def test_jump_table_is_found_and_named_for_m2c(self):
        base = 0x80001000
        code = asm(0x3C018009, 0x00220821, 0x8C2287D0, 0x00400008, 0x00000000,   # ... lw v0,tbl(at); jr v0
                   0x03E00008, 0x24020001, 0x03E00008, 0x24020002)
        table_off = 0x800887D0 - base
        image = code + bytes(table_off - len(code)) + asm(base + 0x14, base + 0x1C, 0)
        # The shared analyser recognizes the dispatch from the containing image.
        tables = m2c_bridge.find_jump_tables(image, base, base, len(code))
        self.assertEqual(tables, [(0x800887D0, [base + 0x14, base + 0x1C])])
        text = m2c_bridge.emit_asm("f", code, base, jump_tables=tables, macros=False)
        self.assertIn("%lo(jtbl_800887D0)($at)", text)
        self.assertIn("glabel jtbl_800887D0", text)
        self.assertIn(f".word .L{base + 0x1C:08X}", text)

    def test_jump_table_in_another_image_is_found_through_it(self):
        """An overlay's table may stand in the resident image: the function's own image does not hold it."""
        base = 0x80101000
        code = asm(0x3C018000, 0x00220821, 0x8C2229B0, 0x00400008, 0x00000000,   # lw v0,0x800029B0(at); jr v0
                   0x03E00008, 0x24020001, 0x03E00008, 0x24020002)
        resident = bytes(0x29B0) + asm(base + 0x14, base + 0x1C, 0)               # first byte at 0x80000000
        self.assertEqual(m2c_bridge.find_jump_tables(code, base, base, len(code)), [])
        tables = m2c_bridge.find_jump_tables(code, base, base, len(code), also=[(resident, 0x80000000)])
        self.assertEqual(tables, [(0x800029B0, [base + 0x14, base + 0x1C])])
        # the function's own image answers for the addresses it holds, whatever stands behind it
        shadow = bytes(base - 0x80000000) + asm(*([0xFFFFFFFF] * 9))
        self.assertEqual(m2c_bridge.find_jump_tables(code, base, base, len(code), also=[(shadow, 0x80000000), (resident, 0x80000000)]),
                         tables)

    def test_external_branch_has_faithful_and_liftable_forms(self):
        base = 0x80001000
        code = asm(0x10000007, 0x00000000)  # b 0x80001020; nop
        faithful = m2c_bridge.emit_asm("f", code, base, macros=False)
        liftable = m2c_bridge.emit_asm("f", code, base, macros=False, external="stub")
        self.assertIn("L_80001020", faithful)
        self.assertIn(".L80001020:", liftable)
        self.assertIn("j L_80001020", liftable)

    def test_real_m2c_smoke_when_installed(self):
        if m2c_bridge.find_m2c() is None:
            self.skipTest("m2c is optional")
        result = m2c_bridge.draft("fixture", asm(0x03E00008, 0x24020001), 0x80001000)
        self.assertEqual(result["status"], "ok", result["error"])
        self.assertIn("return 1", result["source"])


class RepairTest(unittest.TestCase):
    SOURCE = "void g(); /* extern */\ns32 f(s32 arg0) {\n    s32 t;\n\n    *(arg0 | 4) = sp10;\n    return g(t);\n}\n"

    def test_prepare_unprototypes_callees(self):
        text = autodraft.prepare("s32 g(s32, ? *);                    /* extern */\nvoid f(void) {\n}\n")
        self.assertIn("s32 g();", text)
        self.assertNotIn("?", text.split("#endif")[-1])

    def test_repairs(self):
        fixed = autodraft.repair(self.SOURCE, "x.c:5: `sp10' undeclared (first use this function)")
        self.assertIn("{\n    s32 sp10;\n", fixed)
        fixed = autodraft.repair(self.SOURCE, "x.c:5: invalid type argument of `unary *'")
        self.assertIn("*(s32 *)(arg0 | 4) = sp10;", fixed)
        fixed = autodraft.repair(self.SOURCE, "x.c:6: void value not ignored as it ought to be")
        self.assertTrue(fixed.startswith("s32 g();"))
        self.assertIsNone(autodraft.repair(self.SOURCE, "x.c:9: parse error"))

    def test_a_missing_declaration_goes_to_the_function_body_not_a_nested_block(self):
        source = "s32 f(s32 arg0) {\n    s32 t;\n\n    t = sp10;\n    if (arg0 != 0) {\n        t = 1;\n    }\n    return t;\n}\n"
        fixed = autodraft.repair(source, "x.c:4: `sp10' undeclared (first use this function)")
        self.assertTrue(fixed.startswith("s32 f(s32 arg0) {\n    s32 sp10;\n"), fixed)

    def test_a_call_through_a_loaded_function_pointer_is_parenthesised(self):
        line = "    temp = *M2C_FIELD(arg0, void *(***)(void *, M2C_UNK), 4)(arg0, 0x118);"
        source = "void *f(void *arg0) {\n    void *temp;\n\n" + line + "\n    return temp;\n}\n"
        for message in ("cfe: Error: x.c, line 4: Non-function name referenced in function call.",
                        "x.c:4: called object is not a function"):
            fixed = autodraft.repair(source, message)
            self.assertIn("temp = (*M2C_FIELD(arg0, void *(***)(void *, M2C_UNK), 4))(arg0, 0x118);", fixed)
        # already parenthesised, or not a call: left alone
        self.assertEqual(autodraft.call_through_loaded_pointer("x = (*M2C_FIELD(a, s32 (**)(), 0))(a);"),
                         "x = (*M2C_FIELD(a, s32 (**)(), 0))(a);")
        self.assertEqual(autodraft.call_through_loaded_pointer("x = *M2C_FIELD(a, s32 **, 0) + 1;"),
                         "x = *M2C_FIELD(a, s32 **, 0) + 1;")

    def test_a_guessed_prototype_in_a_function_pointer_cast_is_dropped_at_any_depth(self):
        source = "void f(void *a, void *b) {\n    M2C_FIELD(a, M2C_UNK (**)(M2C_UNK *, void *, void *), 8)(a, b);\n}\n"
        fixed = autodraft.repair(source, "cfe: Error: x.c, line 2: The number of arguments doesn't agree with the "
                                         "number in the declaration.")
        self.assertIn("M2C_FIELD(a, M2C_UNK (**)(), 8)(a, b);", fixed)

    def test_an_integer_minus_an_address_becomes_integer_arithmetic(self):
        source = "s32 *f(s32 *arg0) {\n    *arg0 = 0x8023E000 - &D_802142F0;\n    return arg0;\n}\n"
        fixed = autodraft.repair(source, "cfe: Error: x.c, line 2: Unacceptable operand of '-'.")
        self.assertIn("*arg0 = 0x8023E000 - (s32) &D_802142F0;", fixed)

    def test_a_starred_variable_that_is_called_gets_a_callable_type(self):
        source = "void f(void ***arg0) {\n    **arg0(arg0);\n}\n"
        fixed = autodraft.repair(source, "cfe: Error: x.c, line 2: Non-function name referenced in function call.")
        self.assertIn("    ((s32 (*)()) **arg0)(arg0);", fixed)

    def test_dereferencing_an_integer_variable_treats_it_as_an_address(self):
        source = "void f(s32 arg0) {\n    s32 var_a0;\n    void *p;\n\n    *var_a0 = 0;\n    *p = *arg0;\n}\n"
        fixed = autodraft.repair(source, "cfe: Error: x.c, line 5: Dereferenced a non-pointer.")
        self.assertIn("    *(s32 *)var_a0 = 0;", fixed)
        fixed = autodraft.repair(source, "x.c:6: invalid type argument of `unary *'")
        self.assertIn("    *p = *(s32 *)arg0;", fixed)                  # the pointer is left alone

    def test_arithmetic_on_a_loaded_void_pointer_is_byte_arithmetic(self):
        source = "void f(s32 *arg0, s32 arg1) {\n    M2C_FIELD((M2C_FIELD(arg0, void **, 0x1B4) + (arg1 * 4)), s32 *, 8) = 0;\n}\n"
        fixed = autodraft.repair(source, "cfe: Error: x.c, line 2: Unacceptable operand of '+'.")
        self.assertIn("M2C_FIELD(arg0, s8 **, 0x1B4) + (arg1 * 4)", fixed)

    def test_a_loaded_pointer_called_inside_a_condition_is_parenthesised_not_mangled(self):
        source = "s32 f(void *arg0) {\n    if (*M2C_FIELD(arg0, s32 (***)(void *), 0x1B0)(arg0) == 0) {\n        return 1;\n    }\n    return 0;\n}\n"
        fixed = autodraft.repair(source, "cfe: Error: x.c, line 2: Non-function name referenced in function call.")
        self.assertIn("if ((*M2C_FIELD(arg0, s32 (***)(void *), 0x1B0))(arg0) == 0) {", fixed)

    def test_a_pointer_compared_with_an_integer_compares_the_address(self):
        source = "s32 f(void *arg1, s32 *p) {\n    if ((p[3] == 6) && (arg1 == M2C_FIELD(p, s32 *, 0x10))) {\n        return 1;\n    }\n    return 0;\n}\n"
        fixed = autodraft.repair(source, "cfe: Error: x.c, line 2: Unacceptable operand of == or !=")
        self.assertIn("((s32) arg1 == M2C_FIELD(p, s32 *, 0x10))", fixed)
        self.assertIn("(p[3] == 6)", fixed)

    def test_null_stored_to_an_integer_is_zero(self):
        source = "void f(s32 *t) {\n    *t = NULL;\n}\n"
        self.assertIn("*t = 0;", autodraft.repair(source, "cfe: Error: x.c, line 2: Constants must have arithmetic type."))

    def test_an_address_subtracted_from_a_variable_is_integer_arithmetic(self):
        source = "void f(s32 arg0) {\n    s32 sp0;\n\n    sp0 = arg0 - &D_80180DB0;\n}\n"
        self.assertIn("sp0 = arg0 - (s32) &D_80180DB0;",
                      autodraft.repair(source, "cfe: Error: x.c, line 4: Unacceptable operand of '-'."))

    def test_a_builtin_called_with_other_arguments_is_called_through_a_cast(self):
        source = "void f(s32 a, s32 b) {\n    memcpy(&D_1, a, 0x20, b);\n}\n"
        fixed = autodraft.repair(source, "x.c:2: too many arguments to function `memcpy'")
        self.assertIn("    ((s32 (*)()) memcpy)(&D_1, a, 0x20, b);", fixed)
        declared = "s32 g(s32);\nvoid f(s32 a, s32 b) {\n    g(a, b);\n}\n"         # a declaration is relaxed instead
        self.assertTrue(autodraft.repair(declared, "x.c:3: too many arguments to function `g'").startswith("s32 g();"))

    def test_a_pointer_compared_with_an_address_expression(self):
        source = "void f(s32 v) {\n    s32 sp7C;\n\n    do {\n    } while ((s32) v != ((v * 8) + &sp7C));\n}\n"
        fixed = autodraft.repair(source, "cfe: Error: x.c, line 5: Unacceptable operand of == or !=")
        self.assertIn("} while ((s32) v != ((v * 8) + (s32) &sp7C));", fixed)

    def test_a_word_read_through_a_void_cast(self):
        source = "void f(void) {\n    do {\n    } while (*(void *)0xA4600010 & 3);\n}\n"
        self.assertIn("while (*(s32 *)0xA4600010 & 3);",
                      autodraft.repair(source, "x.c:3: void value not ignored as it ought to be"))

    def test_a_function_pointer_read_from_a_computed_address_is_called(self):
        source = "void f(s32 i) {\n    *D_1 = *(&D_2 + (i * 4))(D_1, i);\n}\n"
        fixed = autodraft.repair(source, "cfe: Error: x.c, line 2: Non-function name referenced in function call.")
        self.assertIn("*D_1 = ((s32 (*)()) *(&D_2 + (i * 4)))(D_1, i);", fixed)

    def test_sweep_rejects_interval_outside_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = autodraft.sweep([("outside", 0x80002000, 8)], bytes(16), 0x80001000,
                                     lambda *_args: None, Path(tmp), jobs=1)
        self.assertEqual(report["rows"][0]["status"], "skipped")
        self.assertIn("outside", report["rows"][0]["error"])

    def test_exact_draft_score_ignores_unowned_alignment(self):
        code = asm(0x03E00008, 0x00000000)
        outcome = SimpleNamespace(
            exact=True, error=None,
            result=SimpleNamespace(compiled_text=code + b"\x00\x00\x00\x00", diff=None))
        lifted = {"asm": "glabel f\n", "source": "void f(void) {}\n", "error": None}
        with tempfile.TemporaryDirectory() as tmp, \
             patch("n64dk.match.autodraft.m2c_bridge.draft", return_value=lifted):
            row = autodraft.draft_one("f", 0x80001000, len(code), code, lambda _a: None,
                                      lambda *_a: outcome, Path(tmp))
        self.assertEqual(row.status, "exact")
        self.assertEqual(row.asm_score["total"], 0)


class MappingTest(unittest.TestCase):
    def test_composite_delta_disagreement_fails_closed(self):
        args = Namespace(project=".", addr=0x800A2078, size=0xF8,
                         image="overlay_2_composite.z64", image_name=None,
                         vram_delta=0x80000000)
        row = {"id": "target", "image": "overlay2", "vram": "0x800A2078", "size": 0xF8,
               "section_vram": "0x8008B8E0", "section_size": 0x32000}
        with patch("n64dk.match.inventory.build", return_value={"functions": [row]}), \
             patch("n64dk.match.layout.from_inventory", return_value=object()), \
             patch("n64dk.match.layout.resolve_image_map",
                   return_value={"selected": {"offset": 0xA2C78,
                                                "authority": "anchored-composite-region"},
                                 "linked_vram": "0x800A2078", "linked_base": "0x8008B8E0"}):
            with self.assertRaisesRegex(ValueError, "conflicts|project mapping"):
                _draft_mapping(args)

    def test_canonical_composite_delta_records_authority(self):
        args = Namespace(project=".", addr=0x800A2078, size=0xF8,
                         image="overlay_2_composite.z64", image_name=None,
                         vram_delta=0x7FFFF400)
        row = {"id": "target", "image": "overlay2", "vram": "0x800A2078", "size": 0xF8,
               "section_vram": "0x8008B8E0", "section_size": 0x32000}
        with patch("n64dk.match.inventory.build", return_value={"functions": [row]}), \
             patch("n64dk.match.layout.from_inventory", return_value=object()), \
             patch("n64dk.match.layout.resolve_image_map",
                   return_value={"selected": {"offset": 0xA2C78,
                                                "authority": "anchored-composite-region"},
                                 "linked_vram": "0x800A2078", "linked_base": "0x8008B8E0"}):
            self.assertEqual(_draft_mapping(args)["authority"], "anchored-composite-region")

    def test_shifted_catalogue_address_normalizes_to_linked_vram(self):
        args = Namespace(project=".", addr=0x800E0B58, size=0x20,
                         image="overlay_4_composite.z64", image_name="overlay4",
                         vram_delta=0x7FFE4C48)
        row = {"id": "target", "image": "overlay4", "vram": "0x800E0B58", "size": 0x20,
               "section_vram": "0x800C77E8", "section_size": 0x21000}
        resolved = {"selected": {"offset": 0xFA670, "authority": "anchored-composite-region"},
                    "linked_vram": "0x800DF2B8", "linked_base": "0x800C5F48"}
        with patch("n64dk.match.inventory.build", return_value={"functions": [row]}), \
             patch("n64dk.match.layout.from_inventory", return_value=object()), \
             patch("n64dk.match.layout.resolve_image_map", return_value=resolved):
            mapping = _draft_mapping(args)
        self.assertEqual(args.addr, 0x800DF2B8)
        self.assertEqual(mapping["catalogue_vram"], "0x800E0B58")
        self.assertEqual(mapping["linked_vram"], "0x800DF2B8")


class IdoRepairTest(unittest.TestCase):
    """The same repairs must fire on IDO front-end diagnostics, which are worded differently."""

    SOURCE = ("void g(s32, s32);  /* extern */\nvoid f(void *arg0, s32 arg1) {\n    void *var_a0;\n    s32 *var_a1;\n\n"
              "    var_a0 = arg0;\n    var_a0 += 0x28;\n    *(arg0 + sp4) = 0;\n    var_a1 = arg1;\n    g(arg1);\n"
              "    (*(M2C_UNK (**)(void *, void *))((s8 *)(arg0) + 8))(arg0);\n}\n")

    def fix(self, line, message):
        return autodraft.repair(self.SOURCE, f"cfe: Error: /tmp/x/f.c, line {line}: {message}\n  src\n ---^")

    def test_undefined_name(self):
        self.assertIn("{\n    s32 sp4;\n", self.fix(8, "'sp4' undefined; reoccurrences will not be reported."))

    def test_dereferenced_non_pointer(self):
        self.assertIn("*(s32 *)(arg0 + sp4) = 0;", self.fix(8, "Dereferenced a non-pointer."))

    def test_void_pointer_arithmetic_becomes_byte_pointer(self):
        fixed = self.fix(7, "Bad operand type for += or -=")
        self.assertIn("    s8 *var_a0;", fixed)
        self.assertIn("void *arg0", fixed)                       # only the variable on that line changes
        self.assertIn("f(s8 *arg0, s32 arg1)", self.fix(8, "Unacceptable operand of '+'."))

    def test_argument_count_disagreement(self):
        message = "Number of arguments doesn't agree with the number in the declaration."
        self.assertIn("(*(M2C_UNK (**)())((s8 *)(arg0) + 8))(arg0);", self.fix(11, message))
        self.assertTrue(self.fix(10, message).startswith("void g();  /* extern */"))

    def test_pointer_integer_assignment_gets_the_target_type(self):
        self.assertIn("    var_a1 = (s32 *) (arg1);", self.fix(9, "illegal combination of pointer and integer"))

    def test_redeclared_stack_slot_keeps_its_first_declaration(self):
        source = "void f(void) {\n    s32 sp1C;\n    f32 sp1C;\n    s32 a;\n\n    a = sp1C;\n}\n"
        ido = "cfe: Error: f.c, line 3: redeclaration of 'sp1C'; previous declaration at line 2 in file 'f.c'"
        gcc = "f.c:3: redeclaration of `sp1C'\nf.c:2: `sp1C' previously declared here"
        for message in (ido, gcc):
            self.assertEqual(autodraft.repair(source, message),
                             "void f(void) {\n    s32 sp1C;\n    s32 a;\n\n    a = sp1C;\n}\n")
        self.assertIsNone(autodraft.repair(source, "f.c:6: redeclaration of `sp1C'"))   # not a declaration line

    def test_dereferenced_void_pointer_becomes_a_word_pointer(self):
        source = "s32 f(void *arg0) {\n    void *var_v0;\n\n    var_v0 = arg0;\n    return *var_v0;\n}\n"
        for message in ("f.c:5: warning: dereferencing `void *' pointer\nf.c:5: void value not ignored as it ought to be",
                        "cfe: Error: f.c, line 5: Reference of an expression of void type or an incomplete type."):
            fixed = autodraft.repair(source, message)
            self.assertIn("    s32 *var_v0;", fixed)
            self.assertIn("void *arg0", fixed)

    def test_gcc_wording_still_works(self):
        fixed = autodraft.repair(self.SOURCE, "f.c:8: `sp4' undeclared (first use this function)")
        self.assertIn("    s32 sp4;", fixed)


if __name__ == "__main__":
    unittest.main()
