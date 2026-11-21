import struct
import unittest

from n64dk.match import causes, walls


def code(*words):
    return struct.pack(f">{len(words)}I", *words)


JAL = lambda target: 0x0C000000 | ((target & 0x0FFFFFFF) >> 2)
JR_RA, NOP = 0x03E00008, 0
BEQZ_V0 = 0x10400002                      # beqz $v0, +2
ADDU_V0_A0 = 0x00801021                   # addu $v0, $a0, $zero
ADDU_V1_A0 = 0x00801821                   # addu $v1, $a0, $zero


class ClassifyTest(unittest.TestCase):
    def test_identical_code_has_no_wall(self):
        same = code(JAL(0x80001000), NOP, JR_RA, NOP)
        self.assertIsNone(walls.classify(same, same)["wall"])

    def test_a_missing_call_is_the_call_set_wall_even_when_registers_differ_too(self):
        retail = code(JAL(0x80001000), NOP, JAL(0x80002000), ADDU_V0_A0, JR_RA, NOP)
        built = code(JAL(0x80001000), NOP, ADDU_V1_A0, NOP, JR_RA, NOP)
        found = walls.classify(retail, built)
        self.assertEqual(found["wall"], "call-set")
        self.assertIn("0x80002000", " ".join(found["evidence"]))

    def test_the_same_number_of_calls_to_a_different_function_is_still_the_call_set_wall(self):
        retail = code(JAL(0x80001000), NOP, JR_RA, NOP)
        built = code(JAL(0x80003000), NOP, JR_RA, NOP)
        self.assertEqual(walls.classify(retail, built)["wall"], "call-set")

    def test_unresolved_calls_are_compared_by_count_only(self):
        retail = code(JAL(0x80001000), NOP, JR_RA, NOP)
        built = code(JAL(0), NOP, ADDU_V0_A0, JR_RA, NOP)
        self.assertNotEqual(walls.classify(retail, built)["wall"], "call-set")

    def test_a_different_number_of_branches_is_the_control_flow_wall(self):
        retail = code(BEQZ_V0, NOP, ADDU_V0_A0, JR_RA, NOP)
        built = code(ADDU_V0_A0, NOP, NOP, JR_RA, NOP)
        found = walls.classify(retail, built)
        self.assertEqual(found["wall"], "control-flow")
        self.assertIn("branch", found["evidence"][0])

    def test_equal_counts_with_other_registers_is_the_register_wall(self):
        retail = code(ADDU_V0_A0, JR_RA, NOP)
        built = code(ADDU_V1_A0, JR_RA, NOP)
        self.assertEqual(walls.classify(retail, built)["wall"], "register-schedule")

    def test_code_equal_under_the_address_mask_is_the_referent_wall(self):
        retail = code(0x3C028001, 0x8C421234, JR_RA, NOP)      # lui $v0, 0x8001; lw $v0, 0x1234($v0)
        built = code(0x3C028002, 0x8C425678, JR_RA, NOP)
        self.assertEqual(walls.classify(retail, built)["wall"], "referent")

    def test_a_permuted_jump_table_is_the_switch_map_wall(self):
        same_shape = code(ADDU_V0_A0, JR_RA, NOP)
        other = code(ADDU_V1_A0, JR_RA, NOP)
        found = walls.classify(same_shape, other, target_tables=[(0x10, 3, [0x20, 0x30, 0x40])],
                               candidate_tables=[(0x10, 3, [0x20, 0x40, 0x30])])
        self.assertEqual(found["wall"], "switch-map")
        self.assertIn("case 1", " ".join(found["evidence"]))


class GateTest(unittest.TestCase):
    def test_register_causes_wait_for_the_call_set(self):
        held = walls.gate("call-set", ["register.pressure", "schedule.delay_slot", "type.width", "control.branch_form"])
        self.assertEqual(held, {"register.pressure", "schedule.delay_slot", "control.branch_form"})

    def test_nothing_waits_at_the_register_wall(self):
        self.assertEqual(walls.gate("register-schedule", ["register.pressure"]), set())

    def test_diagnose_reports_the_call_set_first_and_holds_register_causes_back(self):
        vram = 0x80010000
        retail = code(0x27BDFFE8, 0xAFBF0010, JAL(0x80001000), NOP, JAL(0x80002000), ADDU_V0_A0,
                      0x8FBF0010, JR_RA, 0x27BD0018)
        built = code(0x27BDFFE8, 0xAFBF0010, JAL(0x80001000), NOP, ADDU_V1_A0, NOP,
                     0x8FBF0010, JR_RA, 0x27BD0018)
        ranked = causes.diagnose(retail, built, vram)
        self.assertEqual(ranked[0].id, "call.set")
        later = [c for c in ranked if c.id.startswith(("register.", "schedule."))]
        self.assertTrue(all(c.confidence <= 0.3 for c in later))


class JumpTableTest(unittest.TestCase):
    VRAM = 0x80020000
    # sltiu $v0, $a0, 3; lui $v0, 0x8003; sll $v1, $a0, 2; addu $v0, $v0, $v1; lw $v0, 0x10($v0); jr $v0; nop
    FUNCTION = code(0x2C820003, 0x3C028003, 0x00041880, 0x00431021, 0x8C420010, 0x00400008, NOP,
                    ADDU_V0_A0, JR_RA, NOP)

    def read(self, table):
        def read(address, size):
            return table if address == 0x80030010 and size == len(table) else None
        return read

    def test_the_arms_are_offsets_from_the_function_start(self):
        table = struct.pack(">3I", self.VRAM + 0x1C, self.VRAM + 0x20, self.VRAM + 0x1C)
        self.assertEqual(walls.jump_tables(self.FUNCTION, self.VRAM, self.read(table)), [(0x10, 3, [0x1C, 0x20, 0x1C])])

    def test_a_table_that_cannot_be_read_is_reported_as_unknown_not_as_empty(self):
        self.assertEqual(walls.jump_tables(self.FUNCTION, self.VRAM, lambda address, size: None), [(0x10, 3, None)])

    def test_entries_outside_the_function_mean_the_table_was_misread(self):
        table = struct.pack(">3I", 0x80900000, self.VRAM + 0x20, self.VRAM + 0x1C)
        self.assertEqual(walls.jump_tables(self.FUNCTION, self.VRAM, self.read(table))[0][2], None)


class ScaffoldingTest(unittest.TestCase):
    def names(self, source):
        return {item["device"] for item in walls.scaffolding(source)}

    def test_plain_source_has_none(self):
        self.assertEqual(self.names("s32 f(Obj *o) { s32 i; for (i = 0; i < 4; i++) { o->v[i] = 0; } return 1; }"), set())

    def test_the_devices_are_named(self):
        source = ('void f(s32 a) { volatile s32 x; register Obj *p __asm__("$2"); if (a == 0) {}\n'
                  '  x = x; *(s32 *)((char *)p + 4) = (s32)&D_80001000; goto done; done: ; }')
        self.assertEqual(self.names(source), {"volatile", "pinned-register", "empty-arm", "self-assignment",
                                              "offset-cast", "pointer-as-integer", "goto"})

    def test_words_in_comments_and_strings_do_not_count(self):
        self.assertEqual(self.names('/* volatile goto x; */ void f(void) { g("goto end; volatile"); }'), set())

    def test_an_asm_label_on_a_declaration_is_not_inline_assembly(self):
        self.assertEqual(self.names('void g(void) __asm__("func_80001000"); void f(void) { g(); }'), set())

    def test_storing_a_local_into_a_member_of_the_same_name_is_not_a_self_assignment(self):
        self.assertEqual(self.names('void f(Obj *o, s32 size) { o->size = size; o->v.size = size; }'), set())

    def test_inline_assembly_is_found(self):
        self.assertIn("inline-assembly", self.names('void f(void) { __asm__("multu $4,$5\\n\\tmflo $2"); }'))


if __name__ == "__main__":
    unittest.main()
