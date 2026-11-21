import struct
import unittest

from n64dk.match import macros, normalize


def _asm(*lines):
    return "\n".join([*lines, ""])


class AtBranchTest(unittest.TestCase):
    def test_dead_comparison_becomes_a_macro(self):
        text = "\n".join(["f:", "\tslt\t$2,$4,$5", "\tbne\t$2,$0,$L3", "\tli\t$2,7", "\tj\t$31", "$L3:",
                          "\tli\t$2,1", "\tj\t$31", "\t.end\tf", ""])
        out = normalize.at_branch_macros(text)
        self.assertIn("\tblt\t$4,$5,$L3", out)
        self.assertNotIn("slt\t$2", out)

    def test_immediate_and_unsigned_and_beq(self):
        text = "\n".join(["f:", "\tsltiu\t$3,$4,10", "\tbeq\t$3,$0,$L2", "\tmove\t$2,$0", "$L2:", "\tli\t$3,1",
                          "\tj\t$31", ""])
        self.assertIn("\tbgeu\t$4,10,$L2", normalize.at_branch_macros(text))

    def test_live_comparison_is_kept(self):
        # $2 is the comparison and the return value on the fall-through path
        text = "\n".join(["f:", "\tslt\t$2,$4,$5", "\tbne\t$2,$0,$L3", "\tj\t$31", "$L3:", "\tli\t$2,1",
                          "\tj\t$31", ""])
        self.assertEqual(normalize.at_branch_macros(text), text)
        # read at the branch target
        text = "\n".join(["f:", "\tslt\t$8,$4,$5", "\tbne\t$8,$0,$L3", "\tli\t$8,0", "$L3:", "\taddu\t$2,$8,$4",
                          "\tj\t$31", ""])
        self.assertEqual(normalize.at_branch_macros(text), text)

    def test_unknown_label_is_live(self):
        text = "\n".join(["\tslt\t$8,$4,$5", "\tbne\t$8,$0,$Lmissing", "\tli\t$8,0", "\tj\t$31", ""])
        self.assertEqual(normalize.at_branch_macros(text), text)

    def test_combined_with_symbol_loads(self):
        text = "\n".join(["\tlw\t$4,g", "\tslt\t$8,$4,$5", "\tbne\t$8,$0,$L1", "\tli\t$8,0", "$L1:", "\tli\t$8,2",
                          "\tj\t$31", ""])
        out = normalize.at_macros(text)
        self.assertIn("lui\t$1,%hi(g)", out)
        self.assertIn("\tblt\t$4,$5,$L1", out)

    def test_return_delay_slot_reads_the_comparison(self):
        # KMC GCC -O2 fills the return's delay slot inside .set noreorder: the slot reads $8
        text = _asm("f:", "\tslt\t$8,$4,$5", "\tbne\t$8,$0,$L3", "\tli\t$2,1", "\t.set\tnoreorder",
                    "\t.set\tnomacro", "\tj\t$31", "\tmove\t$2,$8", "\t.set\tmacro", "\t.set\treorder", "$L3:",
                    "\tli\t$8,0", "\tj\t$31", "\t.end\tf")
        self.assertEqual(normalize.at_branch_macros(text), text)

    def test_likely_branch_slot_does_not_kill_the_fall_through(self):
        text = _asm("f:", "\tslt\t$8,$4,$5", "\tbne\t$8,$0,$L3", "\t.set\tnoreorder", "\t.set\tnomacro",
                    "\tbeql\t$6,$0,$L4", "\tli\t$8,1", "\t.set\tmacro", "\t.set\treorder", "\taddu\t$2,$8,$4",
                    "\tj\t$31", "$L4:", "\tli\t$2,0", "\tj\t$31", "$L3:", "\tli\t$8,0", "\tj\t$31", "\t.end\tf")
        self.assertEqual(normalize.at_branch_macros(text), text)

    def test_plain_branch_slot_kills_on_both_paths(self):
        text = _asm("f:", "\tslt\t$8,$4,$5", "\tbne\t$8,$0,$L3", "\t.set\tnoreorder", "\tbne\t$6,$0,$L4",
                    "\tli\t$8,1", "\t.set\treorder", "\taddu\t$2,$8,$4", "\tj\t$31", "$L4:", "\taddu\t$2,$8,$4",
                    "\tj\t$31", "$L3:", "\tli\t$8,0", "\tj\t$31")
        self.assertIn("\tblt\t$4,$5,$L3", normalize.at_branch_macros(text))

    def test_jump_to_a_label_is_followed_and_its_slot_checked(self):
        text = _asm("f:", "\tslt\t$8,$4,$5", "\tbne\t$8,$0,$L3", "\tj\t$L4", "$L3:", "\tli\t$8,0", "$L4:",
                    "\tli\t$8,1", "\tj\t$31")
        self.assertIn("\tblt\t$4,$5,$L3", normalize.at_branch_macros(text))
        text = _asm("f:", "\tslt\t$8,$4,$5", "\tbne\t$8,$0,$L3", "\t.set\tnoreorder", "\tj\t$L4",
                    "\taddu\t$2,$8,$0", "\t.set\treorder", "$L3:", "\tli\t$8,0", "$L4:", "\tli\t$8,1", "\tj\t$31")
        self.assertEqual(normalize.at_branch_macros(text), text)

    def test_jump_table_path_is_live(self):
        # the jump through the table is an unknown path even though every case label kills $8
        text = _asm("f:", "\tslt\t$8,$4,$5", "\tbne\t$8,$0,$L3", "\tsll\t$2,$6,2", "\tlw\t$2,$L11($2)",
                    "\tj\t$2", "\t.rdata", "$L11:", "\t.word\t$L5", "\t.text", "$L5:", "\tli\t$8,0", "\tj\t$31",
                    "$L3:", "\tli\t$8,0", "\tj\t$31")
        self.assertEqual(normalize.at_branch_macros(text), text)
        # the table's bounds check itself (sltu with a literal) is a dead comparison
        text = _asm("f:", "\tsltu\t$2,$4,5", "\tbeq\t$2,$0,$L10", "\tsll\t$2,$4,2", "\tlw\t$2,$L11($2)",
                    "\tj\t$2", "$L10:", "\tmove\t$2,$0", "\tj\t$31")
        self.assertIn("\tbgeu\t$4,5,$L10", normalize.at_branch_macros(text))

    def test_loops(self):
        # read after the loop: the back edge must not hide it
        text = _asm("f:", "\tslt\t$8,$4,$5", "\tbne\t$8,$0,$L3", "$L2:", "\taddiu\t$4,$4,1", "\tbne\t$4,$5,$L2",
                    "\taddu\t$2,$8,$0", "\tj\t$31", "$L3:", "\tli\t$8,0", "\tj\t$31")
        self.assertEqual(normalize.at_branch_macros(text), text)
        # the rewritten branch closes its own loop: the comparison is recomputed at the head
        text = _asm("f:", "$L1:", "\taddiu\t$4,$4,1", "\tslt\t$8,$4,$5", "\tbne\t$8,$0,$L1", "\tli\t$8,0",
                    "\tj\t$31")
        self.assertIn("\tblt\t$4,$5,$L1", normalize.at_branch_macros(text))

    def test_first_operand_read(self):
        for line in ("\tmult\t$8,$6", "\taddu\t$8,$6", "\tsra\t$8,2", "\tnegu\t$8", "\tlwl\t$8,0($6)"):
            with self.subTest(line=line):
                text = _asm("f:", "\tslt\t$8,$4,$5", "\tbne\t$8,$0,$L3", line, "\tj\t$31", "$L3:", "\tli\t$8,0",
                            "\tj\t$31")
                self.assertEqual(normalize.at_branch_macros(text), text)

    def test_noat_and_nomacro_regions_are_left_alone(self):
        for on, off in (("noat", "at"), ("nomacro", "macro")):
            with self.subTest(region=on):
                text = _asm(f"\t.set\t{on}", "\tslt\t$8,$4,$5", "\tbne\t$8,$0,$L3", "\tli\t$8,0", "$L3:",
                            "\tli\t$8,1", "\tj\t$31", f"\t.set\t{off}")
                self.assertEqual(normalize.at_branch_macros(text), text)

    def test_callee_saved_register_is_live_at_return(self):
        text = _asm("f:", "\tslt\t$16,$4,$5", "\tbne\t$16,$0,$L3", "\tj\t$31", "$L3:", "\tli\t$2,1", "\tj\t$31")
        self.assertEqual(normalize.at_branch_macros(text), text)

    def test_local_subroutine_call_is_live(self):
        text = _asm("f:", "\tslt\t$8,$4,$5", "\tbne\t$8,$0,$L3", "\tbal\t$L9", "\tli\t$8,0", "\tj\t$31", "$L3:",
                    "\tli\t$8,0", "\tj\t$31", "$L9:", "\taddu\t$2,$8,$0", "\tj\t$31")
        self.assertEqual(normalize.at_branch_macros(text), text)

    def test_label_defined_twice_is_unknown(self):
        text = _asm("f:", "\tslt\t$8,$4,$5", "\tbne\t$8,$0,$L3", "\tli\t$8,0", "\tj\t$31", "$L3:",
                    "\taddu\t$2,$8,$0", "\tj\t$31", "$L3:", "\tli\t$8,1", "\tj\t$31")
        self.assertEqual(normalize.at_branch_macros(text), text)

    def test_immediates(self):
        text = _asm("f:", "\tslti\t$8,$4,%lo(x)", "\tbne\t$8,$0,$L3", "\tli\t$8,0", "\tj\t$31", "$L3:",
                    "\tli\t$8,1", "\tj\t$31")
        self.assertEqual(normalize.at_branch_macros(text), text)       # a relocation: no branch macro takes it
        text = text.replace("%lo(x)", "-0x10")
        self.assertIn("\tblt\t$4,-0x10,$L3", normalize.at_branch_macros(text))


class MacroModelTest(unittest.TestCase):
    def test_branch_comparisons_through_at_are_counted(self):
        # slt $at,$4,$5; bne $at,$0; sltiu $at,$4,10; beq $at,$0
        code = struct.pack(">4I", 0x0085082A, 0x14200003, 0x2C81000A, 0x10200002)
        self.assertEqual(macros.signatures(code)["at-branch"], 2)

    def test_regs_skips_fields_that_are_not_general_registers(self):
        self.assertEqual(macros._regs(0x04810005), {4})              # bgez $4 (REGIMM rt = 1)
        self.assertEqual(macros._regs(0xC4810000), {4})              # lwc1 $f1,0($4)
        self.assertEqual(macros._regs(0x45010004), set())            # bc1t
        self.assertIn(macros.AT, macros._regs(0x44810000))            # mtc1 $1,$f0

    def test_trim_does_not_trust_a_label_at_zero(self):
        # IRIX as (IDO 5.3) records the local end label at 0: the expansion is not empty
        self.assertEqual(macros._trim([1, 2, 0], 0), ([1, 2], ""))
        self.assertEqual(macros._trim([1, 2, 0, 0], 8), ([1, 2], ""))
        self.assertEqual(macros._trim([1, 0, 0], None), ([1], ""))

    def test_constant_compares_are_not_the_branch_signature(self):
        def model(form):
            row = macros.Expansion("branch", form, "", "reorder", words=2, text=["x", "y"], uses_at=True)
            return macros.Model("as", "d", [], [row])
        self.assertNotIn("at-branch", macros.producible(model("beq-imm")))
        self.assertIn("at-branch", macros.producible(model("blt-reg")))
        self.assertIn("at-branch", macros.producible(model("bltu-imm")))


if __name__ == "__main__":
    unittest.main()
