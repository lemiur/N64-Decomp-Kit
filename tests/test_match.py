"""Matching engine tests.

Pure tests run everywhere.  ROM tests need the fixture ROMs plus the matched projects named in
roms.toml ``[truth]`` (battletanx, battletanx_ga and optionally mk64), whose ``.toolchain``
directories provide the original compilers (used read-only; builds run in temporary
directories).
"""
from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from n64dk import fixtures
from n64dk.commands.match import _mapped_comparison_addr
from n64dk.match import asn64, diff, manifest, normalize, symbols
from n64dk.match.engine import (Target, batch, manifest_row, match_source, non_axis_flags, sweep,
                                sweep_specs)
from n64dk.match.pipeline import BuildSpec, split_flags
from n64dk.match.toolchain import load_config


def words(*ws: int) -> bytes:
    return b"".join(struct.pack(">I", w) for w in ws)


NOP = 0
ADDU_V0_A0_A1 = 0x00851021  # addu $v0,$a0,$a1
ADDU_V0_A1_A0 = 0x00A41021  # addu $v0,$a1,$a0
ADDU_V1_A0_A1 = 0x00851821  # addu $v1,$a0,$a1
JR_RA = 0x03E00008
LUI_AT_8007 = 0x3C018007
LUI_AT_8008 = 0x3C018008
LW_V0_AT = 0x8C220010     # lw $v0,0x10($at)


class MappingTests(unittest.TestCase):
    def test_raw_overlay_applies_catalogue_to_link_shift_before_delta_validation(self):
        resolved = {"linked_vram": "0x800C77E8", "selected": {"authority": "raw-overlay"}}
        linked = _mapped_comparison_addr(resolved, 0x800C77E8, 0x18A0)
        self.assertEqual(linked, 0x800C5F48)
        self.assertEqual(resolved["linked_vram"], "0x800C5F48")

    def test_anchored_composite_mapping_remains_authoritative(self):
        resolved = {"linked_vram": "0x800C5F48", "selected": {"authority": "anchored-composite-region"}}
        self.assertEqual(_mapped_comparison_addr(resolved, 0x800C77E8, 0x18A0), 0x800C5F48)


class DiffTests(unittest.TestCase):
    def kinds(self, a, b):
        return [e.kind for e in diff.diff(words(*a), words(*b), 0x80000000).entries]

    def test_identical(self):
        rep = diff.diff(words(ADDU_V0_A0_A1, JR_RA, NOP), words(ADDU_V0_A0_A1, JR_RA, NOP), 0x80000000)
        self.assertEqual(rep.score, 0)
        self.assertEqual(rep.summary(), "identical")

    def test_operand_order(self):
        self.assertEqual(self.kinds([ADDU_V0_A0_A1, JR_RA], [ADDU_V0_A1_A0, JR_RA]), ["operand-order"])

    def test_register_and_swap(self):
        rep = diff.diff(words(ADDU_V0_A0_A1, JR_RA), words(ADDU_V1_A0_A1, JR_RA), 0x80000000)
        self.assertEqual([e.kind for e in rep.entries], ["register"])
        self.assertEqual(rep.register_swap, {"$v0": "$v1"})

    def test_immediate(self):
        self.assertEqual(self.kinds([LUI_AT_8007, LW_V0_AT], [LUI_AT_8008, LW_V0_AT]), ["immediate"])

    def test_hazard_nop_and_size(self):
        rep = diff.diff(words(ADDU_V0_A0_A1, NOP, JR_RA, NOP), words(ADDU_V0_A0_A1, JR_RA, NOP), 0x80000000)
        self.assertEqual([e.kind for e in rep.entries], ["hazard-nop"])
        self.assertEqual(rep.counts["size"], 1)

    def test_scheduling(self):
        rep = diff.diff(words(ADDU_V1_A0_A1, LUI_AT_8007, JR_RA), words(LUI_AT_8007, ADDU_V1_A0_A1, JR_RA), 0x80000000)
        self.assertTrue(rep.entries)
        self.assertEqual({e.kind for e in rep.entries}, {"scheduling"})


class TextTests(unittest.TestCase):
    def test_asn64_opt_in_kmc_image_table_finalize(self):
        source = (
            "\tsubu\t$sp,$sp,48\n"
            "\tsw\t$20,40($sp)\n\tmove\t$20,$5\n"
            "\tsw\t$18,32($sp)\n\tmove\t$18,$4\n"
            "\tsw\t$31,44($sp)\n\tsw\t$19,36($sp)\n"
            "\tsw\t$17,28($sp)\n\tsw\t$16,24($sp)\n"
            ".L32:\n\tmove\t$4,$0\n\tmove\t$8,$0\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tblez\t$17,.L20\n\tmove\t$7,$11\n"
            "\t.set\tmacro\n\t.set\treorder\n\n.L21:\n"
            "\tlw\t$2,40($18)\n\t#nop\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbnel\t$2,$10,.L24\n\tsll\t$6,$4,2\n"
            "\t.set\tmacro\n\t.set\treorder\n\n"
            "\tlw\t$6,0($7)\n\t#nop\n\tsll\t$6,$6,2\n.L24:\n"
            "\taddu\t$6,$20,$6\n\tlw\t$3,0($6)\n\t#nop\n"
            "\tdiv\t$5,$9,$3\n\taddu\t$3,$3,1\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbne\t$8,$0,.L32\n\tmove\t$2,$9\n"
            "\t.set\tmacro\n\t.set\treorder\n"
            "\tlw\t$31,44($sp)\n\tlw\t$20,40($sp)\n"
            "\tlw\t$19,36($sp)\n\tlw\t$18,32($sp)\n"
            "\tlw\t$17,28($sp)\n\tlw\t$16,24($sp)\n"
            "\taddu\t$sp,$sp,48\n"
        )
        out = asn64._apply_enabled_lanes(
            source, {"KMC-IMAGE-TABLE-FINALIZE"},
        )
        self.assertIn(
            "\tsubu\t$sp,$sp,56\n\tsw\t$18,40($sp)\n"
            "\tmove\t$18,$4\n\tsw\t$20,48($sp)\n",
            out,
        )
        self.assertIn(
            "\tbne\t$2,$10,.L24_MODE_INDEX\n\tmove\t$2,$4\n",
            out,
        )
        self.assertIn(
            "\tj\t.L24\n\tsll\t$2,$2,2\n",
            out,
        )
        self.assertIn("\taddu\t$6,$2,$20\n\tlw\t$2,0($6)\n", out)
        self.assertIn(
            "\tbnel\t$8,$0,.L32\n\tmove\t$4,$0\n"
            "\t.set\tmacro\n\t.set\treorder\n\n\tmove\t$2,$9\n",
            out,
        )
        self.assertIn("\tlw\t$31,52($sp)\n\tlw\t$20,48($sp)\n", out)

    def test_asn64_opt_in_kmc_image_table_update(self):
        source = (
            "\tjal\tobject_children_allocate\n.L12:\n"
            "\tlw\t$2,112($18)\n\tlw\t$3,120($18)\n"
            "\tmove\t$16,$0\n\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tblez\t$3,.L2\n\taddu\t$2,$2,2\n"
            "\t.set\tmacro\n\t.set\treorder\n\n\tsll\t$19,$2,1\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-IMAGE-TABLE-UPDATE"})
        self.assertIn(
            "\tjal\tobject_children_allocate\n\tmove\t$4,$18\n",
            out,
        )
        self.assertIn(
            ".L12:\n\tlw\t$2,112($18)\n\tmove\t$16,$0\n"
            "\tlw\t$3,120($18)\n\taddu\t$2,$2,2\n",
            out,
        )
        self.assertIn("\tblez\t$3,.L2\n\tsll\t$19,$2,1\n", out)

    def test_asn64_opt_in_kmc_image_table_process(self):
        source = (
            "\tbne\t$3,$2,.L2\n\tli\t$3,0x0000005e\t\t# 94\n"
            ".L2:\n\tlw\t$2,0($19)\n\t#nop\n\tsw\t$3,20($2)\n"
            "\t#.set\tvolatile\n\tsw\t$2,16($sp)\n"
            "\t#.set\tnovolatile\n\t#.set\tvolatile\n"
            "\tlw\t$9,16($sp)\n\t#.set\tnovolatile\n"
            "\tlw\t$2,120($19)\n\tmove\t$21,$23\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tblez\t$2,.L4\n\tmove\t$20,$0\n"
            "\t.set\tmacro\n\t.set\treorder\n"
            "\tlw\t$18,32($8)\n\tmove\t$16,$0\n\tdiv\t$17,$21,$18\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tblez\t$18,.L8\n\tmove\t$22,$9\n"
            "\t.set\tmacro\n\t.set\treorder\n\n\tmove\t$4,$19\n"
            ".L23:\n\tmove\t$5,$20\n\tmove\t$6,$16\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tjal\tfunc_800600E0\n\taddu\t$7,$18,-1\n"
            "\t.set\tmacro\n\t.set\treorder\n\n"
            "\tmult\t$16,$17\n\tmflo\t$5\n\t#nop\n\tmove\t$6,$2\n"
            "\t#.set\tvolatile\n\tlw\t$2,16($sp)\n"
            "\t#.set\tnovolatile\n\tsw\t$23,20($fp)\n\tsw\t$2,16($fp)\n"
        )
        out = asn64._apply_enabled_lanes(
            source, {"KMC-IMAGE-TABLE-PROCESS"},
        )
        self.assertIn("\tli\t$2,0x0000005e\t\t# 94\n", out)
        self.assertIn(".L2:\n\tlw\t$3,0($19)\n\t#nop\n\tsw\t$2,20($3)\n", out)
        self.assertIn(
            "\tlw\t$3,120($19)\n\tmove\t$21,$23\n\tmove\t$20,$0\n",
            out,
        )
        self.assertIn("\tblez\t$3,.L4\n\tsw\t$2,20($sp)\n", out)
        self.assertIn("\tblez\t$18,.L8\n\tmove\t$16,$0\n", out)
        self.assertIn(
            "\taddu\t$7,$18,-1\n\tsw\t$8,48($sp)\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tjal\tfunc_800600E0\n\tsw\t$9,52($sp)\n",
            out,
        )
        self.assertIn("\tlw\t$8,48($sp)\n\tlw\t$9,52($sp)\n", out)
        self.assertIn("\tlw\t$10,20($sp)\n", out)
        self.assertIn("\tsw\t$10,16($fp)\n\tsw\t$23,20($fp)\n", out)

    def test_asn64_opt_in_kmc_image_table_decode(self):
        source = (
            "\tlw\t$2,456($23)\n\tlw\t$3,88($23)\n"
            "\t#.set\tvolatile\n\tsw\t$2,16($sp)\n\t#.set\tnovolatile\n"
            "\tli\t$2,0x00000001\t\t# 1\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbne\t$3,$2,.L2\n\tli\t$2,0x000001fe\t\t# 510\n"
            "\t.set\tmacro\n\t.set\treorder\n"
            "\t#.set\tvolatile\n\tsw\t$2,24($sp)\n\t#.set\tnovolatile\n"
            "\t#.set\tvolatile\n\tlw\t$2,16($sp)\n\t#.set\tnovolatile\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tj\t.L3\n\tsw\t$3,28($2)\n"
            "\t.set\tmacro\n\t.set\treorder\n"
            ".L2:\n\t#.set\tvolatile\n\tsw\t$0,24($sp)\n"
            "\t#.set\tnovolatile\n\t#.set\tvolatile\n"
            "\tlw\t$2,16($sp)\n\t#.set\tnovolatile\n\t#nop\n"
            "\tsw\t$0,28($2)\n.L3:\nold allocation\n.L5:\n"
            "old outer\n\tjal\tfunc_8006012C\n\taddu\t$7,$22,-1\n"
            "\t.set\tmacro\n\t.set\treorder\n"
            "\t#.set\tvolatile\n\tlw\t$2,24($sp)\n"
            "\t#.set\tnovolatile\n\t#nop\n"
            "\tbeq\t$2,$0,.L17\n\taddu\t$4,$20,-1\n"
            "\tli\t$16,0x00000001\t\t# 1\n"
        )
        out = asn64._apply_enabled_lanes(
            source, {"KMC-IMAGE-TABLE-DECODE"},
        )
        self.assertIn("\tlw\t$3,88($23)\n\tlw\t$9,456($23)\n", out)
        self.assertIn("\tbne\t$3,$2,.L2\n\tsw\t$9,20($sp)\n", out)
        self.assertIn("\tli\t$9,0x000001fe\t\t# 510\n", out)
        self.assertIn("\tsw\t$0,28($sp)\n", out)
        self.assertIn("\tlw\t$9,28($sp)\n", out)
        self.assertIn("\tbeq\t$9,$0,.L17\n\tnop\n", out)

    def test_asn64_opt_in_kmc_object_table_process(self):
        source = (
            "\tlw\t$2,320($4)\n\t#.set\tvolatile\n\tsw\t$6,20($sp)\n"
            "\t#.set\tnovolatile\n\t#.set\tvolatile\n\tsw\t$7,24($sp)\n"
            "\t#.set\tnovolatile\n\t#.set\tvolatile\n\tsw\t$2,16($sp)\n"
            "\t#.set\tnovolatile\n\tlw\t$16,456($4)\n\tlw\t$18,120($4)\n"
            "\tlw\t$22,112($4)\n\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tblez\t$7,.L2\n\tsll\t$2,$22,1\n\t.set\tmacro\n\t.set\treorder\n"
            "\taddu\t$2,$2,2\n\t#.set\tvolatile\n\tlw\t$21,20($sp)\n"
            "\t#.set\tnovolatile\n\tmove\t$23,$5\n\taddu\t$17,$22,-1\n"
            "\tmult\t$17,$18\n\tmflo\t$fp\n\t#nop\n\tsw\t$2,76($sp)\n"
            "\tsw\t$17,84($sp)\n.L3:\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-OBJECT-TABLE-PROCESS"})
        self.assertIn("\tsw\t$7,28($sp)\n", out)
        self.assertIn("\tlw\t$4,320($4)\n", out)
        self.assertIn("\tblez\t$7,.L2\n\tsw\t$4,36($sp)\n", out)
        self.assertIn("\tsw\t$17,52($sp)\n", out)

    def test_asn64_opt_in_kmc_image_plane_process(self):
        source = (
            "\tsubu\t$sp,$sp,16\n\tmove\t$14,$6\n\tmove\t$24,$0\n"
            "\tmove\t$25,$4\n\tsw\t$16,8($sp)\n\tlw\t$2,308($25)\n"
            "\tlw\t$7,0($7)\n\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tblez\t$2,.L2\n\tmove\t$16,$5\n\t.set\tmacro\n\t.set\treorder\n"
            ".L3:\n\tmove\t$13,$0\n\tsll\t$15,$24,2\n"
            "\taddu\t$15,$7,$15\n.L6:\n\tlw\t$11,0($14)\n"
            "\tbne\t$13,$0,.L9\n\tlw\t$9,-4($14)\n\tj\t.L10\n"
            ".L9:\n\tlw\t$9,4($14)\n.L10:\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-IMAGE-PLANE-PROCESS"})
        self.assertIn("\tmove\t$25,$4\n\tsw\t$16,8($sp)\n", out)
        self.assertIn("\tblez\t$2,.L2\n\tmove\t$24,$0\n", out)
        self.assertIn("\tsll\t$2,$24,2\n\taddu\t$15,$2,$7\n", out)
        self.assertIn("\tbne\t$13,$0,.L9\n\tnop\n", out)

    def test_asn64_opt_in_kmc_image_plane_transform(self):
        source = (
            " #NO_APP\n\tmove\t$4,$16\n\tlw\t$2,0($2)\n"
            "\tmove\t$20,$0\n\tlw\t$18,216($16)\n"
            "\tlw\t$2,32($16)\n\tmove\t$17,$21\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tblez\t$2,.L4\n\taddu\t$18,$18,40\n"
            "\t.set\tmacro\n\t.set\treorder\n"
            "\txor\t$4,$7,$8\n\tsltu\t$4,$4,1\n"
            "\txor\t$2,$5,$6\n\tsltu\t$2,$2,1\n"
            "\tand\t$2,$4,$2\n\t.set\tnoreorder\n"
            "\t.set\tnomacro\n\tbeq\t$2,$0,.L10\n"
            "\tsll\t$3,$5,1\n\t.set\tmacro\n\t.set\treorder\n"
            "\tbne\t$2,$0,.L23\n\tdiv\t$3,$8,$7\n"
            "\taddu\t$2,$20,$21\n"
        )
        out = asn64._apply_enabled_lanes(
            source, {"KMC-IMAGE-PLANE-TRANSFORM"},
        )
        self.assertNotIn("\tmove\t$4,$16\n\tlw\t$2,0($2)\n", out)
        self.assertIn("\tblez\t$2,.L4\n\tmove\t$20,$0\n", out)
        self.assertIn("\txor\t$3,$5,$6\n\tsltu\t$3,$3,1\n", out)
        self.assertIn("\tbne\t$2,$0,.L23\n\tnop\n", out)
        self.assertIn("\taddu\t$2,$21,$20\n", out)

    def test_asn64_opt_in_kmc_menu_player_update(self):
        source = (
            "\tsubu\t$sp,$sp,56\n\tsw\t$18,32($sp)\n"
            "\tmove\t$18,$4\n\tli\t$4,0x00000003\t\t# 3\n"
            "\tsw\t$19,36($sp)\n\tmove\t$19,$5\n"
            "\tsw\t$20,40($sp)\n\tmove\t$20,$6\n"
            "\tsw\t$21,44($sp)\n\tmove\t$21,$7\n"
            "\tsw\t$31,48($sp)\n\tsw\t$17,28($sp)\n"
            "\tbne\t$3,$0,.L7\n\tmove\t$2,$0\n"
            "\tlh\t$2,D_80087A80\n\t#nop\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbgtz\t$2,.L5\n\tmove\t$5,$2\n"
            "\t.set\tmacro\n\t.set\treorder\n\n"
            "\tsll\t$2,$2,2\n\tlhu\t$3,D_80088AE4\n"
            "\taddu\t$2,$2,$5\n"
            "\taddu\t$5,$5,1\n\tsh\t$18,D_8008A740($2)\n"
            "\tsh\t$5,D_80087A80\n"
        )
        out = asn64._apply_enabled_lanes(
            source, {"KMC-MENU-PLAYER-UPDATE"},
        )
        self.assertIn("\tsw\t$19,36($sp)\n\tmove\t$19,$5\n", out)
        self.assertIn("\tbnel\t$3,$0,.L7\n\tmove\t$2,$0\n", out)
        self.assertIn("\tlh\t$4,D_80087A80\n", out)
        self.assertIn("\tsll\t$2,$4,2\n", out)
        self.assertIn(
            "\tsh\t$18,D_8008A740($2)\n\taddu\t$3,$5,1\n",
            out,
        )

    def test_asn64_opt_in_kmc_bitmap_rows_apply(self):
        source = (
            "\tlbu\t$12,0($10)\n\tandi\t$5,$5,0x00ff\n"
            "\tmult\t$5,$12\n\tmflo\t$5\n\t#nop\n"
            "\tlbu\t$2,1($10)\n\t#nop\n\tmult\t$5,$2\n"
            "\tmflo\t$5\n\t#nop\n\tsubu\t$sp,$sp,8\n"
            "\tlbu\t$9,27($sp)\n\tlbu\t$2,35($sp)\n"
            " #APP\n #NO_APP\n\tlbu\t$3,31($sp)\n"
            "\tsubu\t$9,$9,$2\n\taddu\t$9,$9,-1\n"
            "\tsubu\t$12,$9,$12\n\tslt\t$2,$12,$9\n"
            "\taddu\t$5,$5,2\n\t.set\tnoreorder\n"
            "\t.set\tnomacro\n\tbeq\t$2,$0,.L2\n"
            "\taddu\t$11,$10,$5\n\t.set\tmacro\n"
            "\t.set\treorder\n\n\tandi\t$7,$7,0x00ff\n"
            "\tmult\t$9,$7\n\tmflo\t$2\n\t#nop\n"
            "\tli\t$13,0x00000001\t\t# 1\n"
            "\tandi\t$14,$3,0x00ff\n\taddu\t$6,$6,$2\n"
            "\tbeq\t$2,$0,.L18\n\tmove\t$4,$0\n"
        )
        out = asn64._apply_enabled_lanes(
            source, {"KMC-BITMAP-ROWS-APPLY"},
        )
        self.assertIn("\tlbu\t$4,0($10)\n", out)
        self.assertIn("\tsubu\t$sp,$sp,8\n\tlbu\t$3,35($sp)\n", out)
        self.assertIn("\tmove\t$9,$2\n\tlbu\t$3,31($sp)\n", out)
        self.assertIn("\taddu\t$6,$2,$6\n", out)
        self.assertIn("\tblez\t$2,.L18\n\tmove\t$4,$0\n", out)

    def test_asn64_opt_in_kmc_image_rgb_decode(self):
        rows = (
            "\taddu\t$6,$6,1\n\tlw\t$2,0($24)\n\tlw\t$12,0($7)\n"
            "\tlw\t$3,4($24)\n\tlw\t$5,8($24)\n"
            "\taddu\t$2,$4,$2\n\tlw\t$10,0($2)\n"
            "\taddu\t$2,$4,$3\n\tlw\t$11,0($2)\n"
            "\taddu\t$4,$4,$5\n\tlw\t$8,0($4)\n"
            "\t.set\tmacro\n\t.set\treorder\n\n"
            "\taddu\t$15,$8,$25\n\taddu\t$9,$12,2\n"
            "\tlw\t$2,0($4)\n\taddu\t$8,$8,1\n\taddu\t$10,$10,1\n"
            "\taddu\t$5,$5,$2\n\taddu\t$2,$13,$5\n"
            "\tlbu\t$2,0($2)\n\taddu\t$11,$11,1\n"
        )
        out = asn64._apply_enabled_lanes(rows, {"KMC-IMAGE-RGB-DECODE"})
        self.assertIn("\tlw\t$3,0($24)\n\tlw\t$12,0($7)\n", out)
        self.assertIn(
            "\tmove\t$8,$2\n\tmove\t$11,$5\n\tmove\t$10,$9\n"
            "\taddu\t$15,$25,$8\n",
            out,
        )
        self.assertIn(
            "\taddu\t$8,$8,1\n\taddu\t$11,$11,1\n"
            "\taddu\t$5,$5,$2\n\taddu\t$5,$13,$5\n"
            "\tlbu\t$2,0($5)\n\taddu\t$10,$10,1\n",
            out,
        )

    def test_asn64_opt_in_kmc_image_rgba_decode(self):
        prologue = (
            "\tsubu\t$sp,$sp,24\n"
            "\tsw\t$20,16($sp)\n\tsw\t$19,12($sp)\n"
            "\tsw\t$18,8($sp)\n\tsw\t$17,4($sp)\n\tsw\t$16,0($sp)\n"
            "\tlw\t$2,452($4)\n\tlw\t$15,40($sp)\n"
            "\tlw\t$16,112($4)\n\tlw\t$20,8($2)\n"
            "\tlw\t$19,12($2)\n\tlw\t$18,16($2)\n\tlw\t$17,20($2)\n"
            "\tlw\t$2,320($4)\n\tmove\t$24,$5\n\taddu\t$15,$15,-1\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbltz\t$15,.L2\n\taddu\t$14,$2,255\n"
            "\t.set\tmacro\n\t.set\treorder\n"
        )
        rows = (
            "\taddu\t$6,$6,1\n\tlw\t$13,0($7)\n"
            "\tlw\t$2,0($24)\n\tlw\t$3,4($24)\n"
            "\taddu\t$2,$4,$2\n\tlw\t$10,0($2)\n"
            "\taddu\t$2,$4,$3\n\tlw\t$3,8($24)\n"
            "\tlw\t$5,12($24)\n\tlw\t$11,0($2)\n"
            "\taddu\t$2,$4,$3\n\tlw\t$12,0($2)\n"
            "\taddu\t$4,$4,$5\n\tlw\t$9,0($4)\n"
            "\t.set\tmacro\n\t.set\treorder\n\n"
            "\taddu\t$25,$9,$16\n\taddu\t$8,$13,3\n"
            "\taddu\t$12,$12,1\n\taddu\t$5,$5,$2\n"
            "\tsubu\t$5,$14,$5\n\tlbu\t$2,0($5)\n"
            "\taddu\t$10,$10,1\n\taddu\t$11,$11,1\n"
        )
        out = asn64._apply_enabled_lanes(
            prologue + rows, {"KMC-IMAGE-RGBA-DECODE"},
        )
        self.assertIn(
            "\tlw\t$15,40($sp)\n\tmove\t$24,$5\n\tsw\t$20,16($sp)\n",
            out,
        )
        self.assertIn("\tlw\t$5,320($4)\n\taddu\t$15,$15,-1\n", out)
        self.assertIn(
            "\tmove\t$9,$2\n\tmove\t$12,$5\n\tmove\t$11,$8\n"
            "\taddu\t$25,$16,$9\n",
            out,
        )
        self.assertIn("\taddu\t$11,$11,1\n\taddu\t$10,$10,1\n", out)

    def test_asn64_opt_in_kmc_image_rgb_accumulate(self):
        source = (
            "\tsll\t$3,$10,2\n\taddu\t$2,$10,1\n"
            "\tandi\t$10,$2,0x000f\n\taddu\t$2,$3,$25\n"
            "\tlw\t$6,0($2)\n\taddu\t$2,$3,$24\n\tlw\t$2,0($2)\n"
            "\taddu\t$3,$3,$15\n\taddu\t$4,$4,$6\n"
            "\taddu\t$6,$20,$4\n\taddu\t$5,$5,$2\n"
            "\taddu\t$4,$19,$5\n\tlw\t$2,0($3)\n"
            "\tlbu\t$3,0($6)\n\tlbu\t$4,0($4)\n"
            "\taddu\t$7,$7,$2\n\taddu\t$2,$18,$7\n"
            "\tlbu\t$2,0($2)\n\taddu\t$3,$3,$4\n"
            "\taddu\t$2,$2,$3\n"
        )
        out = asn64._apply_enabled_lanes(
            source, {"KMC-IMAGE-RGB-ACCUMULATE"},
        )
        self.assertIn(
            "\tsll\t$3,$10,2\n\taddu\t$2,$3,$25\n"
            "\tlw\t$6,0($2)\n\taddu\t$2,$10,1\n",
            out,
        )
        self.assertIn(
            "\tlbu\t$2,0($4)\n\taddu\t$7,$7,$3\n"
            "\taddu\t$7,$18,$7\n",
            out,
        )
        self.assertTrue(out.endswith(
            "\taddu\t$2,$2,$4\n\taddu\t$2,$2,$3\n",
        ))

    def test_asn64_opt_in_kmc_image_table_init(self):
        source = (
            ".L2:\n\tsll\t$4,$7,2\n\tli\t$6,-65536\n"
            "\tori\t$6,$6,0x492e\n\tlw\t$2,8($17)\n"
            "\taddu\t$12,$4,$2\n\tsra\t$2,$8,16\n\tsw\t$2,0($12)\n"
            "\tlw\t$2,12($17)\n\taddu\t$7,$7,1\n"
            "\tslt\t$2,$7,256\n\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbne\t$2,$0,.L2\n\taddu\t$11,$11,-22554\n"
            "\t.set\tmacro\n\t.set\treorder\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-IMAGE-TABLE-INIT"})
        self.assertIn(
            ".L2:\n\tsll\t$4,$7,2\n\taddiu\t$7,$7,1\n"
            "\tlui\t$6,0xffff\n\tori\t$6,$6,0x492e\n",
            out,
        )
        self.assertIn(
            "\tlw\t$3,8($17)\n\tlui\t$5,0x1\n"
            "\tori\t$5,$5,0xc5a2\n\tsra\t$2,$8,16\n"
            "\taddu\t$3,$4,$3\n\tsw\t$2,0($3)\n",
            out,
        )
        self.assertIn("\tslti\t$2,$7,256\n", out)

    def test_asn64_opt_in_kmc_image_block_decode(self):
        source = (
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbeq\t$5,$2,.L43\n\tli\t$2,0x00000003\t\t# 3\n"
            "\t.set\tmacro\n\t.set\treorder\n\n\tsltu\t$2,$5,3\n"
            ".L43:\n\tlw\t$3,36($16)\n"
            "\tbne\t$2,$0,.L67\n\tli\t$3,0x00000001\t\t# 1\n"
            "\tlw\t$3,120($16)\n.L67:\n\tsw\t$3,124($16)\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-IMAGE-BLOCK-DECODE"})
        self.assertIn("\tbeq\t$5,$2,.L43\n\tsltu\t$2,$5,3\n", out)
        self.assertIn(
            ".L43:\n\tlw\t$3,36($16)\n"
            "\tli\t$2,0x00000003\t\t# 3\n",
            out,
        )
        self.assertIn(
            "\tbne\t$2,$0,.L67\n\tli\t$2,0x00000001\t\t# 1\n"
            "\tlw\t$2,120($16)\n.L67:\n\tsw\t$2,124($16)\n",
            out,
        )

    def test_asn64_opt_in_kmc_frame_record_allocate(self):
        source = (
            "\tsll\t$2,$3,2\n\tlw\t$4,D_8007343C\n"
            "\taddu\t$4,$4,$2\n\tlbu\t$2,13($4)\n\tmove\t$7,$4\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-FRAME-RECORD-ALLOCATE"})
        self.assertIn("\tlw\t$4,D_8007343C\n\tsll\t$2,$3,2\n", out)
        self.assertIn(
            "\taddu\t$3,$2,$4\n\tlbu\t$2,13($3)\n\tmove\t$7,$3\n",
            out,
        )

    def test_asn64_opt_in_kmc_script_opcode_load(self):
        source = (
            "\tlh\t$7,0($2)\n"
            "\tli\t$2,0x00004000\t\t# 16384\n"
            "\tandi\t$3,$7,0xf000\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-SCRIPT-OPCODE-LOAD"})
        self.assertEqual(
            out,
            "\tlh\t$2,0($2)\n\tmove\t$7,$2\n"
            "\tandi\t$3,$2,0xf000\n"
            "\tli\t$2,0x00004000\t\t# 16384\n",
        )

    def test_asn64_opt_in_kmc_stream_chunk_argument_schedule(self):
        source = (
            "\tmove\t$18,$6\n\tbeq\t$3,$2,.L2\n\tsw\t$0,16($sp)\n"
            "\tmove\t$5,$17\n\tlw\t$2,4($2)\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-STREAM-CHUNK-ARG-SCHEDULE"})
        self.assertIn("\tbeq\t$3,$2,.L2\n\tmove\t$18,$6\n", out)
        self.assertIn(
            "\tmove\t$5,$17\n\tsw\t$0,16($sp)\n\tlw\t$2,4($2)\n",
            out,
        )

    def test_asn64_opt_in_kmc_stream_byte_read_schedule(self):
        source = (
            "\tlw\t$16,4($18)\n\tlw\t$17,0($18)\n"
            "\tlbu\t$20,0($17)\n\taddu\t$17,$17,1\n"
            "\taddu\t$16,$16,-1\n\tsll\t$20,$20,8\n"
            "\tlbu\t$4,0($17)\n\taddu\t$20,$20,$4\n"
            "\taddu\t$17,$17,1\n\taddu\t$16,$16,-1\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-STREAM-BYTE-READ-SCHEDULE"})
        self.assertIn("\tlw\t$17,0($18)\n\tlw\t$16,4($18)\n", out)
        self.assertIn(
            "\tlbu\t$2,0($17)\n\taddu\t$16,$16,-1\n"
            "\taddu\t$17,$17,1\n\tsll\t$20,$2,8\n",
            out,
        )
        self.assertIn(
            "\tlbu\t$2,0($17)\n\taddu\t$16,$16,-1\n"
            "\taddu\t$17,$17,1\n\taddu\t$20,$20,$2\n",
            out,
        )

    def test_asn64_opt_in_kmc_middle_product_words(self):
        source = (
            "\tmfhi\t$2\n\tmflo\t$3\n\tsll\t$4,$2,16\n"
            "\tsrl\t$2,$3,16\n\tor\t$4,$4,$2\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-MIDDLE-PRODUCT-WORDS"})
        self.assertEqual(
            out,
            "\tmfhi\t$14\n\tmflo\t$15\n\tsll\t$14,$14,16\n"
            "\tsrl\t$15,$15,16\n\tor\t$4,$14,$15\n",
        )

    def test_asn64_opt_in_kmc_menu_resource_state(self):
        source = (
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbeq\t$2,$3,.L2\n\tsw\t$31,20($sp)\n"
            "\t.set\tmacro\n\t.set\treorder\n\n\tmove\t$4,$5\n"
            "\tlhu\t$3,D_8008AFD4\n\t#nop\n\tsll\t$2,$3,2\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-MENU-RESOURCE-STATE"})
        self.assertIn(
            "\tmove\t$4,$5\n\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbeq\t$2,$3,.L2\n\tsw\t$31,20($sp)\n",
            out,
        )
        self.assertIn("\tlhu\t$2,D_8008AFD4\n\tsll\t$2,$2,2\n", out)

    def test_asn64_opt_in_kmc_indexed_runtime_rebuild(self):
        source = (
            "\tmove\t$4,$19\n\tmove\t$5,$17\n\tsw\t$31,36($sp)\n"
            "\tlw\t$2,D_8008B080($16)\n\t#nop\n\taddu\t$3,$2,$17\n"
            "\tmove\t$18,$17\n\tsll\t$2,$18,1\n"
            "\tsll\t$18,$18,2\n\taddu\t$3,$3,$18\n"
            "\tlw\t$2,D_8008B080($16)\n\t#nop\n\taddu\t$3,$2,$17\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-INDEXED-RUNTIME-REBUILD"})
        self.assertIn("\tmove\t$4,$19\n\tsw\t$31,36($sp)\n", out)
        self.assertEqual(out.count("\tlw\t$3,D_8008B080($16)\n\taddu\t$3,$3,$17\n"), 2)
        self.assertNotIn("\tmove\t$18,$17\n", out)
        self.assertIn("\tsll\t$2,$17,1\n", out)
        self.assertIn("\tsll\t$18,$17,2\n\taddu\t$3,$18,$3\n", out)

    def test_asn64_opt_in_kmc_match_packet_schedule(self):
        source = (
            ".L2:\n\tlbu\t$2,D_80088AF5\n"
            "\t#.set\tvolatile\n\tsb\t$4,D_8008B062\n\t#.set\tnovolatile\n"
            "\tbnel\t$5,$0,.L3\n\tori\t$2,$2,0x0080\n"
            ".L3:\n\t#.set\tvolatile\n\tsb\t$2,D_8008B063\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-MATCH-PACKET-SCHEDULE"})
        self.assertIn(
            ".L2:\n\t#.set\tvolatile\n\tsb\t$4,D_8008B062\n"
            "\t#.set\tnovolatile\n\tlbu\t$4,D_80088AF5\n",
            out,
        )
        self.assertIn("\tori\t$4,$4,0x0080\n", out)
        self.assertIn(
            ".L3:\n\tlhu\t$2,D_800876DC\n\tlhu\t$3,D_80087AAC\n"
            "\tlhu\t$6,D_80088AF0\n\tlhu\t$7,D_80088AEE\n"
            "\t#.set\tvolatile\n\tsb\t$4,D_8008B063\n",
            out,
        )

    def test_asn64_opt_in_kmc_roster_entry_schedule(self):
        source = (
            "\tandi\t$4,$4,0xffff\n\tsll\t$2,$4,5\n"
            "\tla\t$4,D_800816A7\n\taddu\t$2,$2,$4\n"
            "\tsw\t$19,28($sp)\n\tandi\t$19,$5,0xffff\n"
            "\tsll\t$3,$19,1\n\tsubu\t$3,$3,$19\n"
            "\taddu\t$20,$3,$2\n\tmove\t$4,$19\n.L10:\n"
            "\tandi\t$16,$17,0xffff\n\taddu\t$16,$20,$16\n"
            "\tlbu\t$2,0($16)\n\tbne\t$2,$0,.L10\n\tmove\t$4,$19\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-ROSTER-ENTRY-SCHEDULE"})
        self.assertIn(
            "\tsw\t$19,28($sp)\n\tandi\t$19,$5,0xffff\n"
            "\tandi\t$4,$4,0xffff\n",
            out,
        )
        self.assertIn(
            "\tsubu\t$3,$3,$19\n\taddu\t$2,$2,$4\n"
            "\taddu\t$20,$3,$2\n",
            out,
        )
        self.assertIn("\tandi\t$16,$17,0xffff\n.L10:\n\tmove\t$4,$19\n", out)
        self.assertIn("\taddu\t$2,$20,$16\n\tlbu\t$2,0($2)\n", out)

    def test_asn64_opt_in_kmc_load_hazard_move_fill(self):
        source = (
            "\tlw\t$2,D_80082DC8($16)\n"
            "\t#nop\n"
            "\tlw\t$2,8($2)\n"
            "\tmove\t$4,$0\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-LOAD-HAZARD-MOVE-FILL"})
        self.assertEqual(
            out,
            "\tlw\t$2,D_80082DC8($16)\n"
            "\tmove\t$4,$0\n"
            "\tlw\t$2,8($2)\n",
        )
    def test_asn64_opt_in_kmc_record_matrix_clear(self):
        source = (
            "\tandi\t$2,$16,0xffff\n\tsll\t$5,$2,5\n"
            "\taddu\t$5,$5,$2\n\tsll\t$5,$5,2\n"
            "\taddu\t$5,$5,$2\n\tsll\t$5,$5,2\n"
            "\taddu\t$5,$5,$2\n\tsll\t$5,$5,2\n"
            "\tsubu\t$5,$5,$2\n\tsll\t$5,$5,2\n\tandi\t$3,$4,0xffff\n"
            ".L15:\n\taddu\t$4,$4,1\n\tsll\t$2,$3,1\n"
            "\taddu\t$2,$2,$3\n\taddu\t$2,$5,$2\n"
            "\t#.set\tvolatile\n\tsb\t$0,D_80081602($2)\n\t#.set\tnovolatile\n"
            "\tandi\t$2,$4,0xffff\n\tsltu\t$2,$2,55\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbnel\t$2,$0,.L15\n\tandi\t$3,$4,0xffff\n"
            "\tandi\t$4,$17,0xffff\n.L17:\n\tandi\t$5,$16,0xffff\n"
            "\tbne\t$2,$0,.L17\n\tandi\t$4,$17,0xffff\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-RECORD-MATRIX-CLEAR"})
        self.assertIn("\tandi\t$3,$16,0xffff\n\tsll\t$2,$3,5\n", out)
        self.assertIn(
            ".L15:\n\taddu\t$4,$4,1\n\tsll\t$3,$2,1\n"
            "\taddu\t$3,$3,$2\n\taddu\t$3,$3,$5\n"
            "\tandi\t$2,$4,0xffff\n\tsltu\t$2,$2,55\n",
            out,
        )
        self.assertIn("\tsb\t$0,D_80081602($3)\n", out)
        self.assertIn("\tbne\t$2,$0,.L15\n\tandi\t$2,$4,0xffff\n", out)
        self.assertIn("\tandi\t$5,$16,0xffff\n.L17:\n\tandi\t$4,$17,0xffff\n", out)

    def test_asn64_opt_in_kmc_indexed_runtime_apply(self):
        source = (
            "\tandi\t$16,$4,0xffff\n.L2:\n\tandi\t$16,$4,0xffff\n"
            ".L3:\n\tmove\t$17,$16\n\tsll\t$16,$16,1\n"
            "\tlw\t$6,D_8008B078($16)\n\tmove\t$3,$18\n"
            "\tsll\t$2,$3,1\n\taddu\t$2,$2,$18\n\tsll\t$2,$2,2\n"
            "\tlw\t$7,D_8008B07C($16)\n\taddu\t$2,$2,$6\n"
            "\tlw\t$6,4($2)\n\tsll\t$3,$3,2\n\taddu\t$3,$3,$7\n"
            "\tjal\tfunc_8003663C\n\tsw\t$19,0($3)\n"
        )
        out = asn64._apply_enabled_lanes(
            source, {"KMC-INDEXED-RUNTIME-APPLY"},
        )
        self.assertEqual(out.count("\tandi\t$17,$4,0xffff\n"), 2)
        self.assertNotIn("\tmove\t$17,$16\n", out)
        self.assertIn("\tsll\t$16,$17,1\n", out)
        self.assertIn(
            "\tlw\t$3,D_8008B078($16)\n\tsll\t$2,$18,1\n"
            "\taddu\t$2,$2,$18\n\tsll\t$2,$2,2\n"
            "\taddu\t$2,$2,$3\n\tlw\t$3,D_8008B07C($16)\n"
            "\tlw\t$6,4($2)\n\tsll\t$2,$18,2\n"
            "\taddu\t$2,$2,$3\n\tjal\tfunc_8003663C\n"
            "\tsw\t$19,0($2)\n",
            out,
        )

    def test_asn64_opt_in_kmc_resource_best_match_schedule(self):
        source = (
            "\tlbu\t$11,D_800724DA\n\tbeq\t$11,$0,.L3\n"
            "\tlw\t$7,D_800724DC\n\tsltu\t$2,$2,$11\n"
            "\tlbu\t$11,D_800724DA\n\tbeq\t$11,$0,.L3\n"
            "\tlw\t$10,D_800724DC\n\taddu\t$5,$10,$2\n"
            "\tbnel\t$2,$4,.L13\n\tsltu\t$2,$2,$11\n"
        )
        out = asn64._apply_enabled_lanes(
            source, {"KMC-RESOURCE-BEST-MATCH-SCHEDULE"},
        )
        self.assertIn(
            "\tlbu\t$2,D_800724DA\n\tbeq\t$2,$0,.L3\n"
            "\tlw\t$7,D_800724DC\n\tmove\t$5,$2\n"
            "\tsltu\t$2,$2,$5\n",
            out,
        )
        self.assertIn(
            "\tlbu\t$2,D_800724DA\n\tbeq\t$2,$0,.L3\n"
            "\tlw\t$10,D_800724DC\n\tmove\t$7,$2\n"
            "\taddu\t$5,$2,$10\n\tbnel\t$4,$2,.L13\n"
            "\tsltu\t$2,$2,$7\n",
            out,
        )

    def test_asn64_opt_in_kmc_record_lookup_argument_schedule(self):
        source = (
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tjal\tfunc_80028A58\n\tandi\t$4,$16,0xffff\n"
            "\t.set\tmacro\n\t.set\treorder\n\n"
            "\tmove\t$4,$16\n #APP\n #NO_APP\n"
            "\tandi\t$16,$2,0xffff\n #APP\n #NO_APP\n"
            "\tandi\t$4,$4,0xffff\n"
        )
        out = asn64._apply_enabled_lanes(
            source, {"KMC-RECORD-LOOKUP-ARG-SCHEDULE"},
        )
        self.assertIn(
            "\tandi\t$16,$16,0xffff\n\t.set\tnoreorder\n"
            "\t.set\tnomacro\n\tjal\tfunc_80028A58\n"
            "\tmove\t$4,$16\n\t.set\tmacro\n\t.set\treorder\n\n"
            "\tmove\t$4,$16\n #APP\n #NO_APP\n"
            "\tandi\t$16,$2,0xffff\n #APP\n #NO_APP\n",
            out,
        )
        self.assertNotIn("\tandi\t$4,$4,0xffff\n", out)

    def test_asn64_opt_in_kmc_signed_component_schedule(self):
        source = (
            "\tlb\t$2,2($5)\n\t#nop\n\tnor\t$3,$0,$2\n"
            "\tlb\t$2,0($5)\n\tsubu\t$sp,$sp,48\n\t.set\tnoreorder\n"
            ".L4:\n\taddu\t$4,$4,16\n"
            "\t#.set\tvolatile\n\tsb\t$2,0($4)\n\t#.set\tnovolatile\n"
            "\t#.set\tvolatile\n\tsb\t$2,0($4)\n\t#.set\tnovolatile\n"
            "\tlb\t$2,1($5)\n\t#nop\n\t.set\tnoreorder\n"
            ".L7:\n\tla\t$4,D_80078429\n\tmove\t$6,$4\n"
            "\t#.set\tvolatile\n\tsb\t$2,0($6)\n\t#.set\tnovolatile\n"
            "\t#.set\tvolatile\n\tsb\t$2,0($6)\n\t#.set\tnovolatile\n"
            "\tlb\t$2,2($5)\n\t#nop\n\t.set\tnoreorder\n"
            ".L10:\n\tla\t$4,D_8007842A\n\tsb\t$2,0($4)\n"
            "\t#.set\tvolatile\n\tsb\t$2,0($4)\n\t#.set\tnovolatile\n"
        )
        out = asn64._apply_enabled_lanes(
            source, {"KMC-SIGNED-COMPONENT-SCHEDULE"},
        )
        self.assertIn(
            "\tlb\t$2,2($5)\n\tsubu\t$sp,$sp,48\n"
            "\tnor\t$3,$0,$2\n",
            out,
        )
        self.assertIn(
            "\tlb\t$2,0($5)\n\taddu\t$4,$4,16\n"
            "\tmove\t$6,$4\n\t.set\tnoreorder\n",
            out,
        )
        self.assertIn(
            ".L4:\n\t#.set\tvolatile\n\tsb\t$2,0($6)\n",
            out,
        )
        self.assertIn(
            "\tlb\t$2,1($5)\n\tla\t$4,D_80078429\n"
            "\tmove\t$6,$4\n\t.set\tnoreorder\n",
            out,
        )
        self.assertIn(".L7:\n\t#.set\tvolatile\n\tsb\t$2,0($6)\n", out)
        self.assertIn("\tsb\t$2,0($4)\n\t#.set\tnovolatile\n\tlb\t$2,2($5)\n", out)
        self.assertIn(
            "\tlb\t$2,2($5)\n\tla\t$4,D_8007842A\n"
            "\tmove\t$5,$4\n\t.set\tnoreorder\n",
            out,
        )
        self.assertIn(".L10:\n\tsb\t$2,0($5)\n", out)

    def test_asn64_opt_in_kmc_object_cycle_call_schedule(self):
        source = (
            "\taddu\t$6,$6,40\n\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tjal\tfunc_80058F2C\n\taddu\t$6,$20,$6\n"
            "\t.set\tmacro\n\t.set\treorder\n\n"
            "\tmove\t$4,$16\n\tjal\tfunc_80058F2C\n\tnop\n"
            "\taddu\t$21,$21,4\n\tsw\t$0,20($22)\n"
        )
        out = asn64._apply_enabled_lanes(
            source, {"KMC-OBJECT-CYCLE-CALL-SCHEDULE"},
        )
        self.assertIn(
            "\taddu\t$6,$6,40\n\taddu\t$6,$20,$6\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tjal\tfunc_80058F2C\n\taddu\t$21,$21,4\n",
            out,
        )
        self.assertNotIn(
            "\tjal\tfunc_80058F2C\n\taddu\t$6,$20,$6\n", out,
        )

    def test_asn64_opt_in_kmc_memory_pool_initialize(self):
        source = (
            "\tsll\t$3,$4,16\n\tla\t$5,D_8007A400\n"
            "\tsra\t$3,$3,14\n\taddu\t$5,$3,$5\n"
            "\tlw\t$16,0($5)\n\tlw\t$2,D_8007A3F4($3)\n"
            "\tsw\t$16,D_8007A410($3)\n\tsw\t$16,D_8007A40C($3)\n"
            "\tsw\t$16,D_8007A414($3)\n\tsw\t$2,0($16)\n"
            "\tli\t$2,0x00000001\n\tsh\t$2,8($16)\n\tli\t$2,-1\n"
            "\tsw\t$2,20($16)\n\tsw\t$2,24($16)\n"
            "\tsw\t$2,16($16)\n\tsw\t$2,12($16)\n"
            "\tlw\t$2,D_8007A3F8($3)\n\t#nop\n\tsw\t$2,4($16)\n"
            "\tlw\t$17,0($5)\n"
            "\tsll\t$18,$18,16\n\tsra\t$18,$18,14\n"
            "\tsw\t$4,D_8007A408($18)\n"
        )
        out = asn64._apply_enabled_lanes(
            source, {"KMC-MEMORY-POOL-INITIALIZE"},
        )
        self.assertIn(
            "\tsll\t$2,$4,16\n\tla\t$3,D_8007A400\n"
            "\tsra\t$2,$2,14\n\taddu\t$3,$2,$3\n",
            out,
        )
        self.assertIn(
            "\tlw\t$16,0($3)\n\tlw\t$5,D_8007A3F4($2)\n"
            "\tsw\t$16,D_8007A410($2)\n",
            out,
        )
        self.assertIn(
            "\tlw\t$5,D_8007A3F8($2)\n\tli\t$2,0x00000001\n"
            "\tsh\t$2,8($16)\n\tli\t$2,-1\n"
            "\tsw\t$2,24($16)\n\tsw\t$2,20($16)\n",
            out,
        )
        self.assertIn(
            "\tsll\t$2,$18,16\n\tsra\t$2,$2,14\n"
            "\tsw\t$4,D_8007A408($2)\n",
            out,
        )

    def test_asn64_opt_in_kmc_memory_pool_region_schedule(self):
        source = (
            "\t#nop\n\tsll\t$2,$3,3\n\tsubu\t$2,$2,$3\n\tsll\t$2,$2,2\n"
            " #APP\n #NO_APP\n\tsll\t$4,$3,1\n #APP\n #NO_APP\n"
            "\tnor\t$6,$0,$7\n\tmove\t$19,$4\n\taddu\t$5,$5,$7\n"
            "\tmove\t$7,$2\n\tand\t$5,$5,$6\n #APP\n #NO_APP\n"
            "\taddu\t$5,$5,$7\n\taddu\t$5,$5,$19\n\tsw\t$5,D_8007A3F4\n"
            "\tlw\t$2,4($18)\n\t#nop\n\tsubu\t$2,$2,$5\n"
            "\tsw\t$2,D_8007A3F8\n\tlw\t$2,0($18)\n\t#nop\n"
            "\taddu\t$2,$2,$19\n\tsw\t$2,D_8007A400\n"
            ".L2:\n\tmove\t$4,$0\n\tsll\t$17,$17,2\n\tla\t$16,D_8007A3F4\n"
            "\taddu\t$16,$17,$16\n\taddu\t$19,$2,$19\n"
            "\tsw\t$19,D_8007A400($17)\n\tsw\t$2,D_8007A404($17)\n"
            "\tlw\t$2,4($18)\n\tlw\t$3,0($16)\n\t#nop\n"
            "\tsubu\t$2,$2,$3\n\tsw\t$2,D_8007A3F8($17)\n"
        )
        out = asn64._apply_enabled_lanes(
            source, {"KMC-MEMORY-POOL-REGION-SCHEDULE"},
        )
        self.assertIn(
            "\tnor\t$6,$0,$7\n\taddu\t$5,$5,$7\n"
            "\tsll\t$2,$3,3\n\tsubu\t$2,$2,$3\n\tsll\t$2,$2,2\n",
            out,
        )
        self.assertIn(
            "\taddu\t$3,$5,$2\n\taddu\t$3,$3,$4\n"
            "\tsw\t$3,D_8007A3F4\n",
            out,
        )
        self.assertIn(
            ".L2:\n\tmove\t$4,$0\n\tla\t$16,D_8007A3F4\n"
            "\tsll\t$17,$17,2\n",
            out,
        )
        self.assertIn(
            "\tlw\t$4,4($18)\n\tlw\t$5,0($16)\n"
            "\taddu\t$3,$2,$19\n\tsw\t$3,D_8007A400($17)\n"
            "\tsw\t$2,D_8007A404($17)\n\tsubu\t$4,$4,$5\n"
            "\tsw\t$4,D_8007A3F8($17)\n",
            out,
        )

    def test_asn64_opt_in_kmc_row_group_loop_schedule(self):
        source = (
            "\taddu\t$16,$16,$20\n"
            "\tlw\t$2,308($17)\n"
            "\tsll\t$3,$20,2\n"
        )
        out = asn64._apply_enabled_lanes(
            source, {"KMC-ROW-GROUP-LOOP-SCHEDULE"},
        )
        self.assertEqual(
            out,
            "\tsll\t$3,$20,2\n"
            "\tlw\t$2,308($17)\n"
            "\taddu\t$16,$16,$20\n",
        )

    def test_asn64_opt_in_kmc_lookup_decode_schedule(self):
        source = (
            "\tlw\t$3,112($23)\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tblez\t$7,.L2\n\tmove\t$20,$0\n"
            "\t.set\tmacro\n\t.set\treorder\n\n"
            "\t#.set\tvolatile\n\tsw\t$3,28($sp)\n"
            "\t#.set\tnovolatile\n"
            "\tlw\t$2,0($2)\n\tsll\t$3,$6,6\n"
            "\taddu\t$3,$2,$3\n"
            "\tlw\t$2,20($sp)\n\tslt\t$2,$20,$2\n"
        )
        out = asn64._apply_enabled_lanes(
            source, {"KMC-LOOKUP-DECODE-SCHEDULE"},
        )
        self.assertIn(
            "\tlw\t$8,112($23)\n\tmove\t$20,$0\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tblez\t$7,.L2\n\tsw\t$8,28($sp)\n",
            out,
        )
        self.assertIn(
            "\tlw\t$3,0($2)\n\tsll\t$2,$6,6\n"
            "\taddu\t$3,$3,$2\n\tlw\t$8,20($sp)\n"
            "\tslt\t$2,$20,$8\n",
            out,
        )

    def test_asn64_opt_in_kmc_dynamic_alloc_argument_schedule(self):
        source = (
            "\tlw\t$2,4($19)\n"
            "\t#nop\n"
            "\tlw\t$2,0($2)\n"
            "\tli\t$5,0x00000001\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tjal\t$31,$2\n"
            "\tsll\t$6,$16,5\n"
            "\t.set\tmacro\n\t.set\treorder\n\n"
            "\tmove\t$18,$0\n"
        )
        out = asn64._apply_enabled_lanes(
            source, {"KMC-DYNAMIC-ALLOC-ARG-SCHEDULE"},
        )
        self.assertIn(
            "\tli\t$5,0x00000001\n\tlw\t$2,0($2)\n"
            "\tsll\t$6,$16,5\n\t.set\tnoreorder\n"
            "\t.set\tnomacro\n\tjal\t$31,$2\n\tmove\t$18,$0\n",
            out,
        )

    def test_asn64_opt_in_kmc_early_a1_constant(self):
        source = (
            "\tsubu\t$sp,$sp,32\n"
            "\tsw\t$17,20($sp)\n"
            "\tmove\t$17,$0\n"
            "\tsw\t$18,24($sp)\n"
            "\tla\t$18,D_800748FC\n"
            "\tli\t$5,0x00000006\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-EARLY-A1-CONSTANT"})
        self.assertIn(
            "\tsubu\t$sp,$sp,32\n\tli\t$5,0x00000006\n"
            "\tsw\t$17,20($sp)\n",
            out,
        )

    def test_asn64_opt_in_kmc_jalr_loop_argument_schedule(self):
        source = (
            "\tlw\t$2,4($16)\n"
            "\tmove\t$4,$16\n"
            "\tlw\t$2,0($2)\n"
            "\tli\t$5,0x00000001\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tjal\t$31,$2\n"
            "\tli\t$6,0x00000100\n"
            "\t.set\tmacro\n\t.set\treorder\n\n"
            "\taddu\t$17,$17,1\n"
        )
        out = asn64._apply_enabled_lanes(
            source, {"KMC-JALR-LOOP-ARG-SCHEDULE"},
        )
        self.assertIn(
            "\tmove\t$4,$16\n\tli\t$5,0x00000001\n"
            "\tlw\t$2,0($2)\n\tli\t$6,0x00000100\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tjal\t$31,$2\n\taddu\t$17,$17,1\n",
            out,
        )

    def test_asn64_opt_in_kmc_switch_shared_case_split(self):
        source = (
            "\tbeq\t$4,$2,.L9\n"
            "\tbeq\t$4,$2,.L9\n"
            ".L9:\n"
            "\tli\t$17,0x00000009\n"
            ".L11:\n"
            "\tli\t$16,0x0000000d\n"
        )
        out = asn64._apply_enabled_lanes(
            source, {"KMC-SWITCH-SHARED-CASE-SPLIT"},
        )
        self.assertIn("beq\t$4,$2,.L__kmc_shared_case_split", out)
        self.assertEqual(out.count("beq\t$4,$2,.L9"), 1)
        self.assertIn(
            ".L__kmc_shared_case_split:\n\t.set\tnoreorder\n"
            "\t.set\tnomacro\n\tj\t.L11\n\tli\t$17,0x00000009\n",
            out,
        )

    def test_asn64_reorder_branch_gets_nop(self):
        out = asn64.convert("\t.text\n\tbeq\t$4,$0,$L2\n\taddu\t$2,$3,$4\n$L2:\n\tj\t$31\n")
        self.assertIn("\t.set noreorder\n", out)
        body = out.split(".set noreorder\n", 1)[1]
        self.assertIn("beq\t$4,$0,$L2\n\tnop\n", body)

    def test_asn64_reorder_branch_nop_before_move_can_be_disabled(self):
        out = asn64.convert(
            "\t.text\n\tbeq\t$4,$0,$L2\n\tmove\t$2,$3\n$L2:\n\tj\t$31\n",
            {"REORDER-BRANCH-NOP-MOVE"},
        )
        body = out.split(".set noreorder\n", 1)[1]
        self.assertIn("beq\t$4,$0,$L2\n\tmove\t$2,$3\n", body)
        self.assertNotIn("beq\t$4,$0,$L2\n\tnop\n", body)

    def test_asn64_opt_in_kmc_prologue_s2_save_order(self):
        source = (
            "\tsw\t$31,56($sp)\n"
            "\tsw\t$21,44($sp)\n"
            "\tsw\t$18,32($sp)\n"
            "\tsw\t$17,28($sp)\n"
        )
        self.assertIn("$31,56($sp)\n\tsw\t$21,44($sp)\n\tsw\t$18,32($sp)",
                      asn64.convert(source))
        out = asn64.convert(source, enabled={"KMC-PROLOGUE-S2-SAVE-ORDER"})
        self.assertIn("$18,32($sp)\n\tsw\t$31,56($sp)\n\tsw\t$21,44($sp)", out)

    def test_asn64_opt_in_kmc_early_s4_save(self):
        source = (
            "\tsw\t$18,24($sp)\n"
            "\tmove\t$18,$6\n"
            "\tli\t$2,0x3b9a0000\n"
            "\tori\t$2,$2,0xc9f0\n"
            "\tsw\t$31,36($sp)\n"
            "\tsw\t$20,32($sp)\n"
        )
        out = asn64.convert(source, enabled={"KMC-PROLOGUE-EARLY-S4-SAVE"})
        self.assertIn(
            "$20,32($sp)\n\tsw\t$18,24($sp)\n\tmove\t$18,$6\n"
            "\tli\t$2,0x3b9a0000\n\tori\t$2,$2,0xc9f0\n\tsw\t$31,36($sp)",
            out,
        )

    def test_asn64_opt_in_kmc_early_s5_save(self):
        source = (
            "\tsw\t$18,24($sp)\n"
            "\tmove\t$18,$6\n"
            "\tli\t$2,0x3b9a0000\n"
            "\tori\t$2,$2,0xc9f0\n"
            "\tsw\t$31,44($sp)\n"
            "\tsw\t$22,40($sp)\n"
            "\tsw\t$21,36($sp)\n"
        )
        out = asn64.convert(source, enabled={"KMC-PROLOGUE-EARLY-S5-SAVE"})
        self.assertIn(
            "$21,36($sp)\n\tsw\t$18,24($sp)\n\tmove\t$18,$6\n"
            "\tli\t$2,0x3b9a0000\n\tori\t$2,$2,0xc9f0\n"
            "\tsw\t$31,44($sp)\n\tsw\t$22,40($sp)",
            out,
        )

    def test_asn64_opt_in_kmc_jalr_argument_before_store_delay(self):
        source = (
            "\t.set\tnoreorder\n"
            "\t.set\tnomacro\n"
            "\tjal\t$31,$2\n"
            "\taddu\t$5,$16,48\n"
            "\t.set\tmacro\n"
            "\t.set\treorder\n\n"
            "\tsw\t$0,40($16)\n"
        )
        out = asn64.convert(source, enabled={"KMC-JALR-ARG-BEFORE-STORE-DELAY"})
        arg = out.index("\taddu\t$5,$16,48\n")
        call = out.index("\tjal\t$31,$2\n")
        store = out.index("\tsw\t$0,40($16)\n")
        self.assertLess(arg, call)
        self.assertEqual(store, call + len("\tjal\t$31,$2\n"))

    def test_asn64_opt_in_kmc_shift_before_move(self):
        source = "\tmove\t$18,$3\n\tsll\t$2,$18,2\n"
        out = asn64.convert(source, enabled={"KMC-SHIFT-BEFORE-MOVE"})
        self.assertIn("\tsll\t$2,$3,2\n\tmove\t$18,$3\n", out)

    def test_asn64_kmc_shift_before_move_preserves_dependency(self):
        source = "\tmove\t$18,$3\n\tsll\t$3,$18,2\n"
        out = asn64.convert(source, enabled={"KMC-SHIFT-BEFORE-MOVE"})
        self.assertIn(source, out)

    def test_asn64_opt_in_kmc_shift_into_branch_delay(self):
        source = (
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbeq\t$2,$0,.Ldone\n\tmove\t$18,$3\n"
            "\t.set\tmacro\n\t.set\treorder\n\n"
            "\tsll\t$2,$18,2\n"
        )
        out = asn64.convert(source, enabled={"KMC-SHIFT-BEFORE-MOVE"})
        self.assertIn("\tbeq\t$2,$0,.Ldone\n\tsll\t$2,$3,2\n", out)
        self.assertIn("\tmove\t$18,$3\n", out)

    def test_asn64_opt_in_kmc_late_v0_argument(self):
        source = (
            "\tsubu\t$sp,$sp,40\n\tmove\t$2,$6\n"
            "\tsw\t$17,20($sp)\n\tmove\t$17,$4\n"
            "\tandi\t$5,$19,0xffff\n\taddu\t$6,$2,21\n"
        )
        out = asn64.convert(source, enabled={"KMC-LATE-V0-ARG"})
        self.assertIn(
            "\tsubu\t$sp,$sp,40\n\tsw\t$17,20($sp)\n\tmove\t$17,$4\n"
            "\tandi\t$5,$19,0xffff\n\tmove\t$2,$6\n\taddu\t$6,$2,21\n",
            out,
        )

    def test_asn64_opt_in_kmc_shared_div3_constant(self):
        source = (
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbeq\t$3,$2,.L13\n\tsll\t$2,$16,16\n"
            "\t.set\tmacro\n\t.set\treorder\n"
            ".L11:\n\tsll\t$2,$16,16\n\tsra\t$2,$2,15\n"
            "\tli\t$3,0x55550000\n\tori\t$3,$3,0x5556\n"
            ".L13:\n\tsra\t$2,$2,15\n"
            "\tli\t$3,0x55550000\n\tori\t$3,$3,0x5556\n"
        )
        out = asn64.convert(source, enabled={"KMC-SHARED-DIV3-CONSTANT"})
        self.assertEqual(out.count("li\t$3,0x55550000"), 2)
        self.assertIn("beq\t$3,$2,.L13\n\tli\t$3,0x55550000\n", out)
        self.assertIn(
            ".L11:\n\tli\t$3,0x55550000\n\tori\t$3,$3,0x5556\n"
            "\tsll\t$2,$16,16\n",
            out,
        )
        self.assertIn(".L13:\n\tori\t$3,$3,0x5556\n\tsll\t$2,$16,16\n", out)

    def test_asn64_opt_in_kmc_compact_frame(self):
        source = (
            "\tsubu\t$sp,$sp,48\n\tsw\t$16,32($sp)\n"
            "\tsw\t$31,40($sp)\n\tlw\t$3,16($sp)\n"
            "\tlw\t$31,40($sp)\n\taddu\t$sp,$sp,48\n"
        )
        out = asn64.convert(source, enabled={"KMC-COMPACT-FRAME-8"})
        self.assertIn("subu\t$sp,$sp,40", out)
        self.assertIn("sw\t$16,24($sp)", out)
        self.assertIn("sw\t$31,32($sp)", out)
        self.assertIn("lw\t$3,16($sp)", out)
        self.assertIn("addu\t$sp,$sp,40", out)

    def test_asn64_opt_in_kmc_index_table_schedule(self):
        source = (
            "\tla\t$3,D_table\n\tandi\t$19,$19,0xffff\n"
            "\tsll\t$18,$19,1\n\taddu\t$18,$18,$3\n"
            "\taddu\t$3,$3,8524\n\tlhu\t$4,0($18)\n"
            "\tandi\t$20,$20,0xffff\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-INDEX-TABLE-SCHEDULE"})
        self.assertIn("\tandi\t$19,$19,0xffff\n\tla\t$3,D_table\n", out)
        self.assertIn(
            "\tlhu\t$4,0($18)\n\tandi\t$20,$20,0xffff\n"
            "\taddu\t$3,$3,8524\n",
            out,
        )

    def test_asn64_opt_in_kmc_late_s0_setup(self):
        source = (
            "\tsubu\t$sp,$sp,48\n\tsw\t$16,16($sp)\n"
            " #APP\n #NO_APP\n\taddu\t$16,$6,-1\n"
            "\tsw\t$18,24($sp)\n\tsw\t$20,32($sp)\n"
            "\tlhu\t$2,D_table($2)\n\tsll\t$5,$21,1\n"
            "\taddu\t$5,$5,16\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-LATE-S0-SETUP"})
        self.assertIn("\tsw\t$20,32($sp)\n\tsw\t$16,16($sp)\n", out)
        self.assertIn("\taddu\t$5,$5,16\n\taddu\t$16,$6,-1\n", out)

    def test_asn64_opt_in_kmc_preload_bound_save_delay(self):
        source = (
            "\tsubu\t$sp,$sp,32\n\tsw\t$17,20($sp)\n"
            "\tmove\t$17,$4\n\tsw\t$16,16($sp)\n\tmove\t$16,$0\n"
            "\tsw\t$31,24($sp)\n #APP\n #NO_APP\n"
            "\tlhu\t$2,D_bound\n\tsltu\t$2,$16,$2\n\tbeq\t$2,$0,.Ldone\n"
        )
        out = asn64.convert(source, enabled={"KMC-PRELOAD-BOUND-SAVE-DELAY"})
        bound = out.index("lhu\t$2,D_bound")
        stack = out.index("subu\t$sp,$sp,32")
        branch = out.index("beq\t$2,$0,.Ldone")
        save = out.index("sw\t$31,24($sp)")
        self.assertLess(bound, stack)
        self.assertIn("beq\t$2,$0,.Ldone\n\tsw\t$31,24($sp)\n", out)

    def test_asn64_opt_in_kmc_table_s0_save_order(self):
        source = (
            "\tsw\t$20,32($sp)\n\tsw\t$16,16($sp)\n"
            "\tsw\t$17,20($sp)\n\tsw\t$31,44($sp)\n"
            "\tsw\t$19,28($sp)\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-TABLE-S0-SAVE-ORDER"})
        self.assertIn(
            "\tsw\t$20,32($sp)\n\tsw\t$17,20($sp)\n"
            "\tsw\t$31,44($sp)\n\tsw\t$19,28($sp)\n"
            "\tsw\t$16,16($sp)\n",
            out,
        )

    def test_asn64_opt_in_kmc_jal_argument_delay(self):
        source = (
            "\tmove\t$4,$6\n\taddu\t$2,$3,8\n\taddu\t$3,$3,16\n"
            "\tsw\t$2,D_output\n\tsw\t$3,D_output\n"
            "\tjal\tcallback\n\tli\t$6,0xb9000000\n"
        )
        out = asn64.convert(source, enabled={"KMC-JAL-ARG-DELAY"})
        self.assertIn("\tjal\tcallback\n\tmove\t$4,$6\n", out)
        self.assertLess(out.index("sw\t$3,D_output"), out.index("jal\tcallback"))

    def test_asn64_opt_in_kmc_staged_saved_registers(self):
        source = (
            "\tmove\t$19,$4\n\tsw\t$31,32($sp)\n\tsw\t$18,24($sp)\n"
            "\tsw\t$17,20($sp)\n\tsw\t$16,16($sp)\n"
            " #APP\n #NO_APP\n\tmove\t$17,$5\n"
            "\tandi\t$3,$17,0xffff\n\tmove\t$16,$0\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-STAGED-SAVED-REGISTERS"})
        self.assertIn("\tmove\t$19,$4\n\tsw\t$17,20($sp)\n\tmove\t$17,$5\n", out)
        self.assertIn(
            "\tsw\t$16,16($sp)\n\tmove\t$16,$0\n"
            "\tsw\t$31,32($sp)\n\tsw\t$18,24($sp)\n",
            out,
        )

    def test_asn64_opt_in_kmc_jal_argument_order(self):
        source = (
            "\tandi\t$5,$5,0xffff\n\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tjal\tlookup\n\tandi\t$4,$19,0xffff\n"
            "\t.set\tmacro\n\t.set\treorder\n"
        )
        out = asn64.convert(source, enabled={"KMC-JAL-ARG-ORDER"})
        first = out.index("andi\t$4,$19,0xffff")
        call = out.index("jal\tlookup")
        second = out.index("andi\t$5,$5,0xffff")
        self.assertLess(first, call)
        self.assertEqual(second, call + len("jal\tlookup\n\t"))

    def test_asn64_opt_in_kmc_strip_loop_schedule(self):
        source = (
            "\tandi\t$7,$2,0xffff\n\tmove\t$8,$0\n\tli\t$9,0xb1000000\n"
            "\taddu\t$6,$6,2\n\taddu\t$16,$16,2\n"
            "\tandi\t$2,$8,0xffff\n\tsltu\t$2,$2,6\n"
            "\tbne\t$2,$0,.Lloop\n\tlw\t$31,20($sp)\n"
        )
        out = asn64.convert(source, enabled={"KMC-STRIP-LOOP-SCHEDULE"})
        self.assertIn("\tmove\t$8,$0\n\tandi\t$7,$2,0xffff\n", out)
        self.assertIn("\tbne\t$2,$0,.Lloop\n\taddu\t$16,$16,2\n", out)

    def test_asn64_opt_in_kmc_audio_slot_schedule(self):
        source = (
            "\tsll\t$2,$2,2\n\tlbu\t$3,31($4)\n"
            "\taddu\t$5,$5,$4\n\taddu\t$3,$4,$2\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbeq\t$2,$7,.Linc\n\tli\t$2,0x00000001\n"
            "\t.set\tmacro\n\t.set\treorder\n\n #APP\n #NO_APP\n"
            "\tj\t.Ldone\n.Linc:\n\taddu\t$2,$6,1\n"
            ".Lnext:\n\tmove\t$6,$2\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-AUDIO-SLOT-SCHEDULE"})
        self.assertIn("\tlbu\t$3,31($4)\n\tsll\t$2,$2,2\n", out)
        self.assertIn("\taddu\t$5,$4,$5\n\taddu\t$3,$2,$4\n", out)
        self.assertIn("\tbeq\t$2,$7,.Lnext\n\taddu\t$2,$6,1\n", out)
        self.assertIn("\tj\t.Ldone\n\tli\t$2,0x00000001\n", out)
        self.assertNotIn(".Linc:", out)

    def test_asn64_opt_in_kmc_fp_pair_words(self):
        source = (
            "\ts.d\t$f22,8($sp)\n"
            "\tl.d\t$f20,0($sp)\n"
            "\tl.d\t$f4,0($t0)\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-FP-PAIR-WORDS"})
        self.assertEqual(
            out,
            "\ts.s\t$f23,8($sp)\n\ts.s\t$f22,12($sp)\n"
            "\tl.s\t$f21,0($sp)\n\tl.s\t$f20,4($sp)\n"
            "\tl.s\t$f5,0($t0)\n\tl.s\t$f4,4($t0)\n",
        )

    def test_asn64_opt_in_kmc_play_queue_schedule(self):
        source = (
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbeq\t$2,$0,.Lwrap\n\taddu\t$2,$4,1\n"
            "\t.set\tmacro\n\t.set\treorder\n\n\tmove\t$5,$3\n"
            "\tlbu\t$2,D_80088AFA\n #APP\n #NO_APP\n\t#nop\n"
            "\tmove\t$5,$2\n\taddu\t$4,$5,1\n"
            "\tli\t$2,-2004353024\n\tori\t$2,$2,0x8889\n"
            "\tmult\t$4,$2\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-PLAY-QUEUE-SCHEDULE"})
        self.assertIn("\tbeq\t$2,$0,.Lwrap\n\tmove\t$5,$3\n", out)
        self.assertIn("\t.set\treorder\n\n\taddu\t$2,$4,1\n", out)
        self.assertIn(
            "\tli\t$3,-2004353024\n\tori\t$3,$3,0x8889\n"
            "\tmove\t$5,$2\n\taddu\t$4,$5,1\n\tmult\t$4,$3\n",
            out,
        )

    def test_asn64_opt_in_kmc_archive_decode_schedule(self):
        source = (
            "\tsw\t$fp,680($sp)\n\tsh\t$7,646($sp)\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tjal\tfunc_80063530\n\tmove\t$fp,$6\n"
            "\tmove\t$23,$3\n\tsltu\t$2,$2,$23\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbeq\t$2,$0,.L6\n\tmove\t$17,$4\n"
            "\tmove\t$4,$22\n\tlhu\t$2,646($sp)\n"
            "\tandi\t$5,$17,0xffff\n\tandi\t$6,$23,0xffff\n"
            "\tsll\t$7,$fp,16\n\tsll\t$2,$2,16\n"
            "\tsra\t$2,$2,16\n\tsw\t$2,16($sp)\n"
            "\tlhu\t$2,638($sp)\n\tsra\t$7,$7,16\n"
            "\tli\t$3,0x00000010\n\tsw\t$3,24($sp)\n"
            "\tsll\t$2,$2,16\n\tsra\t$2,$2,16\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tjal\tfunc_80017840\n\tsw\t$2,20($sp)\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-ARCHIVE-DECODE-SCHEDULE"})
        self.assertIn("\tsw\t$fp,680($sp)\n\tmove\t$fp,$6\n", out)
        self.assertIn("\tjal\tfunc_80063530\n\tsh\t$7,646($sp)\n", out)
        self.assertIn(
            "\tmove\t$17,$4\n\tsltu\t$2,$2,$3\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbeq\t$2,$0,.L6\n\tmove\t$23,$3\n",
            out,
        )
        self.assertIn(
            "\tlhu\t$8,646($sp)\n\tandi\t$6,$23,0xffff\n"
            "\tsll\t$7,$fp,16\n\tsll\t$2,$8,16\n"
            "\tlhu\t$8,638($sp)\n",
            out,
        )
        self.assertIn(
            "\tsw\t$2,20($sp)\n\tli\t$2,0x00000010\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tjal\tfunc_80017840\n\tsw\t$2,24($sp)\n",
            out,
        )

    def test_asn64_opt_in_kmc_object_vector_schedule(self):
        source = (
            "\tmflo\t$3\n\t#nop\n\tli\t$8,0x00040000\n"
            "\t#.set\tvolatile\n\tsw\t$3,0($5)\n"
            "\t#.set\tnovolatile\n\t#.set\tvolatile\n"
            "\tlw\t$2,32($4)\n\t#.set\tnovolatile\n"
            "\tori\t$8,$8,0xcccc\n\tmflo\t$8\n\t#nop\n"
            "\tj\t.L4\n.L2:\n\tli\t$8,0x00040000\n"
            "\t#.set\tvolatile\n\tlw\t$3,32($4)\n"
            "\t#.set\tnovolatile\n\tori\t$8,$8,0xcccc\n"
            "\tmflo\t$8\n\t#nop\n\taddu\t$6,$6,-2048\n"
            ".L4:\n\t#.set\tvolatile\n\tsw\t$8,4($5)\n"
            "\t#.set\tnovolatile\n\tli\t$3,0x00040000\n"
            "\t#.set\tvolatile\n\tlw\t$4,36($4)\n"
            "\t#.set\tnovolatile\n\tori\t$3,$3,0xcccc\n"
            "\tmflo\t$3\n\t#nop\n\tandi\t$2,$6,0x0fff\n"
            "\t#.set\tvolatile\n\tsw\t$3,8($5)\n"
            "\t#.set\tnovolatile\n\tj\t$31\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-OBJECT-VECTOR-SCHEDULE"})
        self.assertEqual(out.count("\tmflo\t"), 4)
        self.assertEqual(out.count("\tmflo\t$3\n\tnop\n\tnop\n"), 2)
        self.assertEqual(out.count("\tmflo\t$8\n\tnop\n\tnop\n"), 2)
        self.assertLess(out.index("\tsw\t$3,0($5)"), out.index("\tlw\t$2,32($4)"))
        self.assertLess(out.index("\tlw\t$2,32($4)"), out.index("\tli\t$8,0x00040000"))
        self.assertIn(
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tj\t.L4\n\tsw\t$8,4($5)\n",
            out,
        )
        self.assertLess(out.index("\tsw\t$8,4($5)", out.index(".L2:")), out.index(".L4:"))
        self.assertIn("\tli\t$3,0x00040000\n\tori\t$3,$3,0xcccc\n", out)
        self.assertIn("\tj\t$31\n\tsw\t$3,8($5)\n", out)

    def test_asn64_opt_in_kmc_menu_activation_schedule(self):
        source = (
            "\tmove\t$16,$2\n\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tjal\tfunc_8001FA38\n\tli\t$4,0x00000003\n"
            "\t.set\tmacro\n\t.set\treorder\n"
            "\tandi\t$16,$16,0xffff\n\tslt\t$3,$3,$16\n"
            "\tbne\t$2,$0,.L6\n\tlw\t$2,36($17)\n"
            "\tj\t.L7\n\tli\t$4,0x0000000d\n"
            ".L6:\n\tlw\t$2,36($17)\n\tli\t$4,0x00000003\n"
            ".L7:\n\tjal\tfunc_8001FA68\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-MENU-ACTIVATION-SCHEDULE"})
        self.assertIn(
            "\tli\t$4,0x00000003\n\t.set\tnoreorder\n"
            "\t.set\tnomacro\n\tjal\tfunc_8001FA38\n"
            "\tandi\t$16,$2,0xffff\n",
            out,
        )
        self.assertNotIn("\tmove\t$16,$2\n", out)
        self.assertNotIn("\tandi\t$16,$16,0xffff\n", out)
        self.assertIn(
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbne\t$2,$0,.L6\n\tli\t$4,0x00000003\n",
            out,
        )
        self.assertEqual(out.count("\tli\t$4,0x00000003\n"), 2)

    def test_asn64_opt_in_kmc_combine_mode_schedule(self):
        source = (
            "\tmul.s\t$f0,$f0,$f6\n"
            "\tsw\t$4,D_800887D4\n\tsw\t$5,D_800887D8\n"
            "\tsw\t$6,D_800887DC\n\tsw\t$7,D_800887E0\n"
            "\tsw\t$10,D_800887E4\n"
            "\ttrunc.w.s $f2,$f0,$3\n\tmfc1\t$2,$f2\n\t#nop\n"
            "\tandi\t$2,$2,0x0fff\n\tor\t$2,$8,$2\n\tsw\t$2,4($9)\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-COMBINE-MODE-SCHEDULE"})
        self.assertLess(out.index("\tsw\t$2,4($9)"), out.index("\tsw\t$4,D_800887D4"))

        constants = asn64.convert(
            "\tli.d\t$f0,4.294967296e9\n\tli.s\t$f6,4.0\n"
            "\tli.d\t$f2,4.294967296e9\n\tli.d\t$f4,4.294967296e9\n",
            enabled={"KMC-LID-PAIR", "KMC-COMBINE-MODE-SCHEDULE"},
        )
        self.assertEqual(constants.count("\t.double "), 1)
        self.assertEqual(constants.count("\t.word 0x41f00000,0x00000000\n"), 2)
        self.assertIn("RODATA_SYM_2:\n\t.align 2\n", constants)
        self.assertIn("RODATA_SYM_3:\n\t.align 2\n", constants)

    def test_asn64_opt_in_kmc_display_page_schedule(self):
        source = (
            "\tsubu\t$sp,$sp,32\n\tsw\t$17,20($sp)\n"
            "\tla\t$17,D_80064B88\n\tli\t$6,-2147155968\n"
            "\tori\t$6,$6,0x0f78\n\tlw\t$3,D_80085038\n"
            "\tlw\t$5,D_8008AF98\n\tli\t$2,-2147024896\n"
            "\tori\t$2,$2,0x5d00\n\tandi\t$4,$4,0xffff\n"
            "\tsw\t$16,16($sp)\n\tsll\t$16,$4,2\n"
            "\tsw\t$2,D_80064B78\n\tla\t$2,D_8008B350\n"
            "\tsw\t$31,24($sp)\n\tsw\t$6,D_80064B70\n"
            "\tsw\t$3,0($17)\n\taddu\t$3,$3,$5\n"
            "\tsw\t$3,D_80064B8C\n\taddu\t$3,$16,$2\n"
            "\tli\t$2,0x00000001\n\tsw\t$2,D_80086170($16)\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tjal\tfunc_8004D814\n\tmove\t$4,$17\n"
            "\t.set\tmacro\n\t.set\treorder\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-DISPLAY-PAGE-SCHEDULE"})
        self.assertTrue(out.startswith("\tlw\t$3,D_80085038\n\tlw\t$5,D_8008AF98\n"))
        self.assertIn("\tlui\t$2,0x8005\n\taddiu\t$2,$2,0x0f78\n", out)
        self.assertIn(
            "\tsw\t$3,D_80064B8C\n\tla\t$3,D_8008B350\n"
            "\taddu\t$3,$16,$3\n",
            out,
        )
        self.assertIn(
            "\tmove\t$4,$17\n\tla\t$2,D_80086170\n"
            "\taddu\t$16,$16,$2\n\tli\t$2,0x00000001\n"
            "\tsw\t$2,0($16)\n\t.set\tnoreorder\n"
            "\t.set\tnomacro\n\tjal\tfunc_8004D814\n\tnop\n",
            out,
        )

    def test_asn64_opt_in_kmc_strided_copy_argument_order(self):
        source = (
            "\tmove\t$6,$17\n\tmove\t$4,$16\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tjal\tfunc_8002E17C\n\tmove\t$5,$19\n"
            "\t.set\tmacro\n\t.set\treorder\n"
        )
        out = asn64._apply_enabled_lanes(source, {"KMC-STRIDED-COPY-ARG-ORDER"})
        self.assertEqual(
            out,
            "\tmove\t$4,$16\n\tmove\t$5,$19\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tjal\tfunc_8002E17C\n\tmove\t$6,$17\n"
            "\t.set\tmacro\n\t.set\treorder\n",
        )

    def test_asn64_opt_in_kmc_double_load_pair(self):
        out = asn64.convert("\tli.d\t$f6,5.0e-1\n", enabled={"KMC-LID-PAIR"})
        self.assertIn("\tlwc1 $f7, %lo(RODATA_SYM_0)($at)\n", out)
        self.assertIn("\tlwc1 $f6, %lo(RODATA_SYM_0+4)($at)\n", out)
        self.assertNotIn("\tldc1 ", out)

    def test_asn64_mfp32_does_not_enable_title_specific_double_layout(self):
        source = "# Cc1 arguments: -mips3 -mfp32\n\n\tli.d\t$f6,5.0e-1\n"
        out = asn64.convert(source)
        self.assertIn("\tldc1 $f6, %lo(RODATA_SYM_0)($at)\n", out)
        self.assertNotIn("\tlwc1 $f7", out)

    def test_asn64_opt_in_kmc_double_literal_align4(self):
        out = asn64.convert(
            "\tli.s\t$f4,1.0\n\tli.d\t$f6,5.0e-1\n",
            enabled={"KMC-LID-PAIR", "KMC-LID-ALIGN4"},
        )
        self.assertIn("RODATA_SYM_1:\n\t.align 2\n", out)
        self.assertIn("\t.word 0x3fe00000,0x00000000\n", out)
        self.assertNotIn("\t.double ", out)
        self.assertIn("\tlwc1 $f7, %lo(RODATA_SYM_1)($at)\n", out)
        self.assertIn("\tlwc1 $f6, %lo(RODATA_SYM_1+4)($at)\n", out)

    def test_asn64_opt_in_kmc_double_literal_absolute_pad(self):
        out = asn64.convert(
            "\tli.d\t$f2,4096.0\n\t.section\t.rodata\n"
            "\t.align\t3\n.Ltable:\n\t.word\t.Ltable\n",
            enabled={"KMC-LID-PAIR", "KMC-LID-ALIGN4", "KMC-LID-PAD4"},
        )
        self.assertIn("\t.word 0x40b00000,0x00000000\n\t.space 4\n", out)
        self.assertIn("\t.align\t2\n.Ltable:\n", out)

    def test_asn64_uses_absolute_alignment_for_revisited_rodata(self):
        source = (
            "\t.section\t.rodata\n\t.align\t2\n.Lchars:\n"
            "\t.string\t\")!@#$%^&*(\"\n\t.text\n"
            "\t.section\t.rodata\n\t.align\t3\n.Ltable:\n"
            "\t.word\t.Ltable,.Ltable\n"
        )
        out = asn64.convert(source, section_bases={".rodata": 0x8008BC1C})
        self.assertIn(
            "\t.string\t\")!@#$%^&*(\"\n\t.text\n"
            "\t.section\t.rodata\n\t.space 1 # ASN64 absolute .align 3\n"
            ".Ltable:\n",
            out,
        )

    def test_asn64_keeps_equivalent_section_alignment(self):
        source = (
            "\t.section\t.rodata\n\t.align\t2\n.Lword:\n"
            "\t.word\t1\n\t.align\t3\n.Ldouble:\n\t.double\t1.0\n"
        )
        out = asn64.convert(source, section_bases={".rodata": 0x80001000})
        self.assertIn("\t.word\t1\n\t.align\t3\n.Ldouble:\n", out)

    def test_asn64_opt_in_kmc_absolute_symbol_pair(self):
        out = asn64.convert(
            "\tlw\t$2,D_80087B64\n\tsw\t$3,D_80087B68\n",
            enabled={"KMC-ABS-SYM-PAIR"},
        )
        self.assertIn("\tlui $2,%hi(D_80087B64)\n", out)
        self.assertIn("\tlw $2,%lo(D_80087B64)($2)\n", out)
        self.assertIn("\tlui $at,%hi(D_80087B68)\n", out)
        self.assertIn("\tsw $3,%lo(D_80087B68)($at)\n", out)

    def test_asn64_absolute_symbol_pair_consumes_hilo_hazard_slots(self):
        out = asn64.convert(
            "\tmflo\t$6\n\t#nop\n\tlw\t$2,D_80087B64\n\tmult\t$6,$2\n",
            enabled={"KMC-ABS-SYM-PAIR"},
        )
        self.assertIn(
            "\tmflo\t$6\n\t#nop\n\t.set noat\n"
            "\tlui $2,%hi(D_80087B64)\n"
            "\tlw $2,%lo(D_80087B64)($2)\n\t.set at\n\tmult\t$6,$2\n",
            out,
        )

    def test_asn64_noreorder_kept_and_div_expanded(self):
        out = asn64.convert("\t.set\tnoreorder\n\tjal\tf\n\tnop\n\t.set\treorder\n\tdiv\t$2,$4,$5\n")
        self.assertNotIn("jal\tf\n\tnop\n\tnop", out)
        self.assertIn("break 0x7", out)
        self.assertIn("mflo $2", out)

    def test_asn64_reorder_branch_delay_consumes_hilo_hazard_slot(self):
        out = asn64.convert(
            "\tdiv\t$4,$6,$5\n\tmfhi\t$2\n"
            "\tbne\t$2,$0,.L1\n\tdiv\t$3,$8,$7\n.L1:\n"
        )
        self.assertIn(
            "\tbne\t$2,$0,.L1\n\tnop\n\t.set noat\n\tdiv $0,$8,$7\n",
            out,
        )
        self.assertNotIn("\tbne\t$2,$0,.L1\n\tnop\n\tnop\n\tdiv", out)

    def test_asn64_symbol_base_load(self):
        out = asn64.convert("\tlw\t$2,D_80001234($4)\n")
        self.assertIn("lui $at,%hi(D_80001234)", out)
        self.assertIn("addu $at,$at,$4", out)
        self.assertNotIn("%hi(", asn64.convert("\tlw\t$2,D_80001234($4)\n", {"MEM-SYM-BASE"}))

    def test_asn64_symbol_base_expansion_consumes_hilo_hazard_slots(self):
        out = asn64.convert(
            "\tmflo\t$5\n"
            "\tlw\t$2,D_80001234($7)\n"
            "\tmult\t$5,$2\n"
        )
        self.assertIn(
            "\tmflo\t$5\n"
            "\t.set noat\n"
            "\tlui $at,%hi(D_80001234)\n"
            "\taddu $at,$at,$7\n"
            "\tlw $2,%lo(D_80001234)($at)\n"
            "\t.set at\n"
            "\tmult\t$5,$2\n",
            out,
        )

    def test_builtin_normalizer(self):
        out = normalize.builtin("\tjal\tf\n\tmove\t$2,$3\n\t.set\tnoreorder\n\tj\t$31\n\taddu\t$2,$2,1\n\t.set\treorder\n")
        self.assertTrue(out.startswith("\t.set\tnoreorder\n"))
        self.assertIn("\tjal\tf\n\tnop\n", out)
        self.assertNotIn("j\t$31\n\tnop", out)          # compiler-filled slot untouched
        self.assertNotIn(".set\treorder", out)


class SymbolTests(unittest.TestCase):
    def test_sources(self):
        m = symbols.SymbolMap()
        symbols.load_splat("foo = 0x80001000; // type:func size:0x40\nbar=0x80002000;\n", m)
        symbols.load_toml({"section": [{"name": ".main", "vram": 1, "functions": [
            {"name": "baz", "vram": 0x80003000, "size": 8}]}]}, m)
        self.assertEqual((m["foo"], m["bar"], m["baz"]), (0x80001000, 0x80002000, 0x80003000))
        self.assertEqual(m.sizes["foo"], 0x40)
        self.assertNotIn(".main", dict(m))
        self.assertIn("func_80012345", m)
        self.assertEqual(m["D_1000138"], 0x1000138)
        self.assertNotIn("unknown_name", m)
        extra = symbols.from_source('/* LDSYM gThing=0x80123456 */\nextern int x __asm__("D_80100000");\n', m)
        self.assertEqual(extra, {"gThing": 0x80123456, "D_80100000": 0x80100000})


class ManifestTests(unittest.TestCase):
    def test_source_and_unit_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src/code/a").mkdir(parents=True)
            (root / "src/code/a/u.c").write_text("")
            (root / "kmc.tsv").write_text(
                "# ROM_START\tSIZE\tSOURCE\tKMC_GCC_FLAGS\tSHA256_TEXT\n"
                "0x1000\t0x20\tsrc/x.c\t--assembler=asn64 --section-base=.rodata:0x80071000 -O2 -G0\t" + "a" * 64 + "\n"
                "0x2000\t0x10\tsrc/y.c\t-O2 -G0\t" + "b" * 64 + "\n")
            (root / "units.tsv").write_text("# unit\tvram\tsize\tstatus\na/u\t0x80080000\t0x40\tfast\n")
            rows = manifest.parse(root / "kmc.tsv", root, lambda r: r + 0x80070000, kmc_lane="kmc+hook")
            self.assertEqual([u.spec.pipeline for u in rows], ["kmc+asn64", "kmc+hook"])
            self.assertEqual(rows[0].vram, 0x80071000)
            self.assertEqual(rows[0].spec.section_bases, {".rodata": 0x80071000})
            self.assertEqual(rows[0].spec.flags, ["-O2", "-G0"])
            units = manifest.parse(root / "units.tsv", root, None, pipeline="kmc",
                                   status_flags={"fast": ["-mcpu=r8000"]})
            self.assertEqual(units[0].source, root / "src/code/a/u.c")
            self.assertEqual(units[0].spec.flags[-1], "-mcpu=r8000")
            self.assertEqual(units[0].spec.pipeline, "kmc+builtin")

    def test_flag_roundtrip(self):
        spec = split_flags(["--assembler=asn64", "--symbol=x:0x80001000", "--normalizer-legacy", "-O2"], "kmc")
        self.assertEqual(spec.pipeline, "kmc+asn64")
        self.assertEqual(spec.normalizer_args, ["--legacy"])
        again = split_flags(spec.manifest_flags().split(), "kmc")
        self.assertEqual((again.pipeline, again.symbols, again.flags), (spec.pipeline, spec.symbols, spec.flags))

    def test_sweep_order(self):
        specs = sweep_specs(["kmc+builtin", "ido5.3"])
        self.assertEqual(specs[0].flags, ["-O2", "-G0", "-mips3", "-mgp32", "-mfp32"])
        self.assertEqual(len(specs), 50)                        # 24 per pipeline, and IDO's two -mips1 settings last
        self.assertEqual(specs[24].pipeline, "ido5.3")
        self.assertEqual(non_axis_flags(["-O2", "-G", "0", "-DX", "-mips2", "-nostdinc", "-g"]), ["-DX", "-nostdinc"])
        extended = sweep_specs(["kmc+builtin"], extended=True)
        self.assertGreater(len(extended), 24)
        self.assertTrue(any("-mcpu=r4600" in spec.flags for spec in extended))
        self.assertFalse(any("-mcpu=r4600" in spec.flags and "-mips3" not in spec.flags for spec in extended))


# ---------------------------------------------------------------------------- ROM tests
def _project(name: str):
    root = fixtures.truth(name)
    image = fixtures.rom(name)
    if root is None or image is None or not Path(root, ".toolchain").is_dir():
        return None
    return Path(root), image


def _bt_toolchain():
    root = fixtures.truth("battletanx")
    return Path(root, ".toolchain") if root and Path(root, ".toolchain").is_dir() else None


@unittest.skipUnless(_project("battletanx"), "BattleTanx ROM/project not configured")
class BattleTanxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, image = _project("battletanx")
        cls.cfg = load_config(cls.root)
        cls.cfg.normalizer = "python3 tools/normalize_kmc_gcc_asm.py"
        cls.target = Target.from_rom(image)
        cls.syms = symbols.load(symbols.discover(cls.root))

    def _rows(self, name, lane="kmc+hook", n=None, pred=None):
        units = manifest.parse(self.root / "config/us" / name, self.root, self.target.rom_to_vram, kmc_lane=lane)
        units = [u for u in units if pred is None or pred(u)]
        return units[:n] if n else units

    def test_batch_kmc_hook_and_asn64(self):
        units = self._rows("matching_kmc_units.tsv", n=40) + \
            self._rows("matching_kmc_units.tsv", pred=lambda u: u.spec.pipeline == "kmc+asn64", n=10)
        outs = batch(units, self.cfg, self.target, self.syms)
        self.assertEqual([o.name for o in outs if not o.exploratory_exact], [])
        self.assertTrue(all(o.exact or o.as_dict()["proof"]["verdict"] == "assumed" for o in outs))
        self.assertTrue(all(o.digest_ok for o in outs))

    def test_batch_ido_and_asm(self):
        units = self._rows("matching_units.tsv", n=40) + self._rows("matching_asm_units.tsv")
        outs = batch(units, self.cfg, self.target, self.syms)
        self.assertEqual([o.name for o in outs if not o.exploratory_exact], [])
        self.assertTrue(all(o.exact or o.as_dict()["proof"]["verdict"] == "assumed" for o in outs))

    def test_sweep_finds_asn64(self):
        src = self.root / "src/code/effects/break_object_into_debris.c"
        res = sweep(src, self.cfg, self.target, 0x800A0930, 0x1A4, self.syms,
                    ["kmc+builtin", "kmc+asn64", "kmc+as"])
        hits = [outcome for outcome in res.tried if outcome.exploratory_exact]
        self.assertTrue(hits)
        self.assertEqual(hits[0].spec.pipeline, "kmc+asn64")
        self.assertEqual(hits[0].as_dict()["proof"]["verdict"], "assumed")
        with self.assertRaisesRegex(ValueError, "strictly verified"):
            manifest_row(hits[0], self.root, src, self.target, self.syms)

    def test_rodata_and_diff(self):
        src = self.root / "src/code/audio/audio_state_helpers.c"
        with tempfile.TemporaryDirectory() as tmp:
            ok = match_source(src, BuildSpec("kmc+hook", ["-O2", "-G0", "-mips3", "-mgp32", "-mfp32"]),
                              self.cfg, self.target, 0x800AB91C, 0xE0, self.syms, Path(tmp))
            self.assertTrue(ok.exact, ok.as_dict())
            self.assertEqual(ok.result.bases[".rodata"], 0x80073110)  # recovered from the ROM
            bad = match_source(src, BuildSpec("kmc+as", ["-O2", "-G0", "-mips3", "-mgp32", "-mfp32"]),
                               self.cfg, self.target, 0x800AB91C, 0xE0, self.syms, Path(tmp))
            self.assertFalse(bad.exact)
            self.assertGreater(bad.result.diff.score, 0)


@unittest.skipUnless(_project("battletanx_ga"), "Global Assault ROM/project not configured")
class GlobalAssaultTests(unittest.TestCase):
    def test_batch_sample(self):
        root, image = _project("battletanx_ga")
        cfg = load_config(root)
        cfg.normalizer = "python3 tools/normalize_kmc_gcc_asm.py"
        cfg.include = [root / "include", root / "src/code/include"]
        target = Target.from_rom(image)
        syms = symbols.load([root / "config/us/symbol_addrs.txt"])
        rodata = manifest.load_section_map(root / "config/us/unit_rodata.tsv")
        units = manifest.parse(root / "config/us/clean_normalized_units.tsv", root, target.rom_to_vram,
                               "kmc", "kmc+hook", rodata_map=rodata,
                               status_flags={"matched-mcpu-r8000": ["-mcpu=r8000"]})[:30]
        units += manifest.parse(root / "config/us/asn64_units.tsv", root, target.rom_to_vram,
                                "kmc+asn64", rodata_map=rodata)
        outs = batch(units, cfg, target, syms, infer_externals=True)
        self.assertEqual([o.name for o in outs if not o.exact], [])


@unittest.skipUnless(fixtures.truth("mk64") and fixtures.rom("mk64") and _bt_toolchain(),
                     "Mario Kart 64 source tree or ROM not configured")
class MarioKartTests(unittest.TestCase):
    def test_libultra_located_and_exact(self):
        root = Path(fixtures.truth("mk64"))
        cfg = load_config(root, toolchain_dir=_bt_toolchain())
        cfg.include = [root / "include", root / "src", root / "include/libc"]
        target = Target.from_rom(fixtures.rom("mk64"))
        flags = "-G 0 -nostdinc -DTARGET_N64 -D_LANGUAGE_C -mips2 -non_shared -Wab,-r4300_mul " \
                "-Xcpluscomm -signed -32".split()
        with tempfile.TemporaryDirectory() as tmp:
            for name, opt in (("guRotateF", ["-O3"]), ("osCreateThread", []), ("__osGetCurrFaultedThread", [])):
                o = match_source(root / f"src/os/{name}.c", BuildSpec("ido5.3", flags + opt), cfg, target,
                                 None, None, symbols.SymbolMap(), Path(tmp))
                self.assertTrue(o.exact, (name, o.error, o.result and o.result.as_dict()))


if __name__ == "__main__":
    unittest.main()
