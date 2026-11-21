"""Project generator on a hand-built export tree (no toolchain, no ROM file, no build)."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from n64dk.match import projectgen as pg

HEADER = ("image\tfunction\tvram\trom_offset\tsize\tcode_size\toutcome\thow\tsource\tdistance\treproduced_by"
          "\tunit_class")
DELTA = 0x80000400 - 0x1000
ROM = bytes((i * 7 + 3) & 0xFF for i in range(0x2000))     # stands in for the user's ROM
ROM_SHA1 = hashlib.sha1(ROM).hexdigest()
C1 = b"int f1(void) { return 1; }\n"
ASM_HEAD = ".macro glabel label\n    .global \\label\n    \\label:\n.endm\n.set noreorder\n.section .text\n"


def row(function, vram, outcome, how, source, reproduced, distance=""):
    return "\t".join(["main", function, f"0x{vram:08X}", f"0x{vram - DELTA:X}", "0x20", "0x20", outcome, how,
                      source, distance, reproduced, "game"])


def write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())


def make_export(root: Path, *, f4_exact: bool = False, f1_text: bytes = C1) -> Path:
    export = root / "export"
    rows = [HEADER,
            row("f1", 0x80000400, "exact", "kmc+as -O2 -G0 (draft)", "src/main/f1.c", "C"),
            row("f2", 0x80000420, "awaiting-symbols", "kmc+as -O2 -G0 (draft)", "src_awaiting_symbols/main/f2.c", "C"),
            row("f3", 0x80000440, "assembly", "gnu-as -march=vr4300 -mabi=o64", "asm/main/f3.s", "assembly (hand-written)"),
            (row("f4", 0x80000460, "exact", "kmc+as -O1 (draft after variant search)", "src/main/f4.c", "C")
             if f4_exact else
             row("f4", 0x80000460, "open", "kmc+as -O2 -G0", "asm_not_decompiled/main/f4.s", "assembly (not decompiled)", "12")),
            row("f5", 0x80000480, "open", "kmc+as -O2 -G0", "asm_not_decompiled/main/f5.s", "assembly (not decompiled)", "40"),
            row("f6", 0x800004A0, "does-not-build", "kmc+as -O2 -G0", "", "nothing"),
            row("f7", 0x800004C0, "donor", "donor unit Lib: src/x.c", "donors/main/src/Lib/src/x.c", "C")]
    write(export / "FUNCTIONS.tsv", "\n".join(rows) + "\n")
    write(export / "src/main/f1.c", f1_text)
    write(export / "src_awaiting_symbols/main/f2.c", "extern int D_80001000;\nint f2(void) { return D_80001000; }\n")
    write(export / "asm/main/f3.s", ASM_HEAD + "glabel f3\n    jr $ra\n    nop\n")
    if f4_exact:
        write(export / "src/main/f4.c", "int f4(int a) { return a + 4; }\n")
    else:
        write(export / "asm_not_decompiled/main/f4.s", ASM_HEAD + "glabel f4\n    addiu $v0, $a0, 4\n    jr $ra\n    nop\n")
    write(export / "asm_not_decompiled/main/f5.s",
          "/* raw words: this range does not assemble from mnemonics */\nglabel f5\n.word 0x12345678 /* x */\n")
    write(export / "donors/main/src/Lib/src/x.c", '#include "x.h"\nint f7(void) { return X; }\n')
    write(export / "donors/main/src/Lib/include/x.h", "#define X 7\n")
    write(export / "donors/main/src/Lib/blob.bin", bytes(range(64)))
    write(export / "donors/main/proposed_Lib_kmc_as.tsv",
          "# kmc+as rows proposed by donor transplant from Lib; SOURCE paths are relative to\n"
          "# src/Lib/ (the source and the headers it includes, in the donor's layout)\n"
          "# ROM_START\tSIZE\tSOURCE\tFLAGS\n0x10C0\t0x20\tsrc/x.c\t-O2 -Iinclude\n")
    write(export / "README.md", f"# T\n\n- ROM SHA-1: `{ROM_SHA1}`\n")
    write(export / "config/n64dk.toml", '[images.main]\npath = "baserom.z64"\ncatalogue = "survey/functions.toml"\n')
    write(export / "symbols/symbol_addrs.txt", "f1 = 0x80000400; // type:func\n")
    write(export / "manifests/assumptions.tsv", "# image\tunit\tkind\tname\taddress\nmain\tf2\tsymbol\tD_80001000\t0x80001000\n")
    write(export / "receipts/main/f1-80000400.json",
          json.dumps({"source": {"name": "f1.c", "sha256": hashlib.sha256(C1).hexdigest()}}))
    return export


def tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def generate(export: Path, out: Path, **more) -> dict:
    return pg.generate(export, out, images={"main": (ROM, DELTA)}, toolchain_lock={"digest": "d" * 64}, **more)


class GenerateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.export = make_export(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_statuses_and_layout(self):
        out = self.root / "project"
        report = generate(self.export, out)
        self.assertEqual(report["counts"], {"exact": 2, "assumed": 1, "assembly": 2, "raw": 2})
        files = tree(out)
        for path in ("src/main/f1.c", "src_assumed/main/f2.c", "asm/main/f3.s", "asm/generated/main/f4.s",
                     "raw/main/f5.s", "raw/main/f6.s", "donors/main/src/Lib/src/x.c", "donors/main/src/Lib/include/x.h",
                     "receipts/main/f1-80000400.json", "provenance/main/f1.json", "unresolved/main/f2.md",
                     "unresolved/main/f4.md", "unresolved/main/f5.md", "Makefile", "tools/extract_raw.py",
                     "tools/verify.sh", "tools/verify_receipts.sh", "recipes.tsv", "STATUS.tsv", "STATUS.md",
                     "toolchain.lock.json", "include/units_main.h", "include/macro.inc", "linker/main_text.ld",
                     "symbols/symbol_addrs.txt", "README.md", pg.MANIFEST):
            self.assertIn(path, files)
        self.assertNotIn("donors/main/src/Lib/blob.bin", files)              # binaries are never copied
        self.assertNotIn("unresolved/main/f1.md", files)
        self.assertNotIn("unresolved/main/f3.md", files)                     # hand-written assembly is settled
        self.assertIn("D_80001000", files["unresolved/main/f2.md"].decode())
        raw = files["raw/manifest.tsv"].decode()
        self.assertIn(f"# rom_sha1 {ROM_SHA1}", raw)
        f5 = hashlib.sha256(ROM[0x1080:0x10A0]).hexdigest()
        self.assertIn(f"main\tf5\trom\t0x1080\t0x20\t{f5}", raw)
        self.assertIn('.incbin "build/raw/main/f5.bin"', files["raw/main/f5.s"].decode())
        status = {line.split("\t")[0]: line.split("\t") for line in files["STATUS.tsv"].decode().splitlines()[1:]}
        self.assertEqual(len(status), 7)
        self.assertEqual(status["main/f5"][6:8], ["raw", "raw-words"])
        self.assertEqual(status["main/f7"][6:8], ["exact", "donor-c"])
        self.assertEqual(status["main/f1"][12], "present")
        verify = files["tools/verify.sh"].decode()
        self.assertIn("--pipeline kmc+as '--flags=-O2 -G0'", verify)
        self.assertIn("unit exact main__f7 donors/main/src/Lib --image-name main --addr 800004C0 --size 20", verify)
        self.assertIn("'--flags=-O2 -Iinclude'", verify)
        self.assertNotIn("main__f5", verify)                                  # raw units are extracted, not verified
        self.assertIn("verify-receipt receipts/main/f1-80000400.json --source src/main/f1.c", files["tools/verify_receipts.sh"].decode())
        makefile = files["Makefile"].decode()
        self.assertIn("\n\t$(PYTHON) tools/extract_raw.py --rom $(ROM) $(IMAGES) --out build/raw\n", makefile)
        self.assertIn("\nPROJECT ?= ../project\n", makefile)
        manifest = json.loads(files[pg.MANIFEST])
        self.assertEqual(manifest["units"]["main/f4"]["status"], "assembly")
        self.assertEqual(manifest["files"]["src/main/f1.c"], hashlib.sha256(C1).hexdigest())

    def test_reused_donor_source_keeps_each_target_placement(self):
        manifest = self.export / "donors/main/proposed_Lib_kmc_as.tsv"
        manifest.write_text(manifest.read_text() +
                            "0x10E0\t0x20\tsrc/x.c\t-O1 -Iinclude\n")
        recipes = pg.donor_recipes(self.export)
        source = "donors/main/src/Lib/src/x.c"
        first = pg.Unit("main", "f7", 0x800004C0, 0x10C0, 0x20, 0x20,
                        "donor", source=source, kind="donor-c")
        second = pg.Unit("main", "f8", 0x800004E0, 0x10E0, 0x20, 0x20,
                         "donor", source=source, kind="donor-c")
        self.assertEqual(pg.parse_recipe(first, recipes)["unit_offset"], "0x10C0")
        self.assertEqual(pg.parse_recipe(first, recipes)["flags"], "-O2 -Iinclude")
        self.assertEqual(pg.parse_recipe(second, recipes)["unit_offset"], "0x10E0")
        self.assertEqual(pg.parse_recipe(second, recipes)["flags"], "-O1 -Iinclude")
    def test_a_section_the_manifest_places_is_in_the_units_recipe(self):
        # the function table names the compiler recipe; the address of a jump table the strict comparison
        # placed by its contents is in manifests/queue.tsv, and the link needs it
        write(self.export / "manifests/placed_sections.tsv",
              "# image\tvram\tsection\taddress\torigin\n"
              "main\t0x80000400\t.rodata\t0x80001F00\tsection-content\n")
        self.assertEqual(pg.queue_placements(self.export), {("main", 0x80000400): ["--section-base=.rodata:0x80001F00"]})
        _files, units, _manifest = pg.build_plan(self.export)
        by_name = {unit.function: unit for unit in units}
        self.assertEqual(by_name["f1"].recipe["flags"], "-O2 -G0 --section-base=.rodata:0x80001F00")
        self.assertNotIn("--section-base", by_name["f2"].recipe["flags"])

    def test_no_rom_bytes_in_the_tree(self):
        out = self.root / "project"
        generate(self.export, out)
        windows = {ROM[k:k + 32] for k in range(0, len(ROM) - 32, 4)}
        for path, data in tree(out).items():
            self.assertNotIn("raw words", data.decode(errors="replace"), path)
            for k in range(0, max(0, len(data) - 31)):
                self.assertNotIn(data[k:k + 32], windows, path)

    def test_generating_twice_is_byte_identical(self):
        generate(self.export, self.root / "one")
        generate(self.export, self.root / "two")
        self.assertEqual(tree(self.root / "one"), tree(self.root / "two"))

    def test_regeneration_rewrites_only_what_changed(self):
        out = self.root / "project"
        first = generate(self.export, out)
        self.assertEqual(first["conflicts"], [])
        again = generate(self.export, out)
        self.assertEqual((again["written"], again["removed"], again["changed_units"]), ([], [], []))
        newer = make_export(self.root / "newer", f4_exact=True, f1_text=C1 + b"/* edited */\n")
        report = generate(newer, out)
        self.assertEqual(report["changed_units"], ["main/f1", "main/f4"])
        self.assertEqual(report["invalidated_receipts"], ["main/f1"])
        self.assertIn("src/main/f4.c", report["written"])
        for path in ("asm/generated/main/f4.s", "unresolved/main/f4.md", "receipts/main/f1-80000400.json"):
            self.assertIn(path, report["removed"])
        for path in ("src_assumed/main/f2.c", "provenance/main/f2.json", "raw/main/f6.s"):
            self.assertNotIn(path, report["written"])
        self.assertFalse((out / "asm" / "generated").exists())                 # emptied folders pruned
        generate(newer, self.root / "fresh")
        self.assertEqual(tree(out), tree(self.root / "fresh"))                 # same as generating from scratch

    def test_hand_edits_are_never_overwritten(self):
        out = self.root / "project"
        generate(self.export, out)
        mine = b"/* my assembly for f6 */\n"
        (out / "raw/main/f6.s").write_bytes(mine)
        report = generate(self.export, out)
        self.assertIn("raw/main/f6.s", report["conflicts"])
        self.assertEqual((out / "raw/main/f6.s").read_bytes(), mine)
        self.assertIn("raw/main/f6.s", generate(self.export, out)["conflicts"])  # still protected next time
        forced = generate(self.export, out, force=True)
        self.assertIn("raw/main/f6.s", forced["written"])

    def test_link_layout_objects_are_attached(self):
        layout = self.root / "layout"
        write(layout / "objects.tsv", "segment\tposition\tpositions\tunit\tkind\tobject\tarchive\tmember\ttext_start"
                                      "\ttext_size\tsections\tnames\nmain\t0\t0\tmain_80000400\tc\tbuild/main/main_80000400.o"
                                      "\t\t\t0x80000400\t0x40\t.text\t\n")
        write(layout / "link.ld", "SECTIONS\n{\n}\n")
        out = self.root / "project"
        generate(self.export, out, link_layout=layout)
        files = tree(out)
        status = {line.split("\t")[0]: line.split("\t") for line in files["STATUS.tsv"].decode().splitlines()[1:]}
        self.assertEqual(status["main/f2"][9], "main_80000400")
        self.assertEqual(status["main/f3"][9], "-")
        self.assertEqual(files["linker/link_layout.ld"], b"SECTIONS\n{\n}\n")


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.export = make_export(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_resolve_export(self):
        self.assertEqual(pg.resolve_export(self.root), self.export.resolve())   # a run directory holding export/
        with self.assertRaises(pg.GenerationError):
            pg.resolve_export(self.root / "nothing")

    def test_refused_outputs(self):
        rom = self.root / "roms" / "game.z64"
        write(rom, b"\0" * 64)
        (self.root / "repo" / ".git").mkdir(parents=True)
        write(self.root / "busy" / "notes.txt", "mine")
        write(self.root / "withrom" / "x.z64", b"\0")
        for out in (self.export / "project", self.root, self.root / "roms" / "project", self.root / "repo" / "project",
                    self.root / "busy", self.root / "withrom"):
            with self.subTest(out=out), self.assertRaises(pg.GenerationError):
                pg.generate(self.export, out, roms=[rom])
        self.assertFalse((self.root / "repo" / "project").exists())

    def test_generated_project_in_a_git_tree_regenerates_in_place(self):
        out = self.root / "work" / "project"
        generate(self.export, out)
        (self.root / "work" / ".git").mkdir()
        self.assertEqual(generate(self.export, out)["conflicts"], [])


if __name__ == "__main__":
    unittest.main()
