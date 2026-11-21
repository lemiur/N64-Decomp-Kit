"""relink-image on a tiny synthetic image: hand-built ELF objects, a stand-in ROM, the kit's relocator
(no toolchain, no linker, no ROM file)."""
from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from n64dk.elf import R_MIPS_26, R_MIPS_HI16, R_MIPS_LO16, Elf
from n64dk.match import relink as rl

BASE = 0x80000400
DELTA = BASE - 0x1000
ALLOC, WRITE, EXEC = 0x2, 0x1, 0x4


def words(*values) -> bytes:
    return struct.pack(f">{len(values)}I", *values)


C_BYTES = b"\x27\xbd\xff\xe8\x03\xe0\x00\x08"
# A: jal func_80000410; nop; lui v0, %hi(D_80000430); lw v0, %lo(D_80000430)(v0)
IMAGE = (words(0x0C000104, 0, 0x3C028000, 0x8C420430)
         + words(0x03E00008, 0, 0, 0)                   # B: hand-written assembly
         + C_BYTES                                       # C: 8 bytes
         + words(1, 2)                                   # data no unit owns
         + words(0x11223344)                             # D_80000430: A's initialised .data
         + b"\0\0\0\0ROMDATA!")                          # more data
ROM = bytes(0x1000) + IMAGE + bytes(0x40)
ROM_SHA1 = hashlib.sha1(ROM).hexdigest()
SYMBOLS = "func_80000410 = 0x80000410;\nD_80000430 = 0x80000430;\n"


def object_a(callee: str = "func_80000410") -> bytes:
    return rl.write_elf(
        [(".text", words(0x0C000000, 0, 0x3C020000, 0x8C420000), ALLOC | EXEC, 4, 0),
         (".data", words(0x11223344), ALLOC | WRITE, 4, 0)],
        [("func_80000400", 0, 16, ".text", 1, 2), ("D_80000430", 0, 4, ".data", 1, 1), (callee, 0, 0, None, 1, 0)],
        {".text": [(0, R_MIPS_26, callee), (8, R_MIPS_HI16, "D_80000430"), (12, R_MIPS_LO16, "D_80000430")]})


def object_b(tail: bytes = b"") -> bytes:                # GNU as aligns .text to 16
    return rl.write_elf([(".text", words(0x03E00008, 0, 0, 0) + tail, ALLOC | EXEC, 16, 0)],
                        [("func_80000410", 0, 16, ".text", 1, 2)])


OBJECT_C = rl.write_elf([(".text", C_BYTES, ALLOC | EXEC, 4, 0)], [("func_80000420", 0, 8, ".text", 1, 2)])


def unit(function, vram, size, status, kind, file, pipeline="", flags="", **recipe):
    return {"unit": f"main/{function}", "image": "main", "function": function, "vram": f"0x{vram:08X}",
            "offset": f"0x{vram - DELTA:X}", "size": f"0x{size:X}", "code_size": f"0x{size:X}", "status": status,
            "kind": kind, "file": file, "recipe": {"pipeline": pipeline, "flags": flags, **recipe}}


def units(a_status="exact", c_linked=False):
    c = (unit("func_80000420", BASE + 0x20, 8, "assembly", "generated", "asm/generated/main/func_80000420.s", "as")
         if c_linked else unit("func_80000420", BASE + 0x20, 8, "raw", "none", "raw/main/func_80000420.s"))
    return [unit("func_80000400", BASE, 0x10, a_status, "c", "src/main/func_80000400.c", "kmc+as", "-O2"),
            unit("func_80000410", BASE + 0x10, 0x10, "assembly", "handwritten", "asm/main/func_80000410.s", "as"), c]


def write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())


class Fixture:
    def __init__(self, root: Path, rows=None, symbols=SYMBOLS, objects=None):
        self.project, self.build = root / "project", root / "build"
        write(self.project / "GENERATED.json", "{}\n")
        for row in rows or units():
            write(self.project / "provenance" / "main" / f"{row['function']}.json", json.dumps(row))
            write(self.project / row["file"], f"/* {row['function']} */\n")
        write(self.project / "config" / "layout.json",
              json.dumps({"main": {"vram": "0x80000400", "end": "0x80000440", "rom": "0x1000", "rom_end": "0x1040"}}))
        write(self.project / "symbols" / "symbol_addrs.txt", symbols)
        write(self.project / "raw" / "manifest.tsv", f"# raw units\n# rom_sha1 {ROM_SHA1}\n")
        self.objects = (objects if objects is not None
                        else {"func_80000400.c": object_a(), "func_80000410.s": object_b()})
        self.calls = []

    def build_fn(self, source, spec, cfg, workdir):
        self.calls.append((Path(source).name, spec.pipeline, list(spec.flags)))
        data = self.objects.get(Path(source).name)
        if data is None:
            raise RuntimeError("cc1: error: does not build")
        Path(workdir).mkdir(parents=True, exist_ok=True)
        path = Path(workdir) / (Path(source).stem + ".o")
        path.write_bytes(data)
        return path

    def run(self, rom=ROM, **options):
        options.setdefault("linker", "kit")
        return rl.relink(self.project, "main", self.build, data=rom, delta=DELTA, is_rom=True,
                         rom_sha1=hashlib.sha1(rom).hexdigest(), build_fn=self.build_fn, **options)

    def tree(self) -> dict:
        return {p.relative_to(self.project).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(self.project.rglob("*")) if p.is_file()}


class RelinkTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_identical_with_coverage_and_a_raw_unit(self):
        fx = Fixture(self.root)
        report = fx.run(keep_image=True)
        self.assertEqual(report["outcome"], "identical")
        self.assertEqual(report["coverage"],
                         {"exact": 20, "assumed": 0, "assembly": 16, "raw-unit": 8, "raw-other": 20})
        self.assertEqual((report["source_text_bytes"], report["source_data_bytes"]), (32, 4))
        self.assertEqual(report["sections"]["placed_by_declared_symbols"], 1)      # A's .data from D_80000430
        self.assertEqual(fx.calls, [("func_80000400.c", "kmc+as", ["-O2"]), ("func_80000410.s", "as", [])])
        self.assertEqual((fx.build / "relinked.bin").read_bytes(), IMAGE)
        self.assertTrue(report["level"].startswith("unchanged"))
        self.assertTrue(any("raw stand-ins" in item for item in report["level_blockers"]))
        self.assertEqual(report["units"]["raw_members_by_reason"], {"raw": 1})
        script = (fx.build / "link.ld").read_text()
        self.assertIn(" 0x80000400 : SUBALIGN(1) { obj/0001_func_80000400.o(.text) }", script)
        self.assertIn(" 0x80000430 : SUBALIGN(1) { obj/0001_func_80000400.o(.data) }", script)
        self.assertIn(" 0x80000420 : SUBALIGN(1) { raw/raw.o(.raw_80000420) }", script)
        self.assertIn("/DISCARD/", script)
        rows = (fx.build / "units.tsv").read_text().splitlines()
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[1].split("\t")[6:11], ["yes", "16", "4", "0", "0"])
        self.assertEqual(json.loads((fx.build / rl.MARKER).read_text())["outcome"], "identical")

    def test_every_unit_linked_reaches_level_3_for_the_image(self):
        objects = {"func_80000400.c": object_a(), "func_80000410.s": object_b(), "func_80000420.s": OBJECT_C}
        report = Fixture(self.root, units(c_linked=True), objects=objects).run()
        self.assertEqual(report["outcome"], "identical")
        self.assertEqual(report["coverage"]["raw-unit"], 0)
        self.assertEqual(report["level_blockers"], [])
        self.assertTrue(report["level"].startswith("3 (pinned relink) for image main"))

    def test_first_difference_and_its_owner(self):
        rom = bytearray(ROM)
        rom[0x100F] ^= 1                                  # the low byte of A's lw
        fx = Fixture(self.root)
        with self.assertRaises(rl.RelinkError):           # not the ROM the project describes
            fx.run(rom=bytes(rom))
        report = fx.run(rom=bytes(rom), any_rom=True)
        self.assertEqual((report["outcome"], report["differing_bytes"]), ("different", 1))
        first = report["first_difference"]
        self.assertEqual((first["vram"], first["rom_offset"], first["class"], first["units"]),
                         ("0x8000040F", "0x100F", "exact", ["func_80000400"]))
        self.assertEqual(report["differing_by_class"], {"exact": 1})
        self.assertEqual((fx.build / "units.tsv").read_text().splitlines()[1].split("\t")[10], "1")

    def test_an_undeclared_name_is_reported_and_bound_to_a_sentinel(self):
        objects = {"func_80000400.c": object_a("missing_function"), "func_80000410.s": object_b()}
        fx = Fixture(self.root, objects=objects)
        report = fx.run()
        self.assertEqual(report["symbols"]["unresolved"], ["missing_function"])
        self.assertEqual(report["outcome"], "different")
        self.assertEqual(report["first_difference"]["image_offset"], "0x0")
        self.assertIn("PROVIDE(missing_function = 0x8FFF0000);", (fx.build / "link.ld").read_text())

    def test_object_definitions_win_over_declared_addresses_and_disagreements_are_listed(self):
        report = Fixture(self.root, symbols="func_80000410 = 0x80000414;\nD_80000430 = 0x80000430;\n").run()
        self.assertEqual(report["outcome"], "identical")
        self.assertEqual(report["symbols"]["declared_vs_linked"],
                         ["func_80000410: linked 0x80000410, declared 0x80000414"])

    def test_a_unit_that_does_not_build_is_a_raw_stand_in(self):
        fx = Fixture(self.root, objects={"func_80000400.c": object_a()})
        report = fx.run()
        self.assertEqual(report["outcome"], "identical")                  # A's call bound to the declared address
        self.assertEqual(report["coverage"]["assembly"], 0)
        self.assertEqual(report["coverage"]["raw-unit"], 24)
        self.assertEqual(report["units"]["raw_members_by_reason"], {"build-failed": 1, "raw": 1})
        self.assertIn("PROVIDE(func_80000410 = 0x80000410);", (fx.build / "link.ld").read_text())

    def test_assumed_units_are_raw_unless_linked_on_request(self):
        fx = Fixture(self.root, units(a_status="assumed"))
        report = fx.run()
        self.assertEqual([call[0] for call in fx.calls], ["func_80000410.s"])
        self.assertEqual(report["coverage"],
                         {"exact": 0, "assumed": 0, "assembly": 16, "raw-unit": 24, "raw-other": 24})
        linked = fx.run(assumed="link")
        self.assertEqual(linked["outcome"], "identical")
        self.assertEqual(linked["coverage"]["assumed"], 20)
        self.assertTrue(any("assumed" in item for item in linked["level_blockers"]))

    def test_nonzero_bytes_past_a_slot_are_a_spill(self):
        objects = {"func_80000400.c": object_a(), "func_80000410.s": object_b(words(0xDEADBEEF, 0, 0, 0))}
        report = Fixture(self.root, objects=objects).run()
        self.assertEqual(report["outcome"], "identical")                  # the slot's bytes are right
        self.assertEqual(len(report["spills"]), 1)
        self.assertTrue(any("problems" in item for item in report["level_blockers"]))

    def test_rom_bytes_only_under_the_build_directory(self):
        fx = Fixture(self.root)
        before = fx.tree()
        fx.run()
        self.assertEqual(fx.tree(), before)
        for path in fx.project.rglob("*"):
            if path.is_file():
                self.assertNotIn(b"ROMDATA!", path.read_bytes(), path)
        raw = Elf((fx.build / "raw" / "raw.o").read_bytes())
        self.assertEqual(raw.section(".raw_80000434").data, IMAGE[0x34:])
        holding = sorted(p.relative_to(fx.build).as_posix() for p in fx.build.rglob("*")
                         if p.is_file() and b"ROMDATA!" in p.read_bytes())
        self.assertEqual(holding, ["raw/raw.o"])                 # the linked image is kept only on request
        self.assertFalse((fx.build / "relinked.bin").exists())

    def test_build_directory_refusals_and_reuse(self):
        fx = Fixture(self.root)
        with self.assertRaises(rl.RelinkError):           # inside the project tree
            rl.relink(fx.project, "main", fx.project / "build", data=ROM, delta=DELTA, is_rom=True,
                      build_fn=fx.build_fn, linker="kit")
        self.assertFalse((fx.project / "build").exists())
        write(self.root / "other" / "notes.txt", "mine\n")
        with self.assertRaises(rl.RelinkError):           # a non-empty directory that is not a relink build
            rl.relink(fx.project, "main", self.root / "other", data=ROM, delta=DELTA, is_rom=True,
                      build_fn=fx.build_fn, linker="kit")
        self.assertEqual(fx.run()["outcome"], "identical")
        self.assertEqual(fx.run()["outcome"], "identical")                 # an earlier relink build is reused

    def test_gnu_ld_path_script_and_collection(self):
        fx = Fixture(self.root)
        members = rl.read_members(fx.project, "main")
        plan_units = rl.targets(members)
        plan_units[0].obj, plan_units[1].obj = "obj/a.o", "obj/b.o"
        objects = {0: object_a(), 1: object_b()}
        layout = rl.plan(plan_units, objects, (BASE, BASE + 0x40), rl.declared_symbols(fx.project, "main"))
        script = rl.linker_script("main", layout)
        self.assertIn(" 0x80000410 : SUBALIGN(1) { obj/b.o(.text) }", script)
        self.assertIn(" 0x80000434 : SUBALIGN(1) { raw/raw.o(.raw_80000434) }", script)
        raw = {p.out: IMAGE[p.vram - BASE:p.vram - BASE + p.size] for p in layout.pieces if p.target < 0}
        linked, problems = rl.kit_link(objects, layout, raw)
        self.assertEqual(problems, [])
        # what ld writes: one section per piece at its address
        sections = [(p.out, linked[p.out], ALLOC, 1, p.vram) for p in layout.pieces if p.out in linked]
        collected, problems = rl.collect_linked(rl.write_elf(sections, executable=True), layout)
        self.assertEqual((collected, problems), (linked, []))
        self.assertEqual(rl.compare_image(layout, collected, IMAGE).total, 0)
        moved = [(name, body, flags, align, address + 4 if name == layout.pieces[0].out else address)
                 for name, body, flags, align, address in sections]
        collected, problems = rl.collect_linked(rl.write_elf(moved, executable=True), layout)
        self.assertNotIn(layout.pieces[0].out, collected)
        self.assertTrue(any("planned 0x80000400" in item for item in problems))

    def test_donor_rows_share_one_object_and_include_paths_follow_the_donor_root(self):
        recipe = {"pipeline": "kmc+as", "flags": "-O2", "unit_offset": "0x1010", "unit_size": "0x10"}
        donor = "donors/main/src/tree/src/x.c"
        rows = [rl.Member("main/f1", "f1", BASE + 0x10, 0x1010, 8, 8, "exact", "donor-c", donor, recipe),
                rl.Member("main/f2", "f2", BASE + 0x18, 0x1018, 8, 8, "assumed", "donor-c", donor, recipe)]
        found = rl.targets(rows)
        self.assertEqual(len(found), 1)
        self.assertEqual((found[0].vram, found[0].slot, found[0].status, found[0].reason_code),
                         (BASE + 0x10, 0x10, "assumed", "assumed"))
        self.assertEqual(rl.targets(rows, assumed="link")[0].reason, "")
        asm = rl.Member("main/f3", "f3", BASE + 0x20, 0x1020, 8, 8, "assembly", "donor-asm", "donors/main/src/t/bcmp.s",
                        {"pipeline": "", "flags": ""})
        self.assertEqual(rl.targets([asm])[0].pipeline, "as")        # as tools/verify.sh assembles it
        bare = rl.Member("main/f4", "f4", BASE + 0x28, 0x1028, 8, 8, "assumed", "donor-c", donor, {"pipeline": ""})
        self.assertEqual(rl.targets([bare], assumed="link")[0].reason_code, "no-recipe")
        self.assertEqual(rl._rooted(["-Iinclude", "-I", "sub", "-I/abs", "-O2"], Path("/r")),
                         ["-I/r/include", "-I", "/r/sub", "-I/abs", "-O2"])


if __name__ == "__main__":
    unittest.main()


class LikelyStepTest(unittest.TestCase):
    def test_each_placement_and_word_class_names_its_step(self):
        from n64dk.match.relink import likely_step
        code = {"class": "exact", "placement": "slot", "section": ".text"}
        jal = lambda t: (0x0C000000 | (t >> 2) & 0x3FFFFFF).to_bytes(4, "big")      # noqa: E731
        self.assertIn("link: a call", likely_step(code, jal(0x80001000), jal(0x80001040), 1))
        self.assertIn("link: an immediate", likely_step(code, bytes.fromhex("3C018001"), bytes.fromhex("3C018002"), 3))
        self.assertIn("compile", likely_step(code, bytes.fromhex("00851021"), bytes.fromhex("00A41021"), 0))
        self.assertIn("layout", likely_step({"class": "hole (no piece owns it)"}, b"", b"", 0))
        self.assertIn("placement", likely_step({"class": "exact", "placement": "scratch", "section": ".data"}, b"", b"", 0))
        self.assertIn("data", likely_step({"class": "exact", "placement": "slot", "section": ".data"}, b"", b"", 0))



class OverlappingOverlayTest(unittest.TestCase):
    """Two overlays loaded at the same VRAM (0x80100000) from different ROM offsets: each image is
    linked and compared on its own; swapping their units is caught."""
    VRAM = 0x80100000
    ROMS = {"ovl1": 0x2000, "ovl2": 0x3000}
    CODE = {"ovl1": words(0x27BDFFE8, 0x03E00008, 0x27BD0018, 0), "ovl2": words(0x24020001, 0x03E00008, 0, 0)}

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        rom = bytearray(0x4000)
        for image, offset in self.ROMS.items():
            rom[offset:offset + 16] = self.CODE[image]
        self.rom = bytes(rom)

    def tearDown(self):
        self._tmp.cleanup()

    def project(self, swap: bool = False) -> tuple[Path, dict]:
        project = self.root / ("swapped" if swap else "project")
        write(project / "GENERATED.json", "{}\n")
        layout, objects = {}, {}
        for image, offset in self.ROMS.items():
            function = f"{image}_func_{self.VRAM:08X}"
            row = {"unit": f"{image}/{function}", "image": image, "function": function, "vram": f"0x{self.VRAM:08X}",
                   "offset": f"0x{offset:X}", "size": "0x10", "code_size": "0x10", "status": "exact", "kind": "c",
                   "file": f"src/{image}/{function}.c", "recipe": {"pipeline": "kmc+as", "flags": "-O2"}}
            write(project / "provenance" / image / f"{function}.json", json.dumps(row))
            write(project / row["file"], f"/* {function} */\n")
            layout[image] = {"vram": f"0x{self.VRAM:08X}", "end": f"0x{self.VRAM + 0x10:08X}",
                             "rom": f"0x{offset:X}", "rom_end": f"0x{offset + 0x10:X}"}
            source = "ovl2" if (swap and image == "ovl1") else image
            objects[f"{function}.c"] = rl.write_elf([(".text", self.CODE[source], ALLOC | EXEC, 4, 0)],
                                                    [(function, 0, 16, ".text", 1, 2)])
        write(project / "config" / "layout.json", json.dumps(layout))
        write(project / "symbols" / "symbol_addrs.txt", "")
        write(project / "raw" / "manifest.tsv", f"# raw units\n# rom_sha1 {hashlib.sha1(self.rom).hexdigest()}\n")
        return project, objects

    def relink(self, project: Path, objects: dict, image: str) -> dict:
        def build_fn(source, spec, cfg, workdir):
            Path(workdir).mkdir(parents=True, exist_ok=True)
            path = Path(workdir) / (Path(source).stem + ".o")
            path.write_bytes(objects[Path(source).name])
            return path
        return rl.relink(project, image, self.root / f"build_{project.name}_{image}", data=self.rom,
                         delta=self.VRAM - self.ROMS[image], is_rom=True, rom_sha1=hashlib.sha1(self.rom).hexdigest(),
                         build_fn=build_fn, linker="kit")

    def test_each_overlay_links_at_the_shared_vram_and_matches_its_own_rom_range(self):
        project, objects = self.project()
        for image in self.ROMS:
            report = self.relink(project, objects, image)
            self.assertEqual(report["outcome"], "identical", (image, report.get("first_difference")))
            self.assertEqual(report["extent"]["vram"], f"0x{self.VRAM:08X}")

    def test_the_other_overlays_unit_at_the_same_vram_is_caught(self):
        project, objects = self.project(swap=True)
        report = self.relink(project, objects, "ovl1")
        self.assertEqual(report["outcome"], "different")
        self.assertIn("compile", report["first_difference"]["likely_step"])
        self.assertEqual(self.relink(project, objects, "ovl2")["outcome"], "identical")


class AssemblyLaneTest(unittest.TestCase):
    """Generated and hand-written assembly is rebuilt with the assembler the export proved it with."""

    def test_recorded_modern_assembler_selects_the_gnu_as_lane(self):
        from n64dk.match import projectgen as pg
        old = rl.Member("main/f", "f", BASE, 0, 8, 8, "assembly", "generated", "asm/generated/main/f.s",
                        {"pipeline": "as", "flags": "", "assembler": pg.ASSEMBLY_RECIPE})
        kmc = rl.Member("main/g", "g", BASE + 8, 8, 8, 8, "assembly", "handwritten", "asm/main/g.s",
                        {"pipeline": "as", "flags": ""})
        bare = rl.Member("main/h", "h", BASE + 16, 16, 8, 8, "assembly", "generated", "asm/generated/main/h.s", {})
        lanes = {t.members[0].function: t.pipeline for t in rl.targets([old, kmc, bare])}
        self.assertEqual(lanes, {"f": "gnu-as", "g": "as", "h": "gnu-as"})

    def test_generated_projects_record_the_gnu_as_lane(self):
        from n64dk.match import pipeline
        self.assertIn("gnu-as", pipeline.PIPELINES)



class RawUnitPinsTest(unittest.TestCase):
    """A raw stand-in's recipe still declares addresses other units refer to, when their origin is independent."""

    def test_only_independently_recorded_addresses_of_unlinked_units_are_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            write(project / "donors" / "main" / "pin_origins.json", json.dumps({
                "0x80001000": {"gMask": "transfer: 3 other function(s), e.g. func_80002000",
                               "gSelf": "self: only this unit's own bytes"},
                "0x80003000": {"gLinked": "transfer: 2 other function(s)"}}))
            flags = "--symbol=gMask:0x80010000 --symbol=gSelf:0x80010004 --symbol=gNone:0x80010008"
            raw = rl.Target("donors/main/src/t/a.s@80001000", [], "assumed", "donor-asm", "donors/main/src/t/a.s",
                            "as", flags, 0x80001000, 0x10)
            linked = rl.Target("donors/main/src/t/b.s@80003000", [], "exact", "donor-asm", "donors/main/src/t/b.s",
                               "as", "--symbol=gLinked:0x80010010", 0x80003000, 0x10)
            pins = rl.raw_unit_pins([raw, linked], {1: b""}, project, "main")
            self.assertEqual(pins, {"gMask": 0x80010000})


class UnallocatedRodataTest(unittest.TestCase):
    """KMC GCC 2.7.2's assembler can emit .rodata with flags WRITE only; it is still placed at its declared base."""

    def test_rodata_without_the_alloc_flag_is_placed_at_its_declared_base(self):
        code = words(0x3C010000, 0xD4200000, 0x03E00008, 0)
        obj = rl.write_elf([(".text", code, ALLOC | EXEC, 4, 0), (".rodata", b"\x3f\xf0" + b"\0" * 6, 0x1, 8, 0)],
                           [("f", 0, 16, ".text", 1, 2)])
        t = rl.Target("main/f", [], "exact", "c", "src/main/f.c", "kmc+as", "", BASE, 16)
        t.bases[".rodata"] = BASE + 0x100
        layout = rl.plan([t], {0: obj}, (BASE, BASE + 0x200), {})
        placed = {p.section: (p.vram, p.placement) for p in layout.pieces}
        self.assertEqual(placed[".rodata"], (BASE + 0x100, "declared base"))


class PaddingYieldTest(unittest.TestCase):
    """A data section's trailing zero padding gives way to the next unit's declared section; nothing else does."""

    def layout(self, first_section=".rodata", tail=b"\0" * 8, placement="declared base"):
        plan = rl.Plan(BASE, BASE + 0x40, BASE + 0x1000)
        plan.pieces = [rl.Piece(".a", 0, "a.o", first_section, 1, BASE, 0x20, 0x20, "exact", "declared base"),
                       rl.Piece(".b", 1, "b.o", ".rodata", 1, BASE + 0x18, 0x10, 0x10, "exact", placement)]
        linked = {".a": b"\x11" * 0x18 + tail, ".b": b"\x22" * 0x10}
        return plan, linked

    def test_zero_tail_yields_to_a_declared_neighbour(self):
        plan, linked = self.layout()
        result = rl.compare_image(plan, linked, b"\x11" * 0x18 + b"\x22" * 0x10 + b"\0" * 0x18)
        self.assertEqual((result.total, result.yielded, result.overlaps), (0, 8, []))

    def test_non_zero_bytes_and_code_never_yield(self):
        for plan, linked in (self.layout(tail=b"\0" * 7 + b"\1"), self.layout(first_section=".text"),
                             self.layout(placement="scratch")):
            result = rl.compare_image(plan, linked, b"\x11" * 0x18 + b"\x22" * 0x10 + b"\0" * 0x18)
            self.assertEqual(result.yielded, 0)
            self.assertTrue(result.overlaps)


class SymbolOrderTest(unittest.TestCase):
    """An object whose symbol table lists a local symbol after the first global one (GNU ld refuses it)."""

    def elf(self, bindings, first_global):
        import struct
        symtab = b"".join(struct.pack(">IIIBBH", 0, 0, 0, binding << 4, 0, 0) for binding in bindings)
        header = bytearray(0x34)
        header[:4] = b"\x7fELF"
        struct.pack_into(">I", header, 0x20, 0x34 + len(symtab))            # e_shoff
        struct.pack_into(">HH", header, 0x2E, 40, 2)                        # e_shentsize, e_shnum
        null = bytes(40)
        section = struct.pack(">IIIIIIIIII", 0, 2, 0, 0, 0x34, len(symtab), 0, first_global, 4, 16)
        return bytes(header) + symtab + null + section

    def test_a_local_symbol_after_sh_info_is_detected(self):
        from n64dk.match import relink
        self.assertFalse(relink.locals_after_globals(self.elf([0, 0, 0, 1, 1], 3)))
        self.assertTrue(relink.locals_after_globals(self.elf([0, 0, 0, 1, 0, 1, 0], 3)))
        self.assertFalse(relink.locals_after_globals(b"not an object"))
