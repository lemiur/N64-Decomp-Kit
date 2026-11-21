import json
import tempfile
import unittest
from pathlib import Path

from n64dk.interop import PROVENANCE, maps, objdiff, splat, symaddrs
from n64dk.interop.splat import YamlSubsetError, load_yaml


# A splat-style configuration written for these tests in the shapes public configurations use:
# options with lists and a block scalar, dict and list segments, flow subsegments with "auto",
# dict subsegments, a nested data group, bss-only segments sharing a ROM start, two overlays
# sharing one VRAM range, data segments without vram, comments, and a final [end] entry.
CONFIG = """\
name: Test Title (Synthetic)
sha1: 0123456789abcdef0123456789abcdef01234567
options:
  basename: test
  target_path: baserom.z64
  platform: n64
  compiler: KMC
  find_file_boundaries: True
  symbol_addrs_path:
    - symbol_addrs.txt
  auto_link_sections: [".data", ".rodata", ".bss"]
  generated_c_preamble: |
    #include "ultra64.h"
    #include "include_asm.h"
  asm_inc_header: ""
  subalign: null
  custom_unknown_option: 0x10   # carried, not read
segments:
  - name: header
    type: code
    start: 0x0
    vram: 0x8F000000
    subsegments:
      - [auto, c, rom_header]
      - [0x0, .data, rom_header]
  - name: boot
    type: code
    start: 0x1000
    vram: 0x80000400
    bss_size: 0x100
    subsegments:
      - [0x1000, hasm, entry]
      - [0x1060, c, main]   # two functions
      - [0x10A0, c, util]
      - [0x10C0, .data, main]
      - { type: .bss, vram: 0x80000500, name: main }
      - start: 0x10D0
        type: .data
        name: group
        subsegments:
          - [0x10D0]
          - [0x10D8, ci8, tex, 4, 2]
  - name: ovl_a
    type: code
    start: 0x2000
    vram: 0x80100000
    exclusive_ram_id: overlays
    subsegments:
      - [0x2000, c, ovl_a]
  - name: ovl_b
    type: code
    start: 0x3000
    vram: 0x80100000
    exclusive_ram_id: overlays
    subsegments:
      - [0x3000, c, ovl_b]
  - name: buffer
    type: code
    start: 0x4000
    vram: 0x80200000
    bss_size: 0x1000
    subsegments:
      - { start: 0x4000, type: .bss, vram: 0x80200000, name: buffer }
  - name: assets
    type: bin
    start: 0x4000
  - [0x4800, bin, more_assets]
  - [0x5000]
"""

SYMBOLS = """\
// text
entry_point = 0x80000400; // type:func size:0x60
main_func = 0x80000460; // type:func defined:true
helper = 0x80000480; //
util_func = 0x800004A0; // type:func size:0x20 name_end:util_end force_migration:true
D_800004C0 = 0x800004C0; // type:data size:0x10
ignored_thing = 0x80000400; // ignore:true
ovl_a_func = 0x80100000; // type:func size:0x40 rom:0x2000
ovl_b_func = 0x80100000; // type:func size:0x80 rom:0x3000
ovl_b_second = 0x80100080; // type:func segment:ovl_b
ambiguous_func = 0x80100100; // type:func
outside_func = 0x80300000; // type:func
odd = 0x80000470; // 0x10  (player, true)
"""


def _import(include_untyped=True):
    return splat.import_config(splat.read_config(CONFIG), [symaddrs.SymbolFile.parse(SYMBOLS)],
                               include_untyped=include_untyped)


class YamlSubsetTest(unittest.TestCase):
    def test_reads_the_shapes_splat_configurations_use(self):
        data = load_yaml(CONFIG)
        options = data["options"]
        self.assertEqual(options["symbol_addrs_path"], ["symbol_addrs.txt"])
        self.assertEqual(options["auto_link_sections"], [".data", ".rodata", ".bss"])
        self.assertEqual(options["generated_c_preamble"], '#include "ultra64.h"\n#include "include_asm.h"\n')
        self.assertEqual(options["asm_inc_header"], "")
        self.assertIsNone(options["subalign"])
        self.assertIs(options["find_file_boundaries"], True)
        self.assertEqual(options["custom_unknown_option"], 0x10)
        self.assertEqual(data["sha1"], "0123456789abcdef0123456789abcdef01234567")
        boot = data["segments"][1]
        self.assertEqual(boot["subsegments"][0], [0x1000, "hasm", "entry"])
        self.assertEqual(boot["subsegments"][4], {"type": ".bss", "vram": 0x80000500, "name": "main"})
        self.assertEqual(boot["subsegments"][5]["subsegments"], [[0x10D0], [0x10D8, "ci8", "tex", 4, 2]])
        self.assertEqual(data["segments"][0]["subsegments"][0], ["auto", "c", "rom_header"])
        self.assertEqual(data["segments"][-1], [0x5000])

    def test_sequence_at_its_keys_indentation_and_multiline_flow(self):
        data = load_yaml("a:\n- 1\n- two\nb: [1,\n   2, 3]\nc: 'it''s'\nd: \"x\\ty\"\n")
        self.assertEqual(data, {"a": [1, "two"], "b": [1, 2, 3], "c": "it's", "d": "x\ty"})

    def test_yaml_1_1_scalars(self):
        data = load_yaml("a: yes\nb: off\nc: ~\nd: 0x1F\ne: 010\nf: 1.5\ng: 1_000\n")
        self.assertEqual(data, {"a": True, "b": False, "c": None, "d": 0x1F, "e": 8, "f": 1.5, "g": 1000})

    def test_refuses_what_it_does_not_implement(self):
        for text in ("a: &x 1\nb: *x\n", "base: {a: 1}\nother:\n  <<: {b: 2}\n", "a: 1\n---\nb: 2\n",
                     "a: !!str 1\n", "a:\n\t- 1\n"):
            with self.subTest(text=text), self.assertRaises(YamlSubsetError):
                load_yaml(text)

    def test_json_configuration_accepted(self):
        config = splat.read_config(json.dumps({"options": {"platform": "n64"},
                                               "segments": [{"name": "x", "type": "code", "start": 4096,
                                                             "vram": 0x80000400,
                                                             "subsegments": [[4096, "c", "x"]]}, [8192]]}))
        kit = splat.import_config(config)
        self.assertEqual(list(kit.images), ["x"])
        self.assertEqual(kit.images["x"].rom_size, 0x1000)

    def test_dump_then_load_is_identity(self):
        data = load_yaml(CONFIG)
        self.assertEqual(load_yaml(splat.dump_yaml(data)), data)


class SymbolAddrsTest(unittest.TestCase):
    def test_unchanged_round_trip_is_byte_identical(self):
        text = SYMBOLS + "\n// trailing comment\nD_00000000 = 0x00000000; // ignore:true\nD_00000000 = 0x00000000; // ignore:true\n"
        self.assertEqual(symaddrs.SymbolFile.parse(text).render(), text)

    def test_unknown_fields_survive_a_change(self):
        parsed = symaddrs.SymbolFile.parse(SYMBOLS)
        util = parsed.by_name()["util_func"][0]
        util.set("size", "0x24")
        rendered = parsed.render()
        line = next(l for l in rendered.splitlines() if l.startswith("util_func"))
        self.assertEqual(line, "util_func = 0x800004A0; // type:func size:0x24 name_end:util_end force_migration:true")
        odd = symaddrs.SymbolFile.parse(rendered).by_name()["odd"][0]
        self.assertEqual(odd.free_text, ["0x10", "(player,", "true)"])
        self.assertEqual(rendered.count("\n"), SYMBOLS.count("\n"))

    def test_attributes_read(self):
        entries = symaddrs.SymbolFile.parse(SYMBOLS).by_name()
        self.assertEqual(entries["ovl_b_func"][0].rom, 0x3000)
        self.assertEqual(entries["ovl_b_second"][0].segment, "ovl_b")
        self.assertTrue(entries["ignored_thing"][0].ignored)
        self.assertIsNone(entries["helper"][0].type)


class ImportTest(unittest.TestCase):
    def test_overlays_sharing_vram_stay_distinct_by_vram_and_rom(self):
        kit = _import()
        self.assertEqual(sorted(kit.images), ["boot", "ovl_a", "ovl_b"])
        self.assertEqual(kit.images["ovl_a"].segment.identity, (0x80100000, 0x2000))
        self.assertEqual(kit.images["ovl_b"].segment.identity, (0x80100000, 0x3000))
        self.assertEqual(kit.overlapping(), [("ovl_a", "ovl_b")])
        self.assertEqual([(f.name, f.vram, f.size) for f in kit.images["ovl_a"].functions],
                         [("ovl_a_func", 0x80100000, 0x40)])
        self.assertEqual([(f.name, f.vram, f.size) for f in kit.images["ovl_b"].functions],
                         [("ovl_b_func", 0x80100000, 0x80), ("ovl_b_second", 0x80100080, 0xF80)])
        reasons = {row["name"]: row["reason"] for row in kit.unassigned}
        self.assertTrue(reasons["ambiguous_func"].startswith("ambiguous: ovl_a, ovl_b"))
        self.assertEqual(reasons["outside_func"], "outside every imported code segment")

    def test_addresses_unchanged_and_sizes_marked(self):
        boot = {f.name: f for f in _import().images["boot"].functions}
        self.assertEqual(boot["entry_point"].vram, 0x80000400)
        self.assertEqual((boot["entry_point"].size, boot["entry_point"].size_source), (0x60, "attribute"))
        self.assertEqual((boot["main_func"].size, boot["main_func"].size_source), (0x10, "inferred: next function"))
        self.assertEqual((boot["helper"].size, boot["helper"].symbol_type),
                         (0x20, "untyped symbol in a text unit"))
        self.assertEqual(boot["util_func"].size, 0x20)
        self.assertNotIn("D_800004C0", boot)
        self.assertNotIn("ignored_thing", boot)
        self.assertEqual(sorted(_import(include_untyped=False).images["boot"].functions[i].name for i in range(3)),
                         ["entry_point", "main_func", "util_func"])

    def test_skipped_segments_say_why(self):
        skipped = {row["name"]: row["reason"] for row in _import().skipped_segments}
        self.assertEqual(skipped["header"], "no text subsegment with a known start")
        self.assertTrue(skipped["buffer"].startswith("no text subsegment") or skipped["buffer"].startswith("no ROM"))
        self.assertEqual(skipped["assets"], "no explicit vram")
        self.assertEqual(_import().rom_end, 0x5000)

    def test_provenance_everywhere_and_never_an_exact_status(self):
        kit = _import()
        texts = [kit.registry_toml(), *(kit.catalogue_toml(n) for n in kit.images), json.dumps(kit.as_dict())]
        for text in texts:
            self.assertIn(PROVENANCE, text)
            self.assertNotIn("exact", text.lower())
            self.assertNotIn("status", text.lower())
        for image in kit.images.values():
            self.assertTrue(all(f.provenance.startswith(PROVENANCE) for f in image.functions))

    def test_written_import_reads_back_through_the_kit_registry(self):
        from n64dk.match import images as imagemod
        kit = _import()
        with tempfile.TemporaryDirectory() as tmp:
            written = splat.write_import(kit, Path(tmp) / "out")
            self.assertEqual(written[0].name, "n64dk.imported.toml")
            registry = imagemod.load_registry(written[0])
            self.assertEqual(sorted(registry), ["boot", "ovl_a", "ovl_b"])
            self.assertEqual(registry["ovl_b"].vram_delta, 0x80100000 - 0x3000)
            self.assertEqual(registry["ovl_b"].raw_segments, [(0x3000, 0x3000, 0x1000, "copy")])
            self.assertEqual(registry["ovl_b"].rows(), [("ovl_b_func", 0x80100000, 0x80),
                                                        ("ovl_b_second", 0x80100080, 0xF80)])
            with self.assertRaises(FileExistsError):
                splat.write_import(kit, Path(tmp) / "out")


class ExportTest(unittest.TestCase):
    MARKER = b"ROMBYTES-MUST-NOT-LEAK" * 8

    def _project(self, root: Path) -> Path:
        rom = bytearray(0x5000)
        rom[0x2000:0x2000 + len(self.MARKER)] = self.MARKER
        (root / "baserom.z64").write_bytes(bytes(rom))
        (root / "cat_a.toml").write_text(
            '[[section]]\nname = "a"\nrom = 0x2000\nvram = 0x80100000\nsize = 0x100\n\nfunctions = [\n'
            '    { name = "func_80100000", vram = 0x80100000, size = 0x40 },\n'
            '    { name = "a_second", vram = 0x80100040, size = 0x20 },\n]\n')
        (root / "cat_b.toml").write_text(
            '[[section]]\nname = "b"\nrom = 0x3000\nvram = 0x80100000\nsize = 0x200\n\nfunctions = [\n'
            '    { name = "func_80100000", vram = 0x80100000, size = 0x80 },\n]\n')
        (root / "n64dk.toml").write_text(
            '[images.ovl_a]\npath = "baserom.z64"\nvram_delta = 0x800FE000\ncatalogue = "cat_a.toml"\n'
            'raw_segments = [[0x2000, 0x2000, 0x100, "copy"]]\n\n'
            '[images.ovl_b]\npath = "baserom.z64"\nvram_delta = 0x800FD000\ncatalogue = "cat_b.toml"\n'
            'raw_segments = [[0x3000, 0x3000, 0x200, "copy"]]\n\n'
            '[images.raw]\npath = "seg.bin"\nvram_delta = 0x80200000\n')
        return root

    def test_export_round_trip_keeps_addresses_and_overlays_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, skipped = splat.kit_images(self._project(Path(tmp)))
        self.assertEqual([i.name for i in images], ["ovl_a", "ovl_b"])
        self.assertEqual(skipped[0]["image"], "raw")
        yaml_text = splat.export_yaml(images, title="Synthetic", rom_end=0x5000)
        symbols_text, report = splat.export_symbol_addrs(images)
        self.assertEqual(report["rom_attribute_on"], ["ovl_a", "ovl_b"])
        self.assertEqual(report["renamed"], [{"name": "func_80100000", "image": "ovl_b",
                                              "written_as": "func_80100000_ovl_b"}])
        for text in (yaml_text, symbols_text):
            self.assertNotIn(self.MARKER.decode(), text)
            self.assertNotIn(self.MARKER[:16].hex(), text.lower())
        back = splat.import_config(splat.read_config(yaml_text), [symaddrs.SymbolFile.parse(symbols_text)])
        self.assertEqual({n: (i.vram, i.rom_start, i.rom_size) for n, i in back.images.items()},
                         {"ovl_a": (0x80100000, 0x2000, 0x100), "ovl_b": (0x80100000, 0x3000, 0x200)})
        self.assertEqual([(f.vram, f.size) for f in back.images["ovl_a"].functions],
                         [(0x80100000, 0x40), (0x80100040, 0x20)])
        self.assertEqual([(f.name, f.vram, f.size) for f in back.images["ovl_b"].functions],
                         [("func_80100000_ovl_b", 0x80100000, 0x80)])
        self.assertIn(PROVENANCE, back.provenance)
        self.assertIn("exported by n64dk", back.provenance)              # the export's note survives
        self.assertEqual(back.unassigned, [])

    def test_existing_symbol_file_is_kept_and_extended(self):
        existing = symaddrs.SymbolFile.parse("// mine\nfunc_80100000 = 0x80100000; // type:func size:0x40 custom:keep\n"
                                             "a_second = 0x80100044; // type:func\n")
        image = splat.KitImage("ovl_a", 0x2000, 0x80100000, 0x100,
                               [("func_80100000", 0x80100000, 0x40), ("a_second", 0x80100040, 0x20),
                                ("a_third", 0x80100060, 0x10)])
        text, report = splat.export_symbol_addrs([image], existing)
        self.assertTrue(text.startswith("// mine\nfunc_80100000 = 0x80100000; // type:func size:0x40 custom:keep\n"))
        self.assertIn("a_third = 0x80100060; // type:func size:0x10", text)
        self.assertEqual(report["conflicts"], [{"name": "a_second", "kit": "0x80100040", "existing": ["0x80100044"]}])
        self.assertEqual((report["added"], report["kept_existing"]), (1, 1))


GNU_MAP = """\
Archive member included to satisfy reference by file (symbol)

Memory Configuration

Name             Origin             Length             Attributes
*default*        0x0000000000000000 0xffffffffffffffff

Linker script and memory map

                0x0000000000000000                __romPos = 0x0
                0x0000000080000400                . = 0x80000400

.main           0x0000000080000400      0x140 load address 0x0000000000001000
                0x0000000080000400                _mainSegmentStart = ADDR (.main)
 *(.text)
 .text          0x0000000080000400       0xc0 build/src/main.o
                0x0000000080000400                main_func
                0x0000000080000460                other_func
 .text          0x00000000800004c0        0x0 build/src/empty.o
 .text.unlikely_long_section_name
                0x00000000800004c0       0x40 build/src/long.o
                0x00000000800004c0                long_func
 *fill*         0x0000000080000500       0x10
 .data          0x0000000080000510       0x30 /lib/libultra.a(osSetTime.o)
                0x0000000080000510                osTimeData
 COMMON         0x0000000080000540        0x0 build/src/main.o
                0x0000000000001140                __romPos = (__romPos + SIZEOF (.main))

.overlay_with_a_long_name
                0x0000000080100000       0x20 load address 0x0000000000002000
 .text          0x0000000080100000       0x20 build/src/ovl.o
                0x0000000080100000                ovl_func
OUTPUT(build/test.elf elf32-tradbigmips)
"""


class MapsTest(unittest.TestCase):
    def test_gnu_map_symbols_objects_and_rom(self):
        facts = maps.read_map(GNU_MAP)
        self.assertEqual(facts.format, "gnu-ld")
        self.assertTrue(facts.format_verified)
        symbols = {s.name: s for s in facts.symbols if s.kind == "symbol"}
        self.assertEqual(sorted(symbols), ["long_func", "main_func", "osTimeData", "other_func", "ovl_func"])
        self.assertEqual((symbols["other_func"].address, symbols["other_func"].rom), (0x80000460, 0x1060))
        self.assertEqual(symbols["other_func"].object, "build/src/main.o")
        self.assertEqual((symbols["long_func"].object, symbols["long_func"].rom), ("build/src/long.o", 0x10C0))
        self.assertEqual(symbols["osTimeData"].object, "/lib/libultra.a(osSetTime.o)")
        self.assertEqual((symbols["ovl_func"].output_section, symbols["ovl_func"].rom), (".overlay_with_a_long_name", 0x2000))
        assignments = {s.name for s in facts.symbols if s.kind == "assignment"}
        self.assertEqual(assignments, {"__romPos", "_mainSegmentStart"})
        order = maps.object_order(facts)
        self.assertEqual([o["object"] for o in order[".main"]],
                         ["build/src/main.o", "build/src/long.o", "/lib/libultra.a(osSetTime.o)", "build/src/main.o"])
        self.assertEqual(order[".main"][0]["provenance"], PROVENANCE)
        self.assertEqual(len(facts.output_sections), 2)

    def test_map_symbols_as_symbol_file(self):
        text = maps.read_map(GNU_MAP).symbol_file().render()
        self.assertIn("main_func = 0x80000400; // rom:0x1000", text)
        self.assertNotIn("__romPos", text)

    def test_column_map_with_and_without_header(self):
        facts = maps.read_map("ADDRESS  SYMBOL      OBJECT\n80000400 main_func main.o\n0x80000480 helper util.o\n"
                              "garbage line here\n")
        self.assertEqual(facts.format, "columns")
        self.assertFalse(facts.format_verified)
        self.assertEqual([(s.name, s.address, s.object) for s in facts.symbols],
                         [("main_func", 0x80000400, "main.o"), ("helper", 0x80000480, "util.o")])
        self.assertEqual(facts.skipped_lines, 1)
        reordered = maps.read_column_map("main.o .text 0x80000400 main_func\n",
                                         ["object", "section", "address", "symbol"])
        self.assertEqual((reordered.symbols[0].name, reordered.symbols[0].section), ("main_func", ".text"))


class ObjdiffTest(unittest.TestCase):
    def test_units_from_tree_and_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in ("build/expected/main/a.o", "build/expected/main/b.o", "build/base/main/a.o", "src/main/a.c"):
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_bytes(b"")
            view = objdiff.from_export(root)
            self.assertTrue(view["generated"])
            self.assertEqual(view["document"]["units"], [
                {"name": "main/a", "target_path": "build/expected/main/a.o", "base_path": "build/base/main/a.o",
                 "metadata": {"source_path": "src/main/a.c"}},
                {"name": "main/b", "target_path": "build/expected/main/b.o"}])
            minimal = objdiff.project(objdiff.units_from_tree(root, "build/expected", "build/base"), minimal=True)
            self.assertEqual(set(minimal), {"units"})
            path = objdiff.write(root / "objdiff.json", minimal)
            with self.assertRaises(FileExistsError):
                objdiff.write(path, minimal)

    def test_no_objects_no_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            view = objdiff.from_export(tmp)
        self.assertFalse(view["generated"])
        self.assertIn("no per-unit objects", view["reason"])


if __name__ == "__main__":
    unittest.main()
