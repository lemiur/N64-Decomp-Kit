"""Synthetic corpus: generation, linker script, ROM, ground truth and the builder with injected tools."""
import json
import re
import struct
import tempfile
import unittest
from pathlib import Path

from n64dk.elf import Elf
from n64dk.match import synthetic as syn
from n64dk.match.pipeline import BuildError
from n64dk.rom import Rom


def linked_elf(sections, symbols) -> bytes:
    """An ELF32 big-endian executable-like file: ``sections`` [(name, type, data or NOBITS size)],
    ``symbols`` [(name, value, size, section index, info)]; .symtab, .strtab, .shstrtab appended."""
    names = [name for name, _t, _d in sections] + [".symtab", ".strtab", ".shstrtab"]
    shstr = b"\0" + b"".join(name.encode() + b"\0" for name in names)
    strtab = b"\0" + b"".join(name.encode() + b"\0" for name, *_rest in symbols)
    symtab = bytes(16) + b"".join(
        struct.pack(">IIIBBH", strtab.index(b"\0" + name.encode() + b"\0") + 1, value, size, info, 0, index)
        for name, value, size, index, info in symbols)
    bodies = [data if kind != 8 else b"" for _n, kind, data in sections] + [symtab, strtab, shstr]
    offsets, cursor = [], 52
    for body in bodies:
        offsets.append(cursor)
        cursor += len(body)
    headers = [bytes(40)]
    for number, (name, kind, data) in enumerate(sections):
        size = data if kind == 8 else len(data)
        headers.append(struct.pack(">10I", shstr.index(b"\0" + name.encode() + b"\0") + 1, kind, 3, 0,
                                   offsets[number], size, 0, 0, 16, 0))
    base = len(sections)
    symtab_index = base + 1
    headers.append(struct.pack(">10I", shstr.index(b"\0.symtab\0") + 1, 2, 0, 0, offsets[base], len(symtab),
                               symtab_index + 1, 1, 4, 16))
    headers.append(struct.pack(">10I", shstr.index(b"\0.strtab\0") + 1, 3, 0, 0, offsets[base + 1], len(strtab),
                               0, 0, 1, 0))
    headers.append(struct.pack(">10I", shstr.index(b"\0.shstrtab\0") + 1, 3, 0, 0, offsets[base + 2], len(shstr),
                               0, 0, 1, 0))
    return (b"\x7fELF\x01\x02\x01" + bytes(9) + struct.pack(">HHIIIIIHHHHHH", 2, 8, 1, 0x80000400, 0, cursor, 0, 52, 0,
                                                             0, 40, len(headers), len(headers) - 1)
            + b"".join(bodies) + b"".join(headers))


def small_program() -> syn.Program:
    return syn.Program(1, [
        syn.Unit("entry", "asm", "main", "\t.set\tnoreorder\n", [syn.FunctionTruth("_start", "entry", "main", "entry", "asm")]),
        syn.Unit("u0", "c", "main", "int syn_main;\n",
                 [syn.FunctionTruth("syn_main", "u0", "main", "call", "void (void)"),
                  syn.FunctionTruth("syn_loop_1", "u0", "main", "call", "int (int)", ("loop",)),
                  syn.FunctionTruth("syn_helper_2", "u0", "main", "static-call", "int (int)", ("static",), True)],
                 [syn.DataTruth("syn_buffer_3", "u0", "int[16]", "bss", True),
                  syn.DataTruth("syn_grid_4", "u0", "int[4][4]", "data")]),
        syn.Unit("ovl", "c", "overlay", "int ovl;\n", [syn.FunctionTruth("syn_overlay_5", "ovl", "overlay", "call",
                                                                          "int (int)", ("overlay",))])],
        ["loop", "static", "overlay"])


MAIN, BSS, OVL = 1, 2, 3
SYMBOLS = [("_start", 0x80000400, 0, MAIN, 0x10), ("syn_main", 0x80000410, 0x10, MAIN, 0x12),
           ("syn_loop_1", 0x80000420, 0, MAIN, 0x12), ("syn_grid_4", 0x80000430, 0x10, MAIN, 0x11),
           ("syn_buffer_3", 0x80000440, 0x40, BSS, 0x01), ("syn_overlay_5", 0x80400000, 0, OVL, 0x12),
           ("main_TEXT_START", 0x80000400, 0, MAIN, 0x10), ("main_TEXT_END", 0x80000430, 0, MAIN, 0x10),
           ("main_SDATA_START", 0x80000440, 0, MAIN, 0x10), ("main_SDATA_END", 0x80000440, 0, MAIN, 0x10),
           ("main_DATA_END", 0x80000440, 0, MAIN, 0x10), ("main_BSS_START", 0x80000440, 0, BSS, 0x10),
           ("main_SBSS_START", 0x80000440, 0, BSS, 0x10), ("main_SBSS_END", 0x80000440, 0, BSS, 0x10),
           ("main_BSS_END", 0x80000480, 0, BSS, 0x10), ("ovl_TEXT_START", 0x80400000, 0, OVL, 0x10),
           ("ovl_TEXT_END", 0x80400010, 0, OVL, 0x10), ("ovl_DATA_END", 0x80400010, 0, OVL, 0x10)]
LINKED = linked_elf([(".main", 1, bytes(range(64))), (".main.bss", 8, 0x40), (".ovl", 1, b"\x22" * 16)], SYMBOLS)


class GenerateTest(unittest.TestCase):
    def test_deterministic(self):
        one, two = syn.generate(5), syn.generate(5)
        self.assertEqual(one.as_dict(), two.as_dict())
        self.assertEqual([u.source for u in one.units], [u.source for u in two.units])
        self.assertNotEqual([u.source for u in one.units], [u.source for u in syn.generate(6).units])

    def test_feature_coverage_and_shape(self):
        program = syn.generate(11)
        self.assertLessEqual(set(syn.FEATURES), set(program.features))
        self.assertEqual(program.units[0].name, "entry")
        self.assertIn("ovl", [u.name for u in program.units])
        self.assertEqual(next(u for u in program.units if u.name == "ovl").segment, "overlay")
        for unit in program.units:
            for function in unit.functions:
                marker = f"{function.name}:" if unit.language == "asm" else f"{function.name}("
                self.assertIn(marker, unit.source, function.name)
            if unit.language == "c":
                self.assertEqual(unit.source.count("{"), unit.source.count("}"), unit.name)
                self.assertEqual(unit.source.count("("), unit.source.count(")"), unit.name)
                self.assertNotIn("//", unit.source)
                self.assertNotIn("$", unit.source)
                self.assertIn("class synthetic", unit.source)
                self.assertTrue(re.search(r"int syn_knr_\d+\(\);", unit.source))

    def test_calls_between_units(self):
        program = syn.generate(11)
        owner = {f.name: u.name for u in program.units for f in u.functions}
        cross = next(f for u in program.units for f in u.functions if "multi-unit" in f.features)
        source = next(u.source for u in program.units if u.name == cross.unit)
        callee = re.search(cross.name + r"\(int v\)\n\{\n    return (\w+)\(", source).group(1)
        self.assertNotEqual(owner[callee], cross.unit)

    def test_function_pointers_only(self):
        program = syn.generate(11)
        u0 = next(u for u in program.units if u.name == "u0")
        body = u0.source.split("void syn_main(void)\n{", 1)[1]
        pointers = [f.name for u in program.units for f in u.functions if f.reached_by == "pointer"]
        self.assertTrue(pointers)
        self.assertIn("syn_table[i](i)", body)
        for name in pointers:
            self.assertNotIn(name + "(", body)
        self.assertIn("syn_table", [d.name for d in u0.data])

    def test_small_request(self):
        program = syn.generate(3, features=("loop",), units=1, extra=0)
        self.assertEqual([u.name for u in program.units], ["entry", "u0"])
        names = [f.name for u in program.units for f in u.functions]
        self.assertEqual(names, ["_start", "syn_loop_0", "syn_main"])
        self.assertNotIn("asm_fill", program.units[0].source)


class LayoutTest(unittest.TestCase):
    def test_linker_script(self):
        script = syn.linker_script([("entry.o", "main"), ("u0.o", "main"), ("ovl.o", "overlay")])
        self.assertIn("ENTRY(_start)", script)
        self.assertIn("EXCLUDE_FILE(ovl.o)", script)
        self.assertIn(".ovl 0x80400000 : AT(0x1000 + SIZEOF(.main))", script)
        self.assertLess(script.index("entry.o(.text)"), script.index("*(EXCLUDE_FILE(ovl.o) .text"))
        self.assertIn("_gp = . + 0x7FF0;", script)
        plain = syn.linker_script([("entry.o", "main"), ("u0.o", "main")])
        self.assertNotIn("EXCLUDE_FILE", plain)
        self.assertNotIn(".ovl", plain)

    def test_rom(self):
        rom = syn.make_rom(b"\x11" * 16, b"\x22" * 8, 0x80000400)
        self.assertEqual(len(rom), syn.ROM_MINIMUM)
        self.assertEqual(rom[0x1000:0x1010], b"\x11" * 16)
        self.assertEqual(rom[0x1010:0x1018], b"\x22" * 8)
        parsed = Rom(rom)
        self.assertEqual(parsed.header.entry, 0x80000400)
        self.assertEqual(parsed.header.title, "N64DK SYNTHETIC")

    def test_truth_from_elf(self):
        truth = syn.truth_from_elf(small_program(), Elf(LINKED))
        rows = {row["name"]: row for row in truth["functions"]}
        self.assertEqual((rows["_start"]["vram"], rows["_start"]["size"], rows["_start"]["size_source"]),
                         (0x80000400, 0x10, "next start"))
        self.assertEqual((rows["syn_main"]["size"], rows["syn_main"]["size_source"]), (0x10, "symbol"))
        self.assertEqual(rows["syn_loop_1"]["size"], 0x10)
        self.assertFalse(rows["syn_helper_2"]["present"])
        self.assertEqual(rows["syn_overlay_5"]["size"], 0x10)
        regions = {row["name"]: row["region"] for row in truth["data"]}
        self.assertEqual(regions, {"syn_buffer_3": "bss", "syn_grid_4": "data"})
        self.assertEqual(truth["layout"]["entry"], 0x80000400)
        self.assertEqual(truth["layout"]["main"]["size"], 64)
        self.assertEqual(syn.region(truth["layout"], 0x80400004), "ovl-text")
        self.assertEqual(syn.region(truth["layout"], 0x90000000), "outside")


class BuildTest(unittest.TestCase):
    def tools(self, fail_unit=None, fail_link=False):
        calls = []

        def build(path, spec, cfg, workdir):
            calls.append((Path(path).name, spec.pipeline, list(spec.flags)))
            if Path(path).stem == fail_unit:
                raise BuildError("cc1: parse error")
            Path(workdir).mkdir(parents=True, exist_ok=True)
            obj = Path(workdir) / "x.o"
            obj.write_bytes(b"object")
            return obj

        def link(cfg, folder, script, objects, output, timeout):
            calls.append(("link", tuple(objects)))
            self.assertTrue((Path(folder) / script).is_file())
            if fail_link:
                raise RuntimeError("link failed: undefined")
            path = Path(folder) / output
            path.write_bytes(LINKED)
            return path
        return build, link, calls

    def test_built(self):
        build, link, calls = self.tools()
        with tempfile.TemporaryDirectory() as folder:
            manifest = syn.build_program(small_program(), "kmc+as", ["-O2", "-G0"], None, folder, build=build, link=link)
            self.assertEqual(manifest["status"], "built")
            self.assertEqual(calls[0], ("entry.s", "as", []))
            self.assertEqual(calls[1], ("u0.c", "kmc+as", ["-O2", "-G0"]))
            self.assertEqual(calls[-1], ("link", ("entry.o", "u0.o", "ovl.o")))
            self.assertTrue((Path(folder) / "rom.z64").is_file())
            stored = json.loads((Path(folder) / "manifest.json").read_text())
            self.assertEqual(syn.assert_synthetic(stored)["class"], "synthetic")
            self.assertEqual(stored["layout"]["ovl"]["rom"], 0x1000 + 64)
            self.assertEqual(stored["absent_functions"], ["syn_helper_2"])
            self.assertEqual(syn.discovery_truth(stored),
                             [("_start", 0x80000400, 0x10), ("syn_main", 0x80000410, 0x10),
                              ("syn_loop_1", 0x80000420, 0x10)])
            rom = (Path(folder) / "rom.z64").read_bytes()
            self.assertEqual(rom[0x1000:0x1040], bytes(range(64)))

    def test_build_failure_is_recorded(self):
        build, link, _calls = self.tools(fail_unit="u0")
        with tempfile.TemporaryDirectory() as folder:
            manifest = syn.build_program(small_program(), "ido5.3", ["-O2"], None, folder, build=build, link=link)
            self.assertEqual(manifest["status"], "build-failed")
            self.assertEqual(manifest["errors"][0]["unit"], "u0")
            self.assertTrue((Path(folder) / "manifest.json").is_file())

    def test_link_failure_is_recorded(self):
        build, link, _calls = self.tools(fail_link=True)
        with tempfile.TemporaryDirectory() as folder:
            manifest = syn.build_program(small_program(), "kmc+as", ["-O2"], None, folder, build=build, link=link)
            self.assertEqual(manifest["status"], "link-failed")
            self.assertEqual(manifest["errors"][0]["stage"], "link")

    def test_retail_is_refused(self):
        for bad in ({"class": "retail", "schema": syn.SCHEMA}, {"class": "synthetic"}, None):
            with self.assertRaises(ValueError):
                syn.assert_synthetic(bad)


if __name__ == "__main__":
    unittest.main()
