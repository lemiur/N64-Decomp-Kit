"""Build graph and link-order reconstruction on hand-built catalogues (no toolchain, no ROM)."""
from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from n64dk.analysis import linklayout as ll

TEXT = 0x80000400


def two_anchored_and_one_free() -> ll.LinkGraph:
    """A and B are placed in .text, A directly before B; C has no placed piece at all."""
    graph = ll.LinkGraph()
    graph.add_segment("main", TEXT, 0x1000)
    graph.add_unit("A", "main")
    graph.add_piece("A", ".text", TEXT, 0x10)
    graph.add_unit("B", "main")
    graph.add_piece("B", ".text", TEXT + 0x10, 0x10)
    graph.add_unit("C", "main")
    return graph


def words(*values: int) -> bytes:
    return struct.pack(f">{len(values)}I", *values)


class OrderTests(unittest.TestCase):
    def test_ambiguous_order_is_retained(self):
        solution = ll.solve(two_anchored_and_one_free())
        main = solution["segments"]["main"]
        self.assertEqual(main["orderings"], 3)                      # C before A, between, after B
        self.assertTrue(main["complete"])
        self.assertTrue(solution["consistent"])
        self.assertEqual(main["order"], ["A", "B", "C"])            # representative: placed units first
        self.assertEqual(main["positions"]["C"], [0, 2])
        self.assertEqual(main["ambiguous_pair_count"], 2)           # (A, C) and (B, C)
        first, second = main["edges"]
        self.assertTrue(first["forced"])
        self.assertEqual(first["evidence"], 1.0)
        self.assertFalse(second["forced"])
        self.assertEqual(second["evidence"], 0.0)
        self.assertLess(second["confidence"], 0.5)
        rows = {row["unit"]: row for row in ll.manifest(two_anchored_and_one_free(), solution)}
        self.assertEqual(rows["C"]["positions"], "0-2")
        self.assertEqual(rows["A"]["object"], "build/main/A.o")

    def test_data_order_resolves_the_position_text_leaves_open(self):
        graph = two_anchored_and_one_free()
        graph.add_piece("A", ".data", 0x80001000, 0x10)
        graph.add_piece("C", ".data", 0x80001010, 0x10)
        graph.add_piece("B", ".data", 0x80001020, 0x10)
        main = ll.solve(graph)["segments"]["main"]
        self.assertEqual(main["orderings"], 1)
        self.assertEqual(main["order"], ["A", "C", "B"])
        self.assertTrue(all(edge["forced"] and edge["evidence"] == 1.0 for edge in main["edges"]))
        self.assertIn(".data", main["edges"][0]["provenance"][0])
        self.assertEqual(main["positions"], {})

    def test_contradiction_is_reported_with_its_constraints(self):
        graph = ll.LinkGraph()
        graph.add_segment("main", TEXT)
        for name, text, rodata in (("A", TEXT, 0x80002008), ("B", TEXT + 0x10, 0x80002000)):
            graph.add_unit(name, "main")
            graph.add_piece(name, ".text", text, 0x10)
            graph.add_piece(name, ".rodata", rodata, 8, confidence=0.8, provenance="test rodata")
        solution = ll.solve(graph)
        self.assertFalse(solution["consistent"])
        cycles = [c for c in solution["contradictions"] if c["kind"] == "cycle"]
        self.assertEqual(len(cycles), 1)
        rejected = [c for c in graph.constraints if c.status == "rejected"]
        self.assertEqual(sorted(c.id for c in rejected), sorted(cycles[0]["rejected"]))
        self.assertEqual({c.section for c in rejected}, {".rodata"})          # the weaker evidence
        self.assertEqual({c.kind for c in rejected}, {"before", "adjacent"})
        sections = {c["section"] for c in cycles[0]["constraints"]}
        self.assertEqual(sections, {".text", ".rodata"})                     # both sides are reported
        self.assertEqual(solution["segments"]["main"]["order"], ["A", "B"])
        self.assertIn("CONTRADICTION (cycle)", ll.linker_script(graph, solution))

    def test_archive_members_stay_together(self):
        graph = two_anchored_and_one_free()
        graph.add_unit("L2", "main")
        self.assertEqual(ll.solve(graph)["segments"]["main"]["orderings"], 12)
        graph.group(("C", "L2"), 0.6, "test archive")
        main = ll.solve(graph)["segments"]["main"]
        self.assertEqual(main["orderings"], 6)                              # block of two, three places, two orders
        order = main["order"]
        self.assertEqual(abs(order.index("C") - order.index("L2")), 1)

    def test_sizes_choose_the_gap_a_piece_fits(self):
        graph = ll.LinkGraph()
        graph.add_segment("main", TEXT, sections={".data": (0x80001000, 0x80001048)})
        graph.add_unit("A", "main")
        graph.add_piece("A", ".text", TEXT, 0x10)
        graph.add_piece("A", ".data", 0x80001000, 0x10)
        graph.add_unit("B", "main")
        graph.add_piece("B", ".text", TEXT + 0x10, 0x10)
        graph.add_piece("B", ".data", 0x80001030, 0x10)
        graph.add_unit("C", "main")
        graph.add_piece("C", ".data", None, 0x20, 4, provenance="test: size known, place unknown")
        main = ll.solve(graph)["segments"]["main"]
        self.assertEqual(main["orderings"], 1)
        self.assertEqual(main["order"], ["A", "C", "B"])
        self.assertEqual([h["kind"] for h in graph.holes], ["unexplained"])

    def test_unsatisfiable_group_is_set_aside_and_reported(self):
        graph = two_anchored_and_one_free()
        graph.add_unit("D", "main")
        graph.add_piece("D", ".text", TEXT + 0x20, 0x10)
        graph.group(("A", "D"), 0.6, "test archive")                         # B lies between them in .text
        solution = ll.solve(graph)
        kinds = [c["kind"] for c in solution["contradictions"]]
        self.assertIn("unsatisfiable", kinds)
        self.assertEqual(solution["segments"]["main"]["order"][:3], ["A", "B", "D"])


class OverlayTests(unittest.TestCase):
    def test_exclusive_overlays_have_separate_orders(self):
        graph = ll.LinkGraph()
        graph.add_segment("ovl1", 0x80400000, 0x100000, exclusive="ovl")
        graph.add_segment("ovl2", 0x80400000, 0x110000, exclusive="ovl")
        graph.add_unit("X1", "ovl1")
        graph.add_piece("X1", ".text", 0x80400000, 0x20)
        graph.add_unit("X2", "ovl1")
        graph.add_piece("X2", ".text", 0x80400020, 0x10)
        graph.add_unit("Y1", "ovl2")
        graph.add_piece("Y1", ".text", 0x80400000, 0x40)                      # same addresses: not an overlap
        claim = graph.before("X2", "Y1", 0.9, "test: an order across exclusive overlays")
        self.assertEqual(claim.status, "invalid")
        solution = ll.solve(graph)
        kinds = [c["kind"] for c in solution["contradictions"]]
        self.assertEqual(kinds, ["exclusive-overlays"])
        self.assertEqual(solution["segments"]["ovl1"]["order"], ["X1", "X2"])
        self.assertEqual(solution["segments"]["ovl2"]["order"], ["Y1"])
        script = ll.linker_script(graph, solution)
        self.assertIn(".ovl1 0x80400000 : AT(ovl1_ROM_START)", script)
        self.assertIn(".ovl2 0x80400000 : AT(ovl2_ROM_START)", script)
        self.assertIn("overlay group ovl: mutually exclusive with ovl2", script)
        self.assertIn("__romPos = 0x110000;", script)

    def test_cross_segment_claim_is_invalid(self):
        graph = ll.LinkGraph()
        graph.add_segment("main", TEXT)
        graph.add_segment("other", 0x80300000)
        graph.add_unit("A", "main")
        graph.add_unit("Z", "other")
        graph.before("A", "Z", 1.0, "test")
        self.assertEqual([c["kind"] for c in ll.solve(graph)["contradictions"]], ["cross-segment"])


class GapTests(unittest.TestCase):
    def test_infer_gap(self):
        self.assertEqual(ll.infer_gap(0x34, 0x40)["alignments"], [16])
        self.assertEqual(ll.infer_gap(0x34, 0x38)["alignment_min"], 8)
        self.assertEqual(ll.infer_gap(0x40, 0x40)["kind"], "abutting")
        self.assertEqual(ll.infer_gap(0x34, 0x40, data=b"\0\0\0\1" * 3)["kind"], "content")
        self.assertEqual(ll.infer_gap(0x60, 0x90, data=bytes(0x30))["kind"], "zero-fill")
        self.assertEqual(ll.infer_gap(0x40, 0x3C)["kind"], "overlap")

    def test_alignment_gap_inference_and_holes(self):
        base = 0x80000000
        image = bytearray(0x100)
        image[0xA0:0xC0] = b"\x11" * 0x20
        graph = ll.LinkGraph()
        graph.add_segment("main", base, sections={".text": (base, base + 0xD0)})
        for name, start, size in (("A", 0x00, 0x34), ("B", 0x40, 0x20), ("C", 0x90, 0x10), ("D", 0xC0, 0x10)):
            graph.add_unit(name, "main")
            graph.add_piece(name, ".text", base + start, size)
        graph.derive(lambda _segment, vram, size: bytes(image[vram - base:vram - base + size]))
        self.assertEqual(len(graph.alignments), 1)
        self.assertEqual(graph.alignments[0]["unit"], "B")
        self.assertEqual(graph.alignments[0]["alignment_min"], 16)
        self.assertTrue(graph.alignments[0]["zero"])
        adjacent = {c.units for c in graph.constraints if c.kind == "adjacent"}
        self.assertIn(("A", "B"), adjacent)
        self.assertNotIn(("B", "C"), adjacent)
        self.assertEqual([(h["kind"], h["size"]) for h in graph.holes], [("zero-fill", 0x30), ("content", 0x20)])
        raw = graph.units["raw_main_800000A0"]
        self.assertEqual(raw.kind, "raw")
        self.assertIn(("C", "raw_main_800000A0"), adjacent)
        self.assertIn(("raw_main_800000A0", "D"), adjacent)
        main = ll.solve(graph)["segments"]["main"]
        self.assertEqual(main["order"], ["A", "B", "C", "raw_main_800000A0", "D"])
        self.assertEqual(main["orderings"], 1)

    def test_overlapping_exact_pieces_are_reported(self):
        graph = ll.LinkGraph()
        graph.add_segment("main", TEXT)
        graph.add_unit("A", "main")
        graph.add_piece("A", ".text", TEXT, 0x20)
        graph.add_unit("B", "main")
        graph.add_piece("B", ".text", TEXT + 0x10, 0x20)
        self.assertIn("overlap", [c["kind"] for c in ll.solve(graph)["contradictions"]])


class ScriptTests(unittest.TestCase):
    def test_linker_script_text(self):
        graph = two_anchored_and_one_free()
        graph.units["B"].archive, graph.units["B"].member = "libgultra_rom", "sched.o"
        graph.set_symbol("_gp", 0x80001234, "test")
        script = ll.linker_script(graph, ll.solve(graph))
        self.assertIn("grants no match credit", script)
        self.assertIn("OUTPUT_ARCH (mips)", script)
        self.assertIn("_gp = 0x80001234;", script)
        self.assertIn("__romPos = 0x1000;", script)
        self.assertIn(".main 0x80000400 : AT(main_ROM_START)", script)
        self.assertLess(script.index("build/main/A.o(.text);"), script.index("*libgultra_rom.a:sched.o(.text);"))
        self.assertLess(script.index("*libgultra_rom.a:sched.o(.text);"), script.index("build/main/C.o(.text);"))
        self.assertLess(script.index("main_TEXT_END = .;"), script.index("main_DATA_START = .;"))
        self.assertIn("ALTERNATIVE: C is at position 2", script)
        self.assertIn("UNORDERED: no constraint places B directly before C", script)
        self.assertIn(".main.noload (NOLOAD) :", script)
        self.assertIn("build/main/A.o(.bss);", script)
        self.assertIn("build/main/A.o(COMMON);", script)
        self.assertIn("_end = .;", script)
        self.assertIn("/DISCARD/", script)
        self.assertEqual(script.count("{"), script.count("}"))

    def test_manifest_and_graph_files(self):
        graph = two_anchored_and_one_free()
        solution = ll.solve(graph)
        with tempfile.TemporaryDirectory() as directory:
            path = ll.write_manifest(ll.manifest(graph, solution), Path(directory) / "objects.tsv")
            lines = path.read_text().splitlines()
        self.assertEqual(lines[0].split("\t"), list(ll.MANIFEST_COLUMNS))
        self.assertEqual(len(lines), 4)
        document = graph.as_dict()
        self.assertEqual(document["credit"], ll.CREDIT)
        self.assertEqual({u["name"] for u in document["units"]}, {"A", "B", "C"})
        summary = ll.summary(graph, solution)
        self.assertEqual(summary["orderings"], 3)
        self.assertEqual(summary["unit_kinds"], {"c": 3})

    def test_bad_input_is_refused(self):
        graph = two_anchored_and_one_free()
        with self.assertRaises(ValueError):
            graph.add_unit("E", "main", kind="lib")
        with self.assertRaises(ValueError):
            graph.add_piece("A", ".weird", TEXT, 4)
        with self.assertRaises(ValueError):
            graph.add_constraint("adjacent", ("A", "B"), 1.0, "no section")
        with self.assertRaises(ValueError):
            graph.before("A", "A", 1.0, "self")


class ExtractorTests(unittest.TestCase):
    def test_private_data_references_order_units(self):
        base = 0x80000000
        f1 = words(0x3C018000, 0x8C221000, 0x8C231100, 0x03E00008, 0, 0)   # lw v0,0x1000 / v1,0x1100 (at=0x8000xxxx)
        f2 = words(0x3C018000, 0x8C221010, 0x8C231100, 0x03E00008, 0, 0)   # lw v0,0x1010 / v1,0x1100
        image = (f1 + f2).ljust(0x1200, b"\0")
        functions = {"U1": [("f1", base, 0x18)], "U2": [("f2", base + 0x18, 0x18)]}
        references = ll.code_references(image, base, functions, 0x80001000, 0x80001200)
        self.assertEqual(sorted(references["U1"]), [(0x80001000, "load"), (0x80001100, "load")])
        private, counts = ll.private_addresses(references)
        self.assertEqual(private, {"U1": [0x80001000], "U2": [0x80001010]})
        self.assertEqual(counts, {"addresses": 3, "shared": 1})
        graph = ll.LinkGraph()
        graph.add_segment("main", base)
        graph.add_unit("U1", "main")
        graph.add_piece("U1", ".text", base, 0x18)
        graph.add_unit("U2", "main")
        graph.add_piece("U2", ".text", base + 0x18, 0x18)
        self.assertEqual(ll.reference_pieces(graph, {".data": (0x80001000, 0x80001200)}, private), 2)
        graph.derive()
        data = [c for c in graph.constraints if c.section == ".data"]
        self.assertEqual([(c.kind, c.units, c.confidence) for c in data], [("before", ("U1", "U2"), 0.6)])

    def test_infer_data_split(self):
        order = ["U0", "U1", "U2", "U3"]
        addresses = {"U0": [0x100, 0x200], "U1": [0x110], "U2": [0x120, 0x210], "U3": [0x130, 0x220]}
        found = ll.infer_data_split(order, addresses, 0x100, 0x300, stores={0x100, 0x110})
        self.assertEqual(found["split"], 0x200)
        self.assertEqual(found["sections"], {".data": (0x100, 0x200), ".rodata": (0x200, 0x300)})
        self.assertEqual((found["score"], found["unsplit_score"]), (0, 3))
        one = ll.infer_data_split(order, {"U0": [0x100], "U1": [0x110]}, 0x100, 0x300)
        self.assertIsNone(one["split"])

    def test_sdk_membership_names_members_and_groups_an_archive(self):
        tags = ["ultralib:2.0K:libgultra_rom", "ultralib:2.0L:libgultra_rom"]
        entries = {"osCreateThread": [{"obj": "createthread.c", "sources": tags}],
                   "osStartThread": [{"obj": "startthread.c", "sources": tags[:1]}],
                   "osSetThreadPri": [{"obj": "setthreadpri.c", "sources": tags}],
                   "game_fn": [{"obj": "800FE710_x.c", "sources": ["some_project"]}]}
        names = {TEXT: "osCreateThread", TEXT + 0x10: "osStartThread", TEXT + 0x18: "osSetThreadPri",
                 TEXT + 0x20: "game_fn"}
        members = ll.sdk_membership(names, entries)
        self.assertEqual(sorted(members), [TEXT, TEXT + 0x10, TEXT + 0x18])
        self.assertEqual(members[TEXT]["member"], "createthread.o")
        self.assertEqual(members[TEXT]["archive"], "libgultra_rom")
        graph = ll.LinkGraph()
        graph.add_segment("main", TEXT)
        functions = {"U1": [("a", TEXT, 0x10)], "U2": [("b", TEXT + 0x10, 0x8), ("c", TEXT + 0x18, 0x8)],
                     "U3": [("d", TEXT + 0x20, 0x10)]}
        for unit, rows in functions.items():
            graph.add_unit(unit, "main")
            graph.add_piece(unit, ".text", rows[0][1], sum(size for _n, _v, size in rows))
        stats = ll.apply_membership(graph, functions, members)
        self.assertEqual(stats, {"units_with_member": 1, "units_spanning_members": 1, "groups": 1})
        self.assertEqual(graph.units["U1"].object_path(), "*libgultra_rom.a:createthread.o")
        self.assertEqual([h["kind"] for h in graph.hints], ["split"])
        groups = [c for c in graph.constraints if c.kind == "group"]
        self.assertEqual([c.units for c in groups], [("U1", "U2")])
        self.assertEqual(ll.solve(graph)["segments"]["main"]["order"], ["U1", "U2", "U3"])

    def test_source_names_and_catalogue_tags(self):
        graph = ll.LinkGraph()
        graph.add_segment("main", TEXT)
        graph.add_unit("U1", "main")
        functions = {"U1": [("a", TEXT, 0x10), ("b", TEXT + 0x10, 0x10)]}
        named = ll.apply_source_names(graph, functions, [
            {"vram": f"0x{TEXT:08X}", "source_names": ["game.c"]},
            {"vram": f"0x{TEXT + 0x10:08X}", "source_names": ["other.c"]}])
        self.assertEqual(named, 1)
        self.assertEqual([h["kind"] for h in graph.hints], ["split"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "functions.toml"
            path.write_text('functions = [\n    { name = "f", vram = 0x80000400, size = 0x60 }, # asm\n'
                            '    { name = "g", vram = 0x80000460, size = 0x8 }, # stub\n]\n')
            self.assertEqual(ll.catalogue_tags(path), {0x80000400: "asm", 0x80000460: "stub"})


if __name__ == "__main__":
    unittest.main()
