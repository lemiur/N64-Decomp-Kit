import json
import struct
import tempfile
import unittest
from pathlib import Path

from n64dk.analysis import boundaries as b
from n64dk.analysis import flow

BASE = 0x80001000
ZERO, V0, V1, A0, S0, SP, RA = 0, 2, 3, 4, 16, 29, 31


def addiu(rt, rs, imm):
    return (9 << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def addu(rd, rs, rt):
    return (rs << 21) | (rt << 16) | (rd << 11) | 0x21


def sw(rt, off, base=SP):
    return (43 << 26) | (base << 21) | (rt << 16) | (off & 0xFFFF)


def lw(rt, off, base=SP):
    return (35 << 26) | (base << 21) | (rt << 16) | (off & 0xFFFF)


def jal(target):
    return (3 << 26) | ((target >> 2) & 0x03FFFFFF)


def beq(rs, rt, here, target):
    return (4 << 26) | (rs << 21) | (rt << 16) | (((target - here - 4) >> 2) & 0xFFFF)


JR_RA, NOP = 0x03E00008, 0


def image(layout: dict[int, list[int]], size: int = 0x600) -> b.Image:
    data = bytearray(size)
    for address, words in layout.items():
        for k, word in enumerate(words):
            struct.pack_into(">I", data, address - BASE + 4 * k, word)
    return b.Image(bytes(data), BASE)


# A (0x1000, called by F) calls C, branches into B2; B1 and B2 are interior fragments of A.
A = [addiu(SP, SP, -0x18), sw(RA, 0x14), jal(BASE + 0x40), NOP,
     beq(V0, ZERO, BASE + 0x10, BASE + 0x28), NOP, addiu(V0, ZERO, 1), addiu(V0, V0, 2)]
B1 = [addu(V1, V0, V0), addu(V0, V1, ZERO)]                                    # reads caller's $v0
B2 = [lw(RA, 0x14), JR_RA, addiu(SP, SP, 0x18), NOP, NOP, NOP]                 # restore without save
C = [addu(V0, S0, A0), JR_RA, NOP]                                             # called, reads $s0: -O3 static
E = [JR_RA, addiu(V0, ZERO, 1), JR_RA, addiu(V0, ZERO, 2)]                     # 0x1208 is called: split
F = [addiu(SP, SP, -0x18), sw(RA, 0x14), jal(BASE + 0x208), NOP, jal(BASE + 0x400), NOP, jal(BASE),
     NOP, lw(RA, 0x14), JR_RA, addiu(SP, SP, 0x18)]
G = [JR_RA, addiu(V0, ZERO, 3)]                                                # called, no row: add
H = [JR_RA, NOP]
P = [JR_RA, NOP, NOP, NOP]                                                     # trailing padding: shrink

LAYOUT = {BASE: A, BASE + 0x20: B1, BASE + 0x28: B2, BASE + 0x40: C, BASE + 0x200: E, BASE + 0x300: F,
          BASE + 0x400: G, BASE + 0x500: H, BASE + 0x580: P}
CATALOGUE = [("func_A", BASE, 0x20), ("func_B1", BASE + 0x20, 0x8), ("func_B2", BASE + 0x28, 0x18),
             ("func_C", BASE + 0x40, 0xC), ("func_E", BASE + 0x200, 0x10), ("func_F", BASE + 0x300, 0x2C),
             ("func_H", BASE + 0x500, 0x8), ("func_P", BASE + 0x580, 0x10)]


def kinds(ops):
    return sorted(op["kind"] + (":" + op["to"] if op.get("to") else "") for op in ops)


class SolverTest(unittest.TestCase):
    def setUp(self):
        self.image = image(LAYOUT)
        self.ops = b.solve(self.image, CATALOGUE)

    def op(self, kind, to=None):
        hits = [o for o in self.ops if o["kind"] == kind and o.get("to") == to]
        self.assertTrue(hits, (kind, to, kinds(self.ops)))
        return hits[0]

    def test_merge_chain_closed(self):
        merge = self.op("merge")
        self.assertEqual(merge["absorbed"], ["func_B1", "func_B2"])
        self.assertEqual(merge["insert"], [{"name": "func_A", "vram": "0x80001000", "size": "0x00000040"}])
        self.assertTrue(merge["closed"])
        self.assertGreaterEqual(merge["confidence"], 0.8)
        self.assertEqual(merge["affects"]["functions"], -2)
        self.assertTrue(flow.entry_evidence(self.image.flow(BASE, 0x40))["complete"])

    def test_called_nonstandard_abi_not_merged(self):
        merge = self.op("merge")
        self.assertNotIn("func_C", merge["absorbed"])
        retype = self.op("retype", "nonstandard-abi")
        self.assertEqual(retype["target"]["name"], "func_C")
        self.assertEqual(retype["flags"], ["needs-tu"])
        self.assertEqual(retype["insert"], [])

    def test_split_at_called_interior(self):
        split = self.op("split")
        self.assertEqual(split["at"], ["0x80001208"])
        self.assertEqual([r["vram"] for r in split["insert"]], ["0x80001200", "0x80001208"])
        self.assertEqual(split["insert"][1]["name"], "func_80001208")

    def test_add_missing_entry(self):
        add = self.op("add")
        self.assertEqual(add["insert"], [{"name": "func_80001400", "vram": "0x80001400", "size": "0x00000008"}])
        self.assertEqual(add["released"][0]["class"], "padding")

    def test_shrink_padding_unless_convention_keeps_it(self):
        shrink = self.op("shrink")
        self.assertEqual(shrink["insert"][0]["size"], "0x00000008")
        self.assertEqual(shrink["released"], [{"vram": "0x80001588", "size": "0x00000008", "class": "padding"}])
        kept = b.solve(self.image, CATALOGUE, keep_padding=True)
        self.assertNotIn("shrink", kinds(kept))

    def test_blocked_by_verified_interval(self):
        ops = b.solve(self.image, CATALOGUE, verified=b.Verified([(BASE, BASE + 0x20, "src/a.c")]))
        merge = [o for o in ops if o["kind"] == "merge"][0]
        self.assertEqual(merge["status"], "blocked")
        self.assertEqual(merge["affects"]["manifest_rows"], ["src/a.c 0x80001000+0x00000020"])
        new, applied, skipped = b.apply(CATALOGUE, ops, 0.8)
        self.assertIn(("func_B1", BASE + 0x20, 8), new)
        self.assertTrue(any(s["skip_reason"].startswith("blocked") for s in skipped))
        # padding inside a verified unit is the unit's (no shrink)
        ops = b.solve(self.image, CATALOGUE, verified=b.Verified([(BASE + 0x580, BASE + 0x590, "src/p.c")]))
        self.assertNotIn("shrink", kinds(ops))

    def test_retype_padding_data_jump_table(self):
        layout = dict(LAYOUT)
        layout[BASE + 0x100] = [0, 0, 0, 0]
        layout[BASE + 0x110] = [BASE + 0x200, BASE + 0x208, BASE + 0x300]
        layout[BASE + 0x120] = [0xFC000000, 0xFFFFFFFF, 0x7C000000, 0xFC000001]
        img = image(layout)
        cat = CATALOGUE + [("pad", BASE + 0x100, 0x10), ("tbl", BASE + 0x110, 0xC), ("dat", BASE + 0x120, 0x10)]
        ops = b.solve(img, cat)
        for to in ("padding", "jump-table", "data"):
            self.assertIn("retype:" + to, kinds(ops))

    def test_inverse_round_trip(self):
        new, applied, _skipped = b.apply(CATALOGUE, self.ops, 0.8)
        self.assertNotEqual(new, CATALOGUE)
        self.assertEqual({o["kind"] for o in applied}, {"merge", "split", "shrink", "add", "retype"})
        back, inv_applied, inv_skipped = b.apply(new, b.invert(applied), 0.8)
        self.assertEqual(back, sorted(CATALOGUE, key=lambda r: (r[1], r[0])))
        self.assertEqual(inv_skipped, [])
        # operations are self-contained JSON
        self.assertEqual(json.loads(json.dumps(self.ops)), self.ops)

    def test_apply_respects_threshold_and_staleness(self):
        _new, applied, skipped = b.apply(CATALOGUE, self.ops, 1.01)
        self.assertEqual(applied, [])
        stale = b.apply([r for r in CATALOGUE if r[0] != "func_B1"], self.ops, 0.8)[2]
        self.assertTrue(any(s["skip_reason"].startswith("stale") for s in stale))

    def test_also_image_calls_count(self):
        cat = [("func_A", BASE, 0x20), ("func_X", BASE + 0x20, 0x8)]
        caller = b.Image(struct.pack(">I", jal(BASE + 0x20)) + b"\0" * 12, 0x80400000)
        lone = b.solve(self.image, cat)
        self.assertIn("merge", kinds(lone))
        called = b.solve(self.image, cat, also=[caller])
        self.assertNotIn("merge", kinds(called))


FP = 30
MOVE_FP_SP, MOVE_SP_FP = addu(FP, SP, ZERO), addu(SP, FP, ZERO)


def j(target):
    return (2 << 26) | ((target >> 2) & 0x03FFFFFF)


class CertainCorrectionTest(unittest.TestCase):
    """The two corrections decompile-run applies to its own catalogue (measured on an unoptimised title)."""
    # a function that ends in an endless loop; the compiler still emitted its epilogue after the loop
    LOOPS = [addiu(SP, SP, -0x18), sw(RA, 0x14), sw(FP, 0x10), MOVE_FP_SP, jal(BASE + 0x100), NOP,
             j(BASE + 0x18), NOP]
    EPILOGUE = [MOVE_SP_FP, lw(RA, 0x14), lw(FP, 0x10), addiu(SP, SP, 0x18), JR_RA, NOP]
    CALLEE = [JR_RA, NOP]
    CALLER = [addiu(SP, SP, -0x18), sw(RA, 0x14), jal(BASE), NOP, lw(RA, 0x14), JR_RA, addiu(SP, SP, 0x18)]

    def solve(self, epilogue, loops=None):
        loops = loops or self.LOOPS
        layout = {BASE: loops, BASE + 0x20: epilogue, BASE + 0x100: self.CALLEE, BASE + 0x200: self.CALLER}
        rows = [("func_L", BASE, 0x20), ("func_T", BASE + 0x20, 4 * len(epilogue)), ("func_C", BASE + 0x100, 8),
                ("func_R", BASE + 0x200, 0x1C)]
        return rows, b.solve(image(layout), rows)

    def test_the_unreachable_epilogue_of_an_endless_function_is_merged_into_it(self):
        rows, ops = self.solve(self.EPILOGUE)
        merge = [op for op in ops if op["kind"] == "merge"][0]
        self.assertIn("never returns", merge["certain"])
        self.assertEqual(merge["confidence"], 0.99)
        new, applied = b.apply_certain(rows, ops)
        self.assertEqual(len(applied), 1)
        self.assertIn(("func_L", BASE, 0x38), new)
        self.assertNotIn(("func_T", BASE + 0x20, 0x18), new)

    def test_another_frame_or_a_register_the_function_never_saved_is_not_certain(self):
        for epilogue in ([MOVE_SP_FP, lw(RA, 0x14), lw(FP, 0x10), addiu(SP, SP, 0x20), JR_RA, NOP],     # other frame
                         [MOVE_SP_FP, lw(RA, 0x14), lw(S0, 0x0C), addiu(SP, SP, 0x18), JR_RA, NOP],     # $s0 not saved
                         [MOVE_SP_FP, lw(RA, 0x14), addu(V0, ZERO, ZERO), addiu(SP, SP, 0x18), JR_RA, NOP]):  # work
            rows, ops = self.solve(epilogue)
            self.assertEqual(b.apply_certain(rows, ops)[1], [], epilogue)

    def test_a_function_that_returns_does_not_take_a_following_epilogue(self):
        returns = [addiu(SP, SP, -0x18), sw(RA, 0x14), sw(FP, 0x10), MOVE_FP_SP, lw(RA, 0x14), lw(FP, 0x10),
                   JR_RA, addiu(SP, SP, 0x18)]
        self.assertIsNone(b.unreachable_epilogue(returns, self.EPILOGUE))
        self.assertIsNotNone(b.unreachable_epilogue(self.LOOPS, self.EPILOGUE))

    def test_a_function_cut_at_an_inner_return_is_joined_when_only_the_whole_is_closed(self):
        ops = b.solve(image(LAYOUT), CATALOGUE)
        merge = [op for op in ops if op["kind"] == "merge"][0]
        # func_A branches into func_B2 and falls into func_B1; neither B part can be entered; the whole closes
        self.assertIn("not a closed body", merge.get("certain", ""))
        new, applied = b.apply_certain(CATALOGUE, ops)
        self.assertIn(("func_A", BASE, 0x40), new)
        self.assertNotIn(("func_B1", BASE + 0x20, 0x8), new)

    def test_a_call_that_never_returns_in_front_of_a_function_is_split_off(self):
        stub = [jal(BASE + 0x100), addu(A0, ZERO, ZERO), NOP, NOP]
        body = [addu(V1, ZERO, ZERO), addiu(SP, SP, -0x18), sw(RA, 0x14), jal(BASE + 0x100), NOP, lw(RA, 0x14), JR_RA,
                addiu(SP, SP, 0x18)]
        layout = {BASE: stub + body, BASE + 0x100: self.CALLEE, BASE + 0x200: self.CALLER}
        rows = [("func_S", BASE, 0x30), ("func_C", BASE + 0x100, 8), ("func_R", BASE + 0x200, 0x1C)]
        new, applied = b.apply_certain(rows, b.solve(image(layout), rows))
        self.assertEqual([op["kind"] for op in applied], ["split"])
        self.assertIn(("func_S", BASE, 8), new)
        self.assertIn(("func_80001010", BASE + 0x10, 0x20), new)
        # a call inside an ordinary function (after $ra is saved) is nothing of the kind
        self.assertIsNone(b.call_stub(image({BASE: self.CALLER}), BASE, 0x1C))

    def test_a_stray_word_in_front_of_a_function_is_released_and_the_function_starts_after_it(self):
        closed = [addiu(SP, SP, -0x18), sw(RA, 0x14), jal(BASE + 0x100), NOP, lw(RA, 0x14), JR_RA,
                  addiu(SP, SP, 0x18)]
        stray = [addu(V1, S0, A0)]                    # decodes, reads registers nothing set: not this function's
        body = [addiu(SP, SP, -0x18), sw(RA, 0x14), jal(BASE + 0x100), NOP, lw(RA, 0x14), JR_RA, addiu(SP, SP, 0x18)]
        layout = {BASE: closed, BASE + 0x1C: stray + body, BASE + 0x100: self.CALLEE, BASE + 0x200: self.CALLER}
        rows = [("func_L", BASE, 0x1C), ("func_T", BASE + 0x1C, 0x20), ("func_C", BASE + 0x100, 8),
                ("func_R", BASE + 0x200, 0x1C)]
        ops = b.solve(image(layout), rows)
        new, applied = b.apply_certain(rows, ops)
        self.assertEqual([op["kind"] for op in applied], ["shrink"])
        self.assertIn(("func_80001020", BASE + 0x20, 0x1C), new)
        self.assertIn(("func_L", BASE, 0x1C), new)                    # the function before it is left alone
        self.assertEqual(applied[0]["released"], [{"vram": "0x8000101C", "size": "0x00000004", "class": "data"}])


class CatalogueFileTest(unittest.TestCase):
    TOML = ('# header\n[[section]]\nname = ".main"\nfunctions = [\n'
            '    { name = "func_A", vram = 0x80001000, size = 0x20 }, # c\n'
            '    { name = "func_B1", vram = 0x80001020, size = 0x8 }, # leaf\n'
            '    { name = "func_B2", vram = 0x80001028, size = 0x18 },\n'
            ']\n')
    SYMS = ("func_A = 0x80001000; // type:func size:0x20\n"
            "D_80002000 = 0x80002000; // type:data\n"
            "func_B1 = 0x80001020; // type:func size:0x8\n"
            "func_B2 = 0x80001028; // type:func size:0x18\n")
    TSV = ("# ROM_START\tSIZE\tVRAM\tNAME\tEVIDENCE\n"
           "0x1000\t0x20\t0x80001000\tfunc_A\tretail\n0x1020\t0x8\t0x80001020\tfunc_B1\tretail\n"
           "0x1028\t0x18\t0x80001028\tfunc_B2\tretail\n")

    def round_trip(self, text, name):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / name
            src.write_text(text)
            rows = b.read_catalogue(src)
            self.assertEqual([r[0] for r in rows], ["func_A", "func_B1", "func_B2"])
            same = b.write_catalogue(Path(tmp) / "out" / name, rows, like=src)
            self.assertEqual(same.read_text(), text)
            with self.assertRaises(ValueError):
                b.write_catalogue(src, rows, like=src)
            ops = [o for o in b.solve(image(LAYOUT), rows) if o["kind"] == "merge"]
            new, applied, _ = b.apply(rows, ops, 0.8)
            merged = b.write_catalogue(Path(tmp) / "m" / name, new, like=src)
            self.assertEqual(b.read_catalogue(merged), [("func_A", BASE, 0x40)])
            back = b.apply(b.read_catalogue(merged), b.invert(applied), 0.8)[0]
            restored = b.write_catalogue(Path(tmp) / "r" / name, back, like=src)
            self.assertEqual(b.read_catalogue(restored), rows)
            return merged.read_text()

    def test_toml(self):
        merged = self.round_trip(self.TOML, "functions.toml")
        self.assertIn('{ name = "func_A", vram = 0x80001000, size = 0x40 },', merged)

    def test_symbol_addrs(self):
        merged = self.round_trip(self.SYMS, "symbol_addrs.txt")
        self.assertIn("D_80002000", merged)
        self.assertIn("func_A = 0x80001000; // type:func size:0x40", merged)

    def test_tsv(self):
        merged = self.round_trip(self.TSV, "functions.tsv")
        self.assertIn("0x1000\t0x40\t0x80001000\tfunc_A", merged)

    def test_shift(self):
        rows = [("f", 0x80005000, 8)]
        ops = [{"kind": "add", "confidence": 1, "remove": [], "insert": [{"name": "g", "vram": "0x80001000",
                                                                         "size": "0x8"}], "released": []}]
        shifted = b.shift_operations(ops, 0x18A0)
        self.assertEqual(shifted[0]["insert"][0]["vram"], "0x800028A0")
        self.assertEqual(b.shift_rows(b.shift_rows(rows, 0x18A0), -0x18A0), rows)


class CommandTest(unittest.TestCase):
    def test_boundary_solve_and_ownership_commands(self):
        import contextlib
        import io
        from n64dk import cli
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cat").mkdir()
            (root / "img.bin").write_bytes(image(LAYOUT).data)
            cat = root / "cat" / "functions.toml"
            cat.write_text("functions = [\n" + "".join(
                f'    {{ name = "{n}", vram = 0x{v:08X}, size = 0x{s:X} }},\n' for n, v, s in CATALOGUE) + "]\n")
            (root / "m.tsv").write_text("0x40\t0xC\tsrc/c.c\t-O2\n")
            (root / "verdicts.tsv").write_text("src/c.c\texact\n")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                base = ["boundary-solve", str(root / "img.bin"), "--vram-delta", hex(BASE), "--catalogue", str(cat),
                        "--manifest", str(root / "m.tsv"), "--verdicts", str(root / "verdicts.tsv")]
                self.assertEqual(cli.main(base + ["--out", str(root / "cat" / "x")]), 2)   # inside catalogue dir
                self.assertEqual(cli.main(base + ["--out", str(root / "out")]), 0)
                ops, header = b.load_operations(root / "out" / "operations.json")
                self.assertTrue(any(o["status"] == "blocked" and o.get("to") == "nonstandard-abi" for o in ops))
                inverse, _ = b.load_operations(root / "out" / "inverse.json")
                corrected = b.read_catalogue(root / "out" / "functions.toml")
                self.assertEqual(b.apply(corrected, inverse, 0.8)[0], sorted(CATALOGUE, key=lambda r: r[1]))
                build = ["ledger-build", str(root / "img.bin"), "--vram-delta", hex(BASE), "--catalogue", str(cat),
                         "--c-manifest", str(root / "m.tsv")]
                self.assertEqual(cli.main(build + ["--out", str(root / "before.tsv")]), 0)
                self.assertEqual(cli.main(build + ["--operations", str(root / "out" / "operations.json"),
                                                   "--out", str(root / "after.tsv")]), 0)
                # before: the called but uncatalogued body at 0x80001400 is a gap; after the add it is owned
                self.assertEqual(cli.main(["ledger-check", str(root / "before.tsv"), "--explain"]), 1)
                self.assertEqual(cli.main(["ledger-check", str(root / "after.tsv")]), 0)
                self.assertEqual(cli.main(["ledger-diff", str(root / "before.tsv"), str(root / "after.tsv")]), 0)
            self.assertIn("reclassified as fragment of func_A", buf.getvalue())


if __name__ == "__main__":
    unittest.main()


class EmptyCatalogueTest(unittest.TestCase):
    """A survey can write a function table for a segment with no functions (Daikatana, 2026-11-18): it is an
    empty catalogue, not an unreadable one that stops the whole title's run."""

    def test_a_toml_table_that_lists_no_function_reads_as_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "functions_seg.toml"
            path.write_text('# Function boundaries for seg: main segment\n[[section]]\nname = ".main"\n'
                            'rom = 0x00000000\nvram = 0x80230F90\nsize = 0x0\n\nfunctions = [\n]\n')
            self.assertEqual(b.catalogue_format(path), "toml")
            self.assertEqual(b.read_catalogue(path), [])

    def test_text_that_is_no_catalogue_is_still_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.toml"
            path.write_text("title = 'x'\n")
            with self.assertRaises(ValueError):
                b.catalogue_format(path)
