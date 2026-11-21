"""Data/.rodata/BSS solver (n64dk.analysis.datasolve) on hand-built big-endian MIPS images.

Every image is 0x80 bytes at 0x80001000: function f (0x00-0x20), function g (0x20-0x40), then
0x40 bytes that no function covers."""
import struct
import unittest
from types import SimpleNamespace

from n64dk.analysis import datasolve
from n64dk.analysis.datasolve import Table

BASE = 0x80001000
F, G = BASE, BASE + 0x20
ROWS = [("f", F, 0x20), ("g", G, 0x20)]
AT, V0, V1, A0, T6, GP = 1, 2, 3, 4, 14, 28
JR_RA, NOP = 0x03E00008, 0


def I(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def lui(rt, imm): return I(15, 0, rt, imm)
def addiu(rt, rs, imm): return I(9, rs, rt, imm)
def ori(rt, rs, imm): return I(13, rs, rt, imm)
def lw(rt, imm, rs): return I(35, rs, rt, imm)
def sw(rt, imm, rs): return I(43, rs, rt, imm)
def lwc1(ft, imm, rs): return I(49, rs, ft, imm)
def ldc1(ft, imm, rs): return I(53, rs, ft, imm)


def image(f=(), g=(), data=b""):
    def words(ws):
        ws = list(ws) + [NOP] * (8 - len(ws))
        assert len(ws) == 8
        return b"".join(struct.pack(">I", w) for w in ws)
    data = bytes(data) + bytes(0x40 - len(data))
    assert len(data) == 0x40
    return words(f) + words(g) + data


def be(*ws):
    return b"".join(struct.pack(">I", w) for w in ws)


def by_start(solution):
    return {r.start: r for r in solution.rows}


class OwnershipTest(unittest.TestCase):
    def test_owner_is_the_single_referencing_function(self):
        img = image(f=[lui(AT, 0x8000), lw(V0, 0x1040, AT), lui(AT, 0x8000), sw(V0, 0x1048, AT), JR_RA],
                    g=[lui(AT, 0x8000), lw(V1, 0x1048, AT), JR_RA],
                    data=be(1, 2, 3))
        sol = datasolve.solve(img, BASE, ROWS)
        rows = by_start(sol)
        self.assertEqual(sorted(rows), [BASE + 0x40, BASE + 0x48])
        first, second = rows[BASE + 0x40], rows[BASE + 0x48]
        self.assertEqual((first.owner_function, first.kind, first.min_size, first.size), (F, "object", 4, 8))
        self.assertFalse(first.written)
        self.assertIsNone(second.owner_function)
        self.assertEqual(second.functions, (F, G))
        self.assertTrue(second.written)
        self.assertEqual(second.section_hint, ".data")
        self.assertEqual(second.size, 0x38)
        s = sol.summary
        self.assertEqual((s["non_code_bytes"], s["owned"], s["conflicting"]), (0x40, 8, 0))
        self.assertEqual(s["owned"] + s["conflicting"] + s["unowned"], s["non_code_bytes"])
        self.assertEqual((s["unowned_unclaimed"], s["unowned_adjacent"]), (0, 0x38))
        self.assertEqual((s["single_owner_rows"], s["shared_rows"]), (1, 1))
        self.assertIs(s["exact"], False)
        self.assertIsNone(sol.bss)

    def test_reference_into_code_is_reported_not_owned(self):
        img = image(f=[lui(AT, 0x8000), lw(V0, 0x1024, AT), lui(A0, 0x8000), addiu(A0, A0, 0x1020), JR_RA])
        sol = datasolve.solve(img, BASE, ROWS)
        self.assertEqual(sol.rows, [])
        self.assertEqual(len(sol.code_overlap), 1)          # the load; the address of g is a function pointer
        self.assertEqual(sol.code_overlap[0]["address"], "0x80001024")


class KindTest(unittest.TestCase):
    def test_strings_floats_and_an_o32_double(self):
        img = image(f=[lui(AT, 0x8000), lwc1(1, 0x1060, AT), lwc1(0, 0x1064, AT), lwc1(2, 0x1068, AT),
                       ldc1(4, 0x1070, AT), JR_RA],
                    g=[lui(A0, 0x8000), addiu(A0, A0, 0x1050), JR_RA],
                    data=bytes(0x10) + b"Hello\0\0\0" + bytes(8)
                    + be(0x3FF00000, 0, 0x40000000, 0, 0x40000000, 0))
        rows = by_start(datasolve.solve(img, BASE, ROWS))
        self.assertEqual(sorted(rows), [BASE + 0x50, BASE + 0x60, BASE + 0x68, BASE + 0x70])
        text = rows[BASE + 0x50]
        self.assertEqual((text.kind, text.min_size, text.owner_function, text.section_hint), ("string", 6, G, ".rodata?"))
        double = rows[BASE + 0x60]
        self.assertEqual((double.kind, double.min_size, double.size), ("f64", 8, 8))
        self.assertIn("hilo-pair:f64*2", double.evidence)
        self.assertEqual((rows[BASE + 0x68].kind, rows[BASE + 0x68].min_size), ("f32", 4))
        self.assertEqual((rows[BASE + 0x70].kind, rows[BASE + 0x70].min_size), ("f64", 8))
        sol = datasolve.solve(img, BASE, ROWS)
        self.assertEqual(sol.summary["owned"], 6 + 8 + 4 + 8)
        self.assertEqual(sol.summary["unowned_unclaimed"], 0x10)

    def test_without_the_string_source_a_string_is_only_address_taken(self):
        img = image(g=[lui(A0, 0x8000), addiu(A0, A0, 0x1050), JR_RA], data=bytes(0x10) + b"Hello\0\0\0")
        row = by_start(datasolve.solve(img, BASE, ROWS, sources=("hilo",)))[BASE + 0x50]
        self.assertEqual((row.kind, row.min_size), ("address-taken", 1))

    def test_unpaired_halves_stay_two_f32(self):
        # same registers' parity reversed: not the o32 double form
        img = image(f=[lui(AT, 0x8000), lwc1(0, 0x1060, AT), lwc1(1, 0x1064, AT), JR_RA])
        rows = by_start(datasolve.solve(img, BASE, ROWS))
        self.assertEqual([rows[a].kind for a in sorted(rows)], ["f32", "f32"])

    def test_jump_table_shape_without_indirect_analysis(self):
        img = image(f=[lui(V1, 0x8000), addiu(V1, V1, 0x1060), JR_RA],
                    data=bytes(0x20) + be(F + 0x8, F + 0xC, F + 0x10, 0))
        row = by_start(datasolve.solve(img, BASE, ROWS))[BASE + 0x60]
        self.assertEqual((row.kind, row.min_size, row.owner_function), ("jtbl-shape", 12, F))

    def test_pointer_array_and_its_targets(self):
        img = image(g=[lui(A0, 0x8000), addiu(A0, A0, 0x1040), JR_RA],
                    data=be(BASE + 0x50, BASE + 0x58, 0, 0) + b"abcd\0\0\0\0" + b"wxyz\0\0\0\0")
        sol = datasolve.solve(img, BASE, ROWS)
        rows = by_start(sol)
        self.assertEqual((rows[BASE + 0x40].kind, rows[BASE + 0x40].min_size), ("pointers", 8))
        for target in (BASE + 0x50, BASE + 0x58):
            self.assertEqual((rows[target].kind, rows[target].data_refs, rows[target].functions), ("string", 1, ()))
        self.assertEqual(sol.summary["data_only_rows"], 2)
        ablated = datasolve.partition(datasolve.collect(img, BASE, ROWS), sources=("hilo", "string"))
        self.assertEqual(sorted(by_start(ablated)), [BASE + 0x40])


class TableTest(unittest.TestCase):
    def test_a_reference_inside_a_definite_table_is_a_conflict(self):
        img = image(g=[lui(AT, 0x8000), lw(V0, 0x1068, AT), JR_RA],
                    data=bytes(0x20) + be(F + 0x8, F + 0xC, F + 0x10, F + 0x14))
        sol = datasolve.solve(img, BASE, ROWS, tables=[Table(BASE + 0x60, 0x10, F, True, "switch-definite")])
        rows = by_start(sol)
        table = rows[BASE + 0x60]
        self.assertEqual((table.kind, table.min_size, table.size, table.owner_function), ("jtbl", 0x10, 8, F))
        self.assertIn("exact-count", table.evidence)
        self.assertEqual(sol.conflicts[0]["type"], "table-overlap")
        self.assertEqual((sol.conflicts[0]["start"], sol.conflicts[0]["end"]), ("0x80001068", "0x80001070"))
        s = sol.summary
        self.assertEqual((s["owned"], s["conflicting"], s["unowned_unclaimed"], s["unowned_adjacent"]), (8, 8, 0x20, 0x10))
        self.assertEqual(s["unowned_zero_bytes"], 0x30)
        self.assertEqual(s["owned"] + s["conflicting"] + s["unowned"], s["non_code_bytes"])

    def test_tables_from_indirect(self):
        sites = [SimpleNamespace(table=0x80002000, count=4, kind="switch", status="definite", function=F),
                 SimpleNamespace(table=0x80002100, count=3, kind="switch-unbounded", status="finite", function=G),
                 SimpleNamespace(table=0x80002200, count=5, kind="pointer-table", status="finite", function=G),
                 SimpleNamespace(table=None, count=0, kind="unknown", status="unresolved", function=G)]
        tables = datasolve.tables_from_indirect(SimpleNamespace(sites=sites))
        self.assertEqual([(t.address, t.size, t.exact) for t in tables],
                         [(0x80002000, 16, True), (0x80002100, 12, False)])


class SmallDataAndBssTest(unittest.TestCase):
    def test_gp_relative_references_with_gp_set_in_the_image(self):
        img = image(f=[lui(GP, 0x8000), ori(GP, GP, 0x1050), lw(V0, -8, GP), sw(V0, 4, GP), JR_RA])
        sol = datasolve.solve(img, BASE, ROWS)
        self.assertEqual(sol.gp["value"], "0x80001050")
        rows = by_start(sol)
        self.assertEqual(sorted(rows), [BASE + 0x48, BASE + 0x54])     # _gp itself is not an object
        self.assertEqual(rows[BASE + 0x48].evidence, ("gprel:word",))
        self.assertTrue(rows[BASE + 0x54].written)
        given = datasolve.solve(img, BASE, ROWS, gp=BASE + 0x60)
        self.assertEqual((given.gp["basis"], sorted(by_start(given))), ("given", [BASE + 0x58, BASE + 0x64]))

    def test_without_gp_no_small_data_is_guessed(self):
        img = image(f=[lw(V0, -8, GP), JR_RA])
        sol = datasolve.solve(img, BASE, ROWS)
        self.assertIsNone(sol.gp["value"])
        self.assertIn("used", sol.gp["basis"])
        self.assertEqual(sol.rows, [])

    def test_bss_inferred_from_references_past_the_image(self):
        img = image(f=[lui(AT, 0x8000), lw(V0, 0x1100, AT), sw(V0, 0x1108, AT), lui(AT, 0x8010), sw(V0, 0, AT), JR_RA])
        sol = datasolve.solve(img, BASE, ROWS)
        self.assertEqual(sol.bss, (BASE + 0x80, BASE + 0x110))
        self.assertIn("inferred", sol.bss_basis)
        rows = by_start(sol)
        self.assertEqual([(rows[a].region, rows[a].size, rows[a].min_size) for a in sorted(rows)],
                         [("bss", 8, 4), ("bss", 8, 4)])
        self.assertEqual(rows[BASE + 0x108].section_hint, ".bss")
        self.assertEqual(sol.external, {"hilo": 1})
        self.assertEqual(sol.summary["bss_evidenced"], 8)
        given = datasolve.solve(img, BASE, ROWS, bss=(BASE + 0x80, 0x80200000))
        self.assertEqual((given.bss_basis, len(given.rows)), ("given", 3))
        off = datasolve.solve(img, BASE, ROWS, sources=("hilo",))
        self.assertEqual((off.rows, off.external), ([], {"hilo": 3}))

    def test_extent_excludes_bytes_past_the_initialised_image(self):
        img = image(f=[lui(AT, 0x8000), lw(V0, 0x1070, AT), JR_RA])
        sol = datasolve.solve(img, BASE, ROWS, extent=(BASE, BASE + 0x60), bss=(BASE + 0x60, BASE + 0x80))
        self.assertEqual(sol.summary["non_code_bytes"], 0x20)
        self.assertEqual([r.region for r in sol.rows], ["bss"])


class ScoringTest(unittest.TestCase):
    def test_compare_counts_starts_and_size_bounds(self):
        img = image(f=[lui(AT, 0x8000), lw(V0, 0x1040, AT), lui(AT, 0x8000), sw(V0, 0x1048, AT), JR_RA])
        sol = datasolve.solve(img, BASE, ROWS)
        score = datasolve.compare(sol.rows, [(BASE + 0x40, 8, "data"), (BASE + 0x44, 4, "data"),
                                             (BASE + 0x48, 2, "rodata"), (BASE + 0x200, 4, "data")],
                                  lo=BASE, hi=BASE + 0x80)
        self.assertEqual((score["predicted"], score["truth"], score["true_starts"]), (2, 3, 2))
        self.assertEqual((score["precision"], score["recall"]), (1.0, 0.6667))
        self.assertEqual((score["size_equal"], score["size_upper_bound_holds"]), (1, 2))
        self.assertEqual(score["min_size_overstated"], 1)                # a word access on a 2-byte object
        self.assertEqual(score["by_truth_label"]["data"], {"truth": 2, "found": 1})

    def test_report_says_hypothesis(self):
        img = image(f=[lui(AT, 0x8000), lw(V0, 0x1040, AT), JR_RA])
        sol = datasolve.solve(img, BASE, ROWS)
        tsv = sol.to_tsv()
        self.assertTrue(tsv.startswith("# hypothesis"))
        self.assertIn("0x80001040\t0x40\t0x4\tobject\tdata\t-\t0x80001000", tsv)
        self.assertEqual(sol.as_dict()["summary"]["exact"], False)


if __name__ == "__main__":
    unittest.main()
