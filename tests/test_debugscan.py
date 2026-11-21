import struct
import unittest

from n64dk.analysis import debugscan


def hdrr(endian=">", **fields):
    values = {name: 0 for name in debugscan.HDRR_FIELDS}
    values.update(magic=0x7009, vstamp=0x030B, **fields)
    return struct.pack(endian + "hH" + "i" * 23, *(values[n] for n in debugscan.HDRR_FIELDS))


class EcoffTest(unittest.TestCase):
    def test_a_header_with_its_string_table_is_decoded(self):
        names = b"\x00main\x00osInit\x00gameLoop\x00"
        # file layout: [header][external strings][external symbols]
        header = hdrr(issExtMax=len(names), cbSsExtOffset=debugscan.HDRR_SIZE, iextMax=3,
                      cbExtOffset=debugscan.HDRR_SIZE + len(names) + 3)
        blob = b"\xff" * 64 + header + names + b"\x00" * 3 + b"\x00" * 48 + b"\xee" * 32
        result = debugscan.ecoff(blob)
        self.assertEqual(len(result["found"]), 1)
        entry = result["found"][0]
        self.assertEqual((entry["offset"], entry["endian"], entry["tables"]), (64, "big", "inside"))
        self.assertEqual(entry["names"], ["main", "osInit", "gameLoop"])

    def test_little_endian_and_outside_tables(self):
        header = hdrr("<", isymMax=10, cbSymOffset=0x10000)
        result = debugscan.ecoff(b"\x00" * 8 + header)
        self.assertEqual([(f["endian"], f["tables"]) for f in result["found"]], [("little", "outside")])

    def test_implausible_headers_are_rejected_with_reasons(self):
        bad = [hdrr(isymMax=-1), hdrr(), hdrr(isymMax=4, cbSymOffset=0x100, ipdMax=4, cbPdOffset=0x110)]
        blob = b"".join(b + b"\x00" * 4 for b in bad)
        result = debugscan.ecoff(blob)
        self.assertEqual(result["found"], [])
        self.assertEqual(result["rejected"].get("count out of range"), 1)
        self.assertEqual(result["rejected"].get("no symbols, procedures or externals"), 1)
        self.assertEqual(result["rejected"].get("tables overlap"), 1)

    def test_random_magic_bytes_are_not_headers(self):
        self.assertEqual(debugscan.ecoff(b"\x70\x09" + b"\x00" * 200)["found"], [])


def stab(strx, typ, desc=0, value=0, endian=">"):
    return struct.pack(endian + "IBBhI", strx, typ, 0, desc, value)


class StabsTest(unittest.TestCase):
    def test_a_stabs_run_and_its_strings(self):
        strings = b"\x00main.c\x00main:F1\x00x:p1\x00int:t1=r1;0;-1;\x00"
        offs = {s: strings.index(s.encode()) for s in ("main.c", "main:F1", "x:p1", "int:t1=r1;0;-1;")}
        run = (stab(offs["main.c"], 0x64, value=0x80000400) + stab(offs["int:t1=r1;0;-1;"], 0x80)
               + stab(offs["main:F1"], 0x24, value=0x80000400) + stab(offs["x:p1"], 0xA0, value=0)
               + b"".join(stab(0, 0x44, desc=n, value=4 * n) for n in range(6)))
        blob = b"\xff" * 16 + run + strings
        result = debugscan.stabs(blob)
        self.assertEqual(len(result["found"]), 1)
        entry = result["found"][0]
        self.assertEqual((entry["offset"], entry["entries"]), (16, 10))
        self.assertEqual(entry["sources"], ["main.c"])
        self.assertEqual(entry["functions"], ["main"])

    def test_a_run_without_a_naming_string_table_is_rejected(self):
        run = stab(0, 0x24, value=4) + stab(0, 0x28) + b"".join(stab(0, 0, desc=n) for n in range(8))
        result = debugscan.stabs(b"\xff" * 8 + run + b"\xff" * 8)
        self.assertEqual(result["found"], [])

    def test_zero_padding_is_not_a_run_and_is_scanned_once(self):
        result = debugscan.stabs(b"\x00" * 120000)
        self.assertEqual(result["found"], [])
        self.assertEqual(result["rejected"]["structureless run"], 2)       # once per endianness


class StringsTest(unittest.TestCase):
    def test_tools_dates_and_messages(self):
        blob = (b"\x00GCC: (GNU) 2.7.2\x00junk\x00@(#)libultra 2.0I\x00Mar 14 1998\x0012:30:01\x00"
                b"Assertion failed: x > 0\x00score %d\x00")
        result = debugscan.build_strings(blob)
        self.assertIn("gcc", result["tool_kinds"])
        self.assertIn("sccs", result["tool_kinds"])
        self.assertEqual([d["text"] for d in result["dates"]], ["Mar 14 1998"])
        self.assertEqual([t["text"] for t in result["times"]], ["12:30:01"])
        self.assertEqual(result["counts"]["assertions"], 1)
        self.assertEqual(result["counts"]["format_strings"], 1)

    def test_short_or_embedded_names_are_not_tools(self):
        result = debugscan.build_strings(b"\x00KMC\x00$Id: PHILIDOR.C,v 1.2 1998/01/17 ab Exp $\x00KMC GCC 2.7.2\x00")
        kinds = [t["kind"] for t in result["tools"]]
        self.assertEqual(kinds.count("kmc"), 1)                 # "KMC GCC 2.7.2" only
        self.assertNotIn("ido", kinds)                          # PHILIDOR is not IDO
        self.assertIn("rcs", kinds)

    def test_scan_runs_every_scanner(self):
        self.assertEqual(set(debugscan.scan(b"\x00" * 64)), {"ecoff", "stabs", "strings"})


if __name__ == "__main__":
    unittest.main()
