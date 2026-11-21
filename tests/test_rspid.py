"""RSP microcode and boot identification (wave 4, workstream 8): synthetic images only.

The RSP "code" here is four hand-picked words (a COP2 vector op, an LWC2, an ADDI and a J) repeated,
enough for the kit's RSP detector; the version strings are the public string formats.  No microcode
or ROM bytes.
"""
import contextlib
import io
import json
import re
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

from n64dk import rom as rommod
from n64dk.analysis import rspid

BASE = 0x80000400                       # main image: ROM 0x1000 is VRAM 0x80000400
RSP_WORDS = (0x4A000010, 0xC8011000, 0x20010001, 0x08000400)
F3DEX2_STRING = b"RSP Gfx ucode F3DEX       fifo 2.08  Yoshitaka Yasumoto 1999 Nintendo.\0"
MAPPING = rspid.Mapping(0x1000, BASE, 0x5000)


def vram(rom: int) -> int:
    return BASE + rom - 0x1000


def rsp_text(size: int, jump: int = 0x400) -> bytes:
    words = [RSP_WORDS[i % 4] if i % 4 != 3 else 0x08000000 | jump for i in range(size // 4)]
    return struct.pack(f">{len(words)}I", *words)


def put(buf: bytearray, off: int, words) -> None:
    struct.pack_into(f">{len(words)}I", buf, off, *words)


def image(size: int = 0x6000, data_end: int = 0x3100, table: bool = True, code: bool = True,
          ostask: bool = False, string: bytes = F3DEX2_STRING) -> bytearray:
    """Text A 0x2000..0x2400 (graphics), text B 0x2400..0x24D0 (boot-sized), DMEM data at 0x3000
    with its version string at 0x3040; references in a (text, data) table at 0x4000, in lui/addiu
    pairs at 0x5000 and, optionally, in an OSTask template at 0x4800."""
    buf = bytearray(size)
    buf[0x2000:0x2400] = rsp_text(0x400)
    buf[0x2400:0x24D0] = rsp_text(0xD0)
    if string:
        buf[0x3000:0x3040] = b"\x11" * 0x40
        buf[0x3040:0x3040 + len(string)] = string
    if table:
        put(buf, 0x4000, [vram(0x2000), vram(0x3000), 0, 0, vram(data_end)])
    if code:   # lui a0,0x8000; addiu a0,a0,0x1800 (ROM 0x2400); lui a1,0x8000; addiu a1,a1,0x18D0 (0x24D0)
        put(buf, 0x5000, [0x3C048000, 0x24841800, 0x3C058000, 0x24A518D0])
    if ostask:
        put(buf, 0x4800, [1, 0, vram(0x2400), 0xD0, vram(0x2000), 0x1000, vram(0x3000), 0x800] + [0] * 8)
    return buf


def block(report: dict, kind: str, rom: int) -> dict:
    return next(b for b in report["blocks"] if b["kind"] == kind and b["rom"] == rom)


class TableTest(unittest.TestCase):
    def test_table_holds_names_patterns_and_sizes_only(self):
        for row in rspid.KNOWN:
            for value in (row.name, row.family, row.task, row.source):
                self.assertIsInstance(value, str)
            self.assertTrue(row.pattern is None or isinstance(row.pattern, str))
            for size in (row.text_size, row.data_size):
                self.assertTrue(size is None or isinstance(size, int))
            if row.pattern is None:
                self.assertIsNotNone(row.text_size, row.name)   # a row without a string is a size row
            else:
                re.compile(row.pattern)

    def test_version_strings(self):
        cases = {
            "RSP Gfx ucode F3DEX       fifo 2.08  Yoshitaka Yasumoto 1999 Nintendo.":
                ("gspF3DEX2.fifo", "F3DEX2", "2.08"),
            "RSP Gfx ucode F3DEX.NoN   fifo 2.06  Yoshitaka Yasumoto 1998 Nintendo.":
                ("gspF3DEX2.NoN.fifo", "F3DEX2", "2.06"),
            "RSP Gfx ucode F3DZEX.NoN  fifo 2.06H Yoshitaka Yasumoto 1998 Nintendo.":
                ("gspF3DZEX2.NoN.fifo", "F3DZEX2", "2.06H"),
            "RSP Gfx ucode F3DLX.Rej   fifo 2.03  Yoshitaka Yasumoto 1998 Nintendo.":
                ("gspF3DLX2.Rej.fifo", "F3DLX2", "2.03"),
            "RSP Gfx ucode S2DEX       fifo 2.05  Yoshitaka Yasumoto 1998 Nintendo.":
                ("gspS2DEX2.fifo", "S2DEX2", "2.05"),
            "RSP Gfx ucode F3DEX       1.23 Yoshitaka Yasumoto Nintendo.": ("gspF3DEX", "F3DEX", "1.23"),
            "RSP Gfx ucode L3DEX       1.21 Yoshitaka Yasumoto Nintendo.": ("gspL3DEX", "L3DEX", "1.21"),
            "RSP Gfx ucode F3DEX       Yoshitaka Yasumoto Nintendo.": ("gspF3DEX", "F3DEX", None),
            "RSP SW Version: 2.0D, 04-01-96": ("gspFast3D", "F3D", "2.0D"),
        }
        for text, (name, family, version) in cases.items():
            got = rspid.match_string(text)
            self.assertIsNotNone(got, text)
            self.assertEqual((got["name"], got["family"], got["version"]), (name, family, version), text)
            self.assertEqual(got["basis"], "version-string")
        for text in ("RSP Gfx ucode F3DEXBG.NoN fifo 2.08 Yoshitaka Yasumoto 1999 Nintendo.",
                     "RSP Audio ucode something", "RSP SW Version: unknown"):
            self.assertIsNone(rspid.match_string(text), text)

    def test_size_candidates(self):
        self.assertEqual(rspid.size_candidates("text", 0xD0), ["rspboot"])
        self.assertEqual(rspid.size_candidates("data", 0x420), ["gspF3DEX2.fifo"])
        self.assertEqual(rspid.size_candidates("text", 0x123), [])


class DigestTest(unittest.TestCase):
    def test_rel_digest_ignores_imem_targets_and_padding(self):
        a, b = rsp_text(0x40, 0x400), rsp_text(0x40, 0x520) + bytes(16)
        self.assertNotEqual(rspid.hashlib.sha256(a).hexdigest(), rspid.hashlib.sha256(b).hexdigest())
        self.assertEqual(rspid.rel_digest(a, "text"), rspid.rel_digest(b, "text"))
        self.assertNotEqual(rspid.rel_digest(a, "text"), rspid.rel_digest(rsp_text(0x44), "text"))
        self.assertEqual(rspid.rel_digest(b"abc\0\0", "data"), rspid.rel_digest(b"abc", "data"))

    def test_address_constants(self):
        refs = rspid.address_constants(bytes(image()), 0x1000, 0x6000, MAPPING)
        self.assertEqual(refs[0x2000], [(0x4000, "word")])
        self.assertEqual(refs[0x2400], [(0x5004, "hi/lo")])
        self.assertEqual(refs[0x24D0], [(0x500C, "hi/lo")])
        excluded = rspid.address_constants(bytes(image()), 0x1000, 0x6000, MAPPING, [(0x4000, 0x4008)])
        self.assertNotIn(0x2000, excluded)


class IdentifyTest(unittest.TestCase):
    def test_string_names_data_and_pairs_text_size_second(self):
        rep = rspid.identify(bytes(image()), mapping=MAPPING)
        data = block(rep, "data", 0x3000)
        self.assertEqual((data["end"], data["extent"], data["status"]), (0x3100, "exact", "named"))
        self.assertEqual((data["identity"]["name"], data["identity"]["version"]), ("gspF3DEX2.fifo", "2.08"))
        text = block(rep, "text", 0x2000)
        self.assertEqual((text["end"], text["extent"], text["status"]), (0x2400, "exact", "paired"))
        self.assertEqual((text["identity"]["family"], text["data_rom"]), ("F3DEX2", 0x3000))
        self.assertIn("adjacent pointer words", text["identity"]["basis"])
        boot = block(rep, "text", 0x2400)
        self.assertEqual((boot["end"], boot["size"], boot["status"]), (0x24D0, 0xD0, "candidate-by-size"))
        self.assertEqual(boot["size_candidates"], ["rspboot"])
        self.assertEqual(boot["vram"], f"0x{vram(0x2400):08X}")
        self.assertEqual(rep["summary"]["text_bytes"], {"candidate-by-size": 0xD0, "paired": 0x400})

    def test_string_wins_over_size(self):
        rep = rspid.identify(bytes(image(data_end=0x3420)), mapping=MAPPING)
        data = block(rep, "data", 0x3000)
        self.assertEqual((data["size"], data["size_candidates"]), (0x420, ["gspF3DEX2.fifo"]))
        self.assertEqual((data["status"], data["identity"]["basis"]), ("named", "version-string"))

    def test_ostask_template_names_boot_by_role(self):
        rep = rspid.identify(bytes(image(ostask=True)), mapping=MAPPING)
        self.assertEqual([t["rom"] for t in rep["tasks"]], [0x4800])
        self.assertEqual(rep["tasks"][0]["type"], "M_GFXTASK")
        boot = block(rep, "text", 0x2400)
        self.assertEqual((boot["status"], boot["identity"]["name"]), ("role", "rspboot"))
        text = block(rep, "text", 0x2000)
        self.assertEqual(text["status"], "paired")
        self.assertIn("OSTask template at ROM 0x4800", text["identity"]["basis"])

    def test_unidentified_without_string_or_reference(self):
        rep = rspid.identify(bytes(image(table=False, code=False, string=b"")), mapping=MAPPING)
        texts = [b for b in rep["blocks"] if b["kind"] == "text"]
        self.assertEqual(len(texts), 1)
        self.assertEqual((texts[0]["rom"], texts[0]["end"], texts[0]["extent"]), (0x2000, 0x24C0, "estimated"))
        self.assertEqual(texts[0]["status"], "unidentified")
        self.assertIn("64-byte estimate", texts[0]["reason"])
        self.assertEqual(rspid.identify(bytes(image(table=False, code=False, string=b"")))["summary"]["blocks"],
                         {"text:unidentified": 1})

    def test_unlisted_string_is_reported_not_named(self):
        odd = b"RSP Gfx ucode F3DEXBG.NoN fifo 2.08  Yoshitaka Yasumoto 1999 Nintendo.\0"
        rep = rspid.identify(bytes(image(string=odd)), mapping=MAPPING)
        data = block(rep, "data", 0x3000)
        self.assertEqual(data["status"], "named-unlisted")
        self.assertIsNone(data["identity"]["name"])
        self.assertEqual(block(rep, "text", 0x2000)["status"], "paired")


class BootTest(unittest.TestCase):
    def boot_image(self, lo: int) -> bytearray:
        buf = bytearray(0x1100)
        buf[0x40:0x1000] = bytes((i * 7) & 0xFF for i in range(0xFC0))
        put(buf, 0x1000, [0x3C088000, 0x25080000 | lo, 0x01000008, 0, 0x11111111])
        return buf

    def test_entry_routine_digests(self):
        a, b = rspid.identify_boot(bytes(self.boot_image(0x1234))), rspid.identify_boot(bytes(self.boot_image(0x5678)))
        self.assertEqual((a["entry_routine"]["size"], a["entry_routine"]["end_basis"]),
                         (0x10, "first jr/jalr and its delay slot"))
        self.assertNotEqual(a["entry_routine"]["sha256"], b["entry_routine"]["sha256"])
        self.assertEqual(a["entry_routine"]["rel_digest"], b["entry_routine"]["rel_digest"])

    def test_ipl3_by_the_kits_cic_table(self):
        buf = self.boot_image(0)
        self.assertEqual(rspid.identify_boot(bytes(buf))["ipl3"]["status"], "unidentified")
        with mock.patch.dict(rommod.CIC_BY_IPL3_CRC, {zlib.crc32(bytes(buf[0x40:0x1000])): "6102"}):
            ipl3 = rspid.identify_boot(bytes(buf))["ipl3"]
        self.assertEqual((ipl3["status"], ipl3["cic"], ipl3["chips"]), ("named", "6102", "CIC-NUS-6102/7101"))


class ExplainAndCensusTest(unittest.TestCase):
    def test_explanations(self):
        rep = rspid.identify(bytes(image()), mapping=MAPPING)
        rep["boot"] = rspid.identify_boot(bytes(image()))
        causes = {(e["unit"], e["rom"]): e for e in rspid.explain(rep)}
        self.assertIn("gspF3DEX2.fifo 2.08", causes[("rsp", 0x2000)]["cause"])
        self.assertIn("candidate rspboot by size only", causes[("rsp", 0x2400)]["cause"])
        self.assertIn("named by its version string", causes[("rsp", 0x3000)]["cause"])
        self.assertEqual(causes[("boot", 0x40)]["status"], "unidentified")

    def test_census_command(self):
        root = Path(tempfile.mkdtemp(prefix="n64dk-test-rspid-"))
        project = root / "synthetic"
        (project / "survey").mkdir(parents=True)
        rom = image(size=0x101000)
        put(rom, 0, [0x80371240, 0, BASE])
        (project / "baserom.z64").write_bytes(bytes(rom))
        (project / "n64dk.toml").write_text('[images.main]\npath = "baserom.z64"\n')
        (project / "survey" / "layout.json").write_text(json.dumps(
            {"main": {"kind": "rom", "rom": "0x1000", "rom_end": "0x6000", "vram": f"0x{BASE:08X}"}}))
        out = root / "census.json"
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(rspid.main(["census", "--out", str(out), str(project)]), 0)
        lines = [json.loads(line) for line in printed.getvalue().splitlines()]
        self.assertEqual(lines[0]["title"], "synthetic")
        self.assertIn("gspF3DEX2.fifo 2.08", lines[0]["named"])
        result = json.loads(out.read_text())
        self.assertEqual(result["summary"]["titles"], 1)
        self.assertEqual(result["titles"][0]["summary"]["blocks"],
                         {"data:named": 1, "text:candidate-by-size": 1, "text:paired": 1})


if __name__ == "__main__":
    unittest.main()
