import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

from n64dk import media, rom as rommod

SIZE = 0x101000


def image(title=b"SYNTHETIC TITLE", code=b"NXYE", version=0, size=SIZE, ipl=b"") -> bytes:
    """A hand-built cartridge image in z64 order: header fields only, no retail bytes."""
    data = bytearray(size)
    struct.pack_into(">IIIIII", data, 0, 0x80371240, 0x0000000F, 0x80000400, 0x0000144B, 0x12345678, 0x9ABCDEF0)
    data[0x20:0x20 + len(title)] = title
    data[0x3B:0x3F] = code
    data[0x3F] = version
    data[0x40:0x40 + len(ipl)] = ipl
    return bytes(data)


def v64(data: bytes) -> bytes:
    out = bytearray(data)
    out[0::2], out[1::2] = data[1::2], data[0::2]
    return bytes(out)


def n64(data: bytes) -> bytes:
    words = struct.unpack(f">{len(data) // 4}I", data)
    return struct.pack(f"<{len(words)}I", *words)


class ByteOrderTest(unittest.TestCase):
    def test_three_orders_read_the_same_header(self):
        base = image()
        for order, data in (("z64", base), ("v64", v64(base)), ("n64", n64(base))):
            with self.subTest(order=order):
                report = media.capability_report(data, checksum=False)
                self.assertEqual((report["media"], report["byte_order"]), ("cartridge", order))
                self.assertEqual(report["header"]["title"], "SYNTHETIC TITLE")
                self.assertEqual(report["header"]["game_id"], "XY")
                self.assertTrue(report["supported"])
                json.dumps(report)

    def test_unknown_first_word_fails_precisely(self):
        data = bytearray(image())
        data[3] = 0x41                                        # 80371241: a non-standard PI word
        report = media.capability_report(bytes(data))
        self.assertFalse(report["supported"])
        self.assertEqual(report["media"], "unknown")
        self.assertIn("80371241", report["unsupported"][0])
        self.assertIn("resembles z64 order", report["unsupported"][0])
        with self.assertRaises(media.UnsupportedMedia) as caught:
            media.require_supported(bytes(data))
        self.assertIs(caught.exception.report["supported"], False)
        empty = media.capability_report(b"")
        self.assertIn("(empty)", empty["unsupported"][0])

    def test_too_small_for_the_kit(self):
        report = media.capability_report(image(size=0x1000))
        self.assertEqual(report["media"], "cartridge")
        self.assertFalse(report["supported"])
        self.assertIn("smaller than header + IPL3", report["unsupported"][0])
        self.assertFalse(report["capabilities"]["survey"])


class HeaderFactsTest(unittest.TestCase):
    def test_video_standard_from_country_code(self):
        for code, standard in ((b"NXYE", "NTSC"), (b"NXYJ", "NTSC"), (b"NXYP", "PAL"), (b"NXYD", "PAL"),
                               (b"NXYQ", "unknown")):
            with self.subTest(code=code):
                video = media.capability_report(image(code=code), checksum=False)["video"]
                self.assertEqual(video["standard"], standard)
                self.assertIn("convention", video["basis"])

    def test_unknown_ipl3_is_reported_not_guessed(self):
        report = media.capability_report(image(), checksum=False)
        self.assertIsNone(report["cic"]["value"])
        self.assertIsNone(report["checksum_ok"])
        self.assertTrue(any("not a retail boot code" in x for x in report["homebrew"]["indicators"]))
        self.assertIn("unrelocated", report["capabilities"]["boot_entry"])

    def test_known_ipl3_family_and_checksum(self):
        ipl = bytes(range(256)) * 15
        data = image(ipl=ipl)
        crc = zlib.crc32(data[0x40:0x1000])
        with mock.patch.dict(rommod.CIC_BY_IPL3_CRC, {crc: "6102"}), \
                mock.patch.object(rommod, "compute_checksum", return_value=(0x12345678, 0x9ABCDEF0)) as computed:
            report = media.capability_report(data)
            self.assertEqual(report["cic"]["value"], "6102")
            self.assertTrue(report["checksum_ok"])
            self.assertEqual(report["homebrew"]["indicators"], [])
            computed.return_value = (1, 2)
            modified = media.capability_report(data)
        self.assertFalse(modified["checksum_ok"])
        self.assertTrue(any("checksum does not match" in x for x in modified["homebrew"]["indicators"]))

    def test_save_type_and_expansion_pak(self):
        retail = media.capability_report(image(), checksum=False)
        self.assertEqual(retail["save_type"]["value"], "unknown")
        self.assertEqual(retail["expansion_pak"]["value"], "not determinable statically")
        homebrew = media.capability_report(image(code=b"NEDE", version=0x50), checksum=False)
        self.assertEqual(homebrew["save_type"]["value"], "FlashRAM 1 Mbit")
        self.assertIn("assumed", homebrew["save_type"]["basis"])
        self.assertIn("advanced homebrew header (game id ED)", homebrew["homebrew"]["indicators"])

    def test_ique_is_never_claimed(self):
        report = media.capability_report(image(code=b"NXYC"), checksum=False)
        self.assertEqual(report["ique"]["detected"], False)
        self.assertEqual(report["ique"]["status"], "not detected: unsupported")
        self.assertTrue(report["ique"]["weak_indicators"][0].startswith("country code C (China"))
        self.assertEqual(report["ique"]["weak_indicators"][-1], "non-retail IPL3")
        blank = media.capability_report(image(title=b"", code=b"\0\0\0\0"), checksum=False)
        self.assertFalse(blank["ique"]["detected"])
        self.assertIn("all blank", blank["ique"]["weak_indicators"][0])
        self.assertEqual(blank["video"]["standard"], "unknown")
        self.assertTrue(blank["supported"])                 # still a cartridge image the kit can read
        retail_like = media.capability_report(image(), checksum=False)
        self.assertEqual(retail_like["ique"]["weak_indicators"], [])

    def test_categories(self):
        disk_part = media.capability_report(image(code=b"EXYJ"), checksum=False)
        self.assertFalse(disk_part["supported"])
        expandable = media.capability_report(image(code=b"CXYJ"), checksum=False)
        self.assertTrue(expandable["supported"])
        self.assertTrue(any("64DD disk part" in w for w in expandable["warnings"]))
        self.assertEqual(expandable["media_category"]["meaning"], "expandable game: Game Pak part")


class DiskImageTest(unittest.TestCase):
    def test_64dd_sizes_are_unsupported_with_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            for size, kind in ((0x3DEC800, "retail"), (0x435B0C0, "development")):
                path = Path(tmp) / f"disk_{size:X}.ndd"
                with open(path, "wb") as stream:          # sparse: nothing is written but the length
                    stream.truncate(size)
                with self.subTest(kind=kind):
                    report = media.capability_report(path)
                    self.assertEqual((report["media"], report["supported"]), ("64dd-disk", False))
                    self.assertIn(kind, report["unsupported"][0])
                    self.assertIn("size", report["unsupported"][0])
                    with self.assertRaises(media.UnsupportedMedia):
                        media.survey_hook(path)

    def test_cartridge_from_a_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.v64"
            path.write_bytes(v64(image()))
            report = media.survey_hook(path)
        self.assertEqual((report["source"], report["byte_order"], report["size"]), ("synthetic.v64", "v64", SIZE))


if __name__ == "__main__":
    unittest.main()
