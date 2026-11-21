"""relink-cartridge: a whole ROM assembled from linked images, exactly repacked segments and
preserved inputs, each counted apart (synthetic: a fake per-image relink, a zlib segment)."""
import json
import tempfile
import unittest
import zlib
from pathlib import Path

from n64dk.match import cartridge as cart

MAIN = bytes(range(16)) * 4                          # 64 bytes at ROM 0x1000
SEGMENT = b"segment code " * 20                       # decoded payload of the compressed segment
STREAM = zlib.compress(SEGMENT, 9)


def rom_with(stream: bytes = STREAM) -> bytes:
    rom = bytearray(0x3000)
    rom[0:4] = b"\x80\x37\x12\x40"
    rom[0x1000:0x1000 + len(MAIN)] = MAIN
    rom[0x2000:0x2000 + len(stream)] = stream
    return bytes(rom)


class Relinks:
    """Stands in for relink.relink: writes relinked.bin (possibly altered) and a minimal report."""

    def __init__(self, altered: dict | None = None):
        self.altered = altered or {}

    def __call__(self, tree, image, build, *, data, delta, is_rom, keep_image, **_options):
        linked = self.altered.get(image, data if not is_rom else MAIN)
        Path(build).mkdir(parents=True, exist_ok=True)
        (Path(build) / "relinked.bin").write_bytes(linked)
        expected = MAIN if is_rom else data
        return {"outcome": "identical" if linked == expected else "different", "differing_bytes": 0,
                "coverage": {}, "source_text_bytes": len(linked), "source_data_bytes": 0,
                "extent": {"rom_offset": "0x1000"}}


class CartridgeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.init = self.root / "init"
        (self.init / "survey").mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def run_with(self, rom: bytes, relinks: Relinks, stream_len: int):
        (self.init / "survey" / "layout.json").write_text(json.dumps({"code_segments": [
            {"name": "seg", "rom": "0x2000", "rom_end": f"0x{0x2000 + stream_len:X}", "vram": "0x80100000",
             "codec": "zlib"}]}))
        images = {"main": (rom, 0x80000400 - 0x1000, True), "seg": (SEGMENT, 0x80100000, False)}
        return cart.relink_cartridge(self.root / "tree", self.init, self.root / "build", rom=rom, images=images,
                                     relink_fn=relinks)

    def test_identical_cartridge_with_an_exact_repack(self):
        report = self.run_with(rom_with(), Relinks(), len(STREAM))
        self.assertTrue(report["rom_sha1_match"])
        self.assertEqual(report["bytes_by_provenance"]["linked"], len(MAIN))
        self.assertEqual(report["bytes_by_provenance"]["repacked"], len(STREAM))
        self.assertEqual(report["bytes_by_provenance"]["preserved-compressed"], 0)
        self.assertEqual(report["compressed_payload_reconstructed"], len(SEGMENT))
        self.assertFalse((self.root / "build" / "cartridge.z64").exists())      # no ROM bytes kept by default

    def test_a_stream_no_encoder_reproduces_is_preserved_not_repacked(self):
        stream = zlib.compress(SEGMENT, 1) + b"\0"                                  # trailing byte in the slice
        report = self.run_with(rom_with(stream), Relinks(), len(stream))
        self.assertTrue(report["rom_sha1_match"])                                 # only thanks to the preserved input
        self.assertEqual(report["bytes_by_provenance"]["repacked"], 0)
        self.assertEqual(report["bytes_by_provenance"]["preserved-compressed"], len(stream))
        self.assertGreater(report["preserved_inputs_needed"], 0)

    def test_a_changed_unit_changes_the_cartridge(self):
        changed = bytearray(MAIN)
        changed[5] ^= 0xFF
        report = self.run_with(rom_with(), Relinks({"main": bytes(changed)}), len(STREAM))
        self.assertFalse(report["rom_sha1_match"])
        self.assertEqual(report["first_difference"], {"rom_offset": "0x1005", "owner": "linked:main"})

    def test_a_relinked_payload_that_differs_is_preserved(self):
        report = self.run_with(rom_with(), Relinks({"seg": SEGMENT[:-1] + b"!"}), len(STREAM))
        self.assertEqual(report["bytes_by_provenance"]["repacked"], 0)
        self.assertIn("differs", report["images"]["seg"]["placed"])


    def test_an_uncompressed_copied_segment_is_linked_at_its_rom_offset(self):
        rom = bytearray(rom_with())
        rom[0x2000:0x2000 + len(SEGMENT)] = SEGMENT
        (self.init / "survey" / "layout.json").write_text(json.dumps({"code_segments": [
            {"name": "seg", "rom": "0x2000", "rom_end": f"0x{0x2000 + len(SEGMENT):X}", "codec": None}]}))
        images = {"main": (bytes(rom), 0x80000400 - 0x1000, True), "seg": (SEGMENT, 0x80100000, False)}
        report = cart.relink_cartridge(self.root / "tree", self.init, self.root / "build", rom=bytes(rom),
                                       images=images, relink_fn=Relinks())
        self.assertTrue(report["rom_sha1_match"])
        self.assertEqual(report["bytes_by_provenance"]["linked"], len(MAIN) + len(SEGMENT))
        self.assertIn("uncompressed", report["images"]["seg"]["placed"])
        changed = cart.relink_cartridge(self.root / "tree", self.init, self.root / "build2", rom=bytes(rom),
                                        images=images, relink_fn=Relinks({"seg": SEGMENT[:-1] + b"!"}))
        self.assertEqual(changed["first_difference"]["owner"], "linked:seg")


if __name__ == "__main__":
    unittest.main()
