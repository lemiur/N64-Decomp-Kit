"""Unit classes kept apart in exports (round 12)."""
import json
import tempfile
import unittest
from pathlib import Path

from n64dk.analysis import unitclass


class UnitClassTest(unittest.TestCase):
    def project(self) -> Path:
        root = Path(tempfile.mkdtemp(prefix="n64dk-test-unitclass-"))
        survey = root / "survey"
        survey.mkdir()
        (survey / "layout.json").write_text(json.dumps({"entry": "0x80000400"}))
        (survey / "report.json").write_text(json.dumps({"rsp_in_text": ["0x80050000..0x80050800"]}))
        (survey / "functions.toml").write_text(
            'functions = [\n'
            '    { name = "func_80000400", vram = 0x80000400, size = 0x6c }, # asm\n'
            '    { name = "func_80001000", vram = 0x80001000, size = 0x20 }, # asm\n'
            '    { name = "func_80002000", vram = 0x80002000, size = 0x40 }, # c\n'
            '    { name = "osSendMesg", vram = 0x80003000, size = 0x40 }, # c\n'
            ']\n')
        (survey / "sdk_symbols.txt").write_text("osSendMesg = 0x80003000; // sdk\n")
        return root

    def test_classes(self):
        evidence = unitclass.load(self.project())
        self.assertEqual(unitclass.classify(0x80000400, 0x6C, evidence), "boot")
        self.assertEqual(unitclass.classify(0x80001000, 0x20, evidence), "handwritten")
        self.assertEqual(unitclass.classify(0x80002000, 0x40, evidence), "game")
        self.assertEqual(unitclass.classify(0x80003000, 0x40, evidence), "sdk")
        self.assertEqual(unitclass.classify(0x8004FFF0, 0x20, evidence), "rsp")       # overlaps the RSP range

    def test_missing_survey_is_game(self):
        evidence = unitclass.load(Path(tempfile.mkdtemp(prefix="n64dk-test-unitclass-")))
        self.assertEqual(unitclass.classify(0x80000400, 4, evidence), "game")


if __name__ == "__main__":
    unittest.main()
