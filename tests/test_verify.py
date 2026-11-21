"""Strict verification: the cases where the exploration oracle reports a false ``exact``.

The ELF fixtures are tiny relocatable MIPS objects (assembled once with GNU as, embedded so the
tests need no toolchain).  TRUTH calls ``goodFunction``, reads ``gRight`` and takes the address
of a local ``table`` in ``.data`` holding 1,2,3,4.  WRONG calls ``otherFunction`` and reads
``gWrong``.  SHORTDATA is TRUTH with table = 1,0,0,0.  LOOSE has ``.rodata`` that ``.text``
never references.
"""
import base64
import tempfile
import unittest
from pathlib import Path

from n64dk.elf import Elf
from n64dk.match import compare as cmp
from n64dk.match import verify
from n64dk.match.symbols import SymbolMap
from n64dk.match.engine import Outcome, Target, _compare_at
from n64dk.match.pipeline import BuildSpec

TRUTH = (
    "f0VMRgECAQAAAAAAAAAAAAABAAgAAAABAAAAAAAAAAAAAAFoIAARAQA0AAAAAAAoAAgABwAAAAAAAAAAAAAAAAwAAAAAAAAAPAIA"
    "AIxCAAA8BAAAJIQAAAPgAAgAAAAAAAAAAQAAAAIAAAADAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAAABAAAAAAAA"
    "AAAAAAAAAwAAAwAAAAAAAAAAAAAAAAMAAAQAAAABAAAAAAAAAAAAAAADAAAABwAAAAAAAAAAEAAAAQAAAAkAAAAAAAAAABAAAAAA"
    "AAAWAAAAAAAAAAAQAAAAAHRhYmxlAGYAZ29vZEZ1bmN0aW9uAGdSaWdodAAAAAAAAAAAAAAGBAAAAAgAAAcFAAAADAAABwYAAAAQ"
    "AAACBQAAABQAAAIGAC5zeW10YWIALnN0cnRhYgAuc2hzdHJ0YWIALnJlbC50ZXh0AC5kYXRhAC5ic3MAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB8AAAABAAAABgAAAAAAAABAAAAAIAAAAAAAAAAAAAAAEAAAAAAAAAAbAAAACQAA"
    "AEAAAAAAAAABEAAAACgAAAAFAAAAAQAAAAQAAAAIAAAAJQAAAAEAAAADAAAAAAAAAGAAAAAQAAAAAAAAAAAAAAAQAAAAAAAAACsA"
    "AAAIAAAAAwAAAAAAAABwAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAABAAAAAgAAAAAAAAAAAAAAcAAAAIAAAAAGAAAABQAAAAQAAAAQ"
    "AAAACQAAAAMAAAAAAAAAAAAAAPAAAAAdAAAAAAAAAAAAAAABAAAAAAAAABEAAAADAAAAAAAAAAAAAAE4AAAAMAAAAAAAAAAAAAAA"
    "AQAAAAA=")

WRONG = (
    "f0VMRgECAQAAAAAAAAAAAAABAAgAAAABAAAAAAAAAAAAAAFoIAARAQA0AAAAAAAoAAgABwAAAAAAAAAAAAAAAAwAAAAAAAAAPAIA"
    "AIxCAAA8BAAAJIQAAAPgAAgAAAAAAAAAAQAAAAIAAAADAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAAABAAAAAAAA"
    "AAAAAAAAAwAAAwAAAAAAAAAAAAAAAAMAAAQAAAABAAAAAAAAAAAAAAADAAAABwAAAAAAAAAAEAAAAQAAAAkAAAAAAAAAABAAAAAA"
    "AAAXAAAAAAAAAAAQAAAAAHRhYmxlAGYAb3RoZXJGdW5jdGlvbgBnV3JvbmcAAAAAAAAAAAAGBAAAAAgAAAcFAAAADAAABwYAAAAQ"
    "AAACBQAAABQAAAIGAC5zeW10YWIALnN0cnRhYgAuc2hzdHJ0YWIALnJlbC50ZXh0AC5kYXRhAC5ic3MAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB8AAAABAAAABgAAAAAAAABAAAAAIAAAAAAAAAAAAAAAEAAAAAAAAAAbAAAACQAA"
    "AEAAAAAAAAABEAAAACgAAAAFAAAAAQAAAAQAAAAIAAAAJQAAAAEAAAADAAAAAAAAAGAAAAAQAAAAAAAAAAAAAAAQAAAAAAAAACsA"
    "AAAIAAAAAwAAAAAAAABwAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAABAAAAAgAAAAAAAAAAAAAAcAAAAIAAAAAGAAAABQAAAAQAAAAQ"
    "AAAACQAAAAMAAAAAAAAAAAAAAPAAAAAeAAAAAAAAAAAAAAABAAAAAAAAABEAAAADAAAAAAAAAAAAAAE4AAAAMAAAAAAAAAAAAAAA"
    "AQAAAAA=")

SHORTDATA = (
    "f0VMRgECAQAAAAAAAAAAAAABAAgAAAABAAAAAAAAAAAAAAFoIAARAQA0AAAAAAAoAAgABwAAAAAAAAAAAAAAAAwAAAAAAAAAPAIA"
    "AIxCAAA8BAAAJIQAAAPgAAgAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAAABAAAAAAAA"
    "AAAAAAAAAwAAAwAAAAAAAAAAAAAAAAMAAAQAAAABAAAAAAAAAAAAAAADAAAABwAAAAAAAAAAEAAAAQAAAAkAAAAAAAAAABAAAAAA"
    "AAAWAAAAAAAAAAAQAAAAAHRhYmxlAGYAZ29vZEZ1bmN0aW9uAGdSaWdodAAAAAAAAAAAAAAGBAAAAAgAAAcFAAAADAAABwYAAAAQ"
    "AAACBQAAABQAAAIGAC5zeW10YWIALnN0cnRhYgAuc2hzdHJ0YWIALnJlbC50ZXh0AC5kYXRhAC5ic3MAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB8AAAABAAAABgAAAAAAAABAAAAAIAAAAAAAAAAAAAAAEAAAAAAAAAAbAAAACQAA"
    "AEAAAAAAAAABEAAAACgAAAAFAAAAAQAAAAQAAAAIAAAAJQAAAAEAAAADAAAAAAAAAGAAAAAQAAAAAAAAAAAAAAAQAAAAAAAAACsA"
    "AAAIAAAAAwAAAAAAAABwAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAABAAAAAgAAAAAAAAAAAAAAcAAAAIAAAAAGAAAABQAAAAQAAAAQ"
    "AAAACQAAAAMAAAAAAAAAAAAAAPAAAAAdAAAAAAAAAAAAAAABAAAAAAAAABEAAAADAAAAAAAAAAAAAAE4AAAAMAAAAAAAAAAAAAAA"
    "AQAAAAA=")

LOOSE = (
    "f0VMRgECAQAAAAAAAAAAAAABAAgAAAABAAAAAAAAAAAAAADwIAARAQA0AAAAAAAoAAgABwAAAAAAAAAAAAAAAAPgAAgkAgABAAAA"
    "AAAAAADerb7vEjRWeAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAAABAAAAAAAAAAAAAAAAAwAAAgAAAAAAAAAAAAAAAAMA"
    "AAMAAAAAAAAAAAAAAAADAAAEAAAAAQAAAAAAAAAAEAAAAQBmAAAuc3ltdGFiAC5zdHJ0YWIALnNoc3RydGFiAC50ZXh0AC5kYXRh"
    "AC5ic3MALnJvZGF0YQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABsAAAABAAAABgAAAAAAAABA"
    "AAAAEAAAAAAAAAAAAAAAEAAAAAAAAAAhAAAAAQAAAAMAAAAAAAAAUAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAJwAAAAgAAAADAAAA"
    "AAAAAFAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAACwAAAABAAAAAgAAAAAAAABQAAAACAAAAAAAAAAAAAAABAAAAAAAAAABAAAAAgAA"
    "AAAAAAAAAAAAWAAAAGAAAAAGAAAABQAAAAQAAAAQAAAACQAAAAMAAAAAAAAAAAAAALgAAAADAAAAAAAAAAAAAAABAAAAAAAAABEA"
    "AAADAAAAAAAAAAAAAAC7AAAANAAAAAAAAAAAAAAAAQAAAAA=")

BASE, TEXT, DATA = 0x80000000, 0x80000400, 0x80000800
SYMBOLS = {"goodFunction": 0x80001000, "otherFunction": 0x80001100, "gRight": 0x80100000, "gWrong": 0x80100004}


def elf(blob: str) -> Elf:
    return Elf(base64.b64decode(blob))


def rom_from(truth: Elf, bases: dict) -> bytes:
    rom = bytearray(0x2000)
    placed = {".text": TEXT, **bases}
    for name, base in placed.items():
        section = cmp.find_section(truth, name)
        if section is not None and section.size:
            data = truth.relocate(section.name, placed, SymbolMap(SYMBOLS))
            rom[base - BASE:base - BASE + len(data)] = data
    return bytes(rom)


def strict(candidate, rom, symbols, bases=None, size=0x20, **kw):
    return verify.strict_compare(candidate, rom, lambda v: v - BASE, TEXT, size, SymbolMap(symbols), bases, **kw)


class StrictCompareTest(unittest.TestCase):
    def setUp(self):
        self.rom = rom_from(elf(TRUTH), {".data": DATA})

    def test_fully_declared_unit_is_exact(self):
        result = strict(elf(TRUTH), self.rom, SYMBOLS, {".data": DATA})
        self.assertEqual((result.verdict, result.assumptions, result.conflicts), ("exact", [], []))
        self.assertEqual(result.sections[0]["status"], "equal")

    def test_undeclared_addresses_are_assumptions_not_proof(self):
        result = strict(elf(TRUTH), self.rom, {})
        self.assertEqual(result.verdict, "assumed")
        self.assertTrue(result.text_equal)
        self.assertEqual({(a.kind, a.name, a.address) for a in result.assumptions},
                         {("symbol", "goodFunction", 0x80001000), ("symbol", "gRight", 0x80100000),
                          ("section", ".data", DATA)})

    def test_the_exploration_oracle_accepts_the_wrong_callee(self):
        loose = cmp.compare_object(elf(WRONG), self.rom, lambda v: v - BASE, TEXT, 0x20, SymbolMap({}))
        self.assertTrue(loose.exact)            # documents the hazard strict_compare exists for

    def test_wrong_callee_and_global_are_a_mismatch_when_declared(self):
        self.assertEqual(strict(elf(WRONG), self.rom, SYMBOLS, {".data": DATA}).verdict, "mismatch")

    def test_assumption_that_collides_with_a_declared_symbol_is_a_conflict(self):
        known = {"goodFunction": 0x80001000, "gRight": 0x80100000}      # the wrong names are not declared
        result = strict(elf(WRONG), self.rom, known, {".data": DATA})
        self.assertEqual(result.verdict, "mismatch")
        self.assertEqual(len(result.conflicts), 2)
        self.assertIn("goodFunction", result.conflicts[0] + result.conflicts[1])

    def test_call_must_land_on_a_function_start(self):
        result = strict(elf(TRUTH), self.rom, {"gRight": 0x80100000}, {".data": DATA},
                        function_starts={0x80002000})
        self.assertEqual(result.verdict, "mismatch")
        self.assertIn("not a function start", result.conflicts[0])

    def test_trailing_zeros_do_not_match_real_data(self):
        result = strict(elf(SHORTDATA), self.rom, SYMBOLS, {".data": DATA})
        self.assertEqual(result.verdict, "assumed")                    # cannot tell padding from missing data
        self.assertEqual(result.sections[0]["status"], "unverified")
        owned = strict(elf(SHORTDATA), self.rom, SYMBOLS, {".data": DATA}, section_sizes={".data": 16})
        self.assertEqual((owned.verdict, owned.sections[0]["status"]), ("mismatch", "different"))
        loose = cmp.compare_object(elf(SHORTDATA), self.rom, lambda v: v - BASE, TEXT, 0x20, SymbolMap(SYMBOLS),
                                   {".data": DATA})
        self.assertTrue(loose.exact)            # the hazard

    def test_unreferenced_data_is_reported_not_skipped(self):
        loose_elf = elf(LOOSE)
        rom = rom_from(loose_elf, {})
        result = strict(loose_elf, rom, {}, size=8)
        self.assertEqual(result.verdict, "assumed")
        self.assertIn("was not compared", " ".join(result.notes))
        placed = rom_from(loose_elf, {".rodata": DATA})
        self.assertEqual(strict(loose_elf, placed, {}, {".rodata": DATA}, size=8).verdict, "exact")

    def test_data_outside_the_image_is_unverified_not_different(self):
        """A range found by code density can end before its data: the bytes are absent, which
        proves nothing either way (it was reported as a difference before)."""
        short = self.rom[:DATA - BASE]                                  # the image stops before .data
        result = strict(elf(TRUTH), short, SYMBOLS, {".data": DATA})
        self.assertTrue(result.text_equal)
        self.assertEqual((result.verdict, result.sections[0]["status"], result.sections[0]["held"]),
                         ("assumed", "unverified", 0))
        self.assertIn("holds 0 of its", " ".join(result.notes))

    def test_extra_code_after_the_owned_size_is_a_mismatch(self):
        self.assertEqual(strict(elf(TRUTH), self.rom, SYMBOLS, {".data": DATA}, size=0x18).verdict, "mismatch")

    def test_candidate_ldsym_marker_is_not_strict_proof_authority(self):
        with tempfile.TemporaryDirectory() as name:
            source = Path(name) / "candidate.c"
            source.write_text("/* LDSYM goodFunction=0x80001000 */\nvoid f(void) {}\n")
            spec = BuildSpec("kmc+builtin", [], section_bases={".data": DATA})
            outcome = Outcome("f", TEXT, 0x20, spec, proof_required=True)
            target = Target(self.rom, BASE, (0, len(self.rom)))
            _compare_at(outcome, elf(TRUTH), source, spec, target, TEXT, 0x20,
                        SymbolMap({"gRight": 0x80100000}), False, None,
                        {".data": 16}, False, True, None)
        self.assertTrue(outcome.exploratory_exact)
        self.assertFalse(outcome.exact)
        self.assertEqual(outcome.proof.verdict, "assumed")
        self.assertEqual([item.name for item in outcome.proof.assumptions], ["goodFunction"])


class LedgerTest(unittest.TestCase):
    def test_cross_unit_contradiction_and_proposals(self):
        def verdict(address):
            return verify.Verdict("assumed", True, assumptions=[verify.Assumption("symbol", "gShared", address)])
        report = verify.reconcile_assumptions([
            {"unit": "a.c", "verdict": verdict(0x80100000)}, {"unit": "b.c", "verdict": verdict(0x80100010)},
            {"unit": "c.c", "verdict": verify.Verdict("assumed", True, assumptions=[
                verify.Assumption("symbol", "gOnly", 0x80200000)])}])
        self.assertEqual(set(report["contradictions"]), {"gShared"})
        self.assertEqual(report["proposed_symbols"], ["gOnly = 0x80200000;"])

    def test_jump_targets_outside(self):
        import struct
        code = struct.pack(">4I", 0x08000408, 0, 0x08000101, 0)      # j 0x80001020 ; nop ; j 0x80000404 ; nop
        self.assertEqual(verify.jump_targets_outside(code, 0x80000400), [(0x80000400, 0x80001020)])


if __name__ == "__main__":
    unittest.main()


class CorroboratedSectionTest(unittest.TestCase):
    """A donor unit's own section: its contents at the derived address confirm that address."""

    def test_contents_found_at_the_derived_address_confirm_it(self):
        from n64dk.match import transplant
        rom = rom_from(elf(TRUTH), {".data": DATA})
        symbols = {"goodFunction": SYMBOLS["goodFunction"], "gRight": SYMBOLS["gRight"]}
        first = strict(elf(TRUTH), rom, symbols)
        self.assertEqual(first.verdict, "assumed")
        confirmed = transplant.corroborated_sections(first, elf(TRUTH))
        self.assertEqual(confirmed, {".data": DATA})
        self.assertEqual(strict(elf(TRUTH), rom, symbols, confirmed).verdict, "exact")

    def test_a_section_of_zeros_confirms_nothing(self):
        from n64dk.match import transplant
        rom = rom_from(elf(SHORTDATA), {".data": DATA})
        first = strict(elf(SHORTDATA), rom, {"goodFunction": SYMBOLS["goodFunction"], "gRight": SYMBOLS["gRight"]})
        self.assertTrue(first.text_equal)
        self.assertEqual(transplant.corroborated_sections(first, elf(SHORTDATA)), {})

    def test_different_code_confirms_nothing(self):
        from n64dk.match import transplant
        rom = rom_from(elf(TRUTH), {".data": DATA})
        first = strict(elf(WRONG), rom, {})
        if not first.text_equal:
            self.assertEqual(transplant.corroborated_sections(first, elf(WRONG)), {})

    def test_fixed_addresses(self):
        from n64dk.match import transplant
        self.assertTrue(transplant.fixed_address(0x80000308))
        self.assertTrue(transplant.fixed_address(0xA4400010))
        self.assertFalse(transplant.fixed_address(0x80000400))
        self.assertFalse(transplant.fixed_address(0x8009D030))


class ConfirmSectionsTest(unittest.TestCase):
    """One helper applies the content rule for resolved C, receipt issue and receipt replay."""

    def test_a_confirmed_base_makes_the_comparison_exact_and_is_reported(self):
        from n64dk.match import transplant
        rom = rom_from(elf(TRUTH), {".data": DATA})
        symbols = {"goodFunction": SYMBOLS["goodFunction"], "gRight": SYMBOLS["gRight"]}
        verdict, confirmed = transplant.confirm_sections(lambda bases: strict(elf(TRUTH), rom, symbols, bases),
                                                         elf(TRUTH), {})
        self.assertEqual((verdict.verdict, confirmed), ("exact", {".data": DATA}))

    def test_a_given_base_is_never_replaced_and_nothing_is_reported(self):
        from n64dk.match import transplant
        rom = rom_from(elf(TRUTH), {".data": DATA})
        symbols = {"goodFunction": SYMBOLS["goodFunction"], "gRight": SYMBOLS["gRight"]}
        verdict, confirmed = transplant.confirm_sections(lambda bases: strict(elf(TRUTH), rom, symbols, bases),
                                                         elf(TRUTH), {".data": DATA})
        self.assertEqual((verdict.verdict, confirmed), ("exact", {}))

    def test_a_section_of_zeros_stays_an_assumption(self):
        from n64dk.match import transplant
        rom = rom_from(elf(SHORTDATA), {".data": DATA})
        symbols = {"goodFunction": SYMBOLS["goodFunction"], "gRight": SYMBOLS["gRight"]}
        verdict, confirmed = transplant.confirm_sections(lambda bases: strict(elf(SHORTDATA), rom, symbols, bases),
                                                         elf(SHORTDATA), {})
        self.assertEqual(confirmed, {})
        self.assertNotEqual(verdict.verdict, "exact")
