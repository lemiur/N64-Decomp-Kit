"""``decompile-status`` with what a run's ``report.json`` adds: the discovery verdict and its
denominator warning, the small-data entry and per-image totals.

Synthetic run directories only: a temporary case store, a rounds log and a hand-written
``report.json``.  No toolchain, no ROM, no machine lock."""
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from n64dk.commands import runstatus
from n64dk.match import resolve as resolvemod
from n64dk.match import solver

OLD_KEYS = {"run", "session", "mode", "rounds", "stops", "credit", "remaining_by_cause"}
NEW_KEYS = {"report", "discovery", "small_data", "image_totals"}

DENOMINATOR = ("the function catalogue is incomplete even after repair: every count and percentage in "
               "this report is over the catalogue, not over the title's code")
GP_WARNING = ("the project declares _gp = 0x80099990; the code that sets $gp says 0x80012340: "
              "the project's value is used")


def _case(name, vram, *, image="main", status="open", score=10, top="register.permutation", size=0x40, **extra):
    return resolvemod.Case(resolvemod.case_id(image, vram, size), image, name, vram, size, "t" + name,
                           status=status, top_cause=top, best_score=score,
                           best_digest=("d" + name) if score is not None else None, recipe="ido5.3 -O2", **extra)


def _report() -> dict:
    """The shape ``decompile-run`` writes (``n64dk.commands.run``), two images."""
    return {
        "project": "<project>", "status": "incomplete", "seconds": 12.5, "failed_stages": {"aux": ["ledger"]},
        "discovery": {
            "verdict_before": "incomplete", "verdict": "incomplete", "rom_reasons": [], "placed": [],
            "warning": DENOMINATOR,
            "images": {
                "main": {"verdict": "incomplete", "coverage": 0.91, "functions": 100,
                         "reasons": ["12 call targets outside the catalogue"],
                         "repaired_functions": 7, "repaired_bytes": 640, "verdict_after": "incomplete",
                         "coverage_after": 0.95, "functions_after": 107,
                         "reasons_after": ["3 call targets outside the catalogue"]},
                "aux": {"verdict": "complete", "coverage": 1.0, "functions": 4, "verdict_after": "complete",
                        "coverage_after": 1.0, "functions_after": 4, "reasons_after": []}}},
        "small_data": {"value": "0x80012340", "confidence": "high", "sites": 2, "project_gp": "0x80099990",
                       "reason": "set to 0x80012340 at 2 site(s); 40 GP-relative uses in 9 function(s)",
                       "warning": GP_WARNING},
        "images": {
            "main": {"functions": 107,
                     "resolve": {"statuses": {"exact": 2, "open": 1}, "exact_bytes": 128},
                     "queue": {"strict_exact": 2, "assumed": 1, "assembly": 0},
                     "donors": {"verified": [{"functions": ["a", "b"]}, {"functions": ["c"]}], "assumed": []},
                     "ledger": {"totals": {"extent": "0x80000000..0x80001000", "extent_bytes": 4096,
                                           "bytes": {"c": 128}, "sum_equals_extent": True, "code_bytes": 4000,
                                           "c_bytes": 128, "asm_bytes": 3872, "functions": 107,
                                           "functions_by_status": {"c": 2, "asm": 105},
                                           "functions_matched_c": 2, "functions_asm": 105},
                                "violations": {"overlap": 1}, "file": "<run>/main/ownership.tsv"},
                     "solver": {"mode": "normal", "coverage": {"functions": 107, "attempted": 100,
                                                               "not_attempted": 7, "unexplained": 0,
                                                               "reasons": {"too large": 7}}}},
            "aux": {"functions": 4, "resolve": {"statuses": {"exact": 4}, "exact_bytes": 64},
                    "queue": {"strict_exact": 4, "assumed": 0}, "ledger": {"error": "boom"}}}}


class RunStatusReportTest(unittest.TestCase):
    def setUp(self):
        self.run = Path(tempfile.mkdtemp(prefix="n64dk-runstatus-test-"))
        self.addCleanup(shutil.rmtree, self.run, True)
        store = resolvemod.CaseStore(self.run / "state")
        try:
            for case in (_case("a", 0x80000000, status="exact", score=0, strict={"verdict": "exact"}),
                         _case("b", 0x80000040),
                         _case("c", 0x80100000, image="aux", status="exact", score=0, strict={"verdict": "exact"})):
                store.put(case)
        finally:
            store.close()
        log = solver.RoundsLog(self.run / "rounds.jsonl")
        log.append({"kind": "round", "session": "s", "image": "main", "round": 1, "sieve": "sweep",
                    "techniques": {"sweep": {"cases": 2, "builds": 40, "exact": 1}}, "cascade": {},
                    "gains": {"new_exact": 1, "awaiting_declarations": 0, "evidence": 2}})
        log.append({"kind": "stop", "session": "s", "image": "main", "after_round": 1, "reason": "round limit"})

    def _write(self, report) -> None:
        text = report if isinstance(report, str) else json.dumps(report)
        (self.run / "report.json").write_text(text)

    def test_without_a_report_only_empty_keys_are_added(self):
        result = runstatus.status(self.run)
        self.assertTrue(OLD_KEYS <= set(result))
        self.assertEqual(set(result) - OLD_KEYS, NEW_KEYS)
        self.assertEqual((result["report"], result["discovery"], result["small_data"], result["image_totals"]),
                         (None, None, None, {}))
        text = runstatus.render(result)
        self.assertIn("Round 1 [main]: sweep +1 exact", text)
        for absent in ("report.json", "Discovery", "Small data", "Per image"):
            self.assertNotIn(absent, text)

    def test_json_carries_verdict_warning_small_data_and_totals(self):
        self._write(_report())
        result = runstatus.status(self.run)
        self.assertEqual(set(result) - OLD_KEYS, NEW_KEYS)
        self.assertEqual(result["report"], {"file": str(self.run / "report.json"), "status": "incomplete",
                                            "seconds": 12.5, "failed_stages": {"aux": ["ledger"]}})
        found = result["discovery"]
        self.assertEqual((found["verdict"], found["verdict_before"], found["warning"]),
                         ("incomplete", "incomplete", DENOMINATOR))
        self.assertEqual(found["images"]["main"], {
            "verdict": "incomplete", "verdict_after": "incomplete", "coverage": 0.91, "coverage_after": 0.95,
            "functions_after": 107, "repaired_functions": 7,
            "reasons_after": ["3 call targets outside the catalogue"]})
        self.assertEqual(found["images"]["aux"]["reasons_after"], [])
        self.assertEqual(result["small_data"], _report()["small_data"])
        self.assertEqual(result["image_totals"]["main"], {
            "functions": 107, "resolved": {"exact": 2, "open": 1}, "exact_bytes": 128, "queued_strict_exact": 2,
            "awaiting_symbols": 1, "assembly": 0, "donor_verified": 3, "errors": [],
            "ledger": {"extent_bytes": 4096, "code_bytes": 4000, "c_bytes": 128, "asm_bytes": 3872,
                       "functions": 107, "functions_matched_c": 2, "functions_asm": 105,
                       "sum_equals_extent": True, "violations": {"overlap": 1}},
            "coverage": {"functions": 107, "attempted": 100, "not_attempted": 7, "unexplained": 0}})
        aux = result["image_totals"]["aux"]
        self.assertEqual((aux["functions"], aux["errors"], aux["donor_verified"]), (4, ["ledger"], 0))
        self.assertNotIn("ledger", aux)
        self.assertNotIn("coverage", aux)
        # the case store is still read live, next to the report's figures
        self.assertEqual(result["credit"]["exact_c"], 2)
        json.dumps(result)

    def test_text_shows_the_report_before_the_credit_line(self):
        self._write(_report())
        text = runstatus.render(runstatus.status(self.run))
        for expected in (
                "report.json (last finished invocation): incomplete, failed stages {'aux': ['ledger']}",
                "Discovery: catalogue incomplete\n",
                "  main: incomplete, coverage 0.95, 107 functions (+7 repaired)",
                "    - 3 call targets outside the catalogue",
                "  aux: complete, coverage 1.0, 4 functions\n",
                f"  WARNING: {DENOMINATOR}",
                "Small data: _gp 0x80012340 (confidence high, 2 setting sites): set to 0x80012340 at 2 site(s)",
                "  project declares _gp = 0x80099990",
                f"  WARNING: {GP_WARNING}",
                "Per image:",
                "  main: 107 functions; resolved {'exact': 2, 'open': 1}; exact bytes 128; queued strictly exact 2; "
                "awaiting symbols 1; donor-verified 3",
                "    ledger: C 128 bytes, assembly 3872 bytes of 4000 code bytes; functions matched C 2, "
                "assembly 105; violations {'overlap': 1}",
                "    coverage: 100 of 107 attempted, 7 not attempted, 0 unexplained",
                "  aux: 4 functions; resolved {'exact': 4}; exact bytes 64; queued strictly exact 4; "
                "awaiting symbols 0; donor-verified 0",
                "    stage errors: ledger",
                "Round 1 [main]: sweep +1 exact",
                "remaining queue by primary cause:"):
            self.assertIn(expected, text)
        self.assertNotIn("before repair", text)              # same verdict before and after
        self.assertLess(text.index("Stopped [main]"), text.index("Discovery:"))
        self.assertLess(text.index(DENOMINATOR), text.index("exact C "))

    def test_image_filter_limits_the_report_views(self):
        self._write(_report())
        result = runstatus.status(self.run, "aux")
        self.assertEqual(set(result["image_totals"]), {"aux"})
        self.assertEqual(set(result["discovery"]["images"]), {"aux"})
        self.assertEqual(result["report"]["failed_stages"], {"aux": ["ledger"]})
        self.assertEqual(result["discovery"]["verdict"], "incomplete")    # the run's verdict, over every image
        self.assertEqual(runstatus.status(self.run, "main")["report"]["failed_stages"], {})

    def test_reading_the_report_writes_nothing(self):
        self._write(_report())

        def listing():
            return sorted((str(p.relative_to(self.run)), p.stat().st_mtime_ns) for p in self.run.rglob("*"))
        before = listing()
        runstatus.render(runstatus.status(self.run))
        self.assertEqual(listing(), before)

    def test_a_malformed_report_is_reported_not_raised(self):
        for text, error in (("{not json", "JSONDecodeError"), ("[1, 2]", "does not hold an object")):
            self._write(text)
            result = runstatus.status(self.run)
            self.assertIn(error, result["report"]["error"])
            self.assertEqual((result["discovery"], result["small_data"], result["image_totals"]), (None, None, {}))
            rendered = runstatus.render(result)
            self.assertIn("report.json unreadable: ", rendered)
            self.assertIn("remaining queue by primary cause:", rendered)

    def test_failed_stages_and_an_absent_gp(self):
        self._write({"status": "complete", "failed_stages": {}, "images": {},
                     "discovery": {"error": "OSError: no catalogue"},
                     "small_data": {"value": None, "confidence": None, "reason": "no code sets $gp", "sites": 0}})
        text = runstatus.render(runstatus.status(self.run))
        self.assertIn("report.json (last finished invocation): complete\n", text)
        self.assertIn("Discovery: failed: OSError: no catalogue", text)
        self.assertIn("Small data: no _gp established (confidence none, 0 setting sites): no code sets $gp", text)
        self.assertNotIn("Per image:", text)
        self._write({"status": "complete", "small_data": {"error": "ValueError: bad word"}})
        self.assertIn("Small data: failed: ValueError: bad word", runstatus.render(runstatus.status(self.run)))

    def test_an_older_report_without_the_newer_entries(self):
        self._write({"project": "<project>", "images": {"main": {"functions": 5}}})
        result = runstatus.status(self.run)
        self.assertEqual((result["discovery"], result["small_data"]), (None, None))
        self.assertEqual(result["report"]["status"], None)
        self.assertEqual(result["image_totals"], {"main": {
            "functions": 5, "resolved": None, "exact_bytes": None, "queued_strict_exact": None,
            "awaiting_symbols": None, "assembly": None, "donor_verified": 0, "errors": []}})
        text = runstatus.render(result)
        self.assertIn("report.json (last finished invocation): -", text)
        self.assertIn("  main: 5 functions;", text)

    def test_command_json_output_has_the_new_keys(self):
        from n64dk import cli
        self._write(_report())
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            cli.main(["--format", "json", "decompile-status", str(self.run)])
        printed = json.loads(buffer.getvalue())
        self.assertTrue((OLD_KEYS | NEW_KEYS) <= set(printed))
        self.assertEqual(printed["discovery"]["warning"], DENOMINATOR)
        self.assertEqual(printed["small_data"]["project_gp"], "0x80099990")
        self.assertEqual(printed["image_totals"]["main"]["exact_bytes"], 128)


if __name__ == "__main__":
    unittest.main()
