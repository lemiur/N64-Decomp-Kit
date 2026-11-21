import json
import tempfile
import unittest
from pathlib import Path

from n64dk.commands import batch


class SlugTest(unittest.TestCase):
    def test_names_are_directory_safe_and_unique(self):
        taken: set[str] = set()
        self.assertEqual(batch.title_slug("JEOPARDY!", "abcdef0123", taken), "jeopardy")
        self.assertEqual(batch.title_slug("Jeopardy", "1234567890", taken), "jeopardy_12345678")
        self.assertEqual(batch.title_slug("", "ffff", taken), "rom")
        self.assertEqual(batch.title_slug(None, "ffff0000", taken), "rom_ffff0000")
        self.assertEqual(len(taken), 4)


class TallyTest(unittest.TestCase):
    IMAGES = {"main": {"functions": 100, "donor_verified": 10,
                       "resolved": {"exact": 30, "assumed": 2, "near": 8, "open": 40, "does-not-build": 5}},
              "seg_1": {"functions": 20, "donor_verified": 0, "resolved": {"exact": 4}}}

    def test_identical_is_donors_plus_exact_plus_assumed_over_every_image(self):
        found = batch.tally(self.IMAGES)
        self.assertEqual((found["functions"], found["donors"], found["exact"], found["assumed"]), (120, 10, 34, 2))
        self.assertEqual(found["identical"], 46)
        self.assertEqual(found["does_not_build"], 5)
        self.assertAlmostEqual(found["share"], 46 / 120, places=4)

    def test_assembly_is_accounted_for_apart_from_what_was_matched_from_c(self):
        found = batch.tally({"main": {"functions": 50, "donor_verified": 5, "donor_assumed": 3,
                                      "resolved": {"exact": 10, "assumed": 2, "assembly": 4, "open": 26}}})
        self.assertEqual((found["assumed"], found["identical"], found["assembly"], found["accounted"]), (5, 20, 4, 24))
        self.assertAlmostEqual(found["share"], 0.4, places=4)
        self.assertAlmostEqual(found["accounted_share"], 0.48, places=4)

    def test_an_image_without_results_counts_its_functions_only(self):
        self.assertEqual(batch.tally({"main": {"functions": 7}})["identical"], 0)
        self.assertEqual(batch.tally({})["share"], 0.0)

    def test_no_store_means_no_stored_tally(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(batch.stored_tally(Path(tmp), self.IMAGES))


class TimeLimitTest(unittest.TestCase):
    def test_a_step_that_does_not_finish_is_stopped_and_the_alarm_is_cleared(self):
        import signal
        import time
        if not hasattr(signal, "SIGALRM"):
            self.skipTest("no alarm signal on this platform")
        with self.assertRaises(TimeoutError):
            with batch.time_limit(1, "too slow"):
                time.sleep(5)
        self.assertEqual(signal.alarm(0), 0)
        with batch.time_limit(0, "no limit"):
            pass


class InputTest(unittest.TestCase):
    def test_a_donors_file_that_is_not_there_stops_the_batch_before_any_title(self):
        import argparse
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(donors=str(Path(tmp) / "missing.toml"), rom=[], out=tmp)
            with self.assertRaises(SystemExit) as raised:
                batch.decompile_batch(args)
            self.assertIn("no such file", str(raised.exception))


class DiagnosisTest(unittest.TestCase):
    def test_every_function_gets_one_place_and_unmatched_ones_a_kind_of_reason(self):
        from n64dk.match import resolve as resolvemod
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            store = resolvemod.CaseStore(out / "state")
            rows = [(0x80001000, 0x40, "exact"), (0x80001040, 0x40, "assembly"), (0x80001080, 0x40, "near"),
                    (0x800010C0, 0x40, "open"), (0x80001100, 0x200, "open"), (0x80001300, 0x40, "does-not-build"),
                    (0x80001340, 0x40, "blocked-boundary"), (0x80002000, 0x40, "open"), (0x80003000, 0x40, "open")]
            for vram, size, status in rows:
                store.put(resolvemod.Case(resolvemod.case_id("main", vram, size), "main", f"func_{vram:08X}", vram,
                                          size, "t", status=status))
            store.close()
            (out / "main").mkdir()
            (out / "main" / "stage_donors.json").write_text(json.dumps(
                {"verified": [{"function_vrams": [0x80003000, 0x80004000]}], "assumed": []}))
            (out / "report.json").write_text(json.dumps({"images": {"main": {"resolve": {"unknown_toolchain": {
                "stretches": [{"start": "0x80002000", "end": "0x80002FFC"}]}}}}}))
            found = batch.diagnosis(out, {"main": {}})
        self.assertEqual(found["matched from C"], 3)                 # one exact, two functions from a donor unit
        self.assertEqual(found["hand-written, kept as assembly"], 1)
        self.assertEqual([found[reason] for reason in batch.REASONS], [1, 1, 1, 1, 1, 1])
        text = batch.markdown({"titles": [{"title": "a", "seconds": 1.0, "final": batch.tally({}), "diagnosis": found}],
                               "passes": 1, "seconds": 1, "total": batch._report([], 1, 0.0, Path("/x"))["total"]})
        self.assertIn("## Where every function ended", text)
        self.assertIn("compiler not in the catalogue", text)

    def test_no_store_no_diagnosis(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(batch.diagnosis(Path(tmp), {"main": {}}))


class ReportTest(unittest.TestCase):
    def test_totals_and_table(self):
        titles = [{"title": "a", "seconds": 10.0, "final": batch.tally(TallyTest.IMAGES)},
                  {"title": "b", "seconds": 1.0, "error": "pass 1: no code segment"}]
        report = batch._report(titles, 2, 0.0, Path("/x/library"))
        self.assertEqual((report["total"]["functions"], report["total"]["identical"]), (120, 46))
        text = batch.markdown(report)
        self.assertIn("| a | 120 | 10 | 34 | 2 | 46 | 38.3% | 0 | 38.3% | 10 |", text)
        self.assertIn("failed: pass 1: no code segment", text)
        json.dumps(report, default=str)


class SourceTreeTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        self.out = Path(tempfile.mkdtemp(prefix="n64dk-test-batch-"))
        self.row = {"title": "one", "project": str(self.out / "one"), "run": str(self.out / "one.run")}

    def importer(self, tree, project, where, recipes, toolchain=None, jobs=4):
        self.asked.append((tree.name, recipes))
        if tree.name == "other":                             # a tree of some other title
            return {"source": str(tree), "donors": None, "reason": "fewer than 8 functions", "units_in_tree": 5}
        where.mkdir(parents=True, exist_ok=True)
        (where / "donors.toml").write_text("[donor.tree_mine]\nroot = \"x\"\n")
        return {"source": str(tree), "donors": str(where / "donors.toml"), "units_in_tree": 9, "compiled": 8,
                "target_functions_found": 40, "macros": ["RELEASE"]}

    def test_a_tree_that_gives_the_title_functions_joins_its_donors_and_the_others_are_reported(self):
        from pathlib import Path
        self.asked = []
        base = self.out / "base.toml"
        base.write_text("[donor.sdk]\nroot = \"y\"\n")
        again = batch.offer_trees(self.row, [Path("/t/mine"), Path("/t/other")], self.out, str(base),
                                  importer=self.importer, recipes_of=lambda _run: ["gcc -O2"])
        self.assertTrue(again)
        self.assertEqual(self.asked, [("mine", ["gcc -O2"]), ("other", ["gcc -O2"])])
        merged = Path(self.row["donors"]).read_text()
        self.assertLess(merged.index("[donor.sdk]"), merged.index("[donor.tree_mine]"))
        self.assertEqual([bool(item["donors"]) for item in self.row["source_trees"]], [True, False])
        text = batch.markdown({"titles": [{**self.row, "seconds": 1.0, "final": batch.tally({})}], "passes": 1,
                               "seconds": 1, "total": batch._shares({"functions": 0, "donors": 0, "exact": 0, "assumed": 0})})
        self.assertIn("| one | mine | 9 | 8 | 40 | RELEASE | donor of this title |", text)
        self.assertIn("| one | other | 5 | 0 | 0 | - | fewer than 8 functions |", text)

    def test_no_tree_gives_anything_and_the_title_keeps_its_donors(self):
        from pathlib import Path
        self.asked = []
        self.assertFalse(batch.offer_trees(self.row, [Path("/t/other")], self.out, None,
                                           importer=self.importer, recipes_of=lambda _run: ["gcc -O2"]))
        self.assertNotIn("donors", self.row)

    def test_a_run_that_matched_nothing_names_no_recipe_and_no_tree_is_built(self):
        from pathlib import Path
        self.asked = []
        self.assertFalse(batch.offer_trees(self.row, [Path("/t/mine")], self.out, None,
                                           importer=self.importer, recipes_of=lambda _run: []))
        self.assertEqual(self.asked, [])
        self.assertIn("no candidate recipe", self.row["source_trees"][0]["reason"])


class EverythingTest(unittest.TestCase):
    def test_everything_is_the_batch_at_the_kitchen_sink_effort_with_export(self):
        import argparse
        from unittest import mock
        parser = argparse.ArgumentParser()
        batch.decompile_everything.arguments(parser)
        args = parser.parse_args(["a.z64", "--out", "/nonexistent/out"])
        self.assertIsNone(args.effort)
        with mock.patch.object(batch, "decompile_batch", side_effect=lambda a: (a.effort, a.export)) as run:
            self.assertEqual(batch.decompile_everything(args), ("exhaustive", True))
            run.assert_called_once()
        args = parser.parse_args(["a.z64", "--out", "/nonexistent/out", "--effort", "deep"])
        with mock.patch.object(batch, "decompile_batch", side_effect=lambda a: a.effort):
            self.assertEqual(batch.decompile_everything(args), "deep")


class BudgetTest(unittest.TestCase):
    def test_an_effort_mode_sets_the_budget_unless_the_batch_gives_one(self):
        import argparse
        self.assertEqual(batch.budget_argv(argparse.Namespace(budget=None, effort=None)), ["--budget", "100"])
        self.assertEqual(batch.budget_argv(argparse.Namespace(budget=None, effort="exhaustive")), [])
        self.assertEqual(batch.budget_argv(argparse.Namespace(budget=50, effort="exhaustive")), ["--budget", "50"])
        parser = argparse.ArgumentParser()
        batch.decompile_everything.arguments(parser)
        self.assertIsNone(parser.parse_args(["a.z64", "--out", "/nonexistent/out"]).budget)


class ExportTitleTest(unittest.TestCase):
    def test_a_failed_export_is_recorded_and_never_raised(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            run, tree = Path(tmp) / "t.run", Path(tmp) / "t.export"
            with mock.patch("n64dk.commands.export.export", side_effect=ValueError("broken")), \
                    mock.patch("n64dk.commands.runstatus.status", return_value={"ok": 1}):
                record, status = batch.export_title(Path(tmp), run, tree, None)
            self.assertEqual((record["out"], status), (str(tree), {"ok": 1}))
            self.assertIn("ValueError: broken", record["error"])
            with mock.patch("n64dk.commands.export.export",
                            return_value={"functions": 3, "matched": 1, "from_c": 1, "out": "x"}) as export, \
                    mock.patch("n64dk.commands.runstatus.status", side_effect=OSError("gone")):
                record, status = batch.export_title(Path(tmp), run, tree, "/tc")
            self.assertEqual(record, {"out": str(tree), "functions": 3, "matched": 1, "from_c": 1})
            self.assertEqual(export.call_args.kwargs["toolchain"], "/tc")
            self.assertIn("OSError", status["error"])


if __name__ == "__main__":
    unittest.main()
