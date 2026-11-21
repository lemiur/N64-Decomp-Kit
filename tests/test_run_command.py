import shutil
import atexit
import argparse
import tempfile
import unittest
from pathlib import Path

from n64dk.commands import run


def _scratch(prefix: str = "n64dk-test-") -> str:
    """A scratch directory removed when the test process exits."""
    path = tempfile.mkdtemp(prefix=prefix)
    atexit.register(shutil.rmtree, path, True)
    return path



class RunCommandTest(unittest.TestCase):
    def test_project_without_images_is_rejected_with_guidance(self):
        project = Path(_scratch())
        (project / "n64dk.toml").write_text("[toolchains]\nroot = \"/nonexistent\"\n")
        args = argparse.Namespace(project=str(project), out=_scratch(), image=[], all_images=True)
        with self.assertRaisesRegex(SystemExit, r"\[images\.<name>\]"):
            run.decompile_run(args)

    def test_settings_merge_defaults_and_per_image_overrides(self):
        project = Path(_scratch())
        (project / "n64dk.toml").write_text(
            '[run]\npipeline = "ido5.3"\nflags = "-O2"\n[run.images.overlay]\nflags = "-O1"\n'
            'recipes = { "m.tsv" = "ido7.1" }\n')
        self.assertEqual(run._settings(project, "main"), {"pipeline": "ido5.3", "flags": "-O2"})
        self.assertEqual(run._settings(project, "overlay"),
                         {"pipeline": "ido5.3", "flags": "-O1", "recipes": {"m.tsv": "ido7.1"}})

    def test_stage_results_are_checkpointed(self):
        folder = Path(_scratch())
        calls = []
        first, cached = run._stage(folder, "x", lambda: calls.append(1) or {"n": 1}, False)
        again, cached_again = run._stage(folder, "x", lambda: calls.append(1) or {"n": 2}, False)
        redo, _ = run._stage(folder, "x", lambda: calls.append(1) or {"n": 3}, True)
        self.assertEqual((first, cached, again, cached_again, redo, len(calls)),
                         ({"n": 1}, False, {"n": 1}, True, {"n": 3}, 2))

    def test_stage_checkpoint_is_invalidated_by_input_identity(self):
        folder = Path(_scratch())
        calls = []
        first, _ = run._stage(folder, "x", lambda: calls.append(1) or {"n": 1}, False, "image-a")
        cached, was_cached = run._stage(folder, "x", lambda: calls.append(2) or {"n": 2}, False, "image-a")
        changed, changed_cached = run._stage(folder, "x", lambda: calls.append(3) or {"n": 3}, False, "image-b")
        self.assertEqual((first, cached, was_cached, changed, changed_cached, calls),
                         ({"n": 1}, {"n": 1}, True, {"n": 3}, False, [1, 3]))

    def test_generated_state_inside_source_tree_is_rejected(self):
        project = Path(_scratch())
        (project / "n64dk.toml").write_text("[toolchains]\nroot='/none'\n")
        args = argparse.Namespace(project=str(project), out=str(project / "generated"), image=[], all_images=True,
                                  allow_project_output=False)
        with self.assertRaisesRegex(SystemExit, "source tree"):
            run.decompile_run(args)

    def test_report_markdown_lists_every_image_and_remaining_causes(self):
        report = {"project": "/p", "seconds": 1.5, "images": {"main": {
            "functions": 10, "donors": {"verified": [{"functions": ["a", "b"]}]},
            "resolve": {"statuses": {"exact": 3}, "exact_bytes": 64},
            "queue": {"exported": 3, "strict_exact": 3},
            "hard_queue": [{"cause": "type.width", "functions": 2, "bytes": 80}],
            "ledger": {"error": "boom"}}}}
        text = run._markdown(report)
        self.assertIn("## main", text)
        self.assertIn("recovered from donors (verified): 2", text)
        self.assertIn("| type.width | 2 | 80 |", text)
        self.assertIn("stages with errors: ['ledger']", text)


class RegistrationTest(unittest.TestCase):
    """The command line must reach ``decompile_run`` with every option it documents."""

    def test_command_is_registered_with_its_options(self):
        import contextlib
        import io
        from n64dk import cli
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), self.assertRaises(SystemExit) as stop:
            cli.main(["decompile-run", "--help"])
        self.assertEqual(stop.exception.code, 0)
        text = buffer.getvalue()
        for option in ("--project", "--out", "--donors", "--workers", "--search-ceiling", "--refine-command",
                       "--refine-rounds", "--max-size", "--budget"):
            self.assertIn(option, text)

    def test_resolve_hard_documents_the_refine_and_padding_options(self):
        import contextlib
        import io
        from n64dk import cli
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), self.assertRaises(SystemExit):
            cli.main(["resolve-hard", "--help"])
        for option in ("--refine-command", "--refine-names", "--keep-padding", "--workers", "--search-ceiling"):
            self.assertIn(option, buffer.getvalue())


class DeclareAssumedTest(unittest.TestCase):
    def test_only_self_describing_addresses_are_declared_and_the_verifier_decides(self):
        import argparse
        import shutil
        import tempfile
        from n64dk.commands import run as runmod
        from n64dk.match import resolve as resolvemod
        root = Path(tempfile.mkdtemp(prefix="n64dk-declare-test-"))
        self.addCleanup(shutil.rmtree, root, True)
        store = resolvemod.CaseStore(root / "state")
        for index, assumption in enumerate([
                {"kind": "symbol", "name": "D_80046354", "address": "0x80046354"},
                {"kind": "symbol", "name": "gPlayer", "address": "0x80050000"},          # a claim, not a restatement
                {"kind": "symbol", "name": "D_80046354", "address": "0x80046358"},      # name and address disagree
                {"kind": "section", "name": ".rodata", "address": "0x80060000"}]):
            case = resolvemod.Case(resolvemod.case_id("main", 0x80001000 + 0x40 * index, 0x20), "main", f"f{index}",
                                   0x80001000 + 0x40 * index, 0x20, "digest", status="assumed",
                                   strict={"verdict": "assumed", "assumptions": [assumption]})
            store.put(case)
        store.close()
        seen = {}

        def fake_resolve_hard(options):
            seen["symbols"], seen["only"], seen["refine"] = options.symbols, options.only_statuses, options.refine
            return {"statuses": {"exact": 1, "assumed": 3}}

        def options(**overrides):
            return argparse.Namespace(symbols=["symbol_addrs.txt"], **overrides)
        result = runmod.declare_assumed(fake_resolve_hard, options, root / "state", "main", root / "main", {})
        declared = (root / "main" / "declared_symbols.txt").read_text()
        self.assertEqual(declared, "D_80046354 = 0x80046354;\n")
        self.assertEqual((result["declared"], result["waiting"], result["promoted"], result["still_assumed"]), (1, 4, 1, 3))
        self.assertEqual(seen["symbols"], ["symbol_addrs.txt", str(root / "main" / "declared_symbols.txt")])
        self.assertEqual((seen["only"], seen["refine"]), (("assumed",), None))

    def test_nothing_to_declare_runs_nothing(self):
        import shutil
        import tempfile
        from n64dk.commands import run as runmod
        root = Path(tempfile.mkdtemp(prefix="n64dk-declare-test-"))
        self.addCleanup(shutil.rmtree, root, True)
        result = runmod.declare_assumed(lambda options: self.fail("no pass expected"), None, root / "state", "main",
                                        root / "main", {})
        self.assertEqual(result, {"declared": 0, "promoted": 0, "waiting": 0})


if __name__ == "__main__":
    unittest.main()


class ResumeSweepTest(unittest.TestCase):
    def test_finished_cases_are_skipped_and_calibration_samples_are_not(self):
        import shutil
        import tempfile
        from pathlib import Path
        from n64dk.commands.run import finished_cases
        from n64dk.match import resolve as resolvemod
        root = Path(tempfile.mkdtemp(prefix="n64dk-resume-test-"))
        self.addCleanup(shutil.rmtree, root, True)
        store = resolvemod.CaseStore(root)
        try:
            for vram, stages, image in ((0x80001000, [{"stage": "drafts", "variants": []}], "main"),
                                        (0x80002000, [{"stage": "provisional", "recipe": "kmc -O2"}], "main"),
                                        (0x80003000, [], "seg")):
                store.put(resolvemod.Case(resolvemod.case_id(image, vram, 0x40), image, f"f{vram:x}", vram, 0x40,
                                          "t", status="open", stages=stages))
        finally:
            store.close()
        self.assertEqual(finished_cases(root, "main"), {0x80001000})
        self.assertEqual(finished_cases(root / "missing", "main"), set())


class ProjectGpTest(unittest.TestCase):
    def test_the_project_gp_comes_from_the_files_the_resolver_reads(self):
        project = Path(_scratch())
        (project / "match.txt").write_text("_gp = 0x80100000;\nfunc_a = 0x80000400;\n")
        (project / "run.txt").write_text("_gp = 0x80200000;\n")
        (project / "n64dk.toml").write_text('[match]\nsymbols = ["match.txt"]\n')
        self.assertEqual(run.project_gp(project, "main"), 0x80100000)       # [match] symbols, as the resolver
        (project / "n64dk.toml").write_text('[match]\nsymbols = ["match.txt"]\n[run]\nsymbols = ["run.txt"]\n')
        self.assertEqual(run.project_gp(project, "main"), 0x80200000)       # [run] symbols win, as in a run
        (project / "n64dk.toml").write_text('[run]\nsymbols = ["missing.txt"]\n')
        self.assertIsNone(run.project_gp(project, "main"))

    def test_a_run_without_a_value_leaves_no_gp_file_behind(self):
        out = Path(_scratch())
        self.assertEqual(run.write_gp_symbols(out, None), [])                 # nothing there: no error
        written = run.write_gp_symbols(out, "_gp = 0x80100000;\n")
        self.assertEqual(Path(written[0]).read_text(), "_gp = 0x80100000;\n")
        self.assertEqual(run.write_gp_symbols(out, None), [])                 # --small-data off afterwards
        self.assertFalse((out / "small_data" / "gp_symbols.txt").exists())


class SmallDataProvenanceTest(unittest.TestCase):
    """Functions verified against a _gp recovered from code are reported as such (strict verification
    itself is unchanged)."""

    def test_exact_cases_in_the_small_data_lanes_are_counted(self):
        from types import SimpleNamespace
        cases = [SimpleNamespace(status=status, recipe=recipe) for status, recipe in (
            ("exact", "kmc+asg -O2 -G8"), ("exact", "kmc+gas -O2 -G8"), ("exact", "kmc+as -O2"),
            ("assumed", "kmc+asg -O2 -G8"), ("open", "kmc+gas -O2 -G8"), ("exact", None))]
        self.assertEqual(run.small_data_lane_exact(cases), 2)

    def test_the_report_says_where_the_gp_came_from(self):
        recovered = {"value": "0x80100000", "confidence": "high", "reason": None, "sites": 2}
        run.note_small_data(recovered, 3, True)
        self.assertEqual((recovered["gp_source"], recovered["exact_small_data_lane"]), ("recovered", 3))
        self.assertIn("small_data/gp_symbols.txt", recovered["gp_note"])
        declared = {"value": "0x80100000", "confidence": "high", "project_gp": "0x80100000"}
        run.note_small_data(declared, 3, True)
        self.assertEqual(declared["gp_source"], "project")
        self.assertNotIn("gp_note", declared)
        unreadable = {"value": "0x80100000", "confidence": "high", "project_gp_error": "OSError: x"}
        run.note_small_data(unreadable, 1, True)
        self.assertEqual(unreadable["gp_source"], "unknown")
        low = {"value": "0x80100000", "confidence": "low"}             # not declared to the resolver
        run.note_small_data(low, 0, False)
        self.assertEqual((low["gp_source"], low["exact_small_data_lane"]), (None, 0))
        run.note_small_data(None, 1, True)                              # --small-data off: no entry, no error
        run.note_small_data({"error": "ValueError: x"}, 0, False)

    def test_the_markdown_report_names_the_gp_source(self):
        recovered = {"value": "0x80100000", "confidence": "high"}
        run.note_small_data(recovered, 3, True)
        text = run._markdown({"project": "/p", "images": {}, "small_data": recovered})
        self.assertIn("_gp 0x80100000, recovered from the code that sets $gp", text)
        self.assertIn("3 exact functions built in a small-data lane", text)
        declared = {"value": "0x80100000", "project_gp": "0x80100000", "warning": "the project declares _gp"}
        run.note_small_data(declared, 0, True)
        text = run._markdown({"project": "/p", "images": {}, "small_data": declared})
        self.assertIn("declared by the project", text)
        self.assertIn("WARNING: the project declares _gp", text)
        self.assertNotIn("Small data", run._markdown({"project": "/p", "images": {}}))


class HeadlineTest(unittest.TestCase):
    def test_observation_errors_are_reported_apart_from_failures(self):
        item = {"functions": 3, "solver": {"observation_error": "ValueError: x"}, "resolve": {"statuses": {}},
                "donors": {"error": "boom"}}
        head = run._headline(item)
        self.assertEqual(head["errors"], ["donors"])
        self.assertEqual(head["observation_errors"], ["solver"])


class SkipStageTest(unittest.TestCase):
    def test_calibration_can_be_skipped(self):
        # run.py consults "calibrate" in skip_stage; the parser must accept it (round 12 ablation finding)
        import argparse
        parser = argparse.ArgumentParser()
        run.decompile_run.arguments(parser)
        args = parser.parse_args(["--project", "/nonexistent", "--out", "/nonexistent/out", "--skip-stage", "calibrate"])
        self.assertEqual(args.skip_stage, ["calibrate"])
