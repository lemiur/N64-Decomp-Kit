"""Ablation switches of decompile-run (product gate Q15, ABLATION_REPORT.md "no off switch").

Each subsystem the report found without an off switch now has one (``--skip-stage NAME`` or
``N64DK_DISABLE_<NAME>=1``).  These tests hold the four promises: the default is unchanged, the
switch reaches the subsystem, the run records what it switched off, and a switched-off run can
never claim more credit (strict verification untouched, no inherited case store, attribution
flags a run that is not what its variant says).  Fakes only: no compiler, no ROM, no lock."""
import argparse
import contextlib
import io
import json
import os
import shutil
import struct
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from n64dk.commands import resolve as resolvecmd
from n64dk.commands import run
from n64dk.match import attribution as at
from n64dk.match import iterate
from n64dk.match import resolve as resolvemod
from n64dk.match import solver
from n64dk.match import transplant


def _scratch(test) -> Path:
    path = Path(tempfile.mkdtemp(prefix="n64dk-ablation-test-"))
    test.addCleanup(shutil.rmtree, path, True)
    return path


def _parser():
    parser = argparse.ArgumentParser()
    run.decompile_run.arguments(parser)
    return parser


def _clean_environ():
    """os.environ without any N64DK_DISABLE_* variable (the tests set their own)."""
    return {k: v for k, v in os.environ.items() if not k.startswith(run.DISABLE_PREFIX)}


BASE = 0x80001000
TARGET = struct.pack(">3I", 0x84820004, 0x03E00008, 0x00000000)     # lh v0,4(a0); jr ra; nop
WRONG = struct.pack(">3I", 0x94820004, 0x03E00008, 0x00000000)      # lhu
SOURCE = "s32 read(Thing *thing) {\n    u16 value;\n\n    value = thing->field;\n    return value;\n}\n"


def fake_evaluate(text):
    """Pretend compiler (as tests/test_resolve.py): exact when ``value`` is declared s16."""
    exact = "s16 value;" in text
    report = {"counts": {} if exact else {"instruction": 1},
              "entries": [] if exact else [{"compiled": "lhu $v0, 0x4($a0)", "retail": "lh $v0, 0x4($a0)"}]}
    outcome = {"proof": {"verdict": "exact"}} if exact else None
    return iterate.Evaluation(0 if exact else 3, exact, report, outcome=outcome), (TARGET if exact else WRONG)


class ParserAndEnvironmentTest(unittest.TestCase):
    def test_every_switch_is_a_skip_stage_choice_and_the_old_choices_remain(self):
        for name in ("discovery", "boundaries", "donors", "calibrate", "resolve", "ledger", *run.ABLATION_SWITCHES):
            args = _parser().parse_args(["--project", "p", "--out", "o", "--skip-stage", name])
            self.assertEqual(args.skip_stage, [name])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            _parser().parse_args(["--project", "p", "--out", "o", "--skip-stage", "no-such-subsystem"])
        # the subsystems ABLATION_REPORT.md lists without a switch (overlays and ledger excepted)
        for name in ("declare", "context", "draft-portfolio", "draft", "recipes", "cross-recipes", "search",
                     "frame-fit", "twin", "handasm", "rodata-settlement", "at-loads", "round-search",
                     "round-recipe-sweep", "round-refine", "refine", "cascade-neighbours"):
            self.assertIn(name, run.ABLATION_SWITCHES)

    def test_command_line_and_environment_are_merged_with_their_source(self):
        found = run.disabled_subsystems(["ledger", "search"], {"N64DK_DISABLE_DRAFT_PORTFOLIO": "1",
                                                               "N64DK_DISABLE_TWIN": "0",
                                                               "N64DK_DISABLE_SEARCH": "1", "PATH": "/bin"})
        self.assertEqual(found, {"draft-portfolio": "env N64DK_DISABLE_DRAFT_PORTFOLIO", "ledger": "--skip-stage",
                                 "search": "--skip-stage"})
        self.assertEqual(run.disabled_subsystems(None, {}), {})

    def test_a_mistyped_or_malformed_variable_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "no such ablation switch"):
            run.disabled_subsystems([], {"N64DK_DISABLE_SERACH": "1"})
        with self.assertRaisesRegex(SystemExit, "no such ablation switch"):
            run.disabled_subsystems([], {"N64DK_DISABLE_LEDGER": "1"})          # stages: --skip-stage only
        with self.assertRaisesRegex(SystemExit, "set it to 1"):
            run.disabled_subsystems([], {"N64DK_DISABLE_SEARCH": "yes"})


class PolicyAndOptionsTest(unittest.TestCase):
    def test_the_effort_mode_loses_only_what_a_switch_names(self):
        deep = solver.policy("deep")
        self.assertIs(run.ablated_policy(deep, []), deep)
        self.assertIs(run.ablated_policy(deep, ["ledger", "twin"]), deep)        # nothing of the mode's
        cut = run.ablated_policy(deep, ["search", "declare", "cascade-neighbours"])
        self.assertEqual(cut.techniques, ("recipe-sweep", "refine"))
        self.assertEqual(cut.cascade, ("recipes",))
        self.assertTrue(cut.refine)
        self.assertEqual((cut.name, cut.budget, cut.max_rounds), (deep.name, deep.budget, deep.max_rounds))
        solver.check_policy(cut)
        quiet = run.ablated_policy(deep, ["refine"])
        self.assertFalse(quiet.refine)
        self.assertNotIn("refine", quiet.techniques)
        self.assertEqual(run.ablated_policy(deep, ["round-refine"]).refine, True)
        self.assertNotIn("refine", run.ablated_policy(deep, ["round-refine"]).techniques)

    def _options(self, disabled=None):
        args = _parser().parse_args(["--project", "p", "--out", "o"])
        solver.apply_policy(args, solver.policy(None))
        if disabled is not None:
            args.disabled = disabled
        spec = types.SimpleNamespace(path="/rom.z64", vram_delta=None, delta_source="boot", catalogue="/c.txt",
                                     catalogue_shift=0, name="main")
        factory = run.resolve_options_for(args, {}, mode=solver.policy(None), project=Path("/p"), spec=spec,
                                          jobs=2, events=Path("/o/events.jsonl"), out=Path("/o"), manifests=[],
                                          recipes={}, recovered=set())
        return factory()

    def test_default_options_are_the_hard_coded_ones(self):
        options = self._options()
        self.assertEqual((options.context, options.no_draft, options.no_recipes, options.draft_portfolio,
                          options.disabled), ("auto", False, False, True, ()))

    def test_switches_reach_the_resolve_options_of_every_call(self):
        options = self._options(["context", "draft", "draft-portfolio", "recipes", "search"])
        self.assertEqual((options.context, options.no_draft, options.no_recipes, options.draft_portfolio),
                         (None, True, True, False))
        self.assertEqual(options.disabled, ("context", "draft", "draft-portfolio", "recipes", "search"))
        self.assertFalse(resolvecmd._enabled(options, "search"))
        self.assertTrue(resolvecmd._enabled(options, "twin"))
        self.assertTrue(resolvecmd._enabled(argparse.Namespace(), "twin"))     # resolve-hard on its own

    def test_declare_leaves_the_stage_plan(self):
        args = _parser().parse_args(["--project", "p", "--out", "o", "--declare-assumed"])
        solver.apply_policy(args, solver.policy(None))
        self.assertIn("declare", solver.stage_plan(solver.policy(None), args, {}))
        args.disabled = ["declare"]
        self.assertNotIn("declare", solver.stage_plan(solver.policy(None), args, {"declare_assumed": True}))


class LadderTest(unittest.TestCase):
    """A rung switched off can only cost the match; strict verification is never touched."""

    def setUp(self):
        self.store = resolvemod.CaseStore(_scratch(self))
        self.addCleanup(self.store.close)

    def test_without_the_search_the_ladder_stops_at_the_draft(self):
        with mock.patch.object(resolvemod.itermod, "descend", side_effect=AssertionError("searched")):
            case = resolvemod.resolve(self.store, "main", "read", BASE, TARGET, fake_evaluate, source=SOURCE,
                                      jobs=1, search=False)
        self.assertEqual(case.status, "near")
        skipped = [s for s in case.stages if s["stage"] == "iterate-skipped"]
        self.assertIn("switched off", skipped[0]["reason"])
        self.assertFalse([s for s in case.stages if s["stage"] == "iterate"])
        on = resolvemod.resolve(self.store, "main", "read", BASE, TARGET, fake_evaluate, source=SOURCE, jobs=1,
                                force=True)
        self.assertEqual(on.status, "exact")                    # the default still searches

    def test_without_drafts_x_recipes_a_sibling_recipe_is_not_tried(self):
        narrow = SOURCE.replace("u16 value;", "u8 value;")

        def default(text):
            return iterate.Evaluation(6 if "u8 value;" in text else 3, False, {"counts": {"register": 1}}), WRONG

        def sibling(text):
            exact = "u8 value;" in text
            return iterate.Evaluation(0 if exact else 5, exact, {"counts": {}},
                                      outcome={"proof": {"verdict": "exact"}} if exact else None), \
                (TARGET if exact else WRONG)

        def ladder(store, **switches):
            return resolvemod.resolve(store, "main", "read", BASE, TARGET, default, jobs=1, pipeline="ido5.3",
                                      recipe_label="ido5.3 -O2", search=False,
                                      draft=lambda family: [("plain", SOURCE), ("narrow", narrow)],
                                      recipes=lambda family: [("ido7.1 -O2", "ido7.1", sibling)], **switches)
        off = ladder(self.store, drafts_x_recipes=False)
        self.assertNotEqual(off.status, "exact")
        self.assertFalse([s for s in off.stages if s["stage"] == "drafts-x-recipes"])
        other = resolvemod.CaseStore(_scratch(self))
        self.addCleanup(other.close)
        self.assertEqual(ladder(other).status, "exact")


class RunnerTest(unittest.TestCase):
    def _runner(self, disabled=()):
        args = types.SimpleNamespace(flags="-O2", pipeline="kmc+builtin", context=None, draft_options="",
                                     disabled=tuple(disabled))
        return resolvecmd._Runner(args, None, None, {}, _scratch(self))

    def test_handasm_and_the_at_load_lane_switch_off(self):
        self.assertTrue(hasattr(self._runner().drafter("f", BASE, 12, fake_evaluate, {}), "assembly"))
        runner = self._runner(["handasm", "at-loads"])
        self.assertFalse(hasattr(runner.drafter("f", BASE, 12, fake_evaluate, {}), "assembly"))
        self.assertFalse(runner.at_loads(BASE, 12))              # no evidence read: the lane is never offered


class RodataSettlementTest(unittest.TestCase):
    def _run(self, **switch):
        job = types.SimpleNamespace(vram=0x80000000, size=4, whole=False)      # what run() reads of a Job
        row = {"verdict": "mismatch", "donor": "d", "donor_unit": "u.c", "target_vram": 0x80000000, "size": 4,
               "transferred": {}, "notes": []}
        target = types.SimpleNamespace(rom=b"\0" * 16, delta=0x80000000, vram_to_rom=lambda v: v - 0x80000000)
        settle = mock.Mock(return_value={"promoted": 1})
        with mock.patch.multiple(transplant, scan=mock.Mock(return_value=[]), source_order=mock.Mock(return_value={}),
                                 corroborate=mock.Mock(return_value=0),
                                 transfer=mock.Mock(return_value={"symbols": {}, "contradictions": [],
                                                                  "suspicious": []}),
                                 plan=mock.Mock(return_value=([job], [])), alternative_order=lambda j: 0,
                                 distinct_alternatives=lambda alts: list(alts),
                                 verify_job=mock.Mock(side_effect=lambda *a, **k: dict(row)),
                                 select_rows=lambda rows: rows, settle_rodata=settle,
                                 symbol_file=mock.Mock(return_value="#\n#\n")):
            report = transplant.run(None, target, [("f", 0x80000000, 4)], object(), jobs=1, **switch)
        return report, settle

    def test_settlement_runs_by_default_and_not_when_switched_off(self):
        report, settle = self._run()
        self.assertEqual(settle.call_count, 1)
        self.assertEqual(report["rodata_settlement"], {"promoted": 1})
        report, settle = self._run(rodata_settlement=False)
        self.assertEqual(settle.call_count, 0)
        self.assertEqual(report["rodata_settlement"], {"switched_off": True, "promoted": 0})


class RecordAndResumeTest(unittest.TestCase):
    def test_the_invocation_records_the_switches_and_a_resume_must_keep_them(self):
        out = _scratch(self)
        with mock.patch.dict(os.environ, {**_clean_environ(), "N64DK_DISABLE_TWIN": "1"}, clear=True):
            first = _parser().parse_args(["--project", "/p", "--out", str(out), "--skip-stage", "search"])
            run.prepare_invocation(first)
            run.save_invocation(first, out)
        saved = json.loads((out / "solver" / "invocation.json").read_text())
        self.assertEqual(saved["disabled"], ["search", "twin"])
        self.assertEqual(saved["disabled_from"], {"search": "--skip-stage", "twin": "env N64DK_DISABLE_TWIN"})
        with mock.patch.dict(os.environ, _clean_environ(), clear=True), \
                self.assertRaisesRegex(SystemExit, "--resume: the run was started with"):
            run.prepare_invocation(_parser().parse_args(["--resume", str(out)]))
        with mock.patch.dict(os.environ, {**_clean_environ(), "N64DK_DISABLE_TWIN": "1"}, clear=True):
            again = _parser().parse_args(["--resume", str(out)])
            run.prepare_invocation(again)
        self.assertEqual(again.disabled, ["search", "twin"])

    def test_a_run_saved_before_the_switches_resumes_with_its_skipped_stages(self):
        out = _scratch(self)
        (out / "solver").mkdir()
        (out / "solver" / "invocation.json").write_text(json.dumps(
            {"project": "/p", "out": str(out), "effort": "normal", "skip_stage": ["ledger"]}))
        with mock.patch.dict(os.environ, _clean_environ(), clear=True):
            args = _parser().parse_args(["--resume", str(out)])
            run.prepare_invocation(args)
        self.assertEqual(args.disabled, ["ledger"])

    def test_an_ablation_run_needs_an_empty_case_store(self):
        out = _scratch(self)
        run.check_ablation_store(out, ["search"], False)                     # no store yet
        (out / "state").mkdir()
        run.check_ablation_store(out, ["search"], False)                     # an empty one
        (out / "state" / "cases.sqlite").write_bytes(b"")
        with self.assertRaisesRegex(SystemExit, "case store already holds cases"):
            run.check_ablation_store(out, ["search"], False)
        run.check_ablation_store(out, ["search"], True)                      # its own resume
        run.check_ablation_store(out, [], False)                             # all-on: unchanged behaviour

    def test_the_markdown_report_opens_with_what_was_switched_off(self):
        report = {"project": "p", "images": {}, "ablation": {
            "disabled": {"search": "--skip-stage", "twin": "env N64DK_DISABLE_TWIN"},
            "switched_off": ["search", "twin"],
            "techniques_removed": ["search"], "cascade_removed": []}}
        text = run._markdown(report)
        self.assertIn("ABLATION RUN: switched off search (--skip-stage), twin (env N64DK_DISABLE_TWIN); effort mode "
                      "without techniques search", text)
        report["ablation"] = {"disabled": {"ledger": "--skip-stage"}, "switched_off": []}
        self.assertNotIn("ABLATION RUN", run._markdown(report))


def _case(name, vram, *, status="open", score=10, size=0x40, strict=None):
    return resolvemod.Case(resolvemod.case_id("main", vram, size), "main", name, vram, size, "t" + name,
                           status=status, top_cause="register.permutation", best_score=score,
                           best_digest="d" + name, recipe="ido5.3 -O2", strict=strict or {})


class RunWiringTest(unittest.TestCase):
    """decompile-run end to end with resolve-hard replaced by a fake (as tests/test_solver.py)."""

    ROWS = [("f0", 0x80000000, 0x40), ("f1", 0x80000040, 0x40), ("f2", 0x80000080, 0x40)]

    def setUp(self):
        self.project = _scratch(self)
        (self.project / "n64dk.toml").write_text(
            '[images.main]\npath = "main.bin"\nvram_delta = 0x80000000\ncatalogue = "main.txt"\n')
        words = [0] * 0x40
        words[0x10 + 2] = (3 << 26) | ((0x80000000 >> 2) & 0x03FFFFFF)          # f1 calls f0
        (self.project / "main.bin").write_bytes(b"".join(w.to_bytes(4, "big") for w in words))
        (self.project / "main.txt").write_text("".join(f"{n} = 0x{v:08X}; // type:func size:0x{z:X}\n"
                                                       for n, v, z in self.ROWS))
        self.calls = []

        def fake(options):
            self.calls.append(options)
            store = resolvemod.CaseStore(Path(options.state))
            try:
                for name, vram, size in self.ROWS:
                    case = _case(name, vram, size=size, status="exact" if name == "f0" else "open",
                                 score=0 if name == "f0" else 8, strict={"verdict": "exact"} if name == "f0" else {})
                    case.builds = 12
                    store.put(case)
            finally:
                store.close()
            return {"statuses": {}, "builds": 0}
        patcher = mock.patch.object(resolvecmd, "resolve_hard", fake)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, *argv, environ=None):
        with mock.patch.dict(os.environ, {**_clean_environ(), **(environ or {})}, clear=True):
            args = _parser().parse_args(list(argv))
            run.prepare_invocation(args)
            return run._decompile_run(args)

    def _base(self, out):
        return ("--project", str(self.project), "--out", str(out), "--all-images",
                "--skip-stage", "boundaries", "--skip-stage", "ledger")

    def test_an_all_on_run_is_unchanged_and_says_nothing_was_switched_off(self):
        out = self.project / ".n64dk" / "all"
        result = self._run(*self._base(out))
        self.assertNotIn("switched_off", result)
        options = self.calls[0]
        self.assertEqual((options.context, options.draft_portfolio, options.no_draft, options.no_recipes),
                         ("auto", True, False, False))
        report = json.loads((out / "report.json").read_text())
        self.assertEqual((report["ablation"]["switched_off"], report["ablation"]["techniques_removed"]), ([], []))
        self.assertNotIn("ABLATION RUN", (out / "report.md").read_text())
        self._run(*self._base(out))                       # an all-on rerun in the same directory: allowed as before

    def test_a_switched_off_run_records_it_everywhere_and_cannot_reuse_a_store(self):
        out = self.project / ".n64dk" / "minus"
        result = self._run(*self._base(out), "--skip-stage", "search",
                           environ={"N64DK_DISABLE_CONTEXT": "1"})
        self.assertEqual(result["switched_off"], ["context", "search"])
        options = self.calls[0]
        self.assertIsNone(options.context)
        self.assertIn("search", options.disabled)
        report = json.loads((out / "report.json").read_text())
        self.assertEqual(report["ablation"]["disabled"],
                         {"boundaries": "--skip-stage", "context": "env N64DK_DISABLE_CONTEXT",
                          "ledger": "--skip-stage", "search": "--skip-stage"})
        self.assertEqual(report["ablation"]["techniques_removed"], ["search"])
        self.assertIn("ABLATION RUN: switched off", (out / "report.md").read_text())
        events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
        self.assertIn("ablation", [event["event"] for event in events])
        saved = json.loads((out / "solver" / "invocation.json").read_text())
        self.assertEqual(saved["disabled"], ["boundaries", "context", "ledger", "search"])
        with self.assertRaisesRegex(SystemExit, "case store already holds cases"):
            self._run(*self._base(out), "--skip-stage", "search", environ={"N64DK_DISABLE_CONTEXT": "1"})


class AttributionTest(unittest.TestCase):
    AVAILABLE = list(run.SKIP_STAGES)

    def _checked(self, label, subsystem, kind, disabled):
        found = at.Run(label, {}, subsystem=subsystem, kind=kind)
        at.check_switches(found, {"disabled": {name: "--skip-stage" for name in disabled},
                                  "available": self.AVAILABLE})
        return found

    def test_a_run_is_held_to_what_its_variant_says(self):
        self.assertTrue(self._checked("all", "all", "base", []).complete)
        base = self._checked("all", "all", "base", ["twin"])
        self.assertFalse(base.complete)
        self.assertIn("switched off twin", base.notes[0])
        self.assertTrue(self._checked("-search", "search", "minus", ["search"]).complete)
        self.assertFalse(self._checked("-search", "search", "minus", []).complete)            # forgot the switch
        self.assertFalse(self._checked("-search", "search", "minus", ["search", "twin"]).complete)
        self.assertTrue(self._checked("-small-data", "small-data", "minus", []).complete)      # not a switch
        self.assertFalse(self._checked("+egraph", "egraph", "plus", ["search"]).complete)
        old = at.Run("all", {}, kind="base")
        at.check_switches(old, None)                                     # a report made before the switches
        self.assertTrue(old.complete)
        self.assertEqual(at.restrict(base, set()).disabled, ("twin",))

    def test_load_run_reads_the_record_from_report_json(self):
        root = _scratch(self)
        (root / "export").mkdir()
        (root / "run").mkdir()
        (root / "export" / "FUNCTIONS.tsv").write_text(
            "image\tfunction\tvram\trom_offset\tsize\tcode_size\toutcome\thow\tsource\tdistance\treproduced_by\n"
            "main\tf1\t0x80000400\t0x400\t0x10\t0x10\texact\tkmc+as -O2 (draft)\t\t\t\n")
        (root / "run" / "report.json").write_text(json.dumps({"status": "complete", "ablation": {
            "disabled": {"twin": "env N64DK_DISABLE_TWIN"}, "available": self.AVAILABLE}}))
        loaded = at.load_run("all", root / "export", root / "run", subsystem="all", kind="base")
        self.assertFalse(loaded.complete)
        self.assertEqual(loaded.disabled, ("twin",))


if __name__ == "__main__":
    unittest.main()
