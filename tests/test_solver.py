"""The evidence-driven fixed-point solver (n64dk.match.solver) and its decompile-run wiring.

Everything here uses fakes: a temporary case store and fake sweep/technique callables.  No
compiler runs and no machine lock is taken."""
import argparse
import contextlib
import io
import json
import shutil
import tempfile
import types
import unittest
from pathlib import Path

from n64dk.match import resolve as resolvemod
from n64dk.match import solver


def _scratch(test) -> Path:
    path = Path(tempfile.mkdtemp(prefix="n64dk-solver-test-"))
    test.addCleanup(shutil.rmtree, path, True)
    return path


def _case(name, vram, *, status="open", score=10, top="register.permutation", size=0x40, recipe="ido5.3 -O2",
          digest=None, image="main", **extra):
    return resolvemod.Case(resolvemod.case_id(image, vram, size), image, name, vram, size, "t" + name,
                           status=status, top_cause=top, best_score=score,
                           best_digest=digest if digest is not None else ("d" + name if score is not None else None),
                           recipe=recipe, **extra)


def _parser():
    from n64dk.commands import run
    parser = argparse.ArgumentParser()
    run.decompile_run.arguments(parser)
    return parser


class World:
    """A fake title: the sweep stores open cases, each technique has a scripted outcome per case."""

    def __init__(self, root: Path, cases, script=None, sweep_exact=()):
        self.root, self.cases, self.script = root, cases, dict(script or {})
        self.sweep_exact = set(sweep_exact)
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.sweeps = 0

    def store(self):
        return resolvemod.CaseStore(self.root)

    def sweep(self):
        self.sweeps += 1
        store = self.store()
        try:
            for case in self.cases:
                case = resolvemod.Case(**json.loads(json.dumps(case.__dict__)))
                case.builds = 20
                if case.name in self.sweep_exact:
                    case.status, case.best_score, case.strict = "exact", 0, {"verdict": "exact"}
                store.put(case)
        finally:
            store.close()
        return {}

    def technique(self, name, decisions, calls_left):
        self.calls.append((name, tuple(sorted(d.name for d in decisions))))
        store = self.store()
        calls = 0
        try:
            for decision in decisions:
                outcome, builds = self.script.get((decision.name, name), ("none", 5))
                case = store.get(decision.case_id)
                case.builds += builds
                if outcome == "exact":
                    case.status, case.best_score, case.strict = "exact", 0, {"verdict": "exact"}
                elif outcome == "improve" and (case.best_score or 10) > 1:
                    case.best_score = max(1, (case.best_score or 10) - 3)
                    case.best_digest = (case.best_digest or "") + "+"
                if name == "refine":
                    calls += 3
                store.put(case)
        finally:
            store.close()
        return {"model_calls": calls}


def _solve(world, mode, log, observations, budget=None, rounds=None, start_round=1, model=False, session="s1",
           neighbours=None, declare=None):
    return solver.solve_image(image="main", store_root=world.root, mode=mode, budget=budget or solver.Budget(),
                              log=log, observations=observations,
                              hooks=solver.Hooks(world.sweep, world.technique, declare, neighbours),
                              session=session, rounds=rounds, start_round=start_round, model=model)


class PolicyTableTest(unittest.TestCase):
    def test_every_mode_is_valid_and_bounded(self):
        for mode in solver.MODES.values():
            solver.check_policy(mode)
            self.assertGreater(mode.budget, 0)
            for recipes in (mode.cross_recipes or resolvemod.CROSS_RECIPES, mode.recipe_sweep):
                self.assertLessEqual(recipes * solver.CROSS_DRAFTS, solver.MAX_COMBINATIONS)
            if mode.name != "exhaustive":                    # the kitchen sink runs every technique
                self.assertFalse(mode.unit_recipes, "nearest-unit retry measured 0/165: off below exhaustive")
        self.assertEqual([m.name for m in solver.MODES.values()], ["quick", "normal", "deep", "exhaustive"])
        self.assertLess(solver.MODES["quick"].budget, solver.MODES["normal"].budget)
        self.assertLessEqual(solver.MODES["deep"].budget, solver.MODES["exhaustive"].budget)
        self.assertEqual((solver.MODES["normal"].max_rounds, solver.MODES["normal"].schedule), (1, "observe"))
        self.assertEqual({solver.MODES[m].schedule for m in ("deep", "exhaustive")}, {"order"})
        exhaustive = solver.MODES["exhaustive"]
        self.assertEqual((exhaustive.max_size, exhaustive.require_reasons), (0, True))
        self.assertGreater(exhaustive.max_rounds, solver.MODES["deep"].max_rounds)
        # kitchen sink: nothing skipped to save builds, every opt-in technique on
        self.assertEqual((exhaustive.recipe_gate, exhaustive.search_economics), (False, False))
        self.assertTrue(exhaustive.regional_calibration and exhaustive.unit_recipes and exhaustive.region_recipes)
        self.assertTrue(all(m.recipe_gate and not m.regional_calibration
                            for m in solver.MODES.values() if m.name != "exhaustive"))

    def test_policies_that_break_the_rules_are_refused(self):
        from dataclasses import replace
        normal = solver.MODES["normal"]
        with self.assertRaisesRegex(ValueError, "exceeds"):
            solver.check_policy(replace(normal, recipe_sweep=64))
        with self.assertRaisesRegex(ValueError, "budget"):
            solver.check_policy(replace(normal, budget=0))
        with self.assertRaisesRegex(ValueError, "unknown techniques"):
            solver.check_policy(replace(normal, techniques=("search", "cartesian")))
        with self.assertRaisesRegex(ValueError, "unknown effort mode"):
            solver.policy("ludicrous")

    def test_large_function_threshold_is_the_resolvers(self):
        from n64dk.commands import resolve as resolvecmd
        self.assertEqual(solver.LARGE_FUNCTION, resolvecmd.LARGE_FUNCTION)


class DefaultEqualsLegacyTest(unittest.TestCase):
    """The default (no --effort) must be decompile-run exactly as it was."""

    LEGACY_PLAN = ["boundaries", "calibrate", "resolve", "queue", "ledger"]

    def test_normal_fills_exactly_the_old_defaults(self):
        self.assertEqual({k: getattr(solver.MODES["normal"], k) for k in solver.OPTION_FIELDS},
                         solver.LEGACY_DEFAULTS)
        args = _parser().parse_args(["--project", "p", "--out", "o"])
        solver.apply_policy(args, solver.policy(None))
        self.assertEqual({k: getattr(args, k) for k in solver.OPTION_FIELDS}, solver.LEGACY_DEFAULTS)
        self.assertEqual((args.budget, args.max_size, args.rounds, args.schedule), (200, 0, 1, "observe"))

    def test_default_stage_plan_equals_the_current_one(self):
        args = _parser().parse_args(["--project", "p", "--out", "o"])
        solver.apply_policy(args, solver.policy(None))
        self.assertEqual(solver.stage_plan(solver.policy(None), args, {}), self.LEGACY_PLAN)
        args = _parser().parse_args(["--project", "p", "--out", "o", "--donors", "d.toml", "--declare-assumed",
                                     "--skip-stage", "ledger"])
        solver.apply_policy(args, solver.policy(None))
        self.assertEqual(solver.stage_plan(solver.policy(None), args, {}),
                         ["boundaries", "donors", "calibrate", "resolve", "declare", "queue"])
        deep = _parser().parse_args(["--project", "p", "--out", "o", "--effort", "deep"])
        solver.apply_policy(deep, solver.policy("deep"))
        self.assertIn("rounds", solver.stage_plan(solver.policy("deep"), deep, {}))

    def _options(self, args, mode, settings=None):
        from n64dk.commands import run
        spec = types.SimpleNamespace(path="/rom.z64", vram_delta=None, delta_source="boot", catalogue="/c.txt",
                                     catalogue_shift=0, name="main")
        factory = run.resolve_options_for(args, settings or {}, mode=mode, project=Path("/p"), spec=spec, jobs=2,
                                          events=Path("/o/events.jsonl"), out=Path("/o"), manifests=[], recipes={},
                                          recovered=set())
        return vars(factory())

    def test_resolve_options_of_the_default_equal_the_legacy_options(self):
        legacy = _parser().parse_args(["--project", "p", "--out", "o"])
        for key, value in solver.LEGACY_DEFAULTS.items():       # the literal parser defaults before modes
            setattr(legacy, key, value)
        current = _parser().parse_args(["--project", "p", "--out", "o"])
        solver.apply_policy(current, solver.policy(None))
        self.assertEqual(self._options(current, solver.policy(None)), self._options(legacy, solver.policy("normal")))
        explicit = _parser().parse_args(["--project", "p", "--out", "o", "--effort", "normal"])
        solver.apply_policy(explicit, solver.policy("normal"))
        self.assertEqual(self._options(explicit, solver.policy("normal")),
                         self._options(legacy, solver.policy("normal")))

    def test_an_explicit_option_wins_over_the_mode_and_quick_never_asks_the_refiner(self):
        args = _parser().parse_args(["--project", "p", "--out", "o", "--effort", "deep", "--budget", "50"])
        solver.apply_policy(args, solver.policy("deep"))
        self.assertEqual((args.budget, args.rounds), (50, solver.MODES["deep"].max_rounds))
        settings = {"refine": {"command": "true"}}
        quick = _parser().parse_args(["--project", "p", "--out", "o", "--effort", "quick"])
        solver.apply_policy(quick, solver.policy("quick"))
        self.assertIsNone(self._options(quick, solver.policy("quick"), settings)["refine"])
        normal = _parser().parse_args(["--project", "p", "--out", "o"])
        solver.apply_policy(normal, solver.policy(None))
        self.assertIsNotNone(self._options(normal, solver.policy("normal"), settings)["refine"])
        self.assertEqual(self._options(quick, solver.policy("quick"))["max_size"], 0x200)


class CauseTest(unittest.TestCase):
    def test_primary_cause_follows_the_review_table(self):
        expect = {
            "uncertain boundary": _case("a", 0x100, status="blocked-boundary", top="boundary.fragment", score=None),
            "handwritten": _case("b", 0x200, status="probable-handwritten", top="assembly.handwritten", score=None),
            "no buildable draft": _case("c", 0x300, status="does-not-build", score=None, top="semantic.unknown"),
            "link/address mismatch": _case("d", 0x400, status="assumed", score=0, top=""),
            "unknown toolchain": _case("e", 0x500, top="register.pressure"),
            "large far-off": _case("f", 0x600, score=90, top="type.width"),
            "recipe uncertain": _case("g", 0x700, top="compiler.family"),
            "wrong load width": _case("h", 0x800, top="type.signedness"),
            "global offsets differ": _case("i", 0x900, top="global.declaration"),
            "register-only residual": _case("j", 0xA00, top="register.lifetime"),
            "branch topology": _case("k", 0xB00, top="control.branch_likely"),
            "wrong stack frame": _case("l", 0xC00, top="frame.size"),
        }
        for cause, case in expect.items():
            self.assertEqual(solver.primary_cause(case, unknown={0x500}), cause, case.name)
        self.assertEqual(solver.primary_cause(_case("m", 0xD00, size=0x800, score=5, top="type.width")),
                         "large far-off")
        self.assertEqual(set(solver.CAUSES), set(expect) | {"no discovered image"})

    def test_only_the_causes_own_techniques_are_ever_applicable(self):
        mode = solver.MODES["exhaustive"]
        observations = solver.Observations()
        cases = [_case(f"f{i}", 0x1000 + 0x100 * i, status=status, score=score, top=top)
                 for i, (status, score, top) in enumerate([
                     ("open", 10, "register.permutation"), ("open", 10, "compiler.family"),
                     ("does-not-build", None, ""), ("assumed", 0, ""), ("open", 30, "frame.size"),
                     ("blocked-boundary", None, "boundary.false_split"), ("open", 120, "semantic.unknown")])]
        for case in cases:
            decision = solver.decide(case, mode, observations, model=True)
            allowed = solver.CAUSES[decision.cause][1]
            self.assertTrue(decision.technique is None or decision.technique in allowed, decision)
            self.assertTrue(all(row["technique"] in allowed for row in decision.ranked))
        boundary = solver.decide(cases[5], mode, observations, model=True)
        self.assertIsNone(boundary.technique)
        self.assertIn("never applied unattended", boundary.reason)
        # a style fingerprint is not a build test: the stretch still gets the recipe sweep (2026-11-18)
        unknown = solver.decide(cases[0], mode, observations, unknown={cases[0].vram})
        self.assertEqual((unknown.cause, unknown.technique), ("unknown toolchain", "recipe-sweep"))

    def test_exclusions_are_recorded_with_their_reason(self):
        mode = solver.MODES["deep"]
        big = _case("big", 0x100, size=0x800, score=5, top="compiler.family")
        usable, excluded = solver.applicable(big, "recipe uncertain", mode)
        self.assertEqual(usable, [])
        self.assertIn("964 builds", excluded["search"])
        far = _case("far", 0x200, score=60, top="type.width")
        self.assertIn("search ceiling", solver.applicable(far, "wrong load width", mode)[1]["search"])
        closed = _case("closed", 0x300, closed_families={"dclosed": list(solver.itermod.FAMILIES)})
        self.assertIn("closed", solver.applicable(closed, "register-only residual", mode)[1]["search"])
        self.assertIn("no refiner", solver.applicable(closed, "register-only residual", mode, model=False)[1]["refine"])
        named = _case("named", 0x400, status="assumed", score=0,
                      strict={"assumptions": [{"kind": "symbol", "name": "gThing", "address": "0x80001000"}]})
        self.assertIn("a person has to declare", solver.applicable(named, "link/address mismatch", mode)[1]["declare"])
        described = _case("described", 0x500, status="assumed", score=0,
                          strict={"assumptions": [{"kind": "symbol", "name": "D_80001000", "address": "0x80001000"}]})
        self.assertEqual(solver.applicable(described, "link/address mismatch", mode)[0], ["declare"])
        normal = solver.applicable(_case("n", 0x600), "register-only residual", solver.MODES["normal"])[0]
        self.assertEqual(normal, ["search"])                   # predicted only: normal runs one round

    def test_technique_options_are_single_dimension_overrides(self):
        mode = solver.MODES["exhaustive"]
        self.assertEqual(solver.technique_options("search", mode),
                         {"only_statuses": ("open", "near"), "write_scope": False})
        sweep = solver.technique_options("recipe-sweep", mode)
        self.assertEqual((sweep["fresh"], sweep["cross_recipes"]), (True, mode.recipe_sweep))
        self.assertNotIn("draft_options", sweep)
        self.assertTrue(solver.technique_options("refine", mode)["skip_sweep"])


class RankingTest(unittest.TestCase):
    def test_ranking_learns_from_recorded_outcomes(self):
        mode = solver.MODES["deep"]
        observations = solver.Observations()
        case = _case("x", 0x100, top="compiler.family")
        first = observations.rank(case, "recipe uncertain", ["recipe-sweep", "search"], mode)
        for i in range(6):                                  # search keeps winning on this cause
            before = _case(f"s{i}", 0x1000 + i * 0x100, top="compiler.family")
            after = _case(f"s{i}", 0x1000 + i * 0x100, top="compiler.family", status="exact", score=0)
            observations.record(before, after, "search", "recipe uncertain", 2, 10)
            failed = _case(f"r{i}", 0x2000 + i * 0x100, top="compiler.family")
            observations.record(failed, failed, "recipe-sweep", "recipe uncertain", 2, 80)
        after = observations.rank(case, "recipe uncertain", ["recipe-sweep", "search"], mode)
        self.assertGreater(after[0]["p_exact"], first[0]["p_exact"])
        self.assertEqual(after[0]["technique"], "search")
        self.assertEqual(observations.cost("recipe-sweep", mode), 80.0)
        observed = observations.rank(case, "recipe uncertain", ["recipe-sweep", "search"], solver.MODES["normal"])
        self.assertEqual([r["technique"] for r in observed], ["recipe-sweep", "search"])   # observe: table order

    def test_measured_priors_are_used(self):
        observations = solver.Observations()
        p = observations.probability("register-only residual", "ido", "<=0x40", "search")
        self.assertAlmostEqual(p, 17 / 127, places=3)
        self.assertLess(observations.probability("no buildable draft", "ido", "<=0x40", "draft-options"), 0.02)

    def test_negative_facts_block_repeats_until_new_evidence(self):
        mode = solver.MODES["deep"]
        observations = solver.Observations()
        case = _case("x", 0x100)
        observations.record(case, case, "search", "register-only residual", 2, 30)
        decision = solver.decide(case, mode, observations)
        self.assertIsNone(decision.technique)
        self.assertIn("changed nothing", decision.excluded["search"])
        self.assertTrue(observations.mark_evidence(case.case_id, 3))
        self.assertEqual(solver.decide(case, mode, observations).technique, "search")
        moved = _case("x", 0x100, digest="dother")              # another best source: not the same attempt
        observations.evidence.clear()
        self.assertEqual(solver.decide(moved, mode, observations).technique, "search")

    def test_a_plateau_sets_a_technique_aside_only_when_ranking_orders_work(self):
        observations = solver.Observations()
        for i in range(solver.PLATEAU_ATTEMPTS):
            other = _case(f"p{i}", 0x1000 + 0x100 * i, digest=f"dp{i}")
            observations.record(other, other, "search", "register-only residual", 2, 30)
        self.assertTrue(observations.plateaued("register-only residual", "search"))
        fresh = _case("fresh", 0x9000)
        self.assertIn("plateaued", solver.decide(fresh, solver.MODES["deep"], observations).excluded["search"])
        observe = solver.Policy(**{**solver.MODES["deep"].as_dict(), "schedule": "observe"})
        self.assertEqual(solver.decide(fresh, observe, observations).technique, "search")

    def test_ranking_changes_order_never_credit(self):
        cases = [_case(f"f{i}", 0x1000 + 0x100 * i, top="compiler.family") for i in range(6)]
        script = {("f0", "search"): ("exact", 7), ("f1", "recipe-sweep"): ("exact", 40),
                  ("f2", "recipe-sweep"): ("improve", 40), ("f3", "search"): ("exact", 7)}
        outcomes = []
        for favour in ("search", "recipe-sweep"):
            root = _scratch(self)
            world = World(root / "state", cases, script)
            observations = solver.Observations(root / "obs.json")
            for i in range(8):                                  # seed: the favoured technique looks better
                seed = _case(f"seed{i}", 0x80000 + i * 0x100, top="compiler.family")
                won = _case(f"seed{i}", 0x80000 + i * 0x100, top="compiler.family", status="exact", score=0)
                observations.record(seed, won, favour, "recipe uncertain", 0, 5)
            log = solver.RoundsLog(root / "rounds.jsonl")
            stop = _solve(world, solver.MODES["exhaustive"], log, observations, rounds=20)
            self.assertTrue(stop["reason"].startswith("fixed point"), stop)
            store = world.store()
            try:
                final = store.all()
            finally:
                store.close()
            first_round = [c for c in world.calls]
            outcomes.append(({c.case_id for c in final if c.status == "exact"}, solver.credit(final),
                             first_round[0][0]))
        self.assertEqual(outcomes[0][0], outcomes[1][0])
        self.assertEqual(outcomes[0][1], outcomes[1][1])
        self.assertEqual({outcomes[0][2], outcomes[1][2]}, {"search", "recipe-sweep"})   # the order did change

    def test_planning_never_touches_a_case(self):
        cases = [_case(f"f{i}", 0x1000 + 0x100 * i, top=top)
                 for i, top in enumerate(["compiler.family", "type.width", "frame.size"])]
        frozen = [json.dumps(c.__dict__, sort_keys=True) for c in cases]
        solver.plan(cases, solver.MODES["exhaustive"], solver.Observations(), model=True)
        self.assertEqual([json.dumps(c.__dict__, sort_keys=True) for c in cases], frozen)


class FixedPointTest(unittest.TestCase):
    def setUp(self):
        self.root = _scratch(self)
        self.cases = [_case(f"f{i}", 0x1000 + 0x100 * i) for i in range(4)]

    def _world(self, script=None, **kw):
        return World(self.root / "state", self.cases, script, **kw)

    def test_stops_at_the_fixed_point_and_logs_every_round(self):
        world = self._world({("f0", "search"): ("exact", 9), ("f1", "search"): ("improve", 9)})
        log = solver.RoundsLog(self.root / "rounds.jsonl")
        stop = _solve(world, solver.MODES["deep"], log, solver.Observations(self.root / "obs.json"), rounds=10)
        rounds = [r for r in log.read() if r["kind"] == "round"]
        self.assertTrue(stop["reason"].startswith("fixed point"), stop)
        self.assertEqual(rounds[0]["sieve"], "sweep")
        self.assertEqual(rounds[1]["techniques"]["search"]["exact"], 1)
        self.assertEqual(rounds[1]["gains"]["new_exact"], 1)
        self.assertEqual(rounds[1]["techniques"]["search"]["builds"], 9 + 9 + 5 + 5)
        self.assertEqual(rounds[1]["gains"]["negative_facts"], 2)                       # f2 and f3
        # f1 improved (new best source) and is searched again in round 3; the others are not repeated
        self.assertEqual(world.calls[1], ("search", ("f1",)))
        self.assertEqual(rounds[-1]["gains"]["new_exact"], 0)
        self.assertEqual(log.read()[-1]["kind"], "stop")

    def test_stops_on_the_round_limit(self):
        world = self._world()
        log = solver.RoundsLog(self.root / "rounds.jsonl")
        stop = _solve(world, solver.MODES["normal"], log, solver.Observations())
        self.assertEqual(stop["after_round"], 1)
        self.assertIn("round limit", stop["reason"])
        self.assertEqual(world.calls, [])                      # normal: the sweep only

    def test_stops_on_each_budget(self):
        world = self._world()
        log = solver.RoundsLog(self.root / "rounds.jsonl")
        stop = _solve(world, solver.MODES["deep"], log, solver.Observations(), budget=solver.Budget(builds=50))
        self.assertIn("build budget 50", stop["reason"])       # the sweep alone spent 80
        self.assertEqual(world.calls, [])

        root = _scratch(self)
        world = World(root / "state", self.cases)
        ticks = iter(range(0, 10_000, 100))
        budget = solver.Budget(seconds=150, clock=lambda: next(ticks))
        stop = _solve(world, solver.MODES["deep"], solver.RoundsLog(root / "rounds.jsonl"), solver.Observations(),
                      budget=budget)
        self.assertIn("time budget", stop["reason"])

        root = _scratch(self)
        cases = [_case(f"g{i}", 0x1000 + 0x100 * i, top="frame.size") for i in range(3)]
        world = World(root / "state", cases, {(c.name, "search"): ("none", 1) for c in cases})
        mode = solver.Policy(**{**solver.MODES["deep"].as_dict(), "max_rounds": 6})
        stop = _solve(world, mode, solver.RoundsLog(root / "rounds.jsonl"), solver.Observations(),
                      budget=solver.Budget(model_calls=5), model=True)
        self.assertIn("model-call budget 5", stop["reason"])

        root = _scratch(self)
        world = World(root / "state", self.cases)
        stop = _solve(world, solver.MODES["deep"], solver.RoundsLog(root / "rounds.jsonl"), solver.Observations(),
                      budget=solver.Budget(builds=100))      # 80 by the sweep, 20 left: less than one search
        self.assertIn("budget", stop["reason"])
        self.assertEqual(world.calls, [])

    def test_resume_continues_without_redoing_finished_rounds(self):
        world = self._world({("f0", "search"): ("improve", 3), ("f0", "refine"): ("exact", 0)})
        log = solver.RoundsLog(self.root / "rounds.jsonl")
        observations = solver.Observations(self.root / "obs.json")
        stop = _solve(world, solver.MODES["deep"], log, observations, rounds=2)
        self.assertIn("round limit", stop["reason"])
        self.assertEqual(world.sweeps, 1)
        calls_before = list(world.calls)
        last, stopped = log.progress("s1", "main")
        self.assertEqual((last, stopped["after_round"]), (2, 2))
        again = solver.Observations(self.root / "obs.json")        # reloaded from disk
        self.assertEqual(again.facts.keys(), observations.facts.keys())
        stop = _solve(world, solver.MODES["deep"], log, again, rounds=12, start_round=last + 1, model=True)
        self.assertEqual(world.sweeps, 1)                          # round 1 is not redone
        rounds = [r["round"] for r in log.read() if r["kind"] == "round"]
        self.assertEqual(rounds[:2], [1, 2])
        self.assertEqual(rounds[2], 3)
        self.assertNotIn(("search", ("f1", "f2", "f3")), world.calls[len(calls_before):])   # facts kept
        self.assertTrue(stop["reason"].startswith("fixed point"))

    def test_cascade_declares_and_marks_neighbours(self):
        cases = [_case("a", 0x1000), _case("b", 0x1100),
                 _case("c", 0x1200, status="assumed", score=0,
                       strict={"verdict": "assumed",
                               "assumptions": [{"kind": "symbol", "name": "D_80001000", "address": "0x80001000"}]})]
        world = World(self.root / "state", cases, {("a", "search"): ("exact", 4)})

        def declare():
            store = world.store()
            try:
                case = store.get(cases[2].case_id)
                case.status, case.strict, case.builds = "exact", {"verdict": "exact"}, case.builds + 1
                store.put(case)
            finally:
                store.close()
            return {"names": ["D_80001000"]}
        log = solver.RoundsLog(self.root / "rounds.jsonl")
        observations = solver.Observations()
        stop = _solve(world, solver.MODES["deep"], log, observations, declare=declare,
                      neighbours=lambda vram: {0x1100} if vram == 0x1000 else set())
        rounds = [r for r in log.read() if r["kind"] == "round"]
        self.assertEqual(rounds[0]["cascade"]["declare"]["promoted"], 1)
        self.assertEqual(rounds[0]["gains"]["declarations"], 1)
        self.assertEqual(rounds[1]["cascade"]["neighbours"]["marked"], 1)       # a became exact: b has evidence
        self.assertEqual(observations.evidence[cases[1].case_id], 3)
        self.assertIn(("search", ("b",)), world.calls)          # b searched again after the evidence
        self.assertTrue(stop["reason"].startswith("fixed point"))


class CoverageAndDossierTest(unittest.TestCase):
    def test_exhaustive_records_a_reason_for_every_function_not_attempted(self):
        rows = [("donor", 0x80000000, 0x20), ("cased", 0x80000020, 0x40), ("sized", 0x80000060, 0x500),
                ("manifest", 0x80000560, 0x20), ("outside", 0x80900000, 0x20), ("tiny", 0x80000580, 2),
                ("big", 0x80000600, 0x600), ("lost", 0x80000C00, 0x20)]
        cases = [_case("cased", 0x80000020), _case("big", 0x80000600, size=0x600)]
        found = solver.coverage(rows, cases, recovered={0x80000000},
                                left_out={0x80000060: "larger than the run's --max-size 0x400"},
                                covered=lambda vram: vram == 0x80000560, image_size=0x2000, delta=0x80000000)
        entries = {e["name"]: e for e in found["entries"]}
        for entry in found["entries"]:
            if not entry["attempted"]:
                self.assertTrue(entry["reason"], entry)
        self.assertIn("donor", entries["donor"]["reason"])
        self.assertIn("--max-size", entries["sized"]["reason"])
        self.assertIn("manifest", entries["manifest"]["reason"])
        self.assertIn("no discovered image", entries["outside"]["reason"])
        self.assertIn("minimum", entries["tiny"]["reason"])
        self.assertIn("964", entries["big"]["limited"])
        self.assertTrue(entries["lost"]["unexplained"])
        self.assertEqual((found["functions"], found["attempted"], found["unexplained"]), (8, 2, 1))

    def test_exhaustive_flags_a_function_without_a_reason(self):
        from n64dk.commands import run
        out = _scratch(self)
        store = resolvemod.CaseStore(out / "state")
        store.put(_case("a", 0x80000000))
        store.close()
        rows = [("a", 0x80000000, 0x40), ("b", 0x80000040, 0x40)]
        common = dict(rows=rows, data=bytes(0x80), delta=0x80000000, recovered=set(), manifest_rows=[],
                      stretches=solver.Stretches(), ceiling=40, model=False, stop={"reason": "fixed point"},
                      boundary_file=out / "none.json")
        loose = run.solver_outputs(out, "main", solver.MODES["normal"], solver.Observations(), **common)
        strict = run.solver_outputs(out, "main", solver.MODES["exhaustive"], solver.Observations(), **common)
        self.assertIsNone(loose["coverage_warning"])
        self.assertIn("1 have none", strict["coverage_warning"])
        self.assertFalse(any(p.name.endswith(("-wal", "-shm")) for p in (out / "state").iterdir()))

    def test_dossier_holds_what_was_tried_and_why_automation_stopped(self):
        case = _case("x", 0x80001000, top="compiler.family", stages=[
            {"stage": "drafts", "variants": [{"label": "plain", "score": 30}, {"label": "context", "score": 22}]},
            {"stage": "recipe", "tried": 9, "chosen": "ido7.1 -O2", "score": 22},
            {"stage": "iterate", "families": ["types"], "builds": 120, "start": 22, "final": 14}],
            causes=[{"cause": "compiler.family", "confidence": 0.8}])
        observations = solver.Observations()
        observations.record(case, case, "recipe-sweep", "recipe uncertain", 2, 70)
        decision = solver.decide(case, solver.MODES["deep"], observations)
        record = solver.dossier(case, decision, observations, related=[{"name": "y", "relation": "calls"}],
                                stop="fixed point")
        for key in ("image", "vram", "size", "boundary", "best_recipe", "recipes_tried", "best_score", "residual",
                    "primary_cause", "ladder", "solver_attempts", "builds", "related_exact", "why_stopped"):
            self.assertIn(key, record)
        self.assertEqual(record["primary_cause"], "recipe uncertain")
        self.assertEqual(record["recipes_tried"], ["ido7.1 -O2"])
        self.assertEqual(record["builds"]["solver_rounds"], 70)
        self.assertEqual([row["technique"] for row in record["ladder"]], ["drafts", "recipe", "iterate"])
        self.assertEqual(record["next_action"], "search")

    def test_dossier_names_the_region_and_what_would_reopen_it(self):
        case = _case("x", 0x80001000, top="frame.size", stages=[
            {"stage": "frame-fit", "tried": 6, "chosen": "pad 8 top", "score": 30}])
        region = {"start": "0x80000000", "end": "0x80002000", "verdict": "unknown",
                  "ranked": [{"recipe": "kmc+as -O2", "share": 0.4}],
                  "component_mismatches": [{"component": "assembler: global loads expanded through $at"}],
                  "unknown_marks": ["at=direct-load"]}
        observations = solver.Observations()
        decision = solver.decide(case, solver.MODES["normal"], observations)
        record = solver.dossier(case, decision, observations, region=region)
        self.assertEqual(record["toolchain_region"]["verdict"], "unknown")
        self.assertTrue(record["transformations_of_best_source"]["scaffolding"])
        reopen = " | ".join(record["reopen_on"])
        self.assertIn("global loads expanded through $at", reopen)
        self.assertIn("stack frame", reopen)
        self.assertIsNone(solver.dossier(case, decision, observations)["toolchain_region"])

    def test_call_graph_links_callers_and_callees(self):
        delta = 0x80000000
        words = [0] * 16
        words[1] = (3 << 26) | ((0x80000020 >> 2) & 0x03FFFFFF)     # f calls g
        data = b"".join(w.to_bytes(4, "big") for w in words)
        graph = solver.call_graph(data, delta, [("f", 0x80000000, 0x20), ("g", 0x80000020, 0x20)])
        self.assertEqual(graph, {0x80000000: {0x80000020}, 0x80000020: {0x80000000}})

    def test_stretches(self):
        found = solver.Stretches({"stretches": [{"start": "0x80001000", "end": "0x80002000"}]})
        self.assertIn(0x80001800, found)
        self.assertNotIn(0x80003000, found)
        self.assertFalse(solver.Stretches(None))


class StatusTest(unittest.TestCase):
    def test_status_is_read_only_and_prints_the_round_table(self):
        from n64dk.commands import runstatus
        run = _scratch(self)
        store = resolvemod.CaseStore(run / "state")
        for case in [_case("a", 0x1000, status="exact", score=0, strict={"verdict": "exact"}),
                     _case("b", 0x1100, top="frame.size"), _case("c", 0x1200, status="blocked-boundary",
                                                                 top="boundary.fragment", score=None)]:
            store.put(case)
        store.close()
        log = solver.RoundsLog(run / "rounds.jsonl")
        log.append({"kind": "round", "session": "s", "image": "main", "round": 1, "sieve": "sweep",
                    "techniques": {"sweep": {"cases": 3, "builds": 60, "exact": 1}}, "cascade": {},
                    "gains": {"new_exact": 1, "awaiting_declarations": 0, "evidence": 3}})
        log.append({"kind": "round", "session": "s", "image": "main", "round": 2, "sieve": "search",
                    "techniques": {"search": {"cases": 1, "builds": 30, "exact": 1}}, "cascade": {},
                    "gains": {"new_exact": 1, "awaiting_declarations": 0, "evidence": 1}})
        log.append({"kind": "stop", "session": "s", "image": "main", "after_round": 2, "reason": "fixed point"})
        solver.write_json(run / "solver" / "invocation.json", {"effort": "deep"})

        def listing():
            return sorted((str(p.relative_to(run)), p.stat().st_mtime_ns) for p in run.rglob("*"))
        before = listing()
        result = runstatus.status(run)
        text = runstatus.render(result)
        self.assertEqual(listing(), before)
        self.assertFalse(any(name.endswith(("-wal", "-shm")) for name, _t in listing()))
        self.assertIn("Round 2 [main]: search +1 exact", text)
        self.assertEqual(result["mode"], "deep")
        self.assertEqual(set(result["remaining_by_cause"]), {"wrong stack frame", "uncertain boundary"})
        self.assertEqual(result["remaining_by_cause"]["wrong stack frame"]["next"], {"search": 1})
        self.assertEqual(result["credit"]["exact_c"], 1)

    def test_command_prints_text(self):
        from n64dk import cli
        run = _scratch(self)
        store = resolvemod.CaseStore(run / "state")
        store.put(_case("b", 0x1100))
        store.close()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            cli.main(["decompile-status", str(run)])
        self.assertIn("register-only residual", buffer.getvalue())


class RunWiringTest(unittest.TestCase):
    """``decompile-run`` end to end with ``resolve-hard`` replaced by a fake (no compiler, no lock)."""

    ROWS = [("f0", 0x80000000, 0x40), ("f1", 0x80000040, 0x40), ("f2", 0x80000080, 0x40)]

    def setUp(self):
        from n64dk.commands import resolve as resolvecmd
        self.project = _scratch(self)
        (self.project / "n64dk.toml").write_text(
            '[images.main]\npath = "main.bin"\nvram_delta = 0x80000000\ncatalogue = "main.txt"\n')
        words = [0] * 0x40
        words[0x10 + 2] = (3 << 26) | ((0x80000000 >> 2) & 0x03FFFFFF)          # f1 calls f0
        (self.project / "main.bin").write_bytes(b"".join(w.to_bytes(4, "big") for w in words))
        (self.project / "main.txt").write_text("".join(f"{n} = 0x{v:08X}; // type:func size:0x{z:X}\n"
                                                       for n, v, z in self.ROWS))
        self.calls = []
        original = resolvecmd.resolve_hard

        def fake(options):
            self.calls.append(options)
            store = resolvemod.CaseStore(Path(options.state))
            try:
                if options.only_names is None and not getattr(options, "only_statuses", None):
                    for name, vram, size in self.ROWS:            # the sweep
                        case = _case(name, vram, size=size, status="exact" if name == "f0" else "open", score=0
                                     if name == "f0" else 8, strict={"verdict": "exact"} if name == "f0" else {})
                        case.builds = 12
                        store.put(case)
                else:                                              # a later technique: f1 matches
                    names = {n for item in options.only_names for n in item.split(",")}
                    for name, vram, size in self.ROWS:
                        if name in names:
                            case = store.get(resolvemod.case_id("main", vram, size))
                            case.builds += 6
                            if name == "f1":
                                case.status, case.best_score, case.strict = "exact", 0, {"verdict": "exact"}
                            store.put(case)
            finally:
                store.close()
            return {"statuses": {}, "builds": 0}
        resolvecmd.resolve_hard = fake
        self.addCleanup(setattr, resolvecmd, "resolve_hard", original)

    def _run(self, *argv):
        from n64dk.commands import run
        args = _parser().parse_args(list(argv))
        run.prepare_invocation(args)
        return run._decompile_run(args)

    def test_default_run_is_one_sweep_with_the_legacy_options(self):
        out = self.project / ".n64dk" / "run"
        self._run("--project", str(self.project), "--out", str(out), "--all-images",
                  "--skip-stage", "boundaries", "--skip-stage", "ledger")
        self.assertEqual(len(self.calls), 1)
        options = self.calls[0]
        self.assertEqual((options.budget, options.max_size, options.cross_recipes, options.region_recipes,
                          options.search_economics, options.only_names), (200, 0, None, False, False, None))
        log = solver.RoundsLog(out / "rounds.jsonl").read()
        self.assertEqual([r["kind"] for r in log], ["round", "stop"])
        self.assertIn("round limit", log[-1]["reason"])
        dossier = json.loads((out / "solver" / "hard_queue_main.json").read_text())
        self.assertEqual(sorted(r["name"] for r in dossier), ["f1", "f2"])
        self.assertEqual({r["next_action"] for r in dossier}, {"search"})   # predicted, not run
        coverage = json.loads((out / "solver" / "coverage_main.json").read_text())
        self.assertEqual((coverage["attempted"], coverage["unexplained"]), (3, 0))

    def test_a_finished_normal_run_continues_as_deep_from_the_same_round_one(self):
        out = self.project / ".n64dk" / "run"
        common = ["--skip-stage", "boundaries", "--skip-stage", "ledger"]
        self._run("--project", str(self.project), "--out", str(out), "--all-images", *common)
        self.assertEqual(len(self.calls), 1)
        self._run("--resume", str(out), "--effort", "deep")
        sweeps = [o for o in self.calls if o.only_names is None]
        self.assertEqual(len(sweeps), 1)                           # round 1 is not redone
        self.assertTrue(all(o.budget == 200 for o in self.calls))  # the run's own per-function budget
        log = solver.RoundsLog(out / "rounds.jsonl").read()
        rounds = [r for r in log if r["kind"] == "round"]
        self.assertEqual([(r["round"], r["mode"]) for r in rounds][:2], [(1, "normal"), (2, "deep")])
        self.assertEqual(len({r["session"] for r in log}), 1)
        self.assertTrue(log[-1]["reason"].startswith("fixed point"))

    def test_deep_run_reaches_its_fixed_point_and_resume_redoes_nothing(self):
        from n64dk.commands import runstatus
        out = self.project / ".n64dk" / "run"
        self._run("--project", str(self.project), "--out", str(out), "--all-images", "--effort", "deep",
                  "--skip-stage", "boundaries", "--skip-stage", "ledger")
        later = [o for o in self.calls if o.only_names]
        self.assertTrue(later and all(o.write_scope is False and o.refine is None for o in later))
        log = solver.RoundsLog(out / "rounds.jsonl").read()
        rounds = [r for r in log if r["kind"] == "round"]
        self.assertEqual(rounds[1]["techniques"]["search"]["exact"], 1)
        self.assertTrue(log[-1]["reason"].startswith("fixed point"), log[-1])
        report = (out / "report.md").read_text()
        self.assertIn("effort `deep`", report)
        observations = solver.Observations(out / "solver" / "observations.json")
        self.assertIn(resolvemod.case_id("main", 0x80000040, 0x40), observations.facts)
        # f1 calls f0, which was exact from the sweep; f2 has nothing related
        dossier = {r["name"]: r for r in json.loads((out / "solver" / "hard_queue_main.json").read_text())}
        self.assertEqual(list(dossier), ["f2"])
        self.assertIn("changed nothing", dossier["f2"]["excluded_actions"]["search"])
        count = len(self.calls)
        self._run("--resume", str(out))
        self.assertEqual(len(self.calls), count)                  # finished rounds are not redone
        text = runstatus.render(runstatus.status(out))
        self.assertIn("Round 2 [main]: search +1 exact", text)


class InvocationTest(unittest.TestCase):
    def test_resume_takes_the_saved_invocation_and_new_budgets(self):
        from n64dk.commands import run
        out = _scratch(self)
        first = _parser().parse_args(["--project", "/p", "--out", str(out), "--effort", "deep", "--budget", "77",
                                      "--budget-builds", "10"])
        run.prepare_invocation(first)
        run.save_invocation(first, out)
        saved = json.loads((out / "solver" / "invocation.json").read_text())
        self.assertNotIn("budget_builds", saved)
        again = _parser().parse_args(["--resume", str(out), "--budget-builds", "500", "--rounds", "9"])
        mode = run.prepare_invocation(again)
        self.assertEqual((mode.name, again.project, again.out, again.budget, again.budget_builds, again.rounds),
                         ("deep", "/p", str(out), 77, 500, 9))
        normal = _scratch(self)
        start = _parser().parse_args(["--project", "/p", "--out", str(normal)])
        run.prepare_invocation(start)
        run.save_invocation(start, normal)
        deeper = _parser().parse_args(["--resume", str(normal), "--effort", "deep"])
        mode = run.prepare_invocation(deeper)
        self.assertEqual((mode.name, deeper.rounds, deeper.schedule, deeper.budget),
                         ("deep", solver.MODES["deep"].max_rounds, "order", 200))   # round 1's options kept
        with self.assertRaisesRegex(SystemExit, "invocation.json"):
            run.prepare_invocation(_parser().parse_args(["--resume", str(_scratch(self))]))
        with self.assertRaisesRegex(SystemExit, "--project and --out"):
            run.prepare_invocation(_parser().parse_args([]))

    def test_cli_documents_the_new_options(self):
        from n64dk import cli
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), self.assertRaises(SystemExit):
            cli.main(["decompile-run", "--help"])
        for option in ("--effort", "--rounds", "--schedule", "--budget-time", "--budget-builds",
                       "--budget-model-calls", "--resume"):
            self.assertIn(option, buffer.getvalue())

    def test_durations(self):
        self.assertEqual([solver.parse_duration(x) for x in ("90", "90s", "30m", "2h", None)],
                         [90.0, 90.0, 1800.0, 7200.0, None])


class SettingFieldsTest(unittest.TestCase):
    """recipe_gate and regional_calibration: command line > [run] setting > the run's starting mode,
    and a resume under another --effort keeps the values the run was started with (review A 6-7)."""

    def _gate(self, args, mode, settings):
        from n64dk.commands import run
        spec = types.SimpleNamespace(path="/rom.z64", vram_delta=None, delta_source="boot", catalogue="/c.txt",
                                     catalogue_shift=0, name="main")
        factory = run.resolve_options_for(args, settings, mode=mode, project=Path("/p"), spec=spec, jobs=2,
                                          events=Path("/o/events.jsonl"), out=Path("/o"), manifests=[], recipes={},
                                          recovered=set())
        return factory().recipe_gate

    def test_one_precedence_rule_for_both_settings(self):
        exhaustive = solver.policy("exhaustive")
        args = _parser().parse_args(["--project", "p", "--out", "o", "--effort", "exhaustive"])
        self.assertIsNone(args.regional_calibration)            # neither flag given: None, not store_false's True
        solver.apply_policy(args, exhaustive)
        self.assertEqual(args.mode_settings, {"recipe_gate": False, "regional_calibration": True})
        self.assertFalse(solver.setting(args, {}, "recipe_gate", exhaustive))
        self.assertTrue(solver.setting(args, {"recipe_gate": True}, "recipe_gate", exhaustive))
        self.assertTrue(self._gate(args, exhaustive, {"recipe_gate": True}))           # [run] beats the mode
        self.assertTrue(solver.setting(args, {}, "regional_calibration", exhaustive))
        # the old `or` chain kept exhaustive's regional calibration on whatever [run] said
        self.assertFalse(solver.setting(args, {"regional_calibration": False}, "regional_calibration", exhaustive))
        explicit = _parser().parse_args(["--project", "p", "--out", "o", "--effort", "exhaustive", "--recipe-gate",
                                         "--no-regional-calibration"])
        solver.apply_policy(explicit, exhaustive)
        off = {"recipe_gate": False, "regional_calibration": True}
        self.assertTrue(solver.setting(explicit, off, "recipe_gate", exhaustive))      # the command line beats [run]
        self.assertFalse(solver.setting(explicit, off, "regional_calibration", exhaustive))
        self.assertTrue(self._gate(explicit, exhaustive, off))

    def test_a_resume_under_another_effort_keeps_the_starting_values(self):
        from n64dk.commands import run
        out = _scratch(self)
        start = _parser().parse_args(["--project", "/p", "--out", str(out)])
        run.prepare_invocation(start)
        run.save_invocation(start, out)
        saved = json.loads((out / "solver" / "invocation.json").read_text())
        self.assertEqual(saved["mode_settings"], {"recipe_gate": True, "regional_calibration": False})
        again = _parser().parse_args(["--resume", str(out), "--effort", "exhaustive"])
        mode = run.prepare_invocation(again)
        self.assertEqual(mode.name, "exhaustive")
        self.assertTrue(solver.setting(again, {}, "recipe_gate", mode))                # normal's gate, not exhaustive's
        self.assertFalse(solver.setting(again, {}, "regional_calibration", mode))
        self.assertTrue(self._gate(again, mode, {}))
        self.assertFalse(solver.setting(again, {"recipe_gate": False}, "recipe_gate", mode))   # [run] still wins
        # and the other way: an exhaustive run continued as normal keeps the gate off
        kitchen = _scratch(self)
        first = _parser().parse_args(["--project", "/p", "--out", str(kitchen), "--effort", "exhaustive"])
        run.prepare_invocation(first)
        run.save_invocation(first, kitchen)
        lighter = _parser().parse_args(["--resume", str(kitchen), "--effort", "normal"])
        mode = run.prepare_invocation(lighter)
        self.assertEqual((mode.name, solver.setting(lighter, {}, "recipe_gate", mode),
                          solver.setting(lighter, {}, "regional_calibration", mode)), ("normal", False, True))

    def test_an_invocation_saved_before_the_settings_resumes_with_its_own_mode(self):
        from n64dk.commands import run
        out = _scratch(self)
        first = _parser().parse_args(["--project", "/p", "--out", str(out), "--effort", "exhaustive"])
        run.prepare_invocation(first)
        run.save_invocation(first, out)
        path = out / "solver" / "invocation.json"
        saved = json.loads(path.read_text())
        del saved["mode_settings"]                                # as a round-11 run saved it
        path.write_text(json.dumps(saved))
        again = _parser().parse_args(["--resume", str(out), "--effort", "deep"])
        mode = run.prepare_invocation(again)
        self.assertEqual(mode.name, "deep")
        self.assertEqual(again.mode_settings, {"recipe_gate": False, "regional_calibration": True})


if __name__ == "__main__":
    unittest.main()


class LedgerAndSieveTest(unittest.TestCase):
    def _cases(self):
        drafted = _case("a", 0x80001000, status="exact", score=0,
                        stages=[{"stage": "drafts", "variants": [{"label": "plain", "score": 0}]}])
        crossed = _case("b", 0x80001040, status="exact", score=0,
                        stages=[{"stage": "drafts", "variants": [{"label": "plain", "score": 30}]},
                                {"stage": "drafts-x-recipes", "tried": 12, "chosen": "kmc -O1", "score": 0}])
        searched = _case("c", 0x80001080, status="exact", score=0,
                         stages=[{"stage": "drafts", "variants": [{"label": "plain", "score": 6}]},
                                 {"stage": "iterate", "builds": 40, "start": 6, "final": 0,
                                  "helpful_families": ["lifetime", "types"]}])
        left = _case("d", 0x800010C0, status="open", score=22,
                     stages=[{"stage": "drafts", "variants": [{"label": "plain", "score": 22}]},
                             {"stage": "iterate-skipped", "ceiling": 20, "score": 22}])
        return [drafted, crossed, searched, left]

    def test_sieve_accounts_for_every_case_once(self):
        table = solver.sieve(self._cases(), recovered=5)
        settled = {name.split(" ")[0]: row["settled"] for name, row in table["stages"].items()}
        self.assertEqual((settled["3"], settled["4"], settled["2"], settled["7"]), (5, 1, 1, 1))
        self.assertEqual(table["unresolved"], 1)
        self.assertEqual(sum(settled.values()) - 5 + table["unresolved"], 4)

    def test_attempt_ledger_has_every_method_and_a_final_row(self):
        observations = solver.Observations()
        observations.facts["main:800010C0:40"] = [{"technique": "search", "round": 2, "outcome": "no-change",
                                                   "builds": 30, "calls": 0, "digest": "dd"}]
        rows = solver.attempts(self._cases(), observations)
        left = [r for r in rows if r["name"] == "d"]
        self.assertEqual([r["method"] for r in left], ["drafts", "iterate-skipped", "search", "final"])
        self.assertEqual(left[-1]["outcome"]["status"], "open")
        self.assertEqual(rows, solver.attempts(self._cases(), observations))      # rebuilt identically
