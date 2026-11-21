"""The unresolved dossier (match.dossier), reason retry kinds (match.reasons.RETRY) and the declare
pass's deduplication (commands.run.declare_assumed): records only, no toolchain, no ROM."""
import argparse
import inspect
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from n64dk.commands import export as exportmod
from n64dk.commands import run as runmod
from n64dk.match import dossier
from n64dk.match import iterate as itermod
from n64dk.match import reasons
from n64dk.match import resolve as resolvemod
from n64dk.match import solver


def _case(status="open", *, vram=0x80001000, size=0x40, score=10, top="register.permutation", stages=(),
          recipe="ido5.3 -O2", strict=None, digest="d0", closed=None, next_step="", image="main"):
    return resolvemod.Case(resolvemod.case_id(image, vram, size), image, f"func_{vram:08X}", vram, size, "t" * 64,
                           status=status, top_cause=top, causes=[{"cause": top, "confidence": 0.9, "next": "hint"}]
                           if top else [], best_score=score, best_digest=digest, stages=list(stages),
                           closed_families=closed or {}, recipe=recipe, strict=strict or {}, next=next_step)


def _scratch(test) -> Path:
    root = Path(tempfile.mkdtemp(prefix="n64dk-dossier-test-"))
    test.addCleanup(shutil.rmtree, root, True)
    return root


def _states(rows) -> dict:
    return {row["technique"]: row["state"] for row in rows}


class ExclusionTest(unittest.TestCase):
    """Every exclusion the solver writes maps to a ledger state (a guard against drift)."""

    def excluded(self, case, mode, *, model=False, observations=None):
        decision = solver.decide(case, mode, observations or solver.Observations(), model=model)
        return decision.cause, decision.excluded

    def test_each_solver_exclusion_has_its_state(self):
        quick, normal, deep = solver.MODES["quick"], solver.MODES["normal"], solver.MODES["deep"]
        found = []
        # not in the mode: draft-options is exhaustive's only
        _cause, excluded = self.excluded(_case("does-not-build", score=None, top="source.missing"), quick)
        found.append(("draft-options", excluded["draft-options"], quick, "not-in-mode"))
        self.assertIn("exhaustive", dossier.classify_exclusion("draft-options", excluded["draft-options"], quick)[1])
        # not run on this status
        _cause, excluded = solver.applicable(_case("assumed", score=0), "register-only residual", normal)
        found.append(("search", excluded["search"], normal, "inapplicable"))
        # larger than 0x400
        _cause, excluded = self.excluded(_case(size=0x800, score=100), normal)
        found.append(("recipe-sweep", excluded["recipe-sweep"], normal, "inapplicable"))
        # above the search ceiling
        _out, excluded = solver.applicable(_case(score=100), "wrong load width", normal)
        found.append(("search", excluded["search"], normal, "inapplicable"))
        # every family closed for the best source
        _out, excluded = solver.applicable(_case(closed={"d0": list(itermod.FAMILIES)}), "register-only residual",
                                           normal)
        found.append(("search", excluded["search"], normal, "exhausted"))
        # no refiner; refiner already asked
        _out, excluded = solver.applicable(_case(), "register-only residual", normal)
        found.append(("refine", excluded["refine"], normal, "inapplicable"))
        _out, excluded = solver.applicable(_case(stages=[{"stage": "refine"}]), "register-only residual", normal,
                                           model=True)
        found.append(("refine", excluded["refine"], normal, "exhausted"))
        # declare without a self-describing name
        _out, excluded = solver.applicable(_case("assumed", score=0, strict={"assumptions": [
            {"kind": "symbol", "name": "gPlayer", "address": "0x80050000"}]}), "link/address mismatch", deep)
        found.append(("declare", excluded["declare"], deep, "inapplicable"))
        # a negative fact on the same source
        observations = solver.Observations()
        case = _case()
        observations.facts[case.case_id] = [{"technique": "search", "outcome": "no-change", "digest": "d0", "round": 2}]
        _cause, excluded = self.excluded(case, deep, observations=observations)
        found.append(("search", excluded["search"], deep, "exhausted"))
        # plateaued for its cause
        observations = solver.Observations()
        observations.cells[solver._key("register-only residual", "ido", "<=0x40", "search")] = {
            "attempts": 12, "exact": 0, "improved": 0, "builds": 100, "calls": 0}
        _cause, excluded = self.excluded(_case(), deep, observations=observations)
        found.append(("search", excluded["search"], deep, "exhausted"))
        # an ablation switch took it out of the mode
        ablated = runmod.ablated_policy(deep, ["round-search"])
        _cause, excluded = self.excluded(_case(), ablated)
        found.append(("search", excluded["search"], ablated, "switched-off"))
        for name, why, mode, expected in found:
            with self.subTest(why=why):
                self.assertEqual(dossier.classify_exclusion(name, why, mode)[0], expected)

    def test_a_switched_off_refiner_is_switched_off(self):
        state, _why = dossier.classify_exclusion("refine", "no refiner configured", solver.MODES["deep"], ["refine"])
        self.assertEqual(state, "switched-off")


class LadderTest(unittest.TestCase):
    def test_round_one_rungs_come_from_the_stages(self):
        case = _case(stages=[{"stage": "drafts", "variants": [{"label": "plain", "score": 30}, {"label": "c", "score": 12}]},
                             {"stage": "drafts-x-recipes", "tried": 32, "chosen": "ido7.1 -O2", "score": 12},
                             {"stage": "recipe-skipped", "reason": "no unoptimised-code evidence"},
                             {"stage": "iterate", "builds": 150, "start": 12, "final": 10}],
                     closed={"d0": list(itermod.FAMILIES)})
        rows = {row["technique"]: row for row in dossier.ladder(case)}
        self.assertEqual((rows["ladder:drafts"]["state"], rows["ladder:drafts"]["builds"]), ("attempted", 2))
        self.assertEqual(rows["ladder:drafts-x-recipes"]["builds"], 32)
        self.assertEqual(rows["ladder:recipe-retry"]["state"], "withheld")
        self.assertEqual((rows["ladder:search"]["state"], rows["ladder:search"]["builds"]), ("exhausted", 150))
        self.assertEqual(rows["ladder:twin"]["state"], "not-run")

    def test_a_large_function_and_a_far_off_draft(self):
        large = _case(size=0x800, score=400, stages=[{"stage": "iterate-skipped", "score": 400, "ceiling": 1}])
        states = _states(dossier.ladder(large) + [dossier.structural(large)])
        self.assertEqual(states["ladder:search"], "inapplicable")
        self.assertEqual(states["ladder:drafts-x-recipes"], "inapplicable")
        self.assertEqual(states[dossier.STRUCTURAL], "unavailable")
        far = _case(score=90, stages=[{"stage": "iterate-skipped", "score": 90, "ceiling": 40}])
        self.assertEqual(_states(dossier.ladder(far))["ladder:search"], "inapplicable")
        self.assertEqual(dossier.structural(far)["state"], "inapplicable")

    def test_switches_and_target_findings(self):
        states = _states(dossier.ladder(_case(), disabled=["search", "frame-fit", "draft"]))
        self.assertEqual((states["ladder:search"], states["ladder:frame-fit"], states["ladder:drafts"]),
                         ("switched-off",) * 3)
        blocked = dossier.ladder(_case("blocked-boundary", top="boundary.embedded_data"))
        self.assertEqual({row["state"] for row in blocked}, {"inapplicable"})
        self.assertIn("boundary.embedded_data", blocked[0]["why"])

    def test_later_rounds_from_the_hard_queue_record(self):
        record = {"primary_cause": "register-only residual", "ranked_actions": [{"technique": "search"}],
                  "excluded_actions": {"refine": "no refiner configured"},
                  "solver_attempts": [{"technique": "search", "builds": 40, "outcome": "improved"}]}
        rows = {row["technique"]: row for row in dossier.later(_case(), record, solver.MODES["deep"])}
        self.assertEqual((rows["search"]["state"], rows["search"]["attempts"], rows["search"]["builds"]),
                         ("eligible", 1, 40))
        self.assertEqual(rows["refine"]["state"], "inapplicable")
        self.assertEqual(rows["declare"]["state"], "inapplicable")          # does not act on its cause
        self.assertIn("register-only residual", rows["declare"]["why"])


class CostTest(unittest.TestCase):
    def stats(self):
        cells = {solver._key("register-only residual", "ido", "<=0x40", "search"):
                 {"attempts": 4, "exact": 1, "improved": 2, "builds": 200, "calls": 0},
                 solver._key("wrong stack frame", "gcc", "<=0x100", "search"):
                 {"attempts": 1, "exact": 0, "improved": 0, "builds": 100, "calls": 0}}
        rounds = [{"kind": "round", "techniques": {"search": {"cases": 5, "builds": 300}}},
                  {"kind": "round", "techniques": {"sweep": {"cases": 10, "builds": 900}}}]
        profile = [{"stage": "round:search", "wall_s": 30.0, "builds": {"requested": 300}},
                   {"stage": "resolve", "wall_s": 90.0, "builds": {"requested": 900}}]
        return dossier.technique_stats(cells, rounds, profile)

    def test_measured_builds_and_seconds(self):
        stats = self.stats()
        search = stats["techniques"]["search"]
        self.assertEqual((search["attempts"], search["builds"], search["seconds_per_build"]), (5, 300, 0.1))
        self.assertEqual(stats["techniques"]["sweep"]["builds_per_attempt"], 90.0)
        cost = dossier.expected_cost(stats, "search", "register-only residual", "ido", "<=0x40")
        self.assertEqual((cost["builds"], cost["seconds"], cost["attempts"]), (50.0, 5.0, 4))
        self.assertIn("same cause, recipe family and size", cost["basis"])
        other = dossier.expected_cost(stats, "search", "branch topology", "ido", "<=0x40")
        self.assertEqual((other["builds"], other["basis"]), (60.0, "this run: every case"))

    def test_never_invented(self):
        cost = dossier.expected_cost(self.stats(), "recipe-sweep", "recipe uncertain", "ido", "<=0x40")
        self.assertEqual((cost["builds"], cost["seconds"]), (dossier.UNMEASURED, dossier.UNMEASURED))
        empty = dossier.technique_stats({}, [], [])
        self.assertEqual(empty, {"techniques": {}, "by_cause": {}, "cells": {}})
        # builds measured, seconds not (no profile record): seconds stay unmeasured
        stats = dossier.technique_stats({solver._key("c", "ido", "<=0x40", "search"):
                                         {"attempts": 1, "builds": 10, "calls": 0}}, [], [])
        self.assertEqual(dossier.expected_cost(stats, "search", "c", "ido", "<=0x40")["seconds"], dossier.UNMEASURED)


class EntryTest(unittest.TestCase):
    def test_a_round_limited_case_can_retry_now_at_a_measured_cost(self):
        stats = CostTest().stats()
        record = {"primary_cause": "register-only residual", "next_action": "search",
                  "why_stopped": "round limit: effort normal allows 1 round(s)",
                  "ranked_actions": [{"technique": "search"}], "excluded_actions": {"refine": "no refiner configured"},
                  "reopen_on": ["a caller, callee or catalogue neighbour becomes exact"]}
        entry = dossier.case_entry(_case(), record, solver.MODES["normal"], stats)
        self.assertEqual((entry["reason_code"], entry["retry"]), ("round-limit", "now"))
        self.assertEqual(entry["eligible"][0]["technique"], "search")
        self.assertEqual(entry["eligible"][0]["cost"]["builds"], 50.0)
        self.assertIn("a caller, callee or catalogue neighbour becomes exact", entry["evidence_change"])
        self.assertEqual([row["technique"] for row in entry["ledger"]], list(dossier.TECHNIQUES))
        self.assertTrue({row["state"] for row in entry["ledger"]} <= set(dossier.STATES))

    def test_the_evidence_change_names_the_case_detail(self):
        assumed = _case("assumed", score=0, strict={"verdict": "assumed", "assumptions": [
            {"kind": "symbol", "name": "D_80123456", "address": "0x80123456"}]})
        entry = dossier.case_entry(assumed, {"primary_cause": "link/address mismatch"}, solver.MODES["normal"],
                                   dossier.technique_stats({}, [], []))
        self.assertEqual((entry["reason_code"], entry["retry"]), ("needs-symbol", "evidence"))
        self.assertEqual(entry["evidence_change"][0], "a declaration of D_80123456 at 0x80123456")
        far = _case(score=120, top="control.branch_form",
                    stages=[{"stage": "iterate-skipped", "score": 120, "ceiling": 40}])
        entry = dossier.case_entry(far, {"primary_cause": "large far-off"}, solver.MODES["normal"], {})
        self.assertEqual(entry["reason_code"], "far-off-draft")
        self.assertEqual(entry["evidence_change"][0], "a draft scoring at most 40 (best is 120)")
        region = {"primary_cause": "recipe uncertain", "toolchain_region": {"range": "0x80001000..0x80002000",
                                                                             "component_mismatches": ["assembler"]}}
        entry = dossier.case_entry(_case(top="compiler.family"), region, solver.MODES["normal"], {})
        self.assertEqual(entry["reason_code"], "no-recipe-identified")
        self.assertTrue(entry["evidence_change"][0].startswith("a recipe for region 0x80001000..0x80002000"))

    def test_not_a_c_target_is_never_retried(self):
        entry = dossier.case_entry(_case(), {"primary_cause": "register-only residual",
                                             "ranked_actions": [{"technique": "search"}]}, solver.MODES["normal"], {},
                                   unit_class="rsp")
        self.assertEqual((entry["reason_code"], entry["retry"]), ("rsp-microcode", "never"))

    def test_entries_without_a_case(self):
        entry = dossier.other_entry("main", "f", 0x80002000, 0x40, outcome="not attempted",
                                    reason="larger than the run's --max-size 0x200")
        self.assertEqual((entry["reason_code"], entry["retry"], entry["ledger"]), ("too-large", "evidence", []))
        donor = dossier.other_entry("main", "g", 0x80003000, 0x40, outcome="donor-awaiting-symbols",
                                    awaiting={".rodata": "0x80335B90"})
        self.assertEqual(donor["reason_code"], "needs-symbol")
        self.assertEqual(donor["evidence_change"][0], "a declared base of section .rodata (0x80335B90 was inferred)")


class BuildTest(unittest.TestCase):
    def build(self, mode=None):
        cases = [_case(), _case("exact", vram=0x80001100, score=0),
                 _case(vram=0x80001200, size=0x800, score=500, stages=[{"stage": "iterate-skipped", "score": 500,
                                                                         "ceiling": 1}])]
        records = {"main": {0x80001000: {"primary_cause": "register-only residual", "next_action": "search",
                                         "why_stopped": "round limit", "ranked_actions": [{"technique": "search"}]},
                            0x80001200: {"primary_cause": "large far-off", "why_stopped": "fixed point",
                                         "excluded_actions": {"recipe-sweep": "larger than 0x400: drafted"}}}}
        coverage = {"main": [{"name": "func_80001000", "vram": "0x80001000", "size": 0x40, "attempted": True},
                             {"name": "small", "vram": "0x80001300", "size": 0x4, "attempted": False,
                              "reason": "smaller than the resolver's minimum of 4 bytes"},
                             {"name": "lib", "vram": "0x80001400", "size": 0x40, "attempted": False,
                              "reason": "recovered from a donor (proven source of another title)"},
                             {"name": "waits", "vram": "0x80001500", "size": 0x40, "attempted": False,
                              "reason": "recovered from a donor (proven source of another title)"}]}
        donors = {"main": {0x80001500: {"D_80200000": "0x80200000"}}}
        rounds = [{"kind": "stop", "image": "main", "reason": "round limit: effort deep allows 4 round(s)"}]
        return dossier.build(cases=cases, records=records, coverage=coverage, donors=donors, cells={}, rounds=rounds,
                             profile=[], mode=mode or solver.MODES["deep"])

    def test_every_unsettled_function_once(self):
        found = self.build()
        names = [entry["name"] for entry in found["functions"]]
        self.assertEqual(names, ["func_80001000", "func_80001200", "small", "waits"])   # exact and donor left out
        summary = found["summary"]
        self.assertEqual(summary["functions"], 4)
        self.assertEqual(summary["by_reason"]["too-large"]["functions"], 1)
        self.assertEqual(summary["large_functions"], {"cases": 1, "unavailable": 1, "search_inapplicable": 1})
        self.assertEqual(summary["eligible"]["search"]["functions"], 1)
        self.assertEqual(summary["eligible"]["search"]["unmeasured"], 1)
        self.assertFalse(summary["exhaustion"]["complete"])
        self.assertEqual(summary["exhaustion"]["stops"], {"main": "round limit: effort deep allows 4 round(s)"})
        # one case's cause has no use for it, the other is larger than 0x400
        self.assertEqual(summary["ledger"]["recipe-sweep"].get("inapplicable"), 2)
        self.assertEqual(summary["ledger"][dossier.STRUCTURAL], {"inapplicable": 1, "unavailable": 1})
        json.dumps(found)                                              # serialisable as it stands

    def test_markdown(self):
        text = dossier.markdown(self.build())
        for heading in ("# Unresolved functions", "## Why", "## Eligible next techniques", "## Techniques per case",
                        "## Exhaustion", "## Functions"):
            self.assertIn(heading, text)
        self.assertIn("| main | func_80001000 | 0x80001000 | 0x40 | round-limit | search (unmeasured) | now |", text)
        self.assertNotIn("\t", text)
        self.assertIn("no unresolved dossier", dossier.markdown(None))
        many = {"mode": "normal", "disabled": [], "summary": self.build()["summary"],
                "functions": [dossier.other_entry("main", f"f{i}", 0x80000000 + 4 * i, 4, outcome="not attempted",
                                                  reason="smaller than the resolver's minimum") for i in range(501)]}
        rendered = dossier.markdown(many).splitlines()
        self.assertEqual(sum(line.startswith("| main | f") for line in rendered), dossier._MARKDOWN_ROWS)
        self.assertIn("500 of 501 shown", "\n".join(rendered))


class FilesTest(unittest.TestCase):
    def test_collect_write_and_load_from_a_run_directory(self):
        run = _scratch(self)
        store = resolvemod.CaseStore(run / "state")
        store.put(_case())
        store.close()
        (run / "solver").mkdir()
        (run / "solver" / "invocation.json").write_text(json.dumps({"effort": "deep", "disabled": ["twin"]}))
        (run / "solver" / "hard_queue_main.json").write_text(json.dumps([
            {"vram": "0x80001000", "primary_cause": "register-only residual", "ranked_actions": [{"technique": "search"}],
             "next_action": "search", "why_stopped": "budget: build budget 10 reached"}]))
        (run / "solver" / "coverage_main.json").write_text(json.dumps({"entries": [
            {"name": "out", "vram": "0x80009000", "size": 0x40, "attempted": False,
             "reason": "no discovered image: the range is outside the image's bytes"}]}))
        (run / "solver" / "observations.json").write_text(json.dumps({"cells": {
            solver._key("register-only residual", "ido", "<=0x40", "search"): {"attempts": 2, "builds": 80, "calls": 0}}}))
        (run / "rounds.jsonl").write_text(json.dumps({"kind": "round", "techniques": {"search": {"cases": 2}}})
                                          + "\n{torn line\n")
        (run / "profile.jsonl").write_text(json.dumps({"stage": "round:search", "wall_s": 8.0,
                                                       "builds": {"requested": 80}}) + "\n")
        headline = dossier.write(run)
        self.assertEqual((headline["functions"], headline["cases_with_eligible_left"]), (2, 1))
        found = dossier.load(run)
        self.assertEqual((found["mode"], found["disabled"]), ("deep", ["twin"]))
        first = found["functions"][0]
        self.assertEqual((first["reason_code"], first["eligible"][0]["cost"]["builds"],
                          first["eligible"][0]["cost"]["seconds"]), ("budget-exhausted", 40.0, 4.0))
        self.assertEqual(found["functions"][1]["reason_code"], "outside-image")
        self.assertIsNone(dossier.load(_scratch(self)))

    def test_hooks_are_report_only(self):
        source = inspect.getsource(runmod._decompile_run)
        self.assertIn("dossiermod.write(out, mode", source)
        self.assertIn("observation_error", source[source.index("dossiermod.write"):][:600])
        export_source = inspect.getsource(exportmod.export)
        self.assertIn("UNRESOLVED.md", export_source)
        self.assertIn("dossiermod.load(run) or dossiermod.collect(run", export_source)
        text = runmod._markdown({"project": "p", "images": {}, "unresolved": {
            "file": "x", "functions": 3, "by_retry": {"now": 1}, "cases_with_eligible_left": 1}})
        self.assertIn("Unresolved dossier: 3 functions not settled", text)
        self.assertEqual(exportmod._scrub(dossier.markdown(None), []), dossier.markdown(None))


class RetryTest(unittest.TestCase):
    def test_every_code_has_a_retry_kind_and_change(self):
        self.assertEqual(set(reasons.RETRY), set(reasons.CODES))
        for code, (when, change) in reasons.RETRY.items():
            self.assertIn(when, ("now", "evidence", "never"), code)
            self.assertTrue(change and "\t" not in change, code)
        self.assertEqual(reasons.retry_of(""), ("", ""))


class DeclareDedupTest(unittest.TestCase):
    """A case verified against the same declarations, source and recipe is not built again."""

    def setUp(self):
        self.root = _scratch(self)
        self.calls = []

    def put(self, vram, name):
        store = resolvemod.CaseStore(self.root / "state")
        try:
            store.put(_case("assumed", vram=vram, score=0, digest=f"s{vram:X}", strict={"verdict": "assumed",
                      "assumptions": [{"kind": "symbol", "name": name, "address": "0x" + name[2:]}]}))
        finally:
            store.close()

    def declare(self, verified):
        def resolve_hard(options):
            self.calls.append(options)
            return {"statuses": {"assumed": 1}}

        def options(**overrides):
            return argparse.Namespace(symbols=[], **overrides)
        return runmod.declare_assumed(resolve_hard, options, self.root / "state", "main", self.root / "main", {},
                                      verified)

    def test_an_address_named_callee_is_declared_and_another_name_is_not(self):
        store = resolvemod.CaseStore(self.root / "state")
        try:
            store.put(_case("assumed", vram=0x80001000, score=0, digest="s1", strict={"verdict": "assumed", "assumptions": [
                {"kind": "symbol", "name": "func_80003000", "address": "0x80003000", "call": True},
                {"kind": "symbol", "name": "D_80200000", "address": "0x80200000"},
                {"kind": "symbol", "name": "osSendMesg", "address": "0x80004000", "call": True},
                {"kind": "symbol", "name": "func_80005000", "address": "0x80005004", "call": True}]}))
        finally:
            store.close()
        result = self.declare({})
        text = (self.root / "main" / "declared_symbols.txt").read_text()
        self.assertEqual(result["declared"], 2)
        self.assertEqual(text, "func_80003000 = 0x80003000;\nD_80200000 = 0x80200000;\n")

    def test_repeats_are_skipped_and_changes_reverified(self):
        verified = {}
        self.put(0x80001000, "D_80200000")
        self.put(0x80001100, "D_80200000")
        self.declare(verified)
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(hasattr(self.calls[0], "only_names"))         # the first pass is today's pass
        self.assertEqual(len(verified), 2)
        result = self.declare(verified)                                  # nothing changed: no pass, no build
        self.assertEqual((len(self.calls), result["unchanged"]), (1, 2))
        self.put(0x80001200, "D_80200000")                               # a new case, same declarations
        self.declare(verified)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual((self.calls[1].only_names, self.calls[1].write_scope), (["func_80001200"], False))
        self.put(0x80001300, "D_80200400")                               # new declarations: every case again
        self.declare(verified)
        self.assertEqual(len(self.calls), 3)
        self.assertFalse(hasattr(self.calls[2], "only_names"))

    def test_without_a_memo_every_pass_verifies_every_case(self):
        self.put(0x80001000, "D_80200000")
        self.declare(None)
        self.declare(None)
        self.assertEqual(len(self.calls), 2)


if __name__ == "__main__":
    unittest.main()
