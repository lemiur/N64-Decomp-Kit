"""The multi-hypothesis world model (``n64dk.match.worldmodel``): stable ids, evidence and
contradictions with provenance, the beam, settlement by strict-exact results with propagation,
the queries, persistence behind the evidence store, and the read-only import of a synthetic run
directory.  No toolchain, no ROM."""
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from n64dk.match import evidence_store as es
from n64dk.match import resolve as resolvemod
from n64dk.match import worldmodel as wm


def _ev(key, weight=0.5, basis="heuristic", note=""):
    return wm.evidence(key, "test", basis=basis, weight=weight, note=note)


def _case(name, vram, *, image="main", status="open", size=0x40, recipe="ido5.3 -O2", **extra):
    return resolvemod.Case(resolvemod.case_id(image, vram, size), image, name, vram, size, "t" + name,
                           status=status, best_score=0 if status == "exact" else 12, recipe=recipe, **extra)


def _snapshot(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns,
                                             hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(root.rglob("*")) if p.is_file()}


class WorldModelTest(unittest.TestCase):
    def _model(self, **kw):
        """A catalogue premise, two recipe alternatives for one region, and a hypothesis resting on
        the leading recipe."""
        model = wm.WorldModel(**kw)
        premise = model.add("catalogue", "main:catalogue", {"verdict": "complete"}, confidence=0.7,
                            evidence=[_ev("report.json#/discovery", 0.7, "target-derived")])
        ido = model.add("recipe", "main:region:A", {"recipe": "ido5.3 -O2"}, confidence=0.6,
                        evidence=[_ev("regions#A", 0.6, "target-derived")], depends_on=[premise.id],
                        cases=["c1", "c2"], cache_keys=["main/stage_calibrate.json"],
                        consequences=[{"case": "c1", "predicts": "exact", "recipe": "ido5.3 -O2"}],
                        cost_to_test={"builds": 40, "note": "calibrate the region"})
        kmc = model.add("recipe", "main:region:A", {"recipe": "kmc+as -O2"}, confidence=0.3,
                        evidence=[_ev("regions#A", 0.3, "target-derived", "second")], depends_on=[premise.id],
                        cases=["c1", "c2"])
        context = model.add("draft-context", "main:c3", {"prototypes": "from region A"}, confidence=0.6,
                            depends_on=[ido.id], cases=["c3"], cache_keys=["k3"])
        return model, premise, ido, kmc, context

    def test_ids_are_content_hashes_and_adding_again_merges(self):
        model, premise, ido, _kmc, _context = self._model()
        self.assertEqual(ido.id, wm.hypothesis_id("recipe", "main:region:A", {"recipe": "ido5.3 -O2"}))
        self.assertEqual(ido.id, wm.WorldModel().add("recipe", "main:region:A", {"recipe": "ido5.3 -O2"}).id)
        self.assertNotEqual(ido.id, wm.hypothesis_id("recipe", "main:region:B", {"recipe": "ido5.3 -O2"}))
        merged = model.add("recipe", "main:region:A", {"recipe": "ido5.3 -O2"}, cases=["c9"],
                           evidence=[_ev("calibration", 0.5, "derived")])
        self.assertEqual(merged.id, ido.id)
        self.assertEqual(merged.cases, ("c1", "c2", "c9"))
        self.assertAlmostEqual(merged.confidence, 1 - 0.4 * 0.5)            # noisy-or
        self.assertEqual([e["key"] for e in merged.evidence], ["regions#A", "calibration"])
        self.assertEqual(merged.status, "open")
        with self.assertRaises(KeyError):
            model.add("x", "y", 1, depends_on=["h-unknown"])
        with self.assertRaises(ValueError):                                  # no cycles
            model.add("catalogue", "main:catalogue", {"verdict": "complete"}, depends_on=[ido.id])
        with self.assertRaises(ValueError):
            wm.evidence("k", "s", basis="proof")
        heuristic = model.add("x", "capped", True, confidence=0.9, evidence=[_ev("a", 0.9)])
        heuristic = model.add_evidence(heuristic.id, _ev("b", 1.0))
        self.assertEqual(heuristic.confidence, wm.HEURISTIC_CAP)   # only a strict-exact result reaches 1
        self.assertEqual(heuristic.status, "supported")

    def test_a_contradiction_lowers_confidence_splits_and_never_overwrites(self):
        model, _premise, ido, _kmc, _context = self._model()
        after = model.add_contradiction(ido.id, _ev("marks#A", 0.5, "target-derived", "unknown marks"),
                                        alternatives=[{"recipe": "sn 2.7.2 -O2"}])
        self.assertAlmostEqual(after.confidence, 0.3)
        self.assertEqual(after.status, "split")
        self.assertEqual([e["key"] for e in after.evidence], ["regions#A"])  # evidence kept
        self.assertEqual(after.contradictions[0]["note"], "unknown marks")
        sn = model.get(wm.hypothesis_id("recipe", "main:region:A", {"recipe": "sn 2.7.2 -O2"}))
        self.assertEqual((sn.split_from, sn.cases, sn.depends_on), (ido.id, ido.cases, ido.depends_on))
        self.assertEqual(sn.evidence[0]["basis"], "derived")
        history = model.history(ido.id)
        self.assertEqual([h["status"] for h in history], ["open", "split"])
        self.assertEqual(history[-1]["event"], "contradiction")
        self.assertEqual(len(model.repository.history(wm.KIND, ido.id)), 2)

    def test_the_beam_retires_the_weakest_and_restores_one_that_climbs(self):
        model = wm.WorldModel(beam=2)
        low = model.add("image-base", "seg:window", {"base": "0x80100000"}, confidence=0.3)
        mid = model.add("image-base", "seg:window", {"base": "0x80200000"}, confidence=0.5)
        high = model.add("image-base", "seg:window", {"base": "0x80300000"}, confidence=0.7)
        self.assertEqual([model.get(h.id).status for h in (low, mid, high)], ["retired", "open", "open"])
        model.add_evidence(low.id, _ev("calls", 0.9, "target-derived"))
        self.assertEqual([model.get(h.id).status for h in (low, mid, high)], ["supported", "retired", "open"])
        self.assertEqual(model.leader("image-base", "seg:window").id, low.id)
        self.assertEqual(len(model.alternatives("seg:window")), 3)            # retired, never deleted

    def test_a_failed_operation_leaves_the_model_as_the_store_left_it(self):
        model, _premise, ido, _kmc, _context = self._model()
        before = model.export()
        with self.assertRaises(RuntimeError):
            with model._atomic():
                model.add("x", "y", 1)
                model.add_contradiction(ido.id, _ev("marks#A", 0.9))
                raise RuntimeError("simulated failure")
        self.assertEqual(model.export(), before)
        self.assertEqual(len(model.history(ido.id)), 1)

    def test_settle_supports_without_reopening_anything(self):
        model, _premise, ido, _kmc, _context = self._model()
        result = model.settle({"case": "c1", "verdict": "exact", "key": "cases#c1", "recipe": "ido5.3  -O2"})
        self.assertEqual(result.settled, (ido.id,))
        self.assertEqual((result.reopen, result.invalidate), ((), ()))
        self.assertEqual(model.get(ido.id).status, "supported")
        self.assertEqual(model.get(ido.id).evidence[-1]["basis"], "strict-exact")

    def test_settle_contradicts_propagates_and_returns_work_to_redo(self):
        model, _premise, ido, kmc, context = self._model()
        result = model.settle({"case": "c1", "verdict": "exact", "key": "cases#c1", "source": "strict verifier",
                               "recipe": "kmc+as -O2"})
        self.assertEqual(result.settled, (ido.id,))
        self.assertEqual(model.get(ido.id).status, "contradicted")
        self.assertEqual(model.get(ido.id).confidence, 0.0)
        self.assertEqual(model.get(ido.id).contradictions[0]["basis"], "strict-exact")
        dependent = model.get(context.id)
        self.assertEqual(dependent.contradictions[0]["key"], "hypothesis:" + ido.id)
        self.assertAlmostEqual(dependent.confidence, 0.3)
        self.assertEqual(result.reopen, ("c2", "c3"))                           # c1 itself is exact
        self.assertEqual(result.invalidate, ("k3", "main/stage_calibrate.json"))
        self.assertEqual(set(result.changed), {ido.id, context.id})
        self.assertEqual(model.leader("recipe", "main:region:A").id, kmc.id)
        nothing = model.settle({"case": "c2", "verdict": "different", "key": "cases#c2"})
        self.assertEqual((nothing.settled, nothing.reopen), ((), ()))
        self.assertIn("not strict-exact", nothing.note)

    def test_queries(self):
        model, premise, ido, kmc, context = self._model()
        (why,) = model.why("main:region:A")
        self.assertEqual(why["believed"], {"recipe": "ido5.3 -O2"})
        self.assertEqual(why["because"]["evidence"][0]["key"], "regions#A")
        self.assertEqual(why["because"]["premises"][0]["id"], premise.id)
        self.assertEqual([a["id"] for a in why["alternatives"]], [kmc.id])
        self.assertEqual(model.why("nothing here"), [])
        self.assertEqual(model.dependents(premise.id), sorted([ido.id, kmc.id]))
        self.assertEqual(model.dependents(premise.id, transitive=True), sorted([ido.id, kmc.id, context.id]))
        before = model.export()
        found = model.what_if(ido.id)
        self.assertEqual(model.export(), before)                                # changes nothing
        self.assertTrue(found["is_leader"])
        self.assertEqual(found["next_leader"]["id"], kmc.id)
        self.assertEqual(found["dependents"], [context.id])
        self.assertEqual(found["cases"], ["c1", "c2", "c3"])
        self.assertEqual(found["cache_keys"], ["k3", "main/stage_calibrate.json"])
        self.assertEqual(found["predictions"], [{"hypothesis": ido.id, "case": "c1", "predicts": "exact"}])
        triggers = model.reopen_triggers("c3")
        self.assertEqual((triggers[0]["hypothesis"], triggers[0]["via"]), (context.id, None))
        self.assertEqual({(t["hypothesis"], t["via"]) for t in triggers[1:]},
                         {(ido.id, context.id), (premise.id, context.id)})
        self.assertTrue(any("settles the prediction" in t for row in model.reopen_triggers("c1")
                            for t in row["triggers"]))

    def test_the_model_lives_behind_the_evidence_store(self):
        root = Path(tempfile.mkdtemp(prefix="n64dk-worldmodel-test-"))
        self.addCleanup(shutil.rmtree, root, True)
        path = root / es.DATABASE
        repo = es.SQLiteRepository(path)
        try:
            model, premise, ido, _kmc, _context = self._model(repository=repo)
            model.add_contradiction(ido.id, _ev("marks#A", 0.2))
            exported = model.export()
            self.assertIn("hypothesis:" + premise.id, repo.latest(wm.KIND, ido.id).depends_on)
            marked = repo.invalidate("report.json#/discovery")      # an evidence key the premise cites
            self.assertEqual(marked[0], "hypothesis:" + premise.id)
            self.assertEqual(set(marked), {"hypothesis:" + h.id for h in (premise, ido, _kmc, _context)})
        finally:
            repo.close()
        inspect = es.SQLiteRepository(path, readonly=True)
        try:
            again = wm.WorldModel(inspect)
            self.assertEqual(again.export(), exported)
            self.assertEqual(again.leader("recipe", "main:region:A").id, ido.id)
            with self.assertRaises(es.StoreError):
                again.add("x", "y", 1)
        finally:
            inspect.close()


class FromRunTest(unittest.TestCase):
    """``from_run`` over a synthetic decompile-run directory."""

    def setUp(self):
        self.run = Path(tempfile.mkdtemp(prefix="n64dk-worldmodel-run-"))
        self.addCleanup(shutil.rmtree, self.run, True)
        for folder in ("solver", "discovery", "main", "small_data"):
            (self.run / folder).mkdir()
        report = {
            "status": "complete",
            "discovery": {"verdict": "complete", "verdict_before": "incomplete", "rom_reasons": [],
                          "placed": [{"name": "seg_1000", "rom": "0x1000", "rom_end": "0x2000", "vram": "0x80400000",
                                      "landed": 12, "functions": 5}],
                          "images": {"main": {"verdict": "incomplete", "verdict_after": "complete", "coverage": 0.9,
                                              "coverage_after": 1.0, "functions_after": 3, "repaired_functions": 1,
                                              "reasons_after": []}}},
            "small_data": {"value": "0x80012340", "confidence": "high", "sites": 2, "project_gp": "0x80099990",
                           "reason": "set at 2 sites", "warning": "the project declares another _gp"},
            "images": {"main": {"functions": 3, "calibrate": {"chosen": {"label": "ido5.3 -O2", "pipeline": "ido5.3",
                                                                          "flags": "-O2"}}}}}
        (self.run / "report.json").write_text(json.dumps(report))
        (self.run / "solver" / "coverage_main.json").write_text(json.dumps({"entries": [
            {"name": "f1", "vram": "0x80001000", "size": 64, "attempted": True, "status": "exact"},
            {"name": "f2", "vram": "0x80001040", "size": 32, "attempted": True, "status": "open"},
            {"name": "func_80001060", "vram": "0x80001060", "size": 32, "attempted": False, "reason": "x"}]}))
        (self.run / "discovery" / "repair_main.json").write_text(json.dumps({"added": [
            {"start": "0x80001060", "end": "0x80001080", "size": 32, "evidence": ["call target",
                                                                                  "leads into the next function"],
             "confidence": "high", "moved_from": "0x80001064"}]}))
        (self.run / "main" / "toolchain_regions.json").write_text(json.dumps({"regions": [
            {"start": "0x80001000", "end": "0x80001100", "verdict": "known", "proven": {"ido5.3 -O2": 1},
             "ranked": [{"recipe": "ido5.3 -O2", "share": 0.7}, {"recipe": "kmc+as -O2", "share": 0.2}],
             "component_mismatches": []}]}))
        (self.run / "main" / "stage_calibrate.json").write_text("{}")
        (self.run / "discovery" / "overlays.json").write_text(json.dumps({"main": {"current": "0x80000000", "runs": [
            {"start": "0x0", "end": "0x1000", "base": "0x80000000", "calls": 10, "landed": 8, "landed_at_current": 8},
            {"start": "0x1000", "end": "0x2000", "base": "0x80100000", "calls": 10, "landed": 9,
             "landed_at_current": 2}]}}))
        (self.run / "small_data" / "gp_symbols.txt").write_text("_gp = 0x80012340;\n")
        store = resolvemod.CaseStore(self.run / "state")
        try:
            store.put(_case("f1", 0x80001000, status="exact", strict={"verdict": "exact"}))
            store.put(_case("f2", 0x80001040, size=32, recipe="kmc+gas -O2 -G8"))
        finally:
            store.close()

    def test_imports_the_facts_a_run_holds_without_writing(self):
        before = _snapshot(self.run)
        model = wm.from_run(self.run)
        self.assertEqual(_snapshot(self.run), before)
        f1 = resolvemod.case_id("main", 0x80001000, 0x40)
        f2 = resolvemod.case_id("main", 0x80001040, 32)
        (start,) = model.why("main:0x80001000")
        self.assertEqual((start["believed"], start["because"]["status"]), ({"start": True}, "supported"))
        self.assertIn(f"state/cases.sqlite#{f1}", [e["key"] for e in start["because"]["evidence"]])
        (moved,) = model.why("main:0x80001064")
        self.assertEqual(moved["believed"], {"start": False})
        self.assertEqual(moved["alternatives"][0]["status"], "split")
        (added,) = model.why("main:0x80001060")
        self.assertEqual(added["because"]["evidence"][0]["source"], "discovery repair")
        (recipe,) = model.why("main:region:0x80001000..0x80001100")
        self.assertEqual(recipe["believed"], {"recipe": "ido5.3 -O2"})
        catalogue = wm.hypothesis_id("catalogue", "main:catalogue", {"verdict": "complete"})
        self.assertEqual(recipe["because"]["premises"][0]["id"], catalogue)
        affected = model.what_if(catalogue)
        self.assertIn(f1, affected["cases"])
        self.assertIn("main/stage_calibrate.json", affected["cache_keys"])
        (calibrated,) = model.why("main:image")
        self.assertEqual(calibrated["believed"], {"recipe": "ido5.3 -O2"})
        (gp,) = model.why("program:_gp")
        self.assertEqual(gp["believed"], {"value": "0x80099990"})               # the project's value is used
        recovered = model.get(wm.hypothesis_id("gp", "program:_gp", {"value": "0x80012340"}))
        self.assertEqual(recovered.contradictions[0]["basis"], "declared")
        self.assertEqual(recovered.cache_keys, ("small_data/gp_symbols.txt",))
        self.assertIn(gp["because"]["id"], [t["hypothesis"] for t in model.reopen_triggers(f2)])
        (window,) = model.why("main:window:0x1000..0x2000")
        self.assertEqual(window["believed"], {"base": "0x80100000"})
        (placed,) = model.why("seg_1000:image")
        self.assertEqual(placed["believed"], {"vram": "0x80400000"})
        exported = model.export()
        self.assertEqual(json.loads(json.dumps(exported)), exported)
        self.assertEqual(wm.from_run(self.run).export(), exported)               # deterministic


if __name__ == "__main__":
    unittest.main()
