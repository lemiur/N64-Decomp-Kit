import math
import unittest

from n64dk.match import activeid
from n64dk.match.activeid import ActiveId, Option, OutcomeTable, Probe

A = "kmc+as -O2 -G0 -mips3"
B = "kmc+asn64 -O2 -G0 -mips3"          # one compiler and level, another assembler: A's sibling
C = "ido5.3 -O2 -mips2"
F = "kmc+as -O0 -G0 -mips3"
OTHERS = ["ido5.3 -O1", "ido5.3 -O2", "ido5.3 -g", "ido7.1 -O1", "ido7.1 -O2", "ido7.1 -g", "custom:mipspro744 -O2"]


def build_under(truth):
    return lambda function, recipe: "exact" if recipe == truth else "far"


class LabelTest(unittest.TestCase):
    def test_parts_and_relations(self):
        self.assertEqual(activeid.parts("kmc+as -O2 -G0 -mips3"), ("kmc+as", "-O2", "-G0 -mips3"))
        self.assertEqual(activeid.class_key(C), "ido5.3 -O2")
        self.assertEqual(activeid.default_relation(A, A), "same")
        self.assertEqual(activeid.default_relation(A, B), "sibling")
        self.assertEqual(activeid.default_relation(A, F), "family")
        self.assertEqual(activeid.default_relation("ido5.3 -O2", "ido7.1 -O2"), "family")
        self.assertEqual(activeid.default_relation(A, C), "other")

    def test_bucket(self):
        self.assertEqual(activeid.bucket((True, 0)), "exact")
        self.assertEqual(activeid.bucket((False, 10)), "near")
        self.assertEqual(activeid.bucket((False, 500)), "far")
        self.assertEqual(activeid.bucket(False), "far")
        self.assertIsNone(activeid.bucket(None))


class PosteriorTest(unittest.TestCase):
    def test_update_is_bayes_rule(self):
        state = ActiveId([A, C])
        state.update(Option("build", "f", A), "exact")
        self.assertAlmostEqual(state.posterior[A], 0.45 / (0.45 + 0.01), places=9)
        state.update(Option("build", "f", A), "far")
        self.assertAlmostEqual(state.posterior[A], (0.45 * 0.2) / (0.45 * 0.2 + 0.01 * 0.9), places=9)
        self.assertAlmostEqual(sum(state.posterior.values()), 1.0, places=12)

    def test_prior_is_respected(self):
        state = ActiveId([A, C], prior={A: 3, C: 1})
        self.assertAlmostEqual(state.posterior[A], 0.75, places=12)

    def test_contradiction_is_absorbed_by_smoothing(self):
        hard = {"same": {"exact": 1.0, "near": 0.0, "far": 0.0}, "sibling": {"exact": 1.0, "near": 0.0, "far": 0.0},
                "family": {"exact": 0.0, "near": 0.0, "far": 1.0}, "other": {"exact": 0.0, "near": 0.0, "far": 1.0}}
        state = ActiveId([A, C], table=OutcomeTable(rows=hard))
        state.update(Option("build", "f", A), "exact")
        self.assertGreater(state.posterior[A], 0.99)
        self.assertGreater(state.posterior[C], 0.0)
        for name in ("g", "h"):                               # two builds contradict it
            state.update(Option("build", name, A), "far")
        self.assertTrue(all(p > 0 and math.isfinite(math.log(p)) for p in state.posterior.values()))
        self.assertGreater(state.posterior[C], state.posterior[A])

    def test_many_observations_do_not_underflow(self):
        state = ActiveId([A, C])
        for n in range(400):
            state.update(Option("build", f"f{n}", A), "exact")
        self.assertAlmostEqual(state.posterior[A], 1.0, places=9)
        self.assertTrue(math.isfinite(state.log[C]))                 # kept in log space: never zero

    def test_learned_table(self):
        history = [(A, A, "exact")] * 100 + [(A, C, "far")] * 10
        table = activeid.learn_table(history)
        self.assertAlmostEqual(table.rows["same"]["exact"], (100 + 20 * 0.45 + 1) / (100 + 20 + 3), places=9)
        self.assertIn(f"{A}|{A}", table.pairs)
        self.assertNotIn(f"{A}|{C}", table.pairs)               # 10 < min_pair
        self.assertAlmostEqual(sum(table.likelihood(A, C).values()), 1.0, places=12)
        again = OutcomeTable.from_dict(table.as_dict())
        self.assertEqual(again.likelihood(A, A), table.likelihood(A, A))


class SelectionTest(unittest.TestCase):
    def test_a_probe_separates_two_statically_tied_recipes(self):
        separating = Probe("P1", {A: "x", B: "y"})
        lumping = ("P2", {A: "x", B: "x", C: "z"})             # tells C from A and B only
        state = ActiveId([A, B, C], prior={A: 0.45, B: 0.45, C: 0.10}, scope="region 0x80100000")
        report = state.run([], lambda f, r: None, probes=[lumping, separating],
                           probe_fn=lambda p: {"P1": "x", "P2": "x"}[p.name])
        self.assertEqual(report["steps"][0]["probe"], "P1")
        self.assertTrue(report["steps"][0]["why"]["separates"])
        self.assertEqual(report["steps"][0]["why"]["predicts"]["y"], [B])
        self.assertEqual(report["decision"], A)
        self.assertEqual(report["stopped"], "confident")
        self.assertEqual((report["builds"], report["probes"]), (0, 2))
        self.assertEqual([c["hypothesis"] for c in report["candidates_before"]][:2], [A, B])
        self.assertFalse(report["proof"])

    def test_eig_prefers_the_separating_observation(self):
        state = ActiveId([A, B])
        blind = Probe("same-for-both", {A: "x", B: "x"})
        sharp = Probe("sharp", {A: "x", B: "y"})
        self.assertLess(state.gain(Option("probe", probe=blind))[0], 1e-9)
        self.assertFalse(state.why(Option("probe", probe=blind))["separates"])
        option, bits, _predictive = state.select([], [blind, sharp], set(), {})
        self.assertEqual(option.probe.name, "sharp")
        self.assertGreater(bits, 0.5)

    def test_eig_builds_where_the_mass_is(self):
        unlikely = "ido5.3 -g"
        state = ActiveId([unlikely, A, B], prior={unlikely: 0.001, A: 0.4995, B: 0.4995})
        option, _bits, _p = state.select(["f"], [], set(), {})
        self.assertIn(option.recipe, (A, B))

    def test_lane_mark_probes_and_region_prior(self):
        marks = {"kmc+as -O2": {"m1", "m2"}, "kmc+asn64 -O2": {"m1"}}
        probes = activeid.probes_from_lane_marks(marks, [A, B])
        self.assertEqual([p.name for p in probes], ["mark:m2"])
        self.assertEqual(probes[0].distinguishes, {A: "present", B: "absent"})
        prior = activeid.prior_from_region({"ranked": [{"recipe": "kmc+as -O2", "share": 0.8},
                                                       {"recipe": "ido5.3 -O2", "share": 0.2}]}, [A, B, C])
        self.assertAlmostEqual(sum(prior.values()), 1.0, places=12)
        self.assertGreater(prior[A], prior[C])
        self.assertGreater(prior[C], prior[B])
        self.assertEqual(activeid.prior_from_region(None, [A, B]), {A: 0.5, B: 0.5})


class StoppingTest(unittest.TestCase):
    def test_stops_when_nothing_is_informative(self):
        table = OutcomeTable(relation=lambda truth, built: "other")      # every recipe predicts the same
        report = ActiveId([A, B], table=table).run(["f", "g"], build_under(A))
        self.assertEqual(report["stopped"], "no observation is informative enough")
        self.assertEqual(report["builds"], 0)
        self.assertIsNone(report["decision"])

    def test_stops_at_the_budget(self):
        report = ActiveId([A, B, C], budget=2).run(["f", "g", "h"], lambda f, r: "near")
        self.assertEqual(report["stopped"], "budget")
        self.assertEqual(report["builds"], 2)

    def test_stops_when_nothing_is_left(self):
        report = ActiveId([A, B]).run([], lambda f, r: "far")
        self.assertEqual(report["stopped"], "nothing left to observe")

    def test_fewer_builds_than_uniform_calibration_for_the_same_certainty(self):
        candidates = OTHERS + [F, B, A]                        # the truth is tried last by the uniform order
        prior = {A: 0.4, B: 0.4, **{c: 0.025 for c in OTHERS + [F]}}     # statically tied A and B
        functions = [f"f{n}" for n in range(10)]
        report = ActiveId(candidates, prior=prior).run(functions, build_under(A), compare_uniform=True)
        self.assertEqual(report["stopped"], "confident")
        self.assertEqual(report["decision"], A)
        self.assertTrue(report["uniform"]["reached"])
        self.assertEqual(report["uniform"]["decision"], A)
        self.assertLess(report["builds"], report["uniform"]["builds"])
        self.assertEqual(report["builds_avoided"], report["uniform"]["builds"] - report["builds"])
        for step in report["steps"]:
            self.assertIn(step["outcome"], ("exact", "far"))

    def test_deterministic(self):
        candidates = OTHERS + [F, B, A]
        one = ActiveId(candidates).run(["f", "g", "h"], build_under(B))
        two = ActiveId(candidates).run(["f", "g", "h"], build_under(B))
        self.assertEqual(one, two)


if __name__ == "__main__":
    unittest.main()
