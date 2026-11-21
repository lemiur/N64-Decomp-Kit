import json
import math
import tempfile
import unittest
from pathlib import Path

from n64dk.match import calibration as cal


def repeat(score, positives, negatives, title=""):
    return [(score, 1, title)] * positives + [(score, 0, title)] * negatives


class ScoreTest(unittest.TestCase):
    def test_brier(self):
        self.assertEqual(cal.brier([(1.0, 1), (0.0, 0)]), 0.0)
        self.assertAlmostEqual(cal.brier([(0.5, 1), (0.5, 0)]), 0.25)

    def test_ece_of_a_calibrated_and_an_overconfident_score(self):
        self.assertAlmostEqual(cal.ece(repeat(0.7, 7, 3)), 0.0, places=9)
        self.assertAlmostEqual(cal.ece(repeat(0.9, 5, 5)), 0.4, places=9)
        mixed = repeat(0.15, 0, 4) + repeat(0.85, 4, 0)            # bins [0.1,0.2) and [0.8,0.9)
        self.assertAlmostEqual(cal.ece(mixed), 0.15, places=9)

    def test_scores_that_are_not_probabilities_are_refused(self):
        with self.assertRaises(ValueError):
            cal.ece([(3.5, 1), (0.2, 0)])
        with self.assertRaises(ValueError):
            cal.brier([(-0.1, 0)])
        curve = cal.reliability([(3.5, 1), (0.2, 0), (7.0, 1), (1.0, 0)], bins=2, strategy="quantile")
        self.assertEqual([b["n"] for b in curve], [2, 2])            # quantile bins work on any scale
        self.assertEqual(cal.confusion([(3.5, 1), (0.2, 0)], 1.0)["tp"], 1)

    def test_reliability_bins_and_wilson(self):
        curve = cal.reliability(repeat(0.15, 1, 3) + repeat(0.95, 9, 1), bins=10)
        self.assertEqual([(b["n"], b["observed"]) for b in curve], [(4, 0.25), (10, 0.9)])
        lo, hi = cal.wilson(0, 5)
        self.assertAlmostEqual(lo, 0.0, places=12)
        self.assertAlmostEqual(hi, 0.4345, places=3)
        self.assertEqual(cal.wilson(0, 0), (0.0, 1.0))

    def test_quantile_bins_keep_equal_scores_together(self):
        curve = cal.reliability(repeat(0.5, 2, 2) + repeat(0.6, 1, 0), bins=4, strategy="quantile")
        self.assertEqual([b["n"] for b in curve], [4, 1])


class IsotonicTest(unittest.TestCase):
    def test_pool_adjacent_violators(self):
        fit = cal.Isotonic.fit([(0.1, 1), (0.2, 0), (0.3, 1), (0.4, 1)])
        self.assertEqual([(b[0], b[1], b[2], b[3]) for b in fit.blocks], [(0.1, 0.2, 0.5, 2), (0.3, 0.4, 1.0, 2)])
        self.assertEqual(fit.predict(0.15), 0.5)
        self.assertEqual(fit.predict(0.25), 0.5)                       # between blocks: the lower one
        self.assertEqual(fit.predict(0.0), 0.5)
        self.assertEqual(fit.predict(0.35), 1.0)

    def test_monotone_and_ties_pooled(self):
        rows = [(s / 20, (s * 7) % 3 == 0) for s in range(20)] + [(0.5, 1), (0.5, 0), (0.5, 0)]
        fit = cal.Isotonic.fit(rows)
        values = [b[2] for b in fit.blocks]
        self.assertEqual(values, sorted(values))
        self.assertEqual(len({b[0] for b in fit.blocks}), len(fit.blocks))
        self.assertEqual(sum(b[3] for b in fit.blocks), len(rows))
        again = cal.Isotonic.from_list(json.loads(json.dumps(fit.as_list())))
        self.assertEqual([again.predict(s / 40) for s in range(40)], [fit.predict(s / 40) for s in range(40)])


class DecisionTest(unittest.TestCase):
    ROWS = [(0.9, 1), (0.8, 1), (0.6, 0), (0.4, 1), (0.2, 0)]

    def test_confusion_with_missed_positives(self):
        c = cal.confusion(self.ROWS, 0.5, missed=2)
        self.assertEqual((c["tp"], c["fp"], c["fn"], c["tn"]), (2, 1, 3, 1))
        self.assertAlmostEqual(c["precision"], 2 / 3)
        self.assertAlmostEqual(c["recall"], 2 / 5)
        self.assertAlmostEqual(c["false_positive_rate"], 0.5)

    def test_cost_weighted_threshold(self):
        self.assertEqual(cal.choose_threshold(self.ROWS, 1, 1)["threshold"], 0.8)     # tie 0.8 / 0.4: higher
        self.assertEqual(cal.choose_threshold(self.ROWS, 5, 1)["threshold"], 0.8)
        self.assertEqual(cal.choose_threshold(self.ROWS, 1, 5)["threshold"], 0.4)
        self.assertEqual(cal.choose_threshold(self.ROWS, 100, 0)["threshold"], math.inf)   # accept nothing
        with self.assertRaises(ValueError):
            cal.choose_threshold(self.ROWS, -1, 1)

    def test_drift_names_the_worst_title(self):
        rows = repeat(0.7, 7, 3, "good") * 2 + repeat(0.9, 5, 5, "bad") * 2 + repeat(0.5, 1, 1, "tiny")
        d = cal.drift(rows, min_records=20)
        self.assertEqual(d["worst"], "bad")
        self.assertAlmostEqual(d["spread"], 0.4, places=4)
        self.assertEqual(d["skipped_titles"], ["tiny"])


class FreezeTest(unittest.TestCase):
    def sources(self):
        dev = repeat(0.9, 9, 1, "devA") + repeat(0.6, 3, 7, "devA") + repeat(0.3, 1, 9, "devB")
        held = repeat(0.9, 2, 8, "heldX") + repeat(0.3, 0, 10, "heldX")
        return {"boundaries": {"records": dev + held, "probability": True}}, held

    def test_fit_on_development_only_and_evaluate_without_refit(self):
        sources, held = self.sources()
        frozen = cal.freeze(sources, ["devA", "devB"], cost_fp=1, cost_fn=1)
        entry = frozen["sources"]["boundaries"]
        self.assertEqual(entry["titles"], ["devA", "devB"])            # held-out records were not used
        self.assertEqual(entry["development_in_sample"]["records"], 30)
        self.assertEqual(entry["threshold"], 0.9)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frozen.json"
            cal.save_frozen(frozen, path)
            result = cal.evaluate_frozen(path, "boundaries", held)
        self.assertEqual(result["threshold"], 0.9)                     # frozen, although held-out would differ
        self.assertEqual(result["held_out_titles"], ["heldX"])
        self.assertEqual(result["mapped"]["at_threshold"]["precision"], 0.2)
        self.assertIn("raw", result)

    def test_refuses_development_titles_and_tampering(self):
        sources, _held = self.sources()
        frozen = cal.freeze(sources, ["devA", "devB"], cost_fp=1, cost_fn=1)
        with self.assertRaises(cal.FrozenError):
            cal.evaluate_frozen(frozen, "boundaries", [(0.9, 1, "devA")])
        tampered = json.loads(json.dumps(frozen))
        tampered["sources"]["boundaries"]["threshold"] = 0.1
        with self.assertRaises(cal.FrozenError):
            cal.load_frozen(tampered)
        with self.assertRaises(cal.FrozenError):
            cal.evaluate_frozen(frozen, "absent", [(0.9, 1, "heldX")])
        with self.assertRaises(cal.FrozenError):
            cal.freeze(sources, [], 1, 1)

    def test_per_source_development_titles_and_missed_positives(self):
        sources = {"a": {"records": repeat(0.8, 4, 1, "t1") + repeat(0.8, 1, 4, "t2"), "development": ["t1"],
                         "missed": {"t1": 2, "t2": 7}},
                   "b": {"records": repeat(0.5, 1, 1, "t3")}}
        frozen = cal.freeze(sources, ["t3"], cost_fp=1, cost_fn=1)
        self.assertEqual(frozen["development_titles"], ["t1", "t3"])
        self.assertEqual(frozen["sources"]["a"]["development_in_sample"]["missed"], 2)
        self.assertFalse(frozen["sources"]["a"]["probability_claimed"])
        result = cal.evaluate_frozen(frozen, "a", repeat(0.8, 1, 4, "t2"), missed={"t2": 7})
        self.assertEqual(result["mapped"]["at_threshold"]["fn"], 7)
        with self.assertRaises(cal.FrozenError):                  # t3 fitted source b: refused for a too
            cal.evaluate_frozen(frozen, "a", [(0.8, 1, "t3")])

    def test_digest_is_stable(self):
        sources, _held = self.sources()
        one = cal.freeze(sources, ["devB", "devA"], 1, 1)
        two = cal.freeze(sources, ["devA", "devB"], 1, 1)
        self.assertEqual(one["digest"], two["digest"])
        self.assertEqual(cal.load_frozen(one), one)


if __name__ == "__main__":
    unittest.main()
