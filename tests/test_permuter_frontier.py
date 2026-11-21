import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from n64dk.match import permuter
from n64dk.match import symbols as symmod

SOURCE = """
int g(int);
int f(int a, int b) {
    int x;
    int y;
    x = a + b;
    y = a - b;
    if (x > y) {
        x = g(x);
    }
    return x + y;
}
"""


@unittest.skipUnless(permuter.available(), "pycparser not importable")
class FrontierTest(unittest.TestCase):
    """The search keeps the best few results that compile to different code, not different text."""

    def run_search(self, **options):
        compiled = []

        def fake_match(path, spec, cfg, target, vram, size, syms, work, includes=None, infer_externals=True, **kw):
            text = Path(path).read_text()
            # what it "compiles to" depends only on how many characters the body has, in steps of 16:
            # many different texts share one output, as real sources do
            state = len(text) // 16
            compiled.append(state)
            score = 1 + abs(state - 9)
            diff = SimpleNamespace(score=score)
            return SimpleNamespace(exact=False, result=SimpleNamespace(diff=diff),
                                   digest=hashlib.sha1(str(state).encode()).hexdigest())

        original = permuter.match_source
        permuter.match_source = fake_match
        try:
            with tempfile.TemporaryDirectory() as name:
                source = Path(name) / "f.c"
                source.write_text(SOURCE)
                return permuter.permute(source, None, None, None, 0x80001000, None, symmod.SymbolMap(), [],
                                        function="f", iterations=48, jobs=2, seed=7, out_dir=Path(name) / "out",
                                        **options), compiled
        finally:
            permuter.match_source = original

    def test_states_in_the_frontier_compile_to_different_code(self):
        result, _ = self.run_search(keep=3)
        outputs = [state for _score, state, _text in result.frontier]
        self.assertEqual(len(outputs), len(set(outputs)))
        self.assertLessEqual(len(outputs), 3)
        self.assertGreaterEqual(len(outputs), 1)

    def test_the_frontier_is_ordered_best_first_and_holds_the_best_result(self):
        result, _ = self.run_search(keep=4)
        scores = [score for score, _state, _text in result.frontier]
        self.assertEqual(scores, sorted(scores))
        self.assertEqual(scores[0], result.best_score)

    def test_islands_search_from_the_same_start_and_still_return_one_result(self):
        result, compiled = self.run_search(islands=3)
        self.assertEqual(result.iterations, 48)
        self.assertLessEqual(result.best_score, result.base_score)
        self.assertTrue(compiled)


if __name__ == "__main__":
    unittest.main()
