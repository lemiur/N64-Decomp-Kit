import subprocess
import tempfile
import unittest
from pathlib import Path

from n64dk.match import regress

HEAD = "image\tfunction\tvram\trom_offset\tsize\tcode_size\toutcome\thow\tsource\tdistance\treproduced_by\n"


def row(fn, vram, outcome, how, size="0x10", source="", distance=""):
    return f"main\t{fn}\t{vram}\t0x1000\t{size}\t{size}\t{outcome}\t{how}\t{source}\t{distance}\t\n"


class CompareTest(unittest.TestCase):
    def write(self, folder, rows, sources=None):
        folder.mkdir()
        (folder / "FUNCTIONS.tsv").write_text(HEAD + "".join(rows))
        for name, text in (sources or {}).items():
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            (folder / name).write_text(text)
        return folder

    def test_gains_losses_transitions_and_first_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.write(Path(tmp) / "a", [
                row("f1", "0x80000400", "open", "kmc+as -O2"),
                row("f2", "0x80000410", "exact", "kmc+as -O2 (draft)", source="src/f2.c"),
                row("f3", "0x80000420", "near", "kmc+as -O2", distance="12"),
                row("f4", "0x80000430", "exact", "kmc+as -O2 (draft)"),
                row("f5", "0x80000440", "open", "kmc+as -O2")], {"src/f2.c": "int a;"})
            b = self.write(Path(tmp) / "b", [
                row("f1", "0x80000400", "exact", "kmc+gas -O2 -G8 (draft)"),
                row("f2", "0x80000410", "near", "kmc+as -O2 (draft)", source="src/f2.c", distance="3"),
                row("f3", "0x80000420", "near", "kmc+as -O2", distance="5"),
                row("f4", "0x80000430", "exact", "kmc+as -O2 (donor)"),
                row("f6", "0x80000450", "open", "kmc+as -O2")], {"src/f2.c": "int b;"})
            result = regress.compare(a, b)
            self.assertEqual((result["strict_before"], result["strict_after"]), (2, 2))
            self.assertEqual([(e["function"], e["first_difference"]) for e in result["gained"]], [("f1", "recipe")])
            self.assertEqual([(e["function"], e["first_difference"]) for e in result["lost"]], [("f2", "source")])
            self.assertEqual(result["transitions"]["open -> exact"], 1)
            self.assertEqual(result["transitions"]["open -> absent"], 1)
            self.assertEqual(result["denominator"]["main"]["only_before"], 1)
            self.assertEqual(result["denominator"]["main"]["only_after"], 1)
            causes = result["causes"]
            self.assertEqual(causes["discovery"], 2)                  # f5 gone, f6 new
            self.assertEqual(causes["stage"], 1)                      # f4 draft -> donor
            self.assertEqual(causes["verification"], 1)               # f3 distance only
            self.assertIn("Lost strict functions", regress.markdown(result))

    def test_a_boundary_change_is_discovery(self):
        a = {"size": "0x10", "rom_offset": "0x1000", "how": "x"}
        b = {"size": "0x14", "rom_offset": "0x1000", "how": "x"}
        self.assertEqual(regress.first_difference(a, b), "discovery")


class BisectTest(unittest.TestCase):
    def test_binary_search_finds_the_first_bad_commit(self):
        commits = [f"c{i}" for i in range(10)]
        first, tried = regress.search(commits, lambda c: int(c[1:]) >= 6)
        self.assertEqual(first, "c6")
        self.assertLessEqual(len(tried), 4)
        self.assertEqual(regress.search(commits, lambda c: False)[0], None)

    def test_a_worktree_check_never_touches_the_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
            subprocess.run(git[:3] + ["init", "-q"], check=True)
            for i in range(4):
                (repo / "v.txt").write_text(str(i))
                subprocess.run(git + ["add", "v.txt"], check=True)
                subprocess.run(git + ["commit", "-qm", f"c{i}"], check=True)
            log = subprocess.run(git[:3] + ["rev-list", "--reverse", "HEAD"], capture_output=True, text=True).stdout.split()
            candidates = regress.commits(repo, log[0], log[-1])
            self.assertEqual(len(candidates), 3)
            import os
            from unittest import mock
            with mock.patch.dict(os.environ, {"N64DK_STATE_DIR": str(Path(tmp) / "state")}):   # a private lock
                first, _tried = regress.search(candidates, regress.worktree_check(repo, "test $(cat v.txt) -lt 2", 60))
            self.assertEqual(first, log[2])
            self.assertEqual((repo / "v.txt").read_text(), "3")                 # the checkout untouched
            listed = subprocess.run(git[:3] + ["worktree", "list"], capture_output=True, text=True).stdout
            self.assertEqual(len(listed.strip().splitlines()), 1)              # every worktree removed


if __name__ == "__main__":
    unittest.main()
