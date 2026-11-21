"""decompile-run --blind-forbid: blind isolation enforced around the run (product gate Q13)."""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from n64dk.commands import run


class BlindRunTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="n64dk-blind-run-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        # hermetic: the guard reads every N64DK_* variable and walks what it names (a toolchain root
        # exported by the job wrapper would be hashed twice per test)
        clean = mock.patch.dict(os.environ, {k: v for k, v in os.environ.items() if not k.startswith("N64DK_")},
                                clear=True)
        clean.start()
        self.addCleanup(clean.stop)
        self.truth = self.root / "public_src" / "title"
        (self.truth / "src").mkdir(parents=True)
        (self.truth / "src" / "game.c").write_text("int secret(void) { return 42; }\n" * 4)
        self.project = self.root / "proj"
        self.project.mkdir()
        (self.project / "n64dk.toml").write_text('[run]\nsymbols = ["symbols.txt"]\n')
        (self.project / "symbols.txt").write_text("func_80001000 = 0x80001000;\n")

    def args(self, **extra):
        return SimpleNamespace(project=str(self.project), out=str(self.root / "out"),
                               blind_forbid=[str(self.truth)], donors=None, **extra)

    def test_a_clean_blind_run_records_both_receipts(self):
        with mock.patch.object(run, "_decompile_run", return_value={"images": {}}) as body:
            result = run._blind_run(self.args())
        body.assert_called_once()
        self.assertTrue(result["blind"]["ok"])
        self.assertTrue(result["blind"]["enforced"])
        for name in ("blind_entry.json", "blind_exit.json"):
            receipt = json.loads((self.root / "out" / "solver" / name).read_text())
            self.assertTrue(receipt["ok"])
        self.assertEqual(result["blind"]["entry_digest"],
                         json.loads((self.root / "out" / "solver" / "blind_entry.json").read_text())["digest"])

    def test_a_reachable_tree_refuses_before_any_stage(self):
        (self.project / "n64dk.toml").write_text(f'[run]\nsymbols = ["{self.truth / "src" / "game.c"}"]\n')
        with mock.patch.object(run, "_decompile_run") as body, self.assertRaises(SystemExit) as stop:
            run._blind_run(self.args())
        self.assertEqual(stop.exception.code, 3)
        body.assert_not_called()
        receipt = json.loads((self.root / "out" / "solver" / "blind_violation.json").read_text())
        self.assertFalse(receipt["ok"])

    def test_a_tree_made_reachable_during_the_run_fails_it(self):
        def body(_args):
            (self.project / "n64dk.toml").write_text(f'[run]\nsymbols = ["{self.truth / "src" / "game.c"}"]\n')
            return {"images": {}}
        with mock.patch.object(run, "_decompile_run", side_effect=body), self.assertRaises(SystemExit) as stop:
            run._blind_run(self.args())
        self.assertEqual(stop.exception.code, 3)
        self.assertTrue((self.root / "out" / "solver" / "blind_entry.json").is_file())
        self.assertFalse(json.loads((self.root / "out" / "solver" / "blind_violation.json").read_text())["ok"])

    def test_the_option_is_saved_for_resume(self):
        self.assertNotIn("blind_forbid", run._NOT_SAVED)

    def test_a_tree_that_does_not_exist_refuses_the_run(self):
        # a typo, or a relative tree resumed from another directory, used to pass as "enforced"
        args = self.args()
        args.blind_forbid = [str(self.root / "public_src" / "no-such-title")]
        with mock.patch.object(run, "_decompile_run") as body, self.assertRaises(SystemExit) as stop:
            run._blind_run(args)
        body.assert_not_called()
        self.assertIn("no such tree", str(stop.exception.code))

    def test_the_saved_trees_are_absolute(self):
        args = SimpleNamespace(project=str(self.project), blind_forbid=[os.path.relpath(self.truth)])
        run.save_invocation(args, self.root / "out")
        saved = json.loads((self.root / "out" / "solver" / "invocation.json").read_text())
        self.assertEqual(saved["blind_forbid"], [str(self.truth.resolve())])

    def test_resume_cannot_add_a_tree_the_run_did_not_start_with(self):
        run_dir = self.root / "out"
        (run_dir / "solver").mkdir(parents=True)
        (run_dir / "solver" / "invocation.json").write_text(json.dumps({"project": str(self.project),
                                                                        "blind_forbid": None}))
        args = SimpleNamespace(resume=str(run_dir), blind_forbid=[str(self.truth)], rounds=None, effort=None,
                               schedule=None, project=None, out=None)
        with self.assertRaises(SystemExit) as stop:                 # it used to be dropped silently
            run.prepare_invocation(args)
        self.assertIn("was not enforced when the run started", str(stop.exception.code))


if __name__ == "__main__":
    unittest.main()
