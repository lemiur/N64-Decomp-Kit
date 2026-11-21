"""Ways around a blind run's isolation guard (wave-4 review W4_02), each closed by the strict check
(``check_isolation(strict=True)``, what ``decompile-run --blind-forbid`` uses) or by the run
directory's provenance record (``governance.run_provenance``).  Temporary trees only; no ROM, no
toolchain, no kit run."""
import argparse
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from n64dk import governance as gov
from n64dk.commands import run

TRUTH_C = "void func_80001000(void) {\n    gSomeGlobalCounter = gSomeGlobalCounter + 1;\n    return;\n}\n"
SYMBOLS = "".join(f"func_8000{i:04X} = 0x8000{i:04X}; // gTitleObject{i}\n" for i in range(8))


class StrictCheckTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="n64dk-blind-bypass-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.truth = self.root / "public_src" / "title"
        (self.truth / "src").mkdir(parents=True)
        (self.truth / "src" / "game.c").write_text(TRUTH_C)
        (self.truth / "include").mkdir()
        (self.truth / "include" / "ultra64.h").write_text("/* shared SDK header */\n" + "#define X 1\n" * 8)
        (self.truth / "symbol_addrs.txt").write_text(SYMBOLS)
        self.project = self.root / "work" / "proj"
        self.project.mkdir(parents=True)
        (self.project / "n64dk.toml").write_text('[run]\nsymbols = ["symbols.txt"]\n')
        (self.project / "symbols.txt").write_text("func_80001000 = 0x80001000;\n")
        self.forbidden = [("title", self.truth)]

    def check(self, config=None, env=None, forbidden=None, **kw):
        return gov.check_isolation(forbidden or self.forbidden, config=config or {}, project=self.project,
                                   env=env or {}, base=self.root, **kw)

    def both(self, config=None, env=None, **kw):
        return self.check(config, env, **kw), self.check(config, env, strict=True, **kw)

    def verdicts(self, receipt):
        return [v["verdict"] for v in receipt["violations"]]

    def test_without_strict_the_receipt_is_unchanged(self):
        plain = self.check()
        self.assertEqual(plain, self.check(strict=False))
        self.assertNotIn("strict", plain)
        self.assertNotIn("includes_checked", plain["walk"])
        self.assertEqual(len(plain["limits"]), 4)
        strict = self.check(strict=True)
        self.assertTrue(strict["ok"], strict["violations"])
        self.assertTrue(strict["strict"])
        self.assertEqual(gov.verify_receipt(strict), [])

    def test_a_symlink_below_the_third_level_is_found(self):
        deep = self.project / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        os.symlink(self.truth / "symbol_addrs.txt", deep / "symbol_addrs.txt")   # symbols.discover finds it
        plain, strict = self.both()
        self.assertTrue(plain["ok"])                                             # the gap: never walked
        self.assertFalse(strict["ok"])
        self.assertEqual(self.verdicts(strict), ["symlink inside forbidden"])

    def test_a_truncated_walk_fails(self):
        many = self.project / "many"
        many.mkdir()
        for index in range(10):
            (many / f"f{index}.bin").write_bytes(b"x")
        plain, strict = self.both(file_limit=5)
        self.assertTrue(plain["ok"])
        self.assertTrue(plain["walk"]["truncated"])
        self.assertFalse(strict["ok"])
        self.assertTrue(any(v.startswith("walk truncated") for v in self.verdicts(strict)))

    def test_a_forbidden_tree_larger_than_the_limit_or_missing_fails(self):
        strict = self.check(strict=True, file_limit=3)
        self.assertTrue(any("not all of it is indexed" in v for v in self.verdicts(strict)))
        ghost = [("ghost", self.root / "no_such_tree")]
        self.assertTrue(self.check(forbidden=ghost)["ok"])                     # blind against nothing
        missing = self.check(forbidden=ghost, strict=True)
        self.assertEqual(self.verdicts(missing), ["the forbidden tree does not exist"])

    def test_a_donors_file_of_any_name_is_read(self):
        table = f'[donor.a]\nroot = "{self.truth / "src"}"\n'
        (self.root / "donors.list").write_text(table)
        plain, strict = self.both({"donors": str(self.root / "donors.list")})
        self.assertTrue(plain["ok"])
        self.assertFalse(strict["ok"])
        self.assertEqual(strict["violations"][0]["key"], "config.donors>donor.a.root")

    def test_a_bare_relative_name_is_a_path_when_it_exists(self):
        (self.root / "donors").write_text(f'[donor.a]\nroot = "{self.truth}"\n')
        plain, strict = self.both({"donors": "donors", "pipeline": "ido5.3"})
        self.assertTrue(plain["ok"])
        self.assertFalse(strict["ok"])
        self.assertNotIn("config.pipeline", [row["key"] for row in strict["checked"]])   # no such file

    def test_a_shared_scratch_directory_is_not_an_input(self):
        shared = self.root / "shared_tmp"
        shared.mkdir()
        os.symlink(self.truth / "src", shared / "link")
        with mock.patch.object(gov, "_scratch_dirs", return_value={shared.resolve()}):
            plain, strict = self.both({"library": str(shared)})
            tmpdir = self.check({"pipelines": {"ido": {"env": {"TMPDIR": str(shared)}}}}, strict=True)
        self.assertTrue(plain["ok"])                                             # the gap: never walked
        self.assertEqual(self.verdicts(strict), ["a shared scratch directory named as an input: not checkable"])
        self.assertTrue(tmpdir["ok"], tmpdir["violations"])                      # a scratch key stays scratch

    def test_search_path_variables_outside_n64dk(self):
        cases = {"CPATH": (str(self.truth / "include"), False),
                 "PYTHONPATH": (str(self.root / "public_src"), False),           # contains the tree: importable
                 "PATH": (os.pathsep.join(["/usr/bin", str(self.truth / "tools")]), False),
                 "TMPDIR": (str(self.truth / "tmp"), False)}
        for name, (value, ok) in cases.items():
            with self.subTest(variable=name):
                plain, strict = self.both(env={name: value})
                self.assertTrue(plain["ok"])
                self.assertEqual(strict["ok"], ok, strict["violations"])
        outer = self.check(env={"PATH": str(self.root / "public_src")}, strict=True)
        self.assertTrue(outer["ok"], outer["violations"])                        # PATH: only its own files

    def test_an_include_that_leaves_its_tree_is_followed(self):
        (self.project / "src").mkdir()
        (self.project / "src" / "x.c").write_text(
            '#include "../../../public_src/title/include/ultra64.h"\n'
            '#include "../include/own.h"\n#include <ultra64.h>\nint x;\n')
        (self.project / "game.ld").write_text(f"INCLUDE {self.truth / 'include' / 'ultra64.h'}\n")
        plain, strict = self.both()
        self.assertTrue(plain["ok"])
        self.assertFalse(strict["ok"])
        self.assertEqual(self.verdicts(strict), ["include inside forbidden"] * 2)
        self.assertEqual(strict["walk"]["includes_checked"], 3)

    def test_the_run_directory_is_checked_not_walked(self):
        out = self.project / ".n64dk" / "run"
        (out / "state" / "sources").mkdir(parents=True)
        for index in range(10):
            (out / "state" / "sources" / f"{index}.c").write_text("int x;\n")
        walked = self.check(strict=True, file_limit=8)
        self.assertTrue(any(v.startswith("walk truncated") for v in self.verdicts(walked)))
        skipped = self.check(strict=True, file_limit=8, outputs={"out": str(out)})
        self.assertTrue(skipped["ok"], skipped["violations"])
        [row] = [row for row in skipped["checked"] if row["key"] == "output.out"]
        self.assertEqual(row["verdict"], "outside (checked, not walked)")
        self.assertFalse(self.check(strict=True, outputs={"out": str(self.truth / "out")})["ok"])

    def test_a_copied_symbol_list_fails_and_a_named_file_is_hashed(self):
        (self.project / "syms_copy.txt").write_text(SYMBOLS)
        plain, strict = self.both()
        self.assertTrue(plain["ok"])
        self.assertEqual([row["same_as"] for row in plain["shared_files"]], ["title/symbol_addrs.txt"])
        self.assertEqual(self.verdicts(strict), ["copy of a forbidden symbol list or linker script"])
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "ctx.c").write_text(TRUTH_C)
        (self.project / "syms_copy.txt").unlink()
        plain, strict = self.both({"context_file": str(elsewhere / "ctx.c")})
        self.assertTrue(plain["ok"])                                             # a file, not a directory
        self.assertEqual(self.verdicts(strict), ["copy of forbidden source"])


class ProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="n64dk-blind-provenance-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.truth = self.root / "public_src" / "title"
        self.truth.mkdir(parents=True)
        self.other = self.root / "public_src" / "other"
        self.other.mkdir()
        self.forbidden = [("title", self.truth)]

    def test_fresh_unknown_and_non_blind_directories(self):
        out = self.root / "run"
        self.assertEqual(gov.run_provenance(out, self.forbidden)["verdict"], "fresh")
        out.mkdir()
        self.assertTrue(gov.run_provenance(out, self.forbidden)["ok"])
        (out / "main").mkdir()
        (out / "main" / "stage_donors.json").write_text("{}")
        unknown = gov.run_provenance(out, self.forbidden)
        self.assertEqual((unknown["ok"], unknown["verdict"]), (False, "unknown provenance"))
        gov.record_session(out, blind=False, status="not blind")
        tainted = gov.run_provenance(out, self.forbidden)
        self.assertFalse(tainted["ok"])
        self.assertIn("session 1 was not blind", tainted["problems"])

    def test_blind_sessions_cover_only_the_trees_they_forbade(self):
        out = self.root / "run"
        (out / "solver").mkdir(parents=True)
        index = gov.record_session(out, blind=True, forbidden=self.forbidden)
        (out / "state").mkdir()
        self.assertTrue(gov.run_provenance(out, self.forbidden)["ok"])
        self.assertEqual(gov.run_provenance(out, self.forbidden)["interrupted"], 1)
        wider = gov.run_provenance(out, self.forbidden + [("other", self.other)])
        self.assertFalse(wider["ok"])
        self.assertIn(str(self.other.resolve()), wider["problems"][0])
        gov.update_session(out, index, status="violated")
        self.assertFalse(gov.run_provenance(out, self.forbidden)["ok"])

    def test_a_blind_session_over_unknown_state_is_not_blind(self):
        out = self.root / "run"
        (out / "state").mkdir(parents=True)
        gov.record_session(out, blind=True, forbidden=self.forbidden)
        data = json.loads(gov.provenance_path(out).read_text())
        self.assertEqual((data["sessions"][0]["blind"], data["sessions"][0]["prior_state"]), (False, True))
        self.assertFalse(gov.run_provenance(out, self.forbidden)["ok"])


class BlindRunDirectoryTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="n64dk-blind-rundir-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        # the guard reads this process's environment: keep it to what the test controls
        environment = mock.patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.truth = self.root / "public_src" / "title"
        (self.truth / "src").mkdir(parents=True)
        (self.truth / "src" / "game.c").write_text(TRUTH_C)
        self.project = self.root / "proj"
        self.project.mkdir()
        (self.project / "n64dk.toml").write_text('[run]\nsymbols = ["symbols.txt"]\n')
        (self.project / "symbols.txt").write_text("func_80001000 = 0x80001000;\n")
        self.out = self.root / "out"

    def args(self, *forbid, **extra):
        return SimpleNamespace(project=str(self.project), out=str(self.out), donors=None,
                               blind_forbid=[str(p) for p in (forbid or (self.truth,))], **extra)

    def solved(self, _args):
        """What a run leaves behind: a case store with a solved function in it."""
        (self.out / "state").mkdir(parents=True, exist_ok=True)
        (self.out / "state" / "cases.sqlite").write_bytes(b"solved")
        return {"images": {}}

    def test_state_from_an_unrecorded_run_refuses_a_blind_run(self):
        (self.out / "main").mkdir(parents=True)
        (self.out / "main" / "stage_donors.json").write_text('{"verified": []}')
        with mock.patch.object(run, "_decompile_run") as body, self.assertRaises(SystemExit) as stop:
            run._blind_run(self.args())
        self.assertEqual(stop.exception.code, 3)
        body.assert_not_called()
        receipt = json.loads((self.out / "solver" / "blind_violation.json").read_text())
        self.assertEqual(receipt["verdict"], "unknown provenance")
        self.assertFalse(receipt["ok"])

    def test_a_non_blind_session_taints_its_directory_even_when_it_fails(self):
        plain = SimpleNamespace(project=str(self.project), out=str(self.out), allow_project_output=False)
        with mock.patch.object(run, "_decompile_run", side_effect=self.solved):
            run._unguarded_run(plain)                                  # creates the directory: marked at the end
        with mock.patch.object(run, "_decompile_run", side_effect=RuntimeError("killed")), \
                self.assertRaises(RuntimeError):
            run._unguarded_run(plain)                                  # existing directory: marked first
        sessions = json.loads(gov.provenance_path(self.out).read_text())["sessions"]
        self.assertEqual([s["blind"] for s in sessions], [False, False])
        with mock.patch.object(run, "_decompile_run") as body, self.assertRaises(SystemExit) as stop:
            run._blind_run(self.args())
        self.assertEqual(stop.exception.code, 3)
        body.assert_not_called()

    def test_a_refused_project_output_is_not_marked(self):
        inside = SimpleNamespace(project=str(self.project), out=str(self.project / "generated"),
                                 allow_project_output=False)
        (self.project / "generated").mkdir()
        with mock.patch.object(run, "_decompile_run", side_effect=SystemExit("source tree")), \
                self.assertRaises(SystemExit):
            run._unguarded_run(inside)
        self.assertFalse((self.project / "generated" / "solver").exists())

    def test_blind_sessions_continue_but_a_wider_forbid_needs_a_fresh_directory(self):
        with mock.patch.object(run, "_decompile_run", side_effect=self.solved) as body:
            first = run._blind_run(self.args())
            again = run._blind_run(self.args())
        self.assertEqual(body.call_count, 2)
        self.assertEqual((first["blind"]["run_dir"]["verdict"], again["blind"]["run_dir"]["verdict"]),
                         ("fresh", "blind sessions only"))
        sessions = json.loads(gov.provenance_path(self.out).read_text())["sessions"]
        self.assertEqual([(s["blind"], s["status"]) for s in sessions], [(True, "clean"), (True, "clean")])
        other = self.root / "public_src" / "other"
        other.mkdir()
        with mock.patch.object(run, "_decompile_run") as body, self.assertRaises(SystemExit) as stop:
            run._blind_run(self.args(self.truth, other))
        self.assertEqual(stop.exception.code, 3)
        body.assert_not_called()

    def test_a_tree_reached_during_the_run_marks_the_session(self):
        def body(_args):
            (self.project / "n64dk.toml").write_text(f'[run]\nsymbols = ["{self.truth / "src" / "game.c"}"]\n')
            return {"images": {}}
        with mock.patch.object(run, "_decompile_run", side_effect=body), self.assertRaises(SystemExit):
            run._blind_run(self.args())
        sessions = json.loads(gov.provenance_path(self.out).read_text())["sessions"]
        self.assertEqual(sessions[0]["status"], "violated")

    def test_resume_keeps_forbidden_trees_and_refuses_new_ones(self):
        parser = argparse.ArgumentParser()
        run.decompile_run.arguments(parser)
        self.out.mkdir()
        start = parser.parse_args(["--project", str(self.project), "--out", str(self.out)])
        run.prepare_invocation(start)
        run.save_invocation(start, self.out)
        again = parser.parse_args(["--resume", str(self.out), "--blind-forbid", str(self.truth)])
        with self.assertRaises(SystemExit):              # never silently dropped: a session that was
            run.prepare_invocation(again)                # never checked cannot be made blind by a resume
        blind_out = self.root / "blind_out"
        blind_out.mkdir()
        relative = os.path.relpath(self.truth)
        start = parser.parse_args(["--project", str(self.project), "--out", str(blind_out),
                                   "--blind-forbid", relative])
        run.prepare_invocation(start)
        run.save_invocation(start, blind_out)
        saved = json.loads((blind_out / "solver" / "invocation.json").read_text())
        self.assertEqual(saved["blind_forbid"], [str(self.truth.resolve())])      # resumable from anywhere
        resumed = parser.parse_args(["--resume", str(blind_out)])
        run.prepare_invocation(resumed)
        self.assertEqual(resumed.blind_forbid, [str(self.truth.resolve())])


if __name__ == "__main__":
    unittest.main()
