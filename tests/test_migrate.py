"""Versions, the format registry and migration (``n64dk.migrate``): one version source of truth, every
registered current version readable from the module that writes it, and read-old / migrate / refuse
answers for runs, caches, stores and documents.  Synthetic inputs only; nothing outside a temporary
directory is written; no git is run."""
import hashlib
import importlib.util
import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from n64dk import migrate as mig
from n64dk.match import evidence_store as es
from n64dk.match import resolve as resolvemod


def _snapshot(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns,
                                             hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(root.rglob("*")) if p.is_file()}


class MigrateTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="n64dk-migrate-test-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def run_dir(self, invocation: dict) -> Path:
        run = self.root / "run"
        (run / "solver").mkdir(parents=True)
        (run / "solver" / "invocation.json").write_text(json.dumps(invocation))
        (run / "report.json").write_text(json.dumps({"status": "complete"}))
        (run / "rounds.jsonl").write_text('{"kind": "round"}\n{"kind": "ro')
        store = resolvemod.CaseStore(run / "state")
        try:
            store.put(resolvemod.Case(resolvemod.case_id("main", 0x80001000, 0x20), "main", "f", 0x80001000, 0x20, "t"))
        finally:
            store.close()
        return run

    def test_one_version_and_semantic_bumps(self):
        info = mig.version_info()
        self.assertTrue(info["consistent"], info["problems"])
        self.assertEqual(mig.bump("0.1.0", "minor"), "0.2.0")
        self.assertEqual(mig.bump("0.1.9", "patch"), "0.1.10")
        self.assertEqual(mig.bump("1.4.2", "major"), "2.0.0")
        with self.assertRaises(ValueError):
            mig.parse_semver("v1")
        project = self.root / "p"
        project.mkdir()
        (project / "pyproject.toml").write_text('[project]\nversion = "9.9.9"\n')
        self.assertFalse(mig.version_info(project)["consistent"])

    def test_registry_reads_current_versions_from_the_writing_modules(self):
        rows = mig.registry()
        self.assertEqual(len({r["name"] for r in rows}), len(rows))
        for fmt in mig.FORMATS:
            self.assertIn(fmt.old, ("read-old", "migrate", "refuse"))
            if fmt.current and not fmt.current.startswith("="):
                module = fmt.current.partition(":")[0]
                with self.subTest(format=fmt.name):
                    if importlib.util.find_spec(module) is None:
                        self.assertIsNone(fmt.current_value())       # a module this tree does not have
                    else:
                        self.assertIsNotNone(fmt.current_value())
        self.assertEqual(next(r for r in rows if r["name"] == "evidence.store")["current"], es.SCHEMA_VERSION)

    def test_version_comparison(self):
        self.assertEqual(mig.compare_versions("n64dk.tv/1", "n64dk.tv/1"), "current")
        self.assertEqual(mig.compare_versions("n64dk.tv/2", "n64dk.tv/1"), "newer")
        self.assertEqual(mig.compare_versions("n64dk.operations.v1", "n64dk.operations.v2"), "older")
        self.assertEqual(mig.compare_versions("n64dk-proof-receipt/1", "n64dk.tv/1"), "foreign")
        self.assertEqual(mig.compare_versions(1, 2), "older")

    def test_a_current_run_checks_clean_and_check_writes_nothing(self):
        run = self.run_dir({"effort": "exhaustive", "mode_settings": {"recipe_gate": False}})
        before = _snapshot(run)
        report = mig.migrate(run, check=True)
        self.assertEqual(_snapshot(run), before)
        self.assertEqual((report["kind"], report["ok"], report["pending"]), ("run", True, 0))
        status = {item["format"]: item["status"] for item in report["items"]}
        self.assertEqual(status["run.invocation"], "current")
        self.assertEqual(status["run.cases"], "read-old")
        self.assertEqual(status["run.observations"], "absent")
        rounds = next(i for i in report["items"] if i["format"] == "run.rounds")
        self.assertIn("1 cut lines", rounds["explanation"])

    def test_a_v1_invocation_migrates_with_a_backup(self):
        run = self.run_dir({"effort": "exhaustive", "budget": 400})
        report = mig.migrate(run, check=True)
        self.assertEqual(report["pending"], 1)
        done = mig.migrate(run, check=False)
        self.assertEqual(next(i for i in done["items"] if i["format"] == "run.invocation")["status"], "migrated")
        saved = json.loads((run / "solver" / "invocation.json").read_text())
        self.assertEqual(saved["mode_settings"], {"recipe_gate": False, "regional_calibration": True})
        self.assertEqual(saved["budget"], 400)
        self.assertEqual(json.loads((run / "solver" / "invocation.json.v1.bak").read_text()),
                         {"effort": "exhaustive", "budget": 400})
        self.assertEqual(mig.migrate(run, check=True)["pending"], 0)

    def test_newer_or_foreign_data_is_refused_with_an_explanation(self):
        run = self.run_dir({"effort": "ludicrous"})
        report = mig.migrate(run)
        self.assertFalse(report["ok"])
        self.assertIn("not a mode of this kit", report["items"][0]["explanation"])
        db = sqlite3.connect(run / "state" / "cases.sqlite")
        db.execute("UPDATE cases SET payload = json_set(payload, '$.future_field', 1)")
        db.commit()
        db.close()
        cases = mig.check_store(run / "state" / "cases.sqlite")[0]
        self.assertEqual(cases["status"], "refused")
        self.assertIn("future_field", cases["explanation"])
        document = self.root / "tv.json"
        document.write_text(json.dumps({"schema": "n64dk.tv/7"}))
        self.assertEqual(mig.migrate(document)["items"][0]["status"], "refused")
        document.write_text(json.dumps({"schema": "n64dk.tv/1"}))
        self.assertEqual(mig.migrate(document)["items"][0]["status"], "current")
        unknown = self.root / "nothing"
        unknown.mkdir()
        self.assertEqual(mig.migrate(unknown)["items"][0]["status"], "refused")

    def test_stores_evidence_and_verification(self):
        path = self.root / "store" / es.DATABASE
        es.SQLiteRepository(path, schema=1).close()
        self.assertEqual(mig.migrate(path)["items"][0]["status"], "migratable")
        self.assertEqual(mig.migrate(path, check=False)["items"][0]["status"], "migrated")
        self.assertEqual(mig.migrate(path)["items"][0]["status"], "current")
        receipts = self.root / "verification.sqlite"
        db = sqlite3.connect(receipts)
        db.execute("CREATE TABLE receipts (receipt_id TEXT PRIMARY KEY, source TEXT)")
        db.commit()
        db.close()
        self.assertEqual(mig.migrate(receipts)["items"][0]["status"], "migratable")
        self.assertEqual(mig.migrate(receipts, check=False)["items"][0]["status"], "migrated")
        db = sqlite3.connect(receipts)
        columns = {row[1] for row in db.execute("PRAGMA table_info(receipts)")}
        db.close()
        self.assertLessEqual({"image_path", "image_name", "target_offset", "catalogue_vram"}, columns)
        other = self.root / "index.sqlite"
        db = sqlite3.connect(other)
        db.execute("CREATE TABLE scripts (root TEXT)")
        db.commit()
        db.close()
        self.assertEqual(mig.migrate(other)["items"][0]["status"], "read-old")

    def test_content_cache(self):
        cache = self.root / "cache"
        (cache / "objects" / "ab").mkdir(parents=True)
        (cache / "refs").mkdir()
        digest = "ab" + "0" * 62
        (cache / "objects" / "ab" / digest[2:]).write_bytes(b"x")
        (cache / "refs" / "one").write_text(json.dumps({"key": "k", "digest": digest}))
        (cache / "refs" / "two").write_text(json.dumps({"key": "k2", "digest": "cd" + "0" * 62}))
        report = mig.migrate(cache)
        self.assertEqual((report["kind"], report["items"][0]["status"]), ("cache", "read-old"))
        self.assertIn("1 to missing objects", report["items"][0]["explanation"])
        (cache / "refs" / "three").write_text("{not json")
        self.assertEqual(mig.migrate(cache)["items"][0]["status"], "refused")

    def test_changelog_from_git_log_text(self):
        text = "abcdef1234567890\x1f2026-11-17\x1fSecond review wave\n0123456789abcdef\x1f2026-11-16\x1fReview fixes\n"
        entries = mig.parse_log(text)
        self.assertEqual([e["commit"] for e in entries], ["abcdef1234", "0123456789"])
        section = mig.changelog(entries, "0.2.0", "2026-11-18")
        self.assertTrue(section.startswith("## 0.2.0 (2026-11-18)"))
        self.assertIn("- Second review wave (abcdef1234, 2026-11-17)", section)
        calls = []

        def runner(argv, **_kwargs):
            calls.append(argv)
            return SimpleNamespace(returncode=0, stdout=text, stderr="")
        self.assertEqual(mig.git_log("repo", "v0.1.0", runner=runner), text)
        self.assertEqual(calls[0][:4], ["git", "-C", "repo", "log"])
        self.assertIn("v0.1.0..HEAD", calls[0])


if __name__ == "__main__":
    unittest.main()
