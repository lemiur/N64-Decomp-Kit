"""The versioned, content-addressed evidence store (``n64dk.match.evidence_store``).

Both stores behind the one ``Repository`` interface; SQLite in rollback-journal mode with
sidecar-free read-only inspection, migrations, crash safety, invalidation, deterministic export
that refuses ROM-derived content, and a read-only import of a synthetic run directory."""
import hashlib
import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from n64dk.match import evidence_store as es


def _snapshot(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns,
                                             hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(root.rglob("*")) if p.is_file()}


def _sidecars(path: Path) -> list[str]:
    return [suffix for suffix in ("-wal", "-shm", "-journal") if Path(f"{path}{suffix}").exists()]


def _operations(repo) -> None:
    """The same history on any store: two versions, dependencies, a blob, an invalidation."""
    repo.put("fact", "a", {"value": 1}, provenance={"source": "test"})
    repo.put("fact", "a", {"value": 2}, provenance={"source": "test"})
    repo.put("derived", "b", [1, 2, 3], depends_on=["fact:a"])
    repo.put("derived", "c", "text", depends_on=["derived:b"])
    repo.put("other", "d", None, depends_on=["file:x"])
    repo.put_blob(b"plain evidence")
    repo.invalidate("fact:a", cause="test")


class EvidenceStoreTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="n64dk-evidence-test-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def stores(self):
        """(name, factory) for each implementation of the interface."""
        counter = iter(range(1000))
        return [("memory", es.MemoryRepository),
                ("sqlite", lambda: es.SQLiteRepository(self.root / f"s{next(counter)}" / es.DATABASE))]

    def test_both_stores_implement_the_interface(self):
        for name, make in self.stores():
            with self.subTest(store=name):
                repo = make()
                try:
                    self.assertIsInstance(repo, es.Repository)
                finally:
                    repo.close()

    def test_records_are_versioned_append_only_and_content_addressed(self):
        for name, make in self.stores():
            with self.subTest(store=name):
                repo = make()
                try:
                    first = repo.put("fact", "a", {"value": 1, "list": (1, 2)}, provenance={"file": "x.json"})
                    again = repo.put("fact", "a", {"list": [1, 2], "value": 1}, provenance={"file": "x.json"})
                    self.assertEqual(again, first)                      # same content: no new version
                    second = repo.put("fact", "a", {"value": 2}, provenance={"file": "x.json"})
                    self.assertEqual((first.version, second.version), (1, 2))
                    self.assertEqual(second.parents, (first.record_id,))
                    self.assertNotEqual(first.record_id, second.record_id)
                    self.assertEqual([r.payload for r in repo.history("fact", "a")],
                                     [{"list": [1, 2], "value": 1}, {"value": 2}])
                    self.assertEqual(repo.latest("fact", "a"), second)
                    self.assertIsNone(repo.latest("fact", "missing"))
                    self.assertEqual([r.handle for r in repo.records()], ["fact:a"])
                    with self.assertRaises(ValueError):
                        repo.put("fact", "nan", float("nan"))
                    with self.assertRaises(ValueError):
                        repo.put("", "a", 1)
                    digest = repo.put_blob(b"abc")
                    self.assertEqual(digest, hashlib.sha256(b"abc").hexdigest())
                    self.assertEqual(repo.get_blob(digest), b"abc")
                    self.assertFalse(repo.blob_is_rom_derived(digest))
                    repo.put_blob(b"abc", rom_derived=True)          # a flag is raised, never lowered
                    repo.put_blob(b"abc")
                    self.assertTrue(repo.blob_is_rom_derived(digest))
                finally:
                    repo.close()

    def test_an_exception_inside_a_transaction_leaves_nothing(self):
        for name, make in self.stores():
            with self.subTest(store=name):
                repo = make()
                try:
                    repo.put("fact", "kept", 1)
                    with self.assertRaises(RuntimeError):
                        with repo.transaction():
                            repo.put("fact", "lost", 2)
                            repo.put("fact", "kept", 3)
                            repo.put_blob(b"lost blob")
                            repo.invalidate("fact:kept")
                            raise RuntimeError("simulated failure mid-transaction")
                    self.assertIsNone(repo.latest("fact", "lost"))
                    self.assertEqual(repo.latest("fact", "kept").payload, 1)
                    self.assertEqual(len(repo.history("fact", "kept")), 1)
                    self.assertFalse(repo.is_stale("fact", "kept"))
                    with self.assertRaises(KeyError):
                        repo.get_blob(hashlib.sha256(b"lost blob").hexdigest())
                    repo.put("fact", "after", 4)                     # the store is usable afterwards
                    self.assertEqual(repo.latest("fact", "after").version, 1)
                    if name == "sqlite":
                        self.assertEqual(_sidecars(repo.path), [])
                finally:
                    repo.close()

    def test_a_writer_that_dies_mid_transaction_leaves_its_journal_to_roll_back(self):
        path = self.root / "live" / es.DATABASE
        copy = self.root / "crashed" / es.DATABASE
        copy.parent.mkdir()
        repo = es.SQLiteRepository(path)
        try:
            self.assertEqual(repo.db.execute("PRAGMA journal_mode").fetchone()[0].lower(), "delete")
            repo.put("fact", "committed", 1)
            block = repo.transaction()
            block.__enter__()
            try:
                repo.put("fact", "uncommitted", 2)
                journal = Path(f"{path}-journal")
                self.assertTrue(journal.is_file())                  # rollback journal, never a WAL
                self.assertFalse(Path(f"{path}-wal").exists())
                shutil.copyfile(path, copy)                         # the disk as a crash would leave it
                shutil.copyfile(journal, Path(f"{copy}-journal"))
            finally:
                block.__exit__(RuntimeError, RuntimeError("stop"), None)
        finally:
            repo.close()
        if es._hot_journal(Path(f"{copy}-journal")):                # synced: pages may be in the file
            with self.assertRaises(es.StoreError):                  # read-only refuses a hot journal
                es.SQLiteRepository(copy, readonly=True)
        else:                                                       # not yet synced: the file is untouched
            with es.SQLiteRepository(copy, readonly=True) as early:  # and SQLite ignores the journal
                self.assertIsNone(early.latest("fact", "uncommitted"))
        recovered = es.SQLiteRepository(copy)                       # a writable open rolls it back
        try:
            self.assertIsNone(recovered.latest("fact", "uncommitted"))
            self.assertEqual(recovered.latest("fact", "committed").payload, 1)
        finally:
            recovered.close()
        inspect = es.SQLiteRepository(copy, readonly=True)
        try:
            self.assertEqual(inspect.latest("fact", "committed").payload, 1)
        finally:
            inspect.close()

    def test_a_journal_without_its_header_magic_is_not_hot(self):
        path = self.root / "store" / es.DATABASE
        repo = es.SQLiteRepository(path)
        repo.put("fact", "a", 1)
        repo.close()
        journal = Path(f"{path}-journal")
        journal.write_bytes(bytes(512))                             # zeroed header: never synced
        with es.SQLiteRepository(path, readonly=True) as view:
            self.assertEqual(view.latest("fact", "a").payload, 1)
        journal.write_bytes(es.JOURNAL_MAGIC + bytes(504))          # synced header: hot (even with nRec 0:
        with self.assertRaises(es.StoreError):                      # rollback still truncates to the old size)
            es.SQLiteRepository(path, readonly=True)

    def test_a_persist_mode_journal_zeroed_at_commit_is_not_hot(self):
        path = self.root / "store" / es.DATABASE
        repo = es.SQLiteRepository(path)
        repo.put("fact", "a", 1)
        repo.close()
        raw = sqlite3.connect(path, isolation_level=None)           # another tool, journal_mode=PERSIST
        try:
            self.assertEqual(raw.execute("PRAGMA journal_mode=PERSIST").fetchone()[0].lower(), "persist")
            raw.execute("CREATE TABLE other_tool (x INTEGER)")
        finally:
            raw.close()
        journal = Path(f"{path}-journal")
        self.assertGreater(journal.stat().st_size, 0)               # kept, its header zeroed at commit
        self.assertFalse(es._hot_journal(journal))
        with es.SQLiteRepository(path, readonly=True) as view:      # the committed state is readable
            self.assertEqual(view.latest("fact", "a").payload, 1)
            self.assertIsNotNone(view.db.execute("SELECT 1 FROM sqlite_master WHERE name = 'other_tool'")
                                 .fetchone())

    def test_a_read_only_view_takes_sqlite_locks_against_a_live_writer(self):
        path = self.root / "store" / es.DATABASE
        repo = es.SQLiteRepository(path)
        views = []
        try:
            repo.put("fact", "a", 1)
            views.append(es.SQLiteRepository(path, readonly=True))
            self.assertEqual(views[0].latest("fact", "a").payload, 1)
            with repo.transaction():                                # RESERVED; journal written, not synced
                repo.put("fact", "a", 2)
                journal = Path(f"{path}-journal")
                self.assertTrue(journal.is_file())
                self.assertFalse(es._hot_journal(journal))
                views.append(es.SQLiteRepository(path, readonly=True))   # admitted: the file is untouched
                for view in views:
                    self.assertEqual(view.latest("fact", "a").payload, 1)
            for view in views:                                      # the commit is seen whole, never a
                self.assertEqual(view.latest("fact", "a").payload, 2)    # stale cache of the old file
            early = es.open_readonly(path, timeout=0)
            try:
                repo.db.execute("BEGIN EXCLUSIVE")                  # a writer holding its commit lock
                try:
                    with self.assertRaises(sqlite3.OperationalError):   # busy: no read through the lock
                        early.execute("SELECT COUNT(*) FROM records").fetchone()
                finally:
                    repo.db.execute("ROLLBACK")
                self.assertEqual(early.execute("SELECT COUNT(*) FROM records").fetchone()[0], 2)
            finally:
                early.close()
        finally:
            for view in views:
                view.close()
            repo.close()
        self.assertEqual(_sidecars(path), [])

    def test_read_only_inspection_writes_nothing(self):
        path = self.root / "store" / es.DATABASE
        repo = es.SQLiteRepository(path)
        _operations(repo)
        repo.close()
        before = _snapshot(path.parent)
        inspect = es.SQLiteRepository(path, readonly=True)
        try:
            self.assertEqual(_sidecars(path), [])
            self.assertEqual(inspect.latest("fact", "a").payload, {"value": 2})
            self.assertTrue(inspect.is_stale("derived", "c"))
            self.assertTrue(inspect.export_lines())
            for write in (lambda: inspect.put("fact", "x", 1), lambda: inspect.put_blob(b"x"),
                          lambda: inspect.invalidate("fact:a")):
                with self.assertRaises(es.StoreError):
                    write()
        finally:
            inspect.close()
        self.assertEqual(_snapshot(path.parent), before)

    def test_a_wal_database_is_refused_read_only_and_converted_when_writable(self):
        path = self.root / "wal" / es.DATABASE
        path.parent.mkdir()
        raw = sqlite3.connect(path)
        raw.execute("PRAGMA journal_mode=WAL")
        raw.execute("CREATE TABLE foreign_table (x INTEGER)")
        raw.commit()
        raw.close()
        with self.assertRaises(es.StoreError):
            es.SQLiteRepository(path, readonly=True)
        repo = es.SQLiteRepository(path)
        try:
            self.assertEqual(repo.schema, es.SCHEMA_VERSION)
            repo.put("fact", "a", 1)
        finally:
            repo.close()
        inspect = es.SQLiteRepository(path, readonly=True)
        try:
            self.assertEqual(inspect.latest("fact", "a").payload, 1)
        finally:
            inspect.close()
        self.assertEqual(_sidecars(path), [])

    def test_migration_from_schema_1_keeps_every_record(self):
        path = self.root / "old" / es.DATABASE
        old = es.SQLiteRepository(path, schema=1)
        try:
            record = old.put("fact", "a", {"value": 1}, provenance={"file": "f.json"})
            with self.assertRaises(es.SchemaError):
                old.put("fact", "b", 2, depends_on=["fact:a"])
        finally:
            old.close()
        with self.assertRaises(es.SchemaError):                     # read-only never migrates
            es.SQLiteRepository(path, readonly=True)
        repo = es.SQLiteRepository(path)
        try:
            self.assertEqual(repo.schema, es.SCHEMA_VERSION)
            self.assertEqual(repo.latest("fact", "a"), record)
            self.assertEqual(repo.put("fact", "a", {"value": 1}, provenance={"file": "f.json"}), record)
            repo.put("fact", "b", 2, depends_on=["fact:a"])
            self.assertEqual(repo.invalidate("fact:a"), ["fact:a", "fact:b"])
            versions = [row[0] for row in repo.db.execute("SELECT version FROM schema_version ORDER BY version")]
            self.assertEqual(versions, list(range(1, es.SCHEMA_VERSION + 1)))
            repo.db.execute("INSERT INTO schema_version VALUES (?, 0, 'from the future')", (es.SCHEMA_VERSION + 1,))
        finally:
            repo.close()
        for readonly in (False, True):
            with self.subTest(readonly=readonly), self.assertRaises(es.SchemaError):
                es.SQLiteRepository(path, readonly=readonly)

    def test_invalidation_marks_dependents_transitively_until_a_new_version(self):
        for name, make in self.stores():
            with self.subTest(store=name):
                repo = make()
                try:
                    repo.put("fact", "a", 1)
                    repo.put("derived", "b", 2, depends_on=["fact:a"])
                    repo.put("derived", "c", 3, depends_on=["derived:b"])
                    repo.put("derived", "loop", 4, depends_on=["derived:c", "derived:loop"])
                    repo.put("other", "d", 5, depends_on=["file:x"])
                    self.assertEqual(repo.invalidate("fact:a"), ["fact:a", "derived:b", "derived:c", "derived:loop"])
                    self.assertEqual(repo.stale(), ["derived:b", "derived:c", "derived:loop", "fact:a"])
                    self.assertFalse(repo.is_stale("other", "d"))
                    self.assertEqual(repo.invalidate("file:x"), ["other:d"])
                    self.assertEqual(repo.invalidate("nothing:depends"), [])
                    renewed = repo.put("derived", "b", 2, depends_on=["fact:a"])     # same content, but stale
                    self.assertEqual(renewed.version, 2)
                    self.assertFalse(repo.is_stale("derived", "b"))
                    self.assertTrue(repo.is_stale("derived", "c"))
                finally:
                    repo.close()

    def test_export_is_deterministic_and_refuses_rom_derived_content(self):
        exports = []
        for name, make in self.stores():
            with self.subTest(store=name):
                repo = make()
                try:
                    _operations(repo)
                    exports.append(repo.export())
                    self.assertEqual(repo.export(), exports[-1])
                    repo.put_blob(b"bytes copied from a ROM", rom_derived=True)
                    with self.assertRaises(es.ExportRefused):
                        repo.export()
                    text = repo.export(self.root / f"export_{name}.jsonl", omit_rom_derived=True)
                    rows = [json.loads(line) for line in text.splitlines()]
                    rom = [r for r in rows if r.get("rom_derived")]
                    self.assertEqual(len(rom), 1)
                    self.assertNotIn("data", rom[0])
                    self.assertNotIn("Ynl0ZXMgY29waWVk", text)        # base64 of the flagged bytes
                    self.assertEqual((self.root / f"export_{name}.jsonl").read_text(), text)
                    repo.put("note", "from-rom", {"words": [1]}, provenance={"rom_derived": True})
                    with self.assertRaises(es.ExportRefused):
                        repo.export_lines()
                    row = [json.loads(line) for line in repo.export_lines(omit_rom_derived=True)
                           if '"from-rom"' in line][0]
                    self.assertEqual((row["payload"], row["omitted"]), (None, "rom-derived"))
                finally:
                    repo.close()
        self.assertEqual(exports[0], exports[1])                      # memory and SQLite agree byte for byte
        rows = [json.loads(line) for line in exports[0].splitlines()]
        self.assertEqual(rows[0], {"type": "header", "format": es.FORMAT, "schema": es.SCHEMA_VERSION})
        self.assertEqual([r["type"] for r in rows], ["header", "blob"] + ["record"] * 5 + ["stale"] * 3)
        self.assertFalse(any("created_ns" in line for line in exports[0].splitlines()))

    def test_import_of_a_run_directory_reads_only_and_is_idempotent(self):
        run = self.root / "run"
        (run / "solver").mkdir(parents=True)
        (run / "main").mkdir()
        (run / "discovery").mkdir()
        (run / "solver" / "invocation.json").write_text(json.dumps({"effort": "exhaustive", "budget": 400}))
        (run / "report.json").write_text(json.dumps({"status": "complete", "images": {"main": {"functions": 2}}}))
        (run / "solver" / "coverage_main.json").write_text(json.dumps({"functions": 2, "entries": []}))
        (run / "main" / "stage_ledger.json").write_text(json.dumps({"totals": {"extent_bytes": 64}}))
        (run / "main" / "stage_ledger.meta.json").write_text(json.dumps({"identity": "x"}))
        (run / "discovery" / "repair_main.json").write_text(json.dumps({"added": [], "audits": []}))
        (run / "rounds.jsonl").write_text('{"kind": "round", "round": 1}\n{"kind": "sto')
        (run / "events.jsonl").write_text('{"event": "image-start"}\n')
        before = _snapshot(run)
        self.assertEqual(es.run_files(run), [
            "solver/invocation.json", "report.json", "discovery/repair_main.json", "solver/coverage_main.json",
            "main/stage_ledger.json", "rounds.jsonl"])
        repo = es.SQLiteRepository(self.root / "store" / es.DATABASE)
        try:
            result = es.import_run(repo, run, label="t")
            self.assertEqual(len(result["imported"]), 6)
            self.assertEqual(result["skipped"], {})
            rounds = repo.latest("run-file", "t/rounds.jsonl")
            self.assertEqual(rounds.payload, [{"kind": "round", "round": 1}])
            self.assertEqual(rounds.provenance["unreadable_lines"], 1)
            report = repo.latest("run-file", "t/report.json")
            self.assertEqual(report.provenance["sha256"], hashlib.sha256((run / "report.json").read_bytes()).hexdigest())
            self.assertIn("run-file:t/solver/invocation.json", report.depends_on)
            again = es.import_run(repo, run, label="t")
            self.assertEqual([row["version"] for row in again["imported"]], [1] * 6)
            marked = repo.invalidate("run-file:t/solver/invocation.json", cause="a resumed invocation")
            self.assertEqual(len(marked), 6)
            self.assertTrue(repo.is_stale("run-file", "t/main/stage_ledger.json"))
            small = es.import_run(repo, run, label="small", limit=10)
            self.assertIn("report.json", small["skipped"])
        finally:
            repo.close()
        self.assertEqual(_snapshot(run), before)
        with self.assertRaises(FileNotFoundError):
            es.import_run(es.MemoryRepository(), self.root / "store")


if __name__ == "__main__":
    unittest.main()
