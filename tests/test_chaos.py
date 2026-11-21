"""Chaos: processes killed outright (SIGKILL: no handler runs) in the middle of checkpoints,
stale leases and locks, truncated result files, a corrupted cache object, an interrupted
evidence-store transaction and missing optional tools.  What must hold every time:

* resume reproduces or recomputes: never reads a torn checkpoint as data;
* nothing is ever credited from a damaged artifact (no exact candidate appears, a corrupt object is
  never returned);
* no descendant process survives a job that dies, where the kit promises it (guarded tools).

Tests marked ``expectedFailure`` document gaps found by this file (CHAOS_TESTING.md): C1, C2,
C2b and C4 are fixed and their tests now pass; C3 (a native tool outliving a job killed outright
under the default death-guard scope) is a recorded design choice.
Fixing C3 would turn its test into an unexpected success, which fails the suite until the marker is removed.

Every child here is a short synthetic Python process; every tool is ``sleep``; all are reaped."""
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest import mock

from n64dk import safety
from n64dk.match.cache import CacheError, ContentCache


def gone(pid: int, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while True:
        try:
            with open(f"/proc/{pid}/stat") as handle:
                if handle.read().rsplit(")", 1)[1].split()[0] == "Z":
                    return True
        except (FileNotFoundError, ProcessLookupError):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


class ChaosCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="n64dk-chaos-"))
        self.state = self.root / "state"
        self.overrides = {"N64DK_STATE_DIR": str(self.state), "N64DK_TOOL_FLOOR_MB": "0",
                          "N64DK_PAUSE_LIMIT_S": "0", "N64DK_TOKEN_DEADLINE_S": "30"}
        self.patch = mock.patch.dict(os.environ, self.overrides)
        self.patch.start()
        self.children: list[subprocess.Popen] = []
        self.strays: list[int] = []

    def tearDown(self):
        for child in self.children:
            if child.poll() is None:
                child.kill()
            child.wait()
            if child.stdout:
                child.stdout.close()
        for pid in self.strays:                       # never leave a descendant of the test itself
            if not gone(pid, 0):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.patch.stop()
        shutil.rmtree(self.root, ignore_errors=True)   # each case left ~8 MB in a RAM-backed /tmp

    def child(self, script: str, drop=()) -> subprocess.Popen:
        """Start ``script``; return once it printed its first line (``ready``)."""
        env = {**os.environ, "PYTHONPATH": os.getcwd()}
        for name in drop:
            env.pop(name, None)
        process = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True, env=env)
        self.children.append(process)
        line = process.stdout.readline().strip()
        self.assertEqual(line, "ready", f"child did not start (exit {process.poll()})")
        return process

    def kill(self, process: subprocess.Popen) -> None:
        process.send_signal(signal.SIGKILL)
        process.wait(timeout=10)


# ---- leases and the machine lock -------------------------------------------------------------
class LeaseChaosTest(ChaosCase):
    def test_sigkill_leaves_a_stale_lease_that_never_blocks_the_next_job(self):
        job = self.child("import time\nfrom pathlib import Path\nfrom n64dk import safety\n"
                         "with safety.heavy_job('chaos', Path('.'), force=True):\n"
                         "    print('ready', flush=True)\n    time.sleep(60)\n")
        self.kill(job)                                               # no handler runs: the lease stays
        self.assertEqual([lease["state"] for lease in safety.leases()], ["stale"])
        self.assertFalse(safety.cancel(job.pid)["cancelled"])        # nothing alive is touched
        self.assertEqual(safety.leases(), [])
        with safety.machine_lock():                                  # the kernel released the flock
            pass


# ---- descendants -----------------------------------------------------------------------------
TOOL = ("import sys\nfrom n64dk import safety\nprint('ready', flush=True)\n"
        "safety.run_tool(['sh', '-c', 'echo $$ > {pidfile}.tmp; mv {pidfile}.tmp {pidfile}; exec sleep 30'], timeout=60)\n")


class DescendantTest(ChaosCase):
    def tool_after_sigkill(self, scope: str | None) -> int:
        pidfile = self.root / "tool.pid"
        if scope:
            os.environ["N64DK_PDEATHSIG_SCOPE"] = scope
        job = self.child(TOOL.format(pidfile=pidfile), drop=() if scope else ("N64DK_PDEATHSIG_SCOPE",))
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not pidfile.is_file():
            time.sleep(0.02)
        tool = int(pidfile.read_text())
        self.strays.append(tool)
        self.kill(job)
        return tool

    def test_a_guarded_tool_dies_with_a_job_killed_outright(self):
        self.assertTrue(gone(self.tool_after_sigkill("all"), 3), "the tool outlived its SIGKILLed job")

    @unittest.expectedFailure
    def test_an_unguarded_tool_dies_with_a_job_killed_outright(self):
        """Finding C3: under the default N64DK_PDEATHSIG_SCOPE=long a native tool is not guarded
        (by design, to save ~0.7 ms per launch); a SIGKILLed job leaves it running until it ends
        by itself.  The chaos run measures whether real compiler passes do."""
        self.assertTrue(gone(self.tool_after_sigkill(None), 0.5), "the tool outlived its SIGKILLed job")


# ---- run checkpoints (decompile-run stages, rounds log) -----------------------------------------
class CheckpointTest(ChaosCase):
    def setUp(self):
        super().setUp()
        from n64dk.commands import run as runmod
        from n64dk.match import solver
        self.stage, self.solver = runmod._stage, solver
        self.folder = self.root / "solver"

    def test_a_crashed_stage_leaves_no_checkpoint(self):
        def compute():
            raise RuntimeError("killed in the middle")
        with self.assertRaises(RuntimeError):
            self.stage(self.folder, "calibrate", compute, False, "id-1")
        self.assertFalse((self.folder / "stage_calibrate.json").exists())

    def test_a_checkpoint_without_its_identity_record_is_recomputed(self):
        self.folder.mkdir()
        (self.folder / "stage_calibrate.json").write_text(json.dumps({"chosen": "stale"}))   # meta never written
        result, cached = self.stage(self.folder, "calibrate", lambda: {"chosen": "fresh"}, False, "id-1")
        self.assertEqual((result, cached), ({"chosen": "fresh"}, False))
        again, cached = self.stage(self.folder, "calibrate", lambda: {"chosen": "other"}, False, "id-1")
        self.assertEqual((again, cached), ({"chosen": "fresh"}, True))                      # reproduced

    def truncated(self):
        self.folder.mkdir()
        (self.folder / "stage_calibrate.json").write_text('{"chosen": {"label": "kmc')
        (self.folder / "stage_calibrate.meta.json").write_text(json.dumps({"identity": "id-1"}))

    def test_a_truncated_checkpoint_is_never_read_as_data(self):
        self.truncated()
        result, cached = self.stage(self.folder, "calibrate", lambda: {"chosen": "fresh"}, False, "id-1")
        self.assertFalse(cached)
        self.assertEqual(json.loads((self.folder / "stage_calibrate.json").read_text()), {"chosen": "fresh"})

    def test_a_truncated_checkpoint_is_recomputed(self):
        """Finding C1 (fixed): a stage file that does not parse is recomputed, not turned into a
        stage error that lets the run go on without the stage."""
        self.truncated()
        result, cached = self.stage(self.folder, "calibrate", lambda: {"chosen": "fresh"}, False, "id-1")
        self.assertEqual((result, cached), ({"chosen": "fresh"}, False))

    def test_sigkill_while_checkpointing_leaves_the_old_or_the_new_file(self):
        script = (f"from pathlib import Path\nfrom n64dk.commands.run import _stage\n"
                  f"folder = Path({str(self.folder)!r})\nn = 0\n"
                  "while True:\n"
                  "    n += 1\n"
                  "    _stage(folder, 'calibrate', lambda: {'n': n, 'pad': 'x' * 400000}, True, 'id-1')\n"
                  "    if n == 1:\n        print('ready', flush=True)\n")
        job = self.child(script)
        time.sleep(0.3)
        self.kill(job)
        payload = json.loads((self.folder / "stage_calibrate.json").read_text())       # never torn
        self.assertGreaterEqual(payload["n"], 1)
        meta_present = (self.folder / "stage_calibrate.meta.json").is_file()
        result, cached = self.stage(self.folder, "calibrate", lambda: {"n": -1}, False, "id-1")
        if cached:                                      # the checkpoint and its identity were both written:
            self.assertTrue(meta_present)               # the resume reproduces it exactly
            self.assertEqual(result["n"], payload["n"])
        else:                                           # killed between the checkpoint and its identity record:
            self.assertFalse(meta_present)              # recomputed, never the new result under an old identity
            self.assertEqual(result, {"n": -1})

    def test_a_torn_last_line_of_the_rounds_log_is_ignored(self):
        log = self.solver.RoundsLog(self.root / "rounds.jsonl")
        log.path.write_text(json.dumps({"session": "s", "image": "main", "kind": "round", "round": 1}) + "\n"
                            + '{"session": "s", "image": "main", "kind": "round", "rou')
        self.assertEqual(log.progress("s", "main"), (1, None))

    def test_the_first_record_after_a_torn_line_is_kept(self):
        """Finding C4 (low: a SIGKILL cannot tear one small write; power loss or a full disk can):
        ``append`` does not start a new line after a torn one, so the resumed run's first record is
        glued to the fragment and lost."""
        log = self.solver.RoundsLog(self.root / "rounds.jsonl")
        log.path.write_text('{"session": "s", "image": "main", "kind": "round", "rou')
        log.append({"session": "s", "image": "main", "kind": "round", "round": 2})
        self.assertEqual(log.progress("s", "main"), (2, None))

    def test_a_crash_between_checkpoint_and_identity_never_mislabels(self):
        """R2: the stage file is replaced before its identity record.  A crash in between used to
        leave the new result under the old identity, reused once the inputs went back."""
        self.stage(self.folder, "calibrate", lambda: {"chosen": "for-1"}, False, "id-1")
        with mock.patch.object(Path, "write_text", side_effect=OSError("killed before the identity")):
            with self.assertRaises(OSError):
                self.stage(self.folder, "calibrate", lambda: {"chosen": "for-2"}, False, "id-2")
        result, cached = self.stage(self.folder, "calibrate", lambda: {"chosen": "for-1 again"}, False, "id-1")
        self.assertEqual((result, cached), ({"chosen": "for-1 again"}, False))

    def test_a_full_disk_leaves_no_temporary(self):
        with mock.patch("os.fsync", side_effect=OSError(28, "No space left on device")):
            with self.assertRaises(OSError):
                self.stage(self.folder, "calibrate", lambda: {"chosen": "x"}, False, "id-1")
        self.assertEqual(list(self.folder.iterdir()), [])


# ---- matching sessions (SQLite, WAL) -------------------------------------------------------------
class SessionChaosTest(ChaosCase):
    def test_sigkill_during_appends_keeps_the_store_whole_and_credits_nothing(self):
        store_root = self.root / "sessions"
        script = (
            "import time\nfrom n64dk.match.session import SessionCandidate, SessionStore\n"
            f"store = SessionStore({str(store_root)!r})\n"
            "session = store.create({'chaos': True})\nn = 0\n"
            "while True:\n"
            "    n += 1\n"
            "    store.append(session, SessionCandidate(f'c{n}', None, 'ab' * 32, 'label', 'h', 'kmc', ('-O2',),\n"
            "                 'evaluated', False, 100 + n, {'insn': n}, {'pad': 'x' * 2000}, None, time.time_ns()))\n"
            "    if n == 1:\n        print('ready', flush=True)\n")
        job = self.child(script)
        time.sleep(0.3)
        self.kill(job)
        from n64dk.match.session import SessionStore
        store = SessionStore(store_root)                        # a writer's death is recovered on open
        try:
            self.assertEqual(store.db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            [session] = [row[0] for row in store.db.execute("SELECT session_id FROM sessions")]
            rows = store.candidates(session)
            self.assertGreaterEqual(len(rows), 1)
            self.assertFalse([row for row in rows if row.exact])  # nothing credited by a crash
        finally:
            store.close()


# ---- the content cache ------------------------------------------------------------------------
class CacheChaosTest(ChaosCase):
    def test_sigkill_during_puts_never_leaves_a_torn_object(self):
        cache_root = self.root / "cache"
        job = self.child(f"import os\nfrom n64dk.match.cache import ContentCache\n"
                         f"cache = ContentCache({str(cache_root)!r})\nn = 0\n"
                         "while True:\n    cache.put(os.urandom(1 << 18))\n    n += 1\n"
                         "    if n == 1:\n        print('ready', flush=True)\n")
        time.sleep(0.3)
        self.kill(job)
        cache = ContentCache(cache_root)
        objects = [p for p in cache.objects.rglob("*") if p.is_file() and not p.name.startswith(".")]
        self.assertGreaterEqual(len(objects), 1)
        for path in objects:
            digest = path.parent.name + path.name
            self.assertEqual(cache.get(digest), path.read_bytes())              # every object verifies

    def corrupt(self):
        cache = ContentCache(self.root / "cache")
        digest, produced = cache.resolve("unit-key", lambda: b"object bytes")
        path = cache.path(digest)
        path.write_bytes(b"object bytez")
        return cache

    def test_a_corrupt_object_is_never_returned(self):
        cache = self.corrupt()
        digest = cache.get_ref("unit-key")
        with self.assertRaises(CacheError):
            cache.get(digest)                                                   # get never returns it
        self.assertEqual(cache.put(b"object bytes"), digest)                    # a put heals it
        self.assertEqual(cache.get(digest), b"object bytes")

    def test_a_corrupt_object_is_rebuilt(self):
        """Finding C2 (fixed): ``resolve`` re-hashes a hit; a corrupt object is rebuilt and replaced."""
        cache = self.corrupt()
        digest, produced = cache.resolve("unit-key", lambda: b"object bytes")
        self.assertTrue(produced)
        self.assertEqual(cache.get(digest), b"object bytes")

    def test_a_truncated_ref_fails_closed(self):
        cache = ContentCache(self.root / "cache")
        cache.resolve("unit-key", lambda: b"object bytes")
        [ref] = list(cache.refs.iterdir())
        ref.write_text(ref.read_text()[:20])
        with self.assertRaises(ValueError):
            cache.get_ref("unit-key")

    def test_a_lock_left_by_a_killed_holder_fails_closed(self):
        cache_root = self.root / "cache"
        job = self.child(f"import time\nfrom n64dk.match.cache import ContentCache\n"
                         f"with ContentCache({str(cache_root)!r}).lock('unit-key'):\n"
                         "    print('ready', flush=True)\n    time.sleep(60)\n")
        self.kill(job)
        with ContentCache(cache_root).lock("unit-key", timeout=0.2):            # finding C2b fixed: the
            pass                                                                # dead holder's lock is broken

    def test_a_live_holder_keeps_its_lock(self):
        cache_root = self.root / "cache"
        job = self.child(f"import time\nfrom n64dk.match.cache import ContentCache\n"
                         f"with ContentCache({str(cache_root)!r}).lock('unit-key'):\n"
                         "    print('ready', flush=True)\n    time.sleep(60)\n")
        try:
            with self.assertRaises(CacheError):
                with ContentCache(cache_root).lock("unit-key", timeout=0.3):
                    pass
        finally:
            self.kill(job)

    def test_a_truncated_ref_is_a_miss_for_resolve(self):
        cache = ContentCache(self.root / "cache")
        cache.resolve("unit-key", lambda: b"object bytes")
        [ref] = list(cache.refs.iterdir())
        ref.write_text(ref.read_text()[:20])
        digest, produced = cache.resolve("unit-key", lambda: b"object bytes")
        self.assertTrue(produced)
        self.assertEqual(cache.get_ref("unit-key"), digest)

    def stale_lock(self, cache: ContentCache) -> Path:
        path = cache.locks / cache._safe_key("unit-key")
        path.mkdir()
        (path / cache.OWNER).write_text("{torn")                               # judged stale
        return path

    def test_a_lock_that_cannot_be_broken_still_times_out(self):
        """R2: a stale lock whose rename fails (permissions, a busy directory on a Windows mount)
        made every waiter spin without sleeping and past its deadline."""
        cache = ContentCache(self.root / "cache")
        self.stale_lock(cache)
        calls = []

        def refuse(*args):
            calls.append(args)
            if len(calls) > 1000:
                raise RuntimeError("spinning past the deadline")
            raise PermissionError(13, "rename refused")
        with mock.patch.object(Path, "rename", side_effect=refuse):
            with self.assertRaises(CacheError):
                with cache.lock("unit-key", timeout=0.2):
                    pass
        self.assertLess(len(calls), 100)

    def test_a_break_whose_cleanup_fails_still_takes_the_lock(self):
        cache = ContentCache(self.root / "cache")
        self.stale_lock(cache)
        with mock.patch.object(Path, "rmdir", side_effect=OSError(16, "Device or resource busy")):
            with cache.lock("unit-key", timeout=0.2):                          # litter, not an error
                pass

    def test_a_second_breaker_leaves_a_fresh_lock_alone(self):
        """R2: two waiters judged one dead lock stale; the first broke it and took a fresh lock
        under the same name.  The second's break must re-judge, not rename the fresh one away."""
        cache = ContentCache(self.root / "cache")
        path = cache.locks / cache._safe_key("unit-key")
        with cache.lock("unit-key"):                                            # owned and alive
            self.assertFalse(cache._break(path))
            self.assertTrue((path / cache.OWNER).is_file())
        path.mkdir()                                       # between a holder's mkdir and its record
        self.assertFalse(cache._break(path))
        self.assertTrue(path.is_dir())

    def test_without_proc_a_live_holder_keeps_its_lock(self):
        """R2: with no start time (no /proc) every lock read as stale, so any waiter broke a live
        holder's lock.  The pid alone is asked instead."""
        cache = ContentCache(self.root / "cache")
        path = cache.locks / cache._safe_key("unit-key")
        with mock.patch.object(safety, "_start_time", return_value=None):
            with cache.lock("unit-key"):
                self.assertFalse(cache._stale(path))
            dead = subprocess.Popen(["true"])
            dead.wait()
            path.mkdir()
            (path / cache.OWNER).write_text(json.dumps({"pid": dead.pid, "start_time": None}))
            self.assertTrue(cache._stale(path))

    def test_a_lock_directory_that_will_not_go_is_left_ownerless(self):
        """R2: on a Windows mount the release's rmdir can fail (a handle still open); that must not
        turn a finished body into an error."""
        cache = ContentCache(self.root / "cache")
        with mock.patch.object(Path, "rmdir", side_effect=OSError(39, "Directory not empty")):
            with cache.lock("unit-key"):
                pass
        self.assertFalse((cache.locks / cache._safe_key("unit-key") / cache.OWNER).exists())


# ---- the evidence store (rollback journal) ------------------------------------------------------
@unittest.skipUnless(find_spec("n64dk.match.evidence_store"), "evidence store not in this tree")
class EvidenceChaosTest(ChaosCase):
    def test_sigkill_inside_a_transaction_rolls_back(self):
        from n64dk.match import evidence_store as es
        path = self.root / "evidence" / es.DATABASE
        job = self.child(f"import time\nfrom n64dk.match import evidence_store as es\n"
                         f"repo = es.SQLiteRepository({str(path)!r})\n"
                         "repo.put('fact', 'a', {'v': 1})\n"
                         "with repo.transaction():\n"
                         "    repo.put('fact', 'a', {'v': 2})\n"
                         "    repo.put_blob(b'x' * 100000)\n"
                         "    print('ready', flush=True)\n    time.sleep(60)\n")
        self.kill(job)
        journal = Path(f"{path}-journal")
        if journal.is_file() and es._hot_journal(journal):
            with self.assertRaises(es.StoreError):                              # no read through a hot journal
                es.SQLiteRepository(path, readonly=True)
        else:                                       # not synced yet: the database file is untouched
            with es.SQLiteRepository(path, readonly=True) as early:
                self.assertEqual(early.latest("fact", "a").payload, {"v": 1})
        repo = es.SQLiteRepository(path)                                        # rolls the journal back
        try:
            self.assertEqual(repo.latest("fact", "a").payload, {"v": 1})
            self.assertEqual(repo.db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        finally:
            repo.close()
        with es.SQLiteRepository(path, readonly=True) as view:
            self.assertEqual(view.latest("fact", "a").version, 1)


# ---- missing optional tools ---------------------------------------------------------------------
class MissingToolTest(ChaosCase):
    def test_without_setpriv_the_guard_falls_back_to_prctl(self):
        with mock.patch.object(safety, "SETPRIV", str(self.root / "no-setpriv")), \
                mock.patch.dict(os.environ, {"N64DK_PDEATHSIG": "setpriv"}):
            self.assertEqual(safety.pdeathsig_mode(), "prctl")
            self.assertEqual(safety.death_guarded(["sleep", "1"]), ["sleep", "1"])

    def test_a_missing_tool_fails_at_once_and_returns_its_slot(self):
        started = time.monotonic()
        with self.assertRaises(FileNotFoundError):
            safety.run_tool([str(self.root / "no-such-compiler"), "-c", "x.c"], timeout=30)
        self.assertLess(time.monotonic() - started, 5)
        with safety.token("light", deadline=0, weight=safety.capacity("light")):    # every slot free again
            pass

    def test_a_missing_tool_in_a_guarded_launch_is_not_disguised(self):
        os.environ["N64DK_PDEATHSIG_SCOPE"] = "all"
        with self.assertRaises(FileNotFoundError):                             # not setpriv's exit 127
            safety.run_tool([str(self.root / "no-such-compiler")], timeout=30)


if __name__ == "__main__":
    unittest.main()
