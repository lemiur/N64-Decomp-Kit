"""Content-addressed build jobs, the local worker loop and the coordinator's verification
(``n64dk.workers``).  The "compiler" is a small Python script written into a temporary toolchain
folder: it turns ``word XXXXXXXX`` lines (from the source and its included headers) into a minimal
big-endian MIPS ELF object.  No toolchain, no ROM; every subprocess is short and reaped."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from n64dk import workers
from n64dk.match.cache import ContentCache

FAKECC = r'''
import os, struct, sys, time
args = sys.argv[1:]
out = args[args.index("-o") + 1]
include = args[args.index("-I") + 1]
source = args[-1]
flags = [a for a in args if a.startswith("-D")]
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.log"), "a") as log:
    log.write("run\n")
words, relocs = [], []
def read(path):
    for line in open(path):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "#include":
            read(os.path.join(include, parts[1].strip('"')))
        elif parts[0] == "word":
            words.append(int(parts[1], 16))
        elif parts[0] == "reloc":
            relocs.append(4 * len(words))
        elif parts[0] == "sleep":
            print(os.getpid(), flush=True)
            open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tool.pid"), "w").write(str(os.getpid()))
            time.sleep(float(parts[1]))
        elif parts[0] == "fail":
            sys.stderr.write("error: asked to fail\n")
            sys.exit(1)
read(source)
if "-DINVERT" in flags and words:
    words[0] ^= 1
code = b"".join(struct.pack(">I", w) for w in words)
shstr = b"\0.text\0.rel.text\0.symtab\0.strtab\0.shstrtab\0"
strtab = b"\0f\0"
symtab = bytes(16) + struct.pack(">IIIBBH", 1, 0, len(code), 0x12, 0, 1)
rel = b"".join(struct.pack(">II", off, (1 << 8) | 4) for off in relocs)
body = [code, rel, symtab, strtab, shstr]
offsets, position = [], 52
for part in body:
    offsets.append(position)
    position += len(part)
sections = [bytes(40),
            struct.pack(">10I", 1, 1, 6, 0, offsets[0], len(code), 0, 0, 4, 0),
            struct.pack(">10I", 7, 9, 0, 0, offsets[1], len(rel), 3, 1, 4, 8),
            struct.pack(">10I", 17, 2, 0, 0, offsets[2], len(symtab), 4, 1, 4, 16),
            struct.pack(">10I", 25, 3, 0, 0, offsets[3], len(strtab), 0, 0, 1, 0),
            struct.pack(">10I", 33, 3, 0, 0, offsets[4], len(shstr), 0, 0, 1, 0)]
header = (b"\x7fELF\x01\x02\x01" + bytes(9) + struct.pack(">HHIIIIIHHHHHH", 1, 8, 1, 0, 0, position, 0,
                                                            52, 0, 0, 40, len(sections), 5))
open(out, "wb").write(header + b"".join(body) + b"".join(sections))
'''

CODE = bytes.fromhex("27bdffe8afbf001403e0000800000000")          # four words


def source_of(code: bytes, extra: str = "") -> bytes:
    return ("".join(f"word {code[i:i + 4].hex()}\n" for i in range(0, len(code), 4)) + extra).encode()


class WorkerCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="n64dk-workers-test-"))
        self.env = mock.patch.dict(os.environ, {
            "N64DK_STATE_DIR": str(self.root / "state"), "N64DK_TOOL_FLOOR_MB": "0", "N64DK_PAUSE_LIMIT_S": "0",
            "N64DK_TOKENS_COMPILE": "4", "N64DK_TOKEN_DEADLINE_S": "30"})
        self.env.start()
        self.toolchain = self.root / "toolchain"
        self.toolchain.mkdir()
        (self.toolchain / "fakecc.py").write_text(FAKECC)
        self.recipe = workers.Recipe("fake-cc", (sys.executable, str(self.toolchain / "fakecc.py"), "{flags}",
                                                 "-I", "{include}", "-o", "{output}", "{source}"),
                                     toolchain_digest="fake-1", allowed_flags=frozenset({"-O2", "-DINVERT"}))
        self.recipes = {self.recipe.name: self.recipe}
        self.cache = ContentCache(self.root / "cache")
        self.spool = workers.Spool(self.root / "spool")

    def tearDown(self):
        self.env.stop()

    def job(self, source=None, target=CODE, **options):
        return workers.make_job(self.recipe, self.cache, source if source is not None else source_of(CODE),
                                target, **options)

    def runs(self) -> int:
        log = self.toolchain / "runs.log"
        return len(log.read_text().splitlines()) if log.is_file() else 0


class JobIdentityTest(WorkerCase):
    def test_equal_work_has_one_name(self):
        a = self.job(headers={"b.h": b"word 1\n", "a.h": b"word 2\n"}, flags=("-O2",))
        b = self.job(headers={"a.h": b"word 2\n", "b.h": b"word 1\n"}, flags=("-O2",))
        self.assertEqual(a.job_id, b.job_id)
        for changed in (self.job(flags=("-DINVERT",)), self.job(source=source_of(CODE, "word 0\n")),
                        self.job(target=CODE[:8]), self.job(limits={"timeout_s": 5}),
                        self.job(headers={"a.h": b"word 3\n"})):
            self.assertNotEqual(changed.job_id, a.job_id)
        other = workers.Recipe("fake-cc", self.recipe.argv, "fake-2", self.recipe.allowed_flags)
        self.assertNotEqual(other.identity(), self.recipe.identity())     # another toolchain, another recipe

    def test_unsafe_header_paths_and_unknown_limits_are_refused(self):
        for name in ("../x.h", "/etc/passwd", "a/../../b.h", "", "a\\b.h"):
            with self.assertRaises(workers.JobError, msg=name):
                self.job(headers={name: b""})
        with self.assertRaises(workers.JobError):
            self.job(limits={"network": 1})


class WorkerLoopTest(WorkerCase):
    def test_build_receipt_and_local_verification(self):
        job_id, new = self.spool.submit(self.job(flags=("-O2",)))
        self.assertTrue(new)
        [receipt] = workers.run_worker(self.spool, self.cache, self.recipes, worker="w1")
        self.assertEqual(receipt["status"], "built", receipt)
        self.assertEqual(self.spool.pending(), [])
        self.assertEqual(list((self.spool.root / "claims").iterdir()), [])
        check = workers.verify(self.spool, self.cache, job_id, CODE)
        self.assertEqual(check.verdict, "exact-locally", check.reasons)
        self.assertFalse(check.credit)
        self.assertFalse(check.as_dict()["credit"])

    def test_a_different_build_is_different(self):
        job_id, _ = self.spool.submit(self.job(flags=("-DINVERT",)))
        workers.run_worker(self.spool, self.cache, self.recipes)
        self.assertEqual(workers.verify(self.spool, self.cache, job_id, CODE).verdict, "different")

    def test_duplicates_are_built_once(self):
        first, new = self.spool.submit(self.job())
        again, new_again = self.spool.submit(self.job())
        self.assertEqual(first, again)
        self.assertTrue(new)
        self.assertFalse(new_again)
        workers.run_worker(self.spool, self.cache, self.recipes)
        workers.run_worker(self.spool, self.cache, self.recipes)
        self.assertEqual(self.runs(), 1)
        self.assertEqual(self.spool.submit(self.job()), (first, False))      # done: not queued again

    def test_only_allowlisted_recipes_and_flags_run(self):
        stranger = workers.Recipe("other-cc", ("/bin/sh", "-c", "touch {output}"), "x")
        cases = [workers.make_job(stranger, self.cache, b"", CODE),                            # not registered here
                 workers.Job("fake-cc", "0" * 64, self.cache.put(b""), target=workers.sha256(CODE)),  # other identity
                 self.job(flags=("-fplugin=evil.so",))]                                          # flag not allowed
        for job in cases:
            self.spool.submit(job)
        receipts = workers.run_worker(self.spool, self.cache, self.recipes)
        self.assertEqual([r["status"] for r in receipts], ["rejected"] * 3)
        self.assertEqual(self.runs(), 0)

    def test_inputs_must_be_present_and_intact(self):
        job = self.job(source=source_of(CODE, "word 00000001\n"))
        self.cache.path(job.source).unlink()
        self.spool.submit(job)
        [receipt] = workers.run_worker(self.spool, self.cache, self.recipes)
        self.assertEqual(receipt["status"], "rejected")
        self.assertIn("missing or corrupt input", receipt["reason"])

    def test_headers_are_materialised_inside_the_job_folder(self):
        job = self.job(source=b'#include "sub/a.h"\nword 00000000\n', target=bytes.fromhex("1234567800000000"),
                       headers={"sub/a.h": b"word 12345678\n"})
        job_id, _ = self.spool.submit(job)
        workers.run_worker(self.spool, self.cache, self.recipes)
        self.assertEqual(workers.verify(self.spool, self.cache, job_id, bytes.fromhex("1234567800000000")).verdict,
                         "exact-locally")

    def test_failures_become_receipts(self):
        failing, _ = self.spool.submit(self.job(source=b"fail\n"))
        slow, _ = self.spool.submit(self.job(source=b"sleep 30\n", limits={"timeout_s": 0.5}))
        broken = workers.Recipe("broken-cc", ("/nonexistent/cc", "{source}"), "none")
        missing, _ = self.spool.submit(workers.make_job(broken, self.cache, b"", CODE))
        receipts = {r["job"]: r for r in workers.run_worker(self.spool, self.cache, {**self.recipes, "broken-cc": broken})}
        self.assertEqual(receipts[failing]["status"], "build-failed")
        self.assertIn(b"asked to fail", self.cache.get(receipts[failing]["stderr"]))
        self.assertEqual(receipts[slow]["status"], "timeout")
        self.assertEqual(receipts[missing]["status"], "tool-missing")
        pid = int((self.toolchain / "tool.pid").read_text())
        self.assertTrue(wait_gone(pid), "the timed-out tool survived")
        for job_id in (failing, slow, missing):
            self.assertEqual(workers.verify(self.spool, self.cache, job_id, CODE).verdict, "unverified")


class CoordinatorTest(WorkerCase):
    def built(self):
        job_id, _ = self.spool.submit(self.job())
        workers.run_worker(self.spool, self.cache, self.recipes)
        return job_id

    def test_an_altered_receipt_is_rejected(self):
        job_id = self.built()
        path = self.spool.root / "receipts" / f"{job_id}.json"
        receipt = json.loads(path.read_text())
        receipt["object"] = self.cache.put(b"another object")
        path.write_text(json.dumps(receipt))
        self.assertEqual(workers.verify(self.spool, self.cache, job_id, CODE).verdict, "rejected")

    def test_a_corrupt_or_missing_object_is_never_exact(self):
        job_id = self.built()
        receipt = self.spool.receipt(job_id)
        target = self.cache.path(receipt["object"])
        data = target.read_bytes()
        target.write_bytes(data[:-1] + bytes([data[-1] ^ 0xFF]))   # a real change, whatever the last byte
        self.assertEqual(workers.verify(self.spool, self.cache, job_id, CODE).verdict, "rejected")
        target.unlink()
        self.assertEqual(workers.verify(self.spool, self.cache, job_id, CODE).verdict, "unverified")

    def test_an_altered_job_or_another_target_is_rejected(self):
        job_id = self.built()
        self.assertEqual(workers.verify(self.spool, self.cache, job_id, CODE[:8]).verdict, "rejected")
        path = self.spool.root / "jobs" / f"{job_id}.json"
        job = json.loads(path.read_text())
        job["flags"] = ["-DINVERT"]
        path.write_text(json.dumps(job))
        self.assertEqual(workers.verify(self.spool, self.cache, job_id, CODE).verdict, "rejected")

    def test_relocations_need_the_kits_verifier(self):
        job_id, _ = self.spool.submit(self.job(source=b"reloc\nword 0c000000\n", target=bytes.fromhex("0c000000")))
        workers.run_worker(self.spool, self.cache, self.recipes)
        check = workers.verify(self.spool, self.cache, job_id, bytes.fromhex("0c000000"))
        self.assertEqual(check.verdict, "unverified")
        self.assertIn("relocations", check.reasons[0])


def wait_gone(pid: int, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with open(f"/proc/{pid}/stat") as handle:
                if handle.read().rsplit(")", 1)[1].split()[0] == "Z":
                    return True
        except (FileNotFoundError, ProcessLookupError):
            return True
        time.sleep(0.05)
    return False


class ClaimTest(WorkerCase):
    def test_sigkill_of_a_worker_takes_its_tool_and_leaves_a_stale_claim(self):
        """A worker killed outright mid-build: its tool dies with it (parent-death guard, which the
        worker's entry point turns on), no receipt appears, and the claim is stale (taken over)."""
        job_id, _ = self.spool.submit(self.job(source=source_of(CODE, "sleep 30\n")))
        recipes = self.root / "recipes.json"
        recipes.write_text(json.dumps([{"name": self.recipe.name, "argv": list(self.recipe.argv),
                                        "toolchain_digest": self.recipe.toolchain_digest,
                                        "allowed_flags": sorted(self.recipe.allowed_flags)}]))
        env = {**os.environ, "PYTHONPATH": os.getcwd()}
        env.pop("N64DK_PDEATHSIG_SCOPE", None)                   # the entry point must set it itself
        worker = subprocess.Popen([sys.executable, "-m", "n64dk.workers", str(self.spool.root),
                                   str(self.cache.root), str(recipes), "--once"], env=env,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pidfile, tool = self.toolchain / "tool.pid", None
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and not pidfile.is_file():
                time.sleep(0.02)
            time.sleep(0.05)
            tool = int(pidfile.read_text())
            worker.send_signal(signal.SIGKILL)
            worker.wait(timeout=10)
            self.assertTrue(wait_gone(tool), "the worker's tool outlived it")
            self.assertIsNone(self.spool.receipt(job_id))
            self.assertEqual(self.spool.stale_claims(), [job_id])
            self.assertEqual(workers.verify(self.spool, self.cache, job_id, CODE).verdict, "unverified")
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait()
            if tool is not None and not wait_gone(tool, 0):
                try:                                        # it may die between the check and the kill
                    os.kill(tool, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_a_fresh_claim_is_not_taken_over_by_a_second_taker(self):
        job_id, _ = self.spool.submit(self.job())
        claim = self.spool.root / "claims" / job_id
        claim.mkdir()
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        (claim / "owner.json").write_text(json.dumps({"pid": dead.pid, "start_time": "1", "worker": "gone"}))
        self.assertTrue(self.spool.claim(job_id, "first"))              # takes the dead claim over
        # a second taker that judged the old claim dead earlier must re-judge it before renaming
        self.assertFalse(self.spool._take_over(claim, job_id))
        owner = json.loads((claim / "owner.json").read_text())
        self.assertEqual(owner["worker"], "first")

    def test_a_killed_worker_is_redone_identically(self):
        job_id, _ = self.spool.submit(self.job())
        claim = self.spool.root / "claims" / job_id
        claim.mkdir()
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        (claim / "owner.json").write_text(json.dumps({"pid": dead.pid, "start_time": "1", "worker": "gone"}))
        self.assertEqual(self.spool.stale_claims(), [job_id])
        [receipt] = workers.run_worker(self.spool, self.cache, self.recipes)
        self.assertEqual(receipt["status"], "built")
        other = ContentCache(self.root / "cache-reference")
        reference = workers.Spool(self.root / "reference")
        reference.submit(workers.make_job(self.recipe, other, source_of(CODE), CODE))
        [again] = workers.run_worker(reference, other, self.recipes)
        self.assertEqual(again["object"], receipt["object"])                 # reproduced, not merely accepted

    def test_a_live_claim_is_left_alone(self):
        job_id, _ = self.spool.submit(self.job())
        self.assertTrue(self.spool.claim(job_id, "me"))                      # this (live) process holds it
        self.assertEqual(workers.run_worker(self.spool, self.cache, self.recipes), [])
        self.spool.release(job_id)
        self.assertEqual(len(workers.run_worker(self.spool, self.cache, self.recipes)), 1)


if __name__ == "__main__":
    unittest.main()
