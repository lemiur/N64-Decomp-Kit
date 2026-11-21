"""Content-addressed build jobs and a local worker loop: a prototype (design: DISTRIBUTED_WORKERS.md).

A **job** names everything a build depends on by digest: the source, every header it may include
(relative path -> digest), the recipe (an allowlisted name plus the identity of its command line and
toolchain), the flags (a subset of the recipe's allowed flags), the target bytes it should match and
the limits.  ``job_id`` is the SHA-256 of the canonical job, so equal work has one name and is done
once (duplicate suppression at submit and at receipt).

A **worker** executes only recipes registered on its own machine (the allowlist is local: a job can
name a recipe, never a command line), in a fresh temporary directory, through ``safety.run_tool``
(machine tokens, memory ceiling, own process group, timeout), with a minimal environment.  It
stores the object in the content cache and writes a **receipt** (digests of the object and of the
tool's output, exit status, the receipt's own digest).

The **coordinator** never trusts a receipt: ``verify`` checks the job's content address, the
receipt's digest, the target, re-reads the object from the cache (which recomputes its digest) and
compares it itself.  ``exact-locally`` is not credit: credit comes only from the kit's strict
verifier on the exported project.  ``Verification.credit`` is always False.

Transport: none in code.  The spool is a local directory (``jobs/``, ``claims/``, ``receipts/``).
DISTRIBUTED_WORKERS.md describes a transport (pull over SSH/rsync of the spool and the cache) that
would move these same files unchanged.

    python -m n64dk.workers SPOOL CACHE RECIPES.json [--once]    # one local worker
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from . import safety
from .match.cache import CacheError, ContentCache

JOB_SCHEMA = "n64dk.worker.job/1"
RECEIPT_SCHEMA = "n64dk.worker.receipt/1"
DEFAULT_LIMITS = {"timeout_s": 120, "max_output_bytes": 4 << 20}
CLAIM_GRACE_S = 30.0          # a claim without an owner record older than this is stale
OUTPUT_KEEP = 64 << 10        # bytes of stdout/stderr kept in the cache per run


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value) -> str:
    return sha256(canonical(value).encode())


class JobError(ValueError):
    """A job or receipt that cannot be trusted as written."""


@dataclass(frozen=True)
class Recipe:
    """One allowlisted way to build.  ``argv`` words may contain ``{source}``, ``{output}`` and
    ``{include}`` (paths inside the job's temporary directory); a word that is exactly ``{flags}``
    expands to the job's flags, each of which must be in ``allowed_flags``.  ``toolchain_digest``
    names the binaries ``argv`` runs (for example ``pipeline.toolchain_digest``)."""
    name: str
    argv: tuple
    toolchain_digest: str
    allowed_flags: frozenset = frozenset()
    output: str = "out.o"
    source_name: str = "source.c"

    def identity(self) -> str:
        return _digest({"name": self.name, "argv": list(self.argv), "toolchain": self.toolchain_digest,
                        "allowed_flags": sorted(self.allowed_flags), "output": self.output,
                        "source_name": self.source_name})

    @classmethod
    def from_dict(cls, value: dict) -> "Recipe":
        return cls(value["name"], tuple(value["argv"]), value["toolchain_digest"],
                   frozenset(value.get("allowed_flags", ())), value.get("output", "out.o"),
                   value.get("source_name", "source.c"))


@dataclass(frozen=True)
class Job:
    recipe: str
    recipe_identity: str
    source: str
    headers: tuple = ()           # ((relative path, digest), ...) sorted
    flags: tuple = ()
    target: str = ""
    target_size: int = 0
    limits: tuple = ()            # sorted (name, value) pairs

    def as_dict(self) -> dict:
        return {"schema": JOB_SCHEMA, "recipe": self.recipe, "recipe_identity": self.recipe_identity,
                "source": self.source, "headers": [list(pair) for pair in self.headers], "flags": list(self.flags),
                "target": self.target, "target_size": self.target_size, "limits": dict(self.limits)}

    @property
    def job_id(self) -> str:
        return _digest(self.as_dict())

    def limit(self, name: str):
        return dict(self.limits).get(name, DEFAULT_LIMITS[name])

    @classmethod
    def from_dict(cls, value: dict) -> "Job":
        if value.get("schema") != JOB_SCHEMA:
            raise JobError(f"unknown job schema {value.get('schema')!r}")
        return cls(value["recipe"], value["recipe_identity"], value["source"],
                   tuple(tuple(pair) for pair in value.get("headers", ())), tuple(value.get("flags", ())),
                   value.get("target", ""), int(value.get("target_size", 0)),
                   tuple(sorted(dict(value.get("limits", {})).items())))


def safe_header_path(name: str) -> bool:
    """A header path a job may ask for: relative, POSIX, no ``..``, no empty or dot parts."""
    if not name or "\\" in name or "\0" in name or name.startswith("/"):
        return False
    parts = PurePosixPath(name).parts
    return bool(parts) and all(part not in ("", ".", "..") for part in parts)


def make_job(recipe: Recipe, cache: ContentCache, source: bytes, target: bytes, *, headers: dict | None = None,
             flags=(), limits: dict | None = None) -> Job:
    """Store the inputs in ``cache`` and describe the job by their digests."""
    pairs = []
    for name, data in sorted((headers or {}).items()):
        if not safe_header_path(name):
            raise JobError(f"unsafe header path {name!r}")
        pairs.append((name, cache.put(data)))
    unknown = set(limits or {}) - set(DEFAULT_LIMITS)
    if unknown:
        raise JobError(f"unknown limits {sorted(unknown)}")
    return Job(recipe.name, recipe.identity(), cache.put(source), tuple(pairs), tuple(flags),
               sha256(target), len(target), tuple(sorted((limits or {}).items())))


# ---- the spool: jobs, claims, receipts ------------------------------------------------------
def _write_once(path: Path, text: str) -> bool:
    """Create ``path`` with ``text`` atomically unless it exists (then keep the first): True when
    this call created it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".incoming-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(name, path)                    # fails when the file exists: first writer wins
            return True
        except FileExistsError:
            return False
    finally:
        os.unlink(name)


class Spool:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        for sub in ("jobs", "claims", "receipts"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def submit(self, job: Job) -> tuple[str, bool]:
        """(job id, whether new work was queued).  An equal job already queued or done is not
        queued again."""
        job_id = job.job_id
        if (self.root / "receipts" / f"{job_id}.json").is_file():
            return job_id, False
        return job_id, _write_once(self.root / "jobs" / f"{job_id}.json", canonical(job.as_dict()))

    def job(self, job_id: str) -> Job:
        """The job stored under ``job_id``; JobError when its content no longer matches the name."""
        path = self.root / "jobs" / f"{job_id}.json"
        try:
            job = Job.from_dict(json.loads(path.read_text()))
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise JobError(f"job {job_id[:12]} unreadable: {error}") from None
        if job.job_id != job_id:
            raise JobError(f"job {job_id[:12]} was altered: its content hashes to {job.job_id[:12]}")
        return job

    def pending(self) -> list[str]:
        done = {p.stem for p in (self.root / "receipts").glob("*.json")}
        return sorted(p.stem for p in (self.root / "jobs").glob("*.json") if p.stem not in done)

    def receipt(self, job_id: str) -> dict | None:
        path = self.root / "receipts" / f"{job_id}.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except ValueError:
            return {"schema": RECEIPT_SCHEMA, "job": job_id, "status": "unreadable"}

    def write_receipt(self, receipt: dict) -> bool:
        return _write_once(self.root / "receipts" / f"{receipt['job']}.json", canonical(receipt))

    # -- claims: one worker per job; a claim whose owner died is taken over
    def _owner_alive(self, folder: Path) -> bool:
        try:
            owner = json.loads((folder / "owner.json").read_text())
        except (OSError, ValueError):
            try:
                return time.time() - folder.stat().st_mtime < CLAIM_GRACE_S   # being written right now
            except OSError:
                return False
        return owner.get("start_time") is not None and safety._start_time(owner.get("pid", -1)) == owner["start_time"]

    def _take_over(self, folder: Path, job_id: str) -> bool:
        """Remove a dead owner's claim.  Two takers can both judge one claim dead; the first takes
        it over and writes a fresh claim under the same name, which the second must not rename
        away.  So the claim is judged again under a non-blocking flock that serialises takers
        (a busy flock: another taker is at work, this one gives up)."""
        import errno
        import fcntl
        try:
            handle = open(self.root / "claims" / ".breaking", "a+b")
        except OSError:
            return False
        with handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                    return False
            if self._owner_alive(folder):
                return False
            grave = folder.with_name(f".stale-{job_id}-{time.time_ns()}")
            try:
                folder.rename(grave)
            except OSError:
                return False
        shutil.rmtree(grave, ignore_errors=True)
        return True

    def claim(self, job_id: str, worker: str) -> bool:
        folder = self.root / "claims" / job_id
        for _attempt in range(2):
            try:
                folder.mkdir()
            except FileExistsError:
                if self._owner_alive(folder) or not self._take_over(folder, job_id):
                    return False
                continue
            pid = os.getpid()
            record = {"pid": pid, "start_time": safety._start_time(pid), "worker": worker, "claimed": time.time()}
            temporary = folder / ".owner.tmp"
            temporary.write_text(json.dumps(record))
            os.replace(temporary, folder / "owner.json")
            return True
        return False

    def release(self, job_id: str) -> None:
        shutil.rmtree(self.root / "claims" / job_id, ignore_errors=True)

    def stale_claims(self) -> list[str]:
        return sorted(p.name for p in (self.root / "claims").iterdir()
                      if p.is_dir() and not p.name.startswith(".") and not self._owner_alive(p))


# ---- the worker -----------------------------------------------------------------------------
def _receipt(job_id: str, worker: str, job: Job | None, status: str, *, reason: str = "", **fields) -> dict:
    body = {"schema": RECEIPT_SCHEMA, "job": job_id, "worker": worker, "status": status, "reason": reason,
            "recipe": job.recipe if job else None, "recipe_identity": job.recipe_identity if job else None,
            "object": None, "object_size": None, "stdout": None, "stderr": None, "returncode": None,
            "seconds": None, **fields}
    body["receipt"] = _digest(body)
    return body


def _expand(recipe: Recipe, job: Job, paths: dict) -> list[str]:
    out = []
    for word in recipe.argv:
        if word == "{flags}":
            out.extend(job.flags)
            continue
        for key, value in paths.items():
            word = word.replace("{" + key + "}", str(value))
        out.append(word)
    return out


def execute(job: Job, job_id: str, recipes: dict, cache: ContentCache, worker: str, runner=None) -> dict:
    """Build one job and return its receipt (never raises for a bad job: it is a receipt)."""
    runner = runner or safety.run_tool
    recipe = recipes.get(job.recipe)
    if recipe is None:
        return _receipt(job_id, worker, job, "rejected", reason=f"recipe {job.recipe!r} is not allowlisted here")
    if recipe.identity() != job.recipe_identity:
        return _receipt(job_id, worker, job, "rejected",
                        reason="recipe identity differs (other command line or toolchain on this worker)")
    extra = [flag for flag in job.flags if flag not in recipe.allowed_flags]
    if extra:
        return _receipt(job_id, worker, job, "rejected", reason=f"flags not allowed by the recipe: {extra}")
    bad = [name for name, _digest_ in job.headers if not safe_header_path(name)]
    if bad:
        return _receipt(job_id, worker, job, "rejected", reason=f"unsafe header paths {bad}")
    try:
        source = cache.get(job.source)
        headers = [(name, cache.get(digest)) for name, digest in job.headers]
    except (OSError, ValueError, CacheError) as error:
        return _receipt(job_id, worker, job, "rejected", reason=f"missing or corrupt input: {error}")
    with tempfile.TemporaryDirectory(prefix="n64dk-worker-") as folder:
        root = Path(folder)
        include = root / "include"
        include.mkdir()
        (root / recipe.source_name).write_bytes(source)
        for name, data in headers:
            path = include / PurePosixPath(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        output = root / recipe.output
        argv = _expand(recipe, job, {"source": root / recipe.source_name, "output": output, "include": include})
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C", "LC_ALL": "C",
               "HOME": folder, "TMPDIR": folder}
        started = time.monotonic()
        try:
            result = runner(argv, kind="compile", timeout=job.limit("timeout_s"), cwd=folder, env=env)
        except subprocess.TimeoutExpired:
            return _receipt(job_id, worker, job, "timeout", reason=f"over {job.limit('timeout_s')} s",
                            seconds=round(time.monotonic() - started, 3))
        except OSError as error:
            return _receipt(job_id, worker, job, "tool-missing", reason=str(error))
        seconds = round(time.monotonic() - started, 3)
        logs = {name: cache.put(bytes(data or b"")[-OUTPUT_KEEP:]) if isinstance(data, (bytes, bytearray))
                else cache.put(str(data or "").encode()[-OUTPUT_KEEP:])
                for name, data in (("stdout", result.stdout), ("stderr", result.stderr))}
        if result.returncode != 0:
            return _receipt(job_id, worker, job, "build-failed", reason=f"exit {result.returncode}",
                            returncode=result.returncode, seconds=seconds, **logs)
        if not output.is_file():
            return _receipt(job_id, worker, job, "build-failed", reason="the recipe wrote no output",
                            returncode=0, seconds=seconds, **logs)
        size = output.stat().st_size
        if size > job.limit("max_output_bytes"):
            return _receipt(job_id, worker, job, "rejected", reason=f"output of {size} bytes over the limit",
                            returncode=0, seconds=seconds, **logs)
        digest = cache.put(output.read_bytes())
        return _receipt(job_id, worker, job, "built", object=digest, object_size=size, returncode=0,
                        seconds=seconds, **logs)


def run_worker(spool: Spool, cache: ContentCache, recipes: dict, *, worker: str | None = None,
               limit: int | None = None, runner=None) -> list[dict]:
    """Do the pending jobs once (at most ``limit``); returns the receipts this call wrote.  A job
    another live worker holds is left alone; one whose worker died is taken over."""
    worker = worker or f"local-{os.getpid()}"
    written = []
    for job_id in spool.pending():
        if limit is not None and len(written) >= limit:
            break
        if spool.receipt(job_id) is not None or not spool.claim(job_id, worker):
            continue
        try:
            try:
                job = spool.job(job_id)
            except JobError as error:
                receipt = _receipt(job_id, worker, None, "rejected", reason=str(error))
            else:
                receipt = execute(job, job_id, recipes, cache, worker, runner)
            if spool.write_receipt(receipt):
                written.append(receipt)
        finally:
            spool.release(job_id)
    return written


# ---- the coordinator ----------------------------------------------------------------------
@dataclass
class Verification:
    job: str
    verdict: str                      # exact-locally | different | unverified | rejected
    reasons: list = field(default_factory=list)
    object: str | None = None

    @property
    def credit(self) -> bool:
        return False                  # credit comes only from the kit's strict verifier

    def as_dict(self) -> dict:
        return {"job": self.job, "verdict": self.verdict, "reasons": list(self.reasons), "object": self.object,
                "credit": False, "proof": False}


def text_compare(object_bytes: bytes, target: bytes) -> tuple[str, list]:
    """The object's ``.text`` against the target, byte for byte.  An object whose ``.text`` has
    relocations is ``unverified`` here: resolving them needs the project's symbols (the kit's
    ``compare.compare_object``), which a coordinator passes as ``compare``."""
    from .elf import Elf
    try:
        elf = Elf(object_bytes)
    except Exception as error:                                # noqa: BLE001 - any parse failure
        return "unverified", [f"not a readable ELF object: {error}"]
    text = elf.section(".text")
    if text is None:
        return "unverified", ["the object has no .text"]
    relocations = elf.relocations(text)
    if relocations:
        return "unverified", [f"{len(relocations)} relocations in .text: strict comparison needs the kit's verifier"]
    if text.data == target:
        return "exact-locally", []
    first = next((i for i, (a, b) in enumerate(zip(text.data, target)) if a != b), min(len(text.data), len(target)))
    return "different", [f"{len(text.data)} vs {len(target)} bytes, first difference at 0x{first:X}"]


def verify(spool: Spool, cache: ContentCache, job_id: str, target: bytes, *, compare=text_compare) -> Verification:
    """Check a receipt without trusting it: the job's content address, the receipt's digest and
    fields, the target, the object (re-read and re-hashed) and a local comparison."""
    try:
        job = spool.job(job_id)
    except JobError as error:
        return Verification(job_id, "rejected", [str(error)])
    receipt = spool.receipt(job_id)
    if receipt is None:
        return Verification(job_id, "unverified", ["no receipt yet"])
    body = {key: value for key, value in receipt.items() if key != "receipt"}
    if receipt.get("receipt") != _digest(body):
        return Verification(job_id, "rejected", ["the receipt was altered (its digest does not match)"])
    if receipt.get("job") != job_id or receipt.get("recipe") != job.recipe or \
            receipt.get("recipe_identity") != job.recipe_identity:
        return Verification(job_id, "rejected", ["the receipt is for another job or recipe"])
    if sha256(target) != job.target:
        return Verification(job_id, "rejected", ["the target given is not the job's target"])
    if receipt.get("status") != "built":
        return Verification(job_id, "unverified", [f"worker status {receipt.get('status')}: {receipt.get('reason')}"])
    digest = receipt.get("object") or ""
    try:
        data = cache.get(digest)                  # recomputes the digest
    except (FileNotFoundError, ValueError):
        return Verification(job_id, "unverified", ["the object is not in this cache"], digest)
    except CacheError:
        return Verification(job_id, "rejected", ["the cached object is corrupt"], digest)
    if len(data) != receipt.get("object_size"):
        return Verification(job_id, "rejected", ["object size differs from the receipt"], digest)
    verdict, reasons = compare(data, target)
    return Verification(job_id, verdict, list(reasons), digest)


# ---- command line: one local worker ----------------------------------------------------------
def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="python -m n64dk.workers", description="one local build worker")
    parser.add_argument("spool")
    parser.add_argument("cache")
    parser.add_argument("recipes", help="JSON list of recipes (name, argv, toolchain_digest, allowed_flags)")
    parser.add_argument("--once", action="store_true", help="do the pending jobs, then exit")
    parser.add_argument("--poll", type=float, default=2.0)
    args = parser.parse_args(argv)
    # every tool this worker starts dies with it, even under SIGKILL (safety's parent-death guard)
    os.environ["N64DK_PDEATHSIG_SCOPE"] = "all"
    recipes = {r.name: r for r in map(Recipe.from_dict, json.loads(Path(args.recipes).read_text()))}
    spool, cache = Spool(args.spool), ContentCache(args.cache)
    while True:
        for receipt in run_worker(spool, cache, recipes):
            print(json.dumps({"job": receipt["job"][:16], "status": receipt["status"]}), flush=True)
        if args.once:
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
