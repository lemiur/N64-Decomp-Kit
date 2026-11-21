"""Local content-addressed storage for reproducible build artifacts.

The cache belongs outside Git.  It never infers trust from a filename: callers
must verify an artifact through the exact oracle before crediting it.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable


class CacheError(RuntimeError):
    pass


class ContentCache:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.objects = self.root / "objects"
        self.refs = self.root / "refs"
        self.locks = self.root / "locks"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.refs.mkdir(parents=True, exist_ok=True)
        self.locks.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def path(self, digest: str) -> Path:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("expected a lowercase SHA-256 digest")
        return self.objects / digest[:2] / digest[2:]

    def put(self, data: bytes) -> str:
        digest = self.digest(data)
        target = self.path(digest)
        if target.is_file() and self.digest(target.read_bytes()) == digest:
            return digest
        # absent, or present but corrupt: write the right bytes and replace atomically, so one
        # damaged file never blocks its key (CHAOS_TESTING.md finding C2)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".incoming-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, target)
        finally:
            try:
                os.unlink(name)
            except FileNotFoundError:
                pass
        return digest

    def get(self, digest: str) -> bytes:
        data = self.path(digest).read_bytes()
        if self.digest(data) != digest:
            raise CacheError(f"corrupt cached object {digest}")
        return data

    def prune(self, max_mb: int | None = None) -> dict:
        """Keep the stored objects under a size (``N64DK_CACHE_MAX_MB``), least recently used
        first.  A reference to a removed object reads as a miss and is rebuilt."""
        from ..safety import prune
        return prune(self.objects, max_mb)

    def contains(self, digest: str) -> bool:
        return self.path(digest).is_file()

    @staticmethod
    def _safe_key(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def get_ref(self, key: str) -> str | None:
        path = self.refs / self._safe_key(key)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text())
        digest = payload.get("digest", "")
        return digest if self.contains(digest) else None

    def _verified_ref(self, key: str) -> str | None:
        try:
            digest = self.get_ref(key)
            if digest is not None:
                self.get(digest)
            return digest
        except (ValueError, OSError, CacheError):
            return None

    def put_ref(self, key: str, digest: str) -> None:
        if not self.contains(digest):
            raise CacheError(f"cannot reference missing object {digest}")
        target = self.refs / self._safe_key(key)
        payload = json.dumps({"key": key, "digest": digest}, sort_keys=True).encode()
        fd, name = tempfile.mkstemp(prefix=".incoming-ref-", dir=self.refs)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, target)
        finally:
            try:
                os.unlink(name)
            except FileNotFoundError:
                pass

    OWNER = "owner.json"
    UNOWNED_GRACE_S = 30.0          # a lock directory whose owner never wrote its record
    BREAKING = ".breaking"          # flock'd by whoever breaks a lock (the kernel drops it on death)

    @staticmethod
    def _alive(pid: int, started) -> bool:
        """Whether the process ``pid`` that recorded ``started`` still runs.  Without /proc (no
        start time on either side) only the pid can be asked: a reused pid then reads as alive,
        which keeps the lock (the waiter times out) instead of breaking a live one."""
        from ..safety import _start_time
        if pid <= 0:
            return False
        if started is not None and _start_time(os.getpid()) is not None:
            return _start_time(pid) == started
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            return True                 # EPERM: it exists, under another user
        return True

    def _stale(self, path: Path) -> bool:
        """Whether a lock's holder is gone: its recorded process (pid + kernel start time) no longer
        exists, or no owner was ever recorded and the directory is older than the grace period.
        An owner record that cannot be read (permissions) is not judged: the lock is kept."""
        try:
            owner = json.loads((path / self.OWNER).read_text())
        except FileNotFoundError:
            try:
                return time.time() - path.stat().st_mtime > self.UNOWNED_GRACE_S
            except OSError:
                return False
        except ValueError:
            return True                 # written by replace, so only a crash of the disk tears it
        except OSError:
            return False
        if not isinstance(owner, dict) or not isinstance(owner.get("pid"), int):
            return True
        return not self._alive(owner["pid"], owner.get("start_time"))

    @contextmanager
    def _breaking(self):
        """Serialise breakers; yields whether this caller may break now.  Two waiters can both
        judge one dead lock stale; the first breaks it and takes a fresh lock under the same name,
        which the second must not rename away.  Each breaker therefore re-judges the lock under
        this flock, right before its rename.  The flock is never waited for (a busy one means
        another breaker is at work: retry after the poll interval), so the caller's deadline
        holds.  Where flock is refused (some network mounts) breaking stays unserialised."""
        import errno
        import fcntl
        try:
            handle = open(self.locks / self.BREAKING, "a+b")
        except OSError:
            handle = None
        if handle is None:
            yield True
            return
        with handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                    yield False
                    return
            yield True

    def _break(self, path: Path) -> bool:
        """Remove a stale lock (renamed first, then emptied and removed) and say whether it was.
        A rename that fails (permissions, a busy directory on a Windows mount) leaves the lock
        for the caller's deadline; a cleanup that fails leaves only a ``.stale-*`` directory."""
        with self._breaking() as allowed:
            if not allowed or not self._stale(path):
                return False
            grave = path.with_name(f".stale-{path.name}-{os.getpid()}-{time.monotonic_ns()}")
            try:
                path.rename(grave)
            except OSError:
                return False
        try:
            for child in grave.iterdir():
                child.unlink(missing_ok=True)
            grave.rmdir()
        except OSError:
            pass
        return True

    @contextmanager
    def lock(self, key: str, timeout: float = 300.0):
        """Exclusive per-key lock: a directory holding its owner's pid and start time.  A lock
        whose holder died (SIGKILL, OOM) is broken instead of failing every later caller
        (CHAOS_TESTING.md finding C2b).  The deadline holds whether or not a break succeeds."""
        from ..safety import _start_time
        path = self.locks / self._safe_key(key)
        deadline = time.monotonic() + timeout
        while True:
            try:
                path.mkdir()
                break
            except FileExistsError:
                if self._stale(path) and self._break(path):
                    continue
                if time.monotonic() >= deadline:
                    raise CacheError(f"timed out waiting for cache key {key}")
                time.sleep(0.05)
        record = path / f".{self.OWNER}.tmp"
        record.write_text(json.dumps({"pid": os.getpid(), "start_time": _start_time(os.getpid())}))
        os.replace(record, path / self.OWNER)
        try:
            yield
        finally:
            try:                        # best effort: a directory that will not go (an open handle
                (path / self.OWNER).unlink(missing_ok=True)     # on a Windows mount) is ownerless
                path.rmdir()                                    # and broken after the grace period
            except OSError:
                pass

    def resolve(self, key: str, producer: Callable[[], bytes]) -> tuple[str, bool]:
        """Return the object's digest and whether ``producer`` actually ran.  A hit counts only
        when its object re-hashes to its digest; a corrupt object or an unreadable ref is a miss,
        rebuilt and replaced (CHAOS_TESTING.md finding C2)."""
        hit = self._verified_ref(key)
        if hit is not None:
            return hit, False
        with self.lock(key):
            hit = self._verified_ref(key)
            if hit is not None:
                return hit, False
            digest = self.put(producer())
            self.put_ref(key, digest)
            return digest, True
