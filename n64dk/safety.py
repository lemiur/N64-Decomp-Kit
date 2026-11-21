"""Keeping the workstation usable while the kit works.

A decompilation sweep starts many short compiler processes from several workers.  On a machine
with little memory the failure is not an error message: memory runs out, the system swaps, the
disk sits at 100% and everything on the machine stops, the kit included.  Planning the number of
workers from free memory at the start (``n64dk.resources``) is not enough on its own: another
program, a second sweep or a step that grows later can still push the machine over.  This module
adds the rest, all enforced rather than advised:

* **one heavy job at a time** — a machine-wide lock (``machine_lock``); a second heavy command
  refuses to start (or waits, when asked) instead of doubling the load;
* **preflight** — a heavy command does not start when available memory, memory pressure or free
  disk space is below a threshold (``preflight``);
* **pause under pressure** — before every compiler launch the kit checks memory pressure and
  waits while the machine is short of memory (``throttle``), so a sweep slows down instead of
  driving the machine into swap;
* **a memory ceiling per child** — every compiler process gets an address-space limit
  (``limit_child``): one that runs away fails alone, as a build failure;
* **leases** — every heavy job registers itself (``JobLease``); ``n64dk jobs`` lists them, tells
  stale ones apart and cancels a job's whole process group.

Everything is configured by environment variables, each named where it is read, and all of it
can be switched off for a machine that does not need it (``N64DK_SAFETY=off``).
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

PRESSURE_FILE = Path("/proc/pressure/memory")
MEMINFO = Path("/proc/meminfo")


def enabled() -> bool:
    return os.environ.get("N64DK_SAFETY", "on").lower() not in ("off", "0", "false", "no")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def state_dir() -> Path:
    root = os.environ.get("N64DK_STATE_DIR") or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "n64dk"
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- measurements

def memory_pressure(path: Path = PRESSURE_FILE) -> float | None:
    """Share of the last ten seconds some task waited for memory (``some avg10``), in percent;
    None where the kernel does not report it."""
    try:
        for line in path.read_text().splitlines():
            if line.startswith("some "):
                return float(dict(item.split("=") for item in line.split()[1:])["avg10"])
    except (OSError, KeyError, ValueError):
        return None
    return None


def available_mb(path: Path = MEMINFO) -> int | None:
    try:
        for line in path.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def free_disk_mb(where: Path) -> int | None:
    where = Path(where)
    while not where.exists() and where != where.parent:
        where = where.parent
    try:
        return shutil.disk_usage(where).free // (1024 * 1024)
    except OSError:
        return None


# --------------------------------------------------------------------------- preflight

def preflight(where: Path, *, min_available_mb: int | None = None, max_pressure: float | None = None,
              min_free_mb: int | None = None, pressure=memory_pressure, available=available_mb,
              free=free_disk_mb) -> dict:
    """Whether a heavy job may start.  ``{"ok": bool, "reasons": [...], measurements...}``."""
    floor = min_available_mb if min_available_mb is not None else _int_env("N64DK_MIN_AVAILABLE_MB", 3072)
    ceiling = max_pressure if max_pressure is not None else float(os.environ.get("N64DK_MAX_MEMORY_PRESSURE", "10"))
    disk = min_free_mb if min_free_mb is not None else _int_env("N64DK_MIN_FREE_DISK_MB", 2048)
    found = {"available_mb": available(), "memory_pressure": pressure(), "free_disk_mb": free(where)}
    reasons = []
    if found["available_mb"] is not None and found["available_mb"] < floor:
        reasons.append(f"only {found['available_mb']} MB of memory available (at least {floor} needed)")
    if found["memory_pressure"] is not None and found["memory_pressure"] > ceiling:
        reasons.append(f"memory pressure {found['memory_pressure']:.1f}% over the last ten seconds (limit {ceiling:g}%)")
    if found["free_disk_mb"] is not None and found["free_disk_mb"] < disk:
        reasons.append(f"only {found['free_disk_mb']} MB free on the output disk (at least {disk} needed)")
    return {"ok": not reasons or not enabled(), "reasons": reasons, **found,
            "thresholds": {"min_available_mb": floor, "max_memory_pressure": ceiling, "min_free_disk_mb": disk}}


# --------------------------------------------------------------------------- pause under pressure

_last_check = [0.0, False]
_paused = {"count": 0, "seconds": 0.0}


def throttle(*, pressure=memory_pressure, available=available_mb, sleep=time.sleep, clock=time.monotonic) -> float:
    """Wait while the machine is short of memory; returns the seconds waited.  Called before
    every compiler launch.  Checks at most twice a second; waits at most
    ``N64DK_PAUSE_LIMIT_S`` (default 600) before letting the launch go ahead anyway."""
    if not enabled():
        return 0.0
    now = clock()
    if now - _last_check[0] < 0.5 and not _last_check[1]:
        return 0.0
    ceiling = float(os.environ.get("N64DK_PAUSE_PRESSURE", "25"))
    floor = _int_env("N64DK_PAUSE_AVAILABLE_MB", 768)
    limit = float(os.environ.get("N64DK_PAUSE_LIMIT_S", "600"))

    def short() -> bool:
        level, free = pressure(), available()
        return (level is not None and level > ceiling) or (free is not None and free < floor)

    waited, started = 0.0, clock()
    _last_check[0] = now
    _last_check[1] = short()
    while _last_check[1] and waited < limit:
        sleep(2.0)
        waited = clock() - started
        _last_check[1] = short()
    if waited:
        _paused["count"] += 1
        _paused["seconds"] += waited
    return waited


def pauses() -> dict:
    return dict(_paused)


# --------------------------------------------------------------------------- child processes

def limit_child(pid: int, legacy: bool = False) -> bool:
    """Give a just-started compiler process an address-space ceiling
    (``N64DK_CHILD_MEMORY_MB``, default 2048; 0 for none).  A process that outgrows it fails on
    its own, as a build failure, instead of taking the machine's memory.  Returns whether the
    limit was set."""
    if not enabled():
        return False
    # an emulator (qemu-irix) reserves the guest's whole 2 GB address space up front: the limit
    # is on address space, not on memory used, so it gets a ceiling it can reserve within
    megabytes = (_int_env("N64DK_CHILD_MEMORY_MB_LEGACY", 8192) if legacy
                 else _int_env("N64DK_CHILD_MEMORY_MB", 2048))
    if megabytes <= 0:
        return False
    try:
        import resource
        limit = megabytes * 1024 * 1024
        resource.prlimit(pid, resource.RLIMIT_AS, (limit, limit))
        return True
    except (ImportError, OSError, ValueError, AttributeError):
        return False


# --------------------------------------------------------------------------- machine-wide worker tokens

CAPACITIES = {"compile": None, "model": 1, "legacy": None, "light": 4}    # None: the machine's worker ceiling


def capacity(kind: str) -> int:
    """Slots of ``kind`` on this machine: ``N64DK_TOKENS_<KIND>``, else the class default
    (``compile``: twice the job ceiling, i.e. two unmeasured tools; ``legacy``: the job ceiling;
    ``model``: 1; ``light``: 4)."""
    value = os.environ.get(f"N64DK_TOKENS_{kind.upper()}")
    if value and value.isdigit() and int(value) > 0:
        return int(value)
    if CAPACITIES.get(kind):
        return CAPACITIES[kind]
    from .resources import plan_jobs
    jobs = max(1, plan_jobs(0).jobs)
    return 2 * jobs if kind == "compile" else jobs


def log_event(event: str, **fields) -> None:
    """Append one line to the machine's event log (waits, refusals, overruns)."""
    try:
        with (state_dir() / "events.jsonl").open("a") as sink:
            sink.write(json.dumps({"time": round(time.time(), 3), "event": event, "pid": os.getpid(), **fields}) + "\n")
    except OSError:
        pass


@contextmanager
def tool_tokens(legacy: bool = False, family: str | None = None):
    """The slots one tool process takes: ``compile`` slots weighted by the tool's measured peak
    (``tool_weight``; unmeasured tools take 2 of the default 4), after available memory covers its
    expected peak, and for an emulated tool a ``legacy`` token as well.  Always taken in that
    order, so two holders cannot deadlock."""
    weight = tool_weight(family) if family else 2
    expected = (tool_profile().get(family) or {}).get("peak_rss_mb", 256.0) if family else 256.0
    admit_memory(expected)
    with token("compile", weight=weight):
        if legacy:
            with token("legacy"):
                yield
        else:
            yield


@contextmanager
def token(kind: str = "compile", deadline: float | None = None, weight: int = 1):
    """Hold ``weight`` of the machine's ``kind`` slots: every n64dk process and thread shares the
    same slot files, so the weights held at once never exceed ``capacity(kind)``.  Slots are
    taken all at once or not at all (no holding while waiting, so no deadlock).  Waits (polling)
    up to ``deadline`` seconds (``N64DK_TOKEN_DEADLINE_S``, default 3600), then raises ``Busy``.
    A slot held by a process that dies is released by the kernel."""
    if not enabled():
        yield None
        return
    slots = capacity(kind)
    weight = max(1, min(weight, slots))
    folder = state_dir() / "tokens" / kind
    folder.mkdir(parents=True, exist_ok=True)
    limit = deadline if deadline is not None else float(os.environ.get("N64DK_TOKEN_DEADLINE_S", "3600"))
    started = time.monotonic()
    held: list = []
    while len(held) < weight:
        for number in range(slots):
            candidate = (folder / f"slot-{number}").open("a+")
            try:
                fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held.append(candidate)
                if len(held) == weight:
                    break
            except BlockingIOError:
                candidate.close()
        if len(held) < weight:
            for handle in held:                  # never hold part of a grant while waiting
                fcntl.flock(handle, fcntl.LOCK_UN)
                handle.close()
            held = []
            if time.monotonic() - started > limit:
                log_event("token-refused", kind=kind, weight=weight, waited=round(time.monotonic() - started, 1))
                raise Busy(f"no {weight} {kind} slot(s) free after {limit:g} s ({slots} on this machine)")
            time.sleep(0.02)
    waited = time.monotonic() - started
    if waited > 1.0:
        log_event("token-wait", kind=kind, weight=weight, waited=round(waited, 2), slots=slots)
    try:
        yield held
    finally:
        for handle in held:
            try:
                fcntl.flock(handle, fcntl.LOCK_UN)
            finally:
                handle.close()


# --------------------------------------------------------------------------- measured tool weights

LIGHT_MB, MEDIUM_MB = 128, 512
_PROFILE: dict = {"loaded": 0.0, "families": {}}


def tool_profile(refresh_after: float = 60.0) -> dict:
    """The machine's measured peak RSS per tool family (``tool_profile.json`` in the state
    directory, written by ``compiler-probes --tool-profile``), re-read at most once a minute."""
    if time.monotonic() - _PROFILE["loaded"] > refresh_after:
        path = state_dir() / "tool_profile.json"
        try:
            _PROFILE["families"] = json.loads(path.read_text()).get("families", {}) if path.is_file() else {}
        except (OSError, ValueError):
            _PROFILE["families"] = {}
        _PROFILE["loaded"] = time.monotonic()
    return _PROFILE["families"]


def save_tool_profile() -> Path:
    """Merge this process's measured peaks into the machine's tool profile (peak kept as the
    maximum ever seen, samples added), under a lock: forked workers merge too.  A process's peaks
    are merged once (cleared afterwards)."""
    path = state_dir() / "tool_profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / "tool_profile.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return _merge_tool_profile(path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _merge_tool_profile(path: Path) -> Path:
    try:
        families = json.loads(path.read_text()).get("families", {}) if path.is_file() else {}
    except (OSError, ValueError):
        families = {}
    with _USAGE_LOCK:
        for family, entry in TOOL_USAGE.items():
            if "peak_rss_mb" not in entry:
                continue
            known = families.setdefault(family, {"peak_rss_mb": 0.0, "samples": 0})
            known["peak_rss_mb"] = round(max(known["peak_rss_mb"], entry["peak_rss_mb"]), 1)
            known["samples"] += entry.get("peak_samples", 0)
            entry.pop("peak_rss_mb", None)
            entry.pop("peak_samples", None)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"families": families}, indent=1, sort_keys=True))
    os.replace(temporary, path)
    _PROFILE["loaded"] = 0.0
    return path


def tool_weight(family: str) -> int:
    """Compile slots one tool of ``family`` takes: from its measured peak when the profile holds at
    least ``N64DK_PROFILE_MIN_SAMPLES`` (5) samples (1 up to 128 MB, 2 up to 512 MB, every slot
    beyond), else 2 (with the default 4 slots: two tools at once, the unmeasured default)."""
    entry = tool_profile().get(family)
    minimum = _int_env("N64DK_PROFILE_MIN_SAMPLES", 5)
    if not entry or entry.get("samples", 0) < minimum:
        return 2
    peak = entry.get("peak_rss_mb", 0.0)
    return 1 if peak <= LIGHT_MB else 2 if peak <= MEDIUM_MB else capacity("compile")


def admit_memory(expected_mb: float, limit: float = 600.0) -> None:
    """Wait (at most ``limit`` s) until available memory covers ``N64DK_TOOL_FLOOR_MB`` (1536) plus
    a tool's expected peak, before it starts."""
    if not enabled():
        return
    floor = _int_env("N64DK_TOOL_FLOOR_MB", 1536)
    started = time.monotonic()
    while available_mb() < floor + expected_mb and time.monotonic() - started < limit:
        time.sleep(0.2)
    waited = time.monotonic() - started
    if waited > 1.0:
        log_event("memory-wait", expected_mb=round(expected_mb, 1), waited=round(waited, 1))


# --------------------------------------------------------------------------- one heavy job at a time

class Busy(RuntimeError):
    """Another heavy job holds the machine-wide lock."""


@contextmanager
def machine_lock(wait: bool = False, path: Path | None = None):
    """Hold the machine-wide heavy-job lock for the duration.  Raises ``Busy`` at once when
    another job holds it, unless ``wait``."""
    if not enabled():
        yield None
        return
    lock = Path(path) if path else state_dir() / "heavy.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    handle = lock.open("a+")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            holder = lock.with_suffix(".holder")
            who = holder.read_text().strip() if holder.is_file() else "another n64dk job"
            raise Busy(f"another heavy job is running ({who}); run one at a time, or pass --wait-for-lock") from None
        lock.with_suffix(".holder").write_text(f"pid {os.getpid()}\n")
        yield lock
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            handle.close()


# --------------------------------------------------------------------------- leases

def _start_time(pid: int) -> str | None:
    """The kernel's start time of ``pid`` (ticks since boot): with the pid, it names one process
    even after the pid is reused."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19]
    except (OSError, IndexError):
        return None


class JobLease:
    """A heavy job's record in ``<state>/jobs``: what runs, since when, its process group, and a
    heartbeat.  Removed when the job ends; one left behind by a job that died is stale."""

    def __init__(self, command: str, argv: list[str] | None = None, heartbeat: float = 30.0):
        self.pid = os.getpid()
        self.path = state_dir() / "jobs" / f"{self.pid}-{_start_time(self.pid) or 0}.json"
        self.record = {"pid": self.pid, "pgid": os.getpgid(self.pid), "start_time": _start_time(self.pid),
                       "command": command, "argv": list(argv or []), "cwd": os.getcwd(),
                       "started": time.time(), "heartbeat": time.time()}
        self.interval = heartbeat
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _write(self) -> None:
        self.record["heartbeat"] = time.time()
        self.record["paused"] = pauses()
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.record))
        os.replace(temporary, self.path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write()

        def beat():
            while not self._stop.wait(self.interval):
                try:
                    self._write()
                except OSError:
                    pass
        self._thread = threading.Thread(target=beat, name="n64dk-lease", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.path.unlink(missing_ok=True)
        return False


def leases(stale_after: float = 300.0) -> list[dict]:
    """Registered jobs, each with ``state``: ``running``, ``stale`` (its process is gone or is
    another process now) or ``silent`` (alive, no heartbeat for ``stale_after`` seconds)."""
    folder = state_dir() / "jobs"
    out = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        alive = _start_time(record["pid"]) == record.get("start_time") and record.get("start_time") is not None
        record["state"] = ("stale" if not alive else
                           "silent" if time.time() - record.get("heartbeat", 0) > stale_after else "running")
        record["lease"] = str(path)
        out.append(record)
    return out


def cancel(pid: int, grace: float = 10.0) -> dict:
    """Stop a registered job and everything it started: SIGTERM to its process group, SIGKILL
    after ``grace`` seconds.  Only a job with a live lease is touched (the start time must match:
    a reused pid is left alone)."""
    record = next((item for item in leases() if item["pid"] == pid), None)
    if record is None:
        return {"pid": pid, "cancelled": False, "reason": "no job registered with this pid"}
    if record["state"] == "stale":
        Path(record["lease"]).unlink(missing_ok=True)
        return {"pid": pid, "cancelled": False, "reason": "the job is gone; its stale lease was removed"}
    group = record.get("pgid") or pid
    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and _start_time(pid) == record.get("start_time"):
        time.sleep(0.2)
    killed = False
    if _start_time(pid) == record.get("start_time"):
        try:
            os.killpg(group, signal.SIGKILL)
            killed = True
        except ProcessLookupError:
            pass
    Path(record["lease"]).unlink(missing_ok=True)
    return {"pid": pid, "cancelled": True, "process_group": group, "forced": killed}


# --------------------------------------------------------------------------- bounded caches

def prune(directory: Path, max_mb: int | None = None, pattern: str = "*") -> dict:
    """Keep a cache directory under ``max_mb`` (``N64DK_CACHE_MAX_MB``, default 4096): files
    are removed least recently used first.  Only files matching ``pattern`` directly or in
    subdirectories of ``directory`` are considered."""
    limit = (max_mb if max_mb is not None else _int_env("N64DK_CACHE_MAX_MB", 4096)) * 1024 * 1024
    directory = Path(directory)
    if not directory.is_dir():
        return {"files": 0, "bytes": 0, "removed": 0}
    files = []
    for path in directory.rglob(pattern):
        try:
            if path.is_file():
                info = path.stat()
                files.append((max(info.st_atime, info.st_mtime), info.st_size, path))
        except OSError:
            continue
    total = sum(size for _t, size, _p in files)
    removed = freed = 0
    for _when, size, path in sorted(files):
        if total - freed <= limit:
            break
        try:
            path.unlink()
            removed += 1
            freed += size
        except OSError:
            pass
    return {"files": len(files), "bytes": total - freed, "removed": removed, "freed_bytes": freed}


# --------------------------------------------------------------------------- one call for a command

_held = threading.local()


# Processes a SIGTERM must take down with the job: tool process groups (compilers run in their
# own sessions, so a signal to the job's group does not reach them) and forked workers.
_TOOL_GROUPS: set[int] = set()
_CHILDREN: set[int] = set()
_REGISTRY_LOCK = threading.Lock()


def register_tool(pid: int) -> None:
    with _REGISTRY_LOCK:
        _TOOL_GROUPS.add(pid)


def unregister_tool(pid: int) -> None:
    with _REGISTRY_LOCK:
        _TOOL_GROUPS.discard(pid)


def register_child(pid: int) -> None:
    """A forked worker of this job: SIGTERM is forwarded to it (it runs the same handler)."""
    with _REGISTRY_LOCK:
        _CHILDREN.add(pid)


def unregister_child(pid: int) -> None:
    """A forked worker that has been reaped: its pid may be reused, so it is no longer signalled."""
    with _REGISTRY_LOCK:
        _CHILDREN.discard(pid)


def forked() -> None:
    """Call first thing in a forked worker: the parent's registrations are not its own, and the
    worker stops (SIGTERM: its tools' groups killed, see ``_terminate``) when its parent dies."""
    if sys.platform.startswith("linux") and pdeathsig_mode() != "off":
        try:
            parent = os.getppid()
            _libc().prctl(1, signal.SIGTERM, 0, 0, 0)        # PR_SET_PDEATHSIG
            if os.getppid() != parent or parent == 1:
                os.kill(os.getpid(), signal.SIGTERM)
        except Exception:                                    # noqa: BLE001 - a guard, never a failure
            pass
    with _REGISTRY_LOCK:
        _TOOL_GROUPS.clear()
        _CHILDREN.clear()
    with _USAGE_LOCK:                       # the parent's counters are the parent's to report
        TOOL_USAGE.clear()
    try:
        from .match import pipeline
        pipeline.reset_counters()
    except Exception:
        pass


def _terminate(owner: int, lease_path: Path | None) -> None:
    """SIGTERM inside a heavy job: stop at once.  Unwinding with an exception is not enough: a
    thread pool's exit waits for every queued task, so the job ran on (measured: a decompile-run
    outlived its timeout).  Kill the tools' process groups, forward the signal to forked workers,
    remove the lease in the owning process (the kernel releases the flock), and exit."""
    with _REGISTRY_LOCK:
        groups, children = list(_TOOL_GROUPS), list(_CHILDREN)
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except OSError:
            pass
    for child in children:
        try:
            os.kill(child, signal.SIGTERM)
        except OSError:
            pass
    if os.getpid() == owner and lease_path is not None:
        try:
            lease_path.unlink(missing_ok=True)
        except OSError:
            pass
    os._exit(143)


# --------------------------------------------------------------------------- per-tool usage

_USAGE_LOCK = threading.Lock()
TOOL_USAGE: dict[str, dict] = {}      # family -> count, wall_s, cpu_s, timeouts (+ peak_rss_mb when measured)


def tool_family(argv) -> str:
    """The tool a command line runs: the program itself, or the program an emulator runs."""
    words = [str(word) for word in argv]
    name = os.path.basename(words[0]) if words else "?"
    if name == "wibo" and len(words) > 1:
        return os.path.basename(words[1])
    if name == "qemu-irix":
        index = 1
        while index < len(words) and words[index].startswith("-"):
            index += 2 if words[index] == "-L" else 1
        return "irix:" + (os.path.basename(words[index]) if index < len(words) else "?")
    return name


def record_usage(family: str, wall: float, usage, timed_out: bool = False) -> None:
    with _USAGE_LOCK:
        entry = TOOL_USAGE.setdefault(family, {"count": 0, "wall_s": 0.0, "cpu_s": 0.0, "timeouts": 0})
        entry["count"] += 1
        entry["wall_s"] += wall
        entry["timeouts"] += timed_out
        if usage is not None:
            # CPU time is the tool's own (and its passes').  Its ru_maxrss is not: a child forked
            # from this process inherits the parent's RSS high-water mark across exec, so peak
            # RSS is only measured through measure_rss() (a small intermediate process).
            entry["cpu_s"] += usage.ru_utime + usage.ru_stime


def record_peak(family: str, peak_mb: float) -> None:
    with _USAGE_LOCK:
        entry = TOOL_USAGE.setdefault(family, {"count": 0, "wall_s": 0.0, "cpu_s": 0.0, "timeouts": 0})
        entry["peak_rss_mb"] = max(entry.get("peak_rss_mb", 0.0), peak_mb)
        entry["peak_samples"] = entry.get("peak_samples", 0) + 1


# --- parent death ---------------------------------------------------------------------------
# Tools run in their own sessions so a job can kill their whole group; nothing kills them when
# the n64dk process itself is killed outright (SIGKILL, the OOM killer) outside a heavy-job scope.
# The parent-death signal closes that gap on Linux: the kernel sends the tool SIGKILL when the
# thread that started it exits.  ``setpriv`` (util-linux) sets it with one extra exec and no code
# in the forked child; ``prctl`` sets it from the forked child (ctypes) before exec.
# Which tools get it (``N64DK_PDEATHSIG_SCOPE``): ``long`` (default) guards what can run long
# or stay large when orphaned: emulated compilers, the draft generator, the permuter and the model
# call; a native compiler pass lives milliseconds, and guarding it costs about 0.7 ms of CPU per
# launch (measured, bench/safety), several percent of a run.  ``all`` guards every tool.  Forked
# Python workers always get SIGTERM on parent death (``forked``), so they stop their own tools.
PDEATHSIG_MODES = ("setpriv", "prctl", "off")
DEFAULT_PDEATHSIG = "setpriv"
SETPRIV = "/usr/bin/setpriv"
LONG_FAMILIES = ("m2c", "permuter", "qemu", "wibo")
_LIBC = None


def _libc():
    global _LIBC
    if _LIBC is None:
        import ctypes
        _LIBC = ctypes.CDLL(None, use_errno=True)
    return _LIBC


def guard_wanted(argv=(), legacy: bool = False, long: bool = False) -> bool:
    """Whether a tool launched with ``argv`` gets the parent-death signal."""
    if os.environ.get("N64DK_PDEATHSIG_SCOPE") == "all":
        return True
    family = tool_family(argv) if argv else ""
    first = os.path.basename(str(argv[0])) if argv else ""
    return legacy or long or any(name in family or name in first for name in LONG_FAMILIES)


def pdeathsig_mode() -> str:
    """``N64DK_PDEATHSIG`` (setpriv, prctl, off); ``off`` where the platform has neither."""
    mode = os.environ.get("N64DK_PDEATHSIG", DEFAULT_PDEATHSIG)
    if mode not in PDEATHSIG_MODES:
        mode = DEFAULT_PDEATHSIG
    if not sys.platform.startswith("linux"):
        return "off"
    if mode == "setpriv" and not os.access(SETPRIV, os.X_OK):
        return "prctl"
    return mode


def child_setup(guard: bool = True):
    """``preexec_fn`` for ``prctl`` mode (else None): SIGKILL on parent death, and exit at once if
    the parent is already gone (it may have died between fork and prctl)."""
    if not guard or pdeathsig_mode() != "prctl":
        return None
    libc, parent = _libc(), os.getpid()

    def setup():
        libc.prctl(1, signal.SIGKILL, 0, 0, 0)            # PR_SET_PDEATHSIG
        if os.getppid() != parent:
            os._exit(127)
    return setup


def death_guarded(argv, guard: bool = True) -> list[str]:
    """The command under ``setpriv --pdeathsig KILL`` in setpriv mode when guarded, else unchanged.
    A program that certainly cannot start (a bare name not on PATH, an absolute path that does not
    exist) is left unwrapped, so its launch fails as it always did instead of as setpriv's exit 127."""
    import shutil
    words = [str(word) for word in argv]
    if not guard or pdeathsig_mode() != "setpriv" or not words:
        return words
    program = words[0]
    if (os.sep not in program and shutil.which(program) is None) or \
            (os.path.isabs(program) and not os.path.exists(program)):
        return words
    return [SETPRIV, "--pdeathsig", "KILL", "--"] + words


def measured(argv, guard: bool = False) -> tuple[list[str], Path | None]:
    """The command as launched: under ``setpriv --pdeathsig KILL`` (see :func:`pdeathsig_mode`),
    and with ``N64DK_MEASURE_RSS=1`` under ``/usr/bin/time`` writing the tool's peak RSS (KB) to a
    file, so its high-water mark is its own and not inherited from this process."""
    words = [str(word) for word in argv]
    if os.environ.get("N64DK_MEASURE_RSS") != "1" or not os.path.exists("/usr/bin/time"):
        return death_guarded(words, guard), None
    import tempfile
    handle, name = tempfile.mkstemp(prefix="n64dk-rss-")
    os.close(handle)
    return death_guarded(["/usr/bin/time", "-f", "%M", "-o", name, *words], guard), Path(name)


def collect_peak(family: str, report: Path | None) -> None:
    if report is None:
        return
    try:
        value = report.read_text().strip().splitlines()[-1]
        record_peak(family, int(value) / 1024)
    except (OSError, ValueError, IndexError):
        pass
    finally:
        report.unlink(missing_ok=True)


def communicate(process, family: str, input=None, timeout: float | None = None):
    """``Popen.communicate`` that reaps the child with ``wait4`` so its peak RSS and CPU time are
    kept (``TOOL_USAGE``).  On timeout the process group is killed and ``TimeoutExpired`` raised."""
    import subprocess
    started = time.monotonic()
    outputs: dict[str, object] = {}

    def drain(name, stream):
        outputs[name] = stream.read()
        stream.close()
    readers = [threading.Thread(target=drain, args=(name, stream), daemon=True)
               for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)) if stream is not None]
    for reader in readers:
        reader.start()
    if process.stdin is not None:
        try:
            if input is not None:
                process.stdin.write(input)
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
    reaped: dict = {}

    def reap():
        try:
            _pid, status, usage = os.wait4(process.pid, 0)
            reaped.update(status=status, usage=usage)
        except ChildProcessError:
            reaped.update(status=None, usage=None)
    waiter = threading.Thread(target=reap, daemon=True)
    waiter.start()
    waiter.join(timeout)
    timed_out = waiter.is_alive()
    if timed_out:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
        waiter.join()
    for reader in readers:
        reader.join(5)
    status = reaped.get("status")
    process.returncode = os.waitstatus_to_exitcode(status) if status is not None else -9
    record_usage(family, time.monotonic() - started, reaped.get("usage"), timed_out)
    if timed_out:
        raise subprocess.TimeoutExpired(process.args, timeout, outputs.get("stdout"), outputs.get("stderr"))
    return outputs.get("stdout"), outputs.get("stderr")


def flush_usage(folder: Path | None = None) -> None:
    """Write this process's tool usage (and the build counters ``extra`` sources register) to
    ``N64DK_USAGE_DIR`` (or ``folder``) so a run can add up what its forked workers did."""
    if os.environ.get("N64DK_MEASURE_RSS") == "1":
        save_tool_profile()                      # this worker's measured peaks
    target = folder or (Path(os.environ["N64DK_USAGE_DIR"]) if os.environ.get("N64DK_USAGE_DIR") else None)
    if target is None:
        return
    target.mkdir(parents=True, exist_ok=True)
    payload = {"pid": os.getpid(), "tools": TOOL_USAGE, "builds": _build_counters()}
    temporary = target / f".{os.getpid()}.tmp"
    temporary.write_text(json.dumps(payload))
    os.replace(temporary, target / f"{os.getpid()}-{time.time_ns()}.json")


def _build_counters() -> dict:
    try:
        from .match import pipeline
        return pipeline.build_counters()
    except Exception:          # the build module is not loaded in every process
        return {}


def run_tool(argv, *, kind: str = "light", timeout: float | None = None, input=None, cwd=None, env=None,
             text: bool = False, capture_output: bool = True, check: bool = False):
    """``subprocess.run`` for every external tool the kit starts: a machine-wide token of
    ``kind`` (``light``: preprocessors, binutils queries, the draft generator; ``compile``;
    ``legacy`` for emulated tools, which also take a compile token), its own session, the per-child
    memory ceiling, registration so a SIGTERM to the job kills it, and on timeout the whole process
    group killed.  Returns a ``subprocess.CompletedProcess``; raises ``subprocess.TimeoutExpired``
    like ``subprocess.run``."""
    import subprocess
    throttle()
    if kind == "light":
        tokens = token("light")
    elif kind == "legacy":
        tokens = tool_tokens(legacy=True, family=tool_family(argv))
    else:
        tokens = tool_tokens(legacy=False, family=tool_family(argv)) if kind == "compile" else token(kind)
    pipe = subprocess.PIPE if capture_output else None
    family = tool_family(argv)
    guard = guard_wanted(argv, legacy=kind in ("legacy", "model"))
    argv, report = measured(argv, guard)
    with tokens:
        process = subprocess.Popen([str(a) for a in argv], cwd=cwd, env=env, text=text, start_new_session=True,
                                   stdin=subprocess.PIPE if input is not None else None, stdout=pipe, stderr=pipe,
                                   preexec_fn=child_setup(guard))
        limit_child(process.pid, kind == "legacy")
        register_tool(process.pid)
        try:
            stdout, stderr = communicate(process, family, input, timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
            raise
        finally:
            unregister_tool(process.pid)
            collect_peak(family, report)
    result = subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
    if check:
        result.check_returncode()
    return result


@contextmanager
def heavy_job(command: str, where: Path, *, wait_for_lock: bool = False, force: bool = False,
              argv: list[str] | None = None):
    """Everything a heavy command needs around it: preflight (unless ``force``), the machine
    lock, and a lease.  Yields the preflight record.  A heavy command run from inside another
    (a batch running each title) shares the outer one's lock and lease."""
    if getattr(_held, "depth", 0):
        _held.depth += 1
        try:
            yield {"ok": True, "reasons": [], "nested": True}
        finally:
            _held.depth -= 1
        return
    check = preflight(where)
    if not check["ok"] and not force:
        raise Busy("not starting: " + "; ".join(check["reasons"]) + " (pass --force-resources to override)")
    previous = None
    owner = os.getpid()
    current: dict = {"lease": None}
    if threading.current_thread() is threading.main_thread():
        # SIGTERM (``n64dk jobs --cancel``, a timeout, a service manager): tools, workers and
        # this process stop at once, and the lease goes with them (see _terminate)
        def stop(_signum, _frame):
            _terminate(owner, current["lease"])
        previous = signal.signal(signal.SIGTERM, stop)
    try:
        with machine_lock(wait=wait_for_lock):
            with JobLease(command, argv) as lease:
                current["lease"] = lease.path
                _held.depth = 1
                try:
                    yield check
                finally:
                    _held.depth = 0
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
