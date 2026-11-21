"""Host doctor: can this machine run the kit, and what is missing, by kind.

``n64dk doctor --host`` (the ``--host`` mode of the existing ``doctor`` command, which otherwise
checks a project) needs no project and changes nothing.  Every check has one row with stable keys
``category``, ``name``, ``status``, ``ok``, ``required``, ``detail``, ``remediation``:

=====================  ==========================================================  =========
category               what                                                        required
=====================  ==========================================================  =========
required_dependency    Python >= 3.11, rabbitizer >= 1.16 (pyproject.toml), the    yes
                       stdlib modules some Python builds omit (sqlite3, zlib)
optional_capability    m2c, pycparser, PyYAML (not needed), z3, wibo, splat,       no
                       spimdisasm, git, host gcc, the built-in emulator
compilers              ``match.compilers.status`` of the toolchain root            no
host                   operating system, fcntl, WSL (``/proc/version``), CPU       Windows
                                                                                   only
disk                   free space of the state directory and ``--out`` against     yes
                       ``N64DK_MIN_FREE_DISK_MB`` (2048, as ``safety.preflight``)
memory                 MemTotal against ``N64DK_MIN_AVAILABLE_MB`` (3072, as       total:
                       ``safety.preflight``): a machine below it can never start   yes;
                       a heavy job; MemAvailable below it is reported, not failed  now: no
output_path            ``--out`` inside the kit repository, inside a source        yes
                       project (outside its ``.n64dk``), beside ROM images, or
                       not writable (Windows drives under WSL: a warning)
schema                 stored formats against this kit's constants: a run's        no
                       ``donor_index.json`` (``transplant.INDEX_VERSION``) and
                       fixture snapshots (``fixtures.SNAPSHOT_FORMAT``)
=====================  ==========================================================  =========

The exit status is 1 exactly when a required check is not ``ok``.  Statuses: ``ok``, ``missing``,
``insufficient``, ``unsafe``, ``stale``, ``warn``, ``unknown``, ``fail``.  No stored format of this
kit version needs a manual migration: the two versioned stores are rebuilt automatically when
stale, so ``stale`` is informational.

All probing goes through ``Probes`` so tests replace it; nothing here runs a compiler.  The
separate ``synthetic_match`` (``python -m n64dk.commands.doctor smoke-match``, used by
``tools/smoke_test.sh``) does: it compiles a one-line function with the first installed KMC GCC
2.7.2 / IDO 5.3 / IDO 7.1 through the kit's own build pipeline and compares the object's code with
the two instructions every one of them emits for it, or reports SKIP when none is installed.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import struct
import sys
from pathlib import Path

DOCTOR_SCHEMA = 1
REQUIRED_PYTHON = (3, 11)
REQUIRED_PACKAGES = {"rabbitizer": (1, 16)}
REQUIRED_STDLIB = {"sqlite3": "result, session and verification stores",
                   "zlib": "IPL3/CIC identification and compressed streams"}
OPTIONAL_MODULES = {
    "m2c": "first C drafts (m2c_bridge); install the [draft] extra",
    "pycparser": "permuter-style source search; install the [search] extra",
    "yaml": "not needed: splat configurations are read by n64dk.interop.splat's own YAML subset reader",
    "z3": "symbolic tier of the behaviour oracle (blocked without it; other tiers work)",
}
OPTIONAL_TOOLS = {
    "wibo": "runs the Windows executables of the SN compiler lanes (or [toolchains] wibo)",
    "gcc": "host C preprocessor (gcc -E) used by the SN lanes",
    "git": "pinned ground-truth snapshots, workspace audit and the project doctor",
    "splat": "boundary corroboration against splat (optional research)",
    "spimdisasm": "advanced disassembly metadata (optional research)",
}
ROM_SUFFIXES = (".z64", ".n64", ".v64", ".ndd", ".d64", ".rom")
STATUSES = ("ok", "missing", "insufficient", "unsafe", "stale", "warn", "unknown", "fail")
KIT_ROOT = Path(__file__).resolve().parents[2]


def _version_tuple(text: str | None) -> tuple[int, ...]:
    out = []
    for part in re.split(r"[.+-]", text or ""):
        if not part.isdigit():
            break
        out.append(int(part))
    return tuple(out)


class Probes:
    """Everything the doctor asks the machine.  Tests subclass it."""

    def python_version(self) -> tuple[int, int, int]:
        return tuple(sys.version_info[:3])

    def has_module(self, name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            return False

    def module_version(self, name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    def which(self, name: str) -> str | None:
        return shutil.which(name)

    def platform(self) -> dict:
        return {"system": platform.system(), "machine": platform.machine(), "release": platform.release()}

    def read_text(self, path: str) -> str | None:
        try:
            return Path(path).read_text(errors="replace")
        except OSError:
            return None

    def env(self, name: str) -> str | None:
        return os.environ.get(name)

    def free_mb(self, path: Path) -> int | None:
        from .. import safety
        return safety.free_disk_mb(path)

    def memory(self):
        from ..resources import memory_snapshot
        return memory_snapshot()

    def compilers(self, root: Path | None) -> dict:
        from ..match import compilers
        return compilers.status(root)

    def listdir(self, path: Path) -> list[str]:
        try:
            return sorted(os.listdir(path))
        except OSError:
            return []

    def is_dir(self, path: Path) -> bool:
        return Path(path).is_dir()

    def is_file(self, path: Path) -> bool:
        return Path(path).is_file()

    def writable(self, path: Path) -> bool:
        return os.access(path, os.W_OK)

    def head(self, path: Path, size: int = 256) -> bytes | None:
        try:
            with open(path, "rb") as stream:
                return stream.read(size)
        except OSError:
            return None

    def glob(self, path: Path, pattern: str) -> list[Path]:
        return sorted(Path(path).glob(pattern)) if Path(path).is_dir() else []

    def home(self) -> Path:
        return Path.home()


def _row(category, name, status, required, detail=None, remediation=None) -> dict:
    assert status in STATUSES, status
    return {"category": category, "name": name, "status": status, "ok": status == "ok",
            "required": bool(required), "detail": detail, "remediation": remediation}


def _int_env(probes: Probes, name: str, default: int) -> int:
    try:
        return int(probes.env(name) or default)
    except ValueError:
        return default


# --------------------------------------------------------------------------- checks by category

def dependency_checks(p: Probes) -> list[dict]:
    rows = []
    version = p.python_version()
    rows.append(_row("required_dependency", "python", "ok" if version >= REQUIRED_PYTHON else "missing", True,
                     ".".join(map(str, version)) + f" (needs >= {'.'.join(map(str, REQUIRED_PYTHON))})",
                     None if version >= REQUIRED_PYTHON else
                     "use Python 3.11 or newer for the kit's virtual environment"))
    for name, minimum in REQUIRED_PACKAGES.items():
        found = p.has_module(name)
        have = p.module_version(name) if found else None
        recent = found and (have is None or _version_tuple(have) >= minimum)
        rows.append(_row("required_dependency", name, "ok" if recent else "missing", True,
                         (f"{name} {have}" if have else (f"{name} (version unknown)" if found else "not importable")),
                         None if recent else f"pip install '{name}>={'.'.join(map(str, minimum))}' in the kit's environment"))
    for name, use in REQUIRED_STDLIB.items():
        found = p.has_module(name)
        rows.append(_row("required_dependency", name, "ok" if found else "missing", True, use,
                         None if found else f"use a Python build that includes the {name} module"))
    return rows


def optional_checks(p: Probes) -> list[dict]:
    rows = []
    for name, use in OPTIONAL_MODULES.items():
        found = p.has_module(name)
        rows.append(_row("optional_capability", name, "ok" if found else "missing", False, use,
                         None if found or name == "yaml" else f"pip install {name} (optional)"))
    for name, use in OPTIONAL_TOOLS.items():
        path = p.which(name)
        rows.append(_row("optional_capability", name, "ok" if path else "missing", False,
                         f"{use}; found at {path}" if path else use,
                         None if path else f"install {name} on PATH (optional)"))
    rows.append(_row("optional_capability", "emulator", "ok", False,
                     "built-in interpreter (n64dk.emu, pure Python) for boot emulation and the behaviour oracle; "
                     "no external emulator backend is integrated"))
    return rows


def compiler_checks(p: Probes, root: Path | None) -> list[dict]:
    if root is None:
        return [_row("compilers", "toolchain root", "missing", False,
                     "no toolchain root given (--toolchain-dir or N64DK_TOOLCHAIN): no compiler can be used",
                     "n64dk compilers --toolchain-dir DIR --fetch  (downloads the public decomp.me compiler archive)")]
    try:
        status = p.compilers(root)
    except Exception as error:                           # a broken root is a finding, not a crash
        return [_row("compilers", "toolchain root", "unknown", False, f"{type(error).__name__}: {error}")]
    usable = [row["id"] for row in status.get("compilers", []) if row.get("usable")]
    rows = [_row("compilers", "usable compilers", "ok" if usable else "missing", False,
                 {"root": status.get("root"), "known": status.get("known"), "installed": status.get("installed"),
                  "usable": usable, "installed_without_lane": status.get("installed_without_lane", {})},
                 None if usable else "n64dk compilers --toolchain-dir DIR --fetch")]
    return rows


def host_checks(p: Probes) -> tuple[list[dict], dict]:
    info = p.platform()
    system, machine = info.get("system", ""), (info.get("machine") or "").lower()
    version_text = p.read_text("/proc/version") or ""
    lowered = version_text.lower()
    wsl = "microsoft" in lowered
    wsl_version = (2 if ("microsoft-standard" in lowered or "wsl2" in lowered) else 1) if wsl else None
    facts = {**info, "wsl": wsl, "wsl_version": wsl_version}
    rows = []
    if system == "Linux":
        rows.append(_row("host", "operating system", "ok", True, f"Linux{' (WSL' + str(wsl_version) + ')' if wsl else ''}"))
    elif system == "Darwin":
        rows.append(_row("host", "operating system", "warn", False,
                         "macOS: untested; /proc probes report unknown and the compiler archives are Linux binaries"))
    elif system == "Windows":
        rows.append(_row("host", "operating system", "fail", True,
                         "native Windows: the kit needs POSIX process control (fcntl, process groups)",
                         "run the kit inside WSL 2 or a Linux machine"))
    else:
        rows.append(_row("host", "operating system", "warn", False, f"{system or 'unknown'}: untested"))
    has_fcntl = p.has_module("fcntl")
    rows.append(_row("host", "fcntl", "ok" if has_fcntl else "fail", True,
                     "machine lock and leases (n64dk.safety)", None if has_fcntl else "use a POSIX host"))
    if wsl:
        rows.append(_row("host", "wsl", "ok" if wsl_version == 2 else "warn", False,
                         f"WSL {wsl_version} (from /proc/version)" + (
                             "" if wsl_version == 2 else
                             "; WSL 1 does not run 32-bit Linux executables, which some period compilers are (assumed)"),
                         None if wsl_version == 2 else "convert the distribution to WSL 2 (wsl --set-version)"))
    if machine in ("x86_64", "amd64"):
        rows.append(_row("host", "cpu architecture", "ok", False, machine))
    else:
        rows.append(_row("host", "cpu architecture", "warn", False,
                         f"{machine or 'unknown'}: the compiler archives the kit fetches are x86-64 Linux binaries "
                         "(they need an x86-64 host or user-mode emulation; assumed)"))
    return rows, facts


def state_directory(p: Probes) -> Path:
    """``safety.state_dir()``'s location, computed without creating it."""
    root = p.env("N64DK_STATE_DIR") or Path(p.env("XDG_CACHE_HOME") or p.home() / ".cache") / "n64dk"
    return Path(root)


def disk_checks(p: Probes, out: Path | None) -> list[dict]:
    floor = _int_env(p, "N64DK_MIN_FREE_DISK_MB", 2048)
    rows = []
    places = [("state directory", state_directory(p))] + ([("output directory", out)] if out else [])
    for label, where in places:
        free = p.free_mb(where)
        if free is None:
            rows.append(_row("disk", label, "unknown", False, f"{where}: free space not measurable"))
        else:
            ok = free >= floor
            rows.append(_row("disk", label, "ok" if ok else "insufficient", True,
                             f"{where}: {free} MB free (heavy jobs need {floor} MB: N64DK_MIN_FREE_DISK_MB)",
                             None if ok else "free disk space or point N64DK_STATE_DIR / --out at a larger disk"))
    return rows


def memory_checks(p: Probes) -> list[dict]:
    floor = _int_env(p, "N64DK_MIN_AVAILABLE_MB", 3072)
    snap = p.memory()
    total, available = getattr(snap, "total_mb", None), getattr(snap, "available_mb", None)
    if total is None:
        return [_row("memory", "total memory", "unknown", False, "MemTotal not readable (/proc/meminfo)")]
    rows = [_row("memory", "total memory", "ok" if total >= floor else "insufficient", True,
                 f"{total} MB in all; heavy jobs start only with {floor} MB available (N64DK_MIN_AVAILABLE_MB)",
                 None if total >= floor else
                 "this machine can never meet the floor: lower N64DK_MIN_AVAILABLE_MB knowingly, and use --workers 1")]
    if available is not None:
        swap = getattr(snap, "swap_used_mb", None)
        try:
            from ..resources import plan_jobs
            jobs = plan_jobs(0, snapshot=snap).jobs
        except Exception:
            jobs = None
        rows.append(_row("memory", "available memory", "ok" if available >= floor else "insufficient", False,
                         f"{available} MB available now" + (f", {swap} MB swap in use" if swap else "") +
                         (f"; the resource plan allows {jobs} compiler job(s)" if jobs else ""),
                         None if available >= floor else "heavy jobs will wait or refuse until memory frees"))
    return rows


def _nearest_existing(p: Probes, path: Path) -> Path:
    while not p.is_dir(path) and path != path.parent:
        path = path.parent
    return path


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def output_checks(p: Probes, out: Path | None, kit_root: Path = KIT_ROOT) -> list[dict]:
    if out is None:
        return []
    out = Path(os.path.abspath(Path(out).expanduser()))
    problems = []
    if p.is_file(kit_root / "pyproject.toml") and p.is_dir(kit_root / "n64dk") and _within(out, kit_root):
        problems.append(f"inside the kit repository {kit_root}: generated files become repository litter")
    home = p.home()
    for ancestor in [out, *out.parents]:
        if ancestor in (Path(ancestor.anchor), home):
            continue
        names = set(p.listdir(ancestor)) if p.is_dir(ancestor) else set()
        if "n64dk.toml" in names and not _within(out, ancestor / ".n64dk"):
            problems.append(f"inside the source project {ancestor} (outside its .n64dk state directory)")
            break
        if ".git" in names and ("Makefile" in names or "src" in names) and ancestor != kit_root:
            problems.append(f"inside the source tree {ancestor}")
            break
    for folder in dict.fromkeys([out, out.parent]):
        roms = [n for n in p.listdir(folder) if n.lower().endswith(ROM_SUFFIXES)] if p.is_dir(folder) else []
        if roms:
            problems.append(f"{folder} holds ROM images ({', '.join(roms[:3])}{', ...' if len(roms) > 3 else ''}): "
                            "the ROM folder is input only")
    existing = _nearest_existing(p, out)
    if not p.writable(existing):
        problems.append(f"{existing} is not writable")
    rows = [_row("output_path", "output directory", "unsafe" if problems else "ok", True,
                 "; ".join(problems) if problems else f"{out}",
                 "choose a directory outside the kit, the project sources and the ROM folder "
                 "(or <project>/.n64dk/...)" if problems else None)]
    if re.match(r"^/mnt/[a-z]/", out.as_posix()):
        rows.append(_row("output_path", "windows drive", "warn", False,
                         "a Windows drive through WSL (DrvFs): slow file metadata, and file locks are not "
                         "shared with Windows processes (assumed); prefer the Linux filesystem"))
    return rows


def schema_checks(p: Probes, run: Path | None, kit_root: Path = KIT_ROOT) -> list[dict]:
    rows = []
    if run is not None:
        index = Path(run) / "donor_index.json"
        if p.is_file(index):
            try:
                from ..match.transplant import INDEX_VERSION
            except Exception as error:
                rows.append(_row("schema", "donor index", "unknown", False, f"cannot import the kit: {error}"))
            else:
                head = (p.head(index) or b"").decode("utf-8", "replace")
                m = re.search(r'"version"\s*:\s*(\d+)', head)
                found = int(m.group(1)) if m else None
                rows.append(_row("schema", "donor index", "ok" if found == INDEX_VERSION else "stale", False,
                                 f"{index}: version {found}, this kit writes {INDEX_VERSION}",
                                 None if found == INDEX_VERSION else
                                 "none needed: the index is rebuilt (every donor re-indexed) on the next run"))
        else:
            rows.append(_row("schema", "donor index", "ok", False, f"{index}: absent (nothing stored to migrate)"))
    from ..fixtures import SNAPSHOT_FORMAT
    stale, current = [], 0
    for stamp in p.glob(kit_root / "build" / "truth", "*/.n64dk-snapshot"):
        first = (p.head(stamp, 64) or b"").decode("utf-8", "replace").splitlines()[:1]
        if first == [f"format {SNAPSHOT_FORMAT}"]:
            current += 1
        else:
            stale.append(stamp.parent.name)
    rows.append(_row("schema", "fixture snapshots", "stale" if stale else "ok", False,
                     f"{current} current (format {SNAPSHOT_FORMAT})" + (f", stale: {', '.join(stale)}" if stale else ""),
                     "none needed: stale snapshots are re-extracted on next use" if stale else None))
    return rows


# --------------------------------------------------------------------------- the report

def run_checks(*, out=None, run=None, toolchain_dir=None, probes: Probes | None = None,
               kit_root: Path = KIT_ROOT) -> dict:
    p = probes or Probes()
    root = toolchain_dir or p.env("N64DK_TOOLCHAIN")
    root = Path(root).expanduser() if root else None
    host_rows, host = host_checks(p)
    checks = (dependency_checks(p) + optional_checks(p) + compiler_checks(p, root) + host_rows +
              disk_checks(p, Path(out) if out else None) + memory_checks(p) +
              output_checks(p, Path(out) if out else None, kit_root) +
              schema_checks(p, Path(run) if run else None, kit_root))
    failures = [f"{row['category']}:{row['name']}" for row in checks if row["required"] and not row["ok"]]
    summary: dict[str, dict[str, int]] = {}
    for row in checks:
        summary.setdefault(row["category"], {}).setdefault(row["status"], 0)
        summary[row["category"]][row["status"]] += 1
    return {"doctor": "host", "schema": DOCTOR_SCHEMA, "ok": not failures, "exit_code": 1 if failures else 0,
            "required_failures": failures, "summary": summary, "host": host,
            "toolchain_root": str(root) if root else None, "checks": checks}


def render_text(report: dict) -> str:
    lines = [f"n64dk host doctor: {'OK' if report['ok'] else 'REQUIRED CHECKS FAILED'}"]
    for row in report["checks"]:
        mark = "ok  " if row["ok"] else ("FAIL" if row["required"] else row["status"][:4])
        detail = row["detail"] if isinstance(row["detail"], str) else json.dumps(row["detail"], default=str)
        lines.append(f"  [{mark:4}] {row['category']}/{row['name']}: {detail}")
        if row["remediation"] and not row["ok"]:
            lines.append(f"         -> {row['remediation']}")
    return "\n".join(lines)


def host_doctor(args) -> int:
    """``doctor --host``: print the report, return the exit status."""
    report = run_checks(out=getattr(args, "out", None), run=getattr(args, "run", None),
                        toolchain_dir=getattr(args, "toolchain_dir", None))
    if getattr(args, "format", "text") == "json":
        json.dump(report, sys.stdout, indent=1, default=str)
        sys.stdout.write("\n")
    else:
        print(render_text(report))
    return report["exit_code"]


def host_arguments(p) -> None:
    """Arguments of the --host mode (added to the doctor command's parser)."""
    p.add_argument("--host", action="store_true",
                   help="check this machine only (no project): dependencies, optional capabilities, compilers, "
                        "host, disk, memory, output path, stored schemas; exit 1 on a required failure")
    p.add_argument("--out", help="with --host: an output directory to vet")
    p.add_argument("--run", help="with --host: a run directory whose stored formats to check")
    p.add_argument("--toolchain-dir", help="with --host: toolchain root (default: N64DK_TOOLCHAIN)")


# --------------------------------------------------------------------------- synthetic match

SMOKE_SOURCE = "int n64dk_smoke_zero(void) { return 0; }\n"
JR_RA = 0x03E00008
MOVE_V0_ZERO = {0x00001021: "addu $v0,$zero,$zero", 0x00001025: "or $v0,$zero,$zero",
                0x0000102D: "daddu $v0,$zero,$zero"}
SMOKE_LANES = (("kmc", "kmc+as"), ("ido53", "ido5.3"), ("ido71", "ido7.1"))


def synthetic_match(toolchain_root, workdir) -> dict:
    """Compile ``SMOKE_SOURCE`` with the first installed of KMC GCC 2.7.2, IDO 5.3, IDO 7.1 through
    the kit's build pipeline and compare its code with ``jr $ra`` + ``move $v0,$zero`` (zero padding
    allowed).  PASS / FAIL / SKIP (no such compiler)."""
    if toolchain_root is None:
        return {"status": "SKIP", "reason": "no toolchain root (--toolchain-dir or N64DK_TOOLCHAIN)"}
    from ..elf import Elf
    from ..match.pipeline import BuildSpec, build
    from ..match.toolchain import PROFILES, Config
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    cfg = Config(project=work, toolchain_root=Path(toolchain_root).expanduser())
    for key, pipeline in SMOKE_LANES:
        if cfg.toolchain(key) is None:
            continue
        source = work / "n64dk_smoke.c"
        source.write_text(SMOKE_SOURCE)
        try:
            obj = build(source, BuildSpec(pipeline, list(PROFILES[key].default_flags)), cfg, work / "build")
            text = Elf.load(obj).relocate(".text", {".text": 0x80000400}, {})
        except Exception as error:
            return {"status": "FAIL", "pipeline": pipeline, "error": f"{type(error).__name__}: {error}"[-600:]}
        words = struct.unpack(f">{len(text) // 4}I", text[:len(text) // 4 * 4])
        ok = len(words) >= 2 and words[0] == JR_RA and words[1] in MOVE_V0_ZERO and not any(words[2:])
        return {"status": "PASS" if ok else "FAIL", "pipeline": pipeline,
                "flags": list(PROFILES[key].default_flags), "words": [f"{w:08X}" for w in words[:8]],
                "move_form": MOVE_V0_ZERO.get(words[1]) if len(words) > 1 else None}
    return {"status": "SKIP", "reason": "none of KMC GCC 2.7.2, IDO 5.3, IDO 7.1 is installed under the root"}


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m n64dk.commands.doctor")
    sub = parser.add_subparsers(dest="action", required=True)
    smoke = sub.add_parser("smoke-match", help="synthetic one-function compile and compare")
    smoke.add_argument("--toolchain-dir")
    smoke.add_argument("--work", required=True)
    args = parser.parse_args(argv)
    root = args.toolchain_dir or os.environ.get("N64DK_TOOLCHAIN")
    result = synthetic_match(root, args.work)
    print(json.dumps(result, indent=1))
    return {"PASS": 0, "FAIL": 1, "SKIP": 2}[result["status"]]


if __name__ == "__main__":
    sys.exit(_main())
