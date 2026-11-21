"""High-level correctness, experiment, search, integration and orchestration commands."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import shlex
import difflib
import csv
import struct
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

from ..cli import command
from ..match.inventory import build as build_inventory
from ..match.layout import (boundary_proposals, configured_image_path, from_inventory,
                            physical_collisions, preflight, resolve_image_map)
from ..match.registry import resolve_store, standard_store
from ..match.results import ResultIndex
from ..match.session import MatchSession, SessionStore
from ..match.toolchain import LAYOUT, PROFILES, load_config


def _digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _common_project(p):
    p.add_argument("--project", default=".")
    p.add_argument("--state-root")
    p.add_argument("--database")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--summary-only", action="store_true")


def _image_map_args(p):
    p.add_argument("--project", default=".")
    p.add_argument("--image", required=True)
    p.add_argument("--function", required=True)
    p.add_argument("--image-file")


@command("image-map-explain", "Explain raw, linked and catalogue address transforms and fail on ambiguity")
def image_map_explain(args):
    project = Path(args.project).resolve(); inventory = build_inventory(project)
    image_key = args.image.replace("_", "")
    rows = [row for row in inventory["functions"]
            if row["image"].replace("_", "") == image_key and
            (row["name"] == args.function or row["vram"].lower() == args.function.lower())]
    if len(rows) != 1: raise ValueError(f"expected one catalogue function, found {len(rows)}")
    row = rows[0]
    supplied = Path(args.image_file).expanduser() if args.image_file else None
    image_path = ((project / supplied).resolve() if supplied and not supplied.is_absolute()
                  else supplied.resolve() if supplied else configured_image_path(project, row["image"]))
    if not image_path.is_file(): raise ValueError(f"canonical image is missing: {image_path}")
    layout = from_inventory(row); mapping = resolve_image_map(image_path, layout)
    preflight_error = None
    try:
        check = preflight(image_path, layout, row.get("catalogue_status", ""))
    except ValueError as error:
        preflight_error = str(error)
        check = {"ok": False, "error": preflight_error,
                 "mapping": {**layout.as_dict(), "linked_vram": mapping.get("linked_vram"),
                             "linked_base": mapping.get("linked_base"),
                             "target_digest": mapping.get("target_digest")},
                 "entry_preflight": {"entry_status": "unknown", "call_target_evidence": []},
                 "warnings": ["mapping resolved, but structural preflight rejected the interval"]}
    collisions = physical_collisions(image_path, layout, mapping, inventory["functions"])
    selected_collisions = [item for item in collisions
                           if item["candidate_authority"] == mapping["selected"]["authority"]]
    agreements = {"catalogue": True,
                  "manifest": row.get("coverage_authority") == "manifest",
                  "splat": any(args.image.lower().replace("_", "") in str(item).lower().replace("_", "")
                               for item in inventory.get("splat", [])),
                  "symbols": bool(check["entry_preflight"].get("call_target_evidence")),
                  "raw_composite_digest": mapping["raw_target_digest"] == mapping["target_digest"]}
    errors = []
    if preflight_error:
        errors.append(preflight_error)
    if selected_collisions:
        errors.append("selected physical target collides with an existing owner: " +
                      ", ".join(item["owner"] for item in selected_collisions))
    return {"function": row["name"], "image": args.image, "mapping": {
                key: value for key, value in mapping.items() if key not in {"selected", "candidates"}},
            "selected": {key: value for key, value in mapping["selected"].items() if key != "code"},
            "candidates": [{key: value for key, value in item.items() if key != "code"}
                           for item in mapping["candidates"]],
            "agreements": agreements, "competing_mapping_collisions": collisions,
            "preflight": check, "ambiguous": False, "safe": not errors,
            "errors": errors}


image_map_explain.arguments = _image_map_args


@command("image-resolve", "Resolve one logical image/function to an authoritative coordinate tuple")
def image_resolve(args):
    return image_map_explain(args)


image_resolve.arguments = _image_map_args


def _address_explain_args(p):
    p.add_argument("address", type=lambda value: int(value, 0))
    p.add_argument("--project", default=".")
    p.add_argument("--image", required=True)
    p.add_argument("--size", type=lambda value: int(value, 0))
    p.add_argument("--image-file")


@command("address-explain", "Explain a catalogue or linked VRAM in a logical image")
def address_explain(args):
    project = Path(args.project).resolve(); inventory = build_inventory(project)
    normalized = args.image.replace("_", "").replace("-", "").lower()
    rows = [row for row in inventory["functions"]
            if row["image"].replace("_", "").replace("-", "").lower() == normalized
            and (args.size is None or int(row["size"]) == args.size)]
    supplied = Path(args.image_file).expanduser() if args.image_file else None
    image_path = ((project / supplied).resolve() if supplied and not supplied.is_absolute()
                  else supplied.resolve() if supplied else configured_image_path(project, args.image))
    matches = []
    for row in rows:
        try:
            mapping = resolve_image_map(image_path, from_inventory(row))
        except ValueError:
            # An unrelated bad catalogue interval must not hide a unique valid resolution.
            continue
        if int(row["vram"], 0) == args.address:
            matches.append((row, mapping, "catalogue_vram")); continue
        if mapping.get("linked_vram") and int(mapping["linked_vram"], 0) == args.address:
            matches.append((row, mapping, "linked_vram"))
    if len(matches) != 1:
        raise ValueError(f"address resolved to {len(matches)} intervals in logical image {args.image}")
    row, mapping, space = matches[0]
    return {"identity": f"{row['image']}:{mapping['linked_vram']}:{int(row['size'])}",
            "matched_space": space, "function": row["name"], "image": row["image"],
            "coordinates": mapping["coordinates"], "catalogue_vram": mapping["catalogue_vram"],
            "linked_vram": mapping["linked_vram"], "raw_rom": mapping["raw_rom"],
            "target_digest": mapping["target_digest"],
            "mapping_authority": mapping["selected"]["authority"], "ambiguous": False}


address_explain.arguments = _address_explain_args


def _project_workbench_config(project: Path) -> dict:
    path = project / "n64dk.toml"
    if not path.is_file(): return {}
    return tomllib.loads(path.read_text()).get("workbench", {})


def _python_path(value: str | os.PathLike[str] | None = None) -> str:
    """Canonicalize spelling without dereferencing a virtualenv interpreter symlink."""
    selected = os.fspath(value) if value is not None else os.path.join(os.path.realpath(sys.prefix), "bin", "python")
    return os.path.abspath(os.path.expanduser(selected))


def _worktree_snapshot(project: Path) -> tuple[dict[str, bytes], set[str]]:
    tracked_run = subprocess.run(["git", "ls-files", "-z"], cwd=project,
                                 stdout=subprocess.PIPE, check=True, timeout=60)
    tracked = {}
    for raw in tracked_run.stdout.split(b"\0"):
        if not raw: continue
        name = os.fsdecode(raw); path = project / name
        tracked[name] = path.read_bytes() if path.is_file() else b""
    untracked_run = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "-z"],
                                   cwd=project, stdout=subprocess.PIPE, check=True, timeout=60)
    untracked = {os.fsdecode(raw) for raw in untracked_run.stdout.split(b"\0") if raw}
    for name in untracked:
        path = project / name
        if path.is_file(): tracked[name] = path.read_bytes()
    return tracked, untracked


def _restore_worktree_snapshot(project: Path, tracked: dict[str, bytes], untracked: set[str]) -> None:
    for name, content in tracked.items():
        path = project / name
        if not path.is_file() or path.read_bytes() != content:
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
    current = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "-z"],
                             cwd=project, stdout=subprocess.PIPE, check=True,
                             timeout=60).stdout.split(b"\0")
    for raw in current:
        if not raw: continue
        name = os.fsdecode(raw)
        if name not in untracked:
            path = project / name
            if path.is_file() or path.is_symlink(): path.unlink()


def _transaction_diff(project: Path, before: dict[str, bytes], prior_untracked: set[str]) -> dict:
    changed, patches = [], []
    names = set(before)
    now_untracked = {os.fsdecode(raw) for raw in subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=project,
        stdout=subprocess.PIPE, check=True, timeout=60).stdout.split(b"\0") if raw}
    names.update(now_untracked - prior_untracked)
    for name in sorted(names):
        old = before.get(name, b""); path = project / name
        new = path.read_bytes() if path.is_file() else b""
        if old == new: continue
        changed.append(name)
        try:
            patches.extend(difflib.unified_diff(old.decode().splitlines(), new.decode().splitlines(),
                                                f"a/{name}", f"b/{name}", lineterm=""))
        except UnicodeDecodeError:
            patches.append(f"Binary files differ: {name}")
    return {"paths": changed, "unified": "\n".join(patches)[-50000:]}


def _canonical_env() -> dict[str, str]:
    env = dict(os.environ); prefix = Path(sys.prefix).resolve()
    env["VIRTUAL_ENV"] = str(prefix)
    env["PATH"] = str(prefix / "bin") + os.pathsep + env.get("PATH", "")
    return env


def _doctor_args(p):
    p.add_argument("--project", default=".")
    p.add_argument("--profile", choices=("matching", "gpu-research", "full"), default="matching")
    p.add_argument("--fix-user", action="store_true",
                   help="print user-local remediation commands; never mutates the host")
    p.add_argument("--skip-functional", action="store_true", help="skip functional smoke tests")
    from .doctor import host_arguments
    host_arguments(p)


def _doctor_functional_smokes(project: Path) -> list[dict]:
    smokes = []
    def record(name, ok, detail=None): smokes.append({"name": name, "ok": bool(ok), "detail": detail})
    try:
        with tempfile.TemporaryDirectory(prefix="n64dk-doctor-map-") as name:
            root = Path(name); raw = root / "overlay_9.z64"; composite = root / "overlay_9_composite.z64"
            code = struct.pack(">III", 0x24020000, 0x03E00008, 0)
            raw_data = bytearray(0x1100); raw_data[0x1000:0x100C] = code; raw.write_bytes(raw_data)
            composite_data = bytearray(0x400); struct.pack_into(">I", composite_data, 8, 0x80000400)
            composite_data[0x200:0x20C] = code; composite.write_bytes(composite_data)
            layout = __import__("n64dk.match.layout", fromlist=["Layout"]).Layout(
                "overlay9", 0x1000, 0x80002000, 12, 0x80002000, 12)
            mapping = resolve_image_map(composite, layout)
            record("raw/composite mapping", mapping["selected"]["offset"] == 0x200,
                   {"selected": f"0x{mapping['selected']['offset']:X}"})
    except Exception as error: record("raw/composite mapping", False, str(error))
    try:
        parser = __import__("argparse").ArgumentParser(add_help=False); _compiler_explain_args(parser)
        one = parser.parse_args(["session", "--candidate", "one"])
        pair = parser.parse_args(["session", "--from", "one", "--to", "two"])
        record("compiler-explain contract", bool(one.candidate and pair.from_candidate and pair.to_candidate))
    except Exception as error: record("compiler-explain contract", False, str(error))
    try:
        with tempfile.TemporaryDirectory(prefix="n64dk-doctor-sqlite-") as name:
            path = Path(name) / "state.sqlite"; first = sqlite3.connect(path); second = sqlite3.connect(path)
            first.execute("PRAGMA journal_mode=WAL"); first.execute("CREATE TABLE smoke(value INTEGER)"); first.commit()
            second.execute("INSERT INTO smoke VALUES (1)"); second.commit()
            ok = first.execute("SELECT COUNT(*) FROM smoke").fetchone()[0] == 1
            first.close(); second.close(); record("SQLite schema/write locking", ok)
    except Exception as error: record("SQLite schema/write locking", False, str(error))
    try:
        with tempfile.TemporaryDirectory(prefix="n64dk-doctor-stale-") as name:
            root = Path(name); source = root / "func_80002000.c"; source.write_text("int func_80002000(void){return 0;}\n")
            index = ResultIndex(root / "results.sqlite")
            payload = {"exact": True, "pipeline": "fixture", "flags": [], "sha256_text": "a" * 64,
                       "replay": {"version": "fixture"}}
            identifier = index.record_direct(payload, {"addr": 0x80002000, "size": 12,
                "rom_sha256": "b" * 64, "target_text_sha256": "a" * 64, "vram_delta": 0x80001000}, source,
                "func_80002000", "valid_current_target")
            source.unlink(); rows = index.query(function="func_80002000", missing_source=True)
            index.close(); record("stale observation invalidation", bool(rows and rows[0]["observation_id"] == identifier))
    except Exception as error: record("stale observation invalidation", False, str(error))

    # Exercise one real, already-exact project unit when a compact manifest member is available.
    compile_smoke = None
    inventory = build_inventory(project)
    for manifest in sorted(project.glob("config/**/matching_overlay*_kmc_units.tsv")):
        number = re.search(r"overlay(\d+)", manifest.name)
        if not number: continue
        image = project / "build/us/audit" / f"overlay_{number.group(1)}_composite.z64"
        if not image.is_file(): continue
        with manifest.open(newline="") as stream:
            for values in csv.reader((line for line in stream if line.strip() and not line.startswith("#")), delimiter="\t"):
                if len(values) < 4: continue
                try: rom, size = int(values[0], 0), int(values[1], 0)
                except ValueError: continue
                if size > 0x20 or not (project / values[2]).is_file(): continue
                rows = [row for row in inventory["functions"] if row["image"] == "overlay" + number.group(1)
                        and int(row["rom"], 0) == rom and int(row["size"]) == size]
                if len(rows) != 1: continue
                pipeline = "kmc+asn64" if "--assembler=asn64" in values[3] else "kmc+builtin"
                command_line = [sys.executable, "-m", "n64dk", "--format", "json", "match", values[2],
                    "--project", str(project), "--rom", str(image), "--addr", rows[0]["vram"],
                    "--size", f"0x{size:X}", "--pipeline", pipeline, "--flags", values[3]]
                run = subprocess.run(command_line, cwd=project, text=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, timeout=120)
                try: output = json.loads(run.stdout)
                except json.JSONDecodeError: output = {}
                compile_smoke = {"source": values[2], "exit_code": run.returncode,
                                 "exact": bool(output.get("exact")), "stderr": run.stderr[-1000:]}
                break
        if compile_smoke: break
    record("compile-and-compare fixture", compile_smoke is None or bool(compile_smoke["exact"]),
           compile_smoke or "skipped: project has no compact exact manifest fixture")
    return smokes


@command("doctor", "Validate project, toolchains, host tools and optional research profiles")
def doctor(args):
    if getattr(args, "host", False):                  # machine checks only (n64dk.commands.doctor)
        from .doctor import host_doctor
        return host_doctor(args)
    project = Path(args.project).expanduser().resolve()
    cfg = load_config(project)
    workbench = _project_workbench_config(project)
    checks = []

    def add(group, name, ok, found=None, remediation=None, required=True):
        checks.append({"group": group, "name": name, "ok": bool(ok), "required": required,
                       "found": str(found) if found else None, "remediation": remediation})

    required_tools = {"python": sys.executable, "git": shutil.which("git"),
                      "make": shutil.which("make"), "sqlite3": shutil.which("sqlite3")}
    for name, path in required_tools.items():
        add("required matching", name, path, Path(path).resolve() if path else None,
            f"install {name} under ~/.local/bin and add it to PATH")
    rom = cfg.find_rom()
    add("required project build", "project ROM", rom and rom.is_file(), rom,
        "set [match] rom in n64dk.toml or provide a clean project ROM")
    for key, profile in PROFILES.items():
        root = cfg.toolchain(key)
        add("required matching", profile.name, root is not None, root,
            f"install {profile.name} at {(cfg.toolchain_root or project / '.toolchain') / LAYOUT[key]}")
    gnu = cfg.gnu_as()
    add("required matching", "MIPS GNU assembler", gnu is not None, gnu[0] if gnu else None,
        "install a MIPS binutils toolchain under .toolchain/mips-binutils")
    sn_root = cfg.toolchains.get("sn") or ((cfg.toolchain_root / LAYOUT["sn"])
                                           if cfg.toolchain_root else None)
    add("required matching", "asn64", bool(sn_root and (sn_root / "asn64.exe").is_file()),
        sn_root / "asn64.exe" if sn_root else None, "place asn64.exe inside sn-gcc-2.8")
    for name in ("splat", "spimdisasm"):
        add("optional corpus research", name, shutil.which(name), shutil.which(name),
            f"install {name} in the active virtual environment", required=False)
    if args.profile in {"gpu-research", "full"}:
        add("optional NVIDIA GPU", "nvidia-smi", shutil.which("nvidia-smi"), shutil.which("nvidia-smi"),
            "install a compatible NVIDIA driver; GPU packages remain optional", required=False)
        for package in ("torch", "cupy", "onnxruntime"):
            found = __import__("importlib").util.find_spec(package) is not None
            add("optional NVIDIA GPU", package, found, package if found else None,
                f"install {package} only in a dedicated gpu-research environment", required=False)
    dependency_groups = workbench.get("dependencies", {})
    python_profiles = workbench.get("python", {})
    selected_pythons = {"matching": _python_path(python_profiles.get("matching")),
                        "project-test": _python_path(python_profiles.get("project_test")),
                        "gpu": _python_path(python_profiles.get("gpu"))}
    import_probes = ("elftools", "capstone", "PIL", "networkx", "zstandard", "lz4",
                     "numpy", "scipy", "sklearn")
    interpreter_probes = []
    for label, executable in (("qualified", selected_pythons["matching"]),
                              ("system", shutil.which("python3"))):
        if not executable: continue
        for package in import_probes:
            probe = subprocess.run([executable, "-c", f"import {package}; print(getattr({package}, '__version__', 'ok'))"],
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   timeout=30)
            interpreter_probes.append({"interpreter": label, "python": str(Path(executable).resolve()),
                "package": package, "ok": probe.returncode == 0,
                "output": (probe.stdout or probe.stderr)[-500:]})
    for name in ("jq", "rg", "cmake", "ninja", "meson", "ccache", "dot"):
        add("optional corpus research", name, shutil.which(name), shutil.which(name),
            f"install {name} for accelerated research workflows", required=False)
    sqlite_features = {"json": False, "wal": False}
    try:
        probe_db = sqlite3.connect(":memory:")
        sqlite_features["json"] = probe_db.execute("SELECT json_valid('{}')").fetchone()[0] == 1
        sqlite_features["wal"] = sqlite3.sqlite_version_info >= (3, 7, 0)
        probe_db.close()
    except sqlite3.DatabaseError: pass
    for group, packages in dependency_groups.items():
        python_key = "project-test" if group == "project_test" else "gpu" if "gpu" in group else "matching"
        executable = selected_pythons[python_key]
        for package in packages:
            probe = subprocess.run([executable, "-c", f"import {package}"], text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            add("project-test dependencies" if group == "project_test" else "optional " + group,
                package, probe.returncode == 0, executable,
                f"{executable} -m pip install {package}", required=group == "project_test")
    cache = project / ".n64dk"
    cache.mkdir(exist_ok=True)
    add("required matching", "writable state", os.access(cache, os.W_OK), cache)
    free = shutil.disk_usage(project).free
    add("required project build", "disk space >= 2 GiB", free >= 2 * 1024**3,
        f"{free / 1024**3:.1f} GiB free")
    locks = [str(path) for path in cache.rglob("*.lock") if time.time() - path.stat().st_mtime > 3600]
    prefix = str(Path(sys.prefix).resolve())
    kit_venv = str((Path(__file__).resolve().parents[2] / ".venv").resolve())
    add("required matching", "canonical virtual environment", prefix == kit_venv, prefix,
        f"invoke {kit_venv}/bin/python or resolve the configured interpreter path")
    functional = [] if args.skip_functional else _doctor_functional_smokes(project)
    for smoke in functional:
        add("functional smoke", smoke["name"], smoke["ok"], smoke.get("detail"),
            "repair the failing functional subsystem before trusting matching results")
    optional_disabled = []
    if not shutil.which("splat"): optional_disabled.append({"capability": "Splat boundary corroboration",
                                                             "commands": ["image-map-explain", "boundary-audit"]})
    if not shutil.which("spimdisasm"): optional_disabled.append({"capability": "advanced disassembly metadata",
                                                                  "commands": ["investigate", "corpus-index"]})
    required_failed = [row for row in checks if row["required"] and not row["ok"]]
    return {"profile": args.profile, "project": str(project), "ok": not required_failed,
            "checks": checks, "stale_locks": locks, "active_workers": [],
            "canonical_paths": {"python": str(Path(sys.executable).resolve()), "prefix": prefix,
                                "project": str(project)},
            "selected_python": selected_pythons, "integration_hooks": workbench.get("hooks", {}),
            "interpreter_import_probes": interpreter_probes, "sqlite_features": sqlite_features,
            "functional_smokes": functional, "optional_capabilities_disabled": optional_disabled,
            "remediation": [row["remediation"] for row in checks
                            if not row["ok"] and row["remediation"]] if args.fix_user else []}


doctor.arguments = _doctor_args


def _cap_args(p):
    p.add_argument("--project", default=".")
    p.add_argument("--state-root")
    p.add_argument("--invalidate", action="store_true")
    p.add_argument("--pipeline")
    p.add_argument("--option")
    p.add_argument("--stage", choices=("preprocessor", "compiler", "assembler", "linker"))


@command("compiler-capabilities", "Inspect or selectively invalidate compiler-option capability records")
def compiler_capabilities(args):
    store = SessionStore(args.state_root or standard_store(args.project))
    try:
        removed = store.invalidate_capabilities(args.pipeline, args.option, args.stage) if args.invalidate else 0
        return {"removed": removed, "capabilities": store.capabilities()}
    finally:
        store.close()


compiler_capabilities.arguments = _cap_args


def _load_matrix(path: str) -> dict:
    text = Path(path).read_text()
    if path.endswith(".json"):
        return json.loads(text)
    try:
        import tomllib
        if path.endswith(".toml"):
            return tomllib.loads(text)
    except Exception:
        pass
    # Deliberately small YAML subset: scalar keys, inline JSON lists, and axes mapping.
    result, axes, section = {}, {}, None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip(): continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if section in {"constraints"} and stripped.startswith("-"):
            result.setdefault(section, []).append(stripped[1:].strip()); continue
        key, _, value = stripped.partition(":")
        if indent == 0:
            section = key if not value.strip() else None
            if value.strip(): result[key] = value.strip()
        elif section == "axes":
            axes[key] = json.loads(value.strip().replace("'", '"'))
    if axes: result["axes"] = axes
    return result


def _matrix_args(p):
    p.add_argument("session_id")
    p.add_argument("matrix")
    p.add_argument("--project", default=".")
    p.add_argument("--state-root")
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--limit", type=int, default=256)


@command("match-matrix", "Run a declarative bounded source/prototype experiment matrix")
def match_matrix(args):
    spec = _load_matrix(args.matrix)
    axes = spec.get("axes", {})
    if not axes:
        raise ValueError("matrix requires a non-empty axes mapping")
    store = SessionStore(resolve_store(args.session_id, args.project, args.state_root))
    try:
        parent = store.best(args.session_id)
        template = store.cache.get(parent.source_digest).decode()
        names = list(axes)
        variants = []
        for values in itertools.product(*(axes[name] for name in names)):
            mapping = dict(zip(names, values))
            allowed = True
            for constraint in spec.get("constraints", []):
                found = re.fullmatch(r"(\w+)\s*(==|!=)\s*([^ ]+)(?:\s+implies\s+(\w+)\s*(==|!=)\s*([^ ]+))?", constraint)
                if not found: continue
                left, op, value, right, rop, rvalue = found.groups()
                premise = (str(mapping.get(left)) == value) if op == "==" else (str(mapping.get(left)) != value)
                if premise and right:
                    consequence = (str(mapping.get(right)) == rvalue) if rop == "==" else (str(mapping.get(right)) != rvalue)
                    allowed &= consequence
                elif not right:
                    allowed &= premise
            if not allowed: continue
            source = template
            for name, value in mapping.items():
                source = source.replace("{{" + name + "}}", str(value))
            variants.append({"label": ",".join(f"{k}={v}" for k, v in mapping.items()), "source": source})
            if len(variants) >= args.limit: break
        from .match import _session_evaluator
        transaction = MatchSession(store, _session_evaluator(store.context(args.session_id))).try_proposal(
            args.session_id, {"hypothesis": spec.get("hypothesis", "declarative matrix"),
                              "variants": variants}, args.jobs, False)
        transaction["matrix"] = {"axes": axes, "generated": len(variants),
                                 "object_classes": transaction["object_classes"]}
        return transaction
    finally:
        store.close()


match_matrix.arguments = _matrix_args


def _prototype_args(p):
    p.add_argument("session_id")
    p.add_argument("--symbol", required=True)
    p.add_argument("--project", default=".")
    p.add_argument("--state-root")
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--types", default="void,s8,u8,s16,u16,s32,u32,void*")
    p.add_argument("--budget", type=int, default=96)
    p.add_argument("--dimensions", default="prototype,lifetime,narrowing")


@command("prototype-matrix", "Explore evidence-bounded ABI and prototype type variants")
def prototype_matrix(args):
    from ..match.typed_explore import (declaration_context, lifetime_graph, lifetime_variants,
                                       narrowing_variants, prototype_variants, source_shape)
    store = SessionStore(resolve_store(args.session_id, args.project, args.state_root))
    try:
        parent = store.best(args.session_id); source = store.cache.get(parent.source_digest).decode()
        evidence = parent.outcome.get("diff", {}).get("entries", [])
        hints = {"signed-halfword": any("sra" in str(row) and "0x10" in str(row) for row in evidence),
                 "unsigned-halfword": any("andi" in str(row) and "FFFF" in str(row).upper() for row in evidence),
                 "signed-byte": any("sra" in str(row) and "0x18" in str(row) for row in evidence),
                 "unsigned-byte": any("andi" in str(row) and "0xFF" in str(row) for row in evidence)}
        dimensions = {value.strip() for value in args.dimensions.split(",")}
        graph = lifetime_graph(parent.outcome, store.context(args.session_id)); variants = []
        if "prototype" in dimensions:
            variants += prototype_variants(source, args.symbol, args.types.split(","), args.budget)
        if not variants and "prototype" in dimensions:
            raise ValueError("symbol declaration was not found in retained source")
        if "lifetime" in dimensions and (parent.counts.get("register") or parent.counts.get("scheduling")):
            variants += lifetime_variants(source, graph, args.budget - len(variants))
        if "narrowing" in dimensions and (hints["signed-halfword"] or hints["unsigned-halfword"] or
                                            parent.counts.get("instruction")):
            variants += narrowing_variants(source, args.budget - len(variants))
        unique = {}; shapes = {}
        for variant in variants:
            digest = hashlib.sha256(variant["source"].encode()).hexdigest()
            unique.setdefault(digest, variant)
            shapes.setdefault(source_shape(variant["source"]), []).append(variant["label"])
        variants = list(unique.values())[:args.budget]
        from .match import _session_evaluator
        result = MatchSession(store, _session_evaluator(store.context(args.session_id))).try_proposal(
            args.session_id, {"hypothesis": "prototype width controls normalization and carrier allocation",
                              "variants": variants}, args.jobs, False)
        classes = result["object_classes"]
        best = result.get("transaction_best")
        result["abi_evidence"] = hints; result["value_lifetime_graph"] = graph
        result["declaration_context"] = declaration_context(source, args.symbol)
        result["adaptive_selection"] = {"requested": sorted(dimensions),
            "selected": sorted({item.get("family") for item in variants if item.get("family")}),
            "reason": "calls/prototypes affect return and argument registers; register/scheduling residuals select lifetime; extension residuals select narrowing"}
        result["family_report"] = {"family": "prototype/lifetime/narrowing",
            "trials": len(result["transaction_results"]), "unique_objects": len(classes),
            "source_shape_classes": len(shapes), "best_score": best.get("score") if best else None,
            "collapsed_families": [labels for labels in shapes.values() if len(labels) > 1],
            "still_discriminating": ["call-result ownership", "post-call materialization",
                                     "branch result versus function result"]}
        return result
    finally: store.close()


prototype_matrix.arguments = _prototype_args


def _group_db(project: Path):
    root = project / ".n64dk"; root.mkdir(exist_ok=True)
    db = sqlite3.connect(root / "group-sessions.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS groups (group_id TEXT PRIMARY KEY,kind TEXT,payload TEXT,created_ns INTEGER)")
    db.commit(); return db


def _group_start_args(p):
    p.add_argument("--project", default=".")
    p.add_argument("--unit", required=True)
    p.add_argument("--functions", required=True)


@command("family-session-start", "Create a persistent cross-function family experiment context")
def family_session_start(args):
    project = Path(args.project).resolve(); unit = Path(project, args.unit).resolve()
    payload = {"project": str(project), "unit": str(unit), "unit_digest": _digest(unit),
               "functions": args.functions.split(","), "sessions": {}, "kind": "family"}
    group_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
    db = _group_db(project); db.execute("INSERT OR REPLACE INTO groups VALUES (?,?,?,?)",
        (group_id, "family", json.dumps(payload), time.time_ns())); db.commit(); db.close()
    return {"family_id": group_id, **payload}


family_session_start.arguments = _group_start_args


def _tu_start_args(p):
    p.add_argument("unit")
    p.add_argument("--project", default=".")
    p.add_argument("--manifest-row", help="manifest row text or path:line")
    p.add_argument("--rom", help="physical image; defaults to configured logical --image-name")
    p.add_argument("--addr", type=lambda value: int(value, 0), required=True,
                   help="catalogue VRAM for the complete translation-unit text interval")
    p.add_argument("--size", type=lambda value: int(value, 0), required=True)
    p.add_argument("--image-name", required=True)
    p.add_argument("--pipeline")
    p.add_argument("--flags")
    p.add_argument("--config")
    p.add_argument("--toolchain-dir")
    p.add_argument("--symbols", action="append", default=[])
    p.add_argument("-I", "--include", action="append", default=[])
    p.add_argument("--normalizer")
    p.add_argument("--vram-delta", type=lambda value: int(value, 0))
    p.add_argument("--state-root")
    p.add_argument("--no-infer", action="store_true")
    p.add_argument("--strict-entry", action="store_true", default=True)
    p.add_argument("--jobs", type=int, default=1)


@command("tu-session-start", "Create a translation-unit session preserving source and manifest context")
def tu_session_start(args):
    from .match import _session_context, _session_evaluator, _setup, _spec
    from ..match.registry import SessionRegistry
    project = Path(args.project).resolve(); unit = Path(project, args.unit).resolve()
    if not unit.is_file():
        raise FileNotFoundError(unit)
    cfg, target, _symbols, symbol_files = _setup(args)
    spec = _spec(args, cfg); source = unit.read_bytes()
    context = _session_context(args, cfg, target, symbol_files, unit, spec, source)
    context.update({"translation_unit": str(unit.relative_to(project)),
                    "manifest_row": args.manifest_row,
                    "comparison_scope": "complete owned TU text/data/relocations",
                    "inventory_image": args.image_name})
    store_root = Path(args.state_root).resolve() if args.state_root else standard_store(project)
    store = SessionStore(store_root)
    try:
        session_id, baseline = MatchSession(store, _session_evaluator(context, strict=True)).create(
            context, source, spec.pipeline, tuple(spec.flags))
        registry = SessionRegistry(project)
        try:
            registry.register(session_id, store_root, context, baseline.source_digest,
                              "translation-unit")
        finally:
            registry.close()
    finally:
        store.close()
    payload = {"project": str(project), "unit": str(unit), "unit_digest": _digest(unit),
               "manifest_row": args.manifest_row, "kind": "translation-unit",
               "match_session_id": session_id, "state_root": str(store_root),
               "image": args.image_name, "catalogue_vram": f"0x{args.addr:08X}",
               "linked_vram": f"0x{context['addr']:08X}", "size": args.size}
    group_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
    db = _group_db(project); db.execute("INSERT OR REPLACE INTO groups VALUES (?,?,?,?)",
        (group_id, "translation-unit", json.dumps(payload), time.time_ns())); db.commit(); db.close()
    return {"tu_session_id": group_id, **payload, "baseline": baseline.summary(True),
            "regression_policy": "strictly compare complete owned TU text/data/relocations"}


tu_session_start.arguments = _tu_start_args


def _code_query_args(p):
    p.add_argument("--project", default=".")
    p.add_argument("--symbol")
    p.add_argument("--address")
    p.add_argument("--type-conflicts", action="store_true")
    p.add_argument("--all-projects", action="store_true")
    p.add_argument("--limit", type=int, default=100)


@command("code-query", "Search declarations, definitions, calls, manifests, candidates and sessions")
def code_query(args):
    query = args.symbol or args.address
    if not query: raise ValueError("--symbol or --address is required")
    roots = [Path(args.project).resolve()]
    if args.all_projects: roots = [path for path in Path(args.project).resolve().parent.iterdir() if path.is_dir()]
    hits = []
    matches = []
    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "--json", "-n", "--glob", "!build/**", "--glob", "!.git/**", query,
               *map(str, roots)]
        run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             timeout=120)
        for line in run.stdout.splitlines():
            try: value = json.loads(line)
            except json.JSONDecodeError: continue
            if value.get("type") != "match": continue
            data = value["data"]
            matches.append((data["path"]["text"], data["line_number"], data["lines"]["text"].strip()))
    else:
        ignored = {"build", ".git", ".venv", ".n64dk", "__pycache__"}
        for root in roots:
            for path in root.rglob("*"):
                if not path.is_file() or any(part in ignored for part in path.relative_to(root).parts):
                    continue
                try: lines = path.read_text(errors="strict").splitlines()
                except (UnicodeError, OSError): continue
                matches.extend((str(path), number, text.strip())
                               for number, text in enumerate(lines, 1) if query in text)
    for path, line_number, text in matches:
        kind = ("manifest" if "matching_" in path else "catalogue" if "functions_" in path
                else "candidate" if ".tmp" in path else "declaration" if re.search(r"\bextern\b|;\s*$", text)
                else "definition" if "{" in text else "call-or-reference")
        hits.append({"authority": {"manifest": 0, "catalogue": 1, "definition": 2}.get(kind, 3),
                     "kind": kind, "path": path, "line": line_number, "text": text})
    hits.sort(key=lambda row: (row["authority"], row["path"], row["line"]))
    conflicts = sorted({re.sub(r"\s+", " ", row["text"]) for row in hits if row["kind"] == "declaration"})
    return {"query": query, "count": len(hits), "results": hits[:args.limit],
            "type_conflicts": conflicts if args.type_conflicts and len(conflicts) > 1 else []}


code_query.arguments = _code_query_args


def _asm_query_args(p):
    p.add_argument("--sequence", required=True)
    p.add_argument("--project", default=".")
    p.add_argument("--all-projects", action="store_true")
    p.add_argument("--register-renamed", action="store_true")
    p.add_argument("--limit", type=int, default=100)


@command("asm-query", "Search normalized opcode sequences and delay-slot forms across projects")
def asm_query(args):
    wanted = [item.strip() for item in args.sequence.split(",")]
    roots = [Path(args.project).resolve()]
    if args.all_projects: roots = [p for p in roots[0].parent.iterdir() if p.is_dir()]
    results = []
    for root in roots:
        for path in itertools.chain(root.glob("asm/**/*.s"), root.glob("src/**/*.s")):
            lines = path.read_text(errors="ignore").splitlines(); ops = []
            for index, line in enumerate(lines):
                value = line.split("#", 1)[0].strip()
                if not value or value.endswith(":") or value.startswith("."): continue
                ops.append((value.split(None, 1)[0], index + 1))
            for index in range(len(ops) - len(wanted) + 1):
                if [x[0] for x in ops[index:index + len(wanted)]] == wanted:
                    results.append({"project": root.name, "path": str(path), "line": ops[index][1],
                                    "classification": "register-renamed" if args.register_renamed else "opcode-exact"})
                    if len(results) >= args.limit: return {"count": len(results), "results": results}
    return {"count": len(results), "results": results}


asm_query.arguments = _asm_query_args


def _boundary_args(p):
    p.add_argument("function")
    p.add_argument("--project", default=".")
    p.add_argument("--image", required=True)
    p.add_argument("--strict", action="store_true")


@command("boundary-audit", "Reconcile catalogue, Splat, CFG, prologue, delay-slot and ownership evidence")
def boundary_audit(args):
    inventory = build_inventory(args.project); rows = [x for x in inventory["functions"] if x["name"] == args.function]
    if len(rows) != 1: raise ValueError(f"function resolved to {len(rows)} rows")
    layout = from_inventory(rows[0]); error = None
    try: report = preflight(args.image, layout, rows[0]["catalogue_status"], args.strict)
    except ValueError as caught:
        error = str(caught); report = {"ok": False, "error": error,
            "entry_preflight": {"entry_status": "reject"}, "preflight_version": "image-map-v3"}
    return {"function": rows[0], "preflight": report, "splat": inventory["splat"],
            "decision": report["entry_preflight"]["entry_status"], "rule_version": report["preflight_version"],
            "correction_proposals": boundary_proposals(args.image, layout) if error else []}


boundary_audit.arguments = _boundary_args


def _boundary_correct_args(p):
    p.add_argument("function")
    p.add_argument("--project", default=".")
    p.add_argument("--new-start", type=lambda x: int(x, 0))
    p.add_argument("--new-size", type=lambda x: int(x, 0))
    p.add_argument("--merge-with")
    p.add_argument("--output", required=True)


@command("boundary-correct", "Prepare a non-applying boundary and denominator correction transaction")
def boundary_correct(args):
    inventory = build_inventory(args.project); rows = [x for x in inventory["functions"] if x["name"] == args.function]
    if len(rows) != 1: raise ValueError("function selector is ambiguous")
    project = Path(args.project).resolve(); row = rows[0]
    merge = None
    if args.merge_with:
        merge_rows = [item for item in inventory["functions"]
                      if item["name"] == args.merge_with and item["image"] == row["image"]]
        if len(merge_rows) != 1: raise ValueError("merge target is missing or ambiguous in the same image")
        merge = merge_rows[0]
    starts = [int(row["vram"], 0)] + ([int(merge["vram"], 0)] if merge else [])
    ends = [int(row["vram"], 0) + int(row["size"])] + \
           ([int(merge["vram"], 0) + int(merge["size"])] if merge else [])
    new_start = args.new_start if args.new_start is not None else min(starts)
    new_size = args.new_size if args.new_size is not None else max(ends) - new_start
    if new_start % 4 or new_size <= 0 or new_size % 4:
        raise ValueError("corrected function interval must be positive and 4-byte aligned")
    evidence = next((value for value in row.get("mapping_evidence", []) if value.startswith("catalogue:")), None)
    if not evidence: raise ValueError("inventory row has no authoritative catalogue source")
    relative = evidence.split(":", 2)[1]; catalogue = project / relative
    before_text = catalogue.read_text(); after_lines = []
    target_seen = merge_seen = False
    for line in before_text.splitlines(keepends=True):
        if re.search(rf'\bname\s*=\s*"{re.escape(row["name"])}"', line):
            line = re.sub(r"vram\s*=\s*0x[0-9A-Fa-f]+", f"vram = 0x{new_start:x}", line)
            line = re.sub(r"size\s*=\s*0x[0-9A-Fa-f]+", f"size = 0x{new_size:x}", line)
            target_seen = True
        if merge and re.search(rf'\bname\s*=\s*"{re.escape(merge["name"])}"', line):
            merge_seen = True
            continue
        after_lines.append(line)
    if not target_seen or (merge and not merge_seen):
        raise ValueError("catalogue row text could not be located for a reversible edit")
    after_text = "".join(after_lines)
    forward = "".join(difflib.unified_diff(before_text.splitlines(True), after_text.splitlines(True),
                                            fromfile=str(relative), tofile=str(relative)))
    reverse = "".join(difflib.unified_diff(after_text.splitlines(True), before_text.splitlines(True),
                                            fromfile=str(relative), tofile=str(relative)))
    affected_manifests = sorted({part.split(":", 2)[1] for item in ([row] + ([merge] if merge else []))
                                 for part in item.get("mapping_evidence", []) if part.startswith("manifest:")})
    before = inventory["summary"]; denominator_delta = -1 if merge else 0
    transaction = {"function": rows[0], "requested": {"new_start": args.new_start,
                   "new_size": args.new_size, "merge_with": args.merge_with},
                   "resolved_interval": {"start": f"0x{new_start:X}", "size": f"0x{new_size:X}",
                                         "end": f"0x{new_start + new_size:X}"},
                   "accounting": {"functions_before": before["functions"],
                                  "functions_after": before["functions"] + denominator_delta,
                                  "matched_before": before["covered"], "matched_after": before["covered"],
                                  "denominator_delta": denominator_delta,
                                  "code_bytes_before": sum(int(item["size"]) for item in ([row] + ([merge] if merge else []))),
                                  "code_bytes_after": new_size,
                                  "decompilation_gain": 0}, "apply": False,
                   "catalogue": str(relative), "catalogue_sha256": hashlib.sha256(before_text.encode()).hexdigest(),
                   "forward_patch": forward, "reverse_patch": reverse,
                   "affected_manifests": affected_manifests,
                   "affected": ["catalogue", "manifests", "progress", "ownership ledger",
                                "sessions", "Splat review"]}
    if args.output != "-":
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(transaction, indent=2) + "\n")
    return transaction


boundary_correct.arguments = _boundary_correct_args


def _self_test_args(p):
    p.add_argument("--project", default=".")
    p.add_argument("--pattern")


@command("self-test", "Run the kit test suite with a canonical top-level discovery path")
def self_test(args):
    kit = Path(__file__).resolve().parents[2]
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", str(kit / "tests"), "-t", str(kit), "-v"]
    if args.pattern: cmd += ["-p", args.pattern]
    run = subprocess.run(cmd, cwd=kit, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, timeout=1800)
    return {"ok": run.returncode == 0, "exit_code": run.returncode, "command": cmd,
            "output": run.stdout[-16000:]}


self_test.arguments = _self_test_args


def _family_try_args(p):
    p.add_argument("group_id")
    p.add_argument("--project", default=".")
    p.add_argument("--proposal", required=True, help="shared proposal JSON")
    p.add_argument("--sessions", help="FUNCTION=SESSION_ID comma list")
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--verify-command")


def _group_payload(project: Path, group_id: str):
    db = _group_db(project); row = db.execute("SELECT payload FROM groups WHERE group_id=?", (group_id,)).fetchone(); db.close()
    if not row: raise KeyError(f"unknown group session {group_id}")
    return json.loads(row[0])


@command("family-session-try", "Apply one shared hypothesis transactionally across a function family")
def family_session_try(args):
    project = Path(args.project).resolve(); payload = _group_payload(project, args.group_id)
    proposal = json.loads(Path(args.proposal).read_text() if Path(args.proposal).is_file() else args.proposal)
    sessions = dict(item.split("=", 1) for item in (args.sessions or "").split(",") if item)
    results = []
    from .match import _session_evaluator
    for function in payload.get("functions", []):
        session_id = sessions.get(function) or payload.get("sessions", {}).get(function)
        if not session_id:
            results.append({"function": function, "status": "needs-session"}); continue
        store = SessionStore(resolve_store(session_id, project))
        try:
            value = MatchSession(store, _session_evaluator(store.context(session_id))).try_proposal(
                session_id, proposal, max(1, args.jobs // max(1, len(sessions))), False)
            results.append({"function": function, "status": "evaluated",
                            "transaction_best": value["transaction_best"],
                            "session_best": value["session_best"]})
        finally: store.close()
    verification = None
    if args.verify_command:
        run = subprocess.run(args.verify_command, cwd=project, shell=True, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800)
        verification = {"exit_code": run.returncode, "output": run.stdout[-12000:]}
    return {"group_id": args.group_id, "results": results, "verification": verification,
            "shared_hypothesis": proposal.get("hypothesis")}


family_session_try.arguments = _family_try_args


@command("tu-session-try", "Edit one function in retained translation-unit context and verify the whole unit")
def tu_session_try(args):
    project = Path(args.project).resolve(); payload = _group_payload(project, args.group_id)
    unit = Path(payload["unit"]); original = unit.read_text(); proposal = json.loads(
        Path(args.proposal).read_text() if Path(args.proposal).is_file() else args.proposal)
    source = proposal.get("source", original)
    for edit in proposal.get("edits", []):
        old, new = edit["find"], edit.get("replace", "")
        if old not in source: raise ValueError(f"TU edit text not found: {old[:40]}")
        source = source.replace(old, new, 1)
    if _digest(unit) != payload["unit_digest"]:
        raise RuntimeError("translation unit changed since session start; start a new TU session")
    session_id = payload.get("match_session_id")
    if not session_id:
        raise RuntimeError("legacy plan-only TU session has no strict matching session; recreate it")
    from .match import _session_evaluator
    store = SessionStore(Path(payload["state_root"]))
    try:
        transaction = MatchSession(store, _session_evaluator(store.context(session_id), strict=True)).try_proposal(
            session_id, {"source": source, "hypothesis": proposal.get("hypothesis",
                "translation-unit context experiment"), "variants": [{"label": "tu-candidate"}]},
            args.jobs, True)
    finally:
        store.close()
    best = transaction.get("transaction_best")
    verification = {"strict": True, "transaction_best": best,
                    "all_owned_sections_exact": bool(best and best.get("exact")),
                    "candidate_count": len(transaction.get("transaction_results", []))}
    if args.verify_command:
        with tempfile.TemporaryDirectory(prefix="n64dk-tu-hook-") as name:
            staged = Path(name) / unit.name; staged.write_text(source)
            env = dict(os.environ); env["N64DK_CANDIDATE_SOURCE"] = str(staged)
            run = subprocess.run(args.verify_command, cwd=project, shell=True, env=env, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            verification["project_hook"] = {"exit_code": run.returncode,
                                             "output": run.stdout[-12000:]}
    return {"tu_session_id": args.group_id, "original_digest": payload["unit_digest"],
            "candidate_digest": hashlib.sha256(source.encode()).hexdigest(),
            "match_session_id": session_id, "verification": verification,
            "transaction": transaction, "project_mutated": False,
            "regression_scope": "complete owned TU text/data/relocations"}


tu_session_try.arguments = _family_try_args


def _compiler_explain_args(p):
    p.add_argument("session_id")
    p.add_argument("--candidate", action="append", help="explain one candidate (repeat twice for legacy pair mode)")
    p.add_argument("--from", dest="from_candidate", help="pairwise baseline candidate")
    p.add_argument("--to", dest="to_candidate", help="pairwise comparison candidate")
    p.add_argument("--project", default=".")
    p.add_argument("--state-root")
    p.add_argument("--passes", default="greg,lreg,sched")
    p.add_argument("--artifact-dir")


@command("compiler-explain", "Compare retained candidates at CSE, allocation, reload and scheduling boundaries")
def compiler_explain(args):
    store = SessionStore(resolve_store(args.session_id, args.project, args.state_root))
    try:
        legacy = args.candidate or []
        if args.from_candidate or args.to_candidate:
            if not (args.from_candidate and args.to_candidate) or legacy:
                raise ValueError("use either --candidate ID or the complete --from ID --to ID pair")
            identifiers = [args.from_candidate, args.to_candidate]; mode = "pairwise"
        elif len(legacy) in {1, 2}:
            identifiers = legacy; mode = "single" if len(legacy) == 1 else "pairwise"
        else:
            raise ValueError("supply one --candidate ID or --from ID --to ID")
        candidates = [store.get(args.session_id, value) for value in identifiers]
        missing = [identifier for identifier, value in zip(identifiers, candidates) if value is None]
        if missing: raise ValueError("unknown or stale candidate: " + ", ".join(missing))
        artifacts = []
        root = Path(args.artifact_dir).resolve() if args.artifact_dir else None
        if root: root.mkdir(parents=True, exist_ok=True)
        dump_context = dict(store.context(args.session_id))
        if root: dump_context["artifact_dir"] = str(root)
        from .match import _session_evaluator
        evaluator = _session_evaluator(dump_context) if root else None
        for candidate in candidates:
            try: source = store.cache.get(candidate.source_digest)
            except (KeyError, FileNotFoundError):
                raise ValueError(f"stale candidate {candidate.candidate_id}: retained source is unavailable")
            source_path = None
            if root:
                source_path = root / f"{candidate.candidate_id}.c"; source_path.write_bytes(source)
            dump_outcome = None
            if evaluator:
                dump_outcome = evaluator(source, candidate.pipeline,
                                         tuple(candidate.flags) + (("-da",) if candidate.pipeline.startswith("kmc") else ()),
                                         candidate.candidate_id)
            emitted = [str(path) for path in (root / candidate.candidate_id).glob("*")] if root and (root / candidate.candidate_id).is_dir() else []
            excerpts = {}
            for artifact in (Path(value) for value in emitted):
                if any(token in artifact.name for token in args.passes.split(",")):
                    excerpts[artifact.name] = artifact.read_text(errors="replace")[-4000:]
            artifacts.append({"candidate_id": candidate.candidate_id, "source": str(source_path) if source_path else None,
                              "object_digest": candidate.outcome.get("sha256_text"),
                              "residual": candidate.counts, "passes_requested": args.passes.split(","),
                              "dump_compile_exact": dump_outcome.exact if dump_outcome else None,
                              "artifacts": emitted, "pass_excerpts": excerpts})
        if mode == "pairwise":
            left, right = candidates
            stage = ("instruction-selection/CSE" if left.counts.get("instruction") != right.counts.get("instruction")
                     else "allocation/reload" if left.counts.get("register") != right.counts.get("register")
                     else "scheduling" if left.counts.get("scheduling") != right.counts.get("scheduling") else "object-equivalent")
        else:
            left = candidates[0]
            stage = ("scheduling" if left.counts.get("scheduling") else
                     "allocation/reload" if left.counts.get("register") else
                     "instruction-selection/CSE" if left.counts.get("instruction") else "object-equivalent")
        causal = []
        if stage == "scheduling":
            causal.append({"cause": "operation has no modeled dependency on the intervening call",
                           "evidence": "residual contains scheduling motion around a call boundary",
                           "discriminators": ["callee prototype", "genuine data dependency",
                                              "result ownership", "declaration visibility", "scheduler pass dump"]})
        return {"session_id": args.session_id, "mode": mode, "candidates": artifacts,
                "first_discriminating_stage": stage, "causal_explanation": causal,
                "note": "raw pass dumps are retained when emitted by the configured compiler recipe"}
    finally: store.close()


compiler_explain.arguments = _compiler_explain_args


def _synthesize_args(p):
    p.add_argument("session_id", nargs="?")
    p.add_argument("--candidate", action="append")
    p.add_argument("--function")
    p.add_argument("--candidate-a")
    p.add_argument("--candidate-b")
    p.add_argument("--database", default=".n64dk/results.sqlite")
    p.add_argument("--project", default=".")
    p.add_argument("--state-root")
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--budget", type=int, default=32)


@command("frontier-synthesize", "Compile a minimal cross-product from complementary retained frontiers")
def frontier_synthesize(args):
    if not args.session_id:
        if not (args.function and args.candidate_a and args.candidate_b):
            raise ValueError("supply SESSION and two --candidate values, or --function with --candidate-a/--candidate-b")
        project = Path(args.project).resolve(); inventory = build_inventory(project)
        database = Path(args.database); database = database if database.is_absolute() else project / database
        index = ResultIndex(database)
        try:
            observations = index.query(function=args.function, limit=100000, project=project, inventory=inventory)
        finally: index.close()
        selected = [next((row for row in observations if row["observation_id"] == identifier), None)
                    for identifier in (args.candidate_a, args.candidate_b)]
        if any(row is None for row in selected): raise ValueError("observation is stale, invalid, or not current-target reproducible")
        if any(not row.get("live_validity", {}).get("source_digest_resolves") for row in selected):
            raise ValueError("both observations require recoverable content-addressed source")
        replay = selected[0]["payload"].get("replay") or {}; image = Path(replay.get("image", ""))
        rows = [row for row in inventory["functions"] if row["name"] == args.function]
        if len(rows) != 1 or not image.is_file(): raise ValueError("cannot reconstruct a unique current target session")
        cfg = load_config(project); layout = from_inventory(rows[0]); check = preflight(image, layout, rows[0].get("catalogue_status", ""), True)
        context = {"source_name": f"{args.function}.c", "source_suffix": ".c", "rom": str(image),
            "rom_sha256": hashlib.sha256(image.read_bytes()).hexdigest(), "project": str(project),
            "toolchain_root": str(cfg.toolchain_root) if cfg.toolchain_root else None,
            "toolchains": {key: str(value) for key, value in cfg.toolchains.items()},
            "include": [str(value) for value in cfg.include], "symbols": [str(value) for value in cfg.symbols],
            "normalizer": cfg.normalizer,
            "vram_delta": int(check["mapping"].get("linked_vram") or layout.function_vram, 0) - int(check["selected_image_offset"], 0),
            "addr": int(check["mapping"].get("linked_vram") or layout.function_vram, 0),
            "catalogue_addr": layout.function_vram,
            "size": layout.function_size, "infer_externals": True,
            "inventory_name": args.function, "inventory_image": rows[0]["image"], "layout_preflight": check,
            "target_text_sha256": check["mapping"].get("target_digest")}
        sources = [Path(row["live_validity"]["resolved_source"]).read_bytes() for row in selected]
        pipeline = selected[0]["payload"].get("pipeline") or replay.get("pipeline") or "kmc+asn64"
        flags = selected[0]["payload"].get("flags") or replay.get("flags") or []
        if isinstance(flags, str): flags = shlex.split(flags)
        store = SessionStore(standard_store(project))
        try:
            session = MatchSession(store, __import__("n64dk.commands.match", fromlist=["_session_evaluator"])._session_evaluator(context))
            session_id, first = session.create(context, sources[0], pipeline, tuple(flags))
            transaction = session.try_proposal(session_id, {"hypothesis": "import complementary historical frontier",
                "variants": [{"label": "observation-b", "source": sources[1].decode()}]}, args.jobs, False)
            args.session_id = session_id; args.candidate = [first.candidate_id,
                transaction["transaction_results"][0]["candidate_id"]]
        finally: store.close()
    store = SessionStore(resolve_store(args.session_id, args.project, args.state_root))
    try:
        rows = [store.get(args.session_id, value) for value in (args.candidate or [])]
        if len(rows) < 2 or any(row is None for row in rows): raise ValueError("two retained candidates are required")
        sources = [store.cache.get(row.source_digest).decode().splitlines(keepends=True) for row in rows]
        base = sources[0]; differing = [i for i in range(min(map(len, sources))) if len({s[i] for s in sources}) > 1]
        variants = []
        for choices in itertools.product(range(len(sources)), repeat=min(len(differing), 8)):
            lines = list(base)
            for index, choice in zip(differing, choices): lines[index] = sources[choice][index]
            variants.append({"label": "cross-" + "".join(map(str, choices)), "source": "".join(lines)})
            if len(variants) >= args.budget: break
        from .match import _session_evaluator
        value = MatchSession(store, _session_evaluator(store.context(args.session_id))).try_proposal(
            args.session_id, {"hypothesis": "complementary frontier source choices may coexist",
                              "variants": variants}, args.jobs, False)
        value["synthesis"] = {"source_dimensions": len(differing), "variants": len(variants),
                              "exhausted": len(variants) < args.budget}
        return value
    finally: store.close()


frontier_synthesize.arguments = _synthesize_args


def _attribute_args(p):
    p.add_argument("session_id")
    p.add_argument("--candidate", required=True)
    p.add_argument("--project", default=".")
    p.add_argument("--state-root")


@command("residual-attribution", "Map residual instructions to likely controlling source values and lifetimes")
def residual_attribution(args):
    from ..match.typed_explore import call_result_ownership, lifetime_graph
    store = SessionStore(resolve_store(args.session_id, args.project, args.state_root))
    try:
        row = store.get(args.session_id, args.candidate)
        if row is None: raise ValueError("unknown candidate")
        source = store.cache.get(row.source_digest).decode()
        calls = re.findall(r"\b([A-Za-z_]\w*)\s*\([^;{}]*\)", source)
        declarations = re.findall(r"(?m)^\s*([\w *]+\s+(?:func_[0-9A-Fa-f]{8}|[A-Za-z_]\w*)\s*\([^;]*\)\s*;)", source)
        expressions = [{"line": number, "text": line.strip()} for number, line in enumerate(source.splitlines(), 1)
                       if re.search(r"\(s16\)|\(u16\)|<<\s*16|>>\s*16|func_[0-9A-Fa-f]{8}\s*\(", line)]
        entries = row.outcome.get("diff", {}).get("entries", [])
        attributed = []
        for entry in entries:
            compiled = entry.get("compiled", ""); retail = entry.get("retail", "")
            mechanism = ("narrow return or promotion" if any(x in compiled + retail for x in ("sll", "sra", "andi"))
                         else "carrier lifetime/register ownership" if entry.get("kind") == "register"
                         else "control-flow source form" if compiled.startswith("b") or retail.startswith("b")
                         else "instruction selection")
            specific = []
            if "sra" in retail + compiled:
                specific = ["explicit (s16) cast", "declared width of the result carrier",
                            "callee return width", "shift-pair sign extension"]
            elif re.search(r"move\s+\$s[0-7],\s*\$v0", retail + compiled):
                specific = ["call result remains live across a later call", "named carrier ownership",
                            "callee-saved register pressure"]
            attributed.append({"target_offset": entry.get("target_offset"), "compiled": compiled,
                               "retail": retail, "likely_source_values": calls[-3:],
                               "likely_source_expressions": specific, "source_lines": expressions[:12],
                               "declarations": declarations[:12], "mechanism": mechanism})
        ownership = call_result_ownership(row.outcome, store.context(args.session_id))
        if ownership.get("transfers"):
            attributed = [{**item, "ranked_data_flow": ownership["transfers"],
                           "controlling_source_span": next((line for line in expressions
                               if sum(call in line["text"] for call in calls) >= 2), None),
                           "semantic_or_allocation": "different call-result ownership"}
                          for item in attributed]
        return {"candidate_id": row.candidate_id, "attribution": attributed,
                "value_lifetime_graph": lifetime_graph(row.outcome, store.context(args.session_id)),
                "call_result_ownership": ownership,
                "confidence": "heuristic-def-use", "next_discriminators":
                ["carrier reuse", "narrow prototype", "unified return path"]}
    finally: store.close()


residual_attribution.arguments = _attribute_args


def _neg_args(p):
    p.add_argument("--project", default=".")
    p.add_argument("--state-root")
    p.add_argument("--residual")
    p.add_argument("--compiler")
    p.add_argument("--limit", type=int, default=100)


@command("negatives-query", "Retrieve equivalent objects and disproved hypotheses before compilation")
def negatives_query(args):
    roots = [Path(args.state_root)] if args.state_root else [standard_store(args.project)]
    output = []
    for root in roots:
        if not (root / "sessions.sqlite").is_file(): continue
        store = SessionStore(root)
        try:
            for row in store.db.execute("SELECT session_id,family,conclusion,payload FROM session_experiments ORDER BY created_ns DESC"):
                payload = json.loads(row[3])
                if args.residual and args.residual not in json.dumps(payload): continue
                if args.compiler and args.compiler not in json.dumps(payload): continue
                output.append({"session_id": row[0], "hypothesis": row[1], "conclusion": row[2], "payload": payload})
                if len(output) >= args.limit: break
        finally: store.close()
    return {"count": len(output), "negative_results": output}


negatives_query.arguments = _neg_args


def _integrate_args(p):
    p.add_argument("candidate_id")
    p.add_argument("--session", required=True)
    p.add_argument("--project", default=".")
    p.add_argument("--state-root")
    p.add_argument("--output", required=True)
    p.add_argument("--verify-twice", action="store_true", default=True,
                   help="independently compile twice (the integration default)")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--destination")
    p.add_argument("--manifest")
    p.add_argument("--affected-gate")
    p.add_argument("--full-gate")
    p.add_argument("--project-tests")


@command("integrate-winner", "Prepare and independently verify an exact-winner integration transaction")
def integrate_winner(args):
    store = SessionStore(resolve_store(args.session, args.project, args.state_root))
    try:
        row = store.get(args.session, args.candidate_id)
        if row is None or not row.exact: raise ValueError("candidate is not a retained exact result")
        project = Path(args.project).resolve(); context = store.context(args.session); source = store.cache.get(row.source_digest)
        inventory = build_inventory(project)
        catalogue_addr = context.get("catalogue_addr", context["addr"])
        mapping_rows = [item for item in inventory["functions"]
                        if int(item["vram"], 0) == catalogue_addr and int(item["size"]) == context["size"]
                        and (not context.get("inventory_image") or item["image"] == context["inventory_image"])]
        if len(mapping_rows) != 1: raise ValueError("winner target does not resolve to one authoritative catalogue interval")
        inventory_row = mapping_rows[0]; layout = from_inventory(inventory_row)
        image_path = Path(context["rom"]); canonical_map = resolve_image_map(image_path, layout)
        canonical_check = preflight(image_path, layout, inventory_row.get("catalogue_status", ""), strict_entry=True)
        recorded_offset = context["addr"] - context["vram_delta"]
        collisions = physical_collisions(image_path, layout, canonical_map, inventory["functions"])
        recorded_collisions = [item for item in collisions if int(item["candidate_offset"], 0) == recorded_offset]
        if recorded_offset != canonical_map["selected"]["offset"]:
            detail = (f"; selected bytes are already owned by {recorded_collisions[0]['owner']} "
                      f"at {recorded_collisions[0]['physical_interval']}"
                      if recorded_collisions else "")
            raise ValueError(f"winner used stale physical offset 0x{recorded_offset:X}; canonical offset is "
                             f"0x{canonical_map['selected']['offset']:X}{detail}")
        selected_collisions = [item for item in collisions
                               if item["candidate_authority"] == canonical_map["selected"]["authority"]]
        if selected_collisions:
            raise ValueError("candidate target bytes are already owned by " +
                             ", ".join(item["owner"] for item in selected_collisions))
        context_digest = context.get("target_text_sha256")
        if context_digest and context_digest != canonical_map["target_digest"]:
            raise ValueError("winner target digest disagrees with the current canonical physical interval")
        from .match import _session_evaluator
        evaluator = _session_evaluator(context, strict=True); receipts = []
        for index in range(2):
            result = evaluator(source, row.pipeline, row.flags, f"integration-verification-{index + 1}")
            receipts.append(result.as_dict())
            if not result.exact:
                verdict = result.as_dict().get("proof", {}).get("verdict", "missing")
                raise RuntimeError(f"fresh independent strict verification failed ({verdict})")
        root = Path(args.output).resolve(); root.mkdir(parents=True, exist_ok=True)
        candidate = root / f"{inventory_row['name']}.c"; candidate.write_bytes(source)
        manifest_row = None
        if args.destination:
            destination = Path(args.destination)
            if not destination.is_absolute(): destination = Path(args.project).resolve() / destination
            relative = destination.relative_to(Path(args.project).resolve())
            raw_rom = int(inventory_row["rom"], 0)
            manifest_row = "\t".join((f"0x{raw_rom:X}",
                                      f"0x{context['size']:X}", str(relative), " ".join(row.flags),
                                      row.outcome.get("sha256_text") or context.get("target_text_sha256") or ""))
        workbench = _project_workbench_config(Path(args.project).resolve())
        hooks = workbench.get("hooks", {})
        affected_gate = args.affected_gate or hooks.get("affected_gate")
        full_gate = args.full_gate or hooks.get("full_gate")
        project_tests = args.project_tests or hooks.get("project_tests")
        regeneration = [hooks[key] for key in ("attack", "report", "progress") if hooks.get(key)]
        transaction = {"states": ["candidate_exact", "raw_image_verified", "composite_image_verified",
                      "physical_ownership_verified", "symbol_boundary_verified", "independently_recompiled",
                      "manifest_simulated", "affected_gate_pending", "full_gate_pending", "not_yet_fully_verified"],
                      "candidate": str(candidate), "source_digest": row.source_digest,
                      "target": {key: context.get(key) for key in ("rom_sha256", "target_text_sha256", "addr", "size", "vram_delta")},
                      "pipeline": row.pipeline, "flags": list(row.flags), "verifications": receipts,
                      "planned_updates": ["canonical source", "manifest", "catalogue", "progress", "provenance"],
                      "manifest_row": manifest_row, "applied": False}
        transaction["mapping_evidence"] = {"raw_rom": canonical_map["raw_rom"],
            "composite_rom": f"0x{canonical_map['selected']['offset']:X}",
            "linked_vram": canonical_map["linked_vram"], "catalogue_vram": canonical_map["catalogue_vram"],
            "target_digest": canonical_map["target_digest"], "image_digest": canonical_map["image_digest"],
            "collision_count": len(selected_collisions), "preflight": canonical_check}
        if args.destination:
            transaction["image_mapping"] = {"raw_rom": f"0x{raw_rom:X}",
                "composite_rom": f"0x{canonical_map['selected']['offset']:X}",
                "linked_vram": f"0x{context['addr']:08X}",
                "catalogue_vram": f"0x{catalogue_addr:08X}",
                "image": context.get("inventory_image")}
        transaction["project_hooks"] = {"regeneration": regeneration, "affected_gate": affected_gate,
                                         "full_gate": full_gate, "project_tests": project_tests}
        policy = _resolved_gate_policy(project, "integration")
        # Explicit command-line hooks participate in the same named policy without weakening
        # any other required stage.
        policy["commands"].update({"affected_gate": affected_gate, "full_gate": full_gate,
                                   "project_tests": project_tests})
        policy["missing"] = [name for name in policy["required_hooks"]
                             if not policy["commands"].get(name)]
        transaction["gate_policy"] = policy
        if affected_gate:
            gate_env = _canonical_env(); gate_env["N64DK_CANDIDATE_SOURCE"] = str(candidate)
            if manifest_row: gate_env["N64DK_MANIFEST_ROW"] = manifest_row
            focused = subprocess.run(affected_gate, cwd=project, shell=True, text=True, env=gate_env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     timeout=1800)
            transaction["preintegration_gate"] = {"command": affected_gate,
                "exit_code": focused.returncode, "output": focused.stdout[-12000:]}
            if focused.returncode: raise RuntimeError("focused preintegration project gate failed")
            transaction["states"][7] = "affected_gate_passed"
        receipt = root / "integration-transaction.json"; receipt.write_text(json.dumps(transaction, indent=2) + "\n")
        if args.apply:
            if not (args.destination and args.manifest) or policy["missing"]:
                raise ValueError("--apply requires destination/manifest and a complete integration gate policy; missing: "
                                 + ", ".join(policy["missing"]))
            destination = Path(args.destination)
            if not destination.is_absolute(): destination = project / destination
            manifest = Path(args.manifest)
            if not manifest.is_absolute(): manifest = project / manifest
            dirty = subprocess.run(["git", "status", "--porcelain", "--", str(destination), str(manifest)],
                                   cwd=project, text=True, stdout=subprocess.PIPE,
                                   timeout=60).stdout.strip()
            if dirty: raise ValueError("integration targets contain pre-existing edits; transaction refused")
            tracked_before, untracked_before = _worktree_snapshot(project)
            old_destination = destination.read_bytes() if destination.exists() else None
            old_manifest = manifest.read_bytes() if manifest.exists() else b""
            try:
                destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(source)
                manifest.parent.mkdir(parents=True, exist_ok=True)
                manifest.write_bytes(old_manifest + ((b"" if old_manifest.endswith(b"\n") or not old_manifest else b"\n") + manifest_row.encode() + b"\n"))
                gates = []
                for index, command_text in enumerate(regeneration):
                    run = subprocess.run(command_text, cwd=project, shell=True, text=True, env=_canonical_env(),
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         timeout=1800)
                    gates.append({"gate": f"regeneration-{index + 1}", "exit_code": run.returncode,
                                  "output": run.stdout[-12000:]})
                    if run.returncode: raise RuntimeError("regeneration hook failed")
                for label, command_text in (("affected", affected_gate), ("full", full_gate),
                                            ("project-tests", project_tests)):
                    run = subprocess.run(command_text, cwd=project, shell=True, text=True, env=_canonical_env(),
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         timeout=1800)
                    gates.append({"gate": label, "exit_code": run.returncode, "output": run.stdout[-12000:]})
                    if run.returncode: raise RuntimeError(f"{label} gate failed")
                transaction["applied"] = True; transaction["gates"] = gates
                transaction["states"][-1] = "integration_ready"
                transaction["tracked_diff"] = _transaction_diff(project, tracked_before, untracked_before)
                from ..match.source_index import SourceIndex
                index = SourceIndex(project)
                try: transaction["cache_invalidation"] = {"source_index": index.refresh()}
                finally: index.close()
                invalidated = 0
                for sweep in project.glob(".n64dk/**/sweeps.sqlite"):
                    try:
                        sweep_db = sqlite3.connect(sweep)
                        changed = sweep_db.execute("""UPDATE tranche_items SET status='already_owned',
                            phase='invalidated-after-integration',updated_ns=? WHERE name=?""",
                            (time.time_ns(), inventory_row["name"])).rowcount
                        sweep_db.commit(); sweep_db.close(); invalidated += changed
                    except sqlite3.DatabaseError: pass
                transaction["cache_invalidation"]["tranche_rows"] = invalidated
            except Exception:
                _restore_worktree_snapshot(project, tracked_before, untracked_before)
                transaction["rolled_back"] = True
                receipt.write_text(json.dumps(transaction, indent=2) + "\n")
                raise
            receipt.write_text(json.dumps(transaction, indent=2) + "\n")
        return transaction
    finally: store.close()


integrate_winner.arguments = _integrate_args


def _integration_audit_args(p):
    p.add_argument("--project", default=".")
    p.add_argument("--database", default=".n64dk/results.sqlite")
    p.add_argument("--exact-unintegrated", action="store_true")
    p.add_argument("--limit", type=int, default=100)


@command("integration-audit", "Find exact current-target winners absent from authoritative manifests")
def integration_audit(args):
    project = Path(args.project).resolve(); database = Path(args.database)
    if not database.is_absolute(): database = project / database
    inventory = build_inventory(project)
    index = ResultIndex(database)
    try: rows = index.query(exact_only=True, limit=100000, project=project, inventory=inventory,
                            not_in_manifest=True)
    finally: index.close()
    by_name = {row["name"]: row for row in inventory["functions"]}
    actionable = [row for row in rows if row.get("observation_class") == "currently_reproducible_exact"
                  and row.get("function") in by_name
                  and by_name[row["function"]]["coverage_authority"] != "manifest"
                  and row.get("source") and Path(row["source"]).is_file()]
    return {"count": len(actionable), "candidates": actionable[:args.limit],
            "requirements": ["fresh independent recompilation", "current image and target digest",
                             "source availability", "no aggregate ownership conflict"]}


integration_audit.arguments = _integration_audit_args


def _gate_args(p):
    p.add_argument("baseline")
    p.add_argument("current")


@command("gate-delta", "Compare baseline and current matching-gate outcomes without hiding pre-existing failures")
def gate_delta(args):
    def load(path):
        value = json.loads(Path(path).read_text())
        rows = value.get("results", value if isinstance(value, list) else [])
        return {str(row.get("name") or row.get("source") or i): bool(row.get("exact")) for i, row in enumerate(rows)}
    before, after = load(args.baseline), load(args.current)
    preexisting = sorted(key for key, ok in before.items() if not ok and not after.get(key, False))
    new = sorted(key for key, ok in after.items() if not ok and before.get(key, True))
    resolved = sorted(key for key, ok in after.items() if ok and not before.get(key, True))
    return {"pre_existing_failures": len(preexisting), "new_failures": len(new),
            "resolved_failures": len(resolved), "pre_existing": preexisting,
            "new": new, "resolved": resolved}


gate_delta.arguments = _gate_args


def _project_gate_args(p):
    p.add_argument("--project", default=".")
    p.add_argument("--hook", default="full_gate")
    p.add_argument("--machine", action="store_true")
    p.add_argument("--affected-unit")


@command("project-gate", "Run a configured project gate with compact per-image machine output")
def project_gate(args):
    project = Path(args.project).resolve(); hooks = _project_workbench_config(project).get("hooks", {})
    command_text = hooks.get(args.hook)
    if not command_text: raise ValueError(f"workbench.hooks.{args.hook} is not configured")
    run = subprocess.run(command_text, cwd=project, shell=True, text=True, env=_canonical_env(),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800)
    artifact_dir = project / ".n64dk" / "gate-artifacts"; artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / f"{args.hook}-{time.time_ns()}.log"; artifact.write_text(run.stdout)
    lines = run.stdout.splitlines(); failures = [line for line in lines if re.search(r"\b(FAIL|ERROR|DIFF)\b", line)]
    images = {}
    for line in lines:
        found = re.search(r"\b(resident|overlay\d+|loader)\b.*?(\d+)\s*/\s*(\d+)", line, re.I)
        if found: images[found.group(1).lower()] = {"exact": int(found.group(2)), "total": int(found.group(3))}
    affected = [line for line in lines if args.affected_unit and args.affected_unit in line]
    return {"ok": run.returncode == 0, "exit_code": run.returncode, "hook": args.hook,
            "images": images, "failures": failures[-100:], "affected_unit": affected[-20:],
            "line_count": len(lines), "full_output_artifact": str(artifact),
            "output_tail": None if args.machine else run.stdout[-16000:]}


project_gate.arguments = _project_gate_args


_DEFAULT_GATE_POLICY = {
    "claim": ["affected_gate"],
    "winner": ["affected_gate", "full_gate", "project_tests"],
    "integration": ["attack", "report", "progress", "affected_gate", "full_gate", "project_tests"],
    "boundary": ["attack", "report", "progress", "full_gate", "project_tests"],
}


def _resolved_gate_policy(project: Path, stage: str) -> dict:
    config = _project_workbench_config(project)
    configured = config.get("gate_policy", {})
    requested = configured.get(stage, _DEFAULT_GATE_POLICY.get(stage))
    if not isinstance(requested, list) or not requested:
        raise ValueError(f"workbench.gate_policy.{stage} must be a non-empty hook list")
    hooks = config.get("hooks", {})
    missing = [name for name in requested if not hooks.get(name)]
    payload = {"stage": stage, "required_hooks": requested,
               "commands": {name: hooks.get(name) for name in requested}, "missing": missing}
    payload["policy_digest"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def _gate_policy_args(p):
    p.add_argument("--project", default=".")
    p.add_argument("--stage", choices=tuple(_DEFAULT_GATE_POLICY), default="winner")
    p.add_argument("--run", action="store_true")
    p.add_argument("--affected-unit")


@command("gate-policy", "Inspect or execute the fail-closed gate policy for a state transition")
def gate_policy(args):
    project = Path(args.project).resolve(); policy = _resolved_gate_policy(project, args.stage)
    policy["ok"] = not policy["missing"]
    policy["executed"] = False; policy["results"] = []
    if not args.run:
        return policy
    if policy["missing"]:
        raise ValueError("gate policy is incomplete; missing hooks: " + ", ".join(policy["missing"]))
    artifact_root = project / ".n64dk" / "gate-artifacts" / policy["policy_digest"][:16]
    artifact_root.mkdir(parents=True, exist_ok=True)
    env = _canonical_env()
    if args.affected_unit:
        env["N64DK_AFFECTED_UNIT"] = args.affected_unit
    for hook in policy["required_hooks"]:
        command_text = policy["commands"][hook]
        run = subprocess.run(command_text, cwd=project, shell=True, text=True, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800)
        artifact = artifact_root / f"{len(policy['results']) + 1:02d}-{hook}.log"
        artifact.write_text(run.stdout)
        policy["results"].append({"hook": hook, "exit_code": run.returncode,
                                  "artifact": str(artifact), "tail": run.stdout[-4000:]})
        if run.returncode:
            policy["ok"] = False
            break
    policy["executed"] = True
    policy["passed_hooks"] = [row["hook"] for row in policy["results"] if row["exit_code"] == 0]
    return policy


gate_policy.arguments = _gate_policy_args


def _scheduler_args(p):
    p.add_argument("--project", default=".")
    p.add_argument("--jobs", type=int, help="compile workers (default: CPU count minus two, at least one)")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--limit", type=int, default=500)


@command("scheduler-plan", "Build persistent discovery, exploitation, family, recovery, boundary and verification lanes")
def scheduler_plan(args):
    project = Path(args.project).resolve(); inventory = build_inventory(project); rows = inventory["functions"]
    jobs = args.jobs if args.jobs is not None else max(1, (os.cpu_count() or 2) - 2)
    lane = {name: [] for name in ("discovery", "exploitation", "family-propagation", "recovery", "boundary-audit", "integration")}
    for row in rows:
        if row["state"] == "conflict" or row["ownership_state"] == "fragment": lane["boundary-audit"].append(row)
        elif row["state"] == "remaining" and row.get("candidate_source"): lane["exploitation"].append(row)
        elif row["state"] == "remaining": lane["discovery"].append(row)
    for values in lane.values(): values[:] = values[:args.limit]
    metrics = {"catalogue_considered": len(rows), "excluded_fragments": sum(x["ownership_state"] == "fragment" for x in rows),
               "assembly_owned": sum(x["ownership_state"].startswith("assembly") for x in rows),
               "aggregate_owned": sum(x["ownership_state"] == "manifest_exact_member" for x in rows),
               "usable_candidate_source": sum(bool(x.get("candidate_source")) for x in rows),
               "selected": sum(map(len, lane.values())), "compiled": 0, "unique_sources": 0,
               "unique_objects": 0, "exact": 0, "improved": 0, "unchanged": 0, "invalid_mappings": 0}
    plan = {"cpu": {"compile_workers": jobs, "reserved_orchestration": 1, "reserved_verification": 1},
            "gpu": {"enabled": args.gpu, "role": "bounded retrieval and ranking only"},
            "lanes": {key: [row["id"] for row in value] for key, value in lane.items()}, "metrics": metrics}
    plan_id = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:20]
    db = sqlite3.connect(project / ".n64dk" / "scheduler.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS plans (plan_id TEXT PRIMARY KEY,payload TEXT,created_ns INTEGER,updated_ns INTEGER)")
    now = time.time_ns(); db.execute("INSERT OR REPLACE INTO plans VALUES (?,?,?,?)",
                                    (plan_id, json.dumps(plan), now, now)); db.commit(); db.close()
    return {"plan_id": plan_id, **plan}


scheduler_plan.arguments = _scheduler_args


def _scheduler_status_args(p):
    p.add_argument("plan_id")
    p.add_argument("--project", default=".")


@command("scheduler-status", "Inspect persistent lane allocation and exact-match throughput metrics")
def scheduler_status(args):
    db = sqlite3.connect(Path(args.project).resolve() / ".n64dk" / "scheduler.sqlite")
    row = db.execute("SELECT payload,created_ns,updated_ns FROM plans WHERE plan_id=?", (args.plan_id,)).fetchone(); db.close()
    if not row: raise KeyError(f"unknown scheduler plan {args.plan_id}")
    return {"plan_id": args.plan_id, **json.loads(row[0]), "created_ns": row[1], "updated_ns": row[2]}


scheduler_status.arguments = _scheduler_status_args


def _scaffold_args(p):
    p.add_argument("function")
    p.add_argument("--project", default=".")
    p.add_argument("--image", required=True)
    p.add_argument("--image-name")
    p.add_argument("--output")


@command("source-scaffold", "Generate an immediately compilable ordinary-C starter from target evidence")
def source_scaffold(args):
    from ..match.reconstruct import scaffold
    inventory = build_inventory(args.project); rows = [row for row in inventory["functions"]
        if row["name"] == args.function and (not args.image_name or row["image"] == args.image_name)]
    if len(rows) != 1: raise ValueError(f"function resolved to {len(rows)} catalogue intervals")
    row = rows[0]; mapping = resolve_image_map(args.image, from_inventory(row)); data = Path(args.image).read_bytes()
    offset = mapping["selected"]["offset"]
    linked_vram = int(mapping.get("linked_vram") or row["vram"], 0)
    source, evidence = scaffold(
        row["name"], data[offset:offset + int(row["size"])], linked_vram)
    evidence["coordinates"] = {"catalogue_vram": row["vram"],
                               "linked_vram": f"0x{linked_vram:X}",
                               "selected_image_offset": f"0x{offset:X}"}
    output = Path(args.output).resolve() if args.output and args.output != "-" else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True); output.write_text(source)
        output.with_suffix(output.suffix + ".evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return {"source": source if not output else str(output), "output": "stdout" if args.output == "-" else str(output) if output else None,
            "evidence": evidence,
            "ordinary_c": True, "compile_ready": True}


source_scaffold.arguments = _scaffold_args


def _layout_infer_args(p):
    p.add_argument("--project", default=".")
    p.add_argument("--image", required=True)
    p.add_argument("--image-name", required=True)
    p.add_argument("--function", action="append", required=True)


@command("structure-layout-infer", "Aggregate typed structure-offset evidence across related functions")
def structure_layout_infer(args):
    from ..match.reconstruct import scaffold, shared_layout
    inventory = build_inventory(args.project); data = Path(args.image).read_bytes(); observations = []
    for name in args.function:
        rows = [row for row in inventory["functions"] if row["name"] == name and row["image"] == args.image_name]
        if len(rows) != 1: continue
        row = rows[0]; mapping = resolve_image_map(args.image, from_inventory(row)); offset = mapping["selected"]["offset"]
        _source, evidence = scaffold(name, data[offset:offset + int(row["size"])], int(row["vram"], 0))
        observations.append(evidence)
    return {"functions": [row["function"] for row in observations], **shared_layout(observations)}


structure_layout_infer.arguments = _layout_infer_args


def _prototype_solve_args(p):
    p.add_argument("--project", default=".")
    p.add_argument("--conflicts-only", action="store_true")


@command("prototype-solve", "Solve and report project-wide declaration constraints before matching")
def prototype_solve(args):
    from ..match.reconstruct import prototype_constraints
    result = prototype_constraints(Path(args.project).resolve())
    if args.conflicts_only: result.pop("results", None)
    return result


prototype_solve.arguments = _prototype_solve_args


def _tu_reconstruct_args(p):
    p.add_argument("unit")
    p.add_argument("--project", default=".")
    p.add_argument("--manifest-row")


@command("tu-reconstruct", "Prepare a first-class translation-unit reconstruction and whole-unit comparison plan")
def tu_reconstruct(args):
    project = Path(args.project).resolve(); unit = (project / args.unit).resolve(); text = unit.read_text(errors="replace")
    functions = re.findall(r"(?m)^\s*[A-Za-z_]\w*(?:\s*\*)?\s+(func_[0-9A-Fa-f]{8})\s*\(", text)
    declarations = re.findall(r"(?m)^\s*(?:extern\s+|static\s+)?[^;{}]+;", text)
    return {"unit": str(unit), "digest": hashlib.sha256(unit.read_bytes()).hexdigest(),
            "functions": functions, "declaration_order": declarations,
            "manifest_row": args.manifest_row, "dimensions": ["declaration order", "visibility",
                "typedefs", "macros", "inlining", "literal pools", "function order"],
            "comparison_scope": ["every owned function", "data sections", "relocations"],
            "apply": False}


tu_reconstruct.arguments = _tu_reconstruct_args


def _snapshot_args(p):
    p.add_argument("--project", default=".")
    p.add_argument("--function", required=True)
    p.add_argument("--image")
    p.add_argument("--database", default=".n64dk/results.sqlite")


@command("workbench-snapshot", "Emit one compact ownership, boundary, source, history and next-action snapshot")
def workbench_snapshot(args):
    project = Path(args.project).resolve(); inventory = build_inventory(project)
    rows = [row for row in inventory["functions"] if row["name"] == args.function]
    if len(rows) != 1: raise ValueError("function selector is ambiguous")
    git = subprocess.run(["git", "status", "--short"], cwd=project, text=True,
                         stdout=subprocess.PIPE, timeout=60)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project, text=True,
                          stdout=subprocess.PIPE, timeout=60)
    result = {"function": {k: rows[0].get(k) for k in ("name", "image", "vram", "size", "state", "ownership_state", "owner", "reason")},
              "git": {"head": head.stdout.strip(), "dirty_files": len(git.stdout.splitlines())}}
    if args.image:
        try: result["preflight"] = preflight(args.image, from_inventory(rows[0]), rows[0]["catalogue_status"])
        except ValueError as error: result["preflight"] = {"ok": False, "error": str(error)}
    database = Path(project, args.database) if not Path(args.database).is_absolute() else Path(args.database)
    if database.is_file():
        index = ResultIndex(database)
        try:
            history = index.query(function=args.function, limit=10)
            result["history"] = history; result["bracketed_target"] = index.brackets(args.function)
        finally: index.close()
    result["next_commands"] = [f"n64dk code-query --project {project} --symbol {args.function}",
                               f"n64dk match-inventory --project {project} --explain {args.function}",
                               f"n64dk result-query --project {project} --function {args.function}"]
    return result


workbench_snapshot.arguments = _snapshot_args


def _lifetime_args(p):
    p.add_argument("session_id")
    p.add_argument("--project", default=".")
    p.add_argument("--state-root")
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--variants-per-round", type=int, default=16)
    p.add_argument("--summary-only", action="store_true", help="omit frontier and per-round detail")
    p.add_argument("--best-only", action="store_true", help="include only the compact session best")


@command("register-lifetime-explore", "Explore bounded carrier, scope, declaration and return-lifetime hypotheses")
def register_lifetime_explore(args):
    from ..match.explore import explore
    from .match import _session_evaluator
    store = SessionStore(resolve_store(args.session_id, args.project, args.state_root))
    try:
        session = MatchSession(store, _session_evaluator(store.context(args.session_id)))
        result = explore(store, session, args.session_id, args.jobs, args.rounds,
                         args.variants_per_round, True)
        result["allocation_hypotheses"] = [
            {"name": "formal reuse", "expected": "preserve or destructively reuse the incoming carrier"},
            {"name": "named result carrier", "expected": "extend the returned value live range"},
            {"name": "scope shortening", "expected": "remove interference before the residual site"},
            {"name": "joined return", "expected": "force convergence on one return register"},
        ]
        if args.best_only:
            result["best"] = store.best(args.session_id).summary(False)
        if args.summary_only or args.best_only:
            result.pop("frontier", None)
            result.pop("round_reports", None)
        return result
    finally: store.close()


register_lifetime_explore.arguments = _lifetime_args


def _control_args(p):
    p.add_argument("session_id")
    p.add_argument("--project", default=".")
    p.add_argument("--state-root")
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--budget", type=int, default=32)
    p.add_argument("--stop-on-exact", action="store_true")


@command("control-flow-explore", "Classify branch targets and test bounded loop/exit source forms")
def control_flow_explore(args):
    from ..match.controlflow import classify_branches, variants
    from .match import _session_evaluator
    store = SessionStore(resolve_store(args.session_id, args.project, args.state_root))
    try:
        parent = store.best(args.session_id); context = store.context(args.session_id)
        source = store.cache.get(parent.source_digest).decode()
        branch_evidence = classify_branches(context, parent.outcome)
        generated = variants(source, args.budget)
        if not generated:
            return {"session_id": args.session_id, "branch_targets": branch_evidence,
                    "variants": 0, "reason": "no bounded simple control-flow region found"}
        transaction = MatchSession(store, _session_evaluator(context)).try_proposal(
            args.session_id, {"parent": parent.candidate_id,
                              "hypothesis": "branch target shape selects loop exit or shared cleanup",
                              "variants": generated}, args.jobs, True)
        original_offsets = {row.get("target_offset", row.get("offset"))
                            for row in parent.outcome.get("diff", {}).get("entries", [])}
        for result in transaction["transaction_results"]:
            child = store.get(args.session_id, result["candidate_id"])
            child_offsets = {row.get("target_offset", row.get("offset"))
                             for row in child.outcome.get("diff", {}).get("entries", [])} if child else set()
            result["cfg_scope"] = "preserved" if child_offsets <= original_offsets else "unrelated-block-change"
            result["eligible_frontier"] = result["cfg_scope"] == "preserved"
            if not result["eligible_frontier"]:
                store.db.execute("UPDATE session_candidates SET status = 'rejected-cfg' "
                                 "WHERE session_id = ? AND candidate_id = ?",
                                 (args.session_id, result["candidate_id"]))
        store.db.commit()
        accepted = [row for row in transaction["transaction_results"] if row["eligible_frontier"]]
        transaction["transaction_best"] = min(
            accepted,
            key=lambda row: row.get("score") if row.get("score") is not None else 1 << 30,
            default=None,
        )
        transaction["rejected_unrelated_cfg"] = sum(not row["eligible_frontier"]
                                                     for row in transaction["transaction_results"])
        transaction["branch_targets"] = branch_evidence
        transaction["forms_tested"] = [row["form"] for row in generated]
        return transaction
    finally: store.close()


control_flow_explore.arguments = _control_args


def _clone_args(p):
    p.add_argument("--source-session", required=True)
    p.add_argument("--source-candidate")
    p.add_argument("--target-session", required=True)
    p.add_argument("--project", default=".")
    p.add_argument("--jobs", type=int, default=8)


@command("clone-family-transfer", "Align a reconstructed family member and test a renamed sibling candidate")
def clone_family_transfer(args):
    import difflib
    source_store = SessionStore(resolve_store(args.source_session, args.project))
    target_store = SessionStore(resolve_store(args.target_session, args.project))
    try:
        source_row = source_store.get(args.source_session, args.source_candidate) if args.source_candidate else source_store.best(args.source_session)
        target_row = target_store.best(args.target_session)
        source = source_store.cache.get(source_row.source_digest).decode()
        target = target_store.cache.get(target_row.source_digest).decode()
        source_name = re.search(r"\b(func_[0-9A-Fa-f]{8})\s*\(", source)
        target_name = re.search(r"\b(func_[0-9A-Fa-f]{8})\s*\(", target)
        if not source_name or not target_name: raise ValueError("family sources need identifiable function names")
        candidate = source.replace(source_name.group(1), target_name.group(1))
        similarity = difflib.SequenceMatcher(None, source.splitlines(), target.splitlines()).ratio()
        from .match import _session_evaluator
        result = MatchSession(target_store, _session_evaluator(target_store.context(args.target_session))).try_proposal(
            args.target_session, {"parent": target_row.candidate_id,
                                  "hypothesis": "clone family shares CFG and differs only in case body",
                                  "variants": [{"label": "renamed-family-transfer", "source": candidate}]},
            args.jobs, False)
        result["family_alignment"] = {"source_function": source_name.group(1),
                                      "target_function": target_name.group(1),
                                      "line_similarity": round(similarity, 4),
                                      "method": "function rename plus target recompilation"}
        return result
    finally:
        source_store.close(); target_store.close()


clone_family_transfer.arguments = _clone_args
