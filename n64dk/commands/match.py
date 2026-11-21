"""Matching commands: one-shot relocated compare / configuration sweep, manifest batches and
the permuter driver."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

from ..cli import command
from ..match import manifest as manmod
from ..match import symbols as symmod
from ..match.engine import Target, batch, default_pipelines, manifest_row, match_source, sweep
from ..match.pipeline import (PIPELINES, BuildSpec, dependency_digest, split_flags,
                              scratch_dir, toolchain_digest)
from ..match.session import MatchSession, SessionStore
from ..match.registry import SessionRegistry, resolve_store, standard_store
from ..match.layout import (Layout, configured_image_path, from_inventory, preflight,
                            resolve_image_map)
from ..match.toolchain import Config, PROFILES, load_config
from ..match import inventory as inventorymod
from ..rom import Rom


def _sqlite_retry(action, attempts: int = 5, delay: float = 0.1):
    """Retry transient SQLite contention and return a visible warning on persistent failure."""
    import time
    for attempt in range(attempts):
        try:
            action()
            return None
        except sqlite3.OperationalError as error:
            transient = any(word in str(error).lower() for word in ("locked", "busy"))
            if transient and attempt + 1 < attempts:
                time.sleep(delay * (2 ** attempt))
                continue
            suffix = f" (after {attempt + 1} attempts)" if transient else ""
            return f"history not recorded: {error}{suffix}"
        except (sqlite3.Error, OSError) as error:
            return f"history not recorded: {type(error).__name__}: {error}"
    return "history not recorded"


def _exact_project_evidence(cfg, args, outcome, source, spec, target):
    if not outcome.exact or outcome.vram is None:
        return {}
    try:
        inventory = inventorymod.build(cfg.project)
        catalogue_vram = getattr(args, "_catalogue_addr", outcome.vram)
        rows = [row for row in inventory["functions"] if int(row["vram"], 0) == catalogue_vram
                and int(row["size"]) == outcome.size]
    except Exception:
        rows = []
    selected = None
    for row in rows:
        try:
            check = preflight(cfg.rom, from_inventory(row), row.get("catalogue_status", ""))
        except Exception:
            continue
        if int(check["selected_image_offset"], 0) == outcome.vram - target.delta:
            selected = (row, check); break
    if selected is None and len(rows) == 1:
        row = rows[0]; selected = (row, None)
    if selected is None:
        return {"exact_evidence": {"target_text": outcome.result.text.match if outcome.result else True,
                                   "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                                   "generated_text_sha256": outcome.as_dict().get("sha256_text")}}
    row, check = selected; layout = from_inventory(row)
    target_digest = outcome.as_dict().get("sha256_text")
    relative = str(Path(source).resolve().relative_to(cfg.project)) if Path(source).resolve().is_relative_to(cfg.project) else str(source)
    project_row = "\t".join((f"0x{layout.rom:X}", f"0x{outcome.size:X}", relative,
                             spec.manifest_flags(), target_digest or ""))
    profile_key = {"ido5.3": "ido53", "ido7.1": "ido71", "sn": "sn"}.get(spec.pipeline, "kmc")
    compiler_root = cfg.toolchain(profile_key); compiler_hasher = hashlib.sha256()
    if compiler_root:
        for relative_name in PROFILES[profile_key].required:
            compiler_hasher.update(relative_name.encode()); compiler_hasher.update(hashlib.sha256((compiler_root / relative_name).read_bytes()).digest())
    compiler_digest = compiler_hasher.hexdigest() if compiler_root else None
    catalogue_vram = getattr(args, "_catalogue_addr", outcome.vram)
    return {"image_mapping": {"raw_rom": f"0x{layout.rom:X}",
                              "composite_rom": check.get("selected_image_offset") if check else None,
                              "linked_vram": f"0x{outcome.vram:08X}",
                              "catalogue_vram": f"0x{catalogue_vram:08X}", "image": row["image"]},
            "project_manifest_row": project_row,
            "exact_evidence": {"target_text_sha256": target_digest,
                               "complete_owned_sections": all(section.match for section in outcome.result.sections) if outcome.result else True,
                               "expected_manifest_hash": target_digest,
                               "compiler_binary_digest": compiler_digest,
                               "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                               "relocations_checked": bool(outcome.result and outcome.result.sections),
                               "mapping_preflight": check}}


def _replay_record(cfg, args, outcome, source: Path, spec: BuildSpec, target: Target) -> dict:
    profile_key = {"ido5.3": "ido53", "ido7.1": "ido71", "sn": "sn"}.get(spec.pipeline, "kmc")
    root = cfg.toolchain(profile_key); binaries = {}
    if root:
        for name in PROFILES[profile_key].required:
            path = root / name
            if path.is_file(): binaries[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    allowed_environment = {key: os.environ[key] for key in ("LANG", "LC_ALL", "TZ", "TMPDIR", "WIBO")
                           if key in os.environ}
    return {"version": "n64dk-replay-v1", "source": str(source),
            "source_digest": hashlib.sha256(source.read_bytes()).hexdigest(),
            "toolchain_binaries": binaries, "pipeline": spec.pipeline, "flags": list(spec.flags),
            "include_paths": [str(path) for path in cfg.include],
            "symbol_files": [str(path) for path in (args.symbols or cfg.symbols)],
            "image": str(Path(args.rom).resolve()), "image_digest": hashlib.sha256(target.rom).hexdigest(),
            "address": outcome.vram, "size": outcome.size, "vram_delta": target.delta,
            "target_digest": hashlib.sha256(target.rom[outcome.vram - target.delta:
                outcome.vram - target.delta + outcome.size]).hexdigest(),
            "environment": allowed_environment}


def _common(p, rom_required: bool):
    p.add_argument("--rom", help="physical retail/composite image (may be omitted when --image-name "
                   "selects a configured [images] entry)" + ("" if rom_required else
                   " (otherwise defaults to n64dk.toml [match] rom or baserom*.z64)"))
    p.add_argument("--project", help="project directory: working directory for builds, "
                   "default include/, .toolchain/ and symbol files (default: current directory)")
    p.add_argument("--config", help="n64dk.toml (default: <project>/n64dk.toml)")
    p.add_argument("--toolchain-dir", help="directory holding kmc-gcc-2.7.2/, ido5.3/, ... ")
    p.add_argument("--symbols", action="append", default=[],
                   help="symbol file: splat symbol_addrs.txt or catalogue TOML (repeatable)")
    p.add_argument("-I", "--include", action="append", default=[], help="include directory (repeatable)")
    p.add_argument("--normalizer", help="assembly normalizer hook, e.g. 'python3 tools/norm.py {input} {output}'")
    p.add_argument("--vram-delta", type=lambda s: int(s, 0), help="VRAM - ROM offset of the code segment")
    p.add_argument("--image-name", help="logical image identity for overlapping overlay address spaces")
    p.add_argument("--jobs", type=int, default=None, help="parallel builds (default: CPU count)")


def _mapped_comparison_addr(resolved: dict, catalogue_addr: int, catalogue_shift: int = 0) -> int:
    comparison = int(resolved["linked_vram"], 0) if resolved.get("linked_vram") else catalogue_addr
    if catalogue_shift and resolved.get("selected", {}).get("authority") == "raw-overlay":
        comparison = catalogue_addr - catalogue_shift
        resolved["linked_vram"] = f"0x{comparison:X}"
    return comparison


def _setup(args):
    cfg = load_config(args.project, args.config, args.toolchain_dir)
    if args.normalizer:
        cfg.normalizer = args.normalizer
    if args.include:
        cfg.include = list(dict.fromkeys([Path(cfg.project, i).resolve() for i in args.include] + cfg.include))
    rom_path = args.rom
    if rom_path is None and getattr(args, "image_name", None):
        rom_path = configured_image_path(cfg.project, args.image_name)
    if rom_path is None:
        rom_path = cfg.find_rom()
    if rom_path is None:
        raise SystemExit("no ROM: pass --rom or set [match] rom in n64dk.toml")
    image = Rom.load(rom_path)
    cfg.rom = Path(rom_path).expanduser().resolve()
    # Downstream commands must see the concrete immutable image, not the optional spelling
    # supplied by the caller.  This also makes replay/session identity identical whether the
    # image was selected by --rom or by a logical [images] entry.
    args.rom = str(cfg.rom)
    if image.checksum_ok() is False:
        raise SystemExit(f"{rom_path}: header checksum mismatch; not a clean base image")
    selected_delta = args.vram_delta if args.vram_delta is not None else cfg.vram_delta
    if getattr(args, "addr", None) is not None and getattr(args, "size", None) is not None:
        inventory = inventorymod.build(cfg.project)
        rows = [row for row in inventory["functions"] if int(row["vram"], 0) == args.addr
                and int(row["size"]) == args.size]
        if args.image_name:
            known = {row["image"] for row in inventory["functions"]}
            if known and args.image_name not in known:
                raise SystemExit(f"--image-name {args.image_name!r} is not an inventory image "
                                 f"(images: {', '.join(sorted(known))})")
            rows = [row for row in rows if row["image"] == args.image_name]
            if not rows and known:
                raise SystemExit(f"no catalogue function at 0x{args.addr:08X} size 0x{args.size:X} "
                                 f"in image {args.image_name!r}")
        if len(rows) > 1:
            choices = "; ".join(f"{row['image']}:{row['name']} (rom {row.get('rom')})" for row in rows)
            raise SystemExit(f"0x{args.addr:08X} size 0x{args.size:X} exists in {len(rows)} images: "
                             f"{choices}; pass --image-name")
        if len(rows) == 1:
            resolved = resolve_image_map(rom_path, from_inventory(rows[0]))
            comparison_addr = _mapped_comparison_addr(
                resolved, args.addr, int(getattr(args, "catalogue_shift", 0) or 0))
            # A raw overlay contains linked bytes, while its catalogue may deliberately retain
            # historical/survey addresses.  The explicit catalogue shift is independent mapping
            # evidence and must be applied before validating the raw image delta.
            canonical_delta = comparison_addr - resolved["selected"]["offset"]
            if args.vram_delta is not None and args.vram_delta != canonical_delta:
                raise ValueError(f"explicit vram_delta 0x{args.vram_delta:X} conflicts with canonical "
                                 f"image mapping 0x{canonical_delta:X}")
            selected_delta = canonical_delta
            args._resolved_mapping = resolved
            args._resolved_inventory_row = rows[0]
            args._catalogue_addr = args.addr
            args._comparison_addr = comparison_addr
    target = Target.from_rom(image, selected_delta)
    sym_files = [Path(s) for s in args.symbols] or cfg.symbols or symmod.discover(cfg.project)
    syms = symmod.load(sym_files)
    # real names (sin, cos, ...) must resolve to one address and agree with the code's identity
    # where an SDK signature named it (NAMES.tsv beside the project): a swapped pair is refused,
    # because a relocated compare cannot tell two same-shaped library functions apart
    names_tsv = Path(cfg.project) / "NAMES.tsv"
    identities = symmod.identities_from_names(names_tsv.read_text()) if names_tsv.is_file() else {}
    try:
        args._name_findings = symmod.require_consistent_names(syms, identities)
    except symmod.NameError_ as error:
        raise symmod.NameError_(f"symbol names: {error}") from None
    resolved = getattr(args, "_resolved_mapping", None)
    row = getattr(args, "_resolved_inventory_row", None)
    if resolved and row and resolved.get("linked_base"):
        catalogue_base = int(row["section_vram"], 0)
        linked_base = int(resolved["linked_base"], 0)
        shift = catalogue_base - linked_base
        section_end = catalogue_base + int(row["section_size"])
        if shift:
            adjusted = syms.copy()
            for name, address in list(adjusted.items()):
                if catalogue_base <= address < section_end:
                    adjusted[name] = address - shift
            syms = adjusted
            args._coordinate_transform = {"catalogue_base": f"0x{catalogue_base:X}",
                                          "linked_base": f"0x{linked_base:X}",
                                          "catalogue_to_linked": f"-{shift:#x}"}
    return cfg, target, syms, sym_files


def _spec(args, cfg) -> BuildSpec:
    pipe = args.pipeline
    if pipe is None:
        avail = default_pipelines(cfg)
        pipe = avail[0] if avail else "kmc+builtin"
    flags = shlex.split(args.flags) if args.flags is not None else None
    if flags is None:
        key = {"ido5.3": "ido53", "ido7.1": "ido71", "sn": "sn"}.get(pipe, "kmc")
        flags = list(PROFILES[key].default_flags)
    return split_flags(flags, pipe)


def _match_args(p):
    p.add_argument("source", help="C (or assembly) source file")
    _common(p, rom_required=True)
    p.add_argument("--addr", type=lambda s: int(s, 16),
                   help="VRAM of the unit's .text (hex; default: search the ROM for the compiled code)")
    p.add_argument("--size", type=lambda s: int(s, 16), help="text size in bytes (hex; default: compiled size)")
    p.add_argument("--pipeline", choices=PIPELINES, help="build pipeline (default: first available)")
    p.add_argument("--flags", help="compiler flags (default: the toolchain profile's standard flags)")
    p.add_argument("--sweep", action="store_true",
                   help="search pipelines x flags x rodata placement for the first exact configuration")
    p.add_argument("--pipelines", help="comma-separated pipelines for --sweep (default: all available)")
    p.add_argument("--all", action="store_true", help="with --sweep: do not stop at the first exact result")
    p.add_argument("--extended-flags", action="store_true",
                   help="with --sweep: add bounded GCC CPU and optimization-pass discriminators")
    p.add_argument("--no-infer", action="store_true",
                   help="do not recover unresolved external symbols from the retail code")
    p.add_argument("--lines", type=int, default=24, help="diff lines shown in text output")


@command("match", "Build a source with an original toolchain and compare it relocated against the ROM")
def match(args):
    cfg, target, syms, _files = _setup(args)
    source = Path(args.source).resolve()
    if args.sweep:
        pipes = args.pipelines.split(",") if args.pipelines else None
        base = split_flags(shlex.split(args.flags), "kmc+builtin") if args.flags else None
        comparison_addr = getattr(args, "_comparison_addr", args.addr)
        res = sweep(source, cfg, target, comparison_addr, args.size, syms, pipes, args.jobs,
                    stop_at_first=not args.all, base_spec=base, extended_flags=args.extended_flags)
        out = {"source": str(source), "vram": f"0x{(res.exact.vram if res.exact else comparison_addr or 0):08X}",
               "catalogue_vram": f"0x{args.addr:08X}" if args.addr is not None else None,
               "tried": len(res.tried),
               "pipelines": pipes or default_pipelines(cfg), "exact": res.exact is not None}
        if res.exact is not None:
            o = res.exact
            out.update({"pipeline": o.spec.pipeline, "flags": " ".join(o.spec.flags),
                        "bases": {k: f"0x{v:08X}" for k, v in o.result.bases.items()},
                        "manifest_row": manifest_row(o, cfg.project, source, target, syms)})
            if args.all:
                out["all_exact"] = [f"{x.spec.pipeline} {' '.join(x.spec.flags)}" for x in res.tried if x.exact]
        else:
            best = res.best()
            if best is not None:
                out.update({"best_pipeline": best.spec.pipeline, "best_flags": " ".join(best.spec.flags),
                            "best_diff": best.result.diff.summary() if best.result.diff else best.error})
        if args.format == "json":
            out["results"] = [{"pipeline": o.spec.pipeline, "flags": " ".join(o.spec.flags), "exact": o.exact,
                               "score": (o.result.diff.score if o.result and o.result.diff else
                                         (0 if o.exact else None)), "error": o.error} for o in res.tried]
        return out
    spec = _spec(args, cfg)
    with scratch_dir() as tmp:
        comparison_addr = getattr(args, "_comparison_addr", args.addr)
        o = match_source(source, spec, cfg, target, comparison_addr, args.size, syms, Path(tmp),
                         infer_externals=not args.no_infer,
                         strict_verification=args.addr is not None and args.size is not None)
    payload = o.as_dict()
    if getattr(args, "_catalogue_addr", None) is not None:
        payload["catalogue_vram"] = f"0x{args._catalogue_addr:08X}"
        payload["linked_vram"] = f"0x{o.vram:08X}"
        payload["coordinate_transform"] = getattr(args, "_coordinate_transform", None)
    payload["replay"] = _replay_record(cfg, args, o, source, spec, target)
    if o.exact:
        payload["manifest_row"] = manifest_row(o, cfg.project, source, target, syms)
        payload.update(_exact_project_evidence(cfg, args, o, source, spec, target))
    # Direct observations enter the same durable history as sessions immediately.
    def record_history():
        from ..match.results import ResultIndex
        address, size = o.vram, o.size
        target_offset = address - target.delta
        target_digest = (hashlib.sha256(target.rom[target_offset:target_offset + size]).hexdigest()
                         if 0 <= target_offset <= len(target.rom) - size else None)
        history = ResultIndex(cfg.project / ".n64dk" / "results.sqlite")
        try:
            history.record_direct(payload, {"rom_sha256": hashlib.sha256(target.rom).hexdigest(),
                "target_text_sha256": target_digest, "vram_delta": target.delta,
                "addr": address, "size": size}, source,
                source.stem if re.fullmatch(r"func_[0-9A-Fa-f]{8}", source.stem) else None,
                "valid_current_target" if target_digest else "unknown_provenance")
        finally: history.close()
    warning = _sqlite_retry(record_history)
    if warning is None:
        payload["history"] = "recorded"
    else:
        payload["history"] = warning
    if args.format == "json":
        return payload
    lines = [f"{'MATCH' if o.exact else 'DIFF'} {source.name} @ 0x{o.vram:08X} "
             f"[{spec.pipeline} {' '.join(spec.flags)}]"]
    if o.error:
        lines.append(f"error: {o.error}")
    if o.result:
        r = o.result
        lines.append(f".text 0x{r.text.size:X} bytes (compiled 0x{r.compiled_size:X})"
                     + (f", first difference at +0x{r.text.first_diff:X}" if r.text.first_diff is not None else ""))
        for s in r.sections:
            lines.append(f"{s.name} @ 0x{s.vram:08X} 0x{s.size:X} bytes: {'match' if s.match else 'DIFF'}")
        if r.inferred_symbols:
            lines.append("symbols recovered from the ROM: " +
                         ", ".join(f"{k}=0x{v:08X}" for k, v in sorted(r.inferred_symbols.items())))
        if r.error:
            lines.append(r.error)
        if r.diff:
            lines.append("diff: " + r.diff.summary())
            lines += r.diff.lines(args.lines)
        if o.exact:
            lines.append("manifest row: " + manifest_row(o, cfg.project, source, target, syms))
    return "\n".join(lines)


match.arguments = _match_args


def _session_context(args, cfg, target, sym_files, source: Path, spec: BuildSpec,
                     source_bytes: bytes) -> dict:
    rom_path = Path(args.rom).expanduser().resolve()
    toolchain_hasher = hashlib.sha256()
    for key, root in sorted(cfg.toolchains.items()):
        toolchain_hasher.update(key.encode())
        if root.is_dir():
            for item in sorted(path for path in root.iterdir() if path.is_file()):
                toolchain_hasher.update(item.name.encode()); toolchain_hasher.update(hashlib.sha256(item.read_bytes()).digest())
    if cfg.toolchain_root:
        toolchain_hasher.update(str(cfg.toolchain_root.resolve()).encode())
    profile_key = {"ido5.3": "ido53", "ido7.1": "ido71", "sn": "sn"}.get(spec.pipeline, "kmc")
    profile_root = cfg.toolchain(profile_key)
    if profile_root:
        for relative in PROFILES[profile_key].required:
            binary = profile_root / relative
            toolchain_hasher.update(relative.encode()); toolchain_hasher.update(hashlib.sha256(binary.read_bytes()).digest())
    context = {
        "source_name": source.name,
        "source_path": str(source.resolve()),
        "source_suffix": source.suffix or ".c",
        "rom": str(rom_path),
        "rom_sha256": hashlib.sha256(target.rom).hexdigest(),
        "project": str(cfg.project),
        "toolchain_root": str(cfg.toolchain_root) if cfg.toolchain_root else None,
        "toolchains": {key: str(value) for key, value in cfg.toolchains.items()},
        "include": [str(value) for value in cfg.include],
        "symbols": [str(Path(value).resolve()) for value in sym_files],
        "normalizer": cfg.normalizer,
        "vram_delta": target.delta,
        "addr": getattr(args, "_comparison_addr", args.addr),
        "catalogue_addr": getattr(args, "_catalogue_addr", args.addr),
        "size": args.size,
        "infer_externals": not args.no_infer,
        "default_pipeline": spec.pipeline,
        "default_flags": list(spec.flags),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "compiler_recipe": {"pipeline": spec.pipeline, "flags": list(spec.flags)},
        "toolchain_digest": toolchain_hasher.hexdigest(),
        "pipeline_toolchain_digest": toolchain_digest(cfg, spec.pipeline),
        "include_closure_digest": dependency_digest(source, spec, cfg),
    }
    if args.addr is not None and args.size is not None:
        layout = (from_inventory(args._resolved_inventory_row)
                  if getattr(args, "_resolved_inventory_row", None) else
                  Layout("standalone", 0, target.delta, len(target.rom), args.addr, args.size,
                         "standalone-explicit-range"))
        context["layout_preflight"] = preflight(rom_path, layout, strict_entry=args.strict_entry)
        rom_offset = context["addr"] - target.delta
        target_bytes = target.rom[rom_offset:rom_offset + args.size]
        context["target_text_sha256"] = hashlib.sha256(target_bytes).hexdigest()
        context["image_identity"] = rom_path.name
        context["catalogue_boundary_revision"] = None
        resolved_row = getattr(args, "_resolved_inventory_row", None)
        resolved_mapping = getattr(args, "_resolved_mapping", None)
        if resolved_row and resolved_mapping:
            context.update({"catalogue_base": int(resolved_row["section_vram"], 0),
                            "linked_base": int(resolved_mapping.get("linked_base") or
                                               resolved_row["section_vram"], 0),
                            "section_size": int(resolved_row["section_size"]),
                            "inventory_image": resolved_row["image"]})
    return context


def _session_runtime(context: dict):
    rom_path = Path(context["rom"])
    image = Rom.load(rom_path)
    if hashlib.sha256(image.data).hexdigest() != context["rom_sha256"]:
        raise RuntimeError("session ROM changed; start a new matching session")
    cfg = Config(
        project=Path(context["project"]),
        toolchain_root=(Path(context["toolchain_root"])
                        if context.get("toolchain_root") else None),
        toolchains={key: Path(value) for key, value in context.get("toolchains", {}).items()},
        include=[Path(value) for value in context.get("include", [])],
        symbols=[Path(value) for value in context.get("symbols", [])],
        normalizer=context.get("normalizer"),
        vram_delta=context["vram_delta"],
        rom=rom_path,
    )
    source_path = Path(context.get("source_path", ""))
    if source_path.is_file():
        spec = BuildSpec(context["default_pipeline"], tuple(context.get("default_flags", ())))
        current_dependencies = dependency_digest(source_path, spec, cfg)
        if current_dependencies != context.get("include_closure_digest"):
            raise RuntimeError("session include closure changed; start a new matching session")
    current_toolchain = toolchain_digest(cfg, context["default_pipeline"])
    if current_toolchain != context.get("pipeline_toolchain_digest"):
        raise RuntimeError("session compiler toolchain changed; start a new matching session")
    target = Target.from_rom(image, context["vram_delta"])
    syms = symmod.load([Path(value) for value in context.get("symbols", [])])
    catalogue_base = context.get("catalogue_base")
    linked_base = context.get("linked_base")
    section_size = context.get("section_size")
    if catalogue_base is not None and linked_base is not None and section_size is not None:
        shift = int(catalogue_base) - int(linked_base)
        if shift:
            adjusted = syms.copy()
            for name, address in list(adjusted.items()):
                if int(catalogue_base) <= address < int(catalogue_base) + int(section_size):
                    adjusted[name] = address - shift
            syms = adjusted
    return cfg, target, syms


def _session_evaluator(context: dict, strict: bool = True):
    cfg, target, syms = _session_runtime(context)

    def evaluate(source: bytes, pipeline: str, flags: tuple[str, ...], label: str):
        spec = split_flags(list(flags), pipeline)
        with tempfile.TemporaryDirectory(prefix="n64dk-session-") as name:
            root = Path(name)
            source_path = root / ("candidate" + context.get("source_suffix", ".c"))
            source_path.write_bytes(source)
            workdir = root / "build"
            workdir.mkdir()
            outcome = match_source(
                source_path, spec, cfg, target, context.get("addr"), context.get("size"),
                syms, workdir, infer_externals=context.get("infer_externals", True),
                name=label, strict_verification=strict,
            )
            artifact_dir = context.get("artifact_dir")
            if artifact_dir:
                destination = Path(artifact_dir) / re.sub(r"[^A-Za-z0-9_.-]", "_", label)
                destination.mkdir(parents=True, exist_ok=True)
                for artifact in workdir.rglob("*"):
                    if artifact.is_file() and artifact.suffix not in {".o"}:
                        shutil.copy2(artifact, destination / artifact.name)
            return outcome
    return evaluate


def _session_start_args(p):
    p.add_argument("source", help="baseline C/assembly source file, or - for stdin")
    p.add_argument("--source-name", default="stdin.c",
                   help="logical filename when source is stdin (default: stdin.c)")
    _common(p, rom_required=True)
    p.add_argument("--addr", type=lambda value: int(value, 16),
                   help="VRAM of the unit's .text (hex)")
    p.add_argument("--size", type=lambda value: int(value, 16), help="text size in bytes (hex)")
    p.add_argument("--pipeline", choices=PIPELINES)
    p.add_argument("--flags")
    p.add_argument("--no-infer", action="store_true")
    p.add_argument("--state-root", help="session store (default: <project>/.n64dk/match-sessions)")
    p.add_argument("--details", action="store_true", help="include the full instruction diff")
    p.add_argument("--strict-entry", action="store_true",
                   help="reject likely internal fragments using entry and delay-slot evidence")
    _session_output_args(p)


def _session_output_args(p):
    group = p.add_mutually_exclusive_group()
    group.add_argument("--summary-only", action="store_true", help="omit candidate collections")
    group.add_argument("--best-only", action="store_true", help="return only the best candidate")
    group.add_argument("--best-details", action="store_true", help="return only the best candidate with its outcome")
    group.add_argument("--candidate", help="return one candidate ID with full details")
    group.add_argument("--exact-only", action="store_true", help="return exact candidates only")


def _project_session_output(value: dict, args, store: SessionStore | None = None) -> dict:
    if getattr(args, "best_only", False) or getattr(args, "best_details", False):
        return {"session_id": value["session_id"], "best": value.get("best") or value.get("baseline")}
    candidate_id = getattr(args, "candidate", None)
    if candidate_id:
        if store is None:
            raise ValueError("candidate selection is unavailable")
        candidate = store.get(value["session_id"], candidate_id)
        if candidate is None:
            raise ValueError(f"unknown candidate {candidate_id}")
        return {"session_id": value["session_id"], "candidate": candidate.summary(True)}
    if getattr(args, "summary_only", False):
        return {key: item for key, item in value.items()
                if key not in {"frontier", "results", "candidates", "context", "experiments"}}
    if getattr(args, "exact_only", False):
        result = dict(value)
        for key in ("frontier", "results", "candidates"):
            if key in result:
                result[key] = [row for row in result[key] if row.get("exact")]
        return result
    return value


@command("match-session-start", "Start a persistent standalone matching session")
def match_session_start(args):
    cfg, target, syms, sym_files = _setup(args)
    source = Path(args.source).resolve() if args.source != "-" else Path(args.source_name)
    source_bytes = source.read_bytes() if args.source != "-" else sys.stdin.buffer.read()
    spec = _spec(args, cfg)
    context = _session_context(args, cfg, target, sym_files, source, spec, source_bytes)
    store_root = Path(args.state_root).resolve() if args.state_root else standard_store(cfg.project)
    store = SessionStore(store_root)
    try:
        session = MatchSession(store, _session_evaluator(context))
        session_id, baseline = session.create(context, source_bytes,
                                              spec.pipeline, tuple(spec.flags))
        registry = SessionRegistry(cfg.project)
        try:
            registry.register(session_id, store_root, context, baseline.source_digest, "standalone")
        finally:
            registry.close()
        result = {
            "session_id": session_id,
            "state_root": str(store_root),
            "baseline": baseline.summary(args.details or args.best_details),
            "frontier": [baseline.summary(args.details or args.best_details)] if baseline.status == "evaluated" else [],
        }
        return _project_session_output(result, args, store)
    finally:
        store.close()


match_session_start.arguments = _session_start_args


def _read_proposal(value: str) -> dict:
    if value == "-":
        return json.load(sys.stdin)
    if value.lstrip().startswith("{"):
        return json.loads(value)
    return json.loads(Path(value).read_text())


def _session_try_args(p):
    p.add_argument("session_id")
    p.add_argument("--state-root")
    p.add_argument("--project", default=".")
    p.add_argument("--proposal", default="-",
                   help="JSON object, JSON file, or - for stdin (default)")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--details", action="store_true", help="include full outcome and diff entries")
    p.add_argument("--transaction-results", action="store_true")
    p.add_argument("--transaction-best", action="store_true")
    p.add_argument("--session-best", action="store_true")
    p.add_argument("--changed-only", action="store_true")
    p.add_argument("--object-classes", action="store_true")
    p.add_argument("--why-not-better", action="store_true")
    _session_output_args(p)


@command("match-session-try", "Apply in-memory edits and evaluate source/flag variants as one transaction")
def match_session_try(args):
    store = SessionStore(resolve_store(args.session_id, args.project, args.state_root))
    try:
        context = store.context(args.session_id)
        session = MatchSession(store, _session_evaluator(context))
        value = session.try_proposal(args.session_id, _read_proposal(args.proposal),
                                     args.jobs, args.details or args.best_details)
        selected = [name for name in ("transaction_results", "transaction_best", "session_best",
                    "object_classes", "why_not_better") if getattr(args, name, False)]
        if args.changed_only:
            value["transaction_results"] = [row for row in value["transaction_results"]
                                             if not row.get("cache_hit") and
                                             (row.get("delta", {}).get("score") or
                                              row.get("delta", {}).get("counts"))]
        if selected:
            value = {"session_id": args.session_id, **{key: value.get(key) for key in selected}}
        return _project_session_output(value, args, store)
    finally:
        store.close()


match_session_try.arguments = _session_try_args


def _session_status_args(p):
    p.add_argument("session_id")
    p.add_argument("--state-root")
    p.add_argument("--project", default=".")
    p.add_argument("--candidates", action="store_true")
    p.add_argument("--details", action="store_true", help="include full outcome and diff entries")
    _session_output_args(p)


@command("match-session-status", "Inspect a standalone matching session and its residual frontier")
def match_session_status(args):
    store = SessionStore(resolve_store(args.session_id, args.project, args.state_root))
    try:
        session = MatchSession(store, lambda *_args: None)
        value = session.status(args.session_id, args.candidates or args.exact_only,
                               args.details or args.best_details)
        registry = SessionRegistry(args.project)
        try:
            historical = registry.same_target(store.context(args.session_id))
        finally:
            registry.close()
        value["project_best"] = historical[0] if historical else None
        value["historical_sessions"] = len(historical)
        return _project_session_output(value, args, store)
    finally:
        store.close()


match_session_status.arguments = _session_status_args


def _session_export_args(p):
    p.add_argument("session_id")
    p.add_argument("output")
    p.add_argument("--state-root")
    p.add_argument("--project", default=".")
    p.add_argument("--candidate-id", help="default: current best candidate")


@command("match-session-export", "Export a retained session candidate without editing project source")
def match_session_export(args):
    store = SessionStore(resolve_store(args.session_id, args.project, args.state_root))
    try:
        candidate = (store.get(args.session_id, args.candidate_id)
                     if args.candidate_id else store.best(args.session_id))
        if candidate is None:
            raise ValueError("candidate does not exist in this session")
        source_bytes = store.cache.get(candidate.source_digest)
        if args.output == "-":
            return {"output": "stdout", "source": source_bytes.decode("utf-8", "replace"),
                    "evidence": candidate.as_dict(), "candidate": candidate.as_dict()}
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(source_bytes)
        evidence = output.with_suffix(output.suffix + ".evidence.json")
        evidence.write_text(json.dumps(candidate.as_dict(), indent=2, sort_keys=True) + "\n")
        return {"output": str(output), "evidence": str(evidence),
                "candidate": candidate.as_dict()}
    finally:
        store.close()


match_session_export.arguments = _session_export_args


def _winner_prepare_args(p):
    p.add_argument("session_id")
    p.add_argument("output_dir")
    p.add_argument("--state-root")
    p.add_argument("--project", default=".")
    p.add_argument("--candidate-id", help="default: exact best candidate")


@command("match-session-prepare-winner", "Independently recompile an exact candidate and prepare an integration transaction")
def match_session_prepare_winner(args):
    store = SessionStore(resolve_store(args.session_id, args.project, args.state_root))
    try:
        candidate = (store.get(args.session_id, args.candidate_id)
                     if args.candidate_id else store.best(args.session_id))
        if candidate is None or not candidate.exact:
            raise ValueError("integration preparation requires an exact retained candidate")
        context = store.context(args.session_id)
        source = store.cache.get(candidate.source_digest)
        verified = _session_evaluator(context, strict=True)(source, candidate.pipeline, candidate.flags,
                                               "independent-winner-verification")
        if not verified.exact:
            raise RuntimeError("independent recompilation did not reproduce the exact match")
        root = Path(args.output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        canonical_name = (f"{context.get('inventory_name')}.c" if context.get("inventory_name") else
                          f"func_{int(context.get('catalogue_addr', context['addr'])):08X}.c"
                          if context.get("addr") is not None else "winner.c")
        source_path = root / canonical_name
        source_path.write_bytes(source)
        receipt = {
            "proof": "independent exact recompilation",
            "session_id": args.session_id,
            "candidate_id": candidate.candidate_id,
            "source_sha256": candidate.source_digest,
            "pipeline": candidate.pipeline,
            "flags": list(candidate.flags),
            "target": {key: context.get(key) for key in
                       ("rom_sha256", "vram_delta", "addr", "catalogue_addr", "size",
                        "inventory_id", "inventory_image")},
            "outcome": verified.as_dict(),
            "project_edits_applied": False,
            "historical_source_name": context.get("source_name"),
            "canonical_source_name": canonical_name,
        }
        evidence = root / "winner.evidence.json"
        evidence.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        queue = root / "exact-winner.jsonl"
        queue.write_text(json.dumps({key: receipt[key] for key in
                                    ("session_id", "candidate_id", "source_sha256", "pipeline", "flags")},
                                    sort_keys=True) + "\n")
        return {"source": str(source_path), "evidence": str(evidence), "export_queue": str(queue),
                "exact": True, "project_edits_applied": False}
    finally:
        store.close()


match_session_prepare_winner.arguments = _winner_prepare_args


def _session_explore_args(p):
    p.add_argument("session_id")
    p.add_argument("--project", default=".")
    p.add_argument("--state-root")
    p.add_argument("--strategy", choices=("residual-guided",), default="residual-guided")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--variants-per-round", type=int, default=12)
    p.add_argument("--stop-on-exact", action="store_true")
    p.add_argument("--best-details", action="store_true")
    p.add_argument("--summary-only", action="store_true", help="omit frontier and per-round detail")
    p.add_argument("--best-only", action="store_true", help="return compact run summary and session best")
    p.add_argument("--target-window", help="target residual byte window START:END")
    p.add_argument("--freeze-prefix", type=lambda value: int(value, 0),
                   help="freeze residual behavior before this target offset")
    p.add_argument("--freeze-suffix", type=lambda value: int(value, 0),
                   help="freeze residual behavior at and after this target offset")
    p.add_argument("--dimensions", default="iterate,spelling,prototype,lifetime,narrowing,control-flow,flags")


@command("match-session-explore", "Run bounded multi-round residual-guided source experiments")
def match_session_explore(args):
    from ..match.explore import explore
    store = SessionStore(resolve_store(args.session_id, args.project, args.state_root))
    try:
        session = MatchSession(store, _session_evaluator(store.context(args.session_id)))
        window = None
        if args.target_window:
            try: left, right = args.target_window.split(":", 1); window = (int(left, 0), int(right, 0))
            except (ValueError, TypeError): raise ValueError("--target-window must be START:END")
        elif args.freeze_prefix is not None and args.freeze_suffix is not None:
            window = (args.freeze_prefix, args.freeze_suffix)
        if window and not 0 <= window[0] < window[1]: raise ValueError("residual window must be increasing")
        result = explore(store, session, args.session_id, args.jobs, args.rounds,
                         args.variants_per_round, args.stop_on_exact, window,
                         {value.strip() for value in args.dimensions.split(",") if value.strip()})
        if args.best_details:
            result["best"] = store.best(args.session_id).summary(True)
        elif args.best_only:
            result["best"] = store.best(args.session_id).summary(False)
        if args.summary_only or args.best_only:
            result.pop("frontier", None)
            result.pop("round_reports", None)
        return result
    finally:
        store.close()


match_session_explore.arguments = _session_explore_args


def _session_explain_args(p):
    p.add_argument("session_id")
    p.add_argument("--project", default=".")
    p.add_argument("--state-root")
    p.add_argument("--from", dest="from_id", required=True)
    p.add_argument("--to", dest="to_id", required=True)


@command("match-session-explain", "Explain source and residual changes between two candidates")
def match_session_explain(args):
    from ..match.explore import explain
    store = SessionStore(resolve_store(args.session_id, args.project, args.state_root))
    try:
        return explain(store, args.session_id, args.from_id, args.to_id)
    finally:
        store.close()


match_session_explain.arguments = _session_explain_args


def _batch_args(p):
    p.add_argument("manifests", nargs="+", metavar="MANIFEST[=PIPELINE]",
                   help="manifest TSV; optional =PIPELINE overrides the pipeline taken from its header")
    _common(p, rom_required=False)
    p.add_argument("--kmc-lane", choices=("hook", "builtin", "as", "asn64"), default=None,
                   help="assembler lane for KMC rows without --assembler= (default: hook when a "
                        "normalizer is configured, else builtin)")
    p.add_argument("--source-template", help="unit-row source path, e.g. 'src/code/{unit}.c'")
    p.add_argument("--flags", help="compiler flags for unit rows (default: toolchain profile)")
    p.add_argument("--status-flags", action="append", default=[], metavar="STATUS=FLAGS",
                   help="extra compiler flags for unit rows with this status (repeatable)")
    p.add_argument("--rodata-map", help="TSV UNIT VRAM SIZE fixing .rodata placement of unit rows")
    p.add_argument("--infer-externals", action="store_true",
                   help="recover symbols no symbol file or batch unit defines from the retail code")
    p.add_argument("--only", action="append", default=[], help="verify only units whose name contains this")


def _strict_batch(units, cfg, target, syms, jobs) -> list[dict]:
    """Compatibility entry point for strict per-unit verdicts used by audits and callers."""
    from ..match.unit_verify import verify_units
    return verify_units(units, cfg, target.rom, target.vram_to_rom, syms,
                        cfg.project / ".n64dk", min(jobs or 4, 4))


def _strict_summary(rows: list[dict]) -> dict:
    from types import SimpleNamespace
    from ..match import verify as verifymod
    counts: dict[str, int] = {}
    for row in rows:
        key = "exact_strict" if row["verdict"] == "exact" else row["verdict"]
        counts[key] = counts.get(key, 0) + 1
    ledger = verifymod.reconcile_assumptions([
        {"unit": row["unit"], "verdict": SimpleNamespace(assumptions=[
            verifymod.Assumption(item["kind"], item["name"], int(item["address"], 16))
            for item in row.get("assumptions", [])])} for row in rows])
    return {"verdicts": counts,
            "all_exact_strict": all(row["verdict"] == "exact" for row in rows),
            "assumed_symbols": ledger["proposed_symbols"],
            "contradictions": ledger["contradictions"]}


@command("match-batch", "Rebuild and verify every unit of one or more matching manifests in parallel")
def match_batch(args):
    cfg, target, syms, sym_files = _setup(args)
    lane = args.kmc_lane or ("hook" if cfg.normalizer else "builtin")
    status_flags = {}
    for item in args.status_flags:
        k, _, v = item.partition("=")
        status_flags[k] = shlex.split(v)
    rodata = manmod.load_section_map(Path(cfg.project, args.rodata_map)) if args.rodata_map else None
    units = []
    for spec in args.manifests:
        path, _, pipe = spec.partition("=")
        if pipe and pipe not in PIPELINES and pipe != "kmc" and not pipe.startswith("custom:"):
            raise SystemExit(f"unknown pipeline {pipe!r}")
        units += manmod.parse(Path(cfg.project, path), cfg.project, target.rom_to_vram, pipe or None,
                              f"kmc+{lane}", args.source_template,
                              shlex.split(args.flags) if args.flags else None, status_flags, rodata)
    if args.only:
        units = [u for u in units if any(o in u.name for o in args.only)]
    done = [0]

    def progress(_o):
        done[0] += 1
        if args.format != "json" and sys.stderr.isatty():
            print(f"\r{done[0]}/{len(units)}", end="", file=sys.stderr, flush=True)

    outs = batch(units, cfg, target, syms, args.jobs, args.infer_externals, progress)
    from ..match.verification_store import record as record_verification
    # Verification receipts are consumed by catalogue-space inventory.  Batch units build at
    # linked VRAM, which differs for shifted overlays, so retain the explicit coordinate
    # transform instead of making inventory guess from a filename or applying a game bias.
    catalogue_vrams = {}
    wanted = {(unit.vram, unit.size) for unit in units}
    if wanted:
        inventory = inventorymod.build(cfg.project)
        candidates: dict[tuple[int, int], list[int]] = {}
        section_shifts: dict[tuple[str, str, int], int] = {}
        for row in inventory.get("functions", []):
            if args.image_name and row["image"] != args.image_name:
                continue
            try:
                section_key = (row["image"], row["section_vram"], int(row["section_size"]))
                if section_key not in section_shifts:
                    mapping = resolve_image_map(cfg.rom, from_inventory(row), data=target.rom)
                    section_shifts[section_key] = (int(row["vram"], 0) -
                        int(mapping.get("linked_vram") or row["vram"], 0))
                linked = int(row["vram"], 0) - section_shifts[section_key]
            except (OSError, ValueError):
                continue
            key = (linked, int(row["size"]))
            if key in wanted:
                candidates.setdefault(key, []).append(int(row["vram"], 0))
        ambiguous = {key: values for key, values in candidates.items() if len(set(values)) != 1}
        if ambiguous:
            raise ValueError("batch target coordinates are ambiguous; pass --image-name")
        catalogue_vrams = {key: values[0] for key, values in candidates.items() if values}
    receipts = record_verification(cfg.project, cfg, target, units, outs,
                                   getattr(args, "image_name", None), catalogue_vrams)
    exact = [o for o in outs if o.exact]
    by_pipe: dict[str, list[int]] = {}
    for o in outs:
        by_pipe.setdefault(o.spec.pipeline, [0, 0])
        by_pipe[o.spec.pipeline][0] += 1
        by_pipe[o.spec.pipeline][1] += o.exact
    assumed = [o for o in outs if o.exploratory_exact and not o.exact]
    summary = {"units": len(outs), "exact": len(exact), "assumed": len(assumed),
               "failed": len(outs) - len(exact), "verification_receipts": receipts,
               "text_bytes": sum(o.size for o in exact),
               "data_bytes": sum(s.size for o in exact for s in (o.result.sections if o.result else [])),
               "by_pipeline": {k: f"{v[1]}/{v[0]}" for k, v in sorted(by_pipe.items())},
               "symbol_files": [str(f) for f in sym_files]}
    if args.format == "json":
        summary["results"] = [o.as_dict() for o in outs]
        return summary
    if sys.stderr.isatty():
        print(file=sys.stderr)
    lines = [f"{'MATCH' if o.exact else 'ASSUM' if o.exploratory_exact else 'FAIL '} [{o.spec.pipeline}] {o.name}: 0x{o.vram:08X} (0x{o.size:X} bytes)"
             + ("" if o.exact else " -- " + (o.error or (o.result.error if o.result and o.result.error else "")
                                             or (o.result.diff.summary() if o.result and o.result.diff else
                                                 "digest mismatch" if o.digest_ok is False else "section mismatch")))
             for o in outs]
    lines.append(f"Verified {summary['exact']}/{summary['units']} units, {summary['text_bytes']} code bytes, "
                 f"{summary['data_bytes']} data bytes ({', '.join(f'{k} {v}' for k, v in summary['by_pipeline'].items())})")
    print("\n".join(lines))
    return 0 if not summary["failed"] else 1


match_batch.arguments = _batch_args


def _permute_args(p):
    p.add_argument("source", help="C source containing the function")
    _common(p, rom_required=True)
    p.add_argument("--addr", required=True, type=lambda s: int(s, 16), help="VRAM of the unit's .text (hex)")
    p.add_argument("--size", type=lambda s: int(s, 16), help="text size (hex)")
    p.add_argument("--function", help="C name of the function to permute (default: the one at --addr)")
    p.add_argument("--pipeline", choices=PIPELINES)
    p.add_argument("--flags")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--seed", type=int)
    p.add_argument("--max-mutations", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--islands", type=int, default=1, help="independent walkers from the same start (default 1)")
    p.add_argument("--keep", type=int, default=4, help="best results kept that compile to different code")
    p.add_argument("--out", help="directory for best.c")
    p.add_argument("--replay-log", help="append every generated candidate and oracle score as JSONL")
    p.add_argument("--proposal-log", help="generate uncompiled candidates here and exit")
    p.add_argument("--parent-score", type=int, help="base score recorded in --proposal-log rows")


@command("match-permute", "Random AST permutation search scored by the relocated instruction diff")
def match_permute(args):
    from ..match import permuter
    if not permuter.available():
        raise SystemExit("pycparser is not importable; install it or add it to PYTHONPATH")
    cfg, target, syms, _files = _setup(args)
    spec = _spec(args, cfg)
    comparison_addr = getattr(args, "_comparison_addr", args.addr)
    if args.proposal_log:
        if args.parent_score is None:
            raise SystemExit("--proposal-log requires --parent-score")
        generated = permuter.generate_pool(
            Path(args.source).resolve(), syms, cfg.include, args.function, comparison_addr,
            args.iterations, args.seed, args.max_mutations, args.parent_score,
            Path(args.proposal_log))
        return {"generated": generated, "proposal_log": args.proposal_log,
                "compiled": 0}
    res = permuter.permute(Path(args.source).resolve(), spec, cfg, target, comparison_addr, args.size, syms,
                           cfg.include, args.function, args.iterations, args.jobs or 4, args.seed,
                           args.max_mutations, args.temperature, Path(args.out) if args.out else None,
                           Path(args.replay_log) if args.replay_log else None,
                           islands=args.islands, keep=args.keep)
    strict = None
    if res.exact:
        with tempfile.TemporaryDirectory(prefix="n64dk-permute-verify-") as name:
            source = Path(name) / "best.c"; source.write_text(res.best_source)
            strict = match_source(source, spec, cfg, target, comparison_addr, args.size, syms,
                                  Path(name) / "build", strict_verification=True)
    return {"base_score": res.base_score, "best_score": res.best_score,
            "exploratory_exact": res.exact, "exact": bool(strict and strict.exact),
            "strict_verification": strict.as_dict() if strict else None,
            "iterations": res.iterations, "improvements": res.history,
            "frontier": [{"score": score, "output": state[:16], **({"source": text} if args.format == "json" else {})}
                         for score, state, text in res.frontier],
            **({"best_source": res.best_source} if args.format == "json" else {})}


match_permute.arguments = _permute_args
