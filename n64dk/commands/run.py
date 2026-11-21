"""``decompile-run``: one command that drives every image of a project through the pipeline.

For each image named in the project's ``[images.<name>]`` configuration::

    discovery  ->  boundaries  ->  donors  ->  resolve  ->  strict queue  ->  ledger

* **discovery**   the catalogue checked against call, pointer and prologue evidence
                  (``n64dk.analysis.coverage``); an incomplete one is repaired into ``<out>/discovery``;

* **boundaries**  boundary operations the catalogue needs (reported; never applied to the project);
* **donors**      functions that are machine-code-identical to code already matched in another
                  local project are rebuilt from that proven source and verified;
* **resolve**     everything else climbs the resolution ladder (draft portfolio with recovered
                  declarations, compiler-family and per-unit recipe selection, cause diagnosis,
                  variant search), each function ending as a durable case;
* **data**        (opt-in, ``--data-stage``) every data reference that keeps a unit waiting, grouped
                  and placed only from evidence independent of that unit; ``replay`` re-verifies
                  strictly the cases whose placements are established (``n64dk.match.datastage``);
* **queue**       exact cases re-verified with the strict oracle are exported as sources plus
                  manifest rows plus the declarations they still assume;
* **ledger**      byte ownership and its violations for the image.

Everything is written under ``--out`` (never into the project's sources).  The run is
resumable: cases, the donor index and each stage's result are kept, so a second invocation
continues where the first stopped.  Progress is appended to ``<out>/events.jsonl``.

Per-image build settings come from ``n64dk.toml``::

    [run]                         # defaults for every image
    pipeline = "ido5.3"
    flags = "-O2 -mips2 -non_shared -G 0"
    symbols = ["symbol_addrs.txt"]
    [run.images.main]             # overrides and, per manifest, the pipeline its rows used
    recipes = { "config/matching_ido_units.tsv" = "ido5.3", "config/matching_kmc_units.tsv" = "kmc+as" }
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import tomllib
from pathlib import Path

from ..cli import command
from ..resources import plan_jobs
from ..match import images as imagemod
from ..match import resolve as resolvemod
from ..match import solver


# --- ablation switches (product gate Q15; ABLATION_REPORT.md "no off switch").  Every subsystem
# below is on by default and each switch only turns one off: ``--skip-stage NAME`` or, for a run
# started by a script, ``N64DK_DISABLE_<NAME>=1`` (NAME upper case, ``-`` as ``_``).  A run records
# what it switched off (solver/invocation.json, report.json ``ablation``, report.md, events.jsonl).
# A switch never touches strict verification: a disabled subsystem can only cost credit.
ABLATION_SWITCHES = {
    "declare": "declare-assumed: round 1 ([run] declare_assumed / --declare-assumed) and the cascade step",
    "context": "context synthesis (recovered structs, globals, prototypes for m2c)",
    "draft": "m2c drafting (a function then needs a source from elsewhere: in a fresh run only donors settle)",
    "draft-portfolio": "the draft portfolio (one draft, with context when there is one)",
    "draft-forms": "normal forms of the first drafts (n64dk.match.draftforms) in the portfolio",
    "redraft": "recipe-specific rewrites of the first drafts (n64dk.match.redraft) in the portfolio",
    "recipes": "per-function recipe selection (recipe retry, drafts x recipes; also in calibration)",
    "cross-recipes": "drafts x recipes (the near-miss drafts rebuilt under sibling recipes)",
    "search": "the variant search, in the ladder and as a later-round technique",
    "hand-forms": "hand-matching rewrites chosen by the near-miss diagnosis (n64dk.match.handforms)",
    "frame-fit": "frame fit (temporaries inlined, frame padding computed from the target's frame)",
    "twin": "the twin pass (source of an exact function with the same masked machine code)",
    "handasm": "hand-written assembly kept as source when it reassembles to the same bytes",
    "at-loads": "the evidence-gated kmc+atload assembler lane",
    "rodata-settlement": "donor rows settled by end-to-end read-only data chains",
    "round-search": "the later-round search technique only",
    "round-recipe-sweep": "the later-round recipe-sweep technique",
    "round-draft-options": "the later-round draft-options technique",
    "round-refine": "the later-round refine technique (round 1 may still ask a configured refiner)",
    "refine": "the refiner everywhere (round 1 and later rounds)",
    "cascade-neighbours": "the neighbours cascade step (evidence marks for the callers/callees of new exacts)",
}
SKIP_STAGES = ("discovery", "boundaries", "donors", "calibrate", "resolve", "ledger", *ABLATION_SWITCHES)
DISABLE_PREFIX = "N64DK_DISABLE_"
# later-round techniques and cascade steps an effort mode loses with a switch
_TECHNIQUES_OFF = {"declare": ("declare",), "search": ("search",), "round-search": ("search",),
                   "recipes": ("recipe-sweep",), "round-recipe-sweep": ("recipe-sweep",),
                   "draft": ("recipe-sweep", "draft-options"), "draft-portfolio": ("draft-options",),
                   "round-draft-options": ("draft-options",), "round-refine": ("refine",), "refine": ("refine",)}
_CASCADE_OFF = {"declare": ("declare",), "cascade-neighbours": ("neighbours",)}


def disabled_subsystems(skip_stage=None, environ=None) -> dict[str, str]:
    """Every stage or subsystem switched off, with where the switch came from: ``--skip-stage``
    (any choice) or ``N64DK_DISABLE_<NAME>=1`` (ablation switches only).  An unknown or malformed
    ``N64DK_DISABLE_*`` variable is refused: a typo must not leave a run claiming all-on."""
    environ = os.environ if environ is None else environ
    found = {name: "--skip-stage" for name in (skip_stage or ())}
    for key in sorted(environ):
        if not key.startswith(DISABLE_PREFIX):
            continue
        value, name = environ[key], key[len(DISABLE_PREFIX):].lower().replace("_", "-")
        if name not in ABLATION_SWITCHES:
            raise SystemExit(f"{key}: no such ablation switch; one of " + ", ".join(
                DISABLE_PREFIX + n.upper().replace("-", "_") for n in ABLATION_SWITCHES))
        if value == "1":
            found.setdefault(name, f"env {key}")
        elif value not in ("", "0"):
            raise SystemExit(f"{key}={value!r}: set it to 1 to switch the subsystem off, or leave it unset")
    return dict(sorted(found.items()))


def enabled(args, name: str) -> bool:
    """False when ``name`` is switched off for this run (``args.disabled``, set by prepare_invocation)."""
    return name not in (getattr(args, "disabled", None) or ())


def ablated_policy(mode, disabled) -> "solver.Policy":
    """``mode`` without the later-round techniques, cascade steps and refiner that ``disabled``
    switches off (the same mode when nothing applies)."""
    import dataclasses
    disabled = set(disabled or ())
    techniques = {t for name in disabled for t in _TECHNIQUES_OFF.get(name, ())}
    cascade = {c for name in disabled for c in _CASCADE_OFF.get(name, ())}
    changes = {}
    if techniques & set(mode.techniques):
        changes["techniques"] = tuple(t for t in mode.techniques if t not in techniques)
    if cascade & set(mode.cascade):
        changes["cascade"] = tuple(c for c in mode.cascade if c not in cascade)
    if "refine" in disabled and mode.refine:
        changes["refine"] = False
    return dataclasses.replace(mode, **changes) if changes else mode


def ablation_record(args, mode) -> dict:
    """What this run switched off, for report.json: the switches, the effort mode's losses, and
    the other off-options a reader of an ablation table needs (discovery, small data)."""
    full = solver.policy(getattr(args, "effort", None))
    disabled = dict(getattr(args, "disabled_from", None) or {})
    return {"disabled": disabled, "switched_off": [name for name in disabled if name in ABLATION_SWITCHES],
            "techniques_removed": [t for t in full.techniques if t not in mode.techniques],
            "cascade_removed": [c for c in full.cascade if c not in mode.cascade],
            "refine_removed": bool(full.refine and not mode.refine),
            "discovery": getattr(args, "discovery", None) or "repair",
            "small_data": getattr(args, "small_data", None) or "auto",
            "data_stage": getattr(args, "data_stage", None) or "off",
            "available": list(SKIP_STAGES)}


def check_ablation_store(out: Path, disabled, resume: bool) -> None:
    """An ablation run starts from an empty case store: an exact case left by an earlier run in
    the same directory is returned as it stands, so a run with a subsystem off would inherit that
    subsystem's credit.  (A resume keeps its own disabled set; prepare_invocation checks it.)"""
    state = out / "state"
    if disabled and not resume and state.is_dir() and any(state.iterdir()):
        raise SystemExit(f"{out}: the case store already holds cases; a run with subsystems switched off "
                         f"({', '.join(sorted(disabled))}) must start in a new --out (or --resume its own run)")


def _settings(project: Path, image: str) -> dict:
    path = project / "n64dk.toml"
    data = tomllib.loads(path.read_text()) if path.is_file() else {}
    run = dict(data.get("run") or {})
    per_image = (run.pop("images", {}) or {}).get(image, {})
    return {**run, **per_image}


def project_symbols(project: Path, image: str) -> dict[str, int]:
    """The symbols the project declares for ``image``, explicitly (never a name that only spells an
    address), from the files the resolver reads: ``[run]`` symbols, else ``[match]`` symbols, else the
    conventional files."""
    from ..match import symbols as symmod
    from ..match.toolchain import load_config
    files = [project / s for s in (_settings(project, image).get("symbols") or [])]
    if not files:
        cfg = load_config(project)
        files = cfg.symbols or symmod.discover(cfg.project)
    return dict(symmod.load([Path(f) for f in files if Path(f).is_file()]))


def project_gp(project: Path, image: str) -> int | None:
    """``_gp`` as the project declares it (:func:`project_symbols`); None when absent."""
    return project_symbols(project, image).get("_gp")


def write_gp_symbols(out: Path, line: str | None) -> list[str]:
    """The run's small-data symbol file, as extra symbols for the resolver: written when a value is
    established, removed otherwise (an earlier run's value in the same directory is not this run's,
    and decompile-export copies whatever file is there)."""
    path = out / "small_data" / "gp_symbols.txt"
    if not line:
        path.unlink(missing_ok=True)
        return []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(line)
    return [str(path)]


def small_data_lane_exact(cases) -> int:
    """Exact cases built in a small-data lane (``kmc+asg``, ``kmc+gas``): the assembler ran at the
    compiler's -G, so their objects can hold GP-relative relocations, verified against ``_gp``."""
    from ..match.pipeline import SMALL_DATA_LANES
    return sum(1 for case in cases if case.status == "exact"
               and str(case.recipe or "").partition(" ")[0] in SMALL_DATA_LANES)


def note_small_data(entry, exact: int, recovered: bool) -> None:
    """Where the ``_gp`` of the small-data lanes came from, in the report's ``small_data`` entry.
    Verification is unchanged (strict, with the relocations): this only says that functions verified
    against a ``_gp`` recovered from code, not declared by the project, verify again only with
    ``small_data/gp_symbols.txt``."""
    if not isinstance(entry, dict):
        return
    entry["exact_small_data_lane"] = exact
    entry["gp_source"] = ("project" if entry.get("project_gp") else "unknown" if entry.get("project_gp_error")
                          else "recovered" if recovered else None)
    if entry["gp_source"] == "recovered" and exact:
        entry["gp_note"] = (f"{exact} exact functions were built in a small-data lane and verified against the _gp "
                            "recovered from the code that sets $gp (small_data/gp_symbols.txt), which the project "
                            "does not declare: they verify again only with that file")


def _stage(folder: Path, name: str, compute, redo: bool, identity: str = ""):
    """Checkpointed stage result, invalidated when its immutable inputs change."""
    path = folder / f"stage_{name}.json"
    meta = folder / f"stage_{name}.meta.json"
    if path.is_file() and not redo:
        # a checkpoint that does not parse (a full disk or power loss tore it) is recomputed, never
        # trusted and never turned into a stage error (CHAOS_TESTING.md finding C1)
        try:
            if not identity or (meta.is_file() and json.loads(meta.read_text()).get("identity") == identity):
                return json.loads(path.read_text()), True
        except (ValueError, OSError):
            pass
    result = compute()
    folder.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=1, default=str)
    with tempfile.NamedTemporaryFile("w", dir=folder, prefix=f".{name}-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())                  # durable before it replaces the old file
        except BaseException:
            temporary.unlink(missing_ok=True)          # a full disk leaves no partial temporary
            raise
    # the old identity record goes first: a crash between the two replaces must leave a checkpoint
    # with no identity (recomputed), never the new result under the old identity
    meta.unlink(missing_ok=True)
    temporary.replace(path)
    if identity:
        temporary_meta = meta.with_suffix(".tmp")
        temporary_meta.write_text(json.dumps({"identity": identity}))
        temporary_meta.replace(meta)
    return result, False


def _profiled(profiler, name: str, image: str, compute):
    """``compute`` measured as one stage of the run's profile (``n64dk.profiling``), with what it
    produced recorded as the stage's gain."""
    def run():
        with profiler.stage(name, image) as record:
            result = compute()
            if isinstance(result, dict):
                record["gain"] = {key: result[key] for key in (
                    "functions_recovered", "functions_awaiting_symbols", "statuses", "promoted", "declared",
                    "chosen", "rodata_settlement") if key in result and not isinstance(result[key], list)}
                if isinstance(record["gain"].get("chosen"), dict):
                    record["gain"]["chosen"] = record["gain"]["chosen"].get("label")
            return result
    return run


def _identity(**values) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()


CALIBRATION_SIZES = (0x18, 0x80)      # small enough to match from a draft, large enough to depend on the recipe


RECIPE_DOUBT = 0.4         # share of an image whose marks contradict its configured recipe that triggers calibration
DOUBT_SAMPLE = 96


def calibrate_recipe(resolve_hard, resolve_options, state: Path, image: str, rows, skip, sample: int,
                     delta: int | None = None, region: tuple[int, int] | None = None) -> dict:
    """Resolve a spread sample of small functions with every available recipe tried on each, and
    return the recipe most of the byte-identical ones were built with.

    A title's compiler and optimisation level are not known in advance, and the configured recipe
    is a guess (for a project made by ``decompile-init``, a fingerprint).  The sample is taken
    across the whole address range, skipping what donors already recovered, so library code built
    with another compiler does not decide the answer.  The cases it produces stay in the store."""
    low, high = CALIBRATION_SIZES
    eligible = [row for row in rows if low <= row[2] <= high and row[1] not in skip]
    if not eligible:
        return {"sample": 0, "identical": 0, "recipes": {}, "chosen": None, "reason": "no functions in the size window"}
    every = max(1, len(eligible) // sample)
    extra = {}
    if region is not None:                               # only this region's functions
        extra["only_names"] = [",".join(sorted({row[0] for row in eligible}))]
    result = resolve_hard(resolve_options(min_size=low, max_size=high, every=every, limit=sample,
                                          cross_recipes=64, refine=None, budget=40, provisional=True,
                                          write_scope=False if region is not None else True, **extra))
    store = resolvemod.CaseStore(state)
    try:
        cases = [case for case in store.all() if case.image == image and case.status in ("exact", "assumed")
                 and low <= case.size <= high and (region is None or region[0] <= case.vram <= region[1])]
    finally:
        store.close()
    evidence = recipe_evidence(cases)
    tally = evidence["votes"]
    ranked = sorted(tally.items(), key=lambda item: (-item[1], item[0]))
    chosen = None
    if ranked and ranked[0][1] >= 3:
        pipeline, _, flags = ranked[0][0].partition(" ")
        chosen = {"label": ranked[0][0], "pipeline": pipeline, "flags": flags, "matched": ranked[0][1]}
    return {"sample": result.get("functions"), "identical": len(cases), "recipes": dict(ranked[:8]),
            "chosen": chosen, "version_evidence": {k: evidence[k] for k in ("insensitive", "discriminating", "unchecked")}, "seconds": result.get("seconds"),
            "reason": None if chosen else "fewer than three sample functions were byte-identical under one recipe "
                                          "and not under its sibling versions"}


def recipe_evidence(cases) -> dict:
    """Votes for recipes from matched calibration cases, without the version-insensitive ones: a
    function exact under both IDO 5.3 and 7.1 (stage ``version-check``) is evidence for neither,
    so it must not lock a title or region to the version tried first
    (``n64dk.match.compilers.version_evidence``)."""
    from ..match import compilers as compmod
    rows = []
    for case in cases:
        if not case.recipe:
            continue
        check = next((st for st in reversed(case.stages) if st.get("stage") == "version-check"), None)
        rows.append((case.recipe, None if check is None else check.get("exact_under")))
    return compmod.version_evidence(rows)


REGION_MIN_FUNCTIONS = 40      # a region calibrated on its own needs this many functions
REGION_MIN_SAMPLE = 24


def calibrate_regions(resolve_hard, resolve_options, state: Path, image: str, rows, skip, sample: int,
                      delta: int, data: bytes, emit=None) -> dict:
    """Regional calibration: the image's toolchain regions (``n64dk.match.toolchain_regions``)
    each calibrated on a sample of their own small functions, so a region built by another recipe
    than the rest is not decided by the majority.  Returns ``{"regions": [...], "map": [...]}``
    where ``map`` holds the regions that chose a recipe (start, end, pipeline, flags) for
    ``resolve-hard``'s ``region_map``.  Regions without enough functions or a choice keep the
    image's recipe."""
    from ..match import toolchain_regions as tcmod
    found = tcmod.reconstruct(data, delta, rows)
    regions = [r for r in found.get("regions") or [] if r["functions"] >= REGION_MIN_FUNCTIONS
               and r["verdict"] != "assembly"]
    if len(regions) < 2:
        return {"regions": [], "map": [], "reason": "fewer than two regions large enough to calibrate apart"}
    total = sum(r["functions"] for r in regions)
    out, mapping = [], []
    for region in regions:
        lo, hi = int(region["start"], 16), int(region["end"], 16)
        inside = [row for row in rows if lo <= row[1] <= hi]
        share = max(REGION_MIN_SAMPLE, round(sample * region["functions"] / total))
        found_here = calibrate_recipe(resolve_hard, resolve_options, state, image, inside, skip, share, delta,
                                      region=(lo, hi))
        chosen = found_here.get("chosen")
        out.append({"start": region["start"], "end": region["end"], "functions": region["functions"],
                    "verdict": region["verdict"], "sample": found_here.get("sample"),
                    "identical": found_here.get("identical"), "recipes": found_here.get("recipes"),
                    "version_evidence": found_here.get("version_evidence"),
                    "chosen": (chosen or {}).get("label")})
        if chosen:
            mapping.append({"start": lo, "end": hi, "pipeline": chosen["pipeline"], "flags": chosen["flags"],
                            "label": chosen["label"], "matched": chosen.get("matched"),
                            "evidence": found_here.get("version_evidence")})
        if emit:
            emit("stage", image=image, stage="calibrate-region", start=region["start"], chosen=(chosen or {}).get("label"))
    return {"regions": out, "map": mapping}


def refine_settings(settings: dict, args):
    """``[run.refine]`` (or a per-image override) merged with the command line; None when no
    refiner command is configured."""
    from ..match import refine as refinemod
    built = refinemod.Settings.from_table(settings.get("refine"), getattr(args, "refine_command", None))
    if built is None:
        return None
    for key in ("rounds", "max_cases", "jobs", "max_score"):
        value = getattr(args, f"refine_{key}", None)
        if value is not None:
            setattr(built, key, value)
    if getattr(args, "refine_known_source", None):
        built.known_source = [*built.known_source, *args.refine_known_source]
    return built


def declare_assumed(resolve_hard, resolve_options, state: Path, image: str, folder: Path, overrides: dict,
                    verified: dict | None = None, runtime_helpers: dict | None = None) -> dict:
    """Declare the addresses that byte-identical (``assumed``) cases lean on and verify those cases
    again.

    Only a symbol whose name states its own address (``D_80123456`` or ``func_80123456`` at 0x80123456)
    is declared: the declaration says nothing the name did not, and the address was read from the
    target's own instructions.  Call targets count like data: a draft names a callee the catalogue does
    not list by the address its ``jal`` encodes, and a case calling one stayed ``assumed`` for want of
    that line (measured on a title: nine byte-identical functions, six of them for a callee's name).  A compiler runtime helper (``runtime_helpers``: name -> the addresses whose code is
    that helper, ``n64dk.match.runtimehelpers``) is declared too, at an address where the helper's own code
    stands: the compiler writes that name, the source cannot.  Any other assumption (a named symbol, a
    section) is left for a person.  The verdict still comes from the strict verifier; a case it does not
    call exact stays ``assumed``.

    ``verified`` (optional, kept by the caller for one invocation): case id -> (best source, recipe,
    digest of the declarations) of each case a pass left ``assumed``.  Such a case is verified again
    only when one of the three changed: the same source, recipe and declarations give the same
    strict verdict, so a later pass (the solver's declare cascade) does not rebuild it."""
    store = resolvemod.CaseStore(state)
    pending = []
    try:
        names: dict[str, int] = {}
        waiting = 0
        for case in store.all():
            if case.image != image or case.status != "assumed":
                continue
            waiting += 1
            pending.append(case)
            for item in (case.strict or {}).get("assumptions", []):
                try:
                    address = int(str(item.get("address")), 16)
                except ValueError:
                    continue
                if item.get("kind") == "symbol" and (
                        item.get("name") in (f"D_{address:08X}", f"func_{address:08X}")
                        or address in (runtime_helpers or {}).get(item.get("name"), ())):
                    names[item["name"]] = address
    finally:
        store.close()
    if not names:
        return {"declared": 0, "promoted": 0, "waiting": waiting}
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "declared_symbols.txt"
    text = "".join(f"{name} = 0x{address:08X};\n" for name, address in sorted(names.items(), key=lambda r: r[1]))
    # the data stage's lines (n64dk.match.datastage) stay: exact cases were verified with them
    from ..match.datastage import DATA_STAGE_MARK
    kept = [line for line in path.read_text().splitlines() if DATA_STAGE_MARK in line] if path.is_file() else []
    text += "".join(line + "\n" for line in kept)
    path.write_text(text)
    declarations = hashlib.sha256(text.encode()).hexdigest()

    def key(case) -> tuple:
        return (case.best_digest, case.recipe, declarations)
    todo = [case for case in pending if verified is None or tuple(verified.get(case.case_id) or ()) != key(case)]
    if not todo:                         # every case was verified against these declarations already
        return {"declared": len(names), "file": str(path), "waiting": waiting, "promoted": 0,
                "still_assumed": waiting, "unchanged": waiting}
    extra = {}
    if len(todo) < len(pending):         # only the cases not verified yet; the sweep's scope file stays
        extra = {"only_names": [",".join(sorted({case.name for case in todo}))], "write_scope": False}
    options = resolve_options(**overrides, refine=None, only_statuses=("assumed",), **extra)
    options.symbols = list(options.symbols) + [str(path)]
    report = resolve_hard(options)
    statuses = report.get("statuses") or {}
    if verified is not None:
        store = resolvemod.CaseStore(state)
        try:
            for case in todo:
                after = store.get(case.case_id)
                if after is not None and after.status == "assumed":
                    verified[case.case_id] = key(after)
        finally:
            store.close()
    return {"declared": len(names), "file": str(path), "waiting": waiting, "promoted": statuses.get("exact", 0),
            "still_assumed": statuses.get("assumed", 0), "unchanged": len(pending) - len(todo)}


def data_stage(out: Path, folder: Path, spec, project: Path, *, data: bytes, delta: int, rows, manifests,
               gp_found, homes: dict, redo: bool, base_identity: str) -> dict:
    """The opt-in data stage of one image (:mod:`n64dk.match.datastage`), checkpointed on what it
    reads: the image, the cases, the donor report, the project's symbols and both _gp records.
    Writes ``<out>/<image>/data_ownership.json`` and ``data_replay.json``; builds nothing."""
    from ..match import byteledger as ledgermod
    from ..match import datastage
    from ..match.ownership import gp_provenance
    notes: list[str] = []
    try:
        declared = project_symbols(project, spec.name)
    except Exception as error:                               # noqa: BLE001 - reported; no declaration is used
        declared = None
        notes.append(f"project symbols unreadable: {type(error).__name__}: {error}"[-300:])
    provenance = gp_provenance((declared or {}).get("_gp"), gp_found)
    if declared is None:
        provenance.update(status="unknown", value=None, basis=None)     # the declared side is unknown
    cases, donor_rows = datastage.load_units(out / "state", spec.name, folder / "donors" / "report.json")
    manifest_rows = []
    for manifest in manifests:
        cls = "asm" if "asm" in Path(manifest).name.lower() else "c"
        try:
            manifest_rows += [(r.vram, r.size, f"{Path(manifest).name}:{r.source}", r.cls)
                              for r in ledgermod.read_manifest(manifest, delta, cls)]
        except Exception as error:                           # noqa: BLE001 - reported; its rows are left out
            notes.append(f"{Path(manifest).name}: {type(error).__name__}: {error}"[-300:])
    loader = {delta + image_offset for _rom, image_offset, _size, _how in spec.raw_segments}
    if spec.extent:
        loader.add(spec.extent[0])
    identity = _identity(base=base_identity, units=datastage.units_digest(cases, donor_rows),
                         declared=sorted((declared or {}).items()), gp=provenance, homes=sorted(homes.items()),
                         manifest_rows=manifest_rows)

    def compute():
        tables = ()
        try:
            from ..analysis import datasolve, indirect
            tables = datasolve.tables_from_indirect(indirect.analyse_image(data, delta, rows))
        except Exception as error:                           # noqa: BLE001 - placed without switch tables
            notes.append(f"switch tables: {type(error).__name__}: {error}"[-300:])
        document = datastage.run_image(
            image=spec.name, data=data, delta=delta, rows=rows, cases=cases, donor_rows=donor_rows,
            manifest_rows=manifest_rows, declared=declared or {}, provenance=provenance, loader_starts=loader,
            homes=homes, rom_of=lambda address: spec.rom_of(address - delta)[0], extent=spec.extent, tables=tables)
        document["notes"] = notes
        return datastage.write(folder, document)
    result, cached = _stage(folder, "data", compute, redo, identity)
    return {**result, "cached": cached}


def replay_data(resolve_hard, resolve_options, folder: Path, overrides: dict) -> dict:
    """Re-verify, strictly and in one batch, the cases whose every data placement the data stage
    established (``<folder>/data_replay.json``, ``replay: resolve``), with those placements added to
    ``declared_symbols.txt``.  A unit replayed before with the same placements is not built again.
    The verdict is the strict verifier's: a case it does not call exact stays ``assumed``."""
    from ..match import datastage
    entries = json.loads((folder / "data_replay.json").read_text())
    todo = [e for e in entries if e.get("replay") == "resolve" and e.get("origin") == "case"]
    done_path = folder / "data_replayed.json"
    done = json.loads(done_path.read_text()) if done_path.is_file() else {}
    fresh = [e for e in todo if done.get(e["unit"]) != e.get("digest")]
    result = {"units": 0, "already_replayed": len(todo) - len(fresh), "promoted": 0}
    if not fresh:
        return result
    lines = {name: (int(address, 16), f"{entry['unit']}: {entry['why'].get(name, '')}")
             for entry in fresh for name, address in entry["declare"].items()}
    merged = datastage.merge_declared(folder, lines)
    declared = set(merged["declared"])
    labels = sorted({e["label"] for e in fresh if set(e["declare"]) <= declared})
    result.update(declared=len(declared), added=merged["added"], conflicts=merged["conflicts"], file=merged["file"])
    if labels:
        # a pass over a named set: the sweep's scope file (what was left out, and why) stays
        options = resolve_options(**overrides, refine=None, only_statuses=("assumed",), only_names=[",".join(labels)],
                                  write_scope=False)
        options.symbols = list(options.symbols) + [merged["file"]]
        statuses = resolve_hard(options).get("statuses") or {}
        result.update(units=len(labels), promoted=statuses.get("exact", 0), still_assumed=statuses.get("assumed", 0))
    for entry in fresh:
        done[entry["unit"]] = entry.get("digest")
    done_path.write_text(json.dumps(done, indent=1, sort_keys=True) + "\n")
    return result


def resolve_options_for(args, settings: dict, *, mode, project: Path, spec, jobs: int, events: Path, out: Path,
                        manifests: list[str], recipes: dict, recovered: set[int],
                        small_data_symbols: list[str] | tuple = ()):
    """The ``resolve-hard`` options of one image, as a function of per-call overrides.  ``mode`` is
    the effort policy; the options themselves come from ``args`` after ``solver.apply_policy``
    (``normal`` fills exactly the defaults decompile-run had before effort modes)."""
    disabled = tuple(sorted(getattr(args, "disabled", None) or ()))

    def resolve_options(**overrides):
        options = argparse.Namespace(
            project=str(project), config=None, toolchain_dir=args.toolchain_dir, normalizer=None, include=[],
            rom=str(spec.path), vram_delta=spec.vram_delta if spec.delta_source == "declared" else None,
            jobs=jobs, workers=args.workers,
            worker_memory_mb=args.worker_memory_mb, memory_reserve_mb=args.memory_reserve_mb,
            refine=refine_settings(settings, args) if mode.refine else None, refine_command=None, keep_padding=False,
            search_ceiling=(args.search_ceiling if args.search_ceiling is not None
                            else settings.get("search_ceiling", resolvemod.SEARCH_CEILING)),
            symbols=[str(project / s) for s in settings.get("symbols", [])],
            catalogue=str(spec.catalogue), catalogue_shift=spec.catalogue_shift,
            manifest=[m + ("=" + recipes[str(Path(m).resolve())] if str(Path(m).resolve()) in recipes else "")
                      for m in manifests],
            limit=args.limit, max_size=args.max_size, min_size=4, every=1, holdout=0, show=5,
            image_name=spec.name, pipeline=settings.get("pipeline", args.pipeline),
            flags=settings.get("flags", args.flags), context=None if "context" in disabled else "auto",
            context_parts="structs,globals,prototypes", proto_confidence=0.9, budget=args.budget,
            context_headers=[str(project / h) for h in settings.get("context_headers", [])],
            state=str(out / "state"), events=str(events), no_draft="draft" in disabled,
            no_recipes="recipes" in disabled, draft_portfolio="draft-portfolio" not in disabled,
            draft_options=settings.get("draft_options", ""), force=False,
            skip_vrams=recovered, addr=None, size=None, cross_recipes=getattr(args, "cross_recipes", None),
            only_names=getattr(args, "only_names", None), order=getattr(args, "order", None),
            region_recipes=bool(getattr(args, "region_recipes", False) or
                                settings.get("region_recipes", False)),
            unit_recipes=bool(getattr(args, "unit_recipes", False) or
                              settings.get("unit_recipes", False)),
            skip_unknown_search=bool(getattr(args, "skip_unknown_search", False) or
                                     settings.get("skip_unknown_search", False)),
            search_economics=bool(getattr(args, "search_economics", False) or
                                  settings.get("search_economics", False)),
            # command line, else [run] recipe_gate, else the mode the run was started with
            recipe_gate=solver.setting(args, settings, "recipe_gate", mode))
        options.extra_symbols = list(small_data_symbols)
        options.disabled = disabled                  # read by resolve-hard (n64dk.commands.resolve)
        for key, value in overrides.items():
            setattr(options, key, value)
        return options

    return resolve_options


@command("decompile-run", "Drive every configured image through boundaries, donors, resolution, strict queue and ledger")
def decompile_run(args):
    """One heavy job at a time, with a preflight and a lease (see ``n64dk.safety``)."""
    from .. import safety
    prepare_invocation(args)
    try:
        with safety.heavy_job("decompile-run", Path(args.out), wait_for_lock=getattr(args, "wait_for_lock", False),
                              force=getattr(args, "force_resources", False), argv=sys.argv[1:]) as check:
            if getattr(args, "run_mode", None):
                result = _mode_run(args)
            elif getattr(args, "blind_forbid", None):
                result = _blind_run(args)
            else:
                result = _unguarded_run(args)
    except safety.Busy as error:
        raise SystemExit(str(error)) from None
    if isinstance(result, dict) and not check.get("nested"):
        result["preflight"] = check
        result["paused"] = safety.pauses()
    return result


def _unguarded_run(args):
    """A run that is not blind marks its run directory as such (``governance.record_session``)
    before it does any work, so no later ``--blind-forbid`` run reuses its checkpoints, case store,
    donor index or answer cache (they may have been informed by any tree).  A directory this run
    creates is marked when the run ends; one it leaves unmarked (killed first) has state and no
    record, which a blind run refuses just the same."""
    from .. import governance as gov
    out = Path(args.out).resolve()
    project = Path(args.project).resolve()
    if ((out == project or project in out.parents) and not (project / ".n64dk" == out or
            project / ".n64dk" in out.parents) and not getattr(args, "allow_project_output", False)):
        return _decompile_run(args)              # which refuses it before writing anything
    plan = getattr(args, "_run_mode_plan", None)            # --mode source-tree (n64dk.runmodes)
    fields = ({"run_mode": plan.mode, "trees": [{k: row.get(k) for k in ("label", "root", "tree_digest")}
                                                for row in plan.recorded]} if plan is not None else {})
    # a --mode run has already refused a project-internal --out (_mode_run): it is marked before any
    # work even when it creates the directory, so its session is never mistaken for unknown state
    marked = out.is_dir() or plan is not None
    if marked:
        gov.record_session(out, blind=False, status="not blind", **fields)
    try:
        return _decompile_run(args)
    finally:
        if not marked and out.is_dir():
            gov.record_session(out, blind=False, status="not blind", **fields)


def _mode_run(args):
    """``--mode blind|research|source-tree`` (``n64dk.runmodes``): what the run may learn from,
    enforced when it executes.  The plan identifies the title (ROM SHA-1, header name and game code,
    ``--title``), takes the corpus manifest's trees that concern it, classifies every donor and, in a
    blind run, looks for names of the title's trees in the project's metadata.  A donor or a name the
    mode does not permit, or a run directory filled under a less strict mode
    (``governance.mode_reuse``), refuses the run (exit 3, ``solver/run_mode_violation.json``)
    before any stage.  Blind and research runs then go through the strict isolation guard with
    the title's trees forbidden (``_blind_run``); a source-tree run records the trees it may use."""
    from .. import governance as gov
    from .. import runmodes
    out = Path(args.out).resolve()
    project = Path(args.project).resolve()
    if ((out == project or project in out.parents) and not (project / ".n64dk" == out or
            project / ".n64dk" in out.parents) and not getattr(args, "allow_project_output", False)):
        raise SystemExit("refusing to write generated run state inside the project source tree; "
                         "use <project>/.n64dk/... or --allow-project-output")
    mode = args.run_mode
    plan = runmodes.build_plan(args)                    # SystemExit when no run of this mode is possible
    reuse = gov.mode_reuse(out, mode)
    plan.run_dir = {k: reuse.get(k) for k in ("verdict", "sessions", "modes")}
    plan.problems += [f"run directory: {problem}" for problem in reuse.get("problems", [])]
    if plan.problems:
        plan.write(out, "refused", "run_mode_violation.json")
        gov.record_session(out, blind=mode != "source-tree", status="refused", run_mode=mode,
                           forbidden=[("", root) for root in plan.forbidden_roots()], reason=plan.problems[0])
        print(f"{mode} run refused: {'; '.join(plan.problems[:3])}; receipt in solver/run_mode_violation.json",
              file=sys.stderr)
        raise SystemExit(3)
    args._run_mode_plan = plan                          # not JSON: never saved, never part of the guard's config
    if mode == "source-tree":
        result = _unguarded_run(args)
    else:
        args.blind_forbid = plan.forbidden_roots()      # saved resolved with the invocation
        result = _blind_run(args)
    plan.write(out, "complete")
    if isinstance(result, dict):
        result["run_mode"] = plan.summary()
    return result


def _blind_run(args):
    """``--blind-forbid TREE``: a blind run, enforced rather than asserted.  The isolation check
    (``governance.blind_guard``, strict) runs on the run's own configuration, project and
    environment before any stage, and again when the run ends; both receipts are written to
    ``solver/``.  A reachable forbidden tree refuses the run (exit 3) before anything runs; one that
    became reachable during the run fails it (exit 3) and the results must not be reported as blind.

    The run directory is checked first (``governance.run_provenance``): its checkpoints, case store,
    donor index and answer cache are reused by the stages, so a directory that holds state from a
    session that was not blind against these trees (or from an unrecorded one) refuses the run
    (exit 3); every blind session is recorded in ``solver/blind_provenance.json``."""
    from .. import governance as gov
    out = Path(args.out).resolve()
    forbidden = [(f"forbid:{Path(p).name}", Path(p).expanduser().resolve()) for p in args.blind_forbid]
    missing = [str(root) for _label, root in forbidden if not root.exists()]
    if missing:                                     # a tree that is not there is checked against nothing
        raise SystemExit(f"--blind-forbid: no such tree: {', '.join(missing)}; a blind run checked against "
                         "nothing is not blind")
    config = {}
    for key, value in vars(args).items():
        if key in _NOT_SAVED or key == "blind_forbid" or callable(value):     # the forbidden list itself
            continue                                                          # is not a route to the trees
        try:
            json.dumps(value)
        except TypeError:
            continue
        config[key] = value
    # the run directory is an output: checked against the trees but not walked (the run fills it with
    # candidate sources); what it held before this run is decided by its provenance record
    config.pop("out", None)
    provenance = gov.run_provenance(out, forbidden)          # before this run writes anything there
    (out / "solver").mkdir(parents=True, exist_ok=True)

    def record(name: str, receipt: dict) -> None:
        solver.write_json(out / "solver" / name, receipt)

    run_mode = getattr(args, "run_mode", None)              # --mode blind / research (n64dk.runmodes)
    mode_fields = {"run_mode": run_mode} if run_mode else {}

    def refused(reason: str) -> None:
        # a --mode run records the attempt it refused before any work, so a corrected rerun can use
        # the directory (without --mode the record stays as it was)
        if run_mode:
            gov.record_session(out, blind=True, forbidden=forbidden, status="refused", reason=reason, **mode_fields)

    if not provenance["ok"]:
        record("blind_violation.json", {**provenance, "violations": [
            {"key": "out", "value": str(out), "verdict": problem} for problem in provenance.get("problems", [])]})
        refused(provenance["verdict"])
        print(f"blind run refused: the run directory is not blind against these trees "
              f"({'; '.join(provenance.get('problems', [])[:3])}); receipt in solver/blind_violation.json",
              file=sys.stderr)
        raise SystemExit(3)
    limit = getattr(args, "blind_walk_limit", None) or 50_000
    session = None
    try:
        shared = []
        for spec in getattr(args, "blind_shared_library", None) or []:
            try:
                shared.append(gov.SharedLibrary.load(*gov.parse_pin(spec)))
            except (OSError, ValueError) as exc:
                print(f"blind run refused: shared library manifest {spec}: {exc}", file=sys.stderr)
                refused("shared library manifest refused")
                raise SystemExit(2) from None
        with gov.blind_guard(forbidden, config=config, project=args.project, shared=shared, strict=True,
                             file_limit=limit, outputs={"out": str(out)},
                             **({"mode": run_mode} if run_mode else {})) as guard:
            record("blind_entry.json", guard.receipt)
            session = gov.record_session(out, blind=True, forbidden=forbidden, status="entered",
                                         entry_digest=guard.receipt.get("digest"), **mode_fields)
            result = _decompile_run(args)
    except gov.BlindViolation as error:
        receipt = getattr(error, "receipt", {}) or {}
        record("blind_violation.json", receipt)
        if session is not None:                  # reached during the run: its state is not blind
            gov.update_session(out, session, status="violated", exit_digest=receipt.get("digest"))
        else:
            refused("a forbidden tree is reachable")
        reached = "; ".join(f"{v.get('key')}: {v.get('verdict')}" for v in (receipt.get("violations") or [])[:5])
        print(f"blind run refused: a forbidden tree is reachable ({reached}); receipt in solver/blind_violation.json",
              file=sys.stderr)
        raise SystemExit(3) from None
    record("blind_exit.json", guard.exit_receipt)
    gov.update_session(out, session, status="clean", exit_digest=(guard.exit_receipt or {}).get("digest"))
    if isinstance(result, dict):
        result["blind"] = {"enforced": True, "forbidden": [label for label, _root in forbidden],
                           "entry_digest": guard.receipt.get("digest"),
                           "exit_digest": (guard.exit_receipt or {}).get("digest"), "ok": True,
                           "run_dir": {"verdict": provenance["verdict"], "sessions_before": provenance["sessions"],
                                       "interrupted_before": provenance.get("interrupted", 0)}}
    return result


# options a resumed invocation takes from its own command line; every other one comes from the
# invocation that started the run (solver/invocation.json)
RESUME_FROM_COMMAND_LINE = ("budget_time", "budget_builds", "budget_model_calls", "wait_for_lock", "force_resources",
                            "jobs", "workers", "worker_memory_mb", "memory_reserve_mb", "blind_walk_limit")
_NOT_SAVED = {"_fn", "format", "resume", "command", "redo"} | set(RESUME_FROM_COMMAND_LINE)


def prepare_invocation(args) -> solver.Policy:
    """Merge ``--resume RUN_DIR`` with the invocation that started the run, check the effort mode
    and fill every option the command line left unset from it.  The subsystems switched off
    (``--skip-stage``, ``N64DK_DISABLE_*``) are fixed when the run starts: a resume under another
    environment is refused, so no round of one run is measured with another set."""
    saved = None
    if getattr(args, "resume", None):
        run_dir = Path(args.resume).resolve()
        path = run_dir / "solver" / "invocation.json"
        if not path.is_file():
            raise SystemExit(f"--resume {run_dir}: no solver/invocation.json; only a run made by this version "
                             "of decompile-run can be resumed")
        saved = json.loads(path.read_text())
        rounds, effort, schedule = (getattr(args, "rounds", None), getattr(args, "effort", None),
                                    getattr(args, "schedule", None))
        data_stage_now = getattr(args, "data_stage", None)        # opt-in; it can join a resumed run
        # --blind-forbid comes from the run's start: a tree named only now would be silently dropped
        # below, and a session that was never checked cannot be made blind afterwards
        started_blind = {str(Path(p).expanduser().resolve()) for p in (saved.get("blind_forbid") or [])}
        added = sorted({str(Path(p).expanduser().resolve()) for p in (getattr(args, "blind_forbid", None) or [])}
                       - started_blind)
        if added:
            raise SystemExit(f"--resume {run_dir}: --blind-forbid {', '.join(added)} was not enforced when the run "
                             "started; start a fresh run to make it blind")
        # --mode is the run's: what it learnt so far was learnt under the mode it started with
        given_mode = getattr(args, "run_mode", None)
        if given_mode is not None and given_mode != saved.get("run_mode"):
            raise SystemExit(f"--resume {run_dir}: the run was started with --mode {saved.get('run_mode') or 'none'}; "
                             f"a resume cannot make it {given_mode}: start a fresh --out")
        for key, value in saved.items():
            if key not in _NOT_SAVED:
                setattr(args, key, value)
        if "mode_settings" not in saved and saved.get("effort") in solver.MODES:
            # saved before the mode's settings were recorded: they are the starting mode's
            args.mode_settings = solver.mode_settings(solver.MODES[saved["effort"]])
        if effort is not None and effort != saved.get("effort"):
            # another mode for the rounds still to come (e.g. a finished normal run continued as deep):
            # its rounds, scheduling, techniques and cascade apply; the options the run was started
            # with (budget per function, sizes, recipes, the recipe gate and regional calibration)
            # stay as they were, so round 1 is comparable
            args.effort, args.rounds, args.schedule = effort, None, None
        if rounds is not None:
            args.rounds = rounds
        if schedule is not None:
            args.schedule = schedule
        if data_stage_now is not None:
            args.data_stage = data_stage_now
        args.out, args.redo = str(run_dir), False
    if not getattr(args, "project", None) or not getattr(args, "out", None):
        raise SystemExit("decompile-run needs --project and --out (or --resume RUN_DIR)")
    found = disabled_subsystems(getattr(args, "skip_stage", None))
    if saved is not None:
        # a run saved before the switches existed disabled exactly its --skip-stage stages
        before = sorted(saved["disabled"] if "disabled" in saved else
                        disabled_subsystems(saved.get("skip_stage"), {}))
        if sorted(found) != before:
            raise SystemExit(f"--resume: the run was started with {before or 'nothing'} switched off, this "
                             f"invocation switches off {sorted(found) or 'nothing'}; resume with the same "
                             "N64DK_DISABLE_* environment")
    args.disabled, args.disabled_from = sorted(found), found
    mode = solver.policy(getattr(args, "effort", None))
    solver.check_policy(mode)
    args.effort = mode.name
    solver.apply_policy(args, mode)
    return mode


def save_invocation(args, out: Path) -> None:
    keep = {}
    for key, value in vars(args).items():
        if key in _NOT_SAVED or callable(value):
            continue
        try:
            json.dumps(value)
        except TypeError:
            continue
        keep[key] = value
    for key in ("project", "donors", "donor_index", "library", "toolchain_dir", "corpus_manifest"):  # from anywhere
        if isinstance(keep.get(key), str) and keep[key]:
            keep[key] = str(Path(keep[key]).expanduser().resolve())
    if isinstance(keep.get("blind_forbid"), list):        # a relative tree must not move with --resume's cwd
        keep["blind_forbid"] = [str(Path(p).expanduser().resolve()) for p in keep["blind_forbid"]]
    solver.write_json(out / "solver" / "invocation.json", keep)


def _decompile_run(args):
    if getattr(args, "donors", None) and not Path(args.donors).is_file():
        raise SystemExit(f"--donors {args.donors}: no such file")
    project = Path(args.project).resolve()
    out = Path(args.out).resolve()
    if (out == project or project in out.parents) and not (project / ".n64dk" == out or
            project / ".n64dk" in out.parents) and not getattr(args, "allow_project_output", False):
        raise SystemExit("refusing to write generated run state inside the project source tree; "
                         "use <project>/.n64dk/... or --allow-project-output")
    out.mkdir(parents=True, exist_ok=True)
    from ..profiling import Profiler
    profiler = Profiler(out)
    registry = imagemod.load_registry(project)
    if not registry:
        raise SystemExit("no [images.<name>] tables in the project's n64dk.toml: declare each image (path, "
                         "vram_delta for linear images, catalogue, manifests)")
    names = sorted(registry) if args.all_images or not args.image else args.image
    events = out / "events.jsonl"
    # --- effort mode, rounds log, observations and budgets (n64dk.match.solver); the mode loses
    # the techniques and cascade steps an ablation switch turns off
    # (switched: the ablation switches proper; a --skip-stage of discovery, boundaries, ... is recorded
    # but changes nothing it did not change before)
    switched = [name for name in getattr(args, "disabled", None) or () if name in ABLATION_SWITCHES]
    mode = ablated_policy(solver.policy(getattr(args, "effort", None)), switched)
    resume = bool(getattr(args, "resume", None))
    check_ablation_store(out, switched, resume)
    solver_dir = out / "solver"
    log = solver.RoundsLog(out / "rounds.jsonl")
    session = (log.latest_session() if resume else None) or f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    if not resume:
        save_invocation(args, out)
    run_mode_plan = getattr(args, "_run_mode_plan", None)      # --mode (n64dk.runmodes); None: unchanged
    if run_mode_plan is not None:
        run_mode_plan.write(out, "entered")
    observations = solver.Observations(solver_dir / "observations.json")
    if not resume:
        observations.cells, observations.facts, observations.evidence, observations.plateaus = {}, {}, {}, {}
    budget = solver.Budget(solver.parse_duration(getattr(args, "budget_time", None)),
                           getattr(args, "budget_builds", None), getattr(args, "budget_model_calls", None))
    resource_plan = plan_jobs(args.jobs, reserve_mb=args.memory_reserve_mb or None,
                              memory_per_job_mb=args.worker_memory_mb or None)
    jobs = resource_plan.jobs
    started = time.time()

    def emit(kind: str, **fields) -> None:
        with events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": round(time.time(), 3), "event": kind, **fields}) + "\n")

    report = {"project": str(project), "images": {}, "resources": resource_plan.as_dict(),
              "ablation": ablation_record(args, mode),
              **({"run_mode": run_mode_plan.summary()} if run_mode_plan is not None else {})}
    emit("resource-plan", **resource_plan.as_dict())
    if switched:
        emit("ablation", disabled=report["ablation"]["disabled"],
             techniques_removed=report["ablation"]["techniques_removed"],
             cascade_removed=report["ablation"]["cascade_removed"])
    # --- 0. discovery: is each catalogue complete enough to decompile against?  An incomplete one
    # is repaired from call, pointer and prologue evidence into the run directory (never the
    # project), and the report says so: a percentage over a catalogue that misses code is not one.
    repaired: dict[str, Path] = {}
    placed: dict = {}                    # segments the discovery stage placed from the ROM
    names = list(names)
    discovery_mode = getattr(args, "discovery", None) or "repair"
    if discovery_mode != "off" and "discovery" not in args.skip_stage:
        from .discovery import run_audit
        try:
            with profiler.stage("discovery", "*"):
                found = run_audit(project, names, out / "discovery", repair=discovery_mode == "repair")
            report["discovery"] = discovery_summary(found)
            repaired = {name: Path(img["repair"]["catalogue"]) for name, img in found["images"].items()
                        if (img.get("repair") or {}).get("catalogue")}
            for name, info in (found.get("extra_images") or {}).items():
                placed[name] = imagemod.ImageSpec(name, Path(info["bin"]), int(info["vram"], 16),
                                                  Path(info["catalogue"]))
                if args.all_images or not args.image:
                    names.append(name)
        except Exception as error:                           # noqa: BLE001 - reported; the run goes on
            report["discovery"] = {"error": f"{type(error).__name__}: {error}"[-300:]}
        emit("stage", image="*", stage="discovery", result=report["discovery"])

    # --- 0b. small data: the program's _gp from its own startup code (sites that set $gp, all
    # agreeing, nonzero).  Declared to the resolver only through a symbol file in the run
    # directory; a project that declares _gp itself keeps its value.  --small-data off skips it.
    small_data_symbols: list[str] = []
    gp_found = None                      # the inferred _gp's record, kept apart from the declared one
    if (getattr(args, "small_data", None) or "auto") != "off":
        from ..analysis import gp as gpmod
        try:
            found_gp = gp_found = gpmod.for_project(project)
            small_data_symbols = write_gp_symbols(out, gpmod.symbol_line(found_gp))
            report["small_data"] = {k: found_gp[k] for k in ("value", "confidence", "reason")}
            report["small_data"]["sites"] = len(found_gp["sites"])
            # a project that declares _gp itself keeps its value (the resolver adds the recovered one
            # only when absent); a disagreement is reported, never resolved silently.  A symbol file
            # that cannot be read is reported beside the recovered value, never instead of it.
            try:
                declared = project_gp(project, names[0] if names else "main")
            except Exception as error:                       # noqa: BLE001 - reported; the value stands
                declared = None
                report["small_data"]["project_gp_error"] = f"{type(error).__name__}: {error}"[-300:]
            if declared is not None:
                report["small_data"]["project_gp"] = f"0x{declared:08X}"
                if found_gp.get("value") and int(found_gp["value"], 16) != declared:
                    report["small_data"]["warning"] = (f"the project declares _gp = 0x{declared:08X}; the code that "
                                                       f"sets $gp says {found_gp['value']}: the project's value is used")
            # declared and inferred _gp tracked apart (n64dk.match.ownership.gp_provenance): agree,
            # declared-only, inferred-only, inferred-low, conflict or none
            from ..match.ownership import gp_provenance
            report["small_data"]["provenance"] = ("unknown" if "project_gp_error" in report["small_data"]
                                                  else gp_provenance(declared, found_gp)["status"])
        except Exception as error:                           # noqa: BLE001 - reported; the run goes on
            report["small_data"] = {"error": f"{type(error).__name__}: {error}"[-300:]}
        emit("stage", image="*", stage="small-data", result=report["small_data"])
    if not small_data_symbols:
        write_gp_symbols(out, None)          # --small-data off, or the stage failed: no earlier run's value

    def resolve_spec(name: str):
        if name in placed:
            return placed[name]
        spec = imagemod.resolve(project, name)
        if name in repaired:
            import dataclasses
            spec = dataclasses.replace(spec, catalogue=repaired[name], _rows=None)
        return spec
    # a library of earlier runs: every other title in it is a donor, and this run joins it at the end
    from ..match import library as libmod
    library = Path(args.library).expanduser().resolve() if getattr(args, "library", None) else None
    first = imagemod.resolve(project, "main" if "main" in registry else sorted(registry)[0])
    title_key = libmod.rom_key(Path(first.path).read_bytes())
    specs_seen = []
    # every function start of the title, whichever image it is in: a unit may call across images
    title_starts: set[int] = set()
    for name in sorted(set(registry) | set(placed)):
        try:
            title_starts.update(vram for _n, vram, _s in resolve_spec(name).rows())
        except Exception:                                    # noqa: BLE001 - an image without a catalogue
            pass
    small_lane_exact: dict[str, int] = {}        # per image: exact cases built in a small-data lane
    image_spans: dict[str, tuple[int, int]] = {}  # every image's VRAM range, for the data stage (lazily)
    for name in names:
        spec = resolve_spec(name)
        settings = _settings(project, spec.name)
        folder = out / spec.name
        rows = spec.rows()
        data, delta = spec.load(), spec.delta
        emit("image-start", image=spec.name, functions=len(rows))
        image_report: dict = {"functions": len(rows),
                              # what the image ran with (command line > [run] > the run's starting mode)
                              "effective_settings": {key: solver.setting(args, settings, key, mode)
                                                     for key in solver.SETTING_FIELDS}}
        manifests = [str(m) for m in spec.manifests]
        stage_identity = _identity(image=spec.name, image_digest=hashlib.sha256(data).hexdigest(),
                                   delta=delta, rows=rows, manifests=manifests, settings=settings,
                                   min_confidence=args.min_confidence,
                                   **({"switched_off": switched} if switched else {}))   # all-on: unchanged
        recipes = {str((project / k).resolve()): v for k, v in (settings.get("recipes") or {}).items()}

        # --- 1. boundaries (a report, and the corrections that are certain)
        def boundaries():
            from ..analysis import boundaries as boundmod
            from ..match import byteledger as ownmod
            # Manifest rows are ownership claims, not rebuild receipts.  The unattended boundary
            # audit is report-only and must not label those ranges as independently verified.
            verified = boundmod.Verified([])
            operations = boundmod.solve(boundmod.Image(data, delta), rows, verified=verified)
            new_rows, applied, skipped = boundmod.apply(rows, operations, args.min_confidence)
            folder.mkdir(parents=True, exist_ok=True)
            boundmod.dump_operations(folder / "boundary_operations.json", operations, image=spec.name)
            kinds: dict[str, int] = {}
            for op in operations:
                kinds[op.get("kind", "?")] = kinds.get(op.get("kind", "?"), 0) + 1
            result = {"operations": kinds, "would_apply": len(applied), "functions_after": len(new_rows),
                      "file": str(folder / "boundary_operations.json")}
            # The operations the solver calls certain (an unreachable epilogue cut off its function, a
            # stray word in front of one) are applied to this run's catalogue; the project's is not
            # touched.  Whatever they change is then proved or refuted by the strict comparison like
            # any other function: a wrong correction can only fail to match.
            certain_rows, certain = boundmod.apply_certain(rows, operations)
            if certain and spec.catalogue:
                path = out / "discovery" / f"{spec.name}.boundaries{Path(spec.catalogue).suffix}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.unlink(missing_ok=True)
                boundmod.write_catalogue(path, boundmod.shift_rows(certain_rows, spec.catalogue_shift), spec.catalogue)
                result.update(certain_applied=len(certain), catalogue=str(path),
                              certain=[{"kind": op["kind"], "remove": op["remove"], "insert": op["insert"],
                                        "why": op["certain"]} for op in certain])
            return result
        if "boundaries" not in args.skip_stage:
            try:
                image_report["boundaries"], _cached = _stage(folder, "boundaries", _profiled(profiler, "boundaries", spec.name, boundaries), args.redo,
                                                               stage_identity)
            except Exception as error:                       # a stage failure is reported, not fatal
                image_report["boundaries"] = {"error": str(error)[-300:]}
            corrected = (image_report.get("boundaries") or {}).get("catalogue")
            if corrected and Path(corrected).is_file():
                import dataclasses
                repaired[spec.name] = Path(corrected)        # every later reading of this image's catalogue
                spec = dataclasses.replace(spec, catalogue=Path(corrected), _rows=None)
                rows = spec.rows()
                image_report["functions"] = len(rows)
                stage_identity = _identity(base=stage_identity, catalogue=hashlib.sha256(
                    Path(corrected).read_bytes()).hexdigest())
                emit("stage", image=spec.name, stage="boundaries",
                     certain_applied=image_report["boundaries"].get("certain_applied"))
            emit("stage", image=spec.name, stage="boundaries", result=image_report["boundaries"])

        if name in repaired and Path(repaired[name]).is_file():
            # the catalogue this image ran with is not the project's: the export lists the run's functions
            imagemod.note_run_catalogue(out, spec.name, repaired[name])

        # --- 2. donors
        recovered: set[int] = set()
        library_identity = libmod.identity(library, title_key)
        if run_mode_plan is not None:            # only the library entries the run's mode admitted
            library_identity = run_mode_plan.library_identity(library_identity)
        if (args.donors or library_identity) and "donors" not in args.skip_stage:
            def donors():
                from ..match import transplant
                from ..match.engine import Target
                explicit = (transplant.load_donors(args.donors, args.donor or None, args.exclude_donor or None)
                            if args.donors else [])
                if run_mode_plan is not None:    # enforced here too: a donor not admitted at the start fails
                    run_mode_plan.check_explicit(explicit)
                    specs = explicit + run_mode_plan.library_donors(library, title_key)
                else:
                    specs = explicit + libmod.donors(library, title_key)
                index, _stats = transplant.build_index(specs, args.donor_index or out / "donor_index.json")
                target = Target(data, delta, (0, len(data)))
                skip = [(v, v + s) for v, s in ((r.vram, r.size) for m in manifests
                                                for r in _rows_of(m, delta))]
                from ..match.toolchain import load_config
                toolchain = args.toolchain_dir or os.environ.get("N64DK_TOOLCHAIN") or \
                    load_config(project, None, None).toolchain_root
                result = transplant.run(None, target, rows, index, skip=skip, out_dir=folder / "donors",
                                        jobs=min(jobs, 4), toolchain_root=Path(toolchain) if toolchain else None,
                                        other_starts=title_starts - {v for _n, v, _s in rows},
                                        rodata_settlement=enabled(args, "rodata-settlement"),
                                        target_symbols=project_symbols(project, spec.name))
                keep = {k: v for k, v in result.items()
                        if not k.startswith("_") and (not isinstance(v, (list, dict)) or k in ("verdicts", "timing_s", "rodata_settlement"))}
                by_name = {n: v for n, v, _s in rows}
                keep["verified"] = [{"functions": r.get("functions", []), "vram": r.get("target_vram"),
                                     "size": r.get("size"), "donor": r.get("donor"), "unit": r.get("donor_unit"),
                                     "source_kind": ("assembly" if Path(str(r.get("donor_unit", ""))).suffix.lower()
                                                     in (".s", ".asm") else "c"),
                                     # the functions that matched a donor function, and every other
                                     # catalogued function wholly inside the bytes the unit was compared over
                                     "function_vrams": sorted(
                                         {by_name[n] for n in r.get("functions", []) if n in by_name}
                                         | set(transplant.functions_inside(int(str(r.get("target_vram")), 16),
                                                                           r.get("size") or 0, rows, data, delta)))}
                                    for r in result.get("rows", [])
                                    if r.get("verdict") == transplant.EXACT_VERDICT]
                keep["assumed"] = [{"functions": r.get("functions", []), "vram": r.get("target_vram"),
                                    "size": r.get("size"), "donor": r.get("donor"), "unit": r.get("donor_unit"),
                                    "source_kind": ("assembly" if Path(str(r.get("donor_unit", ""))).suffix.lower()
                                                    in (".s", ".asm") else "c"),
                                    "awaiting": r.get("awaiting") or {},
                                    "function_vrams": [v for _n, v, _s in rows
                                                       if int(str(r.get("target_vram")), 16) <= v
                                                       < int(str(r.get("target_vram")), 16) + (r.get("size") or 0)]}
                                   for r in result.get("rows", [])
                                   if r.get("verdict") == transplant.ASSUMED_VERDICT]
                return keep
            try:
                image_report["donors"], _cached = _stage(folder, "donors", _profiled(profiler, "donors", spec.name, donors), args.redo,
                                                          _identity(base=stage_identity, donors=args.donors,
                                                                    selected=args.donor,
                                                                    excluded=args.exclude_donor,
                                                                    library=library_identity))
                # Only a verified unit takes its functions out of the resolve stage.  A unit that is
                # byte-identical but still waits for an address recovers nothing yet: its functions are
                # resolved like any other, and one the resolver proves strictly is credited to that proof
                # (measured: eight functions of a title, exact from drafts without donors, were exported
                # as waiting once a donor unit covered them).
                for row in image_report["donors"].get("verified", []):
                    recovered.update(int(v) for v in row.get("function_vrams", []))
            except Exception as error:
                image_report["donors"] = {"error": str(error)[-300:]}
            emit("stage", image=spec.name, stage="donors",
                 verified=len(image_report["donors"].get("verified", [])) if isinstance(image_report["donors"], dict) else 0)

        # --- 3. resolve
        resolve_options = resolve_options_for(args, settings, mode=mode, project=project, spec=spec, jobs=jobs,
                                              events=events, out=out, manifests=manifests, recipes=recipes,
                                              recovered=recovered, small_data_symbols=small_data_symbols)

        plan = solver.stage_plan(mode, args, settings)
        held = {"overrides": {}}

        def runtime_helpers(data=data, delta=delta, rows=rows) -> dict:
            # compiler support routines found by their own code (once per image)
            if "runtime_helpers" not in held:
                from ..match import runtimehelpers

                def read(address: int):
                    offset = address - delta
                    return int.from_bytes(data[offset:offset + 4], "big") if 0 <= offset <= len(data) - 4 else None
                held["runtime_helpers"] = runtimehelpers.helpers(read, [v for _n, v, _s in rows])
            return held["runtime_helpers"]
        if "resolve" in plan:
            from .resolve import resolve_hard
            round1 = solver_dir / f"round1_{spec.name}.json"
            last, stopped = log.progress(session, spec.name) if resume else (0, None)

            def sweep():
                """Round 1: today's decompile-run resolve stage, unchanged (calibrate, sweep, declare)."""
                chosen_recipe = None
                # --- 3a. calibrate: measure the title's recipe on a sample of its own functions
                sample = args.calibrate if args.calibrate is not None else int(settings.get("calibrate", 0) or 0)
                if sample == 0 and "calibrate" not in args.skip_stage and settings.get("pipeline"):
                    # No calibration asked for: the configured recipe is taken on trust.  When the code's
                    # own marks say most of the image was built another way, measure instead.
                    try:
                        from ..match import idioms
                        table = idioms.Table.load()
                        marked = [(vram, idioms.features(data[vram - delta:vram - delta + size], vram))
                                  for _name, vram, size in rows if size >= 8]
                        found = idioms.doubt(marked, table, f"{settings['pipeline']} {settings.get('flags', '')}")
                        if found["share"] >= RECIPE_DOUBT:
                            image_report["recipe_doubt"] = found
                            sample = DOUBT_SAMPLE
                            emit("stage", image=spec.name, stage="recipe-doubt", share=found["share"])
                    except Exception:                            # noqa: BLE001 - a hint, never a reason to stop
                        pass
                # command line, else [run] regional_calibration (false turns exhaustive's off), else
                # the mode the run was started with
                regional = solver.setting(args, settings, "regional_calibration", mode)
                if sample > 0 and "calibrate" not in args.skip_stage and regional:
                    def calibrate_by_region():
                        return calibrate_regions(resolve_hard, resolve_options, out / "state", spec.name, rows,
                                                 recovered, sample, delta, data, emit)
                    try:
                        image_report["calibrate_regions"], _cached = _stage(
                            folder, "calibrate_regions", _profiled(profiler, "calibrate-regions", spec.name,
                                                                   calibrate_by_region), args.redo,
                            _identity(base=stage_identity, sample=sample, donors=sorted(recovered)))
                        held["region_map"] = image_report["calibrate_regions"].get("map") or []
                    except Exception as error:
                        image_report["calibrate_regions"] = {"error": str(error)[-300:]}
                if sample > 0 and "calibrate" not in args.skip_stage:
                    def calibrate():
                        return calibrate_recipe(resolve_hard, resolve_options, out / "state", spec.name, rows, recovered,
                                                sample, delta)
                    try:
                        image_report["calibrate"], _cached = _stage(
                            folder, "calibrate", _profiled(profiler, "calibrate", spec.name, calibrate), args.redo,
                            _identity(base=stage_identity, sample=sample, donors=sorted(recovered)))
                        chosen_recipe = image_report["calibrate"].get("chosen")
                    except Exception as error:
                        image_report["calibrate"] = {"error": str(error)[-300:]}
                    emit("stage", image=spec.name, stage="calibrate",
                         chosen=(chosen_recipe or {}).get("label"))
                overrides = held["overrides"] = {}
                if chosen_recipe:
                    overrides = held["overrides"] = {"pipeline": chosen_recipe["pipeline"], "flags": chosen_recipe["flags"]}
                if held.get("region_map"):
                    overrides = held["overrides"] = {**overrides, "region_map": held["region_map"]}
                try:
                    call = dict(overrides)
                    if resume:
                        # Round 1 was cut short: a case the interrupted sweep finished is not swept
                        # again (that would continue it from its stored source, a second and different
                        # pass, and repeat builds).  Calibration samples still marked provisional, and
                        # functions without a case, are swept as an uninterrupted run would sweep them.
                        finished = finished_cases(out / "state", spec.name)
                        if finished:
                            # applied after the selection (size window, --limit), so the resumed
                            # sweep covers exactly the functions the original one selected
                            call["resume_finished"] = finished
                            image_report["resume_skipped_finished"] = len(finished)
                    image_report["resolve"] = _profiled(profiler, "resolve", spec.name,
                                                        lambda: resolve_hard(resolve_options(**call)))()
                except Exception as error:
                    image_report["resolve"] = {"error": str(error)[-300:]}
                # --- 3b. declare: settle byte-identical cases that only wait for an address to be declared
                if (args.declare_assumed or settings.get("declare_assumed")) and enabled(args, "declare") \
                        and "error" not in image_report["resolve"]:
                    try:
                        image_report["declare"] = _profiled(profiler, "declare", spec.name, lambda: declare_assumed(
                            resolve_hard, resolve_options, out / "state", spec.name, folder, overrides,
                            (held.setdefault("declare_verified", {}) if os.environ.get("N64DK_DECLARE_MEMO") == "1" else None),
                            runtime_helpers=runtime_helpers()))()
                        moved = image_report["declare"].get("promoted", 0)
                        statuses = image_report["resolve"].get("statuses")
                        if moved and isinstance(statuses, dict):
                            statuses["exact"] = statuses.get("exact", 0) + moved
                            statuses["assumed"] = max(0, statuses.get("assumed", 0) - moved)
                    except Exception as error:
                        image_report["declare"] = {"error": str(error)[-300:]}
                    emit("stage", image=spec.name, stage="declare", promoted=image_report["declare"].get("promoted"))

                solver.write_json(round1, {**{key: image_report.get(key) for key in
                                              ("recipe_doubt", "calibrate", "resolve", "declare") if key in image_report},
                                           "overrides": held["overrides"]})
                return {"model_calls": int(((image_report.get("resolve") or {}).get("refine") or {}).get("calls") or 0),
                        "declared": isinstance(image_report.get("declare"), dict)
                        and "error" not in image_report["declare"]}

            if last >= 1 and round1.is_file():               # a resume: round 1 is not redone
                saved = json.loads(round1.read_text())
                held["overrides"] = saved.pop("overrides", None) or {}
                image_report.update(saved)

            def technique(name, decisions, calls_left):
                """One later-round technique on a named set of stored cases (n64dk.match.solver)."""
                options = resolve_options(**held["overrides"], **solver.technique_options(name, mode),
                                          only_names=[",".join(sorted({d.name for d in decisions}))])
                if name == "refine":
                    if options.refine is None:
                        return {"error": "no refiner configured"}
                    if calls_left is not None:
                        cap = max(1, calls_left // max(1, options.refine.rounds))
                        options.refine.max_cases = min(options.refine.max_cases or cap, cap)
                else:
                    options.refine = None                    # a build technique never asks the refiner
                result = _profiled(profiler, f"round:{name}", spec.name, lambda: resolve_hard(options))()
                return {"model_calls": int(((result.get("refine") or {}).get("calls")) or 0),
                        "error": result.get("error")}

            def declare():
                # profiled as a round stage, so the unresolved dossier has its seconds per build
                result = _profiled(profiler, "round:declare", spec.name, lambda: declare_assumed(
                    resolve_hard, resolve_options, out / "state", spec.name, folder, held["overrides"],
                    (held.setdefault("declare_verified", {}) if os.environ.get("N64DK_DECLARE_MEMO") == "1" else None),
                    runtime_helpers=runtime_helpers()))()
                listing = folder / "declared_symbols.txt"
                from ..match.datastage import DATA_STAGE_MARK
                names = [line.split(" = ")[0] for line in listing.read_text().splitlines()
                         if " = " in line and DATA_STAGE_MARK not in line] if listing.is_file() else []
                image_report.setdefault("declare_rounds", []).append(result)
                return {**result, "names": names}

            graph: dict[int, set[int]] = {}

            def neighbours(vram: int) -> set[int]:
                if not graph:                                  # rows and cases are both in link space
                    graph.update(solver.call_graph(data, delta, rows) or {0: set()})
                return graph.get(vram, set())

            ceiling = (args.search_ceiling if args.search_ceiling is not None
                       else settings.get("search_ceiling", resolvemod.SEARCH_CEILING))
            model = bool(mode.refine and refine_settings(settings, args) is not None)
            finished = stopped is not None and str(stopped.get("reason", "")).startswith(
                ("fixed point", "nothing applicable"))
            if finished:
                image_report["solver_stop"] = stopped
            else:
                image_report["solver_stop"] = solver.solve_image(
                    image=spec.name, store_root=out / "state", mode=mode, budget=budget, log=log,
                    observations=observations, hooks=solver.Hooks(sweep, technique, declare, neighbours),
                    session=session, rounds=args.rounds, start_round=last + 1,
                    unknown=lambda: solver.Stretches((image_report.get("resolve") or {}).get("unknown_toolchain")),
                    ceiling=ceiling, model=model, emit=emit)

        # --- 3b'. data (opt-in, --data-stage report|replay): every data reference that keeps a case or
        #     a donor unit waiting, grouped by symbol, section and image, placed only from evidence
        #     independent of the unit that needs it (n64dk.match.datastage); replay re-verifies strictly
        #     the cases whose every data placement is established
        data_mode = getattr(args, "data_stage", None) or "off"
        if data_mode != "off":
            if not image_spans:
                for other in sorted(set(registry) | set(placed)):
                    try:
                        image_spans[other] = tuple(resolve_spec(other).span())
                    except Exception:                        # noqa: BLE001 - an image without a range
                        continue
            try:
                image_report["data"] = _profiled(profiler, "data", spec.name, lambda: data_stage(
                    out, folder, spec, project, data=data, delta=delta, rows=rows, manifests=manifests,
                    gp_found=gp_found, homes={k: v for k, v in image_spans.items() if k != spec.name},
                    redo=args.redo, base_identity=stage_identity))()
            except Exception as error:                       # noqa: BLE001 - a report, never the run's failure
                image_report["data"] = {"observation_error": f"{type(error).__name__}: {error}"[-300:]}
            emit("stage", image=spec.name, stage="data",
                 groups=(image_report["data"].get("summary") or {}).get("groups_by_status"))
            if data_mode == "replay" and "resolve" in plan and image_report["data"].get("replay_units"):
                from .resolve import resolve_hard
                try:
                    image_report["data_replay"] = _profiled(profiler, "data-replay", spec.name, lambda: replay_data(
                        resolve_hard, resolve_options, folder, held["overrides"]))()
                    moved = image_report["data_replay"].get("promoted", 0)
                    statuses = (image_report.get("resolve") or {}).get("statuses")
                    if moved and isinstance(statuses, dict):
                        statuses["exact"] = statuses.get("exact", 0) + moved
                        statuses["assumed"] = max(0, statuses.get("assumed", 0) - moved)
                except Exception as error:
                    image_report["data_replay"] = {"error": f"{type(error).__name__}: {error}"[-300:]}
                emit("stage", image=spec.name, stage="data-replay", promoted=image_report["data_replay"].get("promoted"))

        # --- 3c. solver outputs (read the store, build nothing): coverage, primary causes, dossiers
        if "resolve" in plan:
            try:
                image_report["solver"] = solver_outputs(
                    out, spec.name, mode, observations, rows=rows, data=data, delta=delta, recovered=recovered,
                    manifest_rows=[(r.vram, r.size) for m in manifests for r in _rows_of(m, delta)],
                    stretches=solver.Stretches((image_report.get("resolve") or {}).get("unknown_toolchain")),
                    ceiling=ceiling, model=model, stop=image_report.get("solver_stop") or {},
                    boundary_file=folder / "boundary_operations.json")
            except Exception as error:                       # an observation failure never costs the run
                image_report["solver"] = {"observation_error": f"{type(error).__name__}: {error}"[-300:]}

        # --- 3d. toolchain regions (builds nothing): ranked recipes per region, component mismatches,
        #     abstention and dossiers, with the run's exact cases as proof
        if "resolve" in plan:
            try:
                from ..match import toolchain_regions as tcmod
                store = resolvemod.CaseStore(out / "state", readonly=True)
                try:
                    image_cases = [case for case in store.all() if case.image == spec.name]
                finally:
                    store.close()
                from ..match import transform_classes as transforms
                tally: dict[str, int] = {}
                for case in image_cases:
                    if case.status == "exact":
                        found = transforms.provenance(case.stages)
                        key = found["class"] or "as drafted"
                        tally[key] = tally.get(key, 0) + 1
                        tally["with frame scaffolding"] = tally.get("with frame scaffolding", 0) + found["scaffolding"]
                image_report["exact_provenance"] = tally
                regions = tcmod.reconstruct(data, delta, rows, cases=image_cases)
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "toolchain_regions.json").write_text(json.dumps(regions, indent=1) + "\n")
                image_report["toolchain_regions"] = {**(regions.get("summary") or {}),
                                                     "file": str(folder / "toolchain_regions.json")}
            except Exception as error:                       # noqa: BLE001 - an analysis, never the run's failure
                image_report["toolchain_regions"] = {"observation_error": f"{type(error).__name__}: {error}"[-300:]}

        # --- 4. strict queue
        store = resolvemod.CaseStore(out / "state")
        try:
            image_report["queue"] = resolvemod.export_queue(store, out / "queue", spec.name, not args.strict_only)
            image_cases = [c for c in store.all() if c.image == spec.name]
            image_report["hard_queue"] = [
                {k: g[k] for k in ("cause", "functions", "bytes")}
                for g in resolvemod.hard_queue(image_cases)["queue"]]
            small_lane_exact[spec.name] = small_data_lane_exact(image_cases)
        finally:
            store.close()

        # --- 5. ledger
        if "ledger" not in args.skip_stage:
            def ledger():
                from ..match import byteledger as ownmod
                manifest_rows = []
                for manifest in manifests:
                    cls = "asm" if "asm" in Path(manifest).name.lower() else "c"
                    manifest_rows += ownmod.read_manifest(manifest, delta, cls)
                built = ownmod.build(ownmod.ImageInfo.of(spec.name, data, delta), rows, manifest_rows, data=data)
                violations = built.check()
                folder.mkdir(parents=True, exist_ok=True)
                built.to_tsv(folder / "ownership.tsv")
                kinds: dict[str, int] = {}
                for violation in violations:
                    kinds[violation.kind] = kinds.get(violation.kind, 0) + 1
                return {"totals": built.totals(), "violations": kinds, "file": str(folder / "ownership.tsv")}
            try:
                image_report["ledger"], _cached = _stage(folder, "ledger", _profiled(profiler, "ledger", spec.name, ledger), args.redo, stage_identity)
            except Exception as error:
                image_report["ledger"] = {"error": str(error)[-300:]}
        report["images"][spec.name] = image_report
        specs_seen.append(spec)
        emit("image-done", image=spec.name)
    note_small_data(report.get("small_data"), sum(small_lane_exact.values()), bool(small_data_symbols))
    # The tables under queue/ are rewritten by each image's export: one more export, of every
    # image, leaves them complete (and the library below describes every image, not the last).
    if len(specs_seen) > 1:
        store = resolvemod.CaseStore(out / "state")
        try:
            report["queue"] = {k: v for k, v in resolvemod.export_queue(
                store, out / "queue", None, not args.strict_only).items() if k != "skipped"}
        except Exception as error:
            report["queue"] = {"error": str(error)[-300:]}
        finally:
            store.close()
    # this run as a donor for other titles (strictly exact functions only)
    try:
        symbol_files = [project / s for s in (_settings(project, names[0]).get("symbols") or [])]
        report["donor"] = libmod.write_donor(out, project, specs_seen, title_key, symbol_files)
        if library is not None:
            report["donor"]["registered"] = libmod.register(library, out, title_key)
    except Exception as error:
        report["donor"] = {"error": str(error)[-300:]}
    # the unresolved dossier (n64dk.match.dossier): every function not settled with its reason, its
    # technique ledger, the eligible next techniques at their measured cost and the evidence that
    # would justify a retry.  Report-only: it reads the run's records and builds nothing.
    try:
        from ..match import dossier as dossiermod
        report["unresolved"] = dossiermod.write(out, mode, disabled=getattr(args, "disabled", None) or (),
                                                project=project)
    except Exception as error:                            # noqa: BLE001 - a report, never the run's failure
        report["unresolved"] = {"observation_error": f"{type(error).__name__}: {error}"[-300:]}
    report["seconds"] = round(time.time() - started, 1)
    report["profile"] = profiler.summary()
    if os.environ.get("N64DK_MEASURE_RSS") == "1":
        from .. import safety as safetymod
        report["tool_profile"] = str(safetymod.save_tool_profile())
    report["failed_stages"] = {name: _headline(item)["errors"] for name, item in report["images"].items()
                               if _headline(item)["errors"]}
    report["status"] = "incomplete" if report["failed_stages"] else "complete"
    report["observation_errors"] = {name: _headline(item)["observation_errors"] for name, item in report["images"].items()
                                    if _headline(item)["observation_errors"]}
    if getattr(args, "receipts", False):              # opt-in: a proof receipt per settled result
        try:                                              # (n64dk.match.proofreceipt; rebuilds each once)
            from .verifyreceipt import issue_receipts
            report["receipts"] = issue_receipts(out)
        except Exception as error:                        # noqa: BLE001 - a report, never the run's failure
            report["receipts"] = {"error": f"{type(error).__name__}: {error}"[-300:]}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    (out / "report.md").write_text(_markdown(report))
    return {"status": report["status"], "failed_stages": report["failed_stages"],
            "discovery": {k: v for k, v in (report.get("discovery") or {}).items() if k != "images"},
            "report": str(out / "report.md"), "events": str(events),
            "images": {name: _headline(item) for name, item in report["images"].items()},
            "seconds": report["seconds"], **({"switched_off": switched} if switched else {})}


def solver_outputs(out: Path, image: str, mode, observations, *, rows, data: bytes, delta: int, recovered,
                   manifest_rows, stretches, ceiling, model: bool, stop: dict, boundary_file: Path) -> dict:
    """Coverage of every catalogued function and a dossier for every unresolved one, from the
    case store (read-only).  Every address is in link space (``ImageSpec.rows``), as cases are."""
    store = resolvemod.CaseStore(out / "state", readonly=True)
    try:
        cases = [case for case in store.all() if case.image == image]
    finally:
        store.close()
    left_out = {}
    scope = out / "state" / f"scope_{image}.json"
    if scope.is_file():
        try:
            left_out = {int(row["vram"], 16): row["reason"] for row in json.loads(scope.read_text()).get("left_out", [])}
        except (ValueError, KeyError, TypeError):
            left_out = {}

    def covered(vram: int) -> bool:
        return any(start <= vram < start + size for start, size in manifest_rows)
    found = solver.coverage(rows, cases, recovered=recovered, left_out=left_out, covered=covered,
                            image_size=len(data), delta=delta)
    solver.write_json(out / "solver" / f"coverage_{image}.json", found)
    graph = solver.call_graph(data, delta, rows)
    order = sorted(vram for _n, vram, _s in rows)
    position = {vram: i for i, vram in enumerate(order)}
    exact = {case.vram: case.name for case in cases if case.status == "exact"}
    boundary = solver.boundary_operations(boundary_file)
    recipes = solver.recipe_confidence(cases)
    try:
        from ..match import toolchain_regions as tcmod
        reconstruction = tcmod.reconstruct(data, delta, rows, cases=cases)
    except Exception:                                        # noqa: BLE001 - dossier context, never the run's failure
        reconstruction = {}
    records, by_cause = [], {}
    for case in sorted(cases, key=lambda c: c.vram):
        if case.status not in solver.UNRESOLVED:
            continue
        decision = solver.decide(case, mode, observations, unknown=stretches, ceiling=ceiling, model=model)
        index = position.get(case.vram, -1)
        adjacent = {order[i] for i in (index - 1, index + 1) if 0 <= i < len(order)} if index >= 0 else set()
        related = sorted({(exact[v], "calls or is called by") for v in graph.get(case.vram, set()) if v in exact}
                         | {(exact[v], "adjacent in the catalogue") for v in adjacent if v in exact})
        records.append(solver.dossier(case, decision, observations,
                                      related=[{"name": n, "relation": r} for n, r in related],
                                      boundary=boundary.get(case.vram, []), stop=stop.get("reason") or "",
                                      image_recipes=recipes,
                                      region=tcmod.region_of(reconstruction, case.vram) if reconstruction else None))
        group = by_cause.setdefault(decision.cause, {"functions": 0, "bytes": 0, "next": {}})
        group["functions"] += 1
        group["bytes"] += case.size
        key = decision.technique or "stop"
        group["next"][key] = group["next"].get(key, 0) + 1
    solver.write_json(out / "solver" / f"hard_queue_{image}.json", records)
    ledger_path = out / "solver" / f"attempts_{image}.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ledger_path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        for entry in solver.attempts(cases, observations):
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    temporary.replace(ledger_path)
    sieve_table = solver.sieve(cases, recovered=len(recovered))
    warning = None
    if mode.require_reasons and found["unexplained"]:
        warning = (f"effort {mode.name} requires a recorded reason for every function not attempted: "
                   f"{found['unexplained']} have none (see coverage_{image}.json)")
    return {"mode": mode.name, "stop": stop.get("reason"), "after_round": stop.get("after_round"),
            "coverage_warning": warning,
            "coverage": {k: found[k] for k in ("functions", "attempted", "not_attempted", "unexplained", "reasons")},
            "remaining_by_cause": dict(sorted(by_cause.items(), key=lambda kv: -kv[1]["bytes"])),
            "sieve": sieve_table,
            "files": {"rounds": str(out / "rounds.jsonl"), "coverage": str(out / "solver" / f"coverage_{image}.json"),
                      "hard_queue": str(out / "solver" / f"hard_queue_{image}.json"),
                      "attempts": str(ledger_path)}}


def finished_cases(state: Path, image: str) -> set[int]:
    """Link-space addresses of the cases of ``image`` whose ladder finished (a case is stored only
    when its ladder ends), less calibration samples still marked provisional."""
    if not Path(state).is_dir():
        return set()
    store = resolvemod.CaseStore(state, readonly=True)
    try:
        return {case.vram for case in store.all() if case.image == image
                and not any(stage.get("stage") == "provisional" for stage in case.stages)}
    finally:
        store.close()


def discovery_summary(found: dict) -> dict:
    """Per image: the audit's verdict before and after repair, and what the repair added."""
    images = {}
    for name, img in found["images"].items():
        repair = img.get("repair") or {}
        images[name] = {"verdict": img.get("verdict"), "coverage": img.get("coverage"),
                        "functions": img.get("functions"), "reasons": img.get("reasons"),
                        "missing_call_targets": (img.get("direct_calls") or {}).get("missing"),
                        "unmapped_call_targets": (img.get("direct_calls") or {}).get("unmapped"),
                        "framed_functions_outside": (img.get("prologues_outside") or {}).get("walk_as_functions"),
                        "repaired_functions": repair.get("added", 0), "repaired_bytes": repair.get("bytes", 0),
                        "verdict_after": repair.get("verdict_after", img.get("verdict")),
                        "coverage_after": repair.get("coverage_after", img.get("coverage")),
                        "functions_after": repair.get("functions_after", img.get("functions")),
                        # what remains after repair (round 11: the summary used to carry only the
                        # pre-repair reasons, which overstated what was still missing)
                        "reasons_after": repair.get("reasons_after", img.get("reasons"))}
    verdict = found.get("verdict")
    return {"verdict_before": found.get("verdict_before", verdict), "verdict": verdict,
            "rom_reasons": found.get("rom_reasons"),
            "placed": [{k: p[k] for k in ("name", "rom", "rom_end", "vram", "landed", "functions")}
                       for p in found.get("placed") or []],
            "warning": (None if verdict != "incomplete" else
                        "the function catalogue is incomplete even after repair: every count and percentage in "
                        "this report is over the catalogue, not over the title's code"),
            "images": images}


def _rows_of(manifest: str, delta: int):
    from ..match import byteledger as ownmod
    try:
        return ownmod.read_manifest(manifest, delta, "c")
    except Exception:
        return []


def _headline(item: dict) -> dict:
    resolve = item.get("resolve") or {}
    donors = item.get("donors") or {}
    queue = item.get("queue") or {}
    return {"functions": item.get("functions"),
            "donor_verified": sum(len(r.get("function_vrams") or r.get("functions") or [])
                                  for r in donors.get("verified", [])),
            "donor_assumed": sum(len(r.get("function_vrams", [])) for r in donors.get("assumed", [])),
            "resolved": resolve.get("statuses"), "exact_bytes": resolve.get("exact_bytes"),
            "queued": queue.get("strict_exact"), "queued_strict_exact": queue.get("strict_exact"),
            "awaiting_symbols": queue.get("assumed"), "refine": resolve.get("refine"),
            "calibrate": item.get("calibrate"), "declare": item.get("declare"),
            "errors": [k for k, v in item.items() if isinstance(v, dict) and "error" in v],
            # an observation stage that failed (solver accounting, toolchain regions) changes no
            # result, but it must not fail silently either
            "observation_errors": [k for k, v in item.items() if isinstance(v, dict) and "observation_error" in v]}


def _markdown(report: dict) -> str:
    lines = [f"# decompile-run report", "", f"Project: `{report['project']}`  ", f"Wall time: {report.get('seconds')} s", ""]
    run_mode = report.get("run_mode") or {}
    if run_mode.get("mode"):
        lines += [f"Run mode: {run_mode['mode']} (forbidden trees: {', '.join(run_mode.get('forbidden_trees') or []) or 'none'}; "
                  f"recorded trees: {', '.join(run_mode.get('recorded_trees') or []) or 'none'}; library entries left "
                  f"out: {len(run_mode.get('library_excluded') or [])}); see solver/run_mode.json", ""]
    ablation = report.get("ablation") or {}
    if ablation.get("switched_off"):
        # an ablation run says so first: its counts are not the all-on run's
        lines += ["ABLATION RUN: switched off " + ", ".join(f"{name} ({source})" for name, source in
                                                       sorted((ablation.get("disabled") or {}).items()))
                  + "".join(f"; effort mode without {label} {', '.join(items)}" for label, items in
                            (("techniques", ablation.get("techniques_removed")),
                             ("cascade steps", ablation.get("cascade_removed"))) if items), ""]
    found = report.get("discovery") or {}
    if found.get("verdict"):
        lines.append(f"Discovery: catalogue {found['verdict']}"
                     + (f" (before repair: {found['verdict_before']})" if found.get("verdict_before") != found["verdict"] else ""))
        for name, img in (found.get("images") or {}).items():
            lines.append(f"- {name}: {img['verdict_after']}, coverage {img['coverage_after']}"
                         + (f", +{img['repaired_functions']} functions from call/pointer/prologue evidence"
                            if img.get("repaired_functions") else ""))
            if img.get("verdict_after") != "complete":
                lines += [f"  - {reason}" for reason in (img.get("reasons_after") or [])[:4]]
        if found.get("warning"):
            lines.append(f"- WARNING: {found['warning']}")
        lines.append("")
    small = report.get("small_data") or {}
    if small.get("gp_source") in ("recovered", "project", "unknown"):
        source = {"recovered": f"{small.get('value')}, recovered from the code that sets $gp "
                               "(`small_data/gp_symbols.txt`; the project does not declare it)",
                  "project": f"{small.get('project_gp')}, declared by the project",
                  "unknown": f"{small.get('value')} recovered from code; whether the project declares its own is "
                             "unknown (its symbol files could not be read)"}[small["gp_source"]]
        lines += [f"Small data: _gp {source}; {small.get('exact_small_data_lane', 0)} exact functions built in a "
                  "small-data lane were verified against it", *([f"- WARNING: {small['warning']}"]
                                                                if small.get("warning") else []), ""]
    for name, item in report["images"].items():
        head = _headline(item)
        lines += [f"## {name}", "", f"- catalogued functions: {head['functions']}",
                  f"- recovered from donors (verified): {head['donor_verified']}",
                  f"- identical to a donor unit, awaiting symbols: {head.get('donor_assumed') or 0} "
                  "(`donors/awaiting_symbols.txt`)",
                  f"- resolution outcomes this run: {head['resolved']}",
                  f"- exact bytes from resolution: {head['exact_bytes']}",
                  f"- queued for integration (strictly exact): {head['queued_strict_exact']}",
                  f"- byte-identical, awaiting symbol declarations: {head.get('awaiting_symbols') or 0} "
                  "(`queue/assumed.tsv`; review `queue/symbols.assumed.txt`, add it to the symbol file, rerun)"]
        if head.get("calibrate"):
            info = head["calibrate"]
            picked = (info.get("chosen") or {}).get("label")
            lines.append(f"- recipe calibration: {info.get('identical')} of {info.get('sample')} sample functions "
                         f"byte-identical; " + (f"sweep recipe `{picked}`" if picked
                                               else f"configured recipe kept ({info.get('reason')})"))
        if head.get("refine"):
            info = head["refine"]
            lines.append(f"- refiner: asked about {info.get('attempted')} cases in {info.get('calls')} calls; "
                         f"{info.get('improved')} improved, {info.get('exact')} became exact, "
                         f"{info.get('assumed')} byte-identical awaiting symbols, {info.get('errors')} call errors")
        if item.get("exact_provenance"):
            lines.append(f"- exact functions by what their source went through (n64dk.match.transform_classes): "
                         f"{item['exact_provenance']}")
        if (item.get("toolchain_regions") or {}).get("functions_by_verdict"):
            lines.append(f"- toolchain regions: {item['toolchain_regions']['functions_by_verdict']} "
                         f"(`{item['toolchain_regions'].get('file')}`)")
        if item.get("boundaries"):
            lines.append(f"- boundary operations proposed: {item['boundaries'].get('operations')}")
        if item.get("ledger") and "totals" in item["ledger"]:
            lines.append(f"- ledger violations: {item['ledger'].get('violations')}")
        if isinstance(item.get("data"), dict) and item["data"].get("summary"):
            info = item["data"]["summary"]
            lines.append(f"- data stage: {info.get('groups')} groups of unresolved data references "
                         f"{info.get('groups_by_status')}; units {info.get('units_by_replay')} "
                         f"(`{item['data'].get('file')}`; placements are hypotheses until verified strictly)")
        if isinstance(item.get("data_replay"), dict) and "units" in item["data_replay"]:
            info = item["data_replay"]
            lines.append(f"- data replay: {info.get('promoted', 0)} of {info.get('units', 0)} re-verified cases became "
                         f"exact (strict verifier); {info.get('already_replayed', 0)} replayed before")
        solved = item.get("solver") or {}
        if solved.get("mode"):
            lines.append(f"- effort `{solved['mode']}`: stopped after round {solved.get('after_round')} "
                         f"({solved.get('stop')}); rounds in `rounds.jsonl`, dossiers in "
                         f"`solver/hard_queue_{name}.json`")
            cover = solved.get("coverage") or {}
            if solved.get("coverage_warning"):
                lines.append(f"- WARNING: {solved['coverage_warning']}")
            lines.append(f"- coverage: {cover.get('attempted')} of {cover.get('functions')} catalogued functions "
                         f"attempted, {cover.get('not_attempted')} not attempted (each with a recorded reason), "
                         f"{cover.get('unexplained')} unexplained")
            if solved.get("sieve"):
                lines += ["", "| sieve stage | settled | builds |", "|---|---:|---:|"]
                lines += [f"| {name} | {row['settled']} | {row['builds']} |"
                          for name, row in solved["sieve"]["stages"].items() if row["settled"] or row["builds"]]
                lines.append(f"| unresolved | {solved['sieve']['unresolved']} | |")
            if solved.get("remaining_by_cause"):
                lines += ["", "| primary cause | functions | bytes | next action |", "|---|---:|---:|---|"]
                lines += [f"| {cause} | {g['functions']} | {g['bytes']} | "
                          + ", ".join(f"{k} {v}" for k, v in sorted(g["next"].items(), key=lambda kv: -kv[1]))
                          + " |" for cause, g in solved["remaining_by_cause"].items()]
        if item.get("hard_queue"):
            lines += ["", "| remaining cause | functions | bytes |", "|---|---:|---:|"]
            lines += [f"| {g['cause']} | {g['functions']} | {g['bytes']} |" for g in item["hard_queue"]]
        if head["errors"]:
            lines.append(f"- stages with errors: {head['errors']}")
        lines.append("")
    unresolved = report.get("unresolved") or {}
    if unresolved.get("file"):
        lines += [f"Unresolved dossier: {unresolved.get('functions')} functions not settled "
                  f"(retry {unresolved.get('by_retry')}); "
                  f"{unresolved.get('cases_with_eligible_left')} still have an eligible technique "
                  f"(`solver/unresolved_dossier.json`)", ""]
    elif unresolved.get("observation_error"):
        lines += [f"Unresolved dossier: not written ({unresolved['observation_error']})", ""]
    return "\n".join(lines) + "\n"


def _args(p):
    p.add_argument("--project", help="project directory (required unless --resume)")
    p.add_argument("--image", action="append", default=[], help="logical image name (repeatable)")
    p.add_argument("--all-images", action="store_true")
    p.add_argument("--out", help="output directory for state, queue, reports (not a source tree; required unless "
                                 "--resume)")
    p.add_argument("--effort", choices=sorted(solver.MODES), default=None,
                   help="effort mode (default normal: exactly the single pass decompile-run always made); deep and "
                        "exhaustive add later rounds until a fixed point; see docs/solver.md")
    p.add_argument("--rounds", type=int, help="most rounds (default: the effort mode's)")
    p.add_argument("--schedule", choices=("observe", "order"),
                   help="observe: record the ranking's predictions only; order: let it order later rounds' work "
                        "(default: the effort mode's; never changes what is credited)")
    p.add_argument("--budget-time", help="stop starting work after this long: seconds, or 30m, 2h")
    p.add_argument("--budget-builds", type=int, help="stop starting work after this many builds (all images)")
    p.add_argument("--budget-model-calls", type=int, help="stop asking the refiner after this many calls")
    p.add_argument("--resume", metavar="RUN_DIR",
                   help="continue a run from its case store and rounds.jsonl (finished rounds are not redone)")
    p.add_argument("--declare-assumed", action="store_true",
                   help="declare self-describing data addresses (D_<address>) that byte-identical cases lean on "
                        "and verify those cases again (default: [run] declare_assumed)")
    p.add_argument("--allow-project-output", action="store_true",
                   help="allow --out in the project outside .n64dk (unsafe for generated state)")
    p.add_argument("--donors", help="donors TOML for transplant (see docs/donors.example.toml)")
    p.add_argument("--blind-forbid", action="append", metavar="TREE",
                   help="enforce a blind run: refuse (exit 3) before any stage when TREE (a source tree that must "
                        "not inform the run; repeatable) is reachable from the run's configuration, project or "
                        "environment, and check again at the end; receipts in solver/blind_*.json")
    p.add_argument("--blind-walk-limit", type=int, default=None, metavar="N",
                   help="entries a blind run's isolation check walks per configured directory and indexes per "
                        "forbidden tree (default 50000); a blind run whose check would stop short is refused")
    p.add_argument("--blind-shared-library", action="append", metavar="MANIFEST@sha256:HEX",
                   help="with --blind-forbid: a pinned shared public library (shared-library-pin) whose content "
                        "copies inside a forbidden tree do not count as reaching it (repeatable)")
    p.add_argument("--mode", dest="run_mode", choices=("blind", "research", "source-tree"), default=None,
                   help="what the run may learn from, enforced (default: no mode, unchecked as before). blind: the "
                        "ROM, the kit and other titles' donors; the corpus manifest's trees of this title are "
                        "forbidden (strict --blind-forbid), a donor that is this title or another revision of it "
                        "is refused, and the project's symbol files and catalogues may hold no name those trees "
                        "define. research: public general knowledge (SDK, other titles' trees, research notes), "
                        "never this title's source tree. source-tree: the title's own tree is allowed and "
                        "recorded. Saved with the invocation, recorded in report.json, solver/run_mode.json and "
                        "the run directory's provenance; a directory filled under a less strict mode is refused "
                        "(exit 3)")
    p.add_argument("--corpus-manifest", metavar="FILE",
                   help="with --mode: a n64dk.corpus/1 manifest whose entries concerning this title (by --title, "
                        "the ROM header's name or game code, or its SHA-1) are forbidden (blind, research) or "
                        "recorded (source-tree)")
    p.add_argument("--title", dest="run_title", action="append", metavar="NAME",
                   help="with --mode: the title's name as the corpus manifest and donors spell it (repeatable; the "
                        "ROM header's internal name is always used too)")
    p.add_argument("--donor-index", help="donor index file (default: <out>/donor_index.json)")
    p.add_argument("--library", help="directory of finished runs: every other title registered there is used "
                                     "as a donor, and this run registers its own strictly exact functions when it ends")
    p.add_argument("--donor", action="append", default=[], help="use only these donors")
    p.add_argument("--exclude-donor", action="append", default=[],
                   help="leave these donors out (always exclude the target project itself)")
    p.add_argument("--toolchain-dir")
    p.add_argument("--pipeline", default="kmc+builtin", help="default pipeline when [run] does not set one")
    p.add_argument("--flags", default="-O2 -G0 -mips3 -mgp32 -mfp32")
    p.add_argument("--jobs", type=int, default=0,
                   help="parallel builds in total (default: adaptive, capped at 4; set "
                        "N64DK_ALLOW_OVERSUBSCRIBE=1 for an explicit unsafe override)")
    p.add_argument("--calibrate", type=int,
                   help="functions sampled to measure the title's recipe before the sweep "
                        "(default: [run] calibrate, else none)")
    p.add_argument("--refine-command",
                   help="external author for unresolved cases: reads a prompt on stdin, prints C "
                        "(default: [run.refine] command; none configured means the stage is skipped)")
    p.add_argument("--refine-rounds", type=int)
    p.add_argument("--refine-max-cases", type=int)
    p.add_argument("--refine-max-score", type=int)
    p.add_argument("--refine-jobs", type=int, help="refiner calls at once (default 1)")
    p.add_argument("--refine-known-source", action="append", default=[], metavar="DIR",
                   help="folder of known source (an imported tree's split/): prompts holding its lines are not "
                        "sent, and answers reproducing it are flagged as recollection; repeatable")
    p.add_argument("--workers", type=int, default=0,
                   help="worker processes for the resolve stage (default 0: automatic, from hardware "
                        "threads and free memory)")
    p.add_argument("--worker-memory-mb", type=int, default=0,
                   help="memory estimate used by automatic worker selection (0: adaptive)")
    p.add_argument("--memory-reserve-mb", type=int, default=0,
                   help="memory excluded from automatic workers (0: adaptive)")
    p.add_argument("--budget", type=int, default=None, help="maximum builds per function (default: the effort "
                                                                 "mode's; normal 200)")
    p.add_argument("--search-ceiling", type=int, default=None,
                   help="skip the variant search when the best draft scores above this "
                        "(default: [run] search_ceiling, else the resolver's measured default; 0: always search)")
    p.add_argument("--limit", type=int, default=0, help="at most this many functions per image (smallest first)")
    p.add_argument("--order", default=None,
                   help="attempt order within a pass: size (default), reverse, shuffle:SEED; for determinism checks")
    p.add_argument("--regional-calibration", action="store_true", default=None,
                   help="calibrate each toolchain region (n64dk.match.toolchain_regions) on its own functions and "
                        "start every function from its region's recipe (default: [run] regional_calibration, else "
                        "the effort mode's: on in exhaustive only)")
    p.add_argument("--no-regional-calibration", dest="regional_calibration", action="store_false",
                   help="never calibrate by region, whatever [run] or the effort mode say")
    p.add_argument("--region-recipes", action="store_true", default=None,
                   help="enable the experimental region-recipe retry (measured 3/126; off by default)")
    p.add_argument("--unit-recipes", action="store_true", default=None,
                   help="enable the experimental nearest-unit recipe retry (measured 0/165; off by default)")
    p.add_argument("--skip-unknown-search", action="store_true",
                   help="experimental: skip variant search in unknown-toolchain regions; advisory only by default")
    p.add_argument("--recipe-gate", dest="recipe_gate", action="store_true", default=None,
                   help="run the wide recipe retry only for functions with unoptimised-code evidence (in the "
                        "function or its neighbourhood) or another compiler family (default; [run] recipe_gate). "
                        "Measured offline: 0 exact lost on 3 held-out blind titles, 67K builds saved")
    p.add_argument("--no-recipe-gate", dest="recipe_gate", action="store_false",
                   help="run the wide recipe retry on every far function (round-10 behaviour)")
    p.add_argument("--search-economics", action="store_true", default=None,
                   help="stop the variant search in a stratum (recipe x first-draft distance) whose searches have "
                        "returned nothing after a fixed spend, probing it now and then (n64dk.match.economics)")
    p.add_argument("--only-names", action="append",
                   help="attempt only these functions: comma-separated names or @FILE (one per line); repeatable")
    p.add_argument("--max-size", type=lambda s: int(s, 0), default=None,
                   help="attempt only functions up to this size (default: the effort mode's; 0: no limit)")
    p.add_argument("--min-confidence", type=float, default=0.8)
    p.add_argument("--strict-only", action="store_true", help="queue only strictly exact cases")
    p.add_argument("--skip-stage", action="append", default=[],
                   choices=list(SKIP_STAGES), metavar="STAGE",
                   help="skip a stage or switch one subsystem off for an ablation run (repeatable): "
                        + ", ".join(SKIP_STAGES) + ".  `calibrate` also stops recipe-doubt calibration, "
                        "which --calibrate 0 does not.  The ablation switches (from `declare` on) are also "
                        "read as N64DK_DISABLE_<NAME>=1; what a run switched off is in report.json `ablation`")
    p.add_argument("--discovery", choices=("repair", "audit", "off"), default=None,
                   help="discovery gate before every stage (default repair): audit each catalogue against call, "
                        "pointer and prologue evidence; repair adds evidence-backed functions to an incomplete "
                        "catalogue in the run directory (never the project); audit only reports")
    p.add_argument("--small-data", choices=("auto", "off"), default=None,
                   help="auto (default): recover the program's _gp from the code that sets $gp and offer the "
                        "small-data (-G) assembler lanes to functions that address through $gp; off: neither")
    p.add_argument("--data-stage", choices=("off", "report", "replay"), default=None,
                   help="data stage after resolution (default off). report: group every data reference that keeps "
                        "a case or donor unit waiting by symbol, section and image, and place it only from evidence "
                        "independent of that unit (OUT/IMAGE/data_ownership.json, data_replay.json); replay: also "
                        "re-verify strictly the waiting cases whose every data placement is established")
    p.add_argument("--redo", action="store_true", help="recompute checkpointed stages")
    p.add_argument("--receipts", action="store_true",
                   help="write a proof receipt for every settled result into OUT/receipts when the run ends "
                        "(rebuilds each once; see issue-receipts, verify-run)")


def _all_args(p):
    _args(p)
    _guard_arguments(p)


def _guard_arguments(p):
    p.add_argument("--wait-for-lock", action="store_true",
                   help="wait for another heavy n64dk job to finish instead of refusing to start")
    p.add_argument("--force-resources", action="store_true",
                   help="start even when memory, memory pressure or disk space are below the thresholds")


decompile_run.arguments = _all_args
