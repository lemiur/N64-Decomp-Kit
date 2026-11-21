"""``decompile-receipt RUN``: the exhaustive-treatment receipt of a finished ``decompile-run``.

Reads the run directory and writes nothing into it: ``solver/invocation.json`` (the command and
the effort mode), ``report.json`` (stage records, discovery, small data, ledgers, resources,
profile), ``rounds.jsonl`` (rounds, skips, stops and budgets), ``solver/coverage_<image>.json``
and ``solver/hard_queue_<image>.json`` (coverage and dossiers), ``events.jsonl`` and the case
store, opened read-only (``CaseStore(readonly=True)``: immutable when no write-ahead log exists).

The receipt lists what ran, what is left and why, and checks the run against the contract of an
exhaustive treatment (``CLAUSES``): each clause is ``satisfied``, ``not satisfied`` or ``not
verifiable`` from what the run recorded, with the keys of the evidence it was judged on.  A
clause the run cannot evidence is never reported as satisfied.

``--out DIR`` writes ``receipt.json`` and ``receipt.md`` there (both are left out of the run
digest when ``DIR`` is the run directory itself); ``--require`` exits 4 when a clause is not
satisfied.  The receipt holds no ROM bytes: only counts, names, addresses and digests."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from ..cli import command, emit
from ..match import solver

RECEIPT = "n64dk.exhaustive-receipt.v1"
SATISFIED, NOT_SATISFIED, NOT_VERIFIABLE = "satisfied", "not satisfied", "not verifiable"
OWN_FILES = ("receipt.json", "receipt.md")
SIDECARS = ("-wal", "-shm", "-journal")
FAILED_KEYS = ("error", "observation_error")
FIXED_POINT = ("fixed point", "nothing applicable")
LEDGER_BREAKS = ("gap", "overlap", "invariant", "outside")

CLAUSES = {
    "W0": "the run finished: report.json was written after the last event and no writer is attached",
    "W1": "the run used --effort exhaustive with the mode's settings (recipe gate off, regional calibration on)",
    "W2": "every image-discovery method ran (catalogue audit and repair with ROM placement, small-data "
          "recovery, overlay base inference)",
    "W3": "every catalogue is complete after repair, so every count is over the title's code",
    "W4": "every catalogued function has a case (an attempt with a recorded outcome) or a recorded reason it "
          "was not attempted",
    "W5": "every unresolved case has a dossier",
    "W6": "every skipped or withheld technique has a recorded reason",
    "W7": "no stage failed",
    "W8": "no stage failed silently: every planned stage of every started image left a result or an error",
    "W9": "every byte of every image has exactly one owner in the byte ledger",
    "W10": "every ROM byte has exactly one ownership row",
    "W11": "every exact case is strictly verified",
    "W12": "every image ended at its fixed point, not at a budget or round limit",
    "W13": "every unresolved case names a primary cause and a next action or why automation stopped",
    "W14": "the tool chain is identified by content digest",
    "W15": "memory and swap were recorded and swap did not grow",
    "W16": "the run is reproducible from its own record (invocation and report kept)",
}


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path):
    """(value, error): a file's JSON, or None and why not ("absent" when there is no file)."""
    if not path.is_file():
        return None, "absent"
    try:
        return json.loads(path.read_text()), None
    except (OSError, ValueError) as error:                   # UnicodeDecodeError is a ValueError
        return None, f"{type(error).__name__}: {error}"[-200:]


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _list(value) -> list:
    return value if isinstance(value, list) else []


def _failed(value) -> str | None:
    if isinstance(value, dict):
        for key in FAILED_KEYS:
            if value.get(key):
                return str(value[key])[:300]
    return None


def run_files(run: Path) -> list[dict]:
    """Every file one or two levels down (not the case store's sources, not SQLite sidecars, not
    the receipt's own files), with its size and SHA-256: what the run digest is over."""
    candidates = []
    for path in sorted(run.iterdir()):
        if path.is_file():
            if path.name not in OWN_FILES:
                candidates.append(path)
        elif path.is_dir():
            candidates += sorted(p for p in path.iterdir() if p.is_file())
    return [{"path": p.relative_to(run).as_posix(), "bytes": p.stat().st_size, "sha256": file_digest(p)}
            for p in candidates if not p.name.endswith(SIDECARS)]


def _cases(run: Path):
    if not (run / "state" / "cases.sqlite").is_file():
        return None, "no case store (state/cases.sqlite)"
    try:
        from ..match import resolve as resolvemod
        store = resolvemod.CaseStore(run / "state", readonly=True)
        try:
            return store.all(), None
        finally:
            store.close()
    except Exception as error:                               # noqa: BLE001 - reported, the receipt goes on
        return None, f"{type(error).__name__}: {error}"[-200:]


def _events(run: Path) -> list[dict]:
    path = run / "events.jsonl"
    out = []
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                out.append(value)
    return out


def _state(value, planned: bool, why_not: str) -> dict:
    failure = _failed(value)
    if failure:
        return {"state": "failed", "detail": failure}
    if value is not None:
        return {"state": "completed"}
    if planned:
        return {"state": "missing", "detail": "planned, but the report holds no record of it"}
    return {"state": "not run", "detail": why_not}


# --------------------------------------------------------------------------- sections

def stage_table(run: Path, report: dict, invocation: dict) -> dict:
    """Run-level and per-image stages: completed, failed (with the error), missing (planned with no
    record: a silent failure), not run (with why) or unavailable (not part of decompile-run)."""
    skip = set(_list(invocation.get("skip_stage")))
    discovery_mode = invocation.get("discovery") or "repair"
    overlays = (run / "discovery" / "overlays.json").is_file()
    run_level = {
        "discovery": _state(report.get("discovery"), discovery_mode != "off" and "discovery" not in skip,
                            "--discovery off" if discovery_mode == "off" else "--skip-stage discovery"),
        "small-data": _state(report.get("small_data"), (invocation.get("small_data") or "auto") != "off",
                             "--small-data off"),
        "donor-export": _state(report.get("donor"), True, ""),
        "overlay-base-inference": ({"state": "completed", "detail": "discovery/overlays.json (written outside "
                                                                    "decompile-run, by discovery-audit --overlays)"}
                                   if overlays else
                                   {"state": "unavailable", "detail": "not a decompile-run stage: only discovery-audit "
                                                                      "--overlays infers per-overlay bases"}),
        "behaviour-oracle": {"state": "unavailable", "detail": "n64dk.match.oracle is evidence only and is not wired "
                                                               "into decompile-run (its symbolic tier needs a solver "
                                                               "that is not installed)"},
        "proof-receipts": _state(report.get("receipts"), False, "opt-in: a replayable proof receipt per settled "
                                                                "result is written only when asked for"),
    }
    if run_level["discovery"]["state"] == "completed" and discovery_mode != "repair":
        run_level["discovery"]["detail"] = f"--discovery {discovery_mode}: audited, not repaired"
    donors = bool(invocation.get("donors") or invocation.get("library"))
    resolve = "resolve" not in skip
    images = {}
    for image, item in sorted(_dict(report.get("images")).items()):
        item = _dict(item)
        regional = _dict(item.get("effective_settings")).get("regional_calibration")
        calibrate_why = ("no calibration sample (--calibrate, [run] calibrate) and no recipe doubt"
                         + ("; regional calibration, on in this mode, runs only inside calibration" if regional else ""))
        images[image] = {
            "boundaries": _state(item.get("boundaries"), "boundaries" not in skip, "--skip-stage boundaries"),
            "donors": _state(item.get("donors"), donors and "donors" not in skip,
                             "--skip-stage donors" if donors else "no --donors or --library given"),
            "calibrate": _state(item.get("calibrate"), False, calibrate_why),
            "calibrate-regions": _state(item.get("calibrate_regions"), False,
                                        calibrate_why if regional else "regional calibration off"),
            "resolve": _state(item.get("resolve"), resolve, "--skip-stage resolve"),
            "declare": _state(item.get("declare"), False, "first-round declaration is opt-in (--declare-assumed, "
                                                          "[run] declare_assumed); deep and exhaustive declare in "
                                                          "the cascade after each round"),
            "rounds": _state(item.get("solver_stop"), resolve, "--skip-stage resolve"),
            "solver-outputs": _state(item.get("solver"), resolve, "--skip-stage resolve"),
            "toolchain-regions": _state(item.get("toolchain_regions"), resolve, "--skip-stage resolve"),
            "queue": _state(item.get("queue"), True, ""),
            "ledger": _state(item.get("ledger"), "ledger" not in skip, "--skip-stage ledger"),
        }
    return {"run": run_level, "images": images}


def counts(cases, report: dict, coverage: dict) -> dict:
    """Case-store outcomes (decompile-status's credit), what was never attempted and why, and the
    donors.  "raw" here is the functions with no case: the export leaves them as assembly or raw
    words; the export's own byte categories are decompile-export's, not the run's."""
    out = {"definitions": {
        "exact_c": "strictly verified C (case store, as decompile-status counts it)",
        "awaiting_declarations": "byte-identical, waiting for declared symbols (assumed)",
        "handwritten_assembly": "kept as assembly that reassembles; never credited as C",
        "unresolved": "every other case status but stale",
        "not_attempted": "catalogued functions with no case (the export's assembly or raw-word fallback)"}}
    if cases is not None:
        by_status: dict[str, int] = {}
        for case in cases:
            by_status[case.status] = by_status.get(case.status, 0) + 1
        out["cases"] = {"total": len(cases), "by_status": dict(sorted(by_status.items())), **solver.credit(cases)}
        out["by_image"] = {image: solver.credit([c for c in cases if c.image == image])
                           for image in sorted({c.image for c in cases})}
    reasons: dict[str, int] = {}
    for found in coverage.values():
        for reason, number in _dict(found.get("reasons")).items():
            reasons[reason] = reasons.get(reason, 0) + int(number or 0)
    out["not_attempted"] = {"functions": sum(int(f.get("not_attempted") or 0) for f in coverage.values()),
                            "unexplained": sum(int(f.get("unexplained") or 0) for f in coverage.values()),
                            "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1]))}
    out["donor_verified"] = sum(len(_list(_dict(r).get("functions")))
                                for item in _dict(report.get("images")).values()
                                for r in _list(_dict(_dict(item).get("donors")).get("verified")))
    return out


def receipt(run: Path) -> dict:
    run = Path(run)
    if not run.is_dir():
        raise SystemExit(f"{run}: no such run directory")
    report, report_error = _load(run / "report.json")
    if report is not None and not isinstance(report, dict):
        report, report_error = None, "report.json does not hold an object"
    invocation, invocation_error = _load(run / "solver" / "invocation.json")
    invocation = _dict(invocation)
    rep = _dict(report)
    rounds = solver.RoundsLog(run / "rounds.jsonl").read()
    cases, cases_error = _cases(run)
    events = _events(run)
    names = set(_dict(rep.get("images")))
    for prefix in ("coverage_", "hard_queue_"):
        names |= {p.stem[len(prefix):] for p in (run / "solver").glob(f"{prefix}*.json")} \
            if (run / "solver").is_dir() else set()
    coverage, dossiers = {}, {}
    for image in sorted(names):
        found, _error = _load(run / "solver" / f"coverage_{image}.json")
        if isinstance(found, dict):
            coverage[image] = found
        elif isinstance(_dict(_dict(_dict(rep.get("images")).get(image)).get("solver")).get("coverage"), dict):
            coverage[image] = rep["images"][image]["solver"]["coverage"]
        records, _error = _load(run / "solver" / f"hard_queue_{image}.json")
        if isinstance(records, list):
            dossiers[image] = [r for r in records if isinstance(r, dict)]
    files = run_files(run)
    effort = invocation.get("effort")
    mode = solver.MODES.get(effort) if isinstance(effort, str) else None
    found = _dict(rep.get("discovery"))
    graph = {"verdict": found.get("verdict"), "verdict_before": found.get("verdict_before"),
             "rom_reasons": _list(found.get("rom_reasons")), "placed": _list(found.get("placed")),
             "images": {name: {k: _dict(img).get(k) for k in ("verdict", "verdict_after", "coverage_after",
                                                                "functions_after", "repaired_functions")}
                        for name, img in sorted(_dict(found.get("images")).items())}}
    catalogue = {}
    for image, cover in sorted(coverage.items()):
        rows = [[e.get("name"), e.get("vram"), e.get("size")] for e in _list(cover.get("entries")) if isinstance(e, dict)]
        if rows:
            catalogue[image] = {"functions": len(rows), "digest": sha256_text(canonical(rows))}
    audit = run / "discovery" / "discovery_audit.json"
    stages = stage_table(run, rep, invocation)
    sessions = list(dict.fromkeys(r.get("session") for r in rounds if r.get("session")))
    last_budget = {}
    for record in rounds:
        if record.get("image") and isinstance(record.get("budget"), dict):
            last_budget[record["image"]] = record["budget"]
    profile = _dict(rep.get("profile"))
    requested = sum(int(_dict(row).get("builds_requested") or 0) for row in profile.values())
    distinct = sum(int(_dict(row).get("distinct_objects") or 0) for row in profile.values())
    by_cause: dict[str, dict] = {}
    for records in dossiers.values():
        for record in records:
            group = by_cause.setdefault(str(record.get("primary_cause") or "none recorded"), {"functions": 0, "bytes": 0})
            group["functions"] += 1
            group["bytes"] += int(record.get("size") or 0)
    body = {
        "receipt": RECEIPT,
        "run": run.name,
        "status": rep.get("status") if report is not None else None,
        "report_error": None if report is not None else report_error,
        "command": {"effort": effort, "mode": mode.as_dict() if mode else None,
                    "mode_settings": invocation.get("mode_settings"), "invocation": invocation or None,
                    "invocation_error": None if invocation else invocation_error, "sessions": sessions,
                    "effective_settings": {name: _dict(item).get("effective_settings")
                                           for name, item in sorted(_dict(rep.get("images")).items())}},
        "toolchain": {"toolchain_dir": invocation.get("toolchain_dir"),
                      "recorded_digests": tool_digests(run, rep) or None,
                      "manifest": manifest_summary(run),
                      "note": "decompile-run records the tool chain directory, not digests of the compilers, "
                              "assemblers and linkers it ran; a toolchain_manifest.json beside the report "
                              "(n64dk.match.toolmanifest.build) supplies them",
                      "recipes_of_cases": _tally(c.recipe for c in cases or [] if c.recipe),
                      "recipes_of_exact": _tally(c.recipe for c in cases or [] if c.recipe and c.status == "exact"),
                      "region_verdicts": {name: _dict(_dict(item).get("toolchain_regions")).get("functions_by_verdict")
                                          for name, item in sorted(_dict(rep.get("images")).items())}},
        "image_graph": {"digest": sha256_text(canonical(graph)) if found else None,
                        "audit_file_sha256": file_digest(audit) if audit.is_file() else None,
                        "note": "digest over the run's recorded discovery summary (verdicts, placements), not over "
                                "image bytes", **graph},
        "catalogue": {"digest": sha256_text(canonical(catalogue)) if catalogue else None, "images": catalogue,
                      "note": "digest over the (name, address, size) rows the solver's coverage saw"},
        "stages": stages,
        "counts": counts(cases, rep, coverage),
        "case_store_error": cases_error,
        "ledger": {name: {"totals": _dict(_dict(item).get("ledger")).get("totals"),
                          "violations": _dict(_dict(item).get("ledger")).get("violations"),
                          "error": _failed(_dict(item).get("ledger"))}
                   for name, item in sorted(_dict(rep.get("images")).items())},
        "ledger_note": "built from the project's manifests over the run's catalogue: this run's own exact cases are "
                       "not ledger rows (decompile-export's FUNCTIONS.tsv places them)",
        "unresolved": {"by_primary_cause": dict(sorted(by_cause.items(), key=lambda kv: -kv[1]["bytes"])),
                       "by_image": {name: _dict(_dict(item).get("solver")).get("remaining_by_cause")
                                    for name, item in sorted(_dict(rep.get("images")).items())}},
        "budgets": {**{key: invocation.get(key) for key in ("budget", "max_size", "search_ceiling", "cross_recipes",
                                                            "rounds", "schedule")},
                    "stops": [{k: r.get(k) for k in ("session", "image", "after_round", "reason", "budget")}
                              for r in rounds if r.get("kind") == "stop"],
                    "last_round_budget": last_budget,
                    "note": "--budget-time, --budget-builds and --budget-model-calls come from each invocation's "
                            "command line and are not saved; the rounds log records each round's budget state"},
        "cache": {"builds_requested": requested, "distinct_objects": distinct,
                  "requests_without_a_new_object": max(0, requested - distinct),
                  "by_stage": {name: {k: _dict(row).get(k) for k in ("builds_requested", "distinct_objects",
                                                                     "tool_invocations", "wall_s")}
                               for name, row in sorted(profile.items())},
                  "note": "the profiler's counts; a request without a new object was served by deduplication or a "
                          "cache"},
        "resources": {"plan": rep.get("resources"), "tool_profile": rep.get("tool_profile"),
                      "seconds": rep.get("seconds"),
                      "note": "memory and swap are recorded once, when the run starts; nothing records them at the end"},
        "files": files,
        "run_digest": sha256_text(canonical(files)),
    }
    body["contract"] = contract(run, rep if report is not None else None, report_error, invocation, rounds, cases,
                                coverage, dossiers, events, stages, mode)
    summary = {SATISFIED: 0, NOT_SATISFIED: 0, NOT_VERIFIABLE: 0}
    for row in body["contract"]:
        summary[row["status"]] += 1
    body["contract_summary"] = summary
    return body


def _manifest(run: Path) -> dict:
    found, _error = _load(run / "toolchain_manifest.json")
    return _dict(found)


def tool_digests(run: Path, report: dict) -> dict:
    """``component/file -> sha256`` the run recorded: ``report.json``'s ``toolchain_digests`` (no
    decompile-run writes it yet) or the installed components of a ``toolchain_manifest.json``
    beside the report (``n64dk.match.toolmanifest.build``).  Empty when neither exists."""
    out = {str(k): str(v) for k, v in _dict(report.get("toolchain_digests")).items() if v}
    for component in _list(_manifest(run).get("components")):
        component = _dict(component)
        if not component.get("installed"):
            continue
        for row in _list(component.get("files")):
            row = _dict(row)
            if row.get("sha256") and row.get("file"):
                out[f"{component.get('ident')}/{row['file']}"] = str(row["sha256"])
    return dict(sorted(out.items()))


def manifest_summary(run: Path) -> dict | None:
    manifest = _manifest(run)
    if not manifest:
        return None
    return {"schema": manifest.get("schema"), "digest": manifest.get("digest"),
            "components_installed": sum(1 for c in _list(manifest.get("components")) if _dict(c).get("installed")),
            "recipes": {lane: _dict(row).get("identity") for lane, row in sorted(_dict(manifest.get("recipes")).items())}}


def _tally(values) -> dict:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


# --------------------------------------------------------------------------- the contract

def _clause(clause: str, status: str, evidence: list[str], detail: str) -> dict:
    return {"clause": clause, "text": CLAUSES[clause], "status": status, "evidence": evidence, "detail": detail}


def contract(run: Path, report: dict | None, report_error, invocation: dict, rounds: list, cases, coverage: dict,
             dossiers: dict, events: list, stages: dict, mode) -> list[dict]:
    rep = report or {}
    images = _dict(rep.get("images"))
    out = []

    # W0 finished
    wal = Path(run / "state" / "cases.sqlite-wal")
    if report is None:
        out.append(_clause("W0", NOT_SATISFIED, ["report.json"], f"no readable report.json ({report_error}): the run "
                                                                 "has not finished an invocation"))
    else:
        problems = []
        events_path, report_path = run / "events.jsonl", run / "report.json"
        if events_path.is_file() and events_path.stat().st_mtime > report_path.stat().st_mtime + 1:
            problems.append("events were written after report.json: a later invocation has not finished, so the "
                            "report describes an earlier one")
        if wal.exists() and wal.stat().st_size > 0:
            problems.append("the case store has a write-ahead log: a writer may be attached")
        out.append(_clause("W0", NOT_SATISFIED if problems else SATISFIED,
                           ["report.json", "events.jsonl", "state/cases.sqlite-wal"],
                           "; ".join(problems) or f"report status {rep.get('status')}"))

    # W1 effort exhaustive
    effort = invocation.get("effort")
    if not invocation:
        out.append(_clause("W1", NOT_VERIFIABLE, ["solver/invocation.json"], "no invocation.json"))
    elif effort != "exhaustive":
        out.append(_clause("W1", NOT_SATISFIED, ["solver/invocation.json#/effort"], f"effort {effort}"))
    else:
        overridden = sorted(name for name, item in images.items()
                            if _dict(_dict(item).get("effective_settings")).get("recipe_gate") is True
                            or _dict(_dict(item).get("effective_settings")).get("regional_calibration") is False)
        out.append(_clause("W1", NOT_SATISFIED if overridden else SATISFIED,
                           ["solver/invocation.json#/effort", "report.json#/images/*/effective_settings"],
                           f"the command line or [run] turned the recipe gate on or regional calibration off for "
                           f"{', '.join(overridden)}" if overridden else "effort exhaustive, settings as the mode's"))

    # W2 discovery methods
    run_stages = stages["run"]
    missing = []
    if run_stages["discovery"]["state"] != "completed" or (invocation.get("discovery") or "repair") != "repair":
        missing.append("catalogue audit and repair with ROM placement (" + run_stages["discovery"]["state"] + ")")
    if run_stages["small-data"]["state"] != "completed":
        missing.append("small-data recovery (" + run_stages["small-data"]["state"] + ")")
    if run_stages["overlay-base-inference"]["state"] != "completed":
        missing.append("overlay base inference (not a decompile-run stage)")
    out.append(_clause("W2", NOT_SATISFIED if missing else SATISFIED,
                       ["report.json#/discovery", "report.json#/small_data", "discovery/overlays.json"],
                       ("did not run: " + "; ".join(missing)) if missing else "all ran"))

    # W3 catalogue complete
    verdict = _dict(rep.get("discovery")).get("verdict")
    out.append(_clause("W3", SATISFIED if verdict == "complete" else NOT_VERIFIABLE if verdict is None else NOT_SATISFIED,
                       ["report.json#/discovery/verdict"],
                       f"discovery verdict after repair: {verdict}" + (
                           f" (before: {_dict(rep.get('discovery')).get('verdict_before')})" if verdict else
                           " (no discovery record)")))

    # W4 every function a case or a reason
    without = sorted(name for name in images if name not in coverage)
    unexplained = {name: int(found.get("unexplained") or 0) for name, found in coverage.items()
                   if int(found.get("unexplained") or 0)}
    blank = sum(1 for found in coverage.values() for e in _list(found.get("entries"))
                if isinstance(e, dict) and e.get("attempted") is False and not e.get("reason"))
    bad = bool(without or unexplained or blank)
    out.append(_clause("W4", NOT_VERIFIABLE if not coverage and not images else NOT_SATISFIED if bad else SATISFIED,
                       ["solver/coverage_<image>.json"],
                       "; ".join(filter(None, [f"no coverage record for {', '.join(without)}" if without else "",
                                               f"unexplained functions {unexplained}" if unexplained else "",
                                               f"{blank} not attempted without a reason" if blank else ""]))
                       or f"{sum(int(f.get('functions') or 0) for f in coverage.values())} functions accounted for"))

    # W5 dossiers
    if cases is None:
        out.append(_clause("W5", NOT_VERIFIABLE, ["state/cases.sqlite"], "no readable case store"))
    else:
        unresolved = {c.case_id for c in cases if c.status in solver.UNRESOLVED}
        filed = {str(r.get("case")) for records in dossiers.values() for r in records}
        lacking = sorted(unresolved - filed)
        out.append(_clause("W5", NOT_SATISFIED if lacking else SATISFIED,
                           ["state/cases.sqlite", "solver/hard_queue_<image>.json"],
                           f"{len(lacking)} of {len(unresolved)} unresolved cases have no dossier"
                           + (f" (first: {', '.join(lacking[:3])})" if lacking else "")))

    # W6 skipped techniques have reasons
    unreasoned = []
    for record in rounds:
        for name, row in _dict(record.get("skipped")).items():
            if not _dict(row).get("reason"):
                unreasoned.append(f"round {record.get('round')} [{record.get('image')}] {name}")
    for records in dossiers.values():
        for r in records:
            for name, why in _dict(r.get("excluded_actions")).items():
                if not why:
                    unreasoned.append(f"{r.get('name')}: {name}")
            if not r.get("next_action") and not r.get("why_stopped"):
                unreasoned.append(f"{r.get('name')}: no next action and no stop reason")
    if not rounds and not dossiers:
        out.append(_clause("W6", NOT_VERIFIABLE, ["rounds.jsonl", "solver/hard_queue_<image>.json"],
                           "no rounds log and no dossiers"))
    else:
        out.append(_clause("W6", NOT_SATISFIED if unreasoned else SATISFIED,
                           ["rounds.jsonl#/skipped", "solver/hard_queue_<image>.json#/excluded_actions"],
                           f"{len(unreasoned)} without a reason (first: {'; '.join(unreasoned[:3])})" if unreasoned
                           else "every skip and exclusion carries a reason"))

    # W7 no stage failed
    if report is None:
        out.append(_clause("W7", NOT_VERIFIABLE, ["report.json"], "no report"))
    else:
        failed = [f"{image}/{name}: {row.get('detail')}"[:160] for image, table in stages["images"].items()
                  for name, row in table.items() if row["state"] == "failed"]
        failed += [f"{name}: {row.get('detail')}"[:160] for name, row in run_stages.items() if row["state"] == "failed"]
        hidden = sorted(f"{image}/{key}" for image, item in images.items() for key, value in _dict(item).items()
                        if isinstance(value, dict) and value.get("observation_error"))
        out.append(_clause("W7", NOT_SATISFIED if failed else SATISFIED,
                           ["report.json#/failed_stages", "report.json#/images/*/<stage>/error"],
                           ("; ".join(failed[:4]) + (f"; observation errors not reflected in report.status: "
                                                     f"{', '.join(hidden)}" if hidden else "")) if failed
                           else "no stage recorded an error"))

    # W8 no silent failure
    started = sorted({str(e.get("image")) for e in events if e.get("event") == "image-start" and e.get("image")})
    absent = [image for image in started if image not in images]
    silent = [f"{image}/{name}" for image, table in stages["images"].items() for name, row in table.items()
              if row["state"] == "missing"]
    silent += [name for name, row in run_stages.items() if row["state"] == "missing"]
    if report is None:
        out.append(_clause("W8", NOT_SATISFIED, ["report.json", "events.jsonl"],
                           f"no report: {len(started)} started image(s) left no record"))
    else:
        out.append(_clause("W8", NOT_SATISFIED if absent or silent else SATISFIED,
                           ["events.jsonl#image-start", "report.json#/images"],
                           "; ".join(filter(None, [f"started but absent from the report: {', '.join(absent)}"
                                                   if absent else "",
                                                   f"planned without a record: {', '.join(silent[:6])}"
                                                   if silent else ""])) or "every planned stage left a record"))

    # W9 image ledgers
    broken = []
    for name, item in sorted(images.items()):
        ledger = _dict(_dict(item).get("ledger"))
        if not ledger or _failed(ledger):
            broken.append(f"{name}: no ledger" + (f" ({_failed(ledger)})" if _failed(ledger) else ""))
            continue
        kinds = {k: v for k, v in _dict(ledger.get("violations")).items() if k in LEDGER_BREAKS and v}
        if _dict(ledger.get("totals")).get("sum_equals_extent") is not True or kinds:
            broken.append(f"{name}: sum equals extent {_dict(ledger.get('totals')).get('sum_equals_extent')}, "
                          f"violations {kinds}")
    out.append(_clause("W9", NOT_VERIFIABLE if not images else NOT_SATISFIED if broken else SATISFIED,
                       ["report.json#/images/*/ledger", "<image>/ownership.tsv"],
                       "; ".join(broken[:4]) or "every image ledger sums to its extent without gap or overlap"))

    # W10 ROM-wide
    reasons = _list(_dict(rep.get("discovery")).get("rom_reasons"))
    out.append(_clause("W10", NOT_SATISFIED if reasons else NOT_VERIFIABLE, ["report.json#/discovery/rom_reasons"],
                       ("; ".join(str(r) for r in reasons[:3])) if reasons else
                       "decompile-run keeps ledgers of image extents only: ROM bytes outside every image (header, boot "
                       "code, assets, padding) have no ownership rows in the run"))

    # W11 strict verification of exact cases
    if cases is None:
        out.append(_clause("W11", NOT_VERIFIABLE, ["state/cases.sqlite"], "no readable case store"))
    else:
        exact = [c for c in cases if c.status == "exact"]
        loose = [c.case_id for c in exact if (c.strict or {}).get("verdict") != "exact"]
        out.append(_clause("W11", NOT_SATISFIED if loose else SATISFIED, ["state/cases.sqlite#strict"],
                           f"{len(loose)} of {len(exact)} exact cases without a strict exact verdict"))

    # W12 fixed point
    stops = {name: _dict(_dict(item).get("solver_stop")).get("reason") for name, item in images.items()}
    early = {name: reason for name, reason in stops.items() if not str(reason or "").startswith(FIXED_POINT)}
    out.append(_clause("W12", NOT_VERIFIABLE if not images else NOT_SATISFIED if early else SATISFIED,
                       ["report.json#/images/*/solver_stop", "rounds.jsonl#stop"],
                       "; ".join(f"{name}: {reason or 'no stop record'}" for name, reason in sorted(early.items()))[:400]
                       or "every image reached its fixed point"))

    # W13 cause and next step per unresolved case
    records = [r for rows in dossiers.values() for r in rows]
    vague = [str(r.get("name")) for r in records
             if not r.get("primary_cause") or not (r.get("next_action") or r.get("why_stopped"))]
    out.append(_clause("W13", NOT_VERIFIABLE if not records and cases is None else NOT_SATISFIED if vague else SATISFIED,
                       ["solver/hard_queue_<image>.json#/primary_cause"],
                       f"{len(vague)} of {len(records)} dossiers lack a cause or a next step" if vague
                       else f"{len(records)} dossiers each name a cause and a next step or a stop reason"))

    # W14 tool chain digests
    digests = tool_digests(run, rep)
    out.append(_clause("W14", SATISFIED if digests else NOT_VERIFIABLE,
                       ["report.json#/toolchain_digests", "toolchain_manifest.json#/components/*/files",
                        "solver/invocation.json#/toolchain_dir"],
                       f"{len(digests)} tool file digests recorded" if digests else
                       "the run records the tool chain directory only, not digests of the tools it ran"))

    # W15 resources
    memory = _dict(_dict(rep.get("resources")).get("memory"))
    out.append(_clause("W15", NOT_VERIFIABLE, ["report.json#/resources/memory"],
                       (f"recorded at the start only: swap used {memory.get('swap_used_mb')} MB, available "
                        f"{memory.get('available_mb')} MB; swap at the end is not recorded") if memory
                       else "no memory record"))

    # W16 reproducible
    keep = [] if invocation.get("effort") else ["solver/invocation.json with its effort"]
    keep += [] if report is not None else ["report.json"]
    out.append(_clause("W16", NOT_SATISFIED if keep else SATISFIED, ["solver/invocation.json", "report.json"],
                       ("missing: " + ", ".join(keep)) if keep else
                       "invocation and report kept (decompile-run --resume RUN continues from them)"))
    return out


# --------------------------------------------------------------------------- output

def render(result: dict) -> str:
    """The receipt as Markdown."""
    lines = [f"# Exhaustive-treatment receipt: {result['run']}", "",
             f"Run status: {result.get('status') or 'no report'}; effort {result['command'].get('effort')}; "
             f"run digest `{result['run_digest'][:16]}`", "", "## Contract", "",
             "| clause | status | text | detail | evidence |", "|---|---|---|---|---|"]
    for row in result["contract"]:
        lines.append(f"| {row['clause']} | {row['status']} | {row['text']} | {_cell(row['detail'])} | "
                     f"{_cell(', '.join(row['evidence']))} |")
    summary = result["contract_summary"]
    lines += ["", f"{summary[SATISFIED]} satisfied, {summary[NOT_SATISFIED]} not satisfied, "
                  f"{summary[NOT_VERIFIABLE]} not verifiable from the run.", "", "## Stages", ""]
    for name, row in result["stages"]["run"].items():
        lines.append(f"- {name}: {row['state']}" + (f" ({row['detail']})" if row.get("detail") else ""))
    for image, table in result["stages"]["images"].items():
        lines.append(f"- {image}: " + ", ".join(f"{name} {row['state']}" for name, row in table.items()))
    counted = result["counts"]
    lines += ["", "## Outcomes", ""]
    if "cases" in counted:
        c = counted["cases"]
        lines.append(f"- cases {c['total']}: exact C {c['exact_c']} ({c['exact_c_bytes']} bytes), awaiting "
                     f"declarations {c['awaiting_declarations']}, hand-written assembly {c['handwritten_assembly']}, "
                     f"unresolved {c['unresolved']}")
    else:
        lines.append(f"- cases: {result.get('case_store_error')}")
    lines.append(f"- not attempted {counted['not_attempted']['functions']} (unexplained "
                 f"{counted['not_attempted']['unexplained']}); donor-verified {counted['donor_verified']}")
    for cause, group in result["unresolved"]["by_primary_cause"].items():
        lines.append(f"- unresolved, {cause}: {group['functions']} functions, {group['bytes']} bytes")
    for image, ledger in result["ledger"].items():
        totals = _dict(ledger.get("totals"))
        lines.append(f"- ledger {image}: " + (f"failed: {ledger['error']}" if ledger.get("error") else
                                              f"extent {totals.get('extent_bytes')} bytes, C {totals.get('c_bytes')}, "
                                              f"assembly {totals.get('asm_bytes')}, violations {ledger.get('violations')}"))
    lines.append(f"- {result['ledger_note']}")
    budgets, cache = result["budgets"], result["cache"]
    lines += ["", "## Budgets, cache, resources, identity", "",
              f"- per-function builds {budgets.get('budget')}, rounds {budgets.get('rounds')}, search ceiling "
              f"{budgets.get('search_ceiling')}, schedule {budgets.get('schedule')}"]
    lines += [f"- stop [{s.get('image')}] after round {s.get('after_round')}: {s.get('reason')}" for s in budgets["stops"]]
    lines += [f"- builds requested {cache['builds_requested']}, distinct objects {cache['distinct_objects']}",
              f"- tool chain: {result['toolchain']['note']}",
              f"- image graph digest `{result['image_graph']['digest']}`; catalogue digest "
              f"`{result['catalogue']['digest']}`",
              f"- {len(result['files'])} files in the run digest"]
    return "\n".join(lines) + "\n"


def _cell(text) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def _write(path: Path, text: str) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@command("decompile-receipt", "Exhaustive-treatment receipt of a finished decompile-run: what ran, what is left and "
                              "which contract clauses its own records satisfy (read-only)")
def decompile_receipt(args):
    result = receipt(Path(args.run))
    if getattr(args, "out", None):
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        _write(out / "receipt.json", json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
        _write(out / "receipt.md", render(result))
    code = 4 if getattr(args, "require", False) and result["contract_summary"][NOT_SATISFIED] else 0
    if getattr(args, "format", "text") == "text":
        print(render(result), end="")
        return code
    if code:
        emit(result, "json")
        return code
    return result


def _args(p):
    p.add_argument("run", help="the --out directory of a finished decompile-run")
    p.add_argument("--out", help="write receipt.json and receipt.md here")
    p.add_argument("--require", action="store_true", help="exit 4 when a contract clause is not satisfied")


decompile_receipt.arguments = _args
