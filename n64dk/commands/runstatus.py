"""``decompile-status RUN_DIR``: the rounds of a ``decompile-run`` and what is left, read-only.

Reads ``rounds.jsonl``, ``solver/invocation.json``, ``solver/observations.json`` and the case
store, which is opened read-only (``CaseStore(readonly=True)``: immutable when no write-ahead log
is present), so inspecting a run, finished or still running, writes nothing anywhere.

When the run has a ``report.json`` (written when an invocation finishes), its discovery verdict
with the denominator warning, its small-data entry and per-image totals are shown as well; they
describe the last finished invocation, while the rounds and the queue are read live."""
from __future__ import annotations

import json
from pathlib import Path

from ..cli import command
from ..match import resolve as resolvemod
from ..match import solver


def round_line(record: dict) -> str:
    """``Round N [image]: <sieve> +k exact, ...``"""
    parts = []
    for name, row in (record.get("techniques") or {}).items():
        parts.append(f"{name} +{row.get('exact', 0)} exact ({row.get('cases', 0)} cases, {row.get('builds', 0)} builds"
                     + (f", {row['model_calls']} model calls" if row.get("model_calls") else "") + ")")
    declare = (record.get("cascade") or {}).get("declare") or {}
    if declare.get("ran"):
        parts.append(f"declare +{declare.get('promoted', 0)} exact ({declare.get('new_names', 0)} new declarations)")
    neighbours = (record.get("cascade") or {}).get("neighbours") or {}
    if neighbours.get("marked"):
        parts.append(f"neighbours: {neighbours['marked']} cases with new evidence")
    for name, row in (record.get("skipped") or {}).items():
        parts.append(f"{name} skipped for {row.get('cases')} cases ({row.get('reason')})")
    gains = record.get("gains") or {}
    return (f"Round {record.get('round')} [{record.get('image')}]: " + ("; ".join(parts) or record.get("sieve", "-"))
            + f" | new exact {gains.get('new_exact', 0)}, awaiting declarations "
              f"{gains.get('awaiting_declarations', 0):+d}, evidence {gains.get('evidence', 0)}")


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _list(value) -> list:
    return value if isinstance(value, list) else []


def read_report(run: Path) -> dict | None:
    """``<run>/report.json``, or None when there is none (a first invocation still running).  An
    unreadable or malformed file is returned as ``{"error": ...}``, never raised."""
    path = Path(run) / "report.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as error:                   # UnicodeDecodeError is a ValueError
        return {"error": f"{type(error).__name__}: {error}"[-300:]}
    return data if isinstance(data, dict) else {"error": "report.json does not hold an object"}


def discovery_view(found, image: str | None = None) -> dict | None:
    """The report's discovery entry: the verdict, the verdict before repair, the denominator
    warning and, per image, the verdict and coverage after repair (the reasons when not complete)."""
    found = _dict(found)
    if not found:
        return None
    if found.get("error"):
        return {"error": found["error"]}
    images = {}
    for name, img in _dict(found.get("images")).items():
        if (image is not None and name != image) or not isinstance(img, dict):
            continue
        after = img.get("verdict_after", img.get("verdict"))
        images[name] = {"verdict": img.get("verdict"), "verdict_after": after,
                        "coverage": img.get("coverage"),
                        "coverage_after": img.get("coverage_after", img.get("coverage")),
                        "functions_after": img.get("functions_after", img.get("functions")),
                        "repaired_functions": img.get("repaired_functions") or 0,
                        "reasons_after": [] if after == "complete"
                        else _list(img.get("reasons_after", img.get("reasons")))[:4]}
    return {"verdict": found.get("verdict"), "verdict_before": found.get("verdict_before", found.get("verdict")),
            "warning": found.get("warning"), "images": images}


LEDGER_TOTALS = ("extent_bytes", "code_bytes", "c_bytes", "asm_bytes", "functions", "functions_matched_c",
                 "functions_asm", "sum_equals_extent")


def image_totals(item) -> dict:
    """One image's totals from ``report.json``: catalogued functions, resolution outcomes, exact
    bytes, the strict queue, donors, the ledger's byte classes and the solver's coverage."""
    item = _dict(item)
    resolve, queue, donors = _dict(item.get("resolve")), _dict(item.get("queue")), _dict(item.get("donors"))
    out = {"functions": item.get("functions"), "resolved": resolve.get("statuses"),
           "exact_bytes": resolve.get("exact_bytes"), "queued_strict_exact": queue.get("strict_exact"),
           "awaiting_symbols": queue.get("assumed"), "assembly": queue.get("assembly"),
           "donor_verified": sum(len(_list(r.get("functions"))) for r in _list(donors.get("verified"))
                                 if isinstance(r, dict)),
           "errors": sorted(k for k, v in item.items() if isinstance(v, dict) and "error" in v)}
    ledger = _dict(item.get("ledger"))
    totals = _dict(ledger.get("totals"))
    if totals:
        out["ledger"] = {**{k: totals[k] for k in LEDGER_TOTALS if k in totals},
                         "violations": _dict(ledger.get("violations"))}
    coverage = _dict(_dict(item.get("solver")).get("coverage"))
    if coverage:
        out["coverage"] = {k: coverage.get(k) for k in ("functions", "attempted", "not_attempted", "unexplained")}
    return out


def report_view(run: Path, image: str | None = None) -> dict:
    """The keys ``status`` adds from ``report.json``: ``report`` (file, status, failed stages, or
    the read error), ``discovery``, ``small_data`` and ``image_totals``; None / {} when absent."""
    report = read_report(run)
    if report is None:
        return {"report": None, "discovery": None, "small_data": None, "image_totals": {}}
    head = {"file": str(Path(run) / "report.json")}
    if "error" in report and len(report) == 1:
        return {"report": {**head, "error": report["error"]}, "discovery": None, "small_data": None,
                "image_totals": {}}
    failed = _dict(report.get("failed_stages"))
    head.update({"status": report.get("status"), "seconds": report.get("seconds"),
                 "failed_stages": {k: v for k, v in failed.items() if image is None or k == image}})
    small = report.get("small_data")
    return {"report": head, "discovery": discovery_view(report.get("discovery"), image),
            "small_data": dict(small) if isinstance(small, dict) else None,
            "image_totals": {name: image_totals(item) for name, item in _dict(report.get("images")).items()
                             if image is None or name == image}}


def status(run: Path, image: str | None = None, show: int = 5) -> dict:
    run = Path(run)
    log = solver.RoundsLog(run / "rounds.jsonl")
    records = log.read()
    session = log.latest_session()
    current = [r for r in records if r.get("session") == session and (image is None or r.get("image") == image)]
    invocation = {}
    path = run / "solver" / "invocation.json"
    if path.is_file():
        invocation = json.loads(path.read_text())
    mode = solver.policy(invocation.get("effort"))
    observations = solver.Observations(run / "solver" / "observations.json")     # read, never saved here
    ceiling = invocation.get("search_ceiling")
    ceiling = resolvemod.SEARCH_CEILING if ceiling is None else ceiling
    store = resolvemod.CaseStore(run / "state", readonly=True)
    try:
        cases = [c for c in store.all() if image is None or c.image == image]
    finally:
        store.close()
    stretches: dict[str, solver.Stretches] = {}
    for case in cases:
        if case.image not in stretches:
            saved = run / "solver" / f"round1_{case.image}.json"
            info = None
            if saved.is_file():
                try:
                    info = (json.loads(saved.read_text()).get("resolve") or {}).get("unknown_toolchain")
                except ValueError:
                    info = None
            stretches[case.image] = solver.Stretches(info)
    groups: dict[str, dict] = {}
    for case in cases:
        if case.status not in solver.UNRESOLVED:
            continue
        decision = solver.decide(case, mode, observations, unknown=stretches[case.image], ceiling=ceiling,
                                 model=bool(invocation.get("refine_command")))
        group = groups.setdefault(decision.cause, {"functions": 0, "bytes": 0, "next": {}, "cases": []})
        group["functions"] += 1
        group["bytes"] += case.size
        key = decision.technique or "stop"
        group["next"][key] = group["next"].get(key, 0) + 1
        group["cases"].append((case.best_score if case.best_score is not None else 1 << 30, case.size, case.name,
                               decision.technique or decision.reason))
    for group in groups.values():
        group["cases"].sort()
        group["examples"] = [{"name": n, "score": None if s == 1 << 30 else s, "size": z, "next": t}
                             for s, z, n, t in group.pop("cases")[:show]]
    return {"run": str(run), "session": session, "mode": mode.name,
            "rounds": [round_line(r) for r in current if r.get("kind") == "round"],
            "stops": [{"image": r.get("image"), "after_round": r.get("after_round"), "reason": r.get("reason")}
                      for r in current if r.get("kind") == "stop"],
            "credit": solver.credit(cases),
            "remaining_by_cause": dict(sorted(groups.items(), key=lambda kv: -kv[1]["bytes"])),
            **report_view(run, image)}


def render_report(result: dict) -> list[str]:
    """Text lines for the keys ``report_view`` adds; none when the run has no ``report.json``."""
    head = result.get("report")
    if not head:
        return []
    if head.get("error"):
        return ["", f"report.json unreadable: {head['error']}"]
    lines = ["", f"report.json (last finished invocation): {head.get('status') or '-'}"
                 + (f", failed stages {head['failed_stages']}" if head.get("failed_stages") else "")]
    found = result.get("discovery")
    if found and found.get("error"):
        lines.append(f"Discovery: failed: {found['error']}")
    elif found and found.get("verdict"):
        lines.append(f"Discovery: catalogue {found['verdict']}"
                     + (f" (before repair: {found['verdict_before']})"
                        if found.get("verdict_before") not in (None, found["verdict"]) else ""))
        for name, img in found.get("images", {}).items():
            lines.append(f"  {name}: {img['verdict_after']}, coverage {img['coverage_after']}, "
                         f"{img['functions_after']} functions"
                         + (f" (+{img['repaired_functions']} repaired)" if img.get("repaired_functions") else ""))
            lines += [f"    - {reason}" for reason in img.get("reasons_after") or []]
        if found.get("warning"):
            lines.append(f"  WARNING: {found['warning']}")
    small = result.get("small_data")
    if small and small.get("error"):
        lines.append(f"Small data: failed: {small['error']}")
    elif small:
        value = small.get("value")
        lines.append("Small data: " + (f"_gp {value}" if value else "no _gp established")
                     + f" (confidence {small.get('confidence') or 'none'}, {small.get('sites', 0)} setting sites)"
                     + (f": {small['reason']}" if small.get("reason") else ""))
        if small.get("project_gp"):
            lines.append(f"  project declares _gp = {small['project_gp']}")
        if small.get("project_gp_error"):
            lines.append(f"  project _gp unreadable: {small['project_gp_error']}")
        if small.get("warning"):
            lines.append(f"  WARNING: {small['warning']}")
    if result.get("image_totals"):
        lines.append("Per image:")
    for name, row in (result.get("image_totals") or {}).items():
        lines.append(f"  {name}: {row.get('functions')} functions; resolved {row.get('resolved')}; exact bytes "
                     f"{row.get('exact_bytes')}; queued strictly exact {row.get('queued_strict_exact')}; awaiting "
                     f"symbols {row.get('awaiting_symbols') or 0}; donor-verified {row.get('donor_verified') or 0}")
        ledger = row.get("ledger")
        if ledger:
            lines.append(f"    ledger: C {ledger.get('c_bytes')} bytes, assembly {ledger.get('asm_bytes')} bytes of "
                         f"{ledger.get('code_bytes')} code bytes; functions matched C "
                         f"{ledger.get('functions_matched_c')}, assembly {ledger.get('functions_asm')}; "
                         f"violations {ledger.get('violations')}")
        cover = row.get("coverage")
        if cover:
            lines.append(f"    coverage: {cover.get('attempted')} of {cover.get('functions')} attempted, "
                         f"{cover.get('not_attempted')} not attempted, {cover.get('unexplained')} unexplained")
        if row.get("errors"):
            lines.append(f"    stage errors: {', '.join(row['errors'])}")
    return lines


def render(result: dict) -> str:
    lines = [f"run {result['run']} (effort {result['mode']}, session {result['session']})", ""]
    lines += result["rounds"] or ["no completed round"]
    lines += [f"Stopped [{s['image']}] after round {s['after_round']}: {s['reason']}" for s in result["stops"]]
    lines += render_report(result)
    credit = result["credit"]
    lines += ["", f"exact C {credit['exact_c']} ({credit['exact_c_bytes']} bytes); awaiting declarations "
                  f"{credit['awaiting_declarations']}; hand-written assembly {credit['handwritten_assembly']}; "
                  f"unresolved {credit['unresolved']}", "", "remaining queue by primary cause:"]
    for cause, group in result["remaining_by_cause"].items():
        nexts = ", ".join(f"{k} {v}" for k, v in sorted(group["next"].items(), key=lambda kv: -kv[1]))
        lines.append(f"  {cause}: {group['functions']} functions, {group['bytes']} bytes; next: {nexts}")
        for example in group["examples"]:
            lines.append(f"    {example['name']} size {example['size']} score {example['score']}: {example['next']}")
    return "\n".join(lines)


@command("decompile-status", "Read-only rounds table and remaining queue by primary cause of a decompile-run")
def decompile_status(args):
    result = status(Path(args.run_dir), getattr(args, "image", None), getattr(args, "show", 5))
    if getattr(args, "format", "text") == "text":
        print(render(result))
        return 0
    return result


def _args(p):
    p.add_argument("run_dir", help="the --out directory of a decompile-run")
    p.add_argument("--image", help="only this image")
    p.add_argument("--show", type=int, default=5, help="example functions per cause")


decompile_status.arguments = _args
