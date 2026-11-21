"""``decompile-dashboard RUN --out DIR``: a local, static, read-only HTML dashboard of a finished
``decompile-run`` (round 12, item 15).

Everything comes from the run's own JSON and tables:
- ``report.json``;
- ``profile.jsonl`` and ``rounds.jsonl``;
- ``discovery/discovery_audit.json`` and ``<image>/toolchain_regions.json``;
- the strict queue's TSV files;
- the case store, opened read-only;
- the review log.

No image is ever opened, so no ROM byte can reach the page.  No source text or assembly is shown,
only counts, names, addresses and causes.  Absolute paths are cut to their last component.  The
page is one HTML file with inline CSS and no script, font or link to anything outside it.  The
run directory is not written: an ``--out`` inside it is refused.  ``dashboard.json`` beside the
page holds the same model for other tools."""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

from ..cli import command

FUNNEL = ("exact", "donor-verified", "assumed", "assembly", "raw", "open")
COLORS = {"exact": "#2e7d32", "donor-verified": "#66bb6a", "assumed": "#f9a825", "assembly": "#1565c0",
          "raw": "#6a1b9a", "open": "#bdbdbd", "c-verified": "#2e7d32", "c-declared": "#81c784",
          "asm-proven": "#1565c0", "asm-declared": "#64b5f6", "data": "#8d6e63", "jump-table": "#a1887f",
          "padding": "#e0e0e0", "fragment": "#ef6c00", "function-unmatched": "#bdbdbd", "unknown": "#757575"}
_ABSOLUTE = re.compile(r"^(?:/|[A-Za-z]:[\\/])")
_EMBEDDED = re.compile(r"(?:/(?:home|Users|mnt|tmp|root|var|opt|srv|private)/|[A-Za-z]:\\)[^\s\"'<>]*")


# ------------------------------------------------------------------------------ reading
def _json(path: Path):
    try:
        return json.loads(path.read_text()) if path.is_file() else None
    except (OSError, ValueError):
        return None


def _jsonl(path: Path) -> list:
    out = []
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                out.append(value)
    return out


def portable(value):
    """Absolute paths cut to their last component, recursively."""
    if isinstance(value, str):
        if _ABSOLUTE.match(value):
            return Path(value.replace("\\", "/")).name
        return _EMBEDDED.sub(lambda m: Path(m.group().replace("\\", "/")).name, value)
    if isinstance(value, dict):
        return {portable(k): portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    return value


def _d(value) -> dict:
    return value if isinstance(value, dict) else {}


def _l(value) -> list:
    return value if isinstance(value, list) else []


def _num(value) -> float:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _raw(solver: dict) -> int:
    """Functions with no case that no donor recovered: ``decompile-export`` leaves these as
    assembly or raw words.  (``coverage.not_attempted`` minus its donor reasons.)"""
    coverage = _d(solver.get("coverage"))
    donor = sum(int(_num(v)) for k, v in _d(coverage.get("reasons")).items() if "donor" in str(k))
    return max(0, int(_num(coverage.get("not_attempted"))) - donor)


def _case_statuses(run: Path) -> dict:
    from ..match import resolve as resolvemod
    out: dict = {}
    if not (run / "state" / "cases.sqlite").is_file():
        return out
    store = resolvemod.CaseStore(run / "state", readonly=True)
    try:
        for case in store.all():
            row = out.setdefault(case.image, {})
            row[case.status] = row.get(case.status, 0) + 1
    finally:
        store.close()
    return out


def collect(run) -> dict:
    """The dashboard model of one run (portable: no absolute path, no image byte)."""
    run = Path(run)
    report = _d(_json(run / "report.json"))
    audit = _d(_json(run / "discovery" / "discovery_audit.json"))
    profile = _jsonl(run / "profile.jsonl")
    rounds = _jsonl(run / "rounds.jsonl")
    statuses = _case_statuses(run)
    images = {}
    for name, item in sorted(_d(report.get("images")).items()):
        item = _d(item)
        queue, donors, resolve = _d(item.get("queue")), _d(item.get("donors")), _d(item.get("resolve"))
        functions = int(_num(item.get("functions")))
        solver = _d(item.get("solver"))
        raw = _raw(solver)
        funnel = {"exact": int(_num(queue.get("strict_exact"))),
                  "donor-verified": int(_num(donors.get("functions_recovered"))),
                  "assumed": int(_num(queue.get("assumed"))) + int(_num(donors.get("functions_awaiting_symbols"))),
                  "assembly": int(_num(queue.get("assembly"))), "raw": raw}
        funnel["open"] = max(0, functions - sum(funnel.values()))
        ledger = _d(_d(item.get("ledger")).get("totals"))
        regions_file = _d(_json(run / name / "toolchain_regions.json"))
        units: dict = {}
        for row in _l(donors.get("verified")):
            if isinstance(row, dict) and row.get("unit"):
                entry = units.setdefault(row["unit"], {"functions": 0, "bytes": 0, "kind": row.get("source_kind"),
                                                       "donor": row.get("donor")})
                entry["functions"] += len(_l(row.get("functions"))) or 1
                entry["bytes"] += int(_num(row.get("size")))
        node = _d(_d(audit.get("images")).get(name))
        images[name] = {
            "functions": functions, "funnel": funnel, "case_statuses": statuses.get(name, {}),
            "resolve": {k: resolve.get(k) for k in ("statuses", "builds", "seconds", "exact_bytes")},
            "ledger": {"extent": ledger.get("extent"), "extent_bytes": ledger.get("extent_bytes"),
                       "bytes": _d(ledger.get("bytes")), "sum_equals_extent": ledger.get("sum_equals_extent"),
                       "violations": _d(_d(item.get("ledger")).get("violations"))},
            "toolchain_regions": {"summary": {k: _d(item.get("toolchain_regions")).get(k)
                                              for k in ("regions", "verdicts", "functions_by_verdict")},
                                  "regions": [{"start": r.get("start"), "end": r.get("end"), "functions": r.get("functions"),
                                               "bytes": r.get("bytes"), "lane": r.get("lane"), "verdict": r.get("verdict"),
                                               "top": _d((_l(r.get("ranked")) or [{}])[0])}
                                              for r in _l(regions_file.get("regions")) if isinstance(r, dict)]},
            "graph": {"vram": node.get("vram"), "end": node.get("end"), "bytes": node.get("bytes"),
                      "coverage": node.get("coverage"), "verdict": node.get("verdict"),
                      "calls": _d(node.get("direct_calls")), "pointers": _d(node.get("pointers")),
                      "islands": _d(node.get("islands")).get("count"),
                      "overlays": node.get("overlays")},
            "units": {"count": len(units), "needs_tu": statuses.get(name, {}).get("needs-tu", 0),
                      "largest": sorted(({"unit": u, **v} for u, v in units.items()),
                                        key=lambda r: (-r["functions"], r["unit"]))[:12]},
            "hard_cases": [r for r in _l(item.get("hard_queue")) if isinstance(r, dict)],
            "remaining_by_cause": _d(solver.get("remaining_by_cause")),
            "stages": _d(_d(solver.get("sieve")).get("stages")),
            "coverage": _d(solver.get("coverage")),
        }
    stages: dict = {}
    for row in profile:
        entry = stages.setdefault(row.get("stage") or "?", {"wall_s": 0.0, "child_cpu_s": 0.0, "python_cpu_s": 0.0,
                                                           "requested": 0, "cache_hits": 0, "cache_misses": 0,
                                                           "distinct_objects": 0, "peak_rss_mb": 0.0})
        builds = _d(row.get("builds"))
        for key in ("wall_s", "child_cpu_s", "python_cpu_s"):
            entry[key] = round(entry[key] + _num(row.get(key)), 2)
        for key in ("requested", "cache_hits", "cache_misses", "distinct_objects"):
            entry[key] += int(_num(builds.get(key)))
        entry["peak_rss_mb"] = max(entry["peak_rss_mb"], _num(row.get("peak_rss_mb")))
    try:
        from ..match import review as reviewmod
        items = reviewmod.queue(run, include_decided=True)
        state = reviewmod.replay(run)
        review = {"open": sum(1 for i in items if not i.decision), "decided": sum(1 for i in items if i.decision),
                  "by_kind": {k: sum(1 for i in items if i.kind == k and not i.decision)
                              for k in ("assumed", "near", "donor")},
                  "transactions": state["transactions"], "problems": state["problems"],
                  "next": [{"id": i.id, "kind": i.kind, "name": i.name, "size": i.size, "score": i.score}
                           for i in items if not i.decision][:15]}
    except Exception as error:                                   # the dashboard never fails on one section
        review = {"error": f"{type(error).__name__}: {error}"[:200]}
    model = {"run": f"{run.parent.name}/{run.name}", "status": report.get("status"), "seconds": report.get("seconds"),
             "failed_stages": report.get("failed_stages"), "images": images,
             "discovery": {k: _d(report.get("discovery")).get(k) for k in ("verdict", "verdict_before", "warning")},
             "queue": _d(report.get("queue")), "resources": _d(report.get("resources")),
             "report_profile": _d(report.get("profile")), "profile": stages,
             "rounds": [{k: r.get(k) for k in ("kind", "image", "round", "seconds", "gains", "reason")}
                        for r in rounds], "review": review}
    return portable(model)


# ------------------------------------------------------------------------------ rendering
def _e(value) -> str:
    return html.escape("" if value is None else str(value))


def _table(head, rows) -> str:
    if not rows:
        return "<p class=muted>none recorded</p>"
    out = ["<table><tr>" + "".join(f"<th>{_e(h)}</th>" for h in head) + "</tr>"]
    for row in rows:
        out.append("<tr>" + "".join(f"<td>{_e(c)}</td>" for c in row) + "</tr>")
    return "".join(out) + "</table>"


def _bar(parts: list[tuple[str, float]]) -> str:
    total = sum(v for _k, v in parts if v > 0)
    if total <= 0:
        return "<p class=muted>no data</p>"
    cells, legend = [], []
    for key, value in parts:
        if value <= 0:
            continue
        width = 100.0 * value / total
        cells.append(f'<span class=seg style="width:{width:.3f}%;background:{COLORS.get(key, "#90a4ae")}" '
                     f'title="{_e(key)}: {_e(value)}"></span>')
        legend.append(f'<span class=key><i style="background:{COLORS.get(key, "#90a4ae")}"></i>{_e(key)} {_e(value)}</span>')
    return f"<div class=bar>{''.join(cells)}</div><div class=legend>{''.join(legend)}</div>"


def _section(title: str, body: str) -> str:
    return f"<section><h2>{_e(title)}</h2>{body}</section>"


CSS = """body{font:14px/1.4 system-ui,sans-serif;margin:24px;color:#212121;max-width:1200px}
h1{font-size:22px}h2{font-size:17px;border-bottom:1px solid #ddd;padding-bottom:4px}h3{font-size:15px;margin:12px 0 4px}
table{border-collapse:collapse;margin:6px 0}td,th{border:1px solid #ddd;padding:3px 8px;text-align:left}
th{background:#f5f5f5}.bar{display:flex;height:18px;border:1px solid #ccc;margin:6px 0}.seg{display:block;height:100%}
.legend .key{margin-right:12px;white-space:nowrap}.legend i{display:inline-block;width:10px;height:10px;margin-right:4px}
.muted{color:#757575}.warn{color:#b71c1c}section{margin-bottom:22px}"""


def render(model: dict) -> str:
    imgs = model.get("images") or {}
    parts = [f"<h1>Run {_e(model.get('run'))}</h1>",
             f"<p>status {_e(model.get('status'))}; {_e(model.get('seconds'))} s; discovery "
             f"{_e((model.get('discovery') or {}).get('verdict'))}"
             + (f"; <span class=warn>failed stages {_e(model.get('failed_stages'))}</span>"
                if model.get("failed_stages") else "") + "</p>",
             "<p class=muted>Generated from the run's own records; read-only; no image bytes, no source text. "
             "Strict exact (the run's strict queue) is the only C credit; donor-verified is identical code from another "
             "program's proven source; assumed waits for declared symbols; assembly is reassembling, verified assembly; "
             "raw is catalogued functions with no case and no donor (export fallback, nothing verified); open is the "
             "catalogue minus all of these (floored at 0).</p>"]
    body = []
    for name, img in imgs.items():
        f = img["funnel"]
        body.append(f"<h3>{_e(name)} ({_e(img['functions'])} catalogued functions)</h3>"
                    + _bar([(k, f.get(k, 0)) for k in FUNNEL])
                    + _table(["case status", "cases"], sorted(img["case_statuses"].items())))
    parts.append(_section("Funnel: exact / donor-verified / assumed / assembly / raw / open", "".join(body)))
    body = []
    for name, img in imgs.items():
        led = img["ledger"]
        body.append(f"<h3>{_e(name)} {_e(led.get('extent'))} ({_e(led.get('extent_bytes'))} bytes; sum equals extent: "
                    f"{_e(led.get('sum_equals_extent'))}; violations {_e(led.get('violations') or 'none')})</h3>"
                    + _bar(sorted((k, _num(v)) for k, v in (led.get("bytes") or {}).items())))
    parts.append(_section("Byte ledger", "".join(body)))
    rows = [[n, g["graph"].get("vram"), g["graph"].get("end"), g["graph"].get("bytes"), g["functions"],
             g["graph"].get("coverage"), g["graph"].get("verdict"), (g["graph"].get("calls") or {}).get("cross_image"),
             (g["graph"].get("calls") or {}).get("missing"), (g["graph"].get("calls") or {}).get("unmapped")]
            for n, g in imgs.items()]
    parts.append(_section("Image graph (nodes; edges as call counts)", _table(
        ["image", "vram", "end", "bytes", "functions", "coverage", "discovery", "calls to other images",
         "missing call targets", "unmapped"], rows)))
    rows = [[n, "not run for this run (discovery-audit --overlays)" if g["graph"].get("overlays") is None
             else json.dumps(g["graph"]["overlays"], default=str)[:300]] for n, g in imgs.items()]
    parts.append(_section("Overlays", _table(["image", "overlay inference"], rows)))
    rows = [[n, c.get("targets"), c.get("missing"), c.get("interior"), c.get("cross_image"), c.get("unmapped"),
             (g["graph"].get("pointers") or {}).get("to_uncatalogued_code"), g["graph"].get("islands")]
            for n, g in imgs.items() for c in [g["graph"].get("calls") or {}]]
    parts.append(_section("Call graph summary", _table(
        ["image", "direct call targets", "missing", "interior", "cross-image", "unmapped",
         "pointers to uncatalogued code", "code islands"], rows)))
    body = []
    for name, img in imgs.items():
        tr = img["toolchain_regions"]
        body.append(f"<h3>{_e(name)}: {_e(tr['summary'])}</h3>" + _table(
            ["start", "end", "functions", "bytes", "lane", "verdict", "top recipe", "share"],
            [[r["start"], r["end"], r["functions"], r["bytes"], r["lane"], r["verdict"], r["top"].get("recipe"),
              r["top"].get("share")] for r in tr["regions"]]))
    parts.append(_section("Toolchain regions", "".join(body)))
    body = []
    for name, img in imgs.items():
        u = img["units"]
        body.append(f"<h3>{_e(name)}: {_e(u['count'])} donor units; {_e(u['needs_tu'])} cases need a translation unit</h3>"
                    + _table(["unit", "functions", "bytes", "kind", "donor"],
                             [[r["unit"], r["functions"], r["bytes"], r["kind"], r["donor"]] for r in u["largest"]]))
    parts.append(_section("Translation units", "".join(body)))
    body = []
    for name, img in imgs.items():
        body.append(f"<h3>{_e(name)}</h3>"
                    + _table(["cause", "functions", "bytes"],
                             [[r.get("cause"), r.get("functions"), r.get("bytes")] for r in img["hard_cases"]])
                    + _table(["remaining by primary cause", "functions", "bytes", "next"],
                             [[c, v.get("functions"), v.get("bytes"), v.get("next")]
                              for c, v in img["remaining_by_cause"].items()]))
    parts.append(_section("Hard-case clusters", "".join(body)))
    body = []
    for name, img in imgs.items():
        rows = [[s, _d(v).get("settled"), _d(v).get("builds"),
                 round(_d(v).get("builds", 0) / _d(v)["settled"], 1) if _d(v).get("settled") else "-"]
                for s, v in img["stages"].items()]
        body.append(f"<h3>{_e(name)}: coverage {_e(img['coverage'].get('attempted'))} of "
                    f"{_e(img['coverage'].get('functions'))} attempted</h3>"
                    + _table(["stage", "settled", "builds", "builds per settled"], rows))
    parts.append(_section("Stage yield", "".join(body)))
    prof = model.get("profile") or {}
    parts.append(_section("Build cost", _table(
        ["stage", "wall s", "child CPU s", "python CPU s", "builds requested", "distinct objects"],
        [[s, v["wall_s"], v["child_cpu_s"], v["python_cpu_s"], v["requested"], v["distinct_objects"]]
         for s, v in prof.items()])))
    parts.append(_section("Cache", _table(
        ["stage", "hits", "misses", "hit rate"],
        [[s, v["cache_hits"], v["cache_misses"],
          f"{v['cache_hits'] / (v['cache_hits'] + v['cache_misses']):.1%}" if v["cache_hits"] + v["cache_misses"] else "-"]
         for s, v in prof.items()])))
    res = model.get("resources") or {}
    mem = _d(res.get("memory"))
    parts.append(_section("Resources", _table(
        ["jobs", "max jobs", "limited by", "memory total MB", "available MB", "swap used MB", "peak RSS MB (stages)"],
        [[res.get("jobs"), res.get("max_jobs"), res.get("limited_by"), mem.get("total_mb"), mem.get("available_mb"),
          mem.get("swap_used_mb"), max((v["peak_rss_mb"] for v in prof.values()), default=None)]])))
    rv = model.get("review") or {}
    if rv.get("error"):
        body = f"<p class=warn>review queue unavailable: {_e(rv['error'])}</p>"
    else:
        body = (f"<p>{_e(rv.get('open'))} open, {_e(rv.get('decided'))} decided; {_e(rv.get('by_kind'))}; "
                f"{_e(rv.get('transactions'))} review transaction(s)"
                + (f"; <span class=warn>log problems {_e(rv.get('problems'))}</span>" if rv.get("problems") else "")
                + "</p>" + _table(["item", "kind", "function", "size", "score"],
                                  [[i["id"], i["kind"], i["name"], i["size"], i["score"]] for i in rv.get("next", [])]))
    parts.append(_section("Review queue", body))
    return ("<!DOCTYPE html><html lang=en><head><meta charset=utf-8>"
            f"<title>n64dk run {_e(model.get('run'))}</title><style>{CSS}</style></head><body>"
            + "".join(parts) + "</body></html>\n")


def audit(page: str) -> dict:
    """Self-checks of a generated page: external references, scripts, absolute paths, control
    characters."""
    lowered = page.lower()
    return {"bytes": len(page.encode()), "external": sum(lowered.count(s) for s in ("http://", "https://", "src=",
                                                                                     "<link", "@import", "url(")),
            "scripts": lowered.count("<script"), "machine_paths": sorted(set(_EMBEDDED.findall(page)))[:10],
            "control_characters": sum(1 for ch in page if ord(ch) < 32 and ch not in "\n\t\r")}


def write(run, out) -> dict:
    run, out = Path(run).resolve(), Path(out).resolve()
    if out == run or run in out.parents:
        raise ValueError("--out must be outside the run directory: the dashboard never writes into a run")
    model = collect(run)
    page = render(model)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page)
    (out / "dashboard.json").write_text(json.dumps(model, indent=1, sort_keys=True, default=str) + "\n")
    return {"out": out.name, "files": ["index.html", "dashboard.json"], "audit": audit(page),
            "images": sorted(model["images"])}


@command("decompile-dashboard", "Static, local, read-only HTML dashboard of a decompile-run (no ROM bytes)")
def decompile_dashboard(args):
    try:
        return write(args.run, args.out)
    except ValueError as error:
        raise SystemExit(str(error)) from None


def _args(p):
    p.add_argument("run", help="the --out directory of a decompile-run")
    p.add_argument("--out", required=True, help="output directory (outside the run)")


decompile_dashboard.arguments = _args
