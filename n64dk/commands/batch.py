"""``decompile-batch``: a set of ROMs, one unattended run each, everything matched shared.

``decompile-init`` and ``decompile-run`` handle one title.  A collection is more than the sum of
its titles: studios reused engines, publishers reused middleware, and every title links some
build of the SDK.  This command runs the titles one after another with one library
(``decompile-run --library``): what is strictly exact in one title is offered, as verified
donor units, to every other.

    n64dk decompile-batch a.z64 b.z64 ... --out DIR [--donors donors.toml] [--refine-command CMD]

Layout::

    <out>/<title>/           the project ``decompile-init`` writes (the ROM is linked, never written)
    <out>/<title>.run/       that title's run output and state
    <out>/library/           one donor description per finished title
    <out>/batch.json, batch.md

* **Order.**  Titles run in the order given.  With ``--passes 2`` (the default) every title is
  visited again after all have run once, so the first titles also receive what later ones
  matched.  A second visit recomputes the donor stage (its inputs changed) and continues the
  stored cases; it never lowers a result.
* **One at a time.**  Titles are never run in parallel: memory, not CPU, is the limit of a sweep.
* **Resumable.**  A title whose project exists is not surveyed again (``--force-init`` does);
  ``decompile-run`` checkpoints its own stages.
* **A failing title is reported and the batch goes on.**
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from ..cli import command

GOOD = ("exact", "assumed", "assembly")


class time_limit:
    """Stop a step that does not finish: one title's survey must not hold up a whole batch.
    Uses the alarm signal where the platform has one (main thread only); elsewhere it is no limit."""

    def __init__(self, seconds: int, message: str):
        self.seconds, self.message, self.previous = int(seconds or 0), message, None

    def __enter__(self):
        import signal
        if self.seconds > 0 and hasattr(signal, "SIGALRM"):
            def late(*_args):
                raise TimeoutError(self.message)
            try:
                self.previous = signal.signal(signal.SIGALRM, late)
                signal.alarm(self.seconds)
            except ValueError:                                # not the main thread
                self.previous = None
        return self

    def __exit__(self, *_exc):
        import signal
        if self.previous is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.previous)
        return False


def title_slug(title: str | None, sha1: str | None, taken: set[str]) -> str:
    """A directory name for a title: its header name, made unique with the ROM's digest."""
    base = re.sub(r"[^a-z0-9]+", "_", (title or "").lower()).strip("_") or "rom"
    slug = base if base not in taken else f"{base}_{(sha1 or '')[:8] or len(taken)}"
    while slug in taken:
        slug += "_"
    taken.add(slug)
    return slug


def _shares(out: dict) -> dict:
    """``identical``: matched from C (donor units, exact, byte-identical awaiting symbols).
    ``accounted``: those plus hand-written code kept as assembly that assembles to the same bytes."""
    out["identical"] = out["donors"] + out["exact"] + out["assumed"]
    out["share"] = round(out["identical"] / out["functions"], 4) if out["functions"] else 0.0
    out["accounted"] = out["identical"] + out.get("assembly", 0)
    out["accounted_share"] = round(out["accounted"] / out["functions"], 4) if out["functions"] else 0.0
    return out


def tally(images: dict) -> dict:
    """One title's headline numbers from ``decompile-run``'s per-image summary."""
    out = {"functions": 0, "donors": 0, "exact": 0, "assumed": 0, "assembly": 0, "near": 0, "open": 0,
           "does_not_build": 0}
    for item in images.values():
        resolved = item.get("resolved") or {}
        out["functions"] += item.get("functions") or 0
        out["donors"] += item.get("donor_verified") or 0
        out["assumed"] += item.get("donor_assumed") or 0
        for key, status in (("exact", "exact"), ("assumed", "assumed"), ("assembly", "assembly"), ("near", "near"),
                            ("open", "open"),
                            ("does_not_build", "does-not-build")):
            out[key] += resolved.get(status, 0)
    return _shares(out)


def stored_tally(run_out: Path, images: dict) -> dict | None:
    """The same numbers from the case store, which holds every pass's results (a later pass only
    reports the cases it touched)."""
    from ..match import resolve as resolvemod
    if not (run_out / "state" / "cases.sqlite").is_file():
        return None
    store = resolvemod.CaseStore(run_out / "state", readonly=True)
    try:
        cases = store.all()
    finally:
        store.close()
    out = {"functions": 0, "donors": 0, "exact": 0, "assumed": 0, "assembly": 0, "near": 0, "open": 0,
           "does_not_build": 0}
    for name, item in images.items():
        donated, waiting = set(), set()
        stage = run_out / name / "stage_donors.json"
        if stage.is_file():
            stored = json.loads(stage.read_text())
            for row in stored.get("verified", []):
                donated.update(row.get("function_vrams") or [])
            for row in stored.get("assumed", []):
                waiting.update(row.get("function_vrams") or [])
            waiting -= donated
        out["functions"] += item.get("functions") or 0
        out["donors"] += len(donated)
        out["assumed"] += len(waiting)
        donated |= waiting
        for case in cases:
            if case.image != name or case.vram in donated:
                continue
            key = {"does-not-build": "does_not_build"}.get(case.status, case.status)
            if key in out:
                out[key] += 1
    return _shares(out)


# Why a function is not matched, in the order the question is asked.  Each class names what would
# have to change for it: a compiler to add, the catalogue to fix, or a better draft.
REASONS = ("compiler not in the catalogue", "catalogue: boundaries or no draft", "draft does not compile",
           "draft within reach", "small function, draft far off", "larger function, draft far off")
SMALL_FUNCTION = 0x80


def diagnosis(run_out: Path, images: dict) -> dict | None:
    """Where every function of a finished run ended, and for the unmatched ones the kind of
    reason: ``{"matched from C", "hand-written, kept as assembly", <REASONS>...}`` counts."""
    from ..match import resolve as resolvemod
    run_out = Path(run_out)
    if not (run_out / "state" / "cases.sqlite").is_file():
        return None
    unknown: dict[str, list[tuple[int, int]]] = {}
    try:
        stored = json.loads((run_out / "report.json").read_text())
        for name, item in (stored.get("images") or {}).items():
            found = ((item.get("resolve") or {}).get("unknown_toolchain") or {}).get("stretches") or []
            unknown[name] = [(int(row["start"], 16), int(row["end"], 16)) for row in found]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    store = resolvemod.CaseStore(run_out / "state", readonly=True)
    try:
        cases = store.all()
    finally:
        store.close()
    out = {"matched from C": 0, "hand-written, kept as assembly": 0, **{reason: 0 for reason in REASONS}}
    for name in images:
        donated = set()
        stage = run_out / name / "stage_donors.json"
        if stage.is_file():
            held = json.loads(stage.read_text())
            for row in held.get("verified", []) + held.get("assumed", []):
                donated.update(row.get("function_vrams") or [])
        out["matched from C"] += len(donated)
        for case in cases:
            if case.image != name or case.vram in donated:
                continue
            if case.status in ("exact", "assumed"):
                out["matched from C"] += 1
            elif case.status == "assembly":
                out["hand-written, kept as assembly"] += 1
            elif any(start <= case.vram <= end for start, end in unknown.get(name, [])):
                out[REASONS[0]] += 1
            elif case.status in ("blocked-boundary", "needs-tu", "needs-source", "probable-handwritten", "stale"):
                out[REASONS[1]] += 1
            elif case.status == "does-not-build":
                out[REASONS[2]] += 1
            elif case.status == "near":
                out[REASONS[3]] += 1
            else:
                out[REASONS[4] if case.size <= SMALL_FUNCTION else REASONS[5]] += 1
    return out


def markdown(report: dict) -> str:
    lines = ["# decompile-batch report", "", f"Titles: {len(report['titles'])}; passes: {report['passes']}; "
             f"wall time: {report['seconds']} s", "",
             "`identical` is matched from C; `assembly` is hand-written code kept as assembly source that",
             "assembles to the same bytes; `accounted` is both.", "",
             "| title | functions | donors | exact | assumed | identical | share | assembly | accounted | seconds |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["titles"]:
        final = row.get("final")
        if not final:
            lines.append(f"| {row['title']} | | | | | failed: {str(row.get('error'))[:60]} | | | | |")
            continue
        lines.append(f"| {row['title']} | {final['functions']:,} | {final['donors']:,} | {final['exact']:,} | "
                     f"{final['assumed']:,} | {final['identical']:,} | {final['share']:.1%} | "
                     f"{final.get('assembly', 0):,} | {final.get('accounted_share', final['share']):.1%} | "
                     f"{row['seconds']:.0f} |")
    total = report["total"]
    lines += [f"| **all** | **{total['functions']:,}** | **{total['donors']:,}** | **{total['exact']:,}** | "
              f"**{total['assumed']:,}** | **{total['identical']:,}** | **{total['share']:.1%}** | "
              f"**{total.get('assembly', 0):,}** | **{total.get('accounted_share', total['share']):.1%}** | |", ""]
    rows = [(row["title"], row["diagnosis"]) for row in report["titles"] if row.get("diagnosis")]
    if rows:
        keys = ["matched from C", "hand-written, kept as assembly", *REASONS]
        lines += ["## Where every function ended", "",
                  "For the unmatched ones, what would have to change: a compiler the catalogue lacks, the",
                  "catalogue itself, or the draft (within reach: 10 or fewer differences).", "",
                  "| title | " + " | ".join(keys) + " |", "|---|" + "---:|" * len(keys)]
        whole = {key: 0 for key in keys}
        for title, found in rows:
            lines.append(f"| {title} | " + " | ".join(f"{found.get(key, 0):,}" for key in keys) + " |")
            for key in keys:
                whole[key] += found.get(key, 0)
        count = sum(whole.values()) or 1
        lines += ["| **all** | " + " | ".join(f"**{whole[key]:,}**" for key in keys) + " |",
                  "| share | " + " | ".join(f"{whole[key] / count:.1%}" for key in keys) + " |", ""]
    offered = [(row["title"], item) for row in report["titles"] for item in row.get("source_trees") or []]
    if offered:
        lines += ["## Source trees", "",
                  "Each tree was measured against each title after its first run: units, include directories,",
                  "recipes and configuration macros come from building it and comparing with the title.", "",
                  "| title | tree | units | compiled | title's functions found | macros set | result |",
                  "|---|---|---:|---:|---:|---|---|"]
        for title, item in offered:
            lines.append(f"| {title} | {Path(str(item.get('source'))).name} | {item.get('units_in_tree') or 0:,} | "
                         f"{item.get('compiled') or 0:,} | {item.get('target_functions_found') or 0:,} | "
                         f"{' '.join(item.get('macros') or []) or '-'} | "
                         f"{'donor of this title' if item.get('donors') else item.get('reason') or 'not used'} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def offer_trees(row: dict, trees: list[Path], out: Path, donors: str | None, toolchain: str | None = None,
                jobs: int = 4, importer=None, recipes_of=None) -> bool:
    """Measure every source tree against a title that has just run and write the title's donors
    file with those that give it anything (after the batch's own donors).  True when the title
    should run again with them.  Nobody says which tree belongs to which title: a tree that
    shares nothing with a title is measured, reported and left out."""
    from . import sourcedonor
    importer, recipes_of = importer or sourcedonor.import_tree, recipes_of or sourcedonor.recipes_of_run
    recipes = recipes_of(Path(row["run"]))
    row["source_trees"] = []
    tables = []
    for tree in trees:
        if not recipes:
            row["source_trees"].append({"source": str(tree), "donors": None,
                                        "reason": "the run matched nothing, so it names no candidate recipe"})
            continue
        found = importer(tree, Path(row["project"]), out / "trees" / row["title"] / tree.name, recipes,
                         toolchain=toolchain, jobs=jobs)
        row["source_trees"].append({key: found.get(key) for key in (
            "source", "name", "donors", "reason", "units_in_tree", "compiled", "failed_units", "units_by_recipe",
            "macros", "target_functions_found", "target_functions_compared")})
        if found.get("donors"):
            tables.append(Path(found["donors"]).read_text())
    if not tables:
        return False
    merged = out / "trees" / f"{row['title']}.donors.toml"
    merged.parent.mkdir(parents=True, exist_ok=True)
    merged.write_text((Path(donors).read_text() + "\n" if donors else "") + "\n".join(tables))
    row["donors"] = str(merged)
    return True


LEGACY_BUDGET = 100                  # the batch's builds per function before effort modes reached it


def budget_argv(args) -> list[str]:
    """``--budget`` for each title's run: the batch's own when given; otherwise the effort mode's
    (exhaustive: 400), and the batch's historical 100 only when no mode is chosen."""
    budget = getattr(args, "budget", None)
    if budget is None and not getattr(args, "effort", None):
        budget = LEGACY_BUDGET
    return [] if budget is None else ["--budget", str(budget)]


def export_title(project: Path, run: Path, tree: Path, toolchain: str | None) -> tuple[dict, dict]:
    """A finished run's export tree and status table.  A failure is recorded, never raised: the
    run itself succeeded, so the title is not failed and the batch goes on."""
    from .export import export as export_tree
    from .runstatus import status as run_status
    try:
        record = {"out": str(tree), **{k: v for k, v in export_tree(project, run, tree, toolchain=toolchain).items()
                                       if k in ("functions", "matched", "from_c")}}
    except Exception as error:                               # noqa: BLE001 - a report, never a reason to stop
        record = {"out": str(tree), "error": f"{type(error).__name__}: {error}"[-300:]}
    try:
        status = run_status(run, None, 3)
    except Exception as error:                               # noqa: BLE001 - a report, never a reason to stop
        status = {"error": f"{type(error).__name__}: {error}"[-300:]}
    return record, status


@command("decompile-batch", "Run decompile-init and decompile-run over a set of ROMs, one at a time, sharing every "
                            "strictly exact function between them through a library")
def decompile_batch(args):
    """One heavy job at a time, with a preflight and a lease (see ``n64dk.safety``); every
    title's run shares them."""
    import sys
    from .. import safety
    try:
        with safety.heavy_job("decompile-batch", Path(args.out), wait_for_lock=getattr(args, "wait_for_lock", False),
                              force=getattr(args, "force_resources", False), argv=sys.argv[1:]):
            return _decompile_batch(args)
    except safety.Busy as error:
        raise SystemExit(str(error)) from None


def _decompile_batch(args):
    if args.donors and not Path(args.donors).is_file():
        # A file named on the command line that is not there is the operator's mistake, not a
        # condition of one title: every title would run without donors and report nothing.
        raise SystemExit(f"--donors {args.donors}: no such file")
    from ..rom import Rom
    from . import blind
    from .run import decompile_run
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    library = Path(args.library).resolve() if args.library else out / "library"
    started, taken, titles = time.time(), set(), []
    for rom_path in args.rom:
        rom_path = Path(rom_path)
        try:
            rom = Rom.load(rom_path)
            name, digest = rom.header.title, rom.sha1
        except Exception as error:
            titles.append({"title": rom_path.name, "rom": str(rom_path), "error": f"cannot read the ROM: {error}"[-300:],
                           "seconds": 0.0})
            continue
        slug = title_slug(name or rom_path.stem, digest, taken)
        titles.append({"title": slug, "rom": str(rom_path), "sha1": digest, "project": str(out / slug),
                       "run": str(out / f"{slug}.run"), "passes": [], "seconds": 0.0})

    def run_arguments(row: dict) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        decompile_run.arguments(parser)
        argv = ["--project", row["project"], "--all-images", "--out", row["run"], "--library", str(library),
                *budget_argv(args)]
        if args.max_size:
            argv += ["--max-size", hex(args.max_size)]
        if getattr(args, "search_economics", False):
            argv += ["--search-economics"]
        if row.get("donors") or args.donors:
            argv += ["--donors", str(Path(row.get("donors") or args.donors).resolve())]
        if args.toolchain_dir:
            argv += ["--toolchain-dir", args.toolchain_dir]
        if args.workers:
            argv += ["--workers", str(args.workers)]
        if args.refine_command:
            argv += ["--refine-command", args.refine_command]
        if args.refine_jobs:
            argv += ["--refine-jobs", str(args.refine_jobs)]
        if getattr(args, "effort", None):
            argv += ["--effort", args.effort]
        return parser.parse_args(argv)

    trees = [Path(tree).resolve() for tree in (getattr(args, "source_tree", None) or [])]
    for tree in trees:
        if not tree.is_dir():
            raise SystemExit(f"--source-tree {tree}: no such directory")

    for number in range(1, max(1, args.passes) + 1):
        for row in titles:
            if row.get("error") and "project" not in row:
                continue
            began = time.time()
            try:
                project = Path(row["project"])
                if not (project / "n64dk.toml").is_file() or (args.force_init and number == 1):
                    with time_limit(args.init_timeout, f"the survey did not finish in {args.init_timeout} s"):
                        blind.build_project(Path(row["rom"]), project, calibrate=args.calibrate,
                                            toolchain=args.toolchain_dir or os.environ.get("N64DK_TOOLCHAIN"))
                result = decompile_run(run_arguments(row))
                if trees and "source_trees" not in row and offer_trees(
                        row, trees, out, args.donors, args.toolchain_dir, args.workers or 4):
                    result = decompile_run(run_arguments(row))      # the same visit again, with the trees' units
                images = result.get("images") or {}
                row["passes"].append({"pass": number, **tally(images), "status": result.get("status")})
                row["final"] = stored_tally(Path(row["run"]), images) or tally(images)
                try:
                    row["diagnosis"] = diagnosis(Path(row["run"]), images)
                except Exception:                            # noqa: BLE001 - a report, never a reason to stop
                    row["diagnosis"] = None
                if getattr(args, "export", False):
                    # the tree ready to commit, and the run's status table beside it
                    row["export"], row["status"] = export_title(
                        project, Path(row["run"]), out / f"{row['title']}.export",
                        args.toolchain_dir or os.environ.get("N64DK_TOOLCHAIN"))
                row.pop("error", None)
            except (Exception, SystemExit) as error:        # one title must not stop the others
                row["error"] = f"pass {number}: {error}"[-300:]
            row["seconds"] = round(row["seconds"] + time.time() - began, 1)
            report = _report(titles, number, started, library)
            (out / "batch.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
            (out / "batch.md").write_text(markdown(report))
    report = _report(titles, max(1, args.passes), started, library)
    return {"titles": len(titles), "failed": [row["title"] for row in titles if row.get("error")],
            "total": report["total"], "report": str(out / "batch.md"), "library": str(library),
            "seconds": report["seconds"]}


def _report(titles: list[dict], passes: int, started: float, library: Path) -> dict:
    total = {"functions": 0, "donors": 0, "exact": 0, "assumed": 0, "identical": 0, "assembly": 0, "accounted": 0}
    for row in titles:
        for key in total:
            total[key] += (row.get("final") or {}).get(key, 0)
    total["share"] = round(total["identical"] / total["functions"], 4) if total["functions"] else 0.0
    total["accounted_share"] = round(total["accounted"] / total["functions"], 4) if total["functions"] else 0.0
    return {"titles": titles, "passes": passes, "total": total, "library": str(library),
            "seconds": round(time.time() - started, 1)}


def _args(p):
    p.add_argument("rom", nargs="+", help="ROM images (read, never written), in the order to run them")
    p.add_argument("--out", required=True, help="directory for the projects, the runs, the library and the report")
    p.add_argument("--library", help="library directory to share with other batches (default: <out>/library)")
    p.add_argument("--passes", type=int, default=2,
                   help="visits per title; a second visit gives early titles what later ones matched (default: 2)")
    p.add_argument("--donors", help="donors TOML used for every title in addition to the library")
    p.add_argument("--source-tree", action="append", metavar="DIR",
                   help="a C source tree that may belong to one of the titles (repeatable; any layout).  After a "
                        "title's first run every tree is measured against it; one that gives it functions becomes "
                        "a donor of that title and the title runs again (see source-donor)")
    p.add_argument("--toolchain-dir", help="toolchain root (default: N64DK_TOOLCHAIN)")
    p.add_argument("--calibrate", type=int, default=96,
                   help="functions sampled per title to measure its recipe")
    p.add_argument("--budget", type=int, default=None,
                   help="maximum builds per function (default: the --effort mode's, or 100 without --effort)")
    p.add_argument("--max-size", type=lambda s: int(s, 0), default=0, help="largest function attempted (0: no limit)")
    p.add_argument("--search-economics", action="store_true",
                   help="stop searching strata whose searches have returned nothing (see decompile-run)")
    p.add_argument("--workers", type=int, help="worker processes of each run")
    p.add_argument("--refine-command", help="refiner asked about the cases each run leaves (see decompile-run)")
    p.add_argument("--refine-jobs", type=int)
    p.add_argument("--init-timeout", type=int, default=1800,
                   help="seconds one title's survey may take before it is reported as failed (0: no limit)")
    p.add_argument("--force-init", action="store_true", help="survey every ROM again even when its project exists")
    p.add_argument("--effort", choices=("quick", "normal", "deep", "exhaustive"),
                   help="decompile-run effort mode of every title (default: decompile-run's)")
    p.add_argument("--export", action="store_true",
                   help="after each title's run, write its export tree (<out>/<title>.export) and status")


def _all_args(p):
    from .run import _guard_arguments
    _args(p)
    _guard_arguments(p)


decompile_batch.arguments = _all_args


@command("decompile-everything", "Everything the kit can do for a set of ROMs: init, run at the kitchen-sink effort "
                                 "(exhaustive), library sharing, export and status, one title at a time, resumable")
def decompile_everything(args):
    """``decompile-batch`` with ``--effort exhaustive`` and ``--export`` unless given otherwise."""
    args.effort = args.effort or "exhaustive"
    args.export = True
    return decompile_batch(args)


decompile_everything.arguments = _all_args
