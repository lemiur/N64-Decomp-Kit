"""Ablation, attribution and honest coverage accounting (round 12 addendum, item S).

Input: several runs of one title made with the same code, the same function sample and the same
machine.  One run has every subsystem on (``base``), at best a repeat of it (``repeat``: the noise
floor), and the others each change one subsystem: ``minus`` runs switch one off, ``plus`` runs
switch on one that ``base`` leaves off.  Each run gives its export's ``FUNCTIONS.tsv`` (outcome,
``how`` = recipe and settling-stage hint, distance), and optionally its ``events.jsonl`` (builds per
case) and ``profile.jsonl`` (builds and wall time per stage).

Per subsystem (one comparison against ``base``):

* exact functions added (settled with the subsystem, not without), regressions (settled without
  it, not with it), made buildable, made closer, builds added or saved, wall time;
* the per-pair diff of ``regress.compare`` (strict counts, transitions, first differing stage,
  denominator per image) is attached unchanged;
* functions present in only one of the two tables, or attempted in only one (``--only-names``,
  discovery repair, boundaries), are **denominator changes**: never counted as gains or losses;
* functions whose outcome differs between ``base`` and ``repeat`` are **noise**: never attributed.

Per settled function of ``base`` (strict C: ``regress.STRICT``):

(a) its first settling stage, read from its own record (donor, twin, refine, region recipe,
    variant search, draft; plus the assembler lane that built it);
(b) every path seen settling it in any run (a function exact in several ablation runs, by different
    stages or lanes, has several capable paths);
(c) the subsystems whose removal loses it (its marginal set).

Nothing is double counted.  (a) is a partition.  (c) is folded into one exclusive class per
function: ``unique:<subsystem>`` (exactly one removal loses it), ``joint`` (two or more removals
each lose it: jointly necessary), ``redundant`` (no single removal loses it), ``noise``,
``denominator`` (some ablation did not attempt it or does not list it).  The classes sum to the
settled count.  The per-subsystem marginal columns are counterfactuals and are labelled as not
additive.

Nothing here builds, runs or changes a verdict: it reads finished runs.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import regress

STRICT = frozenset(regress.STRICT)                      # strict C: exact, donor
ASSEMBLY = frozenset({"assembly", "donor-assembly"})
AWAITING = frozenset({"awaiting-symbols", "donor-awaiting-symbols", "donor-assembly-awaiting-symbols"})
NOT_ATTEMPTED = "not attempted"
BROKEN = "does-not-build"
# the ``how`` hint of an exact function (commands.export._origin) -> settling stage
ORIGINS = {"external author, strictly verified": "refine", "source of an identical function": "twin",
           "draft, recipe of its region": "region-recipe", "draft after variant search": "search",
           "draft": "draft"}
# assembler lanes that only a subsystem offers
LANES = {"kmc+asg": "small-data", "kmc+gas": "small-data", "kmc+atmacro": "at-macros", "kmc+atload": "at-loads"}
# which subsystem a settling path belongs to (for overlap: a path whose subsystem was removed and
# that still settles is covered by another path)
PATH_SUBSYSTEM = {"donors": "donors", "region-recipe": "region-recipes", "refine": "refine",
                  "small-data": "small-data", "at-macros": "at-macros"}
KINDS = ("base", "repeat", "minus", "plus")


@dataclass
class Run:
    label: str                                   # variant name ("all", "-small-data", "+egraph")
    rows: dict                                   # (image, vram text) -> FUNCTIONS.tsv row
    export: Path | None = None
    subsystem: str = ""
    kind: str = "base"
    builds: int | None = None                    # builds requested (profile.jsonl)
    wall_s: float | None = None                  # the job's wall time (heavy.sh), else the stages' sum
    stages: dict = field(default_factory=dict)   # stage -> {"builds", "wall_s"}
    case_builds: dict = field(default_factory=dict)   # (image, vram text) -> builds of its case
    complete: bool = True
    notes: list = field(default_factory=list)
    disabled: tuple = ()                         # what the run switched off (report.json ``ablation``)


# ---------------------------------------------------------------------------------------------
# reading runs


def _key_of_case(case: str) -> tuple[str, str] | None:
    """``main:8000DDC4:34`` -> ``("main", "0x8000DDC4")`` (FUNCTIONS.tsv keys)."""
    parts = str(case).rsplit(":", 2)
    if len(parts) != 3:
        return None
    try:
        return parts[0], f"0x{int(parts[1], 16):08X}"
    except ValueError:
        return None


def read_events(path: Path) -> dict:
    """Builds per case: the last ``case-*`` event of each case carries the case's builds."""
    out = {}
    path = Path(path)
    if not path.is_file():
        return out
    with path.open() as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = str(event.get("event", ""))
            if kind.startswith("case-") and isinstance(event.get("builds"), int):
                key = _key_of_case(event.get("case", ""))
                if key is not None:
                    out[key] = event["builds"]
    return out


def read_profile(path: Path) -> tuple[int | None, float | None, dict]:
    """(builds requested, wall seconds, per stage) from ``profile.jsonl`` (one record per stage and image)."""
    path = Path(path)
    if not path.is_file():
        return None, None, {}
    builds, wall, stages = 0, 0.0, {}
    with path.open() as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if "stage" not in record:
                continue
            count = int(((record.get("builds") or {}).get("requested")) or 0)
            seconds = float(record.get("wall_s") or 0.0)
            item = stages.setdefault(record["stage"], {"builds": 0, "wall_s": 0.0})
            item["builds"] += count
            item["wall_s"] = round(item["wall_s"] + seconds, 3)
            builds += count
            wall += seconds
    return builds, round(wall, 3), stages


def check_switches(run: Run, ablation: dict | None) -> None:
    """Hold a run to what its variant says it is, from the run's own record of what it switched
    off (report.json ``ablation``, decompile-run's ablation switches): a base, repeat or plus run
    switches nothing off; a minus run switches off at most its own subsystem, and that subsystem
    when it is a switch the run knew.  A run that does not match is marked incomplete with a note,
    and every comparison and summary it enters says so.  A report without the record (made before
    the switches) is taken as it is."""
    if not isinstance(ablation, dict):
        return
    run.disabled = tuple(sorted(ablation.get("disabled") or ()))
    allowed = {run.subsystem} if run.kind == "minus" else set()
    extra = [name for name in run.disabled if name not in allowed]
    if extra:
        run.complete = False
        run.notes.append(f"{run.kind} run {run.label} switched off {', '.join(extra)}, which its variant does not name")
    if run.kind == "minus" and run.subsystem in (ablation.get("available") or ()) and \
            run.subsystem not in run.disabled:
        run.complete = False
        run.notes.append(f"minus run {run.label} did not switch {run.subsystem} off")


def load_run(label: str, export: Path, run: Path | None = None, *, subsystem: str = "", kind: str = "base",
             job: Path | None = None) -> Run:
    """A finished run: its export (required), its run directory (events, profile, report) and the
    heavy.sh ``.post.json`` of its job (exit status, seconds)."""
    rows, root = regress._table(Path(export))
    out = Run(label, rows, root, subsystem, kind)
    if run is not None:
        run = Path(run)
        out.case_builds = read_events(run / "events.jsonl")
        out.builds, out.wall_s, out.stages = read_profile(run / "profile.jsonl")
        try:
            report = json.loads((run / "report.json").read_text())
            if report.get("status") not in (None, "complete"):
                out.complete = False
                out.notes.append(f"run status {report.get('status')}: {report.get('failed_stages')}")
            check_switches(out, report.get("ablation"))
        except (OSError, ValueError):
            out.complete = False
            out.notes.append("no report.json: the run did not finish")
    if job is not None and Path(job).is_file():
        try:
            post = json.loads(Path(job).read_text())
            out.wall_s = float(post.get("seconds", out.wall_s or 0))
            if post.get("exit") not in (0, None):
                out.complete = False
                out.notes.append(f"job exit {post.get('exit')}")
        except (OSError, ValueError):
            pass
    return out


def restrict(run: Run, names: set[str] | None) -> Run:
    """``run`` with only the functions named in ``names`` (the ``--only-names`` sample): stages that
    act on the whole image (donors) would otherwise add functions no other run attempts."""
    if names is None:
        return run
    rows = {key: row for key, row in run.rows.items() if row.get("function") in names}
    return Run(run.label, rows, run.export, run.subsystem, run.kind, run.builds, run.wall_s, run.stages,
               {key: value for key, value in run.case_builds.items() if key in rows}, run.complete, list(run.notes),
               run.disabled)


def read_names(path: Path) -> set[str]:
    """A ``--only-names`` file: one name per line (``#`` comments, blank lines and commas allowed)."""
    out = set()
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0]
        out.update(item.strip() for item in line.split(",") if item.strip())
    return out


def load_variants(folder: Path) -> list[Run]:
    """The runs of one title as ``plan.sh`` lays them out: ``variants.tsv`` (variant, subsystem,
    kind, flags, environment) and ``<variant>/{run,export}``, ``<variant>/job.run.post.json``."""
    folder = Path(folder)
    runs = []
    with (folder / "variants.tsv").open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            where = folder / row["variant"]
            if not (where / "export" / "FUNCTIONS.tsv").is_file():
                continue
            runs.append(load_run(row["variant"], where / "export", where / "run", subsystem=row["subsystem"],
                                 kind=row["kind"], job=where / "job.run.post.json"))
    return runs


# ---------------------------------------------------------------------------------------------
# one function


def outcome(row: dict | None) -> str:
    return (row or {}).get("outcome") or "absent"


def settled(row: dict | None, kinds=STRICT) -> bool:
    return outcome(row) in kinds


def attempted(row: dict | None) -> bool:
    return row is not None and outcome(row) != NOT_ATTEMPTED


def distance(row: dict | None) -> int | None:
    try:
        return int(str((row or {}).get("distance") or "").strip())
    except ValueError:
        return None


def settling_stage(row: dict) -> str:
    """The stage that settled a function, from its own record (``FUNCTIONS.tsv``)."""
    result = outcome(row)
    if result.startswith("donor"):
        return "donors"
    if result == "assembly":
        return "handwritten-assembly"
    hint = regress._stage(row.get("how", ""))
    return ORIGINS.get(hint, hint or "unstated")


def lane(row: dict) -> str:
    """The assembler lane class of the recipe that built it (``standard`` unless a subsystem's lane)."""
    if outcome(row).startswith("donor"):
        return "donor"
    recipe = regress._recipe(row.get("how", ""))
    return LANES.get(recipe.split(" ")[0] if recipe else "", "standard")


def path(row: dict) -> str:
    """A settling path: stage and lane (``draft@small-data``)."""
    return f"{settling_stage(row)}@{lane(row)}"


def path_subsystems(row: dict) -> set[str]:
    out = set()
    stage, kind = settling_stage(row), lane(row)
    for item in (stage, kind):
        if item in PATH_SUBSYSTEM:
            out.add(PATH_SUBSYSTEM[item])
    return out


# ---------------------------------------------------------------------------------------------
# one comparison


def noise(base: Run, repeat: Run | None, kinds=STRICT) -> dict:
    """Functions on which two runs of the same configuration disagree: ``settled`` (settled in one,
    not the other: never attributed) and ``outcome`` (any outcome differs: never counted as made
    closer or buildable)."""
    if repeat is None:
        return {"settled": set(), "outcome": set()}
    keys = set(base.rows) | set(repeat.rows)
    moved = {key for key in keys if settled(base.rows.get(key), kinds) != settled(repeat.rows.get(key), kinds)}
    changed = moved | {key for key in keys if attempted(base.rows.get(key)) and attempted(repeat.rows.get(key))
                       and outcome(base.rows.get(key)) != outcome(repeat.rows.get(key))}
    return {"settled": moved, "outcome": changed}


def compare_pair(base: Run, other: Run, unstable: dict | None = None, kinds=STRICT) -> dict:
    """What the subsystem ``other`` changes adds or costs, read in the direction of the subsystem:
    for a ``minus`` run the subsystem is on in ``base``; for a ``plus`` run it is on in ``other``."""
    unstable = unstable or {"settled": set(), "outcome": set()}
    on, off = (base, other) if other.kind != "plus" else (other, base)
    keys = set(base.rows) | set(other.rows)
    only = {"only_with": 0, "only_without": 0, "attempted_only_with": 0, "attempted_only_without": 0}
    added, lost, buildable, broken, closer, farther, within_noise = [], [], [], [], [], [], []
    added_assembly = lost_assembly = 0
    for key in sorted(keys):
        a, b = on.rows.get(key), off.rows.get(key)
        if a is None or b is None:
            only["only_with" if b is None else "only_without"] += 1
            continue
        if attempted(a) != attempted(b):
            only["attempted_only_with" if attempted(a) else "attempted_only_without"] += 1
            continue
        if not attempted(a):
            continue
        sa, sb = settled(a, kinds), settled(b, kinds)
        if sa != sb and key in unstable["settled"]:
            within_noise.append(key)
            continue
        if sa and not sb:
            added.append(key)
        elif sb and not sa:
            lost.append(key)
        elif not sa and not sb:
            if outcome(b) == BROKEN and outcome(a) != BROKEN:
                buildable.append(key)
            elif outcome(a) == BROKEN and outcome(b) != BROKEN:
                broken.append(key)
            else:
                da, db = distance(a), distance(b)
                if da is not None and db is not None and da != db and key not in unstable["outcome"]:
                    (closer if da < db else farther).append(key)
        added_assembly += int(settled(a, ASSEMBLY) and not settled(b, ASSEMBLY))
        lost_assembly += int(settled(b, ASSEMBLY) and not settled(a, ASSEMBLY))

    def delta(x, y):
        return None if x is None or y is None else round(x - y, 3)

    builds = delta(on.builds, off.builds)
    wall = delta(on.wall_s, off.wall_s)
    stage_builds = {name: (on.stages.get(name, {}).get("builds", 0) - off.stages.get(name, {}).get("builds", 0))
                    for name in sorted(set(on.stages) | set(off.stages))}
    result = {
        "subsystem": other.subsystem or other.label, "variant": other.label, "kind": other.kind,
        "complete": base.complete and other.complete, "notes": base.notes + other.notes,
        "added": len(added), "regressions": len(lost), "net": len(added) - len(lost),
        "made_buildable": len(buildable), "made_unbuildable": len(broken),
        "made_closer": len(closer), "made_farther": len(farther), "within_noise": len(within_noise),
        "added_assembly": added_assembly, "regressions_assembly": lost_assembly,
        "builds_added": builds if builds is not None and builds > 0 else 0,
        "builds_saved": -builds if builds is not None and builds < 0 else 0,
        "builds_known": builds is not None,
        "builds_per_added": (round(builds / len(added), 1) if builds is not None and added else None),
        "builds_on_added": sum(on.case_builds.get(key, 0) for key in added),
        "wall_s_added": wall, "stage_builds_added": {k: v for k, v in stage_builds.items() if v},
        "denominator": only,
        "functions": {"added": [_name(on, k) for k in added][:40], "regressions": [_name(off, k) for k in lost][:40]},
    }
    if base.export is not None and other.export is not None:
        diff = regress.compare(off.export, on.export)        # from without to with the subsystem
        result["regress"] = {k: diff[k] for k in ("strict_before", "strict_after", "transitions", "denominator",
                                                   "causes", "changed")}
        result["regress"]["gained"], result["regress"]["lost"] = len(diff["gained"]), len(diff["lost"])
    return result


def _name(run: Run, key) -> str:
    row = run.rows.get(key) or {}
    return f"{key[0]}:{row.get('function', key[1])}"


# ---------------------------------------------------------------------------------------------
# one title


def attribute(runs: list[Run], kinds=STRICT) -> dict:
    """Attribution for one title (see the module docstring).  ``runs`` holds one ``base``, at most
    one ``repeat`` and any number of ``minus``/``plus`` runs."""
    bases = [run for run in runs if run.kind == "base"]
    if len(bases) != 1:
        raise ValueError(f"one base run is needed, {len(bases)} given")
    base = bases[0]
    repeat = next((run for run in runs if run.kind == "repeat"), None)
    minus = [run for run in runs if run.kind == "minus"]
    plus = [run for run in runs if run.kind == "plus"]
    unstable = noise(base, repeat, kinds)
    done = sorted(key for key, row in base.rows.items() if settled(row, kinds))

    first = Counter(settling_stage(base.rows[key]) for key in done)                         # (a)
    first_paths = Counter(path(base.rows[key]) for key in done)
    capable_count, capable_paths = Counter(), Counter()                                    # (b)
    for key in done:
        seen = {path(run.rows[key]) for run in runs if settled(run.rows.get(key), kinds)}
        capable_count[len(seen)] += 1
        for item in seen:
            capable_paths[item] += 1
    needed_by, outside = {key: set() for key in done}, {key: set() for key in done}       # (c)
    for run in minus:
        for key in done:
            row = run.rows.get(key)
            if row is None or not attempted(row):
                outside[key].add(run.subsystem)
            elif not settled(row, kinds):
                needed_by[key].add(run.subsystem)
    partition, overlap = Counter(), Counter()
    for key in done:
        if key in unstable["settled"]:
            partition["noise"] += 1
        elif outside[key]:
            partition["denominator"] += 1
        elif not needed_by[key]:
            partition["redundant"] += 1
        elif len(needed_by[key]) == 1:
            partition[f"unique:{next(iter(needed_by[key]))}"] += 1
        else:
            partition["joint"] += 1
            for a in needed_by[key]:
                for b in needed_by[key]:
                    if a < b:
                        overlap[f"{a} & {b}"] += 1
    assert sum(partition.values()) == len(done)
    # a path whose subsystem was removed, and the function still settled: another path covers it
    covered = Counter()
    for run in minus:
        for key in done:
            if run.subsystem in path_subsystems(base.rows[key]) and settled(run.rows.get(key), kinds):
                covered[run.subsystem] += 1
    pairs = [compare_pair(base, run, unstable, kinds) for run in minus + plus]
    marginal = {run.subsystem: sum(1 for key in done if run.subsystem in needed_by[key]
                                   and key not in unstable["settled"] and not outside[key]) for run in minus}
    attempted_count = sum(1 for row in base.rows.values() if attempted(row))
    return {
        "base": base.label, "repeat": repeat.label if repeat else None, "complete": all(r.complete for r in runs),
        "coverage": {"catalogue": len(base.rows), "attempted": attempted_count, "settled": len(done),
                     "settled_of_attempted": round(len(done) / attempted_count, 4) if attempted_count else None,
                     "assembly": sum(1 for row in base.rows.values() if settled(row, ASSEMBLY)),
                     "awaiting": sum(1 for row in base.rows.values() if settled(row, AWAITING))},
        "noise": {"outcome_differs": len(unstable["outcome"]), "settled_differs": len(unstable["settled"])},
        "first_stage": dict(first.most_common()), "first_path": dict(first_paths.most_common()),
        "capable_paths": {"functions_by_number_of_paths": dict(sorted(capable_count.items())),
                          "functions_per_path_not_additive": dict(capable_paths.most_common())},
        "partition": dict(sorted(partition.items())),
        "marginal_not_additive": marginal, "joint_overlap": dict(overlap.most_common()),
        "covered_by_another_path": dict(covered),
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------------------------
# several titles


def summary(titles: dict[str, dict]) -> dict:
    """Per subsystem over titles (titles are disjoint: their counts add), titles affected."""
    out: dict[str, dict] = {}
    numeric = ("added", "regressions", "net", "made_buildable", "made_unbuildable", "made_closer", "made_farther",
               "within_noise", "added_assembly", "regressions_assembly", "builds_added", "builds_saved",
               "builds_on_added")
    for title, result in titles.items():
        for pair in result["pairs"]:
            item = out.setdefault(pair["subsystem"], {"kind": pair["kind"], "titles": [], "titles_affected": 0,
                                                      "incomplete": [], "wall_s_added": 0.0,
                                                      **{key: 0 for key in numeric}})
            item["titles"].append(title)
            for key in numeric:
                item[key] += pair[key] or 0
            item["wall_s_added"] = round(item["wall_s_added"] + (pair["wall_s_added"] or 0), 1)
            item["titles_affected"] += int(bool(pair["added"] or pair["regressions"]))
            if not pair["complete"]:
                item["incomplete"].append(title)
    partition = Counter()
    for result in titles.values():
        partition.update(result["partition"])
    return {"subsystems": out, "partition": dict(sorted(partition.items())),
            "settled": sum(result["coverage"]["settled"] for result in titles.values())}


def markdown(titles: dict[str, dict], total: dict) -> str:
    lines = ["# Ablation and attribution", "",
             "Strict C functions (exact, donor).  Counts in the partition add up to the settled total; the "
             "marginal and per-path columns are counterfactuals and do not add.", "",
             "## Per subsystem (all titles)", "",
             "| subsystem | kind | added | regressions | net | buildable | closer | within noise | builds added | "
             "builds saved | wall s | titles affected |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, item in sorted(total["subsystems"].items()):
        flag = " (incomplete: " + ", ".join(item["incomplete"]) + ")" if item["incomplete"] else ""
        lines.append(f"| {name}{flag} | {item['kind']} | {item['added']} | {item['regressions']} | {item['net']} | "
                     f"{item['made_buildable']} | {item['made_closer']} | {item['within_noise']} | "
                     f"{item['builds_added']} | {item['builds_saved']} | {item['wall_s_added']} | "
                     f"{item['titles_affected']}/{len(item['titles'])} |")
    lines += ["", f"## Exclusive attribution of {total['settled']} settled functions", "",
              "| class | functions |", "|---|---:|"]
    lines += [f"| {key} | {value} |" for key, value in total["partition"].items()]
    for title, result in titles.items():
        cov = result["coverage"]
        lines += ["", f"## {title}", "",
                  f"Catalogue {cov['catalogue']}, attempted {cov['attempted']}, settled {cov['settled']} "
                  f"({cov['settled_of_attempted']} of attempted); assembly {cov['assembly']}, awaiting "
                  f"{cov['awaiting']}.  Noise (base vs repeat): {result['noise']['settled_differs']} settled "
                  f"differently, {result['noise']['outcome_differs']} with another outcome.", "",
                  "First settling stage: " + ", ".join(f"{k} {v}" for k, v in result["first_stage"].items()), "",
                  "Functions by number of capable paths: " + ", ".join(
                      f"{k}: {v}" for k, v in result["capable_paths"]["functions_by_number_of_paths"].items()), "",
                  "| comparison | added | regressions | buildable | closer | denominator (only with / only without "
                  "/ attempted only with / without) | builds added | regress.compare causes |",
                  "|---|---:|---:|---:|---:|---|---:|---|"]
        for pair in result["pairs"]:
            d = pair["denominator"]
            causes = ", ".join(f"{k} {v}" for k, v in (pair.get("regress") or {}).get("causes", {}).items())
            lines.append(f"| {pair['variant']} | {pair['added']} | {pair['regressions']} | {pair['made_buildable']} | "
                         f"{pair['made_closer']} | {d['only_with']} / {d['only_without']} / "
                         f"{d['attempted_only_with']} / {d['attempted_only_without']} | "
                         f"{pair['builds_added'] - pair['builds_saved']} | {causes} |")
    return "\n".join(lines) + "\n"
