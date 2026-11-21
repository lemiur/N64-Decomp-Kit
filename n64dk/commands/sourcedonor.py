"""``source-donor``: any C source tree as a donor, with its build found by measurement."""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from ..cli import command

RECIPE_MATCHES = 3         # a recipe of the run is a candidate when this many of its matches used it
MAX_RECIPES = 6           # from the run; one more, at the neighbouring instruction set, may be added
LEVELS = ("-O0", "-O1", "-O2", "-O3", "-g")
ISA_NEIGHBOUR = {"-mips2": "-mips1", "-mips1": "-mips2", "-mips3": "-mips2"}


def recipes_of_run(run: Path, limit: int = MAX_RECIPES) -> list[str]:
    """The recipes a finished run matched functions under, most used first: what the title's own
    code says about how it was built."""
    store = Path(run) / "state" / "cases.sqlite"
    if not store.is_file():
        return []
    tally: dict[str, int] = {}
    with closing(sqlite3.connect(f"file:{store}?mode=ro", uri=True)) as db:
        for (payload,) in db.execute("select payload from cases where status in ('exact', 'assumed')"):
            recipe = json.loads(payload).get("recipe")
            if recipe:
                tally[recipe] = tally.get(recipe, 0) + 1
    ranked = sorted(tally.items(), key=lambda item: (-item[1], item[0]))
    return with_levels([recipe for recipe, count in ranked if count >= RECIPE_MATCHES], limit)


def with_levels(recipes: list[str], limit: int = MAX_RECIPES) -> list[str]:
    """``recipes``, then the first one at the other usual levels: a title's library and its game
    code are often the same compiler at two levels, and a run that matched few library functions
    by itself says nothing about the second."""
    out = list(recipes[:limit])
    if out:
        parts = out[0].split()
        level = next((index for index, part in enumerate(parts) if part in LEVELS), None)
        if level is not None:
            plain = "-g" if parts[0].startswith(("ido", "custom:ido", "custom:mipspro")) else "-O0"
            for other in ("-O2", plain, "-O1"):
                variant = " ".join([*parts[:level], other, *parts[level + 1:]])
                if variant not in out and len(out) < limit:
                    out.append(variant)
        # ... and at the neighbouring instruction-set level: one part of a title is often built
        # for an older one (a library, a separately written module)
        isa = next((index for index, part in enumerate(parts) if part in ISA_NEIGHBOUR), None)
        if isa is not None:
            variant = " ".join([*parts[:isa], ISA_NEIGHBOUR[parts[isa]], *parts[isa + 1:]])
            if variant not in out:
                out.append(variant)
    return out


MIN_SHARED = 8             # target functions a tree must give before it is taken as a donor of the title


def import_tree(source: Path, project: Path, out: Path, recipes: list[str], toolchain: str | None = None,
                name: str | None = None, jobs: int = 4, defines: list[str] | None = None,
                max_macros: int = 48, keep_work: bool = False, emit=None, by_function: bool = True,
                prepare_command: str | None = None, whole_units_too: bool = False,
                min_shared: int | None = None) -> dict:
    """Describe ``source`` as a donor of the title surveyed in ``project``.  Writes
    ``out/donors.toml`` when the tree gives the title anything and returns the measurements.

    ``prepare_command`` is run in a copy of the tree first (for headers its own build generates);
    the tree itself is only read.  Objects are kept in ``out/<name>/objects`` between runs, under
    keys that cover everything they depend on."""
    import json
    import shutil
    import time
    from ..match import sourcedonor
    from ..match.toolchain import load_config
    from ..resources import plan_jobs
    started = time.time()
    jobs = plan_jobs(jobs).jobs
    source, out, project = Path(source).resolve(), Path(out).resolve(), Path(project).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(out, None, toolchain or os.environ.get("N64DK_TOOLCHAIN"))
    name = name or "tree_" + "".join(ch if ch.isalnum() else "_" for ch in source.name)
    origin = sourcedonor.provenance(source)
    prepared = None
    read_from = source
    if prepare_command:
        read_from = out / name / "prepared_tree"
        prepared = sourcedonor.prepared_copy(source, read_from, prepare_command)
    tree = sourcedonor.Tree(read_from)
    objects = out / name / "objects"
    cached_before = len(list(objects.glob("*.o"))) if objects.is_dir() else 0
    report = {"source": str(source), "name": name, "candidate_recipes": recipes, "units_in_tree": len(tree.units),
              "provenance": origin, "prepared": prepared}
    if not tree.units:
        return {**report, "donors": None, "reason": "no C file with a function body"}
    signatures = sourcedonor.target_signatures(project)
    if not signatures:
        return {**report, "donors": None, "reason": "the project has no surveyed functions to compare with"}
    work = out / name / "work"
    plan = sourcedonor.calibrate(tree, signatures, recipes, cfg, work, jobs=jobs, macros=defines or [],
                                 max_macros=max_macros, emit=emit, objects=objects)
    report.update(target_functions_compared=sum(len(places) for places in signatures.values()),
                  found_by_recipe=plan["found_by_recipe"], macros=plan["macros"],
                  include_directories=[str(path) for path in plan["includes"]],
                  macro_trials=[row for row in plan["macro_trials"] if row["with"] != row["without"]],
                  macros_not_measured=plan["macros_not_measured"][:40])
    donors = out / "donors.toml"
    donors.unlink(missing_ok=True)
    # A small tree (a compiler's runtime routines, three functions) is a donor too when the caller says how
    # few are enough; the default guards the calibration of a large tree against a recipe found by chance.
    needed = MIN_SHARED if min_shared is None else max(1, min_shared)
    if max(plan["found_by_recipe"].values(), default=0) < needed:
        shutil.rmtree(out / name, ignore_errors=True)
        return {**report, "donors": None,
                "reason": f"fewer than {needed} functions of the title are found in this tree under any candidate recipe"}
    result = sourcedonor.build(tree, plan, out, name, cfg, signatures, by_function=by_function)
    if whole_units_too and by_function:
        # the tree's own units as a second donor: a function that matches only when its unit is
        # compiled together (shared literals, static data, inlined helpers, the unit's layout)
        # is taken through it, with the whole unit standing in the title and strictly verified
        units = sourcedonor.build(tree, plan, out, f"{name}_units", cfg, signatures, by_function=False)
        if units.get("toml"):
            result["toml"] = (result.get("toml") or "") + "\n" + units["toml"]
        result["whole_units"] = {key: units.get(key) for key in ("units_found_in_target", "target_functions_found",
                                                                 "compiled", "failed_units")}
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
        if by_function and prepare_command:                 # the split sources need nothing from the copy
            shutil.rmtree(read_from, ignore_errors=True)
    builder = plan["builder"]
    from .. import safety
    pruned = safety.prune(objects)
    result.update(cache={"pruned": pruned, "objects_before": cached_before,
                         "objects_after": len(list(objects.glob("*.o"))) if objects.is_dir() else 0,
                         "compiles_this_run": builder.builds}, seconds=round(time.time() - started, 1))
    (out / name).mkdir(parents=True, exist_ok=True)
    (out / name / "provenance.json").write_text(json.dumps(
        {"tree": origin, "prepared": prepared, "macros": plan["macros"], "recipes": recipes,
         "kit_note": "functions from this tree count only after strict verification against the title"},
        indent=1, default=str) + "\n")
    if result.get("toml"):
        licence = "; ".join(f"{item['file']}: {item['first_line']}" for item in origin.get("licence_files", [])) \
            or origin.get("licence_note", "")
        donors.write_text("# Written by `n64dk source-donor`.  Every transplant is rebuilt and verified by the run "
                          "that receives it.\n"
                          f"# tree: {origin.get('remote') or origin['path']} at {origin.get('commit') or 'unknown commit'}"
                          f" ({origin.get('modified_files', '?')} files modified)\n# licence: {licence}\n\n"
                          + result["toml"])
    result.pop("toml", None)
    return {**report, **result, "donors": str(donors) if donors.is_file() else None}


@command("source-donor", "Describe any C source tree as a donor for decompile-run --donors: its units, include "
                         "directories, each unit's recipe and its configuration macros are found by building and "
                         "comparing with the target, not read from its build files")
def source_donor(args):
    """One heavy job at a time, with a preflight and a lease (see ``n64dk.safety``)."""
    from .. import safety
    try:
        with safety.heavy_job("source-donor", Path(args.out), wait_for_lock=args.wait_for_lock,
                              force=args.force_resources, argv=sys.argv[1:]):
            return _source_donor(args)
    except safety.Busy as error:
        raise SystemExit(str(error)) from None


def _source_donor(args):
    source = Path(args.source).resolve()
    if not source.is_dir():
        raise SystemExit(f"{source} is not a directory")
    recipes = list(dict.fromkeys([*(args.recipe or []), *(recipes_of_run(Path(args.run)) if args.run else [])]))
    if not recipes:
        raise SystemExit("no candidate recipe: name a finished run of the target (--run) or give --recipe")

    def emit(kind: str, **fields) -> None:
        print(json.dumps({"event": kind, **fields}, default=str), file=sys.stderr, flush=True)

    return import_tree(source, Path(args.project), Path(args.out), recipes, toolchain=args.toolchain_dir,
                       name=args.name, jobs=args.jobs, defines=args.define, max_macros=args.max_macros,
                       keep_work=args.keep_work, emit=emit, by_function=args.units != "whole" and not args.whole_units,
                       prepare_command=args.prepare_command, whole_units_too=args.units == "both",
                       min_shared=getattr(args, "min_shared", None))


def _args(p):
    p.add_argument("--source", required=True, help="the source tree (any layout)")
    p.add_argument("--project", required=True, help="surveyed project of the target title (its functions are what "
                                                    "the tree is compared with)")
    p.add_argument("--out", required=True, help="directory for the image, manifests and donors.toml")
    p.add_argument("--run", help="a finished decompile-run of the target: the recipes it matched under are the candidates")
    p.add_argument("--recipe", action="append", metavar="'PIPELINE FLAGS'",
                   help="a candidate recipe (repeatable), e.g. 'ido5.3 -O2 -mips2'")
    p.add_argument("--define", action="append", metavar="NAME",
                   help="a configuration macro to set without measuring it (repeatable)")
    p.add_argument("--max-macros", type=int, default=48, help="configuration macros measured, most used first")
    p.add_argument("--name", help="donor name (default: from the directory)")
    p.add_argument("--min-shared", type=int, default=None, metavar="N",
                   help=f"functions of the title the tree must give to be taken as a donor (default {MIN_SHARED}); "
                        "lower it for a small tree whose recipe you state with --recipe")
    p.add_argument("--units", choices=["split", "whole", "both"], default="split",
                   help="donor units: one per function (split, the default), the tree's own units (whole), or both "
                        "as two donors (a function is then taken from whichever verifies)")
    p.add_argument("--whole-units", action="store_true",
                   help="keep the tree's own units as the donor's units (a unit is then taken only when all of it "
                        "stands in the title, and its data comes with it); the default is one unit per function")
    p.add_argument("--prepare-command", metavar="SHELL",
                   help="run this in a copy of the tree before reading it (for headers the tree's own build "
                        "generates, e.g. 'make NOEXTRACT=1 build/us/include/text_strings.h'); the tree is never written. "
                        "Without it, missing generated files get stand-ins and functions that need them are refused "
                        "by verification")
    p.add_argument("--keep-work", action="store_true", help="keep the calibration's probe images")
    p.add_argument("--toolchain-dir", help="toolchain root (default: N64DK_TOOLCHAIN)")
    p.add_argument("--jobs", type=int, default=2, help="compilers run at once (capped by the machine's memory)")
    from .run import _guard_arguments
    _guard_arguments(p)


source_donor.arguments = _args


@command("source-donor-account", "Give every function of a title one cause after a source tree was offered to it "
                                 "(verified, refused and why, differs from the tree's function at that place, not C "
                                 "in the tree)")
def source_donor_account(args):
    from ..match import treeaccount
    folder = Path(args.donor_dir).resolve()
    names = (args.donor or folder.name).split(",")
    found = treeaccount.account(Path(args.project).resolve(), Path(args.run).resolve(), folder,
                                names[0] if len(names) == 1 else set(names))
    if args.out:
        Path(args.out).write_text(json.dumps(found, indent=1) + "\n")
    found.pop("rows")
    return found


def _account_args(p):
    p.add_argument("--project", required=True, help="the title's project")
    p.add_argument("--run", required=True, help="the decompile-run output that used the tree's donors")
    p.add_argument("--donor-dir", required=True, help="the tree donor's directory (from source-donor)")
    p.add_argument("--donor", help="donor name, or names separated by commas for a tree imported as several donors "
                                   "(default: the directory name)")
    p.add_argument("--out", help="write every function's cause here (JSON)")


source_donor_account.arguments = _account_args
