"""``review-queue RUN``, ``review-show RUN CASE``, ``review-decide RUN CASE ACTION``: the human
review and acceptance workflow over a finished ``decompile-run`` (``n64dk.match.review``).

The run is read-only apart from ``RUN/review/`` (the append-only transaction log and exports).
Project source is never written.  ``review-show --diff`` and ``review-decide accept-exact --verify``
rebuild the stored candidate with the project the run used (``solver/invocation.json``): the
first for the target-versus-candidate assembly diff, the second for the strict verifier's
verdict; without ``--verify`` an accept is recorded as ``requested`` only."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from ..cli import command
from ..match import review as reviewmod


def _builder(run: Path, toolchain_dir=None):
    """``build(case, text) -> (outcome, target code)`` with the run's project and default recipe."""
    from ..match import images as imagemod
    from ..match import symbols as symmod
    from ..match.engine import Target, match_source
    from ..match.pipeline import split_flags
    from ..match.toolchain import load_config
    invocation = json.loads((run / "solver" / "invocation.json").read_text())
    project = Path(invocation["project"])
    cfg = load_config(project, None, toolchain_dir or os.environ.get("N64DK_TOOLCHAIN"))
    files = [Path(p) for p in (cfg.symbols or symmod.discover(project))]
    survey = project / "survey"
    if survey.is_dir():                       # the declarations the run itself added (as the solver does)
        extra = sorted(survey.glob("symbol_addrs*.txt")) + [survey / "sdk_symbols.txt"]
        files += [p for p in extra if p.is_file() and p not in files]
    syms = symmod.load(files)
    default = f"{invocation.get('pipeline') or 'kmc+builtin'} {invocation.get('flags') or ''}".strip()
    loaded = {}

    def build(case, text):
        if case.image not in loaded:
            spec = imagemod.resolve(project, case.image)
            data = spec.load()
            loaded[case.image] = (data, spec.delta)
        data, delta = loaded[case.image]
        code = data[case.vram - delta:case.vram - delta + case.size]
        pipeline, _, flags = (case.recipe or default).partition(" ")
        with tempfile.TemporaryDirectory(prefix="n64dk-review-") as work:
            path = Path(work) / f"{case.name}.c"
            path.write_text(text)
            outcome = match_source(path, split_flags(flags.split(), pipeline), cfg,
                                   Target(data, delta, (0, len(data))), case.vram, case.size, syms, Path(work), None,
                                   strict_verification=True)
        return outcome, code
    return build


def differ_for(run: Path, toolchain_dir=None):
    from ..match import diff as diffmod
    build = _builder(run, toolchain_dir)

    def differ(case, text):
        outcome, code = build(case, text)
        if outcome.error or outcome.result is None:
            return [f"does not build: {(outcome.error or 'no comparison')[-300:]}"]
        report = diffmod.diff(outcome.result.compiled_text or b"", code, case.vram)
        return [f"score {report.score}; counts {dict(report.counts)}"] + report.lines(80)
    return differ


def verifier_for(run: Path, toolchain_dir=None):
    build = _builder(run, toolchain_dir)

    def verify(case, text):
        outcome, _code = build(case, text)
        if outcome.error or outcome.result is None:
            return {"verdict": "does-not-build", "error": (outcome.error or "no comparison")[-300:]}
        verdict = "exact" if outcome.exact else ("byte-identical-unproven" if outcome.exploratory_exact else "different")
        proof = outcome.proof
        return {"verdict": verdict, "digest": outcome.digest,
                "assumptions": len(getattr(proof, "assumptions", []) or []) if proof is not None else None}
    return verify


def maintainable_for(run: Path, names_path=None, use_egraph: bool = False):
    from ..match import twotrack
    names = {}
    if names_path and Path(names_path).is_file():
        names = twotrack.read_names(Path(names_path).read_text())

    def make(text, case):
        return twotrack.build(text, case.name, names=names,
                              provenance={"case": case.case_id, "status": case.status, "recipe": case.recipe},
                              rewriter=twotrack.egraph_rewriter if use_egraph else None)
    return make


# ------------------------------------------------------------------------------ commands
@command("review-queue", "Review queue of a decompile-run: assumed, near and donor items not yet decided")
def review_queue(args):
    run = Path(args.run)
    items = reviewmod.queue(run, include_decided=args.all)
    state = reviewmod.replay(run)
    result = {"run": run.name, "items": [i.as_dict() for i in items[:args.limit or None]], "total": len(items),
              "by_kind": {k: sum(1 for i in items if i.kind == k) for k in ("assumed", "near", "donor")},
              "transactions": state["transactions"], "log_problems": state["problems"]}
    if getattr(args, "format", "text") == "text":
        print(f"{result['total']} item(s): {result['by_kind']}; {result['transactions']} transaction(s)"
              + (f"; LOG PROBLEMS: {result['log_problems']}" if result["log_problems"] else ""))
        for i in result["items"]:
            print(f"  {i['id']:<28} {i['kind']:<8} {i['name']:<24} size {i['size']:<6} score {i['score']} "
                  f"{i['cause']}" + (f"  [{i['decision']}]" if i["decision"] else ""))
        return 0
    return result


def _queue_args(p):
    p.add_argument("run", help="the --out directory of a decompile-run")
    p.add_argument("--all", action="store_true", help="include decided items")
    p.add_argument("--limit", type=int, default=0)


review_queue.arguments = _queue_args


@command("review-show", "One review item in full: source, diff, residual, receipt, assumptions, provenance, view")
def review_show(args):
    run = Path(args.run)
    differ = differ_for(run, args.toolchain_dir) if args.diff else None
    view = maintainable_for(run, args.names, args.egraph) if args.view else None
    try:
        result = reviewmod.show(run, args.case, differ=differ, maintainable=view)
    except reviewmod.ReviewError as error:
        raise SystemExit(str(error)) from None
    if getattr(args, "format", "text") == "text":
        item = result["item"]
        print(f"{item['id']} {item['name']} ({item['kind']}) size {item['size']} score {item['score']}")
        for key in ("status", "recipe", "proof_verdict", "decision", "residual", "assumptions", "provenance",
                    "transform_classes", "receipt", "behaviour", "history", "note"):
            if result.get(key) not in (None, [], {}):
                print(f"{key}: {json.dumps(result[key], default=str)}")
        for line in result.get("diff") or []:
            print(line)
        if result.get("source"):
            print("---- candidate source ----")
            print(result["source"])
        if result.get("maintainable", {}).get("view"):
            print(f"---- maintainable view ({result['maintainable']['class']}) ----")
            print(result["maintainable"]["view"])
        return 0
    return result


def _show_args(p):
    p.add_argument("run")
    p.add_argument("case", help="case id (image:VRAM:SIZE), donor item id or function name")
    p.add_argument("--diff", action="store_true", help="rebuild the candidate and show the assembly diff")
    p.add_argument("--view", action="store_true", help="add the maintainable view (two-track source)")
    p.add_argument("--names", help="NAMES.tsv for the maintainable view's names")
    p.add_argument("--egraph", action="store_true", help="normalise statement forms in the view (e-graph)")
    p.add_argument("--toolchain-dir")


review_show.arguments = _show_args


@command("review-decide", "Append a review transaction (accept-exact needs --verify to be more than a request)")
def review_decide(args):
    run = Path(args.run)
    payload = {}
    for item in args.set or []:
        key, _, value = item.partition("=")
        payload[key.strip()] = value.strip()
    verify = verifier_for(run, args.toolchain_dir) if args.verify and args.action == "accept-exact" else None
    try:
        record = reviewmod.decide(run, args.case, args.action, payload=payload, reviewer=args.reviewer,
                                  note=args.note, verify=verify)
    except reviewmod.ReviewError as error:
        raise SystemExit(str(error)) from None
    if getattr(args, "format", "text") == "text":
        print(f"#{record['seq']} {record['action']} {record['case']}: {record['status']}"
              + (f" ({record['reason']})" if record.get("reason") else "") + f"  id {record['id'][:16]}")
        return 0
    return record


def _decide_args(p):
    p.add_argument("run")
    p.add_argument("case")
    p.add_argument("action", choices=sorted(reviewmod.ACTIONS))
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="action payload, e.g. reason=..., stage=..., symbol=... type=..., vram=... size=...")
    p.add_argument("--reviewer")
    p.add_argument("--note")
    p.add_argument("--verify", action="store_true", help="accept-exact: rebuild and ask the strict verifier")
    p.add_argument("--toolchain-dir")


review_decide.arguments = _decide_args


def _audit_args(p):
    p.add_argument("paths", nargs="+", help="C/C++ sources or folders of them")
    p.add_argument("--show", type=int, default=20, help="files listed in text output")


@command("source-audit", "List the devices in sources that match bytes without being plausible original source "
         "(volatile, pinned registers, empty arms, byte-offset casts, addresses held in integers, gotos, "
         "decompiler placeholders); a label for cleanup, never a verdict on exactness")
def source_audit(args):
    from collections import Counter
    from pathlib import Path
    from ..match import walls
    files = []
    for given in args.paths:
        path = Path(given)
        files += sorted(p for p in path.rglob("*") if p.suffix in (".c", ".cpp", ".cc")) if path.is_dir() else [path]
    rows, devices = [], Counter()
    for path in files:
        found = walls.scaffolding(path.read_text(errors="replace"))
        for item in found:
            devices[item["device"]] += 1
        if found:
            rows.append({"file": str(path), "devices": found})
    if args.format != "json":
        print(f"source-audit: {len(files)} file(s), {len(rows)} with scaffolding")
        for name, count in devices.most_common():
            print(f"  {name}: {count} file(s)")
        for row in rows[:args.show]:
            print(f"  {row['file']}: " + ", ".join(f"{d['device']} x{d['count']} (line {d['first_line']})"
                                                   for d in row["devices"]))
    return {"files": len(files), "with_scaffolding": len(rows), "by_device": dict(devices), "rows": rows}


source_audit.arguments = _audit_args
