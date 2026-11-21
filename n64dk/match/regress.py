"""Benchmark comparison with causes, and local commit bisection (round 12, product item 10).

``compare`` reads two exports' function tables (``FUNCTIONS.tsv``; an export directory or the
file) and reports, without double counting: strict-exact gains and losses, every category
transition, denominator changes (functions only on one side, per image), recipe and settling-stage
changes, distance changes, and, for each changed function, the *first differing stage* in the
order a run settles a function: discovery (the catalogue row) -> recipe selection -> settling stage
-> source (when both exports carry sources) -> verification.  Resource and cache figures are
compared when both sides give a ``run.json``/post-job record.

``bisect`` finds the first commit at which a check changes, testing each commit in a disposable
``git worktree`` (never the caller's checkout) under the kit's machine-wide heavy-job lock; the
check is a callable (tests) or a shell command run in the worktree with a timeout.
"""
from __future__ import annotations

import csv
import hashlib
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

STRICT = {"exact", "donor"}
STAGES = ("discovery", "recipe", "stage", "source", "verification")
_STAGE_HINT = re.compile(r"\(([^()]*)\)\s*$")


def _table(path: Path) -> tuple[dict, Path | None]:
    path = Path(path)
    tsv = path / "FUNCTIONS.tsv" if path.is_dir() else path
    rows = {}
    with tsv.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows[(row["image"], row["vram"])] = row
    return rows, (path if path.is_dir() else None)


def _recipe(how: str) -> str:
    return _STAGE_HINT.sub("", how or "").strip()


def _stage(how: str) -> str:
    m = _STAGE_HINT.search(how or "")
    return m.group(1).split(";")[0].strip() if m else ""


def _source_digest(root: Path | None, row: dict) -> str | None:
    if root is None or not row.get("source"):
        return None
    path = root / row["source"]
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def first_difference(a: dict | None, b: dict | None, root_a: Path | None = None, root_b: Path | None = None) -> str:
    """The first stage, in settling order, at which two rows of one function differ."""
    if a is None or b is None or (a["size"], a.get("rom_offset")) != (b["size"], b.get("rom_offset")):
        return "discovery"
    if _recipe(a["how"]) != _recipe(b["how"]):
        return "recipe"
    if _stage(a["how"]) != _stage(b["how"]):
        return "stage"
    da, db = _source_digest(root_a, a), _source_digest(root_b, b)
    if da is not None and db is not None and da != db:
        return "source"
    return "verification"


def compare(before: Path, after: Path) -> dict:
    a, root_a = _table(before)
    b, root_b = _table(after)
    keys = set(a) | set(b)
    transitions = Counter()
    gained, lost, changed = [], [], []
    for key in sorted(keys):
        ra, rb = a.get(key), b.get(key)
        oa, ob = (ra or {}).get("outcome", "absent"), (rb or {}).get("outcome", "absent")
        if oa != ob:
            transitions[f"{oa} -> {ob}"] += 1
        sa, sb = oa in STRICT, ob in STRICT
        if oa != ob or (ra and rb and (ra["how"] != rb["how"] or ra.get("distance") != rb.get("distance"))):
            cause = first_difference(ra, rb, root_a, root_b)
            entry = {"image": key[0], "vram": key[1], "function": (rb or ra)["function"], "before": oa, "after": ob,
                     "first_difference": cause}
            if ra and rb and ra.get("distance") != rb.get("distance"):
                entry["distance"] = [ra.get("distance"), rb.get("distance")]
            if sb and not sa:
                gained.append(entry)
            elif sa and not sb:
                lost.append(entry)
            changed.append(entry)
    images = sorted({k[0] for k in keys})
    denominator = {image: {"before": sum(k[0] == image for k in a), "after": sum(k[0] == image for k in b),
                           "only_before": sum(k[0] == image and k not in b for k in a),
                           "only_after": sum(k[0] == image and k not in a for k in b)} for image in images}
    recipes_before = Counter(_recipe(r["how"]) for r in a.values() if r["outcome"] in STRICT)
    recipes_after = Counter(_recipe(r["how"]) for r in b.values() if r["outcome"] in STRICT)
    return {"strict_before": sum(r["outcome"] in STRICT for r in a.values()),
            "strict_after": sum(r["outcome"] in STRICT for r in b.values()),
            "gained": gained, "lost": lost, "transitions": dict(transitions.most_common()),
            "denominator": denominator,
            "causes": dict(Counter(e["first_difference"] for e in changed).most_common()),
            "strict_recipes": {"before": dict(recipes_before), "after": dict(recipes_after)},
            "changed": len(changed)}


def markdown(result: dict, title: str = "") -> str:
    lines = [f"# Benchmark comparison {title}".rstrip(), "",
             f"Strict exact: {result['strict_before']} -> {result['strict_after']} "
             f"(gained {len(result['gained'])}, lost {len(result['lost'])})", "",
             "| image | functions before | after | only before | only after |", "|---|---|---|---|---|"]
    for image, d in result["denominator"].items():
        lines.append(f"| {image} | {d['before']} | {d['after']} | {d['only_before']} | {d['only_after']} |")
    lines += ["", "First differing stage of changed functions: " +
              ", ".join(f"{k} {v}" for k, v in result["causes"].items()), "", "Transitions:"]
    lines += [f"- {k}: {v}" for k, v in result["transitions"].items()]
    if result["lost"]:
        lines += ["", "Lost strict functions:"] + [f"- {e['image']} {e['function']} ({e['before']} -> {e['after']}; "
                                                   f"first difference: {e['first_difference']})" for e in result["lost"]]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# bisection

def commits(repo: Path, good: str, bad: str) -> list[str]:
    """Commits after ``good`` up to and including ``bad``, oldest first."""
    from .. import safety
    out = safety.run_tool(["git", "-C", str(repo), "rev-list", "--reverse", "--ancestry-path", f"{good}..{bad}"],
                          check=True, text=True, timeout=120).stdout.split()
    return out


def search(candidates: list[str], is_bad) -> tuple[str | None, list[tuple[str, bool]]]:
    """Binary search for the first candidate where ``is_bad(commit)`` holds, assuming every
    commit before the first bad one is good.  Returns (first bad or None, the tests run)."""
    lo, hi, tried = 0, len(candidates) - 1, []
    first = None
    while lo <= hi:
        mid = (lo + hi) // 2
        bad = bool(is_bad(candidates[mid]))
        tried.append((candidates[mid], bad))
        if bad:
            first, hi = candidates[mid], mid - 1
        else:
            lo = mid + 1
    return first, tried


def worktree_check(repo: Path, command: str, timeout: float = 3600):
    """``is_bad`` for ``search``: a disposable worktree per commit, the command run there (exit
    status non-zero = bad), the worktree removed afterwards.  Takes the kit's machine-wide lock for
    the duration of each check, so it never runs beside another heavy job."""
    from .. import safety

    def is_bad(commit: str) -> bool:
        work = Path(tempfile.mkdtemp(prefix="n64dk-bisect-"))
        tree = work / "tree"
        safety.run_tool(["git", "-C", str(repo), "worktree", "add", "--detach", str(tree), commit],
                        check=True, timeout=600)
        try:
            with safety.machine_lock(wait=True):
                done = safety.run_tool(["sh", "-c", command], kind="compile", cwd=tree, timeout=timeout)
            return done.returncode != 0
        finally:
            safety.run_tool(["git", "-C", str(repo), "worktree", "remove", "--force", str(tree)], timeout=600)
            shutil.rmtree(work, ignore_errors=True)
    return is_bad
