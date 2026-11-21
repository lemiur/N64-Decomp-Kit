"""The model stage: an agent loop over unresolved functions, with the tool protocol and the playbook.

For each function the stage builds a *work order* (the function, its recipe and lanes, a draft and
its distance, a diagnosis category from the kit's own near-miss analysis, the playbook sections
for that category and compiler family, the project's lessons, matched neighbours, the boundary
check, and the dossier of earlier attempts), opens a work folder holding the order, a session file
and a ``kit`` wrapper for the tool protocol (``n64dk agent-tools``), and hands the order to a
*provider*: any command-line coding agent given by configuration, or a recorded transcript.

Acceptance is the kit's: after the session the stage runs the strict verifier itself on
``final.c``; nothing the agent says is believed.  Results, the transcript and the tool-call log
are kept per attempt; the agent's one-line lesson is appended to the project's lessons file; a
failure leaves a dossier entry (what was tried, how close it came, what was ruled out) that the
next attempt's work order shows.  Budgets: attempts per function, wall time per attempt and in
total, tool calls per attempt.  Resumable: a function with an exact result is never redone and
attempts already made count toward its budget.  Model sessions hold the machine's ``model``
token (``safety.token``), so one runs at a time.

Configuration (``n64dk.toml``)::

    [agent.provider]
    command = ["my-agent", "--print", "--cwd", "{workdir}"]   # prompt on stdin; {prompt_file} also available
    env_names = ["MY_AGENT_KEY"]                              # passed by name; values are never recorded
    timeout = 1800

or ``N64DK_AGENT_COMMAND`` in the environment (a shell-quoted command line).  No vendor is built in.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import agenttools as tools

PLAYBOOK = Path(__file__).resolve().parent / "data" / "playbook"
CATEGORIES = ("boundary", "assisted", "unwritten", "reg-swap", "schedule", "merge-tail", "hoist", "frame", "type",
              "other")
ADDRESS = re.compile(r"\b(?:0x)?[0-9A-Fa-f]{8}\b")
STAGE_SCHEMA = "n64dk.agent-stage/1"


# ---------------------------------------------------------------------- playbook

def playbook_text(name: str) -> str:
    return (PLAYBOOK / name).read_text()


def sections(text: str, keys) -> str:
    """The ``## key`` sections of a playbook file, in the order asked."""
    found, current, out = {}, None, []
    for line in text.splitlines():
        m = re.match(r"^## (\S+)", line)
        if m:
            current = m.group(1)
            found[current] = [line]
        elif current:
            found[current].append(line)
    for key in keys:
        if key in found:
            out.append("\n".join(found[key]).strip())
    return "\n\n".join(out)


def compiler_family(pipeline: str) -> str:
    return "ido" if "ido" in pipeline or "mipspro" in pipeline else "gcc"


def lessons_path(ws: tools.Workspace) -> Path:
    configured = ws.agent.get("lessons")
    return (ws.root / configured) if configured else ws.state / "LESSONS.md"


def append_lesson(ws: tools.Workspace, fn: tools.Function, category: str, lesson: str) -> str | None:
    """Append one general lesson (addresses stripped) to the project's lessons file; None when
    the lesson is empty or already there."""
    line = " ".join(lesson.split())[:600]
    line = re.sub(r"^(?:[-*]\s*)?(?:\(\s*[\w-]+\s*\)\s*)*", "", line)      # the agent's own bullet/category prefix
    line = ADDRESS.sub("ADDR", line)
    if not line:
        return None
    path = lessons_path(ws)
    path.parent.mkdir(parents=True, exist_ok=True)
    have = path.read_text() if path.is_file() else "# Project lessons (appended by the model stage)\n"
    if line in have:
        return None
    entry = f"- ({category}) {line}"
    path.write_text(have.rstrip("\n") + "\n" + entry + "\n")
    return entry


# ---------------------------------------------------------------------- diagnosis

EXTENSIONS = {"sll", "sra", "srl", "andi", "lb", "lbu", "lh", "lhu", "seb", "seh"}


def classify(target: bytes, candidate: bytes | None, *, boundary_agrees: bool = True, fixmul: bool = False) -> tuple[str, list[str]]:
    """The diagnosis category of a near miss (the campaign's categories), with evidence."""
    from .agenttools import _ops
    if not boundary_agrees:
        return "boundary", ["the function's own instructions give another start or size than the catalogue"]
    if fixmul:
        return "assisted", ["fixed-point multiply sequence in fixed registers (inline-assembly macro class)"]
    if not candidate:
        return "unwritten", ["no draft builds"]
    a, b = _ops(target, 0), _ops(candidate.rstrip(b"\0") if len(candidate) > len(target) else candidate, 0)
    mn = lambda ops: [x.split(" ")[0] for x in ops]  # noqa: E731
    if a[:1] != b[:1] and a[:1] and b[:1] and a[0].startswith("addiu $sp") and b[0].startswith("addiu $sp"):
        return "frame", [f"prologue {a[0]} vs {b[0]}"]
    if len(a) != len(b):
        return "merge-tail", [f"{len(a)} target instructions vs {len(b)}"]
    if mn(a) == mn(b):
        return "reg-swap", ["same instructions, registers differ"]
    if sorted(mn(a)) == sorted(mn(b)):
        return "schedule", ["same instructions in another order"]
    diff = [(x, y) for x, y in zip(mn(a), mn(b)) if x != y]
    if diff and all(x in EXTENSIONS or y in EXTENSIONS for x, y in diff):
        return "type", [f"extension/load width differs: {diff[0][0]} vs {diff[0][1]}"]
    loads = lambda ops: sum(1 for x in ops if x.startswith(("lw", "lh", "lb", "lui")))  # noqa: E731
    if loads(a) != loads(b):
        return "hoist", [f"{loads(a)} target loads vs {loads(b)}"]
    return "other", []


# ---------------------------------------------------------------------- work orders

@dataclass
class Budget:
    attempts: int = 2               # agent sessions per function (across runs)
    seconds: float = 1800.0         # wall time per session
    tool_calls: int = 80            # tool calls per session
    total_seconds: float = 0.0      # wall time for the whole stage (0: no limit)


@dataclass
class WorkOrder:
    function: dict
    lanes: list
    category: str
    evidence: list
    causes: list
    draft: dict
    boundary: list
    neighbours: list
    playbook: str
    lessons: str
    dossier: list
    budget: dict
    files: list = field(default_factory=list)

    def render(self) -> str:
        f = self.function
        parts = [self.playbook, "", "---", "",
                 f"# Work order: {f['image']} {f['name']} at real 0x{f['vram']:08X}, size 0x{f['size']:X}",
                 "", "Recipe lanes (`./kit try` builds every one):"]
        parts += [f"- {lane}" for lane in self.lanes]
        parts += ["", f"Diagnosis category: **{self.category}**" +
                  (f" ({'; '.join(self.evidence)})" if self.evidence else "")]
        if self.causes:
            parts += ["Kit causes: " + "; ".join(self.causes[:5])]
        parts += ["", "## Boundary check", *self.boundary]
        d = self.draft
        parts += ["", "## Draft"]
        if d.get("files"):
            parts += [f"Files in this folder: {', '.join(Path(p).name for p in d['files'])}. "
                      f"Best: {d.get('best_file') or '-'} ({d.get('best_status')}, {d.get('best_hunks')} hunks). "
                      "A hint only: write natural C from the disassembly."]
        else:
            parts += ["No draft could be made: write the function from `./kit dis`."]
        parts += ["", "## Neighbours", *self.neighbours[:60]]
        if self.dossier:
            parts += ["", "## Earlier attempts on this function (do not repeat what they ruled out)"]
            for a in self.dossier[-4:]:
                parts.append(f"- attempt {a.get('attempt')}: {a.get('status')}, best {a.get('hunks')} hunks, "
                             f"{a.get('calls')} tool calls; lesson: {a.get('lesson') or '-'}")
                for line in (a.get("notes") or "").splitlines()[:8]:
                    parts.append(f"    {line}")
                for line in a.get("tried", [])[:10]:
                    parts.append(f"    tried: {line}")
        if self.lessons.strip():
            parts += ["", "## This project's lessons so far", self.lessons.strip()]
        b = self.budget
        parts += ["", f"Budget for this session: {b['tool_calls']} tool calls, {int(b['seconds'])} s. "
                  "Leave final.c and LESSON.md before it runs out."]
        return "\n".join(parts) + "\n"


def _draft_and_diagnose(ws: tools.Workspace, fn: tools.Function, folder: Path) -> tuple[dict, bytes | None]:
    """Draft into the folder and build every draft under the first lane: the best one's
    relocated code (for the diagnosis) and a summary."""
    try:
        drafted = tools.tool_draft(ws, fn, [str(folder / "draft.c")], {})
    except Exception as error:  # noqa: BLE001 - a draft is optional
        return dict(files=[], error=str(error)[-300:]), None
    files = drafted.get("files") or []
    best, code = dict(files=files), None
    if not files:
        return best, None
    spec, flags = ws.lanes(fn)[0]
    import tempfile
    for path in files:
        with tempfile.TemporaryDirectory(prefix="n64dk-order-") as tmp:
            try:
                elf = ws.build(Path(path), spec, Path(tmp))
                got = ws.explore(fn, elf, spec)
            except Exception:  # noqa: BLE001
                continue
        want = ws.target(fn)
        hunks = sum(1 for x, y in zip(tools._ops(want, fn.vram), tools._ops(got[:len(want)], fn.vram)) if x != y) + \
            abs(len(want) - len(got.rstrip(b"\0"))) // 4
        if best.get("best_hunks") is None or hunks < best["best_hunks"]:
            best.update(best_file=Path(path).name, best_status="builds", best_hunks=hunks)
            code = got
    if best.get("best_file") is None:
        best.update(best_status="does not build", best_hunks=None)
    return best, code


def build_work_order(ws: tools.Workspace, fn: tools.Function, folder: Path, dossier: list | None = None,
                     budget: Budget | None = None) -> WorkOrder:
    from . import causes as causemod
    from .handforms import fixmul_sites
    budget = budget or Budget()
    folder.mkdir(parents=True, exist_ok=True)
    lanes = ws.lanes(fn)
    lane_labels = [f"{s.pipeline} {f}" for s, f in lanes]
    try:
        boundary = tools.tool_boundary(ws, fn, [], {})
        agrees, boundary_lines = boundary["agrees"] or fn.true_range, boundary["lines"][1:]
    except tools.ToolError as error:
        agrees, boundary_lines = True, [f"(no boundary evidence: {error})"]
    draft, code = _draft_and_diagnose(ws, fn, folder)
    target = ws.target(fn)
    category, evidence = classify(target, code, boundary_agrees=agrees, fixmul=bool(fixmul_sites(target)))
    try:
        cause_ids = [f"{c.id} ({c.confidence:.2f})" for c in
                     causemod.diagnose(target, code, fn.vram, pipeline=lanes[0][0].pipeline)][:6]
    except Exception:  # noqa: BLE001 - diagnosis is advice
        cause_ids = []
    try:
        neighbours = tools.tool_neighbours(ws, fn, [], {})["lines"][1:]
    except Exception as error:  # noqa: BLE001
        neighbours = [f"(neighbours unavailable: {error})"]
    family = compiler_family(lanes[0][0].pipeline)
    playbook = "\n\n".join([playbook_text("BRIEF.md"),
                            "# Compiler notes\n\n" + sections(playbook_text("COMPILERS.md"), [family]),
                            "# Technique for this category\n\n" + sections(playbook_text("TECHNIQUES.md"),
                                                                         [category] + (["unwritten"] if category not in ("unwritten",) else [])),
                            playbook_text("LESSONS.md")])
    lp = lessons_path(ws)
    lessons = "\n".join(lp.read_text().splitlines()[-60:]) if lp.is_file() else ""
    return WorkOrder(dict(image=fn.image, name=fn.name, vram=fn.vram, size=fn.size, true_range=fn.true_range),
                     lane_labels, category, evidence, cause_ids, draft, boundary_lines, neighbours, playbook, lessons,
                     list(dossier or []), asdict(budget), draft.get("files", []))


# ---------------------------------------------------------------------- providers

class NoAgentProvider:
    identity = {"provider": "none"}

    def run(self, prompt: str, workdir: Path, timeout: float) -> dict:
        return dict(transcript="", seconds=0.0, error="no agent configured ([agent.provider] command or "
                                                        "N64DK_AGENT_COMMAND)")


class CommandAgentProvider:
    """A command-line coding agent: the work order on stdin, the work folder as its working
    directory; ``{workdir}`` and ``{prompt_file}`` in the command are substituted.  Secrets reach
    it only through the environment, by name; the command line is checked for credentials."""

    def __init__(self, command: list[str], *, timeout: float = 1800.0, env_names=(), output_limit: int = 4 << 20,
                 runner=None):
        from .providers import refuse_credentials, sha256, canonical
        refuse_credentials(command)
        self.command, self.timeout, self.env_names = list(command), float(timeout), tuple(sorted(env_names))
        self.output_limit, self.runner = int(output_limit), runner
        self.identity = {"provider": "command", "executable": Path(command[0]).name,
                         "argv_digest": sha256(canonical(self.command)), "env_names": list(self.env_names)}

    def run(self, prompt: str, workdir: Path, timeout: float) -> dict:
        from . import refine
        argv = [part.replace("{workdir}", str(workdir)).replace("{prompt_file}", str(workdir / "ORDER.md"))
                for part in self.command]
        env = {name: os.environ[name] for name in self.env_names if name in os.environ}
        began = time.monotonic()
        try:
            text = (self.runner or refine.run_command)(argv, prompt, min(timeout, self.timeout), env, cwd=str(workdir),
                                                       output_limit=self.output_limit)
            return dict(transcript=text, seconds=time.monotonic() - began, error=None)
        except refine.RefineError as error:
            return dict(transcript="", seconds=time.monotonic() - began, error=str(error)[-300:])


class DryRunProvider:
    """Replays a recorded transcript through the real tool protocol (for tests and demonstrations).

    Transcript: ``{"<image>_<VRAM>": [step, ...]}`` (or ``{"*": [...]}``); a step is
    ``{"write": "a.c", "content": "..."}`` or ``{"tool": "try", "files": ["a.c"], "opts": {...}}``.
    Every tool step goes through the same session (logged, counted, budgeted)."""

    identity = {"provider": "dry-run"}

    def __init__(self, transcript: dict | str | Path):
        if not isinstance(transcript, dict):
            transcript = json.loads(Path(transcript).read_text())
        self.transcript = transcript

    def run(self, prompt: str, workdir: Path, timeout: float) -> dict:
        session = json.loads((workdir / tools.SESSION).read_text())
        steps = self.transcript.get(f"{session['image']}_{session['vram']:08X}", self.transcript.get("*", []))
        began, out = time.monotonic(), []
        for step in steps:
            if time.monotonic() - began > timeout:
                out.append("TIMEOUT")
                break
            if "write" in step:
                (workdir / step["write"]).write_text(step["content"])
                out.append(f"$ write {step['write']}")
            elif "tool" in step:
                files = [str(workdir / f) for f in step.get("files", [])]
                result = tools.session_call(workdir / tools.SESSION, step["tool"], files, **step.get("opts", {}))
                out.append(f"$ ./kit {step['tool']} {' '.join(step.get('files', []))}")
                out += result.get("lines", [])
        return dict(transcript="\n".join(out) + "\n", seconds=time.monotonic() - began, error=None)


def provider_from_config(ws: tools.Workspace, kind: str | None = None, transcript: str | None = None):
    if kind == "dry-run":
        if not transcript:
            raise ValueError("the dry-run provider needs a transcript")
        return DryRunProvider(transcript)
    table = dict(ws.agent.get("provider") or {})
    raw = os.environ.get("N64DK_AGENT_COMMAND") or table.get("command")
    if kind == "none" or not raw:
        return NoAgentProvider()
    argv = shlex.split(raw) if isinstance(raw, str) else [str(p) for p in raw]
    return CommandAgentProvider(argv, timeout=float(table.get("timeout", 1800)), env_names=table.get("env_names", ()))


# ---------------------------------------------------------------------- the loop

def _wrapper(folder: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    script = (f"#!/bin/sh\n# the kit's tool protocol for this work order (see ORDER.md)\n"
              f"PYTHONPATH={shlex.quote(str(root))}${{PYTHONPATH:+:$PYTHONPATH}} exec {shlex.quote(sys.executable)} "
              f"-m n64dk agent-tools --session {shlex.quote(str(folder / tools.SESSION))} \"$@\"\n")
    path = folder / "kit"
    path.write_text(script)
    path.chmod(0o755)


def judge(ws: tools.Workspace, fn: tools.Function, folder: Path, flags: str | None = None,
          pipe: str | None = None) -> dict:
    """The stage's own verdict on ``final.c`` (strict verifier; plain C only)."""
    final = folder / "final.c"
    if not final.is_file():
        return dict(status="no-file", accepted=False, hunks=None)
    opts = {k: v for k, v in (("flags", flags), ("pipe", pipe)) if v}
    exact = tools.tool_exact(ws, fn, [str(final)], dict(opts))
    r = (exact.get("results") or [dict(verdict="build-error", accepted=False)])[0]
    if r.get("accepted"):
        return dict(status="exact", accepted=True, hunks=0, recipe=r.get("recipe"), lines=exact["lines"])
    if r.get("assisted"):
        return dict(status="assisted", accepted=False, hunks=0, recipe=r.get("recipe"), lines=exact["lines"])
    tried = tools.tool_try(ws, fn, [str(final)], dict(opts))
    t = (tried.get("results") or [{}])[0]
    return dict(status="near" if t.get("hunks") is not None else "does-not-build", accepted=False,
                hunks=t.get("hunks"), verdict=r.get("verdict"), lines=exact["lines"][:12])


def _session_overrides(folder: Path) -> tuple[str | None, str | None]:
    """The flags/lane of the agent's last exact call on final.c (an extra flag it found)."""
    log = folder / "calls.jsonl"
    flags = pipe = None
    if log.is_file():
        for line in log.read_text().splitlines():
            entry = json.loads(line)
            if entry["tool"] == "exact" and any(f.endswith("final.c") for f in entry["files"]):
                flags, pipe = entry["opts"].get("flags"), entry["opts"].get("pipe")
    return flags, pipe


def _tried(folder: Path) -> tuple[int, list[str]]:
    log = folder / "calls.jsonl"
    if not log.is_file():
        return 0, []
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    lines = []
    for row in rows:
        if row["tool"] in ("try", "exact"):
            for s in row.get("summary", [])[:2]:
                lines.append(" ".join(s.split())[:160])
    return len(rows), lines[-12:]


def run_stage(ws: tools.Workspace, functions: list[tools.Function], provider, budget: Budget | None = None,
              emit=None, clock=time.monotonic) -> dict:
    """Work every function until the strict verifier accepts it or its budget is spent."""
    from .. import safety
    budget = budget or Budget()
    root = ws.state / "stage"
    root.mkdir(parents=True, exist_ok=True)
    started, summary = clock(), {"schema": STAGE_SCHEMA, "functions": 0, "exact": 0, "assisted": 0, "near": 0,
                                 "open": 0, "skipped": 0, "sessions": 0, "rows": []}
    for fn in functions:
        summary["functions"] += 1
        home = root / fn.key
        home.mkdir(parents=True, exist_ok=True)
        dossier_path = home / "dossier.json"
        dossier = json.loads(dossier_path.read_text()) if dossier_path.is_file() else \
            {"function": dict(image=fn.image, name=fn.name, vram=fn.vram, size=fn.size), "status": "open", "attempts": []}
        if dossier["status"] in ("exact", "assisted"):
            summary["skipped"] += 1
            summary["rows"].append(dict(function=fn.key, status=dossier["status"], resumed=True))
            continue
        while len(dossier["attempts"]) < budget.attempts:
            remaining = budget.total_seconds - (clock() - started) if budget.total_seconds else budget.seconds
            if remaining <= 0:
                break
            number = len(dossier["attempts"]) + 1
            folder = home / f"attempt-{number}"
            if folder.exists():                              # an interrupted attempt: start it again cleanly
                folder = home / f"attempt-{number}-{int(time.time())}"
            order = build_work_order(ws, fn, folder, dossier["attempts"], budget)
            session = dict(project=str(ws.root), state=str(ws.state), image=fn.image, key=f"{fn.vram:08X}",
                           vram=fn.vram, size=fn.size, true_range=fn.true_range, tool_calls=budget.tool_calls)
            (folder / tools.SESSION).write_text(json.dumps(session, indent=1))
            prompt = order.render()
            (folder / "ORDER.md").write_text(prompt)
            (folder / "order.json").write_text(json.dumps({k: v for k, v in asdict(order).items() if k != "playbook"},
                                                          indent=1, default=str))
            _wrapper(folder)
            if emit:
                emit("agent-session", function=fn.key, attempt=number, category=order.category)
            with safety.token("model"):
                outcome = provider.run(prompt, folder, min(budget.seconds, remaining))
            if isinstance(provider, NoAgentProvider):        # nothing ran: the order is kept, no budget spent
                summary["no_agent"] = outcome["error"]
                break
            summary["sessions"] += 1
            (folder / "transcript.txt").write_text(outcome.get("transcript") or "")
            flags, pipe = _session_overrides(folder)
            verdict = judge(ws, fn, folder, flags, pipe)
            calls, tried = _tried(folder)
            lesson_file = folder / "LESSON.md"
            lesson = lesson_file.read_text().strip().splitlines()[0] if lesson_file.is_file() and \
                lesson_file.read_text().strip() else ""
            notes = (folder / "NOTES.md").read_text()[:2000] if (folder / "NOTES.md").is_file() else ""
            entry = dict(attempt=number, folder=str(folder), provider=provider.identity, category=order.category,
                         status=verdict["status"], hunks=verdict.get("hunks"), recipe=verdict.get("recipe"),
                         calls=calls, tried=tried, lesson=lesson, notes=notes, seconds=round(outcome["seconds"], 1),
                         error=outcome.get("error"))
            dossier["attempts"].append(entry)
            if lesson:
                entry["lesson_appended"] = append_lesson(ws, fn, order.category, lesson)
            if verdict["status"] in ("exact", "assisted"):
                dossier["status"] = verdict["status"]
                dest = root / ("done" if verdict["status"] == "exact" else "assisted")
                dest.mkdir(exist_ok=True)
                shutil.copyfile(folder / "final.c", dest / f"{fn.key}.c")
                dossier["recipe"] = verdict.get("recipe")
            elif (folder / "final.c").is_file():
                (root / "near").mkdir(exist_ok=True)
                shutil.copyfile(folder / "final.c", root / "near" / f"{fn.key}.c")
                dossier["status"] = "near" if verdict["status"] == "near" else "open"
            dossier_path.write_text(json.dumps(dossier, indent=1, default=str))
            with (root / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(function=fn.key, **{k: entry[k] for k in
                                             ("attempt", "status", "hunks", "recipe", "calls", "category", "seconds",
                                              "error")}), default=str) + "\n")
            if emit:
                emit("agent-result", function=fn.key, attempt=number, status=verdict["status"])
            if verdict["status"] in ("exact", "assisted"):
                break
        status = dossier["status"]
        summary[status if status in ("exact", "assisted", "near") else "open"] += 1
        summary["rows"].append(dict(function=fn.key, status=status, attempts=len(dossier["attempts"])))
    (root / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary
