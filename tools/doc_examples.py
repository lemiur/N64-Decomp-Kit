#!/usr/bin/env python3
"""Documentation as an executable contract: extract the marked shell examples of README.md and
docs/*.md, check every ``n64dk`` command in them against the real CLI parser, and run the ones whose
fixture needs nothing the machine may lack.

A marked example is an HTML comment directly above a fenced code block::

    <!-- example: fixture=overlay-project expect=3 -->
    ```sh
    n64dk image-address --project "$PROJECT" --vram 0x80100010
    ```

Marker keys (``key=value``, space separated):

* ``fixture``  (required) the starting state the example assumes; ``FIXTURES`` lists them.  The
  synthetic ones are built by ``tests/test_doc_examples.py``; the others name real resources.
* ``requires`` comma list of what a run needs beyond the fixture: ``rom``, ``compiler``,
  ``source``, ``run``, ``export``, ``network``, ``model``, ``slow``.  An example with any is
  parse-checked, never executed by the tests, and reported as skipped with that reason.
* ``expect``   exit status of the block's last command: an integer (default 0), ``nonzero`` or
  ``any``.  Every earlier ``n64dk`` command must exit 0.
* ``id``       a stable name (default: ``<file>:<line>``).

Inside the block: ``n64dk ...`` commands (``\\`` continuations joined, ``#`` comments dropped),
``VAR=value`` / ``export VAR=value`` assignments and ``mkdir -p DIR`` are understood; any other line
makes the example *foreign* (parse-checked, not executed in-process).  ``$VAR`` / ``${VAR}`` expand
from the fixture's variables (``WORK``, ``PROJECT``, ``ROM``, ``RUN``, ``EXPORT``, ``TOOLCHAIN``...).

    python tools/doc_examples.py list            # every marked example
    python tools/doc_examples.py check           # markers valid, every n64dk command parses (exit 1 if not)
    python tools/doc_examples.py audit           # also parse UNmarked n64dk lines of the docs (report only)

Executing examples needs fixtures, so it lives in the tests (``run_example`` here does the work).
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
MARKER = re.compile(r"^\s*<!--\s*example:\s*(.*?)\s*-->\s*$")
FENCE = re.compile(r"^(\s*)(```+|~~~+)\s*([A-Za-z0-9_+-]*)\s*$")
SHELL_LANGS = {"", "sh", "bash", "shell", "console", "zsh"}
REQUIREMENTS = {"rom", "compiler", "source", "run", "export", "network", "model", "slow", "project"}
FIXTURES = {
    # synthetic: built by the tests, nothing from the machine
    "none": "an empty work directory ($WORK); N64DK_STATE_DIR points inside it",
    "empty-toolchain": "$WORK plus an empty toolchain root ($TOOLCHAIN)",
    "synthetic-project": "$PROJECT: n64dk.toml with one linear image, its catalogue and image file",
    "overlay-project": "$PROJECT: two overlay images sharing one VRAM range, each with a catalogue",
    # real resources: named for the reader, never built by the tests
    "rom": "$ROM: a retail cartridge image",
    "rom-project": "$PROJECT made by decompile-init from $ROM",
    "run": "$RUN: a finished decompile-run output directory of $PROJECT",
    "export": "$EXPORT: a decompile-export tree of $RUN",
    "source-tree": "$SOURCE: a C source tree of the title (a public decompilation)",
    "project": "a hand-maintained matching project in the current directory",
}
VAR = re.compile(r"\$(\w+)|\$\{(\w+)\}")


class MarkerError(ValueError):
    pass


@dataclass
class Example:
    file: str
    line: int
    fixture: str
    requires: list[str] = field(default_factory=list)
    expect: str = "0"
    ident: str = ""
    text: str = ""
    steps: list[tuple[str, object]] = field(default_factory=list)   # ("n64dk", argv) | ("set", (k, v)) | ("mkdir", path)
    foreign: list[str] = field(default_factory=list)

    @property
    def commands(self) -> list[list[str]]:
        return [value for kind, value in self.steps if kind == "n64dk"]

    def skip_reason(self) -> str | None:
        if self.requires:
            return "needs " + ", ".join(self.requires)
        if self.foreign:
            return "runs commands other than n64dk: " + "; ".join(self.foreign)[:120]
        return None


def parse_marker(text: str, where: str) -> dict:
    values = {}
    for item in text.split():
        key, sep, value = item.partition("=")
        if not sep or not value:
            raise MarkerError(f"{where}: marker item {item!r} is not key=value")
        values[key] = value
    unknown = set(values) - {"fixture", "requires", "expect", "id"}
    if unknown:
        raise MarkerError(f"{where}: unknown marker keys {sorted(unknown)}")
    if "fixture" not in values:
        raise MarkerError(f"{where}: marker has no fixture=")
    if values["fixture"] not in FIXTURES:
        raise MarkerError(f"{where}: unknown fixture {values['fixture']!r} (known: {', '.join(FIXTURES)})")
    requires = [r for r in values.get("requires", "").split(",") if r]
    if set(requires) - REQUIREMENTS:
        raise MarkerError(f"{where}: unknown requirement(s) {sorted(set(requires) - REQUIREMENTS)}")
    expect = values.get("expect", "0")
    if expect not in ("any", "nonzero") and not re.fullmatch(r"\d+", expect):
        raise MarkerError(f"{where}: expect must be an integer, nonzero or any")
    return {"fixture": values["fixture"], "requires": requires, "expect": expect, "id": values.get("id", "")}


def logical_lines(block: str) -> list[str]:
    """Shell lines with ``\\`` continuations joined and comments / blank lines dropped."""
    out, pending = [], ""
    for raw in block.splitlines():
        line = raw.rstrip()
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        line = (pending + line).strip()
        pending = ""
        if not line or line.startswith("#"):
            continue
        out.append(line)
    if pending.strip():
        out.append(pending.strip())
    return out


def _steps(block: str) -> tuple[list, list[str]]:
    steps, foreign = [], []
    for line in logical_lines(block):
        try:
            words = shlex.split(line, comments=True)
        except ValueError as error:
            foreign.append(f"{line} ({error})")
            continue
        if not words:
            continue
        if words[0] == "export":
            words = words[1:]
        if words and all(re.fullmatch(r"[A-Za-z_]\w*=.*", w) for w in words):
            for w in words:
                key, _, value = w.partition("=")
                steps.append(("set", (key, value)))
            continue
        env_prefix = []
        while words and re.fullmatch(r"[A-Za-z_]\w*=.*", words[0]):
            env_prefix.append(words.pop(0))
        if words and words[0] == "n64dk":
            for w in env_prefix:
                key, _, value = w.partition("=")
                steps.append(("set", (key, value)))
            steps.append(("n64dk", words[1:]))
        elif words[:2] == ["mkdir", "-p"] and len(words) >= 3 and not env_prefix:
            steps.extend(("mkdir", w) for w in words[2:])
        else:
            foreign.append(line)
    return steps, foreign


def extract_text(text: str, file: str) -> list[Example]:
    lines = text.splitlines()
    out, pending, k = [], None, 0
    while k < len(lines):
        m = MARKER.match(lines[k])
        if m:
            pending = (parse_marker(m.group(1), f"{file}:{k + 1}"), k + 1)
            k += 1
            continue
        fence = FENCE.match(lines[k])
        if fence:
            close = fence.group(2)
            body, j = [], k + 1
            while j < len(lines) and not lines[j].strip().startswith(close):
                body.append(lines[j])
                j += 1
            if pending is not None:
                marker, mline = pending
                if fence.group(3).lower() not in SHELL_LANGS:
                    raise MarkerError(f"{file}:{mline}: marked block is {fence.group(3)!r}, not a shell block")
                block = "\n".join(body)
                steps, foreign = _steps(block)
                if not any(kind == "n64dk" for kind, _ in steps):
                    raise MarkerError(f"{file}:{mline}: marked block runs no n64dk command")
                out.append(Example(file, k + 1, marker["fixture"], marker["requires"], marker["expect"],
                                   marker["id"] or f"{file}:{k + 1}", block, steps, foreign))
            pending = None
            k = j + 1
            continue
        if pending is not None and lines[k].strip():
            raise MarkerError(f"{file}:{pending[1]}: marker is not directly above a fenced block")
        k += 1
    if pending is not None:
        raise MarkerError(f"{file}:{pending[1]}: marker at the end of the file")
    return out


def documents(root: Path = ROOT) -> list[Path]:
    return [root / "README.md", *sorted((root / "docs").glob("*.md"))]


def extract(paths=None, root: Path = ROOT) -> list[Example]:
    out = []
    for path in paths or documents(root):
        path = Path(path)
        name = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name
        out.extend(extract_text(path.read_text(errors="replace"), name))
    ids = [e.ident for e in out]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise MarkerError(f"duplicate example ids: {duplicates}")
    return out


def expand(word: str, variables: dict[str, str]) -> str:
    return VAR.sub(lambda m: variables.get(m.group(1) or m.group(2), m.group(0)), word)


def placeholder_variables() -> dict[str, str]:
    """Stand-in values for parse-only checks (argparse does not look at files)."""
    return {name: f"/placeholder/{name.lower()}" for name in
            ("WORK", "PROJECT", "ROM", "RUN", "EXPORT", "TOOLCHAIN", "SOURCE", "RECEIPT", "OUT", "HOME")}


def _cli_reference():
    spec = importlib.util.spec_from_file_location("n64dk_cli_reference", Path(__file__).with_name("cli_reference.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module                  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


_PARSER = None


def parser() -> argparse.ArgumentParser:
    global _PARSER
    if _PARSER is None:
        _PARSER = _cli_reference().build_parser()
    return _PARSER


def parse_errors(argv: list[str], variables: dict[str, str] | None = None) -> str | None:
    """None when ``n64dk ARGV`` parses with the real parser, else argparse's message."""
    values = dict(placeholder_variables(), **(variables or {}))
    words = [expand(w, values) for w in argv]
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            parser().parse_args(words)
    except SystemExit as stop:
        if stop.code not in (0, None):
            return (err.getvalue().strip().splitlines() or ["argparse error"])[-1]
    return None


def check(examples: list[Example]) -> list[dict]:
    problems = []
    for example in examples:
        variables = {}
        for kind, value in example.steps:
            if kind == "set":
                variables[value[0]] = value[1]
            elif kind == "n64dk":
                message = parse_errors(value, variables)
                if message:
                    problems.append({"example": example.ident, "argv": value, "error": message})
    return problems


def audit(paths=None, root: Path = ROOT) -> list[dict]:
    """Parse every ``n64dk`` line in every shell block of the docs, marked or not (report only:
    unmarked examples may use placeholders such as ``ROM`` or ``[--flag]`` for a reader)."""
    out = []
    for path in paths or documents(root):
        path = Path(path)
        lines = path.read_text(errors="replace").splitlines()
        k = 0
        while k < len(lines):
            fence = FENCE.match(lines[k])
            if not fence:
                k += 1
                continue
            body, j = [], k + 1
            while j < len(lines) and not lines[j].strip().startswith(fence.group(2)):
                body.append(lines[j])
                j += 1
            if fence.group(3).lower() in SHELL_LANGS:
                for line in logical_lines("\n".join(body)):
                    try:
                        words = shlex.split(line, comments=True)
                    except ValueError:
                        continue
                    while words and re.fullmatch(r"[A-Za-z_]\w*=.*", words[0]):
                        words.pop(0)
                    for start, word in enumerate(words):
                        if word in ("n64dk", ".venv/bin/n64dk") or word.endswith("/bin/n64dk"):
                            argv = words[start + 1:]
                            break
                        if word == "-m" and start + 1 < len(words) and words[start + 1] == "n64dk":
                            argv = words[start + 2:]
                            break
                    else:
                        continue
                    if "|" in argv:
                        argv = argv[:argv.index("|")]
                    message = parse_errors(argv)
                    if message:
                        out.append({"file": path.name, "line": k + 2, "argv": argv, "error": message})
            k = j + 1
    return out


def run_example(example: Example, variables: dict[str, str], cwd: Path) -> dict:
    """Run an example in this process through ``n64dk.cli.main`` with the fixture's variables as
    environment.  Returns ``{"ok", "results": [{"argv", "exit", "stdout", "stderr"}], "why"}``."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from n64dk import cli
    values = dict(variables)
    results = []
    env = {k: v for k, v in values.items()}
    env.setdefault("N64DK_STATE_DIR", str(Path(values.get("WORK", cwd)) / "state"))
    previous = Path.cwd()
    os.chdir(cwd)
    try:
        with mock.patch.dict(os.environ, env):
            commands = [i for i, (kind, _v) in enumerate(example.steps) if kind == "n64dk"]
            for index, (kind, value) in enumerate(example.steps):
                if kind == "set":
                    values[value[0]] = expand(value[1], values)
                    os.environ[value[0]] = values[value[0]]
                    continue
                if kind == "mkdir":
                    Path(expand(value, values)).mkdir(parents=True, exist_ok=True)
                    continue
                argv = [expand(w, values) for w in value]
                out, err = io.StringIO(), io.StringIO()
                try:
                    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                        code = cli.main(argv)
                except SystemExit as stop:
                    code = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
                    if not isinstance(stop.code, int) and stop.code is not None:
                        err.write(str(stop.code))
                results.append({"argv": argv, "exit": code, "stdout": out.getvalue()[-4000:],
                                "stderr": err.getvalue()[-2000:]})
                last = index == commands[-1]
                if not last and code != 0:
                    return {"ok": False, "results": results, "why": f"step exited {code} before the last command"}
    finally:
        os.chdir(previous)
    final = results[-1]["exit"]
    expect = example.expect
    ok = expect == "any" or (expect == "nonzero" and final != 0) or (expect.isdigit() and final == int(expect))
    return {"ok": ok, "results": results, "why": None if ok else f"last command exited {final}, expected {expect}"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("list", "check", "audit"))
    args = ap.parse_args(argv)
    examples = extract()
    if args.action == "list":
        for e in examples:
            print(json.dumps({"id": e.ident, "fixture": e.fixture, "requires": e.requires, "expect": e.expect,
                              "commands": e.commands, "skip": e.skip_reason()}))
        return 0
    if args.action == "check":
        problems = check(examples)
        for p in problems:
            print(json.dumps(p))
        print(f"{len(examples)} marked examples, {sum(len(e.commands) for e in examples)} n64dk commands, "
              f"{len(problems)} not parsing")
        return 1 if problems else 0
    problems = audit()
    for p in problems:
        print(json.dumps(p))
    print(f"{len(problems)} n64dk lines in the docs do not parse with the current CLI (report only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
