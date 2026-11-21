"""Model providers behind one protocol (product item 19).

``n64dk.match.refine`` talks to one external command.  This module puts every source of proposals
behind the same small protocol, so the refinement stage, a benchmark or a comparison never depends
on which model (or none) answers:

* **in**: a *dossier* (``n64dk.refine-dossier/1``): the function's name, recipe, case id, round,
  the prompt ``refine.build_prompt`` wrote and limits.  Its canonical digest is the request digest;
* **out**: a ``Proposal``: the provider's identity (kind, declared model and version, executable
  name and digest, a digest of its non-secret configuration), the request digest, the digest of
  the raw response, the candidate C (``refine.extract_c``), seconds, an error;
* **adapters**: ``LocalCommandProvider`` (any program reading the prompt on stdin, through
  ``refine.run_command``), ``MockProvider`` (deterministic answers, for tests and dry runs) and
  ``NoModelProvider`` (always answers "no model configured"; what a run without a model gets);
* ``as_refine_runner(provider)`` returns a function with ``refine.run_command``'s signature, so
  ``refine.refine(..., run=as_refine_runner(provider))`` uses any provider unchanged;
* **records** (``n64dk.provider-record/1``, JSON lines, ``ProviderLog``) keep identities, digests,
  timings and, when the caller adds them, outcomes (score, exact) - never the prompt or the answer
  text by default (both hold ROM-derived disassembly), and never credentials;
  ``compare(records)`` sets providers side by side on the same requests.

**Credentials are never stored.**  A secret reaches a command only through the environment, by
name (``env_names``); the value is read when the command starts and recorded nowhere.  A command
line that carries something that looks like a secret (``--api-key VALUE``, ``sk-...``,
``Bearer ...``) is refused, and every record is checked (``assert_no_credentials``) before it is
written.  A proposal is a candidate: only the kit's build and strict verifier decide anything.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

DOSSIER_SCHEMA = "n64dk.refine-dossier/1"
RECORD_SCHEMA = "n64dk.provider-record/1"
SECRET_KEY = re.compile(r"(^|[_\-.])(API_?KEY|KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|AUTH|AUTHORIZATION|"
                        r"COOKIE|PRIVATE_?KEY)($|[_\-.])", re.I)
SECRET_VALUE = re.compile(r"(sk-[A-Za-z0-9_\-]{16,}|Bearer\s+\S{12,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|"
                          r"xox[abprs]-[A-Za-z0-9\-]{10,}|AIza[0-9A-Za-z_\-]{30,})")
SAFE_KEYS = {"env_names", "request_digest", "response_digest", "candidate_digest", "config_digest",
             "executable_digest", "identity_digest", "prompt_digest"}


class ProviderError(RuntimeError):
    pass


class CredentialLeak(ValueError):
    """Something that looks like a credential was about to be stored or put on a command line."""


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256(text: str | bytes) -> str:
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def assert_no_credentials(value, where: str = "record") -> None:
    """Raise ``CredentialLeak`` when ``value`` holds a key named like a secret with a non-empty
    value, or a string shaped like a known credential."""
    if isinstance(value, dict):
        for key, inner in value.items():
            if str(key) not in SAFE_KEYS and SECRET_KEY.search(str(key)) and inner not in (None, "", [], {}):
                raise CredentialLeak(f"{where}: field {key!r} looks like a credential")
            assert_no_credentials(inner, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, inner in enumerate(value):
            assert_no_credentials(inner, f"{where}[{index}]")
    elif isinstance(value, str) and SECRET_VALUE.search(value):
        raise CredentialLeak(f"{where}: a value is shaped like a credential")


# --------------------------------------------------------------------------- the protocol

def dossier(name: str, recipe: str, prompt: str, *, case: str | None = None, round_number: int = 1,
            limits: dict | None = None) -> dict:
    """The provider-neutral request for one round of one function."""
    return {"schema": DOSSIER_SCHEMA, "name": name, "recipe": recipe or "", "case": case, "round": int(round_number),
            "prompt": prompt, "prompt_digest": sha256(prompt), "limits": dict(limits or {})}


def validate_dossier(value) -> list[str]:
    if not isinstance(value, dict) or value.get("schema") != DOSSIER_SCHEMA:
        return [f"not a {DOSSIER_SCHEMA} dossier"]
    problems = [f"{key} is required" for key in ("name", "prompt") if not value.get(key)]
    if value.get("prompt") and value.get("prompt_digest") != sha256(value["prompt"]):
        problems.append("prompt_digest does not match the prompt")
    return problems


def request_digest(value: dict) -> str:
    return sha256(canonical(value))


@dataclass(frozen=True)
class Identity:
    provider: str                     # "local-command", "mock", "none", ...
    model: str = ""                   # as declared by whoever configured it
    version: str = ""
    executable: str = ""              # program name, never its arguments
    executable_digest: str = ""       # SHA-256 of the program file, when it can be read
    config_digest: str = ""           # SHA-256 of the non-secret configuration

    def as_dict(self) -> dict:
        out = asdict(self)
        out["identity_digest"] = sha256(canonical(asdict(self)))
        return out


@dataclass(frozen=True)
class Proposal:
    identity: dict
    request_digest: str
    text: str = ""
    response_digest: str | None = None
    candidate: str | None = None
    seconds: float = 0.0
    error: str | None = None

    def record(self, *, include_text: bool = False, name: str | None = None) -> dict:
        """The ``n64dk.provider-record/1`` row: identities and digests; the text only when asked."""
        row = {"schema": RECORD_SCHEMA, "kind": "proposal", "name": name, "identity": self.identity,
               "request_digest": self.request_digest, "response_digest": self.response_digest,
               "candidate_digest": sha256(self.candidate) if self.candidate else None,
               "response_bytes": len(self.text.encode()), "seconds": round(self.seconds, 3), "error": self.error}
        if include_text:
            row["text"] = self.text
        assert_no_credentials(row)
        return row


@runtime_checkable
class Provider(Protocol):
    identity: Identity

    def propose(self, request: dict) -> Proposal: ...


def _proposal(identity: Identity, request: dict, text: str = "", seconds: float = 0.0,
              error: str | None = None) -> Proposal:
    from .refine import extract_c
    problems = validate_dossier(request)
    if problems:
        return Proposal(identity.as_dict(), request_digest(request) if isinstance(request, dict) else "", error="; ".join(problems))
    return Proposal(identity.as_dict(), request_digest(request), text, sha256(text) if text else None,
                    extract_c(text) if text and error is None else None, seconds, error)


class NoModelProvider:
    """What a run without a model gets: every request answered with an error, nothing called."""

    def __init__(self):
        self.identity = Identity("none")

    def propose(self, request: dict) -> Proposal:
        return _proposal(self.identity, request, error="no model configured")


class MockProvider:
    """Deterministic answers: ``answers[name]`` when given, else a trivial unit for the function
    (``int <name>(void) { return 0; }``).  The same request always gets the same answer."""

    def __init__(self, answers: dict[str, str] | None = None, model: str = "mock", version: str = "1"):
        self.answers = dict(answers or {})
        self.identity = Identity("mock", model, version, config_digest=sha256(canonical(sorted(self.answers.items()))))

    def propose(self, request: dict) -> Proposal:
        name = str(request.get("name") or "f") if isinstance(request, dict) else "f"
        text = self.answers.get(name)
        if text is None:
            text = f"```c\n/* mock {request_digest(request)[:12] if isinstance(request, dict) else ''} */\n" \
                   f"int {name}(void) {{\n    return 0;\n}}\n```\n"
        return _proposal(self.identity, request, text)


def refuse_credentials(command: list[str]) -> None:
    """Refuse a provider command line that carries something shaped like a credential (secrets go
    through the environment, by name, never on the command line where logs and records see them)."""
    if not command or any(not isinstance(part, str) or "\x00" in part for part in command):
        raise ValueError("a provider command is a non-empty list of strings")
    for index, part in enumerate(command):
        flag = command[index - 1] if index else ""
        if SECRET_VALUE.search(part) or (flag.startswith("-") and SECRET_KEY.search(flag.lstrip("-"))):
            raise CredentialLeak(f"argument {index} of the command looks like a credential: pass it through "
                                 "the environment (env_names), never on the command line")
        if part.startswith("-") and "=" in part and SECRET_KEY.search(part.split("=", 1)[0].lstrip("-")):
            raise CredentialLeak(f"argument {index} of the command carries a credential")


class LocalCommandProvider:
    """A local program: the prompt on stdin, the answer on stdout (``refine.run_command``: its own
    process group, a timeout, an output limit, a fresh temporary working directory).  Secrets go
    through ``env_names`` (read from this process's environment at call time, recorded by name only)."""

    def __init__(self, command: list[str], *, model: str = "", version: str = "", timeout: float = 600.0,
                 env_names: tuple[str, ...] = (), output_limit: int = 512 * 1024, runner=None):
        refuse_credentials(command)
        self.command, self.timeout, self.output_limit = list(command), float(timeout), int(output_limit)
        self.env_names = tuple(sorted(env_names))
        self.runner = runner
        executable = shutil.which(command[0]) or command[0]
        digest = ""
        try:
            if Path(executable).is_file():
                h = hashlib.sha256()
                with Path(executable).open("rb") as stream:
                    for block in iter(lambda: stream.read(1 << 20), b""):
                        h.update(block)
                digest = h.hexdigest()
        except OSError:
            digest = ""
        config = {"argv_digest": sha256(canonical(self.command)), "timeout": self.timeout,
                  "env_names": list(self.env_names), "output_limit": self.output_limit}
        self.identity = Identity("local-command", model, version, Path(command[0]).name, digest,
                                 sha256(canonical(config)))

    def propose(self, request: dict) -> Proposal:
        problems = validate_dossier(request)
        if problems:
            return _proposal(self.identity, request)
        from . import refine
        run = self.runner or refine.run_command
        env = {name: os.environ[name] for name in self.env_names if name in os.environ}
        began = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix="n64dk-provider-") as cwd:
                text = run(self.command, request["prompt"], self.timeout, env, cwd=cwd,
                           output_limit=self.output_limit)
        except refine.RefineError as error:
            return _proposal(self.identity, request, seconds=time.monotonic() - began, error=str(error)[-200:])
        return _proposal(self.identity, request, text, time.monotonic() - began)


def as_refine_runner(provider, log: "ProviderLog | None" = None, *, name: str = "", recipe: str = ""):
    """A function with ``refine.run_command``'s signature that asks ``provider`` instead (the
    command argument is ignored: the provider owns its configuration).  An error becomes
    ``RefineError``, as a failing command would; every proposal is logged when ``log`` is given."""
    from .refine import RefineError

    def run(_command, prompt, timeout, env=None, cwd=None, output_limit=512 * 1024):
        request = dossier(name or "function", recipe, prompt, limits={"timeout": timeout, "output_limit": output_limit})
        proposal = provider.propose(request)
        if log is not None:
            log.append(proposal, name=name or None)
        if proposal.error:
            raise RefineError(proposal.error)
        return proposal.text

    return run


# --------------------------------------------------------------------------- records and comparison

class ProviderLog:
    """``n64dk.provider-record/1`` rows, one JSON object per line, append-only."""

    def __init__(self, path: str | Path, include_text: bool = False):
        self.path, self.include_text = Path(path), include_text

    def _write(self, row: dict) -> None:
        assert_no_credentials(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(canonical(row) + "\n")

    def append(self, proposal: Proposal, name: str | None = None) -> dict:
        row = proposal.record(include_text=self.include_text, name=name)
        self._write(row)
        return row

    def outcome(self, request: str, identity: dict, *, score: int | None, exact: bool) -> dict:
        """What the kit's build and strict verifier made of a proposal (added by the caller)."""
        row = {"schema": RECORD_SCHEMA, "kind": "outcome", "request_digest": request,
               "identity_digest": identity.get("identity_digest"), "score": score, "exact": bool(exact)}
        self._write(row)
        return row

    def read(self) -> list[dict]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("schema") == RECORD_SCHEMA:
                out.append(row)
        return out


def compare(records: list[dict]) -> dict:
    """Providers side by side: per provider (identity digest) calls, errors, candidates, mean
    seconds, exact and mean score of recorded outcomes; and, on requests more than one provider
    answered, how often their answers were identical and which scored better."""
    providers: dict[str, dict] = {}
    by_request: dict[str, dict[str, dict]] = {}
    outcomes: dict[tuple[str, str], dict] = {}
    for row in records:
        if row.get("kind") == "outcome":
            outcomes[(row.get("request_digest"), row.get("identity_digest"))] = row
    for row in records:
        if row.get("kind") != "proposal":
            continue
        identity = row.get("identity") or {}
        key = identity.get("identity_digest", "?")
        p = providers.setdefault(key, {"provider": identity.get("provider"), "model": identity.get("model"),
                                       "version": identity.get("version"), "calls": 0, "errors": 0, "candidates": 0,
                                       "seconds": 0.0, "outcomes": 0, "exact": 0, "score_total": 0, "scored": 0})
        p["calls"] += 1
        p["errors"] += 1 if row.get("error") else 0
        p["candidates"] += 1 if row.get("candidate_digest") else 0
        p["seconds"] += float(row.get("seconds") or 0)
        found = outcomes.get((row.get("request_digest"), key))
        if found:
            p["outcomes"] += 1
            p["exact"] += 1 if found.get("exact") else 0
            if found.get("score") is not None:
                p["scored"] += 1
                p["score_total"] += int(found["score"])
        by_request.setdefault(row.get("request_digest"), {})[key] = {**row, "outcome": found}
    for p in providers.values():
        p["mean_seconds"] = round(p.pop("seconds") / p["calls"], 3) if p["calls"] else None
        total, scored = p.pop("score_total"), p.pop("scored")
        p["mean_score"] = round(total / scored, 2) if scored else None
    pairs: dict[str, dict] = {}
    for answers in by_request.values():
        keys = sorted(answers)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                pair = pairs.setdefault(f"{a[:12]}|{b[:12]}", {"shared": 0, "identical": 0, "first_better": 0,
                                                              "second_better": 0, "same_score": 0})
                pair["shared"] += 1
                ra, rb = answers[a], answers[b]
                pair["identical"] += 1 if ra.get("response_digest") and ra.get("response_digest") == rb.get(
                    "response_digest") else 0
                sa = (ra.get("outcome") or {}).get("score")
                sb = (rb.get("outcome") or {}).get("score")
                if sa is not None and sb is not None:
                    pair["first_better" if sa < sb else "second_better" if sb < sa else "same_score"] += 1
    return {"providers": providers, "pairs": pairs, "requests": len(by_request)}
