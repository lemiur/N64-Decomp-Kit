"""The stable Python API of the kit (product item 14).

Everything else in ``n64dk`` is internal and may change in any release.  This module is the
contract: its functions, the dataclasses they return, the exceptions they raise and the version
``API_VERSION`` (semantic: MAJOR when something here is removed or changes meaning, MINOR when
something is added, PATCH for fixes).  Each function is the same operation as one CLI command
(``ENTRY_POINTS``): same arguments, same result (``to_dict()`` equals the command's ``--format
json`` output; ``tests/test_contracts.py`` checks it against the operation schema and the CLI).

* ``status(run)``            ``decompile-status``      -> ``StatusResult``
* ``receipt(run)``           ``decompile-receipt``     -> ``ReceiptResult``
* ``operation_schema()``     ``operation-schema``      -> ``OperationSchema``
* ``migrate(path, check)``   ``migrate``               -> ``MigrationReport``
* ``call(command, argv)``    any command, in-process, JSON mode -> its result (untyped)

**Results** keep the fields this version knows as typed attributes and every other key in
``extra``: a newer kit adding a key does not break an older client, and ``to_dict()`` returns all
of it.

**Exceptions** (all ``N64DKError``): ``NotARunDirectory``, ``FormatError`` (data this version
cannot read, e.g. written by a newer kit), ``UnknownCommand``, ``CommandFailed`` (non-zero exit,
with ``exit_code``).

**Deprecation policy.**  A public name is never removed in the release that deprecates it: it is
marked ``@deprecated(since, removal, replacement)``, warns ``DeprecationWarning`` on every call,
is listed in ``DEPRECATIONS``, and survives at least one MINOR release; it is removed only in the
next MAJOR ``API_VERSION``.  A result field is deprecated the same way (it keeps its value until the
MAJOR release).

**Outputs and versions.**  ``VERSIONED_OUTPUTS`` names, per command, the key of its JSON output
that carries a format version; ``UNVERSIONED_OUTPUTS`` lists key commands whose output carries
none yet, and why.
"""
from __future__ import annotations

import argparse
import functools
import warnings
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

API_VERSION = "1.0.0"


# --------------------------------------------------------------------------- exceptions

class N64DKError(Exception):
    """Base of every error the API raises on purpose."""
    exit_code = 1


class NotARunDirectory(N64DKError):
    exit_code = 2


class FormatError(N64DKError):
    """Data in a format this version cannot read (written by a newer kit, or damaged)."""
    exit_code = 3


class UnknownCommand(N64DKError):
    exit_code = 2


class CommandFailed(N64DKError):
    def __init__(self, command: str, exit_code: int, message: str = ""):
        self.command, self.exit_code = command, exit_code
        super().__init__(f"{command} exited {exit_code}" + (f": {message}" if message else ""))


# --------------------------------------------------------------------------- deprecation

DEPRECATIONS: dict[str, dict] = {}


def deprecated(since: str, removal: str, replacement: str | None = None):
    """Mark a public function deprecated (see the policy in the module docstring)."""
    def wrap(fn):
        name = f"{fn.__module__}.{fn.__qualname__}"
        DEPRECATIONS[name] = {"since": since, "removal": removal, "replacement": replacement}
        message = (f"{name} is deprecated since API {since} and is removed in {removal}"
                   + (f"; use {replacement}" if replacement else ""))

        @functools.wraps(fn)
        def inner(*args, **kwargs):
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return fn(*args, **kwargs)
        inner.__deprecated__ = DEPRECATIONS[name]
        return inner
    return wrap


# --------------------------------------------------------------------------- results

def _split(cls, data: dict) -> tuple[dict, dict]:
    if not isinstance(data, dict):
        raise FormatError(f"{cls.__name__}: expected an object")
    names = {f.name for f in fields(cls)} - {"extra"}
    return {k: v for k, v in data.items() if k in names}, {k: v for k, v in data.items() if k not in names}


def _build(cls, values: dict):
    """``cls(**values)``; a missing or misshapen field is a ``FormatError``, not a ``TypeError``."""
    if not isinstance(values, dict):
        raise FormatError(f"{cls.__name__}: expected an object")
    try:
        return cls(**values)
    except TypeError as error:
        raise FormatError(f"{cls.__name__}: {error}") from None


@dataclass(frozen=True)
class Credit:
    exact_c: int = 0
    awaiting_declarations: int = 0
    handwritten_assembly: int = 0
    unresolved: int = 0
    exact_c_bytes: int = 0


@dataclass(frozen=True)
class StatusResult:
    SCHEMA = "n64dk.api.status/1"
    run: str
    session: str | None
    mode: str
    rounds: list
    stops: list
    credit: Credit
    remaining_by_cause: dict
    report: dict | None = None
    discovery: dict | None = None
    small_data: dict | None = None
    image_totals: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "StatusResult":
        known, extra = _split(cls, data)
        credit = known.get("credit") or {}
        credit_known = {k: v for k, v in credit.items() if k in {f.name for f in fields(Credit)}}
        extra_credit = {k: v for k, v in credit.items() if k not in credit_known}
        if extra_credit:
            extra = {**extra, "credit_extra": extra_credit}
        return _build(cls, {**known, "credit": _build(Credit, credit_known), "extra": extra})

    def to_dict(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "extra"}
        out["credit"] = asdict(self.credit)
        extra = dict(self.extra)
        out["credit"].update(extra.pop("credit_extra", {}))
        return {**out, **extra}


@dataclass(frozen=True)
class ContractClause:
    clause: str
    text: str
    status: str
    evidence: list
    detail: str


@dataclass(frozen=True)
class ReceiptResult:
    SCHEMA = "n64dk.api.receipt/1"
    receipt: str
    run: str
    status: str | None
    contract: tuple
    contract_summary: dict
    run_digest: str
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "ReceiptResult":
        known, extra = _split(cls, data)
        if known.get("receipt") != _receipt_version():
            raise FormatError(f"receipt version {known.get('receipt')!r}; this API reads {_receipt_version()!r}")
        clauses = tuple(_build(ContractClause, row) for row in known.get("contract") or [])
        return _build(cls, {**known, "contract": clauses, "extra": extra})

    def clause(self, name: str) -> ContractClause:
        return next(c for c in self.contract if c.clause == name)

    def to_dict(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "extra"}
        out["contract"] = [asdict(c) for c in self.contract]
        return {**out, **self.extra}


@dataclass(frozen=True)
class ArgumentSpec:
    dest: str
    flags: list
    required: bool
    default: object
    choices: list | None
    multiple: bool


@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str
    arguments: tuple

    def argument(self, dest: str) -> ArgumentSpec:
        return next(a for a in self.arguments if a.dest == dest)

    @property
    def dests(self) -> set:
        return {a.dest for a in self.arguments}


@dataclass(frozen=True)
class OperationSchema:
    protocol: str
    commands: tuple
    global_arguments: list
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "OperationSchema":
        known, extra = _split(cls, data)
        if known.get("protocol") != "n64dk.operations.v1":
            raise FormatError(f"operation protocol {known.get('protocol')!r} is not n64dk.operations.v1")
        commands = tuple(CommandSpec(c["name"], c["description"], tuple(ArgumentSpec(**a) for a in c["arguments"]))
                         for c in known.get("commands") or [])
        return cls(known["protocol"], commands, known.get("global_arguments") or [], extra)

    def command(self, name: str) -> CommandSpec:
        for spec in self.commands:
            if spec.name == name:
                return spec
        raise UnknownCommand(name)

    def to_dict(self) -> dict:
        return {"protocol": self.protocol, "global_arguments": self.global_arguments,
                "commands": [{"name": c.name, "description": c.description,
                              "arguments": [asdict(a) for a in c.arguments]} for c in self.commands], **self.extra}


@dataclass(frozen=True)
class MigrationItem:
    format: str
    file: str
    status: str
    found: object
    current: object
    action: str
    explanation: str


@dataclass(frozen=True)
class MigrationReport:
    schema: str
    kit_version: str
    target: str
    kind: str
    check: bool
    items: tuple
    counts: dict
    ok: bool
    pending: int
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "MigrationReport":
        known, extra = _split(cls, data)
        return _build(cls, {**known, "items": tuple(_build(MigrationItem, i) for i in known.get("items") or []),
                            "extra": extra})

    def to_dict(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "extra"}
        out["items"] = [asdict(i) for i in self.items]
        return {**out, **self.extra}


def _receipt_version() -> str:
    from .commands.receipt import RECEIPT
    return RECEIPT


# --------------------------------------------------------------------------- the operations

# API function -> (CLI command, {API parameter: argparse dest})
ENTRY_POINTS: dict[str, tuple[str, dict[str, str]]] = {
    "status": ("decompile-status", {"run": "run_dir", "image": "image", "show": "show"}),
    "receipt": ("decompile-receipt", {"run": "run"}),
    "operation_schema": ("operation-schema", {}),
    "migrate": ("migrate", {"path": "path", "check": "check"}),
}

# command -> the key of its --format json output that carries a format version
VERSIONED_OUTPUTS: dict[str, str] = {
    "operation-schema": "protocol", "decompile-receipt": "receipt", "migrate": "schema", "release-info": "schema",
    "corpus-check": "schema", "blind-check": "schema", "doctor": "schema", "candidate-reconcile": "schema_version",
}

# key commands whose JSON output has no version key yet, and why it was not added here
UNVERSIONED_OUTPUTS: dict[str, str] = {
    "decompile-status": "tests/test_runstatus.py pins the exact key set; add with API 1.1 (the API's StatusResult is "
                        "versioned meanwhile)",
    "decompile-run": "its summary is printed by benchmark scripts that parse it; report.json carries none either",
    "decompile-export": "the export writes files whose formats would need versions first (FUNCTIONS.tsv)",
    "decompile-batch": "per-title rows of decompile-run summaries",
    "decompile-everything": "as decompile-batch",
    "decompile-init": "project files are the contract (n64dk.toml)",
    "discovery-audit": "written to discovery_audit.json inside runs; versioning it needs a migrate entry",
    "resolve-hard": "internal stage result, stored in report.json",
    "match": "per-unit results consumed by match-batch",
    "verify-run": "receipts it reads carry n64dk-proof-receipt/1; its own summary carries none",
}


def _not_a_run(run: Path, needs: str) -> NotARunDirectory:
    return NotARunDirectory(f"{run}: not a decompile-run directory ({needs})")


def status(run: str | Path, image: str | None = None, show: int = 5) -> StatusResult:
    """``decompile-status``: rounds, stops, credit and the remaining queue of a run."""
    run = Path(run)
    if not (run / "state" / "cases.sqlite").is_file():
        raise _not_a_run(run, "no state/cases.sqlite")
    from .commands import runstatus
    return StatusResult.from_dict(runstatus.status(run, image, show))


def receipt(run: str | Path) -> ReceiptResult:
    """``decompile-receipt``: the exhaustive-treatment receipt (read-only)."""
    run = Path(run)
    if not run.is_dir():
        raise _not_a_run(run, "no such directory")
    from .commands import receipt as receiptmod
    return ReceiptResult.from_dict(receiptmod.receipt(run))


def operation_schema() -> OperationSchema:
    """``operation-schema``: every command and its typed arguments."""
    from .operations import command_schema
    return OperationSchema.from_dict(command_schema())


def migrate(path: str | Path, check: bool = True) -> MigrationReport:
    """``migrate``: what a run, cache, store or document needs to be read by this version."""
    from . import migrate as mig
    return MigrationReport.from_dict(mig.migrate(Path(path), check=check))


def call(command: str, argv=()) -> dict:
    """Run any command in this process in JSON mode and return its result.  Untyped (outside this
    module's contract): use the typed functions where one exists."""
    from .cli import COMMANDS, load_commands
    load_commands()
    if command not in COMMANDS:
        raise UnknownCommand(command)
    fn = COMMANDS[command][0]
    parser = argparse.ArgumentParser(prog=f"n64dk {command}", exit_on_error=False)
    arguments = getattr(fn, "arguments", None)
    if arguments:
        arguments(parser)
    try:
        args = parser.parse_args([str(a) for a in argv])
    except (argparse.ArgumentError, SystemExit) as error:
        raise CommandFailed(command, 2, f"bad arguments: {error}") from None
    args.format, args.command, args._fn = "json", command, fn
    try:
        result = fn(args)
    except SystemExit as error:
        raise CommandFailed(command, error.code if isinstance(error.code, int) else 1, str(error.code)) from None
    if isinstance(result, int) or result is None:
        if result:
            raise CommandFailed(command, int(result))
        return {}
    return result
