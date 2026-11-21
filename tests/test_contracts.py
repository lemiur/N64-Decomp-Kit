"""The public contract (``n64dk.api``): the Python API, the CLI and the operation schema agree, and
JSON outputs that claim a format version carry the registered one.  Synthetic run directories
only; no toolchain, no ROM."""
import contextlib
import inspect
import io
import json
import os
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

from n64dk import api, cli, migrate as mig
from n64dk.match import resolve as resolvemod
from n64dk.match import solver

FIXED = "fixed point: a full round brought no new exact function, declaration, evidence or negative fact"


def make_run(root: Path) -> Path:
    """A small finished run: case store, invocation, rounds, coverage, dossiers, events, report."""
    run = root / "run"
    (run / "solver").mkdir(parents=True)
    store = resolvemod.CaseStore(run / "state")
    try:
        for name, vram, status in (("f1", 0x80001000, "exact"), ("f2", 0x80001040, "open")):
            store.put(resolvemod.Case(resolvemod.case_id("main", vram, 0x20), "main", name, vram, 0x20, "t" + name,
                                      status=status, best_score=0 if status == "exact" else 9, recipe="ido5.3 -O2",
                                      strict={"verdict": "exact"} if status == "exact" else {}))
    finally:
        store.close()
    (run / "solver" / "invocation.json").write_text(json.dumps({
        "effort": "exhaustive", "mode_settings": {"recipe_gate": False, "regional_calibration": True},
        "project": "<project>", "skip_stage": []}))
    log = solver.RoundsLog(run / "rounds.jsonl")
    log.append({"kind": "round", "session": "s", "image": "main", "round": 1, "techniques": {}, "cascade": {},
                "skipped": {}, "gains": {"new_exact": 1, "awaiting_declarations": 0, "evidence": 0}})
    log.append({"kind": "stop", "session": "s", "image": "main", "after_round": 1, "reason": FIXED})
    (run / "solver" / "coverage_main.json").write_text(json.dumps({
        "functions": 2, "attempted": 2, "not_attempted": 0, "unexplained": 0, "reasons": {},
        "entries": [{"name": "f1", "vram": "0x80001000", "size": 32, "attempted": True},
                    {"name": "f2", "vram": "0x80001040", "size": 32, "attempted": True}]}))
    (run / "solver" / "hard_queue_main.json").write_text(json.dumps([
        {"name": "f2", "case": resolvemod.case_id("main", 0x80001040, 0x20), "size": 32,
         "primary_cause": "register-only residual", "next_action": "search", "excluded_actions": {}}]))
    (run / "events.jsonl").write_text('{"event": "image-start", "image": "main"}\n')
    (run / "report.json").write_text(json.dumps({
        "status": "complete", "failed_stages": {}, "discovery": {"verdict": "complete", "images": {}},
        "small_data": {"value": None, "reason": "none"}, "donor": {},
        "images": {"main": {"functions": 2, "boundaries": {}, "resolve": {"statuses": {"exact": 1}},
                            "solver_stop": {"reason": FIXED}, "solver": {}, "toolchain_regions": {}, "queue": {},
                            "ledger": {"totals": {"sum_equals_extent": True}, "violations": {}}}}}))
    return run


def cli_json(argv: list[str]):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = cli.main(["--format", "json", *argv])
    text = buffer.getvalue()
    return code, (json.loads(text) if text.strip() else None)


def roundtrip(value):
    return json.loads(json.dumps(value, default=str))


class ContractTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="n64dk-contracts-test-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_api_version_is_semantic_and_registered(self):
        mig.parse_semver(api.API_VERSION)
        self.assertEqual(next(f for f in mig.registry() if f["name"] == "api")["current"], api.API_VERSION)

    def test_every_entry_point_matches_its_command_in_the_operation_schema(self):
        schema = api.operation_schema()
        self.assertEqual(roundtrip(schema.to_dict()), roundtrip(cli_json(["operation-schema"])[1]))
        for function, (command, mapping) in api.ENTRY_POINTS.items():
            with self.subTest(function=function):
                spec = schema.command(command)
                parameters = inspect.signature(getattr(api, function)).parameters
                self.assertEqual(set(parameters), set(mapping), "every API parameter maps to one CLI argument")
                self.assertLessEqual(set(mapping.values()), spec.dests)
                required = {a.dest for a in spec.arguments if a.required or not a.flags}
                self.assertLessEqual(required, set(mapping.values()), "every required CLI argument is in the API")
                for name, dest in mapping.items():
                    default = parameters[name].default
                    if default is inspect.Parameter.empty or (command, dest) in api_default_exceptions():
                        continue
                    self.assertEqual(default, spec.argument(dest).default, f"default of {name}")

    def test_status_and_receipt_equal_the_cli_json(self):
        run = make_run(self.root)
        code, cli_status = cli_json(["decompile-status", str(run)])
        self.assertEqual(code, 0)
        status = api.status(run)
        self.assertIsInstance(status.credit, api.Credit)
        self.assertEqual(status.credit.exact_c, 1)
        self.assertEqual(roundtrip(status.to_dict()), cli_status)
        code, cli_receipt = cli_json(["decompile-receipt", str(run)])
        self.assertEqual(code, 0)
        receipt = api.receipt(run)
        self.assertEqual(roundtrip(receipt.to_dict()), cli_receipt)
        self.assertEqual(receipt.clause("W1").status, "satisfied")
        self.assertEqual(api.call("decompile-status", [str(run)]), runstatus_status(run))

    def test_results_keep_unknown_keys_and_refuse_misshapen_data(self):
        run = make_run(self.root)
        data = api.status(run).to_dict()
        data["added_by_a_newer_kit"] = {"x": 1}
        again = api.StatusResult.from_dict(data)
        self.assertEqual(again.extra["added_by_a_newer_kit"], {"x": 1})
        self.assertEqual(again.to_dict(), data)
        with self.assertRaises(api.FormatError):
            api.StatusResult.from_dict({"run": "x"})
        with self.assertRaises(api.FormatError):
            api.ReceiptResult.from_dict({**api.receipt(run).to_dict(), "receipt": "n64dk.exhaustive-receipt.v9"})
        with self.assertRaises(api.FormatError):
            api.OperationSchema.from_dict({"protocol": "n64dk.operations.v2", "commands": []})

    def test_documented_exceptions(self):
        with self.assertRaises(api.NotARunDirectory):
            api.status(self.root / "missing")
        with self.assertRaises(api.NotARunDirectory):
            api.receipt(self.root / "missing")
        with self.assertRaises(api.UnknownCommand):
            api.call("no-such-command")
        with self.assertRaises(api.UnknownCommand):
            api.operation_schema().command("no-such-command")
        with self.assertRaises(api.CommandFailed) as caught:
            api.call("migrate", [str(self.root / "nothing-here"), "--check"])
        self.assertEqual(caught.exception.exit_code, 3)
        self.assertTrue(issubclass(api.CommandFailed, api.N64DKError))

    def test_versioned_outputs_carry_the_registered_version(self):
        run = make_run(self.root)
        manifest = self.root / "corpus.json"
        manifest.write_text(json.dumps({"schema": "n64dk.corpus/1", "entries": []}))
        forbidden = self.root / "truth"
        forbidden.mkdir()
        outputs = {"operation-schema": [], "decompile-receipt": [str(run)], "migrate": [str(run), "--check"],
                   "release-info": [], "corpus-check": [str(manifest)],
                   "blind-check": ["--forbid", str(forbidden), "--project", str(run), "--no-content"]}
        expected = {"operation-schema": "n64dk.operations.v1", "decompile-receipt": api._receipt_version(),
                    "migrate": mig.REPORT_SCHEMA, "release-info": "n64dk.release-info/1",
                    "corpus-check": "n64dk.corpus/1", "blind-check": "n64dk.blind-isolation/1"}
        for command, argv in outputs.items():
            with self.subTest(command=command):
                quiet = {k: v for k, v in os.environ.items() if not k.startswith("N64DK_")}
                with mock.patch.dict(os.environ, quiet, clear=True):    # blind-check reads N64DK_* paths
                    _code, result = cli_json([command, *argv])
                self.assertEqual(result[api.VERSIONED_OUTPUTS[command]], expected[command])
        self.assertFalse(set(api.VERSIONED_OUTPUTS) & set(api.UNVERSIONED_OUTPUTS))
        commands = {spec.name for spec in api.operation_schema().commands}
        self.assertLessEqual(set(api.VERSIONED_OUTPUTS) | set(api.UNVERSIONED_OUTPUTS), commands)

    def test_the_operation_schema_is_well_formed(self):
        schema = api.operation_schema()
        names = [c.name for c in schema.commands]
        self.assertEqual(len(names), len(set(names)))
        for spec in schema.commands:
            self.assertRegex(spec.name, r"^[a-z0-9]+(-[a-z0-9]+)*$")
            self.assertTrue(spec.description)
        json.dumps(schema.to_dict())

    def test_deprecation_policy(self):
        @api.deprecated("1.0.0", "2.0.0", "n64dk.api.status")
        def old(value):
            return value * 2
        name = f"{old.__module__}.{old.__qualname__}"
        self.addCleanup(api.DEPRECATIONS.pop, name, None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(old(2), 4)
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) and "2.0.0" in str(w.message) for w in caught))
        self.assertEqual(api.DEPRECATIONS[name], {"since": "1.0.0", "removal": "2.0.0",
                                                  "replacement": "n64dk.api.status"})
        major = mig.parse_semver(api.API_VERSION)[0]
        for entry in api.DEPRECATIONS.values():
            self.assertGreater(mig.parse_semver(entry["removal"])[0], major - 1)


def api_default_exceptions() -> set:
    """Where the API's default deliberately differs from the CLI's: the API's migrate only checks
    unless asked to change something; the CLI's --check is a flag."""
    return {("migrate", "check")}


def runstatus_status(run):
    from n64dk.commands import runstatus
    return runstatus.status(run)


if __name__ == "__main__":
    unittest.main()
