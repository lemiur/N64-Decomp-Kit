import hashlib
import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from n64dk import plugins


def make_plugin(root: Path, name: str, script: str, **overrides) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    exe = directory / "run.sh"
    exe.write_text("#!/bin/sh\n" + script + "\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    manifest = {"id": name, "version": "1.0", "capability": "codec", "inputs": ["bytes"], "deterministic": True,
                "weight": "light", "dependencies": [], "provenance": "test", "output_schema": {"required": ["size"]},
                "cache_identity": "v1", "safety": "pure-function", "executable": "run.sh",
                "sha256": hashlib.sha256(exe.read_bytes()).hexdigest()}
    manifest.update(overrides)
    (directory / "plugin.json").write_text(json.dumps(manifest))
    return directory


def local_runner(argv, *, kind, timeout, input, cwd, env, text):
    return subprocess.run(argv, input=input, cwd=cwd, env=env, text=text, capture_output=True, timeout=timeout)


class PluginTest(unittest.TestCase):
    def test_only_allowlisted_directories_are_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_plugin(Path(tmp) / "allowed", "ok", 'echo \'{"result": {"size": 4}}\'')
            make_plugin(Path(tmp) / "project", "sneaky", 'echo \'{"result": {"size": 4}}\'')
            with mock.patch.dict(os.environ, {"N64DK_PLUGIN_DIRS": str(Path(tmp) / "allowed")}):
                registry = plugins.load()
            self.assertEqual([p.ident for p in registry.plugins], ["ok@1.0"])
            self.assertEqual(plugins.load(allow=[]).plugins, [])

    def test_incomplete_or_unpinned_manifests_are_refused_with_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_plugin(Path(tmp), "nofields", "true", capability="teleporter", sha256="0" * 64)
            directory = make_plugin(Path(tmp), "missing", "true")
            manifest = json.loads((directory / "plugin.json").read_text())
            del manifest["output_schema"]
            (directory / "plugin.json").write_text(json.dumps(manifest))
            registry = plugins.load(allow=[tmp])
            reasons = {r["directory"]: r["reasons"] for r in registry.refused}
            self.assertIn("unknown capability 'teleporter'", reasons["nofields"])
            self.assertIn("executable does not have the pinned SHA-256", reasons["nofields"])
            self.assertIn("missing field 'output_schema'", reasons["missing"])
            self.assertEqual(registry.plugins, [])

    def test_a_call_returns_evidence_never_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_plugin(Path(tmp), "ok", 'cat >/dev/null; echo \'{"result": {"size": 4}}\'')
            plugin = plugins.load(allow=[tmp]).plugins[0]
            answer = plugins.call(plugin, {"offset": 16}, runner=local_runner)
            self.assertEqual((answer["proof"], answer["result"]), (False, {"size": 4}))
            self.assertEqual(answer["cache_key"], plugin.cache_key({"offset": 16}))
            self.assertNotEqual(answer["cache_key"], plugin.cache_key({"offset": 17}))

    def test_failures_are_isolated_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = {"crash": "exit 3", "garbage": "echo not-json", "schema": 'echo \'{"result": {"other": 1}}\'',
                     "slow": "sleep 5"}
            for name, script in cases.items():
                make_plugin(Path(tmp), name, script)
            registry = plugins.load(allow=[tmp])
            got = {}
            for plugin in registry.plugins:
                result = plugins.call(plugin, {}, timeout=1, runner=local_runner)
                self.assertIsInstance(result, plugins.PluginFailure)
                got[plugin.manifest["id"]] = result.reason
            self.assertEqual(got, {"crash": "exit-status", "garbage": "output-not-json", "schema": "output-schema",
                                   "slow": "timeout"})

    def test_a_replaced_executable_is_refused_at_call_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = make_plugin(Path(tmp), "ok", 'echo \'{"result": {"size": 4}}\'')
            plugin = plugins.load(allow=[tmp]).plugins[0]
            (directory / "run.sh").write_text("#!/bin/sh\necho evil\n")
            result = plugins.call(plugin, {}, runner=local_runner)
            self.assertEqual(result.reason, "manifest-invalid")

    def test_the_environment_is_not_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_plugin(Path(tmp), "env", 'printf \'{"result": {"size": "%s"}}\' "${SECRET_TOKEN:-none}"')
            plugin = plugins.load(allow=[tmp]).plugins[0]
            with mock.patch.dict(os.environ, {"SECRET_TOKEN": "abc"}):
                result = plugins.call(plugin, {}, runner=local_runner)
            self.assertEqual(result["result"]["size"], "none")

    def test_every_capability_has_a_builtin(self):
        self.assertEqual(set(plugins.builtin_capabilities()), set(plugins.CAPABILITIES))


if __name__ == "__main__":
    unittest.main()
