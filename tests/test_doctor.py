import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from n64dk.commands import doctor as doc
from n64dk.resources import MemorySnapshot

KIT = Path("/kit")
HOME = Path("/h/someone")


class FakeProbes(doc.Probes):
    """A machine described by plain data: nothing real is probed."""

    def __init__(self, **changes):
        self.version = (3, 11, 9)
        self.modules = {"rabbitizer", "sqlite3", "zlib", "fcntl", "m2c", "pycparser"}
        self.versions = {"rabbitizer": "1.16.2"}
        self.tools = {"git": "/usr/bin/git", "gcc": "/usr/bin/gcc"}
        self.plat = {"system": "Linux", "machine": "x86_64", "release": "6.1.0"}
        self.files = {"/proc/version": "Linux version 6.1.0 (gcc) #1 SMP"}
        self.envs = {}
        self.free = {}
        self.default_free = 50_000
        self.snap = MemorySnapshot(16_000, 9_000, 0, 0)
        self.status = {"root": "/tc", "known": 20, "installed": 2, "compilers": [
            {"id": "gcc2.7.2kmc", "usable": True}, {"id": "ido5.3", "usable": False}],
            "installed_without_lane": {"ido5.3": "why"}}
        self.tree = {KIT: ["n64dk", "pyproject.toml", ".git", "tests"], KIT / "n64dk": [], HOME: [],
                     Path("/"): ["h", "kit", "work"], Path("/work"): []}
        self.readonly = set()
        self.heads = {}
        self.globs = {}
        for key, value in changes.items():
            setattr(self, key, value)

    def python_version(self):
        return self.version

    def has_module(self, name):
        return name in self.modules

    def module_version(self, name):
        return self.versions.get(name)

    def which(self, name):
        return self.tools.get(name)

    def platform(self):
        return dict(self.plat)

    def read_text(self, path):
        return self.files.get(path)

    def env(self, name):
        return self.envs.get(name)

    def free_mb(self, path):
        return self.free.get(str(path), self.default_free)

    def memory(self):
        return self.snap

    def compilers(self, root):
        if isinstance(self.status, Exception):
            raise self.status
        return self.status

    def listdir(self, path):
        return list(self.tree.get(Path(path), []))

    def is_dir(self, path):
        return Path(path) in self.tree

    def is_file(self, path):
        path = Path(path)
        return path not in self.tree and path.name in self.tree.get(path.parent, [])

    def writable(self, path):
        return Path(path) not in self.readonly

    def head(self, path, size=256):
        return self.heads.get(Path(path))

    def glob(self, path, pattern):
        return self.globs.get((Path(path), pattern), [])

    def home(self):
        return HOME


def run(probes=None, **kw):
    kw.setdefault("toolchain_dir", "/tc")
    return doc.run_checks(probes=probes or FakeProbes(), kit_root=KIT, **kw)


def rows(report, category):
    return {row["name"]: row for row in report["checks"] if row["category"] == category}


class DoctorTest(unittest.TestCase):
    def test_healthy_machine(self):
        report = run()
        self.assertTrue(report["ok"])
        self.assertEqual((report["exit_code"], report["required_failures"]), (0, []))
        for row in report["checks"]:
            self.assertEqual(set(row), {"category", "name", "status", "ok", "required", "detail", "remediation"})
            self.assertIn(row["status"], doc.STATUSES)
        self.assertEqual(set(report), {"doctor", "schema", "ok", "exit_code", "required_failures", "summary",
                                       "host", "toolchain_root", "checks"})
        json.dumps(report)

    def test_required_dependencies(self):
        for probes, name in ((FakeProbes(modules={"sqlite3", "zlib", "fcntl"}), "rabbitizer"),
                             (FakeProbes(versions={"rabbitizer": "1.15.0"}), "rabbitizer"),
                             (FakeProbes(version=(3, 10, 12)), "python"),
                             (FakeProbes(modules={"rabbitizer", "zlib", "fcntl"}), "sqlite3")):
            with self.subTest(name=name):
                report = run(probes)
                self.assertEqual(report["exit_code"], 1)
                self.assertIn(f"required_dependency:{name}", report["required_failures"])

    def test_optional_and_compilers_never_fail_the_exit_code(self):
        report = run(FakeProbes(modules={"rabbitizer", "sqlite3", "zlib", "fcntl"}, tools={}), toolchain_dir=None)
        self.assertEqual(report["exit_code"], 0)
        optional = rows(report, "optional_capability")
        self.assertEqual(optional["m2c"]["status"], "missing")
        self.assertIsNone(optional["yaml"]["remediation"])           # PyYAML is not needed
        self.assertEqual(optional["emulator"]["status"], "ok")
        self.assertEqual(rows(report, "compilers")["toolchain root"]["status"], "missing")
        self.assertFalse(rows(report, "compilers")["toolchain root"]["required"])

    def test_compiler_status(self):
        ok = rows(run(), "compilers")["usable compilers"]
        self.assertEqual((ok["status"], ok["detail"]["usable"]), ("ok", ["gcc2.7.2kmc"]))
        none = rows(run(FakeProbes(status={"compilers": [{"id": "x", "usable": False}]})), "compilers")
        self.assertEqual(none["usable compilers"]["status"], "missing")
        broken = rows(run(FakeProbes(status=OSError("bad root"))), "compilers")
        self.assertEqual(broken["toolchain root"]["status"], "unknown")

    def test_host_platforms_and_wsl(self):
        windows = run(FakeProbes(plat={"system": "Windows", "machine": "AMD64", "release": "10"},
                                 modules={"rabbitizer", "sqlite3", "zlib"}))
        self.assertIn("host:operating system", windows["required_failures"])
        self.assertIn("host:fcntl", windows["required_failures"])
        wsl2 = run(FakeProbes(files={"/proc/version": "Linux version 5.15.90.1-microsoft-standard-WSL2 (gcc)"}))
        self.assertEqual((wsl2["host"]["wsl"], wsl2["host"]["wsl_version"]), (True, 2))
        self.assertEqual(rows(wsl2, "host")["wsl"]["status"], "ok")
        wsl1 = run(FakeProbes(files={"/proc/version": "Linux version 4.4.0-19041-Microsoft (Microsoft@Microsoft.com)"}))
        self.assertEqual((wsl1["host"]["wsl_version"], rows(wsl1, "host")["wsl"]["status"]), (1, "warn"))
        self.assertEqual(wsl1["exit_code"], 0)
        native = run()
        self.assertFalse(native["host"]["wsl"])
        self.assertNotIn("wsl", rows(native, "host"))
        arm = run(FakeProbes(plat={"system": "Linux", "machine": "aarch64", "release": "6"}))
        self.assertEqual((rows(arm, "host")["cpu architecture"]["status"], arm["exit_code"]), ("warn", 0))
        mac = run(FakeProbes(plat={"system": "Darwin", "machine": "arm64", "release": "23"}))
        self.assertEqual(mac["exit_code"], 0)

    def test_disk_floor(self):
        state = str(HOME / ".cache" / "n64dk")
        low = run(FakeProbes(free={state: 100}))
        self.assertEqual(rows(low, "disk")["state directory"]["status"], "insufficient")
        self.assertEqual(low["exit_code"], 1)
        lowered = run(FakeProbes(free={state: 100}, envs={"N64DK_MIN_FREE_DISK_MB": "50"}))
        self.assertEqual(lowered["exit_code"], 0)
        moved = run(FakeProbes(free={"/elsewhere": 100}, envs={"N64DK_STATE_DIR": "/elsewhere"}))
        self.assertEqual(moved["exit_code"], 1)
        out = run(FakeProbes(free={"/work/out": 10}), out="/work/out")
        self.assertEqual(rows(out, "disk")["output directory"]["status"], "insufficient")

    def test_memory_total_fails_available_only_warns(self):
        small = run(FakeProbes(snap=MemorySnapshot(2048, 1500, 0, 0)))
        self.assertIn("memory:total memory", small["required_failures"])
        busy = run(FakeProbes(snap=MemorySnapshot(8000, 1000, 2048, 1024)))
        memory = rows(busy, "memory")
        self.assertEqual((memory["total memory"]["status"], memory["available memory"]["status"]),
                         ("ok", "insufficient"))
        self.assertIn("1024 MB swap in use", memory["available memory"]["detail"])
        self.assertEqual(busy["exit_code"], 0)
        unknown = run(FakeProbes(snap=MemorySnapshot(None, None, None, None)))
        self.assertEqual(rows(unknown, "memory")["total memory"]["status"], "unknown")

    def test_output_paths(self):
        project = Path("/work/title")
        tree = dict(FakeProbes().tree)
        tree.update({project: ["n64dk.toml", "baserom.z64", ".n64dk"], project / ".n64dk": [],
                     Path("/work/roms"): ["game.z64", "other.v64"], Path("/work/clean"): [],
                     Path("/work/locked"): []})
        probes = FakeProbes(tree=tree, readonly={Path("/work/locked")})
        cases = {"/kit/build/out": "inside the kit repository", "/work/title/out": "inside the source project",
                 "/work/roms/out": "holds ROM images", "/work/roms": "holds ROM images",
                 "/work/locked/out": "is not writable"}
        for out, why in cases.items():
            with self.subTest(out=out):
                row = rows(run(probes, out=out), "output_path")["output directory"]
                self.assertEqual(row["status"], "unsafe")
                self.assertIn(why, row["detail"])
                self.assertTrue(row["required"])
        for out in ("/work/title/.n64dk/run1", "/work/clean/out"):
            with self.subTest(out=out):
                report = run(probes, out=out)
                self.assertEqual(rows(report, "output_path")["output directory"]["status"], "ok", out)
                self.assertEqual(report["exit_code"], 0)
        drive = rows(run(probes, out="/mnt/d/scratch/out"), "output_path")
        self.assertEqual(drive["windows drive"]["status"], "warn")
        self.assertEqual(rows(run(probes), "output_path"), {})

    def test_schema_checks(self):
        from n64dk.fixtures import SNAPSHOT_FORMAT
        from n64dk.match.transplant import INDEX_VERSION
        run_dir = Path("/work/run")
        tree = dict(FakeProbes().tree)
        tree[run_dir] = ["donor_index.json"]
        truth = KIT / "build" / "truth"
        stale_stamp, fresh_stamp = truth / "a-1" / ".n64dk-snapshot", truth / "b-2" / ".n64dk-snapshot"
        probes = FakeProbes(tree=tree, heads={
            run_dir / "donor_index.json": b'{"version": ' + str(INDEX_VERSION - 1).encode() + b', "donors": {}}',
            stale_stamp: f"format {SNAPSHOT_FORMAT - 1}\n/x\nabc\n".encode(),
            fresh_stamp: f"format {SNAPSHOT_FORMAT}\n/x\nabc\n".encode()},
            globs={(truth, "*/.n64dk-snapshot"): [stale_stamp, fresh_stamp]})
        report = run(probes, run=str(run_dir))
        schema = rows(report, "schema")
        self.assertEqual(schema["donor index"]["status"], "stale")
        self.assertEqual(schema["fixture snapshots"]["status"], "stale")
        self.assertIn("a-1", schema["fixture snapshots"]["detail"])
        self.assertEqual(report["exit_code"], 0)                 # rebuilt automatically: not a failure
        probes.heads[run_dir / "donor_index.json"] = b'{"version": ' + str(INDEX_VERSION).encode() + b"}"
        self.assertEqual(rows(run(probes, run=str(run_dir)), "schema")["donor index"]["status"], "ok")


class CommandTest(unittest.TestCase):
    def test_doctor_host_flag_reaches_the_host_doctor(self):
        from n64dk.commands import workbench
        parser = argparse.ArgumentParser()
        workbench.doctor.arguments(parser)
        args = parser.parse_args(["--host", "--out", "x", "--toolchain-dir", "tc"])
        args.format = "json"
        with mock.patch.object(doc, "host_doctor", return_value=1) as called:
            self.assertEqual(workbench.doctor(args), 1)
        called.assert_called_once_with(args)

    def test_host_doctor_prints_json_and_returns_the_exit_code(self):
        report = run(FakeProbes(version=(3, 9, 0)))
        args = argparse.Namespace(format="json", out=None, run=None, toolchain_dir=None)
        stream = io.StringIO()
        with mock.patch.object(doc, "run_checks", return_value=report), contextlib.redirect_stdout(stream):
            code = doc.host_doctor(args)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(stream.getvalue())["required_failures"], ["required_dependency:python"])
        args.format = "text"
        stream = io.StringIO()
        with mock.patch.object(doc, "run_checks", return_value=report), contextlib.redirect_stdout(stream):
            doc.host_doctor(args)
        self.assertIn("REQUIRED CHECKS FAILED", stream.getvalue())

    def test_synthetic_match_skips_without_compilers(self):
        self.assertEqual(doc.synthetic_match(None, "unused")["status"], "SKIP")
        with tempfile.TemporaryDirectory() as tmp:
            result = doc.synthetic_match(tmp, Path(tmp) / "work")
        self.assertEqual(result["status"], "SKIP")


if __name__ == "__main__":
    unittest.main()
