"""Documentation as an executable contract (tools/doc_examples.py, tools/cli_reference.py).

* every marked example in README.md and docs/*.md is well formed and every ``n64dk`` command in it
  parses with the real CLI parser;
* the examples whose fixture is synthetic and that need nothing else are run, in this process,
  against fixtures built here (no ROM, no compiler, no network); the others are skipped with the
  reason their marker gives;
* docs/cli-reference.md is exactly what tools/cli_reference.py generates from the parsers.
"""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tool(name):
    spec = importlib.util.spec_from_file_location(f"n64dk_tool_{name}", ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module                  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


docs = _tool("doc_examples")
reference = _tool("cli_reference")

# ------------------------------------------------------------------ synthetic fixtures


def _catalogue(path: Path, name: str, rows) -> None:
    lines = ["[[section]]", f'name = "{name}"', "rom = 0x0", f"vram = 0x{rows[0][1]:08X}", "size = 0x100", "",
             "functions = ["]
    lines += [f'    {{ name = "{n}", vram = 0x{v:08X}, size = 0x{s:X} }},' for n, v, s in rows]
    (path).write_text("\n".join(lines + ["]", ""]))


def fixture_none(work: Path) -> dict:
    return {"WORK": str(work)}


def fixture_empty_toolchain(work: Path) -> dict:
    (work / "toolchain").mkdir()
    return {"WORK": str(work), "TOOLCHAIN": str(work / "toolchain")}


def fixture_synthetic_project(work: Path) -> dict:
    project = work / "project"
    project.mkdir()
    (project / "main.bin").write_bytes(bytes(0x100))
    _catalogue(project / "functions.toml", "main", [("func_80000400", 0x80000400, 0x20),
                                                    ("func_80000420", 0x80000420, 0x40)])
    (project / "n64dk.toml").write_text('[images.main]\npath = "main.bin"\nvram_delta = 0x80000400\n'
                                        'catalogue = "functions.toml"\nmanifests = []\n')
    return {"WORK": str(work), "PROJECT": str(project)}


def fixture_overlay_project(work: Path) -> dict:
    project = work / "overlays"
    project.mkdir()
    for name, rows in (("ovl_a", [("ovl_a_first", 0x80100000, 0x40), ("ovl_a_second", 0x80100040, 0x20)]),
                       ("ovl_b", [("ovl_b_only", 0x80100000, 0x80)])):
        (project / f"{name}.bin").write_bytes(bytes(0x100))
        _catalogue(project / f"{name}.toml", name, rows)
    (project / "n64dk.toml").write_text(
        '[images.ovl_a]\npath = "ovl_a.bin"\nvram_delta = 0x80100000\ncatalogue = "ovl_a.toml"\n\n'
        '[images.ovl_b]\npath = "ovl_b.bin"\nvram_delta = 0x80100000\ncatalogue = "ovl_b.toml"\n')
    return {"WORK": str(work), "PROJECT": str(project)}


BUILDERS = {"none": fixture_none, "empty-toolchain": fixture_empty_toolchain,
            "synthetic-project": fixture_synthetic_project, "overlay-project": fixture_overlay_project}

# the workflow topics docs/workflows.md must keep covering (item 24 of the product addendum)
REQUIRED_IDS = {"doctor-host", "bare-rom-run", "exhaustive-run", "source-assisted-run", "custom-compiler-run",
                "small-data-run", "overlay-ambiguous", "overlay-run", "resume", "refine-mock", "export",
                "receipts", "project-rebuild", "dashboard"}


class ExtractionTest(unittest.TestCase):
    TEXT = "\n".join([
        "# t", "", "<!-- example: fixture=none id=one -->", "```sh",
        "# a comment", "export A=1", "mkdir -p \"$WORK/x\"", "n64dk --format json jobs \\", "    --stale-after 5",
        "```", "", "```sh", "n64dk not-marked", "```", "",
        "<!-- example: fixture=rom requires=rom,compiler expect=nonzero -->", "", "```bash",
        "N64DK_MAX_JOBS=1 n64dk info \"$ROM\"", "cat file", "```", ""])

    def test_markers_steps_and_continuations(self):
        one, two = docs.extract_text(self.TEXT, "t.md")
        self.assertEqual((one.ident, one.fixture, one.expect), ("one", "none", "0"))
        self.assertEqual(one.steps, [("set", ("A", "1")), ("mkdir", "$WORK/x"),
                                     ("n64dk", ["--format", "json", "jobs", "--stale-after", "5"])])
        self.assertIsNone(one.skip_reason())
        self.assertEqual((two.ident, two.requires, two.expect), ("t.md:18", ["rom", "compiler"], "nonzero"))
        self.assertEqual(two.commands, [["info", "$ROM"]])
        self.assertEqual(two.steps[0], ("set", ("N64DK_MAX_JOBS", "1")))
        self.assertEqual(two.foreign, ["cat file"])
        self.assertEqual(two.skip_reason(), "needs rom, compiler")

    def test_malformed_markers_fail(self):
        bad = {"<!-- example: requires=rom -->\n```sh\nn64dk jobs\n```\n": "no fixture",
               "<!-- example: fixture=nowhere -->\n```sh\nn64dk jobs\n```\n": "unknown fixture",
               "<!-- example: fixture=none requires=moon -->\n```sh\nn64dk jobs\n```\n": "unknown requirement",
               "<!-- example: fixture=none expect=maybe -->\n```sh\nn64dk jobs\n```\n": "expect",
               "<!-- example: fixture=none -->\ntext\n```sh\nn64dk jobs\n```\n": "not directly above",
               "<!-- example: fixture=none -->\n```toml\na = 1\n```\n": "not a shell block",
               "<!-- example: fixture=none -->\n```sh\nls\n```\n": "no n64dk command",
               "<!-- example: fixture=none color=red -->\n```sh\nn64dk jobs\n```\n": "unknown marker keys"}
        for text, why in bad.items():
            with self.subTest(why=why), self.assertRaisesRegex(docs.MarkerError, why):
                docs.extract_text(text, "bad.md")

    def test_parse_check_catches_drift(self):
        self.assertIsNone(docs.parse_errors(["jobs", "--stale-after", "5"]))
        self.assertIsNotNone(docs.parse_errors(["jobs", "--no-such-option"]))
        self.assertIsNotNone(docs.parse_errors(["no-such-command"]))
        self.assertIsNone(docs.parse_errors(["info", "$ROM"]))          # placeholders stand in for files


class DocumentationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = docs.extract()

    def test_markers_are_valid_and_the_workflows_are_covered(self):
        ids = {e.ident for e in self.examples}
        self.assertLessEqual(REQUIRED_IDS, ids, f"missing workflow examples: {sorted(REQUIRED_IDS - ids)}")
        for example in self.examples:
            self.assertIn(example.fixture, docs.FIXTURES)
            if not example.requires and not example.foreign:
                self.assertIn(example.fixture, BUILDERS, f"{example.ident}: runnable example without a fixture builder")

    def test_every_documented_command_parses(self):
        self.assertEqual(docs.check(self.examples), [])

    def test_runnable_examples_run(self):
        for example in self.examples:
            with self.subTest(example=example.ident):
                reason = example.skip_reason()
                if reason:
                    self.skipTest(reason)
                with tempfile.TemporaryDirectory(prefix="n64dk-docex-") as name:
                    work = Path(name)
                    variables = BUILDERS[example.fixture](work)
                    result = docs.run_example(example, variables, ROOT)
                self.assertTrue(result["ok"], f"{example.ident}: {result['why']}\n{result['results'][-1:]}")


class CliReferenceTest(unittest.TestCase):
    def test_parser_matches_the_registry(self):
        from n64dk import cli
        cli.load_commands()
        built = reference.build_parser()
        sub = next(a for a in built._actions if a.dest == "command")
        self.assertEqual(set(sub.choices), set(cli.COMMANDS))

    def test_reference_is_complete_and_reproducible(self):
        from n64dk import cli
        text = reference.render()
        self.assertEqual(text, reference.render())
        for name in cli.COMMANDS:
            self.assertIn(f"\n## {name}\n", text)
        self.assertIn("--host", text)                       # options come from the parsers

    def test_generated_file_is_current(self):
        target = ROOT / "docs" / "cli-reference.md"
        current = target.read_text() if target.is_file() else ""
        self.assertTrue(current == reference.render(),
                        "docs/cli-reference.md is missing or out of date: run python tools/cli_reference.py --write")


if __name__ == "__main__":
    unittest.main()
