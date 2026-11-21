"""The model stage's tool protocol (n64dk.match.agenttools) and the stage loop (n64dk.match.agentstage)
on a synthetic project built by the test itself: a reference C file is compiled with the project's
compiler, its code is placed in an image file, and the catalogue, manifest and recipe are written
into n64dk.toml.  Compiler round trips need N64DK_TOOLCHAIN (KMC GCC); they are skipped without it."""
from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from n64dk.match import agentstage as stage
from n64dk.match import agenttools as tools

PRELUDE = "typedef int s32;\ntypedef short s16;\n"
TARGET = PRELUDE + "s32 mix(s32 a, s32 b) {\n    return a + b * 3;\n}\n"
WRONG = PRELUDE + "s32 mix(s32 a, s32 b) {\n    return a + b * 5;\n}\n"
NEIGHBOUR = PRELUDE + ("typedef struct {\n    s32 unk_00;\n    s16 unk_04;\n} Thing;\n\n"
                       "s32 peek(Thing *t) {\n    return t->unk_00 + t->unk_04;\n}\n")
FLAGS = "-O2 -G0 -mips3 -mgp32 -mfp32"
BASE = 0x80000000
MIX_AT, PEEK_AT = 0x80000100, 0x80000200


def _kmc_ready() -> bool:
    from n64dk.match.toolchain import load_config
    return load_config(Path(tempfile.gettempdir())).toolchain("kmc") is not None


def _compile(folder: Path, name: str, text: str) -> bytes:
    from n64dk.elf import Elf
    from n64dk.match import compare as cmp
    from n64dk.match.pipeline import build, split_flags
    from n64dk.match.toolchain import load_config
    path = folder / name
    path.write_text(text)
    elf = Elf.load(build(path, split_flags(FLAGS.split(), "kmc+builtin"), load_config(folder), folder / ("o_" + name)))
    return cmp.find_section(elf, ".text").data.rstrip(b"\0")


def make_project(root: Path) -> dict:
    """A one-image project: mix() at MIX_AT (unmatched), peek() at PEEK_AT (a matched unit)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(exist_ok=True)
    mix = _compile(root, "ref_mix.c", TARGET)
    peek = _compile(root, "ref_peek.c", NEIGHBOUR)
    image = bytearray(0x400)
    image[0x100:0x100 + len(mix)] = mix
    image[0x200:0x200 + len(peek)] = peek
    (root / "game.bin").write_bytes(bytes(image))
    (root / "src" / "peek.c").write_text(NEIGHBOUR)
    (root / "functions.toml").write_text(
        "[[section]]\nname = \".main\"\nvram = 0x80000000\nsize = 0x400\nfunctions = [\n"
        f"  {{ name = \"mix\", vram = 0x{MIX_AT:08x}, size = 0x{len(mix):x} }},\n"
        f"  {{ name = \"peek\", vram = 0x{PEEK_AT:08x}, size = 0x{len(peek):x} }},\n]\n")
    (root / "units.tsv").write_text("# OFFSET\tSIZE\tSOURCE\tFLAGS\tSHA256\n"
                                    f"0x200\t0x{len(peek):X}\tsrc/peek.c\t{FLAGS}\t0\n")
    (root / "n64dk.toml").write_text(
        "[images.main]\npath = \"game.bin\"\nvram_delta = 0x80000000\ncatalogue = \"functions.toml\"\n"
        "manifests = [{ path = \"units.tsv\", pipeline = \"kmc+builtin\" }]\n\n"
        f"[run]\npipeline = \"kmc+builtin\"\nflags = \"{FLAGS}\"\n")
    return {"mix": len(mix), "peek": len(peek)}


class ClassifyAndBoundaryTests(unittest.TestCase):
    """No compiler needed."""

    def words(self, *values) -> bytes:
        return struct.pack(f">{len(values)}I", *values)

    def test_categories_from_the_difference(self):
        a = self.words(0x27BDFFE8, 0x00851021, 0x03E00008, 0x27BD0018)       # addiu sp,-24; addu v0,a0,a1; jr; addiu
        regs = self.words(0x27BDFFE8, 0x00A41021, 0x03E00008, 0x27BD0018)    # addu v0,a1,a0
        frame = self.words(0x27BDFFE0, 0x00851021, 0x03E00008, 0x27BD0020)
        longer = a + self.words(0x24020001)
        self.assertEqual(stage.classify(a, regs)[0], "reg-swap")
        self.assertEqual(stage.classify(a, frame)[0], "frame")
        self.assertEqual(stage.classify(a, longer)[0], "merge-tail")
        self.assertEqual(stage.classify(a, None)[0], "unwritten")
        self.assertEqual(stage.classify(a, regs, boundary_agrees=False)[0], "boundary")
        self.assertEqual(stage.classify(a, regs, fixmul=True)[0], "assisted")
        swapped = self.words(0x27BDFFE8, 0x03E00008, 0x00851021, 0x27BD0018)
        self.assertEqual(stage.classify(a, swapped)[0], "schedule")

    def test_boundary_finds_the_prologue_before_a_piece(self):
        code = self.words(0x27BDFFE8, 0xAFBF0010, 0x0C000000, 0x00000000, 0x8FBF0010, 0x27BD0018,
                          0x03E00008, 0x00000000, 0x27BDFFF0)
        found = tools.find_boundary(code, BASE, BASE + 8, [dict(name="piece", vram=BASE + 8, size=8)])
        self.assertEqual((found["start"], found["size"], found["frame"]), (BASE, 32, 0x18))
        self.assertEqual([r["name"] for r in found["catalogue"]], ["piece"])

    def test_playbook_is_package_data_and_title_independent(self):
        for name in ("BRIEF.md", "COMPILERS.md", "TECHNIQUES.md", "LESSONS.md"):
            text = stage.playbook_text(name)
            self.assertNotRegex(text, r"\b8[0-9A-F]{7}\b", name)      # no title addresses
            self.assertNotIn("Madden", text)
        self.assertIn("floor_log2(refs)", stage.sections(stage.playbook_text("COMPILERS.md"), ["gcc"]))
        self.assertIn("version siblings", stage.sections(stage.playbook_text("COMPILERS.md"), ["ido"]))
        for category in stage.CATEGORIES:
            self.assertTrue(stage.sections(stage.playbook_text("TECHNIQUES.md"), [category]), category)

    def test_command_provider_refuses_credentials_and_substitutes_the_folder(self):
        from n64dk.match.providers import CredentialLeak
        with self.assertRaises(CredentialLeak):
            stage.CommandAgentProvider(["agent", "--api-key", "abcdefabcdefabcdefabcdef"])
        seen = {}

        def runner(argv, prompt, timeout, env, cwd=None, output_limit=0):
            seen.update(argv=argv, cwd=cwd, prompt=prompt, env=env)
            return "done"
        os.environ["N64DK_TEST_AGENT_SECRET"] = "s3"
        try:
            provider = stage.CommandAgentProvider(["agent", "--dir", "{workdir}"], env_names=["N64DK_TEST_AGENT_SECRET"],
                                                  runner=runner)
            out = provider.run("ORDER", Path("/tmp/w"), 5)
        finally:
            del os.environ["N64DK_TEST_AGENT_SECRET"]
        self.assertEqual(seen["argv"], ["agent", "--dir", "/tmp/w"])
        self.assertEqual(seen["env"], {"N64DK_TEST_AGENT_SECRET": "s3"})
        self.assertEqual(out["transcript"], "done")
        self.assertNotIn("s3", json.dumps(provider.identity))

    def test_lessons_are_appended_once_without_addresses(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "n64dk.toml").write_text("")
            ws = tools.Workspace(tmp)
            fn = tools.Function("main", "f", 0x80001234, 8)
            entry = stage.append_lesson(ws, fn, "type", "- (type) flag at 800A1234 was s16, not s32")
            self.assertEqual(entry, "- (type) flag at ADDR was s16, not s32")
            self.assertIsNone(stage.append_lesson(ws, fn, "type", "flag at 800A1234 was s16, not s32"))
            self.assertIn(entry, stage.lessons_path(ws).read_text())


@unittest.skipUnless(_kmc_ready(), "KMC GCC toolchain not configured (N64DK_TOOLCHAIN)")
class ToolProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="n64dk-agent-tools-")
        cls.root = Path(cls.tmp.name) / "proj"
        cls.sizes = make_project(cls.root)
        (cls.root / "work").mkdir()
        cls.good = cls.root / "work" / "good.c"
        cls.good.write_text(TARGET)
        cls.bad = cls.root / "work" / "bad.c"
        cls.bad.write_text(WRONG)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def ws(self):
        return tools.Workspace(self.root)

    def test_lookup_by_name_address_and_true_range(self):
        ws = self.ws()
        self.assertEqual(ws.lookup("main", "mix").vram, MIX_AT)
        self.assertEqual(ws.lookup("main", f"{MIX_AT:08X}").name, "mix")
        with self.assertRaises(tools.ToolError):
            ws.lookup("main", f"{MIX_AT + 4:08X}")
        fn = ws.lookup("main", f"{MIX_AT + 4:08X}", 8)
        self.assertTrue(fn.true_range)
        self.assertEqual(fn.name, f"func_{MIX_AT + 4:08X}")

    def test_info_lists_the_default_and_sibling_lanes_but_a_unit_its_own(self):
        lanes = tools.run_tool(self.ws(), "info", "main", "mix")["lanes"]
        self.assertEqual([lane.split()[0] for lane in lanes], ["kmc+builtin", "kmc+asn64"])
        self.assertEqual(len(tools.run_tool(self.ws(), "info", "main", "peek")["lanes"]), 1)
        forced = tools.run_tool(self.ws(), "info", "main", "mix", pipe="kmc+asn64")["lanes"]
        self.assertEqual([lane.split()[0] for lane in forced], ["kmc+asn64"])

    def test_dis_try_and_exact(self):
        ws = self.ws()
        self.assertTrue(any("jr $ra" in line for line in tools.run_tool(ws, "dis", "main", "mix")["lines"]))
        tried = tools.run_tool(ws, "try", "main", "mix", [self.good, self.bad])
        self.assertEqual([r["status"] for r in tried["results"]], ["match", "differs"])
        exact = tools.run_tool(ws, "exact", "main", "mix", [self.good, self.bad])
        self.assertEqual([r["accepted"] for r in exact["results"]], [True, False])
        self.assertTrue(any(line.startswith("good.c EXACT") for line in exact["lines"]))
        self.assertTrue(any(line.startswith("bad.c NOT EXACT") for line in exact["lines"]))

    def test_exact_never_credits_asm_or_scaffolding_and_marks_assisted_macros(self):
        folder = self.root / "work"
        (folder / "fix.h").write_text("#define BARRIER() __asm__ volatile(\"\")\n")
        assisted = folder / "assisted.c"
        assisted.write_text('#include "fix.h"\n' + TARGET)
        scaffold = folder / "scaffold.c"
        scaffold.write_text(TARGET + "/* M2C_ERROR */\n")
        res = tools.run_tool(self.ws(), "exact", "main", "mix", [assisted, scaffold])["results"]
        self.assertEqual([(r["accepted"], r["assisted"]) for r in res], [(False, True), (False, False)])
        self.assertTrue(res[1]["rejected"])

    def test_neighbours_and_boundary(self):
        nb = tools.run_tool(self.ws(), "neighbours", "main", "mix")
        self.assertEqual(nb["neighbours"], ["src/peek.c"])
        self.assertIn("Thing", nb["structs"])
        b = tools.run_tool(self.ws(), "boundary", "main", "mix")
        self.assertEqual((b["start"], b["size"]), (MIX_AT, self.sizes["mix"]))
        self.assertTrue(b["agrees"])

    def test_regdump_shows_allocation_dumps(self):
        out = tools.run_tool(self.ws(), "regdump", "main", "mix", [self.good])
        self.assertTrue(any(name.endswith(".lreg") for name in out["dumps"]), out["lines"][:5])

    def test_session_calls_are_logged_and_budgeted(self):
        folder = self.root / "sess"
        folder.mkdir(exist_ok=True)
        (folder / "a.c").write_text(TARGET)
        (folder / tools.SESSION).write_text(json.dumps(dict(project=str(self.root), image="main", key=f"{MIX_AT:08X}",
                                                            vram=MIX_AT, size=self.sizes["mix"], tool_calls=1)))
        first = tools.session_call(folder / tools.SESSION, "try", [str(folder / "a.c")])
        self.assertEqual(first["results"][0]["status"], "match")
        second = tools.session_call(folder / tools.SESSION, "try", [str(folder / "a.c")])
        self.assertTrue(second.get("refused"))
        log = [json.loads(line) for line in (folder / "calls.jsonl").read_text().splitlines()]
        self.assertEqual(len(log), 1)
        self.assertIn("a.c", log[0]["sources"])


@unittest.skipUnless(_kmc_ready(), "KMC GCC toolchain not configured (N64DK_TOOLCHAIN)")
class StageLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="n64dk-agent-stage-")
        self.root = Path(self.tmp.name) / "proj"
        self.sizes = make_project(self.root)
        self.ws = tools.Workspace(self.root)
        self.fn = self.ws.lookup("main", "mix")

    def tearDown(self):
        self.tmp.cleanup()

    def run_with(self, steps, budget=None):
        return stage.run_stage(tools.Workspace(self.root), [self.fn], stage.DryRunProvider({"*": steps}),
                               budget or stage.Budget(attempts=1, seconds=600, tool_calls=20))

    def test_work_order_has_recipe_category_playbook_and_neighbours(self):
        order = stage.build_work_order(self.ws, self.fn, Path(self.tmp.name) / "order")
        text = order.render()
        self.assertIn("kmc+builtin", text)
        self.assertIn(order.category, stage.CATEGORIES)
        self.assertIn("Register priority rule", text)          # gcc family section
        self.assertIn("src/peek.c", text)                      # neighbour
        self.assertIn("./kit exact", text)
        if order.draft.get("files"):                          # m2c available: its draft must build as written
            self.assertEqual(order.draft["best_status"], "builds", order.draft)

    def test_exact_final_is_accepted_and_resume_skips_it(self):
        summary = self.run_with([{"write": "final.c", "content": TARGET}, {"tool": "exact", "files": ["final.c"]},
                                 {"write": "LESSON.md", "content": "multiply by a small constant stays a mult\n"}])
        self.assertEqual(summary["exact"], 1)
        done = self.ws.state / "stage" / "done" / f"{self.fn.key}.c"
        self.assertEqual(done.read_text(), TARGET)
        self.assertIn("multiply by a small constant", stage.lessons_path(self.ws).read_text())
        again = self.run_with([{"write": "final.c", "content": WRONG}])
        self.assertEqual((again["skipped"], again["sessions"]), (1, 0))

    def test_claimed_exact_near_miss_is_not_accepted_and_is_recorded(self):
        summary = self.run_with([{"write": "final.c", "content": WRONG}, {"tool": "try", "files": ["final.c"]},
                                 {"write": "NOTES.md", "content": "ruled out: b * 4 + b\nI am sure this is EXACT\n"}])
        self.assertEqual((summary["exact"], summary["near"]), (0, 1))
        dossier = json.loads((self.ws.state / "stage" / self.fn.key / "dossier.json").read_text())
        attempt = dossier["attempts"][0]
        self.assertEqual(attempt["status"], "near")
        self.assertGreater(attempt["hunks"], 0)
        self.assertTrue(any("DIFFERS" in line for line in attempt["tried"]))
        self.assertIn("ruled out", attempt["notes"])
        self.assertTrue((self.ws.state / "stage" / "near" / f"{self.fn.key}.c").is_file())
        # the next attempt's work order shows what was ruled out
        order = stage.build_work_order(self.ws, self.fn, Path(self.tmp.name) / "o2", dossier["attempts"])
        self.assertIn("ruled out: b * 4 + b", order.render())

    def test_budgets_are_respected_across_runs(self):
        budget = stage.Budget(attempts=2, seconds=600, tool_calls=2)
        steps = [{"write": "final.c", "content": WRONG}] + [{"tool": "try", "files": ["final.c"]}] * 4
        first = self.run_with(steps, budget)
        self.assertEqual(first["sessions"], 2)
        second = self.run_with(steps, budget)
        self.assertEqual(second["sessions"], 0)
        dossier = json.loads((self.ws.state / "stage" / self.fn.key / "dossier.json").read_text())
        self.assertEqual([a["calls"] for a in dossier["attempts"]], [2, 2])   # calls past 2 refused, not logged

    def test_assisted_result_is_not_counted_as_exact(self):
        summary = self.run_with([{"write": "fix.h", "content": "#define B() __asm__ volatile(\"\")\n"},
                                 {"write": "final.c", "content": '#include "fix.h"\n' + TARGET}])
        self.assertEqual((summary["exact"], summary["assisted"]), (0, 1))
        self.assertTrue((self.ws.state / "stage" / "assisted" / f"{self.fn.key}.c").is_file())

    def test_without_an_agent_nothing_is_spent(self):
        summary = stage.run_stage(self.ws, [self.fn], stage.NoAgentProvider(), stage.Budget(attempts=1))
        self.assertEqual(summary["sessions"], 0)
        self.assertIn("no agent configured", summary["no_agent"])

    def test_kit_wrapper_runs_the_tool_protocol(self):
        folder = Path(self.tmp.name) / "wrap"
        folder.mkdir()
        (folder / tools.SESSION).write_text(json.dumps(dict(project=str(self.root), image="main", key="mix",
                                                            vram=MIX_AT, size=self.sizes["mix"], tool_calls=5)))
        stage._wrapper(folder)
        out = subprocess.run(["./kit", "info"], cwd=folder, capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr[-500:])
        self.assertIn("recipe: kmc+builtin", out.stdout)


if __name__ == "__main__":
    unittest.main()
