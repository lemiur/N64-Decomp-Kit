import shutil
import atexit
import os
import sys
import tempfile
import unittest
from pathlib import Path

from n64dk.match import custom_pipeline as cp


def _scratch(prefix: str = "n64dk-test-") -> str:
    """A scratch directory removed when the test process exits."""
    path = tempfile.mkdtemp(prefix=prefix)
    atexit.register(shutil.rmtree, path, True)
    return path



def project_with(text: str) -> Path:
    root = Path(_scratch("n64dk-custom-"))
    (root / "n64dk.toml").write_text(text)
    return root


FAKE = f'''
[pipelines.fake]
family = "ido"
default_flags = ["-O1"]
timeout = 5
steps = [
  "{sys.executable} {{tool}} {{source}} {{object}} {{flags}} {{includes}}",
  "{sys.executable} -c \\"import sys; open(sys.argv[1], 'ab').write(b'POST')\\" {{object}}",
]
[pipelines.fake.vars]
tool = "TOOLPATH"
'''
TOOL = "import sys\nopen(sys.argv[2], 'wb').write((' '.join(sys.argv[3:]) + '|' + open(sys.argv[1]).read()).encode())\n"


class CustomPipelineTest(unittest.TestCase):
    def setUp(self):
        self.tool = Path(_scratch()) / "tool.py"
        self.tool.write_text(TOOL)
        self.project = project_with(FAKE.replace("TOOLPATH", str(self.tool)))

    def test_recipe_is_loaded_and_identified(self):
        recipe = cp.recipe_for("custom:fake", self.project)
        self.assertEqual((recipe.family, recipe.default_flags, recipe.timeout), ("ido", ["-O1"], 5.0))
        self.assertEqual(cp.family_of("custom:fake", self.project), "ido")
        self.assertEqual(cp.family_of("ido7.1"), "ido")
        self.assertEqual(cp.family_of("kmc+asn64"), "gcc")
        self.assertIn(str(self.tool), recipe.digest_material())
        with self.assertRaisesRegex(cp.CustomPipelineError, "known: fake"):
            cp.recipe_for("custom:other", self.project)

    def test_steps_run_in_the_private_directory_with_placeholders(self):
        work = Path(_scratch())
        source = work / "unit.c"
        source.write_text("int x;")
        include = Path(_scratch())
        obj = cp.build(cp.recipe_for("custom:fake", self.project), source, ["-O2", "-DNAME=a b"], [include], work,
                       self.project)
        self.assertEqual(obj, work.resolve() / "unit.o")
        self.assertEqual(obj.read_bytes().decode(), f"-O2 -DNAME=a b -I{include.resolve()}|int x;POST")
        self.assertEqual(sorted(p.name for p in self.project.iterdir()), ["n64dk.toml"])    # nothing leaked

    def test_temporaries_go_to_the_private_directory(self):
        project = project_with(
            '[pipelines.tmp]\nfamily = "ido"\ntimeout = 5\nenv = { TMPDIR = "/tmp" }\nsteps = [\n'
            f'  "{sys.executable} -c \\"import os, sys; open(sys.argv[1], \'w\').write(os.environ[\'TMPDIR\'])\\" {{object}}",\n]\n')
        work = Path(_scratch())
        source = work / "unit.c"
        source.write_text("int x;")
        obj = cp.build(cp.recipe_for("custom:tmp", project), source, [], [], work, project)
        self.assertEqual(obj.read_text(), str(work.resolve()))     # never the shared /tmp

    def test_failures_are_reported_with_the_step(self):
        project = project_with('[pipelines.bad]\nsteps = ["%s -c \\"import sys; print(\'boom\'); sys.exit(3)\\""]\n'
                               % sys.executable)
        work = Path(_scratch())
        (work / "unit.c").write_text("")
        with self.assertRaisesRegex(cp.CustomPipelineError, r"step 1 failed .*exit 3\)[\s\S]*boom"):
            cp.build(cp.recipe_for("custom:bad", project), work / "unit.c", [], [], work, project)
        slow = project_with('[pipelines.slow]\ntimeout = 0.3\nsteps = ["%s -c \\"import time; time.sleep(5)\\""]\n'
                            % sys.executable)
        with self.assertRaisesRegex(cp.CustomPipelineError, "timed out"):
            cp.build(cp.recipe_for("custom:slow", slow), work / "unit.c", [], [], work, slow)
        missing = project_with('[pipelines.none]\nsteps = ["%s -c pass"]\n' % sys.executable)
        with self.assertRaisesRegex(cp.CustomPipelineError, "did not produce"):
            cp.build(cp.recipe_for("custom:none", missing), work / "unit.c", [], [], work, missing)

    def test_bad_configuration_is_rejected(self):
        with self.assertRaisesRegex(cp.CustomPipelineError, "non-empty list"):
            cp.load(project_with("[pipelines.x]\nsteps = []\n"))
        with self.assertRaisesRegex(cp.CustomPipelineError, "family"):
            cp.load(project_with('[pipelines.x]\nfamily = "msvc"\nsteps = ["true"]\n'))


@unittest.skipUnless(os.environ.get("N64DK_TOOLCHAIN") and
                     (Path(os.environ["N64DK_TOOLCHAIN"]) / "kmc-gcc-2.7.2" / "gcc").is_file(),
                     "KMC toolchain not configured")
class RealCompilerTest(unittest.TestCase):
    def test_a_declared_lane_reproduces_the_built_in_kmc_lane(self):
        from n64dk.elf import Elf
        from n64dk.match import compare as cmp
        from n64dk.match.pipeline import build, split_flags
        from n64dk.match.toolchain import load_config
        kmc = Path(os.environ["N64DK_TOOLCHAIN"]) / "kmc-gcc-2.7.2"
        project = project_with(f'''
[pipelines.kmc-declared]
steps = [
  "{{kmc}}/gcc -B{{kmc}}/ -S {{flags}} {{includes}} -o {{workdir}}/{{stem}}.s {{source}}",
  "{{kmc}}/as -mips3 -G0 -o {{object}} {{workdir}}/{{stem}}.s",
]
[pipelines.kmc-declared.vars]
kmc = "{kmc}"
''')
        flags = ["-O2", "-G0", "-mips3", "-mgp32", "-mfp32"]
        text = "int add(int a, int b) { if (a) { return a + b; } return b * 3; }\n"
        work = Path(_scratch())
        (work / "unit.c").write_text(text)
        custom = cp.build(cp.recipe_for("custom:kmc-declared", project), work / "unit.c", flags, [], work, project)
        other = Path(_scratch())
        (other / "unit.c").write_text(text)
        builtin = build(other / "unit.c", split_flags(flags, "kmc+as"), load_config(project, None, None), other)
        self.assertEqual(cmp.find_section(Elf.load(custom), ".text").data,
                         cmp.find_section(Elf.load(builtin), ".text").data)


    def test_the_kit_build_function_runs_a_declared_lane_and_caches_it(self):
        from n64dk.elf import Elf
        from n64dk.match import compare as cmp
        from n64dk.match.pipeline import build, object_cache_stats, split_flags
        from n64dk.match.toolchain import load_config
        kmc = Path(os.environ["N64DK_TOOLCHAIN"]) / "kmc-gcc-2.7.2"
        project = project_with(f'''
[pipelines.kmc-declared]
steps = [
  "{{kmc}}/gcc -B{{kmc}}/ -S {{flags}} -o {{workdir}}/{{stem}}.s {{source}}",
  "{{kmc}}/as -mips3 -G0 -o {{object}} {{workdir}}/{{stem}}.s",
]
[pipelines.kmc-declared.vars]
kmc = "{kmc}"
''')
        (project / "include").mkdir()
        (project / "include" / "k.h").write_text("#define K 7\n")
        source = project / "src.c"
        source.write_text('#include "k.h"\nint f(int a) { return a * K; }\n')
        cfg = load_config(project, None, None)
        spec = split_flags(["-O2", "-G0", "-mips3", "-mgp32", "-mfp32"], "custom:kmc-declared")
        before = object_cache_stats()["hits"]
        first = cmp.find_section(Elf.load(build(source, spec, cfg, Path(_scratch()))), ".text").data
        again = cmp.find_section(Elf.load(build(source, spec, cfg, Path(_scratch()))), ".text").data
        self.assertEqual(first, again)
        self.assertGreater(object_cache_stats()["hits"], before)
        reference = cmp.find_section(Elf.load(build(source, split_flags(spec.flags, "kmc+as"), cfg,
                                                    Path(_scratch()))), ".text").data
        self.assertEqual(first, reference)
        self.assertFalse([p.name for p in project.iterdir() if p.suffix in (".s", ".o")])


if __name__ == "__main__":
    unittest.main()
