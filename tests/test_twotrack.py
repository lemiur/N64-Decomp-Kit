"""twotrack: matching source and maintainable view side by side (round 12, item 11).  No
toolchain; the e-graph test needs the host preprocessor and is skipped without it."""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from n64dk.match import twotrack
from n64dk.match.transform_classes import SCAFFOLD_MARK

HAVE_CPP = shutil.which(os.environ.get("N64DK_CPP", "cpp")) is not None

SOURCE = """/* LDSYM D_80001234=0x80001234 */
typedef int s32;
extern s32 D_80001234;
extern s32 func_80002000(s32 a);

s32 func_80001000(s32 x) {
    s32 pad[2]; """ + SCAFFOLD_MARK + """
    s32 temp;

    temp = func_80002000(x) + D_80001234;
    return temp;
}
"""

NAMES = ("vram\tcanonical\tdisplay\tdisplay_from\toriginal\tsdk\tdonor\tsemantic\tuser\troles\n"
         "0x80001000\tfunc_80001000\tsem_pi_80001000\tsemantic\t\t\t\tsem_pi_80001000 [mmio]\t\tpi\n"
         "0x80002000\tfunc_80002000\tosRecvMesg\tsdk\t\tosRecvMesg [sdk signature]\t\t\t\t\n"
         "0x80004000\tfunc_80004000\tfunc_80004000\tcanonical\t\t\t\t\t\t\n")

TYPEFLOW = """/* typeflow: provisional declarations from machine-code constraints; conflict-free only */
struct T_80003000 {   /* typeflow: observed fields only; true size unknown */
    /* 0x00 */ s32 unk_0;
    /* 0x04 */ u16 unk_4;   /* signedness unproven */
};
extern struct T_80003000 D_80003000;
extern s32 D_80001234;
extern s32 D_80009999;
"""


class NamesTest(unittest.TestCase):
    def test_read_names_keeps_the_field_and_source(self):
        names = twotrack.read_names(NAMES)
        self.assertEqual(names["func_80002000"], {"display": "osRecvMesg", "from": "sdk", "source": "sdk signature"})
        self.assertEqual(names["func_80001000"]["from"], "semantic")
        self.assertEqual(names["func_80004000"]["from"], "canonical")

    def test_semantic_names_are_never_presented_as_original(self):
        track = twotrack.build(SOURCE, "func_80001000", names=twotrack.read_names(NAMES))
        self.assertIn("s32 sem_pi_80001000(s32 x)", track.view)
        self.assertIn("osRecvMesg(x)", track.view)
        semantic = [s for s in track.steps if s.kind == "names" and s.after.startswith("sem_")]
        self.assertEqual(len(semantic), 1)
        self.assertIn("GENERATED", semantic[0].detail)
        self.assertNotIn("original", semantic[0].detail.split("GENERATED")[0])

    def test_semantic_prefix_is_enforced_and_collisions_skipped(self):
        names = {"func_80001000": {"display": "dma_thing", "from": "semantic", "source": "mmio"},
                 "func_80002000": {"display": "temp", "from": "user", "source": "user"}}
        text, steps, notes = twotrack.rename(SOURCE, names)
        self.assertIn("sem_dma_thing", text)
        self.assertIn("func_80002000", text)                       # "temp" is already an identifier
        self.assertTrue(any("already an identifier" in n for n in notes))

    def test_comments_and_strings_are_not_renamed(self):
        source = 'void func_80001000(void) { /* func_80001000 */ const char *s = "func_80001000"; }\n'
        text, _steps, _notes = twotrack.rename(source, {"func_80001000": {"display": "osInit", "from": "sdk",
                                                                          "source": ""}})
        self.assertEqual(text.count("osInit"), 1)
        self.assertEqual(text.count("func_80001000"), 2)


class TracksTest(unittest.TestCase):
    def test_matching_source_is_never_changed(self):
        track = twotrack.build(SOURCE, "func_80001000", names=twotrack.read_names(NAMES), declarations=TYPEFLOW)
        self.assertEqual(track.matching, SOURCE)
        self.assertEqual(track.build_input(), SOURCE)
        self.assertNotEqual(track.view, SOURCE)

    def test_classes_of_each_step(self):
        track = twotrack.build(SOURCE, "func_80001000", names=twotrack.read_names(NAMES))
        by_kind = {}
        for step in track.steps:
            by_kind.setdefault(step.kind, set()).add(step.cls)
        self.assertEqual(by_kind["scaffolding"], {"behaviour-preserving", "unverified"})   # padding, LDSYM
        self.assertEqual(by_kind["names"], {"behaviour-preserving"})
        self.assertEqual(by_kind["comments"], {"presentation-only"})
        self.assertEqual(track.view_class, "unverified")
        self.assertNotIn("LDSYM", track.view.split("*/", 1)[1])
        self.assertNotIn(SCAFFOLD_MARK, track.view)
        self.assertIn("never this file", track.view)

    def test_presentation_only_when_nothing_else_applies(self):
        track = twotrack.build("int f(void) { return 1; }\n", "f")
        self.assertEqual(track.view_class, "presentation-only")
        self.assertTrue(track.view.endswith("int f(void) { return 1; }\n"))

    def test_structures_from_typeflow(self):
        source = SOURCE.replace("return temp;", "return temp + D_80003000.unk_0;")
        track = twotrack.build(source, "func_80001000", declarations=TYPEFLOW, scaffolding=False)
        view = track.view
        self.assertLess(view.index("struct T_80003000 {"), view.index("s32 func_80001000("))
        self.assertIn("extern struct T_80003000 D_80003000;", view)
        self.assertIn("/* typeflow suggests: extern s32 D_80001234; */", view)
        self.assertNotIn("D_80009999", view)                          # not used by the unit
        classes = {s.cls for s in track.steps if s.kind == "structs"}
        self.assertEqual(classes, {"unverified", "presentation-only"})

    def test_rewriter_is_optional_and_failures_are_notes(self):
        def rewriter(text, name):
            start = text.index("return temp;")
            return [(start, start + len("return temp;"), "return (temp);", ("fold",))]
        track = twotrack.build(SOURCE, "func_80001000", rewriter=rewriter)
        self.assertIn("return (temp);", track.view)
        self.assertEqual([s.cls for s in track.steps if s.kind == "egraph"], ["behaviour-preserving"])

        def broken(text, name):
            raise RuntimeError("no preprocessor")
        track = twotrack.build(SOURCE, "func_80001000", rewriter=broken)
        self.assertTrue(any("egraph: not applied" in n for n in track.notes))
        self.assertEqual(track.matching, SOURCE)

    def test_verify_reports_on_the_view_only(self):
        seen = []
        track = twotrack.build(SOURCE, "func_80001000", verify=lambda text: seen.append(text) or True)
        self.assertEqual(track.verified, "exact")
        self.assertEqual(seen, [track.view])
        self.assertEqual(twotrack.build(SOURCE, "func_80001000", verify=lambda text: False).verified, "different")

    def test_save_never_replaces_the_matching_source(self):
        folder = Path(tempfile.mkdtemp(prefix="n64dk-twotrack-"))
        self.addCleanup(shutil.rmtree, folder, True)
        track = twotrack.build(SOURCE, "func_80001000", names=twotrack.read_names(NAMES))
        paths = track.save(folder)
        self.assertEqual(Path(paths["matching"]).read_text(), SOURCE)
        self.assertNotEqual(paths["matching"], paths["view"])
        mapped = json.loads(Path(paths["map"]).read_text())
        self.assertEqual(mapped["schema"], twotrack.SCHEMA)
        self.assertEqual(mapped["class"], "unverified")
        other = twotrack.build(SOURCE.replace("x)", "y)"), "func_80001000")
        with self.assertRaises(twotrack.TwoTrackError):
            other.save(folder)
        self.assertEqual(Path(paths["matching"]).read_text(), SOURCE)


@unittest.skipUnless(HAVE_CPP, "host preprocessor missing")
class EgraphViewTest(unittest.TestCase):
    def test_statement_forms_are_normalised(self):
        source = "typedef int s32;\ns32 f(s32 x) {\n    x = x + 1;\n    return x;\n}\n"
        track = twotrack.build(source, "f", rewriter=twotrack.egraph_rewriter)
        body = track.view.split("*/", 1)[1]
        self.assertTrue(any(form in body for form in ("x++;", "++x;", "x += 1;")), body)
        self.assertEqual(track.matching, source)


if __name__ == "__main__":
    unittest.main()
