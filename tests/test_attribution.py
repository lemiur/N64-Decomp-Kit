"""Ablation attribution (n64dk.match.attribution) on hand-written run folders: FUNCTIONS.tsv,
events.jsonl, profile.jsonl, report.json and heavy.sh post records, laid out as bench/ablation/plan.sh
lays them out.  No run, no toolchain."""
import json
import tempfile
import unittest
from pathlib import Path

from n64dk.match import attribution as at

HEAD = "image\tfunction\tvram\trom_offset\tsize\tcode_size\toutcome\thow\tsource\tdistance\treproduced_by\n"
GP_HOW = "kmc+gas -O2 -G8 (draft; needs the recovered _gp in symbols/gp_symbols.txt)"


def row(fn: str, vram: int, outcome: str, how: str = "kmc+as -O2", distance: str = "") -> str:
    return f"main\t{fn}\t0x{vram:08X}\t0x{vram - 0x80000000:X}\t0x10\t0x10\t{outcome}\t{how}\t\t{distance}\t\n"


def base_rows() -> dict:
    """The all-on run: function name -> row."""
    out = {
        "f1": ("exact", GP_HOW, ""), "f2": ("donor", "donor unit sdk: x.c", ""),
        "f3": ("exact", "kmc+as -O2 (draft after variant search)", ""), "f4": ("exact", "kmc+as -O2 (draft)", ""),
        "f5": ("exact", "kmc+as -O2 (draft)", ""), "f6": ("exact", "kmc+as -O2 (draft)", ""),
        "f7": ("open", "kmc+as -O2", "9"), "f8": ("near", "kmc+as -O2", "4"), "f9": ("near", "kmc+as -O2", "5"),
        "f10": ("open", "kmc+as -O2", "20"), "f11": ("exact", "kmc+as -O2 (source of an identical function)", ""),
        "f12": ("not attempted", "", ""),
    }
    return out


VRAM = {f"f{n}": 0x80000400 + 0x10 * (n - 1) for n in range(1, 13)}
VARIANTS = {   # variant: (subsystem, kind, changes to the base rows, builds, job seconds, events)
    "all": ("all", "base", {}, 1000, 60, {"f1": 30, "f4": 12}),
    "all-repeat": ("all", "repeat", {"f5": ("open", "kmc+as -O2", "3")}, 1010, 61, {}),
    "-small-data": ("small-data", "minus", {"f1": ("open", "kmc+as -O2", "6"), "f4": ("open", "kmc+as -O2", "2"),
                                            "f6": None, "f7": ("exact", "kmc+as -O2 (draft)", ""),
                                            "f8": ("does-not-build", "kmc+as -O2", ""),
                                            "f9": ("near", "kmc+as -O2", "12"),
                                            "f11": ("exact", "kmc+as -O2 (draft)", "")}, 800, 50, {}),
    "-donors": ("donors", "minus", {"f2": ("open", "kmc+as -O2", "30"), "f4": ("near", "kmc+as -O2", "1")},
                900, 55, {}),
    "+egraph": ("egraph", "plus", {"f10": ("exact", "kmc+as -O2 (draft after variant search)", "")}, 1300, 70, {}),
}


def make_title(root: Path, incomplete: str | None = None) -> Path:
    lines = ["variant\tsubsystem\tkind\tflags\tenv"]
    for variant, (subsystem, kind, changes, builds, seconds, events) in VARIANTS.items():
        lines.append(f"{variant}\t{subsystem}\t{kind}\t--flag\t")
        rows = base_rows()
        rows.update(changes)
        export, run = root / variant / "export", root / variant / "run"
        export.mkdir(parents=True)
        run.mkdir()
        (export / "FUNCTIONS.tsv").write_text(HEAD + "".join(
            row(name, VRAM[name], *value) for name, value in rows.items() if value is not None))
        (run / "profile.jsonl").write_text(
            json.dumps({"stage": "discovery", "image": "*", "wall_s": 0.5, "builds": {"requested": 0}}) + "\n" +
            json.dumps({"stage": "resolve", "image": "main", "wall_s": 9.5, "builds": {"requested": builds}}) + "\n")
        (run / "events.jsonl").write_text("".join(
            json.dumps({"event": "case-open", "case": f"main:{VRAM[name]:08X}:10", "builds": 1}) + "\n" +
            json.dumps({"event": "case-exact", "case": f"main:{VRAM[name]:08X}:10", "builds": count}) + "\n"
            for name, count in events.items()) + "not json\n")
        status = "incomplete" if variant == incomplete else "complete"
        (run / "report.json").write_text(json.dumps({"status": status, "failed_stages": {}}))
        (root / variant / "job.run.post.json").write_text(json.dumps({"exit": 0, "seconds": seconds}))
    (root / "variants.tsv").write_text("\n".join(lines) + "\n")
    return root


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def title(self, incomplete=None) -> list:
        return at.load_variants(make_title(self.root / "t", incomplete))


class RecordTest(Fixture):
    def test_stage_lane_and_path_of_a_row(self):
        gp = {"outcome": "exact", "how": GP_HOW}
        self.assertEqual((at.settling_stage(gp), at.lane(gp), at.path(gp)), ("draft", "small-data", "draft@small-data"))
        self.assertEqual(at.settling_stage({"outcome": "donor", "how": "donor unit sdk: x.c"}), "donors")
        self.assertEqual(at.settling_stage({"outcome": "assembly", "how": "gnu-as -march=vr4300"}),
                         "handwritten-assembly")
        self.assertEqual(at.settling_stage({"outcome": "exact", "how": "kmc+as -O2 (external author, strictly "
                                                                       "verified)"}), "refine")
        self.assertEqual(at.path_subsystems(gp), {"small-data"})
        self.assertEqual(at.distance({"distance": " 7 "}), 7)
        self.assertIsNone(at.distance({"distance": ""}))

    def test_events_profile_and_job_are_read(self):
        runs = {run.label: run for run in self.title()}
        base = runs["all"]
        self.assertEqual(at._key_of_case("main:8000DDC4:34"), ("main", "0x8000DDC4"))
        self.assertEqual(base.case_builds, {("main", "0x80000400"): 30, ("main", "0x80000430"): 12})
        self.assertEqual((base.builds, base.wall_s), (1000, 60.0))      # the job's seconds win over the stages'
        self.assertEqual(base.stages["resolve"], {"builds": 1000, "wall_s": 9.5})
        self.assertTrue(base.complete)
        self.assertEqual(len(runs), 5)

    def test_names_restrict_the_universe(self):
        names_file = self.root / "names.txt"
        names_file.write_text("f1\nf2\n# a comment\nf3, f4\n\n")
        names = at.read_names(names_file)
        self.assertEqual(names, {"f1", "f2", "f3", "f4"})
        base = next(run for run in self.title() if run.kind == "base")
        small = at.restrict(base, names)
        self.assertEqual(sorted(row["function"] for row in small.rows.values()), ["f1", "f2", "f3", "f4"])
        self.assertEqual(set(small.case_builds), {("main", "0x80000400"), ("main", "0x80000430")})
        self.assertIs(at.restrict(base, None), base)


class AttributionTest(Fixture):
    def test_the_partition_never_double_counts(self):
        result = at.attribute(self.title())
        self.assertEqual(result["coverage"]["settled"], 7)
        self.assertEqual(result["partition"], {"denominator": 1, "joint": 1, "noise": 1, "redundant": 2,
                                               "unique:donors": 1, "unique:small-data": 1})
        self.assertEqual(sum(result["partition"].values()), result["coverage"]["settled"])
        self.assertEqual(result["first_stage"], {"draft": 4, "donors": 1, "search": 1, "twin": 1})
        self.assertEqual(sum(result["first_stage"].values()), 7)
        # (b): f11 settles by its twin in the base and by its draft once small data is off
        self.assertEqual(result["capable_paths"]["functions_by_number_of_paths"], {1: 6, 2: 1})
        # (c) per subsystem: counterfactual, f4 counts for both, f6 (absent) and f5 (noise) for neither
        self.assertEqual(result["marginal_not_additive"], {"small-data": 2, "donors": 2})
        self.assertEqual(result["joint_overlap"], {"donors & small-data": 1})
        self.assertEqual(result["noise"], {"outcome_differs": 1, "settled_differs": 1})
        self.assertEqual(result["coverage"]["attempted"], 11)

    def test_a_minus_comparison(self):
        pairs = {pair["variant"]: pair for pair in at.attribute(self.title())["pairs"]}
        small = pairs["-small-data"]
        self.assertEqual((small["added"], small["regressions"], small["net"]), (2, 1, 1))
        self.assertEqual((small["made_buildable"], small["made_closer"], small["made_farther"]), (1, 1, 0))
        self.assertEqual(small["denominator"], {"only_with": 1, "only_without": 0, "attempted_only_with": 0,
                                                "attempted_only_without": 0})
        self.assertEqual((small["builds_added"], small["builds_saved"], small["wall_s_added"]), (200, 0, 10.0))
        self.assertEqual(small["builds_on_added"], 42)
        self.assertEqual(small["stage_builds_added"], {"resolve": 200})
        self.assertEqual(small["functions"]["regressions"], ["main:f7"])
        # regress.compare's own view: it counts f6 (absent without the subsystem) as a gain
        self.assertEqual((small["regress"]["strict_before"], small["regress"]["strict_after"]), (5, 7))
        self.assertEqual((small["regress"]["gained"], small["regress"]["lost"]), (3, 1))
        self.assertEqual(small["regress"]["denominator"]["main"]["only_after"], 1)

    def test_a_plus_comparison_and_noise(self):
        pairs = {pair["variant"]: pair for pair in at.attribute(self.title())["pairs"]}
        plus = pairs["+egraph"]
        self.assertEqual((plus["kind"], plus["added"], plus["regressions"], plus["builds_added"]), ("plus", 1, 0, 300))
        donors = pairs["-donors"]
        self.assertEqual((donors["added"], donors["regressions"], donors["builds_added"]), (2, 0, 100))
        self.assertEqual(donors["within_noise"], 0)

    def test_a_settled_change_inside_the_noise_is_not_attributed(self):
        runs = self.title()
        for run in runs:
            if run.label == "-donors":
                run.rows[("main", "0x80000440")] = dict(run.rows[("main", "0x80000440")], outcome="open")
        pairs = {pair["variant"]: pair for pair in at.attribute(runs)["pairs"]}
        self.assertEqual((pairs["-donors"]["added"], pairs["-donors"]["within_noise"]), (2, 1))

    def test_one_base_is_required(self):
        runs = self.title()
        for run in runs:
            run.kind = "base"
        with self.assertRaises(ValueError):
            at.attribute(runs)


class ReportTest(Fixture):
    def test_summary_over_titles_and_markdown(self):
        one = at.attribute(self.title())
        two = at.attribute(at.load_variants(make_title(self.root / "u", incomplete="-donors")))
        total = at.summary({"t": one, "u": two})
        small = total["subsystems"]["small-data"]
        self.assertEqual((small["added"], small["regressions"], small["titles_affected"]), (4, 2, 2))
        self.assertEqual(total["subsystems"]["donors"]["incomplete"], ["u"])
        self.assertEqual(total["settled"], 14)
        self.assertEqual(sum(total["partition"].values()), 14)
        text = at.markdown({"t": one, "u": two}, total)
        self.assertIn("| small-data | minus | 4 | 2 | 2 |", text)
        self.assertIn("donors (incomplete: u)", text)
        self.assertIn("## Exclusive attribution of 14 settled functions", text)
        json.dumps({"t": one, "total": total})                       # the report is plain JSON


if __name__ == "__main__":
    unittest.main()
