"""review: queue, show and durable review transactions over a synthetic run (round 12, item 12).
No toolchain: the verifier and the differ are injected."""
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from n64dk.match import review, twotrack
from n64dk.match.resolve import Case, CaseStore

SOURCE_A = "typedef int s32;\nextern s32 D_1;\ns32 func_80001000(s32 x) {\n    x = x + D_1;\n    return x;\n}\n"
SOURCE_B = "typedef int s32;\ns32 func_80001100(s32 x) {\n    return x * 3;\n}\n"
MACHINE = "/srv/build/title"          # a synthetic absolute path: must never reach the page


def make_run(root: Path) -> Path:
    """A finished run in miniature: case store with sources, report.json, queue tables, profile,
    discovery audit, toolchain regions and one receipt."""
    run = root / "title_base" / "run"
    store = CaseStore(run / "state")
    try:
        a, b = store.save_source(SOURCE_A), store.save_source(SOURCE_B)
        store.put(Case("main:80001000:40", "main", "func_80001000", 0x80001000, 0x40, "t" * 64, status="assumed",
                       best_score=0, best_digest=a, recipe="kmc+as -O2 -G0",
                       stages=[{"stage": "iterate", "helpful_families": ["ast-expr"]},
                               {"stage": "oracle", "result": {"counterexample": {"a0": 7, "target": 1, "candidate": 2}}}],
                       proof={"verdict": "assumed", "assumptions": [{"kind": "symbol", "name": "D_1"}]}))
        store.put(Case("main:80001100:20", "main", "func_80001100", 0x80001100, 0x20, "u" * 64, status="near",
                       best_score=3, best_digest=b, recipe="kmc+as -O2 -G0", top_cause="register.lifetime",
                       stages=[{"stage": "diagnose", "causes": ["register.lifetime"]}]))
        store.put(Case("main:80001200:30", "main", "func_80001200", 0x80001200, 0x30, "v" * 64, status="near",
                       best_score=9, best_digest=b, recipe="kmc+as -O2 -G0"))
        store.put(Case("main:80001300:10", "main", "func_80001300", 0x80001300, 0x10, "w" * 64, status="open",
                       best_score=80, best_digest=b))
        store.put(Case("main:80001400:10", "main", "func_80001400", 0x80001400, 0x10, "x" * 64, status="exact",
                       best_score=0, best_digest=b))
    finally:
        store.close()
    report = {"project": MACHINE, "status": "complete", "seconds": 12.5, "failed_stages": {},
              "images": {"main": {
                  "functions": 40,
                  "donors": {"functions_recovered": 20, "functions_awaiting_symbols": 1,
                             "verified": [{"functions": ["func_80003000"], "vram": "0x80003000", "size": 16,
                                           "donor": "OtherTitle", "unit": "src/a.c", "source_kind": "c",
                                           "awaiting": {"D_9": "0x80009000"}},
                                          {"functions": ["func_80003100", "func_80003110"], "vram": "0x80003100",
                                           "size": 64, "donor": "OtherTitle", "unit": "src/b.c", "source_kind": "c"}]},
                  "resolve": {"statuses": {"assumed": 1, "near": 2, "open": 1, "exact": 1}, "builds": 120,
                              "seconds": 4.0, "exact_bytes": 16},
                  "queue": {"strict_exact": 1, "assumed": 1, "assembly": 3},
                  "toolchain_regions": {"regions": 1, "verdicts": {"known": 1}, "functions_by_verdict": {"known": 40},
                                        "file": MACHINE + "/run/main/toolchain_regions.json"},
                  "hard_queue": [{"cause": "register.lifetime", "functions": 1, "bytes": 32}],
                  "solver": {"remaining_by_cause": {"register-only residual": {"functions": 1, "bytes": 32,
                                                                               "next": {"search": 1}}},
                             "sieve": {"stages": {"3 SDK, library and known-source donors": {"settled": 20, "builds": 0},
                                                  "8 control-flow and expression forms": {"settled": 1, "builds": 90}}},
                             "coverage": {"functions": 40, "attempted": 5, "not_attempted": 22,
                                          "reasons": {"recovered from a donor (proven source of another title)": 21,
                                                      "outside the size limit": 1}}},
                  "ledger": {"totals": {"extent": "0x80001000..0x80002000", "extent_bytes": 4096,
                                        "bytes": {"c-verified": 1000, "asm-proven": 300, "function-unmatched": 2796},
                                        "sum_equals_extent": True}, "violations": {}}}},
              "resources": {"jobs": 2, "max_jobs": 2, "limited_by": ["safety-ceiling"],
                            "memory": {"total_mb": 8000, "available_mb": 5000, "swap_used_mb": 0}},
              "profile": {"resolve": {"wall_s": 4.0}}}
    (run / "report.json").write_text(json.dumps(report))
    (run / "queue" / "main").mkdir(parents=True)
    (run / "queue" / "assumptions.tsv").write_text("# image\tunit\tkind\tname\taddress\n"
                                                   "main\tfunc_80001000\tsymbol\tD_1\t0x80005000\n")
    (run / "queue" / "main" / "func_80001600.s").write_text("glabel func_80001600\n")
    (run / "profile.jsonl").write_text(json.dumps({"stage": "resolve", "image": "main", "wall_s": 4.0, "child_cpu_s": 9.0,
                                                   "python_cpu_s": 1.0, "peak_rss_mb": 210.0,
                                                   "builds": {"requested": 120, "cache_hits": 30, "cache_misses": 90,
                                                              "distinct_objects": 88}}) + "\n")
    (run / "discovery").mkdir()
    (run / "discovery" / "discovery_audit.json").write_text(json.dumps(
        {"images": {"main": {"vram": "0x80001000", "end": "0x80002000", "bytes": 4096, "coverage": 1.0,
                             "verdict": "complete", "direct_calls": {"targets": 12, "missing": 0, "interior": 0,
                                                                      "cross_image": 2, "unmapped": 0},
                             "pointers": {"to_uncatalogued_code": 0}, "islands": {"count": 0}}}}))
    (run / "main").mkdir()
    (run / "main" / "toolchain_regions.json").write_text(json.dumps(
        {"regions": [{"start": "0x80001000", "end": "0x80002000", "functions": 40, "bytes": 4096, "lane": "kmc+as",
                      "verdict": "known", "ranked": [{"recipe": "kmc+as -O2", "share": 0.9}]}]}))
    receipt = run / "receipts" / "main" / "func_80001000-80001000.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"schema": "n64dk-proof-receipt/1", "identity": "abc", "verdict": {"class": "assumed"},
                                   "source": {"sha256": "s"}, "claims": {"byte_identity": True}}))
    return run


def snapshot(run: Path, skip: str = "review") -> dict:
    out = {}
    for path in sorted(run.rglob("*")):
        rel = path.relative_to(run)
        if path.is_file() and rel.parts[0] != skip and not path.name.startswith("cases.sqlite-"):
            out[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="n64dk-review-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.run_dir = make_run(self.root)
        self.clock = iter(f"2026-11-17T00:00:{n:02d}+00:00" for n in range(60)).__next__


class QueueTest(Base):
    def test_order_and_kinds(self):
        items = review.queue(self.run_dir)
        self.assertEqual([i.kind for i in items], ["assumed", "near", "near", "donor"])
        self.assertEqual([i.id for i in items][:3], ["main:80001000:40", "main:80001100:20", "main:80001200:30"])
        self.assertEqual(items[3].id, "donor:main:80003000")
        self.assertEqual(items[3].detail["awaiting"], {"D_9": "0x80009000"})

    def test_decided_items_leave_the_queue(self):
        review.decide(self.run_dir, "main:80001100:20", "reject", payload={"reason": "wrong boundary"}, clock=self.clock)
        ids = [i.id for i in review.queue(self.run_dir)]
        self.assertNotIn("main:80001100:20", ids)
        self.assertIn("main:80001100:20", [i.id for i in review.queue(self.run_dir, include_decided=True)])


class ShowTest(Base):
    def test_everything_a_reviewer_needs(self):
        shown = review.show(self.run_dir, "func_80001000", differ=lambda case, text: ["0x0000 register ..."],
                            maintainable=lambda text, case: twotrack.build(text, case.name))
        self.assertEqual(shown["source"], SOURCE_A)
        self.assertEqual(shown["recipe"], "kmc+as -O2 -G0")
        self.assertEqual(shown["diff"], ["0x0000 register ..."])
        self.assertEqual(len(shown["assumptions"]), 2)
        self.assertEqual(shown["transform_classes"], {"ast-expr": "conditional"})
        self.assertEqual(shown["provenance"]["class"], "conditional")
        self.assertEqual(shown["receipt"]["identity"], "abc")
        self.assertTrue(any("counterexample" in str(b) for b in shown["behaviour"]))
        self.assertEqual(shown["maintainable"]["class"], "presentation-only")
        self.assertEqual(shown["proof_verdict"], "assumed")

    def test_failing_differ_is_shown_not_raised(self):
        def differ(case, text):
            raise RuntimeError("no toolchain")
        shown = review.show(self.run_dir, "main:80001100:20", differ=differ)
        self.assertIn("diff unavailable", shown["diff"][0])

    def test_donor_item_and_unknown_reference(self):
        shown = review.show(self.run_dir, "func_80003000")
        self.assertEqual(shown["item"]["kind"], "donor")
        with self.assertRaises(review.ReviewError):
            review.show(self.run_dir, "func_DEADBEEF")


class DecideTest(Base):
    def test_accept_without_verification_is_only_a_request(self):
        record = review.decide(self.run_dir, "main:80001100:20", "accept-exact", clock=self.clock)
        self.assertEqual(record["status"], "requested")
        self.assertIn("main:80001100:20", [i.id for i in review.queue(self.run_dir)])

    def test_accept_with_strict_verification(self):
        calls = []

        def verify(case, text):
            calls.append((case.case_id, text))
            return {"verdict": "exact"}
        record = review.decide(self.run_dir, "main:80001100:20", "accept-exact", verify=verify, clock=self.clock)
        self.assertEqual(record["status"], "verified")
        self.assertEqual(calls, [("main:80001100:20", SOURCE_B)])
        record = review.decide(self.run_dir, "main:80001200:30", "accept-exact",
                               verify=lambda case, text: {"verdict": "different"}, clock=self.clock)
        self.assertEqual(record["status"], "requested")
        state = review.replay(self.run_dir)["cases"]
        self.assertEqual(state["main:80001100:20"]["decision"], "accept-exact")
        self.assertIsNone(state["main:80001200:30"]["decision"])

    def test_retain_assumed_only_for_assumed(self):
        self.assertEqual(review.decide(self.run_dir, "main:80001100:20", "retain-assumed", clock=self.clock)["status"],
                         "invalid")
        self.assertEqual(review.decide(self.run_dir, "main:80001000:40", "retain-assumed", clock=self.clock)["status"],
                         "recorded")

    def test_payload_validation(self):
        with self.assertRaises(review.ReviewError):
            review.decide(self.run_dir, "main:80001100:20", "reject")
        with self.assertRaises(review.ReviewError):
            review.decide(self.run_dir, "main:80001100:20", "correct-boundary", payload={"vram": "x", "size": "4"})
        with self.assertRaises(review.ReviewError):
            review.decide(self.run_dir, "main:80001100:20", "approve")
        record = review.decide(self.run_dir, "main:80001100:20", "correct-boundary",
                               payload={"vram": "0x80001100", "size": "0x1C"}, clock=self.clock)
        self.assertEqual(record["payload"], {"vram": "0x80001100", "size": 28})

    def test_all_actions_replay(self):
        run = self.run_dir
        review.decide(run, "main:80001100:20", "request-stage", payload={"stage": "recipe-sweep"}, clock=self.clock)
        review.decide(run, "main:80001100:20", "correct-type", payload={"symbol": "D_1", "type": "u16"}, clock=self.clock)
        review.decide(run, "main:80001100:20", "add-evidence", payload={"kind": "debug", "detail": "x"}, clock=self.clock)
        review.decide(run, "main:80001200:30", "mark-handwritten", clock=self.clock)
        review.decide(run, "donor:main:80003000", "defer", note="wait for symbols", clock=self.clock)
        state = review.replay(run)
        self.assertEqual(state["transactions"], 5)
        self.assertEqual(state["problems"], [])
        self.assertEqual(state["cases"]["main:80001100:20"]["requested_stages"], ["recipe-sweep"])
        self.assertEqual(state["cases"]["main:80001100:20"]["corrections"][0]["type"], "u16")
        self.assertEqual(state["cases"]["main:80001200:30"]["decision"], "mark-handwritten")

    def test_hash_chain_detects_tampering_and_truncation(self):
        review.decide(self.run_dir, "main:80001100:20", "defer", clock=self.clock)
        review.decide(self.run_dir, "main:80001200:30", "defer", clock=self.clock)
        path = review.log_path(self.run_dir)
        lines = path.read_text().splitlines()
        record = json.loads(lines[0])
        record["status"] = "verified"
        path.write_text(json.dumps(record, sort_keys=True) + "\n" + lines[1] + "\n")
        self.assertTrue(review.replay(self.run_dir)["problems"])
        with self.assertRaises(review.ReviewError):
            review.decide(self.run_dir, "main:80001100:20", "defer", clock=self.clock)
        path.write_text(lines[0] + "\n" + lines[1][:20])
        state = review.replay(self.run_dir)
        self.assertEqual(state["transactions"], 1)
        self.assertIn("not JSON", state["problems"][0])

    def test_deterministic_and_never_touches_the_run_or_project(self):
        before = snapshot(self.run_dir)
        review.decide(self.run_dir, "main:80001100:20", "export", clock=self.clock)
        review.decide(self.run_dir, "main:80001100:20", "reject", payload={"reason": "r"}, clock=self.clock)
        self.assertEqual(snapshot(self.run_dir), before)
        exported = self.run_dir / "review" / "exports" / "main_80001100_20"
        self.assertEqual((exported / "func_80001100.c").read_text(), SOURCE_B)
        first = review.log_path(self.run_dir).read_text()
        other = make_run(self.root / "again")
        clock = iter(f"2026-11-17T00:00:{n:02d}+00:00" for n in range(60)).__next__
        review.decide(other, "main:80001100:20", "export", clock=clock)
        review.decide(other, "main:80001100:20", "reject", payload={"reason": "r"}, clock=clock)
        self.assertEqual(review.log_path(other).read_text(), first)


if __name__ == "__main__":
    unittest.main()
