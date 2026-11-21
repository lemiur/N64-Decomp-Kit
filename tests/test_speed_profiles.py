"""Speed changes from the round 12 base-bench profiles (W4_03): each is the same result, sooner."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from n64dk.match import asn64, iterate, normalize, pipeline

SOURCE = """#include "types.h"

extern s16 g(s16 index);

s32 f(u16 start) {
    s32 found;
    u16 offset;
    s16 index;

    index = g(start + 1);
    offset = 0;
    found = 0;
    while (found < 5) {
        if (*(s16 *)(index + offset) != -1) {
            found = (u16)found + 1;
        }
        offset++;
    }
    return found;
}
"""


def _files(folder: Path, texts) -> tuple[Path, ...]:
    out = []
    for index, text in enumerate(texts):
        path = folder / f"part{index}.py"
        path.write_text(text)
        out.append(path)
    return tuple(out)


class PipelineCodeDigestTest(unittest.TestCase):
    def test_same_value_as_hashing_the_build_code_on_every_build(self):
        # the object identity (and so every object-cache key) is unchanged
        expected = hashlib.sha256(Path(pipeline.__file__).read_bytes() + Path(asn64.__file__).read_bytes()
                                  + Path(normalize.__file__).read_bytes()).hexdigest()
        self.assertEqual(pipeline.pipeline_code_digest(), expected)
        self.assertEqual(pipeline.pipeline_code_digest(), expected)

    def test_unchanged_files_are_not_read_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = _files(Path(tmp), ["a = 1\n", "b = 2\n", "c = 3\n"])
            first = pipeline.pipeline_code_digest(files)
            self.assertEqual(first, hashlib.sha256(b"a = 1\nb = 2\nc = 3\n").hexdigest())
            with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("read again")):
                self.assertEqual(pipeline.pipeline_code_digest(files), first)

    def test_a_changed_file_changes_the_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = _files(Path(tmp), ["a = 1\n", "b = 2\n", "c = 3\n"])
            first = pipeline.pipeline_code_digest(files)
            files[1].write_text("b = 22\n")                     # another size: a new signature
            second = pipeline.pipeline_code_digest(files)
            self.assertNotEqual(first, second)
            self.assertEqual(second, hashlib.sha256(b"a = 1\nb = 22\nc = 3\n").hexdigest())

    def test_cached_tool_digest_is_the_hex_digest_of_the_bytes(self):
        # the normalizer hook's identity now goes through _file_digest: same hex value as before
        with tempfile.TemporaryDirectory() as tmp:
            (hook,) = _files(Path(tmp), ["#!/bin/sh\n"])
            self.assertEqual(pipeline._file_digest(hook).hex(), hashlib.sha256(b"#!/bin/sh\n").hexdigest())


class DescendFloorTest(unittest.TestCase):
    def test_search_from_the_lowest_rank_builds_only_its_start(self):
        calls = []

        def evaluate(text):
            calls.append(text)
            # byte-identical but not strictly proved; some variants would even be "exact" at the
            # same rank, which a strictly-lower-rank rule could never adopt
            return iterate.Evaluation(0, text != SOURCE and "s16 offset;" in text, {"counts": {}, "entries": []})
        result = iterate.descend(SOURCE, "f", evaluate, jobs=1)
        self.assertEqual(calls, [SOURCE])
        self.assertEqual((result["start_score"], result["final_score"], result["exact"]), (0, 0, False))
        self.assertEqual((result["evaluated"], result["source"], result["sensitivity"]), (1, SOURCE, []))
        self.assertEqual(result["stopped_reason"], "floor")

    def test_resolve_sees_no_improvement_and_no_helpful_family(self):
        # the fields resolve reads: the same as a full search from the floor, which can only keep
        # its start (nothing ranks below it) and has no negative sensitivity delta
        def evaluate(text):
            return iterate.Evaluation(0, False, {"counts": {}, "entries": []})
        result = iterate.descend(SOURCE, "f", evaluate, jobs=1)
        improved = result["final_score"] is not None and result["final_score"] < 0
        helpful = {row["family"] for row in result["sensitivity"] if row.get("delta", 0) < 0}
        self.assertFalse(improved)
        self.assertEqual(helpful, set())
        self.assertEqual(result["source"], SOURCE)

    def test_a_finer_rank_above_the_floor_still_searches(self):
        # match-iterate's evaluator ranks a byte-identical, unproved start at 1: a proved variant
        # (rank 0) is still found
        def evaluate(text):
            proved = "s16 offset;" in text
            return iterate.Evaluation(0, proved, {"counts": {}, "entries": []}, rank_score=0 if proved else 1)
        result = iterate.descend(SOURCE, "f", evaluate, jobs=1)
        self.assertTrue(result["exact"])
        self.assertGreater(result["evaluated"], 1)
        self.assertEqual(result["stopped_reason"], "exact")

    def test_ordinary_search_unchanged(self):
        def evaluate(text):
            score = (0 if "s16 offset;" in text else 10) + \
                (0 if text.index("found = 0;") < text.index("index = g(") else 7)
            return iterate.Evaluation(score, score == 0, {"counts": {}, "entries": []})
        result = iterate.descend(SOURCE, "f", evaluate, jobs=1)
        self.assertTrue(result["exact"])
        self.assertEqual((result["start_score"], result["final_score"]), (17, 0))


if __name__ == "__main__":
    unittest.main()
