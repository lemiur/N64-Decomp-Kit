"""table_ranges' parallel-array search: the round-12 pruning gives exactly the old answer."""
import random
import struct
import unittest

from n64dk.assets import loaders


def reference_pairs(block, form, stream_starts):
    """The pre-round-12 search, verbatim in effect: first (half, lead) by half descending, lead ascending."""
    for half in range(len(block) // 2, 1, -1):
        for lead in range(0, len(block) - 2 * half + 1):
            st, en = block[lead:lead + half], block[lead + half:lead + 2 * half]
            spans = sorted(zip(st, en))
            if all(0 < y - x <= loaders.MAX_RANGE for x, y in spans) and \
                    all(a[1] <= b[0] for a, b in zip(spans, spans[1:])) and \
                    (form == "pi" or sum(x in stream_starts for x in st) >= 2):
                return [(x, y) for x, y in zip(st, en)]
    return []


class ParallelArrayTest(unittest.TestCase):
    def test_same_answer_as_the_unpruned_search(self):
        rnd = random.Random(12)
        rom_size = 0x400000
        for trial in range(300):
            starts = sorted(rnd.sample(range(0x1000, rom_size // 2, 0x10), rnd.randrange(2, 6)))
            ends = [s + rnd.randrange(0x10, 0x800, 0x10) for s in starts]
            noise = [rnd.randrange(0x1000, rom_size, 4) for _ in range(rnd.randrange(0, 6))]
            if trial % 3 == 0:
                words = [0xB0000000 | v for v in noise + starts + ends]           # PI form
            else:
                words = noise + starts + ends
            streams = set(rnd.sample(starts, min(len(starts), rnd.randrange(0, 3)))) if trial % 3 else set()
            data = struct.pack(f">{len(words)}I", *words)
            got = [(r.start, r.end) for r in loaders.table_ranges(data, 0x80100000, rom_size, streams)
                   if r.source == "pairs"]
            forms = [loaders.rom_form(w, rom_size) for w in words]
            expected = []
            i = 0
            while i < len(forms):
                if not forms[i]:
                    i += 1
                    continue
                j = i
                while j < len(forms) and forms[j] and forms[j][1] == forms[i][1]:
                    j += 1
                expected += reference_pairs([f[0] for f in forms[i:j]], forms[i][1], streams)
                i = j
            self.assertEqual(sorted(got), sorted(expected), trial)

    def test_a_larger_half_that_passes_the_span_test_but_overlaps_is_passed_over(self):
        """Half 3, and half 2 at leads 0 and 1, have every span non-empty and short but overlapping:
        the pruning lets them through to the disjointness check, which must reject them; the first
        disjoint answer is half 2 at lead 2 (review D: the random generator does not pin this)."""
        block = [0x1000, 0x1800, 0x2000, 0x3000, 0x2800, 0x3800]
        expected = [(0x2000, 0x2800), (0x3000, 0x3800)]
        for form, words, streams in (("pi", [loaders.PI_BASE | v for v in block], set()),
                                     ("raw", block, {0x1000, 0x1800, 0x2000, 0x3000})):
            self.assertEqual(reference_pairs(block, form, streams), expected, form)
            data = struct.pack(f">{len(words)}I", *words)
            got = [(r.start, r.end) for r in loaders.table_ranges(data, 0x80100000, 0x400000, streams)
                   if r.source == "pairs"]
            self.assertEqual(got, expected, form)


if __name__ == "__main__":
    unittest.main()
