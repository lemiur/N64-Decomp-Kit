"""Per-overlay base inference (round 12)."""
import struct
import unittest
from collections import Counter

from n64dk.analysis import overlays

PROLOGUE = 0x27BDFFE8          # addiu sp,sp,-24
JR_RA, NOP = 0x03E00008, 0


def overlay(link: int, size: int, seed: int) -> tuple[bytes, list[int]]:
    """Functions of irregular sizes (as real code has); each calls the next two functions of its
    own overlay, linked at ``link``.  Returns the bytes and the function start offsets."""
    import random
    rnd = random.Random(seed)
    sizes, total, k = [], 0, 0
    while True:
        step = rnd.randrange(0x60, 0x400, 0x20)
        if total + step > size:
            break
        sizes.append(step)
        total += step
        k += 1
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    words = []
    for k, length in enumerate(sizes):
        body = [PROLOGUE]
        for step in (1, 2):
            callee = link + starts[(k + step) % len(sizes)]
            body.append((3 << 26) | ((callee >> 2) & 0x3FFFFFF))
            body.append(NOP)
        body += [JR_RA, NOP]
        body += [NOP] * (length // 4 - len(body))
        words += body
    words += [NOP] * ((size - total) // 4)
    return struct.pack(f">{len(words)}I", *words), starts


class OverlayTest(unittest.TestCase):
    def test_three_overlays_linked_at_one_address(self):
        link, size, current = 0x80400000, 0x8000, 0x8015C000
        parts = [overlay(link, size, seed) for seed in range(3)]
        data = b"".join(p[0] for p in parts)
        starts = {k * size + s for k, p in enumerate(parts) for s in p[1]}
        runs = overlays.infer(data, current, starts, window=0x4000)
        self.assertEqual([(r.start, r.end) for r in runs], [(0, 0x8000), (0x8000, 0x10000), (0x10000, 0x18000)])
        for k, r in enumerate(runs):
            self.assertEqual((r.base + r.start) & 0xFFFFFFFF, link, k)
            self.assertEqual(r.landed, r.calls)
            self.assertEqual(r.landed_at_current, 0, k)                 # no target lands in the segment now
        rep = overlays.report(data, current, starts, window=0x4000)
        self.assertEqual(rep["moved_runs"], 3)
        self.assertFalse(rep["applied"])

    def test_a_correctly_mapped_segment_is_not_moved(self):
        current = 0x80100000
        data, offsets = overlay(current, 0x10000, 0)
        starts = set(offsets)
        rep = overlays.report(data, current, starts, window=0x4000)
        self.assertEqual(rep["moved_runs"], 0)
        self.assertEqual(rep["calls_landing_now"], rep["calls_landing_inferred"])

    def test_too_little_evidence_gives_nothing(self):
        data, offsets = overlay(0x80400000, 0x200, 0)          # a few functions, a few calls
        rep = overlays.report(data, 0x8015C000, set(offsets), min_votes=64)
        self.assertEqual(rep["runs"], [])

    def test_a_move_beyond_rdram_is_rejected(self):
        link, size, current = 0x83F00000, 0x8000, 0x8015C000          # beyond the 8 MiB of RDRAM
        data, offsets = overlay(link, size, 1)
        rep = overlays.report(data, current, set(offsets), window=0x4000)
        self.assertEqual(rep["moved_runs"], 0)

    # round-12 review D
    def test_a_move_ending_exactly_at_the_top_of_rdram_is_kept(self):
        current = 0x8015C000
        data, offsets = overlay(0x807F8000, 0x8000, 2)                  # ends at 0x80800000
        self.assertEqual(overlays.report(data, current, set(offsets))["moved_runs"], 1)
        data, offsets = overlay(0x807F8010, 0x8000, 2)                  # 16 bytes past it
        self.assertEqual(overlays.report(data, current, set(offsets))["moved_runs"], 0)

    def test_the_current_mapping_is_reported_even_beyond_rdram(self):
        """Only a move is held to RDRAM: unmoved windows (a 16 MiB main image from 0x7FFFF400) stay."""
        current = 0x80900000
        data, offsets = overlay(current, 0x8000, 0)
        rep = overlays.report(data, current, set(offsets), window=0x4000)
        self.assertEqual(rep["moved_runs"], 0)
        self.assertEqual([(r["start"], r["end"]) for r in rep["runs"]], [("0x0", "0x8000")])

    def test_near_ties_come_from_all_votes_not_the_first_eight(self):
        """V2: ``most_common(8)`` kept the first eight of equal counts; the current base, counted
        last, was cut before the near-tie test and the window moved."""
        here = 0x8015C000
        votes = Counter()
        for k in range(10):
            votes[0x80400000 + 0x100 * k] = 10
        votes[here] = 10
        self.assertEqual(overlays._choose(votes, [0x80400000], 0, 0x4000, here, {0}), (here, 10))
        del votes[here]
        self.assertEqual(overlays._choose(votes, [0x80400000], 0, 0x4000, here, {0}), (None, 10))   # 10 near-ties

    def test_calls_leaving_the_segment_do_not_decide_a_near_tie(self):
        """Two bases tied on votes; the lower explains a call at the window's centre.  Calls to code
        above the segment, counted in the distance, favoured the higher base."""
        high, low = 0x80101000, 0x80100000
        mine = [low + 0x4000, high + 0x7000, 0x80400000, 0x80400100, 0x80400200]
        got = overlays._choose(Counter({high: 5, low: 5}), mine, 0, 0x8000, 0x80200000, {0x4000, 0x7000})
        self.assertEqual(got, (low, 5))

    def test_many_calls_to_one_external_function_give_no_run(self):
        """V1 regression: four calls to one libultra function vote once, not four times."""
        jal = (3 << 26) | ((0x80001000 >> 2) & 0x3FFFFFF)
        words = [jal, NOP] * 4 + [NOP] * 56
        data = struct.pack(f">{len(words)}I", *words)
        self.assertEqual(overlays.report(data, 0x80300000, {0, 0x40, 0x80})["runs"], [])

    # round-12 E2 (review D V3/O8, O9)
    def _halves(self, middle: bytes, cross_call: bool = False, tail: bytes = b""):
        """One overlay linked at 0x80400000 whose two halves have ``middle`` (one 0x4000 window)
        between them; with ``cross_call`` the second half also calls a function of the first."""
        link, size, gap, current = 0x80400000, 0x8000, 0x4000, 0x8015C000
        first, s1 = overlay(link, size, 0)
        second, s2 = overlay(link + size + gap, size, 1)
        if cross_call:                                   # in the padding after the second half's first body
            words = list(struct.unpack(f">{len(second) // 4}I", second))
            words[8] = (3 << 26) | (((link + s1[3]) >> 2) & 0x3FFFFFF)
            second = struct.pack(f">{len(words)}I", *words)
        data = first + middle + second + tail
        starts = set(s1) | {size + gap + s for s in s2}
        return overlays.infer(data, current, starts, window=gap), link, data, starts

    def test_windows_without_calls_do_not_split_an_overlay(self):
        """V3: a window with no jal (data, leaf code) between two windows of one base split the
        overlay into two runs.  A trailing call-free window is not taken into the run."""
        runs, link, data, starts = self._halves(bytes(0x4000), cross_call=True, tail=bytes(0x4000))
        self.assertEqual([(r.start, r.end, r.base) for r in runs], [(0, 0x14000, link)])
        self.assertEqual((runs[0].calls_in_run, runs[0].landed_in_run), (runs[0].calls, runs[0].landed))
        rep = overlays.report(data, 0x8015C000, starts, window=0x4000)
        self.assertEqual((rep["moved_runs"], rep["covered_bytes"]), (1, 0x14000))

    def test_a_window_with_calls_and_no_base_still_ends_a_run(self):
        jal = (3 << 26) | ((0x80001000 >> 2) & 0x3FFFFFF)                  # one external function
        middle = struct.pack(f">{0x4000 // 4}I", *([jal, NOP] * 4 + [NOP] * (0x4000 // 4 - 8)))
        runs, link, _data, _starts = self._halves(middle, cross_call=True)
        self.assertEqual([(r.start, r.end, r.base) for r in runs], [(0, 0x8000, link), (0xC000, 0x14000, link)])
        # O9: the second half's call into the first counts as landed over the segment, not in its run
        first, second = runs
        self.assertEqual((first.calls_in_run, first.landed_in_run), (first.calls, first.landed))
        self.assertEqual((second.calls - second.calls_in_run, second.landed - second.landed_in_run), (1, 1))
        self.assertIn("calls_in_run", second.as_dict(0x8015C000))


if __name__ == "__main__":
    unittest.main()
