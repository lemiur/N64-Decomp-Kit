"""Per-overlay base inference: one mapped segment that is really several separately linked overlays.

A loader may keep several overlays one after another in the ROM and load each, when needed, at the
address it was linked for (often the same address for all of them).  Mapped as one segment at one
address, such a range shows call targets that land mid-code and no single shift explains them
(round 11, R3: Tarzan's ``seg_65ED0``).

Each window of the segment (``window`` bytes) votes, from its own ``jal`` sites, for the base
``B`` that makes their targets land on function starts (``vram(offset) = B + offset``): every
target ``T`` and every plausible start offset ``s`` near the window vote ``B = T - s``.  A window's
base is accepted when it is unique, has ``min_votes`` votes and explains at least ``agreement`` of
the window's in-range calls.  Adjacent windows with the same base merge into one overlay, also
across windows that hold no ``jal`` at all; a window with calls but no accepted base joins nothing
and ends a run.  The current mapping's base is reported beside each run so the caller sees which
ranges the inference would move and by how much.

A window's in-range calls are those its base maps anywhere into the segment, not only into its own
overlay: the overlay's extent is what the windows decide, so it is not known when one is judged.
Each run also reports ``calls_in_run``/``landed_in_run``, the same calls measured against the run
itself, so a run whose calls go to other parts of the segment shows it.

The result is evidence for a remapping, not one: ``decompile-init`` and the discovery repair never
apply it on their own (the ledger records it as disabled pending verified ground truth), and a
caller that does must keep the original catalogue.
"""
from __future__ import annotations

import struct
from collections import Counter
from dataclasses import asdict, dataclass

WINDOW = 0x8000
MARGIN = 0x10000
MIN_VOTES = 4
AGREEMENT = 0.6
RDRAM = 0x800000
MAX_CLOSE = 8                  # near-tied bases beyond which a window has no clear base


@dataclass
class Run:
    start: int                 # offset in the segment
    end: int
    base: int                  # vram = base + offset
    votes: int
    calls: int                 # distinct jal targets per window (summed) the base maps into the segment
    landed: int                # of those, landing on a function start
    landed_at_current: int     # landing on a start under the segment's current base
    calls_in_run: int = 0      # of ``calls``, those the base maps into this run (not elsewhere in the segment)
    landed_in_run: int = 0     # of those, landing on a function start

    def as_dict(self, current: int) -> dict:
        row = asdict(self)
        row.update(start=f"0x{self.start:X}", end=f"0x{self.end:X}", base=f"0x{self.base:08X}",
                   vram=f"0x{(self.base + self.start) & 0xFFFFFFFF:08X}", shift=self.base - current,
                   moved=self.base != current)
        return row


def _jals(data: bytes) -> list[tuple[int, int]]:
    out = []
    for off in range(0, len(data) - 3, 4):
        word = struct.unpack_from(">I", data, off)[0]
        if word >> 26 == 3:
            out.append((off, (word & 0x3FFFFFF) << 2))
    return out


def _choose(votes: Counter, mine: list[int], lo: int, hi: int, here: int, starts: set[int]) -> tuple[int | None, int]:
    """A window's base and its votes; ``(None, top count)`` when no single base wins.

    Every base within 10 % of the top count is a near-tie, taken from all of ``votes`` (a fixed
    ``most_common(n)`` cuts equal counts by insertion order, so the current or the true base could
    be dropped before this test).  The current base among them is kept (never move without a clear
    winner).  More than ``MAX_CLOSE`` near-ties is no evidence for any one of them.  Otherwise calls
    inside an overlay go to code near the caller: the base whose landing targets (those it maps onto
    a function start) lie closest to the window wins; still tied, nothing is chosen."""
    if not votes:
        return None, 0
    count = max(votes.values())
    if votes.get(here, 0) * 10 >= count * 9:
        return here, votes[here]
    close = [b for b, c in votes.items() if c * 10 >= count * 9]
    if len(close) > MAX_CLOSE:
        return None, count
    centre = (lo + hi) / 2

    def distance(candidate: int) -> float:
        # only the targets this base explains: a call that leaves the segment (libultra, the main
        # image) is no evidence of where the overlay sits, and its offset would favour higher bases
        offsets = [(t - candidate) & 0xFFFFFFFF for t in mine]
        offsets = [o for o in offsets if o in starts]
        return sum(abs(o - centre) for o in offsets) / len(offsets) if offsets else float("inf")
    scored = sorted((distance(b), b) for b in close)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None, count
    return scored[0][1], votes[scored[0][1]]


def infer(data: bytes, current: int, starts: set[int], window: int = WINDOW, margin: int = MARGIN,
          min_votes: int = MIN_VOTES, agreement: float = AGREEMENT) -> list[Run]:
    """Overlay runs of a segment currently mapped at ``current`` (vram of offset 0); ``starts``
    are plausible function-start offsets.  Targets are taken in the segment's 256 MiB region."""
    # a jal's target lies in the 256 MiB region of the instruction after it (an image may start
    # just below a region boundary: a main image mapped from its header at 0x7FFFF400)
    sites = [(off, ((current + off + 4) & 0xF0000000) | t) for off, t in _jals(data)]
    ordered = sorted(starts)
    windows: list[tuple[int, int, int | None, int, int, int, int]] = []
    call_free: set[int] = set()            # windows without a single jal
    accepted: dict[int, list[int]] = {}    # accepted window -> the targets its base maps into the segment
    for lo in range(0, len(data), window):
        hi = min(len(data), lo + window)
        # one vote per distinct call target: many calls to one function (often outside the segment)
        # would otherwise vote for every base near it (round-12 review)
        mine = sorted({t for off, t in sites if lo <= off < hi})
        if not mine:
            call_free.add(lo)
            windows.append((lo, hi, None, 0, 0, 0, 0))
            continue
        near = [s for s in ordered if lo - margin <= s < hi + margin]
        votes: Counter = Counter()
        for t in mine:
            for s in near:
                votes[(t - s) & 0xFFFFFFFF] += 1
        base, count = _choose(votes, mine, lo, hi, current & 0xFFFFFFFF, starts)
        inside = [t for t in mine if base is not None and 0 <= t - base < len(data)]
        landed = sum(1 for t in inside if (t - base) in starts)
        at_current = sum(1 for t in mine if 0 <= t - current < len(data) and (t - current) in starts)
        here = current & 0xFFFFFFFF
        current_inside = sum(1 for t in mine if 0 <= t - current < len(data))
        # a move needs the current mapping to fail the window (it explains under ``agreement`` of the
        # calls) and the new base to explain clearly more (half again as many landings)
        moving = base is not None and base != here
        weak_move = moving and (at_current >= agreement * max(1, current_inside) or landed < 1.5 * at_current + 1)
        # a move must put the window where code can run: inside RDRAM (8 MiB) for KSEG0/KSEG1
        # addresses (mapped addresses below 0x80000000 are left to the TLB).  Only a move is held
        # to it: the current mapping is reported as it is (a main image of more than 8 MiB mapped
        # from 0x7FFFF400 must not lose its unmoved windows)
        start = None if base is None else (base + lo) & 0xFFFFFFFF
        outside = moving and 0x80000000 <= start < 0xC0000000 and \
            (start & 0x1FFFFFFF) + (hi - lo) > RDRAM
        if base is None or count < min_votes or landed < agreement * max(1, len(inside)) or weak_move or outside:
            windows.append((lo, hi, None, count, len(inside), landed, at_current))
        else:
            windows.append((lo, hi, base, count, len(inside), landed, at_current))
            accepted[lo] = inside
    runs: list[Run] = []
    # adjacent windows with one base are one overlay; so are two runs with one base when only
    # windows without any jal lie between them (data, leaf code: no evidence of another base, and
    # one base is one linear mapping, so the bytes between are mapped the same way).  A window with
    # calls but no accepted base still ends a run.
    bridge = False                         # only call-free windows since the last run's end
    for lo, hi, base, count, calls, landed, at_current in windows:
        if base is None:
            bridge = bridge and lo in call_free
            continue
        if runs and runs[-1].base == base and bridge:
            r = runs[-1]
            r.end, r.votes, r.calls, r.landed = hi, r.votes + count, r.calls + calls, r.landed + landed
            r.landed_at_current += at_current
        else:
            runs.append(Run(lo, hi, base, count, calls, landed, at_current))
        bridge = True
    # the same calls measured against the run itself rather than the whole segment: a target the
    # base maps into another part of the segment (a neighbouring overlay) is counted in ``calls``
    # but not here.  Reported only; acceptance stays per window (see the module notes)
    for r in runs:
        for lo, targets in accepted.items():
            if r.start <= lo < r.end:
                own = [t for t in targets if r.start <= t - r.base < r.end]
                r.calls_in_run += len(own)
                r.landed_in_run += sum(1 for t in own if (t - r.base) in starts)
    return runs


def report(data: bytes, current: int, starts: set[int], **options) -> dict:
    runs = infer(data, current, starts, **options)
    moved = [r for r in runs if r.base != current]
    covered = sum(r.end - r.start for r in runs)
    return {"current": f"0x{current:08X}", "bytes": len(data), "runs": [r.as_dict(current) for r in runs],
            "moved_runs": len(moved), "covered_bytes": covered,
            "calls_landing_now": sum(r.landed_at_current for r in runs),
            "calls_landing_inferred": sum(r.landed for r in runs),
            "applied": False,
            "note": "evidence for a remapping; never applied to a catalogue by itself"}
