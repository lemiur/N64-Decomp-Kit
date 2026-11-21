"""Calibrated confidence and decision quality for the kit's heuristic scores (round 12, item U).

Many stages attach a number to a decision: a boundary operation's ``confidence`` (0.5 .. 0.95,
assigned by rule), a toolchain region's class ``share``, a similarity ``score``, an active-
identification posterior, a catalogue row's kind.  None of these was fitted to outcomes, so none
of them is a probability until measured.  This module measures them against labelled outcomes
and turns them into decisions with stated costs, without ever touching byte evidence:

* ``reliability`` (equal-width or equal-count bins, with Wilson intervals per bin) and
  ``Isotonic`` (pool-adjacent-violators: the non-decreasing step function of observed frequency);
* ``brier`` and ``ece`` (expected calibration error; both refuse scores outside [0, 1]: a score
  that is not on a probability scale is mapped first, by ``Isotonic``, or only thresholded);
* ``confusion`` (precision, recall, false-positive rate at a threshold; positives the method
  never scored count as misses) and ``choose_threshold`` (minimum expected cost for explicit
  false-positive and false-negative costs; ties go to the higher, more conservative threshold);
* ``drift``: per-title ECE and its spread (a calibration fitted on one family of titles may not
  hold on another);
* a **freeze protocol**: ``freeze`` fits the mapping and threshold on development titles only and
  stores them with a SHA-256 digest; ``load_frozen`` refuses a document whose digest does not
  match; ``evaluate_frozen`` applies the frozen mapping and threshold to held-out titles without
  refitting and refuses any record from a title used for fitting.

A record is ``(score, truth, title)``; ``truth`` is 1 or 0.  Nothing here grants credit: a
calibrated probability orders and gates work, and only the strict verifier decides a match.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

FORMAT = "n64dk-calibration-freeze"
VERSION = 1
Z95 = 1.959964


class FrozenError(ValueError):
    """A frozen calibration that is tampered with or evaluated on its own development titles."""


@dataclass(frozen=True)
class Record:
    score: float
    truth: int
    title: str = ""
    item: str = ""


def records(rows) -> list[Record]:
    """``Record``s from records, ``(score, truth[, title[, item]])`` tuples or mappings."""
    out = []
    for row in rows:
        if isinstance(row, Record):
            out.append(row)
        elif isinstance(row, dict):
            out.append(Record(float(row["score"]), int(bool(row["truth"])), str(row.get("title", "")),
                              str(row.get("item", ""))))
        else:
            score, truth, *rest = row
            out.append(Record(float(score), int(bool(truth)), *(str(x) for x in rest[:2])))
    return out


def _probabilities(rows: list[Record]) -> None:
    bad = [r.score for r in rows if not 0.0 <= r.score <= 1.0 or math.isnan(r.score)]
    if bad:
        raise ValueError(f"{len(bad)} scores outside [0, 1] (first {bad[0]}): not on a probability scale; "
                         "map them with Isotonic first or only threshold them")


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials ((0, 1) when n is 0)."""
    if n == 0:
        return 0.0, 1.0
    p, z2 = k / n, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def brier(rows) -> float:
    rows = records(rows)
    _probabilities(rows)
    return sum((r.score - r.truth) ** 2 for r in rows) / len(rows) if rows else float("nan")


def reliability(rows, bins: int = 10, strategy: str = "uniform") -> list[dict]:
    """Reliability curve: per non-empty bin, its range, count, mean score, observed frequency and
    the frequency's Wilson interval.  ``uniform``: equal-width bins over [0, 1] (scores must be
    probabilities); ``quantile``: equal-count bins over the sorted scores (any scale)."""
    rows = sorted(records(rows), key=lambda r: r.score)
    if not rows:
        return []
    groups: list[list[Record]] = []
    if strategy == "uniform":
        _probabilities(rows)
        table: dict[int, list[Record]] = defaultdict(list)
        for r in rows:
            table[min(int(r.score * bins), bins - 1)].append(r)
        groups = [table[k] for k in sorted(table)]
    elif strategy == "quantile":
        size = max(1, math.ceil(len(rows) / bins))
        start = 0
        while start < len(rows):
            end = min(len(rows), start + size)
            while end < len(rows) and rows[end].score == rows[end - 1].score:   # equal scores stay together
                end += 1
            groups.append(rows[start:end])
            start = end
    else:
        raise ValueError(f"unknown binning strategy {strategy}")
    out = []
    for group in groups:
        k, n = sum(r.truth for r in group), len(group)
        lo, hi = wilson(k, n)
        out.append({"lo": group[0].score, "hi": group[-1].score, "n": n, "mean_score": sum(r.score for r in group) / n,
                    "observed": k / n, "wilson": [round(lo, 4), round(hi, 4)]})
    return out


def ece(rows, bins: int = 10) -> float:
    """Expected calibration error over equal-width bins: sum of n_b/N * |observed_b - mean_b|."""
    rows = records(rows)
    curve = reliability(rows, bins, "uniform")
    total = sum(b["n"] for b in curve)
    return sum(b["n"] / total * abs(b["observed"] - b["mean_score"]) for b in curve) if total else float("nan")


class Isotonic:
    """Pool-adjacent-violators fit of P(truth | score): a non-decreasing step function.  Equal
    scores are pooled first; a score between two blocks takes the lower block's value (the
    conservative side); a score below every block takes the first block's value."""

    def __init__(self, blocks: list[tuple[float, float, float, int]] | None = None):
        self.blocks = list(blocks or [])          # (lowest score, highest score, value, count)

    @classmethod
    def fit(cls, rows) -> "Isotonic":
        rows = sorted(records(rows), key=lambda r: r.score)
        stack: list[list] = []                    # [lo, hi, positives, count]
        for r in rows:
            if stack and stack[-1][1] == r.score:
                stack[-1][2] += r.truth
                stack[-1][3] += 1
            else:
                stack.append([r.score, r.score, r.truth, 1])
            while len(stack) > 1 and stack[-2][2] / stack[-2][3] >= stack[-1][2] / stack[-1][3]:   # equal: pooled too
                hi, positives, count = stack[-1][1], stack[-1][2], stack[-1][3]
                stack.pop()
                stack[-1][1] = hi
                stack[-1][2] += positives
                stack[-1][3] += count
        return cls([(lo, hi, positives / count, count) for lo, hi, positives, count in stack])

    def predict(self, score: float) -> float:
        if not self.blocks:
            raise ValueError("empty isotonic fit")
        index = bisect.bisect_right([b[0] for b in self.blocks], score) - 1
        return self.blocks[max(index, 0)][2]

    def apply(self, rows) -> list[Record]:
        return [Record(self.predict(r.score), r.truth, r.title, r.item) for r in records(rows)]

    def as_list(self) -> list[list]:
        return [[lo, hi, value, n] for lo, hi, value, n in self.blocks]       # exact: thresholds compare equal

    @classmethod
    def from_list(cls, blocks) -> "Isotonic":
        return cls([(float(lo), float(hi), float(value), int(n)) for lo, hi, value, n in blocks])


def confusion(rows, threshold: float, missed: int = 0) -> dict:
    """Decision ``score >= threshold`` against truth.  ``missed``: positives the method never
    scored (they are false negatives at every threshold)."""
    rows = records(rows)
    tp = sum(1 for r in rows if r.score >= threshold and r.truth)
    fp = sum(1 for r in rows if r.score >= threshold and not r.truth)
    fn = sum(1 for r in rows if r.score < threshold and r.truth) + missed
    tn = sum(1 for r in rows if r.score < threshold and not r.truth)
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            "precision_wilson": [round(x, 4) for x in wilson(tp, tp + fp)]}


def choose_threshold(rows, cost_fp: float, cost_fn: float, missed: int = 0) -> dict:
    """The threshold among the observed scores (and +inf: accept nothing) that minimises
    ``cost_fp * FP + cost_fn * FN``; ties go to the higher threshold."""
    if cost_fp < 0 or cost_fn < 0:
        raise ValueError("costs must be non-negative")
    rows = records(rows)
    best = None
    for threshold in sorted({r.score for r in rows} | {math.inf}, reverse=True):
        c = confusion(rows, threshold, missed)
        cost = cost_fp * c["fp"] + cost_fn * c["fn"]
        if best is None or cost < best["expected_cost"]:
            best = dict(c, expected_cost=cost, cost_fp=cost_fp, cost_fn=cost_fn)
    return best


def drift(rows, bins: int = 10, min_records: int = 20) -> dict:
    """Per-title ECE (titles with at least ``min_records``), its spread and the worst title."""
    by_title: dict[str, list[Record]] = defaultdict(list)
    for r in records(rows):
        by_title[r.title].append(r)
    per = {title: {"n": len(rs), "ece": round(ece(rs, bins), 4), "observed": round(sum(r.truth for r in rs) / len(rs), 4)}
           for title, rs in sorted(by_title.items()) if len(rs) >= min_records}
    values = [v["ece"] for v in per.values()]
    return {"titles": per, "spread": round(max(values) - min(values), 4) if values else None,
            "worst": max(per, key=lambda t: (per[t]["ece"], t)) if per else None,
            "skipped_titles": sorted(t for t, rs in by_title.items() if len(rs) < min_records)}


# ---- freeze protocol ---------------------------------------------------------------------------
def _digest(payload: dict) -> str:
    body = {k: v for k, v in payload.items() if k != "digest"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def summary(rows, missed: int = 0, threshold: float | None = None) -> dict:
    rows = records(rows)
    out = {"records": len(rows), "positives": sum(r.truth for r in rows), "missed": missed}
    if rows and all(0.0 <= r.score <= 1.0 for r in rows):
        out.update(brier=round(brier(rows), 6), ece=round(ece(rows), 6))
    if threshold is not None:
        out["at_threshold"] = confusion(rows, threshold, missed)
    return out


def _missed(missed, titles) -> int:
    """Positives never scored: an int, or {title: n} summed over ``titles``."""
    if isinstance(missed, dict):
        return sum(int(n) for title, n in missed.items() if title in titles)
    return int(missed or 0)


def freeze(sources: dict, development: list[str], cost_fp: float, cost_fn: float, note: str = "") -> dict:
    """Fit each source on development titles only.  ``sources``: name -> {"records": [...],
    "missed": n or {title: n} (positives never scored), "probability": bool (the score claims to
    be a probability), "development": [titles] (overrides ``development`` for this source),
    "cost_fp"/"cost_fn" (overrides)}.  Returns the frozen document (mapping, threshold on the
    mapped scale, in-sample development metrics, digest).  Every development title of every source
    is refused at evaluation."""
    every: set[str] = set(development)
    document = {"format": FORMAT, "version": VERSION, "note": note, "sources": {}}
    for name, source in sorted(sources.items()):
        titles = sorted(set(source.get("development") or development))
        every |= set(titles)
        rows = [r for r in records(source.get("records") or []) if r.title in titles]
        if not titles or not rows:
            document["sources"][name] = {"status": "no development records", "development_titles": titles}
            continue
        missed = _missed(source.get("missed", 0), titles)
        fp_cost, fn_cost = float(source.get("cost_fp", cost_fp)), float(source.get("cost_fn", cost_fn))
        mapping = Isotonic.fit(rows)
        mapped = mapping.apply(rows)
        chosen = choose_threshold(mapped, fp_cost, fn_cost, missed)
        entry = {"status": "frozen", "probability_claimed": bool(source.get("probability", False)),
                 "isotonic": mapping.as_list(), "threshold": chosen["threshold"], "cost_fp": fp_cost,
                 "cost_fn": fn_cost, "development_in_sample": summary(mapped, missed, chosen["threshold"]),
                 "development_titles": titles, "titles": sorted({r.title for r in rows})}
        if entry["probability_claimed"]:
            entry["development_raw"] = summary(rows, missed)
        document["sources"][name] = entry
    if not every:
        raise FrozenError("no development titles")
    document["development_titles"] = sorted(every)
    document["digest"] = _digest(document)
    return document


def save_frozen(document: dict, path) -> None:
    Path(path).write_text(json.dumps(document, indent=1, sort_keys=True, default=str) + "\n")


def load_frozen(path_or_document) -> dict:
    document = path_or_document if isinstance(path_or_document, dict) else json.loads(Path(path_or_document).read_text())
    if document.get("format") != FORMAT:
        raise FrozenError("not a frozen calibration")
    if document.get("digest") != _digest(document):
        raise FrozenError("frozen calibration digest mismatch: the document changed after freezing")
    return document


def evaluate_frozen(document, name: str, rows, missed: int = 0) -> dict:
    """Held-out metrics of source ``name`` under the frozen mapping and threshold (no refit).
    Refuses records from any development title."""
    document = load_frozen(document)
    entry = document["sources"].get(name) or {}
    if entry.get("status") != "frozen":
        raise FrozenError(f"source {name} was not frozen ({entry.get('status', 'absent')})")
    rows = records(rows)
    used = sorted({r.title for r in rows} & set(document["development_titles"]))
    if used:
        raise FrozenError(f"refusing to evaluate on development titles {used}: they were used for fitting")
    if not rows:
        return {"status": "no held-out records", "digest": document["digest"]}
    missed = _missed(missed, {r.title for r in rows})
    mapping = Isotonic.from_list(entry["isotonic"])
    threshold = entry["threshold"]
    mapped = mapping.apply(rows)
    out = {"status": "evaluated", "digest": document["digest"], "threshold": threshold,
           "held_out_titles": sorted({r.title for r in rows}), "mapped": summary(mapped, missed, threshold),
           "drift": drift(mapped, min_records=10)}
    if entry.get("probability_claimed") and all(0.0 <= r.score <= 1.0 for r in rows):
        out["raw"] = summary(rows, missed)
    return out
