"""Active toolchain identification: choose the next build (or probe) by expected information gain.

Calibration today builds a spread sample of small functions under every candidate recipe and
keeps the recipe most of the byte-identical ones were built with (``decompile-run``
``calibrate_recipe``; per region with ``--regional-calibration``).  Most of those builds teach
nothing: once one recipe has produced several exact functions, building the same functions under
an unrelated compiler only confirms what is known, while two recipes that the static evidence
cannot tell apart (two assemblers behind one compiler, two neighbouring flags) are separated only
by the few functions on which they produce different code.

This module keeps a posterior over candidate hypotheses for one scope (title, image, region, unit
or function) and asks, before every build, which observation is expected to reduce its entropy
the most per unit of cost:

* **Hypotheses** are labels (recipes such as ``kmc+as -O2 -G0 -mips3 -mgp32 -mfp32``, or
  component variants); the **prior** is uniform, or comes from the static regional ranking
  (``prior_from_region``: ``toolchain_regions`` output).
* A **build** of function ``f`` under recipe ``b`` has an outcome bucket (``exact``, ``near``,
  ``far``; the caller's build function returns it, or ``(exact, score)``).  Its likelihood under
  hypothesis ``h`` comes from an ``OutcomeTable``: a row per relation between ``h`` and ``b``
  (``same``, ``sibling``: one compiler and optimisation level, other flags or assembler;
  ``family``; ``other``), optionally a row per (h, b) pair learned from calibration history
  (``learn_table``).  A per-function weight mixes in a recipe-independent "poor draft" row, so a
  function whose draft is bad says less.  Every likelihood is floored and renormalised: no
  observation can drive a hypothesis to zero, so a contradicting observation is absorbed.
* A **probe** is ``(name, distinguishes: {hypothesis: expected output class})`` with an optional
  confusion matrix; the caller runs it (``probe_fn``).  Probes come from elsewhere (the dialect
  and pass probes, compiler probes' lane marks: ``probes_from_lane_marks``); this module only
  scores them.
* **Stopping**: the most probable hypothesis reaches ``threshold``, no observation has gain per
  cost above ``epsilon``, the budget is spent, or nothing is left to observe.

The report lists the candidates before, each observation chosen and why it distinguishes (which
hypotheses predict which outcome), its outcome, the posterior after, and, when asked, the builds a
uniform calibration needs to reach the same certainty with the same build function and the same
model (``compare_uniform``; meant for replays where builds are lookups).

Pure Python, deterministic (ties broken by declared order), no builds of its own: the build and
probe functions are injected.  A posterior is evidence for where to search first; only a rebuilt
function that the strict verifier accepts is a match.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

OUTCOMES = ("exact", "near", "far")
RELATIONS = ("same", "sibling", "family", "other")
# Conservative default: an exact build is strong but not conclusive evidence (sibling recipes often
# produce identical code); a far build under the true recipe is common (drafts are imperfect).
DEFAULT_ROWS = {
    "same": {"exact": 0.45, "near": 0.35, "far": 0.20},
    "sibling": {"exact": 0.15, "near": 0.40, "far": 0.45},
    "family": {"exact": 0.04, "near": 0.26, "far": 0.70},
    "other": {"exact": 0.01, "near": 0.09, "far": 0.90},
}
DEFAULT_NOISE = {"exact": 0.01, "near": 0.29, "far": 0.70}     # a draft too poor to tell recipes apart
FLOOR = 0.005                 # no likelihood below this: no hypothesis is ever ruled out by one observation
THRESHOLD = 0.95
EPSILON = 1e-3                # bits per unit cost below which an observation is not worth making
NEAR_SCORE = 20               # build scores at or below this (and not exact) are "near"

_GCC = ("kmc", "sn", "custom:gcc", "custom:egcs")
_IDO = ("ido", "custom:ido", "custom:mipspro")


# ---- recipe labels ---------------------------------------------------------------------------
def parts(label: str) -> tuple[str, str, str]:
    """``(pipeline, optimisation, other flags)`` of a recipe label (``"kmc+as -O2 -G0 ..."``)."""
    tokens = label.split()
    pipeline = tokens[0] if tokens else ""
    opt = next((t for t in tokens[1:] if re.fullmatch(r"-O\d?|-g\d?", t)), "")
    rest = " ".join(t for t in tokens[1:] if t != opt)
    return pipeline, opt, rest


def compiler(pipeline: str) -> str:
    """The compiler a pipeline runs (``kmc+as`` and ``kmc+asn64`` share KMC GCC)."""
    return "kmc" if pipeline.startswith("kmc") else pipeline


def group(pipeline: str) -> str:
    if pipeline.startswith(_GCC):
        return "gcc"
    if pipeline.startswith(_IDO):
        return "ido"
    return pipeline


def class_key(label: str) -> str:
    """The idioms-table class of a recipe label: pipeline and optimisation level."""
    pipeline, opt, _rest = parts(label)
    return f"{pipeline} {opt}".strip()


def default_relation(truth: str, built: str) -> str:
    if truth == built:
        return "same"
    (pt, ot, _), (pb, ob, _) = parts(truth), parts(built)
    if compiler(pt) == compiler(pb) and ot == ob:
        return "sibling"
    if compiler(pt) == compiler(pb) or (group(pt) == group(pb) and ot == ob):
        return "family"
    return "other"


def _normalised(row: dict, floor: float) -> dict:
    floored = {k: max(float(v), floor) for k, v in row.items()}
    total = sum(floored.values())
    return {k: v / total for k, v in floored.items()}


def _lookup(mapping: dict, label: str):
    """A value keyed by the exact label, else by its class key, else by its pipeline."""
    for key in (label, class_key(label), parts(label)[0]):
        if key in mapping:
            return mapping[key]
    return None


# ---- the outcome model -----------------------------------------------------------------------
@dataclass
class OutcomeTable:
    """P(outcome of building under ``b`` | the true hypothesis is ``h``)."""
    rows: dict = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_ROWS.items()})
    noise: dict = field(default_factory=lambda: dict(DEFAULT_NOISE))
    pairs: dict = field(default_factory=dict)           # "h|b" -> row, learned with enough support
    floor: float = FLOOR
    relation: object = default_relation
    source: str = "default (conservative, not learned)"

    @property
    def outcomes(self) -> tuple[str, ...]:
        return tuple(self.noise)

    def likelihood(self, truth: str, built: str, weight: float = 1.0) -> dict:
        cache = self.__dict__.setdefault("_cache", {})
        key = (truth, built, weight)
        if key not in cache:
            row = self.pairs.get(f"{truth}|{built}") or self.rows[self.relation(truth, built)]
            mixed = {o: weight * row.get(o, 0.0) + (1.0 - weight) * self.noise.get(o, 0.0) for o in self.outcomes}
            cache[key] = _normalised(mixed, self.floor)
        return cache[key]

    def as_dict(self) -> dict:
        return {"rows": self.rows, "noise": self.noise, "pairs": self.pairs, "floor": self.floor,
                "source": self.source}

    @classmethod
    def from_dict(cls, data: dict) -> "OutcomeTable":
        table = cls()
        table.rows = {k: dict(v) for k, v in (data.get("rows") or DEFAULT_ROWS).items()}
        table.noise = dict(data.get("noise") or DEFAULT_NOISE)
        table.pairs = {k: dict(v) for k, v in (data.get("pairs") or {}).items()}
        table.floor = float(data.get("floor", FLOOR))
        table.source = data.get("source", "given")
        return table


def learn_table(history, prior: OutcomeTable | None = None, prior_weight: float = 20.0, alpha: float = 1.0,
                min_pair: int = 30, relation=default_relation) -> OutcomeTable:
    """An outcome table from calibration history: ``(true hypothesis, built recipe, outcome)``
    rows (a finished run's chosen recipe stands for the truth).  Each relation row is the prior
    row worth ``prior_weight`` observations plus the counts, Laplace-smoothed by ``alpha``; a
    (truth, built) pair seen ``min_pair`` times or more gets its own row the same way."""
    prior = prior or OutcomeTable(relation=relation)
    outcomes = prior.outcomes
    by_relation: dict[str, dict[str, int]] = {r: {o: 0 for o in outcomes} for r in prior.rows}
    by_pair: dict[str, dict[str, int]] = {}
    seen = 0
    for truth, built, outcome in history:
        if outcome not in outcomes:
            continue
        seen += 1
        rel = relation(truth, built)
        by_relation.setdefault(rel, {o: 0 for o in outcomes})[outcome] += 1
        by_pair.setdefault(f"{truth}|{built}", {o: 0 for o in outcomes})[outcome] += 1

    def smooth(counts: dict, base: dict) -> dict:
        n = sum(counts.values())
        return {o: (counts[o] + prior_weight * base.get(o, 0.0) + alpha) / (n + prior_weight + alpha * len(outcomes))
                for o in outcomes}

    table = OutcomeTable(rows={rel: smooth(counts, prior.rows.get(rel, prior.rows["other"]))
                               for rel, counts in by_relation.items()},
                         noise=dict(prior.noise), floor=prior.floor, relation=relation,
                         source=f"learned from {seen} observations (prior weight {prior_weight}, alpha {alpha})")
    for key, counts in by_pair.items():
        if sum(counts.values()) >= min_pair:
            truth, _, built = key.partition("|")
            table.pairs[key] = smooth(counts, table.rows[relation(truth, built)])
    return table


def bucket(result, near: int = NEAR_SCORE) -> str | None:
    """A build function's result as an outcome: a bucket name passes through; ``True``/``False``;
    ``(exact, score)``; ``None`` (the build failed: no information)."""
    if result is None or isinstance(result, str):
        return result
    if isinstance(result, bool):
        return "exact" if result else "far"
    exact, score = result
    if exact:
        return "exact"
    return "near" if score is not None and score <= near else "far"


# ---- probes ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Probe:
    """An observation whose expected output class differs between hypotheses.  ``confusion`` is
    P(observed | expected) per expected class; by default ``1 - noise`` on the diagonal."""
    name: str
    distinguishes: dict
    cost: float = 1.0
    noise: float = 0.05
    confusion: dict | None = None

    def classes(self) -> list[str]:
        found = sorted(set(self.distinguishes.values()) | set((self.confusion or {}).keys()))
        return found + ["other"] if "other" not in found else found

    def likelihood(self, truth: str, floor: float = FLOOR) -> dict:
        classes = self.classes()
        expected = _lookup(self.distinguishes, truth)
        if expected is None:                              # the probe says nothing about this hypothesis
            return {c: 1.0 / len(classes) for c in classes}
        if self.confusion and expected in self.confusion:
            row = {c: self.confusion[expected].get(c, 0.0) for c in classes}
        else:
            row = {c: (1.0 - self.noise) if c == expected else self.noise / (len(classes) - 1) for c in classes}
        return _normalised(row, floor)


def as_probe(item) -> Probe:
    """A ``Probe``, or a ``(name, distinguishes)`` / ``(name, distinguishes, cost)`` tuple, or a
    mapping with those keys (the interface the dialect and pass probes are offered through)."""
    if isinstance(item, Probe):
        return item
    if isinstance(item, dict):
        return Probe(item["name"], dict(item["distinguishes"]), float(item.get("cost", 1.0)),
                     float(item.get("noise", 0.05)), item.get("confusion"))
    name, distinguishes, *rest = item
    return Probe(name, dict(distinguishes), float(rest[0]) if rest else 1.0)


LANE_MARK_CONFUSION = {"present": {"present": 0.6, "absent": 0.4, "other": 0.0},
                       "absent": {"present": 0.05, "absent": 0.95, "other": 0.0}}


def probes_from_lane_marks(lane_marks: dict, candidates: list[str], marks=None, cost: float = 0.0) -> list[Probe]:
    """Static probes from the compiler-probes table (``toolchain_regions.load_lane_marks``):
    for each component mark, which candidates can produce it.  A recipe that can produce a mark
    need not show it in every region (asymmetric confusion), one that cannot should not show it.
    Marks no candidate separates are dropped.  ``cost`` 0: the observation is a static scan."""
    out = []
    every = sorted({m for found in lane_marks.values() for m in found}) if marks is None else sorted(marks)
    for mark in every:
        expected = {}
        for label in candidates:
            found = _lookup(lane_marks, label)
            if found is not None:
                expected[label] = "present" if mark in found else "absent"
        if len(set(expected.values())) > 1:
            out.append(Probe(f"mark:{mark}", expected, cost, confusion=LANE_MARK_CONFUSION))
    return out


def prior_from_region(region: dict | None, candidates: list[str], floor: float = 0.02) -> dict:
    """A prior over ``candidates`` from a ``toolchain_regions`` region (its ``ranked`` class
    shares), each class's share split among the candidates of that class, plus ``floor`` for
    every candidate.  Uniform without a ranking."""
    ranked = (region or {}).get("ranked") or []
    mass = {label: floor for label in candidates}
    for entry in ranked:
        members = [label for label in candidates if class_key(label) == entry["recipe"] or label == entry["recipe"]]
        for label in members:
            mass[label] += float(entry["share"]) / len(members)
    total = sum(mass.values())
    return {label: value / total for label, value in mass.items()}


# ---- the engine --------------------------------------------------------------------------------
def entropy(dist: dict) -> float:
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


@dataclass(frozen=True)
class Option:
    kind: str                       # build or probe
    function: str | None = None
    recipe: str | None = None
    probe: Probe | None = None
    weight: float = 1.0
    cost: float = 1.0

    def label(self) -> str:
        return f"probe {self.probe.name}" if self.kind == "probe" else f"build {self.function} under {self.recipe}"


def _functions(functions) -> list[tuple[str, float, float]]:
    out = []
    for item in functions:
        if isinstance(item, str):
            out.append((item, 1.0, 1.0))
        elif isinstance(item, dict):
            out.append((item["name"], float(item.get("weight", 1.0)), float(item.get("cost", 1.0))))
        else:
            name, *rest = item
            out.append((name, float(rest[0]) if rest else 1.0, float(rest[1]) if len(rest) > 1 else 1.0))
    return out


class ActiveId:
    """Posterior over ``candidates`` for one scope, updated by builds and probes."""

    def __init__(self, candidates: list[str], prior: dict | None = None, table: OutcomeTable | None = None,
                 scope: str = "", threshold: float = THRESHOLD, epsilon: float = EPSILON, budget: float = 64.0):
        if not candidates:
            raise ValueError("no candidate hypotheses")
        self.candidates = list(dict.fromkeys(candidates))
        self.table = table or OutcomeTable()
        self.scope, self.threshold, self.epsilon, self.budget = scope, threshold, epsilon, budget
        prior = prior or {}
        raw = {h: max(float(prior.get(h, 0.0)), 1e-6) if prior else 1.0 for h in self.candidates}
        total = sum(raw.values())
        self.initial = {h: raw[h] / total for h in self.candidates}
        self.log = {h: math.log(p) for h, p in self.initial.items()}

    # posterior
    @property
    def posterior(self) -> dict:
        top = max(self.log.values())
        weights = {h: math.exp(v - top) for h, v in self.log.items()}
        total = sum(weights.values())
        return {h: w / total for h, w in weights.items()}

    def ranked(self, limit: int = 8) -> list[dict]:
        post = self.posterior
        order = sorted(self.candidates, key=lambda h: (-post[h], self.candidates.index(h)))
        return [{"hypothesis": h, "p": round(post[h], 4)} for h in order[:limit]]

    def credible_set(self) -> list[str]:
        post, out, mass = self.posterior, [], 0.0
        for h in sorted(self.candidates, key=lambda h: (-post[h], self.candidates.index(h))):
            out.append(h)
            mass += post[h]
            if mass >= self.threshold:
                break
        return out

    # observations
    def rows(self, option: Option) -> dict:
        """hypothesis -> P(outcome | hypothesis) for one option."""
        if option.kind == "probe":
            return {h: option.probe.likelihood(h, self.table.floor) for h in self.candidates}
        return {h: self.table.likelihood(h, option.recipe, option.weight) for h in self.candidates}

    def gain(self, option: Option, post: dict | None = None) -> tuple[float, dict]:
        """Expected entropy reduction in bits and the predictive outcome distribution."""
        post, rows = post or self.posterior, self.rows(option)
        outcomes = list(next(iter(rows.values())))
        predictive = {o: sum(post[h] * rows[h][o] for h in self.candidates) for o in outcomes}
        after = 0.0
        for o, p in predictive.items():
            if p <= 0:
                continue
            after += p * entropy({h: post[h] * rows[h][o] / p for h in self.candidates})
        return max(0.0, entropy(post) - after), predictive

    def why(self, option: Option) -> dict:
        """Which hypotheses predict which outcome (most likely under each), with their mass."""
        post, rows = self.posterior, self.rows(option)
        predicts: dict[str, list[str]] = {}
        mass: dict[str, float] = {}
        for h in self.candidates:
            if post[h] < 0.01:
                continue
            best = max(rows[h], key=lambda o: (rows[h][o], -list(rows[h]).index(o)))
            predicts.setdefault(best, []).append(h)
            mass[best] = mass.get(best, 0.0) + post[h]
        return {"predicts": predicts, "mass": {o: round(m, 4) for o, m in mass.items()},
                "separates": len(predicts) > 1}

    def update(self, option: Option, outcome: str) -> None:
        rows = self.rows(option)
        for h in self.candidates:
            p = rows[h].get(outcome)
            if p is None:                                 # an outcome the model does not know: floor
                p = self.table.floor
            self.log[h] += math.log(p)

    def options(self, functions, probes, done: set, uses: dict) -> list[Option]:
        """Every build not yet made (one representative function per (weight, cost) class and
        recipe: functions of a class are exchangeable under the model) and every unused probe."""
        out = []
        for recipe in self.candidates:
            classes: dict[tuple[float, float], tuple] = {}
            for order, (name, weight, cost) in enumerate(functions):
                if (name, recipe) in done:
                    continue
                key = (weight, cost)
                rank = (-uses.get(name, 0), order)        # reuse a function already drafted
                if key not in classes or rank < classes[key][0]:
                    classes[key] = (rank, name)
            for (weight, cost), (_rank, name) in sorted(classes.items(), key=lambda kv: kv[1][0]):
                out.append(Option("build", name, recipe, None, weight, cost))
        for probe in probes:
            if ("probe", probe.name) not in done:
                out.append(Option("probe", probe=probe, cost=probe.cost))
        return out

    def select(self, functions, probes, done: set, uses: dict):
        """The option with the most expected bits per unit cost: ``(option, bits, predictive)``."""
        functions, probes = _functions(functions), [as_probe(p) for p in probes]
        best = None
        post = self.posterior
        for number, option in enumerate(self.options(functions, probes, done, uses)):
            bits, predictive = self.gain(option, post)
            per_cost = bits / max(option.cost, 1e-3)
            key = (-round(per_cost, 9), option.cost, number)
            if best is None or key < best[0]:
                best = (key, option, bits, predictive)
        return None if best is None else best[1:]

    def run(self, functions, build, probes=(), probe_fn=None, compare_uniform: bool = False,
            max_steps: int = 10_000) -> dict:
        """Observe until a stopping rule holds; returns the report (see the module docstring).

        ``functions``: names, ``(name, weight[, cost])`` or ``{"name", "weight", "cost"}``;
        ``build(function, recipe)`` -> outcome (see ``bucket``); ``probe_fn(probe)`` -> class."""
        functions = _functions(functions)
        probes = [as_probe(p) for p in probes] if probe_fn is not None else []
        before = self.ranked()
        cache: dict[tuple[str, str], str | None] = {}

        def built(name, recipe):
            if (name, recipe) not in cache:
                cache[(name, recipe)] = bucket(build(name, recipe))
            return cache[(name, recipe)]

        done: set = set()
        uses: dict[str, int] = {}
        steps, spent, builds, probes_run, stopped = [], 0.0, 0, 0, None
        for _ in range(max_steps):
            post = self.posterior
            if max(post.values()) >= self.threshold:
                stopped = "confident"
                break
            chosen = self.select(functions, probes, done, uses)
            if chosen is None:
                stopped = "nothing left to observe"
                break
            option, bits, predictive = chosen
            if bits / max(option.cost, 1e-3) <= self.epsilon:
                stopped = "no observation is informative enough"
                break
            if spent + option.cost > self.budget:
                stopped = "budget"
                break
            why = self.why(option)
            if option.kind == "probe":
                outcome = probe_fn(option.probe)
                done.add(("probe", option.probe.name))
                probes_run += 1
            else:
                outcome = built(option.function, option.recipe)
                done.add((option.function, option.recipe))
                uses[option.function] = uses.get(option.function, 0) + 1
                builds += 1
            spent += option.cost
            if outcome is not None:
                self.update(option, outcome)
            steps.append({"step": len(steps) + 1, "observation": option.label(), "kind": option.kind,
                          "function": option.function, "recipe": option.recipe,
                          "probe": option.probe.name if option.probe else None, "cost": option.cost,
                          "expected_gain_bits": round(bits, 4),
                          "predictive": {o: round(p, 4) for o, p in predictive.items()}, "why": why,
                          "outcome": outcome, "posterior": self.ranked(4),
                          "entropy_bits": round(entropy(self.posterior), 4)})
        post = self.posterior
        top = max(self.candidates, key=lambda h: (post[h], -self.candidates.index(h)))
        report = {"scope": self.scope, "candidates_before": before, "steps": steps,
                  "posterior_after": self.ranked(), "decision": top if post[top] >= self.threshold else None,
                  "credible_set": self.credible_set(), "stopped": stopped or "step limit", "builds": builds,
                  "probes": probes_run, "cost": spent, "threshold": self.threshold,
                  "table": self.table.source, "proof": False}
        if compare_uniform:
            uniform = uniform_calibration(self.candidates, functions, built, prior=self.initial, table=self.table,
                                          threshold=self.threshold)
            report["uniform"] = uniform
            report["builds_avoided"] = uniform["builds"] - builds if uniform["reached"] and report["decision"] else None
        return report


def uniform_calibration(candidates, functions, build, prior=None, table=None, threshold: float = THRESHOLD,
                        max_builds: int | None = None) -> dict:
    """Today's shape of calibration under the same model: each function in order built under every
    candidate in order, the posterior updated after each build, until it reaches ``threshold``.
    Returns the builds that took (``reached`` False when the functions ran out first)."""
    state = ActiveId(candidates, prior, table, threshold=threshold)
    functions = _functions(functions)
    limit = max_builds if max_builds is not None else len(functions) * len(state.candidates)
    builds = 0
    for name, weight, _cost in functions:
        for recipe in state.candidates:
            if max(state.posterior.values()) >= threshold or builds >= limit:
                break
            outcome = bucket(build(name, recipe))
            builds += 1
            if outcome is not None:
                state.update(Option("build", name, recipe, None, weight, 1.0), outcome)
        if max(state.posterior.values()) >= threshold or builds >= limit:
            break
    post = state.posterior
    top = max(state.candidates, key=lambda h: (post[h], -state.candidates.index(h)))
    reached = post[top] >= threshold
    return {"builds": builds, "reached": reached, "decision": top if reached else None,
            "posterior_after": state.ranked(4)}
