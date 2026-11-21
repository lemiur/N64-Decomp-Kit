"""Evidence-driven fixed-point solver behind ``decompile-run --effort``.

``decompile-run`` used to make one pass: calibrate, sweep the resolution ladder, settle the
byte-identical cases that wait for a declaration.  This module turns that pass into the first
round of a loop and keeps the evidence each round produces:

* **effort modes** (``MODES``): a declarative policy per mode -- per-function build budget, size
  window, search ceiling, which opt-in techniques run in the first round, how many rounds, which
  techniques later rounds may schedule, which cascade steps run after each round, and whether the
  ranking may reorder work.  ``normal`` is today's ``decompile-run`` exactly (``LEGACY_DEFAULTS``);
* **one primary cause per unresolved case** (``primary_cause``): the review's table of thirteen
  causes, derived from the case's status, its ranked residual causes (``causes.TAXONOMY``), its
  size and score, and whether it lies in a stretch no known compiler explains;
* **applicable techniques only** (``CAUSES``, ``applicable``): each cause lists the techniques
  that can act on it; a cause without one names why automation stops (``STOP``);
* **ranking from this run's own observations** (``Observations``): P(exact | cause, recipe family,
  size bucket, technique) with a prior (measured numbers where they exist), divided by the
  expected cost; negative facts (an attempt that left a case's best source unchanged, a technique
  that plateaued for a cause) are recorded and expire when new evidence reaches the case;
* **the fixed-point loop** (``solve_image``): rounds of scheduled techniques, then the cascade
  (settlement of byte-identical cases, new evidence for callers and callees of new exact
  functions, recipe confidence), until a round brings no new exact function, no new declaration
  and no new evidence, or a budget (time, builds, model calls) or the mode's round limit is hit.
  Rounds go to ``<run>/rounds.jsonl``;
* **coverage and dossiers** (``coverage``, ``dossier``): every catalogued function either has a
  case or a recorded reason it was not attempted; every function still unresolved at the end has
  a JSON record of what was tried and why automation stopped.

Nothing here builds anything: the build work is done by callables ``decompile-run`` passes in
(``resolve-hard`` and ``declare_assumed``), so the loop is tested with fakes.  Credit never comes
from this module: a case is exact only when the strict verifier said so (``resolve._finish``);
the ranking changes the order of work and nothing else.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import handasm
from . import iterate as itermod
from . import resolve as resolvemod

# A function above this size is drafted and built under its recipe only (``commands.resolve``);
# measured: attempting more on such functions gave 0 exact in 964 builds.  Kept in every mode.
LARGE_FUNCTION = 0x400
CROSS_DRAFTS = resolvemod.CROSS_DRAFTS
# drafts x recipes one attempt may build.  Techniques are one dimension each: no mode forms the
# product recipes x flags x transforms.
MAX_COMBINATIONS = 128
MODEL_CALL_COST = 10           # builds one model call is weighed as when ranking (a convention, not measured)
PRIOR_WEIGHT = 10              # pseudo-observations a prior is worth at most
PLATEAU_ATTEMPTS = 12          # attempts of one technique on one cause with no yield before it is set aside
UNRESOLVED = ("open", "near", "does-not-build", "needs-source", "assumed", "blocked-boundary", "needs-tu",
              "probable-handwritten")
ALTERNATIVE_RENDERINGS = "--no-andor,--gotos-only"


# --------------------------------------------------------------------------- effort modes

@dataclass(frozen=True)
class Policy:
    name: str
    budget: int                      # builds per function (resolve --budget)
    max_size: int                    # 0: no size window
    search_ceiling: int | None       # None: [run] search_ceiling, else the resolver's measured default
    cross_recipes: int | None        # None: resolver default (8 recipes x 4 drafts)
    region_recipes: bool             # first round: region-recipe retry (measured 3/126)
    unit_recipes: bool               # first round: nearest-unit retry (measured 0/165; off in every mode)
    search_economics: bool           # first round: stop searching plateaued strata (measured -2.7% builds)
    refine: bool                     # a configured refiner may be asked
    max_rounds: int                  # 1: the first round only (today's decompile-run)
    schedule: str                    # "observe": record predictions, fixed order; "order": ranking orders work
    techniques: tuple[str, ...]      # what later rounds may schedule (and what predictions are made for)
    cascade: tuple[str, ...]         # after each round: declare, neighbours, recipes
    recipe_sweep: int                # alternative recipes in the recipe-sweep technique
    require_reasons: bool            # every function not attempted must carry a specific recorded reason
    # kitchen sink (round 12): everything implemented that skips work is off, everything opt-in on
    regional_calibration: bool = False   # calibrate each toolchain region on its own
    recipe_gate: bool = True             # skip unoptimised recipes without evidence (saves builds)

    def as_dict(self) -> dict:
        return asdict(self)


_LATER = ("search", "recipe-sweep", "declare", "refine")
MODES: dict[str, Policy] = {
    "quick": Policy("quick", budget=60, max_size=0x200, search_ceiling=20, cross_recipes=4, region_recipes=False,
                    unit_recipes=False, search_economics=True, refine=False, max_rounds=1, schedule="observe",
                    techniques=_LATER, cascade=(), recipe_sweep=8, require_reasons=False),
    "normal": Policy("normal", budget=200, max_size=0, search_ceiling=None, cross_recipes=None, region_recipes=False,
                     unit_recipes=False, search_economics=False, refine=True, max_rounds=1, schedule="observe",
                     techniques=_LATER, cascade=(), recipe_sweep=16, require_reasons=False),
    "deep": Policy("deep", budget=300, max_size=0, search_ceiling=None, cross_recipes=None, region_recipes=False,
                   unit_recipes=False, search_economics=False, refine=True, max_rounds=4, schedule="order",
                   techniques=_LATER, cascade=("declare", "neighbours", "recipes"), recipe_sweep=16,
                   require_reasons=False),
    # exhaustive is the kitchen sink: every implemented technique, nothing skipped to save builds
    # (the recipe gate and search economics off, region and unit recipe retries and regional
    # calibration on); the discovery repair and small-data stages are on in every mode
    "exhaustive": Policy("exhaustive", budget=400, max_size=0, search_ceiling=None, cross_recipes=12,
                         region_recipes=True, unit_recipes=True, search_economics=False, refine=True, max_rounds=8,
                         schedule="order", techniques=_LATER + ("draft-options",),
                         cascade=("declare", "neighbours", "recipes"), recipe_sweep=24, require_reasons=True,
                         regional_calibration=True, recipe_gate=False),
}
DEFAULT_MODE = "normal"
# decompile-run's option defaults before effort modes existed; ``normal`` must equal them
LEGACY_DEFAULTS = {"budget": 200, "max_size": 0, "search_ceiling": None, "cross_recipes": None,
                   "region_recipes": False, "unit_recipes": False, "search_economics": False}
OPTION_FIELDS = tuple(LEGACY_DEFAULTS)
# options a project's ``[run]`` table may also set.  One rule for both: an explicit command-line
# option, else the project's setting, else the mode.  apply_policy cannot fill them into ``args`` as
# it does OPTION_FIELDS (a filled option would hide the project's setting), so it records the mode's
# values in ``args.mode_settings``.  That is saved with the invocation and restored on --resume: the
# rounds a resumed run still has to do keep the values the run was started with, whatever --effort
# the resume names.
SETTING_FIELDS = ("recipe_gate", "regional_calibration")


def mode_settings(mode: Policy) -> dict:
    return {key: getattr(mode, key) for key in SETTING_FIELDS}


def setting(args, settings: dict, key: str, mode: Policy) -> bool:
    """The effective value of a SETTING_FIELDS option: the command line (None: not given), else the
    project's ``[run]`` setting, else the mode values the run recorded, else ``mode``."""
    value = getattr(args, key, None)
    if value is not None:
        return bool(value)
    if settings.get(key) is not None:
        return bool(settings[key])
    recorded = getattr(args, "mode_settings", None)
    if isinstance(recorded, dict) and recorded.get(key) is not None:
        return bool(recorded[key])
    return bool(getattr(mode, key))


def policy(name: str | None) -> Policy:
    if not name:
        return MODES[DEFAULT_MODE]
    if name not in MODES:
        raise ValueError(f"unknown effort mode {name!r}: one of {', '.join(MODES)}")
    return MODES[name]


def check_policy(mode: Policy) -> None:
    """Every mode keeps budgets and never forms a product larger than ``MAX_COMBINATIONS``."""
    if mode.budget <= 0:
        raise ValueError(f"{mode.name}: a per-function build budget is required")
    if mode.max_rounds < 1:
        raise ValueError(f"{mode.name}: at least one round")
    for recipes in (mode.cross_recipes or resolvemod.CROSS_RECIPES, mode.recipe_sweep):
        if recipes * CROSS_DRAFTS > MAX_COMBINATIONS:
            raise ValueError(f"{mode.name}: {recipes} recipes x {CROSS_DRAFTS} drafts exceeds {MAX_COMBINATIONS}")
    unknown = set(mode.techniques) - set(TECHNIQUES)
    if unknown:
        raise ValueError(f"{mode.name}: unknown techniques {sorted(unknown)}")
    if mode.schedule not in ("observe", "order"):
        raise ValueError(f"{mode.name}: schedule must be observe or order")


def apply_policy(args, mode: Policy) -> dict:
    """Fill every option the command line left unset (None) from the mode; return what was filled.
    An explicit option always wins over the mode."""
    filled = {}
    for key in OPTION_FIELDS:
        if getattr(args, key, None) is None:
            setattr(args, key, getattr(mode, key))
            filled[key] = getattr(mode, key)
    if getattr(args, "rounds", None) is None:
        args.rounds = mode.max_rounds
        filled["rounds"] = mode.max_rounds
    if getattr(args, "schedule", None) is None:
        args.schedule = mode.schedule
        filled["schedule"] = mode.schedule
    # the mode's SETTING_FIELDS values, recorded once (a resumed run has restored its own already)
    recorded = getattr(args, "mode_settings", None)
    recorded = dict(recorded) if isinstance(recorded, dict) else {}
    missing = {key: value for key, value in mode_settings(mode).items() if key not in recorded}
    if missing:
        args.mode_settings = {**recorded, **missing}
        filled["mode_settings"] = missing
    return filled


def stage_plan(mode: Policy, args, settings: dict) -> list[str]:
    """The stages ``decompile-run`` runs for one image, in order.  ``calibrate`` runs when a
    sample is configured or the code's own marks doubt the configured recipe.  ``rounds`` (later
    rounds of the fixed-point loop) appears only when more than one round is allowed; the solver's
    read-only outputs (coverage, dossiers) are not stages: they build nothing."""
    skip = set(getattr(args, "skip_stage", None) or ()) | set(getattr(args, "disabled", None) or ())
    plan = []
    if "boundaries" not in skip:
        plan.append("boundaries")
    if (getattr(args, "donors", None) or getattr(args, "library", None)) and "donors" not in skip:
        plan.append("donors")
    if "resolve" not in skip:
        plan += ["calibrate", "resolve"]
        if (getattr(args, "declare_assumed", None) or settings.get("declare_assumed")) and "declare" not in skip:
            plan.append("declare")
        rounds = getattr(args, "rounds", None)
        if (rounds if rounds is not None else mode.max_rounds) > 1:
            plan.append("rounds")
    plan.append("queue")
    if "ledger" not in skip:
        plan.append("ledger")
    return plan


# --------------------------------------------------------------------------- causes and techniques

@dataclass(frozen=True)
class Technique:
    kind: str                        # "build" | "model" | "declare"
    statuses: tuple[str, ...]        # case statuses it is run on
    meaning: str


TECHNIQUES: dict[str, Technique] = {
    "search": Technique("build", ("open", "near"),
                        "continue the variant search from the case's best source (families already closed "
                        "for that source are skipped)"),
    "recipe-sweep": Technique("build", ("open", "near", "does-not-build", "needs-source"),
                              "draft again and build the best drafts under more alternative recipes"),
    "draft-options": Technique("build", ("open", "near", "does-not-build", "needs-source"),
                               "draft again with alternative m2c renderings (measured 6/639)"),
    "refine": Technique("model", ("open", "near", "does-not-build", "needs-source"),
                        "ask the configured refiner; every answer is built and strictly verified"),
    "declare": Technique("declare", ("assumed",),
                         "declare self-describing D_<address> symbols and verify again"),
}

CAUSES: dict[str, tuple[str, tuple[str, ...]]] = {
    "no discovered image": ("the code is in no image the survey placed", ()),
    "uncertain boundary": ("the catalogue range is not one whole function", ()),
    "no buildable draft": ("no draft builds", ("refine", "draft-options", "recipe-sweep")),
    "wrong load width": ("a load or store has another width or signedness", ("search", "refine")),
    "global offsets differ": ("a global, constant or field offset differs", ("search", "refine")),
    "register-only residual": ("only register allocation differs", ("search", "refine")),
    "branch topology": ("branches, block order or delay slots differ", ("search", "refine")),
    "wrong stack frame": ("the stack frame differs", ("search", "refine")),
    "recipe uncertain": ("the compiler, its version or its flags are in doubt", ("recipe-sweep", "search")),
    "link/address mismatch": ("byte-identical or nearly so, but an address rests on an undeclared symbol",
                              ("declare", "search")),
    "large far-off": ("far from the target (above the search ceiling) or larger than 0x400",
                      ("recipe-sweep", "refine")),
    "unknown toolchain": ("in a stretch whose code idioms match no compiler in the idiom table (a style fingerprint, "
                          "not a build test)", ("recipe-sweep", "search", "refine")),
    "handwritten": ("hand-written code", ()),
}
STOP = {
    "no discovered image": "no image holds this code: the survey has to place it first",
    "uncertain boundary": "the range is not one whole function; boundary operations are proposed for review and "
                          "never applied unattended",
    "handwritten": "hand-written code: kept as assembly when it reassembles, never credited as C",
}

# measured yields (successes, trials) used as priors; sources are the kit's own records
PRIORS: dict[str, tuple[int, int, str]] = {
    "search": (1, 20, "assumed: every case already had one search in the first round"),
    "recipe-sweep": (1, 20, "assumed"),
    "draft-options": (6, 639, "alternative renderings measured 6/639"),
    "refine": (1, 10, "assumed"),
    "declare": (1, 4, "assumed"),
}
CAUSE_PRIORS: dict[tuple[str, str], tuple[int, int, str]] = {
    ("register-only residual", "search"): (17, 127, "register-only residuals measured 17/127"),
    ("large far-off", "recipe-sweep"): (0, 630, "searches starting above the ceiling measured 0/630"),
}


def size_bucket(size: int) -> str:
    return "<=0x40" if size <= 0x40 else "<=0x100" if size <= 0x100 else "<=0x400" if size <= 0x400 else ">0x400"


def recipe_family(recipe: str) -> str:
    pipeline = (recipe or "").partition(" ")[0]
    if not pipeline:
        return "none"
    if recipe == handasm.RECIPE:
        return "assembly"
    if pipeline.startswith("custom:"):
        return "custom"
    return "ido" if pipeline.startswith("ido") else "gcc"


_BY_TOP = {
    "compiler.family": "recipe uncertain", "assembler.variant": "recipe uncertain",
    "schedule.load_hazard": "recipe uncertain",
    "type.width": "wrong load width", "type.signedness": "wrong load width",
    "global.declaration": "global offsets differ", "type.volatile": "global offsets differ",
    "relocation.target": "link/address mismatch",
    "register.lifetime": "register-only residual", "register.permutation": "register-only residual",
    "register.pressure": "register-only residual",
    "control.branch_form": "branch topology", "control.branch_likely": "branch topology",
    "control.switch": "branch topology", "schedule.delay_slot": "branch topology",
    "semantic.unknown": "branch topology",
    "frame.size": "wrong stack frame",
    "tu.context": "uncertain boundary", "abi.nonstandard": "uncertain boundary",
    "source.missing": "no buildable draft", "assembly.handwritten": "handwritten",
}


def primary_cause(case, *, unknown=frozenset(), ceiling: int | None = resolvemod.SEARCH_CEILING,
                  large: int = LARGE_FUNCTION) -> str:
    """One cause of the review's table for an unresolved case (first match wins)."""
    confidence = {c.get("cause"): c.get("confidence", 0) for c in (case.causes or []) if isinstance(c, dict)}
    top = case.top_cause or ""
    if case.status == "blocked-boundary" or top.startswith("boundary."):
        return "uncertain boundary"
    if case.status in ("probable-handwritten", handasm.STATUS) or top == "assembly.handwritten":
        return "handwritten"
    if case.status == "needs-tu":
        return "handwritten" if confidence.get("assembly.handwritten", 0) >= 0.6 else "uncertain boundary"
    if case.status in ("needs-source", "does-not-build") or case.best_score is None:
        return "no buildable draft"
    if case.status == "assumed":
        return "link/address mismatch"
    if case.vram in unknown:
        return "unknown toolchain"
    if (ceiling and case.best_score > ceiling) or case.size > large:
        return "large far-off"
    return _BY_TOP.get(top, "recipe uncertain")


def _self_describing(case) -> bool:
    for item in (case.strict or {}).get("assumptions", []):
        try:
            address = int(str(item.get("address")), 16)
        except ValueError:
            continue
        if item.get("kind") == "symbol" and item.get("name") == f"D_{address:08X}":
            return True
    return False


def applicable(case, cause: str, mode: Policy, *, ceiling: int | None = resolvemod.SEARCH_CEILING,
               model: bool = False, large: int = LARGE_FUNCTION) -> tuple[list[str], dict[str, str]]:
    """Techniques of ``cause`` this mode allows that can act on this case, and why each other one
    of the cause's techniques cannot."""
    out, excluded = [], {}
    for name in CAUSES[cause][1]:
        technique = TECHNIQUES[name]
        why = None
        if name not in mode.techniques:
            why = f"not used in effort {mode.name}"
        elif case.status not in technique.statuses:
            why = f"not run on {case.status} cases"
        elif name in ("search", "recipe-sweep", "draft-options") and case.size > large:
            why = "larger than 0x400: drafted and built under its recipe only (measured 0 exact in 964 builds)"
        elif name == "search" and (case.best_score is None or (ceiling and case.best_score > ceiling)):
            why = f"best score {case.best_score} above the search ceiling {ceiling}"
        elif name == "search" and case.best_digest and \
                set(itermod.FAMILIES) <= set(case.closed_families.get(case.best_digest, [])):
            why = "every variant family is closed for the case's best source"
        elif name == "recipe-sweep" and mode.recipe_sweep <= 0:
            why = "no alternative recipes in this mode"
        elif name == "refine" and not (model and mode.refine):
            why = "no refiner configured"
        elif name == "refine" and any(stage.get("stage") == "refine" for stage in case.stages):
            why = "the refiner was already asked about this case"
        elif name == "declare" and not _self_describing(case):
            why = "no assumption is a self-describing D_<address> name: a person has to declare it"
        if why:
            excluded[name] = why
        else:
            out.append(name)
    return out, excluded


def technique_options(name: str, mode: Policy) -> dict:
    """``resolve-hard`` option overrides that run one technique on a named set of cases."""
    statuses = TECHNIQUES[name].statuses
    if name == "search":
        return {"only_statuses": statuses, "write_scope": False}
    if name == "recipe-sweep":
        return {"only_statuses": statuses, "write_scope": False, "fresh": True, "cross_recipes": mode.recipe_sweep}
    if name == "draft-options":
        return {"only_statuses": statuses, "write_scope": False, "fresh": True,
                "draft_options": ALTERNATIVE_RENDERINGS}
    if name == "refine":
        return {"only_statuses": statuses, "write_scope": False, "skip_sweep": True}
    return {}


# --------------------------------------------------------------------------- observations and ranking

def _key(*parts) -> str:
    return " | ".join(str(p) for p in parts)


class Observations:
    """Predictions and outcomes of this run, kept beside its rounds (``<run>/solver/observations.json``).

    ``cells``: (cause, recipe family, size bucket, technique) -> attempts, exact, improved, builds,
    model calls.  ``facts``: per case, every technique attempted (round, outcome, builds, best source
    before); an attempt that changed nothing is a negative fact until new evidence reaches the case.
    ``evidence``: case -> round of the latest new evidence (a caller or callee became exact)."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self.cells: dict[str, dict] = {}
        self.facts: dict[str, list[dict]] = {}
        self.evidence: dict[str, int] = {}
        self.plateaus: dict[str, int] = {}
        if self.path and self.path.is_file():
            data = json.loads(self.path.read_text())
            self.cells, self.facts = data.get("cells", {}), data.get("facts", {})
            self.evidence, self.plateaus = data.get("evidence", {}), data.get("plateaus", {})

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"cells": self.cells, "facts": self.facts, "evidence": self.evidence,
                                         "plateaus": self.plateaus}, indent=0, sort_keys=True))
        temporary.replace(self.path)

    # ---- estimates (sums over cells, rebuilt after each change: a round plans thousands of cases)
    def _sums(self) -> tuple[dict, dict]:
        if getattr(self, "_cached", None) is None or self._cached[0] != self._version():
            by_cause: dict[tuple[str, str], list[int]] = {}
            by_technique: dict[str, list[int]] = {}
            for key, row in self.cells.items():
                cause, _family, _bucket, technique = key.split(" | ")
                totals = by_cause.setdefault((cause, technique), [0, 0, 0, 0])
                for i, field_name in enumerate(("attempts", "exact", "builds", "improved")):
                    totals[i] += row[field_name]
                spent = by_technique.setdefault(technique, [0, 0, 0])
                for i, field_name in enumerate(("attempts", "builds", "calls")):
                    spent[i] += row[field_name]
            self._cached = (self._version(), by_cause, by_technique)
        return self._cached[1], self._cached[2]

    def _version(self) -> tuple[int, int]:
        # ``record`` counts changes; a cells mapping replaced from outside has another identity
        return id(self.cells), getattr(self, "_changes", 0)

    def _counts(self, cause: str, family: str | None, bucket: str | None, technique: str) -> tuple[int, int, int]:
        if family is None and bucket is None:
            attempts, exact, builds, _improved = self._sums()[0].get((cause, technique), (0, 0, 0, 0))
            return attempts, exact, builds
        row = self.cells.get(_key(cause, family, bucket, technique))
        return (row["attempts"], row["exact"], row["builds"]) if row else (0, 0, 0)

    def probability(self, cause: str, family: str, bucket: str, technique: str) -> float:
        """Posterior mean of P(exact): the prior, this cause's observations over every family and
        size, and the exact cell's own observations, which count twice."""
        successes, trials, _source = CAUSE_PRIORS.get((cause, technique), PRIORS[technique])
        weight = min(trials, PRIOR_WEIGHT)
        alpha = weight * successes / trials if trials else 0.5
        beta = weight - alpha
        n_all, k_all, _ = self._counts(cause, None, None, technique)
        n_cell, k_cell, _ = self._counts(cause, family, bucket, technique)
        alpha += k_all + k_cell
        beta += (n_all - k_all) + (n_cell - k_cell)
        return round(alpha / (alpha + beta), 4) if alpha + beta else 0.0

    def cost(self, technique: str, mode: Policy) -> float:
        """Expected builds of one attempt: this run's own mean when it has made one, else the mode's."""
        attempts, builds, calls = self._sums()[1].get(technique, (0, 0, 0))
        if attempts:
            return max(1.0, (builds + MODEL_CALL_COST * calls) / attempts)
        return float({"search": mode.budget, "recipe-sweep": CROSS_DRAFTS * mode.recipe_sweep + mode.budget // 2,
                      "draft-options": 12 + mode.budget // 2, "declare": 2,
                      "refine": 3 * MODEL_CALL_COST}[technique])

    def rank(self, case, cause: str, techniques: list[str], mode: Policy) -> list[dict]:
        family, bucket = recipe_family(case.recipe), size_bucket(case.size)
        rows = []
        for order, name in enumerate(techniques):
            p = self.probability(cause, family, bucket, name)
            cost = self.cost(name, mode)
            rows.append({"technique": name, "p_exact": p, "cost": round(cost, 1),
                         "yield_per_build": round(p / cost, 6), "order": order})
        if mode.schedule == "order":
            rows.sort(key=lambda row: (-row["yield_per_build"], row["order"]))
        return rows

    # ---- facts
    def blocked(self, case, technique: str) -> dict | None:
        """The negative fact that rules out repeating ``technique`` on this case, if any: the same
        technique already ran on the same best source and changed nothing, and no evidence arrived
        since."""
        latest = self.evidence.get(case.case_id, 0)
        for fact in reversed(self.facts.get(case.case_id, [])):
            if fact["technique"] == technique and fact["outcome"] == "no-change" \
                    and fact.get("digest") == case.best_digest and fact["round"] >= latest:
                return fact
        return None

    def plateaued(self, cause: str, technique: str) -> bool:
        attempts, exact, _builds, improved = self._sums()[0].get((cause, technique), (0, 0, 0, 0))
        return attempts >= PLATEAU_ATTEMPTS and exact == 0 and improved == 0

    def record(self, before, after, technique: str, cause: str, round_no: int, builds: int, calls: int = 0,
               predicted: float | None = None) -> dict:
        outcome = _outcome(before, after)
        key = _key(cause, recipe_family(before.recipe), size_bucket(before.size), technique)
        row = self.cells.setdefault(key, {"attempts": 0, "exact": 0, "improved": 0, "builds": 0, "calls": 0})
        self._changes = getattr(self, "_changes", 0) + 1
        row["attempts"] += 1
        row["exact"] += outcome == "exact"
        row["improved"] += outcome in ("exact", "improved", "assumed")
        row["builds"] += builds
        row["calls"] += calls
        fact = {"technique": technique, "round": round_no, "outcome": outcome, "builds": builds, "calls": calls,
                "digest": before.best_digest, "score_before": before.best_score,
                "score_after": after.best_score if after else None,
                "status_after": after.status if after else None, "predicted": predicted}
        self.facts.setdefault(before.case_id, []).append(fact)
        if self.plateaued(cause, technique) and _key(cause, technique) not in self.plateaus:
            self.plateaus[_key(cause, technique)] = round_no
        return fact

    def mark_evidence(self, case_id: str, round_no: int) -> bool:
        if self.evidence.get(case_id, -1) >= round_no:
            return False
        self.evidence[case_id] = round_no
        return True


_RANK = {"exact": 0, "assumed": 1, "near": 2}


def _outcome(before, after) -> str:
    if after is None:
        return "error"
    if after.status == "exact" and before.status != "exact":
        return "exact"
    if after.status == "assumed" and before.status not in ("exact", "assumed"):
        return "assumed"
    if _RANK.get(after.status, 3) < _RANK.get(before.status, 3) or (
            after.best_score is not None and (before.best_score is None or after.best_score < before.best_score)):
        return "improved"
    return "no-change"


# --------------------------------------------------------------------------- planning

@dataclass
class Decision:
    case_id: str
    name: str
    vram: int
    size: int
    status: str
    cause: str
    technique: str | None
    p_exact: float | None
    cost: float | None
    ranked: list[dict]
    excluded: dict[str, str]
    reason: str                      # why nothing is scheduled (when technique is None)

    def as_dict(self) -> dict:
        return asdict(self)


def decide(case, mode: Policy, observations: Observations, *, unknown=frozenset(),
           ceiling: int | None = resolvemod.SEARCH_CEILING, model: bool = False) -> Decision:
    cause = primary_cause(case, unknown=unknown, ceiling=ceiling)
    usable, excluded = applicable(case, cause, mode, ceiling=ceiling, model=model)
    kept = []
    for name in usable:
        fact = observations.blocked(case, name)
        if fact is not None:
            excluded[name] = f"round {fact['round']} changed nothing on the same best source"
        elif mode.schedule == "order" and observations.plateaued(cause, name):
            excluded[name] = f"plateaued for {cause}: {PLATEAU_ATTEMPTS}+ attempts without a gain this run"
        else:
            kept.append(name)
    ranked = observations.rank(case, cause, kept, mode)
    if ranked:
        top = ranked[0]
        return Decision(case.case_id, case.name, case.vram, case.size, case.status, cause, top["technique"],
                        top["p_exact"], top["cost"], ranked, excluded, "")
    reason = STOP.get(cause) or ("; ".join(f"{k}: {v}" for k, v in excluded.items())
                                 or "no technique acts on this cause")
    return Decision(case.case_id, case.name, case.vram, case.size, case.status, cause, None, None, None, [],
                    excluded, reason)


def plan(cases, mode: Policy, observations: Observations, **context) -> tuple[list[tuple[str, list[Decision]]],
                                                                               list[Decision]]:
    """Batches ``[(technique, decisions)]`` for one round and every decision made.  In ``order``
    mode batches and their members are ordered by expected exact functions per build; otherwise
    by the technique table and by size, smallest first (the sweep's own order)."""
    decisions = [decide(case, mode, observations, **context) for case in cases if case.status in UNRESOLVED]
    groups: dict[str, list[Decision]] = {}
    for decision in decisions:
        if decision.technique:
            groups.setdefault(decision.technique, []).append(decision)
    table = list(TECHNIQUES)
    batches = []
    for name, members in groups.items():
        if mode.schedule == "order":
            members.sort(key=lambda d: (-(d.p_exact or 0) / max(1.0, d.cost or 1.0), d.size, d.vram))
        else:
            members.sort(key=lambda d: (d.size, d.vram))
        batches.append((name, members))
    if mode.schedule == "order":
        batches.sort(key=lambda batch: (-sum((d.p_exact or 0) / max(1.0, d.cost or 1.0) for d in batch[1])
                                        / len(batch[1]), table.index(batch[0])))
    else:
        batches.sort(key=lambda batch: table.index(batch[0]))
    return batches, decisions


# --------------------------------------------------------------------------- budgets, log, credit

class Budget:
    """Run-level budgets for one invocation.  Checked before every batch: a batch that starts within
    budget runs to its end, so one batch is the most a budget can be overrun by."""

    def __init__(self, seconds: float | None = None, builds: int | None = None, model_calls: int | None = None,
                 clock=time.monotonic):
        self.seconds, self.builds, self.model_calls, self.clock = seconds, builds, model_calls, clock
        self.started = clock()
        self.spent_builds = self.spent_calls = 0

    def spend(self, builds: int = 0, calls: int = 0) -> None:
        self.spent_builds += max(0, builds)
        self.spent_calls += max(0, calls)

    def elapsed(self) -> float:
        return self.clock() - self.started

    def builds_left(self) -> int | None:
        return None if self.builds is None else max(0, self.builds - self.spent_builds)

    def calls_left(self) -> int | None:
        return None if self.model_calls is None else max(0, self.model_calls - self.spent_calls)

    def exhausted(self) -> str | None:
        if self.seconds is not None and self.elapsed() >= self.seconds:
            return f"time budget {self.seconds:g} s reached"
        if self.builds is not None and self.spent_builds >= self.builds:
            return f"build budget {self.builds} reached"
        if self.model_calls is not None and self.spent_calls >= self.model_calls:
            return f"model-call budget {self.model_calls} reached"
        return None

    def as_dict(self) -> dict:
        return {"seconds": self.seconds, "builds": self.builds, "model_calls": self.model_calls,
                "elapsed": round(self.elapsed(), 1), "spent_builds": self.spent_builds,
                "spent_model_calls": self.spent_calls}


def parse_duration(text: str | None) -> float | None:
    """``90``, ``90s``, ``30m``, ``2h`` -> seconds."""
    if text in (None, ""):
        return None
    text = str(text).strip().lower()
    scale = {"s": 1, "m": 60, "h": 3600}.get(text[-1:], None)
    return float(text[:-1]) * scale if scale else float(text)


class RoundsLog:
    """``<run>/rounds.jsonl``: one line per completed round and one per stop, tagged with the
    invocation's session (a resumed invocation continues its session)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def read(self) -> list[dict]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue                                   # a line cut by an interruption
        return out

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torn = False                          # a torn last line (power loss, full disk) must not
        try:                                  # swallow this record (CHAOS_TESTING.md finding C4)
            with self.path.open("rb") as existing:
                existing.seek(0, os.SEEK_END)
                if existing.tell():
                    existing.seek(-1, os.SEEK_END)
                    torn = existing.read(1) != b"\n"
        except FileNotFoundError:
            pass
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(("\n" if torn else "") + json.dumps(record, default=str, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def latest_session(self) -> str | None:
        sessions = [r.get("session") for r in self.read() if r.get("session")]
        return sessions[-1] if sessions else None

    def progress(self, session: str, image: str) -> tuple[int, dict | None]:
        """(last completed round, the stop record if the image stopped) for one image of a session."""
        last, stop = 0, None
        for record in self.read():
            if record.get("session") != session or record.get("image") != image:
                continue
            if record.get("kind") == "round":
                last = max(last, int(record.get("round", 0)))
            elif record.get("kind") == "stop":
                stop = record
        return last, stop


def credit(cases) -> dict:
    """Case-store credit, kept apart: exact C, byte-identical awaiting declarations, hand-written
    assembly kept as assembly, everything else unresolved.  (Generated assembly, data, opaque and
    unknown bytes are the export's and the ledger's categories, not the case store's.)"""
    out = {"exact_c": 0, "awaiting_declarations": 0, "handwritten_assembly": 0, "unresolved": 0,
           "exact_c_bytes": 0}
    for case in cases:
        if case.status == "exact" and (case.strict or {}).get("verdict") == "exact" and case.recipe != handasm.RECIPE:
            out["exact_c"] += 1
            out["exact_c_bytes"] += case.size
        elif case.status == "assumed":
            out["awaiting_declarations"] += 1
        elif case.status == handasm.STATUS:
            out["handwritten_assembly"] += 1
        elif case.status != "stale":
            out["unresolved"] += 1
    return out


def builds_spent(before, after) -> int:
    """Builds one attempt cost, from the stored case before and after it.  A rerun that ended
    further away keeps the earlier case and adds its builds to it; one that ended as far or nearer
    replaces the case, whose count then starts at this attempt."""
    if after is None:
        return 0
    if before is None:
        return after.builds
    if after.updated_ns == before.updated_ns:
        return 0
    if after.builds >= before.builds and after.stages[:len(before.stages)] == before.stages:
        return after.builds - before.builds
    return after.builds


def snapshot(store_root: Path, image: str) -> dict:
    store = resolvemod.CaseStore(store_root)
    try:
        return {case.case_id: case for case in store.all() if case.image == image}
    finally:
        store.close()


# --------------------------------------------------------------------------- the loop

@dataclass
class Hooks:
    """What the loop calls; ``decompile-run`` binds them to ``resolve-hard`` and ``declare_assumed``.

    ``sweep() -> dict``: the first round (calibrate, resolve sweep, configured declaration); may
    return ``model_calls`` and ``declared`` (True when it already settled assumed cases).
    ``technique(name, decisions, calls_left) -> dict``: run one technique on these cases; may return
    ``model_calls`` and ``error``.  ``declare() -> dict``: settle assumed cases; may return
    ``names`` (the declarations it holds).  ``neighbours(vram) -> set[int]``: callers and callees."""
    sweep: object
    technique: object
    declare: object | None = None
    neighbours: object | None = None


def solve_image(*, image: str, store_root: Path, mode: Policy, budget: Budget, log: RoundsLog,
                observations: Observations, hooks: Hooks, session: str, rounds: int | None = None,
                start_round: int = 1, unknown=frozenset(), ceiling: int | None = resolvemod.SEARCH_CEILING,
                model: bool = False, emit=None, clock=time.monotonic) -> dict:
    """Run rounds for one image until the fixed point, the round limit or a budget.  Rounds before
    ``start_round`` are taken as done (a resume).  Returns the stop record."""
    limit = rounds if rounds is not None else mode.max_rounds
    declared_names: set[str] = set()
    round_no = start_round
    stop = None
    while stop is None:
        if round_no > limit:
            stop = f"round limit: effort {mode.name} allows {limit} round(s)"
            break
        why = budget.exhausted()
        if why:
            stop = why
            break
        started = clock()
        before = snapshot(store_root, image)
        record = {"kind": "round", "session": session, "image": image, "round": round_no, "mode": mode.name,
                  "schedule": mode.schedule, "techniques": {}, "cascade": {}, "skipped": {}}
        if round_no == 1:
            result = hooks.sweep() or {}
            after = snapshot(store_root, image)
            spent = sum(builds_spent(before.get(k), c) for k, c in after.items())
            budget.spend(spent, int(result.get("model_calls") or 0))
            record["sieve"] = "sweep"
            record["techniques"]["sweep"] = {"cases": len(after), "builds": spent,
                                             "model_calls": int(result.get("model_calls") or 0),
                                             "exact": _new_exact(before, after)}
            if result.get("declared"):
                for case in after.values():
                    if case.status == "assumed":
                        observations.facts.setdefault(case.case_id, []).append(
                            {"technique": "declare", "round": round_no, "outcome": "no-change", "builds": 0,
                             "calls": 0, "digest": case.best_digest, "score_before": case.best_score,
                             "score_after": case.best_score, "status_after": "assumed", "predicted": None})
        else:
            batches, decisions = plan(before.values(), mode, observations,
                                      unknown=unknown() if callable(unknown) else unknown, ceiling=ceiling,
                                      model=model)
            record["sieve"] = ", ".join(name for name, _members in batches if name != "declare") or "cascade"
            for name, members in batches:
                if name == "declare":
                    continue                               # settlement runs in the cascade below
                why = budget.exhausted()
                if why:
                    record["skipped"][name] = {"cases": len(members), "reason": why}
                    continue
                members = _within_budget(name, members, budget, observations, mode, record)
                if not members:
                    continue
                result = _run_batch(name, members, hooks, budget, observations, store_root, image, round_no)
                record["techniques"][name] = result
            after = snapshot(store_root, image)
        # ---- cascade
        if "declare" in mode.cascade and hooks.declare is not None and not budget.exhausted():
            record["cascade"]["declare"] = _cascade_declare(hooks, store_root, image, observations, budget,
                                                            round_no, declared_names)
            after = snapshot(store_root, image)
        new_exact = _new_exact(before, after)
        if "neighbours" in mode.cascade and hooks.neighbours is not None:
            marked = 0
            by_vram: dict[int, list] = {}
            for other in after.values():
                if other.status in UNRESOLVED:
                    by_vram.setdefault(other.vram, []).append(other)
            for case_id in new_exact_ids(before, after):
                for vram in hooks.neighbours(after[case_id].vram):
                    for other in by_vram.get(vram, ()):
                        marked += observations.mark_evidence(other.case_id, round_no + 1)
            record["cascade"]["neighbours"] = {"marked": marked}
        if "recipes" in mode.cascade:
            record["cascade"]["recipes"] = recipe_confidence(after.values())
        totals = credit(after.values())
        gains = {"exact_c": totals["exact_c"] - credit(before.values())["exact_c"],
                 "awaiting_declarations": totals["awaiting_declarations"]
                 - credit(before.values())["awaiting_declarations"],
                 "handwritten_assembly": totals["handwritten_assembly"]
                 - credit(before.values())["handwritten_assembly"],
                 "new_exact": new_exact,
                 "declarations": int(record["cascade"].get("declare", {}).get("new_names", 0)),
                 "evidence": _evidence(before, after) + int(record["cascade"].get("neighbours", {}).get("marked", 0)),
                 "negative_facts": sum(int(row.get("no_change", 0)) for row in record["techniques"].values())
                 + int(record["cascade"].get("declare", {}).get("no_change", 0))}
        record.update({"gains": gains, "totals": totals, "seconds": round(clock() - started, 1),
                       "budget": budget.as_dict()})
        log.append(record)
        observations.save()
        if emit:
            emit("round", image=image, round=round_no, sieve=record["sieve"], gains=gains)
        progress = gains["new_exact"] + gains["declarations"] + gains["evidence"] + gains["negative_facts"]
        if round_no > 1 and progress == 0:
            if record["skipped"]:
                stop = "budget: what is left does not cover one more attempt (" + "; ".join(
                    f"{k}: {v['reason']}" for k, v in record["skipped"].items()) + ")"
            elif not record["techniques"] and not record["cascade"].get("declare", {}).get("ran"):
                stop = "fixed point: no unresolved case has an applicable technique left in this mode"
            else:
                stop = ("fixed point: a full round brought no new exact function, declaration, evidence or "
                        "negative fact")
        round_no += 1
    final = {"kind": "stop", "session": session, "image": image, "after_round": round_no - 1, "reason": stop,
             "mode": mode.name, "budget": budget.as_dict()}
    log.append(final)
    return final


def new_exact_ids(before: dict, after: dict) -> list[str]:
    return [k for k, c in after.items() if c.status == "exact" and (before.get(k) is None or
                                                                    before[k].status != "exact")]


def _new_exact(before: dict, after: dict) -> int:
    return len(new_exact_ids(before, after))


def _evidence(before: dict, after: dict) -> int:
    """Cases whose status, score or top cause changed (new information about them)."""
    count = 0
    for key, case in after.items():
        old = before.get(key)
        if old is None or (old.status, old.best_score, old.top_cause) != (case.status, case.best_score,
                                                                           case.top_cause):
            count += 1
    return count


def _within_budget(name, members, budget: Budget, observations: Observations, mode: Policy, record) -> list:
    left = budget.builds_left()
    if TECHNIQUES[name].kind == "model":
        calls = budget.calls_left()
        if calls is not None and calls <= 0:
            record["skipped"][name] = {"cases": len(members), "reason": "model-call budget reached"}
            return []
        return members
    if left is None:
        return members
    each = observations.cost(name, mode)
    fit = int(left // max(1.0, each))
    if fit < len(members):
        record["skipped"][name] = {"cases": len(members) - fit,
                                   "reason": f"build budget: {left} builds left, about {each:.0f} per case"}
    return members[:fit]


def _run_batch(name, members, hooks: Hooks, budget: Budget, observations: Observations, store_root, image,
               round_no) -> dict:
    ids = {d.case_id for d in members}
    before = {k: c for k, c in snapshot(store_root, image).items() if k in ids}
    try:
        result = hooks.technique(name, members, budget.calls_left()) or {}
    except Exception as error:                                # a technique failure is recorded, not fatal
        result = {"error": f"{type(error).__name__}: {error}"[-300:]}
    after = {k: c for k, c in snapshot(store_root, image).items() if k in ids}
    calls = int(result.get("model_calls") or 0)
    spent_total, exact, unchanged = 0, 0, 0
    by_id = {d.case_id: d for d in members}
    for key, old in before.items():
        new = after.get(key)
        spent = builds_spent(old, new)
        spent_total += spent
        fact = observations.record(old, new, name, by_id[key].cause, round_no, spent,
                                   calls // max(1, len(members)), by_id[key].p_exact)
        exact += fact["outcome"] == "exact"
        unchanged += fact["outcome"] == "no-change"
    budget.spend(spent_total, calls)
    out = {"cases": len(members), "builds": spent_total, "model_calls": calls, "exact": exact,
           "no_change": unchanged, "predicted_exact": round(sum(d.p_exact or 0 for d in members), 2)}
    if result.get("error"):
        out["error"] = result["error"]
    return out


def _cascade_declare(hooks: Hooks, store_root, image, observations: Observations, budget: Budget, round_no,
                     declared_names: set) -> dict:
    cases = snapshot(store_root, image)
    waiting = [c for c in cases.values() if c.status == "assumed" and _self_describing(c)
               and observations.blocked(c, "declare") is None]
    if not waiting:
        return {"ran": False, "waiting": 0, "new_names": 0}
    try:
        result = hooks.declare() or {}
    except Exception as error:
        result = {"error": f"{type(error).__name__}: {error}"[-300:]}
    after = snapshot(store_root, image)
    spent = unchanged = 0
    for case in waiting:
        new = after.get(case.case_id)
        cost = builds_spent(case, new)
        spent += cost
        fact = observations.record(case, new, "declare", "link/address mismatch", round_no, cost)
        unchanged += fact["outcome"] == "no-change"
    budget.spend(spent)
    names = set(result.get("names") or [])
    fresh = names - declared_names
    declared_names |= names
    out = {"ran": True, "waiting": len(waiting), "builds": spent, "new_names": len(fresh), "no_change": unchanged,
           "promoted": sum(1 for c in waiting if after.get(c.case_id) and after[c.case_id].status == "exact")}
    if result.get("error"):
        out["error"] = result["error"]
    return out


def recipe_confidence(cases) -> dict:
    """Recipes of the image's exact cases, most confirmed first (the recipe confidence update)."""
    tally: dict[str, int] = {}
    for case in cases:
        if case.status == "exact" and case.recipe and case.recipe != handasm.RECIPE:
            tally[case.recipe] = tally.get(case.recipe, 0) + 1
    total = sum(tally.values())
    return {label: {"exact": n, "share": round(n / total, 3)}
            for label, n in sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))[:8]}


# --------------------------------------------------------------------------- call graph, coverage, dossiers

def call_graph(data: bytes, delta: int, rows) -> dict[int, set[int]]:
    """Callers and callees of each catalogued function, from its ``jal`` instructions."""
    starts = {vram for _name, vram, _size in rows}
    links: dict[int, set[int]] = {vram: set() for vram in starts}
    for _name, vram, size in rows:
        offset = vram - delta
        if offset < 0 or offset + size > len(data):
            continue
        for at in range(offset, offset + size - 3, 4):
            word = int.from_bytes(data[at:at + 4], "big")
            if word >> 26 == 3:
                target = ((vram + at - offset) & 0xF0000000) | ((word & 0x03FFFFFF) << 2)
                if target in starts and target != vram:
                    links[vram].add(target)
                    links[target].add(vram)
    return links


class Stretches:
    """The unknown-toolchain stretches ``resolve-hard`` reported (its ``unknown_toolchain`` entry):
    ``vram in Stretches(info)`` tells whether a function lies in one."""

    def __init__(self, info=None):
        rows = (info or {}).get("stretches", []) if isinstance(info, dict) else (info or [])
        self.spans = []
        for row in rows:
            try:
                self.spans.append((int(str(row["start"]), 16), int(str(row["end"]), 16)))
            except (KeyError, TypeError, ValueError):
                continue

    def __contains__(self, vram) -> bool:
        return any(start <= vram <= end for start, end in self.spans)

    def __bool__(self) -> bool:
        return bool(self.spans)


def coverage(rows, cases, *, recovered=frozenset(), left_out=None, covered=None, image_size: int | None = None,
             delta: int = 0, min_size: int = 4, large: int = LARGE_FUNCTION) -> dict:
    """Every catalogued function with what happened to it: a case (and its status), or the reason it
    was not attempted.  Nothing is left out silently: what no rule explains is listed as
    ``unexplained``."""
    left_out = left_out or {}
    by_vram = {case.vram: case for case in cases}
    entries, unexplained = [], 0
    for name, vram, size in sorted(rows, key=lambda r: r[1]):
        entry = {"name": name, "vram": f"0x{vram:08X}", "size": size}
        case = by_vram.get(vram)
        if vram in recovered:
            entry.update(attempted=False, reason="recovered from a donor (proven source of another title)")
        elif case is not None:
            entry.update(attempted=True, status=case.status)
            if size > large:
                entry["limited"] = ("drafted and built under its recipe only: alternative recipes and the variant "
                                    "search are withheld above 0x400 (measured 0 exact in 964 builds)")
        elif vram in left_out:
            entry.update(attempted=False, reason=left_out[vram])
        elif covered is not None and covered(vram):
            entry.update(attempted=False, reason="already matched by a project manifest row")
        elif image_size is not None and not (0 <= vram - delta <= image_size - size):
            entry.update(attempted=False, reason="no discovered image: the range is outside the image's bytes")
        elif size < min_size:
            entry.update(attempted=False, reason=f"smaller than the resolver's minimum of {min_size} bytes")
        else:
            entry.update(attempted=False, reason="unexplained: no case and no recorded reason (the resolve stage "
                                                 "did not run, failed, or the run stopped before this function)",
                         unexplained=True)
            unexplained += 1
        entries.append(entry)
    reasons: dict[str, int] = {}
    for entry in entries:
        if not entry["attempted"]:
            reasons[entry["reason"]] = reasons.get(entry["reason"], 0) + 1
    return {"functions": len(entries), "attempted": sum(e["attempted"] for e in entries),
            "not_attempted": sum(not e["attempted"] for e in entries), "unexplained": unexplained,
            "reasons": reasons, "entries": entries}


_TECHNIQUE_STAGES = ("drafts", "drafts-x-recipes", "frame-fit", "recipe", "iterate", "iterate-skipped", "refine",
                     "unit-recipe", "region-recipe", "twin", "strict-verify", "source", "baseline")


def reopen_on(case, decision: "Decision", region: dict | None) -> list[str]:
    """What new evidence would make this function worth another attempt (the solver's evidence
    triggers, plus what the dossier itself names as missing)."""
    out = ["a caller, callee or catalogue neighbour becomes exact (new evidence marks it for the next round)"]
    if case.status == "assumed" or (case.top_cause or "").startswith(("symbol", "reloc", "data")):
        out.append("a declaration for a symbol it references is settled")
    if region and region.get("verdict") in ("unknown", "mismatch"):
        parts = sorted({m.get("component") for m in region.get("component_mismatches") or [] if m.get("component")})
        out.append("a tool chain that produces " + (", ".join(parts) if parts else "this region's code") +
                   " becomes available (the region is not explained by any installed one)")
    if (case.top_cause or "").startswith("frame"):
        out.append("type or structure evidence that changes the stack frame (a local's size)")
    if case.status in ("does-not-build", "needs-source"):
        out.append("a draft that builds (new declarations or a decompiler rendering)")
    if decision.technique is None and decision.reason:
        out.append("an effort mode that allows a technique excluded here: " + str(decision.reason)[:160])
    return out


def dossier(case, decision: Decision, observations: Observations, *, related=(), boundary=(), stop: str = "",
            image_recipes: dict | None = None, region: dict | None = None) -> dict:
    """The hard-queue record of one function still unresolved at the end of a run."""
    attempted, recipes = [], []
    for stage in case.stages:
        kind = stage.get("stage")
        if kind not in _TECHNIQUE_STAGES:
            continue
        row = {"technique": kind}
        if kind == "drafts":
            row.update(builds=len(stage.get("variants", [])),
                       outcome=min((v["score"] for v in stage.get("variants", []) if v.get("score") is not None),
                                   default=None))
        elif kind in ("drafts-x-recipes", "frame-fit", "recipe"):
            row.update(builds=stage.get("tried"), chosen=stage.get("chosen"), outcome=stage.get("score"))
            if stage.get("chosen"):
                recipes.append(stage["chosen"])
        elif kind == "iterate":
            row.update(builds=stage.get("builds"), outcome=f"{stage.get('start')} -> {stage.get('final')}",
                       families=stage.get("families"))
        elif kind == "iterate-skipped":
            row.update(builds=0, outcome=stage.get("reason") or f"score {stage.get('score')} above ceiling "
                                                                f"{stage.get('ceiling')}")
        elif kind == "refine":
            row.update(model_calls=stage.get("calls"), outcome="improved" if stage.get("improved") else "no gain")
        elif kind in ("unit-recipe", "region-recipe"):
            row.update(recipe=stage.get("recipe"), outcome="kept" if stage.get("kept") else "not kept")
            recipes.append(stage.get("recipe"))
        elif kind == "baseline":
            row.update(builds=1, outcome=stage.get("score") if "score" in stage else "does not build")
        else:
            row.update({k: v for k, v in stage.items() if k != "stage"})
        attempted.append(row)
    solver_rows = observations.facts.get(case.case_id, [])
    boundary_status = "blocked: " + case.top_cause if case.status == "blocked-boundary" else \
        ("proposed operations: " + ", ".join(sorted({op.get("kind", "?") for op in boundary}))
         if boundary else "catalogue range accepted")
    return {"image": case.image, "name": case.name, "case": case.case_id, "vram": f"0x{case.vram:08X}",
            "size": case.size, "status": case.status, "boundary": boundary_status,
            "best_recipe": case.recipe, "recipes_tried": sorted({r for r in recipes if r and r != case.recipe}),
            "image_recipes": image_recipes or {},
            "best_score": case.best_score, "residual": case.causes[:3], "primary_cause": decision.cause,
            "ladder": attempted, "solver_attempts": solver_rows,
            # the stored case's own count (a rerun that ended nearer replaces it) and what the
            # solver's later rounds measured for this case
            "builds": {"case": case.builds, "solver_rounds": sum(int(r.get("builds") or 0) for r in solver_rows),
                       "model_calls": sum(int(r.get("calls") or 0) for r in solver_rows)},
            "related_exact": list(related), "next_action": decision.technique,
            "ranked_actions": decision.ranked, "excluded_actions": decision.excluded,
            "toolchain_region": None if region is None else {
                "range": f"{region.get('start')}..{region.get('end')}", "verdict": region.get("verdict"),
                "ranked": (region.get("ranked") or [])[:2],
                "component_mismatches": sorted({m.get("component") for m in region.get("component_mismatches") or []
                                                if m.get("component")}),
                "unknown_marks": region.get("unknown_marks") or []},
            "transformations_of_best_source": _provenance(case.stages),
            "why_stopped": decision.reason or stop,
            "reopen_on": reopen_on(case, decision, region)}


def _provenance(stages) -> dict:
    from .transform_classes import provenance
    return provenance(stages)


# --------------------------------------------------------------------------- attempt ledger and sieve

# iterate families -> the sieve stage whose kind of change they make
_FAMILY_STAGE = {"types": 7, "fields": 7, "casts": 7, "volatile": 7, "pointer": 7, "globals": 6,
                 "control": 8, "expression": 8, "ast-control": 8, "ast-expr": 8, "ast-cast": 7, "callargs": 8,
                 "lifetime": 9, "order": 9, "scope": 9, "split": 9, "ast-temp": 9, "ast-inline-temp": 9,
                 "ast-stmt-order": 9, "ast-decl-order": 9, "ast-block": 9, "ast-sameline": 9}
SIEVE_STAGES = {
    0: "inventory and byte ledger", 1: "function and image discovery", 2: "regional toolchain reconstruction",
    3: "SDK, library and known-source donors", 4: "exact as first drafted", 5: "primary decompiler drafts",
    6: "compile repair and declaration settlement", 7: "types, signedness, pointers, volatility, stack frames",
    8: "control-flow and expression forms", 9: "registers, lifetimes and scheduling",
    10: "structural and large-function recovery", 11: "model refinement",
    12: "verified assembly, raw-word fallback and dossiers"}


def attempts(cases, observations: Observations) -> list[dict]:
    """The durable attempt ledger of an image: one entry per method that ran on a case, from the
    case's own stages (round 1) and the solver's facts (later rounds), with the recipe, the best
    source digest, the evidence version (round of the latest new evidence), the builds and the
    outcome.  Rebuilt from the store each time, so a resumed run never records an attempt twice."""
    out = []
    for case in sorted(cases, key=lambda c: (c.image, c.vram)):
        evidence = observations.evidence.get(case.case_id, 0)
        base = {"case": case.case_id, "name": case.name, "evidence_version": evidence}
        for stage in case.stages:
            kind = stage.get("stage")
            if kind == "drafts":
                scores = [v.get("score") for v in stage.get("variants") or []]
                out.append({**base, "round": 1, "method": "drafts", "builds": len(scores),
                            "outcome": {"best": min((x for x in scores if x is not None), default=None)}})
            elif kind in ("drafts-x-recipes", "recipe", "frame-fit"):
                out.append({**base, "round": 1, "method": kind, "builds": int(stage.get("tried") or 0),
                            "outcome": {"chosen": stage.get("chosen"), "score": stage.get("score")}})
            elif kind == "iterate":
                out.append({**base, "round": 1, "method": "search", "builds": int(stage.get("builds") or 0),
                            "outcome": {"start": stage.get("start"), "final": stage.get("final"),
                                        "helpful": stage.get("helpful_families") or []}})
            elif kind in ("iterate-skipped", "recipe-skipped"):
                out.append({**base, "round": 1, "method": kind, "builds": 0,
                            "outcome": {k: v for k, v in stage.items() if k != "stage"}})
        for fact in observations.facts.get(case.case_id, []):
            out.append({**base, "round": fact.get("round"), "method": fact.get("technique"),
                        "builds": fact.get("builds"), "model_calls": fact.get("calls"),
                        "digest_before": fact.get("digest"),
                        "outcome": {"result": fact.get("outcome"), "score_before": fact.get("score_before"),
                                    "score_after": fact.get("score_after")}})
        out.append({**base, "round": None, "method": "final", "recipe": case.recipe, "digest": case.best_digest,
                    "builds": case.builds, "outcome": {"status": case.status, "score": case.best_score,
                                                       "top_cause": case.top_cause}})
    return out


def settled_by(case) -> int | None:
    """The sieve stage that settled an exact, assumed or assembly case."""
    if case.status == handasm.STATUS:
        return 12
    if case.status not in ("exact", "assumed"):
        return None
    stages = {s.get("stage"): s for s in case.stages}
    if any(s.get("stage") == "refine" and (s.get("exact") or s.get("score") == 0) for s in case.stages):
        return 11
    drafts = stages.get("drafts") or {}
    if any(v.get("score") == 0 for v in drafts.get("variants") or []):
        return 4
    if (stages.get("drafts-x-recipes") or {}).get("score") == 0 or (stages.get("recipe") or {}).get("score") == 0:
        return 2
    if (stages.get("frame-fit") or {}).get("score") == 0:
        return 7
    search = stages.get("iterate") or {}
    if search.get("final") == 0:
        helpful = search.get("helpful_families") or []
        return min((_FAMILY_STAGE.get(f, 9) for f in helpful), default=9)
    if any(s.get("stage") == "strict-verify" and s.get("origin") == "re-verified" for s in case.stages):
        return 6
    return 5


def sieve(cases, *, recovered: int = 0, discovery: dict | None = None) -> dict:
    """Where every function of an image ended, by sieve stage: functions settled at each stage, the
    builds each stage spent, and what is left unresolved with its primary cause."""
    settled = {number: 0 for number in SIEVE_STAGES}
    settled[3] = recovered
    builds = {number: 0 for number in SIEVE_STAGES}
    left = 0
    for case in cases:
        stage = settled_by(case)
        if stage is not None:
            settled[stage] += 1
        else:
            left += 1
        for row in case.stages:
            kind = row.get("stage")
            if kind == "drafts":
                builds[5] += len(row.get("variants") or [])
            elif kind in ("drafts-x-recipes", "recipe"):
                builds[2] += int(row.get("tried") or 0)
            elif kind == "frame-fit":
                builds[7] += int(row.get("tried") or 0)
            elif kind == "iterate":
                builds[8] += int(row.get("builds") or 0)
    return {"stages": {f"{n} {name}": {"settled": settled[n], "builds": builds[n]} for n, name in SIEVE_STAGES.items()},
            "unresolved": left, "discovery": discovery or {},
            "note": "stage 10 (structural, large functions) is not built: its go/no-go measurement decides "
                    "(docs/large-functions.md); search builds are counted under stage 8"}


def boundary_operations(path: Path) -> dict[int, list[dict]]:
    """Proposed boundary operations by the vram of each catalogue row they would remove."""
    out: dict[int, list[dict]] = {}
    try:
        operations = json.loads(Path(path).read_text()).get("operations", [])
    except (OSError, ValueError, AttributeError):
        return out
    for op in operations:
        for row in op.get("remove", []) or []:
            try:
                vram = int(str(row.get("vram")), 16) if isinstance(row, dict) else None
            except ValueError:
                continue
            if vram is not None:
                out.setdefault(vram, []).append(op)
    return out


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, default=str))
    temporary.replace(path)
