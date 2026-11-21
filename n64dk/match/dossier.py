"""The unresolved dossier: every function a run did not settle, why, what may run next and at what cost.

``decompile-run`` writes ``<out>/solver/unresolved_dossier.json`` at its end (always; it builds
nothing) and ``decompile-export`` renders it as ``UNRESOLVED.md``.  For each function that is not
settled (a case of an unresolved status, a function that was not attempted, or a donor unit that
waits for a declaration) the dossier gives:

1. the reason: ``match.reasons``' code and next step, and the solver's primary cause;
2. the eligible next techniques: the solver's ranked actions for the case (``solver.decide``, as
   the run recorded them in ``solver/hard_queue_<image>.json``), which already leave out what was
   attempted without effect (a negative fact) and what cannot act on the case;
3. the expected cost of each, in builds and seconds, only from what this run measured: builds per
   attempt from ``solver/observations.json`` (by cause, recipe family and size, else by cause,
   else over the technique), seconds per build from ``profile.jsonl`` (the ``round:<technique>``
   stages).  A technique this run never attempted costs ``unmeasured``: the solver's own planning
   convention (``Observations.cost`` without observations) is not a measurement and is not used;
4. the evidence change that would justify a retry (``reasons.RETRY`` with the case's own detail:
   the symbols to declare, the region that needs a recipe, the score the search ceiling needs,
   the boundary operation), plus the solver's ``reopen_on`` list.

It also keeps a **technique ledger** per case: every technique of the kit (the round-1 ladder's
rungs of ``resolve-hard``, the later-round techniques of ``solver.TECHNIQUES``, and large-function
reconstruction, which the kit does not have) with its state: ``attempted``, ``exhausted`` (ran and
may not run again on the same source), ``eligible``, ``not-in-mode`` (another effort mode has it),
``withheld`` (a governor or gate setting), ``switched-off`` (an ablation switch), ``inapplicable``
(with why), ``not-run`` or ``unavailable``.  An exhaustive run is complete when no case has an
eligible technique left; otherwise the dossier says which stop (round limit, budget) left them.

Everything here reads records; nothing builds, reads the ROM, or changes a case or an outcome.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from . import iterate as itermod
from . import reasons as reasonsmod
from . import solver

FILE = "unresolved_dossier.json"
SCHEMA = "n64dk-unresolved-dossier/1"
UNMEASURED = "unmeasured"
LARGE = solver.LARGE_FUNCTION
STRUCTURAL = "large-function-reconstruction"
LADDER = ("ladder:drafts", "ladder:drafts-x-recipes", "ladder:recipe-retry", "ladder:frame-fit", "ladder:search",
          "ladder:twin", "ladder:region-recipe", "ladder:unit-recipe")
LATER = tuple(solver.TECHNIQUES)
TECHNIQUES = LADDER + LATER + (STRUCTURAL,)
STATES = ("attempted", "exhausted", "eligible", "not-in-mode", "withheld", "switched-off", "inapplicable", "not-run",
          "unavailable")
_WHY = 200                                  # characters of one ledger reason
_MARKDOWN_ROWS = 500                        # per-function rows in UNRESOLVED.md (largest first)

# solver.applicable / solver.decide exclusion texts -> ledger state (first matching prefix)
_EXCLUDED = (
    ("not run on ", "inapplicable"),
    ("larger than 0x400", "inapplicable"),
    ("best score ", "inapplicable"),
    ("every variant family is closed", "exhausted"),
    ("no alternative recipes in this mode", "not-in-mode"),
    ("no refiner configured", "inapplicable"),
    ("the refiner was already asked", "exhausted"),
    ("no assumption is a self-describing", "inapplicable"),
    ("plateaued for ", "exhausted"),
)
_NO_CHANGE = re.compile(r"^round \d+ changed nothing")
# statuses the ladder reaches from the target alone, before any draft (match.resolve step 1)
_TARGET_FINDINGS = ("blocked-boundary", "needs-tu", "probable-handwritten")
_STRUCTURAL_WHY = ("not in the kit: sieve stage 10 (structural, large-function recovery) is not built; "
                   "docs/large-functions.md holds its measurement gate")


def _short(text) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= _WHY else text[:_WHY - 3] + "..."


def _row(technique: str, state: str, why: str = "", **extra) -> dict:
    out = {"technique": technique, "state": state}
    if why:
        out["why"] = _short(why)
    out.update({key: value for key, value in extra.items() if value is not None})
    return out


# --------------------------------------------------------------------------- the technique ledger

def classify_exclusion(name: str, why: str, mode, disabled=()) -> tuple[str, str]:
    """Ledger state of a later-round technique the solver excluded, from the solver's own reason."""
    disabled = set(disabled or ())
    if name == "refine" and disabled & {"refine", "round-refine"}:
        return "switched-off", why + " (the refiner is switched off for this run)"
    if why.startswith("not used in effort"):
        full = solver.MODES.get(getattr(mode, "name", ""))
        if full is not None and name in full.techniques:
            return "switched-off", why + " (an ablation switch removed it from the mode)"
        modes = [label for label, policy in solver.MODES.items() if name in policy.techniques and policy.max_rounds > 1]
        return "not-in-mode", why + (f"; effort {', '.join(modes)} uses it" if modes else "")
    if _NO_CHANGE.match(why):
        return "exhausted", why
    for prefix, state in _EXCLUDED:
        if why.startswith(prefix):
            return state, why
    return "inapplicable", why


def _closed_all(case) -> bool:
    closed = (case.closed_families or {}).get(case.best_digest or "", [])
    return bool(case.best_digest) and set(itermod.FAMILIES) <= set(closed)


def ladder(case, disabled=()) -> list[dict]:
    """Round-1 rungs of ``resolve-hard``'s ladder for one case, from the case's stages."""
    disabled = set(disabled or ())
    if case.status in _TARGET_FINDINGS:
        why = f"the ladder stopped at its target finding ({case.status}: {case.top_cause or 'no cause recorded'})"
        return [_row(name, "inapplicable", why) for name in LADDER]
    stages = [stage for stage in case.stages or [] if isinstance(stage, dict)]

    def find(kind):
        return [stage for stage in stages if stage.get("stage") == kind]
    large = case.size > LARGE
    out = []
    # drafts
    drafts = find("drafts")
    failed = [stage for stage in find("source") if stage.get("origin") == "draft" and stage.get("error") is not None]
    if drafts:
        variants = [v for stage in drafts for v in stage.get("variants") or [] if isinstance(v, dict)]
        scores = [v["score"] for v in variants if v.get("score") is not None]
        out.append(_row("ladder:drafts", "attempted", builds=len(variants),
                        result=f"best {min(scores)}" if scores else "no draft built"))
    elif failed:
        out.append(_row("ladder:drafts", "attempted", builds=0, result=_short("no draft: " + str(failed[-1]["error"]))))
    elif "draft" in disabled:
        out.append(_row("ladder:drafts", "switched-off", "m2c drafting is switched off for this run"))
    else:
        out.append(_row("ladder:drafts", "not-run", "the case continued from a stored or supplied source"))
    # drafts x recipes
    rows = find("drafts-x-recipes")
    if rows:
        out.append(_row("ladder:drafts-x-recipes", "attempted", builds=sum(int(r.get("tried") or 0) for r in rows),
                        result=f"chosen {rows[-1].get('chosen')}, score {rows[-1].get('score')}"))
    elif large:
        out.append(_row("ladder:drafts-x-recipes", "inapplicable",
                        "larger than 0x400: alternative recipes are withheld (measured 0 exact in 964 builds)"))
    elif disabled & {"cross-recipes", "recipes"}:
        out.append(_row("ladder:drafts-x-recipes", "switched-off", "drafts x recipes is switched off for this run"))
    else:
        out.append(_row("ladder:drafts-x-recipes", "not-run", "no near-miss draft to rebuild (the first draft was "
                                                              "exact, or none built)"))
    # recipe retry on the baseline source
    rows, skipped = find("recipe"), find("recipe-skipped")
    if rows:
        out.append(_row("ladder:recipe-retry", "attempted", builds=sum(int(r.get("tried") or 0) for r in rows),
                        result=f"chosen {rows[-1].get('chosen')}, score {rows[-1].get('score')}"))
    elif skipped:
        out.append(_row("ladder:recipe-retry", "withheld", f"the recipe gate: {skipped[-1].get('reason')} "
                                                           "(the gate is off in effort exhaustive)"))
    elif large:
        out.append(_row("ladder:recipe-retry", "inapplicable", "larger than 0x400: alternative recipes are withheld"))
    elif "recipes" in disabled:
        out.append(_row("ladder:recipe-retry", "switched-off", "recipe selection is switched off for this run"))
    else:
        out.append(_row("ladder:recipe-retry", "not-run", "the baseline was close enough under a recipe of the "
                                                          "detected compiler family"))
    # frame fit
    rows = find("frame-fit")
    if rows:
        out.append(_row("ladder:frame-fit", "attempted", builds=sum(int(r.get("tried") or 0) for r in rows),
                        result=f"score {rows[-1].get('score')}"))
    elif "frame-fit" in disabled:
        out.append(_row("ladder:frame-fit", "switched-off", "frame fit is switched off for this run"))
    else:
        out.append(_row("ladder:frame-fit", "not-run", "its precondition did not hold on this case"))
    # the variant search inside the ladder
    rows, skipped = find("iterate"), find("iterate-skipped")
    if rows:
        state = "exhausted" if _closed_all(case) else "attempted"
        out.append(_row("ladder:search", state, "every variant family is closed for the best source"
                        if state == "exhausted" else "", builds=sum(int(r.get("builds") or 0) for r in rows),
                        result=f"{rows[0].get('start')} -> {rows[-1].get('final')}"))
    elif skipped:
        last = skipped[-1]
        if last.get("ceiling") is not None and not last.get("reason"):
            why = f"best score {last.get('score')} above the search ceiling {last.get('ceiling')}"
            if large:
                why += " (above 0x400 the ceiling is 1: no search)"
            out.append(_row("ladder:search", "inapplicable", why))
        elif "switched off" in str(last.get("reason")):
            out.append(_row("ladder:search", "switched-off", str(last.get("reason"))))
        else:
            out.append(_row("ladder:search", "withheld", f"search economics: {last.get('reason')}"))
    elif "search" in disabled:
        out.append(_row("ladder:search", "switched-off", "the variant search is switched off for this run"))
    else:
        out.append(_row("ladder:search", "not-run", "no search allowance was left, or the case ended before it"))
    # twin, region and unit recipes
    rows = find("twin")
    out.append(_row("ladder:twin", "attempted", result=f"source of {rows[-1].get('of')}") if rows else
               _row("ladder:twin", "not-run", "no exact function with the same masked machine code"))
    for kind, option in (("region-recipe", "--region-recipes"), ("unit-recipe", "--unit-recipes")):
        rows = find(kind)
        out.append(_row(f"ladder:{kind}", "attempted", result=f"{rows[-1].get('recipe')}: "
                        + ("kept" if rows[-1].get("kept") else "not kept")) if rows else
                   _row(f"ladder:{kind}", "not-run", f"not run on this case ({option}; on in effort exhaustive)"))
    return out


def later(case, record: dict | None, mode, disabled=()) -> list[dict]:
    """Later-round techniques (``solver.TECHNIQUES``) for one case, from its hard-queue record:
    ranked actions are ``eligible``; the solver's exclusions keep their reason."""
    record = record or {}
    cause = record.get("primary_cause") or ""
    excluded = record.get("excluded_actions") or {}
    ranked = [row.get("technique") for row in record.get("ranked_actions") or [] if isinstance(row, dict)]
    facts = [fact for fact in record.get("solver_attempts") or [] if isinstance(fact, dict)]
    acting = solver.CAUSES.get(cause, ("", ()))[1]
    out = []
    for name in LATER:
        mine = [fact for fact in facts if fact.get("technique") == name]
        extra = {"attempts": len(mine), "builds": sum(int(f.get("builds") or 0) for f in mine),
                 "result": mine[-1].get("outcome")} if mine else {}
        if name in ranked:
            out.append(_row(name, "eligible", **extra))
        elif name in excluded:
            state, why = classify_exclusion(name, str(excluded[name]), mode, disabled)
            out.append(_row(name, state, why, **extra))
        elif cause and name not in acting:
            why = solver.STOP.get(cause) or f"does not act on its primary cause ({cause})"
            out.append(_row(name, "inapplicable", why, **extra))
        else:
            out.append(_row(name, "attempted" if mine else "not-run",
                            "" if mine else "no solver decision was recorded for this case", **extra))
    return out


def structural(case) -> dict:
    if case.size > LARGE:
        return _row(STRUCTURAL, "unavailable", _STRUCTURAL_WHY)
    return _row(STRUCTURAL, "inapplicable", "0x400 bytes or smaller")


# --------------------------------------------------------------------------- measured cost

def _zero() -> dict:
    return {"attempts": 0, "builds": 0, "model_calls": 0}


def technique_stats(cells: dict | None, rounds: list | None, profile: list | None) -> dict:
    """What this run measured per later-round technique.  ``cells``: ``Observations.cells``
    (cause | recipe family | size bucket | technique -> attempts, builds, calls); ``rounds``:
    ``rounds.jsonl`` records (cases per technique batch); ``profile``: ``profile.jsonl`` records
    (wall seconds and builds requested of each ``round:<technique>`` stage; ``declare`` and the
    round-1 ``resolve`` stage too)."""
    techniques: dict[str, dict] = {}
    by_cause: dict[str, dict] = {}
    by_cell: dict[str, dict] = {}

    def slot(name: str) -> dict:
        return techniques.setdefault(name, {**_zero(), "round_cases": 0, "profiled_seconds": 0.0,
                                            "profiled_builds": 0})
    for key, row in (cells or {}).items():
        parts = str(key).split(" | ")
        if len(parts) != 4 or not isinstance(row, dict):
            continue
        cause, _family, _bucket, name = parts
        values = (int(row.get("attempts") or 0), int(row.get("builds") or 0), int(row.get("calls") or 0))
        for target in (slot(name), by_cause.setdefault(f"{cause} | {name}", _zero()), by_cell.setdefault(key, _zero())):
            target["attempts"] += values[0]
            target["builds"] += values[1]
            target["model_calls"] += values[2]
    for record in rounds or []:
        if not isinstance(record, dict) or record.get("kind") != "round":
            continue
        for name, row in (record.get("techniques") or {}).items():
            if isinstance(row, dict):
                target = slot(name)
                target["round_cases"] += int(row.get("cases") or 0)
                if name == "sweep":                      # round 1: its own builds (no observation cell)
                    target["attempts"] += int(row.get("cases") or 0)
                    target["builds"] += int(row.get("builds") or 0)
        cascade = (record.get("cascade") or {}).get("declare")
        if isinstance(cascade, dict) and cascade.get("ran"):     # the declare cascade (cells hold its builds)
            slot("declare")["round_cases"] += int(cascade.get("waiting") or 0)
    for record in profile or []:
        if not isinstance(record, dict):
            continue
        stage = str(record.get("stage") or "")
        name = stage[len("round:"):] if stage.startswith("round:") else \
            {"declare": "declare", "resolve": "sweep"}.get(stage)
        if not name:
            continue
        target = slot(name)
        target["profiled_seconds"] = round(target["profiled_seconds"] + float(record.get("wall_s") or 0), 3)
        target["profiled_builds"] += int((record.get("builds") or {}).get("requested") or 0)
    for row in techniques.values():
        row["builds_per_attempt"] = round(row["builds"] / row["attempts"], 2) if row["attempts"] else UNMEASURED
        row["seconds_per_build"] = (round(row["profiled_seconds"] / row["profiled_builds"], 4)
                                    if row["profiled_builds"] and row["profiled_seconds"] else UNMEASURED)
        row["seconds_per_attempt"] = (round(row["profiled_seconds"] / row["round_cases"], 3)
                                      if row["round_cases"] and row["profiled_seconds"] else UNMEASURED)
    return {"techniques": techniques, "by_cause": by_cause, "cells": by_cell}


def expected_cost(stats: dict, name: str, cause: str, family: str, bucket: str) -> dict:
    """Expected builds and seconds of one attempt of ``name`` on a case, from this run's own
    measurements only (the most specific stratum that has an attempt); ``unmeasured`` otherwise."""
    found = None
    for basis, table, key in (("this run: same cause, recipe family and size", stats.get("cells") or {},
                               f"{cause} | {family} | {bucket} | {name}"),
                              ("this run: same cause", stats.get("by_cause") or {}, f"{cause} | {name}"),
                              ("this run: every case", stats.get("techniques") or {}, name)):
        row = table.get(key)
        if row and row.get("attempts"):
            found = (basis, row)
            break
    if found is None:
        return {"builds": UNMEASURED, "seconds": UNMEASURED, "basis": "no attempt of this technique in this run"}
    basis, row = found
    builds = row["builds"] / row["attempts"]
    measured = (stats.get("techniques") or {}).get(name) or {}
    per_build, per_attempt = measured.get("seconds_per_build"), measured.get("seconds_per_attempt")
    if builds > 0 and isinstance(per_build, (int, float)):
        seconds = round(builds * per_build, 1)
    elif isinstance(per_attempt, (int, float)):
        seconds = per_attempt
    else:
        seconds = UNMEASURED
    out = {"builds": round(builds, 1), "seconds": seconds, "basis": basis, "attempts": row["attempts"]}
    if row.get("model_calls"):
        out["model_calls"] = round(row["model_calls"] / row["attempts"], 2)
    return out


# --------------------------------------------------------------------------- evidence that justifies a retry

def _assumed(case) -> list[dict]:
    return [item for item in reasonsmod._assumptions(case) if isinstance(item, dict)] if case is not None else []


def evidence_change(code: str, case=None, record: dict | None = None, *, not_attempted: str = "",
                    awaiting: dict | None = None) -> list[str]:
    """What would justify another attempt: the case's own detail first, then the code's general
    change (``reasons.RETRY``), then the solver's ``reopen_on`` list."""
    record = record or {}
    out = []
    if code in ("needs-symbol", "data-not-owned"):
        for item in _assumed(case):
            kind, name, address = item.get("kind"), item.get("name"), item.get("address")
            if kind == "section":
                out.append(f"a declared base of section {name}" + (f" ({address} was inferred)" if address else ""))
            elif name:
                out.append(f"a declaration of {name}" + (f" at {address}" if address else ""))
        for name, address in sorted((awaiting or {}).items()):
            out.append(f"a declared base of section {name} ({address} was inferred)" if str(name).startswith(".")
                       else f"a declaration of {name} at {address}")
    elif code in ("no-recipe-identified", "style-unrecognised", "assembler-or-schedule"):
        region = record.get("toolchain_region") or {}
        if region.get("range"):
            parts = region.get("component_mismatches") or []
            out.append(f"a recipe for region {region['range']}" + (f" (components that differ: {', '.join(parts)})"
                                                                     if parts else ""))
    elif code == "far-off-draft":
        ceiling = next((stage.get("ceiling") for stage in (case.stages if case is not None else [])
                        if isinstance(stage, dict) and stage.get("stage") == "iterate-skipped"
                        and stage.get("ceiling") is not None), None)
        if ceiling is not None:
            out.append(f"a draft scoring at most {ceiling} (best is {case.best_score})")
    elif code in ("embedded-data", "boundary-uncertain") and record.get("boundary"):
        out.append(f"an accepted boundary operation ({record['boundary']})")
    elif code in ("decompiler-failed", "draft-does-not-build", "strict-verify-failed") and case is not None and case.next:
        out.append(_short(case.next))
    elif not_attempted and case is None:
        out.append(_short(f"the recorded reason no longer holds: {not_attempted}"))
    general = reasonsmod.retry_of(code)[1]
    if general:
        out.append(general)
    for item in record.get("reopen_on") or []:
        if isinstance(item, str) and item not in out:
            out.append(_short(item))
    return out


# --------------------------------------------------------------------------- one entry, the whole dossier

def _family_bucket(case) -> tuple[str, str]:
    return solver.recipe_family(case.recipe), solver.size_bucket(case.size)


def case_entry(case, record: dict | None, mode, stats: dict, *, disabled=(), unit_class: str = "") -> dict:
    """The dossier entry of one unresolved case."""
    record = record or {}
    outcome = "awaiting-symbols" if case.status == "assumed" else case.status
    hints = {key: record.get(key) for key in ("primary_cause", "next_action", "why_stopped")}
    code, step = reasonsmod.explain(outcome, case, unit_class=unit_class, dossier=hints)
    rows = ladder(case, disabled) + later(case, record, mode, disabled) + [structural(case)]
    family, bucket = _family_bucket(case)
    cause = record.get("primary_cause") or ""
    eligible = []
    for row in rows:
        if row["state"] == "eligible":
            eligible.append({"technique": row["technique"],
                             "cost": expected_cost(stats, row["technique"], cause, family, bucket)})
    when = reasonsmod.retry_of(code)[0]
    if eligible and when != "never":                  # not a C target stays never, whatever is eligible
        when = "now"
    return {"image": case.image, "name": case.name, "vram": f"0x{case.vram:08X}", "size": case.size,
            "status": case.status, "best_score": case.best_score, "recipe": case.recipe, "unit_class": unit_class,
            "reason_code": code, "reason": reasonsmod.CODES.get(code, ("", ""))[0], "next_step": step,
            "primary_cause": cause, "why_stopped": _short(record.get("why_stopped")),
            "eligible": eligible, "retry": when,
            # with a technique still eligible no new evidence is needed now; the list says what
            # would justify another attempt once those are spent
            "evidence_change": evidence_change(code, case, record),
            "ledger": rows}


def other_entry(image: str, name: str, vram: int, size: int, *, outcome: str, reason: str = "",
                awaiting: dict | None = None, unit_class: str = "") -> dict:
    """The entry of a function without a case: not attempted, or a donor unit awaiting a declaration."""
    code, step = reasonsmod.explain(outcome, None, unit_class=unit_class, not_attempted=reason)
    return {"image": image, "name": name, "vram": f"0x{vram:08X}", "size": size, "status": outcome,
            "best_score": None, "recipe": "", "unit_class": unit_class, "reason_code": code,
            "reason": reasonsmod.CODES.get(code, ("", ""))[0], "next_step": step, "primary_cause": "",
            "why_stopped": _short(reason), "eligible": [], "retry": reasonsmod.retry_of(code)[0],
            "evidence_change": evidence_change(code, None, None, not_attempted=reason, awaiting=awaiting),
            "ledger": []}


def summarise(entries: list[dict], mode, stats: dict, stops: dict) -> dict:
    by_reason: dict[str, dict] = {}
    by_retry: dict[str, int] = {}
    eligible: dict[str, dict] = {}
    matrix: dict[str, dict[str, int]] = {name: {} for name in TECHNIQUES}
    large = {"cases": 0, "unavailable": 0, "search_inapplicable": 0}
    far_off = 0
    left = 0
    for entry in entries:
        group = by_reason.setdefault(entry["reason_code"], {"functions": 0, "bytes": 0,
                                                            "retry": reasonsmod.retry_of(entry["reason_code"])[0]})
        group["functions"] += 1
        group["bytes"] += int(entry["size"] or 0)
        by_retry[entry["retry"]] = by_retry.get(entry["retry"], 0) + 1
        for item in entry["eligible"]:
            total = eligible.setdefault(item["technique"], {"functions": 0, "expected_builds": 0.0,
                                                            "expected_seconds": 0.0, "unmeasured": 0})
            total["functions"] += 1
            cost = item["cost"]
            if isinstance(cost.get("builds"), (int, float)):
                total["expected_builds"] = round(total["expected_builds"] + cost["builds"], 1)
            else:
                total["unmeasured"] += 1
            if isinstance(cost.get("seconds"), (int, float)):
                total["expected_seconds"] = round(total["expected_seconds"] + cost["seconds"], 1)
        left += bool(entry["eligible"])
        states = {row["technique"]: row["state"] for row in entry["ledger"]}
        for name, state in states.items():
            matrix.setdefault(name, {})
            matrix[name][state] = matrix[name].get(state, 0) + 1
        if entry["ledger"] and int(entry["size"] or 0) > LARGE:
            large["cases"] += 1
            large["unavailable"] += states.get(STRUCTURAL) == "unavailable"
            large["search_inapplicable"] += states.get("ladder:search") == "inapplicable"
        elif states.get("ladder:search") == "inapplicable":
            far_off += 1
    name = getattr(mode, "name", "")
    return {"functions": len(entries), "bytes": sum(int(e["size"] or 0) for e in entries),
            "by_reason": dict(sorted(by_reason.items(), key=lambda kv: (-kv[1]["functions"], kv[0]))),
            "by_retry": by_retry, "eligible": eligible,
            "ledger": {key: value for key, value in matrix.items() if value},
            "large_functions": large, "far_off_drafts": far_off,
            "exhaustion": {"mode": name, "max_rounds": getattr(mode, "max_rounds", None),
                           "cases_with_eligible_left": left, "complete": left == 0, "stops": stops,
                           "note": ("every applicable technique was attempted or is exhausted" if left == 0 else
                                    f"{left} cases still have an eligible technique; the stops say why the run ended")},
            "technique_stats": stats.get("techniques") or {}}


def build(*, cases, records: dict, coverage: dict, donors: dict, cells: dict, rounds: list, profile: list, mode,
          disabled=(), unit_class=None) -> dict:
    """The dossier from records already read: ``cases`` (``resolve.Case``), ``records`` (image ->
    vram -> hard-queue record), ``coverage`` (image -> coverage entries), ``donors`` (image -> vram
    -> awaited names of a donor unit), ``cells`` (observation cells), ``rounds``, ``profile``.
    ``unit_class(vram, size) -> str`` is optional."""
    stats = technique_stats(cells, rounds, profile)

    def unit(vram, size):
        try:
            return unit_class(vram, size) if unit_class is not None else ""
        except Exception:                                    # noqa: BLE001 - a label, never the dossier's failure
            return ""
    entries = []
    seen = set()
    for case in sorted(cases, key=lambda c: (c.image, c.vram)):
        if case.status not in solver.UNRESOLVED:
            continue
        seen.add((case.image, case.vram))
        entries.append(case_entry(case, (records.get(case.image) or {}).get(case.vram), mode, stats,
                                  disabled=disabled, unit_class=unit(case.vram, case.size)))
    for image in sorted(set(coverage) | set(donors)):
        awaiting = donors.get(image) or {}
        listed = set()
        for item in coverage.get(image) or []:
            vram = _vram(item.get("vram"))
            if vram is None or item.get("attempted") or (image, vram) in seen:
                continue
            listed.add(vram)
            size = int(item.get("size") or 0)
            if vram in awaiting:
                entries.append(other_entry(image, str(item.get("name")), vram, size, outcome="donor-awaiting-symbols",
                                           awaiting=awaiting[vram], unit_class=unit(vram, size)))
            elif not str(item.get("reason") or "").startswith("recovered from a donor"):
                entries.append(other_entry(image, str(item.get("name")), vram, size, outcome="not attempted",
                                           reason=str(item.get("reason") or ""), unit_class=unit(vram, size)))
        for vram, names in sorted(awaiting.items()):             # a donor unit no coverage file lists
            if vram not in listed and (image, vram) not in seen:
                entries.append(other_entry(image, f"func_{vram:08X}", vram, 0, outcome="donor-awaiting-symbols",
                                           awaiting=names))
    stops = {}
    for record in rounds or []:
        if isinstance(record, dict) and record.get("kind") == "stop" and record.get("image"):
            stops[record["image"]] = record.get("reason")
    return {"schema": SCHEMA, "mode": getattr(mode, "name", ""), "disabled": sorted(disabled or ()),
            "summary": summarise(entries, mode, stats, stops), "functions": entries}


# --------------------------------------------------------------------------- files

def _vram(text) -> int | None:
    try:
        return int(str(text), 16)
    except ValueError:
        return None


def _json(path: Path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def _jsonl(path: Path) -> list:
    out = []
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue                                         # a line cut by an interruption
    return out


def _images(folder: Path) -> list[str]:
    names = set()
    for prefix in ("hard_queue_", "coverage_"):
        names.update(path.name[len(prefix):-len(".json")] for path in folder.glob(f"{prefix}*.json"))
    return sorted(names)


def _donor_awaiting(run: Path, image: str) -> dict[int, dict]:
    held = _json(run / image / "stage_donors.json", {})
    out: dict[int, dict] = {}
    for row in (held.get("assumed") or []) if isinstance(held, dict) else []:
        if not isinstance(row, dict):
            continue
        for vram in row.get("function_vrams") or []:
            try:
                out[int(vram)] = dict(row.get("awaiting") or {})
            except (TypeError, ValueError):
                continue
    return out


def collect(run: Path, mode=None, *, disabled=None, project: Path | None = None) -> dict:
    """The dossier of a run directory, from its own records (read-only).  ``mode`` and
    ``disabled`` default to what ``solver/invocation.json`` recorded."""
    from . import resolve as resolvemod
    run = Path(run)
    folder = run / "solver"
    invocation = _json(folder / "invocation.json", {})
    if mode is None:
        try:
            mode = solver.policy(invocation.get("effort") if isinstance(invocation, dict) else None)
        except ValueError:
            mode = solver.policy(None)
    if disabled is None:
        disabled = (invocation.get("disabled") if isinstance(invocation, dict) else None) or ()
    cases = []
    if (run / "state" / "cases.sqlite").is_file():
        store = resolvemod.CaseStore(run / "state", readonly=True)
        try:
            cases = store.all()
        finally:
            store.close()
    records, coverage, donors = {}, {}, {}
    for image in _images(folder):
        queue = _json(folder / f"hard_queue_{image}.json", [])
        records[image] = {_vram(r.get("vram")): r for r in queue if isinstance(r, dict) and _vram(r.get("vram")) is not None} \
            if isinstance(queue, list) else {}
        found = _json(folder / f"coverage_{image}.json", {})
        coverage[image] = found.get("entries") or [] if isinstance(found, dict) else []
        donors[image] = _donor_awaiting(run, image)
    observations = _json(folder / "observations.json", {})
    unit_class = None
    if project is not None:
        try:
            from ..analysis import unitclass
            evidence = unitclass.load(Path(project))
            unit_class = lambda vram, size: unitclass.classify(vram, size, evidence)  # noqa: E731
        except Exception:                                    # noqa: BLE001 - a label, never the dossier's failure
            unit_class = None
    return build(cases=cases, records=records, coverage=coverage, donors=donors,
                 cells=observations.get("cells") if isinstance(observations, dict) else {},
                 rounds=_jsonl(run / "rounds.jsonl"), profile=_jsonl(run / "profile.jsonl"), mode=mode,
                 disabled=disabled, unit_class=unit_class)


def write(run: Path, mode=None, *, disabled=None, project: Path | None = None) -> dict:
    """Write ``<run>/solver/unresolved_dossier.json`` and return its headline (for report.json)."""
    found = collect(run, mode, disabled=disabled, project=project)
    path = Path(run) / "solver" / FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(found, separators=(",", ":"), default=str) + "\n")
    temporary.replace(path)
    summary = found["summary"]
    return {"file": str(path), "functions": summary["functions"], "by_retry": summary["by_retry"],
            "cases_with_eligible_left": summary["exhaustion"]["cases_with_eligible_left"],
            "exhaustion_complete": summary["exhaustion"]["complete"]}


def load(run: Path) -> dict | None:
    found = _json(Path(run) / "solver" / FILE, None)
    return found if isinstance(found, dict) and found.get("schema") == SCHEMA else None


# --------------------------------------------------------------------------- UNRESOLVED.md

def _cost(cost: dict) -> str:
    builds, seconds = cost.get("builds"), cost.get("seconds")
    if builds == UNMEASURED:
        return "unmeasured"
    return f"{builds} builds" + (f", {seconds} s" if isinstance(seconds, (int, float)) else ", seconds unmeasured")


def _cell(text) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def markdown(found: dict | None) -> str:
    """``UNRESOLVED.md`` of an export, from a dossier."""
    if not found:
        return "# Unresolved functions\n\nThe run recorded no unresolved dossier.\n"
    summary = found["summary"]
    exhaustion = summary["exhaustion"]
    lines = ["# Unresolved functions", "",
             f"{summary['functions']:,} functions ({summary['bytes']:,} bytes) are not settled after this run "
             f"(effort `{found.get('mode')}`" + (f", switched off: {', '.join(found['disabled'])}" if found.get("disabled")
                                                 else "") + ").  Each has a reason, the techniques of the kit with "
             "what happened to each on it, the eligible next ones with their expected cost (measured in this run, "
             "else `unmeasured`), and the evidence that would justify trying again.  The full record is "
             "`report/unresolved_dossier.json`.  Nothing here is credit.", "",
             "Retry: " + ", ".join(f"{key or 'settled'} {value:,}" for key, value in sorted(summary["by_retry"].items()))
             + " (`now`: a technique is still eligible, only rounds, budget or settings stopped it; `evidence`: the "
             "named change must come first; `never`: not a C target).", "",
             "## Why", "", "| reason code | functions | bytes | retry | evidence that justifies a retry |",
             "|---|---:|---:|---|---|"]
    for code, group in summary["by_reason"].items():
        lines.append(f"| {code} | {group['functions']:,} | {group['bytes']:,} | {group['retry']} | "
                     f"{_cell(reasonsmod.retry_of(code)[1])} |")
    lines += ["", "## Eligible next techniques", ""]
    if summary["eligible"]:
        lines += ["| technique | functions | expected builds (measured) | expected seconds (measured) | unmeasured |",
                  "|---|---:|---:|---:|---:|"]
        lines += [f"| {name} | {row['functions']:,} | {row['expected_builds']:,} | {row['expected_seconds']:,} | "
                  f"{row['unmeasured']:,} |" for name, row in sorted(summary["eligible"].items())]
    else:
        lines.append("No case has an eligible technique left.")
    lines += ["", "## Techniques per case", "",
              "How many unresolved cases each technique reached, by state (functions that were not attempted have no "
              "ledger).", "", "| technique | " + " | ".join(STATES) + " |", "|---" * (len(STATES) + 1) + "|"]
    for name, row in summary["ledger"].items():
        lines.append(f"| {name} | " + " | ".join(f"{row.get(state, 0):,}" for state in STATES) + " |")
    large = summary["large_functions"]
    lines += ["", f"Large functions (over 0x400 bytes): {large['cases']:,} cases; large-function reconstruction "
                  f"is unavailable for {large['unavailable']:,} ({_STRUCTURAL_WHY}).  Far-off drafts (the search "
                  f"withheld above its ceiling, 0x400 bytes or smaller): {summary['far_off_drafts']:,}.", "",
              "## Exhaustion", "",
              f"Effort `{exhaustion['mode']}`: {exhaustion['note']}."]
    for image, reason in sorted((exhaustion.get("stops") or {}).items()):
        lines.append(f"- {image}: {reason}")
    stats = summary.get("technique_stats") or {}
    if stats:
        lines += ["", "## Measured per technique (this run)", "",
                  "| technique | attempts | builds | builds per attempt | seconds per build | seconds per attempt |",
                  "|---|---:|---:|---:|---:|---:|"]
        lines += [f"| {name} | {row.get('attempts', 0):,} | {row.get('builds', 0):,} | {row.get('builds_per_attempt')} | "
                  f"{row.get('seconds_per_build')} | {row.get('seconds_per_attempt')} |" for name, row in sorted(stats.items())]
    functions = sorted(found.get("functions") or [], key=lambda e: (-int(e["size"] or 0), e["image"], e["vram"]))
    shown = f"; {_MARKDOWN_ROWS} of {len(functions):,} shown" if len(functions) > _MARKDOWN_ROWS else ""
    lines += ["", f"## Functions (largest first{shown})",
              "", "| image | function | vram | size | reason | next technique (expected cost) | retry when |",
              "|---|---|---|---:|---|---|---|"]
    for entry in functions[:_MARKDOWN_ROWS]:
        nxt = "; ".join(f"{item['technique']} ({_cost(item['cost'])})" for item in entry["eligible"]) or "none"
        when = "now" if entry["retry"] == "now" else "; ".join(entry["evidence_change"][:2]) or entry["retry"]
        lines.append(f"| {entry['image']} | {_cell(entry['name'])} | {entry['vram']} | 0x{int(entry['size'] or 0):X} | "
                     f"{entry['reason_code']} | {_cell(nxt)} | {_cell(when)} |")
    return "\n".join(lines) + "\n"
