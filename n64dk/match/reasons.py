"""Reason codes: why a function of an export is not settled, from what the run already recorded.

``decompile-export`` gives every function an outcome (``exact``, ``near``, ``open``, ...).  An outcome
says how far a function got, not why it stopped there.  This module gives every function that is
not settled one code of a closed vocabulary (``CODES``) and a next step, derived only from records
the run already keeps:

* the case (``state/cases.sqlite``): status, top cause and ranked residual causes, best score,
  recipe, size, the ladder's stages (``iterate-skipped`` with the search ceiling or the governor's
  reason), and the strict verifier's assumptions;
* the dossier (``solver/hard_queue_<image>.json``): the solver's primary cause, its next action
  and why automation stopped (round limit, budget, no technique left);
* the reason a function was not attempted (``state/scope_<image>.json``,
  ``solver/coverage_<image>.json``);
* the function's unit class (``analysis.unitclass``: boot code and RSP microcode).

Nothing here builds, reads the ROM, or changes an outcome: a code explains, it never grants credit.
The precedence (``explain``) is fixed and the first rule that holds wins, so the same records give
the same code.  ``unknown`` is the code when no record explains a function; it is counted apart.
"""
from __future__ import annotations

import json
from pathlib import Path

# outcomes that are settled: strictly verified C, a verified donor unit, verified assembly
SETTLED = ("exact", "donor", "donor-assembly", "assembly")
UNKNOWN = "unknown"
# above this size a function is drafted and built under its recipe only (``solver.LARGE_FUNCTION``)
LARGE_FUNCTION = 0x400

# code -> (meaning, next step).  The order is the precedence of ``explain`` (first match wins).
CODES: dict[str, tuple[str, str]] = {
    "needs-symbol": ("byte-identical; strict verification assumed the address of a symbol nothing declares",
                     "declare the assumed symbols, then verify again"),
    "data-not-owned": ("byte-identical; a data section base it references was inferred from the ROM, not "
                       "declared: no unit owns that data",
                       "place the data (declare the section bases), then verify again"),
    "source-not-kept": ("matched, but the case store no longer holds its source",
                        "resolve the function again (the run's sources were lost)"),
    "rsp-microcode": ("RSP microcode inside a CPU image: not CPU code, never decompiled as C",
                      "keep it as data or assembly; identify the microcode by its digest"),
    "boot-code": ("the entry routine or its hand-off code", "keep it as assembly"),
    "implausible-size": ("larger than any one function should be (over 128 KB): most likely a discovery boundary "
                         "error that merged code or data; not attempted", "check the function boundaries"),
    "translation-differs": ("unoptimised GCC code read back as C statement by statement; its compiled bytes "
                            "still differ from the target",
                            "compare the translation with the target instruction by instruction: unoptimised "
                            "code is literal, and one C spelling reproduces each form"),
    "too-large": ("larger than 0x400 bytes: drafted under its recipe only, or left out by --max-size",
                  "write or refine the draft by hand; automation withholds search above 0x400"),
    "excluded-by-settings": ("left out by the run's settings (--only-names, --every, --limit)",
                             "run again without the restriction"),
    "covered-by-manifest": ("not attempted: a project manifest row already covers it",
                            "check the manifest row's verification"),
    "outside-image": ("not attempted: its range lies outside every discovered image",
                      "repair the image layout (discovery) first"),
    "too-small": ("not attempted: smaller than the resolver's minimum", "keep it as assembly"),
    "excluded-other": ("not attempted, for a recorded reason no other code names", "read the recorded reason"),
    "embedded-data": ("the catalogued range contains words that are not instructions",
                      "split the data out of the range (boundary operations are proposed, never applied)"),
    "boundary-uncertain": ("the catalogued range is not one whole function (fragment, false split, shared tail)",
                           "review the proposed boundary operations, then resolve again"),
    "handwritten-assembly-unproven": ("hand-written code whose assembly did not reassemble to the same bytes",
                                      "write the assembly by hand or fix the disassembly"),
    "nonstandard-abi": ("register use outside the o32 convention: needs its translation unit",
                        "resolve it with its unit (neighbours, static callers)"),
    "translator-declined": ("unoptimised GCC code that the kit's statement-by-statement translator does not read "
                            "yet; the detail names the construct it stopped at",
                            "teach the translator that construct (n64dk.match.redraft_gcc_debug): settle its C "
                            "spelling with a compiler probe, then check the round trip"),
    "decompiler-failed": ("the decompiler produced no draft", "draft it by hand or with other decompiler options"),
    "draft-does-not-build": ("a draft exists but does not compile", "fix the compile error"),
    "strict-verify-failed": ("byte-identical under the exploration oracle, refused by the strict verifier",
                             "fix the relocation or data the strict verifier names"),
    "style-unrecognised": ("in a stretch whose code idioms (frames, registers, branches) match no compiler in the "
                           "kit's idiom table: a style fingerprint, not a build test; its functions were still "
                           "drafted and tried under the title's recipes",
                           "try every installed compiler on its region (--regional-calibration); a compiler is "
                           "missing only if none of them reproduces a sample"),
    "no-recipe-identified": ("the code's compiler family differs from the recipe it was built with, or it has "
                             "no recipe", "calibrate the recipe of its region"),
    "far-off-draft": ("best draft further than the search ceiling: the variant search was skipped",
                      "a better draft (types, structure) or another recipe"),
    "search-withheld": ("the variant search was withheld by the search-economics governor",
                        "run again without --search-economics"),
    "round-limit": ("automation had a next technique but the effort mode's round limit stopped it",
                    "run again with --effort exhaustive"),
    "budget-exhausted": ("automation had a next technique but a time, build or model-call budget ran out",
                         "run again with a larger budget"),
    "register-allocation-only": ("every remaining difference is register allocation",
                                 "permute statements and temporaries (permuter)"),
    "call-target-differs": ("a call or address resolves elsewhere", "check the callee symbol and its address"),
    "type-or-declaration": ("a type, width, signedness or global declaration differs",
                            "fix the types and declarations"),
    "stack-frame-differs": ("the stack frame size differs", "fix locals: their number, sizes and types"),
    "control-flow-differs": ("branches, branch-likely forms, delay slots or a switch differ",
                             "restructure the control flow"),
    "assembler-or-schedule": ("only nops or assembler expansions differ", "check the assembler and its flags"),
    "needs-tu-context": ("matches only inside its translation unit", "resolve it with its unit"),
    "unrecognised-difference": ("instructions differ without a recognised pattern", "compare by hand"),
    UNKNOWN: ("no record explains why it is not settled", "inspect the run: the case or its dossier is missing"),
}

_NOT_ATTEMPTED = (                       # recorded reason text (commands.resolve, solver.coverage) -> code
    ("larger than the run's --max-size", "too-large"),
    ("implausibly large for one function", "implausible-size"),
    ("not in the --only-names set", "excluded-by-settings"),
    ("sampled out", "excluded-by-settings"),
    ("beyond the run's --limit", "excluded-by-settings"),
    ("already matched by a project manifest row", "covered-by-manifest"),
    ("no discovered image", "outside-image"),
    ("smaller than the resolver's minimum", "too-small"),
    ("unexplained", UNKNOWN),
)

# code -> (when a retry is justified, the change that justifies it).  ``now``: nothing new is
# needed, only rounds, budget or settings; ``evidence``: the named change must happen first;
# ``never``: not a C target, or nothing to retry.  The unresolved dossier (``match.dossier``) adds
# the case's own detail (the symbol names, the region, the score and ceiling).
RETRY: dict[str, tuple[str, str]] = {
    "needs-symbol": ("evidence", "a declaration of each assumed symbol"),
    "data-not-owned": ("evidence", "a declared placement (section base) of the data it references"),
    "source-not-kept": ("now", "none: resolve it again (its source was lost)"),
    "rsp-microcode": ("never", "none: not CPU code"),
    "boot-code": ("never", "none: kept as assembly"),
    "too-large": ("evidence", "a large-function method (not in the kit) or a source supplied for it"),
    "implausible-size": ("evidence", "corrected function boundaries that split the entry"),
    "excluded-by-settings": ("now", "none: a run without the restriction"),
    "covered-by-manifest": ("never", "none: a manifest row covers it"),
    "outside-image": ("evidence", "an image layout that places its range"),
    "too-small": ("never", "none: below the resolver's minimum"),
    "excluded-other": ("evidence", "the recorded reason resolved"),
    "embedded-data": ("evidence", "an accepted boundary operation that splits the data out"),
    "boundary-uncertain": ("evidence", "an accepted boundary operation for its range"),
    "handwritten-assembly-unproven": ("evidence", "an assembly source that reassembles to its bytes"),
    "nonstandard-abi": ("evidence", "its translation unit's neighbours, resolved with it"),
    "translator-declined": ("evidence", "translator support for the construct the detail names"),
    "translation-differs": ("evidence", "the differing form identified and reproduced by a compiler probe"),
    "decompiler-failed": ("evidence", "a draft: another decompiler rendering, the declarations the decompiler "
                                      "needs, or a supplied source"),
    "draft-does-not-build": ("evidence", "a draft that builds (its compile error fixed)"),
    "strict-verify-failed": ("evidence", "the relocation or data fact the strict verifier names"),
    "style-unrecognised": ("now", "none: regional calibration over every installed compiler"),
    "no-recipe-identified": ("evidence", "a recipe for its region (byte-identical calibration samples)"),
    "far-off-draft": ("evidence", "a draft within the search ceiling (types, structure or declarations)"),
    "search-withheld": ("now", "none: a run without --search-economics"),
    "round-limit": ("now", "none: more rounds (an effort mode with more rounds, or --resume with --rounds)"),
    "budget-exhausted": ("now", "none: a larger budget"),
    "register-allocation-only": ("evidence", "new context (a caller or callee becoming exact) or a "
                                             "register-permutation method (not in the kit)"),
    "call-target-differs": ("evidence", "a declaration of the callee or address it references"),
    "type-or-declaration": ("evidence", "declarations of the types and globals it uses"),
    "stack-frame-differs": ("evidence", "type or structure evidence that changes its frame (a local's size)"),
    "control-flow-differs": ("evidence", "a draft with another control-flow structure"),
    "assembler-or-schedule": ("evidence", "the assembler or its flags identified for its region"),
    "needs-tu-context": ("evidence", "its translation unit's neighbours exact"),
    "unrecognised-difference": ("evidence", "a person's comparison"),
    UNKNOWN: ("evidence", "a record that explains it (its case or its dossier)"),
}


def retry_of(code: str) -> tuple[str, str]:
    """``(when, evidence change)`` of a reason code; ``("", "")`` when the function is settled."""
    return RETRY.get(code, ("", "")) if code else ("", "")


_REGISTER =frozenset({"register.permutation", "register.pressure", "register.lifetime"})
_RESIDUAL = {
    "relocation.target": "call-target-differs",
    "type.signedness": "type-or-declaration", "type.width": "type-or-declaration",
    "type.volatile": "type-or-declaration", "global.declaration": "type-or-declaration",
    "frame.size": "stack-frame-differs",
    "control.branch_form": "control-flow-differs", "control.branch_likely": "control-flow-differs",
    "control.switch": "control-flow-differs", "schedule.delay_slot": "control-flow-differs",
    "schedule.load_hazard": "assembler-or-schedule", "assembler.variant": "assembler-or-schedule",
    "tu.context": "needs-tu-context",
    "semantic.unknown": "unrecognised-difference",
}
_LIMIT = 240                             # characters of one next step


def _clean(text: str) -> str:
    """One TSV field: no tab or line break, bounded."""
    text = " ".join(str(text or "").split())
    return text if len(text) <= _LIMIT else text[:_LIMIT - 3] + "..."


def _assumptions(case) -> list[dict]:
    for record in (getattr(case, "strict", None), getattr(case, "proof", None)):
        items = (record or {}).get("assumptions") if isinstance(record, dict) else None
        if items:
            return [item for item in items if isinstance(item, dict)]
    return []


def _not_attempted(reason: str) -> str:
    if not reason:
        return UNKNOWN
    for prefix, code in _NOT_ATTEMPTED:
        if reason.startswith(prefix):
            return code
    return "excluded-other"


def _case_code(case, unit_class: str, dossier: dict) -> tuple[str, str]:
    """(code, residual cause id or '') of a case that is not settled, assumed or exact."""
    status, top = case.status, case.top_cause or ""
    causes = [item for item in (case.causes or []) if isinstance(item, dict)]
    confidence = {item.get("cause"): item.get("confidence") or 0 for item in causes}
    if status == "blocked-boundary" or top.startswith("boundary."):
        return ("embedded-data" if top == "boundary.embedded_data" else "boundary-uncertain"), ""
    if status == "probable-handwritten" or top == "assembly.handwritten" or unit_class == "handwritten" or \
            (status == "needs-tu" and confidence.get("assembly.handwritten", 0) >= 0.6):
        return "handwritten-assembly-unproven", ""
    if status == "needs-tu":
        return "nonstandard-abi", ""
    translator = next((stage for stage in reversed(case.stages or [])
                       if isinstance(stage, dict) and stage.get("stage") == "translator"), None)
    if translator and translator.get("declined") and case.best_score != 0:
        return "translator-declined", str(translator["declined"])
    if status == "needs-source":
        return "decompiler-failed", ""
    if status == "does-not-build" or case.best_score is None:
        return "draft-does-not-build", ""
    if case.best_score == 0:                       # byte-identical, yet neither exact nor assumed
        return "strict-verify-failed", ""
    if dossier.get("primary_cause") == "unknown toolchain":
        return "style-unrecognised", ""
    if top == "compiler.family" or not case.recipe:
        return "no-recipe-identified", ""
    if translator and translator.get("translated"):
        return "translation-differs", ""
    if case.size > LARGE_FUNCTION and status != "near":    # a size, not a score: no residual named
        return "too-large", ""
    stages = [stage for stage in (case.stages or []) if isinstance(stage, dict)]
    if any(stage.get("stage") == "iterate-skipped" and isinstance(stage.get("ceiling"), int)
           and case.best_score > stage["ceiling"] for stage in stages):
        return "far-off-draft", ""
    residual = _residual(case)
    if any(stage.get("stage") == "iterate-skipped" and stage.get("reason") for stage in stages):
        return "search-withheld", residual[1]
    if dossier.get("next_action"):
        why = str(dossier.get("why_stopped") or "")
        if why.startswith("round limit"):
            return "round-limit", residual[1]
        if "budget" in why:
            return "budget-exhausted", residual[1]
    return residual


def _residual(case) -> tuple[str, str]:
    """(code, cause id) of the residual the ladder classified: register-only when every ranked
    cause is a register cause, else the first ranked cause that has a class; ``unknown``."""
    ids = [item.get("cause") for item in (case.causes or []) if isinstance(item, dict)] or \
        ([case.top_cause] if case.top_cause else [])
    if ids and all(cause in _REGISTER for cause in ids):
        return "register-allocation-only", ids[0]
    for cause in ids:
        if cause in _RESIDUAL:
            return _RESIDUAL[cause], cause
    return UNKNOWN, ""


def code_of(outcome: str, case=None, *, unit_class: str = "", not_attempted: str = "",
            dossier: dict | None = None) -> tuple[str, str]:
    """The reason code of one function (``""`` when it is settled) and the residual cause id its
    next step names, if any.  First match wins:

    1. settled outcome: no code;
    2. byte-identical awaiting a declaration: ``data-not-owned`` (a section base assumed) else
       ``needs-symbol``;
    3. ``matched-source-missing``: ``source-not-kept``;
    4. unit class ``rsp`` / ``boot``;
    5. no case: the recorded not-attempted reason (``unknown`` when there is none);
    6. the case's target findings: boundary, hand-written, ABI;
    7. the draft: none, does not build, refused by the strict verifier;
    8. the tool chain: no compiler explains the stretch, wrong or no recipe;
    9. why the search stopped: size, ceiling, governor, round limit, budget (governor, round limit
       and budget also name the residual);
    10. the residual the ladder classified last (register-only, then the first ranked cause with
        a class);
    11. ``unknown``."""
    dossier = dossier or {}
    if outcome in SETTLED:
        return "", ""
    if outcome in ("awaiting-symbols", "donor-awaiting-symbols") or (case is not None and case.status == "assumed"):
        if any(item.get("kind") == "section" for item in _assumptions(case)):
            return "data-not-owned", ""
        return "needs-symbol", ""
    if outcome == "matched-source-missing":
        return "source-not-kept", ""
    if unit_class in ("rsp", "boot"):
        return ("rsp-microcode" if unit_class == "rsp" else "boot-code"), ""
    if case is None:
        return _not_attempted(not_attempted), ""
    return _case_code(case, unit_class, dossier)


def explain(outcome: str, case=None, *, unit_class: str = "", not_attempted: str = "",
            dossier: dict | None = None) -> tuple[str, str]:
    """``(reason_code, next_step)`` of one function; both empty when it is settled."""
    dossier = dossier or {}
    code, cause = code_of(outcome, case, unit_class=unit_class, not_attempted=not_attempted, dossier=dossier)
    if not code:
        return "", ""
    parts = [CODES[code][1]]
    if code in ("needs-symbol", "data-not-owned"):
        names = [str(item.get("name")) for item in _assumptions(case)
                 if item.get("kind") == ("section" if code == "data-not-owned" else "symbol")]
        if names:
            parts[0] += ": " + ", ".join(names[:3]) + (f" (+{len(names) - 3} more)" if len(names) > 3 else "")
    elif cause:                    # the residual's own hint (search-state codes name it as well)
        hint = next((item.get("next") for item in case.causes if isinstance(item, dict)
                     and item.get("cause") == cause and item.get("next")), "")
        parts.append(f"residual {cause}: {hint}" if hint else f"residual {cause}")
    elif code in ("decompiler-failed", "draft-does-not-build", "strict-verify-failed") and case is not None \
            and case.next:
        parts.append(case.next)
    elif case is None and not_attempted and code != UNKNOWN:
        parts.append(not_attempted)
    if dossier.get("next_action"):
        parts.append(f"automation's next action: {dossier['next_action']} "
                     f"(stopped: {dossier.get('why_stopped') or 'not recorded'})")
    return code, _clean("; ".join(parts))


def _vram(text) -> int | None:
    try:
        return int(str(text), 16)
    except ValueError:
        return None


def load_dossiers(run: Path, image: str) -> dict[int, dict]:
    """Address -> the dossier fields a code reads (``solver/hard_queue_<image>.json``); none when the
    file is absent or unreadable."""
    try:
        records = json.loads((Path(run) / "solver" / f"hard_queue_{image}.json").read_text())
    except (OSError, ValueError):
        return {}
    out = {}
    for record in records if isinstance(records, list) else []:
        if isinstance(record, dict) and _vram(record.get("vram")) is not None:
            out[_vram(record["vram"])] = {key: record.get(key) for key in
                                          ("primary_cause", "next_action", "why_stopped")}
    return out


def load_not_attempted(run: Path, image: str) -> dict[int, str]:
    """Address -> the recorded reason a function was not attempted (``solver/coverage_<image>.json``)."""
    try:
        found = json.loads((Path(run) / "solver" / f"coverage_{image}.json").read_text())
    except (OSError, ValueError):
        return {}
    out = {}
    for entry in (found.get("entries") or []) if isinstance(found, dict) else []:
        if isinstance(entry, dict) and not entry.get("attempted") and _vram(entry.get("vram")) is not None:
            out[_vram(entry["vram"])] = str(entry.get("reason") or "")
    return out


def summary(tallies) -> dict:
    """``tallies``: ``(code, size in bytes)`` per function.  Counts of the functions not settled, how
    many carry a specific code and how many ``unknown``."""
    by_code: dict[str, dict[str, int]] = {}
    for code, size in tallies:
        if code:
            group = by_code.setdefault(code, {"functions": 0, "bytes": 0})
            group["functions"] += 1
            group["bytes"] += int(size)
    total = sum(group["functions"] for group in by_code.values())
    unknown = by_code.get(UNKNOWN, {}).get("functions", 0)
    order = sorted(by_code, key=lambda code: (-by_code[code]["functions"], code))
    return {"not_settled": total, "specific": total - unknown, "unknown": unknown,
            "by_code": {code: by_code[code] for code in order}}


def table(tallies) -> list[str]:
    """COVERAGE.md's section on the functions that are not settled, by reason code."""
    found = summary(tallies)
    if not found["not_settled"]:
        return []
    share = found["specific"] / found["not_settled"]
    return ["## Why functions are not settled", "",
            "Every function that is not strictly exact, a verified donor unit or verified assembly has one",
            "`reason_code` in `FUNCTIONS.tsv`, derived from the run's own records by a fixed precedence, and a",
            "`next_step`.  The code explains; it never changes an outcome.", "",
            f"{found['specific']:,} of {found['not_settled']:,} functions have a specific code ({share:.1%}); "
            f"`unknown`: {found['unknown']:,}.", "",
            "| reason code | functions | bytes | meaning |", "|---|---:|---:|---|",
            *[f"| {code} | {group['functions']:,} | {group['bytes']:,} | {CODES[code][0]} |"
              for code, group in found["by_code"].items()], ""]
