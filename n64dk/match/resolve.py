"""Resolution cases: a durable record per unmatched function and the ladder that fills it.

The matching loop keeps answering one question per function -- *what kind of work does this
one need?* -- and without a record the answer is rediscovered every session.  A resolution
case stores it: the verified target identity, the ranked causes (``causes.TAXONOMY``), every
stage that ran, the best candidate, the families that were tried and did nothing, and the
next recommended experiment.

``resolve`` climbs a bounded ladder and stops at the first rung that settles the case:

1. **target**    control flow, liveness and frame analysis of the retail bytes.  A range that
                 is not a whole function (fragment, false split, embedded data), one that uses a
                 non-standard register convention, or probable hand-written code is recorded as
                 such and costs no compile.
2. **source**    the supplied candidate, the case's previous best, or a fresh draft.
3. **baseline**  build and compare; exact ends the case.
4. **diagnose**  typed causes from the residual.
5. **iterate**   variant families justified by the causes first, then the remaining ones,
                 skipping families already closed for this exact source.
6. **classify**  re-diagnose the best candidate; the top cause names the remaining blocker.

A case is keyed by ``IMAGE:VRAM:SIZE`` and carries the digest of its target bytes: when the
boundary, the image or the bytes change, the old case is archived as stale and never reused.
Every case marked ``exact`` has already passed ``verify.strict_compare``.  Integration still
performs an independent clean rebuild and the configured project gates before it becomes credit.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..analysis import flow as flowmod
from . import causes as causemod
from . import handasm
from . import iterate as itermod

BLOCKING = ("boundary.fragment", "boundary.false_split", "boundary.embedded_data")
REWRITES_OF_BEST = 10          # rewrites of the best source tried after the search (one build each)
# The variant search is skipped when the best draft is further than this from the target.
# Measured over 1,712 searches on two titles (IDO and KMC GCC): none of the 630 that started above
# 40 reached an exact match, the furthest success started at 29, and those 630 took 37% of all
# builds.  0 or None removes the ceiling.
SEARCH_CEILING = 40
CROSS_DRAFTS, CROSS_RECIPES = 4, 8      # drafts x recipes tried when no draft is exact under the default recipe

STATUSES = ("exact", "assumed", "assembly", "near", "open", "blocked-boundary", "needs-tu", "probable-handwritten",
            "needs-source", "does-not-build", "stale")
# rough tractability of the top cause, used only to order the queue (higher = sooner)
TRACTABILITY = {"compiler.family": 0.95, "type.signedness": 0.9, "type.width": 0.9, "type.volatile": 0.85, "register.lifetime": 0.8,
                "global.declaration": 0.7, "control.branch_form": 0.6, "control.branch_likely": 0.6,
                "frame.size": 0.6, "register.permutation": 0.5, "schedule.delay_slot": 0.5,
                "register.pressure": 0.4, "source.missing": 0.4, "assembler.variant": 0.3,
                "relocation.target": 0.7, "schedule.load_hazard": 0.3, "control.switch": 0.4,
                "semantic.unknown": 0.2, "tu.context": 0.15, "abi.nonstandard": 0.1,
                "boundary.fragment": 0.0, "boundary.false_split": 0.0, "boundary.shared_tail": 0.05,
                "boundary.embedded_data": 0.0, "assembly.handwritten": 0.0}


def case_id(image: str, vram: int, size: int) -> str:
    return f"{image}:{vram:08X}:{size:X}"


@dataclass
class Case:
    case_id: str
    image: str
    name: str
    vram: int
    size: int
    target_digest: str
    status: str = "open"
    top_cause: str = ""
    causes: list[dict] = field(default_factory=list)
    best_score: int | None = None
    best_digest: str | None = None
    stages: list[dict] = field(default_factory=list)
    closed_families: dict[str, list[str]] = field(default_factory=dict)   # source digest -> families
    next: str = ""
    builds: int = 0
    updated_ns: int = 0
    proof: dict | None = None
    recipe: str = ""                 # pipeline and flags the best candidate was built with
    rom_offset: int | None = None    # file offset of the function in its image, when known
    strict: dict = field(default_factory=dict)   # verify.strict_compare verdict for an exact candidate

    def as_dict(self) -> dict:
        return {"case": self.case_id, "name": self.name, "vram": f"0x{self.vram:08X}", "size": self.size,
                "status": self.status, "top_cause": self.top_cause, "causes": self.causes,
                "recipe": self.recipe, "strict": self.strict,
                "best_score": self.best_score, "best_source": self.best_digest, "builds": self.builds,
                "stages": self.stages, "next": self.next, "proof": self.proof}


class CaseStore:
    """SQLite store (WAL) for cases plus content-addressed candidate sources."""

    def __init__(self, root: str | Path, readonly: bool = False):
        self.root = Path(root)
        self.readonly = readonly
        self.lock = threading.RLock()            # one connection shared by worker threads
        if readonly:
            # Inspection of a finished run leaves no file behind: no WAL or shared-memory file,
            # no directory.  With a WAL file present (a run still writing, or one that stopped
            # before a checkpoint) it must be read through, so the store is opened read-only;
            # without one it is opened immutable, which takes no lock and writes nothing.
            database = self.root / "cases.sqlite"
            if not database.is_file():
                raise FileNotFoundError(database)
            mode = "mode=ro" if database.with_name("cases.sqlite-wal").exists() else "mode=ro&immutable=1"
            self.db = sqlite3.connect(f"{database.resolve().as_uri()}?{mode}", uri=True, timeout=60,
                                      isolation_level=None, check_same_thread=False)
            return
        (self.root / "sources").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "cases.sqlite", timeout=60, isolation_level=None,
                                  check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS cases (case_id TEXT PRIMARY KEY, target_digest TEXT NOT NULL, "
                        "status TEXT NOT NULL, top_cause TEXT NOT NULL, size INTEGER NOT NULL, payload TEXT NOT NULL, "
                        "updated_ns INTEGER NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS stale (case_id TEXT NOT NULL, target_digest TEXT NOT NULL, "
                        "payload TEXT NOT NULL, archived_ns INTEGER NOT NULL)")

    def close(self) -> None:
        self.db.close()

    def get(self, key: str) -> Case | None:
        with self.lock:
            row = self.db.execute("SELECT payload FROM cases WHERE case_id = ?", (key,)).fetchone()
        return Case(**json.loads(row[0])) if row else None

    def put(self, case: Case) -> None:
        case.updated_ns = time.time_ns()
        payload = json.dumps(case.__dict__, sort_keys=True)
        for attempt in range(8):                              # short retry: other processes may hold the lock
            try:
              with self.lock:
                self.db.execute("BEGIN IMMEDIATE")
                self.db.execute("INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(case_id) DO UPDATE SET "
                                "target_digest = excluded.target_digest, status = excluded.status, "
                                "top_cause = excluded.top_cause, size = excluded.size, payload = excluded.payload, "
                                "updated_ns = excluded.updated_ns",
                                (case.case_id, case.target_digest, case.status, case.top_cause, case.size, payload,
                                 case.updated_ns))
                self.db.execute("COMMIT")
                return
            except sqlite3.OperationalError:
                with self.lock:
                    try:
                        self.db.execute("ROLLBACK")
                    except sqlite3.OperationalError:
                        pass
                if attempt == 7:
                    raise
                time.sleep(0.05 * (attempt + 1))

    def archive(self, case: Case) -> None:
        with self.lock:
            self._archive(case)

    def _archive(self, case: Case) -> None:
        self.db.execute("INSERT INTO stale VALUES (?, ?, ?, ?)",
                        (case.case_id, case.target_digest, json.dumps(case.__dict__, sort_keys=True), time.time_ns()))

    def all(self) -> list[Case]:
        with self.lock:
            rows = self.db.execute("SELECT payload FROM cases ORDER BY case_id").fetchall()
        return [Case(**json.loads(row[0])) for row in rows]

    def save_source(self, text: str) -> str:
        digest = hashlib.sha256(text.encode()).hexdigest()
        path = self.root / "sources" / f"{digest}.c"
        if not path.exists():
            temporary = path.with_suffix(f".{time.time_ns()}.tmp")
            temporary.write_text(text)
            temporary.replace(path)
        return digest

    def source(self, digest: str | None) -> str | None:
        path = self.root / "sources" / f"{digest}.c" if digest else None
        return path.read_text() if path is not None and path.is_file() else None


def resolve(store: CaseStore, image: str, name: str, vram: int, target: bytes, evaluate, *,
            source: str | None = None, draft=None, flow: flowmod.Flow | None = None,
            neighbours: set[int] | None = None, budget: int = 300, jobs: int = 8, force: bool = False,
            near: int = 10, emit=None, pipeline: str | None = None, recipes=None, strict=None,
            recipe_label: str = "", rom_offset: int | None = None,
            search_ceiling: int | None = SEARCH_CEILING, frame_fit: bool = True,
            cross_recipes: int | None = None, fresh: bool = False, governor=None, stratum: str = "",
            provisional: bool = False, wide_recipes=None, search: bool = True,
            drafts_x_recipes: bool = True, rewrites=None, near_siblings: bool = True, loop_forms: bool = True,
            hand_forms: bool = True, hand_forms_budget: int = 40,
            region_levels: tuple[str, ...] = ()) -> Case:
    """Run the ladder for one function and store the case.

    ``evaluate(text) -> (iterate.Evaluation, relocated candidate bytes or None)``;
    ``draft(family) -> (source text or None, error or None)`` produces a first candidate when
    none is supplied (``family`` is the detected compiler family or None);
    ``recipes(family) -> [(label, pipeline, evaluate)]`` lists alternative build recipes, tried on
    the baseline source when the bytes carry another compiler family's fingerprint or the
    baseline is far off; ``strict(text, recipe_label) -> dict`` re-verifies an exact candidate
    with the non-circular oracle (its verdict is stored on the case; a ``mismatch`` demotes the
    case to ``open``); ``emit(event_dict)`` receives progress events.  ``fresh`` starts again from
    the drafts under the given recipe instead of continuing the stored case (whose best source
    and recipe may be what kept it from matching); the stored case still stands when the new
    attempt ends further away.  ``wide_recipes`` (optional, ``() -> bool``) gates the wide recipe
    retry after the baseline: when given and false, and the draft builds under a recipe of the
    detected family, the retry is skipped (round 11: it paid only on unoptimised code).
    ``provisional`` marks the case as a sample (recipe calibration):
    the next ordinary resolve of an unsettled provisional case starts fresh in the same way, so a
    function's outcome does not depend on whether calibration happened to sample it.  ``search``,
    ``frame_fit`` and ``drafts_x_recipes`` false switch one rung off (decompile-run ablation switches);
    no switch reaches the strict verdict, so a rung switched off can only cost a match.
    ``rewrites`` (optional, ``text -> [(label, text)]``) gives the recipe's rewrites of a source
    (``n64dk.match.redraft``); they are tried once more on the best source the search ends with, because
    a rewrite keyed on how a loop or a temporary is written may apply only to what the search made of the
    draft (measured on a title: rewrites that matched four functions from such sources found none of them
    from the first drafts).  ``near_siblings`` (default on) tries a near miss under the sibling
    versions of its compiler (``compilers.sibling_recipe``; ``region_levels``: the optimisation
    levels the region already uses) and, for IDO, its local-free forms, before the variant search.
    A provisional (calibration) case that ends exact is rebuilt under the sibling versions at the
    same flags (stage ``version-check``), so a function exact under both versions is not counted
    as evidence for either."""
    size, digest = len(target), hashlib.sha256(target).hexdigest()
    key = case_id(image, vram, size)

    def event(kind: str, **extra) -> None:
        if emit is not None:
            emit({"event": kind, "case": key, "name": name, **extra})

    previous = store.get(key)
    if previous is not None and previous.target_digest != digest:
        previous.status = "stale"
        store.archive(previous)
        event("stale-case-archived", old_target=previous.target_digest[:12])
        previous = None
    if previous is not None and previous.status == "exact" and not force:
        return previous
    if previous is not None and previous.status == "assumed" and not force and source is None:
        # Byte-identical already; the only open question is whether the addresses it leans on have
        # been declared since.  Ask the verifier again with the recipe the match was found under.
        # Re-running the ladder here could only lose the match (it would start from the default
        # recipe), so the stored case is returned whatever the verifier says.
        if strict is not None and previous.best_digest:
            try:
                again = strict(store.source(previous.best_digest) or "", previous.recipe) or {}
            except Exception:
                again = {}
            if again.get("verdict") == "exact":
                previous.strict, previous.status = again, "exact"
                previous.next = ""
                previous.stages.append({"stage": "strict-verify", "verdict": "exact", "assumptions": 0,
                                        "origin": "re-verified"})
                store.put(previous)
                event("case-exact", top_cause=previous.top_cause, score=previous.best_score, builds=previous.builds)
        return previous
    if previous is not None and not provisional and source is None and draft is not None \
            and any(stage.get("stage") == "provisional" for stage in previous.stages):
        fresh = True                     # a calibration sample: draft again under this recipe
    case = Case(key, image, name, vram, size, digest, recipe=recipe_label, rom_offset=rom_offset)
    if provisional:
        case.stages.append({"stage": "provisional", "recipe": recipe_label})
    if previous is not None and not fresh:
        case.closed_families, case.best_digest, case.best_score = (previous.closed_families, previous.best_digest,
                                                                   previous.best_score)
        if source is None and previous.best_digest and previous.recipe and previous.recipe != recipe_label \
                and recipes is not None:
            # the stored best was built under another recipe: continue under that one
            try:
                for label, recipe_pipeline, recipe_evaluate in recipes(None):
                    if label == previous.recipe:
                        evaluate, pipeline, case.recipe = recipe_evaluate, recipe_pipeline, label
                        break
            except Exception:
                pass

    _RANK = {"exact": 0, "assumed": 1, "near": 2}

    def finish(status: str, ranked, next_step: str = "") -> Case:
        done = _finish(status, ranked, next_step)
        if previous is None or force or source is not None:
            return done
        # A rerun never makes a case worse: the earlier result stays when this one is further away.
        # Findings about the target itself (boundary, ABI, hand-written) are new information and stand.
        if done.status in _RANK or done.status in ("open", "does-not-build", "needs-source"):
            before, after = _RANK.get(previous.status, 3), _RANK.get(done.status, 3)
            worse = after > before or (after == before and previous.best_score is not None
                                       and (done.best_score is None or done.best_score > previous.best_score))
            if worse:
                previous.builds = done.builds + previous.builds
                if not provisional:              # drafted again under this recipe: no longer a sample
                    previous.stages = [st for st in previous.stages if st.get("stage") != "provisional"]
                store.put(previous)
                event("case-kept-previous", status=previous.status, score=previous.best_score,
                      rerun_status=done.status, rerun_score=done.best_score)
                return previous
        return done

    def _finish(status: str, ranked, next_step: str = "") -> Case:
        if status in ("exact", "assumed") and case.proof and case.proof.get("verdict") not in (None, "not-run"):
            # the evaluator already verified strictly (engine proof): one verdict, one place
            case.strict = dict(case.proof)
            event("strict-verdict", verdict=case.strict.get("verdict"),
                  assumptions=len(case.strict.get("assumptions", [])))
        elif status == "exact" and strict is not None:
            try:
                case.strict = strict(store.source(case.best_digest) or "", case.recipe) or {}
            except Exception as error:                     # verification trouble must not lose the case
                case.strict = {"verdict": "error", "error": str(error)[-200:]}
            verdict = case.strict.get("verdict")
            event("strict-verdict", verdict=verdict, assumptions=len(case.strict.get("assumptions", [])))
            if verdict == "mismatch":
                status, next_step = "open", "exact under the exploration oracle but not under strict " \
                    "verification: " + "; ".join(case.strict.get("conflicts", [])[:2])
            elif verdict == "assumed":
                status, next_step = "assumed", "declare the assumed symbols/sections, then verify again"
            elif verdict != "exact":
                status, next_step = "open", "strict verification did not produce an exact verdict"
        if provisional and status in ("exact", "assumed") and recipes is not None and case.best_digest:
            from . import compilers as compmod
            current = case.recipe or recipe_label
            under = []
            try:
                best_source = store.source(case.best_digest) or ""
                pipe_now, _, flags_now = (current or "").partition(" ")
                for label, _rp, rev in recipes(None):
                    if label.partition(" ")[0] in compmod.siblings(pipe_now) and label.partition(" ")[2] == flags_now:
                        evaluation, _code = rev(best_source)
                        case.builds += 1
                        if evaluation.exact or evaluation.score == 0:
                            under.append(label)
            except Exception:
                under = None
            case.stages.append({"stage": "version-check", "recipe": current, "exact_under": under})
        if status == "exact" and (case.strict or {}).get("verdict") != "exact":
            # No exploration oracle, historical label, or interrupted verifier may grant exact
            # credit.  Exact is a proof state, not a score or a caller assertion.
            status, next_step = "open", "strict verification proof is missing"
        case.status = status
        case.causes = [c.as_dict() for c in ranked]
        case.top_cause = ranked[0].id if ranked else ""
        case.next = next_step or (ranked[0].next if ranked else "")
        store.put(case)
        event("case-" + status, top_cause=case.top_cause, score=case.best_score, builds=case.builds)
        return case

    # 1. target
    flow = flow or flowmod.analyse(target, vram)
    found = causemod.target_causes(flow, neighbours)
    fingerprint = causemod.compiler_family(flow)
    family = fingerprint["family"] if fingerprint["confidence"] >= 0.6 else None
    case.stages.append({"stage": "target", "shape": flowmod.shapes(flow)[0]["shape"],
                        "entry": flowmod.entry_evidence(flow)["problems"][:3],
                        "compiler_family": fingerprint["family"], "family_confidence": fingerprint["confidence"]})
    by_id = {c.id: c for c in found}
    if not force:
        handwritten = by_id.get("assembly.handwritten") and by_id["assembly.handwritten"].confidence >= 0.8
        if handwritten:
            # Never C: its faithful source is assembly.  Kept when that source assembles back to
            # the same bytes (``draft.assembly``), and reported apart from what was matched from C.
            # This comes before the boundary findings: hand-written routines branch into and fall
            # through to one another, which is what those findings see.
            kept, assemble = None, getattr(draft, "assembly", None)
            if assemble is not None:
                try:
                    kept = assemble()
                except Exception:                           # noqa: BLE001 - the finding itself still stands
                    kept = None
            if kept:
                text_kept, proof_kept = kept
                case.best_digest, case.best_score = store.save_source(text_kept), 0
                case.recipe, case.strict = handasm.RECIPE, dict(proof_kept)
                return finish(handasm.STATUS, found, "hand-written code: kept as assembly source that "
                                                     "assembles to the same bytes")
        blocking = [c for c in found if c.id in BLOCKING and c.confidence >= 0.7]
        if blocking:
            return finish("blocked-boundary", found)
        if by_id.get("abi.nonstandard") and by_id["abi.nonstandard"].confidence >= 0.7:
            return finish("needs-tu", found)
        if handwritten:
            return finish("probable-handwritten", found)

    # 2. source
    text, origin = source, "supplied"
    if text is None and case.best_digest:
        text, origin = store.source(case.best_digest), "previous-best"
    built: dict[str, bytes | None] = {}

    def run(candidate: str) -> itermod.Evaluation:
        evaluation, code = evaluate(candidate)
        built[hashlib.sha256(candidate.encode()).hexdigest()] = code
        return evaluation

    base = None
    scored: list = []
    if text is None and draft is not None:
        drafted = draft(family)
        origin = "draft"
        if isinstance(drafted, list):
            # a portfolio of drafts (e.g. with and without recovered declarations): build each and
            # continue from the closest one, so an unhelpful context can never cost a match
            for label, candidate in drafted:
                evaluation = run(candidate)
                case.builds += 1
                scored.append((evaluation.score if evaluation.score is not None else 1 << 30, len(scored),
                               label, candidate, evaluation))
                if evaluation.exact:
                    break
            if scored:
                scored.sort(key=lambda row: row[:2])
                _score, _n, origin, text, base = scored[0]
                origin = "draft:" + origin
                case.stages.append({"stage": "drafts", "variants": [
                    {"label": row[2], "score": None if row[0] == 1 << 30 else row[0]} for row in scored]})
            error = "no draft variant was produced"
        else:
            text, error = drafted
        if text is None:
            case.stages.append({"stage": "source", "origin": "draft", "error": (error or "")[-300:]})
            return finish("needs-source", causemod.diagnose(target, None, vram, flow, neighbours),
                          "the automatic draft failed: " + (error or "no output")[-200:])
    if text is None:
        return finish("needs-source", causemod.diagnose(target, None, vram, flow, neighbours))

    # 2b. drafts x recipes.  The portfolio was ranked under one recipe; a draft that is a near
    #     miss there is often exact under a sibling (another version of the same compiler, the
    #     flags of a neighbouring unit), and it need not be the draft that ranked first.
    if drafts_x_recipes and recipes is not None and base is not None and not base.exact and scored:
        candidates = [row for row in scored if row[0] != 1 << 30][:CROSS_DRAFTS]
        tried, chosen = 0, None
        for label, recipe_pipeline, recipe_evaluate in list(recipes(family))[:cross_recipes or CROSS_RECIPES]:
            for row in candidates:
                evaluation, code = recipe_evaluate(row[3])
                tried += 1
                case.builds += 1
                if evaluation.score is not None and (base.score is None or evaluation.score < base.score
                                                     or (evaluation.exact and not base.exact)):
                    base, text, origin, chosen = evaluation, row[3], "draft:" + row[2], label
                    evaluate, pipeline = recipe_evaluate, recipe_pipeline
                    case.recipe = label
                    built[hashlib.sha256(text.encode()).hexdigest()] = code
                if base.exact:
                    break
            if base.exact:
                break
        case.stages.append({"stage": "drafts-x-recipes", "tried": tried, "chosen": chosen, "score": base.score})
        if chosen:
            event("recipe-selected", recipe=chosen, score=base.score)

    # 2c. frame fit.  When the candidate's stack frame differs from the target's, every
    #     stack-relative instruction differs and the score says nothing about the rest of the
    #     draft.  The difference is measured, so the repair is computed: substitute the
    #     temporaries that only name sub-expressions, then reserve what is still missing.
    if frame_fit and flow is not None and flow.frame.size and base is None:
        base = run(text)                                       # a supplied source: its baseline, once
        case.builds += 1
    if frame_fit and flow is not None and flow.frame.size and base is not None and not base.exact \
            and base.score is not None:
        from . import draftforms

        def frame_of(candidate: str) -> int | None:
            code = built.get(hashlib.sha256(candidate.encode()).hexdigest())
            if not code:
                return None
            try:
                return flowmod.analyse(code, vram).frame.size
            except Exception:
                return None

        tried, fitted = 0, None
        pool = [text] + [row[3] for row in scored if row[0] != 1 << 30 and row[3] != text][:1]
        for start in pool:
            size = frame_of(start)
            if size is None or size == flow.frame.size:
                continue
            attempts = []
            if size > flow.frame.size:                       # too many locals: fewer temporaries
                for only, label in ((r"temp_\w+", "inlined"), (None, "inlined all")):
                    slim, steps = draftforms.inlined(start, name, only=only)
                    if not steps:
                        continue
                    evaluation = run(slim)
                    tried += 1
                    case.builds += 1
                    attempts.append((evaluation, slim, label))
                    start, size = slim, frame_of(slim) or size
                    if size <= flow.frame.size:
                        break
            if size < flow.frame.size:                       # bytes missing: reserve them
                missing = flow.frame.size - size
                for amount, where in ((missing, "top"), (missing, "bottom"), (missing - 4, "top")):
                    padded = draftforms.pad(start, name, amount, where) if amount > 0 else None
                    if padded is None:
                        continue
                    evaluation = run(padded)
                    tried += 1
                    case.builds += 1
                    attempts.append((evaluation, padded, f"pad {amount} {where}"))
                    if evaluation.exact:
                        break
            for evaluation, candidate, label in attempts:
                if evaluation.score is not None and (evaluation.score < base.score or
                                                     (evaluation.exact and not base.exact)):
                    base, text, fitted = evaluation, candidate, label
                    origin = origin + "+frame"
            if base.exact:
                break
        if tried:
            case.stages.append({"stage": "frame-fit", "tried": tried, "chosen": fitted, "score": base.score,
                                "target_frame": flow.frame.size})
            event("frame-fit", tried=tried, chosen=fitted, score=base.score)
    case.stages.append({"stage": "source", "origin": origin})

    # 3. baseline
    if base is None:
        base = run(text)
        case.builds += 1
    wrong_family = family is not None and pipeline is not None and causemod.pipeline_family(pipeline) != family
    gated = (wide_recipes is not None and not wrong_family and base.score is not None
             and base.score > 4 * near and not wide_recipes())
    if gated:
        case.stages.append({"stage": "recipe-skipped", "reason": "no unoptimised-code evidence in the function "
                                                                  "or its neighbours; family agrees"})
    if recipes is not None and not base.exact and not gated and (wrong_family or base.score is None
                                                                  or base.score > 4 * near):
        tried, chosen = 0, None
        for label, recipe_pipeline, recipe_evaluate in recipes(family):
            evaluation, code = recipe_evaluate(text)
            tried += 1
            case.builds += 1
            if evaluation.score is not None and (base.score is None or evaluation.score < base.score):
                base, chosen, evaluate, pipeline = evaluation, label, recipe_evaluate, recipe_pipeline
                case.recipe = label
                built[hashlib.sha256(text.encode()).hexdigest()] = code
                if evaluation.exact:
                    break
        case.stages.append({"stage": "recipe", "tried": tried, "chosen": chosen, "score": base.score})
        if chosen:
            event("recipe-selected", recipe=chosen, score=base.score)
    if base.score is None:
        # the draft is kept although it does not build: its error is the next thing to fix, by a
        # repair rule or by whoever is asked about the case later
        case.stages.append({"stage": "baseline", "error": (base.error or "")[-900:],
                            "source_digest": store.save_source(text)})
        return finish("does-not-build", causemod.diagnose(target, None, vram, flow, neighbours),
                      "fix the compile error: " + (base.error or "")[-200:])
    # 3b. near miss: sibling versions before source permutation.  A source that is a few
    #     instructions off at the right size is often right for another version of the same
    #     compiler (Sesame Street: the speech library was IDO 7.1 in a title configured IDO 5.3).
    #     For IDO the local-free form of the source (cached field copies read in place) is tried
    #     under the current and the sibling recipes too: it was what IDO 7.1 needed.
    state = {"base": base, "text": text, "evaluate": evaluate, "pipeline": pipeline}

    def near_sibling_stage(stage: str = "near-siblings", extra=()) -> None:
        """Sibling versions (and, for IDO, the local-free forms) of a near miss; updates ``state``."""
        base, text, evaluate, pipeline = state["base"], state["text"], state["evaluate"], state["pipeline"]
        if not (near_siblings and not base.exact and base.score is not None and base.score <= near):
            return
        from . import compilers as compmod
        from . import draftforms
        code_now = built.get(hashlib.sha256(text.encode()).hexdigest())
        # the same size up to a few instructions (alignment padding; nops an assembler fills or drops)
        same_size = code_now is None or abs(len(code_now.rstrip(b"\0")) - len(target.rstrip(b"\0"))) <= 16
        current = case.recipe or recipe_label
        forms = [("source", text)] + [(label, form) for label, form in extra if form != text]
        if causemod.pipeline_family(pipeline) == "ido" or (current or "").startswith("ido"):
            forms += [(label, form) for label, form, _review in draftforms.local_free_drafts(text, name)]
        lanes = [("current", pipeline, evaluate)]
        if same_size and recipes is not None and current:
            try:
                lanes += [(label, rp, rev) for label, rp, rev in recipes(family)
                          if compmod.sibling_recipe(current, label, region_levels)]
            except Exception:
                pass
        tried, chosen = 0, None
        for lane_label, lane_pipeline, lane_evaluate in lanes:
            for form_label, form in forms:
                if lane_label == "current" and (form_label == "source" or (form_label, form) in extra):
                    continue                                   # already built under this recipe
                evaluation, code = lane_evaluate(form)
                tried += 1
                case.builds += 1
                if evaluation.score is not None and (evaluation.score < base.score or
                                                     (evaluation.exact and not base.exact)):
                    base, text, chosen = evaluation, form, (lane_label, form_label)
                    built[hashlib.sha256(form.encode()).hexdigest()] = code
                    if lane_label != "current":
                        evaluate, pipeline, case.recipe = lane_evaluate, lane_pipeline, lane_label
                if base.exact:
                    break
            if base.exact:
                break
        if tried:
            case.stages.append({"stage": stage, "tried": tried, "lanes": [l for l, _p, _e in lanes],
                                "forms": [f for f, _t in forms], "chosen": list(chosen) if chosen else None,
                                "score": base.score})
            event(stage, tried=tried, chosen=chosen, score=base.score)
        state.update(base=base, text=text, evaluate=evaluate, pipeline=pipeline)

    near_sibling_stage()
    base, text, evaluate, pipeline = state["base"], state["text"], state["evaluate"], state["pipeline"]
    case.best_score, case.best_digest = base.score, store.save_source(text)
    case.stages.append({"stage": "baseline", "score": base.score, "exact": base.exact})
    if base.outcome and base.outcome.get("proof"):
        case.proof = base.outcome["proof"]
        case.stages.append({"stage": "strict-verify", "verdict": case.proof.get("verdict"),
                            "assumptions": len(case.proof.get("assumptions", []))})
    event("baseline", score=base.score, exact=base.exact)
    if base.exact:
        return finish("exact", [], "strictly verified; prepare independent integration")
    if base.score == 0 and case.proof and case.proof.get("verdict") == "assumed":
        return finish("assumed", found, "declare or independently prove every strict-verifier assumption")

    # 4. diagnose
    candidate_code = built.get(hashlib.sha256(text.encode()).hexdigest())
    ranked = causemod.diagnose(target, candidate_code, vram, flow, neighbours, pipeline) if candidate_code is not None \
        else found
    case.stages.append({"stage": "diagnose", "causes": [c.id for c in ranked][:6]})

    # 4b. loop form.  When the only difference is a consistent exchange of registers, GCC 2.7.2 ranked
    #     two values the other way round, and references inside a loop count once per loop level: a
    #     loop written with a backward goto is no loop to it, while (1) is (strcpy of the KMC C library,
    #     Sesame Street).  The goto <-> structured rewrites are tried, both directions, before the search.
    # A register-only difference counts too when the exchange is consistent only within each value's
    # lifetime (``register.pressure`` without structural differences, confidence 0.6): the strcpy case.
    registers_only = bool(ranked) and (ranked[0].id == "register.permutation" or
                                       (ranked[0].id == "register.pressure" and ranked[0].confidence >= 0.6))
    if loop_forms and registers_only and not base.exact:
        from . import loopforms
        tried, chosen, scores = 0, None, []
        for label, form in loopforms.variants(text, name):
            evaluation = run(form)
            tried += 1
            case.builds += 1
            scores.append((evaluation.score, label, form))
            if evaluation.score is not None and (evaluation.score < base.score or (evaluation.exact and not base.exact)):
                base, text, chosen = evaluation, form, label
                if evaluation.exact:
                    break
        if tried:
            case.stages.append({"stage": "loop-form", "tried": tried, "chosen": chosen, "score": base.score})
            event("loop-form", tried=tried, chosen=chosen, score=base.score)
        if chosen:
            case.best_score, case.best_digest = base.score, store.save_source(text)
            if base.outcome and base.outcome.get("proof"):
                case.proof = base.outcome["proof"]
            state.update(base=base, text=text, evaluate=evaluate, pipeline=pipeline)
            if not base.exact:                  # the new form may be right for a sibling version
                # every loop form as close as the chosen one is tried under the siblings
                near_sibling_stage("near-siblings-after-loop-form",
                                   [(label, form) for score, label, form in scores if score == base.score])
                base, text, evaluate, pipeline = state["base"], state["text"], state["evaluate"], state["pipeline"]
                case.best_score, case.best_digest = base.score, store.save_source(text)
                if base.outcome and base.outcome.get("proof"):
                    case.proof = base.outcome["proof"]
            if base.exact:
                return finish("exact", [], "strictly verified; prepare independent integration")

    # 4c. hand forms (n64dk.match.handforms): the rewrites that matched functions by hand in the
    #     Madden Football 64 campaign, chosen by the diagnosis (globals as one aggregate or volatile
    #     for schedule/reload causes, chained or split assignments, tails per branch or merged,
    #     locals narrowed to the target's own extension widths, top-tested loops for frame sizes).
    #     Here only a form that is exact by itself is taken: a form that is merely closer would move
    #     the search's start away from the path it finds from the draft (measured: a split variable
    #     the search found from the draft was no longer reached), and the pass is kept to an eighth
    #     of the budget.  Inside the search the same forms are the ``hand`` family, which combines
    #     with the other edits; what the search leaves of the budget goes to the forms of its best
    #     source (stage 5c).
    from . import handforms
    assisted = handforms.fixmul_sites(target)
    if assisted:
        case.stages.append({"stage": "assisted-class", "kind": "fixmul", "sites": assisted[:8],
                            "note": "fixed-point multiply macro (inline asm in the project): assisted, not plain C"})
    hand_causes = [c.id for c in ranked]
    hand_reserve = 0

    def hand_stage(stage: str, start_text: str, start: itermod.Evaluation, allowance: int, greedy: bool):
        """Hand forms of ``start_text``; returns (text, evaluation, chosen, tried)."""
        tried, chosen, best_text, best_eval = 0, [], start_text, start
        for _round in range(2 if greedy else 1):
            improved = False
            for label, form in handforms.forms_for(hand_causes, best_text, name, target):
                if tried >= allowance:
                    break
                evaluation = run(form)
                tried += 1
                case.builds += 1
                better = evaluation.exact or (greedy and evaluation.score is not None and
                                              best_eval.score is not None and evaluation.score < best_eval.score)
                if better:
                    best_text, best_eval, improved = form, evaluation, True
                    chosen.append(label)
                    if evaluation.exact:
                        break
            if best_eval.exact or not improved:
                break
        if tried:
            case.stages.append({"stage": stage, "tried": tried, "chosen": chosen, "score": best_eval.score,
                                "kinds": handforms.kinds_for(hand_causes)})
            event(stage, tried=tried, chosen=chosen, score=best_eval.score)
        return best_text, best_eval, chosen

    if hand_forms and not base.exact and base.score is not None and base.score <= 4 * near and hand_causes:
        hand_reserve = 1                                   # marks the stage as on; nothing is held back
        new_text, new_eval, chosen = hand_stage("hand-forms", text, base,
                                                min(hand_forms_budget, max(2, budget // 8),
                                                    max(0, budget - case.builds)), False)
        if chosen and new_eval.exact:
            base, text = new_eval, new_text
            case.best_score, case.best_digest = base.score, store.save_source(text)
            if base.outcome and base.outcome.get("proof"):
                case.proof = base.outcome["proof"]
            return finish("exact", [], "strictly verified; prepare independent integration")

    # 5. iterate: one search over every family, the ones the causes justify first, so that edits
    #    from different families can combine; families closed for this exact source on an earlier
    #    run are not repeated
    best_text, best = text, base
    source_digest = hashlib.sha256(best_text.encode()).hexdigest()
    families = [f for f in causemod.family_order(ranked, itermod.FAMILIES)
                if f not in case.closed_families.get(source_digest, [])]
    allowance = max(0, budget - case.builds)
    if search_ceiling and best.score is not None and best.score > search_ceiling:
        case.stages.append({"stage": "iterate-skipped", "score": best.score, "ceiling": search_ceiling})
        event("iterate-skipped", score=best.score, ceiling=search_ceiling)
        families = []
    if not search and families:
        case.stages.append({"stage": "iterate-skipped", "reason": "variant search switched off (ablation)"})
        event("iterate-skipped", reason="switched off")
        families = []
    stratum = (governor.stratum(stratum or recipe_label or pipeline or "", best.score)
               if governor is not None else None)
    prediction = governor.predict(stratum) if governor is not None and families and allowance >= 4 \
        and not best.exact else None
    if governor is not None and families and allowance >= 4 and not best.exact:
        allowed, why = governor.allow(stratum)
        if not allowed:
            case.stages.append({"stage": "iterate-skipped", "reason": why, "stratum": stratum})
            event("iterate-skipped", reason=why, stratum=stratum)
            families = []
    if families and allowance >= 4 and not best.exact:
        searched_at = time.monotonic()
        result = itermod.descend(best_text, name, run, tuple(families), jobs=jobs, budget=allowance)
        spent = result.get("evaluated", 0)
        case.builds += spent
        if governor is not None:
            governor.record(stratum, result.get("start_score"), result.get("final_score"), spent,
                            bool(result.get("exact")))
            if prediction is not None:
                governor.observe(prediction, result.get("start_score"), result.get("final_score"), spent,
                                 bool(result.get("exact")), time.monotonic() - searched_at)
        improved = result.get("final_score") is not None and result["final_score"] < best.score
        helpful = {row["family"] for row in result.get("sensitivity", []) if row.get("delta", 0) < 0}
        if not improved:
            closed = set(case.closed_families.get(source_digest, [])) | (set(families) - helpful)
            case.closed_families[source_digest] = sorted(closed)
        case.stages.append({"stage": "iterate", "families": families[:12], "builds": spent,
                            "start": result.get("start_score"), "final": result.get("final_score"),
                            "applied": [s.get("applied") for s in result.get("steps", []) if s.get("applied")],
                            "helpful_families": sorted(helpful)})
        event("iterate", start=result.get("start_score"), final=result.get("final_score"), builds=spent)
        if improved and result.get("source"):
            best_text = result["source"]
            best = itermod.Evaluation(result["final_score"], bool(result.get("exact")))
            case.best_score, case.best_digest = best.score, store.save_source(best_text)
            if best.exact:
                verified, _code = evaluate(best_text)
                case.proof = (verified.outcome or {}).get("proof")
                case.stages.append({"stage": "strict-verify", "verdict":
                                    (case.proof or {}).get("verdict", "missing")})
                if not verified.exact:
                    return finish("assumed", ranked,
                                  "candidate is exploration-exact but lacks strict relocation/data proof")
                event("exact-found", builds=case.builds)
                return finish("exact", [], "strictly verified; prepare independent integration")

    # 5b. the recipe's rewrites of the best source
    if rewrites is not None and not best.exact and best.score is not None and best.score > 0:
        try:
            offered = list(rewrites(best_text))[:REWRITES_OF_BEST]
        except Exception:                                    # a rewrite is optional; the best source stands
            offered = []
        tried, start = 0, best.score
        for label, variant in offered:
            evaluation = run(variant)
            case.builds += 1
            tried += 1
            if evaluation.score is None or evaluation.score >= best.score:
                continue
            best_text, best = variant, evaluation
            case.best_score, case.best_digest = best.score, store.save_source(best_text)
            if best.exact:
                break
        if tried:
            case.stages.append({"stage": "rewrite-best", "tried": tried, "start": start, "final": best.score})
            event("rewrite-best", tried=tried, start=start, final=best.score)
        if best.exact:
            verified, _code = evaluate(best_text)
            case.proof = (verified.outcome or {}).get("proof")
            case.stages.append({"stage": "strict-verify", "verdict": (case.proof or {}).get("verdict", "missing")})
            if not verified.exact:
                return finish("assumed", ranked,
                              "candidate is exploration-exact but lacks strict relocation/data proof")
            event("exact-found", builds=case.builds)
            return finish("exact", [], "strictly verified; prepare independent integration")

    # 5c. hand forms of the search's best source (greedy, from the allowance kept for them)
    if hand_reserve and not best.exact and best.score is not None and best.score > 0:
        new_text, new_eval, chosen = hand_stage("hand-forms-best", best_text, best,
                                                min(hand_forms_budget, max(0, budget - case.builds)), True)
        if chosen:
            best_text, best = new_text, new_eval
            case.best_score, case.best_digest = best.score, store.save_source(best_text)
            if not best.exact:
                state.update(base=best, text=best_text)
                near_sibling_stage("near-siblings-after-hand-forms")
                evaluate, pipeline = state["evaluate"], state["pipeline"]
                if state["base"].score is not None and state["base"].score < best.score:
                    best, best_text = state["base"], state["text"]
                    case.best_score, case.best_digest = best.score, store.save_source(best_text)
        if best.exact:
            verified, _code = evaluate(best_text)
            case.proof = (verified.outcome or {}).get("proof")
            case.stages.append({"stage": "strict-verify", "verdict": (case.proof or {}).get("verdict", "missing")})
            if not verified.exact:
                return finish("assumed", ranked,
                              "candidate is exploration-exact but lacks strict relocation/data proof")
            event("exact-found", builds=case.builds)
            return finish("exact", [], "strictly verified; prepare independent integration")

    # 6. classify what is left
    final_code = built.get(hashlib.sha256(best_text.encode()).hexdigest())
    if final_code is None:
        _evaluation, final_code = evaluate(best_text)
    ranked = causemod.diagnose(target, final_code, vram, flow, neighbours, pipeline) if final_code is not None else ranked
    if case.best_score == 0:
        # byte-identical under the exploration oracle, yet never accepted as exact: keep the
        # strict verdict with the case instead of calling it a near miss
        verified, _code = evaluate(best_text)
        case.proof = (verified.outcome or {}).get("proof") or case.proof
        return finish("assumed", ranked, "declare or independently prove every strict-verifier assumption")
    return finish("near" if (case.best_score or 0) <= near else "open", ranked)


def hard_queue(cases: list[Case]) -> dict:
    """Unresolved cases grouped by top cause and ordered by expected progress."""
    groups: dict[str, dict] = {}
    for case in cases:
        if case.status in ("exact", "stale"):
            continue
        key = case.top_cause or case.status
        group = groups.setdefault(key, {"cause": key, "meaning": causemod.TAXONOMY.get(key, case.status),
                                        "functions": 0, "bytes": 0, "cases": []})
        group["functions"] += 1
        group["bytes"] += case.size
        group["cases"].append((case.best_score if case.best_score is not None else 1 << 30, case.size, case.name,
                               case.case_id, case.status))
    rows = []
    for key, group in groups.items():
        group["cases"].sort()
        group["expected_progress"] = round(group["bytes"] * TRACTABILITY.get(key, 0.2))
        group["best"] = [{"name": n, "case": c, "score": None if s == 1 << 30 else s, "size": size, "status": st}
                         for s, size, n, c, st in group.pop("cases")[:8]]
        rows.append(group)
    rows.sort(key=lambda g: -g["expected_progress"])
    statuses: dict[str, int] = {}
    for case in cases:
        statuses[case.status] = statuses.get(case.status, 0) + 1
    return {"cases": len(cases), "statuses": statuses, "queue": rows}


def export_queue(store: CaseStore, out: Path, image: str | None = None, include_assumed: bool = True) -> dict:
    """Write every exact case as an integration queue: one source per function, a manifest-style
    table (``ROM_START  SIZE  SOURCE  FLAGS`` with the pipeline in a comment per row group) and
    the symbols/sections each unit still needs declared.  Strictly exact cases go to
    ``queue.tsv``.  Cases that are byte-identical but rest on undeclared addresses (status or
    verdict ``assumed``) go, when ``include_assumed``, to ``assumed.tsv`` with every assumption in
    ``assumptions.tsv`` and the declarations that would settle them in ``symbols.assumed.txt``
    (one ``NAME = 0xADDRESS;`` per symbol, for review: declaring them is the operator's decision,
    after which a rerun verifies those cases strictly).  Nothing is written anywhere but ``out``."""
    out = Path(out)
    rows, skipped, assumptions, assembly = [], [], [], []
    placed: list[tuple[str, int, str, str]] = []
    declarations: dict[tuple[str, str], list[str]] = {}
    for case in store.all():
        if image and case.image != image:
            continue
        if case.status == handasm.STATUS:
            # hand-written code kept as assembly source that assembles to the same bytes
            text = store.source(case.best_digest)
            if text is None or (case.strict or {}).get("verdict") != "exact":
                skipped.append({"case": case.case_id, "name": case.name, "strict": "assembly-unproved"})
                continue
            folder = out / case.image
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{case.name}.s").write_text(text)
            assembly.append((case.image, case.rom_offset, case.vram, case.size, f"{case.image}/{case.name}.s"))
            continue
        if case.status not in ("exact", "assumed"):
            continue
        verdict = (case.strict or case.proof or {}).get("verdict", "unverified")
        if case.status == "assumed" and verdict == "exact":
            verdict = "assumed"
        if verdict not in ("exact", "assumed") or (verdict == "assumed" and not include_assumed):
            skipped.append({"case": case.case_id, "name": case.name, "strict": verdict})
            continue
        text = store.source(case.best_digest)
        if text is None:
            skipped.append({"case": case.case_id, "name": case.name, "strict": "source-missing"})
            continue
        folder = out / case.image
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{case.name}.c").write_text(text)
        pipeline, _, flags = case.recipe.partition(" ")
        if verdict == "exact":
            # A section the strict comparison placed by its own contents (a switch's jump table, constants:
            # ``corroborated``).  A link needs that address, or the unit's code differs from the ROM; the recipe
            # itself stays as verified (a receipt rebuilds under it and confirms the section again), so the
            # placements are listed beside the queue.
            for section, address in sorted(((case.strict or case.proof or {}).get("corroborated") or {}).items()):
                placed.append((case.image, case.vram, section, address))
        rows.append((case.image, pipeline, case.rom_offset, case.vram, case.size, f"{case.image}/{case.name}.c",
                     flags, verdict))
        for item in (case.strict or case.proof or {}).get("assumptions", []):
            assumptions.append(f"{case.image}\t{case.name}\t{item.get('kind')}\t{item.get('name')}\t"
                               f"{item.get('address')}")
            if item.get("kind") == "symbol" and item.get("name") and item.get("address"):
                declarations.setdefault((item["name"], str(item["address"])), []).append(case.name)
    rows.sort()

    def table(wanted: str) -> list[str]:
        lines, current = [], None
        for image_name, pipeline, rom_offset, vram, size, source, flags, verdict in rows:
            if verdict != wanted:
                continue
            if (image_name, pipeline) != current:
                current = (image_name, pipeline)
                lines.append(f"# image {image_name}; pipeline {pipeline}; ROM_START SIZE SOURCE FLAGS  "
                             "(# VRAM, strict verdict)")
            start = f"0x{rom_offset:X}" if rom_offset is not None else "?"
            lines.append(f"{start}\t0x{size:X}\t{source}\t{flags}\t# 0x{vram:08X} {verdict}")
        return lines

    if assembly:
        out.mkdir(parents=True, exist_ok=True)
        lines, current = [], None
        for image_name, rom_offset, vram, size, source in sorted(assembly, key=lambda r: (r[0], r[2])):
            if image_name != current:
                current = image_name
                lines.append(f"# image {image_name}; {handasm.RECIPE}; ROM_START SIZE SOURCE  (# VRAM)")
            start = f"0x{rom_offset:X}" if rom_offset is not None else "?"
            lines.append(f"{start}\t0x{size:X}\t{source}\t# 0x{vram:08X}")
        (out / "assembly.tsv").write_text("\n".join(lines) + "\n")
    if rows:
        out.mkdir(parents=True, exist_ok=True)
        (out / "placed_sections.tsv").write_text(
            "# image\tvram\tsection\taddress\torigin  (sections of exact units placed by their own contents)\n"
            + "".join(f"{image_name}\t0x{vram:08X}\t{section}\t{address}\tsection-content\n"
                      for image_name, vram, section, address in sorted(placed)))
        (out / "queue.tsv").write_text("\n".join(table("exact")) + "\n")
        (out / "assumed.tsv").write_text("\n".join(table("assumed")) + "\n")
        (out / "assumptions.tsv").write_text("# image\tunit\tkind\tname\taddress\n" + "\n".join(assumptions) + "\n")
        (out / "symbols.assumed.txt").write_text(
            "// Addresses the strict verifier had to assume.  Review, then append to the project's symbol file.\n" +
            "".join(f"{name} = {address}; // used by {', '.join(sorted(set(users))[:4])}\n"
                    for (name, address), users in sorted(declarations.items(), key=lambda kv: kv[0][1])))
    return {"exported": len(rows), "bytes": sum(r[4] for r in rows),
            "strict_exact": sum(1 for r in rows if r[7] == "exact"),
            "assumed": sum(1 for r in rows if r[7] == "assumed"), "assembly": len(assembly),
            "assembly_bytes": sum(r[3] for r in assembly), "skipped": skipped, "out": str(out)}
