"""Multi-hypothesis world model: what the kit believes about a title, why, and what rests on it.

The kit decides many things about a title before it builds anything -- where functions start,
which compiler and flags built a region, where an overlay is linked, what ``_gp`` is -- and today
each decision is a value in some stage's output.  When a value is wrong, nothing says which cases
were built on it.  This module keeps those decisions as *hypotheses* with alternatives:

* a hypothesis is ``(kind, subject, claim)`` with a stable id (a content hash of the three), a
  confidence, a status (``open``, ``supported``, ``contradicted``, ``split``, ``retired``), the
  evidence for it and the contradictions against it (each with its provenance: the key of the
  record it comes from, the method, and its *basis*), the hypotheses it depends on, the cases and
  cache keys computed under it, its consequences (predictions a strict-exact result can settle)
  and its cost to test;
* alternatives for one subject compete in a bounded beam (``beam``, default 4): the best ``beam``
  live alternatives stay, the rest are retired, never deleted; a strictly supported one is never
  retired;
* evidence raises confidence (noisy-or), a contradiction lowers it and may split the hypothesis
  into alternatives; nothing is overwritten: every change is a new version in the repository
  (``n64dk.match.evidence_store``), so ``history(id)`` shows how a belief evolved;
* ``settle(result)`` takes a strict verifier verdict for one case.  Only ``exact`` settles
  anything (a build that does not match says the source is wrong, not that a premise is).  It
  supports the hypotheses whose consequences it confirms, contradicts those it refutes,
  propagates a contradiction to every dependent, and returns the cases to reopen and the cache
  keys to invalidate;
* queries: ``why(subject)``, ``alternatives(subject)``, ``dependents(id)``, ``what_if(id)`` (the
  hypothesis false: what is affected) and ``reopen_triggers(case)``;
* ``from_run(run)`` imports the facts a finished ``decompile-run`` directory already holds
  (catalogue and repaired function starts, region recipes and the calibrated recipe, overlay and
  placed image bases, ``_gp``), reading only.

Confidence is bookkeeping, never credit: no hypothesis, however supported, makes a function
exact, owns a byte or turns a value recovered from the target into proof.  Basis ``strict-exact``
is reserved for evidence that came from a strict verifier verdict.
"""
from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

from .evidence_store import MemoryRepository, Repository, canonical, handle, sha256

KIND = "hypothesis"
STATUSES = ("open", "supported", "contradicted", "split", "retired")
LIVE = ("supported", "open")
BASES = ("strict-exact", "declared", "target-derived", "heuristic", "derived")
BEAM = 4
SUPPORTED_AT = 0.9            # confidence at which evidence without a strict-exact basis reads as supported
CONTRADICTED_BELOW = 0.1
HEURISTIC_CAP = 0.95          # what no strict-exact result backs never reaches certainty
PREMISE_WEIGHT = 0.5          # how much a contradicted premise lowers each dependent
CONFIDENCE = {"high": 0.8, "medium": 0.6, "low": 0.4}


def hypothesis_id(kind: str, subject: str, claim) -> str:
    return "h" + sha256(canonical([kind, subject, claim]).encode())[:24]


def evidence(key: str, source: str, *, basis: str = "heuristic", weight: float = 0.5, note: str = "") -> dict:
    """One piece of provenance: the key of the record it comes from, the method that produced it,
    its basis (``BASES``), its weight in (0, 1] and a note."""
    if not key or not source:
        raise ValueError("evidence names its key and its source")
    if basis not in BASES:
        raise ValueError(f"basis must be one of {', '.join(BASES)}")
    weight = round(float(weight), 6)
    if not 0.0 < weight <= 1.0:
        raise ValueError("weight must be in (0, 1]")
    return {"key": str(key), "source": str(source), "basis": basis, "weight": weight, "note": str(note)}


def _row(value) -> dict:
    if not isinstance(value, dict):
        raise TypeError("evidence is a dict made by evidence()")
    return evidence(value.get("key"), value.get("source"), basis=value.get("basis", "heuristic"),
                    weight=value.get("weight", 0.5), note=value.get("note", ""))


def _consequence(value) -> dict:
    if not isinstance(value, dict) or not value.get("case"):
        raise ValueError("a consequence names its case")
    out = json.loads(canonical(value))
    out.setdefault("predicts", "exact")
    if out["predicts"] not in ("exact", "not-exact"):
        raise ValueError("a consequence predicts exact or not-exact")
    return out


def _same(a: dict, b: dict) -> bool:
    return (a["key"], a["source"], a["note"]) == (b["key"], b["source"], b["note"])


def _merge(old, new) -> tuple:
    return tuple(dict.fromkeys(tuple(old) + tuple(new)))


def _norm(value):
    if isinstance(value, str):
        text = " ".join(value.split())
        if text.lower().startswith("0x"):
            try:
                return int(text, 16)
            except ValueError:
                return text.lower()
        return text
    return value


@dataclass(frozen=True)
class Hypothesis:
    kind: str
    subject: str
    claim: object
    confidence: float = 0.5
    status: str = "open"
    evidence: tuple = ()
    contradictions: tuple = ()
    depends_on: tuple = ()
    cache_keys: tuple = ()
    cases: tuple = ()
    consequences: tuple = ()
    cost_to_test: dict | None = None
    split_from: str | None = None

    @property
    def id(self) -> str:
        return hypothesis_id(self.kind, self.subject, self.claim)

    def as_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "subject": self.subject, "claim": self.claim,
                "confidence": round(self.confidence, 6), "status": self.status,
                "evidence": [dict(e) for e in self.evidence],
                "contradictions": [dict(c) for c in self.contradictions],
                "depends_on": list(self.depends_on), "cache_keys": list(self.cache_keys), "cases": list(self.cases),
                "consequences": [dict(c) for c in self.consequences], "cost_to_test": self.cost_to_test,
                "split_from": self.split_from}

    @classmethod
    def from_dict(cls, data: dict) -> "Hypothesis":
        return cls(data["kind"], data["subject"], data["claim"], float(data.get("confidence", 0.5)),
                   data.get("status", "open"), tuple(data.get("evidence") or ()),
                   tuple(data.get("contradictions") or ()), tuple(data.get("depends_on") or ()),
                   tuple(data.get("cache_keys") or ()), tuple(data.get("cases") or ()),
                   tuple(data.get("consequences") or ()), data.get("cost_to_test"), data.get("split_from"))


def _cap(h: Hypothesis) -> float:
    return 1.0 if any(e["basis"] == "strict-exact" for e in h.evidence) else HEURISTIC_CAP


def _derive(h: Hypothesis) -> str:
    """The status the evidence implies.  A strict-exact contradiction outranks everything; strict-exact
    evidence supports; ``split`` and ``retired`` are kept until one of those arrives; low confidence
    reads as contradicted only when there is a contradiction (weak evidence alone is just open)."""
    if any(c["basis"] == "strict-exact" for c in h.contradictions):
        return "contradicted"
    if any(e["basis"] == "strict-exact" for e in h.evidence):
        return "supported"
    if h.status in ("split", "retired"):
        return h.status
    if h.confidence < CONTRADICTED_BELOW and h.contradictions:
        return "contradicted"
    if h.confidence >= SUPPORTED_AT:
        return "supported"
    return "open"


def _rank(h: Hypothesis):
    return (h.status != "supported", -h.confidence, h.id)


@dataclass(frozen=True)
class Settlement:
    case: str
    settled: tuple[str, ...]          # hypotheses the result supported or contradicted directly
    changed: tuple[str, ...]          # every hypothesis whose record changed (dependents included)
    reopen: tuple[str, ...]           # cases computed under a belief that changed
    invalidate: tuple[str, ...]       # cache keys computed under a belief that changed
    note: str = ""

    def as_dict(self) -> dict:
        return {"case": self.case, "settled": list(self.settled), "changed": list(self.changed),
                "reopen": list(self.reopen), "invalidate": list(self.invalidate), "note": self.note}


class WorldModel:
    """Hypotheses behind a ``Repository`` (in memory by default).  Every change is a new record
    version; the model's own indexes are rebuilt from the repository's latest versions."""

    def __init__(self, repository: Repository | None = None, *, beam: int = BEAM):
        if beam < 1:
            raise ValueError("the beam keeps at least one alternative")
        self.repository = repository if repository is not None else MemoryRepository()
        self.beam = beam
        self._touched: set[str] | None = None
        self._reload()

    # ---- storage and indexes
    def _reload(self) -> None:
        self._by_id: dict[str, Hypothesis] = {}
        self._groups: dict[tuple[str, str], set[str]] = {}
        self._subjects: dict[str, set[str]] = {}
        self._dependents: dict[str, set[str]] = {}
        self._by_case: dict[str, set[str]] = {}
        for record in self.repository.records(KIND):
            self._index(Hypothesis.from_dict(record.payload))

    @contextlib.contextmanager
    def _atomic(self):
        """One repository transaction.  When it fails the indexes are rebuilt from what the
        repository holds, so the model never shows a change the store rolled back."""
        try:
            with self.repository.transaction():
                yield
        except BaseException:
            self._reload()
            raise

    @staticmethod
    def _cases_of(h: Hypothesis) -> set[str]:
        return set(h.cases) | {c["case"] for c in h.consequences}

    def _index(self, h: Hypothesis) -> None:
        old = self._by_id.get(h.id)
        if old is not None:
            for premise in old.depends_on:
                self._dependents.get(premise, set()).discard(h.id)
            for case in self._cases_of(old):
                self._by_case.get(case, set()).discard(h.id)
        self._by_id[h.id] = h
        self._groups.setdefault((h.kind, h.subject), set()).add(h.id)
        self._subjects.setdefault(h.subject, set()).add(h.kind)
        for premise in h.depends_on:
            self._dependents.setdefault(premise, set()).add(h.id)
        for case in self._cases_of(h):
            self._by_case.setdefault(case, set()).add(h.id)

    def _store(self, h: Hypothesis, event: str, cause: str = "") -> Hypothesis:
        deps = [handle(KIND, premise) for premise in h.depends_on] + [e["key"] for e in h.evidence]
        self.repository.put(KIND, h.id, h.as_dict(), provenance={"event": event, "cause": str(cause)[:300]},
                            depends_on=deps)
        self._index(h)
        if self._touched is not None:
            self._touched.add(h.id)
        return h

    def get(self, hid: str) -> Hypothesis:
        try:
            return self._by_id[hid]
        except KeyError:
            raise KeyError(f"unknown hypothesis {hid}") from None

    def hypotheses(self, kind: str | None = None, subject: str | None = None) -> list[Hypothesis]:
        return sorted((h for h in self._by_id.values() if (kind is None or h.kind == kind)
                       and (subject is None or h.subject == subject)), key=lambda h: (h.kind, h.subject, _rank(h)))

    def history(self, hid: str) -> list[dict]:
        """Every version of one hypothesis: what changed it and what it said then."""
        return [{"version": r.version, "status": r.payload.get("status"), "confidence": r.payload.get("confidence"),
                 "event": r.provenance.get("event"), "cause": r.provenance.get("cause")}
                for r in self.repository.history(KIND, hid)]

    def export(self) -> list[dict]:
        return [self._by_id[hid].as_dict() for hid in sorted(self._by_id)]

    def _group(self, kind: str, subject: str) -> list[Hypothesis]:
        return [self._by_id[hid] for hid in sorted(self._groups.get((kind, subject), ()))]

    def _reaches(self, start: str, target: str) -> bool:
        stack, seen = [start], set()
        while stack:
            current = stack.pop()
            if current == target:
                return True
            if current in seen or current not in self._by_id:
                continue
            seen.add(current)
            stack.extend(self._by_id[current].depends_on)
        return False

    # ---- operations
    def add(self, kind: str, subject: str, claim, *, confidence: float = 0.5, evidence=(), depends_on=(),
            cache_keys=(), cases=(), consequences=(), cost_to_test: dict | None = None,
            split_from: str | None = None) -> Hypothesis:
        """Add a hypothesis, or merge into the one with the same id: its lists grow and the new
        evidence raises its confidence; nothing it already holds is replaced.  ``confidence`` is
        the prior given ``evidence`` for a new hypothesis."""
        for value, what in ((kind, "kind"), (subject, "subject")):
            if not isinstance(value, str) or not value:
                raise ValueError(f"a hypothesis names its {what}")
        claim = json.loads(canonical(claim))
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        rows = tuple(_row(e) for e in evidence)
        predictions = tuple(_consequence(c) for c in consequences)
        hid = hypothesis_id(kind, subject, claim)
        for premise in depends_on:
            if premise not in self._by_id:
                raise KeyError(f"unknown premise {premise}")
            if premise == hid or self._reaches(premise, hid):
                raise ValueError(f"{hid} cannot depend on {premise}: the dependency would form a cycle")
        existing = self._by_id.get(hid)
        with self._atomic():
            if existing is None:
                h = Hypothesis(kind, subject, claim, confidence, "open", rows, (), _merge((), depends_on),
                               _merge((), cache_keys), _merge((), cases), predictions,
                               dict(cost_to_test) if cost_to_test else None, split_from)
                h = replace(h, confidence=min(h.confidence, _cap(h)))
                self._store(replace(h, status=_derive(h)), "add")
            else:
                h = replace(existing, depends_on=_merge(existing.depends_on, depends_on),
                            cache_keys=_merge(existing.cache_keys, cache_keys), cases=_merge(existing.cases, cases),
                            consequences=existing.consequences + tuple(c for c in predictions
                                                                       if c not in existing.consequences),
                            cost_to_test=existing.cost_to_test or (dict(cost_to_test) if cost_to_test else None))
                for row in rows:
                    h = self._absorb(h, row)
                if h != existing:
                    self._store(replace(h, status=_derive(h)), "merge")
            self._beam(kind, subject)
        return self._by_id[hid]

    @staticmethod
    def _absorb(h: Hypothesis, row: dict) -> Hypothesis:
        if any(_same(row, e) for e in h.evidence):
            return h
        h = replace(h, evidence=h.evidence + (row,))
        confidence = 1.0 - (1.0 - h.confidence) * (1.0 - row["weight"])
        return replace(h, confidence=min(confidence, _cap(h)))

    def add_evidence(self, hid: str, row: dict) -> Hypothesis:
        h = self.get(hid)
        row = _row(row)
        with self._atomic():
            new = self._absorb(h, row)
            if new != h:
                self._store(replace(new, status=_derive(new)), "evidence", row["note"] or row["key"])
            self._beam(h.kind, h.subject)
        return self._by_id[hid]

    def add_contradiction(self, hid: str, row: dict, alternatives=()) -> Hypothesis:
        """Record evidence against ``hid``: its confidence falls (to 0 for a strict-exact one), its
        evidence stays.  ``alternatives`` (claims, or ``(claim, confidence)`` pairs) are added for
        the same subject, inheriting its premises, cases and cache keys; ``hid`` is then ``split``."""
        h = self.get(hid)
        row = _row(row)
        with self._atomic():
            if not any(_same(row, c) for c in h.contradictions):
                confidence = 0.0 if row["basis"] == "strict-exact" else h.confidence * (1.0 - row["weight"])
                h = replace(h, contradictions=h.contradictions + (row,), confidence=confidence)
            created = []
            for alternative in alternatives:
                claim, prior = alternative if isinstance(alternative, tuple) else (alternative, None)
                claim = json.loads(canonical(claim))
                if claim == h.claim:
                    continue
                prior = max(0.05, min(0.5, row["weight"])) if prior is None else prior
                note = f"alternative raised when {hid} was contradicted" + (f": {row['note']}" if row["note"] else "")
                created.append(self.add(h.kind, h.subject, claim, confidence=prior,
                                        evidence=(dict(row, basis="derived", note=note),), depends_on=h.depends_on,
                                        cache_keys=h.cache_keys, cases=h.cases, cost_to_test=h.cost_to_test,
                                        split_from=hid).id)
            if created:
                h = replace(h, status="split")
            self._store(replace(h, status=_derive(h)), "contradiction", row["note"] or row["key"])
            self._beam(h.kind, h.subject)
        return self._by_id[hid]

    def _beam(self, kind: str, subject: str) -> None:
        """Keep the best ``beam`` live alternatives for a subject; retire the rest (a strictly
        supported one is never retired; a retired one returns when it ranks in the beam again)."""
        candidates = [h for h in self._group(kind, subject) if h.status in ("supported", "open", "retired")]
        for index, h in enumerate(sorted(candidates, key=_rank)):
            if h.status == "supported" or index < self.beam:
                if h.status == "retired":
                    self._store(replace(h, status=_derive(replace(h, status="open"))), "beam",
                                f"back in the top {self.beam} alternatives for {subject}")
            elif h.status != "retired":
                self._store(replace(h, status="retired"), "beam",
                            f"outside the top {self.beam} alternatives for {subject}")

    def leader(self, kind: str, subject: str) -> Hypothesis | None:
        """The alternative the kit currently acts on: supported first, then the most confident."""
        live = [h for h in self._group(kind, subject) if h.status in LIVE]
        return min(live, key=_rank, default=None)

    def settle(self, result: dict) -> Settlement:
        """Apply one strict verifier verdict: ``{"case", "verdict", "key", "source"}`` plus what the
        result establishes (``start``, ``size``, ``recipe``, ...), compared with each consequence
        that names the case."""
        case = result.get("case")
        if not case:
            raise ValueError("a result names its case")
        if result.get("verdict") != "exact":
            return Settlement(case, (), (), (), (), f"verdict {result.get('verdict')!r} is not strict-exact: a "
                              "build that does not match says the source is wrong, not that a premise is")
        key = result.get("key") or f"case:{case}"
        source = result.get("source") or "strict verifier"
        plan = []
        for hid in sorted(self._by_case.get(case, ())):
            for consequence in self._by_id[hid].consequences:
                if consequence["case"] == case:
                    plan.append((hid, *self._check(consequence, result)))
        groups = {(self._by_id[hid].kind, self._by_id[hid].subject) for hid, _holds, _why in plan}
        leaders = {group: getattr(self.leader(*group), "id", None) for group in groups}
        before = {hid: self._by_id[hid].status for hid, _holds, _why in plan}
        reopen: set[str] = set()
        invalidate: set[str] = set()
        self._touched = set()
        try:
            with self._atomic():
                for hid, holds, why in plan:
                    row = evidence(key, source, basis="strict-exact", weight=1.0, note=f"case {case}: {why}")
                    if holds:
                        self.add_evidence(hid, row)
                    else:
                        self.add_contradiction(hid, row)
                fallen = sorted({hid for hid, _h, _w in plan
                                 if self._by_id[hid].status == "contradicted" and before[hid] != "contradicted"})
                for hid in fallen:
                    reopen |= self._cases_of(self._by_id[hid])
                    invalidate |= set(self._by_id[hid].cache_keys)
                seen, queue = set(fallen), list(fallen)
                while queue:
                    premise = queue.pop(0)
                    for dependent in sorted(self._dependents.get(premise, ())):
                        if dependent in seen:
                            continue
                        seen.add(dependent)
                        self.add_contradiction(dependent, evidence(
                            handle(KIND, premise), "world model", basis="derived", weight=PREMISE_WEIGHT,
                            note=f"premise {premise} contradicted by a strict-exact result for {case}"))
                        reopen |= self._cases_of(self._by_id[dependent])
                        invalidate |= set(self._by_id[dependent].cache_keys)
                        queue.append(dependent)
                for group, old in sorted(leaders.items(), key=lambda kv: kv[0]):
                    new = self.leader(*group)
                    if old is not None and (new is None or new.id != old):
                        for hid in [old] + self.dependents(old, transitive=True):
                            reopen |= self._cases_of(self._by_id[hid])
                            invalidate |= set(self._by_id[hid].cache_keys)
            changed = tuple(sorted(self._touched))
        finally:
            self._touched = None
        reopen.discard(case)                      # the case that settled it is exact already
        return Settlement(case, tuple(sorted({hid for hid, _h, _w in plan})), changed, tuple(sorted(reopen)),
                          tuple(sorted(invalidate)))

    @staticmethod
    def _check(consequence: dict, result: dict) -> tuple[bool, str]:
        if consequence.get("predicts") == "not-exact":
            return False, "no exact match was predicted, and the case is strictly exact"
        for name in sorted(k for k in consequence if k not in ("case", "predicts", "note")):
            if name in result and _norm(consequence[name]) != _norm(result[name]):
                return False, f"strictly exact with {name} {result[name]}, where {consequence[name]} was predicted"
        return True, "strictly exact as predicted"

    # ---- queries
    def alternatives(self, subject: str, kind: str | None = None) -> list[Hypothesis]:
        kinds = sorted(k for k in self._subjects.get(subject, ()) if kind is None or k == kind)
        return [h for k in kinds for h in sorted(self._group(k, subject), key=_rank)]

    def dependents(self, hid: str, transitive: bool = False) -> list[str]:
        self.get(hid)
        if not transitive:
            return sorted(self._dependents.get(hid, ()))
        seen, queue = set(), [hid]
        while queue:
            for dependent in sorted(self._dependents.get(queue.pop(0), ())):
                if dependent not in seen:
                    seen.add(dependent)
                    queue.append(dependent)
        seen.discard(hid)
        return sorted(seen)

    def premises(self, hid: str) -> list[str]:
        """Every hypothesis ``hid`` rests on, transitively."""
        seen, stack = set(), list(self.get(hid).depends_on)
        while stack:
            current = stack.pop()
            if current not in seen and current in self._by_id:
                seen.add(current)
                stack.extend(self._by_id[current].depends_on)
        return sorted(seen)

    def _explain(self, h: Hypothesis, depth: int) -> dict:
        return {"id": h.id, "claim": h.claim, "status": h.status, "confidence": round(h.confidence, 4),
                "evidence": [dict(e) for e in h.evidence], "contradictions": [dict(c) for c in h.contradictions],
                "premises": [self._explain(self._by_id[p], depth - 1) if depth > 0 else {"id": p}
                             for p in h.depends_on if p in self._by_id]}

    def why(self, subject: str, kind: str | None = None, *, depth: int = 2) -> list[dict]:
        """Per kind of belief about ``subject``: what the kit believes, on which evidence (with
        provenance), against which contradictions, resting on which premises, and the alternatives."""
        out = []
        for k in sorted(k for k in self._subjects.get(subject, ()) if kind is None or k == kind):
            lead = self.leader(k, subject)
            out.append({"kind": k, "subject": subject, "believed": None if lead is None else lead.claim,
                        "because": None if lead is None else self._explain(lead, depth),
                        "alternatives": [{"id": h.id, "claim": h.claim, "status": h.status,
                                          "confidence": round(h.confidence, 4)}
                                         for h in sorted(self._group(k, subject), key=_rank)
                                         if lead is None or h.id != lead.id],
                        "reason": None if lead is not None else
                        "no live alternative: every one is contradicted, split or retired"})
        return out

    def what_if(self, hid: str) -> dict:
        """If ``hid`` were false: the hypotheses that rest on it, the cases and cache keys computed
        under them, the predictions at stake and the alternative that would lead.  Changes nothing."""
        h = self.get(hid)
        dependents = self.dependents(hid, transitive=True)
        affected = [h] + [self._by_id[d] for d in dependents]
        lead = self.leader(h.kind, h.subject)
        is_leader = lead is not None and lead.id == hid
        following = None
        if is_leader:
            rest = [x for x in self._group(h.kind, h.subject) if x.id != hid and x.status in LIVE]
            nxt = min(rest, key=_rank, default=None)
            following = None if nxt is None else {"id": nxt.id, "claim": nxt.claim,
                                                  "confidence": round(nxt.confidence, 4)}
        return {"hypothesis": hid, "kind": h.kind, "subject": h.subject, "claim": h.claim, "is_leader": is_leader,
                "next_leader": following, "dependents": dependents,
                "cases": sorted(set().union(*(self._cases_of(x) for x in affected))),
                "cache_keys": sorted(set().union(*(set(x.cache_keys) for x in affected))),
                "predictions": [{"hypothesis": x.id, "case": c["case"], "predicts": c["predicts"]}
                                for x in affected for c in x.consequences]}

    def reopen_triggers(self, case: str) -> list[dict]:
        """What would make ``case`` worth another attempt: each belief it was computed under (and
        each premise of those) being contradicted or losing its lead, and each prediction about it
        that a strict-exact result would settle."""
        direct = sorted(self._by_case.get(case, ()))
        out = []
        for hid in direct:
            h = self._by_id[hid]
            triggers = []
            if case in h.cases:
                triggers.append(f"{h.kind} {h.subject} = {canonical(h.claim)} is contradicted or another "
                                f"alternative becomes the leading one")
            for c in h.consequences:
                if c["case"] == case:
                    triggers.append(f"a strict-exact result for the case settles the prediction ({c['predicts']})")
            out.append({"hypothesis": hid, "kind": h.kind, "subject": h.subject, "claim": h.claim,
                        "status": h.status, "via": None, "triggers": triggers})
        seen = set(direct)
        for hid in direct:
            for premise in self.premises(hid):
                if premise in seen:
                    continue
                seen.add(premise)
                p = self._by_id[premise]
                out.append({"hypothesis": premise, "kind": p.kind, "subject": p.subject, "claim": p.claim,
                            "status": p.status, "via": hid,
                            "triggers": [f"premise of {hid}: {p.kind} {p.subject} = {canonical(p.claim)} is "
                                         f"contradicted"]})
        return out


# --------------------------------------------------------------------------- run directories

def _json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _int(value) -> int | None:
    try:
        return int(str(value), 16) if isinstance(value, str) else int(value)
    except (TypeError, ValueError):
        return None


def _hex(value) -> str | None:
    number = _int(value)
    return None if number is None else f"0x{number:08X}"


def _unit(value, low: float = 0.05, high: float = 0.85) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, number))


def run_cases(run: Path) -> list:
    """The run's cases, read-only (none when it has no case store)."""
    if not (Path(run) / "state" / "cases.sqlite").is_file():
        return []
    from . import resolve as resolvemod
    store = resolvemod.CaseStore(Path(run) / "state", readonly=True)
    try:
        return store.all()
    finally:
        store.close()


def from_run(run: str | Path, model: WorldModel | None = None) -> WorldModel:
    """Hypotheses from what a ``decompile-run`` directory already records, reading only:

    * ``catalogue``     per image, the discovery verdict after repair (``report.json``);
    * ``function-start`` per catalogued function (``solver/coverage_<image>.json``) and per start
      the repair added or moved (``discovery/repair_<image>.json``, with its evidence);
    * ``recipe``        per toolchain region, the ranked recipes and the recipes proven exact in
      it (``<image>/toolchain_regions.json``), and the calibrated recipe of an image;
    * ``image-base``    per overlay window (``discovery/overlays.json``, when
      ``discovery-audit --overlays --out <run>/discovery`` wrote one) and per placed segment;
    * ``gp``            the recovered and the project-declared ``_gp`` (``report.json``);

    then settles every strictly exact case of the case store against the predictions."""
    run = Path(run)
    model = model if model is not None else WorldModel()
    report = _json(run / "report.json")
    report = report if isinstance(report, dict) else {}
    cases = run_cases(run)
    with model._atomic():
        catalogue = _catalogues(model, report)
        _repairs(model, run)
        _function_starts(model, run, cases)
        _recipes(model, run, report, cases, catalogue)
        _image_bases(model, run, report)
        _small_data(model, run, report, cases)
        for case in sorted(cases, key=lambda c: c.case_id):
            if case.status == "exact" and (case.strict or {}).get("verdict") == "exact":
                model.settle({"case": case.case_id, "verdict": "exact", "key": f"state/cases.sqlite#{case.case_id}",
                              "source": "strict verifier (case store)", "start": f"0x{case.vram:08X}",
                              "size": case.size, "recipe": case.recipe})
    return model


def _catalogues(model: WorldModel, report: dict) -> dict[str, str]:
    found = report.get("discovery")
    out = {}
    if not isinstance(found, dict) or found.get("error"):
        return out
    for image, img in sorted((found.get("images") or {}).items()):
        if not isinstance(img, dict):
            continue
        verdict = img.get("verdict_after", img.get("verdict"))
        if not verdict:
            continue
        reasons = img.get("reasons_after", img.get("reasons")) or []
        h = model.add("catalogue", f"{image}:catalogue", {"verdict": verdict}, confidence=0.7, evidence=[
            evidence(f"report.json#/discovery/images/{image}", "discovery audit (call, pointer, prologue and datamap "
                     "evidence)", basis="target-derived", weight=0.7,
                     note=f"coverage {img.get('coverage_after', img.get('coverage'))}, "
                          f"{img.get('functions_after', img.get('functions'))} functions, "
                          f"{img.get('repaired_functions') or 0} added by repair"
                          + (f"; {'; '.join(str(r) for r in reasons[:3])}" if reasons else ""))],
            cost_to_test={"builds": 0, "note": "discovery-audit with another evidence source (a symbol map or "
                                               "source tree) would confirm or refute it"})
        out[image] = h.id
    return out


def _repairs(model: WorldModel, run: Path) -> None:
    folder = run / "discovery"
    if not folder.is_dir():
        return
    for path in sorted(folder.glob("repair_*.json")):
        image = path.stem[len("repair_"):]
        data = _json(path)
        for entry in (data.get("added") if isinstance(data, dict) else None) or []:
            start = _hex(entry.get("start")) if isinstance(entry, dict) else None
            if start is None:
                continue
            weight = CONFIDENCE.get(entry.get("confidence"), 0.5)
            reasons = entry.get("evidence") or ["the repair added this start"]
            rows = [evidence(f"discovery/{path.name}#{start}", "discovery repair", basis="target-derived",
                             weight=weight, note=str(reason)) for reason in reasons]
            model.add("function-start", f"{image}:{start}", {"start": True}, confidence=weight, evidence=rows)
            moved = _hex(entry.get("moved_from"))
            if moved:
                old = model.add("function-start", f"{image}:{moved}", {"start": True}, confidence=0.6, evidence=[
                    evidence(f"discovery/{path.name}#{moved}", "function catalogue before repair", basis="declared",
                             weight=0.6, note="catalogued start before the repair moved it")])
                model.add_contradiction(old.id, evidence(
                    f"discovery/{path.name}#{start}", "discovery repair", basis="target-derived", weight=weight,
                    note=f"the repair moved this start to {start}: a call target leads into it"),
                    alternatives=[({"start": False}, weight)])


def _function_starts(model: WorldModel, run: Path, cases) -> None:
    by_place = {(case.image, case.vram): case for case in cases}
    folder = run / "solver"
    if not folder.is_dir():
        return
    for path in sorted(folder.glob("coverage_*.json")):
        image = path.stem[len("coverage_"):]
        data = _json(path)
        for entry in (data.get("entries") if isinstance(data, dict) else None) or []:
            vram = _int(entry.get("vram")) if isinstance(entry, dict) else None
            if vram is None:
                continue
            start = f"0x{vram:08X}"
            subject = f"{image}:{start}"
            case = by_place.get((image, vram))
            rows = [] if model._groups.get(("function-start", subject)) else [
                evidence(f"solver/{path.name}#{start}", "function catalogue", basis="declared", weight=0.6,
                         note=f"catalogued as {entry.get('name')} ({entry.get('size')} bytes)")]
            model.add("function-start", subject, {"start": True}, confidence=0.6, evidence=rows,
                      cases=[case.case_id] if case else [],
                      consequences=[{"case": case.case_id, "predicts": "exact", "start": start}] if case else [],
                      cost_to_test={"builds": None, "note": "a strictly exact build of the function from this start "
                                                            "settles it"})


def _recipes(model: WorldModel, run: Path, report: dict, cases, catalogue: dict[str, str]) -> None:
    images = sorted({p.parent.name for p in run.glob("*/toolchain_regions.json")} |
                    {name for name, item in (report.get("images") or {}).items()
                     if isinstance(item, dict) and isinstance(item.get("calibrate"), dict)})
    for image in images:
        premise = [catalogue[image]] if image in catalogue else []
        keys = [f"{image}/{name}" for name in ("stage_calibrate_regions.json", "stage_calibrate.json")
                if (run / image / name).is_file()]
        image_cases = sorted((c for c in cases if c.image == image), key=lambda c: c.vram)
        data = _json(run / image / "toolchain_regions.json")
        for region in (data.get("regions") if isinstance(data, dict) else None) or []:
            lo, hi = _int(region.get("start")), _int(region.get("end"))
            if lo is None or hi is None:
                continue
            subject = f"{image}:region:0x{lo:08X}..0x{hi:08X}"
            key = f"{image}/toolchain_regions.json#0x{lo:08X}"
            members = [c.case_id for c in image_cases if lo <= c.vram <= hi]
            ranked = [r for r in region.get("ranked") or [] if isinstance(r, dict) and r.get("recipe")]
            shares = {r["recipe"]: r.get("share") for r in ranked[:model.beam]}
            proven = {k: int(v or 0) for k, v in (region.get("proven") or {}).items() if k}
            for recipe in sorted(set(shares) | set(proven), key=lambda r: (r not in shares, r)):
                rows, prior = [], 0.2             # a recipe only proven, not ranked by the marks
                if recipe in shares:
                    prior = _unit(shares[recipe])
                    rows.append(evidence(key, "toolchain regions (code marks)", basis="target-derived", weight=prior,
                                         note=f"code marks of the region's functions fit this recipe "
                                              f"(share {shares[recipe]})"))
                h = model.add("recipe", subject, {"recipe": recipe}, confidence=prior, evidence=rows,
                              depends_on=premise, cache_keys=keys, cases=members,
                              cost_to_test={"builds": None, "note": "regional calibration on a sample of the "
                                                                    "region (decompile-run --calibrate N)"})
                if proven.get(recipe):
                    count = proven[recipe]
                    model.add_evidence(h.id, evidence(f"{key}/proven", "strictly exact cases in the region",
                                                      basis="derived", weight=min(0.9, count / (count + 2)),
                                                      note=f"{count} function(s) of the region strictly exact under "
                                                           "this recipe"))
                if region.get("verdict") in ("unknown", "mismatch") and recipe in shares:
                    parts = sorted({str(m.get("component")) for m in region.get("component_mismatches") or []
                                    if isinstance(m, dict) and m.get("component")})
                    model.add_contradiction(h.id, evidence(
                        f"{key}/verdict", "toolchain regions (code marks)", basis="target-derived", weight=0.5,
                        note=f"region verdict {region['verdict']}" + (f": {', '.join(parts)}" if parts else "")))
        chosen = ((report.get("images") or {}).get(image) or {}).get("calibrate")
        chosen = chosen.get("chosen") if isinstance(chosen, dict) else None
        if isinstance(chosen, dict):
            label = chosen.get("label") or " ".join(str(chosen.get(k) or "") for k in ("pipeline", "flags")).strip()
            if label:
                model.add("recipe", f"{image}:image", {"recipe": label}, confidence=0.7, evidence=[
                    evidence(f"report.json#/images/{image}/calibrate", "calibration (strict exact count on a sample)",
                             basis="derived", weight=0.7, note=canonical(chosen)[:200])],
                    depends_on=premise, cache_keys=keys, cases=[c.case_id for c in image_cases])


def _image_bases(model: WorldModel, run: Path, report: dict) -> None:
    data = _json(run / "discovery" / "overlays.json")
    for image, found in sorted((data or {}).items() if isinstance(data, dict) else []):
        if not isinstance(found, dict):
            continue
        current = _int(found.get("current"))
        for row in found.get("runs") or []:
            if not isinstance(row, dict):
                continue
            lo, hi, base = _int(row.get("start")), _int(row.get("end")), _int(row.get("base"))
            if None in (lo, hi, base, current):
                continue
            calls = max(1, int(row.get("calls") or 0))
            subject = f"{image}:window:0x{lo:X}..0x{hi:X}"
            key = f"discovery/overlays.json#{subject}"
            here = int(row.get("landed_at_current") or 0)
            model.add("image-base", subject, {"base": f"0x{current:08X}"}, confidence=_unit(here / calls), evidence=[
                evidence(key, "overlay base inference (call targets landing on starts)", basis="target-derived",
                         weight=_unit(here / calls),
                         note=f"{here} of {calls} distinct call targets land on a function start at the current base")])
            if base != current:
                landed = int(row.get("landed") or 0)
                model.add("image-base", subject, {"base": f"0x{base:08X}"}, confidence=_unit(landed / calls),
                          evidence=[evidence(key, "overlay base inference (call targets landing on starts)",
                                             basis="target-derived", weight=_unit(landed / calls),
                                             note=f"{landed} of {calls} distinct call targets land on a function "
                                                  f"start at this base (a report: never applied)")])
    found = report.get("discovery")
    for place in (found.get("placed") if isinstance(found, dict) else None) or []:
        if not isinstance(place, dict) or not place.get("name") or _hex(place.get("vram")) is None:
            continue
        model.add("image-base", f"{place['name']}:image", {"vram": _hex(place.get("vram"))}, confidence=0.6, evidence=[
            evidence(f"report.json#/discovery/placed/{place['name']}", "discovery placement", basis="target-derived",
                     weight=0.6, note=f"ROM {place.get('rom')}..{place.get('rom_end')}: {place.get('landed')} calls "
                                      f"land, {place.get('functions')} functions")])


def _small_data(model: WorldModel, run: Path, report: dict, cases) -> None:
    small = report.get("small_data")
    if not isinstance(small, dict) or small.get("error"):
        return
    from .pipeline import SMALL_DATA_LANES
    lane = sorted(c.case_id for c in cases if str(c.recipe or "").partition(" ")[0] in SMALL_DATA_LANES)
    keys = ["small_data/gp_symbols.txt"] if (run / "small_data" / "gp_symbols.txt").is_file() else []
    recovered, declared = _hex(small.get("value")), _hex(small.get("project_gp"))
    cost = {"builds": None, "note": "a GP-using function built in a small-data lane and strictly exact against it"}
    if recovered:
        weight = CONFIDENCE.get(small.get("confidence"), 0.5)
        h = model.add("gp", "program:_gp", {"value": recovered}, confidence=weight, cache_keys=keys, evidence=[
            evidence("report.json#/small_data", "small-data recovery (code that sets $gp)", basis="target-derived",
                     weight=weight, note=f"{small.get('reason') or ''} ({small.get('sites', 0)} setting sites)")],
            cost_to_test=cost)
        if declared and declared != recovered:
            model.add_contradiction(h.id, evidence("report.json#/small_data/warning", "project symbol files",
                                                   basis="declared", weight=0.5,
                                                   note=small.get("warning") or f"the project declares {declared}"))
    if declared:
        model.add("gp", "program:_gp", {"value": declared}, confidence=0.8, evidence=[
            evidence("report.json#/small_data/project_gp", "project symbol files", basis="declared", weight=0.8,
                     note="the project declares _gp; its value is the one the resolver uses")], cost_to_test=cost)
    used = declared or recovered
    if used:
        model.add("gp", "program:_gp", {"value": used}, cases=lane,
                  consequences=[{"case": c, "predicts": "exact"} for c in lane])
    else:
        model.add("gp", "program:_gp", {"value": None}, confidence=0.5, evidence=[
            evidence("report.json#/small_data", "small-data recovery (code that sets $gp)", basis="target-derived",
                     weight=0.5, note=small.get("reason") or "no _gp established")])
