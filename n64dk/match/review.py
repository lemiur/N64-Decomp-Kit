"""Human review and acceptance workflow over a finished run (round 12, item 12).

Three operations on a ``decompile-run`` directory:

* :func:`queue`: what needs a person: cases waiting for declared symbols (``assumed``), cases
  close to a match (``near``) and donor units waiting for symbols, minus items already decided.
* :func:`show`: one item in full: the candidate source, the target-versus-candidate assembly diff
  (``n64dk.match.diff``; needs a build, so it comes from an injected ``differ``), the residual,
  the proof receipt when one exists (``n64dk.match.proofreceipt``), the strict verifier's
  assumptions, provenance and transformation classes, the recipe, recorded behavioural
  counterexamples, the review history and an optional maintainable view
  (``n64dk.match.twotrack``).
* :func:`decide`: a durable review transaction appended to ``RUN/review/transactions.jsonl``.

Transactions are append-only, one JSON object per line, hash-chained (``prev`` is the previous
record's ``id``, ``id`` the SHA-256 of the record without it) and replayable: :func:`replay`
rebuilds every case's review state from the file alone and reports a broken chain or a truncated
line instead of trusting it.  They are machine-readable on purpose (schema ``n64dk-review/1``):
another agent can read the queue, act and append.

What a decision can and cannot do:

* nothing here edits project source, the case store or any run output outside ``RUN/review``;
* ``accept-exact`` is recorded as ``requested`` unless the injected ``verify(case, source)``
  reports the strict verifier's ``exact`` for the stored candidate, then ``verified``.  It is a
  record of the check, not a new credit path: credit stays with the run's own strict queue;
* ``retain-assumed`` is only valid for a case the strict verifier called ``assumed``;
* ``export`` copies the candidate and its transaction into ``RUN/review/exports/``."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "n64dk-review/1"
ACTIONS: dict[str, tuple[str, ...]] = {
    "accept-exact": (), "retain-assumed": (), "reject": ("reason",), "request-stage": ("stage",),
    "correct-type": ("symbol", "type"), "correct-boundary": ("vram", "size"), "add-evidence": ("kind", "detail"),
    "mark-handwritten": (), "defer": (), "export": ()}
# a decided item leaves the queue (an accept that is only requested does not decide anything)
DECIDING = ("accept-exact", "retain-assumed", "reject", "mark-handwritten")
QUEUE_STATUSES = ("assumed", "near")
BEHAVIOUR_STAGES = ("oracle", "behaviour", "behavioural", "tv", "semantic-check")


class ReviewError(ValueError):
    pass


@dataclass
class Item:
    id: str
    kind: str                 # assumed, near, donor
    image: str
    name: str
    vram: int
    size: int
    score: int | None = None
    cause: str = ""
    detail: dict = field(default_factory=dict)
    decision: str | None = None

    def as_dict(self) -> dict:
        out = asdict(self)
        out["vram"] = f"0x{self.vram:08X}"
        return out


# ------------------------------------------------------------------------------ run access
def review_dir(run) -> Path:
    return Path(run) / "review"


def log_path(run) -> Path:
    return review_dir(run) / "transactions.jsonl"


def _report(run) -> dict:
    path = Path(run) / "report.json"
    try:
        data = json.loads(path.read_text()) if path.is_file() else {}
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_cases(run) -> list:
    """Every case of the run, read-only (an empty list when the run has no case store)."""
    from .resolve import CaseStore
    state = Path(run) / "state"
    if not (state / "cases.sqlite").is_file():
        return []
    store = CaseStore(state, readonly=True)
    try:
        return store.all()
    finally:
        store.close()


def case_source(run, case) -> str | None:
    path = Path(run) / "state" / "sources" / f"{case.best_digest}.c" if case.best_digest else None
    return path.read_text() if path is not None and path.is_file() else None


def donor_items(run) -> list[Item]:
    """Donor units whose code is identical but which wait for symbols (``report.json``)."""
    out = []
    for image, data in sorted((_report(run).get("images") or {}).items()):
        donors = (data or {}).get("donors") if isinstance(data, dict) else None
        for row in (donors or {}).get("verified") or []:
            if not isinstance(row, dict) or not row.get("awaiting"):
                continue
            try:
                vram = int(str(row.get("vram")), 0)
            except ValueError:
                continue
            names = row.get("functions") or [f"func_{vram:08X}"]
            out.append(Item(f"donor:{image}:{vram:08X}", "donor", image, names[0], vram, int(row.get("size") or 0),
                            cause="awaiting symbols",
                            detail={"donor": row.get("donor"), "unit": row.get("unit"),
                                    "functions": names, "awaiting": dict(row["awaiting"])}))
    return out


def _case_item(case) -> Item:
    kind = "assumed" if case.status == "assumed" else "near"
    return Item(case.case_id, kind, case.image, case.name, case.vram, case.size, case.best_score,
                case.top_cause, {"next": case.next, "recipe": case.recipe, "builds": case.builds})


def queue(run, include_decided: bool = False) -> list[Item]:
    """The review queue: assumed cases (largest first), near cases (closest first), donor units."""
    state = replay(run)["cases"]
    every = load_cases(run)
    cases = [c for c in every if c.status in QUEUE_STATUSES]
    assumed = sorted((c for c in cases if c.status == "assumed"), key=lambda c: (-c.size, c.case_id))
    near = sorted((c for c in cases if c.status == "near"),
                  key=lambda c: (c.best_score if c.best_score is not None else 1 << 30, c.case_id))
    with_case = {(c.image, c.vram) for c in every}          # a case supersedes the donor row
    items = [_case_item(c) for c in assumed + near] + [i for i in donor_items(run) if (i.image, i.vram) not in with_case]
    for item in items:
        item.decision = (state.get(item.id) or {}).get("decision")
    return items if include_decided else [i for i in items if not i.decision]


def find(run, ref: str):
    """(item, case or None) for a case id, a donor item id or a function name."""
    cases = load_cases(run)
    for case in cases:
        if case.case_id == ref:
            return (_case_item(case) if case.status in QUEUE_STATUSES else
                    Item(case.case_id, case.status, case.image, case.name, case.vram, case.size, case.best_score,
                         case.top_cause)), case
    for item in donor_items(run):
        if item.id == ref:
            return item, None
    named = [c for c in cases if c.name == ref]
    if len(named) == 1:
        return find(run, named[0].case_id)
    if len(named) > 1:
        raise ReviewError(f"{ref} names {len(named)} cases: " + ", ".join(c.case_id for c in named))
    donors = [i for i in donor_items(run) if ref in i.detail.get("functions", [])]
    if len(donors) == 1:
        return donors[0], None
    raise ReviewError(f"no case or review item {ref!r} in this run")


# ------------------------------------------------------------------------------ show
def _assumptions(run, case) -> list:
    out = list(((case.proof or {}).get("assumptions") or []) if isinstance(case.proof, dict) else [])
    path = Path(run) / "queue" / "assumptions.tsv"
    if path.is_file():
        for line in path.read_text().splitlines():
            cells = line.split("\t")
            if line.startswith("#") or len(cells) < 5:
                continue
            if cells[0] == case.image and cells[1] == case.name:
                out.append({"kind": cells[2], "name": cells[3], "address": cells[4], "from": "queue/assumptions.tsv"})
    return out


def _counterexamples(value, path="", found=None) -> list:
    """Every recorded counterexample or behavioural verdict in a nested stage record."""
    found = [] if found is None else found
    if isinstance(value, dict):
        for key, item in value.items():
            here = f"{path}.{key}" if path else str(key)
            if "counterexample" in str(key) and item:
                found.append({"where": here, "value": item})
            else:
                _counterexamples(item, here, found)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _counterexamples(item, f"{path}[{index}]", found)
    return found


def receipt_summary(run, case) -> dict | None:
    from . import proofreceipt
    path = proofreceipt.receipt_path(Path(run) / "receipts", case.image, case.name, case.vram)
    if not path.is_file():
        return None
    try:
        receipt = proofreceipt.load(path)
    except (OSError, ValueError) as error:
        return {"file": path.name, "error": f"{type(error).__name__}: {error}"[:200]}
    try:
        problems = proofreceipt.check_static(receipt)
    except Exception as error:                     # a malformed receipt is shown as such
        problems = [f"static check failed: {type(error).__name__}: {error}"[:200]]
    return {"file": str(path.relative_to(run)), "identity": receipt.get("identity"),
            "class": (receipt.get("verdict") or {}).get("class"),
            "source_sha256": (receipt.get("source") or {}).get("sha256"),
            "claims": receipt.get("claims"), "static_problems": problems,
            "behavioural": receipt.get("behavioural")}


def show(run, ref: str, *, differ=None, maintainable=None) -> dict:
    """Everything a reviewer needs for one item.  ``differ(case, source)`` returns diff lines
    (a build); ``maintainable(source, case)`` returns a ``twotrack.TwoTrack``.  Both optional."""
    from .transform_classes import FAMILY_CLASS, provenance
    item, case = find(run, ref)
    history = replay(run)["cases"].get(item.id) or {}
    out = {"item": item.as_dict(), "history": history.get("history", []), "decision": history.get("decision")}
    if case is None:
        out["note"] = "a donor unit: identical code from another program, waiting for the listed symbols"
        return out
    source = case_source(run, case)
    prov = provenance(case.stages)
    out.update({
        "status": case.status, "recipe": case.recipe, "builds": case.builds,
        "residual": {"best_score": case.best_score, "top_cause": case.top_cause, "causes": (case.causes or [])[:6],
                     "next": case.next,
                     "diagnose": next((s for s in reversed(case.stages or []) if s.get("stage") == "diagnose"), None)},
        "strict": case.strict or None, "proof_verdict": (case.proof or {}).get("verdict") if isinstance(case.proof, dict)
        else None,
        "assumptions": _assumptions(run, case),
        "provenance": prov,
        "transform_classes": {family: FAMILY_CLASS.get(family, "exploration") for family in prov["families"]},
        "behaviour": [s for s in case.stages or [] if s.get("stage") in BEHAVIOUR_STAGES]
                     + _counterexamples({"stages": case.stages or []}),
        "receipt": receipt_summary(run, case),
        "source": source,
        "stages": [s.get("stage") for s in case.stages or []]})
    if differ is not None and source is not None:
        try:
            out["diff"] = differ(case, source)
        except Exception as error:                     # a failed build is shown, not raised
            out["diff"] = [f"diff unavailable: {type(error).__name__}: {error}"[:300]]
    if maintainable is not None and source is not None:
        try:
            track = maintainable(source, case)
            out["maintainable"] = {"class": track.view_class, "map": track.as_map(), "view": track.view}
        except Exception as error:
            out["maintainable"] = {"error": f"{type(error).__name__}: {error}"[:300]}
    return out


# ------------------------------------------------------------------------------ transactions
def _digest(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "id"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_log(run) -> tuple[list[dict], list[str]]:
    """(valid records in order, problems).  Reading stops trusting the log at the first broken
    link: later records are reported, not applied."""
    path = log_path(run)
    if not path.is_file():
        return [], []
    records, problems, prev = [], [], None
    lines = path.read_text().split("\n")
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            problems.append(f"line {number}: not JSON" + (" (truncated last line)" if number >= len(lines) - 1 else ""))
            break
        if not isinstance(record, dict) or record.get("schema") != SCHEMA:
            problems.append(f"line {number}: not an {SCHEMA} record")
            break
        if record.get("prev") != prev or record.get("id") != _digest(record) or record.get("seq") != len(records) + 1:
            problems.append(f"line {number}: hash chain broken (seq {record.get('seq')})")
            break
        records.append(record)
        prev = record["id"]
    return records, problems


def replay(run) -> dict:
    """Per item: the deciding action (or the latest one), its status and the full history."""
    records, problems = read_log(run)
    cases: dict[str, dict] = {}
    for r in records:
        entry = cases.setdefault(r["case"], {"decision": None, "status": None, "latest": None, "history": [],
                                             "requested_stages": [], "corrections": [], "evidence": []})
        entry["history"].append({"seq": r["seq"], "action": r["action"], "status": r["status"], "at": r.get("at"),
                                 "reviewer": r.get("reviewer"), "id": r["id"]})
        entry["latest"] = r["action"]
        if r["action"] in DECIDING and r["status"] in ("recorded", "verified"):
            entry["decision"], entry["status"] = r["action"], r["status"]
        elif r["action"] == "request-stage":
            entry["requested_stages"].append(r["payload"].get("stage"))
        elif r["action"] in ("correct-type", "correct-boundary"):
            entry["corrections"].append({"action": r["action"], **r["payload"]})
        elif r["action"] == "add-evidence":
            entry["evidence"].append(dict(r["payload"]))
    return {"transactions": len(records), "problems": problems, "cases": cases,
            "head": records[-1]["id"] if records else None}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _verdict(result) -> str | None:
    if isinstance(result, dict):
        return result.get("verdict")
    return result if isinstance(result, str) else ("exact" if result is True else None)


def decide(run, ref: str, action: str, *, payload: dict | None = None, reviewer: str | None = None,
           note: str | None = None, verify=None, clock=None) -> dict:
    """Append one review transaction and return it.  ``verify(case, source)`` (accept-exact only)
    runs the strict verifier on the stored candidate and returns ``{"verdict": ...}``."""
    if action not in ACTIONS:
        raise ReviewError(f"unknown action {action!r}; choose from {', '.join(ACTIONS)}")
    payload = dict(payload or {})
    missing = [k for k in ACTIONS[action] if k not in payload or payload[k] in (None, "")]
    if missing:
        raise ReviewError(f"{action} needs {', '.join(missing)}")
    if action == "correct-boundary":
        try:
            payload["vram"], payload["size"] = f"0x{int(str(payload['vram']), 0):08X}", int(str(payload["size"]), 0)
        except ValueError:
            raise ReviewError("correct-boundary needs a numeric vram and size") from None
    item, case = find(run, ref)
    source = case_source(run, case) if case is not None else None
    status, verdict, reason = "recorded", None, None
    if action == "accept-exact":
        if case is None or source is None:
            status, reason = "invalid", "no stored candidate source to verify"
        elif verify is None:
            status, reason = "requested", "no strict verification was run: recorded as a request"
        else:
            try:
                verdict = verify(case, source)
            except Exception as error:
                verdict = {"verdict": "error", "error": f"{type(error).__name__}: {error}"[:300]}
            status = "verified" if _verdict(verdict) == "exact" else "requested"
            if status == "requested":
                reason = f"strict verification did not say exact ({_verdict(verdict)})"
    elif action == "retain-assumed" and (case is None or case.status != "assumed"):
        status, reason = "invalid", "only a case the strict verifier called assumed can be retained as assumed"
    subject = {"image": item.image, "name": item.name, "vram": f"0x{item.vram:08X}", "size": item.size,
               "item_kind": item.kind}
    if case is not None:
        subject.update({"case_status": case.status, "target_sha256": case.target_digest,
                        "source_sha256": case.best_digest, "recipe": case.recipe})
    path = log_path(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            records, problems = read_log(run)
            if problems:
                raise ReviewError("the review log is damaged; repair it before appending: " + problems[0])
            record = {"schema": SCHEMA, "seq": len(records) + 1, "prev": records[-1]["id"] if records else None,
                      "at": (clock or _now)(), "case": item.id, "action": action, "status": status,
                      "reason": reason, "payload": payload, "reviewer": reviewer, "note": note,
                      "subject": subject, "verdict": verdict}
            folder = review_dir(run) / "exports" / re.sub(r"[^A-Za-z0-9_.-]", "_", item.id)
            if action == "export" and source is not None:
                record["export"] = {"folder": f"review/exports/{folder.name}",
                                    "files": [f"{item.name}.c", "transaction.json"]}
            elif action == "export":
                record["status"], record["reason"] = "invalid", "no stored candidate source to export"
            record["id"] = _digest(record)
            handle.seek(0, os.SEEK_END)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    if record.get("export"):                      # files only under RUN/review, never project source
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{item.name}.c").write_text(source)
        (folder / "transaction.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return record
