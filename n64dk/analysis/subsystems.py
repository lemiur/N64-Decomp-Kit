"""Subsystem hypotheses: which functions work together, and what they are probably for.

A decompilation is organised by subsystem (boot, OS/runtime, renderer, audio, input, storage...),
but a ROM carries no such structure.  :func:`recover` proposes one from five independent kinds of
evidence and says which evidence each proposal rests on:

1. **Call-graph communities.**  Deterministic label propagation over the undirected call graph,
   with an extra half-weight edge between functions that touch the same global (globals touched by
   2..8 functions only).  Hub functions (called from more distinct functions than
   ``max(12, 2% of all functions)``) do not propagate labels; they form the ``shared-utility``
   group.  Singletons are pooled as ``unclustered``.
2. **SDK names** (from the SDK matcher, via :class:`runtime.Model`): ``osAi*``/``al*`` audio,
   ``osVi*`` video, ``osCont*``/``osPfs*``/``osMotor*`` input, ``osPi*``/``osEPi*`` storage-DMA,
   ``osSp*`` RSP tasks, ``gu*``/``gsp*``/``gdp*`` renderer, other ``os*``/``__os*`` OS/runtime.
3. **Runtime calls** recognised by :mod:`runtime` (task types refine RSP tasks into renderer or
   audio).
4. **Hardware roles** of device accesses (:data:`hwmodel.ROLES`).
5. **Thread ownership** (functions reachable from a thread entry inherit the role of the thread's
   own evidence) and **strings** whose address a member takes.  Functions reached from the boot
   entry and from no thread vote ``boot``.

Labels are hypotheses for a human or a later stage, never original names: every
:class:`Subsystem` carries ``hypothesis=True``, a confidence (share of the winning role's votes
times the evidence volume, capped at 1) and its evidence sentences.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from . import hwmodel
from .runtime import Model, hx

NAME_ROLES = (  # first matching prefix wins
    (("osAi", "__osAi", "al", "__al", "n_al", "_n_al"), "audio"),
    (("osVi", "__osVi", "viMgr"), "video"),
    (("osCont", "__osCont", "osPfs", "__osPfs", "osMotor", "__osMotor", "osEeprom", "__osEeprom",
      "__osSi", "osSi"), "input"),
    (("osPi", "osEPi", "__osPi", "__osEPi", "__osDevMgr", "osCartRom", "osFlash", "osSram"), "storage-dma"),
    (("osSp", "__osSp", "osDp", "__osDp"), "rsp-tasks"),
    (("gu", "gsp", "gdp", "gSP", "gDP"), "renderer"),
    (("os", "__os", "__ll", "__ull", "__d", "bzero", "bcopy", "memcpy", "sprintf", "_Printf",
      "_Litob", "_Ldtob"), "os-runtime"),
)
API_ROLES = {"thread-create": "os-runtime", "thread-start": "os-runtime", "queue-create": "os-runtime",
             "queue-send": None, "queue-recv": None, "event": "os-runtime", "vi-event": "video",
             "vi-manager": "video", "pi-manager": "storage-dma", "dma": "storage-dma", "task": "rsp-tasks",
             "interrupt": "os-runtime", "interrupt-mask": "os-runtime", "audio-out": "audio",
             "video-out": "video", "input": "input", "timer": "os-runtime"}
STRING_ROLES = (("audio", "audio"), ("sound", "audio"), ("seq", "audio"), ("music", "audio"),
                ("gfx", "renderer"), ("display", "renderer"), ("rdp", "renderer"), ("rsp", "rsp-tasks"),
                ("cont", "input"), ("pad", "input"), ("pak", "input"), ("dma", "storage-dma"),
                ("rom", "storage-dma"), ("thread", "os-runtime"), ("fault", "os-runtime"))
MIN_VOTES = 2.0


def name_role(name: str | None) -> str | None:
    if not name:
        return None
    for prefixes, role in NAME_ROLES:
        if name.startswith(prefixes):
            return role
    return None


@dataclass
class Subsystem:
    id: int
    label: str
    confidence: float
    functions: list[int]
    bytes: int
    entrypoints: list[str] = field(default_factory=list)
    threads: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    votes: dict[str, float] = field(default_factory=dict)
    hypothesis: bool = True

    def as_dict(self, members: int = 20) -> dict:
        return {"id": self.id, "label": self.label, "hypothesis": self.hypothesis, "confidence": self.confidence,
                "functions": len(self.functions), "bytes": self.bytes,
                "members": [hx(f) for f in self.functions[:members]], "entrypoints": self.entrypoints,
                "threads": self.threads, "votes": {k: round(v, 1) for k, v in sorted(self.votes.items())},
                "evidence": self.evidence[:12]}


def _graph(model: Model, hubs: set[int]) -> dict[int, dict[int, float]]:
    nodes = set(model.functions)
    adj: dict[int, dict[int, float]] = {n: {} for n in nodes}

    def link(a, b, w):
        if a == b or a in hubs or b in hubs:
            return
        adj[a][b] = adj[a].get(b, 0.0) + w
        adj[b][a] = adj[b].get(a, 0.0) + w

    for info in model.functions.values():
        for callee in info.callees:
            if callee in nodes:
                link(info.start, callee, 1.0)
    users: dict[int, list[int]] = {}
    for info in model.functions.values():
        for address in info.globals:
            users.setdefault(address, []).append(info.start)
    for address, functions in users.items():
        if 2 <= len(functions) <= 8:
            for i, a in enumerate(functions):
                for b in functions[i + 1:]:
                    link(a, b, 0.5)
    return adj


def communities(adj: dict[int, dict[int, float]], rounds: int = 20) -> dict[int, int]:
    """Deterministic asynchronous label propagation: visit nodes in address order; a node adopts
    the neighbour label with the largest total weight (ties: keep its own if tied, else the
    smallest label)."""
    label = {n: n for n in adj}
    for _round in range(rounds):
        changed = False
        for node in sorted(adj):
            scores: dict[int, float] = {}
            for neighbour, weight in adj[node].items():
                scores[label[neighbour]] = scores.get(label[neighbour], 0.0) + weight
            if not scores:
                continue
            best_weight = max(scores.values())
            if scores.get(label[node], 0.0) == best_weight:
                continue
            best = min(lab for lab, w in scores.items() if w == best_weight)
            if best != label[node]:
                label[node] = best
                changed = True
        if not changed:
            break
    return label


def _thread_roles(model: Model) -> dict[int, tuple[str, list[str]]]:
    """Role of each thread entry from the API calls and device accesses of what it reaches."""
    reach: dict[int, list[int]] = {}
    for function, owners in model.owners.items():
        for entry in owners:
            reach.setdefault(entry, []).append(function)
    out = {}
    for entry, functions in reach.items():
        votes: Counter = Counter()
        for function in functions:
            info = model.functions[function]
            for call in info.calls:
                if call.target in model.api:
                    name, kind, _how = model.api[call.target]
                    role = _task_role(model, call) if kind == "task" else API_ROLES.get(kind)
                    if role and role != "os-runtime":
                        votes[role] += 1
            for role, n in hwmodel.roles(info.accesses).items():
                votes[role] += n
        if votes:
            role, n = votes.most_common(1)[0]
            out[entry] = (role, [f"thread {model.names.get(entry) or hx(entry)} reaches {n} {role} call(s)/access(es)"])
    return out


def _task_role(model: Model, call) -> str:
    for task in model.tasks.values():
        if any(row["site"] == hx(call.site) for row in task["started"]):
            if "M_GFXTASK" in task.get("type", []):
                return "renderer"
            if "M_AUDTASK" in task.get("type", []):
                return "audio"
    return "rsp-tasks"


def recover(model: Model, *, boot_entry: int | None = None, hub_share: float = 0.02) -> list[Subsystem]:
    functions = model.functions
    if not functions:
        return []
    callers: dict[int, set[int]] = {}
    for info in functions.values():
        for callee in info.callees:
            callers.setdefault(callee, set()).add(info.start)
    hub_limit = max(12, int(hub_share * len(functions)))
    hubs = {f for f in functions if len(callers.get(f, ())) > hub_limit}
    labels = communities(_graph(model, hubs))
    groups: dict[object, list[int]] = {}
    for function, lab in labels.items():
        groups.setdefault(lab, []).append(function)
    pooled: dict[object, list[int]] = {"shared-utility": sorted(hubs)} if hubs else {}
    singles = []
    for lab, members in groups.items():
        members = [m for m in members if m not in hubs]
        if len(members) > 1:
            pooled[lab] = sorted(members)
        else:
            singles += members
    if singles:
        pooled["unclustered"] = sorted(singles)

    thread_roles = _thread_roles(model)
    boot = _boot_set(model, boot_entry)
    out = []
    for number, (key, members) in enumerate(sorted(pooled.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))):
        votes: Counter = Counter()
        evidence: list[str] = []
        for function in members:
            info = functions[function]
            role = name_role(model.names.get(function))
            if role:
                votes[role] += 2
                evidence.append(f"{model.names[function]} at {hx(function)} (SDK name) -> {role}")
            for call in info.calls:
                if call.target in model.api:
                    name, kind, _how = model.api[call.target]
                    role = _task_role(model, call) if kind == "task" else API_ROLES.get(kind)
                    if role:
                        votes[role] += 3
                        evidence.append(f"{hx(function)} calls {name} at {hx(call.site)} -> {role}")
            for role, n in hwmodel.roles(info.accesses).items():
                votes[role] += 2 * n
                evidence.append(f"{hx(function)} accesses {role} hardware ({n} access(es))")
            for entry in model.owners.get(function, ()):
                if entry in thread_roles:
                    votes[thread_roles[entry][0]] += 1
            for text in info.strings:
                lowered = text.lower()
                for word, role in STRING_ROLES:
                    if word in lowered:
                        votes[role] += 1
                        evidence.append(f"{hx(function)} takes the address of the string {text[:30]!r} -> {role}")
                        break
            if function in boot:
                votes["boot"] += 3 if function == boot_entry else 1
        if key in ("shared-utility", "unclustered"):
            label = key
        elif votes and votes.most_common(1)[0][1] >= MIN_VOTES:
            label = votes.most_common(1)[0][0]
        else:
            label = "unknown"
        total = sum(votes.values())
        top = votes.most_common(1)[0][1] if votes else 0
        confidence = round((top / total) * min(1.0, total / 10.0), 2) if total and label not in (
            "unknown", "shared-utility", "unclustered") else 0.0
        threads = sorted({hx(e) for f in members for e in model.owners.get(f, ())})
        member_set = set(members)
        entries = [hx(t["entry"]) + " (thread)" for t in model.threads if t["entry"] in member_set]
        external = Counter(f for f in members for c in callers.get(f, ()) if c not in member_set)
        entries += [hx(f) + f" ({n} outside caller(s))" for f, n in external.most_common(5)]
        size = sum(functions[f].end - functions[f].start for f in members)
        out.append(Subsystem(number, label, confidence, members, size, entries, threads[:12], evidence,
                             dict(votes)))
    return out


def _boot_set(model: Model, entry: int | None) -> set[int]:
    if entry is None or entry not in model.functions:
        return set()
    seen, stack = {entry}, [entry]
    while stack:
        info = model.functions.get(stack.pop())
        if info is None:
            continue
        for callee in info.callees:
            if callee in model.functions and callee not in seen and not model.owners.get(callee):
                seen.add(callee)
                stack.append(callee)
    return seen


def summary(subsystems: list[Subsystem]) -> dict:
    sizes = Counter()
    for s in subsystems:
        sizes[s.label] += len(s.functions)
    return {"subsystems": len(subsystems), "functions_by_label": dict(sizes.most_common()),
            "hypotheses": [s.as_dict(8) for s in subsystems[:40]]}
