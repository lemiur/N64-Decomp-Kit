"""The program's runtime as libultra sets it up: threads, message queues, events, DMA, RSP tasks,
interrupts and the VI/PI/AI managers, recovered from call sites with provenance.

An N64 game is a set of threads that talk through message queues.  Which function is a thread
entry, which queue an interrupt posts to and which code starts DMA or RSP tasks is what the rest
of the kit needs to explain the program's structure (``subsystems.py``), to choose oracle stubs and
to read a function's role.  None of it is visible in the catalogue.

Recognition
    * **By SDK name** (``survey/sdk_symbols.txt`` from the SDK matcher): calls to ``osCreateThread``,
      ``osCreateMesgQueue``, ``osSendMesg``/``osJamMesg``/``osRecvMesg``, ``osSetEventMesg``,
      ``osViSetEvent``, ``osCreateViManager``/``osCreatePiManager``, ``osPiStartDma``/``osEPiStartDma``
      and the raw forms, ``osSpTaskLoad``/``osSpTaskStartGo``/``osSpTaskStart``,
      ``__osSetHWIntrRoutine``, ``osSetIntMask``, ``osAiSetNextBuffer``/``osAiSetFrequency``,
      ``osViSwapBuffer``/``osViSetMode``, ``osContInit``/``osContStartReadData`` and ``osSetTimer``
      (``API``).
    * **By call pattern**, only when the SDK matcher named no ``osCreateThread`` (or no
      ``osCreateMesgQueue``): an unnamed callee whose call sites pass a catalogued function start
      in ``$a2``, a small constant id in ``$a1`` and an 8-aligned non-code constant at ``0x10($sp)``
      is a thread-create candidate; an unnamed small leaf called with two data addresses and a
      count of 1..1024 is a queue-create candidate.  The call shape alone is not enough: an engine's
      own task or process creator has the same shape (one lab title's fallback reported 231 threads
      from 245 call sites).  A candidate is accepted only when its own body corroborates it: a thread
      create stores its incoming ``$a2`` (the entry) at ``0x11C`` of its incoming ``$a0``
      (``OSThread.context.pc``, fixed by the exception handler's save area); a queue create stores
      its ``$a2`` (count) at ``0x10`` and its ``$a1`` (buffer) at ``0x14`` of its ``$a0``
      (``OSMesgQueue.msgCount/msg``).  Stores through stack spills count (``-O0`` code).  Every
      candidate, accepted or rejected, is listed in ``pattern_candidates`` with the reason; accepted
      ones and everything derived from them carry ``confidence_label`` ``low`` and
      ``provenance_kind`` ``pattern``.

Threads are distinct ``(entry, id, priority, stack, thread struct)`` tuples, not call sites: a
thread created identically from several sites is one row whose ``sites`` lists them.

Arguments come from the constant propagation of :mod:`n64dk.analysis.indirect` (``la a2, entry``,
constants spilled to the outgoing stack area), and through wrappers: an argument that is the
caller's own incoming ``$a0..$a3`` is followed to every call site of that caller, three levels
deep (Super Mario 64 creates its threads through ``create_thread``).

Stock libultra is never assumed.  A named API whose body does not read the argument registers
the stock function reads, or whose name does not sit on a catalogued function start, is listed
in ``variants``; its call sites are still interpreted with the stock signature and say so.

Everything is a report with provenance: nothing here changes a catalogue or grants credit.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .. import mips
from . import flow as flowmod
from .datamap import string_at
from .hwmodel import Access, _AccessInterp, accesses_of
from .indirect import MASK, MAX_TABLE, image_reader

ARGS = (4, 5, 6, 7)
STACK_ARGS = (0x10, 0x14, 0x18, 0x1C)
WRAPPER_DEPTH = 3

API: dict[str, tuple[str, dict[str, int]]] = {
    "osCreateThread": ("thread-create", {"thread": 4, "id": 5, "entry": 6, "arg": 7, "stack": 0x10, "pri": 0x14}),
    "osStartThread": ("thread-start", {"thread": 4}),
    "osCreateMesgQueue": ("queue-create", {"queue": 4, "buffer": 5, "count": 6}),
    "osSendMesg": ("queue-send", {"queue": 4, "msg": 5, "flag": 6}),
    "osJamMesg": ("queue-send", {"queue": 4, "msg": 5, "flag": 6}),
    "osRecvMesg": ("queue-recv", {"queue": 4, "msg": 5, "flag": 6}),
    "osSetEventMesg": ("event", {"event": 4, "queue": 5, "msg": 6}),
    "osViSetEvent": ("vi-event", {"queue": 4, "msg": 5, "retrace": 6}),
    "osCreateViManager": ("vi-manager", {"pri": 4}),
    "osCreatePiManager": ("pi-manager", {"pri": 4, "queue": 5, "buffer": 6, "count": 7}),
    "osPiStartDma": ("dma", {"mb": 4, "pri": 5, "direction": 6, "device": 7, "ram": 0x10, "size": 0x14,
                             "queue": 0x18}),
    "osEPiStartDma": ("dma", {"handle": 4, "mb": 5, "direction": 6}),
    "osPiRawStartDma": ("dma", {"direction": 4, "device": 5, "ram": 6, "size": 7}),
    "osEPiRawStartDma": ("dma", {"handle": 4, "direction": 5, "device": 6, "ram": 7, "size": 0x10}),
    "osSpTaskLoad": ("task", {"task": 4}),
    "osSpTaskStartGo": ("task", {"task": 4}),
    "osSpTaskStart": ("task", {"task": 4}),
    "__osSetHWIntrRoutine": ("interrupt", {"interrupt": 4, "handler": 5}),
    "osSetIntMask": ("interrupt-mask", {"mask": 4}),
    "osAiSetNextBuffer": ("audio-out", {"buffer": 4, "size": 5}),
    "osAiSetFrequency": ("audio-out", {"frequency": 4}),
    "osViSwapBuffer": ("video-out", {"framebuffer": 4}),
    "osViSetMode": ("video-out", {"mode": 4}),
    "osContInit": ("input", {"queue": 4, "bitpattern": 5, "status": 6}),
    "osContStartReadData": ("input", {"queue": 4}),
    "osSetTimer": ("timer", {"timer": 4, "queue": 0x18, "msg": 0x1C}),
}
# argument registers the stock function reads (a body that does not read one is a variant)
STOCK_READS = {
    "osCreateThread": {4, 5, 6, 7}, "osStartThread": {4}, "osCreateMesgQueue": {4, 5, 6},
    "osSendMesg": {4, 5, 6}, "osJamMesg": {4, 5, 6}, "osRecvMesg": {4, 5, 6}, "osSetEventMesg": {4, 5, 6},
    "osViSetEvent": {4, 5, 6}, "osPiStartDma": {4, 5, 6, 7}, "osEPiStartDma": {4, 5, 6},
    "osSpTaskLoad": {4}, "osSpTaskStartGo": {4}, "osSetIntMask": {4}, "osAiSetNextBuffer": {4, 5},
    "osViSwapBuffer": {4}, "osCreatePiManager": {4, 5, 6, 7}, "osCreateViManager": {4},
    "osContStartReadData": {4}, "osSetTimer": {4, 6, 7},
}
OS_EVENTS = ("OS_EVENT_SW1", "OS_EVENT_SW2", "OS_EVENT_CART", "OS_EVENT_COUNTER", "OS_EVENT_SP", "OS_EVENT_SI",
             "OS_EVENT_AI", "OS_EVENT_VI", "OS_EVENT_PI", "OS_EVENT_DP", "OS_EVENT_CPU_BREAK",
             "OS_EVENT_SP_BREAK", "OS_EVENT_FAULT", "OS_EVENT_THREADSTATUS", "OS_EVENT_PRENMI")
TASK_TYPES = {1: "M_GFXTASK", 2: "M_AUDTASK", 3: "M_VIDTASK", 4: "M_NJPEGTASK"}
TASK_FIELDS = {0x00: "type", 0x04: "flags", 0x08: "ucode_boot", 0x0C: "ucode_boot_size", 0x10: "ucode",
               0x14: "ucode_size", 0x18: "ucode_data", 0x1C: "ucode_data_size", 0x20: "dram_stack",
               0x24: "dram_stack_size", 0x28: "output_buff", 0x2C: "output_buff_size", 0x30: "data_ptr",
               0x34: "data_size", 0x38: "yield_data_ptr", 0x3C: "yield_data_size"}
_SYM = re.compile(r"^\s*([A-Za-z_.$][\w.$]*)\s*=\s*(0x[0-9A-Fa-f]+)\s*;")
# What a pattern candidate's own body must do: stores (base parameter, offset, stored parameter).
THREAD_PC, THREAD_ID = 0x11C, 0x14               # OSThread.context.pc, OSThread.id
MQ_MSGCOUNT, MQ_MSG = 0x10, 0x14                 # OSMesgQueue.msgCount, OSMesgQueue.msg
PATTERN_RULES = {
    "thread-create": ("osCreateThread?", ((4, THREAD_PC, 6),),
                      "store its $a2 (entry) at 0x11C of its $a0 (OSThread.context.pc)"),
    "queue-create": ("osCreateMesgQueue?", ((4, MQ_MSGCOUNT, 6), (4, MQ_MSG, 5)),
                     "store its $a2 (count) at 0x10 and its $a1 (buffer) at 0x14 of its $a0 "
                     "(OSMesgQueue.msgCount/msg)"),
}
SUPPORTING = {(4, THREAD_ID, 5): "stores its $a1 (id) at 0x14 of its $a0 (OSThread.id)"}


def hx(value) -> str | None:
    return None if value is None else f"0x{value & MASK:08X}"


def _param(value):
    """(register, offset) when ``value`` is the function's own incoming ``$a0..$a3`` plus a constant."""
    if value is not None and value[0] == "lin" and value[2] == 1 and value[1][0] == "in" \
            and value[1][1] == 0 and value[1][2] in ARGS:
        return value[1][2], value[3]
    return None


def level(how: str, resolved: bool = True) -> str:
    """The confidence label of a result: ``low`` for anything derived from a pattern recognition."""
    return "low" if how != "sdk-name" else "high" if resolved else "medium"


# --------------------------------------------------------------------------- per function

@dataclass
class Call:
    site: int
    function: int
    target: int | None
    args: dict                       # location (register number or stack offset) -> abstract value or None


@dataclass
class FunctionInfo:
    start: int
    end: int
    image: str
    calls: list[Call] = field(default_factory=list)
    accesses: list[Access] = field(default_factory=list)
    stores: list[tuple[int, int | None, int]] = field(default_factory=list)
    globals: set[int] = field(default_factory=set)        # %hi/%lo data addresses loaded or stored
    strings: list[str] = field(default_factory=list)      # strings whose address the function takes
    live_in: frozenset | None = None                      # computed for SDK-named functions only
    # (base $aN, offset, stored $aN): the body stores one of its parameters at another parameter + offset
    param_stores: set[tuple[int, int, int]] = field(default_factory=set)

    @property
    def callees(self) -> set[int]:
        return {c.target for c in self.calls if c.target is not None}


class _RuntimeInterp(_AccessInterp):
    """Records every call's argument registers and outgoing stack arguments after its delay slot;
    the entry block starts with ``$a0..$a3`` as the function's parameters."""

    def __init__(self, fl, read_word, max_table: int = MAX_TABLE):
        super().__init__(fl, read_word, max_table)
        self.calls: dict[int, tuple] = {}
        self.param_stores: dict[int, tuple] = {}     # sw site -> (base $aN, offset, stored $aN); last pass wins

    def block(self, block, state):
        if block.number == 0:
            regs = dict(state[0])
            for reg in ARGS:
                regs.setdefault(reg, ("lin", ("in", 0, reg), 1, 0))
            state = (regs, state[1])
        return super().block(block, state)

    def step(self, insn, regs, slots, first):
        w = insn.word
        if insn.op != ".word" and w >> 26 == 43:             # sw, before it executes
            base = _param(regs.get((w >> 21) & 31))
            stored = _param(regs.get((w >> 16) & 31))
            if base is not None and stored is not None and stored[1] == 0:
                simm = (w & 0xFFFF) - 0x10000 if w & 0x8000 else w & 0xFFFF
                self.param_stores[insn.vram] = (base[0], (base[1] + simm) & MASK, stored[0])
            else:
                self.param_stores.pop(insn.vram, None)
        super().step(insn, regs, slots, first)
        index = insn.index
        if index > first:
            prev = self.fl.insns[index - 1]
            if prev.kind == "call":
                args = {reg: regs.get(reg) for reg in ARGS}
                sp = regs.get(flowmod.SP)
                for offset in STACK_ARGS:
                    args[offset] = slots.get((("sp",), (sp[3] + offset) & MASK)) \
                        if sp is not None and sp[0] == "lin" and sp[1] == ("sp",) else None
                self.calls[prev.vram] = (prev.target, args)


def _function(data: bytes, delta: int, vram: int, size: int, image: str, max_table: int) -> FunctionInfo:
    read_word = image_reader(data, delta)
    code = data[vram - delta:vram - delta + size]
    code = code[:len(code) & ~3]
    info = FunctionInfo(vram, vram + len(code), image)
    if len(code) < 4:
        return info
    tables = flowmod.find_jump_tables(code, vram, read_word)
    fl = flowmod.analyse(code, vram, tables)
    interp = _RuntimeInterp(fl, read_word, max_table)
    interp.run()
    for site, (target, args) in sorted(interp.calls.items()):
        if target is None:                                   # jalr: a constant target counts
            value = (interp.sites.get(site) or (None,))[0]
            target = value[1] if value is not None and value[0] == "c" else None
        info.calls.append(Call(site, vram, target, args))
    info.accesses = accesses_of(interp, vram)
    info.stores = list(interp.stores)
    info.param_stores = set(interp.param_stores.values())
    for ref in mips.scan_refs(code, vram):
        if ref.kind in ("load", "store"):
            info.globals.add(ref.address)
        elif ref.kind == "addr":
            offset = ref.address - delta
            if 0 <= offset < len(data):
                found = string_at(data, offset, min(len(data), offset + 256))
                if found:
                    info.strings.append(data[offset:found[0] - 1].decode("latin-1")[:60])
    return info


# --------------------------------------------------------------------------- the model

@dataclass
class Model:
    functions: dict[int, FunctionInfo]
    names: dict[int, str]
    api: dict[int, tuple[str, str, str]]          # address -> (name, kind, how: "sdk-name" | "pattern")
    threads: list[dict] = field(default_factory=list)
    queues: dict[int, dict] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    dma: list[dict] = field(default_factory=list)
    tasks: dict = field(default_factory=dict)
    interrupts: list[dict] = field(default_factory=list)
    managers: list[dict] = field(default_factory=list)
    outputs: list[dict] = field(default_factory=list)
    variants: list[dict] = field(default_factory=list)
    unresolved: list[dict] = field(default_factory=list)
    owners: dict[int, set[int]] = field(default_factory=dict)  # function -> thread entries reaching it
    collisions: int = 0
    pattern_candidates: list[dict] = field(default_factory=list)   # accepted and rejected, with reasons

    def entrypoints(self) -> list[dict]:
        out, seen = [], set()
        for t in self.threads:
            if t["entry"] is not None and t["entry"] not in seen:
                seen.add(t["entry"])
                out.append({"address": hx(t["entry"]), "kind": "thread", "name": self.names.get(t["entry"]),
                            "confidence_label": t["confidence_label"], "provenance_kind": t["provenance_kind"],
                            "provenance": t["provenance"]})
        out += [{"address": hx(i["handler"]), "kind": "interrupt", "name": self.names.get(i["handler"]),
                 "confidence_label": i["confidence_label"], "provenance_kind": i["provenance_kind"],
                 "provenance": i["provenance"]} for i in self.interrupts if i.get("handler") is not None]
        return out

    def as_dict(self) -> dict:
        api_calls = Counter(self.api[c.target][0] for f in self.functions.values() for c in f.calls
                            if c.target in self.api)
        return {"functions": len(self.functions), "address_collisions": self.collisions,
                "api_recognised": {hx(a): {"name": n, "kind": k, "how": how, "confidence": level(how),
                                           "provenance": how} for a, (n, k, how) in self.api.items()},
                "api_calls": dict(sorted(api_calls.items())),
                "pattern_candidates": self.pattern_candidates,
                "thread_count": len(self.threads),
                "thread_entries": len({t["entry"] for t in self.threads}),
                "thread_create_sites": len({s for t in self.threads for s in t["sites"]}),
                "threads": [dict(t, entry=hx(t["entry"])) for t in self.threads],
                "entrypoints": self.entrypoints(),
                "queues": [dict(q, address=hx(a)) for a, q in sorted(self.queues.items())],
                "events": self.events, "dma": self.dma,
                "tasks": [dict(t, address=hx(a)) for a, t in sorted(self.tasks.items())],
                "interrupts": [dict(i, handler=hx(i.get("handler"))) for i in self.interrupts],
                "managers": self.managers, "outputs": self.outputs,
                "variants": self.variants, "unresolved": self.unresolved[:200],
                "thread_ownership": {hx(t["entry"]): sum(1 for o in self.owners.values() if t["entry"] in o)
                                     for t in self.threads if t["entry"] is not None},
                "mmio_accesses": sum(len(f.accesses) for f in self.functions.values())}


def describe(value) -> str:
    if value is None:
        return "unknown at the call"
    kind = value[0]
    if kind == "c":
        return f"constant {hx(value[1])}"
    if kind == "lin":
        sid = value[1]
        if sid == ("sp",):
            return "a stack address"
        if sid[0] == "in":
            return f"${flowmod.NAMES[sid[2]]} at a block entry" if sid[1] else f"parameter ${flowmod.NAMES[sid[2]]}"
        if sid[0] == "call":
            return f"result of the call at {hx(sid[1])}"
        return f"value computed at {hx(sid[1])}"
    if kind == "mem":
        return f"word loaded from {hx(value[1])}"
    if kind == "set":
        return f"one of {value[3]} values from {hx(value[1])} step {value[2]}"
    return f"{kind} value"


class _Builder:
    def __init__(self, model: Model, read_words: list):
        self.m = model
        self.read_words = read_words
        self.callers: dict[int, list[Call]] = {}
        self.thread_keys: dict[tuple, dict] = {}            # (entry, id, pri, stack, thread) -> row
        for info in model.functions.values():
            for call in info.calls:
                if call.target is not None:
                    self.callers.setdefault(call.target, []).append(call)

    def read_word(self, address: int):
        for read in self.read_words:
            word = read(address)
            if word is not None:
                return word
        return None

    def name(self, address: int) -> str:
        return self.m.names.get(address) or hx(address)

    def resolve(self, call: Call, location: int, depth: int = 0, chain: tuple = ()) -> list[tuple]:
        """[(constant or None, provenance chain)] for an argument, through wrappers that pass on
        their own ``$a0..$a3``."""
        value = call.args.get(location)
        here = chain + (f"{'$' + flowmod.NAMES[location] if location < 32 else hex(location) + '($sp)'} "
                        f"at {hx(call.site)} in {self.name(call.function)}",)
        if value is not None and value[0] == "c":
            return [(value[1], here)]
        if value is not None and value[0] == "lin" and value[1][0] == "in" and value[1][1] == 0 \
                and value[1][2] in ARGS and value[2] == 1 and value[3] == 0 and depth < WRAPPER_DEPTH:
            callers = self.callers.get(call.function, [])
            out = []
            for caller in callers:
                out += self.resolve(caller, value[1][2], depth + 1,
                                    here + (f"forwarded parameter of {self.name(call.function)}",))
            return out or [(None, here + (f"parameter of {self.name(call.function)}, which has no known caller",))]
        return [(None, here + (describe(value),))]

    def first(self, call: Call, location: int):
        values = [v for v, _c in self.resolve(call, location) if v is not None]
        return values[0] if len(set(values)) == 1 else None


def _patterns(model: Model, starts: set[int]) -> None:
    """Unnamed thread-create and queue-create candidates (only when the SDK matcher named none).
    The call shape nominates a candidate; only its own body (``PATTERN_RULES``) accepts it."""
    kinds = {kind for _n, kind, _h in model.api.values()}
    by_target: dict[int, list[Call]] = {}
    for info in model.functions.values():
        for call in info.calls:
            if call.target is not None and call.target not in model.api:
                by_target.setdefault(call.target, []).append(call)

    def const(call, loc):
        v = call.args.get(loc)
        return v[1] if v is not None and v[0] == "c" else None

    for target, calls in sorted(by_target.items()):
        if "thread-create" not in kinds:
            hits = [c for c in calls if const(c, 6) in starts and const(c, 5) is not None and const(c, 5) < 0x100
                    and const(c, 0x10) is not None and not const(c, 0x10) & 7 and const(c, 0x10) not in starts]
            if hits and _corroborate(model, target, "thread-create", calls, hits,
                                     distinct_entries=len({const(c, 6) for c in hits})):
                continue
        if "queue-create" not in kinds:
            callee = model.functions.get(target)
            small_leaf = callee is not None and callee.end - callee.start <= 0x60 and not callee.calls
            hits = [c for c in calls if const(c, 4) and const(c, 5) and const(c, 4) not in starts
                    and const(c, 6) is not None and 1 <= const(c, 6) <= 0x400]
            if small_leaf and hits and len(hits) == len(calls):
                _corroborate(model, target, "queue-create", calls, hits)


def _corroborate(model: Model, target: int, kind: str, calls: list, hits: list, **extra) -> bool:
    """Accept a call-shape candidate only when its body does what the API's body must do; record
    the decision either way."""
    name, required, needs = PATTERN_RULES[kind]
    callee = model.functions.get(target)
    stores = callee.param_stores if callee is not None else set()
    accepted = callee is not None and all(r in stores for r in required)
    if callee is None:
        why = "not a catalogued function start: body not checked"
    elif accepted:
        why = None
    else:
        why = f"call shape only: the body does not {needs}"
    model.pattern_candidates.append(dict(
        {"address": hx(target), "api": name, "kind": kind, "accepted": accepted, "call_sites": len(calls),
         "shape_matches": len(hits)}, **extra,
        corroboration=[needs.replace("store ", "stores ", 1)] + [text for key, text in SUPPORTING.items()
                                                                  if key in stores] if accepted else [],
        rejected_because=why, confidence=level("pattern"), provenance="pattern"))
    if accepted:
        model.api[target] = (name, kind, "pattern")
    return accepted


def build(functions: dict[int, FunctionInfo], names: dict[int, str], read_words=(), *, starts=None) -> Model:
    """The runtime model of analysed functions; ``names`` maps addresses to SDK names."""
    starts = set(functions) if starts is None else set(starts)
    api = {address: (name, API[name][0], "sdk-name") for address, name in names.items() if name in API}
    model = Model(functions, dict(names), api)
    _patterns(model, starts)
    b = _Builder(model, list(read_words))
    for info in sorted(functions.values(), key=lambda f: f.start):
        for call in info.calls:
            if call.target not in model.api:
                continue
            name, kind, how = model.api[call.target]
            spec = API.get(name.rstrip("?"), API["osCreateThread"] if kind == "thread-create" else
                           API["osCreateMesgQueue"])[1]
            confidence = 1.0 if how == "sdk-name" else 0.5
            _interpret(b, call, name, kind, how, spec, confidence)
    _variants(model)
    _ownership(model)
    return model


def _interpret(b: _Builder, call: Call, name: str, kind: str, how: str, spec: dict, confidence: float) -> None:
    m = b.m
    where = {"site": hx(call.site), "function": hx(call.function), "caller": b.name(call.function), "api": name,
             "recognised_by": how, "provenance_kind": how, "confidence_label": level(how)}

    def value(param):
        return b.first(call, spec[param]) if param in spec else None

    if kind == "thread-create":
        found = b.resolve(call, spec["entry"])
        for entry, chain in found:
            ok = entry is not None and entry in m.functions
            row = dict(where, entry=entry, entry_name=m.names.get(entry) if entry is not None else None,
                       id=value("id"), pri=value("pri"), stack=hx(value("stack")), thread=hx(value("thread")),
                       confidence=round(confidence * (1.0 if ok else 0.6 if entry is not None else 0.0), 2),
                       confidence_label=level(how, ok), provenance=list(chain))
            if entry is None:
                m.unresolved.append(row)
                continue
            key = (entry, row["id"], row["pri"], row["stack"], row["thread"])   # a thread, not a call site
            same = b.thread_keys.get(key)
            if same is None:
                row["sites"] = [row["site"]]
                b.thread_keys[key] = row
                m.threads.append(row)
            elif row["site"] not in same["sites"]:
                same["sites"].append(row["site"])
    elif kind == "queue-create":
        queue = value("queue")
        if queue is None:
            m.unresolved.append(dict(where, what="queue address", provenance=list(b.resolve(call, 4)[0][1])))
            return
        q = m.queues.setdefault(queue, {"count": None, "created": [], "senders": [], "receivers": [], "events": []})
        q["count"] = value("count")
        q["created"].append(where)
    elif kind in ("queue-send", "queue-recv"):
        queue = value("queue")
        if queue is None:
            m.unresolved.append(dict(where, what="queue address", provenance=list(b.resolve(call, 4)[0][1])))
            return
        q = m.queues.setdefault(queue, {"count": None, "created": [], "senders": [], "receivers": [], "events": []})
        q["senders" if kind == "queue-send" else "receivers"].append(where)
    elif kind in ("event", "vi-event"):
        event = value("event")
        queue = value("queue")
        label = ("VI retrace" if kind == "vi-event" else
                 OS_EVENTS[event] if event is not None and 0 <= event < len(OS_EVENTS) else
                 f"event {event}" if event is not None else "unknown event")
        row = dict(where, event=label, queue=hx(queue), msg=value("msg"))
        m.events.append(row)
        if queue is not None:
            q = m.queues.setdefault(queue, {"count": None, "created": [], "senders": [], "receivers": [], "events": []})
            q["events"].append(label)
    elif kind == "dma":
        row = dict(where, direction=value("direction"), device=hx(value("device")), ram=hx(value("ram")),
                   size=value("size"), queue=hx(value("queue")), handle=hx(value("handle")))
        m.dma.append(row)
        if value("queue") is not None:
            q = m.queues.setdefault(value("queue"), {"count": None, "created": [], "senders": [], "receivers": [],
                                                     "events": []})
            q["senders"].append(dict(where, implicit="DMA completion"))
    elif kind == "task":
        task = value("task")
        if task is None:
            m.unresolved.append(dict(where, what="task address", provenance=list(b.resolve(call, 4)[0][1])))
            return
        t = m.tasks.setdefault(task, {"started": [], "fields": {}})
        fresh = not t["started"]
        t["started"].append(where)
        if fresh:
            _task_fields(b, task, t)
    elif kind == "interrupt":
        handler = value("handler")
        m.interrupts.append(dict(where, interrupt=value("interrupt"), handler=handler,
                                 handler_name=m.names.get(handler) if handler is not None else None,
                                 provenance=list(b.resolve(call, spec["handler"])[0][1])))
    elif kind == "interrupt-mask":
        m.interrupts.append(dict(where, mask=hx(value("mask")), handler=None, provenance=[]))
    elif kind in ("vi-manager", "pi-manager"):
        m.managers.append(dict(where, kind=kind, pri=value("pri"), queue=hx(value("queue"))))
        if kind == "pi-manager" and value("queue") is not None:
            q = m.queues.setdefault(value("queue"), {"count": None, "created": [], "senders": [], "receivers": [],
                                                     "events": []})
            q["receivers"].append(dict(where, implicit="PI manager command queue"))
    elif kind in ("audio-out", "video-out", "input", "timer"):
        row = dict(where, kind=kind)
        for param in spec:
            row[param] = hx(value(param)) if param in ("queue", "buffer", "framebuffer", "timer", "mode") \
                else value(param)
        m.outputs.append(row)
        if kind in ("input", "timer") and value("queue") is not None:
            q = m.queues.setdefault(value("queue"), {"count": None, "created": [], "senders": [], "receivers": [],
                                                     "events": []})
            q["senders"].append(dict(where, implicit=f"{name} completion"))


def _task_fields(b: _Builder, task: int, t: dict) -> None:
    for info in b.m.functions.values():
        for address, constant, site in info.stores:
            offset = address - task
            if offset in TASK_FIELDS and constant is not None:
                t["fields"].setdefault(TASK_FIELDS[offset], set()).add(constant)
    for offset, field_name in TASK_FIELDS.items():
        word = b.read_word(task + offset)
        if word:
            t["fields"].setdefault(field_name, set()).add(word)
    types = sorted(t["fields"].get("type", ()))
    t["type"] = [TASK_TYPES.get(v, f"type {v}") for v in types]
    t["fields"] = {k: sorted(hx(v) for v in vs) for k, vs in t["fields"].items()}


def _variants(m: Model) -> None:
    for address, (name, kind, how) in sorted(m.api.items()):
        if how != "sdk-name":
            m.variants.append({"name": name, "address": hx(address), "why": "recognised by call pattern only",
                               "corroboration": next((c["corroboration"] for c in m.pattern_candidates
                                                      if c["address"] == hx(address)), [])})
            continue
        info = m.functions.get(address)
        if info is None:
            m.variants.append({"name": name, "address": hx(address),
                               "why": "the SDK name is not on a catalogued function start (body not checked)"})
            continue
        expected = STOCK_READS.get(name)
        if not expected or info.end - info.start < 8:
            continue
        if info.live_in is None:
            continue
        reads = set(info.live_in) & set(ARGS)
        missing = sorted(expected - reads)
        if missing:
            m.variants.append({"name": name, "address": hx(address),
                               "why": "does not read " + ", ".join("$" + flowmod.NAMES[r] for r in missing)
                                      + " like the stock function: call sites interpreted with the stock signature"})


def _ownership(m: Model, limit: int = 50000) -> None:
    for thread in m.threads:
        entry = thread["entry"]
        if entry is None or entry not in m.functions:
            continue
        seen, stack = {entry}, [entry]
        while stack and len(seen) < limit:
            info = m.functions.get(stack.pop())
            if info is None:
                continue
            for callee in info.callees:
                if callee in m.functions and callee not in seen:
                    seen.add(callee)
                    stack.append(callee)
        for function in seen:
            m.owners.setdefault(function, set()).add(entry)


# --------------------------------------------------------------------------- images and projects

def analyse_images(images, names: dict[int, str], *, max_table: int = MAX_TABLE) -> Model:
    """``images`` = [(image name, data, delta, rows)]; rows = (name, vram, size)."""
    functions: dict[int, FunctionInfo] = {}
    collisions = 0
    readers = []
    for image, data, delta, rows in images:
        readers.append(image_reader(data, delta))
        for _n, vram, size in sorted(rows, key=lambda r: r[1]):
            if size <= 0 or vram - delta < 0 or vram - delta + size > len(data):
                continue
            if vram in functions:
                collisions += 1
                continue
            info = _function(data, delta, vram, size, image, max_table)
            functions[vram] = info
            code = data[vram - delta:vram - delta + size]
            code = code[:len(code) & ~3]
            if vram in names and names[vram] in STOCK_READS and len(code) >= 8:
                info.live_in = flowmod.analyse(code, vram).live_in
    model = build(functions, names, readers)
    model.collisions = collisions
    return model


def load_names(path) -> dict[int, str]:
    """address -> name from an ``sdk_symbols.txt`` / ``symbol_addrs.txt`` style file."""
    out = {}
    path = Path(path)
    if not path.is_file():
        return out
    for line in path.read_text(errors="replace").splitlines():
        m = _SYM.match(line)
        if m:
            out.setdefault(int(m.group(2), 16), m.group(1))
    return out


def analyse_project(project, images=None) -> Model:
    """Every loadable image of a project, with the SDK names of ``survey/sdk_symbols.txt``."""
    from ..match import images as imagemod
    from . import coverage
    project = Path(project)
    loaded = []
    for name in sorted(imagemod.load_registry(project)):
        if images and name not in images:
            continue
        try:
            spec = imagemod.resolve(project, name)
            loaded.append((name, spec.load(), spec.delta, coverage.rows_of(spec)))
        except Exception:                                    # noqa: BLE001 - an image that does not load
            continue
    return analyse_images(loaded, load_names(project / "survey" / "sdk_symbols.txt"))
