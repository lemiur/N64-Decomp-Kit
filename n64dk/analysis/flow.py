"""Delay-slot-aware control flow, register liveness and frame analysis of one MIPS code range.

Every later stage (boundary solving, prototype evidence, shape classification, residual
attribution) needs the same facts about a range of retail bytes:

* basic blocks with MIPS delay-slot ownership and branch-likely annulment;
* where control leaves the range (returns, jumps and branches outside it, running off the end);
* dominators, natural loops and loop depth;
* which registers are read before they are written (live at entry);
* the stack frame: size, saved and restored registers, and the anomalies that distinguish a
  complete compiler-generated function from an interior fragment or hand-written code;
* what every ``nop`` is for.

``analyse`` computes all of it from bytes only.  Nothing here depends on a game, a compiler
or a project layout; compilers differ in *what* they emit, not in what these facts mean.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import rabbitizer

ZERO, AT, V0, V1, A0, A3, SP, FP, RA = 0, 1, 2, 3, 4, 7, 29, 30, 31
NAMES = ("zero", "at", "v0", "v1", "a0", "a1", "a2", "a3", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7",
         "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "t8", "t9", "k0", "k1", "gp", "sp", "fp", "ra")
ARGS = frozenset(range(4, 8))
SAVED = frozenset([*range(16, 24), FP])
TEMPS = frozenset([AT, V0, V1, *range(8, 16), 24, 25])
CALL_CLOBBERED = frozenset([*TEMPS, *ARGS, RA])
_HILO_READ = {"mfhi", "mflo"}
_HILO_WRITE = {"mult", "multu", "div", "divu", "dmult", "dmultu", "ddiv", "ddivu"}


def reg_name(number: int) -> str:
    return "$" + NAMES[number]


@dataclass
class Insn:
    index: int
    vram: int
    word: int
    op: str
    uses: frozenset[int]
    defs: frozenset[int]
    kind: str = ""            # "", branch, jump, call, return, jr
    likely: bool = False
    target: int | None = None
    text: str = ""

    @property
    def control(self) -> bool:
        return bool(self.kind)


@dataclass
class Block:
    number: int
    first: int                # instruction index, inclusive
    last: int                 # instruction index, inclusive (the delay slot when there is one)
    succs: list[int] = field(default_factory=list)
    preds: list[int] = field(default_factory=list)
    loop_depth: int = 0


@dataclass
class Frame:
    size: int = 0
    saved: dict[int, int] = field(default_factory=dict)       # register -> sp offset
    restored: set[int] = field(default_factory=set)
    arg_spills: dict[int, int] = field(default_factory=dict)  # a0-a3 stored into the caller's home area
    anomalies: list[str] = field(default_factory=list)


@dataclass
class Flow:
    vram: int
    insns: list[Insn]
    blocks: list[Block]
    exits: list[tuple[int, str, int | None]]      # (site, kind, target)
    calls: list[tuple[int, int | None]]           # (site, target or None for jalr)
    live_in: frozenset[int]
    frame: Frame
    loops: list[tuple[int, frozenset[int]]]       # (header block, body blocks)
    dominators: list[frozenset[int]]
    anomalies: list[str]
    reachable: frozenset[int] = frozenset()
    block_live_in: list[frozenset[int]] = field(default_factory=list)

    @property
    def end(self) -> int:
        return self.vram + 4 * len(self.insns)

    def block_of(self, index: int) -> Block:
        for block in self.blocks:
            if block.first <= index <= block.last:
                return block
        raise IndexError(index)

    @property
    def leaf(self) -> bool:
        return not self.calls

    def live_after(self, index: int) -> frozenset[int]:
        """Registers whose current value may still be read after instruction ``index`` has
        executed (for a call: after it returns, i.e. after its delay slot)."""
        block = self.block_of(index)
        live = set()
        for succ in block.succs:
            live |= self.block_live_in[succ]
        position = block.last
        insns = self.insns
        start = index + 1
        if insns[index].control and index + 1 <= block.last:
            start = index + 2                       # the delay slot runs before the call returns
        events = []
        cursor = start
        while cursor <= position:
            insn = insns[cursor]
            if insn.control and cursor + 1 <= position:
                slot = insns[cursor + 1]
                events.append((insn.uses, frozenset()))
                events.append((slot.uses, frozenset() if insn.likely else slot.defs))
                if insn.kind == "call":
                    events.append((frozenset(), CALL_CLOBBERED))
                cursor += 2
            else:
                events.append((insn.uses, CALL_CLOBBERED if insn.kind == "call" else insn.defs))
                cursor += 1
        for uses, defs in reversed(events):
            live = (live - defs) | uses
        return frozenset(live)


def decode(code: bytes, vram: int) -> list[Insn]:
    out = []
    for i in range(0, len(code) - len(code) % 4, 4):
        word, = struct.unpack_from(">I", code, i)
        ins = rabbitizer.Instruction(word, vram + i)
        uses, defs = set(), set()
        if ins.isValid():
            for reads, name in ((ins.readsRs, "rs"), (ins.readsRt, "rt"), (ins.readsRd, "rd")):
                if reads():
                    uses.add(int(getattr(ins, name).value))
            dest = ins.getDestinationGpr()
            if dest is not None:
                defs.add(int(dest.value))
        uses.discard(ZERO)
        defs.discard(ZERO)
        kind, target = "", None
        if ins.isValid() and ins.isBranch():
            kind, target = "branch", ins.getBranchVramGeneric()
        elif ins.isValid() and ins.isJumpWithAddress():
            kind, target = ("call" if ins.doesLink() else "jump"), ins.getInstrIndexAsVram()
        elif ins.isValid() and ins.isJrRa():
            kind = "return"
        elif ins.isValid() and ins.isJrNotRa():
            kind = "jr"
        elif ins.isValid() and ins.isJump() and ins.doesLink():
            kind = "call"                                   # jalr
        op = ins.getOpcodeName() if ins.isValid() else ".word"
        if op in ("bal", "bgezal", "bltzal", "bgezall", "bltzall"):
            kind = "call"
        out.append(Insn(i // 4, vram + i, word, op, frozenset(uses), frozenset(defs), kind,
                        bool(ins.isValid() and ins.isBranchLikely()), target,
                        " ".join(ins.disassemble().split()) if ins.isValid() else f".word 0x{word:08X}"))
    return out


def _unconditional(insn: Insn) -> bool:
    if insn.kind in ("jump", "return", "jr"):
        return True
    if insn.kind != "branch":
        return False
    word = insn.word
    op, rs, rt = word >> 26, (word >> 21) & 31, (word >> 16) & 31
    return (op == 4 and rs == rt) or (op == 1 and rt == 1 and rs == 0)      # beq x,x / bgez $zero


def analyse(code: bytes, vram: int, jump_tables: dict[int, list[int]] | None = None) -> Flow:
    """``jump_tables`` maps the address of a ``jr $reg`` instruction to its target addresses."""
    insns = decode(code, vram)
    count, end = len(insns), vram + 4 * len(insns)
    tables = jump_tables or {}
    anomalies: list[str] = []

    def index_of(address: int) -> int | None:
        return (address - vram) // 4 if vram <= address < end and not address & 3 else None

    leaders = {0}
    delay_slots = set()
    for insn in insns:
        if not insn.control:
            continue
        if insn.index + 1 < count:
            delay_slots.add(insn.index + 1)
        if insn.kind != "call":
            if insn.index + 2 < count:
                leaders.add(insn.index + 2)
            targets = [insn.target] if insn.target is not None else tables.get(insn.vram, [])
            for target in targets:
                at = index_of(target)
                if at is not None:
                    leaders.add(at)
    for at in sorted(leaders & delay_slots):
        anomalies.append(f"branch target inside a delay slot at 0x{vram + 4 * at:08X}")
    order = sorted(leaders)
    blocks = [Block(n, first, (order[n + 1] if n + 1 < len(order) else count) - 1)
              for n, first in enumerate(order)]
    block_at = {block.first: block.number for block in blocks}
    exits: list[tuple[int, str, int | None]] = []
    calls: list[tuple[int, int | None]] = []

    def link(block: Block, address: int | None, site: int, outside_kind: str) -> None:
        at = index_of(address) if address is not None else None
        if at is None:
            exits.append((site, outside_kind, address))
        elif block_at[at] not in block.succs:
            block.succs.append(block_at[at])

    for block in blocks:
        terminator = None
        for index in range(block.first, block.last + 1):
            insn = insns[index]
            if insn.kind == "call":
                calls.append((insn.vram, insn.target))
            elif insn.control:
                terminator = insn
        after = vram + 4 * (block.last + 1)
        if terminator is None:
            link(block, after, insns[block.last].vram, "falls-off-end")
            continue
        if terminator.kind == "return":
            exits.append((terminator.vram, "return", None))
        elif terminator.kind == "jr":
            targets = tables.get(terminator.vram)
            if targets:
                for target in targets:
                    link(block, target, terminator.vram, "jump-table-outside")
            else:
                exits.append((terminator.vram, "computed-jump", None))
        elif terminator.kind == "jump":
            link(block, terminator.target, terminator.vram, "jump-outside")
        else:
            link(block, terminator.target, terminator.vram, "branch-outside")
            if not _unconditional(terminator):
                link(block, after, terminator.vram, "falls-off-end")
    for block in blocks:
        for succ in block.succs:
            blocks[succ].preds.append(block.number)
    reachable, stack = {0} if blocks else set(), [0] if blocks else []
    while stack:
        for succ in blocks[stack.pop()].succs:
            if succ not in reachable:
                reachable.add(succ)
                stack.append(succ)

    def live_site(site: int) -> bool:
        return block_at_site(site) in reachable

    def block_at_site(site: int) -> int:
        index = (site - vram) // 4
        return next(b.number for b in blocks if b.first <= index <= b.last)

    exits = [e for e in exits if live_site(e[0])]
    calls = [c for c in calls if live_site(c[0])]
    dead = [b for b in blocks if b.number not in reachable
            and any(insns[i].word for i in range(b.first, b.last + 1))]
    if dead and not any(i.kind == "jr" and i.vram not in tables for i in insns):
        anomalies.append(f"{len(dead)} block(s) not reachable from the entry (first at "
                         f"0x{insns[dead[0].first].vram:08X}): dead code or a second entry point")

    dominators = _dominators(blocks)
    loops = _loops(blocks, dominators)
    for header, body in loops:
        for number in body:
            blocks[number].loop_depth += 1
    block_live = _live_in(insns, blocks)
    live_in = block_live[0] if block_live else frozenset()
    frame = _frame(insns, exits, calls, vram)
    return Flow(vram, insns, blocks, exits, calls, live_in, frame, loops, dominators, anomalies,
                frozenset(reachable), block_live)


def _dominators(blocks: list[Block]) -> list[frozenset[int]]:
    everything = frozenset(range(len(blocks)))
    dom = [everything] * len(blocks)
    if not blocks:
        return dom
    dom[0] = frozenset([0])
    changed = True
    while changed:
        changed = False
        for block in blocks[1:]:
            preds = [dom[p] for p in block.preds]
            new = (frozenset.intersection(*preds) if preds else frozenset()) | {block.number}
            if new != dom[block.number]:
                dom[block.number], changed = new, True
    return dom


def _loops(blocks: list[Block], dom: list[frozenset[int]]) -> list[tuple[int, frozenset[int]]]:
    found: dict[int, set[int]] = {}
    for block in blocks:
        for succ in block.succs:
            if succ in dom[block.number]:                    # back edge block -> succ
                body, stack = {succ, block.number}, [block.number]
                while stack:
                    for pred in blocks[stack.pop()].preds:
                        if pred not in body:
                            body.add(pred)
                            stack.append(pred)
                found.setdefault(succ, set()).update(body)
    return [(header, frozenset(body)) for header, body in sorted(found.items())]


def _events(insns: list[Insn], block: Block):
    """(uses, definite defs) in execution order for everything that always executes in the
    block.  A call's clobbers follow its delay slot.  The delay slot of a branch-likely is
    left out: it runs only when the branch is taken (see ``_annulled``)."""
    index = block.first
    while index <= block.last:
        insn = insns[index]
        if insn.control and index + 1 <= block.last:
            slot = insns[index + 1]
            yield insn.uses, frozenset()
            if not insn.likely:
                yield slot.uses, slot.defs
            if insn.kind == "call":
                yield frozenset(), CALL_CLOBBERED | insn.defs
            index += 2
        else:
            yield insn.uses, (CALL_CLOBBERED if insn.kind == "call" else insn.defs)
            index += 1


def _annulled(insns: list[Insn], block: Block) -> Insn | None:
    """The delay-slot instruction that executes only on the taken edge, if the block ends in a
    branch-likely."""
    if block.last > block.first and insns[block.last - 1].likely:
        return insns[block.last]
    return None


def _live_in(insns: list[Insn], blocks: list[Block]) -> list[frozenset[int]]:
    vram = insns[0].vram if insns else 0
    use, define, taken = [], [], []
    for block in blocks:
        used, defined = set(), set()
        for uses, defs in _events(insns, block):
            used |= uses - defined
            defined |= defs
        use.append(used)
        define.append(defined)
        slot = _annulled(insns, block)
        target = insns[block.last - 1].target if slot is not None else None
        taken.append((slot, next((b.number for b in blocks if target is not None
                                  and insns[b.first].vram == target), None)))
    live = [set() for _ in blocks]
    changed = True
    while changed:
        changed = False
        for block in reversed(blocks):
            slot, target_block = taken[block.number]
            out = set()
            for succ in block.succs:
                incoming = live[succ]
                if slot is not None and succ == target_block:
                    incoming = slot.uses | (incoming - slot.defs)
                out |= incoming
            if slot is not None and target_block is None:
                out |= slot.uses                         # taken edge leaves the range
            new = use[block.number] | (out - define[block.number])
            if new != live[block.number]:
                live[block.number], changed = new, True
    return [frozenset(item) for item in live]


def _frame(insns: list[Insn], exits, calls, vram: int) -> Frame:
    frame = Frame()
    written: set[int] = set()
    returns = any(kind == "return" for _site, kind, _target in exits)
    for insn in insns:
        word = insn.word
        op, rs, rt, imm = word >> 26, (word >> 21) & 31, (word >> 16) & 31, word & 0xFFFF
        simm = imm - 0x10000 if imm & 0x8000 else imm
        if op in (9, 25) and rs == SP and rt == SP:                     # addiu/daddiu sp, sp, N
            if simm < 0 and not frame.size:
                frame.size = -simm
            elif simm > 0 and frame.size and simm != frame.size:
                frame.anomalies.append(f"stack released by {simm} but allocated {frame.size} at 0x{insn.vram:08X}")
            elif simm > 0 and not frame.size:
                frame.anomalies.append(f"stack released without allocation at 0x{insn.vram:08X}")
        elif op in (43, 63) and rs == SP:                               # sw/sd reg, off(sp)
            if (rt in SAVED or rt == RA) and rt not in written and rt not in frame.saved:
                frame.saved[rt] = simm
            elif rt in ARGS and rt not in written and frame.size and simm >= frame.size:
                frame.arg_spills[rt] = simm
        elif op in (35, 55) and rs == SP and (rt in SAVED or rt == RA):  # lw/ld reg, off(sp)
            if frame.saved.get(rt) == simm:
                frame.restored.add(rt)
            elif rt not in frame.saved:
                frame.anomalies.append(f"{reg_name(rt)} restored from the stack but never saved "
                                       f"(0x{insn.vram:08X})")
        for reg in insn.defs:
            if reg in SAVED and reg not in frame.saved and reg not in written:
                frame.anomalies.append(f"{reg_name(reg)} written without being saved (0x{insn.vram:08X})")
        written |= insn.defs
        if insn.kind == "call":
            written.add(RA)
    if calls and returns and RA not in frame.saved:
        frame.anomalies.append("calls another function but never saves $ra")
    if returns:
        for reg in sorted(set(frame.saved) - frame.restored):
            frame.anomalies.append(f"{reg_name(reg)} saved but not restored before returning")
    return frame


def nop_roles(flow: Flow) -> list[tuple[int, str]]:
    """(address, role) for every all-zero word: ``delay-slot:<kind>``, ``padding``,
    ``load-delay``, ``hilo-hazard`` or ``inline``."""
    insns = flow.insns
    last_control = max((i.index for i in insns if i.control and i.kind != "call"), default=-1)
    out = []
    for insn in insns:
        if insn.word:
            continue
        prev = insns[insn.index - 1] if insn.index else None
        nxt = insns[insn.index + 1] if insn.index + 1 < len(insns) else None
        if prev is not None and prev.control:
            role = "delay-slot:" + prev.kind
        elif last_control >= 0 and insn.index > last_control + 1 and all(i.word == 0 for i in insns[insn.index:]):
            role = "padding"
        elif prev is not None and nxt is not None and prev.op.startswith("l") and prev.defs & nxt.uses \
                and prev.op not in ("lui",):
            role = "load-delay"
        elif any(insns[k].op in _HILO_READ for k in range(max(0, insn.index - 2), insn.index)) and \
                any(insns[k].op in _HILO_WRITE for k in range(insn.index + 1, min(len(insns), insn.index + 3))):
            role = "hilo-hazard"
        else:
            role = "inline"
        out.append((insn.vram, role))
    return out


def entry_evidence(flow: Flow) -> dict:
    """Is this range a complete, independently callable, compiler-generated function?

    ``problems`` are facts a compiler-generated function never exhibits (so the range is an
    interior fragment, a piece of a split function, or hand-written code); ``notes`` are
    weaker observations.  ``arguments`` is the highest argument register read before being
    written plus one (a lower bound; incidental reads are possible)."""
    problems, notes = [], []
    temps = sorted(flow.live_in & TEMPS)
    if temps:
        problems.append("reads " + ", ".join(reg_name(r) for r in temps) + " before writing: uses a caller's "
                        "temporaries, so execution arrives here from inside another body")
    saved_live = sorted((flow.live_in & SAVED) - set(flow.frame.saved))
    if saved_live:
        problems.append("reads " + ", ".join(reg_name(r) for r in saved_live) + " at entry without saving them")
    problems += [a for a in flow.frame.anomalies if "never saved" in a or "without being saved" in a
                 or "without allocation" in a or "never saves $ra" in a]
    notes += [a for a in flow.frame.anomalies if a not in problems] + flow.anomalies
    kinds = {}
    for _site, kind, _target in flow.exits:
        kinds[kind] = kinds.get(kind, 0) + 1
    if kinds.get("falls-off-end"):
        problems.append("control runs off the end of the range into the following bytes")
    if kinds.get("branch-outside"):
        problems.append(f"{kinds['branch-outside']} conditional/relative branch(es) leave the range: the range is "
                        "not a whole function (false split, shared tail or hand-written code)")
    if kinds.get("jump-outside"):
        notes.append(f"{kinds['jump-outside']} `j` to an address outside the range (tail jump, shared exit or "
                     "hand-written code)")
    if not kinds.get("return") and not kinds.get("jump-outside") and not kinds.get("computed-jump") \
            and not problems and not flow.loops:
        notes.append("no return and no loop: check the end of the range")
    arguments = max((r - A0 + 1 for r in flow.live_in & ARGS), default=0)
    return {"complete": not problems, "problems": problems, "notes": notes, "arguments_read": arguments,
            "leaf": flow.leaf, "frame_size": flow.frame.size,
            "saved": [reg_name(r) for r in sorted(flow.frame.saved)], "exits": kinds,
            "loops": len(flow.loops), "blocks": len(flow.blocks)}


def shapes(flow: Flow) -> list[dict]:
    """Ranked source-shape guesses with the evidence for each (several may apply)."""
    insns = [i for i in flow.insns]
    body = [i for i in insns if i.word and not i.control]
    ops = [i.op for i in body]
    loads = [i for i in body if i.op in ("lw", "lh", "lhu", "lb", "lbu", "lwc1", "ldc1", "ld")]
    stores = [i for i in body if i.op in ("sw", "sh", "sb", "swc1", "sdc1", "sd")]
    out = []

    def add(shape, confidence, why):
        out.append({"shape": shape, "confidence": confidence, "evidence": why})

    exits = {kind for _s, kind, _t in flow.exits}
    if flow.leaf and not body:
        add("empty", 0.95, "only a return")
    if flow.leaf and not flow.loops and len(body) <= 3 and not loads and not stores and exits == {"return"}:
        add("constant-or-argument-return", 0.85, f"{len(body)} computing instruction(s), no memory access")
    if flow.leaf and not flow.loops and loads and not stores and len(body) <= 6:
        add("getter", 0.8, f"{len(loads)} load(s), no store, {len(body)} instructions")
    if flow.leaf and not flow.loops and stores and len(body) <= 8 and len(flow.blocks) == 1:
        add("setter-or-initialiser", 0.75, f"{len(stores)} store(s) in one block")
    if flow.leaf and not flow.loops and stores and any(op in ("andi", "ori", "and", "or", "nor", "xori") for op in ops) \
            and len(body) <= 12:
        add("flag-operation", 0.6, "load, bit operation and store")
    if len(flow.calls) == 1 and not flow.loops and len(body) <= 14:
        add("wrapper", 0.7, "one call, no loop")
    if "jump-outside" in exits and not flow.calls and len(body) <= 8:
        add("tail-jump-thunk", 0.7, "ends in `j` to another function")
    if any(kind == "computed-jump" for _s, kind, _t in flow.exits) or \
            any(i.kind == "jr" for i in insns):
        add("switch-dispatcher", 0.75, "indirect `jr` through a register")
    for header, blocks in flow.loops:
        members = [i for b in blocks for i in insns[flow.blocks[b].first:flow.blocks[b].last + 1]]
        if any(i.op == "lw" and i.defs and next(iter(i.defs)) == ((i.word >> 21) & 31) for i in members):
            add("linked-list-walk", 0.7, f"loop at 0x{insns[flow.blocks[header].first].vram:08X} reloads its own "
                "base pointer (`p = p->next`)")
        steps = [i for i in members if i.op == "addiu" and ((i.word >> 21) & 31) == ((i.word >> 16) & 31)
                 and ((i.word >> 21) & 31) != SP]
        if steps:
            stride = sorted({(s.word & 0xFFFF) - (0x10000 if s.word & 0x8000 else 0) for s in steps})
            add("counted-or-pointer-loop", 0.65, f"loop with induction step(s) {stride}")
        if any(i.kind == "call" for i in members):
            add("callback-or-dispatch-loop", 0.5, "loop body contains a call")
    if len(flow.calls) >= 3 and not flow.loops and len(flow.blocks) <= 2:
        add("call-sequence", 0.6, f"{len(flow.calls)} calls in straight-line code")
    if not out:
        add("general", 0.3, f"{len(flow.blocks)} blocks, {len(flow.calls)} calls, {len(flow.loops)} loops")
    return sorted(out, key=lambda row: -row["confidence"])


def find_jump_tables(code: bytes, vram: int, read_word, limit: int = 512) -> dict[int, list[int]]:
    """``jr $reg`` site -> jump-table targets, for use as ``analyse(..., jump_tables=...)``.

    Recognises the compiler patterns ``lui X; addu X, X, idx; lw R, lo(X); jr R`` and
    ``lui X; addiu X, X, lo; addu Y, X, idx; lw R, 0(Y); jr R``, and their 64-bit-register forms
    (``daddu`` / ``daddiu``, as EGCS emits at -mips3).  ``read_word(address)``
    returns the 32-bit word at a virtual address or None; entries are read while they are
    aligned addresses inside the range."""
    words = [struct.unpack_from(">I", code, i)[0] for i in range(0, len(code) - len(code) % 4, 4)]
    end = vram + 4 * len(words)
    out: dict[int, list[int]] = {}
    for index, word in enumerate(words):
        if word >> 26 != 0 or (word & 0x3F) != 8 or (word >> 21) & 31 in (RA, ZERO):
            continue
        register, table = (word >> 21) & 31, None
        for back in range(index - 1, max(-1, index - 8), -1):
            load = words[back]
            if load >> 26 != 35 or (load >> 16) & 31 != register:
                continue
            base, offset = (load >> 21) & 31, (load & 0xFFFF) - (0x10000 if load & 0x8000 else 0)
            wanted, low = {base}, 0
            for earlier in range(back - 1, max(-1, back - 10), -1):
                w = words[earlier]
                op, rs, rt, rd = w >> 26, (w >> 21) & 31, (w >> 16) & 31, (w >> 11) & 31
                if op == 0 and (w & 0x3F) in (0x21, 0x2D) and rd in wanted:  # (d)addu base, a, b
                    wanted |= {rs, rt}
                elif op in (9, 25) and rt in wanted and rs in wanted:         # (d)addiu X, X, lo
                    low += (w & 0xFFFF) - (0x10000 if w & 0x8000 else 0)
                elif op == 15 and rt in wanted:                               # lui X, hi
                    table = (((w & 0xFFFF) << 16) + low + offset) & 0xFFFFFFFF
                    break
            break
        if table is None:
            continue
        targets = []
        while len(targets) < limit:
            value = read_word(table + 4 * len(targets))
            if value is None or value & 3 or not vram <= value < end:
                break
            targets.append(value)
        if targets:
            out[vram + 4 * index] = targets
    return out


def analyse_in_image(image: bytes, vram_delta: int, vram: int, size: int) -> Flow:
    """``analyse`` for a function inside a linear image (file offset = address - vram_delta),
    with its jump tables resolved from the same image."""
    def read_word(address: int):
        offset = address - vram_delta
        return struct.unpack_from(">I", image, offset)[0] if 0 <= offset <= len(image) - 4 else None
    offset = vram - vram_delta
    code = image[offset:offset + size]
    return analyse(code, vram, find_jump_tables(code, vram, read_word))
