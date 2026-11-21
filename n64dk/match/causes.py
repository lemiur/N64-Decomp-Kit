"""Why does this function not match yet?  Ranked, typed causes from a stable taxonomy.

A score says how far a candidate is; it does not say what kind of knowledge is missing.
``diagnose`` turns the retail bytes (and, when there is one, the candidate's relocated bytes)
into a ranked list of causes.  Each cause has a stable identifier, the evidence that triggered
it, the ``iterate`` families able to act on it, and one concrete next experiment.  Several
causes may hold at once; they are ranked, never forced into one label.

The identifiers are the contract between diagnosis, scheduling and reporting::

    boundary.*      the range is not a whole function (fragment, false split, shared tail,
                    embedded data) -- fix the catalogue, do not spend compiles
    abi.nonstandard reads or writes registers no o32 function may (IDO -O3 static functions
                    compiled with their callers, or hand-written code)
    assembly.handwritten   positive evidence of hand-written code
    source.missing  no candidate source yet
    type.*          signedness / width / volatile qualification of a value or field
    global.*        declaration of a global (type, array vs pointer, address)
    register.*      lifetime (call-crossing or not), pressure, or a pure permutation
    schedule.*      delay-slot content, load-use hazards
    control.*       branch form, branch-likely, loop shape, switch
    frame.size      stack frame layout
    assembler.variant      differences only an assembler creates ($at sequences, macros, move encoding)
    compiler.family        the bytes carry another compiler family's fingerprint (IDO vs GCC)
    relocation.target      a call or address resolves to a different symbol
    tu.context      evidence that the function only matches inside its translation unit
    semantic.unknown       instructions differ in ways none of the above explains
"""
from __future__ import annotations

import difflib
from collections import Counter
from dataclasses import dataclass, field

from ..analysis import flow as flowmod
from ..analysis.flow import AT, SAVED, TEMPS, Flow, reg_name

SYSTEM_INSTRUCTIONS = frozenset({"mfc0", "mtc0", "dmfc0", "dmtc0", "eret", "tlbr", "tlbwi", "tlbwr", "tlbp", "cache"})

TAXONOMY = {
    "boundary.fragment": "range starts inside another function's body",
    "boundary.false_split": "control leaves the range by a branch or by running off its end",
    "boundary.shared_tail": "range jumps into code catalogued as another function",
    "boundary.embedded_data": "range contains words that are not instructions",
    "abi.nonstandard": "register use outside the o32 convention",
    "assembly.handwritten": "hand-written assembly",
    "source.missing": "no candidate source",
    "type.signedness": "a value or field has the wrong signedness",
    "type.width": "a value or field has the wrong width",
    "type.volatile": "memory is accessed through a volatile-qualified type in the original",
    "global.declaration": "a global is declared with the wrong type or shape",
    "register.lifetime": "a value is (not) live across a call in the original",
    "register.permutation": "same code with registers consistently exchanged",
    "register.pressure": "different registers without a consistent mapping",
    "schedule.delay_slot": "a delay slot holds a different instruction",
    "schedule.load_hazard": "only nops differ",
    "control.branch_form": "a condition or its arms are inverted",
    "control.branch_likely": "branch-likely form differs",
    "control.switch": "computed jump (switch) present",
    "frame.size": "stack frame size differs",
    "assembler.variant": "difference created by the assembler, not the compiler",
    "compiler.family": "retail code was produced by a different compiler family than the candidate pipeline",
    "relocation.target": "call or address resolves elsewhere",
    "call.set": "the calls made differ in number or in target",
    "control.shape": "the number of branches, returns or computed jumps differs",
    "control.switch_map": "a jump table sends cases to different arms",
    "tu.context": "matches only inside its translation unit",
    "semantic.unknown": "instructions differ without a recognised pattern",
}
FAMILIES_FOR = {
    "type.signedness": ("types", "fields", "casts", "globals", "ast-cast"),
    "type.width": ("types", "fields", "casts", "globals", "ast-cast", "hand"),
    "type.volatile": ("volatile", "globals"),
    "global.declaration": ("globals", "hand"),
    "register.lifetime": ("lifetime", "ast-stmt-order", "scope", "split", "ast-temp", "ast-inline-temp"),
    "register.permutation": ("callargs", "ast-control", "ast-inline-temp", "ast-temp", "ast-decl-order", "order",
                             "lifetime", "ast-stmt-order", "scope", "split", "ast-expr", "expression", "hand"),
    "register.pressure": ("callargs", "ast-inline-temp", "ast-temp", "ast-control", "lifetime", "split", "scope",
                          "ast-expr", "expression", "types", "hand"),
    "schedule.delay_slot": ("volatile", "ast-stmt-order", "lifetime", "order", "ast-sameline", "ast-block",
                            "expression", "hand"),
    "schedule.load_hazard": ("ast-stmt-order", "ast-sameline", "ast-block"),
    "control.branch_form": ("ast-control", "control", "ast-expr", "expression", "hand"),
    "control.branch_likely": ("ast-control", "control", "ast-expr", "expression"),
    "call.set": ("callargs", "split", "ast-inline-temp", "ast-control", "expression"),
    "control.shape": ("ast-control", "control", "ast-block", "ast-expr", "hand"),
    "frame.size": ("ast-inline-temp", "ast-temp", "types", "split", "scope", "volatile", "ast-cast", "hand"),
    "semantic.unknown": ("callargs", "ast-control", "ast-expr", "ast-cast", "ast-inline-temp", "types", "fields",
                         "casts", "expression", "control"),
}
_LOAD_SIGN = {("lh", "lhu"), ("lhu", "lh"), ("lb", "lbu"), ("lbu", "lb"), ("slt", "sltu"), ("sltu", "slt"),
              ("slti", "sltiu"), ("sltiu", "slti"), ("sra", "srl"), ("srl", "sra"), ("div", "divu"),
              ("divu", "div"), ("mult", "multu"), ("multu", "mult"), ("srav", "srlv"), ("srlv", "srav")}
_WIDTH = {"lb": 1, "lbu": 1, "sb": 1, "lh": 2, "lhu": 2, "sh": 2, "lw": 4, "sw": 4, "ld": 8, "sd": 8,
          "lwc1": 4, "swc1": 4, "ldc1": 8, "sdc1": 8}
_MEMORY = set(_WIDTH)
_STORES = {"sb", "sh", "sw", "sd", "swc1", "sdc1"}
_MOVES = {"addu", "or", "daddu"}
_LIKELY = {"beql", "bnel", "beqzl", "bnezl", "bgezl", "bltzl", "blezl", "bgtzl"}
_INVERSE = {"beq": "bne", "bne": "beq", "beqz": "bnez", "bnez": "beqz", "bgez": "bltz", "bltz": "bgez",
            "blez": "bgtz", "bgtz": "blez", "bc1t": "bc1f", "bc1f": "bc1t"}


@dataclass
class Cause:
    id: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    next: str = ""

    @property
    def families(self) -> tuple[str, ...]:
        return FAMILIES_FOR.get(self.id, ())

    def as_dict(self) -> dict:
        return {"cause": self.id, "meaning": TAXONOMY[self.id], "confidence": round(self.confidence, 2),
                "evidence": self.evidence[:6], "families": list(self.families), "next": self.next}


class _Bag:
    def __init__(self):
        self.items: dict[str, Cause] = {}

    def add(self, cause: str, confidence: float, evidence: str, next_step: str = "") -> None:
        item = self.items.setdefault(cause, Cause(cause, 0.0))
        item.confidence = min(0.99, max(item.confidence, confidence) + (0.02 if item.evidence else 0.0))
        if evidence and evidence not in item.evidence:
            item.evidence.append(evidence)
        if next_step and not item.next:
            item.next = next_step

    def ranked(self) -> list[Cause]:
        return sorted(self.items.values(), key=lambda c: -c.confidence)


def compiler_family(flow: Flow) -> dict:
    """Compiler family fingerprint of one function from two toolchain-determined encodings.

    Register copies: IDO (and modern GNU as) encode ``move`` as ``or rd, rs, $zero``; the GNU as
    shipped with KMC/SN GCC encodes it as ``addu`` (``daddu`` when assembling 64-bit).
    Unconditional in-function jumps: IDO emits ``b`` (``beq $zero, $zero``), GCC emits ``j``.
    Returns ``{"family": "ido" | "gcc" | None, "confidence", "moves", "unconditional"}``."""
    moves, jumps = Counter(), Counter()
    for insn in flow.insns:
        word = insn.word
        if word >> 26 == 0 and (word & 0x7FF) in (0x21, 0x25, 0x2D) and (word >> 16) & 31 == 0 \
                and (word >> 21) & 31 and (word >> 11) & 31:
            moves[{0x21: "addu", 0x25: "or", 0x2D: "daddu"}[word & 0x7FF]] += 1
        elif insn.kind == "branch" and word >> 16 == 0x1000:
            jumps["b"] += 1
        elif insn.kind == "jump" and insn.target is not None and flow.vram <= insn.target < flow.end:
            jumps["j"] += 1
    ido = moves["or"] + jumps["b"]
    gcc = moves["addu"] + moves["daddu"] + jumps["j"]
    family, confidence = None, 0.0
    if ido + gcc:
        family = "ido" if ido > gcc else "gcc" if gcc > ido else None
        share = max(ido, gcc) / (ido + gcc)
        confidence = round(min(0.95, 0.5 + 0.1 * (ido + gcc)) * share, 2) if family else 0.0
    return {"family": family, "confidence": confidence, "moves": dict(moves), "unconditional": dict(jumps)}


def pipeline_family(pipeline: str | None) -> str | None:
    if not pipeline or pipeline.startswith("custom:"):
        return None                      # a configured lane: its family is declared, not guessed here
    return "ido" if pipeline.startswith("ido") else "gcc"


def target_causes(flow: Flow, neighbours: set[int] | None = None) -> list[Cause]:
    """Causes visible in the retail bytes alone.  ``neighbours`` are other catalogued starts."""
    bag = _Bag()
    evidence = flowmod.entry_evidence(flow)
    temps = sorted(flow.live_in & TEMPS)
    saved_live = sorted((flow.live_in & SAVED) - set(flow.frame.saved))
    writes_unsaved = [a for a in flow.frame.anomalies if "without being saved" in a]
    restores = [a for a in flow.frame.anomalies if "never saved" in a or "without allocation" in a]
    invalid = [i for i in flow.insns if i.op == ".word"]
    if invalid:
        bag.add("boundary.embedded_data", 0.9 if len(invalid) > 2 else 0.6,
                f"{len(invalid)} word(s) do not decode as instructions (first at 0x{invalid[0].vram:08X})",
                "split the data out of the function range (jump table, literal pool or padding)")
    if restores:
        bag.add("boundary.fragment", 0.9, restores[0],
                "merge with the preceding catalogue entry: this is the tail of a larger function")
    if temps and not saved_live and not writes_unsaved:
        bag.add("boundary.fragment", 0.85, "reads " + ", ".join(reg_name(r) for r in temps) + " before writing them",
                "run fragment-audit: the entry is an interior label of the preceding function")
    if saved_live or writes_unsaved:
        what = ("reads " + ", ".join(reg_name(r) for r in saved_live) + " at entry without saving them") \
            if saved_live else writes_unsaved[0]
        bag.add("abi.nonstandard", 0.8, what,
                "IDO -O3 compiles static functions with their callers and passes values in $s/$t registers: "
                "match the whole translation unit at -O3; otherwise treat as hand-written or as a fragment")
        bag.add("tu.context", 0.6, what, "identify the callers and build them in one translation unit")
        if temps:
            bag.add("boundary.fragment", 0.5, "also reads caller temporaries", "")
    exits = evidence["exits"]
    if exits.get("branch-outside") or exits.get("falls-off-end"):
        bag.add("boundary.false_split", 0.9,
                "; ".join(p for p in evidence["problems"] if "leave the range" in p or "runs off" in p),
                "extend the range to the real end of the function (see fragment-audit merges)")
    outside = [(site, target) for site, kind, target in flow.exits if kind == "jump-outside"]
    if outside:
        inside_other = neighbours is not None and any(target not in neighbours for _s, target in outside)
        bag.add("boundary.shared_tail", 0.75 if inside_other else 0.5,
                f"`j` from 0x{outside[0][0]:08X} to 0x{outside[0][1]:08X}, outside the range"
                + (" and not at a catalogued function start" if inside_other else ""),
                "neither GCC 2.7 nor IDO emits jumps between functions: the range is mis-split or hand-written")
    if any(kind == "computed-jump" for _s, kind, _t in flow.exits):
        bag.add("control.switch", 0.7, "`jr` through a register without a resolved jump table",
                "locate the jump table in rodata and verify it as data owned by this function")
    import rabbitizer
    odd = [i for i in flow.insns if i.word and i.op != ".word"
           and rabbitizer.Instruction(i.word, i.vram).isLikelyHandwritten()]
    kernel = [i for i in flow.insns if (set(i.uses) | set(i.defs)) & {26, 27}]
    # System instructions have no C spelling in these compilers: one is enough to be certain.
    system = [i for i in flow.insns if i.op in SYSTEM_INSTRUCTIONS]
    odd = odd or system
    if odd or kernel:
        bag.add("assembly.handwritten", 0.9 if system else 0.85 if (odd and kernel) or len(odd) > 2 else 0.6,
                (f"{len(odd)} instruction(s) compilers do not emit (first: {odd[0].text})" if odd else "")
                + ("; uses $k0/$k1" if kernel else ""),
                "confirm against SDK assembly sources; credit only with positive provenance")
    return bag.ranked()


def _webs(insns) -> dict[int, list[tuple[int, int, bool, int]]]:
    """register -> [(first index, last index, crosses a call, uses)] in address order (linear
    approximation: a value lives from a definition to the last read before the next definition)."""
    out: dict[int, list[tuple[int, int, bool, int]]] = {}
    open_webs: dict[int, list[int]] = {}
    calls = [i.index for i in insns if i.kind == "call"]

    def close(reg):
        first, last, uses = open_webs.pop(reg)
        crosses = any(first < c < last for c in calls)
        out.setdefault(reg, []).append((first, last, crosses, uses))

    for insn in insns:
        for reg in insn.uses:
            if reg in open_webs:
                open_webs[reg][1] = insn.index
                open_webs[reg][2] += 1
            else:
                open_webs[reg] = [0, insn.index, 1]          # live-in value
        for reg in insn.defs:
            if reg in open_webs:
                close(reg)
            open_webs[reg] = [insn.index, insn.index, 0]
    for reg in list(open_webs):
        close(reg)
    return out


def gcc_priority(refs: int, live_length: int, loop_depth: int = 0) -> float:
    """GCC 2.7.2's allocation priority of a pseudo register (local-alloc/global): ``floor_log2(refs)
    * refs / live_length``, where every reference counts once per enclosing loop level (``flow.c``
    adds ``loop_depth`` per reference, outside loops 1).  Higher priority is allocated first and gets
    the lowest free register; ties go to the lower pseudo number (the value created first)."""
    weighted = refs * (loop_depth + 1)
    if weighted <= 0 or live_length <= 0:
        return 0.0
    return (weighted.bit_length() - 1) * weighted / live_length


def permutation_advice(forward: dict[int, set[int]], candidate_webs) -> str:
    """For a consistent register exchange: which candidate value must gain or lose priority.  The
    candidate's value in register b belongs in register t; a value that must move to a register
    allocated earlier (a lower number among the same class) must outrank the value now there."""
    notes = []
    for built, targets in sorted(forward.items()):
        retail = next(iter(targets))
        if retail == built or not candidate_webs.get(built):
            continue
        first, last, _crosses, uses = max(candidate_webs[built], key=lambda web: web[3])
        priority = gcc_priority(uses, last - first + 1)
        direction = "gain" if retail < built else "lose"
        verb = "a reference (or move it into a loop)" if direction == "gain" else \
            "a reference (or take it out of a loop, or let its rival gain one)"
        notes.append(f"the value in {reg_name(built)} ({uses} refs over {last - first + 1} instructions, priority "
                     f"{priority:.2f}) belongs in {reg_name(retail)}: it must {direction} priority, i.e. "
                     f"{'gain' if direction == 'gain' else 'lose'} {verb}")
        if len(notes) == 2:
            break
    return "; ".join(notes)


def _web_at(webs, reg: int, index: int):
    for web in webs.get(reg, []):
        if web[0] <= index <= web[1]:
            return web
    return None


def diagnose(target: bytes, candidate: bytes | None, vram: int, flow: Flow | None = None,
             neighbours: set[int] | None = None, pipeline: str | None = None) -> list[Cause]:
    """Ranked causes.  ``candidate`` is the relocated candidate code (None: no source yet);
    ``pipeline`` is the build pipeline the candidate used (enables the compiler-family check)."""
    flow = flow or flowmod.analyse(target, vram)
    causes = {c.id: c for c in target_causes(flow, neighbours)}
    bag = _Bag()
    bag.items = causes
    fingerprint = compiler_family(flow)
    used = pipeline_family(pipeline)
    # IDO's own output uses $at (its assembler expands the compiler's compare-and-branch and
    # indexed-address pseudo-ops), so $at differences there are ordinary code differences
    gcc_like = used == "gcc" or (used is None and fingerprint["family"] == "gcc")
    if used and fingerprint["family"] and fingerprint["family"] != used and fingerprint["confidence"] >= 0.6:
        bag.add("compiler.family", 0.6 + 0.35 * fingerprint["confidence"],
                f"retail encodes register copies as {fingerprint['moves']} and unconditional jumps as "
                f"{fingerprint['unconditional']}: {fingerprint['family'].upper()} output, but the candidate was "
                f"built with a {used.upper()} pipeline",
                "rebuild with a pipeline of the detected family (sweep its versions and -O/-g levels)")
    if candidate is None:
        blocked = any(c.confidence >= 0.7 and (c.id.startswith("boundary.") or c.id in
                      ("abi.nonstandard", "assembly.handwritten")) for c in causes.values())
        bag.add("source.missing", 0.4 if blocked else 0.99, "no candidate has been built",
                "resolve the boundary/ABI finding first" if blocked else "draft with m2c, then iterate")
        return bag.ranked()
    if candidate.rstrip(b"\0") == target.rstrip(b"\0") and len(candidate) >= len(target):
        return []
    from . import walls                                   # late: walls imports nothing from here
    wall = walls.classify(target, candidate)
    if wall["wall"] == "call-set":
        bag.add("call.set", 0.95, "; ".join(wall["evidence"][:3]),
                "complete the body or the inline helper that makes the missing call (or remove the extra one) "
                "before working on registers or instruction order")
    elif wall["wall"] == "control-flow":
        bag.add("control.shape", 0.9, "; ".join(wall["evidence"][:3]),
                "find the source construct that adds or removes the branch or return before working on registers")
    left, right = flow.insns, flowmod.decode(candidate, vram)
    matcher = difflib.SequenceMatcher(None, [i.op for i in left], [i.op for i in right], autojunk=False)
    mapping: Counter = Counter()
    register_rows, structural = [], 0
    retail_only, built_only = [], []
    webs_t, webs_c = _webs(left), _webs(right)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            pairs = min(i2 - i1, j2 - j1)
            for k in range(pairs):
                a, b = left[i1 + k], right[j1 + k]
                if (a.op, b.op) in _LOAD_SIGN or (b.op, a.op) in _LOAD_SIGN:
                    bag.add("type.signedness", 0.9, f"0x{a.vram:08X}: retail `{a.text}`, candidate `{b.text}`",
                            "flip the signedness of the value or field read at this site")
                elif a.op in _WIDTH and b.op in _WIDTH and _WIDTH[a.op] != _WIDTH[b.op]:
                    bag.add("type.width", 0.9, f"0x{a.vram:08X}: retail `{a.text}`, candidate `{b.text}`",
                            f"the field or variable is {_WIDTH[a.op] * 8} bits wide")
                elif a.op in _INVERSE and _INVERSE[a.op] == b.op:
                    bag.add("control.branch_form", 0.8, f"0x{a.vram:08X}: retail `{a.op}`, candidate `{b.op}`",
                            "invert the condition and swap the arms, or use an early exit")
                elif (a.op in _LIKELY) != (b.op in _LIKELY) and a.kind == "branch" and b.kind == "branch":
                    bag.add("control.branch_likely", 0.8, f"0x{a.vram:08X}: retail `{a.op}`, candidate `{b.op}`",
                            "branch-likely copies the first instruction of the branch target into the slot: "
                            "change which arm is the fall-through")
                elif a.op in _MOVES and b.op in _MOVES and a.op != b.op and 0 in ((a.word >> 16) & 31, (b.word >> 16) & 31):
                    bag.add("assembler.variant", 0.3 if "compiler.family" in bag.items else 0.85, f"0x{a.vram:08X}: register copy encoded as `{a.op}` in retail, "
                            f"`{b.op}` in the candidate", "the assembler encodes `move`: select the pipeline whose "
                            "assembler uses this encoding")
                else:
                    structural += 1
            retail_only += left[i1 + pairs:i2]
            built_only += right[j1 + pairs:j2]
            continue
        for k in range(i2 - i1):
            a, b = left[i1 + k], right[j1 + k]
            if a.word == b.word:
                continue
            same_regs = (a.word & 0xFC000000) == (b.word & 0xFC000000) and a.uses == b.uses and a.defs == b.defs
            if a.kind == "call" and b.kind == "call" and a.target != b.target:
                bag.add("relocation.target", 0.9, f"0x{a.vram:08X}: retail calls 0x{(a.target or 0):08X}, candidate "
                        f"0x{(b.target or 0):08X}", "check the callee symbol and its address")
            elif a.kind in ("branch", "jump") and same_regs:
                if len(target) == len(candidate.rstrip(b"\0")) or len(target) == len(candidate):
                    bag.add("control.branch_form", 0.5, f"0x{a.vram:08X}: branch target differs",
                            "block order differs: reorder the arms or the exits")
            elif same_regs and a.op == b.op:
                imm_a, imm_b = a.word & 0xFFFF, b.word & 0xFFFF
                if a.op == "addiu" and (a.word >> 21) & 31 == 29 and (a.word >> 16) & 31 == 29:
                    delta = (0x10000 - imm_a) - (0x10000 - imm_b) if imm_a & 0x8000 else 0
                    if delta:
                        bag.add("frame.size", 0.85, f"retail frame is {abs(delta)} bytes "
                                f"{'larger' if delta > 0 else 'smaller'} than the candidate's",
                                "a local, a spilled value or an outgoing-argument slot is missing"
                                if delta > 0 else "remove a temporary or an address-taken local")
                elif (a.uses | b.uses) & {29} and a.op in _MEMORY:
                    bag.add("frame.size", 0.6, f"0x{a.vram:08X}: stack slot offset differs", "")
                elif a.op in ("lui",) or a.op in _MEMORY or a.op in ("addiu", "ori"):
                    bag.add("global.declaration", 0.55, f"0x{a.vram:08X}: retail `{a.text}`, candidate `{b.text}`",
                            "a constant, field offset or symbol address differs at this site")
            elif gcc_like and a.op == b.op and (AT in (a.uses | a.defs)) != (AT in (b.uses | b.defs)):
                side = "retail" if AT in (a.uses | a.defs) else "the candidate"
                bag.add("assembler.variant", 0.8, f"0x{a.vram:08X}: {side} uses $at (`{a.text}` vs `{b.text}`)",
                        "$at is never allocated by the compiler: this sequence is an assembler macro expansion "
                        "(indexed global access, large constant, division check); compare assembler pipelines "
                        "or write the access so the compiler emits the macro form")
            elif a.op == b.op:
                register_rows.append((a, b))
                for x, y in zip(sorted(a.uses | a.defs), sorted(b.uses | b.defs)):
                    if x != y:
                        mapping[(y, x)] += 1
    # --- delay slots and volatile
    def slot_pairs(insns):
        return {i.index: insns[i.index + 1] for i in insns if i.control and i.index + 1 < len(insns)}
    slots_t = slot_pairs(left)
    for index, slot in slots_t.items():
        branch = left[index]
        if slot.word == 0 and index > 0 and left[index - 1].op in _MEMORY:
            before = left[index - 1]
            if any(c.op == before.op and (c.word & 0xFFFF) == (before.word & 0xFFFF) and c.index > 0
                   and right[c.index - 1].control for c in right if c.index < len(right)):
                kind = "store" if before.op in _STORES else "load"
                bag.add("type.volatile", 0.85, f"retail keeps `{before.text}` in front of `{branch.op}` at "
                        f"0x{branch.vram:08X} with a nop in the slot; the candidate moved the {kind} into the slot",
                        "GCC never puts a volatile access in a delay slot: qualify the pointer or global it goes "
                        "through as volatile")
                bag.add("schedule.delay_slot", 0.7, f"delay slot of 0x{branch.vram:08X} differs", "")
    extra_loads = [i for i in retail_only if i.op in _MEMORY and i.op not in _STORES]
    if extra_loads and not [i for i in built_only if i.op in _MEMORY]:
        seen = Counter((i.op, i.word & 0x03E0FFFF) for i in left if i.op in _MEMORY)
        repeated = [i for i in extra_loads if seen[(i.op, i.word & 0x03E0FFFF)] > 1]
        if repeated:
            bag.add("type.volatile", 0.7, f"retail reloads `{repeated[0].text}` where the candidate reuses the value",
                    "a volatile object (or one that may alias a store in between) is reread: try the volatile "
                    "and globals families")
    # --- inserted / deleted rows
    def has_pair(rows, first, second):
        return any(a.op == first and b.op == second and (a.word >> 6) & 31 == (b.word >> 6) & 31
                   and (a.word >> 6) & 31 in (16, 24) for a, b in zip(rows, rows[1:]))
    for rows, side in ((retail_only, "retail"), (built_only, "candidate")):
        if has_pair(rows, "sll", "sra") or has_pair(rows, "sll", "srl") or \
                any(r.op == "andi" and (r.word & 0xFFFF) in (0xFF, 0xFFFF) for r in rows):
            bag.add("type.width", 0.8, f"{side} has an extension (sll/sra or andi) the other side lacks",
                    "a narrow type (s8/u8/s16/u16) is " + ("missing" if side == "retail" else "superfluous"))
        at_rows = [r for r in rows if AT in (r.uses | r.defs)]
        if at_rows and gcc_like:
            bag.add("assembler.variant", 0.7, f"{side}-only instruction uses $at: `{at_rows[0].text}`",
                    "$at is never allocated by the compiler: this sequence is an assembler macro expansion "
                    "(indexed global access, large constant, division check); compare assembler pipelines")
    only_nops = retail_only + built_only
    if only_nops and all(i.word == 0 for i in only_nops) and not structural and not register_rows:
        bag.add("schedule.load_hazard", 0.8, f"{len(only_nops)} nop(s) differ and nothing else",
                "ISA level or assembler reordering differs (-mips1 load delays, hi/lo hazards)")
    real_structural = structural + sum(1 for i in retail_only + built_only if i.word)
    if len(candidate.rstrip(b"\0")) != len(target.rstrip(b"\0")) and not real_structural and not bag.items:
        bag.add("boundary.false_split", 0.5, "sizes differ but the instructions agree: check trailing padding", "")
    # --- registers
    if register_rows:
        forward: dict[int, set[int]] = {}
        for (built, retail), _n in mapping.items():
            forward.setdefault(built, set()).add(retail)
        consistent = all(len(v) == 1 for v in forward.values()) and \
            len({next(iter(v)) for v in forward.values()}) == len(forward)
        first_t, first_c = register_rows[0]
        diff_t = sorted((first_t.uses | first_t.defs) - (first_c.uses | first_c.defs))
        diff_c = sorted((first_c.uses | first_c.defs) - (first_t.uses | first_t.defs))
        lifetime_notes = []
        for reg_t, reg_c in zip(diff_t, diff_c):
            web_t, web_c = _web_at(webs_t, reg_t, first_t.index), _web_at(webs_c, reg_c, first_c.index)
            if (reg_t in SAVED) != (reg_c in SAVED):
                if reg_t in SAVED:
                    born = left[web_t[0]].vram if web_t else first_t.vram
                    lifetime_notes.append(
                        f"retail keeps a value in {reg_name(reg_t)} (set at 0x{born:08X}, live across a call); the "
                        f"candidate has it in {reg_name(reg_c)}, not live across a call: initialise it before the "
                        "call, or keep using it after one")
                else:
                    lifetime_notes.append(
                        f"the candidate keeps a value in {reg_name(reg_c)} across a call where retail uses "
                        f"{reg_name(reg_t)}: retail computes it after the call or recomputes it")
            elif web_t and web_c:
                lifetime_notes.append(
                    f"{reg_name(reg_t)} in retail ({web_t[3]} use(s), {web_t[1] - web_t[0] + 1} instructions) vs "
                    f"{reg_name(reg_c)} in the candidate ({web_c[3]} use(s), {web_c[1] - web_c[0] + 1} instructions)")
        if any("across a call" in n for n in lifetime_notes):
            bag.add("register.lifetime", 0.9, lifetime_notes[0],
                    "move the initialisation across the call (lifetime family)")
        elif consistent and not real_structural:
            pairs = ", ".join(f"{reg_name(b)}->{reg_name(next(iter(t)))}" for b, t in sorted(forward.items())[:6])
            advice = permutation_advice(forward, webs_c)
            bag.add("register.permutation", 0.85, f"consistent exchange {pairs}; first at 0x{first_t.vram:08X}"
                    + (f"; {lifetime_notes[0]}" if lifetime_notes else "") + (f"; {advice}" if advice else ""),
                    "GCC 2.7.2 ranks values by floor_log2(refs)*refs/live_length (ties to the lower pseudo "
                    "number, i.e. the one created first; refs count once per enclosing loop level, and a "
                    "goto-built loop is no loop: write it as while (1) { ... } or the reverse) and gives the "
                    "lowest free register first; IDO by first use: change which value is created or used first")
        else:
            bag.add("register.pressure", 0.6 if not real_structural else 0.35,
                    f"{len(register_rows)} instruction(s) differ only in registers, first at 0x{first_t.vram:08X}"
                    + (f"; {lifetime_notes[0]}" if lifetime_notes else ""),
                    "fix structural differences first; then add or remove a temporary near the first divergence")
    explained = {"type.signedness", "type.width", "control.branch_form", "control.branch_likely",
                 "assembler.variant", "type.volatile", "schedule.delay_slot", "frame.size"}
    if real_structural and not (set(bag.items) & explained):
        bag.add("semantic.unknown", 0.6, f"{real_structural} instruction(s) inserted, deleted or replaced "
                f"(first divergence near 0x{(retail_only[0].vram if retail_only else vram):08X})",
                "compare control flow and expressions around the first divergence")
    elif real_structural > 4:
        bag.add("semantic.unknown", 0.4, f"{real_structural} further structural difference(s)", "")
    family = bag.items.get("compiler.family")
    if family is not None:                                 # another compiler entirely: its differences explain the rest
        for early in ("call.set", "control.shape"):
            if early in bag.items:
                bag.items[early].confidence = min(bag.items[early].confidence, family.confidence - 0.05)
    # the routing order: a cause of a later class is a consequence while an earlier class is open
    for held in walls.gate(wall["wall"], list(bag.items)):
        item = bag.items[held]
        item.confidence = min(item.confidence, 0.3)
        note = f"held back: the {wall['wall']} difference comes first"
        if note not in item.evidence:
            item.evidence.append(note)
    return bag.ranked()


def family_order(causes: list[Cause], available: tuple[str, ...]) -> list[str]:
    """Iterate families ordered by the causes that justify them, the rest after."""
    first: list[str] = []
    for cause in causes:
        for family in cause.families:
            if family in available and family not in first:
                first.append(family)
    return first + [f for f in available if f not in first]
