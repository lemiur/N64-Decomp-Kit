"""Compiler-pass probes and causal attribution of a residual (round 12 addendum, items D and Q).

A near miss is a difference between two instruction streams, and most ways two streams can
differ have one owner among the stages that made them: the scheduler moves instructions, the
register allocator renames them, reload adds stack traffic, the delay-slot filler puts an
instruction behind a branch, the assembler expands macros through ``$at``.  This module names
those owners and reads a residual for them:

* ``PASS_PAIRS``: flag changes that each switch one pass of one compiler family, applied to the
  same source (``PROBE_SOURCE``).  The difference between the two builds is that pass's
  signature on that recipe; ``bench/passprobe/run.py`` builds the pairs on every installed
  recipe and ``learn`` turns the builds into a table of measured signatures;
* ``analyse(target, candidate)``: the dimensions in which two word lists differ (order,
  registers, delay slots, branch-likely, stack traffic, frame size, frame pointer, ``$at`` and
  move encoding, nops, multiplies, loop shape, repeated loads, inserted instructions);
* ``attribute(target, candidate)``: ranked components, each with the evidence for it, the
  evidence against it, the dimensions it explains, whether the residual rules it out, a
  confidence and one next experiment.  The kit's existing diagnoses (``diff.diff`` with
  ``residual.classify``, and ``causes.diagnose``) run as well and their findings are merged into
  the components they name: what they classify is not classified a second time here.

Words are read field by field (``decode``), not through a disassembler, so the names do not
depend on a library's pseudo-instruction settings.  Reporting only: nothing here changes a
build, a score, a verdict or credit.
"""
from __future__ import annotations

import difflib
import struct
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

GPR, FPR = "r", "f"
ZERO, AT, V0, SP, FP, RA = 0, 1, 2, 29, 30, 31
CALLEE_SAVED = frozenset([*range(16, 24), FP, RA])
REGISTER_NAMES = ("zero", "at", "v0", "v1", "a0", "a1", "a2", "a3", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7",
                  "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "t8", "t9", "k0", "k1", "gp", "sp", "fp", "ra")

SPECIAL = {0: "sll", 2: "srl", 3: "sra", 4: "sllv", 6: "srlv", 7: "srav", 8: "jr", 9: "jalr", 12: "syscall",
           13: "break", 15: "sync", 16: "mfhi", 17: "mthi", 18: "mflo", 19: "mtlo", 20: "dsllv", 22: "dsrlv",
           23: "dsrav", 24: "mult", 25: "multu", 26: "div", 27: "divu", 28: "dmult", 29: "dmultu", 30: "ddiv",
           31: "ddivu", 32: "add", 33: "addu", 34: "sub", 35: "subu", 36: "and", 37: "or", 38: "xor", 39: "nor",
           42: "slt", 43: "sltu", 44: "dadd", 45: "daddu", 46: "dsub", 47: "dsubu", 48: "tge", 49: "tgeu",
           50: "tlt", 51: "tltu", 52: "teq", 54: "tne", 56: "dsll", 58: "dsrl", 59: "dsra", 60: "dsll32",
           62: "dsrl32", 63: "dsra32"}
REGIMM = {0: "bltz", 1: "bgez", 2: "bltzl", 3: "bgezl", 16: "bltzal", 17: "bgezal", 18: "bltzall", 19: "bgezall"}
OPCODES = {2: "j", 3: "jal", 4: "beq", 5: "bne", 6: "blez", 7: "bgtz", 8: "addi", 9: "addiu", 10: "slti",
           11: "sltiu", 12: "andi", 13: "ori", 14: "xori", 15: "lui", 20: "beql", 21: "bnel", 22: "blezl",
           23: "bgtzl", 24: "daddi", 25: "daddiu", 26: "ldl", 27: "ldr", 32: "lb", 33: "lh", 34: "lwl", 35: "lw",
           36: "lbu", 37: "lhu", 38: "lwr", 39: "lwu", 40: "sb", 41: "sh", 42: "swl", 43: "sw", 44: "sdl",
           45: "sdr", 46: "swr", 47: "cache", 48: "ll", 49: "lwc1", 52: "lld", 53: "ldc1", 55: "ld", 56: "sc",
           57: "swc1", 60: "scd", 61: "sdc1", 63: "sd"}
FORMATS = {16: "s", 17: "d", 20: "w", 21: "l"}
COP1 = {0: "add", 1: "sub", 2: "mul", 3: "div", 4: "sqrt", 5: "abs", 6: "mov", 7: "neg", 8: "round.l",
        9: "trunc.l", 10: "ceil.l", 11: "floor.l", 12: "round.w", 13: "trunc.w", 14: "ceil.w", 15: "floor.w",
        32: "cvt.s", 33: "cvt.d", 36: "cvt.w", 37: "cvt.l"}
CONDITIONS = ("f", "un", "eq", "ueq", "olt", "ult", "ole", "ule", "sf", "ngle", "seq", "ngl", "lt", "nge", "le",
              "ngt")
SHIFTS = {"sll", "srl", "sra", "dsll", "dsrl", "dsra", "dsll32", "dsrl32", "dsra32"}
LOADS = {"lb", "lh", "lwl", "lw", "lbu", "lhu", "lwr", "lwu", "ld", "ldl", "ldr", "ll", "lld", "lwc1", "ldc1"}
STORES = {"sb", "sh", "swl", "sw", "swr", "sd", "sdl", "sdr", "sc", "scd", "swc1", "sdc1"}
FLOAT_MEMORY = {"lwc1", "ldc1", "swc1", "sdc1"}
PARTIAL_LOADS = {"lwl", "lwr", "ldl", "ldr"}
LIKELY = {"beql", "bnel", "blezl", "bgtzl", "bltzl", "bgezl", "bltzall", "bgezall", "bc1fl", "bc1tl"}
BRANCHES = {"beq", "bne", "blez", "bgtz", "bltz", "bgez", "bltzal", "bgezal", "bc1f", "bc1t"} | LIKELY
JUMPS = {"j", "jal", "jr", "jalr"}
MULTIPLIES = {"mult", "multu", "dmult", "dmultu"}
WIDE = {"dsllv", "dsrlv", "dsrav", "dmult", "dmultu", "ddiv", "ddivu", "dadd", "daddu", "dsub", "dsubu", "dsll",
        "dsrl", "dsra", "dsll32", "dsrl32", "dsra32", "daddi", "daddiu", "ld", "sd"}


@dataclass(frozen=True)
class Insn:
    word: int
    name: str
    operands: tuple = ()          # (class, number) of every register field, in field order
    reads: frozenset = frozenset()
    writes: frozenset = frozenset()
    base: int | None = None       # base register of a memory access
    imm: int = 0                  # immediate, sign-extended
    uimm: int = 0                 # immediate, zero-extended
    shamt: int = 0

    @property
    def control(self) -> bool:
        return self.name in BRANCHES or self.name in JUMPS

    @property
    def likely(self) -> bool:
        return self.name in LIKELY

    @property
    def memory(self) -> bool:
        return self.name in LOADS or self.name in STORES

    def names(self, number: int, kind: str = GPR) -> bool:
        return (kind, number) in self.operands


def decode(word: int) -> Insn:
    """One big-endian MIPS III word, read field by field."""
    word &= 0xFFFFFFFF
    if word == 0:
        return Insn(0, "nop")
    op, rs, rt = word >> 26, (word >> 21) & 31, (word >> 16) & 31
    rd, sa, fn = (word >> 11) & 31, (word >> 6) & 31, word & 63
    uimm = word & 0xFFFF
    imm = uimm - 0x10000 if uimm & 0x8000 else uimm

    def g(number):
        return (GPR, number)

    def f(number):
        return (FPR, number)

    def make(name, operands=(), reads=(), writes=(), **extra):
        return Insn(word, name, tuple(operands), frozenset(reads), frozenset(writes), imm=imm, uimm=uimm, **extra)

    if op == 0:
        name = SPECIAL.get(fn, f"special{fn}")
        if name in SHIFTS:
            return make(name, (g(rt), g(rd)), (g(rt),), (g(rd),), shamt=sa)
        if name == "jr":
            return make(name, (g(rs),), (g(rs),))
        if name == "jalr":
            return make(name, (g(rs), g(rd)), (g(rs),), (g(rd),))
        if name in ("mfhi", "mflo"):
            return make(name, (g(rd),), (), (g(rd),))
        if name in ("mthi", "mtlo"):
            return make(name, (g(rs),), (g(rs),))
        if name in ("syscall", "break", "sync") or name.startswith("special"):
            return make(name)
        if name in MULTIPLIES or name in ("div", "divu", "ddiv", "ddivu") or name.startswith("t"):
            return make(name, (g(rs), g(rt)), (g(rs), g(rt)))
        return make(name, (g(rs), g(rt), g(rd)), (g(rs), g(rt)), (g(rd),))
    if op == 1:
        name = REGIMM.get(rt, f"regimm{rt}")
        return make(name, (g(rs),), (g(rs),), (g(RA),) if name.endswith(("al", "all")) else ())
    if op in (2, 3):
        return make("jal" if op == 3 else "j", (), (), (g(RA),) if op == 3 else ())
    if op == 17:
        if rs == 8:
            return make(("bc1f", "bc1t", "bc1fl", "bc1tl")[rt & 3])
        if rs in (0, 1):
            return make(("mfc1", "dmfc1")[rs], (g(rt), f(rd)), (f(rd),), (g(rt),))
        if rs in (4, 5):
            return make(("mtc1", "dmtc1")[rs - 4], (g(rt), f(rd)), (g(rt),), (f(rd),))
        if rs == 2:
            return make("cfc1", (g(rt),), (), (g(rt),))
        if rs == 6:
            return make("ctc1", (g(rt),), (g(rt),))
        fmt = FORMATS.get(rs)
        if fmt is None:
            return make(f"cop1.{rs}")
        if fn >= 48:
            return make(f"c.{CONDITIONS[fn - 48]}.{fmt}", (f(rt), f(rd)), (f(rt), f(rd)))
        base = COP1.get(fn)
        if base is None:
            return make(f"cop1.{rs}.{fn}")
        if fn <= 3:                                           # add, sub, mul, div: fd = fs op ft
            return make(f"{base}.{fmt}", (f(rt), f(rd), f(sa)), (f(rt), f(rd)), (f(sa),))
        return make(f"{base}.{fmt}", (f(rd), f(sa)), (f(rd),), (f(sa),))
    name = OPCODES.get(op, f"op{op}")
    if name in ("beq", "bne", "beql", "bnel"):
        return make(name, (g(rs), g(rt)), (g(rs), g(rt)))
    if name in ("blez", "bgtz", "blezl", "bgtzl"):
        return make(name, (g(rs),), (g(rs),))
    if name == "lui":
        return make(name, (g(rt),), (), (g(rt),))
    if 8 <= op <= 14 or op in (24, 25):
        return make(name, (g(rs), g(rt)), (g(rs),), (g(rt),))
    if name in FLOAT_MEMORY:
        if name in LOADS:
            return make(name, (g(rs), f(rt)), (g(rs),), (f(rt),), base=rs)
        return make(name, (g(rs), f(rt)), (g(rs), f(rt)), (), base=rs)
    if name in LOADS:
        reads = (g(rs), g(rt)) if name in PARTIAL_LOADS else (g(rs),)
        return make(name, (g(rs), g(rt)), reads, (g(rt),), base=rs)
    if name in STORES:
        return make(name, (g(rs), g(rt)), (g(rs), g(rt)), (g(rt),) if name in ("sc", "scd") else (), base=rs)
    if name == "cache":
        return make(name, (g(rs),), (g(rs),), base=rs)
    return make(name)


def skeleton(word: int) -> int:
    """The word with its register fields cleared (opcode, function, immediates and shift amounts
    kept): two words with the same skeleton differ only in which registers they name."""
    word &= 0xFFFFFFFF
    op = word >> 26
    if word == 0 or op in (2, 3):
        return word
    if op == 0:
        return word & ~0x03FFF800
    if op in (1, 6, 7, 22, 23):
        return word & ~0x03E00000
    if op == 17:
        rs = (word >> 21) & 31
        if rs == 8:
            return word
        if rs in (0, 1, 2, 4, 5, 6):
            return word & ~0x001FF800
        return word & ~0x001FFFC0
    if op == 15:
        return word & ~0x001F0000
    return word & ~0x03FF0000


def register_name(register: tuple) -> str:
    kind, number = register
    return f"$f{number}" if kind == FPR else "$" + REGISTER_NAMES[number]


# ---------------------------------------------------------------------------------------------
# Objects: functions, calls and symbol placement of one relocatable (shared with dialect.py)

SPECIAL_INDEX = {0: "UNDEF", 0xFF00: "ACOMMON", 0xFF01: "TEXT", 0xFF02: "DATA", 0xFF03: "SCOMMON",
                 0xFF04: "SUNDEFINED", 0xFFF1: "ABS", 0xFFF2: "COMMON"}
R_MIPS_26 = 4


class ObjectView:
    """What a probe reads from one object (``n64dk.elf.Elf``): each function's words, the callees
    of its ``jal`` relocations and where a symbol was placed (``COMMON``, ``.bss``, ...)."""

    def __init__(self, elf):
        self.elf = elf
        self.ranges: dict[str, tuple[int, int, int, bool]] = {}     # name -> (section, start, end, sized)
        texts = {s.index for s in elf.sections if s.name == ".text" or s.name.startswith(".text.")}
        found = [sym for sym in elf.symbols if sym.section in texts and sym.name and sym.type in (0, 2)
                 and not sym.name.startswith(("$", ".")) and not sym.name.endswith(".")]
        starts: dict[int, list[int]] = {}
        for sym in found:
            starts.setdefault(sym.section, []).append(sym.value)
        for sym in found:
            size = len(elf.sections[sym.section].data)
            later = [value for value in starts[sym.section] if value > sym.value]
            end = sym.value + sym.size if sym.size else (min(later) if later else size)
            if sym.name not in self.ranges:
                self.ranges[sym.name] = (sym.section, sym.value, min(end, size), bool(sym.size))

    def words(self, name: str) -> list[int] | None:
        found = self.ranges.get(name)
        if found is None:
            return None
        index, start, end, sized = found
        data = self.elf.sections[index].data
        words = [struct.unpack_from(">I", data, offset)[0] for offset in range(start, end - (end - start) % 4, 4)]
        if not sized:                                         # alignment padding is not the function
            while words and words[-1] == 0:
                words.pop()
            if words and decode(words[-1]).control:
                words.append(0)                               # the delay slot the padding swallowed
        return words

    def insns(self, name: str) -> list[Insn] | None:
        words = self.words(name)
        return None if words is None else [decode(word) for word in words]

    def functions(self) -> dict[str, list[int]]:
        return {name: self.words(name) or [] for name in self.ranges}

    def calls(self, name: str) -> list[str]:
        """Callees of the function's ``jal`` instructions, by relocation (a local callee reached
        through a section-relative relocation is named by the function at its target)."""
        found = self.ranges.get(name)
        if found is None:
            return []
        index, start, end, _sized = found
        section = self.elf.sections[index]
        out = []
        for offset, kind, symbol, _addend in self.elf.relocations(section):
            if kind != R_MIPS_26 or not start <= offset < end:
                continue
            word = struct.unpack_from(">I", section.data, offset)[0]
            if word >> 26 != 3:
                continue
            sym = self.elf.symbols[symbol] if symbol < len(self.elf.symbols) else None
            if sym is not None and sym.name and sym.type != 3:
                out.append(sym.name)
                continue
            target = (word & 0x03FFFFFF) << 2
            out.append(next((other for other, (i, s, _e, _z) in self.ranges.items() if i == index and s == target),
                            f"<local+0x{target:X}>"))
        return out

    def place(self, name: str) -> str | None:
        """Where symbol ``name`` lives: a section name, or COMMON/SCOMMON/ACOMMON/UNDEF/ABS."""
        sym = next((s for s in self.elf.symbols if s.name == name), None)
        if sym is None:
            return None
        if sym.section in SPECIAL_INDEX:
            return SPECIAL_INDEX[sym.section]
        if sym.section < len(self.elf.sections):
            return self.elf.sections[sym.section].name
        return f"index:0x{sym.section:X}"


# ---------------------------------------------------------------------------------------------
# Residual dimensions

@dataclass
class Residual:
    target: list[Insn]
    candidate: list[Insn]
    identical: bool = False
    reordered: bool = False                 # the same multiset of words in another order
    nops_only: bool = False                 # equal once nops are removed
    nop_delta: int = 0                      # candidate nops minus target nops
    moved_same: int = 0                     # words on both sides, unaligned (same registers)
    moved_skeleton: int = 0                 # ... counted by skeleton (registers may differ)
    register_rows: int = 0                  # aligned rows that differ only in registers
    mapping: dict | None = None             # consistent candidate -> target register exchange
    extra_target: list[int] = field(default_factory=list)       # unaligned target rows nothing moved
    extra_candidate: list[int] = field(default_factory=list)
    differing: int = 0
    transfer_pairs: int = 0
    slots: list[dict] = field(default_factory=list)
    likely: tuple = (0, 0)
    likely_pairs: int = 0
    covered: tuple = field(default_factory=lambda: (set(), set()))   # rows of differing slots/likely branches
    loop_rows: tuple = field(default_factory=lambda: (set(), set()))  # rows inside loop bodies
    stack: tuple = field(default_factory=lambda: (Counter(), Counter()))
    frame: tuple = (0, 0)
    fp_frame: tuple = (False, False)
    at: tuple = (0, 0)
    at_signatures: tuple = field(default_factory=lambda: ({}, {}))
    moves: tuple = field(default_factory=lambda: (Counter(), Counter()))
    multiplies: tuple = (0, 0)
    loops: tuple = ((0, 0, 0), (0, 0, 0))   # (backward branches, index scalings, pointer steps)
    reloads: list[dict] = field(default_factory=list)

    @property
    def hazard_nops(self) -> int:
        """Nop difference not explained by a delay slot filled on one side only."""
        slots = sum(1 if s["candidate_slot"] == 0 else -1 for s in self.slots if s["kind"] == "filled-vs-empty")
        return self.nop_delta - slots

    def dimensions(self) -> list[str]:
        if self.identical:
            return []
        values = (_role(self.stack[0], "value"), _role(self.stack[1], "value"))
        moves_t, moves_c = self.moves
        checks = (("order", self.moved_skeleton > 0), ("registers", self.register_rows > 0),
                  ("delay-slots", any(not s["likely_changed"] for s in self.slots)),
                  ("branch-likely", self.likely[0] != self.likely[1] or self.likely_pairs > 0),
                  ("stack", values[0] != values[1]), ("frame", self.frame[0] != self.frame[1]),
                  ("frame-pointer", self.fp_frame[0] != self.fp_frame[1]),
                  ("$at", self.at[0] != self.at[1] or self.at_signatures[0] != self.at_signatures[1]),
                  ("move-encoding", bool(moves_t) and bool(moves_c) and set(moves_t) != set(moves_c)),
                  ("nops", self.hazard_nops != 0), ("multiply", self.multiplies[0] != self.multiplies[1]),
                  ("loops", self.loops[0] != self.loops[1]), ("loads", bool(self.reloads)),
                  ("size", len(self.target) != len(self.candidate)),
                  ("inserted", bool(self.extra_target or self.extra_candidate)))
        return [name for name, on in checks if on]


def _role(counter: Counter, role: str) -> int:
    return sum(count for (_direction, kind), count in counter.items() if kind == role)


def _stack(insns: list[Insn]) -> Counter:
    out = Counter()
    for insn in insns:
        if insn.memory and insn.base == SP and len(insn.operands) > 1:
            kind, number = insn.operands[1]
            role = "save" if kind == GPR and number in CALLEE_SAVED else "value"
            out[("load" if insn.name in LOADS else "store", role)] += 1
    return out


def _frame(insns: list[Insn]) -> int:
    for insn in insns:
        if insn.name in ("addiu", "daddiu") and insn.operands == ((GPR, SP), (GPR, SP)) and insn.imm < 0:
            return -insn.imm
    return 0


def _fp_frame(insns: list[Insn]) -> bool:
    return any(insn.name in ("addu", "or", "daddu", "addiu", "daddiu") and (GPR, FP) in insn.writes
               and (GPR, SP) in insn.reads for insn in insns)


def _moves(insns: list[Insn]) -> Counter:
    return Counter(insn.name for insn in insns if insn.name in ("addu", "or", "daddu") and len(insn.operands) == 3
                   and insn.operands[1] == (GPR, ZERO) and insn.operands[0] != (GPR, ZERO)
                   and insn.operands[2] != (GPR, ZERO))


def _loops(insns: list[Insn]) -> tuple[tuple[int, int, int], set[int]]:
    """((backward branches, index scalings, pointer steps), rows inside loop bodies).  A pointer
    step is ``addiu r, r, 2/4/8`` (a step of 1 is taken for a counter)."""
    count = scale = step = 0
    rows: set[int] = set()
    for k, insn in enumerate(insns):
        if insn.name not in BRANCHES or insn.imm >= 0:
            continue
        start = k + 1 + insn.imm
        if start < 0:
            continue
        count += 1
        rows.update(range(start, min(len(insns), k + 2)))
        for body in insns[start:k + 2]:
            if (body.name in ("sll", "dsll") and 1 <= body.shamt <= 3) or body.name in MULTIPLIES:
                scale += 1
            elif (body.name in ("addiu", "daddiu") and len(body.operands) == 2 and body.operands[0] == body.operands[1]
                  and body.imm in (2, 4, 8) and body.operands[0] != (GPR, SP)):
                step += 1
    return (count, scale, step), rows


def _loads(insns: list[Insn]) -> Counter:
    return Counter((insn.name, insn.imm) for insn in insns if insn.name in LOADS and insn.base not in (SP, FP, None))


def _origin(skel: int, insns: list[Insn], branch: int) -> str:
    """Where the instruction that fills a slot on one side sits on the side whose slot is empty."""
    if any(skeleton(insns[n].word) == skel for n in range(max(0, branch - 4), branch)):
        return "before the branch"
    insn = insns[branch]
    if insn.name in BRANCHES:
        target = branch + 1 + insn.imm
        if 0 <= target < len(insns) and skeleton(insns[target].word) == skel:
            return "at the branch target"
    if any(skeleton(insns[n].word) == skel for n in range(branch + 2, min(len(insns), branch + 4))):
        return "after the slot"
    return "elsewhere"


def _bytes(words: list[int]) -> bytes:
    return b"".join(struct.pack(">I", word & 0xFFFFFFFF) for word in words)


def analyse(target: list[int], candidate: list[int]) -> Residual:
    """The dimensions in which ``candidate`` differs from ``target`` (both lists of words)."""
    target = [int(word) & 0xFFFFFFFF for word in target]
    candidate = [int(word) & 0xFFFFFFFF for word in candidate]
    t, c = [decode(word) for word in target], [decode(word) for word in candidate]
    res = Residual(t, c)
    res.identical = target == candidate
    if res.identical:
        return res
    res.reordered = Counter(target) == Counter(candidate)
    res.nop_delta = candidate.count(0) - target.count(0)
    res.nops_only = [w for w in target if w] == [w for w in candidate if w]
    st, sc = [skeleton(word) for word in target], [skeleton(word) for word in candidate]
    matcher = difflib.SequenceMatcher(None, st, sc, autojunk=False)
    forward: dict[tuple, Counter] = {}
    aligned: list[tuple[int, int]] = []
    t_only: list[int] = []
    c_only: list[int] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                aligned.append((i1 + k, j1 + k))
                a, b = t[i1 + k], c[j1 + k]
                if a.word != b.word:
                    res.register_rows += 1
                    for x, y in zip(a.operands, b.operands):
                        forward.setdefault(y, Counter())[x] += 1
            continue
        t_only.extend(range(i1, i2))
        c_only.extend(range(j1, j2))
    if forward and all(len(found) == 1 for found in forward.values()):
        images = [next(iter(found)) for found in forward.values()]
        if len(set(images)) == len(images):
            res.mapping = {y: next(iter(found)) for y, found in forward.items() if next(iter(found)) != y}
    only_t = Counter(target[i] for i in t_only if target[i])
    only_c = Counter(candidate[j] for j in c_only if candidate[j])
    res.moved_same = sum((only_t & only_c).values())
    shared = Counter(st[i] for i in t_only if target[i]) & Counter(sc[j] for j in c_only if candidate[j])
    res.moved_skeleton = sum(shared.values())
    for rows, words, skeletons, out in ((t_only, target, st, res.extra_target),
                                        (c_only, candidate, sc, res.extra_candidate)):
        pool = Counter(shared)
        for index in rows:
            if not words[index]:
                continue
            if pool[skeletons[index]] > 0:
                pool[skeletons[index]] -= 1
            else:
                out.append(index)
    res.differing = len(t_only) + len(c_only) + res.register_rows
    # delay slots and branch-likely
    transfers_t = [k for k, insn in enumerate(t) if insn.control]
    transfers_c = [k for k, insn in enumerate(c) if insn.control]
    if len(transfers_t) == len(transfers_c):
        pairs = list(zip(transfers_t, transfers_c))
    else:
        pairs = [(i, j) for i, j in aligned if t[i].control and c[j].control]
    res.transfer_pairs = len(pairs)
    for i, j in pairs:
        a, b = t[i], c[j]
        if a.likely != b.likely:
            res.likely_pairs += 1
            res.covered[0].add(i)
            res.covered[1].add(j)
        slot_t = target[i + 1] if i + 1 < len(target) else 0
        slot_c = candidate[j + 1] if j + 1 < len(candidate) else 0
        if slot_t == slot_c or (slot_t and slot_c and skeleton(slot_t) == skeleton(slot_c)):
            continue
        res.covered[0].update((i, i + 1))
        res.covered[1].update((j, j + 1))
        kind = "filled-vs-empty" if (slot_t == 0) != (slot_c == 0) else "different-fill"
        origin = ""
        if kind == "filled-vs-empty":
            filled, empty_side, branch = (slot_t, c, j) if slot_t else (slot_c, t, i)
            origin = _origin(skeleton(filled), empty_side, branch)
        res.slots.append({"target_index": i, "candidate_index": j, "branch": a.name, "kind": kind,
                          "target_slot": slot_t, "candidate_slot": slot_c, "origin": origin,
                          "likely_changed": a.likely != b.likely})
    res.likely = (sum(1 for insn in t if insn.likely), sum(1 for insn in c if insn.likely))
    res.stack = (_stack(t), _stack(c))
    res.frame = (_frame(t), _frame(c))
    res.fp_frame = (_fp_frame(t), _fp_frame(c))
    res.at = (sum(1 for insn in t if insn.names(AT)), sum(1 for insn in c if insn.names(AT)))
    from . import macros
    res.at_signatures = (macros.signatures(_bytes(target)), macros.signatures(_bytes(candidate)))
    res.moves = (_moves(t), _moves(c))
    res.multiplies = (sum(1 for insn in t if insn.name in MULTIPLIES), sum(1 for insn in c if insn.name in MULTIPLIES))
    (loops_t, rows_t), (loops_c, rows_c) = _loops(t), _loops(c)
    res.loops, res.loop_rows = (loops_t, loops_c), (rows_t, rows_c)
    loads_t, loads_c = _loads(t), _loads(c)
    for key in sorted(set(loads_t) | set(loads_c)):
        if loads_t[key] != loads_c[key] and min(loads_t[key], loads_c[key]) >= 1:
            res.reloads.append({"load": key[0], "offset": key[1], "target": loads_t[key], "candidate": loads_c[key]})
    return res


# ---------------------------------------------------------------------------------------------
# Components, the pass-signature table and the flag pairs that isolate each pass

COMPONENTS: dict[str, dict] = {
    "scheduling": {"pass": True, "gcc": "cc1 sched1/sched2 (-fschedule-insns, -fschedule-insns2)",
                   "ido": "uopt/ugen and as1 scheduling", "signature": "the same instructions in another order"},
    "delay-slot": {"pass": True, "gcc": "cc1 reorg (-fdelayed-branch); GNU as in .set reorder",
                   "ido": "as1 delay-slot filling", "signature": "a delay slot holds a nop on one side, an instruction "
                                                                 "on the other"},
    "branch-likely": {"pass": True, "gcc": "cc1 reorg at ISA level 2 and above", "ido": "as1/ugen at -mips2",
                      "signature": "beql/bnel on one side where the other has beq/bne"},
    "register-allocation": {"pass": True, "gcc": "cc1 local-alloc/global-alloc", "ido": "uopt register allocation",
                            "signature": "aligned instructions that differ only in their registers"},
    "spill-reload": {"pass": True, "gcc": "cc1 reload (-fcaller-saves, -fforce-mem)", "ido": "ugen spilling",
                     "signature": "stack loads/stores of values on one side only, or another frame size"},
    "frame-pointer": {"pass": True, "gcc": "-fomit-frame-pointer (on at -O1 and above for MIPS)",
                      "ido": "frame layout (alloca, -g)", "signature": "$sp copied into $fp on one side only"},
    "assembler": {"pass": True, "gcc": "the assembler: macros through $at, move encoding, small data",
                  "ido": "as1 macro expansion", "signature": "$at sequences or move encodings differ"},
    "hazard-nops": {"pass": True, "gcc": "ISA level (-mips1 load delays), assembler hazard nops",
                    "ido": "as1 hazard nops (-mips1, -Wab,-r4300_mul)",
                    "signature": "nops that are not delay-slot fills differ"},
    "constant-multiplication": {"pass": True, "gcc": "expand_mult (a product by a constant as shifts and adds)",
                                "ido": "ugen multiplication by a constant",
                                "signature": "mult/multu on one side, shifts and adds on the other"},
    "strength-reduction": {"pass": True, "gcc": "loop strength reduction (-fstrength-reduce)",
                           "ido": "uopt loop optimisation (-Wo,-loopunroll)",
                           "signature": "inside a loop, index scaling on one side, pointer steps on the other"},
    "cse-reuse": {"pass": True, "gcc": "cse/gcse (-fcse-follow-jumps, -frerun-cse-after-loop) and alias analysis",
                  "ido": "uopt common subexpressions and alias analysis",
                  "signature": "the same memory word loaded more often on one side"},
    "level": {"pass": True, "gcc": "optimisation level (-O0/-O1/-O2)", "ido": "optimisation level (-g/-O1/-O2)",
              "signature": "several pass signatures at once, or stack-resident values on one side"},
    "compiler-family": {"pass": False, "gcc": "another compiler", "ido": "another compiler",
                        "signature": "causes.py compiler.family"},
    "source-types": {"pass": False, "gcc": "source: types", "ido": "source: types",
                     "signature": "load width/signedness or extensions differ (causes.py type.*)"},
    "source-volatile": {"pass": False, "gcc": "source: volatile", "ido": "source: volatile",
                        "signature": "causes.py type.volatile"},
    "structural": {"pass": False, "gcc": "source: expressions or control flow", "ido": "source: expressions or "
                   "control flow", "signature": "inserted or deleted instructions no pass signature explains"},
}

NEXT: dict[str, dict[str, str]] = {
    "scheduling": {"gcc": "rebuild with -fno-schedule-insns and, separately, with -fno-schedule-insns2: the switch "
                          "that moves the code toward retail names the pass; otherwise reorder independent statements",
                   "ido": "IDO schedules in uopt/as1 with no switch: compare -O1 and -O2, then reorder independent "
                          "statements"},
    "delay-slot": {"gcc": "toggle -fno-delayed-branch (reorg fills slots at -O1 and above); a volatile access is not "
                          "put in a slot",
                   "ido": "as1 fills slots: compare -O1/-O2 and -mips1/-mips2, or move the statement before the branch"},
    "branch-likely": {"gcc": "branch-likely comes from reorg at -mips2 and above: rebuild at -mips1 or with "
                             "-fno-delayed-branch (GCC 2.x has no -mno-branch-likely); in the source, swap the arms",
                      "ido": "rebuild at -mips1 against -mips2; in the source, swap the arms"},
    "register-allocation": {"any": "no flag isolates allocation: change declaration order, temporaries or lifetimes "
                                   "(iterate families callargs, ast-temp, ast-decl-order, lifetime)"},
    "spill-reload": {"gcc": "shorten the lifetime of the spilled value or reduce values live across calls; toggle "
                            "-fno-caller-saves and -fforce-mem",
                     "ido": "shorten the lifetime of the spilled value; compare -O levels (-g keeps values on the "
                            "stack)"},
    "frame-pointer": {"gcc": "toggle -fomit-frame-pointer / -fno-omit-frame-pointer (-O0 keeps the frame pointer)",
                      "ido": "IDO keeps $fp for variable frames and at -g: compare -g and -O"},
    "assembler": {"any": "compare the assembler lanes that share the compiler (kmc+as, kmc+gas, kmc+asn64, "
                         "kmc+atload: components.shared_stages) and the region's macro signatures (macros.explain)"},
    "hazard-nops": {"gcc": "rebuild at the other ISA level (-mips1 load-delay nops against -mips2/-mips3)",
                    "ido": "toggle -Wab,-r4300_mul (nops around float multiplies) and -mips1/-mips2"},
    "constant-multiplication": {"any": "the compiler turns x*K into shifts and adds when its costs say so: compare "
                                       "compiler versions and -O levels; in the source write the product as shifts "
                                       "or as x*K"},
    "strength-reduction": {"gcc": "toggle -fno-strength-reduce; write the loop as a pointer walk or as an index",
                           "ido": "compare -O1/-O2 and -Wo,-loopunroll,0; write the loop as a pointer walk or as an "
                                  "index"},
    "cse-reuse": {"gcc": "toggle -fno-cse-follow-jumps and -fno-rerun-cse-after-loop; check volatile qualifiers and "
                         "the aliasing facts of the dialect matrix",
                  "ido": "check volatile qualifiers and pointer types (aliasing); compare -O levels"},
    "level": {"gcc": "sweep -O0/-O1/-O2", "ido": "sweep -g/-O1/-O2"},
    "compiler-family": {"any": "rebuild with a pipeline of the detected family"},
    "source-types": {"any": "flip the signedness or width of the value (iterate family types)"},
    "source-volatile": {"any": "qualify the object as volatile (iterate family volatile)"},
    "structural": {"any": "a source-level difference: compare expressions and control flow near the first "
                          "divergence (causes.py)"},
}


def next_step(component: str, family: str | None) -> str:
    table = NEXT.get(component, {})
    return (table.get(family or "") or table.get("any")
            or " / ".join(f"{name}: {text}" for name, text in sorted(table.items())))


def lane_family(lane: str | None) -> str | None:
    """Compiler family of a lane or recipe label (``kmc+as -O2 ...``, ``custom:ido60``)."""
    if not lane:
        return None
    name = lane.split()[0]
    if name.startswith(("ido", "custom:ido", "custom:mipspro")):
        return "ido"
    if name.startswith(("kmc", "sn", "custom:gcc", "custom:egcs")):
        return "gcc"
    return None


@dataclass(frozen=True)
class PassPair:
    """One flag change that switches one pass: ``("add", flag)``, ``("toggle", flag)`` or
    ``("replace", (old, ...), new)``."""
    name: str
    component: str
    family: str
    change: tuple
    note: str = ""

    def apply(self, flags) -> list[str] | None:
        """The flags with the change applied, or None when it does not apply to them."""
        flags = list(flags)
        kind = self.change[0]
        if kind == "add":
            return None if self.change[1] in flags else flags + [self.change[1]]
        if kind == "toggle":
            flag = self.change[1]
            return [item for item in flags if item != flag] if flag in flags else flags + [flag]
        if kind == "replace":
            olds, new = self.change[1], self.change[2]
            hit = next((item for item in flags if item in olds), None)
            if hit is None:
                return None
            return [new if item == hit else item for item in flags]
        raise ValueError(f"unknown change {kind!r}")

    def describe(self) -> str:
        if self.change[0] == "replace":
            return f"{'/'.join(self.change[1])} -> {self.change[2]}"
        return f"{self.change[0]} {self.change[1]}"


PASS_PAIRS: tuple[PassPair, ...] = (
    PassPair("level-O1", "level", "gcc", ("replace", ("-O2", "-O3"), "-O1")),
    PassPair("level-O0", "level", "gcc", ("replace", ("-O1", "-O2", "-O3"), "-O0")),
    PassPair("sched1", "scheduling", "gcc", ("add", "-fno-schedule-insns"), "first scheduling pass"),
    PassPair("sched2", "scheduling", "gcc", ("add", "-fno-schedule-insns2"), "second scheduling pass"),
    PassPair("reorg", "delay-slot", "gcc", ("add", "-fno-delayed-branch"),
             "delay-slot filling and branch-likely conversion"),
    PassPair("strength-reduce", "strength-reduction", "gcc", ("add", "-fno-strength-reduce")),
    PassPair("frame-pointer", "frame-pointer", "gcc", ("add", "-fno-omit-frame-pointer")),
    PassPair("cse-follow-jumps", "cse-reuse", "gcc", ("add", "-fno-cse-follow-jumps")),
    PassPair("rerun-cse", "cse-reuse", "gcc", ("add", "-fno-rerun-cse-after-loop")),
    PassPair("gcse", "cse-reuse", "gcc", ("add", "-fno-gcse"), "EGCS only: a refusal elsewhere is the measurement"),
    PassPair("caller-saves", "spill-reload", "gcc", ("add", "-fno-caller-saves")),
    PassPair("force-addr", "cse-reuse", "gcc", ("add", "-fforce-addr"), "a setting a published GCC 2.8.1 title uses"),
    PassPair("isa-mips1", "hazard-nops", "gcc", ("replace", ("-mips2", "-mips3"), "-mips1"),
             "load-delay nops and no branch-likely"),
    PassPair("no-branch-likely", "branch-likely", "gcc", ("add", "-mno-branch-likely"),
             "a GCC 3 option: expected to be refused by 2.x, the refusal is recorded"),
    PassPair("level-O1", "level", "ido", ("replace", ("-O2", "-O3"), "-O1")),
    PassPair("level-g", "level", "ido", ("replace", ("-O1", "-O2", "-O3"), "-g")),
    PassPair("r4300-mul", "hazard-nops", "ido", ("toggle", "-Wab,-r4300_mul"), "as1 nops around float multiplies"),
    PassPair("isa-mips1", "hazard-nops", "ido", ("replace", ("-mips2", "-mips3"), "-mips1")),
    PassPair("loopunroll", "strength-reduction", "ido", ("add", "-Wo,-loopunroll,0"),
             "uopt loop unrolling off; that the driver accepts it is measured"),
)
ASSEMBLER_VARIANTS = ("kmc+gas", "kmc+asn64", "kmc+atload")     # each against kmc+as with the same flags


def pairs_for(family: str | None) -> list[PassPair]:
    return [pair for pair in PASS_PAIRS if pair.family == family]


PROBE_SOURCE = """\
extern int ext(int);
extern int gtab[64];
extern volatile int vreg;

int pp_sum(int *a, int n)
{
    int i, s = 0;
    for (i = 0; i < n; i++)
        s += a[i] * 3;
    return s;
}

int pp_scale(short *dst, short *src, int n)
{
    int i;
    for (i = 0; i < n; i++)
        dst[i] = src[i] * 10;
    return n;
}

int pp_branch(int a, int b, int *p)
{
    if (a > b) {
        *p = a - b;
        return ext(a);
    } else if (a == b) {
        p[1] = 0;
    }
    return b + *p;
}

int pp_live(int a, int b, int c, int d)
{
    int x = ext(a);
    int y = ext(b + x);
    int z = ext(c * y);
    return x + y + z + ext(d + x * y);
}

int pp_frame(int n)
{
    int buf[8];
    int i;
    for (i = 0; i < 8; i++)
        buf[i] = ext(i + n);
    return buf[n & 7] + buf[0];
}

int pp_sched(int *p, int *q)
{
    int a = p[0], b = q[0], c = p[1], d = q[1];
    return a * b + c * d + (a ^ d);
}

int pp_cse(int a, int *p)
{
    int t = p[2];
    if (a)
        t += p[2] * 2;
    else
        t -= p[3];
    return t + p[2] + gtab[a & 63];
}

float pp_fmul(float a, float b, float c)
{
    return a * b + c * a;
}

int pp_mulk(int a)
{
    return a * 12 + a * 7;
}

int pp_vol(int a)
{
    vreg = a;
    return ext(a) + vreg;
}
"""


# ---------------------------------------------------------------------------------------------
# Attribution

MECHANISM_COMPONENT = {"register-allocation": "register-allocation", "reload-spill-frame": "spill-reload",
                       "scheduling-delay-slot": "scheduling", "types-width-signedness-alias": "source-types",
                       "semantics-or-cfg": "structural"}
CAUSE_COMPONENT = {"register.permutation": "register-allocation", "register.pressure": "register-allocation",
                   "register.lifetime": "register-allocation", "schedule.delay_slot": "delay-slot",
                   "schedule.load_hazard": "hazard-nops", "control.branch_likely": "branch-likely",
                   "assembler.variant": "assembler", "type.signedness": "source-types", "type.width": "source-types",
                   "type.volatile": "source-volatile", "frame.size": "spill-reload", "semantic.unknown": "structural",
                   "control.branch_form": "structural", "global.declaration": "structural",
                   "relocation.target": "structural", "compiler.family": "compiler-family"}
CREATED_BY_WRAPPED = {"source-types", "source-volatile", "compiler-family"}


@dataclass
class Finding:
    component: str
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    contra: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)
    ruled_out: bool = False
    next_experiment: str = ""

    def as_dict(self) -> dict:
        return {"component": self.component, "pass": COMPONENTS.get(self.component, {}).get("pass", False),
                "confidence": round(self.confidence, 2), "evidence": self.evidence[:6], "contra": self.contra[:6],
                "dimensions": list(self.dimensions), "ruled_out": self.ruled_out,
                "next_experiment": self.next_experiment}


def _confidence(strength: float, explains: int, differing: int) -> float:
    """A signature's strength, scaled by the share of the residual it accounts for."""
    share = min(1.0, explains / max(1, differing))
    return min(0.97, strength * (0.55 + 0.45 * share))


def _stack_ratio(insns: list[Insn]) -> float:
    return sum(1 for insn in insns if insn.memory and insn.base in (SP, FP)) / max(1, len(insns))


def _nop_after(insns: list[Insn], predicate) -> bool:
    return any(predicate(insns[k]) and insns[k + 1].word == 0 for k in range(len(insns) - 1))


def wrapped(target: list[int], candidate: list[int], vram: int = 0x80000000,
            pipeline: str | None = None) -> list[tuple[str | None, float, str]]:
    """The existing diagnoses of the same pair, as (component, confidence, text): ``diff.diff`` +
    ``residual.classify`` and ``causes.diagnose``.  A diagnosis that fails is reported, not raised."""
    out: list[tuple[str | None, float, str]] = []
    target_bytes, candidate_bytes = _bytes(target), _bytes(candidate)
    try:
        from . import diff as diffmod
        from . import residual as residualmod
        report = diffmod.diff(candidate_bytes, target_bytes, vram)
        for item in residualmod.classify(residualmod.ResidualSignature.from_report(report)):
            out.append((MECHANISM_COMPONENT.get(item.mechanism.value), item.confidence,
                        f"residual.py {item.mechanism.value} ({item.confidence:.2f})"))
    except Exception as error:                     # noqa: BLE001 - evidence, never fatal
        out.append((None, 0.0, f"residual.py unavailable: {type(error).__name__}"))
    try:
        from . import causes as causemod
        for cause in causemod.diagnose(target_bytes, candidate_bytes, vram, pipeline=pipeline):
            text = f"causes.py {cause.id} ({cause.confidence:.2f})" + (f": {cause.evidence[0]}" if cause.evidence else "")
            out.append((CAUSE_COMPONENT.get(cause.id), cause.confidence, text[:240]))
    except Exception as error:                     # noqa: BLE001
        out.append((None, 0.0, f"causes.py unavailable: {type(error).__name__}"))
    return out


def learned_profile(learned: dict | None, lane: str | None, family: str | None) -> dict:
    if not learned:
        return {}
    by_lane = (learned.get("lanes") or {}).get(lane.split()[0]) if lane else None
    return by_lane or (learned.get("families") or {}).get(family or "") or {}


def attribute(target_words, candidate_words, *, vram: int = 0x80000000, family: str | None = None,
              lane: str | None = None, learned: dict | None = None, pipeline: str | None = None,
              wrap: bool = True) -> list[dict]:
    """Ranked components that can explain how ``candidate_words`` differs from ``target_words``:
    ``[{component, pass, confidence, evidence, contra, dimensions, ruled_out, next_experiment}]``,
    components the residual rules out last (confidence 0, the reason in ``contra``).  ``family``
    ("gcc"/"ido") or ``lane`` selects the next experiments; ``learned`` (``learn`` output) weighs
    each pass by what its flag pair did on the probe corpus.  Empty for identical code."""
    target = [int(word) & 0xFFFFFFFF for word in target_words]
    candidate = [int(word) & 0xFFFFFFFF for word in candidate_words]
    if target == candidate:
        return []
    family = family or lane_family(lane or pipeline)
    res = analyse(target, candidate)
    t, c = res.target, res.candidate
    differing = max(1, res.differing)
    findings: dict[str, Finding] = {}

    def finding(component: str) -> Finding:
        if component not in findings:
            findings[component] = Finding(component, next_experiment=next_step(component, family))
        return findings[component]

    def rule_out(component: str, why: str) -> None:
        item = finding(component)
        item.ruled_out, item.confidence = True, 0.0
        item.contra.append(why)

    # scheduling: instructions at other positions
    slot_fills = [s for s in res.slots if s["kind"] == "filled-vs-empty" and s["origin"] == "before the branch"]
    if res.moved_skeleton:
        item = finding("scheduling")
        item.dimensions = ["order"]
        if res.reordered:
            strength = 0.95
            item.evidence.append("the same instruction words in another order (identical multisets)")
        else:
            renamed = res.moved_skeleton - res.moved_same
            strength = 0.75 if not renamed else 0.6
            item.evidence.append(f"{res.moved_skeleton} instruction(s) sit at other positions on the two sides"
                                 + (f" ({renamed} of them with other registers)" if renamed else ""))
        if len(slot_fills) >= res.moved_skeleton:
            strength *= 0.5
            item.contra.append("every moved instruction is a delay-slot fill (see delay-slot)")
        item.confidence = _confidence(strength, 2 * res.moved_skeleton, differing)
    else:
        rule_out("scheduling", "no instruction changes position")

    # delay slots
    plain_slots = [s for s in res.slots if not s["likely_changed"]]
    transfers = any(insn.control for insn in t) or any(insn.control for insn in c)
    if plain_slots:
        item = finding("delay-slot")
        item.dimensions = ["delay-slots"]
        filled = [s for s in plain_slots if s["kind"] == "filled-vs-empty"]
        for s in plain_slots[:3]:
            where = f"the slot of `{s['branch']}` (retail +0x{4 * s['target_index']:X})"
            if s["kind"] == "filled-vs-empty":
                word = s["target_slot"] or s["candidate_slot"]
                side = "retail" if s["target_slot"] else "the candidate"
                item.evidence.append(f"{where} holds `{decode(word).name}` in {side} and a nop on the other side, "
                                     f"where that instruction sits {s['origin']}")
            else:
                item.evidence.append(f"{where} holds `{decode(s['target_slot']).name}` in retail, "
                                     f"`{decode(s['candidate_slot']).name}` in the candidate")
        if any(not s["target_slot"] and decode(s["candidate_slot"]).memory for s in filled):
            item.contra.append("retail keeps a memory access out of a slot the candidate fills: a volatile object in "
                               "the source gives the same picture (source-volatile)")
        item.confidence = _confidence(0.85 if filled else 0.6, 2 * len(plain_slots), differing)
    elif not transfers:
        rule_out("delay-slot", "neither side has a branch or jump")
    elif res.transfer_pairs and not res.slots:
        rule_out("delay-slot", "every paired delay slot holds the same instruction")

    # branch-likely
    likely_t, likely_c = res.likely
    if likely_t != likely_c or res.likely_pairs:
        item = finding("branch-likely")
        item.dimensions = ["branch-likely"]
        item.evidence.append(f"retail has {likely_t} branch-likely instruction(s), the candidate {likely_c}"
                             + (f"; {res.likely_pairs} paired branch(es) differ in being likely"
                                if res.likely_pairs else ""))
        item.confidence = _confidence(0.9 if res.likely_pairs else 0.75,
                                      2 * max(res.likely_pairs, abs(likely_t - likely_c)), differing)
    elif not likely_t:
        rule_out("branch-likely", "neither side uses a branch-likely instruction")
    else:
        rule_out("branch-likely", "both sides use the same branch-likely instructions")

    # register allocation
    saves = (_role(res.stack[0], "save"), _role(res.stack[1], "save"))
    renamed = res.moved_skeleton - res.moved_same
    if res.register_rows or renamed or saves[0] != saves[1]:
        item = finding("register-allocation")
        item.dimensions = ["registers"]
        if res.register_rows and res.mapping is not None:
            pairs = ", ".join(f"{register_name(y)}->{register_name(x)}" for y, x in sorted(res.mapping.items())[:6])
            item.evidence.append(f"{res.register_rows} aligned instruction(s) differ only in registers, by one "
                                 f"consistent exchange (candidate->retail: {pairs})")
            strength = 0.95 if not (res.extra_target or res.extra_candidate or res.moved_skeleton) else 0.8
        elif res.register_rows:
            item.evidence.append(f"{res.register_rows} aligned instruction(s) differ only in registers, without one "
                                 "consistent mapping")
            strength = 0.55
        else:
            if renamed:
                item.evidence.append(f"{renamed} moved instruction(s) also use other registers")
            strength = 0.55
        if saves[0] != saves[1]:
            item.evidence.append(f"retail saves {saves[0]} callee-saved register(s) on the stack, the candidate "
                                 f"{saves[1]}")
            item.dimensions.append("callee-saved")
        if res.register_rows and res.at[0] != res.at[1] and family != "ido":
            strength *= 0.6
            item.contra.append("the exchanged registers include $at, which the compiler does not allocate: an "
                               "assembler difference (see assembler)")
        item.confidence = _confidence(strength, res.register_rows + renamed + abs(saves[0] - saves[1]), differing)
    else:
        rule_out("register-allocation", "no aligned instruction differs only in its registers")

    # spill / reload / frame
    values = (_role(res.stack[0], "value"), _role(res.stack[1], "value"))
    if values[0] != values[1] or res.frame[0] != res.frame[1]:
        item = finding("spill-reload")
        item.dimensions = [name for name, on in (("stack", values[0] != values[1]),
                                                 ("frame", res.frame[0] != res.frame[1])) if on]
        if values[0] != values[1]:
            item.evidence.append(f"retail has {values[0]} stack load(s)/store(s) of values that are not callee-saved "
                                 f"registers, the candidate {values[1]}")
        if res.frame[0] != res.frame[1]:
            item.evidence.append(f"stack frame of {res.frame[0]} bytes in retail, {res.frame[1]} in the candidate")
        stack_rows = (sum(1 for i in res.extra_target if t[i].base == SP)
                      + sum(1 for j in res.extra_candidate if c[j].base == SP))
        item.confidence = _confidence(0.8 if values[0] != values[1] else 0.55,
                                      max(stack_rows, abs(values[0] - values[1]), 1), differing)
    else:
        rule_out("spill-reload", "stack traffic of values and frame size agree")

    # frame pointer
    fp_t, fp_c = res.fp_frame
    if fp_t != fp_c:
        item = finding("frame-pointer")
        item.dimensions = ["frame-pointer"]
        item.evidence.append(("retail" if fp_t else "the candidate") + " copies $sp into $fp (keeps a frame "
                             "pointer); the other side does not")
        fp_rows = (sum(1 for i in res.extra_target if t[i].names(FP))
                   + sum(1 for j in res.extra_candidate if c[j].names(FP)))
        item.confidence = _confidence(0.92, max(2, fp_rows), differing)
    elif fp_t:
        rule_out("frame-pointer", "both sides keep a frame pointer")
    else:
        rule_out("frame-pointer", "neither side sets up $fp as a frame pointer")

    # assembler: $at forms and move encoding
    sig_t, sig_c = res.at_signatures
    at_forms = {key: (sig_t.get(key, 0), sig_c.get(key, 0)) for key in sorted(set(sig_t) | set(sig_c))
                if sig_t.get(key, 0) != sig_c.get(key, 0)}
    moves_t, moves_c = res.moves
    encoding = bool(moves_t) and bool(moves_c) and set(moves_t) != set(moves_c)
    if at_forms or res.at[0] != res.at[1] or encoding:
        item = finding("assembler")
        item.dimensions = (["$at"] if at_forms or res.at[0] != res.at[1] else []) + (["move-encoding"] if encoding
                                                                                      else [])
        if at_forms:
            item.evidence.append("assembler $at forms differ (retail/candidate): "
                                 + ", ".join(f"{key} {a}/{b}" for key, (a, b) in at_forms.items()))
        elif res.at[0] != res.at[1]:
            item.evidence.append(f"retail names $at in {res.at[0]} instruction(s), the candidate in {res.at[1]}")
        if encoding:
            item.evidence.append(f"register copies encoded as {dict(moves_t)} in retail, {dict(moves_c)} in the "
                                 "candidate")
        strength = 0.85 if family != "ido" or encoding else 0.5
        if family == "ido" and not encoding:
            item.contra.append("IDO's own code uses $at (as1 expands compare-and-branch and address forms): a $at "
                               "difference there is often ordinary code")
        at_rows = (sum(1 for i in res.extra_target if t[i].names(AT))
                   + sum(1 for j in res.extra_candidate if c[j].names(AT)))
        explains = at_rows + abs(res.at[0] - res.at[1]) + (2 if encoding else 0)
        item.confidence = _confidence(strength, max(1, explains), differing)
    else:
        rule_out("assembler", "the $at forms and the move encodings agree")

    # hazard nops
    hazard = res.hazard_nops
    if hazard:
        item = finding("hazard-nops")
        item.dimensions = ["nops"]
        item.evidence.append(f"{'the candidate' if hazard > 0 else 'retail'} has {abs(hazard)} more nop(s) outside "
                             "delay slots" + (" and nothing else differs" if res.nops_only else ""))
        side = c if hazard > 0 else t
        if _nop_after(side, lambda insn: insn.name in ("mul.s", "mul.d")):
            item.evidence.append("nops follow float multiplies: IDO's -Wab,-r4300_mul workaround")
        if _nop_after(side, lambda insn: insn.name in LOADS):
            item.evidence.append("nops follow loads: load delay slots (-mips1)")
        item.confidence = _confidence(0.9 if res.nops_only else 0.55, abs(hazard), differing)
    else:
        rule_out("hazard-nops", "outside delay slots both sides have the same number of nops")

    # multiplication by a constant
    shift_add = ("sll", "srl", "sra", "addu", "subu")
    mul_t = sum(1 for i in res.extra_target if t[i].name in MULTIPLIES)
    mul_c = sum(1 for j in res.extra_candidate if c[j].name in MULTIPLIES)
    sh_t = sum(1 for i in res.extra_target if t[i].name in shift_add)
    sh_c = sum(1 for j in res.extra_candidate if c[j].name in shift_add)
    multiply_fired = (mul_t > mul_c and sh_c > sh_t) or (mul_c > mul_t and sh_t > sh_c)
    if multiply_fired:
        item = finding("constant-multiplication")
        item.dimensions = ["multiply"]
        item.evidence.append("retail multiplies with mult/multu where the candidate uses shifts and adds"
                             if mul_t > mul_c else
                             "the candidate multiplies with mult/multu where retail uses shifts and adds")
        item.confidence = _confidence(0.8, 2 * abs(mul_t - mul_c) + abs(sh_t - sh_c), differing)
    elif res.multiplies[0] == res.multiplies[1]:
        rule_out("constant-multiplication", "both sides multiply equally often" if res.multiplies[0]
                 else "neither side multiplies")

    # loops: strength reduction
    (loops_t, scale_t, step_t), (loops_c, scale_c, step_c) = res.loops
    loop_extra = (sum(1 for i in res.extra_target if i in res.loop_rows[0])
                  + sum(1 for j in res.extra_candidate if j in res.loop_rows[1]))
    loop_fired = False
    if not loops_t and not loops_c:
        rule_out("strength-reduction", "no loop (no backward branch) on either side")
    elif (scale_t - scale_c) * (step_t - step_c) < 0:
        item = finding("strength-reduction")
        item.dimensions = ["loops"]
        item.evidence.append(f"inside loops retail scales an index {scale_t} time(s) and steps a pointer {step_t} "
                             f"time(s), the candidate {scale_c} and {step_c}")
        item.confidence = _confidence(0.7, max(loop_extra, abs(scale_t - scale_c) + abs(step_t - step_c)), differing)
        loop_fired = True
    elif loops_t != loops_c:
        item = finding("strength-reduction")
        item.dimensions = ["loops"]
        item.evidence.append(f"retail has {loops_t} loop(s), the candidate {loops_c} (rotation or unrolling)")
        item.confidence = _confidence(0.45, max(2, loop_extra), differing)
        loop_fired = True
    else:
        rule_out("strength-reduction", "loop bodies scale indexes and step pointers alike")

    # repeated loads
    if res.reloads:
        item = finding("cse-reuse")
        item.dimensions = ["loads"]
        for row in res.reloads[:3]:
            item.evidence.append(f"`{row['load']}` at offset {row['offset']}: {row['target']} load(s) in retail, "
                                 f"{row['candidate']} in the candidate")
        item.confidence = _confidence(0.7, sum(abs(r["target"] - r["candidate"]) for r in res.reloads), differing)
    else:
        rule_out("cse-reuse", "every memory word is loaded equally often on both sides")

    # optimisation level
    fired = sorted(name for name, item in findings.items() if not item.ruled_out and COMPONENTS[name]["pass"])
    ratio_t, ratio_c = _stack_ratio(t), _stack_ratio(c)
    if max(ratio_t, ratio_c) >= 0.35 and min(ratio_t, ratio_c) <= 0.2:
        item = finding("level")
        item.dimensions = ["stack"]
        item.evidence.append(f"{'retail' if ratio_t > ratio_c else 'the candidate'} keeps values on the stack "
                             f"({round(100 * max(ratio_t, ratio_c))}% of its instructions are stack accesses, against "
                             f"{round(100 * min(ratio_t, ratio_c))}%): unoptimised code on one side")
        item.confidence = 0.85
    elif len(fired) >= 3:
        item = finding("level")
        item.dimensions = ["several"]
        item.evidence.append(f"{len(fired)} pass signatures at once ({', '.join(fired)}): a level changes many "
                             "passes together")
        item.confidence = min(0.85, 0.5 + 0.08 * (len(fired) - 3))

    # what no signature explains
    reload_keys = {(row["load"], row["offset"]) for row in res.reloads}

    def explained(insn: Insn, index: int, side: int) -> bool:
        if index in res.covered[side] or (loop_fired and index in res.loop_rows[side]):
            return True
        return (insn.base in (SP, FP) or insn.names(SP) or insn.names(AT) or insn.names(FP)
                or insn.name in MULTIPLIES or insn.name in ("mflo", "mfhi")
                or (multiply_fired and insn.name in shift_add)
                or (insn.name in LOADS and (insn.name, insn.imm) in reload_keys))

    unexplained = ([("retail", t[i], i) for i in res.extra_target if not explained(t[i], i, 0)]
                   + [("the candidate", c[j], j) for j in res.extra_candidate if not explained(c[j], j, 1)])
    if unexplained:
        item = finding("structural")
        item.dimensions = ["inserted"]
        side, insn, index = unexplained[0]
        item.evidence.append(f"{len(unexplained)} inserted or deleted instruction(s) no pass signature accounts "
                             f"for (first: `{insn.name}` at +0x{4 * index:X} in {side})")
        item.confidence = _confidence(0.75, len(unexplained), differing)

    # the kit's existing diagnoses, merged into the components they name
    if wrap:
        for component, confidence, text in wrapped(target, candidate, vram, pipeline):
            if component is None:
                continue
            item = findings.get(component)
            if item is None:
                if component not in CREATED_BY_WRAPPED:
                    continue
                item = finding(component)
                item.confidence = round(0.85 * confidence, 2)
                item.evidence.append(text)
            elif item.ruled_out:
                item.contra.append(f"{text} (not taken: {item.contra[0]})")
            else:
                item.evidence.append(text)
                if component in CREATED_BY_WRAPPED or item.confidence:
                    item.confidence = max(item.confidence, round(0.85 * confidence, 2))
        source = [name for name in ("source-types", "source-volatile") if name in findings]
        if source and "structural" in findings and not findings["structural"].ruled_out:
            findings["structural"].confidence *= 0.5
            findings["structural"].contra.append(f"causes.py reads these rows as {source[0]}")

    # what the pass probes measured for this recipe
    profile = learned_profile(learned, lane, family)
    where = (lane.split()[0] if lane else None) or family or "this family"
    for item in findings.values():
        info = profile.get(item.component)
        if not info or item.ruled_out:
            continue
        toggles = ", ".join(info.get("flags") or []) or "its flag"
        if info.get("supported") and info.get("functions") and not info.get("effect"):
            item.contra.append(f"pass probes: {toggles} changed none of {info['functions']} probe function(s) "
                               f"under {where}")
            item.confidence *= 0.8
        elif info.get("effect"):
            overlap = sorted(set(item.dimensions) & set(info.get("dimensions") or {}))
            if overlap:
                item.evidence.append(f"pass probes: {toggles} produced {', '.join(overlap)} in {info['effect']} probe "
                                     f"function(s) under {where}")
                item.confidence = min(0.97, item.confidence * 1.1)
        if info.get("pairs") and not info.get("supported"):
            item.contra.append(f"{toggles}: not accepted under {where}, so the pass cannot be switched by flag there")

    ranked = sorted(findings.values(), key=lambda item: (item.ruled_out, -item.confidence, item.component))
    return [item.as_dict() for item in ranked]


# ---------------------------------------------------------------------------------------------
# Learning pass signatures from the probe builds; ground truth for evaluations

def _hex_words(values) -> list[int]:
    return [int(value, 16) if isinstance(value, str) else int(value) for value in values]


def _empty_entry() -> dict:
    return {"pairs": [], "flags": [], "supported": False, "functions": 0, "effect": 0, "dimensions": {}, "hits": 0,
            "top": {}}


def learn(results: dict) -> dict:
    """Pass signatures measured by ``bench/passprobe/run.py``: per lane and per family, for each
    component the pairs tried, whether the compiler accepted the flag, how many probe functions
    changed, the residual dimensions the change produced, and how often ``attribute`` put the
    component among its first three (a check on code whose cause is known by construction)."""
    builds = results.get("builds") or []
    base = {build["lane"]: build for build in builds if build.get("role") == "base" and build.get("ok")}
    lanes: dict[str, dict] = {}
    checks = []
    for build in builds:
        role = build.get("role")
        if role in (None, "base") or build.get("skipped"):
            continue
        lane, family = build["lane"], build.get("family")
        component = build.get("component") or "assembler"
        entry = lanes.setdefault(lane, {}).setdefault(component, _empty_entry())
        entry["pairs"].append(role)
        change = build.get("change")
        if change and change not in entry["flags"]:
            entry["flags"].append(change)
        reference = base.get(build.get("base_lane") or lane)
        if not build.get("ok") or reference is None:
            continue
        entry["supported"] = True
        for name, words in sorted((build.get("functions") or {}).items()):
            other = (reference.get("functions") or {}).get(name)
            if other is None:
                continue
            entry["functions"] += 1
            t, c = _hex_words(other), _hex_words(words)
            if t == c:
                continue
            entry["effect"] += 1
            for dimension in analyse(t, c).dimensions():
                entry["dimensions"][dimension] = entry["dimensions"].get(dimension, 0) + 1
            ranked = [row["component"] for row in attribute(t, c, family=family, wrap=False) if not row["ruled_out"]]
            if ranked:
                entry["top"][ranked[0]] = entry["top"].get(ranked[0], 0) + 1
            hit = component in ranked[:3]
            entry["hits"] += int(hit)
            checks.append({"lane": lane, "pair": role, "function": name, "expected": component, "top3": ranked[:3],
                           "hit": hit})
    families: dict[str, dict] = {}
    for lane, components in lanes.items():
        family = lane_family(lane) or "other"
        for component, entry in components.items():
            into = families.setdefault(family, {}).setdefault(component, _empty_entry())
            into["pairs"] += [p for p in entry["pairs"] if p not in into["pairs"]]
            into["flags"] += [f for f in entry["flags"] if f not in into["flags"]]
            into["supported"] = into["supported"] or entry["supported"]
            for key in ("functions", "effect", "hits"):
                into[key] += entry[key]
            for key in ("dimensions", "top"):
                for name, count in entry[key].items():
                    into[key][name] = into[key].get(name, 0) + count
    return {"lanes": lanes, "families": families, "checks": checks}


def summary_markdown(learned: dict) -> str:
    lines = ["| lane | component | pairs | accepted | changed / probe functions | dimensions | attribute top-1 | "
             "expected in top-3 |", "|---|---|---|---|---|---|---|---|"]
    for lane in sorted(learned.get("lanes") or {}):
        for component, entry in sorted(learned["lanes"][lane].items()):
            dims = ", ".join(f"{k} {v}" for k, v in sorted(entry["dimensions"].items(), key=lambda kv: -kv[1])[:4])
            top = ", ".join(f"{k} {v}" for k, v in sorted(entry["top"].items(), key=lambda kv: -kv[1])[:2])
            lines.append(f"| {lane} | {component} | {', '.join(entry['pairs'])} | "
                         f"{'yes' if entry['supported'] else 'no'} | {entry['effect']} / {entry['functions']} | "
                         f"{dims or '-'} | {top or '-'} | {entry['hits']} / {entry['effect']} |")
    return "\n".join(lines) + "\n"


TRUTH_ACCEPTS = {"level": {"level"}, "isa": {"hazard-nops", "branch-likely"}, "small-data": {"assembler"},
                 "assembler": {"assembler"}, "compiler": {"compiler-family", "level"}}


def normalize_flags(flags) -> list[str]:
    """Flags with ``-G N`` written as one token."""
    items, out, k = list(flags), [], 0
    while k < len(items):
        if items[k] == "-G" and k + 1 < len(items):
            out.append("-G" + items[k + 1])
            k += 2
            continue
        out.append(items[k])
        k += 1
    return out


def recipe_delta(a: str, b: str) -> dict:
    """How recipe label ``b`` differs from ``a`` (``lane flag flag ...``)."""
    parts_a, parts_b = a.split(), b.split()
    lane_a, lane_b = (parts_a or [""])[0], (parts_b or [""])[0]
    flags_a, flags_b = normalize_flags(parts_a[1:]), normalize_flags(parts_b[1:])
    return {"lanes": [lane_a, lane_b] if lane_a != lane_b else [],
            "removed": [flag for flag in flags_a if flag not in flags_b],
            "added": [flag for flag in flags_b if flag not in flags_a]}


def flag_truth(flag: str) -> str:
    for pair in PASS_PAIRS:
        if pair.change[0] in ("add", "toggle") and pair.change[1] == flag:
            return pair.component
    if flag.startswith(("-O", "-g")):
        return "level"
    if flag.startswith("-mips"):
        return "isa"
    if flag.startswith("-G"):
        return "small-data"
    return "other"


def truth(delta: dict) -> dict:
    """Ground truth of a recipe difference: its labels, whether it is one label, and the
    components an attribution may name for it."""
    labels: list[str] = []
    if delta.get("lanes"):
        a, b = delta["lanes"]
        labels.append("assembler" if a.startswith("kmc") and b.startswith("kmc") else "compiler")
    for flag in list(delta.get("removed") or []) + list(delta.get("added") or []):
        label = flag_truth(flag)
        if label not in labels:
            labels.append(label)
    accepts: set[str] = set()
    for label in labels:
        accepts |= TRUTH_ACCEPTS.get(label, {label})
    return {"labels": labels, "single": len(labels) == 1, "accepts": sorted(accepts)}


# ---------------------------------------------------------------------------------------------
# Building probes (used by the bench scripts; one compile at a time)

def installed_lanes(cfg, root, assemblers: bool = False) -> dict[str, dict]:
    """Lanes the toolchain root can drive, with their usual flags: the built-in lanes (one per
    compiler; with ``assemblers`` also the assembler variants of KMC GCC) and every configured
    lane of an installed catalogue compiler."""
    from . import compilers, probes
    out: dict[str, dict] = {}
    have = set(cfg.available())
    if "kmc" in have:
        flags = list(probes.FLAG_SETS["kmc+as"][0])
        out["kmc+as"] = {"family": "gcc", "flags": flags, "compiler": "gcc2.7.2kmc"}
        if assemblers:
            out["kmc+atload"] = {"family": "gcc", "flags": flags, "compiler": "gcc2.7.2kmc", "assembler_of": "kmc+as"}
            if cfg.gas29() is not None:
                out["kmc+gas"] = {"family": "gcc", "flags": flags, "compiler": "gcc2.7.2kmc", "assembler_of": "kmc+as"}
            if cfg.gnu_as() is not None:
                out["kmc+asn64"] = {"family": "gcc", "flags": flags, "compiler": "gcc2.7.2kmc",
                                    "assembler_of": "kmc+as"}
    for key, lane in (("ido53", "ido5.3"), ("ido71", "ido7.1")):
        if key in have:
            out[lane] = {"family": "ido", "flags": list(probes.FLAG_SETS[lane][0]), "compiler": lane}
    if "sn" in have:
        out["sn"] = {"family": "gcc", "flags": list(probes.FLAG_SETS["sn"][0]), "compiler": "gcc2.8.1sn"}
    for name, table in compilers.project_pipelines(root).items():
        out["custom:" + name] = {"family": table["family"], "flags": list(table["default_flags"]),
                                 "compiler": table["compiler"]}
    return out


def probe_config(work, root, timeout: float):
    """A scratch project under ``work`` whose configuration declares every installed catalogue
    compiler as a lane, each step limited to ``timeout`` seconds (built-in lanes read
    ``N64DK_TOOL_TIMEOUT``, which the caller sets)."""
    from . import compilers
    from .toolchain import load_config
    project = Path(work) / "project"
    project.mkdir(parents=True, exist_ok=True)
    tables = compilers.toml_tables(compilers.project_pipelines(root))
    tables = tables.replace("\nfamily = ", f"\ntimeout = {float(timeout)}\nfamily = ")
    (project / "n64dk.toml").write_text(f'[toolchains]\nroot = "{root}"\n\n' + tables)
    return load_config(project, None, str(root))


def compile_one(source, lane: str, flags, cfg, work) -> tuple[object, str, float]:
    """(Elf or None, error, seconds): one build of ``source`` through the kit's build path."""
    import tempfile
    import time
    from ..elf import Elf
    from .pipeline import BuildError, build, split_flags
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="n64dk-passprobe-", dir=str(work)) as scratch:
        try:
            built = build(Path(source), split_flags(list(flags), lane), cfg, Path(scratch), includes=[])
            return Elf(Path(built).read_bytes()), "", time.monotonic() - started
        except (BuildError, OSError, ValueError, RuntimeError) as error:
            return None, str(error)[-400:], time.monotonic() - started
