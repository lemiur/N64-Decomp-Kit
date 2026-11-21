"""Differential behaviour oracle: does a candidate's machine code behave like the target's?

Both are executed by the kit's R4300 interpreter (``n64dk.emu``, a bare machine) from identical
states, over the program's own memory image, and their observable effects are compared:

* the result registers (``$v0``, ``$v1``, ``$f0``-``$f3``) and the callee-saved registers
  (``$s0``-``$s8``, ``$sp``, ``$f20``-``$f31``, which must come back unchanged);
* every byte written outside the function's own stack frame;
* the calls made: each ``jal`` target outside the function is replaced by a ``break`` stub that
  records the call (target, ``$a0``-``$a3``, ``$f12``-``$f15`` and four words at ``$sp+16``) and
  behaves like a callee: it returns a value derived from the call's target and ordinal, and leaves
  every caller-saved register (``$at``, ``$a0``-``$a3``, ``$t0``-``$t9``, hi/lo, ``$f1``-``$f19``,
  the FP condition) holding junk derived the same way, identical in both runs, so code that keeps a
  value live in one across a call computes with the junk.  A stub reached by a tail call (``$ra``
  still the caller's) returns from the function.

Inputs come in tiers (``TIERS``): ``concrete`` (seeded random arguments and argument memory),
``boundary`` (0, ±1, the 16- and 32-bit limits and the immediates in the target's own code, ±1,
interleaved; common float values) and ``coverage`` (inputs that reach target instructions no
earlier input reached are kept and mutated, bounded).  Argument memory (the arena) is random, with
the first bytes of every block (where pointer arguments point) drawn from small integers, the
target's immediates ±1, floats and pointers to other blocks.  A ``symbolic`` tier is not available
(no constraint solver is installed).

The verdict is ``differs`` (with the first counterexample), ``no-difference-found`` (with the
number of conclusive inputs and the target coverage reached) or ``unknown``.  It **fails
closed**: a run that exhausts its step budget, loads or stores anywhere but RAM (an effect the
comparison cannot observe), or, for a candidate placed away from the target, reaches the target's
own code, gives ``unknown``, and so do too few conclusive inputs.  A run in which the *target*
faults (``syscall``, RSP access, an unmapped address, a misaligned load or store) says the input
is outside the function's domain and is not counted; the candidate faulting where the target
returned is a difference.
``no-difference-found`` is evidence, **never proof**: it is never an exactness verdict and the
kit never promotes a function on it.
"""
from __future__ import annotations

import hashlib
import random
import struct
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import zip_longest

from ..emu import LINE, M32, M64, Machine, Stop, sx32

RAM_SIZE = 0x01000000                      # 16 MiB: program image below 8 MiB, arena and stack above
IMAGE_LIMIT = 0x80800000
ARENA = 0x80C00000
ARENA_SIZE = 0x10000
STACK = 0x80E00000
STACK_WINDOW = (STACK - 0x10000, STACK + 0x10)   # the function's own frame and its argument home slots
BREAK = 0x0000000D
RETURN_TRAP = Machine.RETURN_TRAP
CANARY = 0x5A5A0000
TIERS = ("concrete", "boundary", "coverage")
SYMBOLIC = "blocked: no constraint solver (z3) is installed; the bounded symbolic tier is not run"
CALLER_SAVED = (1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25)   # $at, $a0-$a3, $t0-$t9
UNOBSERVED = "unobserved access"           # a Stop reason: the run is unknown, never a difference
BLOCK = 0x100                              # pointer arguments point at arena block starts
STRUCTURED = 0x80                          # the leading bytes of each block drawn from the palette
FLOAT_EDGES = (0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 1e-7, -1e7, 3.0e38, 1.2e-38)


class BareMachine(Machine):
    """The oracle's machine.  The interpreter's fast path handles plain RDRAM itself; every other
    access reaches these methods.  One the comparison cannot observe (a store to, or a load or fetch
    from, anything but this RAM: hardware registers, cartridge space, open bus, kseg0/1 above
    16 MiB) stops the run instead of being dropped or reading a constant; ``Runner`` reports it as
    ``unknown``.  (A load from open bus reads 0 for any address, so a wrong address agreed.)
    A misaligned halfword/word/doubleword access stops as the processor's address error does (a
    fault: the target is outside its domain, the candidate differs); without it, ``lw`` from
    ``x+1`` read zero-filled memory as ``lw`` from ``x`` did, and a misaligned store could spill
    into a line the write log never looked at."""

    def __init__(self, rom=None, ram_size: int = RAM_SIZE):
        super().__init__(rom, ram_size=ram_size)
        self.strict_alignment = True

    def read(self, vaddr: int, size: int) -> int:
        if self._phys(vaddr) + size > self.ram_size:
            raise Stop(f"{UNOBSERVED}: {size}-byte load at 0x{vaddr & M32:08X}", self.pc)
        return super().read(vaddr, size)

    def write(self, vaddr: int, size: int, value: int) -> None:
        if self._phys(vaddr) + size > self.ram_size:
            raise Stop(f"{UNOBSERVED}: {size}-byte store at 0x{vaddr & M32:08X}", self.pc)
        super().write(vaddr, size, value)


@dataclass
class Observation:
    status: str                              # returned | fault | unknown
    reason: str = ""
    regs: tuple = ()
    writes: tuple = ()
    calls: tuple = ()
    coverage: frozenset = frozenset()

    def key(self) -> tuple:
        return (self.status, self.regs, self.writes, self.calls)


@dataclass
class Inputs:
    args: tuple                              # $a0..$a3 (32-bit)
    floats: tuple                            # raw bits of $f12, $f13, $f14, $f15
    arena_seed: int
    palette: tuple = ()                      # words the structured part of the arena is drawn from

    def describe(self) -> dict:
        return {"a": [f"0x{a & M32:08X}" for a in self.args], "f": [f"0x{f:08X}" for f in self.floats],
                "arena_seed": self.arena_seed}


def _jal_targets(code: bytes, vram: int) -> set[int]:
    out = set()
    for i in range(0, len(code) - 3, 4):
        word = struct.unpack_from(">I", code, i)[0]
        if word >> 26 == 3:
            out.add(((vram + i + 4) & 0xF0000000) | ((word & 0x3FFFFFF) << 2))
    return out


def _immediates(code: bytes) -> set[int]:
    out = set()
    for i in range(0, len(code) - 3, 4):
        word = struct.unpack_from(">I", code, i)[0]
        op = word >> 26
        if op in (8, 9) and (word >> 21) & 31 == 29:          # a frame size or a frame address
            continue
        if op in (8, 9, 10, 11, 12, 13, 14):                 # addi(u), slti(u), andi, ori, xori
            imm = word & 0xFFFF
            out.update({imm, imm - 0x10000 if imm & 0x8000 else imm})
        elif op == 15:
            out.add((word & 0xFFFF) << 16)
    return out


def _near(code: bytes, limit: int) -> list[int]:
    """The target's own immediates, smallest magnitude first, each with its neighbours: the values
    where a comparison against it flips."""
    out = []
    for imm in sorted(_immediates(code), key=lambda v: (abs(v), v))[:limit]:
        out += [imm - 1, imm, imm + 1]
    return out


def _palette(code: bytes) -> tuple:
    words = [0, 1, 2, 3, 4, 8, 0xFF, 0xFFFF, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]
    words += [struct.unpack(">I", struct.pack(">f", x))[0] for x in (1.0, -1.0, 0.5, 2.0)]
    return tuple(dict.fromkeys(w & M32 for w in words + _near(code, 16)))


@lru_cache(maxsize=4)                        # the two runners ask for the same arena in turn
def _arena(seed: int, palette: tuple) -> bytes:
    """One input's argument memory: random bytes, except the first ``STRUCTURED`` bytes of every
    ``BLOCK`` (where pointer arguments point), drawn word by word from ``palette`` (values a program
    tests against), pointers to other blocks (so a chain of pointers stays in the arena instead of
    faulting) and random words."""
    rnd = random.Random(seed)
    data = bytearray(rnd.randbytes(ARENA_SIZE))
    blocks = ARENA_SIZE // BLOCK
    pool = list(palette) + [ARENA + BLOCK * rnd.randrange(blocks - 32) for _ in range(16)] + \
        [rnd.getrandbits(32) for _ in range(16)]
    per = STRUCTURED // 4
    packed = struct.pack(f">{blocks * per}I", *rnd.choices(pool, k=blocks * per))
    for b in range(blocks):
        data[b * BLOCK:b * BLOCK + STRUCTURED] = packed[b * STRUCTURED:(b + 1) * STRUCTURED]
    return bytes(data)


def _clobber(m: Machine, pc: int, ordinal: int) -> None:
    """What a callee leaves behind: results in ``$v0``/``$v1``/``$f0`` derived from the call's target
    and ordinal, and junk derived the same way in every caller-saved register (both runs make the
    same call, so they get the same junk; a candidate that keeps a value live in one across the call
    where the target saved it computes with the junk and differs).  ``$ra``, ``$sp``, ``$gp`` and the
    callee-saved registers are kept, as a callee that returns keeps them."""
    rnd = random.Random(f"clobber:{pc}:{ordinal}")
    g = m.gpr
    for r in CALLER_SAVED:
        g[r] = sx32(rnd.getrandbits(32))
    for r in range(1, 20):                   # $f1-$f19; $f0 is the result below
        m.fpr[r] = rnd.getrandbits(32)
    m.hi, m.lo = sx32(rnd.getrandbits(32)), sx32(rnd.getrandbits(32))
    m.fcc = bool(rnd.getrandbits(1))
    value = int.from_bytes(hashlib.sha256(f"{pc}:{ordinal}".encode()).digest()[:4], "big")
    g[2], g[3] = sx32(value), sx32(value ^ 0xFFFF)
    m.fpr[0] = value


class Runner:
    """One variant (the target or a candidate) in its own machine; reset before every input.

    ``shadow``: the target's range, given for a candidate.  When the candidate is placed elsewhere,
    the image still holds the target there, and the candidate could reach it (an absolute ``j``
    linked for the target's address, a jump table in the image's .rodata) and agree by running the
    target itself.  The range becomes ``break`` (reaching it: ``unknown``), except its entry, which
    becomes ``j code_vram``: a call to the function's own address is recursion, and reaches the
    candidate as the target's recursion reaches the target."""

    def __init__(self, image: bytes, entry: int, code: bytes, code_vram: int, stubs: set[int],
                 gp: int | None = None, max_steps: int = 200_000, shadow: tuple[int, int] | None = None):
        self.entry, self.max_steps, self.gp = code_vram, max_steps, gp
        self.machine = BareMachine(None, ram_size=RAM_SIZE)
        base = bytearray(RAM_SIZE)
        base[:len(image)] = image
        lo = code_vram & 0x1FFFFFFF
        self.own = (code_vram, code_vram + len(code))
        self.shadow = shadow if shadow is not None and shadow[0] != code_vram and shadow[0] >> 30 == 2 else None
        if self.shadow is not None:
            first, end = self.shadow
            for t in range(first & ~3, end, 4):
                p = t & 0x1FFFFFFF
                if p + 4 <= RAM_SIZE:
                    base[p:p + 4] = struct.pack(">I", BREAK)
            p = first & 0x1FFFFFFF
            if end - first >= 8 and p + 8 <= RAM_SIZE and not (lo < p + 8 and p < lo + len(code)) \
                    and (first + 4) & 0xF0000000 == code_vram & 0xF0000000:
                base[p:p + 8] = struct.pack(">II", (2 << 26) | ((code_vram >> 2) & 0x3FFFFFF), 0)
        base[lo:lo + len(code)] = code                     # after the shadow: where they overlap, the code
        self.stubs = {t for t in stubs if not self.own[0] <= t < self.own[1] and not self.in_shadow(t)}
        for target in self.stubs:
            if target >> 30 != 2:            # kuseg/kseg2 is TLB-mapped: its physical alias is not the callee
                continue
            p = target & 0x1FFFFFFF
            if p + 4 <= RAM_SIZE:
                base[p:p + 4] = struct.pack(">I", BREAK)
        self.base = bytes(base)

    def in_shadow(self, address: int) -> bool:
        return self.shadow is not None and self.shadow[0] <= address < self.shadow[1]

    def release(self) -> None:
        m = self.machine
        if m is not None:
            m._templates.clear()
            m.decoded = {}
            m.code_pages = {}
        self.machine = None
        self.base = b""

    def run(self, inputs: Inputs) -> Observation:
        m = self.machine
        ram = m.ram
        ram[:] = self.base
        arena = _arena(inputs.arena_seed, inputs.palette)
        lo = ARENA & 0x1FFFFFFF
        ram[lo:lo + ARENA_SIZE] = arena
        m.dirty[:] = bytes(len(m.dirty))
        g = m.gpr
        g[:] = [0] * 32
        for i, value in enumerate(inputs.args[:4]):
            g[4 + i] = sx32(value)
        for r in range(16, 24):
            g[r] = sx32(CANARY | r)
        g[30] = sx32(CANARY | 30)
        g[28] = sx32(self.gp) if self.gp is not None else 0
        g[29] = sx32(STACK)
        g[31] = sx32(RETURN_TRAP)
        m.fpr[:] = [0] * 32
        for i, bits in enumerate(inputs.floats[:4]):
            m.fpr[12 + i] = bits & M32
        for r in range(20, 32):
            m.fpr[r] = CANARY | r
        m.hi = m.lo = 0
        m.fcc = False
        m.decoded = {}
        m.code_pages.clear()                     # translated code holds this dict: clear it in place
        m.cop0[:] = [0] * 32
        m.cop0[12] = 0x30000000                  # CU0 | CU1, FR clear
        m.count = 0
        m.tlb[:] = [None] * 32
        m.tlb_next = 31
        m.steps = 0
        m.pc = self.entry
        m.last_call = None
        calls, coverage = [], set()
        try:
            while True:
                result = m.run(max_steps=self.max_steps, stop_after_handoff=1 << 62, stop_at=RETURN_TRAP)
                coverage.update(pc for pc in m.decoded if self.own[0] <= pc < self.own[1])
                reason = result.stop_reason
                if reason == "returned":
                    break
                if reason == "break" and m.pc in self.stubs:
                    if len(calls) >= 4096:
                        return Observation("unknown", "more than 4096 calls", coverage=frozenset(coverage))
                    ordinal = len(calls)
                    # Arguments 5 and later (on the caller's stack) are not logged: the callee's arity is
                    # unknown, and for a callee of four or fewer arguments those words are the caller's
                    # own frame, which equivalent code may lay out differently (a false reject).  A wrong
                    # stack argument therefore goes unseen (a documented limit; round-12 review C, P4).
                    calls.append((m.pc, *(g[r] & M32 for r in (4, 5, 6, 7)),
                                  *(m.fpr[r] & M32 for r in (12, 13, 14, 15))))
                    _clobber(m, m.pc, ordinal)
                    # The callee returns to $ra.  Reached by `jal`, that is the instruction after the delay
                    # slot.  Reached by a tail call (`j`, or `jr` through a register, after the epilogue
                    # restored $ra), it is the caller's return address, RETURN_TRAP: the function has
                    # returned.  (`Machine.run` tests stop_at only on a control transfer, so resuming *at*
                    # RETURN_TRAP would run the exception vector there.)  `j` targets are not stubbed by
                    # themselves: one that is also a `jal` target anywhere in either variant is a stub; any
                    # other runs the image's own code, the same in both runs, rather than turning a
                    # function whose catalogued size is short into a call.
                    if g[31] & M32 == RETURN_TRAP:
                        break
                    m.pc = g[31] & M32
                    continue
                if reason == "break" and self.in_shadow(m.pc):
                    return Observation("unknown", f"the candidate reached the target's own code at 0x{m.pc & M32:08X}",
                                       coverage=frozenset(coverage))
                if reason.startswith(UNOBSERVED):
                    return Observation("unknown", reason, coverage=frozenset(coverage))
                if reason == "step budget exhausted":
                    return Observation("unknown", reason, coverage=frozenset(coverage))
                return Observation("fault", f"{reason} at 0x{m.pc & M32:08X}", coverage=frozenset(coverage))
        except Stop as stop:                                   # raised outside run (should not happen)
            return Observation("unknown", stop.reason, coverage=frozenset(coverage))
        regs = tuple(g[r] & M64 for r in (2, 3, 16, 17, 18, 19, 20, 21, 22, 23, 29, 30)) + \
            tuple(m.fpr[r] & M64 for r in (0, 1, 2, 3, *range(20, 32)))
        writes = []
        # a store the fast path handles marks the line it starts in; a misaligned one would spill into
        # the next line.  BareMachine traps those, and the next line is compared too all the same.
        dirty = m.dirty
        lines = sorted({j for i, d in enumerate(dirty) if d for j in (i, i + 1) if j < len(dirty)})
        for line in lines:
            a, b = line << LINE, (line + 1) << LINE
            vram_a = 0x80000000 | a
            if STACK_WINDOW[0] <= vram_a < STACK_WINDOW[1] and 0x80000000 | b <= STACK_WINDOW[1]:
                continue
            # the state at entry: the base image, with this input's arena over it
            before = bytearray(self.base[a:b])
            if a < lo + ARENA_SIZE and lo < b:
                x, y = max(a, lo), min(b, lo + ARENA_SIZE)
                before[x - a:y - a] = arena[x - lo:y - lo]
            for off in range(a, b):
                if ram[off] != before[off - a] and not STACK_WINDOW[0] <= (0x80000000 | off) < STACK_WINDOW[1]:
                    writes.append((0x80000000 | off, ram[off]))
        return Observation("returned", "", regs, tuple(writes), tuple(calls), frozenset(coverage))


@dataclass
class Verdict:
    verdict: str                             # differs | no-difference-found | unknown
    conclusive: int = 0
    inconclusive: int = 0
    unknown: int = 0
    tiers: dict = field(default_factory=dict)
    coverage: float = 0.0                    # share of target instructions executed
    counterexample: dict | None = None
    reason: str = ""
    symbolic: str = SYMBOLIC
    candidate_unfinished: int = 0            # inputs the target finished and the candidate did not

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "conclusive": self.conclusive, "inconclusive": self.inconclusive,
                "unknown": self.unknown, "tiers": self.tiers, "coverage": round(self.coverage, 3),
                "counterexample": self.counterexample, "reason": self.reason, "symbolic": self.symbolic,
                "proof": False}


def _pointer_args(code: bytes) -> set[int]:
    """Argument registers used as a load/store base (straight line), directly or through a copy
    (``move``, or ``addiu`` of an offset into the same object) made before anything overwrote them."""
    holds = {4: 0, 5: 1, 6: 2, 7: 3}          # register -> the argument it still holds or points into
    out = set()
    for i in range(0, len(code) - 3, 4):
        word = struct.unpack_from(">I", code, i)[0]
        op, rs, rt, rd = word >> 26, (word >> 21) & 31, (word >> 16) & 31, (word >> 11) & 31
        if op in (32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 46, 49, 53, 55, 57, 61, 63) and rs in holds:
            out.add(holds[rs])
        source = None
        if op == 0:
            dest = rd
            if word & 63 in (0x21, 0x25, 0x2D) and rt == 0 and rs in holds:     # move rd, rs
                source = holds[rs]
            elif word & 63 in (0x21, 0x25, 0x2D) and rs == 0 and rt in holds:
                source = holds[rt]
        elif op == 3:                                                    # jal: writes $ra only
            dest = 31
        elif op in (40, 41, 42, 43, 46, 57, 61, 63, 1, 4, 5, 6, 7, 20, 21, 22, 23, 2):
            dest = 0
        else:
            dest = rt
            if op in (9, 25) and rs in holds:                            # addiu rt, rs, offset
                source = holds[rs]
        if dest:
            if source is None:
                holds.pop(dest, None)
            else:
                holds[dest] = source
    return out


def _inputs(tier: str, count: int, seed: int, code: bytes) -> list[Inputs]:
    rnd = random.Random(f"{tier}:{seed}")
    pointers = _pointer_args(code)
    palette = _palette(code)

    def arg(i: int, value: int | None = None) -> int:
        if i in pointers or (value is None and rnd.random() < 0.3):
            return ARENA + BLOCK * rnd.randrange(0, (ARENA_SIZE // BLOCK) - 32)
        return rnd.getrandbits(32) if value is None else value & M32

    def floats(edges: bool = False) -> tuple:
        return tuple(struct.unpack(">I", struct.pack(">f", rnd.choice(FLOAT_EDGES) if edges else
                                                      rnd.uniform(-1000, 1000)))[0] for _ in range(4))

    out = []
    if tier == "concrete":
        for _ in range(count):
            out.append(Inputs(tuple(arg(i) for i in range(4)), floats(), rnd.getrandbits(32), palette))
    elif tier == "boundary":
        generic = [0, 1, -1, 2, 0x7FFF, 0x8000, 0xFFFF, 0x10000, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]
        # interleaved, so the target's own immediates +-1 (where its comparisons flip) are reached
        # by a short tier too; the tier grows (at most doubles) towards one input per value
        mixed = [v for pair in zip_longest(generic, _near(code, 24)) for v in pair if v is not None]
        values = list(dict.fromkeys(v & M32 for v in mixed))
        for k in range(max(count, min(len(values), 2 * count))):
            out.append(Inputs(tuple(arg(i, values[(k + 3 * i) % len(values)] if i not in pointers else None)
                                    for i in range(4)), floats(edges=k % 2 == 1), rnd.getrandbits(32), palette))
    return out


def compare(image: bytes, target: bytes, vram: int, candidate: bytes, candidate_vram: int | None = None,
            gp: int | None = None, per_tier: int = 24, seed: int = 0, min_conclusive: int = 8,
            max_steps: int = 200_000, coverage_rounds: int = 24) -> Verdict:
    """Run ``target`` (at ``vram``) and ``candidate`` (at ``candidate_vram``, default the same
    address) over the program ``image`` (bytes from 0x80000000).  Never proof."""
    left = right = None
    try:
        candidate_vram = vram if candidate_vram is None else candidate_vram
        stubs = _jal_targets(target, vram) | _jal_targets(candidate, candidate_vram)
        image = image[:IMAGE_LIMIT - 0x80000000]
        left = Runner(image, vram, target, vram, stubs, gp, max_steps)
        right = Runner(image, candidate_vram, candidate, candidate_vram, stubs, gp, max_steps,
                       shadow=(vram, vram + len(target)))
        return _compare(left, right, target, seed, per_tier, min_conclusive, coverage_rounds)
    finally:
        # translated instructions hold their machine (a reference cycle around 16 MiB buffers that
        # the cycle collector, counting objects and not bytes, reclaims late): break it now
        for runner in (left, right):
            if runner is not None:
                runner.release()
        import gc
        gc.collect()


def _compare(left: "Runner", right: "Runner", target: bytes, seed: int, per_tier: int, min_conclusive: int,
             coverage_rounds: int) -> Verdict:
    verdict = Verdict("unknown")
    covered: set[int] = set()
    instructions = max(1, len(target) // 4)

    def one(tier: str, inputs: Inputs) -> bool:
        """True when the input showed a difference (the verdict is then final)."""
        a = left.run(inputs)
        covered.update(a.coverage)
        row = verdict.tiers.setdefault(tier, {"inputs": 0, "conclusive": 0})
        row["inputs"] += 1
        if a.status == "unknown":
            verdict.unknown += 1
            return False
        if a.status == "fault":
            verdict.inconclusive += 1                          # outside the target's domain
            return False
        b = right.run(inputs)
        if b.status == "unknown":
            # the target returned and the candidate did not finish: fail closed (round-12 review)
            verdict.unknown += 1
            verdict.candidate_unfinished += 1
            return False
        verdict.conclusive += 1
        row["conclusive"] += 1
        if a.key() != b.key():
            verdict.verdict = "differs"
            verdict.counterexample = {"tier": tier, "inputs": inputs.describe(), "target": _summary(a),
                                      "candidate": _summary(b)}
            return True
        return False

    for tier in ("concrete", "boundary"):
        for inputs in _inputs(tier, per_tier, seed, target):
            if one(tier, inputs):
                verdict.coverage = len(covered) / instructions
                return verdict
    # coverage-guided: mutate the inputs that reached new target instructions
    rnd = random.Random(f"coverage:{seed}")
    corpus = _inputs("concrete", 4, seed + 1, target)
    for _ in range(coverage_rounds):
        parent = rnd.choice(corpus)
        args = list(parent.args)
        i = rnd.randrange(4)
        args[i] = rnd.choice([args[i] ^ (1 << rnd.randrange(32)), args[i] + rnd.choice([-1, 1]),
                              rnd.choice(sorted(_immediates(target)) or [0]) & M32, 0])
        child = Inputs(tuple(a & M32 for a in args), parent.floats, rnd.getrandbits(32), parent.palette)
        known = len(covered)
        if one("coverage", child):
            verdict.coverage = len(covered) / instructions
            return verdict
        if len(covered) > known:
            corpus.append(child)
    verdict.coverage = len(covered) / instructions
    if verdict.candidate_unfinished:
        verdict.reason = (f"the candidate did not finish on {verdict.candidate_unfinished} input(s) the target "
                          "finished (budget, an unobserved access, or the target's own code reached): "
                          "fails closed")
    elif verdict.conclusive >= min_conclusive:
        verdict.verdict = "no-difference-found"
        verdict.reason = f"{verdict.conclusive} conclusive inputs agree; evidence, not proof"
    else:
        verdict.reason = f"only {verdict.conclusive} conclusive inputs (need {min_conclusive}): fails closed"
    return verdict


def _summary(obs: Observation) -> dict:
    return {"status": obs.status, "reason": obs.reason,
            "v0": f"0x{obs.regs[0]:X}" if obs.regs else None,
            "writes": len(obs.writes), "calls": [f"0x{c[0]:08X}" for c in obs.calls[:8]]}


def mutants(code: bytes, limit: int = 8, seed: int = 0) -> list[tuple[str, bytes]]:
    """Single-instruction mutants of ``code`` for measuring false accepts: an immediate changed
    by one, a branch condition inverted, an arithmetic operation swapped.  Some mutants are
    equivalent (dead code, a delay-slot nop), so a mutant judged equal bounds false accepts from
    above.  Sites whose mutant is equivalent by construction are not offered: a result written to
    ``$zero``, ``addu``/``subu`` with ``$zero`` as the second operand (``move``: x + 0 == x - 0),
    a branch to the instruction after its own delay slot (both ways lead there)."""
    rnd = random.Random(seed)
    words = [struct.unpack_from(">I", code, i)[0] for i in range(0, len(code) - len(code) % 4, 4)]
    sites = []
    for i, w in enumerate(words):
        op, rt, rd = w >> 26, (w >> 16) & 31, (w >> 11) & 31
        if op in (8, 9, 10, 11, 12, 13, 14, 32, 33, 35, 36, 37) and rt == 0:
            pass                                               # the result is discarded
        elif op in (8, 9, 10, 11, 12, 13, 14) or op in (32, 33, 35, 36, 37, 40, 41, 43):
            sites.append((i, "immediate+1", (w & 0xFFFF0000) | ((w + 1) & 0xFFFF)))
        if op in (4, 5) and w & 0xFFFF != 1:                   # beq <-> bne
            sites.append((i, "branch-inverted", w ^ (1 << 26)))
        if op == 0 and w & 63 in (0x21, 0x23) and rd != 0 and rt != 0:     # addu <-> subu
            sites.append((i, "addu<->subu", w ^ 0x2))
    rnd.shuffle(sites)
    out = []
    for i, kind, new in sites[:limit]:
        mutated = list(words)
        mutated[i] = new
        out.append((f"{kind}@0x{4 * i:X}", struct.pack(f">{len(mutated)}I", *mutated)))
    return out
