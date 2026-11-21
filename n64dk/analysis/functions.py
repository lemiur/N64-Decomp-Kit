"""Function discovery for the boot-loaded main code segment.

Everything is derived from the ROM image and the boot analysis:

1. The image is the IPL3 copy (ROM 0x1000 -> header entry), clipped at the BSS start
   found in the entry routine.
2. Functions are seeded from the entry point and grown to a fixpoint through ``jal``
   targets and ``j`` tail calls found in walked code.
3. Each function is walked linearly while tracking the furthest in-function branch
   target (jump-table targets included); it ends after the delay slot of the first
   terminator (``jr ra``, ``jr reg``, ``j``/``b``, ``eret``) that no branch skips over.
   Trailing zero words up to the next start are padding and belong to the function.
4. The text extent ends at the first load/store target above the code reached so far,
   or where a linear sweep for further code stops being valid code.
5. Gaps inside text are swept for unreferenced code (must start after a terminator,
   decode as valid compiler-emittable code and end in a terminator); rejected gaps are
   reported as in-text data.
6. Code pointers taken with %hi/%lo pairs and 32-bit words in data that point into text
   add starts where they land on a gap or on a split point (just after a terminator and
   its delay slot), never on jump-table targets.
7. Jump tables are recognised from ``lui``/``addiu``|``lw`` + ``addu`` + ``jr reg``
   with the ``sltiu`` bound when present.  The index addition may be ``daddu``: EGCS
   with 64-bit registers (``-mips3`` without ``-mgp32``) adds the scaled index that way.

Boundary findings (late starts, interior aliases, shared epilogues, cross-function
branches, fall-through) can be computed for the discovered catalogue or for any
catalogue supplied by a project.
"""
from __future__ import annotations

import bisect
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

from .. import boot as bootmod
from .. import mips
from ..rom import Rom

# Registers that legitimately hold a value at function entry (o32): zero, a0-a3,
# s0-s7, gp, sp, fp, ra, and k0/k1 for exception code.
_ENTRY_DEFINED = {0, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31}
_COP0 = {"mtc0", "mfc0", "dmtc0", "dmfc0", "eret", "cache", "tlbwi", "tlbwr", "tlbp", "tlbr", "sync",
         "cfc0", "ctc0"}
_GPR_NAMES = ["zero", "at", "v0", "v1", "a0", "a1", "a2", "a3", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7",
              "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "t8", "t9", "k0", "k1", "gp", "sp", "fp", "ra"]
MAX_TABLE = 1024
_NEVER_TAKEN_ON_ZERO = {"bne", "bnel", "bnez", "bnezl", "bltz", "bltzl", "bgtz", "bgtzl", "bltzal", "bltzall"}


@dataclass
class JumpTable:
    address: int
    count: int
    site: int
    function: int = 0
    bounded: bool = False
    targets: list[int] = field(default_factory=list)

    @property
    def size(self) -> int:
        return self.count * 4


@dataclass
class Body:
    start: int
    end: int  # after the last instruction (delay slot included), padding excluded
    ok: bool = True
    reason: str = ""
    fell_through: bool = False
    calls: set = field(default_factory=set)
    tails: set = field(default_factory=set)
    bals: set = field(default_factory=set)     # calls by a linking branch (hand-written code)
    cross: list = field(default_factory=list)  # (site, target) branches leaving [start, limit)
    tables: list = field(default_factory=list)


@dataclass
class Function:
    start: int
    end: int  # including trailing padding
    body_end: int
    name: str = ""
    kind: str = "c"  # c, O0, asm, leaf, stub
    seed: str = "call"  # entry, call, tail, pointer, data_pointer, sweep
    frame: int = 0
    flags: list = field(default_factory=list)
    reasons: list = field(default_factory=list)   # HANDWRITTEN_REASONS that made it ``asm``

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass
class Finding:
    address: int
    kind: str
    detail: str
    function: int | None = None


@dataclass
class Catalogue:
    vram: int
    rom: int
    image_end: int
    text_start: int
    text_end: int
    functions: list[Function]
    jump_tables: list[JumpTable]
    findings: list[Finding]
    data_regions: list[tuple[int, int]]
    image: "Image | None" = field(default=None, repr=False)

    def function_at(self, vram: int) -> Function | None:
        starts = [f.start for f in self.functions]
        i = bisect.bisect_right(starts, vram) - 1
        if i >= 0 and vram < self.functions[i].end:
            return self.functions[i]
        return None


class Image:
    """The main segment as decoded instructions, cached per word."""

    def __init__(self, data: bytes, vram: int):
        self.data = data
        self.vram = vram
        self.end = vram + len(data) - len(data) % 4
        self._cache: dict[int, object] = {}
        self._valid: dict[tuple[int, bool], bool] = {}

    def __contains__(self, v: int) -> bool:
        return self.vram <= v < self.end

    def word(self, v: int) -> int:
        return struct.unpack_from(">I", self.data, v - self.vram)[0]

    def ins(self, v: int):
        i = self._cache.get(v)
        if i is None:
            i = self._cache[v] = mips.decode(self.word(v), v)
        return i

    def valid(self, v: int, strict: bool = False) -> bool:
        key = (v, strict)
        r = self._valid.get(key)
        if r is None:
            r = self._valid[key] = self._check(v, strict)
        return r

    def _check(self, v: int, strict: bool) -> bool:
        if v not in self:
            return False
        i = self.ins(v)
        if i.getRaw() == 0:
            return True
        if not (i.isValid() and i.isImplemented()):
            return False
        if not strict:
            return True
        if i.isJumpWithAddress():
            t = i.getInstrIndexAsVram()
            # Outside the 8 MiB window the image runs in (RDRAM through KSEG0, or the mapped
            # region of a program run through the TLB), or into the first 8 KiB of KSEG0 below
            # the image (RSP IMEM-style targets of microcode embedded in the segment).
            window = self.vram & 0xFF800000
            if not window <= t < window + 0x800000 and t not in self:
                return False
            if window == 0x80000000 and t < 0x80002000 and t not in self:
                return False
        # Compilers never write $zero except through the canonical nop, and never emit a
        # branch on $zero that can never be taken.
        if i.outputsToGprZero() and not i.isNop():
            return False
        if i.isBranch() and _reads(i) <= {0} and i.getOpcodeName() in _NEVER_TAKEN_ON_ZERO:
            return False
        return True


def _is_terminator(ins) -> bool:
    name = ins.getOpcodeName()
    return (ins.isJrRa() or ins.isJrNotRa() or name in ("eret", "break")
            or (ins.isJumpWithAddress() and not ins.doesLink()) or (ins.isUnconditionalBranch() and not ins.doesLink()))


def _branch_target(ins) -> int | None:
    if ins.isBranch():
        return ins.getBranchVramGeneric() & 0xFFFFFFFF
    if ins.isJumpWithAddress():
        return ins.getInstrIndexAsVram()
    return None


def _written(ins) -> int | None:
    d = ins.getDestinationGpr()
    return mips.reg(d) if d is not None else None


def _reads(ins) -> set[int]:
    out = set()
    if ins.readsRs():
        out.add(mips.reg(ins.rs))
    if ins.readsRt():
        out.add(mips.reg(ins.rt))
    if ins.readsRd():
        out.add(mips.reg(ins.rd))
    return out


def _simm(ins) -> int:
    v = ins.getImmediate()
    return v - 0x10000 if v & 0x8000 else v


WEAK_SEEDS = ("tail", "pointer", "data_pointer", "sweep", "prologue", "branch")     # starts no call vouches for
JR_RA = 0x03E00008


def _after_function_end(img: "Image", at: int, padding: int = 7) -> bool:
    """A function ends right before ``at``: a ``jr ra`` or ``j`` and its delay slot, then at most
    ``padding`` zero words."""
    def ends(word):
        return word == JR_RA or word >> 26 == 2
    q = at
    while q - 4 >= img.vram and img.word(q - 4) == 0 and at - q < 4 * padding:
        q -= 4
    if q < at and q - 4 >= img.vram and ends(img.word(q - 4)):     # its delay slot was a nop
        return True
    return q - 8 >= img.vram and ends(img.word(q - 8))


class Analyser:
    def __init__(self, image: Image, entry: int, text_hint_end: int | None = None, text_lo: int | None = None,
                 scan_evidence: bool = True):
        self.img = image
        self.entry = entry
        self.text_lo = entry if text_lo is None else text_lo
        self.text_hi = text_hint_end or image.end
        self.starts: dict[int, str] = {}
        self.rejected: set[int] = set()          # weak starts dropped because they do not walk as code
        self.bodies: dict[int, Body] = {}
        self.findings: list[Finding] = []
        self.data_regions: list[tuple[int, int]] = []
        self.table_targets: set[int] = set()
        self.tables: dict[int, JumpTable] = {}
        self.const_jumps: dict[int, int] = {}
        self._walks: dict[tuple, Body] = {}
        self._refs: dict[tuple[int, int], list] = {}  # jr reg with a lui/addiu constant target
        # Weak entry evidence used only to resynchronise after stray words: every jal
        # target and every aligned word in the image that points into it.
        self.evidence: set[int] = set()
        self.jal_targets: set[int] = set()
        if not scan_evidence:                   # walking only (coverage audits): no resync evidence
            return
        for _, w in mips.iter_words(image.data):
            if w >> 26 == 3:
                self.jal_targets.add((image.vram & 0xF0000000) | ((w & 0x3FFFFFF) << 2))
            elif image.vram <= w < image.end and not w % 4:
                self.evidence.add(w)
        self.evidence |= self.jal_targets
        self.evidence.update(r.address for r in mips.scan_refs(image.data, image.vram, call_targets=False)
                             if r.kind == "addr")

    # -- walking ---------------------------------------------------------------------------
    def walk(self, start: int, limit: int, strict: bool = False) -> Body:
        key = (start, limit, strict, self.text_hi)
        body = self._walks.get(key)
        if body is None:
            body = self._walks[key] = self._walk(start, limit, strict)
        return body

    def _walk(self, start: int, limit: int, strict: bool) -> Body:
        img = self.img
        body = Body(start, start)
        pc, furthest = start, start
        known: dict[int, int] = {}          # registers holding an address computed from a PC read
        derived: set[int] = set()           # every such address (passed on to local subroutines)
        while pc < limit:
            if not img.valid(pc, strict):
                body.ok, body.reason, body.end = False, f"invalid instruction at 0x{pc:08X}", pc
                return body
            ins = img.ins(pc)
            name = ins.getOpcodeName()
            target = _branch_target(ins)
            # Hand-written code reads its own address (``bltzal $zero`` / ``bal``: $ra = pc + 8) and
            # calls ``jalr`` an address computed from it: a call like any other.
            if name == "jalr" and (mips.reg(ins.rs) in known or derived):
                # the register's own value, or (set before a bal into the subroutine holding this
                # jalr) any address the routine computed from its PC
                for local in ([known[mips.reg(ins.rs)]] if mips.reg(ins.rs) in known else sorted(derived)):
                    if start <= local < limit:
                        furthest = max(furthest, local)
                    elif local in img:
                        body.calls.add(local)
                        body.bals.add(local)
            written = _written(ins)
            if written is not None:
                known.pop(written, None)
            if target is not None and ins.isBranch() and ins.doesLink():
                known[mips.RA] = pc + 8
            elif name in ("addiu", "daddiu") and mips.reg(ins.rs) == mips.RA and mips.RA in known and written:
                known[written] = (known[mips.RA] + _simm(ins)) & 0xFFFFFFFF
                derived.add(known[written])
            if target is not None:
                if ins.doesLink():
                    if ins.isJumpWithAddress():
                        body.calls.add(target)
                    elif ins.getOpcodeName() in ("bltzal", "bltzall") and mips.reg(ins.rs) == 0:
                        pass                       # never taken: only reads the PC (often of a table)
                    elif start <= target < limit:  # bal / bgezal: local subroutine in asm
                        furthest = max(furthest, target)
                    else:
                        # bal into another routine: a call all the same (``bltzal $zero`` is never
                        # taken: it only reads the PC, and its target is often a constant table)
                        body.calls.add(target)
                        body.bals.add(target)
                elif start <= target < limit:
                    furthest = max(furthest, target)
                elif ins.isJumpWithAddress():
                    body.tails.add(target)
                else:
                    body.cross.append((pc, target))
            if ins.isJrNotRa() and pc + 4 < limit:
                table = self.jump_table(start, pc)
                if table is not None:
                    body.tables.append(table)
                    inside = [t for t in table.targets if start <= t < limit]
                    if inside:
                        furthest = max(furthest, max(inside))
                elif pc in self.const_jumps:
                    body.calls.add(self.const_jumps[pc])
            # Dead code after an in-function loop jump or a trap (epilogue, HI/LO read) continues.
            loops = name == "break" or (target is not None and not ins.doesLink() and start <= target <= pc)
            nxt = pc + (4 if name == "break" else 8)
            while loops and nxt < limit and nxt in img and img.word(nxt) == 0 and nxt < pc + 0x40:
                nxt += 4
            if (_is_terminator(ins) and furthest <= pc + (0 if name in ("eret", "break") else 4)
                    and not (loops and nxt < limit and _continues(img, nxt))):
                if name in ("eret", "break"):
                    body.end = pc + 4
                else:
                    if pc + 4 >= limit or not img.valid(pc + 4, strict):
                        body.ok, body.reason, body.end = False, "terminator without delay slot", pc + 4
                        return body
                    body.end = pc + 8
                return body
            pc += 4
        body.end = limit
        body.fell_through = True
        return body

    def jump_table(self, start: int, jr_pc: int) -> JumpTable | None:
        """Recover ``lui``/(``addiu``)/``addu``|``daddu``/``lw``/``jr`` table dispatch ending at ``jr_pc``.

        ``daddu``: EGCS 1.1.2 at ``-O2 -mips3`` (64-bit registers) dispatches through
        ``sltiu; beqz; sll; lui; daddu; lw %lo; jr``.  Missing it left the table words to the
        data-pointer seeds, which split every case label after a ``jr ra`` into a function."""
        img = self.img
        if jr_pc in self.tables:
            return self.tables[jr_pc]
        consts: dict[int, int] = {}
        bases: dict[int, int] = {}
        loaded: dict[int, int] = {}
        bound = None
        for pc in range(max(start, jr_pc - 0x1000), jr_pc, 4):
            ins = img.ins(pc)
            name = ins.getOpcodeName()
            dest = _written(ins)
            value = kind = None
            if name == "lui":
                value, kind = (ins.getImmediate() << 16) & 0xFFFFFFFF, "c"
            elif name in ("addiu", "daddiu") and mips.reg(ins.rs) in consts:   # la at -mips3: lui; daddiu
                value, kind = (consts[mips.reg(ins.rs)] + _simm(ins)) & 0xFFFFFFFF, "c"
            elif name == "ori" and mips.reg(ins.rs) in consts:
                value, kind = consts[mips.reg(ins.rs)] | ins.getImmediate(), "c"
            elif name in ("addu", "daddu"):
                rs, rt = mips.reg(ins.rs), mips.reg(ins.rt)
                if rs in consts or rt in consts:
                    value, kind = consts.get(rs, consts.get(rt)), "b"
            elif name == "lw" and mips.reg(ins.rs) in bases:
                value, kind = (bases[mips.reg(ins.rs)] + _simm(ins)) & 0xFFFFFFFF, "l"
            elif name == "sltiu":
                bound = ins.getImmediate()
            if dest is not None and dest != 0:
                for d in (consts, bases, loaded):
                    d.pop(dest, None)
                if kind == "c":
                    consts[dest] = value
                elif kind == "b":
                    bases[dest] = value
                elif kind == "l":
                    loaded[dest] = value
        reg = mips.reg(img.ins(jr_pc).rs)
        if reg in consts:
            self.const_jumps[jr_pc] = consts[reg]
        address = loaded.get(reg)
        if address is None or address % 4 or address not in img:
            return None
        count = bound if bound and bound <= MAX_TABLE else None
        targets = []
        for i in range(count or MAX_TABLE):
            a = address + 4 * i
            if a + 4 > img.end:
                break
            t = img.word(a)
            if t % 4 or not (start <= t < self.text_hi):
                if count:
                    return None
                break
            targets.append(t)
        if not targets:
            return None
        table = JumpTable(address, len(targets), jr_pc, start, bool(count), targets)
        self.tables[jr_pc] = table
        return table

    # -- helpers -----------------------------------------------------------------------------
    def sorted_starts(self) -> list[int]:
        return sorted(self.starts)

    def limit_for(self, start: int, ordered: list[int]) -> int:
        i = bisect.bisect_right(ordered, start)
        return ordered[i] if i < len(ordered) else self.text_hi

    def split_point(self, v: int) -> bool:
        """True if ``v`` follows a terminator and its delay slot (or padding after one)."""
        p = v - 4
        while p >= self.img.vram and self.img.word(p) == 0 and p > v - 0x40:
            p -= 4
        if p < self.img.vram + 4:
            return False
        if self.img.ins(p).getOpcodeName() in ("eret", "break"):
            return True
        return _is_terminator(self.img.ins(p - 4)) and p + 4 <= v

    def add(self, v: int, seed: str) -> bool:
        if v in self.starts or v not in self.img or v % 4 or not (self.text_lo <= v < self.text_hi):
            return False
        if seed in WEAK_SEEDS and v in self.rejected:
            # Dropped once because it does not walk as code.  The jump that proposed it is still
            # there and would propose it on every pass: growth would never end.
            return False
        self.starts[v] = seed
        return True

    def prologue_seeds(self, lo: int, hi: int) -> int:
        """Seed every stack-frame prologue (``addiu sp, sp, -N``) in [lo, hi) that follows the end
        of a function and walks strictly as one.  Code reached only through pointers the analysis
        cannot resolve (method tables, callbacks) is otherwise never catalogued."""
        added = 0
        for at in range(lo + (-lo % 4), hi, 4):
            word = self.img.word(at)
            if word >> 16 != 0x27BD or not word & 0x8000 or at in self.starts:
                continue
            if any(x <= at < y for x, y in self.data_regions) or not _after_function_end(self.img, at):
                continue
            if self.walk(at, hi, strict=True).ok and self.add(at, "prologue"):
                added += 1
        return added

    # -- driver ------------------------------------------------------------------------------
    def grow(self, seeds_ok=lambda v: True) -> None:
        """Walk all starts with limits from their neighbours until no new call/tail target appears."""
        while True:
            ordered = self.sorted_starts()
            new = False
            bodies = {}
            for s in ordered:
                b = self.walk(s, self.limit_for(s, ordered))
                bodies[s] = b
                if not b.ok and self.starts[s] != "entry":
                    continue
                for t in sorted(b.calls):
                    if seeds_ok(t) and self.img.valid(t) and self.add(t, "call"):
                        new = True
                for t in sorted(b.tails):
                    if seeds_ok(t) and self.img.valid(t) and self.split_point(t) and self.add(t, "tail"):
                        new = True
            self.bodies = bodies
            # A weak start in the delay slot of the code before it is that code's delay slot.
            for i, s in enumerate(ordered[1:], 1):
                before = bodies.get(ordered[i - 1])
                if (self.starts.get(s) in WEAK_SEEDS and before is not None and before.ok and before.end == s
                        and before.fell_through and self.img.ins(s - 4).hasDelaySlot()):
                    self.findings.append(Finding(s, "rejected_start", f"{self.starts[s]} seed: a delay slot"))
                    self.rejected.add(s)
                    del self.starts[s]
                    new = True
            # Drop weak starts whose walk hits invalid code (calls into data, misread tails).
            for s, b in bodies.items():
                if s not in self.starts:
                    continue
                weak = self.starts.get(s) in WEAK_SEEDS
                if weak and (not b.ok or (b.fell_through and self.starts[s] == "sweep" and s not in self.evidence)):
                    reason = b.reason or "falls into the next function"
                    self.findings.append(Finding(s, "rejected_start", f"{self.starts[s]} seed: {reason}"))
                    if not b.ok:                              # what does not walk never will
                        self.rejected.add(s)
                    del self.starts[s]
                    new = True
            if not new:
                return

    def code_refs(self) -> list[mips.Ref]:
        refs = []
        for s, b in self.bodies.items():
            if b.ok and b.end > s:
                got = self._refs.get((s, b.end))
                if got is None:
                    off = s - self.img.vram
                    got = self._refs[(s, b.end)] = mips.scan_refs(self.img.data[off:off + (b.end - s)], s)
                refs += got
        return refs

    def is_prologue(self, v: int) -> bool:
        ins = self.img.ins(v)
        return (ins.getOpcodeName() in ("addiu", "daddiu") and mips.reg(ins.rt) == mips.SP
                and mips.reg(ins.rs) == mips.SP and _simm(ins) < 0)

    def sweep_one(self, pc: int, limit: int, hits: list[int]) -> tuple[int, Body] | None:
        """Next unreferenced function in ``[pc, limit)``: skips zero padding and up to fifteen stray
        words before a stack-frame prologue. Returns None when the bytes are not code."""
        q = pc
        while q < limit and self.img.word(q) == 0:
            q += 4
        if q >= limit:
            return None
        # Objects are often aligned to 16 bytes with garbage rather than zeros: when the words
        # up to the boundary cannot be compiled code, start at the boundary.
        aligned = (q + 15) & ~15
        order = list(range(q, min(q + 64, limit), 4))
        if aligned != q and aligned < limit and _junk_prefix(self.img, q, aligned):
            order.remove(aligned)
            order.insert(0, aligned)
        for cand in order:
            if _continues(self.img, cand):
                continue  # reads a temporary nobody set: stray bytes, not an entry
            b = self.walk(cand, limit, strict=True)
            if cand != q and cand != order[0] and not (self.is_prologue(cand) or cand in self.evidence
                                  or not self.img.valid(cand - 4, strict=True) or _continues(self.img, cand - 4)):
                continue
            i = bisect.bisect_left(hits, cand)
            hit = i < len(hits) and hits[i] < max(b.end, cand + 4)
            if b.ok and not b.fell_through and not hit and b.end - cand >= 8:
                if cand != q:
                    self.findings.append(Finding(q, "stray_words", f"0x{cand - q:X} non-code bytes before 0x{cand:08X}"))
                return cand, b
        return None

    def resync(self, q: int, limit: int, hits: list[int]) -> tuple[int, Body] | None:
        """First strong entry after non-code bytes: a stack-frame prologue, a ``jal`` target or a
        ``jr ra`` leaf right after zero fill, whose walk forms a complete function."""
        for cand in range(q + 4, limit, 4):
            if self.img.word(cand) == 0:
                continue
            after_pad = self.img.word(cand - 4) == 0 and self.img.word(cand - 8) == 0
            if not (cand in self.jal_targets or self.is_prologue(cand) or after_pad) or _continues(self.img, cand):
                continue
            b = self.walk(cand, limit, strict=True)
            if after_pad and not (cand in self.jal_targets or self.is_prologue(cand)):
                # A plain start after zero fill must be a complete leaf returning through jr ra.
                if not (b.ok and b.end - cand >= 12 and self.img.ins(b.end - 8).isJrRa()):
                    continue
            i = bisect.bisect_left(hits, cand)
            if b.ok and not b.fell_through and not (i < len(hits) and hits[i] < b.end):
                return cand, b
        return None

    def find_text_end(self, sweep: bool = True) -> int:
        ends = [b.end for s, b in self.bodies.items() if b.ok or self.starts.get(s) == "entry"]
        top = max(ends)
        if not sweep:
            return top
        refs = self.code_refs()
        data_refs = sorted(r.address for r in refs if r.kind in ("load", "store") and r.address >= top)
        # a data reference may point past the loaded image (bss): the sweep stops at the image's end
        floor = min(data_refs[0], self.img.end) if data_refs else self.img.end
        # Continue past the last reached function while the bytes keep forming functions.
        pc = top
        while pc < floor:
            found = self.sweep_one(pc, floor, data_refs)
            if found is None:
                break
            self.add(found[0], "sweep")
            pc = found[1].end
        return min(pc, floor)

    def sweep_gaps(self, data_hits: set[int]) -> None:
        hits = sorted(data_hits)
        ordered = self.sorted_starts()
        for i, s in enumerate(ordered):
            b = self.bodies.get(s)
            if b is None or not (b.ok or self.starts[s] == "entry"):
                continue
            limit = ordered[i + 1] if i + 1 < len(ordered) else self.text_hi
            pc = b.end
            while pc < limit:
                found = self.sweep_one(pc, limit, hits)
                if found is None:
                    q = pc
                    while q < limit and self.img.word(q) == 0:
                        q += 4
                    if q >= limit:
                        break
                    found = self.resync(q, limit, hits)
                    self.data_regions.append((q, found[0] if found else limit))
                    if found is None:
                        break
                self.add(found[0], "sweep")
                pc = found[1].end

    def pointer_seeds(self, candidates: dict[int, str]) -> bool:
        new = False
        ordered = self.sorted_starts()
        for v, seed in sorted(candidates.items()):
            if v in self.starts or v in self.table_targets or not (self.text_lo <= v < self.text_hi) or v % 4:
                continue
            if any(a <= v < b for a, b in self.data_regions):
                # Bytes the sweep took for data are code after all when code branches to them, or
                # an address pair in code names them and they walk without a junk word.
                if seed not in ("branch", "pointer") or (seed == "pointer" and _junk_prefix(self.img, v, v + 4)):
                    continue
            i = bisect.bisect_right(ordered, v) - 1
            inside = i >= 0 and v < self.bodies.get(ordered[i], Body(0, 0)).end
            if inside and (seed == "branch" or not self.split_point(v)):
                continue                        # a branch into a body is a shared tail, not a start
            if not inside and self.img.word(v) == 0:
                continue
            if self.walk(v, self.limit_for(v, ordered), strict=True).ok and self.add(v, seed):
                new = True
        return new


# ---------------------------------------------------------------------------------------------
def main_image(rom: Rom, info: bootmod.BootInfo | None = None, program=None) -> tuple[Image, bootmod.BootInfo]:
    """The main code segment: the IPL3 copy, or ``program``'s main segment (an
    :class:`n64dk.image.ProgramImage`, e.g. a loader's unpacked output)."""
    data = rom.data
    if program is not None:
        info, data = program.boot_info(), program.main_data(rom)
    info = info or bootmod.analyse(rom)
    size = info.copy_size
    if info.bss_start and info.main_vram < info.bss_start < info.main_vram + size:
        size = info.bss_start - info.main_vram
    size = min(size, len(data) - info.main_rom)
    return Image(data[info.main_rom:info.main_rom + size], info.main_vram), info


def discover(rom: Rom, info: bootmod.BootInfo | None = None, program=None, seeds=()) -> Catalogue:
    """``seeds`` are further addresses known to start functions (for a segment that was not reached
    through a boot entry: its stack-frame prologues), so the call closure does not depend on the
    first function reaching the rest."""
    image, info = main_image(rom, info, program)
    # After IPL3 the entry is the first word of text; a loader's hand-over can be anywhere in it.
    an = Analyser(image, info.entry, text_lo=info.main_vram if program is not None else None)
    an.add(info.entry, "entry")
    # The entry routine hands over through a register (``jr``), which the call closure does not
    # follow.  Where the program's code does not start right behind the entry routine (data lies
    # between), that hand-over is the only way in.
    main_call = getattr(info, "main_call", None)
    if main_call and main_call != info.entry and image.vram <= main_call < image.end and main_call & 3 == 0:
        an.add(main_call, "call")
    for seed in seeds:
        if image.vram <= seed < image.end and seed & 3 == 0:
            an.add(seed, "call")
    # Phase 1: call closure inside the image, then function-shaped prologues reached only through
    # pointers.  A conventionally linked KSEG0 image may contain a non-CPU block after its first
    # text run (Madden's embedded RSP boot code is one example), so do not let an unrelated
    # prologue-looking word beyond the established text bypass the evidence-backed extension
    # pass below.  A TLB-mapped program has no KSEG0 text boundary to lean on; for those images
    # the complete mapped image is the conservative search domain.
    an.grow()
    call_text_end = an.find_text_end()
    mapped = not 0x80000000 <= image.vram < 0x80800000
    if mapped and an.prologue_seeds(an.text_lo, image.end):
        an.grow()
    text_end = an.find_text_end()
    extensions: list[Finding] = []
    for _round in range(4):
        an.text_hi = text_end
        for s in [s for s in an.starts if s >= text_end]:
            del an.starts[s]
        an.grow(lambda v: v < text_end)
        # Once the text extent is established (or extended below), callbacks and method-table
        # entries inside that extent are safe prologue candidates even in a KSEG0 image.
        if an.prologue_seeds(an.text_lo, text_end):
            an.grow(lambda v: v < text_end)
        # Phase 2: fill gaps, then pointer seeds from code and data, to a fixpoint.
        for _ in range(8):
            refs = an.code_refs()
            data_hits = {r.address for r in refs if r.kind in ("load", "store") and an.text_lo <= r.address < text_end}
            an.table_targets = {t for tb in an.tables.values() for t in tb.targets}
            an.data_regions = []
            an.sweep_gaps(data_hits)
            an.grow(lambda v: v < text_end)
            candidates: dict[int, str] = {}
            for b in list(an.bodies.values()):
                if b.ok:
                    for _, t in b.cross:
                        candidates.setdefault(t, "branch")
            for r in refs:
                if r.kind == "addr" and an.text_lo <= r.address < text_end:
                    candidates.setdefault(r.address, "pointer")
            table_spans = [(tb.address, tb.address + tb.size) for tb in an.tables.values()]
            for lo, hi in [(text_end, image.end)] + an.data_regions:
                for off, w in mips.iter_words(image.data[lo - image.vram:hi - image.vram]):
                    a = lo + off
                    if an.text_lo <= w < text_end and not any(x <= a < y for x, y in table_spans):
                        candidates.setdefault(w, "data_pointer")
            if not an.pointer_seeds(candidates):
                break
            an.grow(lambda v: v < text_end)
        # Recovery: calls from accepted code that land past the text end mean CPU code
        # continues after an embedded non-CPU block (e.g. RSP microcode linked into .text).
        # Extend the text and run the phases again; the block becomes in-text data.
        beyond, agreed = _calls_beyond(an, text_end)
        if not beyond:
            break
        old_end = text_end
        an.text_hi = image.end
        for t in beyond:
            an.add(t, "call")
        an.grow()
        # Code found from one call ends where what it reaches ends: no sweep beyond it.
        text_end = max(text_end, an.find_text_end(sweep=agreed))
        if text_end <= old_end:
            break
        extensions.append(Finding(old_end, "text_extended",
                                  f"{len(beyond)} call targets past 0x{old_end:08X}; text extended to 0x{text_end:08X}"))
    # Final data regions after the last growth.
    refs = an.code_refs()
    data_hits = {r.address for r in refs if r.kind in ("load", "store") and an.text_lo <= r.address < text_end}
    an.data_regions = []
    an.sweep_gaps(data_hits)
    an.grow(lambda v: v < text_end)
    an.data_regions = []
    an.sweep_gaps(data_hits)
    cat = _assemble(an, info, image, text_end)
    cat.findings += extensions
    return cat


def _calls_beyond(an: Analyser, text_end: int) -> tuple[list[int], bool]:
    """Call targets past ``text_end`` from successfully walked functions that themselves walk
    as functions, and whether two call sites agree (one stray ``jal`` read from data must not
    move the text end).  A single call counts only when its target opens a stack frame and walks
    strictly to a return (``False``: the text then ends where that code ends)."""
    sites: dict[int, int] = {}
    for s, b in an.bodies.items():
        if s < text_end and b.ok:
            for t in b.calls:
                if text_end <= t < an.img.end and t % 4 == 0:
                    # a bal is a call inside one hand-written unit: it vouches for no text extent
                    sites[t] = sites.get(t, 0) + (0 if t in b.bals else 1)
    good = [t for t in sorted(sites) if sites[t] and an.img.valid(t) and an.walk(t, an.img.end).ok]
    if sum(sites[t] for t in good) >= 2:
        return good, True
    # One call is enough when its target opens a stack frame and walks strictly as a whole
    # function: a stray jal read from data does not land on a prologue that walks.
    return [t for t in good if an.is_prologue(t) and not an.walk(t, an.img.end).fell_through], False


def _assemble(an: Analyser, info, image: Image, text_end: int) -> Catalogue:
    ordered = an.sorted_starts()
    funcs: list[Function] = []
    bodies = {}
    for i, s in enumerate(ordered):
        limit = ordered[i + 1] if i + 1 < len(ordered) else text_end
        b = an.bodies.get(s)
        if b is None or (b.end > limit):
            b = an.walk(s, limit)
        bodies[s] = b
    calls = set().union(*(b.calls for b in bodies.values() if b.ok)) if bodies else set()
    pointers = _pointer_words(image, an.text_lo, text_end, an.data_regions)
    findings = [f for f in _dedupe(an.findings) if f.kind == "rejected_start"]
    for i, s in enumerate(ordered):
        b = bodies[s]
        limit = ordered[i + 1] if i + 1 < len(ordered) else text_end
        end = b.end
        data_after = [a for a, _ in an.data_regions if b.end <= a < limit]
        stop = min(data_after) if data_after else limit
        while end < stop and image.word(end) == 0:
            end += 4
        if end < stop:
            findings.append(Finding(end, "stray_words", f"0x{stop - end:X} non-code bytes before 0x{stop:08X}", s))
        seed = ("entry" if s == info.entry else "call" if s in calls else "pointer" if s in pointers
                else "tail" if an.starts[s] == "tail" else "branch" if an.starts[s] == "branch" else "unreferenced")
        f = Function(s, end, _trim_alignment(image, s, b), name=f"func_{s:08X}", seed=seed)
        classify(image, f)
        if b.fell_through:
            f.flags.append("falls_through")
        if not b.ok:
            f.flags.append("invalid")
        funcs.append(f)
    funcs = _reattach_epilogues(image, funcs, findings)
    findings += _shared_units(image, funcs)
    for a, e in sorted(set(an.data_regions)):
        if e - a < 16 and e in an.starts:
            findings.append(Finding(a, "stray_words", f"0x{e - a:X} non-code bytes before 0x{e:08X}"))
        else:
            findings.append(Finding(a, "in_text_data", f"0x{e - a:X} bytes of non-code inside text"))
    tables = {}
    for t in an.tables.values():
        for f in funcs:
            if f.start <= t.site < f.body_end:
                t.function = f.start
                tables[t.address] = t
                break
    cat = Catalogue(info.main_vram, info.main_rom, image.end, an.text_lo, text_end, funcs,
                    sorted(tables.values(), key=lambda t: t.address), findings,
                    sorted(set(an.data_regions)), image)
    cat.findings += audit(image, [(f.start, f.size) for f in funcs], cat, own=True)
    return cat


def _trim_alignment(image: Image, s: int, b: Body) -> int:
    """End of ``b``'s code: a body that runs into the next start through zero words after a call
    or a terminator (a routine that never returns, then alignment) ends after that delay slot."""
    if not (b.ok and b.fell_through):
        return b.end
    q = b.end
    while q - 4 > s and image.word(q - 4) == 0:
        q -= 4
    if q == b.end:
        return b.end
    last = image.ins(q - 4)
    if not (last.doesLink() or _is_terminator(last)):
        return b.end                            # the zeros are executed: they are code
    return min(b.end, q + 4) if last.hasDelaySlot() else q


def _epilogue_only(image: Image, s: int, e: int, frame: int) -> bool:
    """[s, e) restores registers from the stack, pops a frame of ``frame`` bytes and returns."""
    pops = returns = False
    for pc in range(s, e, 4):
        ins = image.ins(pc)
        n = ins.getOpcodeName()
        if ins.getRaw() == 0:
            continue
        if ins.isJrRa():
            returns = True
        elif n in ("addu", "or", "move", "daddu") and _written(ins) == mips.SP and mips.FP in _reads(ins):
            continue
        elif n in ("lw", "ld", "lwc1", "ldc1") and mips.reg(ins.rs) == mips.SP:
            continue
        elif n in ("addiu", "daddiu") and mips.reg(ins.rt) == mips.SP == mips.reg(ins.rs) and _simm(ins) == frame:
            pops = True
        else:
            return False
    return pops and returns


def _reattach_epilogues(image: Image, funcs: list[Function], findings: list[Finding]) -> list[Function]:
    """An unreferenced range that is nothing but the epilogue of the framed function right before
    it (which ends in a jump back into itself: ``for (;;)`` at -O0) belongs to that function."""
    out: list[Function] = []
    for f in funcs:
        p = out[-1] if out else None
        if (p is not None and f.seed == "unreferenced" and p.frame and p.end == f.start
                and p.body_end >= p.start + 8 and _epilogue_only(image, f.start, f.body_end, p.frame)):
            jump = image.ins(p.body_end - 8)
            back = _branch_target(jump)
            if back is not None and not jump.doesLink() and p.start <= back < p.body_end and _is_terminator(jump):
                findings.append(Finding(f.start, "epilogue_reattached",
                                        f"unreachable epilogue of 0x{p.start:08X} (behind its endless loop)", p.start))
                p.end, p.body_end = f.end, f.body_end
                continue
        out.append(f)
    return out


def _shared_units(image: Image, funcs: list[Function]) -> list[Finding]:
    """A branch from one routine into the middle of another: the two are one unit (hand-written
    code sharing a tail) or one of them is mis-split.  Recorded as a finding; the frameless target
    is hand-written code (``branched_into``)."""
    starts = [f.start for f in funcs]
    out: list[Finding] = []
    for f in funcs:
        for pc in range(f.start, f.body_end, 4):
            ins = image.ins(pc)
            if not ins.isBranch() or ins.doesLink():
                continue
            t = _branch_target(ins)
            if f.start <= t < f.body_end:
                continue
            i = bisect.bisect_right(starts, t) - 1
            g = funcs[i] if i >= 0 and t < funcs[i].body_end else None
            if g is None or g is f or t == g.start:
                continue
            out.append(Finding(pc, "shared_unit", f"0x{f.start:08X} branches to 0x{t:08X} inside 0x{g.start:08X}: "
                                                  "one unit, or a mis-split", f.start))
            if not g.frame and "branched_into" not in g.reasons:
                g.reasons.append("branched_into")
                g.kind = "asm"
    return out


def _dedupe(findings: list[Finding]) -> list[Finding]:
    seen, out = set(), []
    for f in findings:
        key = (f.address, f.kind, f.detail)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


# Code no compiler emits, by named reason (``Function.reasons``).  Each is a fact about the
# instructions, so a routine classified ``asm`` says why, and the reason travels to the reports.
HANDWRITTEN_REASONS = {
    "privileged": "a coprocessor-0, cache, TLB or eret instruction",
    "not_compiler_output": "an encoding the compilers never emit",
    "kernel_register": "uses $k0/$k1, which belong to the exception handler",
    "callee_saved_clobbered": "writes a callee-saved register ($s0-$s7, $fp) without a stack frame",
    "ra_in_register": "keeps the return address in another register",
    "ra_relative": "reads data at an address computed from $ra",
    "pc_read": "reads its own address with a linking branch (bal / bgezal $zero / bltzal)",
    "branch_out": "a branch leaves the body (into a neighbour's code)",
    "branched_into": "entered in its middle by a branch from another routine",
    "tail_jump": "a frameless routine jumps (j) into another routine",
    "call_without_saving_ra": "calls with no stack frame and without keeping $ra anywhere",
    "frame_released_early": "pops its stack frame and goes on running (not at a return)",
}
_CALLEE_SAVED = {16, 17, 18, 19, 20, 21, 22, 23, 30}


def handwritten_reasons(image: Image, f: Function) -> list[str]:
    """The named reasons (``HANDWRITTEN_REASONS``) ``f``'s own instructions give for hand-written
    code; empty for compiler output.  ``f.frame`` must already be known."""
    found: list[str] = []

    def note(reason):
        if reason not in found:
            found.append(reason)
    saved_ra = False
    for pc in range(f.start, f.body_end, 4):
        ins = image.ins(pc)
        if ins.getRaw() == 0:
            continue
        n = ins.getOpcodeName()
        if n in _COP0:
            note("privileged")
        elif ins.notEmittedByCompilers() or ins.isLikelyHandwritten():
            note("not_compiler_output")
        used = _reads(ins) | ({_written(ins)} if _written(ins) is not None else set())
        if used & {26, 27}:
            note("kernel_register")
        if not f.frame and _written(ins) in _CALLEE_SAVED:
            note("callee_saved_clobbered")
        if ins.isBranch() and ins.doesLink():
            note("pc_read")
        elif ins.isBranch():
            target = _branch_target(ins)
            if not f.start <= target < f.body_end and not f.frame:
                note("branch_out")
        if ins.isJumpWithAddress() and not ins.doesLink() and not f.frame \
                and not f.start <= ins.getInstrIndexAsVram() < f.body_end:
            note("tail_jump")
        if ins.doesLink() and not f.frame and not saved_ra:
            note("call_without_saving_ra")
        if (n in ("addiu", "daddiu") and mips.reg(ins.rt) == mips.SP == mips.reg(ins.rs) and _simm(ins) > 0
                and pc + 4 < f.body_end and not image.ins(pc - 4).isJrRa()
                and not any(image.ins(q).isJrRa() for q in range(pc + 4, min(pc + 16, f.body_end), 4))):
            note("frame_released_early")
        if mips.RA in _reads(ins) and not ins.isJrRa():
            saved_ra = True
        if mips.RA in _reads(ins) and not ins.isJrRa() and not ins.doesStore():
            if ins.doesLoad() or (n in ("addiu", "daddiu") and _simm(ins)):
                note("ra_relative")
            elif _written(ins) not in (None, 0, mips.RA):
                note("ra_in_register")
    return found


def classify(image: Image, f: Function) -> None:
    """Set ``kind`` to asm (hand-written), O0 (frame pointer), stub, leaf or c, the frame size and,
    for ``asm``, the reasons (``HANDWRITTEN_REASONS``)."""
    names = []
    calls = False
    for pc in range(f.start, f.body_end, 4):
        ins = image.ins(pc)
        names.append(ins)
        if ins.doesLink():
            calls = True
    head = names[:12]
    for ins in head:
        if ins.getOpcodeName() in ("addiu", "daddiu") and mips.reg(ins.rt) == mips.SP == mips.reg(ins.rs) and _simm(ins) < 0:
            f.frame = -_simm(ins)
            break
    sets_fp = any(_written(i) == mips.FP and mips.SP in _reads(i) and i.getOpcodeName() in ("addu", "or", "daddu", "move", "addiu")
                  for i in head)
    f.reasons = handwritten_reasons(image, f)
    if f.reasons:
        f.kind = "asm"
    elif sets_fp and f.frame:
        f.kind = "O0"
    elif not calls and not f.frame:
        f.kind = "stub" if (f.body_end - f.start) <= 16 else "leaf"
    else:
        f.kind = "c"


# ---------------------------------------------------------------------------------------------
def audit(image: Image, entries: list[tuple[int, int]], cat: Catalogue, own: bool = False) -> list[Finding]:
    """Boundary findings for a catalogue of (start, size) entries over ``image``."""
    out: list[Finding] = []
    entries = sorted(e for e in entries if e[0] in image)
    starts = [s for s, _ in entries]
    start_set = set(starts)
    call_targets: set[int] = set()
    branch_into: dict[int, list[int]] = {}
    for s, size in entries:
        for pc in range(s, min(s + size, image.end), 4):
            ins = image.ins(pc)
            t = _branch_target(ins)
            if t is None:
                continue
            if ins.doesLink():
                call_targets.add(t)
            else:
                branch_into.setdefault(t, []).append(pc)
    pointer_targets = _pointer_words(image, cat.text_start, cat.text_end, cat.data_regions)

    def owner(v):
        i = bisect.bisect_right(starts, v) - 1
        return entries[i] if i >= 0 and v < entries[i][0] + entries[i][1] else None

    prev_end = None
    for idx, (s, size) in enumerate(entries):
        e = s + size
        if prev_end is not None and s < prev_end:
            out.append(Finding(s, "overlap", f"starts 0x{prev_end - s:X} bytes before the previous entry ends", s))
        prev_end = max(prev_end or 0, e)
        if not image.valid(s):
            out.append(Finding(s, "not_code", "first word is not a valid instruction", s))
            continue
        if _junk_prefix(image, s, s + 4, reads=False):
            out.append(Finding(s, "junk_start", "first word is not compiler output (stray or data bytes)", s))
        ended = _control_ends_before(image, s)
        if ended == "delay_slot":
            out.append(Finding(s, "start_in_delay_slot", "entry begins in the delay slot of a jump", s))
        undefined = _reads_before_write(image, s, e)
        referenced = s in call_targets or s in pointer_targets or s == cat.text_start
        from_before = [b for b in branch_into.get(s, []) if b < s and (owner(b) or (0,))[0] != s]
        if ended is False:
            if undefined:
                out.append(Finding(s, "late_start", f"reads {','.join(undefined)} set by the preceding bytes"
                                   f"{'' if referenced else '; not referenced'}", s))
            elif from_before and not referenced:
                out.append(Finding(s, "interior_alias", f"reached only by branches from 0x{from_before[0]:08X}", s))
            elif not own or not referenced:
                out.append(Finding(s, "falls_through", "preceding code continues into this entry", s))
        elif from_before and not referenced:
            out.append(Finding(s, "interior_alias", f"label reached only by branches from 0x{from_before[0]:08X}", s))
        # Branches out of this entry.
        for pc in range(s, min(e, image.end), 4):
            ins = image.ins(pc)
            if ins.doesLink() or not ins.isBranch():
                continue
            t = _branch_target(ins)
            if s <= t < e:
                continue
            g = owner(t)
            if g is None:
                out.append(Finding(pc, "branch_outside", f"branch to 0x{t:08X} outside any entry", s))
            elif t == g[0]:
                out.append(Finding(pc, "branch_to_entry", f"branch into the start of 0x{g[0]:08X}", s))
            elif _is_epilogue(image, t):
                out.append(Finding(pc, "shared_epilogue", f"branch to epilogue at 0x{t:08X} in 0x{g[0]:08X}", s))
            else:
                out.append(Finding(pc, "cross_branch", f"branch to 0x{t:08X} inside 0x{g[0]:08X}", s))
    return out


def _pointer_words(image: Image, text_start: int, text_end: int, data_regions) -> set[int]:
    """Code addresses taken by %hi/%lo pairs in text or stored as words in data."""
    out = set()
    for lo, hi in [(text_end, image.end)] + list(data_regions):
        for _, w in mips.iter_words(image.data[lo - image.vram:hi - image.vram]):
            if text_start <= w < text_end:
                out.add(w)
    off = text_start - image.vram
    for r in mips.scan_refs(image.data[off:text_end - image.vram], text_start):
        if r.kind == "addr":
            out.add(r.address)
    return out


def _control_ends_before(image: Image, v: int):
    """False if control can fall into ``v``; True if it cannot; 'delay_slot' if ``v`` is a delay slot.

    Zero padding and stray non-code words between the last terminator and ``v`` are skipped."""
    if v - 4 < image.vram:
        return True
    prev = image.ins(v - 4)
    if prev.hasDelaySlot() and _is_terminator(prev):
        return "delay_slot"
    p = v - 4
    while p > image.vram and image.word(p) == 0 and p > v - 0x100:
        p -= 4
    # p is the last non-zero word; allow up to three stray words after a terminator.
    for t in range(p, max(image.vram, p - 16) - 4, -4):
        ins = image.ins(t)
        if not _is_terminator(ins):
            continue
        tail = t + (4 if ins.getOpcodeName() in ("eret", "break") else 8)
        if tail > v:
            return "delay_slot"
        return tail > p or _junk_prefix(image, tail, p + 4)
    return False


def _junk_prefix(image: Image, q: int, a: int, reads: bool = True) -> bool:
    """True if some word in ``[q, a)`` is not something a compiler emits at a function start:
    invalid, a misaligned or unaligned-access memory operand, or a read of a temporary
    nobody set."""
    defined = set(_ENTRY_DEFINED)
    for pc in range(q, a, 4):
        if not image.valid(pc, strict=True):
            return True
        ins = image.ins(pc)
        if ins.getRaw() == 0:
            continue
        name = ins.getOpcodeName()
        if name in ("lwl", "lwr", "swl", "swr", "ldl", "ldr", "sdl", "sdr"):
            return True
        if ins.notEmittedByCompilers() and (reads or not (name in _COP0 or name in ("addi", "addiu"))):
            return True
        if ins.doesLoad() or ins.doesStore():
            width = {"lh": 2, "lhu": 2, "sh": 2, "lw": 4, "sw": 4, "lwc1": 4, "swc1": 4, "lwu": 4,
                     "ld": 8, "sd": 8, "ldc1": 8, "sdc1": 8, "ll": 4, "sc": 4, "lld": 8, "scd": 8}.get(name, 1)
            if _simm(ins) % width:
                return True
        if reads and any(r not in defined for r in _reads(ins)):
            return True
        w = _written(ins)
        if w is not None:
            defined.add(w)
    return False


def _continues(image: Image, v: int) -> bool:
    """True if code at ``v`` cannot be a function entry and so continues the code before it:
    it reads HI/LO or a temporary nobody set, or restores ``ra``/pops the frame."""
    if v not in image or image.word(v) == 0:
        return False
    if not all(image.valid(x, strict=True) for x in range(v, min(v + 16, image.end), 4)):
        return False
    ins = image.ins(v)
    name = ins.getOpcodeName()
    if ins.readsHI() or ins.readsLO():
        return True
    if name in ("lw", "ld") and mips.reg(ins.rt) == mips.RA and mips.reg(ins.rs) == mips.SP:
        return True
    if name in ("addiu", "daddiu") and mips.reg(ins.rt) == mips.SP == mips.reg(ins.rs) and _simm(ins) > 0:
        return True
    return bool(_reads_before_write(image, v, v + 32))


def _reads_before_write(image: Image, s: int, e: int) -> list[str]:
    defined = set(_ENTRY_DEFINED)
    bad = []
    for pc in range(s, min(e, s + 32, image.end), 4):
        ins = image.ins(pc)
        if ins.getRaw() == 0:
            continue
        for r in sorted(_reads(ins)):
            if r not in defined and _GPR_NAMES[r] not in bad:
                bad.append(_GPR_NAMES[r])
        if ins.isBranch() or ins.isJump():
            break
        w = _written(ins)
        if w is not None:
            defined.add(w)
    return bad


def _is_epilogue(image: Image, t: int) -> bool:
    for pc in range(t, min(t + 48, image.end), 4):
        ins = image.ins(pc)
        if ins.isJrRa():
            return True
        n = ins.getOpcodeName()
        if ins.isBranch() or ins.isJump() or not (ins.doesLoad() or n in ("addiu", "move", "or", "addu", "nop") or ins.getRaw() == 0):
            return False
    return False


# -- catalogues ----------------------------------------------------------------------------------
_ENTRY_RE = re.compile(r'name\s*=\s*"([^"]+)"\s*,\s*vram\s*=\s*(0x[0-9A-Fa-f]+)\s*,\s*size\s*=\s*(0x[0-9A-Fa-f]+)')
_SYM_RE = re.compile(r'^\s*(\w+)\s*=\s*(0x[0-9A-Fa-f]+)\s*;.*type:func(?:.*size:(0x[0-9A-Fa-f]+))?')


def load_catalogue(path) -> list[tuple[str, int, int]]:
    """Read (name, vram, size) from a recomp-style TOML or a splat symbol_addrs.txt."""
    text = Path(path).read_text()
    out = [(n, int(v, 16), int(s, 16)) for n, v, s in _ENTRY_RE.findall(text)]
    if not out:
        for line in text.splitlines():
            m = _SYM_RE.match(line)
            if m:
                out.append((m.group(1), int(m.group(2), 16), int(m.group(3), 16) if m.group(3) else 0))
    return sorted(set(out), key=lambda x: x[1])


def compare(cat: Catalogue, truth: list[tuple[str, int, int]]) -> dict:
    """Precision/recall of start addresses and exact-size agreement against a reference catalogue.

    Only reference entries inside the discovered text extent are scored. Sizes are compared
    both exactly and with trailing zero padding stripped from both sides.
    """
    lo, hi = cat.text_start, cat.text_end
    ref = {v: s for _, v, s in truth if lo <= v < hi}
    ours = {f.start: f for f in cat.functions}
    common = set(ref) & set(ours)
    exact = sum(1 for v in common if ours[v].size == ref[v])
    body = sum(1 for v in common if ours[v].body_end - v == _strip(cat, v, ref[v]))
    tp = len(common)
    return {"reference": len(ref), "discovered": len(ours), "matched_starts": tp,
            "precision": round(tp / len(ours), 4) if ours else 0.0,
            "recall": round(tp / len(ref), 4) if ref else 0.0,
            "exact_size": exact, "exact_size_rate": round(exact / tp, 4) if tp else 0.0,
            "body_size": body, "body_size_rate": round(body / tp, 4) if tp else 0.0,
            "missed": sorted(set(ref) - common), "extra": sorted(set(ours) - common)}


def _strip(cat: Catalogue, v: int, size: int) -> int:
    """Size with trailing zero padding removed (a delay-slot nop after ``jr ra`` is kept)."""
    img = cat.image
    if img is None:
        return size
    e = v + size
    while e > v + 4 and e <= img.end and img.word(e - 4) == 0 and img.word(e - 8) != mips.JR_RA:
        e -= 4
    return e - v


# -- output ------------------------------------------------------------------------------------------
def to_toml(cat: Catalogue, title: str = "") -> str:
    lines = [f"# Function boundaries{(' for ' + title) if title else ''}: main segment",
             f"# text 0x{cat.text_start:08X}..0x{cat.text_end:08X}, {len(cat.functions)} functions",
             "[[section]]", 'name = ".main"', f"rom = 0x{cat.rom:08X}", f"vram = 0x{cat.vram:08X}",
             f"size = 0x{cat.text_end - cat.vram:X}", "", "functions = ["]
    for f in cat.functions:
        notes = [",".join(f.flags)] if f.flags else []
        if f.end > f.body_end:
            notes.append(f"pad:0x{f.end - f.body_end:X}")
        if f.reasons:
            notes.append("why:" + "+".join(f.reasons))
        lines.append(f'    {{ name = "{f.name}", vram = 0x{f.start:08x}, size = 0x{f.size:x} }},'
                     f' # {f.kind}{(" " + " ".join(notes)) if notes else ""}')
    lines.append("]")
    return "\n".join(lines) + "\n"


RETURN_ONLY = bytes.fromhex("03e0000800000000")              # jr $ra; nop
_FALL_THROUGH = bytes.fromhex("1000000100000000")            # b +1; nop


def merge_dead_returns(cat: Catalogue, data: bytes, vram: int) -> dict:
    """Give a debug build's functions back the code nothing reaches.

    IDO without optimisation ends every function with a return for control falling off the end,
    and compiles each ``return`` as a jump to the epilogue.  In a function with a stack frame that
    is ``b L; nop; L: <epilogue>``.  In a frameless one the epilogue is nothing but ``jr $ra``, so
    every jump to it is itself ``jr $ra; nop`` and the function's code is followed by one or two
    more of them that no path reaches.  A sweep for function starts takes each for an empty
    function.  They are part of the function in front of them: compiled alone, that function
    produces them again, and its interval has to hold them to be matched.

    Only done for an image that *is* such a build (a quarter or more of its framed functions show
    the ``b +1; nop`` fall-through), and only for a return-only range nothing references that
    directly follows a frameless function with no padding between."""
    functions = sorted(cat.functions, key=lambda f: f.start)

    def body(f) -> bytes:
        return data[f.start - vram:f.body_end - vram]

    framed = [f for f in functions if f.frame]
    falling = sum(1 for f in framed if any(body(f)[at:at + 8] == _FALL_THROUGH for at in range(0, len(body(f)) - 7, 4)))
    if len(framed) < 20 or falling * 4 < len(framed):
        return {"merged": 0, "debug_build": False}
    kept, merged = [], 0
    for f in functions:
        before = kept[-1] if kept else None
        if before is not None and f.seed == "unreferenced" and body(f) == RETURN_ONLY and before.end == f.start \
                and before.body_end == before.end and not before.frame and before.kind != "asm":
            before.end, before.body_end = f.end, f.body_end
            merged += 1
            continue
        kept.append(f)
    cat.functions[:] = kept
    return {"merged": merged, "debug_build": True, "framed": len(framed), "fall_through": falling}


def to_symbol_addrs(cat: Catalogue) -> str:
    out = [f"{f.name} = 0x{f.start:08X}; // type:func size:0x{f.size:X}" for f in cat.functions]
    out += [f"jtbl_{t.address:08X} = 0x{t.address:08X}; // type:jtbl size:0x{t.size:X}" for t in cat.jump_tables]
    return "\n".join(out) + "\n"


def to_jump_tables_tsv(cat: Catalogue) -> str:
    rows = ["address\tcount\tsize\tfunction\tsite\tbounded\tfirst_target"]
    for t in cat.jump_tables:
        rows.append(f"0x{t.address:08X}\t{t.count}\t0x{t.size:X}\t0x{t.function:08X}\t0x{t.site:08X}\t{int(t.bounded)}"
                    f"\t0x{t.targets[0]:08X}")
    return "\n".join(rows) + "\n"


def to_findings_tsv(findings: list[Finding]) -> str:
    rows = ["address\tkind\tfunction\tdetail"]
    for f in sorted(findings, key=lambda x: (x.address, x.kind)):
        rows.append(f"0x{f.address:08X}\t{f.kind}\t{'' if f.function is None else f'0x{f.function:08X}'}\t{f.detail}")
    return "\n".join(rows) + "\n"


def summary(cat: Catalogue) -> dict:
    kinds: dict[str, int] = {}
    seeds: dict[str, int] = {}
    for f in cat.functions:
        kinds[f.kind] = kinds.get(f.kind, 0) + 1
        seeds[f.seed] = seeds.get(f.seed, 0) + 1
    fk: dict[str, int] = {}
    for x in cat.findings:
        fk[x.kind] = fk.get(x.kind, 0) + 1
    return {"text": f"0x{cat.text_start:08X}..0x{cat.text_end:08X}",
            "text_size": cat.text_end - cat.text_start,
            "data_start": f"0x{cat.text_end:08X}", "image_end": f"0x{cat.image_end:08X}",
            "functions": len(cat.functions), "kinds": kinds, "seeds": seeds,
            "jump_tables": len(cat.jump_tables), "jump_table_entries": sum(t.count for t in cat.jump_tables),
            "in_text_data": [f"0x{x.address:08X}+0x{int(x.detail.split()[0], 16):X}"
                             for x in cat.findings if x.kind == "in_text_data"],
            "stray_words": sum(1 for x in cat.findings if x.kind == "stray_words"),
            "findings": fk}
