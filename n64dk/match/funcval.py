"""Deterministic emulator-based functional validation: milestones, memory checkpoints, crashes, and
the first divergence between two ROM images.

A *backend* runs a ROM under a fixed configuration and returns a ``Trace``: an ordered list of
milestones, each with the instruction index it was reached at, the PC and two digests (SHA-256 of
all of RDRAM and of the CPU state), plus event counts and how the run stopped.  Outputs are hashes
and counts only.  No screenshot, savestate or memory image is ever written; per-page digests used to
localise a divergence are kept in memory and reported as page addresses.

Backends:

* ``builtin``: the kit's own R4300 interpreter (``n64dk.emu``) from the state IPL3 leaves (the
  first MiB copied to the entry address, boot globals at 0x80000300).  It is always available and
  deterministic: Count advances by a fixed amount per read, RDRAM starts zeroed or filled from a
  fixed seed (``entropy_seed``), and nothing reads the clock.  It reaches the stages the
  interpreter models: the entry routine, PI DMA, decompression and relocation, TLB mapping, the
  first thread dispatch (``eret``) and writes to the VI, SP, MI and AI registers.  It does not model
  interrupts, the RSP or RDP, so a game that waits for a VI or SP interrupt ends in an idle loop
  or exhausts its budget there.  That is the end of what it can validate, reported as such.
* external scriptable emulators (``ares``, ``mupen64plus``, ``simple64``): adapters are declared
  with the binaries they need; ``available()`` is true only when one is installed locally.  None is
  implemented beyond detection: the contract they would have to meet is ``Backend.run``.

Milestones (``Trace.milestones``, in order of instruction index):

``boot``            the IPL3 hand-over state (index 0): RDRAM as IPL3 leaves it
``entry``           the entry routine's first instruction (index 0 on the built-in backend)
``first_dma``       after the first PI DMA completes
``first_vi_write``  the first store to a VI register (the game starting video)
``first_sp_write``  the first store to an SP register (an RSP task about to start)
``handoff``         the first execution of code written after boot (a loader's output)
``main``            the first execution of the hand-over target that ``boot.analyse`` reads
``first_eret``      the first thread dispatch (libultra's scheduler entering a thread)
``instructions:N``  after N instructions (``checkpoints``; measured on a fresh deterministic run)
``watch:<name>``    the first execution of a caller-given address
``end``             where the run stopped

A milestone's ``step`` is the number of instructions executed before it.  For an event inside an
instruction (a DMA, a register store, ``eret``) it is that instruction's index and the state includes
its effect; the interpreter does not track the PC inside an instruction, so it is reported as null
(for ``eret``: the dispatch target).

How a run stopped is classified: ``idle`` (a branch to itself: waiting for an interrupt),
``budget`` (instruction limit), ``exception`` (``syscall``/``break``), ``crash`` (unmapped address,
DMA from outside the ROM), ``emulator-limit`` (an instruction the interpreter does not implement),
``time`` (the wall clock exceeded ``max_seconds``, checked when the run ends).  Reads of open bus are
listed as events.

``first_divergence(rom_a, rom_b)`` runs both images under one configuration, compares the
milestone sequences (name, index, digests), reports the first that differs with the RDRAM pages
that differ there (separating the pages that already differed at boot, i.e. the images' own
difference) and, within an instruction budget, bisects on fresh runs for the first instruction count
after which the CPU states differ.  Both ROMs are supplied by the user locally.  The report holds
addresses, indices and hashes, never bytes.

All of this is evidence about behaviour on the kit's interpreter, never byte evidence and never a
verification verdict.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
import time
from dataclasses import dataclass, field

from .. import emu
from ..rom import Rom

M32, M64 = 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF
PAGE = 0x1000
SCHEMA = "n64dk.funcval/1"
REGIONS = {"vi": (0x04400000, 0x04400038), "sp": (0x04040000, 0x04040020), "mi": (0x04300000, 0x04300010),
           "ai": (0x04500000, 0x04500018)}


@dataclass(frozen=True)
class Config:
    max_instructions: int = 20_000_000
    max_seconds: float = 900.0
    checkpoints: tuple = ()                 # instruction counts
    entropy_seed: int = 0                   # 0: RDRAM zero-filled; otherwise filled from this seed
    watch: tuple = ()                       # (name, vram) pairs
    ram_size: int = emu.RDRAM_SIZE
    main: bool = True                       # read the hand-over target with boot.analyse

    def as_dict(self) -> dict:
        return {"max_instructions": self.max_instructions, "max_seconds": self.max_seconds,
                "checkpoints": list(self.checkpoints), "entropy_seed": self.entropy_seed,
                "watch": [[n, f"0x{a & M32:08X}"] for n, a in self.watch], "ram_size": self.ram_size,
                "main": self.main}


@dataclass
class Milestone:
    name: str
    step: int                               # instructions executed before it (see the module docstring)
    pc: int | None                          # None inside an instruction (a store's side effect)
    rdram: str
    cpu: str

    def as_dict(self) -> dict:
        return {"name": self.name, "step": self.step, "pc": None if self.pc is None else f"0x{self.pc & M32:08X}",
                "rdram_sha256": self.rdram, "cpu_sha256": self.cpu}


@dataclass
class Trace:
    backend: str
    rom_sha256: str
    config: dict
    milestones: list = field(default_factory=list)
    events: dict = field(default_factory=dict)
    stop: dict = field(default_factory=dict)
    cache_key: str = ""
    pages: dict = field(default_factory=dict)          # milestone name -> page digests (in memory only)

    def milestone(self, name: str) -> Milestone | None:
        return next((m for m in self.milestones if m.name == name), None)

    def as_dict(self) -> dict:
        return {"schema": SCHEMA, "backend": self.backend, "rom_sha256": self.rom_sha256, "config": self.config,
                "milestones": [m.as_dict() for m in self.milestones], "events": self.events, "stop": self.stop,
                "cache_key": self.cache_key, "strict": False}


def classify_stop(reason: str) -> str:
    # Machine.run renames an idle loop reached after code written at run time ("ran past the first
    # handoff"); with an unbounded after-handoff budget, as used here, that is its only meaning
    if reason in ("idle loop", "ran past the first handoff"):
        return "idle"
    if reason == "step budget exhausted":
        return "budget"
    if reason in ("syscall", "break"):
        return "exception"
    if reason.startswith("unimplemented") or reason.startswith("FPU") or "ldl/ldr" in reason:
        return "emulator-limit"
    if reason.startswith("unmapped address") or reason.startswith("PI DMA from non-ROM") or \
            reason == "address error":
        return "crash"
    return "other"


def cpu_digest(m, pc: int | None) -> str:
    """SHA-256 of the general, HI/LO and floating-point registers, the FP condition and ``pc``
    (None for a state inside an instruction, where the interpreter does not track the PC)."""
    regs = [r & M64 for r in m.gpr] + [m.hi & M64, m.lo & M64, (pc if pc is not None else M64) & M64] + \
        [f & M64 for f in m.fpr]
    return hashlib.sha256(struct.pack(f">{len(regs)}Q", *regs) + bytes([int(m.fcc)])).hexdigest()


def page_digests(ram) -> list[bytes]:
    view = memoryview(ram)
    return [hashlib.sha256(view[i:i + PAGE]).digest()[:8] for i in range(0, len(ram), PAGE)]


def _fill(m, seed: int, rom: Rom) -> None:
    """Fill RDRAM outside the boot image and the boot globals with bytes derived from ``seed``."""
    phys = rom.entry_vram & 0x1FFFFFFF
    keep = [(0x300, 0x320), (phys, phys + 0x100000)]
    out = bytearray()
    counter = 0
    while len(out) < len(m.ram):
        out += hashlib.sha256(struct.pack(">QQ", seed, counter)).digest()
        counter += 1
    saved = [(a, bytes(m.ram[a:b])) for a, b in keep]
    m.ram[:] = out[:len(m.ram)]
    for a, data in saved:
        m.ram[a:a + len(data)] = data


class _Observed(emu.Machine):
    """The interpreter with observation hooks: first execution of watched addresses and of written
    code (in ``translate``, which runs before an address first executes), PI DMA, hardware register
    stores, thread dispatch."""

    def __init__(self, rom: Rom, ram_size: int, watch: dict[int, str]):
        super().__init__(rom, ram_size=ram_size)
        self.watch = watch
        self.found: list[tuple] = []           # (name, step, pc, rdram, cpu, pages)
        self.names: set[str] = set()
        self.fresh = True
        self.counts = {"vi_writes": 0, "sp_writes": 0, "mi_writes": 0, "ai_writes": 0}

    def record(self, name: str, step: int, pc: int | None) -> None:
        if name in self.names:
            return
        self.names.add(name)
        self.found.append((name, step, pc, hashlib.sha256(self.ram).hexdigest(), cpu_digest(self, pc),
                           page_digests(self.ram)))

    def translate(self, pc: int):
        fn = super().translate(pc)
        # run() sets self.steps just before each instruction executes: here it still holds the
        # previous instruction's index, except before the very first instruction of a fresh machine
        step = self.steps if self.fresh else self.steps + 1
        self.fresh = False
        name = self.watch.get(pc & M32)
        if name is not None:
            self.record(name, step, pc & M32)
        if "handoff" not in self.names:
            try:
                p = self._phys(pc)
            except emu.Stop:
                p = None
            if p is not None and p < self.ram_size and self.dirty[p >> emu.LINE]:
                self.record("handoff", step, pc & M32)
        return fn

    def _pi_dma(self) -> None:
        super()._pi_dma()
        self.record("first_dma", self.steps, None)

    def write(self, vaddr: int, size: int, value: int) -> None:
        super().write(vaddr, size, value)
        try:
            p = self._phys(vaddr)
        except emu.Stop:
            return
        for name, (lo, hi) in REGIONS.items():
            if lo <= p < hi:
                self.counts[f"{name}_writes"] += 1
                if name in ("vi", "sp"):
                    self.record(f"first_{name}_write", self.steps, None)

    def eret(self, pc: int) -> int:
        r = super().eret(pc)
        self.record("first_eret", self.steps, self.jump_now & M32)
        return r


class Backend:
    name = "abstract"

    def available(self) -> bool:
        raise NotImplementedError

    def run(self, rom: Rom, config: Config = Config()) -> Trace:
        raise NotImplementedError


class BuiltinBackend(Backend):
    """The kit's interpreter (see the module docstring)."""
    name = "builtin"

    def available(self) -> bool:
        return True

    def machine(self, rom: Rom, config: Config, watch: dict[int, str] | None = None) -> _Observed:
        m = _Observed(rom, config.ram_size, watch or {})
        if config.entropy_seed:
            _fill(m, config.entropy_seed, rom)
        return m

    def state_at(self, rom: Rom, config: Config, steps: int) -> tuple[str, int, str, list]:
        """(CPU digest, PC, RDRAM digest, page digests) after exactly ``steps`` instructions, or at the
        stop if the run stops earlier; a fresh run, so the result is the deterministic state."""
        m = self.machine(rom, config)
        if steps:
            m.run(max_steps=steps, stop_after_handoff=1 << 62)
        return cpu_digest(m, m.pc & M32), m.pc & M32, hashlib.sha256(m.ram).hexdigest(), page_digests(m.ram)

    def run(self, rom: Rom, config: Config = Config()) -> Trace:
        watch = {a & M32: f"watch:{n}" for n, a in config.watch}
        if config.main:
            try:
                from .. import boot
                main = boot.analyse(rom).main_call
            except Exception:                                  # noqa: BLE001 - the milestone is optional
                main = None
            if main is not None:
                watch.setdefault(main & M32, "main")
        entry = rom.entry_vram & M32
        trace = Trace(self.name, hashlib.sha256(rom.data).hexdigest(), config.as_dict())
        keyed = {k: v for k, v in config.as_dict().items() if k != "max_seconds"}
        trace.cache_key = hashlib.sha256(json.dumps({"schema": SCHEMA, "backend": self.name, "rom": trace.rom_sha256,
                                                     "config": keyed}, sort_keys=True).encode()).hexdigest()
        m = self.machine(rom, config, watch)
        m.record("boot", 0, entry)
        m.names.add("entry")
        m.found.append(("entry",) + m.found[-1][1:])           # the same state: IPL3 jumps to the entry
        started = time.monotonic()
        res = m.run(max_steps=config.max_instructions, stop_after_handoff=1 << 62)
        elapsed = time.monotonic() - started
        m.record("end", res.steps, m.pc & M32)
        found, counts = list(m.found), dict(m.counts)
        del m
        for n in sorted(set(config.checkpoints)):
            if 0 < n <= res.steps:
                cpu, pc, rdram, pages = self.state_at(rom, config, n)
                found.append((f"instructions:{n}", n, pc, rdram, cpu, pages))
        order = {f[0]: (1 if f[0] == "end" else 0, i) for i, f in enumerate(found)}
        found.sort(key=lambda f: (f[1], order[f[0]]))
        for name, step, pc, rdram, cpu, pages in found:
            trace.milestones.append(Milestone(name, step, pc, rdram, cpu))
            trace.pages[name] = pages
        reason = res.stop_reason
        trace.stop = {"reason": reason, "class": "time" if elapsed > config.max_seconds else classify_stop(reason),
                      "pc": f"0x{res.stop_pc:08X}", "steps": res.steps}
        trace.events = {"dmas": len(res.dmas), "dma_bytes": sum(d.size for d in res.dmas),
                        "first_dmas": [{"rom": f"0x{d.rom:08X}", "size": d.size, "ram": f"0x{d.ram:08X}",
                                        "step": d.step} for d in res.dmas[:16]],
                        **counts, "erets": len(res.erets),
                        "first_erets": [[s, f"0x{t:08X}"] for s, t in res.erets[:16]],
                        "probes": [f"0x{p:08X}" for p in res.probes[:16]],
                        "written_bytes": sum(b - a for a, b in res.written)}
        return trace


class ExternalBackend(Backend):
    """A scriptable emulator, used only when installed locally.  Detection only: no adapter is
    implemented, because none of these emulators is installed to validate one against."""

    def __init__(self, name: str, binaries: tuple[str, ...]):
        self.name, self.binaries = name, binaries

    def found(self) -> str | None:
        return next((path for b in self.binaries if (path := shutil.which(b))), None)

    def available(self) -> bool:
        return False

    def run(self, rom: Rom, config: Config = Config()) -> Trace:
        raise NotImplementedError(f"{self.name}: no adapter implemented (binary: {self.found()})")


EXTERNAL = (ExternalBackend("ares", ("ares",)), ExternalBackend("mupen64plus", ("mupen64plus",)),
            ExternalBackend("simple64", ("simple64", "simple64-gui")))


def backends() -> list[dict]:
    """Every backend, whether it can run here, and whether its binary is installed (read-only)."""
    out = [{"name": "builtin", "available": True, "installed": True}]
    for b in EXTERNAL:
        out.append({"name": b.name, "available": b.available(), "installed": b.found() is not None,
                    "note": "adapter not implemented" if b.found() else "not installed"})
    return out


def _pages(indices) -> list[str]:
    return [f"0x{0x80000000 | (i * PAGE):08X}" for i in indices]


def first_divergence(rom_a: Rom, rom_b: Rom, config: Config = Config(), bisect_budget: int = 20_000_000,
                     backend: BuiltinBackend | None = None) -> dict:
    """Compare two images under one configuration (see the module docstring).  ``bisect_budget``
    bounds the instructions the bisection may emulate in total over both images (0: no bisection)."""
    backend = backend or BuiltinBackend()
    ta, tb = backend.run(rom_a, config), backend.run(rom_b, config)
    image = {i for i, (x, y) in enumerate(zip(ta.pages["boot"], tb.pages["boot"])) if x != y}
    out = {"schema": SCHEMA, "config": config.as_dict(), "a": ta.as_dict(), "b": tb.as_dict(),
           "image_pages_differing": len(image), "image_pages_first": _pages(sorted(image)[:16]),
           "stops_differ": ta.stop != tb.stop, "first_milestone": None, "first_cpu_milestone": None,
           "first_instruction": None, "strict": False}
    pairs = list(zip(ta.milestones, tb.milestones))
    first = next((i for i, (x, y) in enumerate(pairs) if (x.name, x.step, x.cpu, x.rdram) != (y.name, y.step,
                                                                                            y.cpu, y.rdram)), None)
    if first is None and len(ta.milestones) != len(tb.milestones):
        first = len(pairs)
    if first is not None:
        ma = ta.milestones[first] if first < len(ta.milestones) else None
        mb = tb.milestones[first] if first < len(tb.milestones) else None
        entry = {"index": first, "a": ma and ma.as_dict(), "b": mb and mb.as_dict()}
        if ma and mb and ma.name == mb.name:
            differing = [i for i, (x, y) in enumerate(zip(ta.pages[ma.name], tb.pages[mb.name])) if x != y]
            beyond = [i for i in differing if i not in image]
            entry.update({"pages_differing": len(differing), "pages_beyond_image_difference": len(beyond),
                          "first_pages_beyond": _pages(beyond[:16])})
        out["first_milestone"] = entry
    cpu_first = next((i for i, (x, y) in enumerate(pairs) if (x.name, x.step, x.cpu) != (y.name, y.step, y.cpu)),
                     None)
    candidates = []
    if cpu_first is not None:
        x, y = pairs[cpu_first]
        out["first_cpu_milestone"] = {"index": cpu_first, "a": x.as_dict(), "b": y.as_dict()}
        candidates = [x.step, y.step]
    candidates += [ta.stop["steps"], tb.stop["steps"]]
    if bisect_budget and (cpu_first is not None or out["stops_differ"]):
        out["first_instruction"] = _bisect(backend, rom_a, rom_b, config, sorted(set(candidates)), bisect_budget)
    return out


def _bisect(backend: BuiltinBackend, rom_a: Rom, rom_b: Rom, config: Config, candidates: list[int],
            budget: int) -> dict:
    """The smallest instruction count after which the two CPU states differ, by bisection over fresh
    deterministic runs, bracketed by the first candidate count at which they differ."""
    spent = 0

    def differs(n: int) -> bool:
        nonlocal spent
        spent += 2 * n
        return backend.state_at(rom_a, config, n)[0] != backend.state_at(rom_b, config, n)[0]

    if differs(0):
        return {"instructions": 0, "exact": True, "note": "the CPU states differ at the entry", "emulated": spent}
    hi = None
    for n in candidates:
        if n and spent + 2 * n <= budget and differs(n):
            hi = n
            break
    if hi is None:
        return {"instructions": None, "exact": False, "emulated": spent,
                "note": "no candidate count with differing CPU states within the budget"}
    lo = 0                                                   # the states agree after lo, differ after hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if spent + 2 * mid > budget:
            return {"instructions": None, "exact": False, "bracket": [lo, hi], "emulated": spent,
                    "note": "bisection budget exhausted"}
        if differs(mid):
            hi = mid
        else:
            lo = mid
    spent += lo
    return {"instructions": hi, "exact": True, "diverging_index": lo,
            "diverging_pc": f"0x{backend.state_at(rom_a, config, lo)[1]:08X}", "emulated": spent}
