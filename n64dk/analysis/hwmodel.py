"""The N64 address space as the hardware sees it: segments, devices, registers, side effects.

Other analyses need two questions answered about an address: *what is there* (cached RDRAM, an
uncached alias of RDRAM, an RCP register, cartridge space, PIF RAM, RSP memory) and *may an access
be removed, reordered, duplicated or replayed* (``is_side_effecting``, ``is_volatile``).  The
behaviour oracle, translation validation and equivalence rewriting must not treat a store to
``VI_ORIGIN`` like a store to a variable.

Sources and confidence
    * MIPS R4300 segments (KUSEG/KSEG0/KSEG1/KSSEG/KSEG3): architectural, **high**.
    * Physical map and RCP register blocks (RDRAM, RDRAM registers, SP DMEM/IMEM, SP, DPC, DPS,
      MI, VI, AI, PI, RI, SI, PIF ROM/RAM) with the register names and offsets of the public
      libultra header ``rcp.h`` (as also documented by n64brew): **high** for block bases and the
      registers listed in ``REGISTERS``.
    * Cartridge domains (domain 2 address 1 = 64DD registers, domain 1 address 1 = 64DD IPL ROM,
      domain 2 address 2 = SRAM/FlashRAM, domain 1 address 2 = cartridge ROM, domain 1 address 3):
      **medium**: the bus layout is public, but what a given cartridge decodes there varies.
    * Side effects: every store to an RCP/RDRAM/SI/PIF register or cartridge space is treated as
      side-effecting; of the loads, only ``SP_SEMAPHORE`` (a read acquires the semaphore) is.
      Reads of other registers are volatile (the value changes under the program) but not
      side-effecting.  **medium**: conservative where the public documentation is silent.

``mmio_accesses(code, vram)`` runs the constant propagation of :mod:`n64dk.analysis.indirect`
over one function and returns every load/store whose address is a device address: exact
(``lui``/``addiu`` pairs, constants in registers), a bounded set, or computed (a device base plus
an unknown index), with the raw instruction kept as evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import flow as flowmod
from .indirect import MASK, MAX_TABLE, _Interp

# --------------------------------------------------------------------------- the map

SEGMENTS = (  # (first, end, name, cached, translated)
    (0x00000000, 0x80000000, "kuseg", None, True),
    (0x80000000, 0xA0000000, "kseg0", True, False),
    (0xA0000000, 0xC0000000, "kseg1", False, False),
    (0xC0000000, 0xE0000000, "ksseg", None, True),
    (0xE0000000, 0x100000000, "kseg3", None, True),
)

# physical [first, end) -> device, kind, confidence
DEVICES = (
    (0x00000000, 0x00800000, "rdram", "memory", "high"),             # 4 MiB, 8 MiB with the expansion pak
    (0x03F00000, 0x04000000, "rdram-regs", "register", "high"),
    (0x04000000, 0x04001000, "sp-dmem", "device-memory", "high"),
    (0x04001000, 0x04002000, "sp-imem", "device-memory", "high"),
    (0x04040000, 0x04040020, "sp", "register", "high"),
    (0x04080000, 0x04080008, "sp-pc", "register", "high"),
    (0x04100000, 0x04100020, "dpc", "register", "high"),
    (0x04200000, 0x04200010, "dps", "register", "high"),
    (0x04300000, 0x04300010, "mi", "register", "high"),
    (0x04400000, 0x04400038, "vi", "register", "high"),
    (0x04500000, 0x04500018, "ai", "register", "high"),
    (0x04600000, 0x04600034, "pi", "register", "high"),
    (0x04700000, 0x04700020, "ri", "register", "high"),
    (0x04800000, 0x0480001C, "si", "register", "high"),
    (0x05000000, 0x06000000, "cart-dom2-addr1", "cartridge", "medium"),   # 64DD registers
    (0x06000000, 0x08000000, "cart-dom1-addr1", "cartridge", "medium"),   # 64DD IPL ROM
    (0x08000000, 0x10000000, "cart-dom2-addr2", "cartridge", "medium"),   # SRAM / FlashRAM
    (0x10000000, 0x1FC00000, "cart-rom", "cartridge", "medium"),          # domain 1 address 2
    (0x1FC00000, 0x1FC007C0, "pif-rom", "pif", "high"),
    (0x1FC007C0, 0x1FC00800, "pif-ram", "pif", "high"),
    (0x1FD00000, 0x80000000, "cart-dom1-addr3", "cartridge", "medium"),
)

REGISTERS = {
    0x03F00000: "RDRAM_CONFIG", 0x03F00004: "RDRAM_DEVICE_ID", 0x03F00008: "RDRAM_DELAY",
    0x03F0000C: "RDRAM_MODE", 0x03F00010: "RDRAM_REF_INTERVAL", 0x03F00014: "RDRAM_REF_ROW",
    0x03F00018: "RDRAM_RAS_INTERVAL", 0x03F0001C: "RDRAM_MIN_INTERVAL", 0x03F00020: "RDRAM_ADDR_SELECT",
    0x03F00024: "RDRAM_DEVICE_MANUF",
    0x04040000: "SP_MEM_ADDR", 0x04040004: "SP_DRAM_ADDR", 0x04040008: "SP_RD_LEN", 0x0404000C: "SP_WR_LEN",
    0x04040010: "SP_STATUS", 0x04040014: "SP_DMA_FULL", 0x04040018: "SP_DMA_BUSY", 0x0404001C: "SP_SEMAPHORE",
    0x04080000: "SP_PC", 0x04080004: "SP_IBIST",
    0x04100000: "DPC_START", 0x04100004: "DPC_END", 0x04100008: "DPC_CURRENT", 0x0410000C: "DPC_STATUS",
    0x04100010: "DPC_CLOCK", 0x04100014: "DPC_BUFBUSY", 0x04100018: "DPC_PIPEBUSY", 0x0410001C: "DPC_TMEM",
    0x04200000: "DPS_TBIST", 0x04200004: "DPS_TEST_MODE", 0x04200008: "DPS_BUFTEST_ADDR",
    0x0420000C: "DPS_BUFTEST_DATA",
    0x04300000: "MI_MODE", 0x04300004: "MI_VERSION", 0x04300008: "MI_INTR", 0x0430000C: "MI_INTR_MASK",
    0x04400000: "VI_STATUS", 0x04400004: "VI_ORIGIN", 0x04400008: "VI_WIDTH", 0x0440000C: "VI_INTR",
    0x04400010: "VI_CURRENT", 0x04400014: "VI_BURST", 0x04400018: "VI_V_SYNC", 0x0440001C: "VI_H_SYNC",
    0x04400020: "VI_LEAP", 0x04400024: "VI_H_START", 0x04400028: "VI_V_START", 0x0440002C: "VI_V_BURST",
    0x04400030: "VI_X_SCALE", 0x04400034: "VI_Y_SCALE",
    0x04500000: "AI_DRAM_ADDR", 0x04500004: "AI_LEN", 0x04500008: "AI_CONTROL", 0x0450000C: "AI_STATUS",
    0x04500010: "AI_DACRATE", 0x04500014: "AI_BITRATE",
    0x04600000: "PI_DRAM_ADDR", 0x04600004: "PI_CART_ADDR", 0x04600008: "PI_RD_LEN", 0x0460000C: "PI_WR_LEN",
    0x04600010: "PI_STATUS", 0x04600014: "PI_BSD_DOM1_LAT", 0x04600018: "PI_BSD_DOM1_PWD",
    0x0460001C: "PI_BSD_DOM1_PGS", 0x04600020: "PI_BSD_DOM1_RLS", 0x04600024: "PI_BSD_DOM2_LAT",
    0x04600028: "PI_BSD_DOM2_PWD", 0x0460002C: "PI_BSD_DOM2_PGS", 0x04600030: "PI_BSD_DOM2_RLS",
    0x04700000: "RI_MODE", 0x04700004: "RI_CONFIG", 0x04700008: "RI_CURRENT_LOAD", 0x0470000C: "RI_SELECT",
    0x04700010: "RI_REFRESH", 0x04700014: "RI_LATENCY", 0x04700018: "RI_RERROR", 0x0470001C: "RI_WERROR",
    0x04800000: "SI_DRAM_ADDR", 0x04800004: "SI_PIF_ADDR_RD64B", 0x04800010: "SI_PIF_ADDR_WR64B",
    0x04800018: "SI_STATUS",
}
READ_SIDE_EFFECTS = {"SP_SEMAPHORE"}            # reading acquires the RSP semaphore

# what software role touching a device suggests (used by subsystems.py)
ROLES = {"vi": "video", "ai": "audio", "si": "input", "pif-ram": "input", "pi": "storage-dma",
         "cart-rom": "storage-dma", "cart-dom2-addr2": "storage-dma", "cart-dom2-addr1": "storage-dma",
         "sp": "rsp-tasks", "sp-pc": "rsp-tasks", "sp-dmem": "rsp-tasks", "sp-imem": "rsp-tasks",
         "dpc": "renderer", "dps": "renderer", "mi": "os-runtime", "ri": "boot", "rdram-regs": "boot",
         "pif-rom": "boot"}

LOADS = {32: 1, 33: 2, 34: 4, 35: 4, 36: 1, 37: 2, 38: 4, 39: 4, 55: 8, 49: 4, 53: 8, 26: 8, 27: 8}
STORES = {40: 1, 41: 2, 42: 4, 43: 4, 46: 4, 63: 8, 57: 4, 61: 8, 44: 8, 45: 8}


@dataclass(frozen=True)
class Region:
    address: int
    segment: str                 # kuseg, kseg0, kseg1, ksseg, kseg3
    physical: int | None         # None when the segment is TLB-translated
    device: str                  # rdram, vi, pi, ..., "tlb-mapped" or "unmapped"
    kind: str                    # memory, register, device-memory, cartridge, pif, mapped, unmapped
    register: str | None = None
    cached: bool | None = None
    confidence: str = "high"

    @property
    def mmio(self) -> bool:
        return self.kind in ("register", "device-memory", "cartridge", "pif")


def segment(address: int) -> tuple[str, bool | None, bool]:
    address &= MASK
    for first, end, name, cached, translated in SEGMENTS:
        if first <= address < end:
            return name, cached, translated
    raise ValueError(address)


def physical(address: int) -> int | None:
    """Physical address of a KSEG0/KSEG1 address; None for TLB-translated segments."""
    address &= MASK
    if 0x80000000 <= address < 0xC0000000:
        return address & 0x1FFFFFFF
    return None


def classify_physical(paddr: int) -> tuple[str, str, str]:
    for first, end, device, kind, confidence in DEVICES:
        if first <= paddr < end:
            return device, kind, confidence
    return "unmapped", "unmapped", "high"


def classify(address: int) -> Region:
    address &= MASK
    name, cached, translated = segment(address)
    if translated:
        return Region(address, name, None, "tlb-mapped", "mapped", None, cached, "high")
    paddr = physical(address)
    device, kind, confidence = classify_physical(paddr)
    register = REGISTERS.get(paddr & ~3) if kind == "register" else None
    return Region(address, name, paddr, device, kind, register, cached, confidence)


def is_volatile(address: int) -> bool:
    """The value may change without this program storing to it: device space, or RDRAM through
    the uncached KSEG1 alias (buffers shared with DMA, the RSP or the RDP)."""
    region = classify(address)
    return region.mmio or (region.device == "rdram" and region.segment == "kseg1")


def is_side_effecting(address: int, store: bool = True) -> bool:
    """An access that changes hardware state: may not be removed, duplicated, reordered or
    replayed.  Stores to registers, PIF RAM and cartridge space; loads of ``SP_SEMAPHORE``.
    Accesses to TLB-mapped addresses are unknown and treated as side-effecting only when the
    caller passes ``store`` and nothing is known (conservative: False for loads)."""
    region = classify(address)
    if region.kind == "mapped":
        return False
    if region.kind in ("register", "pif", "cartridge"):
        return store or region.register in READ_SIDE_EFFECTS
    return False


def describe(address: int) -> str:
    r = classify(address)
    where = r.register or r.device
    return f"0x{address & MASK:08X} {r.segment} {where}" + (f" (physical 0x{r.physical:08X})"
                                                             if r.physical is not None else "")


# --------------------------------------------------------------------------- accesses in code

@dataclass(frozen=True)
class Access:
    site: int
    function: int
    store: bool
    width: int
    form: str                        # exact, set, computed
    addresses: tuple                 # exact or bounded addresses (empty for computed)
    device: str
    register: str | None
    segment: str
    side_effecting: bool
    volatile: bool
    evidence: str                    # raw word, disassembly and base value

    def as_dict(self) -> dict:
        return {"site": f"0x{self.site:08X}", "function": f"0x{self.function:08X}", "store": self.store,
                "width": self.width, "form": self.form, "addresses": [f"0x{a:08X}" for a in self.addresses],
                "device": self.device, "register": self.register, "segment": self.segment,
                "side_effecting": self.side_effecting, "volatile": self.volatile, "evidence": self.evidence}


class _AccessInterp(_Interp):
    """The indirect-control-flow interpreter, recording the base value of every load and store
    before the instruction executes."""

    def __init__(self, fl, read_word, max_table: int = MAX_TABLE):
        super().__init__(fl, read_word, max_table)
        self.accesses: dict[int, tuple] = {}

    def step(self, insn, regs, slots, first):
        w = insn.word
        op = w >> 26
        if insn.op != ".word" and w and (op in LOADS or op in STORES):
            rs = (w >> 21) & 31
            base = ("c", 0) if rs == 0 else regs.get(rs)
            imm = w & 0xFFFF
            self.accesses[insn.vram] = (base, imm - 0x10000 if imm & 0x8000 else imm, op, insn)
        super().step(insn, regs, slots, first)


def _access(site: int, function: int, base, simm: int, op: int, insn) -> Access | None:
    store, width = op in STORES, STORES.get(op) or LOADS.get(op)
    text = f"{insn.word:08X} {insn.text}"
    if base is None:
        return None
    kind = base[0]
    if kind == "c":
        addresses = ((base[1] + simm) & MASK,)
        form = "exact"
        evidence = f"{text}; base {base[1]:#010x}"
    elif kind == "set" and base[3] <= 64:
        addresses = tuple(((base[1] + simm + k * base[2]) & MASK) for k in range(base[3]))
        form = "set"
        evidence = f"{text}; base {base[1]:#010x} + {base[2]}*k, k < {base[3]}"
    elif kind == "lin" and base[1] != ("sp",):
        probe = (base[3] + simm) & MASK
        region = classify(probe)
        if not region.mmio:
            return None
        sid = base[1]
        source = {"in": "a register at block entry", "def": "a value computed in this function",
                  "call": "a call result"}.get(sid[0], "an unknown value")
        return Access(site, function, store, width, "computed", (), region.device, None, region.segment,
                      store and region.kind in ("register", "pif", "cartridge"), True,
                      f"{text}; base {probe:#010x} + {base[2]} * ({source})")
    else:
        return None
    regions = [classify(a) for a in addresses]
    if not any(r.mmio or (r.device == "rdram" and r.segment == "kseg1") for r in regions):
        return None
    first = regions[0]
    registers = {r.register for r in regions if r.register}
    return Access(site, function, store, width, form, addresses, first.device,
                  next(iter(registers)) if len(registers) == 1 else None, first.segment,
                  any(is_side_effecting(a, store) for a in addresses), any(is_volatile(a) for a in addresses),
                  evidence)


def mmio_accesses(code: bytes, vram: int, read_word=None) -> list[Access]:
    """Device and uncached accesses of one function, in address order."""
    code = code[:len(code) & ~3]
    if len(code) < 4:
        return []
    fl = flowmod.analyse(code, vram)
    interp = _AccessInterp(fl, read_word or (lambda _a: None))
    interp.run()
    return accesses_of(interp, vram)


def accesses_of(interp: _AccessInterp, function: int) -> list[Access]:
    out = []
    for site, (base, simm, op, insn) in sorted(interp.accesses.items()):
        access = _access(site, function, base, simm, op, insn)
        if access is not None:
            out.append(access)
    return out


def roles(accesses) -> dict[str, int]:
    """Software role -> number of accesses suggesting it (see ``ROLES``)."""
    out: dict[str, int] = {}
    for access in accesses:
        role = ROLES.get(access.device)
        if role:
            out[role] = out.get(role, 0) + 1
    return out


@dataclass
class Summary:
    accesses: list[Access] = field(default_factory=list)

    def as_dict(self) -> dict:
        by_device: dict[str, int] = {}
        for a in self.accesses:
            by_device[a.device] = by_device.get(a.device, 0) + 1
        return {"accesses": len(self.accesses), "by_device": dict(sorted(by_device.items())),
                "side_effecting": sum(1 for a in self.accesses if a.side_effecting),
                "computed": sum(1 for a in self.accesses if a.form == "computed")}
