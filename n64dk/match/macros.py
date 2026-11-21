"""Declarative assembler-macro model (round 12).

An assembler turns macro instructions (``lw $4,sym``, ``blt $4,$5,L``, ``li $4,0x12345678``,
``div $2,$4,$5``...) into machine instructions, and assemblers differ in how: which register
holds the upper half of an address, whether a branch comparison goes through ``$at``, whether
division is checked, whether small data goes through ``$gp``.  Those differences are exactly the
"assembler component" of a toolchain, and they are measured here rather than guessed:

* ``CORPUS`` declares the macro forms, by family (address materialisation, symbol loads and
  stores, float loads and stores, branch macros, ``li``/``la``/``move``, division and remainder,
  multiplication, unaligned access, set-on-condition, small-data forms under ``-G``);
* ``probe(assembler, flags)`` assembles every form with one installed assembler, in both
  ``.set reorder`` and ``.set noreorder``, and records each expansion as a normalized instruction
  sequence (``Expansion``): the model of that assembler is the resulting table, with provenance
  (the assembler's file digest);
* ``signatures(code, vram)`` counts, in retail machine code, the ``$at`` forms that tell
  assemblers apart (a global loaded through ``$at`` into another register, a branch comparison
  through ``$at``, a float load/store through ``$at``, an indexed access through ``$at``, a
  ``$gp``-relative access), so a region's forms can be compared with what each assembler model
  can produce (``explain``).

Nothing here changes a build; lanes that emulate an assembler variant (``kmc+atload``) are
separate, opt-in recipe components and keep the provenance ``reconstructed``.
"""
from __future__ import annotations

import hashlib
import struct
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

AT = 1
CORPUS: tuple[tuple[str, str, str], ...] = (
    # (family, form, assembly line); "sym" is an external word, "small" a 4-byte .comm under -G
    ("address", "la", "la $4,sym"),
    ("address", "la+offset", "la $4,sym+0x10"),
    ("address", "la-large-offset", "la $4,sym+0x12340"),
    ("load", "lw-symbol", "lw $4,sym"),
    ("load", "lw-symbol+offset", "lw $4,sym+8"),
    ("load", "lh-symbol", "lh $4,sym"),
    ("load", "lbu-symbol", "lbu $4,sym"),
    ("load", "lw-indexed-symbol", "lw $4,sym($5)"),
    ("load", "lw-large-offset", "lw $4,0x12340($5)"),
    ("store", "sw-symbol", "sw $4,sym"),
    ("store", "sh-symbol", "sh $4,sym+2"),
    ("store", "sw-indexed-symbol", "sw $4,sym($5)"),
    ("float", "lwc1-symbol", "lwc1 $f4,sym"),
    ("float", "swc1-symbol", "swc1 $f4,sym"),
    ("float", "l.s-symbol", "l.s $f4,sym"),
    ("float", "l.d-symbol", "l.d $f4,sym"),
    ("float", "s.d-symbol", "s.d $f4,sym"),
    ("float", "li.s", "li.s $f0,1.5"),
    ("float", "li.s-inexact", "li.s $f0,0.1"),
    ("float", "li.d", "li.d $f0,2.25"),
    ("branch", "blt-reg", "blt $4,$5,t_end"),
    ("branch", "bge-reg", "bge $4,$5,t_end"),
    ("branch", "bgtu-reg", "bgtu $4,$5,t_end"),
    ("branch", "blt-imm", "blt $4,10,t_end"),
    ("branch", "bge-imm", "bge $4,10,t_end"),
    ("branch", "bltu-imm", "bltu $4,10,t_end"),     # what kmc+atmacro writes for sltiu (not yet probed)
    ("branch", "bgeu-imm", "bgeu $4,10,t_end"),
    ("branch", "beq-imm", "beq $4,10,t_end"),
    ("branch", "bne-large-imm", "bne $4,0x12345,t_end"),
    ("branch", "ble-imm", "ble $4,10,t_end"),
    ("constant", "li-small", "li $4,100"),
    ("constant", "li-negative", "li $4,-5"),
    ("constant", "li-unsigned16", "li $4,0xFFFF"),
    ("constant", "li-upper", "li $4,0x10000"),
    ("constant", "li-large", "li $4,0x12345678"),
    ("constant", "move", "move $4,$5"),
    ("arith", "div", "div $2,$4,$5"),
    ("arith", "divu", "divu $2,$4,$5"),
    ("arith", "div-machine", "div $0,$4,$5"),
    ("arith", "rem", "rem $2,$4,$5"),
    ("arith", "remu", "remu $2,$4,$5"),
    ("arith", "mul", "mul $2,$4,$5"),
    ("arith", "addu-large-imm", "addu $2,$4,0x12345"),
    ("arith", "abs", "abs $2,$4"),
    ("arith", "neg", "neg $2,$4"),
    ("set", "seq", "seq $2,$4,$5"),
    ("set", "sne", "sne $2,$4,$5"),
    ("set", "sgt", "sgt $2,$4,$5"),
    ("set", "sle", "sle $2,$4,$5"),
    ("unaligned", "ulw", "ulw $4,0($5)"),
    ("unaligned", "usw", "usw $4,0($5)"),
    ("smalldata", "lw-small", "lw $4,small"),
    ("smalldata", "sw-small", "sw $4,small"),
    ("smalldata", "la-small", "la $4,small"),
    ("smalldata", "lwc1-small", "lwc1 $f4,small"),
)


@dataclass
class Expansion:
    family: str
    form: str
    line: str
    mode: str                         # reorder | noreorder
    words: int = 0
    text: list[str] = field(default_factory=list)
    uses_at: bool = False
    uses_gp: bool = False
    error: str = ""


@dataclass
class Model:
    assembler: str
    digest: str
    flags: list[str]
    expansions: list[Expansion]

    def as_dict(self) -> dict:
        return {"assembler": self.assembler, "digest": self.digest, "flags": self.flags,
                "expansions": [asdict(e) for e in self.expansions]}

    def table(self) -> dict[str, str]:
        """form/mode -> expansion text, for comparing models."""
        return {f"{e.form}/{e.mode}": (e.error and f"error: {e.error}") or "; ".join(e.text)
                for e in self.expansions}


def _disassemble(words: list[int], vram: int) -> list[str]:
    import rabbitizer
    out = []
    for i, word in enumerate(words):
        text = rabbitizer.Instruction(word, vram + 4 * i).disassemble()
        out.append(" ".join(text.split()))
    return out


HEADERS = (("\t.comm\tsmall,4,4", "\t.extern\tsym,400"),        # GNU, KMC
           ("\t.comm\tsmall,4", "\t.extern\tsym,400"),
           ("\t.comm\tsmall 4", "\t.extern\tsym 400"),             # IRIX as
           ())


def probe(assembler, flags: list[str], name: str | None = None, timeout: float = 60, convert=None,
          suffix: str = ".o", env: dict | None = None, crlf: bool = False) -> Model:
    """Assemble the corpus with one assembler (each form alone: a form the assembler rejects is
    recorded with its error and does not hide the others).  ``assembler`` is a path or a command
    prefix (``[wibo, asn64.exe]``, ``[ido/cc, -c]``); its last element is the digested file.
    ``convert(obj) -> elf`` turns a foreign object (PSY-Q) into ELF."""
    from ..elf import Elf
    command = [str(a) for a in assembler] if isinstance(assembler, (list, tuple)) else [str(assembler)]
    exe = Path(command[-1] if len(command) == 1 or not command[-1].startswith("-") else command[-2])
    digest = hashlib.sha256(exe.read_bytes()).hexdigest()[:16] if exe.is_file() else "missing"
    model = Model(name or exe.name, digest, list(flags), [])
    import os
    environ = {**os.environ, **(env or {})}

    def assemble(work: str, stem: str, mode: str, header, line: str):
        body = "\n".join(["\t.text", f"\t.set\t{mode}", *header, "\t.globl\tm", "m:", f"\t{line}", "t_end:",
                          "\tnop", ""])
        src, obj = Path(work, f"{stem}.s"), Path(work, f"{stem}{suffix}")
        src.write_bytes(body.replace("\n", "\r\n").encode() if crlf else body.encode())
        # names relative to the working folder: a Windows assembler under a loader takes no Unix paths
        from .. import safety                            # token, own session, memory ceiling, group kill
        proc = safety.run_tool([*command, *flags, "-o", obj.name, src.name], kind="light", text=True,
                               timeout=timeout, cwd=work, env=environ)
        if not proc.returncode and convert is not None and obj.is_file():
            obj = convert(obj)
        message = ((proc.stderr + proc.stdout).strip().splitlines() or ["failed"])[-1][:160]
        return (obj if not proc.returncode and obj.is_file() else None), message

    with tempfile.TemporaryDirectory(prefix="n64dk-macros-") as work:
        header = None
        for candidate in HEADERS:                            # the declaration syntax this assembler takes
            try:
                ok, _message = assemble(work, "header", "reorder", candidate, "nop")
            except (OSError, subprocess.TimeoutExpired):
                ok = None
            if ok is not None:
                header = candidate
                break
        model.flags = list(flags) + ([f"header={'|'.join(h.strip() for h in header)}"] if header else ["header=none"])
        for mode in ("reorder", "noreorder"):
            for index, (family, form, line) in enumerate(CORPUS):
                row = Expansion(family, form, line, mode)
                model.expansions.append(row)
                if header is None:
                    row.error = "no declaration syntax assembles"
                    continue
                try:
                    obj, message = assemble(work, f"{index}{mode}", mode, header, line)
                except (OSError, subprocess.TimeoutExpired) as error:
                    row.error = str(error)[:160]
                    continue
                if obj is None:
                    row.error = message
                    continue
                elf = Elf.load(obj)
                text = elf.section(".text")
                data = text.data if text is not None else b""
                words = [struct.unpack_from(">I", data, i)[0] for i in range(0, len(data) - len(data) % 4, 4)]
                end = next((sym.value for sym in elf.symbols if sym.name == "t_end"), None)
                words, row.error = _trim(words, end)
                row.words = len(words)
                row.text = _disassemble(words, 0)
                row.uses_at = any(_regs(w) & {AT} for w in words)
                row.uses_gp = any(_regs(w) & {28} for w in words)
    return model


def _trim(words: list[int], end: int | None) -> tuple[list[int], str]:
    """(the expansion's words, error): up to the end label when the object keeps it, otherwise with
    trailing nops dropped.  No corpus form expands to nothing, so a label at 0 is not trusted: IRIX
    ``as`` (IDO 5.3) records the local label there (round 12: the probe found no IDO expansion)."""
    if end and end % 4 == 0:
        return words[:end // 4], ""
    words = list(words)
    while len(words) > 1 and words[-1] == 0:
        words.pop()
    return words, "" if words else "empty"


def _regs(word: int) -> set[int]:
    """General registers an instruction word names (fields that are not registers left out)."""
    op, rs, rt = word >> 26, (word >> 21) & 31, (word >> 16) & 31
    if op == 0:
        if (word & 63) in (0x0C, 0x0D):                # syscall/break: a code, not registers
            return set()
        return {rs, rt, (word >> 11) & 31}
    if op in (2, 3):
        return set()
    if op == 1:                                         # REGIMM: rt selects bltz/bgez/bltzal...
        return {rs}
    if op == 17:                                        # COP1: only the moves name a general register
        return {rt} if rs in (0, 1, 2, 4, 5, 6) else set()
    if op in FLOAT_MEM:                                 # lwc1/ldc1/swc1/sdc1: rt is a float register
        return {rs}
    return {rs, rt}


LOADS = {32, 33, 35, 36, 37, 39}
STORES = {40, 41, 43}
FLOAT_MEM = {49, 53, 57, 61}


def signatures(code: bytes, vram: int = 0) -> dict[str, int]:
    """Counts of assembler-distinguishing forms in machine code (straight-line windows):

    * ``at-load-other``: ``lui $at`` then a load ``rD, lo($at)`` with rD != $at (an assembler
      that expands symbol loads through ``$at``; KMC/GNU/ASN64/IRIX ``as`` use rD itself);
    * ``at-load-self``: ``lui rD`` then ``lw rD, lo(rD)`` (the usual expansion; compiler- or
      assembler-made);
    * ``at-store``: ``lui $at`` then a store through ``$at`` (every assembler's store macro);
    * ``at-float``: ``lui $at`` then ``lwc1/ldc1/swc1/sdc1`` through ``$at``;
    * ``at-indexed``: ``lui $at``, ``addu $at, $at, rX``, then an access through ``$at``;
    * ``at-branch``: ``slt/slti/sltu/sltiu $at`` then ``beq/bne $at, $zero`` (a branch macro);
    * ``at-constant``: ``lui $at``/``ori $at`` used as an operand (``li`` through ``$at``);
    * ``gp-relative``: a load, store or ``addiu`` with base ``$gp``.
    """
    words = [struct.unpack_from(">I", code, i)[0] for i in range(0, len(code) - len(code) % 4, 4)]
    out = {k: 0 for k in ("at-load-other", "at-load-self", "at-store", "at-float", "at-indexed", "at-branch",
                          "at-constant", "gp-relative")}
    at_hi = -10
    indexed = -10
    lui_self: dict[int, int] = {}
    slt_at = -10
    for i, w in enumerate(words):
        op, rs, rt, rd, fn = w >> 26, (w >> 21) & 31, (w >> 16) & 31, (w >> 11) & 31, w & 63
        if op == 15:
            if rt == AT:
                at_hi = i
            elif rt:
                lui_self[rt] = i
            continue
        if op == 0 and fn == 0x21 and rd == AT and AT in (rs, rt):
            if i - at_hi <= 3:
                indexed = i
            continue
        if op == 0 and fn in (0x2A, 0x2B) and rd == AT or op in (10, 11) and rt == AT:
            slt_at = i
            continue
        if op in (4, 5) and {rs, rt} == {AT, 0} and i - slt_at <= 2:
            out["at-branch"] += 1
            continue
        if rs == 28 and (op in LOADS | STORES | FLOAT_MEM or op == 9):
            out["gp-relative"] += 1
        if rs == AT and i - max(at_hi, indexed) <= 4:
            if i - indexed <= 3 and (op in LOADS | STORES | FLOAT_MEM):
                out["at-indexed"] += 1
            elif op in FLOAT_MEM:
                out["at-float"] += 1
            elif op in STORES:
                out["at-store"] += 1
            elif op in LOADS and rt != AT:
                out["at-load-other"] += 1
            elif op == 13 and rt != AT:
                out["at-constant"] += 1
            continue
        if op in LOADS and rs == rt and rs in lui_self and i - lui_self[rs] <= 4:
            out["at-load-self"] += 1
        if op == 0 and AT in (rs, rt) and i - at_hi <= 3 and fn in (0x21, 0x23, 0x24, 0x25, 0x2A, 0x2B):
            out["at-constant"] += 1
    return out


# which signature each corpus form produces when an assembler expands it through $at (beq-imm and
# bne-large-imm load the constant into $at and compare against it: not the slt/beq $at form counted)
SIGNATURE_OF_FORM = {"lw-symbol": "at-load-other", "lh-symbol": "at-load-other", "lbu-symbol": "at-load-other",
                     "sw-symbol": "at-store", "sh-symbol": "at-store", "lwc1-symbol": "at-float",
                     "swc1-symbol": "at-float", "l.s-symbol": "at-float", "l.d-symbol": "at-float",
                     "s.d-symbol": "at-float", "lw-indexed-symbol": "at-indexed", "sw-indexed-symbol": "at-indexed",
                     "blt-reg": "at-branch", "bge-reg": "at-branch", "bgtu-reg": "at-branch",
                     "blt-imm": "at-branch", "bge-imm": "at-branch", "bltu-imm": "at-branch",
                     "bgeu-imm": "at-branch", "ble-imm": "at-branch", "lw-small": "gp-relative",
                     "sw-small": "gp-relative", "la-small": "gp-relative", "lwc1-small": "gp-relative"}


def producible(model: Model) -> set[str]:
    """Signatures this assembler model produces from some corpus form."""
    out = set()
    for e in model.expansions:
        if e.error or not e.text:
            continue
        kind = SIGNATURE_OF_FORM.get(e.form)
        if kind == "gp-relative" and e.uses_gp:
            out.add(kind)
        elif kind and e.uses_at:
            if kind == "at-load-other":
                # the destination is $4: through $at only when the load's base is $at
                if any(t.startswith(("lw $a0", "lh $a0", "lbu $a0")) and "($at)" in t for t in e.text):
                    out.add(kind)
            else:
                out.add(kind)
    return out


def explain(region_signatures: dict[str, int], models: list[Model], compiler_emits: set[str] | None = None) -> dict:
    """Which models can produce each signature a region shows (a compiler that writes the
    sequence itself, ``compiler_emits``, covers it whatever the assembler)."""
    compiler_emits = compiler_emits or set()
    out = {}
    for kind, count in region_signatures.items():
        if not count:
            continue
        out[kind] = {"count": count, "compiler": kind in compiler_emits,
                     "assemblers": [m.assembler for m in models if kind in producible(m)]}
    return out
