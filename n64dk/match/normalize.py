"""Assembly normalizers placed between GCC ``-S`` and an old GNU assembler.
``builtin`` makes KMC GCC 2.7.2 output deterministic for GNU as 2.x in the way the
BattleTanx project's normalizer starts out (tools/normalize_kmc_gcc_asm.py, GPL-3.0): GCC
already wraps the transfers whose delay slots it filled in ``.set noreorder`` regions, so
assembler reordering is disabled everywhere else and an explicit ``nop`` is added after each
transfer emitted in reorder mode.  GCC's internal ``b 3f`` of the 64-bit shift helper is
spelled ``bgez $zero,3f`` as the retail assembler expanded it.

Project-specific normalizers run as external hooks (see :func:`run_hook`).
"""
from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

TRANSFER = re.compile(
    r"^(?:b(?:al|eq|ne|gez|gezal|gtz|lez|ltz|ltzal|c1[ft]|c1fl|c1tl|eql|nel|gezl|gtzl|lezl|ltzl)?|"
    r"j|jal|jr|jalr)\s+")
LONG_SHIFT_BRANCH = re.compile(r"^(\s*)b(\s+)3f\b")


def builtin(text: str) -> str:
    out: list[str] = []
    reorder = True
    for line in text.splitlines(keepends=True):
        body = line.split("#", 1)[0].strip()
        if body == ".set\tnoreorder" or re.fullmatch(r"\.set\s+noreorder", body):
            reorder = False
            out.append(line)
            continue
        if re.fullmatch(r"\.set\s+reorder", body):
            reorder = True
            continue  # keep the assembler in noreorder mode
        if reorder and LONG_SHIFT_BRANCH.match(line):
            line = LONG_SHIFT_BRANCH.sub(r"\1bgez\2$0,3f", line, count=1)
            body = line.split("#", 1)[0].strip()
        out.append(line)
        if reorder and TRANSFER.match(body):
            out.append("\tnop\n")
    return "\t.set\tnoreorder\n" + "".join(out)


def run_hook(command: str, src: Path, dst: Path, cwd: Path, extra: list[str] | None = None) -> None:
    """Run an external normalizer.  ``command`` is a shell-style template with ``{input}``,
    ``{output}`` and optional ``{extra}`` placeholders; without placeholders the input and
    output paths are appended."""
    argv = shlex.split(command)
    extra = extra or []
    if not any("{input}" in a or "{output}" in a for a in argv):
        argv = argv + ["{extra}", "{input}", "{output}"]
    final: list[str] = []
    for a in argv:
        if a == "{extra}":
            final.extend(extra)
        else:
            final.append(a.replace("{input}", str(src)).replace("{output}", str(dst)))
    try:
        from .. import safety
        proc = safety.run_tool(final, kind="light", cwd=cwd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("normalizer timed out after 120s") from error
    if proc.returncode:
        raise RuntimeError(f"normalizer failed: {proc.stderr.strip()[-400:]}")


_SYMBOL_LOAD = re.compile(r"^(\s*)(lw|lh|lhu|lb|lbu)(\s+)(\$\d+|\$[a-z][a-z0-9]),\s*([A-Za-z_.$][\w.$]*(?:[+-]\d+)?)\s*(#.*)?$")


def at_symbol_loads(text: str) -> str:
    """Assembler component variant ``atload``: an integer load from a bare symbol (``lw $4,g``,
    which GNU as, KMC as, ASN64 and IRIX as all expand through the destination register) expanded
    through ``$at`` instead: ``lui $at,%hi(g)`` then ``lw $4,%lo(g)($at)``.  Retail code of some
    titles holds exactly this form (round 11, TOOLCHAIN_REGIONS.md: no installed assembler makes
    it).  Behaviour-preserving: the same value reaches the same register; ``$at`` is the
    assembler's temporary.  Everything else is left for the assembler."""
    out = []
    for line in text.splitlines(keepends=True):
        m = _SYMBOL_LOAD.match(line.rstrip("\n"))
        if not m:
            out.append(line)
            continue
        indent, op, _space, reg, sym, _comment = m.groups()
        out.append(f"{indent}.set\tnoat\n{indent}lui\t$1,%hi({sym})\n{indent}{op}\t{reg},%lo({sym})($1)\n"
                   f"{indent}.set\tat\n")
    return "".join(out)


# --------------------------------------------------------------------------- branch macros (round 12)
_INSN = re.compile(r"^\s*([a-z][a-z0-9.]*)\s*(.*?)\s*(#.*)?$")
_LABEL = re.compile(r"^\s*([$.\w]+):\s*$")
_REG = re.compile(r"\$(\d+|[a-z][a-z0-9])\b")
_IS_REG = re.compile(r"\$(\d+|[a-z][a-z0-9]|zero)$")           # a whole operand naming a register ($L5 is a label)
_IMMEDIATE = re.compile(r"-?(0[xX][0-9a-fA-F]+|\d+)$")          # a literal the branch macros take as it stands
_NAMED = {"zero": 0, "at": 1, "v0": 2, "v1": 3, "a0": 4, "a1": 5, "a2": 6, "a3": 7, "t0": 8, "t1": 9, "t2": 10,
          "t3": 11, "t4": 12, "t5": 13, "t6": 14, "t7": 15, "s0": 16, "s1": 17, "s2": 18, "s3": 19, "s4": 20,
          "s5": 21, "s6": 22, "s7": 23, "t8": 24, "t9": 25, "k0": 26, "k1": 27, "gp": 28, "sp": 29, "fp": 30,
          "s8": 30, "ra": 31}
_BRANCH_TO = {"beq", "bne", "beqz", "bnez", "blez", "bgtz", "bltz", "bgez", "beql", "bnel", "blezl", "bgtzl", "bltzl",
              "bgezl", "bc1t", "bc1f", "bc1tl", "bc1fl", "blt", "bge", "bgt", "ble", "bltu", "bgeu", "bgtu", "bleu"}
_LIKELY = {"beql", "bnel", "blezl", "bgtzl", "bltzl", "bgezl", "bc1tl", "bc1fl"}
# control flow the scan does not follow (a local subroutine call, an exception return): live
_OPAQUE = {"bal", "bgezal", "bltzal", "bgezall", "bltzall", "eret"}
# live where the function returns: the results and the callee-saved registers
_LIVE_AT_RETURN = {2, 3, 16, 17, 18, 19, 20, 21, 22, 23, 28, 29, 30, 31}
_MACRO_OF = {("slt", "bne"): "blt", ("slt", "beq"): "bge", ("sltu", "bne"): "bltu", ("sltu", "beq"): "bgeu",
             ("slti", "bne"): "blt", ("slti", "beq"): "bge", ("sltiu", "bne"): "bltu", ("sltiu", "beq"): "bgeu"}


def _regs_of(text: str) -> list[int]:
    out = []
    for name in _REG.findall(text):
        out.append(int(name) if name.isdigit() else _NAMED.get(name, -1))
    return out


# instructions whose first operand is a general register they write in full (everything else is
# treated as reading every register it names and writing none: unknown forms are live, never dead;
# lwl/lwr/ldl/ldr merge into their destination and are left out on purpose)
_WRITES_FIRST = {"addu", "addiu", "subu", "and", "andi", "or", "ori", "xor", "xori", "nor", "sll", "srl", "sra",
                 "sllv", "srlv", "srav", "slt", "slti", "sltu", "sltiu", "lui", "li", "la", "move", "negu", "neg",
                 "not", "lw", "lh", "lhu", "lb", "lbu", "lwu", "ld", "mfhi", "mflo", "mfc1", "dmfc1",
                 "cfc1", "add", "addi", "sub", "abs", "daddu", "daddiu", "dsubu", "dsll", "dsrl", "dsra", "dsll32",
                 "dsrl32", "dsra32", "seq", "sne", "sgt", "sge", "sle", "sgtu", "sgeu", "sleu"}
# operands of the full form where it is not three: fewer is the shorthand that also reads the
# destination ("addu $2,$3", "sra $2,4", "negu $2")
_FULL_OPERANDS = {"mfhi": 1, "mflo": 1, "lui": 2, "li": 2, "la": 2, "move": 2, "negu": 2, "neg": 2, "not": 2,
                  "abs": 2, "lw": 2, "lh": 2, "lhu": 2, "lb": 2, "lbu": 2, "lwu": 2, "ld": 2, "mfc1": 2,
                  "dmfc1": 2, "cfc1": 2}
_WRITES_IF_THREE = {"mul", "div", "divu", "rem", "remu", "mulo", "mulou", "ddiv", "ddivu", "dmul"}


def _parse(line: str):
    """(mnemonic, operand list, written register or None, read registers) of one instruction line."""
    m = _INSN.match(line)
    if not m or m.group(1).startswith("."):
        return None
    op, rest = m.group(1), m.group(2)
    operands = [o.strip() for o in rest.split(",")] if rest else []
    regs = [_regs_of(o) for o in operands]
    writes_first = (op in _WRITES_FIRST or (op in _WRITES_IF_THREE and len(operands) == 3)) and operands \
        and ";" not in rest and operands[0].startswith("$") and len(regs[0]) == 1 \
        and not operands[0].startswith("$f")
    if not writes_first:
        written, reads = None, [r for rs in regs for r in rs]
    else:
        written = regs[0][0]
        reads = [r for rs in regs[1:] for r in rs]
        if len(operands) < _FULL_OPERANDS.get(op, 3):       # shorthand: the destination is read too
            reads.append(written)
    if op in ("jal", "jalr"):                                  # a call reads the argument registers
        reads = reads + [4, 5, 6, 7, 29]
    return op, operands, written, reads


def _set_state(lines: list[str]) -> list[tuple[bool, bool, bool]]:
    """(noreorder, nomacro, noat) in force at each line: the assembler's ``.set`` state is lexical."""
    out, noreorder, nomacro, noat = [], False, False, False
    for line in lines:
        words = line.split("#")[0].split()
        if len(words) == 2 and words[0] == ".set":
            noreorder = {"noreorder": True, "reorder": False}.get(words[1], noreorder)
            nomacro = {"nomacro": True, "macro": False}.get(words[1], nomacro)
            noat = {"noat": True, "at": False}.get(words[1], noat)
        out.append((noreorder, nomacro, noat))
    return out


def _delay_slot(lines: list[str], j: int) -> int | None:
    """Index of the instruction from line ``j`` on (a delay slot), past labels, comments and
    ``.set``/``.loc``; None when anything else (data, the end) comes first."""
    while j < len(lines):
        stripped = lines[j].strip()
        if not stripped or stripped.startswith(("#", ".set", ".loc")) or _LABEL.match(lines[j]):
            j += 1
            continue
        return j if _parse(lines[j]) is not None else None
    return None


def _dead_after(lines: list[str], start: int, reg: int, labels: dict[str, int], seen: set, budget: list,
                state: list[tuple[bool, bool, bool]]) -> bool:
    """Is ``reg`` written before it is read on every path from line ``start``?  Conservative:
    an unknown path (an indirect jump other than the return, a missing label, a local subroutine
    call, the budget) is live.  In ``.set noreorder`` text the instruction after a jump or branch
    is its delay slot: it runs on both paths of a branch, and only on the taken path of a likely
    branch."""
    i = start
    while i < len(lines):
        budget[0] -= 1
        if budget[0] <= 0:
            return False
        if _LABEL.match(lines[i]):
            i += 1
            continue
        parsed = _parse(lines[i])
        if parsed is None:
            if lines[i].strip().startswith(".end"):
                return reg not in _LIVE_AT_RETURN              # end of function
            i += 1
            continue
        op, operands, written, reads = parsed
        if reg in reads or op in _OPAQUE:
            return False
        if op not in _BRANCH_TO and op not in ("j", "jr", "b"):
            if written == reg:
                return True
            i += 1
            continue
        target = operands[-1] if operands else ""
        slot_kills, after = False, i + 1
        if state[i][0]:                                        # noreorder: the next instruction is the delay slot
            s = _delay_slot(lines, i + 1)
            slot = _parse(lines[s]) if s is not None else None
            if slot is None or reg in slot[3] or slot[0] in _OPAQUE or slot[0] in _BRANCH_TO \
                    or slot[0] in ("j", "jr", "b", "jal", "jalr"):
                return False
            slot_kills = slot[2] == reg
            after = s + 1 if op in _LIKELY else s              # a likely branch annuls its slot when not taken
        if op in ("j", "jr") and _IS_REG.match(target):
            # an indirect jump: KMC GCC writes the return as ``j $31``; any other (a jump table) is unknown
            return slot_kills or (target in ("$31", "$ra") and reg not in _LIVE_AT_RETURN)
        if not slot_kills:                                     # the taken path
            if target not in labels:
                return False
            if (target, reg) not in seen:                      # seen: being checked (a loop) or found dead
                seen.add((target, reg))
                if not _dead_after(lines, labels[target], reg, labels, seen, budget, state):
                    return False
        if op in ("j", "b"):
            return True
        i = after                                              # the fall-through path
    return False


def at_branch_macros(text: str) -> str:
    """Assembler component variant ``atbranch``: a comparison into a register that only feeds a
    branch (``slt $2,$4,$5`` then ``bne $2,$0,L``) rewritten as the branch macro (``blt $4,$5,L``),
    which every assembler expands through ``$at``.  Retail code of some titles holds exactly that
    form where KMC GCC writes the comparison into a general register (round 12,
    bench/macros).  Applied only when the register is dead on both paths after the branch, so
    the change is behaviour-preserving, and never under ``.set noreorder`` (explicit delay
    slots), ``.set nomacro`` or ``.set noat`` (``$at`` in use); everything else is left alone."""
    lines = text.splitlines()
    labels: dict[str, int] = {}
    twice = set()
    for index, line in enumerate(lines):
        m = _LABEL.match(line)
        if m:
            if m.group(1) in labels:
                twice.add(m.group(1))
            labels[m.group(1)] = index
    for name in twice:                                         # defined twice (numeric local labels): unknown
        del labels[name]
    state = _set_state(lines)
    out, i = [], 0
    while i < len(lines):
        first = _parse(lines[i]) if not any(state[i]) else None
        second = _parse(lines[i + 1]) if first and i + 1 < len(lines) else None
        if second and (first[0], second[0]) in _MACRO_OF and len(first[1]) == 3 and len(second[1]) == 3:
            dest = first[2]
            a, b, target = first[1][1], first[1][2], second[1][2]
            uses_dest = bool(_IS_REG.match(second[1][0])) and _regs_of(second[1][0]) == [dest] \
                and second[1][1] in ("$0", "$zero")
            plain = bool(_IS_REG.match(a)) and bool(_IS_REG.match(b) or _IMMEDIATE.match(b))
            if dest not in (None, 0, 1) and uses_dest and plain and target in labels:
                # dead on the fall-through path and on the taken path
                if _dead_after(lines, i + 2, dest, labels, set(), [600], state) and \
                        _dead_after(lines, labels[target], dest, labels, set(), [600], state):
                    indent = re.match(r"^(\s*)", lines[i]).group(1)
                    macro = _MACRO_OF[(first[0], second[0])]
                    out.append(f"{indent}{macro}\t{a},{b},{target}")
                    i += 2
                    continue
        out.append(lines[i])
        i += 1
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def at_macros(text: str) -> str:
    """``atload`` and ``atbranch`` together: the assembler variant of the round-12 Batman regions."""
    return at_branch_macros(at_symbol_loads(text))
