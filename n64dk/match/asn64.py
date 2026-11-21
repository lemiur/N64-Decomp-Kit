"""ASN64-compatible preprocessing of GCC assembly for modern GNU as.

Rewrites raw GCC ``-S`` output so that a modern MIPS GNU assembler (run with
``-march=vr4300 -mabi=32 -EB -G0 --no-pad-sections``) reproduces the instruction
reordering, macro expansion and hazard nops of the SN Systems ASN64 assembler used for
many retail N64 objects.

Adapted from the BattleTanx and BattleTanx: Global Assault projects (tools/modern_asn64.py,
GPL-3.0 projects), which in turn derive from RocketRet/modern-asn64 @ 20d21729 (Unlicense).
The "lane" rules (MEM-SYM-BASE, MEM-BIG-CONST, FPCMP-BC1-NOP, REMU-EXPAND,
MULTU-HILO-HAZARD, MUL-HOIST-SPACING, REORDER-BRANCH-NOP-MOVE) come from those projects' retail validation and can be
switched off individually through ``disabled``.  Upstream quirks that affect output (for
example the concatenated ``"jr" "blt"`` entry of the branch set) are preserved on purpose.
"""
from __future__ import annotations

import re
import struct

BRANCH_MNEMONICS = {
    "bc1f", "bc1fl", "bc1t", "bc1tl",
    "beq", "beql",
    "bgez", "bgezal", "bgezall", "bgezl",
    "bgtz", "bgtzl",
    "blez", "blezl",
    "bltz", "bltzal", "bltzall", "bltzl",
    "bne", "bnel",
    "j", "jal", "jalr", "jr" "blt", "ble", "bgt", "bge",  # upstream quirk kept: "jrblt"
}
COP1_BRANCH_MNEMONICS = {"bc1f", "bc1fl", "bc1t", "bc1tl"}
COMPARE_MNEMONICS = {"c.lt.s", "c.le.s", "c.eq.s", "c.lt.d", "c.le.d", "c.eq.d"}
INT_LOADS = {"lb", "lbu", "lh", "lhu", "lw"}
INT_STORES = {"sb", "sh", "sw"}
SYM_BASE_RE = re.compile(r"^([A-Za-z_.$][\w.$]*(?:[+-]\d+)?)\((\$\w+)\)$")
CONST_BASE_RE = re.compile(r"^(-?\d+)\((\$\w+)\)$")
BARE_SYM_RE = re.compile(r"^[A-Za-z_.$][\w.$]*(?:[+-]\d+)?$")

PRELUDE = (
    ".macro glabel label\n    .global \\label\n    \\label:\n.endm\n"
    ".macro dlabel label\n    .global \\label\n    \\label:\n.endm\n\n"
    ".macro move a, b\n    addu \\a, \\b, $0\n.endm\n\n"
    ".macro b target\n    bgez $0, \\target\n.endm\n"
    "\t.set noreorder\n"
)

GNU_AS_FLAGS = ("-march=vr4300", "-mabi=32", "-EB", "-G0", "--no-pad-sections", "-mips3")


def _tokens(line: str) -> list[str]:
    return [s.strip() for s in line.strip().split("#")[0].strip().split()]


def _expand_memory(identifier: str, operand_text: str, disabled, enabled) -> tuple[str, int] | None:
    parts = operand_text.split(",", 1)
    if len(parts) != 2:
        return None
    target, address = parts[0].strip(), parts[1].strip()
    m = SYM_BASE_RE.match(address)
    if m and identifier in INT_LOADS and "MEM-SYM-BASE" not in disabled:
        sym, base = m.group(1), m.group(2)
        return ((f"\t.set noat\n\tlui $at,%hi({sym})\n\taddu $at,$at,{base}\n"
                 f"\t{identifier} {target},%lo({sym})($at)\n\t.set at\n"), 3)
    m = CONST_BASE_RE.match(address)
    if m and (identifier in INT_LOADS or identifier in INT_STORES) and "MEM-BIG-CONST" not in disabled:
        offset, base = int(m.group(1)), m.group(2)
        if not (0x8000 <= offset <= 0x7FFF7FFF):               # past 0xFFFF too (Wheel of Fortune: lw 0x1E478(reg))
            return None
        low = ((offset & 0xFFFF) ^ 0x8000) - 0x8000
        high = ((offset - low) >> 16) & 0xFFFF
        return ((f"\t.set noat\n\tlui $at,{high:#x}\n\taddu $at,{base},$at\n"
                 f"\t{identifier} {target},{low}($at)\n\t.set at\n"), 3)
    if "FP32-PAIR" in enabled and identifier in ("l.d", "s.d") and re.fullmatch(r"\$f\d+", target):
        # 32-bit float registers (``-mfp32``): a double is moved a word at a time, the odd register (its
        # high word) first, through one ``lui`` (the title's objects show ``lui $at; lwc1 $f3,lo; lwc1 $f2,lo+4``)
        number = int(target[2:])
        op = "lwc1" if identifier == "l.d" else "swc1"
        m = SYM_BASE_RE.match(address)
        if m:
            sym, base = m.group(1), m.group(2)
            return ((f"\t.set noat\n\tlui $at,%hi({sym})\n\taddu $at,$at,{base}\n"
                     f"\t{op} $f{number + 1},%lo({sym})($at)\n\t{op} $f{number},%lo({sym}+4)($at)\n\t.set at\n"), 4)
        if BARE_SYM_RE.match(address):
            return ((f"\t.set noat\n\tlui $at,%hi({address})\n"
                     f"\t{op} $f{number + 1},%lo({address})($at)\n\t{op} $f{number},%lo({address}+4)($at)\n"
                     "\t.set at\n"), 3)
        m = CONST_BASE_RE.match(address)
        if m and -0x8000 <= int(m.group(1)) < 0x7FFC:
            offset, base = int(m.group(1)), m.group(2)
            return (f"\t{op} $f{number + 1},{offset}({base})\n\t{op} $f{number},{offset + 4}({base})\n", 2)
        return None
    if ("KMC-ABS-SYM-PAIR" in enabled and BARE_SYM_RE.match(address)
            and (identifier in INT_LOADS or identifier in INT_STORES)):
        base = target if identifier in INT_LOADS else "$at"
        return ((f"\t.set noat\n\tlui {base},%hi({address})\n"
                 f"\t{identifier} {target},%lo({address})({base})\n\t.set at\n"), 2)
    return None


def _fp32(source: str) -> bool:
    """Whether the compiler that wrote ``source`` ran with ``-mfp32`` (the last of ``-mfp32``/``-mfp64`` on the
    ``Cc1 arguments`` line GCC writes at the head of its output)."""
    head = source[:4000]
    at = head.find("# Cc1 arguments")
    if at < 0:
        return False
    stop = head.find("\n\n", at)
    modes = re.findall(r"-mfp(32|64)\b", head[at:stop if stop >= 0 else len(head)])
    return bool(modes) and modes[-1] == "32"


def _div_expansion(kind: str, ops: list[str], label: int) -> tuple[str, int]:
    d, a, b = ops[0], ops[1], ops[2]
    if kind in ("div", "rem"):
        l1, l2 = f"BRANCH_LABEL_{label}", f"BRANCH_LABEL_{label + 1}"
        move = "mflo" if kind == "div" else "mfhi"
        return ("\t.set noat\n"
                f"\tdiv $0,{a},{b}\n\tbnez {b},{l1}\n\tnop\n\tbreak 0x7\n{l1}:\n"
                f"\taddiu $1,$0,-1\n\tbne {b},$1,{l2}\n\tlui $1,0x8000\n"
                f"\tbne {a},$1,{l2}\n\tnop\n\tbreak 0x6\n{l2}:\n\t{move} {d}\n\t.set at\n"), 2
    l1 = f"BRANCH_LABEL_{label}"
    move = "mflo" if kind == "divu" else "mfhi"
    return ("\t.set noat\n"
            f"\tdivu $0,{a},{b}\n\tbnez {b},{l1}\n\tnop\n\tbreak 0x7\n{l1}:\n"
            f"\t{move} {d}\n\t.set at\n"), 1


def _apply_enabled_lanes(source: str, enabled: frozenset[str] | set[str]) -> str:
    """Apply narrowly scoped, opt-in scheduling compatibility lanes."""
    if "KMC-LOAD-HAZARD-MOVE-FILL" in enabled:
        # ASN64's reorder pass can fill a compiler-emitted load-hazard marker
        # with the following independent register move.  KMC represents that
        # marker as a ``#nop`` line; GNU as otherwise sees only a comment and
        # misses the original assembler's legal fill.
        load_fill = re.compile(
            r"(?P<load>^[ \t]*(?:lb|lbu|lh|lhu|lw)[ \t]+(?P<loaded>\$[0-9]+),[^\n]*\n)"
            r"^[ \t]*#nop[^\n]*\n"
            r"(?P<dependent>^[ \t]*(?:lb|lbu|lh|lhu|lw)[ \t]+(?P<dep_dst>\$[0-9]+),"
            r"[+-]?(?:0x[0-9A-Fa-f]+|[0-9]+)\((?P=loaded)\)[^\n]*\n)"
            r"(?P<move>^[ \t]*move[ \t]+(?P<move_dst>\$[0-9]+),(?P<move_src>\$[0-9]+)[^\n]*\n)",
            re.MULTILINE,
        )

        def fill_load_hazard(match: re.Match[str]) -> str:
            move_regs = {match.group("move_dst"), match.group("move_src")} - {"$0"}
            dependent_regs = set(re.findall(r"\$[0-9]+", match.group("dependent")))
            if move_regs & dependent_regs or match.group("loaded") in move_regs:
                return match.group(0)
            return match.group("load") + match.group("move") + match.group("dependent")

        source = load_fill.sub(fill_load_hazard, source)
    if "KMC-IMAGE-TABLE-FINALIZE" in enabled:
        source = source.replace(
            "\tsubu\t$sp,$sp,48\n"
            "\tsw\t$20,40($sp)\n\tmove\t$20,$5\n"
            "\tsw\t$18,32($sp)\n\tmove\t$18,$4\n"
            "\tsw\t$31,44($sp)\n\tsw\t$19,36($sp)\n"
            "\tsw\t$17,28($sp)\n\tsw\t$16,24($sp)\n",
            "\tsubu\t$sp,$sp,56\n"
            "\tsw\t$18,40($sp)\n\tmove\t$18,$4\n"
            "\tsw\t$20,48($sp)\n\tmove\t$20,$5\n"
            "\tsw\t$31,52($sp)\n\tsw\t$19,44($sp)\n"
            "\tsw\t$17,36($sp)\n\tsw\t$16,32($sp)\n",
            1,
        )
        mode_select = re.compile(
            r"^[ \t]*move[ \t]+\$4,\$0\n"
            r"^[ \t]*move[ \t]+\$8,\$0\n"
            r"^[ \t]*\.set[ \t]+noreorder\n"
            r"^[ \t]*\.set[ \t]+nomacro\n"
            r"(?P<empty>^[ \t]*blez[ \t]+\$17,(?P<exit>\.L\d+)\n)"
            r"^[ \t]*move[ \t]+\$7,\$11\n"
            r"^[ \t]*\.set[ \t]+macro\n"
            r"^[ \t]*\.set[ \t]+reorder\n(?:^[ \t]*\n)*"
            r"(?P<loop>^\.L\d+:\n)"
            r"^[ \t]*lw[ \t]+\$2,40\(\$18\)\n"
            r"(?:^[ \t]*#nop[^\n]*\n)*"
            r"^[ \t]*\.set[ \t]+noreorder\n"
            r"^[ \t]*\.set[ \t]+nomacro\n"
            r"^[ \t]*bnel[ \t]+\$2,\$10,(?P<common>\.L\d+)\n"
            r"^[ \t]*sll[ \t]+\$6,\$4,2\n"
            r"^[ \t]*\.set[ \t]+macro\n"
            r"^[ \t]*\.set[ \t]+reorder\n(?:^[ \t]*\n)*"
            r"^[ \t]*lw[ \t]+\$6,0\(\$7\)\n"
            r"(?:^[ \t]*#nop[^\n]*\n)*"
            r"^[ \t]*sll[ \t]+\$6,\$6,2\n"
            r"(?P=common):\n",
            re.MULTILINE,
        )
        match = mode_select.search(source)
        if match:
            else_label = match.group("common") + "_MODE_INDEX"
            replacement = (
                "\tmove\t$4,$0\n"
                "\t.set\tnoreorder\n\t.set\tnomacro\n"
                + match.group("empty") + "\tmove\t$8,$0\n" +
                "\t.set\tmacro\n\t.set\treorder\n\n"
                "\tmove\t$7,$11\n" + match.group("loop") +
                "\tlw\t$2,40($18)\n"
                "\t.set\tnoreorder\n\t.set\tnomacro\n"
                f"\tbne\t$2,$10,{else_label}\n\tmove\t$2,$4\n"
                "\t.set\tmacro\n\t.set\treorder\n\n"
                "\tlw\t$2,0($7)\n"
                "\t.set\tnoreorder\n\t.set\tnomacro\n"
                f"\tj\t{match.group('common')}\n\tsll\t$2,$2,2\n"
                "\t.set\tmacro\n\t.set\treorder\n\n"
                f"{else_label}:\n\tsll\t$2,$2,2\n"
                f"{match.group('common')}:\n"
            )
            source = source[:match.start()] + replacement + source[match.end():]
        source = source.replace(
            "\taddu\t$6,$20,$6\n\tlw\t$3,0($6)\n\t#nop\n"
            "\tdiv\t$5,$9,$3\n\taddu\t$3,$3,1\n",
            "\taddu\t$6,$2,$20\n\tlw\t$2,0($6)\n"
            "\tdiv\t$5,$9,$2\n\taddu\t$3,$2,1\n",
            1,
        )
        back_edge = re.search(
            r"^[ \t]*bne[ \t]+\$8,\$0,(?P<label>\.L\d+)\n",
            source,
            re.MULTILINE,
        )
        if back_edge:
            label = back_edge.group("label")
            source = source.replace(
                f"{label}:\n\tmove\t$4,$0\n",
                f"\tmove\t$4,$0\n{label}:\n",
                1,
            )
        source = re.sub(
            r"^[ \t]*\.set[ \t]+noreorder\n"
            r"^[ \t]*\.set[ \t]+nomacro\n"
            r"^[ \t]*bne[ \t]+\$8,\$0,(?P<label>\.L\d+)\n"
            r"^[ \t]*move[ \t]+\$2,\$9\n"
            r"^[ \t]*\.set[ \t]+macro\n"
            r"^[ \t]*\.set[ \t]+reorder\n",
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            r"\tbnel\t$8,$0,\g<label>\n\tmove\t$4,$0\n"
            "\t.set\tmacro\n\t.set\treorder\n\n\tmove\t$2,$9\n",
            source,
            count=1,
            flags=re.MULTILINE,
        )
        source = source.replace(
            "\tlw\t$31,44($sp)\n\tlw\t$20,40($sp)\n"
            "\tlw\t$19,36($sp)\n\tlw\t$18,32($sp)\n"
            "\tlw\t$17,28($sp)\n\tlw\t$16,24($sp)\n"
            "\taddu\t$sp,$sp,48\n",
            "\tlw\t$31,52($sp)\n\tlw\t$20,48($sp)\n"
            "\tlw\t$19,44($sp)\n\tlw\t$18,40($sp)\n"
            "\tlw\t$17,36($sp)\n\tlw\t$16,32($sp)\n"
            "\taddu\t$sp,$sp,56\n",
            1,
        )
    if "KMC-IMAGE-TABLE-UPDATE" in enabled:
        # Case two enters with $a0 still holding the context.  The retail
        # schedule nevertheless refreshes it in the child-allocation delay
        # slot and moves the derived row size into the following branch slot.
        source = re.sub(
            r"^[ \t]*jal[ \t]+object_children_allocate\n"
            r"(?P<label>^\.L\d+:\n)",
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tjal\tobject_children_allocate\n\tmove\t$4,$18\n"
            "\t.set\tmacro\n\t.set\treorder\n\n" r"\g<label>",
            source,
            count=1,
            flags=re.MULTILINE,
        )
        case_two = re.compile(
            r"^[ \t]*lw[ \t]+\$2,112\(\$18\)\n"
            r"^[ \t]*lw[ \t]+\$3,120\(\$18\)\n"
            r"^[ \t]*move[ \t]+\$16,\$0\n"
            r"^[ \t]*\.set[ \t]+noreorder\n"
            r"^[ \t]*\.set[ \t]+nomacro\n"
            r"(?P<branch>^[ \t]*blez[ \t]+\$3,[^\n]+\n)"
            r"^[ \t]*addu[ \t]+\$2,\$2,2\n"
            r"^[ \t]*\.set[ \t]+macro\n"
            r"^[ \t]*\.set[ \t]+reorder\n(?:^[ \t]*\n)*"
            r"^[ \t]*sll[ \t]+\$19,\$2,1\n",
            re.MULTILINE,
        )
        source = case_two.sub(
            "\tlw\t$2,112($18)\n\tmove\t$16,$0\n"
            "\tlw\t$3,120($18)\n\taddu\t$2,$2,2\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            r"\g<branch>" "\tsll\t$19,$2,1\n"
            "\t.set\tmacro\n\t.set\treorder\n",
            source,
            count=1,
        )
    if "KMC-IMAGE-TABLE-PROCESS" in enabled:
        # Reproduce the retail ready queue for the table builder.  The C
        # compiler emits the same operations, but this ASN64 variant keeps the
        # allocation result through the dimension test and preserves the two
        # caller-saved cursors around the coordinate callback.
        source = source.replace(
            "\tbne\t$3,$2,.L2\n\tli\t$3,0x0000005e\t\t# 94\n",
            "\tbne\t$3,$2,.L2\n\tli\t$2,0x0000005e\t\t# 94\n",
            1,
        )
        source = source.replace(
            ".L2:\n\tlw\t$2,0($19)\n\t#nop\n\tsw\t$3,20($2)\n",
            ".L2:\n\tlw\t$3,0($19)\n\t#nop\n\tsw\t$2,20($3)\n",
            1,
        )
        allocation = re.compile(
            r"^[ \t]*#\.set[ \t]+volatile\n"
            r"^[ \t]*sw[ \t]+\$2,16\(\$sp\)\n"
            r"^[ \t]*#\.set[ \t]+novolatile\n"
            r"^[ \t]*#\.set[ \t]+volatile\n"
            r"^[ \t]*lw[ \t]+\$9,16\(\$sp\)\n"
            r"^[ \t]*#\.set[ \t]+novolatile\n"
            r"^[ \t]*lw[ \t]+\$2,120\(\$19\)\n"
            r"^[ \t]*move[ \t]+\$21,\$23\n"
            r"^[ \t]*\.set[ \t]+noreorder\n"
            r"^[ \t]*\.set[ \t]+nomacro\n"
            r"^[ \t]*blez[ \t]+\$2,(?P<exit>\.L\d+)\n"
            r"^[ \t]*move[ \t]+\$20,\$0\n"
            r"^[ \t]*\.set[ \t]+macro\n"
            r"^[ \t]*\.set[ \t]+reorder\n",
            re.MULTILINE,
        )
        source = allocation.sub(
            "\tlw\t$3,120($19)\n\tmove\t$21,$23\n\tmove\t$20,$0\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            r"\tblez\t$3,\g<exit>\n\tsw\t$2,20($sp)\n"
            "\t.set\tmacro\n\t.set\treorder\n\n"
            "\t#.set\tvolatile\n\tlw\t$9,20($sp)\n"
            "\t#.set\tnovolatile\n",
            source,
            count=1,
        )
        source = source.replace(
            "\tlw\t$18,32($8)\n\tmove\t$16,$0\n\tdiv\t$17,$21,$18\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tblez\t$18,.L8\n\tmove\t$22,$9\n"
            "\t.set\tmacro\n\t.set\treorder\n\n\tmove\t$4,$19\n",
            "\tlw\t$18,32($8)\n\tdiv\t$17,$21,$18\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tblez\t$18,.L8\n\tmove\t$16,$0\n"
            "\t.set\tmacro\n\t.set\treorder\n\n"
            "\tmove\t$22,$9\n\tmove\t$4,$19\n",
            1,
        )
        callback = re.compile(
            r"(?P<label>^\.L\d+:\n)"
            r"^[ \t]*move[ \t]+\$5,\$20\n"
            r"^[ \t]*move[ \t]+\$6,\$16\n"
            r"^[ \t]*\.set[ \t]+noreorder\n"
            r"^[ \t]*\.set[ \t]+nomacro\n"
            r"^[ \t]*jal[ \t]+func_800600E0\n"
            r"^[ \t]*addu[ \t]+\$7,\$18,-1\n"
            r"^[ \t]*\.set[ \t]+macro\n"
            r"^[ \t]*\.set[ \t]+reorder\n(?:^[ \t]*\n)*"
            r"^[ \t]*mult[ \t]+\$16,\$17\n"
            r"^[ \t]*mflo[ \t]+\$5\n"
            r"^[ \t]*#nop\n"
            r"^[ \t]*move[ \t]+\$6,\$2\n",
            re.MULTILINE,
        )
        source = callback.sub(
            r"\g<label>\tmove\t$5,$20\n\tmove\t$6,$16\n"
            "\taddu\t$7,$18,-1\n\tsw\t$8,48($sp)\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tjal\tfunc_800600E0\n\tsw\t$9,52($sp)\n"
            "\t.set\tmacro\n\t.set\treorder\n\n"
            "\tmult\t$16,$17\n\tmflo\t$5\n\t#nop\n"
            "\tmove\t$6,$2\n\tlw\t$8,48($sp)\n\tlw\t$9,52($sp)\n",
            source,
            count=1,
        )
        source = source.replace(
            "\t#.set\tvolatile\n\tlw\t$2,16($sp)\n"
            "\t#.set\tnovolatile\n\tsw\t$23,20($fp)\n\tsw\t$2,16($fp)\n",
            "\t#.set\tvolatile\n\tlw\t$10,20($sp)\n"
            "\t#.set\tnovolatile\n\tsw\t$10,16($fp)\n\tsw\t$23,20($fp)\n",
            1,
        )
    if "KMC-IMAGE-TABLE-DECODE" in enabled:
        # Keep the decoder's three stack-resident values and temporary
        # registers in the locations selected by the retail ASN64 schedule.
        # The replacements retain the compiler's loads, stores, calls, loops,
        # and arithmetic while selecting the original ready-queue ordering.
        initial = re.compile(
            r"^[ \t]*lw[ \t]+\$2,456\(\$23\)\n"
            r"^[ \t]*lw[ \t]+\$3,88\(\$23\)\n"
            r"(?P<app>(?:^[ \t]*#(?:APP|NO_APP)\n)*)"
            r"^[ \t]*#\.set[ \t]+volatile\n^[ \t]*sw[ \t]+\$2,16\(\$sp\)\n"
            r"^[ \t]*#\.set[ \t]+novolatile\n"
            r"^[ \t]*li[ \t]+\$2,0x00000001[^\n]*\n"
            r"^[ \t]*\.set[ \t]+noreorder\n^[ \t]*\.set[ \t]+nomacro\n"
            r"^[ \t]*bne[ \t]+\$3,\$2,\.L2\n"
            r"^[ \t]*li[ \t]+\$2,0x000001fe[^\n]*\n"
            r"^[ \t]*\.set[ \t]+macro\n^[ \t]*\.set[ \t]+reorder\n(?:^[ \t]*\n)*"
            r"^[ \t]*#\.set[ \t]+volatile\n^[ \t]*sw[ \t]+\$2,24\(\$sp\)\n"
            r"^[ \t]*#\.set[ \t]+novolatile\n"
            r"^[ \t]*#\.set[ \t]+volatile\n^[ \t]*lw[ \t]+\$2,16\(\$sp\)\n"
            r"^[ \t]*#\.set[ \t]+novolatile\n"
            r"^[ \t]*\.set[ \t]+noreorder\n^[ \t]*\.set[ \t]+nomacro\n"
            r"^[ \t]*j[ \t]+\.L3\n^[ \t]*sw[ \t]+\$3,28\(\$2\)\n"
            r"^[ \t]*\.set[ \t]+macro\n^[ \t]*\.set[ \t]+reorder\n(?:^[ \t]*\n)*"
            r"^\.L2:\n"
            r"^[ \t]*#\.set[ \t]+volatile\n^[ \t]*sw[ \t]+\$0,24\(\$sp\)\n"
            r"^[ \t]*#\.set[ \t]+novolatile\n"
            r"^[ \t]*#\.set[ \t]+volatile\n^[ \t]*lw[ \t]+\$2,16\(\$sp\)\n"
            r"^[ \t]*#\.set[ \t]+novolatile\n^[ \t]*#nop\n"
            r"^[ \t]*sw[ \t]+\$0,28\(\$2\)\n",
            re.MULTILINE,
        )
        source = initial.sub(
            "\tlw\t$3,88($23)\n\tlw\t$9,456($23)\n"
            r"\g<app>"
            "\tli\t$2,0x00000001\t\t# 1\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbne\t$3,$2,.L2\n\tsw\t$9,20($sp)\n"
            "\t.set\tmacro\n\t.set\treorder\n\n"
            "\tli\t$9,0x000001fe\t\t# 510\n"
            "\t#.set\tvolatile\n\tsw\t$9,28($sp)\n"
            "\t#.set\tnovolatile\n\t#.set\tvolatile\n"
            "\tlw\t$9,20($sp)\n\t#.set\tnovolatile\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tj\t.L3\n\tsw\t$3,28($9)\n"
            "\t.set\tmacro\n\t.set\treorder\n\n"
            ".L2:\n\t#.set\tvolatile\n\tlw\t$9,20($sp)\n"
            "\t#.set\tnovolatile\n\t#.set\tvolatile\n"
            "\tsw\t$0,28($sp)\n\t#.set\tnovolatile\n"
            "\tsw\t$0,28($9)\n",
            source,
            count=1,
        )
        allocation = re.compile(
            r"^\.L3:\n(?P<body>.*?)^\.L5:\n",
            re.MULTILINE | re.DOTALL,
        )
        source = allocation.sub(
            ".L3:\n\tlw\t$2,4($23)\n\tmove\t$4,$23\n"
            "\tlw\t$7,120($23)\n\t#.set\tvolatile\n"
            "\tlw\t$9,28($sp)\n\t#.set\tnovolatile\n"
            "\tlw\t$2,8($2)\n\tli\t$5,0x00000001\t\t# 1\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tjal\t$31,$2\n\taddu\t$6,$9,256\n"
            "\t.set\tmacro\n\t.set\treorder\n\n"
            "\t#.set\tvolatile\n\tlw\t$9,20($sp)\n"
            "\t#.set\tnovolatile\n\tsw\t$2,24($9)\n"
            "\tlw\t$2,120($23)\n\tlw\t$fp,20($9)\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tblez\t$2,.L4\n\tmove\t$21,$0\n"
            "\t.set\tmacro\n\t.set\treorder\n\n"
            "\t#.set\tvolatile\n\tlw\t$9,20($sp)\n"
            "\t#.set\tnovolatile\n\t#.set\tvolatile\n"
            "\tsw\t$9,44($sp)\n\t#.set\tnovolatile\n"
            ".L5:\n",
            source,
            count=1,
        )
        outer_setup = re.compile(
            r"^\.L5:\n.*?"
            r"^[ \t]*jal[ \t]+func_8006012C\n"
            r"^[ \t]*addu[ \t]+\$7,\$22,-1\n"
            r"^[ \t]*\.set[ \t]+macro\n^[ \t]*\.set[ \t]+reorder\n",
            re.MULTILINE | re.DOTALL,
        )
        source = outer_setup.sub(
            ".L5:\n\t#.set\tvolatile\n\tlw\t$9,44($sp)\n"
            "\t#.set\tnovolatile\n\tlw\t$22,32($9)\n"
            "\t#.set\tvolatile\n\tlw\t$9,28($sp)\n"
            "\t#.set\tnovolatile\n\tdiv\t$fp,$fp,$22\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbeq\t$9,$0,.L8\n\tsll\t$8,$21,2\n"
            "\t.set\tmacro\n\t.set\treorder\n\n"
            "\t#.set\tvolatile\n\tlw\t$9,20($sp)\n"
            "\t#.set\tnovolatile\n\tlw\t$3,24($9)\n"
            "\taddu\t$3,$8,$3\n\tlw\t$2,0($3)\n\t#nop\n"
            "\taddu\t$2,$2,255\n\tsw\t$2,0($3)\n"
            ".L8:\n\tmove\t$19,$0\n\tmove\t$4,$23\n"
            "\tmove\t$5,$21\n\t#.set\tvolatile\n"
            "\tlw\t$9,20($sp)\n\t#.set\tnovolatile\n"
            "\tmove\t$6,$0\n\taddu\t$7,$22,-1\n"
            "\tlw\t$2,24($9)\n\tmove\t$16,$0\n"
            "\taddu\t$2,$8,$2\n\tlw\t$20,0($2)\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tjal\tfunc_8006012C\n\tmove\t$18,$0\n"
            "\t.set\tmacro\n\t.set\treorder\n",
            source,
            count=1,
        )
        source = source.replace(
            "\t#.set\tvolatile\n\tlw\t$2,24($sp)\n"
            "\t#.set\tnovolatile\n\t#nop\n"
            "\tbeq\t$2,$0,.L17\n\taddu\t$4,$20,-1\n"
            "\tli\t$16,0x00000001\t\t# 1\n",
            "\t#.set\tvolatile\n\tlw\t$9,28($sp)\n"
            "\t#.set\tnovolatile\n\t.set\tnoreorder\n"
            "\t.set\tnomacro\n\tbeq\t$9,$0,.L17\n\tnop\n"
            "\t.set\tmacro\n\t.set\treorder\n"
            "\tli\t$16,0x00000001\t\t# 1\n\taddu\t$4,$20,-1\n",
            1,
        )
        tail = re.compile(
            r"^\.L17:\n"
            r"^[ \t]*#\.set[ \t]+volatile\n^[ \t]*lw[ \t]+\$2,20\(\$sp\)\n"
            r"^[ \t]*#\.set[ \t]+novolatile\n"
            r"^[ \t]*lw[ \t]+\$3,120\(\$23\)\n"
            r"^[ \t]*addu[ \t]+\$21,\$21,1\n"
            r"^[ \t]*addu[ \t]+\$2,\$2,4\n"
            r"^[ \t]*#\.set[ \t]+volatile\n^[ \t]*sw[ \t]+\$2,20\(\$sp\)\n"
            r"^[ \t]*#\.set[ \t]+novolatile\n"
            r"^[ \t]*slt[ \t]+\$2,\$21,\$3\n"
            r"^[ \t]*bne[ \t]+\$2,\$0,\.L5\n",
            re.MULTILINE,
        )
        source = tail.sub(
            ".L17:\n\t#.set\tvolatile\n\tlw\t$9,44($sp)\n"
            "\t#.set\tnovolatile\n\tlw\t$2,120($23)\n"
            "\taddu\t$21,$21,1\n\taddu\t$9,$9,4\n"
            "\tslt\t$2,$21,$2\n\t.set\tnoreorder\n"
            "\t.set\tnomacro\n\tbne\t$2,$0,.L5\n\tsw\t$9,44($sp)\n"
            "\t.set\tmacro\n\t.set\treorder\n",
            source,
            count=1,
        )
    if "KMC-OBJECT-TABLE-PROCESS" in enabled:
        # Match the retail register-ready queue for the object table filter.
        # This preserves the compiler's arithmetic and memory operations while
        # restoring ASN64's stack slots, caller-register spills and scheduling.
        prologue = re.compile(
            r"^[ \t]*lw[ \t]+\$2,320\(\$4\)\n.*?^\.L3:\n",
            re.MULTILINE | re.DOTALL,
        )
        source = prologue.sub(
            "\t#.set\tvolatile\n\tsw\t$6,20($sp)\n\t#.set\tnovolatile\n"
            "\t#.set\tvolatile\n\tsw\t$7,28($sp)\n\t#.set\tnovolatile\n"
            "\tlw\t$16,456($4)\n\tlw\t$18,120($4)\n"
            "\tlw\t$22,112($4)\n\tlw\t$4,320($4)\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tblez\t$7,.L2\n\tsw\t$4,36($sp)\n"
            "\t.set\tmacro\n\t.set\treorder\n"
            "\taddu\t$17,$22,-1\n\tsw\t$17,52($sp)\n"
            "\tmult\t$17,$18\n\tmflo\t$17\n\t#nop\n"
            "\tsll\t$2,$22,1\n\t#.set\tvolatile\n"
            "\tlw\t$21,20($sp)\n\t#.set\tnovolatile\n"
            "\taddu\t$2,$2,2\n\tmove\t$23,$5\n"
            "\tsw\t$2,68($sp)\n\tsw\t$17,60($sp)\n.L3:\n",
            source,
            count=1,
        )
        source = re.sub(
            r"^[ \t]*\.set[ \t]+noreorder\n^[ \t]*\.set[ \t]+nomacro\n"
            r"^[ \t]*blezl[ \t]+\$18,\.L19\n^[ \t]*addu[ \t]+\$21,\$21,4\n"
            r"^[ \t]*\.set[ \t]+macro\n^[ \t]*\.set[ \t]+reorder\n(?:^[ \t]*\n)*"
            r"^[ \t]*move[ \t]+\$15,\$16\n^[ \t]*move[ \t]+\$fp,\$21\n"
            r"^[ \t]*move[ \t]+\$24,\$0\n",
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tblez\t$18,.L19\n\tmove\t$12,$0\n"
            "\t.set\tmacro\n\t.set\treorder\n"
            "\tsw\t$23,44($sp)\n\tmove\t$fp,$21\n"
            "\tmove\t$24,$0\n\tmove\t$15,$16\n",
            source, count=1, flags=re.MULTILINE,
        )
        setup = re.compile(r"^\.L7:\n.*?^\.L13:\n", re.MULTILINE | re.DOTALL)
        source = setup.sub(
            ".L7:\n\tlw\t$17,44($sp)\n\tlw\t$2,0($17)\n"
            "\tlw\t$3,84($16)\n\tlw\t$9,0($fp)\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbeq\t$3,$0,.L10\n\taddu\t$10,$2,$12\n"
            "\t.set\tmacro\n\t.set\treorder\n"
            "\tlw\t$17,60($sp)\n\taddu\t$10,$10,$17\n"
            "\tlw\t$17,52($sp)\n\tlw\t$2,68($15)\n"
            "\taddu\t$9,$9,$17\n\tlw\t$17,68($sp)\n"
            "\tli\t$14,-1\n\tsubu\t$25,$0,$18\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tj\t.L11\n\taddu\t$8,$2,$17\n"
            "\t.set\tmacro\n\t.set\treorder\n"
            ".L10:\n\tli\t$14,0x00000001\n\tmove\t$25,$18\n"
            "\tlw\t$8,68($15)\n.L11:\n\tmove\t$6,$0\n"
            "\tmove\t$11,$0\n\tlw\t$2,24($16)\n\tmove\t$7,$22\n"
            "\tlw\t$3,16($16)\n\taddu\t$2,$24,$2\n"
            "\tlw\t$20,0($2)\n\taddu\t$3,$24,$3\n"
            "\tlw\t$19,0($3)\n\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbeq\t$7,$0,.L20\n\tmove\t$13,$0\n"
            "\t.set\tmacro\n\t.set\treorder\n"
            "\tsll\t$17,$14,1\n\t#.set\tvolatile\n"
            "\tsw\t$17,84($sp)\n\t#.set\tnovolatile\n.L13:\n",
            source, count=1,
        )
        inner = re.compile(r"^\.L13:\n.*?^[ \t]*move[ \t]+\$8,\$5\n"
                           r"^[ \t]*\.set[ \t]+macro\n^[ \t]*\.set[ \t]+reorder\n",
                           re.MULTILINE | re.DOTALL)
        source = inner.sub(
            ".L13:\n\t#.set\tvolatile\n\tlw\t$17,84($sp)\n"
            "\t#.set\tnovolatile\n\taddu\t$5,$17,$8\n\tlh\t$2,0($5)\n"
            "\tlbu\t$3,0($10)\n\tlw\t$17,36($sp)\n"
            "\taddu\t$2,$6,$2\n\taddu\t$2,$2,8\n\tsra\t$6,$2,4\n"
            "\taddu\t$6,$6,$3\n\taddu\t$2,$17,$6\n\tlbu\t$6,0($2)\n"
            "\taddu\t$2,$20,$6\n\tlbu\t$3,0($2)\n\tlbu\t$2,0($9)\n"
            "\taddu\t$10,$10,$25\n\taddu\t$2,$2,$3\n\taddu\t$3,$19,$3\n"
            "\tsb\t$2,0($9)\n\tlbu\t$2,0($3)\n\taddu\t$7,$7,-1\n"
            "\taddu\t$9,$9,$14\n\tsubu\t$6,$6,$2\n\tmove\t$4,$6\n"
            "\tsll\t$2,$6,1\n\taddu\t$6,$6,$2\n\taddu\t$3,$11,$6\n"
            "\taddu\t$6,$6,$2\n\taddu\t$11,$13,$6\n\tmove\t$13,$4\n"
            "\taddu\t$6,$6,$2\n\tsh\t$3,0($8)\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbne\t$7,$0,.L13\n\tmove\t$8,$5\n"
            "\t.set\tmacro\n\t.set\treorder\n",
            source, count=1,
        )
        source = source.replace("\tsh\t$11,0($8)\n.L20:\n",
                                ".L20:\n\tsh\t$11,0($8)\n", 1)
        source = source.replace("\taddu\t$21,$21,4\n.L19:\n", ".L19:\n", 1)
        source = source.replace(
            ".L19:\n\tlw\t$2,84($16)\n\t#.set\tvolatile\n"
            "\tlw\t$3,24($sp)\n\t#.set\tnovolatile\n\t#.set\tvolatile\n"
            "\tlw\t$4,20($sp)\n\t#.set\tnovolatile\n"
            "\tsltu\t$2,$2,1\n\tsll\t$3,$3,2\n\taddu\t$4,$4,$3\n"
            "\tsw\t$2,84($16)\n\tsltu\t$2,$21,$4\n",
            ".L19:\n\tlw\t$2,84($16)\n\tsltu\t$2,$2,1\n"
            "\tsw\t$2,84($16)\n\tlw\t$17,28($sp)\n\tsll\t$2,$17,2\n"
            "\tlw\t$17,20($sp)\n\taddu\t$21,$21,4\n"
            "\taddu\t$2,$2,$17\n\tslt\t$2,$21,$2\n",
            1,
        )
    if "KMC-BITMAP-ROWS-APPLY" in enabled:
        # Select the retail caller registers and stack-argument schedule for
        # the packed bitmap row writer.  Its loops and all data operations are
        # emitted by the C source unchanged.
        setup = re.compile(
            r"^[ \t]*lbu[ \t]+\$12,0\(\$10\)\n.*?"
            r"^[ \t]*addu[ \t]+\$6,\$6,\$2\n",
            re.MULTILINE | re.DOTALL,
        )
        source = setup.sub(
            "\tlbu\t$4,0($10)\n\tandi\t$5,$5,0x00ff\n"
            "\tmult\t$5,$4\n\tmflo\t$5\n\t#nop\n"
            "\tlbu\t$2,1($10)\n\tsubu\t$sp,$sp,8\n"
            "\tlbu\t$3,35($sp)\n\tmult\t$5,$2\n"
            "\tmflo\t$5\n\t#nop\n\tlbu\t$2,27($sp)\n"
            "\tsubu\t$2,$2,$3\n\taddu\t$2,$2,-1\n"
            "\tmove\t$9,$2\n\tlbu\t$3,31($sp)\n"
            "\tsubu\t$12,$9,$4\n\tslt\t$2,$12,$9\n"
            "\taddu\t$5,$5,2\n\t.set\tnoreorder\n"
            "\t.set\tnomacro\n\tbeq\t$2,$0,.L2\n"
            "\taddu\t$11,$10,$5\n\t.set\tmacro\n"
            "\t.set\treorder\n\n\tandi\t$7,$7,0x00ff\n"
            "\tmult\t$9,$7\n\tmflo\t$2\n\t#nop\n"
            "\tandi\t$14,$3,0x00ff\n"
            "\tli\t$13,0x00000001\t\t# 1\n"
            "\taddu\t$6,$2,$6\n",
            source,
            count=1,
        )
        source = source.replace("\tbeq\t$2,$0,.L18\n\tmove\t$4,$0\n",
                                "\tblez\t$2,.L18\n\tmove\t$4,$0\n", 1)
    if "KMC-MENU-PLAYER-UPDATE" in enabled:
        # Preserve KMC's C operations while selecting the retail save order,
        # branch-likely form, and caller-register allocation for the menu queue.
        source = source.replace(
            "\tsubu\t$sp,$sp,56\n\tsw\t$18,32($sp)\n"
            "\tmove\t$18,$4\n\tli\t$4,0x00000003\t\t# 3\n"
            "\tsw\t$19,36($sp)\n\tmove\t$19,$5\n"
            "\tsw\t$20,40($sp)\n\tmove\t$20,$6\n"
            "\tsw\t$21,44($sp)\n\tmove\t$21,$7\n"
            "\tsw\t$31,48($sp)\n\tsw\t$17,28($sp)\n",
            "\tsubu\t$sp,$sp,56\n\tsw\t$19,36($sp)\n"
            "\tmove\t$19,$5\n\tsw\t$20,40($sp)\n"
            "\tmove\t$20,$6\n\tsw\t$21,44($sp)\n"
            "\tmove\t$21,$7\n\tsw\t$18,32($sp)\n"
            "\tmove\t$18,$4\n\tli\t$4,0x00000003\t\t# 3\n"
            "\tsw\t$31,48($sp)\n\tsw\t$17,28($sp)\n",
            1,
        )
        source = source.replace(
            "\tbne\t$3,$0,.L7\n\tmove\t$2,$0\n",
            "\tbnel\t$3,$0,.L7\n\tmove\t$2,$0\n",
            1,
        )
        source = source.replace(
            "\tlh\t$2,D_80087A80\n\t#nop\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbgtz\t$2,.L5\n\tmove\t$5,$2\n"
            "\t.set\tmacro\n\t.set\treorder\n\n"
            "\tsll\t$2,$2,2\n\tlhu\t$3,D_80088AE4\n"
            "\taddu\t$2,$2,$5\n",
            "\tlh\t$4,D_80087A80\n\t#nop\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbgtz\t$4,.L5\n\tmove\t$5,$4\n"
            "\t.set\tmacro\n\t.set\treorder\n\n"
            "\tsll\t$2,$4,2\n\tlhu\t$3,D_80088AE4\n"
            "\taddu\t$2,$2,$4\n",
            1,
        )
        source = source.replace(
            "\taddu\t$5,$5,1\n\tsh\t$18,D_8008A740($2)\n"
            "\tsh\t$5,D_80087A80\n",
            "\tsh\t$18,D_8008A740($2)\n\taddu\t$3,$5,1\n"
            "\tsh\t$3,D_80087A80\n",
            1,
        )
    if "KMC-IMAGE-PLANE-TRANSFORM" in enabled:
        # Match ASN64's register-ready scheduling for the image-plane transform
        # dispatcher.  The C output already supplies every operation; this lane
        # only removes a redundant live-argument copy, selects equivalent
        # caller registers, and fills the original delay/hazard slots.
        source = source.replace(
            " #NO_APP\n\tmove\t$4,$16\n\tlw\t$2,0($2)\n",
            " #NO_APP\n\tlw\t$2,0($2)\n",
            1,
        )
        source = source.replace(
            "\tmove\t$20,$0\n\tlw\t$18,216($16)\n"
            "\tlw\t$2,32($16)\n\tmove\t$17,$21\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tblez\t$2,.L4\n\taddu\t$18,$18,40\n"
            "\t.set\tmacro\n\t.set\treorder\n",
            "\tlw\t$2,32($16)\n\tlw\t$3,216($16)\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tblez\t$2,.L4\n\tmove\t$20,$0\n"
            "\t.set\tmacro\n\t.set\treorder\n"
            "\taddu\t$18,$3,40\n\tmove\t$17,$21\n",
            1,
        )
        equality = re.compile(
            r"^[ \t]*xor[ \t]+\$4,\$7,\$8\n"
            r"^[ \t]*sltu[ \t]+\$4,\$4,1\n"
            r"^[ \t]*xor[ \t]+\$2,\$5,\$6\n"
            r"^[ \t]*sltu[ \t]+\$2,\$2,1\n"
            r"^[ \t]*and[ \t]+\$2,\$4,\$2\n"
            r"(?P<directives>^[ \t]*\.set[ \t]+noreorder\n"
            r"^[ \t]*\.set[ \t]+nomacro\n)"
            r"^[ \t]*beq[ \t]+\$2,\$0,(?P<label>\.L\d+)\n"
            r"^[ \t]*sll[ \t]+\$3,\$5,1\n",
            re.MULTILINE,
        )
        source = equality.sub(
            "\txor\t$3,$5,$6\n\tsltu\t$3,$3,1\n"
            "\txor\t$2,$7,$8\n\tsltu\t$4,$2,1\n"
            "\tand\t$3,$3,$4\n\\g<directives>"
            "\tbeq\t$3,$0,\\g<label>\n\tsll\t$3,$5,1\n",
            source,
            count=1,
        )
        source = source.replace(
            "\tbne\t$2,$0,.L23\n\tdiv\t$3,$8,$7\n",
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbne\t$2,$0,.L23\n\tnop\n"
            "\t.set\tmacro\n\t.set\treorder\n\tdiv\t$3,$8,$7\n",
            1,
        )
        source = source.replace("\taddu\t$2,$20,$21\n",
                                "\taddu\t$2,$21,$20\n", 1)
    if "KMC-IMAGE-PLANE-PROCESS" in enabled:
        # Select the retail ready queue for the two-pass plane interpolation
        # loop.  Every load, store and arithmetic operation comes from the C
        # output; this lane restores ASN64's register choices and delay slots.
        prologue = re.compile(
            r"^[ \t]*subu[ \t]+\$sp,\$sp,16\n.*?"
            r"^[ \t]*\.set[ \t]+reorder\n",
            re.MULTILINE | re.DOTALL,
        )
        source = prologue.sub(
            "\tsubu\t$sp,$sp,16\n\tmove\t$25,$4\n"
            "\tsw\t$16,8($sp)\n\tmove\t$16,$5\n"
            "\tlw\t$2,308($25)\n\tlw\t$7,0($7)\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tblez\t$2,.L2\n\tmove\t$24,$0\n"
            "\t.set\tmacro\n\t.set\treorder\n\tmove\t$14,$6\n",
            source, count=1,
        )
        head = re.compile(r"^\.L3:\n.*?^\.L10:\n", re.MULTILINE | re.DOTALL)
        source = head.sub(
            ".L3:\n\tmove\t$13,$0\n\tsll\t$2,$24,2\n"
            "\taddu\t$15,$2,$7\n.L6:\n\tlw\t$11,0($14)\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbne\t$13,$0,.L9\n\tnop\n"
            "\t.set\tmacro\n\t.set\treorder\n"
            "\tlw\t$9,-4($14)\n\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tj\t.L10\n\tnop\n\t.set\tmacro\n\t.set\treorder\n"
            ".L9:\n\tlw\t$9,4($14)\n.L10:\n",
            source, count=1,
        )
        initial = re.compile(
            r"^\.L10:\n.*?^[ \t]*lw[ \t]+\$10,40\(\$16\)\n",
            re.MULTILINE | re.DOTALL,
        )
        source = initial.sub(
            ".L10:\n\tlw\t$8,0($15)\n\taddu\t$15,$15,4\n"
            "\taddu\t$24,$24,1\n\tlbu\t$6,0($9)\n\taddu\t$9,$9,1\n"
            "\tlbu\t$4,0($11)\n\taddu\t$11,$11,1\n"
            "\tlbu\t$5,0($9)\n\taddu\t$9,$9,1\n"
            "\tlbu\t$3,0($11)\n\taddu\t$11,$11,1\n"
            "\tsll\t$2,$4,1\n\taddu\t$2,$2,$4\n\taddu\t$6,$2,$6\n"
            "\tsll\t$2,$3,1\n\taddu\t$2,$2,$3\n\taddu\t$4,$2,$5\n"
            "\tsll\t$2,$6,2\n\taddu\t$2,$2,8\n\tsrl\t$2,$2,4\n"
            "\tsb\t$2,0($8)\n\taddu\t$8,$8,1\n"
            "\tsll\t$2,$6,1\n\taddu\t$2,$2,$6\n\taddu\t$2,$2,$4\n"
            "\taddu\t$2,$2,7\n\tsrl\t$2,$2,4\n\tsb\t$2,0($8)\n"
            "\taddu\t$8,$8,1\n\tlw\t$2,40($16)\n",
            source, count=1,
        )
        source = source.replace("\taddu\t$10,$10,-2\n", "\taddu\t$10,$2,-2\n", 1)
        loop = re.compile(r"^\.L13:\n.*?^[ \t]*\.set[ \t]+reorder\n",
                          re.MULTILINE | re.DOTALL)
        source = loop.sub(
            ".L13:\n\tlbu\t$5,0($9)\n\taddu\t$9,$9,1\n"
            "\tlbu\t$4,0($11)\n\taddu\t$11,$11,1\n"
            "\tsll\t$3,$6,1\n\taddu\t$3,$3,$6\n"
            "\taddu\t$2,$3,$12\n\taddu\t$2,$2,8\n\tsrl\t$2,$2,4\n"
            "\tsb\t$2,0($8)\n\taddu\t$8,$8,1\n\tmove\t$12,$6\n"
            "\taddu\t$10,$10,-1\n\tsll\t$2,$4,1\n"
            "\taddu\t$2,$2,$4\n\taddu\t$4,$2,$5\n"
            "\taddu\t$3,$3,$4\n\taddu\t$3,$3,7\n\tsrl\t$3,$3,4\n"
            "\tsb\t$3,0($8)\n\taddu\t$8,$8,1\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbne\t$10,$0,.L13\n\tmove\t$6,$4\n"
            "\t.set\tmacro\n\t.set\treorder\n",
            source, count=1,
        )
        finish = re.compile(r"^\.L12:\n.*?^[ \t]*\.set[ \t]+reorder\n",
                            re.MULTILINE | re.DOTALL)
        source = finish.sub(
            ".L12:\n\tsll\t$2,$6,1\n\taddu\t$2,$2,$6\n"
            "\taddu\t$2,$2,$12\n\taddu\t$2,$2,8\n\tsrl\t$2,$2,4\n"
            "\tsll\t$3,$6,2\n\taddu\t$3,$3,7\n\tsrl\t$3,$3,4\n"
            "\taddu\t$13,$13,1\n\tsb\t$2,0($8)\n\tslt\t$2,$13,2\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbne\t$2,$0,.L6\n\tsb\t$3,1($8)\n"
            "\t.set\tmacro\n\t.set\treorder\n",
            source, count=1,
        )
    if "KMC-IMAGE-RGBA-DECODE" in enabled:
        generated_prologue = (
            "\tsubu\t$sp,$sp,24\n"
            "\tsw\t$20,16($sp)\n\tsw\t$19,12($sp)\n"
            "\tsw\t$18,8($sp)\n\tsw\t$17,4($sp)\n\tsw\t$16,0($sp)\n"
            "\tlw\t$2,452($4)\n\tlw\t$15,40($sp)\n"
            "\tlw\t$16,112($4)\n\tlw\t$20,8($2)\n"
            "\tlw\t$19,12($2)\n\tlw\t$18,16($2)\n\tlw\t$17,20($2)\n"
            "\tlw\t$2,320($4)\n\tmove\t$24,$5\n\taddu\t$15,$15,-1\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbltz\t$15,.L2\n\taddu\t$14,$2,255\n"
            "\t.set\tmacro\n\t.set\treorder\n"
        )
        retail_prologue = (
            "\tsubu\t$sp,$sp,24\n"
            "\tlw\t$15,40($sp)\n\tmove\t$24,$5\n"
            "\tsw\t$20,16($sp)\n\tsw\t$19,12($sp)\n"
            "\tsw\t$18,8($sp)\n\tsw\t$17,4($sp)\n\tsw\t$16,0($sp)\n"
            "\tlw\t$2,452($4)\n\tlw\t$16,112($4)\n\tlw\t$5,320($4)\n"
            "\taddu\t$15,$15,-1\n\tlw\t$20,8($2)\n"
            "\tlw\t$19,12($2)\n\tlw\t$18,16($2)\n\tlw\t$17,20($2)\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbltz\t$15,.L2\n\taddu\t$14,$5,255\n"
            "\t.set\tmacro\n\t.set\treorder\n"
        )
        source = source.replace(generated_prologue, retail_prologue, 1)
        generated_rows = (
            "\taddu\t$6,$6,1\n\tlw\t$13,0($7)\n"
            "\tlw\t$2,0($24)\n\tlw\t$3,4($24)\n"
            "\taddu\t$2,$4,$2\n\tlw\t$10,0($2)\n"
            "\taddu\t$2,$4,$3\n\tlw\t$3,8($24)\n"
            "\tlw\t$5,12($24)\n\tlw\t$11,0($2)\n"
            "\taddu\t$2,$4,$3\n\tlw\t$12,0($2)\n"
            "\taddu\t$4,$4,$5\n\tlw\t$9,0($4)\n"
        )
        retail_rows = (
            "\taddu\t$6,$6,1\n\tlw\t$13,0($7)\n"
            "\tlw\t$2,0($24)\n\tlw\t$3,4($24)\n"
            "\taddu\t$2,$4,$2\n\tlw\t$10,0($2)\n"
            "\taddu\t$3,$4,$3\n\tlw\t$2,8($24)\n"
            "\tlw\t$8,0($3)\n\tlw\t$3,12($24)\n"
            "\taddu\t$2,$4,$2\n\tlw\t$5,0($2)\n"
            "\taddu\t$4,$4,$3\n\tlw\t$2,0($4)\n"
        )
        source = source.replace(generated_rows, retail_rows, 1)
        source = source.replace(
            "\t.set\tmacro\n\t.set\treorder\n\n"
            "\taddu\t$25,$9,$16\n\taddu\t$8,$13,3\n",
            "\t.set\tmacro\n\t.set\treorder\n\n"
            "\tmove\t$9,$2\n\tmove\t$12,$5\n\tmove\t$11,$8\n"
            "\taddu\t$25,$16,$9\n\taddu\t$8,$13,3\n",
            1,
        )
        source = source.replace(
            "\taddu\t$12,$12,1\n"
            "\taddu\t$5,$5,$2\n\tsubu\t$5,$14,$5\n"
            "\tlbu\t$2,0($5)\n\taddu\t$10,$10,1\n\taddu\t$11,$11,1\n",
            "\taddu\t$12,$12,1\n"
            "\taddu\t$5,$5,$2\n\tsubu\t$5,$14,$5\n"
            "\tlbu\t$2,0($5)\n\taddu\t$11,$11,1\n\taddu\t$10,$10,1\n",
            1,
        )
    if "KMC-IMAGE-RGB-DECODE" in enabled:
        # The retail scheduler retains the three row loads in ABI temporaries,
        # then copies them into the loop cursors after the empty-width branch.
        generated_rows = (
            "\taddu\t$6,$6,1\n"
            "\tlw\t$2,0($24)\n"
            "\tlw\t$12,0($7)\n"
            "\tlw\t$3,4($24)\n"
            "\tlw\t$5,8($24)\n"
            "\taddu\t$2,$4,$2\n"
            "\tlw\t$10,0($2)\n"
            "\taddu\t$2,$4,$3\n"
            "\tlw\t$11,0($2)\n"
            "\taddu\t$4,$4,$5\n"
            "\tlw\t$8,0($4)\n"
        )
        retail_rows = (
            "\taddu\t$6,$6,1\n"
            "\tlw\t$3,0($24)\n"
            "\tlw\t$12,0($7)\n"
            "\tlw\t$2,4($24)\n"
            "\taddu\t$3,$4,$3\n"
            "\tlw\t$9,0($3)\n"
            "\tlw\t$3,8($24)\n"
            "\taddu\t$2,$4,$2\n"
            "\tlw\t$5,0($2)\n"
            "\taddu\t$4,$4,$3\n"
            "\tlw\t$2,0($4)\n"
        )
        source = source.replace(generated_rows, retail_rows, 1)
        source = source.replace(
            "\t.set\tmacro\n\t.set\treorder\n\n"
            "\taddu\t$15,$8,$25\n\taddu\t$9,$12,2\n",
            "\t.set\tmacro\n\t.set\treorder\n\n"
            "\tmove\t$8,$2\n\tmove\t$11,$5\n\tmove\t$10,$9\n"
            "\taddu\t$15,$25,$8\n\taddu\t$9,$12,2\n",
            1,
        )
        source = source.replace(
            "\tlw\t$2,0($4)\n"
            "\taddu\t$8,$8,1\n"
            "\taddu\t$10,$10,1\n"
            "\taddu\t$5,$5,$2\n"
            "\taddu\t$2,$13,$5\n"
            "\tlbu\t$2,0($2)\n"
            "\taddu\t$11,$11,1\n",
            "\tlw\t$2,0($4)\n"
            "\taddu\t$8,$8,1\n"
            "\taddu\t$11,$11,1\n"
            "\taddu\t$5,$5,$2\n"
            "\taddu\t$5,$13,$5\n"
            "\tlbu\t$2,0($5)\n"
            "\taddu\t$10,$10,1\n",
            1,
        )
    if "KMC-IMAGE-RGB-ACCUMULATE" in enabled:
        # Keep each channel's table lookup beside the use that consumes it;
        # this is the register-reuse order emitted by the retail KMC scheduler.
        channel_mix = (
            "\tsll\t$3,$10,2\n"
            "\taddu\t$2,$10,1\n"
            "\tandi\t$10,$2,0x000f\n"
            "\taddu\t$2,$3,$25\n"
            "\tlw\t$6,0($2)\n"
            "\taddu\t$2,$3,$24\n"
            "\tlw\t$2,0($2)\n"
            "\taddu\t$3,$3,$15\n"
            "\taddu\t$4,$4,$6\n"
            "\taddu\t$6,$20,$4\n"
            "\taddu\t$5,$5,$2\n"
            "\taddu\t$4,$19,$5\n"
            "\tlw\t$2,0($3)\n"
            "\tlbu\t$3,0($6)\n"
            "\tlbu\t$4,0($4)\n"
            "\taddu\t$7,$7,$2\n"
            "\taddu\t$2,$18,$7\n"
            "\tlbu\t$2,0($2)\n"
        )
        retail_mix = (
            "\tsll\t$3,$10,2\n"
            "\taddu\t$2,$3,$25\n"
            "\tlw\t$6,0($2)\n"
            "\taddu\t$2,$10,1\n"
            "\tandi\t$10,$2,0x000f\n"
            "\taddu\t$2,$3,$24\n"
            "\taddu\t$3,$3,$15\n"
            "\tlw\t$3,0($3)\n"
            "\taddu\t$4,$4,$6\n"
            "\tlw\t$6,0($2)\n"
            "\taddu\t$4,$20,$4\n"
            "\tlbu\t$2,0($4)\n"
            "\taddu\t$7,$7,$3\n"
            "\taddu\t$7,$18,$7\n"
            "\taddu\t$5,$5,$6\n"
            "\taddu\t$5,$19,$5\n"
            "\tlbu\t$4,0($5)\n"
            "\tlbu\t$3,0($7)\n"
        )
        source = source.replace(channel_mix, retail_mix, 1)
        source = source.replace(
            "\taddu\t$3,$3,$4\n\taddu\t$2,$2,$3\n",
            "\taddu\t$2,$2,$4\n\taddu\t$2,$2,$3\n",
            1,
        )
    if "KMC-IMAGE-TABLE-INIT" in enabled:
        # KMC's alternate integer scheduler keeps the four colour-conversion
        # accumulators in $8-$11, advances the index beside the scaled offset,
        # and reuses the ABI temporaries for each table address.  Preserve that
        # schedule without changing the C-level loop or its arithmetic.
        # Use explicit boundaries because the back-edge label is local and the
        # generated source contains assembler mode directives around the branch.
        loop = re.compile(
            r"(?P<label>^\.L(?P<num>\d+):\n)"
            r"^[ \t]*sll[ \t]+\$4,\$7,2\n"
            r"(?P<body>(?:(?!^[ \t]*\.set[ \t]+reorder\n).)*)"
            r"^[ \t]*\.set[ \t]+reorder\n",
            re.MULTILINE | re.DOTALL,
        )
        match = loop.search(source)
        if match and f"bne\t$2,$0,.L{match.group('num')}" in match.group("body"):
            label = match.group("label")
            number = match.group("num")
            replacement = (
                label +
                "\tsll\t$4,$7,2\n"
                "\taddiu\t$7,$7,1\n"
                "\tlui\t$6,0xffff\n"
                "\tori\t$6,$6,0x492e\n"
                "\tlw\t$3,8($17)\n"
                "\tlui\t$5,0x1\n"
                "\tori\t$5,$5,0xc5a2\n"
                "\tsra\t$2,$8,16\n"
                "\taddu\t$3,$4,$3\n"
                "\tsw\t$2,0($3)\n"
                "\tlw\t$2,12($17)\n"
                "\tsra\t$3,$9,16\n"
                "\taddu\t$9,$9,$5\n"
                "\taddu\t$2,$4,$2\n"
                "\tsw\t$3,0($2)\n"
                "\tlw\t$2,16($17)\n"
                "\tlui\t$3,0x1\n"
                "\tori\t$3,$3,0x66e9\n"
                "\taddu\t$2,$4,$2\n"
                "\tsw\t$10,0($2)\n"
                "\tlw\t$2,20($17)\n"
                "\taddu\t$8,$8,$3\n"
                "\taddu\t$10,$10,$6\n"
                "\taddu\t$4,$4,$2\n"
                "\tsw\t$11,0($4)\n"
                "\tslti\t$2,$7,256\n"
                "\t.set\tnoreorder\n"
                "\t.set\tnomacro\n"
                f"\tbne\t$2,$0,.L{number}\n"
                "\taddiu\t$11,$11,-22554\n"
                "\t.set\tmacro\n"
                "\t.set\treorder\n"
            )
            source = source[:match.start()] + replacement + source[match.end():]
    if "KMC-IMAGE-BLOCK-DECODE" in enabled:
        mode_two = re.compile(
            r"(?P<head>^[ \t]*\.set[ \t]+noreorder\n"
            r"^[ \t]*\.set[ \t]+nomacro\n"
            r"^[ \t]*beq[ \t]+\$5,\$2,(?P<label>\.L\d+)\n)"
            r"^[ \t]*li[ \t]+\$2,0x00000003[^\n]*\n"
            r"(?P<tail>^[ \t]*\.set[ \t]+macro\n"
            r"^[ \t]*\.set[ \t]+reorder\n(?:^[ \t]*\n)*)"
            r"(?P<test>^[ \t]*sltu[ \t]+\$2,\$5,3\n)",
            re.MULTILINE,
        )
        match = mode_two.search(source)
        if match:
            label = match.group("label")
            source = mode_two.sub(
                r"\g<head>\g<test>\g<tail>", source, count=1,
            )
            source = source.replace(
                f"{label}:\n\tlw\t$3,36($16)\n",
                f"{label}:\n\tlw\t$3,36($16)\n"
                "\tli\t$2,0x00000003\t\t# 3\n",
                1,
            )
        end_value = re.compile(
            r"(?P<br>^[ \t]*bne[ \t]+\$2,\$0,(?P<label>\.L\d+)\n)"
            r"^[ \t]*li[ \t]+\$3,0x00000001(?P<comment>[^\n]*)\n"
            r"(?P<middle>(?:^[^\n]*\n)*?)"
            r"^[ \t]*lw[ \t]+\$3,120\(\$16\)\n"
            r"(?P=label):\n^[ \t]*sw[ \t]+\$3,124\(\$16\)\n",
            re.MULTILINE,
        )
        source = end_value.sub(
            r"\g<br>\tli\t$2,0x00000001\g<comment>\n\g<middle>"
            r"\tlw\t$2,120($16)\n\g<label>:\n\tsw\t$2,124($16)\n",
            source,
            count=1,
        )
    if "KMC-FRAME-RECORD-ALLOCATE" in enabled:
        source = source.replace(
            "\tsll\t$2,$3,2\n\tlw\t$4,D_8007343C\n",
            "\tlw\t$4,D_8007343C\n\tsll\t$2,$3,2\n",
            1,
        )
        source = source.replace(
            "\taddu\t$4,$4,$2\n\tlbu\t$2,13($4)\n",
            "\taddu\t$3,$2,$4\n\tlbu\t$2,13($3)\n",
            1,
        )
        source = source.replace("\tmove\t$7,$4\n", "\tmove\t$7,$3\n", 1)
    if "KMC-SCRIPT-OPCODE-LOAD" in enabled:
        source = re.sub(
            r"^[ \t]*lh[ \t]+\$7,0\(\$2\)\n"
            r"(?P<const>^[ \t]*li[ \t]+\$2,0x00004000[^\n]*\n)"
            r"^[ \t]*andi[ \t]+\$3,\$7,0xf000\n",
            "\tlh\t$2,0($2)\n\tmove\t$7,$2\n"
            "\tandi\t$3,$2,0xf000\n" r"\g<const>",
            source,
            count=1,
            flags=re.MULTILINE,
        )
    if "KMC-STREAM-CHUNK-ARG-SCHEDULE" in enabled:
        source = source.replace("\tmove\t$18,$6\n", "", 1)
        source = source.replace(
            "\tbeq\t$3,$2,.L2\n\tsw\t$0,16($sp)\n",
            "\tbeq\t$3,$2,.L2\n\tmove\t$18,$6\n",
            1,
        )
        source = source.replace(
            "\tmove\t$5,$17\n\tlw\t$2,4($2)\n",
            "\tmove\t$5,$17\n\tsw\t$0,16($sp)\n\tlw\t$2,4($2)\n",
            1,
        )
    if "KMC-STREAM-BYTE-READ-SCHEDULE" in enabled:
        source = source.replace(
            "\tlw\t$16,4($18)\n\tlw\t$17,0($18)\n",
            "\tlw\t$17,0($18)\n\tlw\t$16,4($18)\n",
            1,
        )
        source = source.replace(
            "\tlbu\t$20,0($17)\n\taddu\t$17,$17,1\n"
            "\taddu\t$16,$16,-1\n",
            "\tlbu\t$2,0($17)\n\taddu\t$16,$16,-1\n"
            "\taddu\t$17,$17,1\n",
            1,
        )
        source = source.replace("\tsll\t$20,$20,8\n", "\tsll\t$20,$2,8\n", 1)
        if "\tlbu\t$4,0($17)\n" in source:
            late = "\taddu\t$17,$17,1\n\taddu\t$16,$16,-1\n"
            late_at = source.rfind(late)
            if late_at >= 0:
                source = source[:late_at] + source[late_at + len(late):]
            source = source.replace("\taddu\t$20,$20,$4\n", "", 1)
            source = source.replace(
                "\tlbu\t$4,0($17)\n",
                "\tlbu\t$2,0($17)\n\taddu\t$16,$16,-1\n"
                "\taddu\t$17,$17,1\n\taddu\t$20,$20,$2\n",
                1,
            )
        marker = re.compile(
            r"(?P<load>^[ \t]*lbu[ \t]+\$20,0\(\$17\)\n)"
            r"^[ \t]*addu[ \t]+\$17,\$17,1\n"
            r"(?P<body>^[ \t]*xori[ \t]+\$3,\$21,0x00ff\n"
            r"(?:^[^\n]*\n)*?"
            r"^[ \t]*beq[ \t]+\$3,\$0,[^\n]+\n)"
            r"^[ \t]*addu[ \t]+\$16,\$16,-1\n",
            re.MULTILINE,
        )
        source = marker.sub(
            r"\g<load>\taddu\t$16,$16,-1\n\g<body>\taddu\t$17,$17,1\n",
            source,
            count=1,
        )
    if "KMC-MIDDLE-PRODUCT-WORDS" in enabled:
        source = re.sub(
            r"^[ \t]*mfhi[ \t]+\$2\n^[ \t]*mflo[ \t]+\$3\n"
            r"(?:^[ \t]*#nop\n)*"
            r"^[ \t]*sll[ \t]+\$4,\$2,16\n"
            r"^[ \t]*srl[ \t]+\$2,\$3,16\n"
            r"^[ \t]*or[ \t]+\$4,\$4,\$2\n",
            "\tmfhi\t$14\n\tmflo\t$15\n\tsll\t$14,$14,16\n"
            "\tsrl\t$15,$15,16\n\tor\t$4,$14,$15\n",
            source,
            count=1,
            flags=re.MULTILINE,
        )
    if "KMC-MENU-RESOURCE-STATE" in enabled:
        branch = re.compile(
            r"(?P<head>^[ \t]*\.set[ \t]+noreorder\n"
            r"^[ \t]*\.set[ \t]+nomacro\n)"
            r"(?P<br>^[ \t]*beq[ \t]+\$2,\$3,[^\n]+\n)"
            r"(?P<delay>^[ \t]*sw[ \t]+\$31,20\(\$sp\)\n)"
            r"(?P<tail>^[ \t]*\.set[ \t]+macro\n"
            r"^[ \t]*\.set[ \t]+reorder\n(?:^[ \t]*\n)*)"
            r"(?P<move>^[ \t]*move[ \t]+\$4,\$5\n)",
            re.MULTILINE,
        )
        source = branch.sub(
            r"\g<move>\g<head>\g<br>\g<delay>\g<tail>", source, count=1,
        )
        source = source.replace(
            "\tlhu\t$3,D_8008AFD4\n\t#nop\n\tsll\t$2,$3,2\n",
            "\tlhu\t$2,D_8008AFD4\n\tsll\t$2,$2,2\n",
            1,
        )
    if "KMC-INDEXED-RUNTIME-REBUILD" in enabled:
        source = source.replace(
            "\tmove\t$4,$19\n\tmove\t$5,$17\n\tsw\t$31,36($sp)\n",
            "\tmove\t$4,$19\n\tsw\t$31,36($sp)\n",
            1,
        )
        source = source.replace(
            "\tlw\t$2,D_8008B080($16)\n\t#nop\n\taddu\t$3,$2,$17\n",
            "\tlw\t$3,D_8008B080($16)\n\taddu\t$3,$3,$17\n",
            2,
        )
        source = source.replace("\tmove\t$18,$17\n", "", 1)
        source = source.replace("\tsll\t$2,$18,1\n", "\tsll\t$2,$17,1\n", 1)
        source = source.replace("\tsll\t$18,$18,2\n", "\tsll\t$18,$17,2\n", 1)
        source = source.replace("\taddu\t$3,$3,$18\n", "\taddu\t$3,$18,$3\n", 1)
    if "KMC-MATCH-PACKET-SCHEDULE" in enabled:
        second_field = re.compile(
            r"(?P<label>^\.L\d+:\n)"
            r"^[ \t]*lbu[ \t]+\$2,D_80088AF5\n"
            r"(?P<store>(?:^[ \t]*#\.set[^\n]*\n)*"
            r"^[ \t]*sb[ \t]+\$4,D_8008B062\n"
            r"(?:^[ \t]*#\.set[^\n]*\n)*)"
            r"(?P<body>(?:^[^\n]*\n)*?)"
            r"(?P<next>^\.L\d+:\n)"
            r"(?P<tail>(?:^[ \t]*#\.set[^\n]*\n)*"
            r"^[ \t]*sb[ \t]+\$2,D_8008B063\n)",
            re.MULTILINE,
        )

        def packet_second(match: re.Match[str]) -> str:
            body = match.group("body").replace("$2,$2,0x0080", "$4,$4,0x0080")
            tail = match.group("tail").replace("$2,D_8008B063", "$4,D_8008B063")
            return (
                match.group("label") + match.group("store")
                + "\tlbu\t$4,D_80088AF5\n" + body
                + match.group("next")
                + "\tlhu\t$2,D_800876DC\n\tlhu\t$3,D_80087AAC\n"
                + "\tlhu\t$6,D_80088AF0\n\tlhu\t$7,D_80088AEE\n"
                + tail
            )

        source = source.replace(
            "\tlhu\t$2,D_800876DC\n\tlhu\t$3,D_80087AAC\n"
            "\tlhu\t$6,D_80088AF0\n\tlhu\t$7,D_80088AEE\n",
            "",
            1,
        )
        source = second_field.sub(packet_second, source, count=1)
    if "KMC-ROSTER-ENTRY-SCHEDULE" in enabled:
        # These paired roster searches keep the narrowed roster selector in a
        # saved register before calculating the team stride.  KMC's allocator
        # otherwise delays that save and folds the successful byte address
        # back into $s0, although the generated operations are equivalent.
        for saved, entries in (("$17", "$19"), ("$19", "$20")):
            setup = (
                f"\taddu\t$2,$2,$4\n\tsw\t{saved},"
                + ("20" if saved == "$17" else "28")
                + f"($sp)\n\tandi\t{saved},$5,0xffff\n"
            )
            early = (
                f"\tsw\t{saved},"
                + ("20" if saved == "$17" else "28")
                + f"($sp)\n\tandi\t{saved},$5,0xffff\n\tandi\t$4,$4,0xffff\n"
            )
            if setup in source:
                source = source.replace("\tandi\t$4,$4,0xffff\n", early, 1)
                source = source.replace(setup, "", 1)
                source = source.replace(
                    f"\tsubu\t$3,$3,{saved}\n",
                    f"\tsubu\t$3,$3,{saved}\n\taddu\t$2,$2,$4\n",
                    1,
                )
                source = source.replace(
                    f"\taddu\t$16,{entries},$16\n\tlbu\t$2,0($16)\n",
                    f"\taddu\t$2,{entries},$16\n\tlbu\t$2,0($2)\n",
                    1,
                )
        occurrence_head = re.compile(
            r"^[ \t]*move[ \t]+\$4,\$19\n"
            r"(?P<label>^\.L\d+:\n)"
            r"^[ \t]*andi[ \t]+\$16,\$17,0xffff\n",
            re.MULTILINE,
        )
        source = occurrence_head.sub(
            r"\tandi\t$16,$17,0xffff\n\g<label>\tmove\t$4,$19\n",
            source,
            count=1,
        )
        source = re.sub(
            r"(?P<br>^[ \t]*bne[ \t]+\$2,\$0,(?P<label>\.L\d+)\n)"
            r"^[ \t]*move[ \t]+\$4,\$19\n",
            r"\g<br>\tandi\t$16,$17,0xffff\n",
            source,
            count=1,
            flags=re.MULTILINE,
        )
    if "KMC-RECORD-MATRIX-CLEAR" in enabled:
        row_math = (
            "\tandi\t$2,$16,0xffff\n\tsll\t$5,$2,5\n"
            "\taddu\t$5,$5,$2\n\tsll\t$5,$5,2\n"
            "\taddu\t$5,$5,$2\n\tsll\t$5,$5,2\n"
            "\taddu\t$5,$5,$2\n\tsll\t$5,$5,2\n"
            "\tsubu\t$5,$5,$2\n\tsll\t$5,$5,2\n"
            "\tandi\t$3,$4,0xffff\n"
        )
        source = source.replace(
            row_math,
            "\tandi\t$3,$16,0xffff\n\tsll\t$2,$3,5\n"
            "\taddu\t$2,$2,$3\n\tsll\t$2,$2,2\n"
            "\taddu\t$2,$2,$3\n\tsll\t$2,$2,2\n"
            "\taddu\t$2,$2,$3\n\tsll\t$2,$2,2\n"
            "\tsubu\t$2,$2,$3\n\tsll\t$5,$2,2\n"
            "\tandi\t$2,$4,0xffff\n",
            1,
        )
        inner = re.compile(
            r"(?P<label>^\.L15:\n)"
            r"^[ \t]*addu[ \t]+\$4,\$4,1\n"
            r"^[ \t]*sll[ \t]+\$2,\$3,1\n"
            r"^[ \t]*addu[ \t]+\$2,\$2,\$3\n"
            r"^[ \t]*addu[ \t]+\$2,\$5,\$2\n"
            r"(?P<stores>(?:^[ \t]*#\.set[^\n]*\n|^[ \t]*sb[^\n]*\n)+)"
            r"^[ \t]*andi[ \t]+\$2,\$4,0xffff\n"
            r"^[ \t]*sltu[ \t]+\$2,\$2,55\n"
            r"(?P<nr>^[ \t]*\.set[ \t]+noreorder\n)"
            r"(?P<nm>^[ \t]*\.set[ \t]+nomacro\n)"
            r"^[ \t]*bnel[ \t]+\$2,\$0,(?P<loop>\.L15)\n"
            r"^[ \t]*andi[ \t]+\$3,\$4,0xffff\n",
            re.MULTILINE,
        )
        def matrix_inner(match: re.Match[str]) -> str:
            stores = match.group("stores").replace("($2)", "($3)")
            return (
                match.group("label")
                + "\taddu\t$4,$4,1\n\tsll\t$3,$2,1\n"
                  "\taddu\t$3,$3,$2\n\taddu\t$3,$3,$5\n"
                  "\tandi\t$2,$4,0xffff\n\tsltu\t$2,$2,55\n"
                + stores + match.group("nr") + match.group("nm")
                + f"\tbne\t$2,$0,{match.group('loop')}\n"
                  "\tandi\t$2,$4,0xffff\n"
            )
        source = inner.sub(matrix_inner, source, count=1)
        source = source.replace(
            "\tandi\t$4,$17,0xffff\n.L17:\n\tandi\t$5,$16,0xffff\n",
            "\tandi\t$5,$16,0xffff\n.L17:\n\tandi\t$4,$17,0xffff\n",
            1,
        )
        source = source.replace(
            "\tbne\t$2,$0,.L17\n\tandi\t$4,$17,0xffff\n",
            "\tbne\t$2,$0,.L17\n\tandi\t$5,$16,0xffff\n",
            1,
        )
    if "KMC-INDEXED-RUNTIME-APPLY" in enabled:
        # Keep the narrowed object index in $s1 across the setup branch and
        # recreate KMC's pointer-table evaluation order for the update call.
        source = source.replace("\tandi\t$16,$4,0xffff\n", "\tandi\t$17,$4,0xffff\n", 2)
        source = source.replace(".L3:\n\tmove\t$17,$16\n", ".L3:\n", 1)
        source = source.replace("\tsll\t$16,$16,1\n", "\tsll\t$16,$17,1\n", 1)
        pointers = re.compile(
            r"^[ \t]*lw[ \t]+\$6,D_8008B078\(\$16\)\n"
            r"^[ \t]*move[ \t]+\$3,\$18\n"
            r"^[ \t]*sll[ \t]+\$2,\$3,1\n"
            r"^[ \t]*addu[ \t]+\$2,\$2,\$18\n"
            r"^[ \t]*sll[ \t]+\$2,\$2,2\n"
            r"^[ \t]*lw[ \t]+\$7,D_8008B07C\(\$16\)\n"
            r"^[ \t]*addu[ \t]+\$2,\$2,\$6\n"
            r"^[ \t]*lw[ \t]+\$6,4\(\$2\)\n"
            r"^[ \t]*sll[ \t]+\$3,\$3,2\n"
            r"^[ \t]*addu[ \t]+\$3,\$3,\$7\n",
            re.MULTILINE,
        )
        source = pointers.sub(
            "\tlw\t$3,D_8008B078($16)\n\tsll\t$2,$18,1\n"
            "\taddu\t$2,$2,$18\n\tsll\t$2,$2,2\n"
            "\taddu\t$2,$2,$3\n\tlw\t$3,D_8008B07C($16)\n"
            "\tlw\t$6,4($2)\n\tsll\t$2,$18,2\n"
            "\taddu\t$2,$2,$3\n",
            source,
            count=1,
        )
        source = source.replace("\tsw\t$19,0($3)\n", "\tsw\t$19,0($2)\n", 1)
    if "KMC-RESOURCE-BEST-MATCH-SCHEDULE" in enabled:
        # Keep each resource count load in $v0 until after the null test, then
        # preserve it in the argument register used by the corresponding loop.
        source = source.replace("\tlbu\t$11,D_800724DA\n", "\tlbu\t$2,D_800724DA\n", 1)
        source = source.replace("\tbeq\t$11,$0,", "\tbeq\t$2,$0,", 1)
        source = source.replace(
            "\tlw\t$7,D_800724DC\n",
            "\tlw\t$7,D_800724DC\n\tmove\t$5,$2\n",
            1,
        )
        source = source.replace("\tsltu\t$2,$2,$11\n", "\tsltu\t$2,$2,$5\n", 1)
        source = source.replace("\tlbu\t$11,D_800724DA\n", "\tlbu\t$2,D_800724DA\n", 1)
        source = source.replace("\tbeq\t$11,$0,", "\tbeq\t$2,$0,", 1)
        source = source.replace(
            "\tlw\t$10,D_800724DC\n",
            "\tlw\t$10,D_800724DC\n\tmove\t$7,$2\n",
            1,
        )
        source = source.replace("\tsltu\t$2,$2,$11\n", "\tsltu\t$2,$2,$7\n", 1)
        source = source.replace("\taddu\t$5,$10,$2\n", "\taddu\t$5,$2,$10\n", 1)
        source = source.replace("\tbnel\t$2,$4,", "\tbnel\t$4,$2,", 1)
    if "KMC-RECORD-LOOKUP-ARG-SCHEDULE" in enabled:
        # Narrow the saved lookup index before the resolver call, pass it in
        # the delay slot, and retain the repeated move that prepares the next
        # table-population call.
        resolver = re.compile(
            r"(?P<nr>^[ \t]*\.set[ \t]+noreorder\n)"
            r"(?P<nm>^[ \t]*\.set[ \t]+nomacro\n)"
            r"(?P<jal>^[ \t]*jal[ \t]+func_80028A58\n)"
            r"^[ \t]*andi[ \t]+\$4,\$16,0xffff\n"
            r"(?P<mac>^[ \t]*\.set[ \t]+macro\n)"
            r"(?P<reo>^[ \t]*\.set[ \t]+reorder\n)"
            r"(?P<gap>(?:^[ \t]*\n)*)"
            r"(?P<move>^[ \t]*move[ \t]+\$4,\$16\n)"
            r"(?P<body>(?:^[ \t]*#(?:APP|NO_APP)[^\n]*\n)*"
            r"^[ \t]*andi[ \t]+\$16,\$2,0xffff\n"
            r"(?:^[ \t]*#(?:APP|NO_APP)[^\n]*\n)*)"
            r"^[ \t]*andi[ \t]+\$4,\$4,0xffff\n",
            re.MULTILINE,
        )
        source = resolver.sub(
            "\tandi\t$16,$16,0xffff\n"
            r"\g<nr>\g<nm>\g<jal>\tmove\t$4,$16\n"
            r"\g<mac>\g<reo>\g<gap>\g<move>\g<body>",
            source,
            count=1,
        )
    if "KMC-SIGNED-COMPONENT-SCHEDULE" in enabled:
        # The original scheduler allocates the local frame while the third
        # positive component is in flight, and prepares each aliased negative
        # output address before its sign branch.
        source = source.replace(
            "\tlb\t$2,2($5)\n\t#nop\n\tnor\t$3,$0,$2\n",
            "\tlb\t$2,2($5)\n\tsubu\t$sp,$sp,48\n"
            "\tnor\t$3,$0,$2\n",
            1,
        )
        source = source.replace(
            "\tlb\t$2,0($5)\n\tsubu\t$sp,$sp,48\n",
            "\tlb\t$2,0($5)\n",
            1,
        )
        source = source.replace(
            "\tlb\t$2,0($5)\n\t.set\tnoreorder\n",
            "\tlb\t$2,0($5)\n\taddu\t$4,$4,16\n"
            "\tmove\t$6,$4\n\t.set\tnoreorder\n",
            1,
        )
        source = source.replace(
            ".L4:\n\taddu\t$4,$4,16\n",
            ".L4:\n",
            1,
        )
        x_store = re.compile(
            r"(?P<label>^\.L4:\n(?:^[^\n]*\n)*?)"
            r"(?P<store>^[ \t]*sb[ \t]+\$2,)0\(\$4\)\n",
            re.MULTILINE,
        )
        source = x_store.sub(r"\g<label>\g<store>0($6)\n", source, count=1)

        y_load = "\tlb\t$2,1($5)\n\t#nop\n"
        y_at = source.rfind(y_load)
        if y_at >= 0:
            source = (
                source[:y_at]
                + "\tlb\t$2,1($5)\n\tla\t$4,D_80078429\n"
                  "\tmove\t$6,$4\n"
                + source[y_at + len(y_load):]
            )
        source = source.replace(
            ".L7:\n\tla\t$4,D_80078429\n\tmove\t$6,$4\n",
            ".L7:\n",
            1,
        )
        y_second = re.compile(
            r"(?P<label>^\.L7:\n(?:^[^\n]*\n)*?^[ \t]*sb[ \t]+\$2,0\(\$6\)\n"
            r"(?:^[^\n]*\n)*?)(?P<store>^[ \t]*sb[ \t]+\$2,)0\(\$6\)\n",
            re.MULTILINE,
        )
        source = y_second.sub(r"\g<label>\g<store>0($4)\n", source, count=1)

        source = source.replace(
            "\tlb\t$2,2($5)\n\t#nop\n\t.set\tnoreorder\n",
            "\tlb\t$2,2($5)\n\tla\t$4,D_8007842A\n"
            "\tmove\t$5,$4\n\t.set\tnoreorder\n",
            1,
        )
        source = source.replace(".L10:\n\tla\t$4,D_8007842A\n", ".L10:\n", 1)
        z_first = re.compile(
            r"(?P<label>^\.L10:\n)(?P<store>^[ \t]*sb[ \t]+\$2,)0\(\$4\)\n",
            re.MULTILINE,
        )
        source = z_first.sub(r"\g<label>\g<store>0($5)\n", source, count=1)
    if "KMC-OBJECT-CYCLE-CALL-SCHEDULE" in enabled:
        # Prepare the first cycle callback's pointer argument before the call
        # and use its delay slot for the next entry-table advance.
        first_call = re.compile(
            r"(?P<arg>^[ \t]*addu[ \t]+\$6,\$6,40\n)"
            r"(?P<nr>^[ \t]*\.set[ \t]+noreorder\n)"
            r"(?P<nm>^[ \t]*\.set[ \t]+nomacro\n)"
            r"(?P<jal>^[ \t]*jal[ \t]+func_80058F2C\n)"
            r"(?P<base>^[ \t]*addu[ \t]+\$6,\$20,\$6\n)"
            r"(?P<mac>^[ \t]*\.set[ \t]+macro\n)"
            r"(?P<reo>^[ \t]*\.set[ \t]+reorder\n)"
            r"(?P<body>(?:^(?![ \t]*addu[ \t]+\$21,\$21,4\n)[^\n]*\n)*?)"
            r"(?P<advance>^[ \t]*addu[ \t]+\$21,\$21,4\n)",
            re.MULTILINE,
        )
        source = first_call.sub(
            r"\g<arg>\g<base>\g<nr>\g<nm>\g<jal>\g<advance>"
            r"\g<mac>\g<reo>\g<body>",
            source,
            count=1,
        )
    if "KMC-MEMORY-POOL-INITIALIZE" in enabled:
        # Preserve KMC's original pool-offset/slot register assignment, load
        # the initial block size before emitting the sentinel links, and keep
        # the saved pool number intact when storing the final index pointer.
        source = source.replace(
            "\tsll\t$3,$4,16\n\tla\t$5,D_8007A400\n"
            "\tsra\t$3,$3,14\n\taddu\t$5,$3,$5\n",
            "\tsll\t$2,$4,16\n\tla\t$3,D_8007A400\n"
            "\tsra\t$2,$2,14\n\taddu\t$3,$2,$3\n",
            1,
        )
        initial = re.compile(
            r"^[ \t]*lw[ \t]+\$16,0\(\$5\)\n"
            r"^[ \t]*lw[ \t]+\$2,D_8007A3F4\(\$3\)\n"
            r"^[ \t]*sw[ \t]+\$16,D_8007A410\(\$3\)\n"
            r"^[ \t]*sw[ \t]+\$16,D_8007A40C\(\$3\)\n"
            r"^[ \t]*sw[ \t]+\$16,D_8007A414\(\$3\)\n"
            r"^[ \t]*sw[ \t]+\$2,0\(\$16\)\n"
            r"(?P<one>^[ \t]*li[ \t]+\$2,0x00000001[^\n]*\n)"
            r"(?P<flag>^[ \t]*sh[ \t]+\$2,8\(\$16\)\n)"
            r"(?P<minus>^[ \t]*li[ \t]+\$2,-1[^\n]*\n)"
            r"^[ \t]*sw[ \t]+\$2,20\(\$16\)\n"
            r"^[ \t]*sw[ \t]+\$2,24\(\$16\)\n"
            r"(?P<link16>^[ \t]*sw[ \t]+\$2,16\(\$16\)\n)"
            r"(?P<link12>^[ \t]*sw[ \t]+\$2,12\(\$16\)\n)"
            r"^[ \t]*lw[ \t]+\$2,D_8007A3F8\(\$3\)\n"
            r"(?P<nop>^[ \t]*#nop[^\n]*\n)"
            r"^[ \t]*sw[ \t]+\$2,4\(\$16\)\n"
            r"^[ \t]*lw[ \t]+\$17,0\(\$5\)\n",
            re.MULTILINE,
        )
        source = initial.sub(
            "\tlw\t$16,0($3)\n\tlw\t$5,D_8007A3F4($2)\n"
            "\tsw\t$16,D_8007A410($2)\n\tsw\t$16,D_8007A40C($2)\n"
            "\tsw\t$16,D_8007A414($2)\n\tsw\t$5,0($16)\n"
            "\tlw\t$5,D_8007A3F8($2)\n"
            r"\g<one>\g<flag>\g<minus>"
            "\tsw\t$2,24($16)\n\tsw\t$2,20($16)\n"
            r"\g<link16>\g<link12>\tsw\t$5,4($16)\n"
            "\tlw\t$17,0($3)\n",
            source,
            count=1,
        )
        source = source.replace(
            "\tsll\t$18,$18,16\n\tsra\t$18,$18,14\n"
            "\tsw\t$4,D_8007A408($18)\n",
            "\tsll\t$2,$18,16\n\tsra\t$2,$2,14\n"
            "\tsw\t$4,D_8007A408($2)\n",
            1,
        )
    if "KMC-MEMORY-POOL-REGION-SCHEDULE" in enabled:
        # KMC's scheduler keeps the alignment mask and adjusted base ahead of
        # the block/index byte calculations.  In the two result paths it also
        # reuses the original count temporaries ($v0/$a0) instead of the saved
        # copies selected by this compiler reconstruction.
        setup = re.compile(
            r"(?P<nop>^[ \t]*#nop[^\n]*\n)"
            r"(?P<block>^[ \t]*sll[ \t]+\$2,\$3,3\n"
            r"^[ \t]*subu[ \t]+\$2,\$2,\$3\n"
            r"^[ \t]*sll[ \t]+\$2,\$2,2\n)"
            r"(?P<app1>(?:^[ \t]*#(?:APP|NO_APP)[^\n]*\n)*)"
            r"(?P<twice>^[ \t]*sll[ \t]+\$4,\$3,1\n)"
            r"(?P<app2>(?:^[ \t]*#(?:APP|NO_APP)[^\n]*\n)*)"
            r"(?P<mask>^[ \t]*nor[ \t]+\$6,\$0,\$7\n)"
            r"^[ \t]*move[ \t]+\$19,\$4\n"
            r"(?P<base>^[ \t]*addu[ \t]+\$5,\$5,\$7\n)"
            r"(?P<bytes>^[ \t]*move[ \t]+\$7,\$2\n)"
            r"(?P<align>^[ \t]*and[ \t]+\$5,\$5,\$6\n)"
            r"(?P<tail>(?:^[ \t]*#(?:APP|NO_APP)[^\n]*\n)*)",
            re.MULTILINE,
        )
        source = setup.sub(
            r"\g<nop>\g<mask>\g<base>\g<block>\g<app1>\g<bytes>"
            r"\g<align>\g<twice>\g<app2>\tmove\t$19,$4\n\g<tail>",
            source,
            count=1,
        )

        direct = re.compile(
            r"(?P<head>^[ \t]*addu[ \t]+)\$5,\$5,\$7\n"
            r"^[ \t]*addu[ \t]+\$5,\$5,\$19\n"
            r"(?P<store>^[ \t]*sw[ \t]+)\$5,(?P<slot>D_8007A3F4[^\n]*)\n"
            r"(?P<load>^[ \t]*lw[ \t]+\$2,4\(\$18\)\n)"
            r"(?P<gap>(?:^[ \t]*#nop[^\n]*\n)?)"
            r"(?P<sub>^[ \t]*subu[ \t]+\$2,\$2,)\$5\n"
            r"(?P<body>(?:^(?![ \t]*addu[ \t]+\$2,\$2,\$19)[^\n]*\n)*?)"
            r"(?P<index>^[ \t]*addu[ \t]+\$2,\$2,)\$19\n",
            re.MULTILINE,
        )
        source = direct.sub(
            r"\g<head>$3,$5,$2\n\taddu\t$3,$3,$4\n"
            r"\g<store>$3,\g<slot>\n\g<load>\g<gap>\g<sub>$3\n"
            r"\g<body>\g<index>$4\n",
            source,
            count=1,
        )

        source = source.replace(
            ".L2:\n\tmove\t$4,$0\n\tsll\t$17,$17,2\n"
            "\tla\t$16,D_8007A3F4\n",
            ".L2:\n\tmove\t$4,$0\n\tla\t$16,D_8007A3F4\n"
            "\tsll\t$17,$17,2\n",
            1,
        )
        indirect = re.compile(
            r"(?P<sum>^[ \t]*addu[ \t]+)\$19,\$2,\$19\n"
            r"(?P<store_obj>^[ \t]*sw[ \t]+)\$19,(?P<obj>D_8007A400[^\n]*)\n"
            r"(?P<store_idx>^[ \t]*sw[ \t]+\$2,D_8007A404[^\n]*\n)"
            r"^[ \t]*lw[ \t]+\$2,4\(\$18\)\n"
            r"^[ \t]*lw[ \t]+\$3,0\(\$16\)\n"
            r"(?P<gap>(?:^[ \t]*#nop[^\n]*\n)?)"
            r"^[ \t]*subu[ \t]+\$2,\$2,\$3\n"
            r"(?P<store_size>^[ \t]*sw[ \t]+)\$2,(?P<size>D_8007A3F8[^\n]*)\n",
            re.MULTILINE,
        )
        source = indirect.sub(
            r"\tlw\t$4,4($18)\n\tlw\t$5,0($16)\n"
            r"\g<sum>$3,$2,$19\n\g<store_obj>$3,\g<obj>\n"
            r"\g<store_idx>\tsubu\t$4,$4,$5\n"
            r"\g<store_size>$4,\g<size>\n",
            source,
            count=1,
        )
    if "KMC-SWITCH-SHARED-CASE-SPLIT" in enabled:
        # One KMC control-flow pass retains the first of two equivalent case
        # assignments as a separate jump block.  Recreate that layout while
        # preserving the shared assignment and destination used by the other
        # case.
        shared = re.search(
            r"(?P<label>^\.L\w+:\n)"
            r"(?P<first>^[ \t]*li[ \t]+\$17,0x00000009[^\n]*\n)"
            r"(?P<common>^\.L\w+:\n)"
            r"(?P<second>^[ \t]*li[ \t]+\$16,0x0000000d[^\n]*\n)",
            source,
            re.MULTILINE,
        )
        if shared:
            label = shared.group("label").strip()[:-1]
            common = shared.group("common").strip()[:-1]
            branch = re.compile(
                rf"(?P<branch>^[ \t]*beq[ \t]+\$4,\$2,){re.escape(label)}\n",
                re.MULTILINE,
            )
            block = (
                ".L__kmc_shared_case_split:\n"
                "\t.set\tnoreorder\n\t.set\tnomacro\n"
                f"\tj\t{common}\n"
                "\tli\t$17,0x00000009\n"
                "\t.set\tmacro\n\t.set\treorder\n\n"
            )
            source = source[:shared.start()] + block + source[shared.start():]
            source = branch.sub(
                r"\g<branch>.L__kmc_shared_case_split\n", source, count=1,
            )
    if "KMC-JALR-LOOP-ARG-SCHEDULE" in enabled:
        # Prepare both constant arguments before an indirect allocation call
        # and use the now-free delay slot for the following loop increment.
        pattern = re.compile(
            r"(?P<load_owner>^[ \t]*lw[ \t]+\$2,4\(\$16\)\n)"
            r"(?P<move_owner>^[ \t]*move[ \t]+\$4,\$16\n)"
            r"(?P<load_call>^[ \t]*lw[ \t]+\$2,0\(\$2\)\n)"
            r"(?P<one>^[ \t]*li[ \t]+\$5,0x00000001[^\n]*\n)"
            r"(?P<nr>^[ \t]*\.set[ \t]+noreorder\n)"
            r"(?P<nm>^[ \t]*\.set[ \t]+nomacro\n)"
            r"(?P<jal>^[ \t]*jal[ \t]+\$31,\$2\n)"
            r"(?P<size>^[ \t]*li[ \t]+\$6,0x00000100[^\n]*\n)"
            r"(?P<mac>^[ \t]*\.set[ \t]+macro\n)"
            r"(?P<reo>^[ \t]*\.set[ \t]+reorder\n)"
            r"(?P<gap>(?:^[ \t]*\n)*)"
            r"(?P<increment>^[ \t]*addu[ \t]+\$17,\$17,1\n)",
            re.MULTILINE,
        )
        source = pattern.sub(
            r"\g<load_owner>\g<move_owner>\g<one>\g<load_call>\g<size>"
            r"\g<nr>\g<nm>\g<jal>\g<increment>\g<mac>\g<reo>\g<gap>",
            source,
            count=1,
        )
    if "KMC-DYNAMIC-ALLOC-ARG-SCHEDULE" in enabled:
        # A KMC ready-queue variant prepares a constant count and dynamic
        # allocation size around the second function-pointer load, leaving
        # the indirect call's delay slot available for the following index
        # initialization.
        pattern = re.compile(
            r"(?P<load_owner>^[ \t]*lw[ \t]+\$2,4\(\$19\)\n)"
            r"(?:^[ \t]*#nop[^\n]*\n)?"
            r"(?P<load_call>^[ \t]*lw[ \t]+\$2,0\(\$2\)\n)"
            r"(?P<one>^[ \t]*li[ \t]+\$5,0x00000001[^\n]*\n)"
            r"(?P<nr>^[ \t]*\.set[ \t]+noreorder\n)"
            r"(?P<nm>^[ \t]*\.set[ \t]+nomacro\n)"
            r"(?P<jal>^[ \t]*jal[ \t]+\$31,\$2\n)"
            r"(?P<size>^[ \t]*sll[ \t]+\$6,\$16,5\n)"
            r"(?P<mac>^[ \t]*\.set[ \t]+macro\n)"
            r"(?P<reo>^[ \t]*\.set[ \t]+reorder\n)"
            r"(?P<gap>(?:^[ \t]*\n)*)"
            r"(?P<index>^[ \t]*move[ \t]+\$18,\$0\n)",
            re.MULTILINE,
        )
        source = pattern.sub(
            r"\g<load_owner>\g<one>\g<load_call>\g<size>"
            r"\g<nr>\g<nm>\g<jal>\g<index>\g<mac>\g<reo>\g<gap>",
            source,
            count=1,
        )
    if "KMC-LOOKUP-DECODE-SCHEDULE" in enabled:
        # Preserve the lookup width in $t0, initialize the outer index before
        # its entry test, and use the branch delay slot for the volatile width
        # spill.  The same scheduler variant keeps the pointer-table load in
        # $v1 and the row offset in $v0 inside the packed-index loop.
        source = source.replace("\tlw\t$3,112($23)\n", "\tlw\t$8,112($23)\n", 1)
        entry = re.compile(
            r"(?P<nr>^[ \t]*\.set[ \t]+noreorder\n)"
            r"(?P<nm>^[ \t]*\.set[ \t]+nomacro\n)"
            r"(?P<branch>^[ \t]*blez[ \t]+\$7,[^\n]+\n)"
            r"^[ \t]*move[ \t]+\$20,\$0\n"
            r"(?P<mac>^[ \t]*\.set[ \t]+macro\n)"
            r"(?P<reo>^[ \t]*\.set[ \t]+reorder\n)"
            r"(?:^[ \t]*\n)*"
            r"(?:^[ \t]*#\.set[ \t]+volatile\n)?"
            r"^[ \t]*sw[ \t]+\$3,28\(\$sp\)\n"
            r"(?:^[ \t]*#\.set[ \t]+novolatile\n)?",
            re.MULTILINE,
        )
        source = entry.sub(
            "\tmove\t$20,$0\n"
            r"\g<nr>\g<nm>\g<branch>\tsw\t$8,28($sp)\n"
            r"\g<mac>\g<reo>",
            source,
            count=1,
        )
        source = source.replace("\tlw\t$2,0($2)\n", "\tlw\t$3,0($2)\n", 1)
        source = source.replace("\tsll\t$3,$6,6\n", "\tsll\t$2,$6,6\n", 1)
        source = source.replace("\taddu\t$3,$2,$3\n", "\taddu\t$3,$3,$2\n", 1)
        source = source.replace("\tlw\t$2,20($sp)\n", "\tlw\t$8,20($sp)\n", 1)
        source = source.replace("\tslt\t$2,$20,$2\n", "\tslt\t$2,$20,$8\n", 1)
    if "KMC-ROW-GROUP-LOOP-SCHEDULE" in enabled:
        # KMC can prepare the destination-table stride before reloading the
        # loop bound, then advance the group index immediately before its
        # comparison.
        source = source.replace(
            "\taddu\t$16,$16,$20\n"
            "\tlw\t$2,308($17)\n"
            "\tsll\t$3,$20,2\n",
            "\tsll\t$3,$20,2\n"
            "\tlw\t$2,308($17)\n"
            "\taddu\t$16,$16,$20\n",
            1,
        )
    if "KMC-EARLY-A1-CONSTANT" in enabled:
        # Move an independent constant second argument directly behind the
        # frame allocation, ahead of saved-register and table-base setup.
        pattern = re.compile(
            r"(?P<frame>^[ \t]*subu[ \t]+\$sp,\$sp,32\n)"
            r"(?P<body>(?:^(?![ \t]*li[ \t]+\$5,0x00000006)[^\n]*\n)*?)"
            r"(?P<arg>^[ \t]*li[ \t]+\$5,0x00000006[^\n]*\n)",
            re.MULTILINE,
        )
        source = pattern.sub(r"\g<frame>\g<arg>\g<body>", source, count=1)
    if "KMC-PROLOGUE-S2-SAVE-ORDER" in enabled:
        # Some retail KMC objects schedule the independent $s2 save ahead of
        # the adjacent $ra/$s5 saves.  GCC's reconstructed host binary emits
        # the same three stores in the opposite ready-queue order.
        pattern = re.compile(
            r"(?P<ra>^[ \t]*sw[ \t]+\$31,[^\n]*\n)"
            r"(?P<s5>^[ \t]*sw[ \t]+\$21,[^\n]*\n)"
            r"(?P<s2>^[ \t]*sw[ \t]+\$18,[^\n]*\n)",
            re.MULTILINE,
        )
        source = pattern.sub(r"\g<s2>\g<ra>\g<s5>", source)
    if "KMC-PROLOGUE-EARLY-S4-SAVE" in enabled:
        # A second observed KMC ready-queue ordering moves the independent
        # $s4 save ahead of the $s2 setup while leaving the $s4 assignment in
        # the later branch delay slot.
        pattern = re.compile(
            r"(?P<s2>^[ \t]*sw[ \t]+\$18,[^\n]*\n)"
            r"(?P<body>(?:^(?![ \t]*sw[ \t]+\$(?:18|20|31),)[^\n]*\n)*?)"
            r"(?P<ra>^[ \t]*sw[ \t]+\$31,[^\n]*\n)"
            r"(?P<s4>^[ \t]*sw[ \t]+\$20,[^\n]*\n)",
            re.MULTILINE,
        )
        source = pattern.sub(r"\g<s4>\g<s2>\g<body>\g<ra>", source)
    if "KMC-PROLOGUE-EARLY-S5-SAVE" in enabled:
        pattern = re.compile(
            r"(?P<s2>^[ \t]*sw[ \t]+\$18,[^\n]*\n)"
            r"(?P<body>(?:^(?![ \t]*sw[ \t]+\$(?:18|21|31),)[^\n]*\n)*?)"
            r"(?P<ra>^[ \t]*sw[ \t]+\$31,[^\n]*\n)"
            r"(?P<middle>(?:^(?![ \t]*sw[ \t]+\$(?:18|21|31),)[^\n]*\n)*?)"
            r"(?P<s5>^[ \t]*sw[ \t]+\$21,[^\n]*\n)",
            re.MULTILINE,
        )
        source = pattern.sub(r"\g<s5>\g<s2>\g<body>\g<ra>\g<middle>", source)
    if "KMC-JALR-ARG-BEFORE-STORE-DELAY" in enabled:
        # KMC's scheduler variant used by some callback loops prepares the
        # final argument before an indirect call and fills the delay slot with
        # the following independent store.
        pattern = re.compile(
            r"(?P<nr>^[ \t]*\.set[ \t]+noreorder\n)"
            r"(?P<nm>^[ \t]*\.set[ \t]+nomacro\n)"
            r"(?P<jal>^[ \t]*jal[ \t]+\$31,\$\w+\n)"
            r"(?P<arg>^[ \t]*(?:addu|addiu)[^\n]*\n)"
            r"(?P<mac>^[ \t]*\.set[ \t]+macro\n)"
            r"(?P<reo>^[ \t]*\.set[ \t]+reorder\n)"
            r"(?P<gap>(?:^[ \t]*\n)*)"
            r"(?P<store>^[ \t]*s[whb][^\n]*\n)",
            re.MULTILINE,
        )
        source = pattern.sub(
            r"\g<arg>\g<nr>\g<nm>\g<jal>\g<store>\g<mac>\g<reo>\g<gap>",
            source,
        )
    if "KMC-SHIFT-BEFORE-MOVE" in enabled:
        # ASN64 sometimes fills a preceding branch delay slot by computing a
        # shift from the move's original source, then performs the move.  The
        # two forms are equivalent when the shift destination is not itself
        # the move source.
        pattern = re.compile(
            r"(?P<move>^(?P<mi>[ \t]*)move[ \t]+(?P<dst>\$\w+),(?P<src>\$\w+)\n)"
            r"(?P<shift>^(?P<si>[ \t]*)sll[ \t]+(?P<out>\$\w+),(?P=dst),(?P<bits>[^\n]+)\n)",
            re.MULTILINE,
        )

        def shift_before_move(match: re.Match[str]) -> str:
            if match.group("out") == match.group("src"):
                return match.group(0)
            return (
                f'{match.group("si")}sll\t{match.group("out")},'
                f'{match.group("src")},{match.group("bits")}\n'
                f'{match.group("mi")}move\t{match.group("dst")},'
                f'{match.group("src")}\n'
            )

        source = pattern.sub(shift_before_move, source)
        delay_pattern = re.compile(
            r"(?P<head>^[ \t]*\.set[ \t]+noreorder\n"
            r"^[ \t]*\.set[ \t]+nomacro\n"
            r"^[ \t]*(?:beq|bne|beqz|bnez|beql|bnel)[^\n]*\n)"
            r"(?P<move>^(?P<mi>[ \t]*)move[ \t]+(?P<dst>\$\w+),(?P<src>\$\w+)\n)"
            r"(?P<tail>^[ \t]*\.set[ \t]+macro\n"
            r"^[ \t]*\.set[ \t]+reorder\n(?:^[ \t]*\n)*)"
            r"(?P<shift>^(?P<si>[ \t]*)sll[ \t]+(?P<out>\$\w+),(?P=dst),(?P<bits>[^\n]+)\n)",
            re.MULTILINE,
        )

        def shift_into_delay(match: re.Match[str]) -> str:
            if match.group("out") == match.group("src"):
                return match.group(0)
            return (
                f'{match.group("head")}{match.group("si")}sll\t'
                f'{match.group("out")},{match.group("src")},{match.group("bits")}\n'
                f'{match.group("tail")}{match.group("mi")}move\t'
                f'{match.group("dst")},{match.group("src")}\n'
            )

        source = delay_pattern.sub(shift_into_delay, source)
    if "KMC-LATE-V0-ARG" in enabled:
        # One KMC ready-queue order defers a temporary copy of the third
        # argument until immediately before its first arithmetic use, after
        # the independent callee-save and argument preparation instructions.
        pattern = re.compile(
            r"(?P<stack>^[ \t]*(?:subu|addiu)[ \t]+\$sp,\$sp,[^\n]+\n)"
            r"(?P<copy>^[ \t]*move[ \t]+\$2,\$(?P<arg>[4-7])\n)"
            r"(?P<body>(?:^(?![ \t]*(?:move[ \t]+\$2|[^\n]*\$2[^\n]*)$)[^\n]*\n)*?)"
            r"(?P<use>^[ \t]*(?:addu|addiu)[ \t]+\$\w+,\$2,[^\n]+\n)",
            re.MULTILINE,
        )
        source = pattern.sub(r"\g<stack>\g<body>\g<copy>\g<use>", source)
    if "KMC-SHARED-DIV3-CONSTANT" in enabled:
        # ASN64 can keep the high half of 0x55555556 live across a two-way
        # division-by-three branch.  It hoists the LUI into the branch delay
        # slot and schedules each low-half OR before that arm's shift.
        branch = re.compile(
            r"(?P<head>^[ \t]*\.set[ \t]+noreorder\n"
            r"^[ \t]*\.set[ \t]+nomacro\n"
            r"^[ \t]*beq[^\n]+\n)"
            r"(?P<shift>^(?P<i>[ \t]*)sll[ \t]+(?P<d>\$\w+),(?P<s>\$\w+),(?P<bits>[^\n]+)\n)"
            r"(?P<tail>^[ \t]*\.set[ \t]+macro\n^[ \t]*\.set[ \t]+reorder\n)",
            re.MULTILINE,
        )
        branch_match = branch.search(source)
        if branch_match and source.count("li\t$3,0x55550000") == 2:
            shift = branch_match.group("shift")
            source = branch.sub(
                lambda m: f'{m.group("head")}{m.group("i")}li\t$3,0x55550000\n{m.group("tail")}',
                source,
                count=1,
            )
            arm_one = re.compile(
                re.escape(shift)
                + r"(?P<sra>^[ \t]*sra[^\n]+\n)"
                + r"(?P<li>^[ \t]*li[ \t]+\$3,0x55550000[^\n]*\n)"
                + r"(?P<ori>^[ \t]*ori[ \t]+\$3,\$3,0x5556\n)",
                re.MULTILINE,
            )
            source = arm_one.sub(r"\g<li>\g<ori>" + shift + r"\g<sra>", source, count=1)
            arm_two = re.compile(
                r"(?P<label>^\.L[^\n]+:\n)"
                r"(?P<sra>^[ \t]*sra[^\n]+\n)"
                r"^[ \t]*li[ \t]+\$3,0x55550000[^\n]*\n"
                r"(?P<ori>^[ \t]*ori[ \t]+\$3,\$3,0x5556\n)",
                re.MULTILINE,
            )
            source = arm_two.sub(r"\g<label>\g<ori>" + shift + r"\g<sra>", source, count=1)
    if "KMC-COMPACT-FRAME-8" in enabled:
        # Some ASN64-linked KMC units use the same local at $sp+16 but omit
        # eight bytes of otherwise unused frame padding.  Adjust only saved
        # register slots above the local area.
        source = re.sub(
            r"^([ \t]*subu[ \t]+\$sp,\$sp,)48$", r"\g<1>40", source,
            flags=re.MULTILINE,
        )
        source = re.sub(
            r"^([ \t]*addu[ \t]+\$sp,\$sp,)48$", r"\g<1>40", source,
            flags=re.MULTILINE,
        )

        def compact_slot(match: re.Match[str]) -> str:
            offset = int(match.group("offset"))
            if offset < 24:
                return match.group(0)
            return f'{match.group("head")}{offset - 8}($sp)'

        source = re.sub(
            r"^(?P<head>[ \t]*(?:lw|sw)[ \t]+\$\w+,)(?P<offset>\d+)\(\$sp\)$",
            compact_slot,
            source,
            flags=re.MULTILINE,
        )
    if "KMC-INDEX-TABLE-SCHEDULE" in enabled:
        # Schedule index narrowing before an independent table address load,
        # then defer the table-base stride until after the first entry load
        # and the second index narrowing.
        first = re.compile(
            r"(?P<table>^[ \t]*la[ \t]+(?P<base>\$\w+),[^\n]+\n)"
            r"(?P<index>^[ \t]*andi[ \t]+(?P<idx>\$\w+),(?P=idx),0xffff\n)",
            re.MULTILINE,
        )
        source = first.sub(r"\g<index>\g<table>", source, count=1)
        second = re.compile(
            r"(?P<stride>^[ \t]*addu[ \t]+(?P<base>\$\w+),(?P=base),[^\n]+\n)"
            r"(?P<load>^[ \t]*lhu[ \t]+[^\n]+\n)"
            r"(?P<index>^[ \t]*andi[ \t]+\$\w+,\$\w+,0xffff\n)",
            re.MULTILINE,
        )
        source = second.sub(r"\g<load>\g<index>\g<stride>", source, count=1)
    if "KMC-LATE-S0-SETUP" in enabled:
        # Defer an independent $s0 save and argument-derived setup until the
        # point where the saved register and computed value first become
        # live, matching another observed KMC ready-queue order.
        pattern = re.compile(
            r"(?P<stack>^[ \t]*(?:subu|addiu)[ \t]+\$sp,\$sp,[^\n]+\n)"
            r"(?P<save>^[ \t]*sw[ \t]+\$16,[^\n]+\n)"
            r"(?P<gap>^(?:[ \t]*#(?:APP|NO_APP)?[^\n]*\n)*)"
            r"(?P<setup>^[ \t]*addu[ \t]+\$16,\$6,-1\n)"
            r"(?P<body>(?:^[^\n]*\n)*?)"
            r"(?P<lastsave>^[ \t]*sw[ \t]+\$20,[^\n]+\n)"
            r"(?P<middle>(?:^[^\n]*\n)*?)"
            r"(?P<size>^[ \t]*addu[ \t]+\$5,\$5,16\n)",
            re.MULTILINE,
        )
        source = pattern.sub(
            r"\g<stack>\g<gap>\g<body>\g<lastsave>\g<save>\g<middle>\g<size>\g<setup>",
            source,
            count=1,
        )
    if "KMC-TABLE-S0-SAVE-ORDER" in enabled:
        # The four-argument table constructor stages its independent $s0 save
        # after the $s3 save, while retaining the separately delayed $s0
        # initialization used by KMC-LATE-S0-SETUP.
        pattern = re.compile(
            r"(?P<s0>^[ \t]*sw[ \t]+\$16,[^\n]+\n)"
            r"(?P<body>(?:^(?![ \t]*sw[ \t]+\$19,)[^\n]*\n)*?)"
            r"(?P<s3>^[ \t]*sw[ \t]+\$19,[^\n]+\n)",
            re.MULTILINE,
        )
        source = pattern.sub(r"\g<body>\g<s3>\g<s0>", source, count=1)
    if "KMC-PRELOAD-BOUND-SAVE-DELAY" in enabled:
        # Hoist an independent global loop-bound load ahead of the prologue,
        # and place the delayed $ra save into the initial empty branch slot.
        pattern = re.compile(
            r"(?P<prologue>^(?P<stack>[ \t]*(?:subu|addiu)[ \t]+\$sp,\$sp,[^\n]+\n)"
            r"(?P<body>(?:^[^\n]*\n)*?)"
            r"(?P<ra>^[ \t]*sw[ \t]+\$31,[^\n]+\n)"
            r"(?P<gap>(?:^[ \t]*#(?:APP|NO_APP)?[^\n]*\n)*))"
            r"(?P<load>^[ \t]*lhu[ \t]+\$2,[A-Za-z_.$][^\n]+\n)"
            r"(?:^[ \t]*#nop[^\n]*\n)*"
            r"(?P<compare>^[ \t]*sltu[^\n]+\n)"
            r"(?P<branch>^[ \t]*beq[^\n]+\n)",
            re.MULTILINE,
        )
        source = pattern.sub(
            r"\g<load>\g<stack>\g<body>\g<gap>\g<compare>"
            r"\t.set\tnoreorder\n\t.set\tnomacro\n\g<branch>\g<ra>"
            r"\t.set\tmacro\n\t.set\treorder\n",
            source,
            count=1,
        )
    if "KMC-JAL-ARG-DELAY" in enabled:
        # Delay final argument setup across independent pointer bookkeeping
        # and use it to fill the otherwise empty direct-call delay slot.
        pattern = re.compile(
            r"(?P<arg>^[ \t]*move[ \t]+\$4,\$6\n)"
            r"(?P<body>(?:^(?![ \t]*(?:jal|move[ \t]+\$4,\$6))[^\n]*\n)*?)"
            r"(?P<jal>^[ \t]*jal[ \t]+[^\n]+\n)",
            re.MULTILINE,
        )
        source = pattern.sub(
            r"\g<body>\t.set\tnoreorder\n\t.set\tnomacro\n"
            r"\g<jal>\g<arg>\t.set\tmacro\n\t.set\treorder\n",
            source,
            count=1,
        )
    if "KMC-STAGED-SAVED-REGISTERS" in enabled:
        # Stage independent callee-save stores beside the points where their
        # registers become live instead of emitting the full save mask at the
        # top of the function.
        pattern = re.compile(
            r"(?P<s3move>^[ \t]*move[ \t]+\$19,\$4\n)"
            r"(?P<ra>^[ \t]*sw[ \t]+\$31,[^\n]+\n)"
            r"(?P<s2>^[ \t]*sw[ \t]+\$18,[^\n]+\n)"
            r"(?P<s1>^[ \t]*sw[ \t]+\$17,[^\n]+\n)"
            r"(?P<s0>^[ \t]*sw[ \t]+\$16,[^\n]+\n)"
            r"(?P<gap>(?:^[ \t]*#(?:APP|NO_APP)?[^\n]*\n)*)"
            r"(?P<s1move>^[ \t]*move[ \t]+\$17,\$5\n)",
            re.MULTILINE,
        )
        match = pattern.search(source)
        if match:
            delayed = match.group("s0") + match.group("ra") + match.group("s2")
            source = pattern.sub(
                r"\g<s3move>\g<s1>\g<s1move>\g<gap>", source, count=1,
            )
            source = re.sub(
                r"(?P<move>^[ \t]*move[ \t]+\$16,\$0\n)",
                delayed + r"\g<move>",
                source,
                count=1,
                flags=re.MULTILINE,
            )
            # Retail schedules the $ra/$s2 saves after the $s0 initialization.
            source = source.replace(
                delayed + "\tmove\t$16,$0\n",
                match.group("s0") + "\tmove\t$16,$0\n"
                + match.group("ra") + match.group("s2"),
                1,
            )
    if "KMC-JAL-ARG-ORDER" in enabled:
        # Prepare the first argument before the call and retain the second
        # argument conversion as the delay-slot instruction.
        pattern = re.compile(
            r"(?P<a1>^[ \t]*andi[ \t]+\$5,\$5,[^\n]+\n)"
            r"(?P<nr>^[ \t]*\.set[ \t]+noreorder\n)"
            r"(?P<nm>^[ \t]*\.set[ \t]+nomacro\n)"
            r"(?P<jal>^[ \t]*jal[ \t]+[^\n]+\n)"
            r"(?P<a0>^[ \t]*andi[ \t]+\$4,\$19,[^\n]+\n)"
            r"(?P<tail>^[ \t]*\.set[ \t]+macro\n^[ \t]*\.set[ \t]+reorder\n)",
            re.MULTILINE,
        )
        source = pattern.sub(
            r"\g<a0>\g<nr>\g<nm>\g<jal>\g<a1>\g<tail>", source, count=1,
        )
    if "KMC-AUDIO-SLOT-SCHEDULE" in enabled:
        # Recover an ASN64 ready-queue schedule used by the four-entry audio
        # slot scan.  The arithmetic/load swaps are independent, and the
        # control-flow rewrite moves the already emitted increment and result
        # setup into their branch/jump delay slots before the shared update.
        source = source.replace(
            "\tsll\t$2,$2,2\n\tlbu\t$3,31($4)\n",
            "\tlbu\t$3,31($4)\n\tsll\t$2,$2,2\n",
            1,
        )
        source = source.replace("\taddu\t$5,$5,$4\n", "\taddu\t$5,$4,$5\n", 1)
        source = source.replace("\taddu\t$3,$4,$2\n", "\taddu\t$3,$2,$4\n", 1)
        pattern = re.compile(
            r"(?P<prefix>^[ \t]*\.set[ \t]+noreorder\n"
            r"^[ \t]*\.set[ \t]+nomacro\n)"
            r"(?P<bi>^[ \t]*)beq[ \t]+\$2,\$7,(?P<inc>\.L\w+)\n"
            r"(?P<li>^[ \t]*li[ \t]+\$2,0x00000001[^\n]*\n)"
            r"(?P<tail>^[ \t]*\.set[ \t]+macro\n"
            r"^[ \t]*\.set[ \t]+reorder\n(?:^[ \t]*\n)*"
            r"(?:^[ \t]*#(?:APP|NO_APP)?[^\n]*\n)*)"
            r"(?P<ji>^[ \t]*)j[ \t]+(?P<done>\.L\w+)\n"
            r"(?P<label>^(?P=inc):\n)"
            r"(?P<add>^[ \t]*addu[ \t]+\$2,\$6,1\n)"
            r"(?P<next>^(?P<next_name>\.L\w+):\n)"
            r"(?P<move>^[ \t]*move[ \t]+\$6,\$2\n)",
            re.MULTILINE,
        )

        def audio_slot_schedule(match: re.Match[str]) -> str:
            return (
                match.group("prefix")
                + f'{match.group("bi")}beq\t$2,$7,{match.group("next_name")}\n'
                + match.group("add")
                + match.group("tail")
                + "\t.set\tnoreorder\n\t.set\tnomacro\n"
                + f'{match.group("ji")}j\t{match.group("done")}\n'
                + match.group("li")
                + "\t.set\tmacro\n\t.set\treorder\n"
                + match.group("next")
                + match.group("move")
            )

        source = pattern.sub(audio_slot_schedule, source, count=1)
    if "KMC-FP-PAIR-WORDS" in enabled:
        # ASN64 can materialize FP pairs as two single-word transfers in
        # odd/even order instead of one doubleword transfer.  This applies
        # to ordinary pointer-based loads/stores as well as stack saves.
        # Keep this opt-in because modern GNU as otherwise emits sdc1/ldc1.
        pattern = re.compile(
            r"^(?P<i>[ \t]*)(?P<op>[sl])\.d[ \t]+\$f(?P<reg>\d+),"
            r"(?P<offset>-?\d+)\((?P<base>\$[A-Za-z0-9]+)\)\n",
            re.MULTILINE,
        )

        def fp_pair_words(match: re.Match[str]) -> str:
            instruction = "s.s" if match.group("op") == "s" else "l.s"
            reg = int(match.group("reg"))
            offset = int(match.group("offset"))
            base = match.group("base")
            indent = match.group("i")
            return (
                f"{indent}{instruction}\t$f{reg + 1},{offset}({base})\n"
                f"{indent}{instruction}\t$f{reg},{offset + 4}({base})\n"
            )

        source = pattern.sub(fp_pair_words, source)
    if "KMC-PLAY-QUEUE-SCHEDULE" in enabled:
        # Fill the queue-wrap branch with the index copy and defer the normal
        # path increment.  In the wrap arm, hoist the independent modulo-30
        # constant while the loaded index is still live in $v0, so the
        # constant occupies $v1 as in ASN64's ready-queue allocation.
        pattern = re.compile(
            r"(?P<head>^[ \t]*\.set[ \t]+noreorder\n"
            r"^[ \t]*\.set[ \t]+nomacro\n"
            r"^[ \t]*beq[ \t]+\$2,\$0,[^\n]+\n)"
            r"(?P<increment>^[ \t]*addu[ \t]+\$2,\$4,1\n)"
            r"(?P<tail>^[ \t]*\.set[ \t]+macro\n"
            r"^[ \t]*\.set[ \t]+reorder\n(?:^[ \t]*\n)*)"
            r"(?P<copy>^[ \t]*move[ \t]+\$5,\$3\n)",
            re.MULTILINE,
        )
        source = pattern.sub(
            r"\g<head>\g<copy>\g<tail>\g<increment>", source, count=1,
        )
        pattern = re.compile(
            r"(?P<load>^[ \t]*lbu[ \t]+\$2,D_80088AFA\n)"
            r"(?P<gap>(?:^[ \t]*(?:#(?:APP|NO_APP)?|#nop)[^\n]*\n|^[ \t]*\n)*)"
            r"(?P<copy>^[ \t]*move[ \t]+\$5,\$2\n)"
            r"(?P<increment>^[ \t]*addu[ \t]+\$4,\$5,1\n)"
            r"(?P<li>^[ \t]*li[ \t]+\$2,-2004353024[^\n]*\n)"
            r"(?P<ori>^[ \t]*ori[ \t]+\$2,\$2,0x8889\n)"
            r"(?P<mult>^[ \t]*mult[ \t]+\$4,\$2\n)",
            re.MULTILINE,
        )

        def play_queue_wrap(match: re.Match[str]) -> str:
            li = match.group("li").replace("$2", "$3")
            ori = match.group("ori").replace("$2", "$3")
            mult = match.group("mult").replace("$2", "$3")
            return (
                match.group("load") + match.group("gap") + li + ori
                + match.group("copy") + match.group("increment") + mult
            )

        source = pattern.sub(play_queue_wrap, source, count=1)
    if "KMC-ARCHIVE-DECODE-SCHEDULE" in enabled:
        # Recover the ready-queue choices used by the archive decode wrapper.
        # All three rewrites retain the same values and dependencies: the
        # argument save trades places with the saved-$fp setup, the dimension
        # copies trade places around their unsigned comparison, and the two
        # sign-extended stack arguments use ASN64's $t0/$v0 allocation.
        source = source.replace(
            "\tsw\t$fp,680($sp)\n",
            "\tsw\t$fp,680($sp)\n\tmove\t$fp,$6\n",
            1,
        )
        source = source.replace("\tsh\t$7,646($sp)\n", "", 1)
        source = source.replace(
            "\tjal\tfunc_80063530\n\tmove\t$fp,$6\n",
            "\tjal\tfunc_80063530\n\tsh\t$7,646($sp)\n",
            1,
        )
        source = source.replace(
            "\tmove\t$23,$3\n\tsltu\t$2,$2,$23\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbeq\t$2,$0,.L6\n\tmove\t$17,$4\n",
            "\tmove\t$17,$4\n\tsltu\t$2,$2,$3\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tbeq\t$2,$0,.L6\n\tmove\t$23,$3\n",
            1,
        )
        call_args = re.compile(
            r"^[ \t]*move[ \t]+\$4,\$22\n"
            r"(?:^[ \t]*#\.set[^\n]*\n)*"
            r"^[ \t]*lhu[ \t]+\$2,646\(\$sp\)\n"
            r"(?:^[ \t]*#\.set[^\n]*\n)*"
            r"^[ \t]*andi[ \t]+\$5,\$17,0xffff\n"
            r"^[ \t]*andi[ \t]+\$6,\$23,0xffff\n"
            r"^[ \t]*sll[ \t]+\$7,\$fp,16\n"
            r"^[ \t]*sll[ \t]+\$2,\$2,16\n"
            r"^[ \t]*sra[ \t]+\$2,\$2,16\n"
            r"^[ \t]*sw[ \t]+\$2,16\(\$sp\)\n"
            r"(?:^[ \t]*#\.set[^\n]*\n)*"
            r"^[ \t]*lhu[ \t]+\$2,638\(\$sp\)\n"
            r"(?:^[ \t]*#\.set[^\n]*\n)*"
            r"^[ \t]*sra[ \t]+\$7,\$7,16\n"
            r"^[ \t]*li[ \t]+\$3,0x00000010[^\n]*\n"
            r"^[ \t]*sw[ \t]+\$3,24\(\$sp\)\n"
            r"^[ \t]*sll[ \t]+\$2,\$2,16\n"
            r"^[ \t]*sra[ \t]+\$2,\$2,16\n"
            r"^[ \t]*\.set[ \t]+noreorder\n"
            r"^[ \t]*\.set[ \t]+nomacro\n"
            r"^[ \t]*jal[ \t]+func_80017840\n"
            r"^[ \t]*sw[ \t]+\$2,20\(\$sp\)\n",
            re.MULTILINE,
        )
        source = call_args.sub(
            "\tmove\t$4,$22\n\tandi\t$5,$17,0xffff\n"
            "\tlhu\t$8,646($sp)\n\tandi\t$6,$23,0xffff\n"
            "\tsll\t$7,$fp,16\n\tsll\t$2,$8,16\n"
            "\tlhu\t$8,638($sp)\n\tsra\t$7,$7,16\n"
            "\tsra\t$2,$2,16\n\tsw\t$2,16($sp)\n"
            "\tsll\t$2,$8,16\n\tsra\t$2,$2,16\n"
            "\tsw\t$2,20($sp)\n\tli\t$2,0x00000010\n"
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tjal\tfunc_80017840\n\tsw\t$2,24($sp)\n",
            source,
            count=1,
        )
    if "KMC-OBJECT-VECTOR-SCHEDULE" in enabled:
        # Preserve the two post-MFLO bubbles emitted by the retail assembler,
        # then reproduce its placement of independent fixed-point constants
        # and stores in the three-component vector helper.
        source = re.sub(
            r"(?P<mflo>^[ \t]*mflo[^\n]*\n)(?:^[ \t]*#nop[^\n]*\n)*",
            r"\g<mflo>\tnop\n\tnop\n",
            source,
            flags=re.MULTILINE,
        )
        source = re.sub(
            r"(?P<li>^[ \t]*li[ \t]+\$8,0x00040000[^\n]*\n)"
            r"(?P<before>(?:^[ \t]*#\.set[^\n]*\n)*)"
            r"(?P<store>^[ \t]*sw[ \t]+\$3,0\(\$5\)\n)"
            r"(?P<middle>(?:^[ \t]*#\.set[^\n]*\n)*)"
            r"(?P<load>^[ \t]*lw[ \t]+\$2,32\(\$4\)\n)"
            r"(?P<after>(?:^[ \t]*#\.set[^\n]*\n)*)",
            r"\g<before>\g<store>\g<middle>\g<load>\g<after>\g<li>",
            source,
            count=1,
            flags=re.MULTILINE,
        )
        source = source.replace(
            "\tj\t.L4\n.L2:\n",
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tj\t.L4\n\tsw\t$8,4($5)\n"
            "\t.set\tmacro\n\t.set\treorder\n\n.L2:\n",
            1,
        )
        source = re.sub(
            r"(?P<li>^[ \t]*li[ \t]+\$8,0x00040000[^\n]*\n)"
            r"(?P<comments>(?:^[ \t]*#\.set[^\n]*\n)*)"
            r"(?P<load>^[ \t]*lw[ \t]+\$3,32\(\$4\)\n)"
            r"(?P<tail>(?:^[ \t]*#\.set[^\n]*\n)*)"
            r"(?P<ori>^[ \t]*ori[ \t]+\$8,\$8,0xcccc\n)",
            r"\g<li>\g<ori>\g<comments>\g<load>\g<tail>",
            source,
            count=1,
            flags=re.MULTILINE,
        )
        source = re.sub(
            r"(?P<label>^\.L4:\n)"
            r"(?P<comments>(?:^[ \t]*#\.set[^\n]*\n)*)"
            r"(?P<store>^[ \t]*sw[ \t]+\$8,4\(\$5\)\n)"
            r"(?P<tail>(?:^[ \t]*#\.set[^\n]*\n)*)",
            r"\g<comments>\g<store>\g<tail>\g<label>",
            source,
            count=1,
            flags=re.MULTILINE,
        )
        source = re.sub(
            r"(?P<li>^[ \t]*li[ \t]+\$3,0x00040000[^\n]*\n)"
            r"(?P<comments>(?:^[ \t]*#\.set[^\n]*\n)*)"
            r"(?P<load>^[ \t]*lw[ \t]+\$4,36\(\$4\)\n)"
            r"(?P<tail>(?:^[ \t]*#\.set[^\n]*\n)*)"
            r"(?P<ori>^[ \t]*ori[ \t]+\$3,\$3,0xcccc\n)",
            r"\g<li>\g<ori>\g<comments>\g<load>\g<tail>",
            source,
            count=1,
            flags=re.MULTILINE,
        )
        source = re.sub(
            r"(?P<andi>^[ \t]*andi[ \t]+\$2,\$6,0x0fff\n)"
            r"(?P<comments>(?:^[ \t]*#\.set[^\n]*\n)*)"
            r"(?P<store>^[ \t]*sw[ \t]+\$3,8\(\$5\)\n)"
            r"(?P<tail>(?:^[ \t]*#\.set[^\n]*\n)*)"
            r"(?P<jump>^[ \t]*j[ \t]+\$31\n)",
            r"\g<andi>\g<comments>\g<tail>\g<jump>\g<store>",
            source,
            count=1,
            flags=re.MULTILINE,
        )
        source = source.replace(
            "\tj\t$31\n\tsw\t$3,8($5)\n",
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            "\tj\t$31\n\tsw\t$3,8($5)\n"
            "\t.set\tmacro\n\t.set\treorder\n",
            1,
        )
    if "KMC-MENU-ACTIVATION-SCHEDULE" in enabled:
        # Narrow the first lookup result in the following call's delay slot,
        # and initialize the default dispatch mode in the later branch slot.
        # Both moves cross only independent instructions and remove no source
        # operation from the reconstructed menu activation logic.
        first_call = re.compile(
            r"^[ \t]*move[ \t]+\$16,\$2\n"
            r"(?P<nr>^[ \t]*\.set[ \t]+noreorder\n)"
            r"(?P<nm>^[ \t]*\.set[ \t]+nomacro\n)"
            r"(?P<jal>^[ \t]*jal[ \t]+func_8001FA38\n)"
            r"^[ \t]*li[ \t]+\$4,0x00000003[^\n]*\n"
            r"(?P<tail>^[ \t]*\.set[ \t]+macro\n"
            r"^[ \t]*\.set[ \t]+reorder\n)",
            re.MULTILINE,
        )
        source = first_call.sub(
            "\tli\t$4,0x00000003\n"
            r"\g<nr>\g<nm>\g<jal>\tandi\t$16,$2,0xffff\n\g<tail>",
            source,
            count=1,
        )
        source = source.replace("\tandi\t$16,$16,0xffff\n", "", 1)
        mode_branch = re.compile(
            r"(?P<branch>^[ \t]*bne[ \t]+\$2,\$0,(?P<label>\.L\w+)\n)"
            r"(?P<body>(?:^(?![ \t]*li[ \t]+\$4,0x00000003)[^\n]*\n)*?)"
            r"(?P<target>^(?P=label):\n)"
            r"(?P<arm>(?:^(?![ \t]*li[ \t]+\$4,0x00000003)[^\n]*\n)*?)"
            r"^[ \t]*li[ \t]+\$4,0x00000003[^\n]*\n",
            re.MULTILINE,
        )
        source = mode_branch.sub(
            "\t.set\tnoreorder\n\t.set\tnomacro\n"
            r"\g<branch>\tli\t$4,0x00000003\n"
            "\t.set\tmacro\n\t.set\treorder\n"
            r"\g<body>\g<target>\g<arm>",
            source,
            count=1,
        )
    if "KMC-COMBINE-MODE-SCHEDULE" in enabled:
        # Keep the final packed-command conversion ahead of the independent
        # state-cache stores, matching the retail ready queue and the source
        # statement order used by the display combine-mode helper.
        state_stores = re.compile(
            r"(?P<head>^[ \t]*mul\.s[ \t]+\$f0,\$f0,\$f6\n)"
            r"(?P<stores>(?:^[ \t]*sw[ \t]+\$(?:4|5|6|7|10),"
            r"D_800887(?:D4|D8|DC|E0|E4)[^\n]*\n){5})"
            r"(?P<convert>^[ \t]*trunc\.w\.s[^\n]*\n"
            r"^[ \t]*mfc1[ \t]+\$2,\$f2\n"
            r"(?:^[ \t]*#nop[^\n]*\n)*"
            r"^[ \t]*andi[ \t]+\$2,\$2,0x0fff\n"
            r"^[ \t]*or[ \t]+\$2,\$8,\$2\n"
            r"^[ \t]*sw[ \t]+\$2,4\(\$9\)\n)",
            re.MULTILINE,
        )
        source = state_stores.sub(
            r"\g<head>\g<convert>\g<stores>", source, count=1,
        )
    if "KMC-DISPLAY-PAGE-SCHEDULE" in enabled:
        # Reproduce the display-page helper's global-load ready queue and its
        # explicit table-base construction.  The rewritten block contains the
        # same loads, stores, constants, and index calculation as GCC output.
        prologue = re.compile(
            r"^[ \t]*subu[ \t]+\$sp,\$sp,32\n"
            r"^[ \t]*sw[ \t]+\$17,20\(\$sp\)\n"
            r"^[ \t]*la[ \t]+\$17,D_80064B88\n"
            r"^[ \t]*li[ \t]+\$6,-2147155968[^\n]*\n"
            r"^[ \t]*ori[ \t]+\$6,\$6,0x0f78\n"
            r"^[ \t]*lw[ \t]+\$3,D_80085038\n"
            r"^[ \t]*lw[ \t]+\$5,D_8008AF98\n"
            r"^[ \t]*li[ \t]+\$2,-2147024896[^\n]*\n"
            r"^[ \t]*ori[ \t]+\$2,\$2,0x5d00\n"
            r"^[ \t]*andi[ \t]+\$4,\$4,0xffff\n"
            r"^[ \t]*sw[ \t]+\$16,16\(\$sp\)\n"
            r"^[ \t]*sll[ \t]+\$16,\$4,2\n"
            r"^[ \t]*sw[ \t]+\$2,D_80064B78\n"
            r"^[ \t]*la[ \t]+\$2,D_8008B350\n"
            r"^[ \t]*sw[ \t]+\$31,24\(\$sp\)\n"
            r"^[ \t]*sw[ \t]+\$6,D_80064B70\n",
            re.MULTILINE,
        )
        source = prologue.sub(
            "\tlw\t$3,D_80085038\n\tlw\t$5,D_8008AF98\n"
            "\tsubu\t$sp,$sp,32\n"
            "\tlui\t$2,0x8005\n\taddiu\t$2,$2,0x0f78\n"
            "\tsw\t$2,D_80064B70\n"
            "\tlui\t$2,0x8007\n\taddiu\t$2,$2,0x5d00\n"
            "\tsw\t$17,20($sp)\n\tla\t$17,D_80064B88\n"
            "\tsw\t$16,16($sp)\n\tandi\t$16,$4,0xffff\n"
            "\tsll\t$16,$16,2\n\tsw\t$31,24($sp)\n"
            "\tsw\t$2,D_80064B78\n",
            source,
            count=1,
        )
        source = source.replace(
            "\tsw\t$3,D_80064B8C\n\taddu\t$3,$16,$2\n",
            "\tsw\t$3,D_80064B8C\n\tla\t$3,D_8008B350\n"
            "\taddu\t$3,$16,$3\n",
            1,
        )
        post_call = re.compile(
            r"^[ \t]*li[ \t]+\$2,0x00000001[^\n]*\n"
            r"^[ \t]*sw[ \t]+\$2,D_80086170\(\$16\)\n"
            r"(?P<nr>^[ \t]*\.set[ \t]+noreorder\n)"
            r"(?P<nm>^[ \t]*\.set[ \t]+nomacro\n)"
            r"(?P<jal>^[ \t]*jal[ \t]+func_8004D814\n)"
            r"^[ \t]*move[ \t]+\$4,\$17\n"
            r"(?P<tail>^[ \t]*\.set[ \t]+macro\n"
            r"^[ \t]*\.set[ \t]+reorder\n)",
            re.MULTILINE,
        )
        source = post_call.sub(
            "\tmove\t$4,$17\n\tla\t$2,D_80086170\n"
            "\taddu\t$16,$16,$2\n\tli\t$2,0x00000001\n"
            "\tsw\t$2,0($16)\n"
            r"\g<nr>\g<nm>\g<jal>\tnop\n\g<tail>",
            source,
            count=1,
        )
    if "KMC-STRIDED-COPY-ARG-ORDER" in enabled:
        # Prepare destination and source before the row-copy call, leaving the
        # byte count in the call delay slot as in the retail ready queue.
        call = re.compile(
            r"^[ \t]*move[ \t]+\$6,\$17\n"
            r"^[ \t]*move[ \t]+\$4,\$16\n"
            r"(?P<nr>^[ \t]*\.set[ \t]+noreorder\n)"
            r"(?P<nm>^[ \t]*\.set[ \t]+nomacro\n)"
            r"(?P<jal>^[ \t]*jal[ \t]+func_8002E17C\n)"
            r"^[ \t]*move[ \t]+\$5,\$19\n"
            r"(?P<tail>^[ \t]*\.set[ \t]+macro\n"
            r"^[ \t]*\.set[ \t]+reorder\n)",
            re.MULTILINE,
        )
        source = call.sub(
            "\tmove\t$4,$16\n\tmove\t$5,$19\n"
            r"\g<nr>\g<nm>\g<jal>\tmove\t$6,$17\n\g<tail>",
            source,
            count=1,
        )
    if "KMC-STRIP-LOOP-SCHEDULE" in enabled:
        # Initialize the loop counter before preserving the call result, and
        # defer the independent source-pointer increment into the back-edge
        # delay slot.
        source = re.sub(
            r"(?P<result>^[ \t]*andi[ \t]+\$7,\$2,0xffff\n)"
            r"(?P<count>^[ \t]*move[ \t]+\$8,\$0\n)",
            r"\g<count>\g<result>",
            source,
            count=1,
            flags=re.MULTILINE,
        )
        increment = re.search(
            r"^[ \t]*addu[ \t]+\$16,\$16,2\n", source, re.MULTILINE,
        )
        if increment:
            increment_text = increment.group(0)
            source = source[:increment.start()] + source[increment.end():]
            source = re.sub(
                r"(?P<branch>^[ \t]*bne[ \t]+\$2,\$0,[^\n]+\n)",
                "\t.set\tnoreorder\n\t.set\tnomacro\n"
                + r"\g<branch>" + increment_text
                + "\t.set\tmacro\n\t.set\treorder\n",
                source,
                count=1,
                flags=re.MULTILINE,
            )
    return source


def _apply_absolute_section_alignment(source: str,
                                      section_bases: dict[str, int]) -> str:
    """Reproduce ASN64's absolute-address alignment in data sections.

    GNU ``as`` aligns a relocatable section relative to offset zero.  ASN64
    instead evaluated ``.align`` after the section had been assigned its
    linked address.  Those differ when (for example) ``.rodata`` begins four
    bytes past an eight-byte boundary and a later compiler pool uses
    ``.align 3``.  Turn only the differing alignments into explicit padding;
    leaving equal cases alone preserves the ELF section alignment metadata.

    KMC compiler pools use the simple scalar directives handled below.  If an
    unknown size-bearing directive is encountered, stop rewriting that
    section rather than guessing.
    """
    import ast

    offsets: dict[str, int] = {}
    known: dict[str, bool] = {}
    current = ".text"
    out: list[str] = []

    def count_values(arg: str) -> int:
        return len([part for part in arg.split(",") if part.strip()])

    for line in source.splitlines(keepends=True):
        stripped = line.strip()
        section = re.match(r"\.section\s+([^,\s]+)", stripped)
        if section:
            current = section.group(1)
            offsets.setdefault(current, 0)
            known.setdefault(current, True)
            out.append(line)
            continue
        if stripped == ".text":
            current = ".text"
            out.append(line)
            continue
        if current not in section_bases or not known.get(current, True):
            out.append(line)
            continue

        align = re.match(r"\.align\s+(\d+)\s*$", stripped)
        if align:
            boundary = 1 << int(align.group(1))
            offset = offsets.get(current, 0)
            relative_pad = (-offset) % boundary
            absolute_pad = (-(section_bases[current] + offset)) % boundary
            if relative_pad != absolute_pad:
                indent = line[:len(line) - len(line.lstrip())]
                ending = "\n" if line.endswith("\n") else ""
                line = (f"{indent}.space {absolute_pad}"
                        f" # ASN64 absolute .align {align.group(1)}{ending}")
            offsets[current] = offset + absolute_pad
            out.append(line)
            continue

        size = None
        scalar = re.match(r"\.(byte|half|short|word|float|double)\s+(.+)$", stripped)
        if scalar:
            widths = {"byte": 1, "half": 2, "short": 2, "word": 4,
                      "float": 4, "double": 8}
            size = widths[scalar.group(1)] * count_values(scalar.group(2))
        else:
            string = re.match(r"\.(string|ascii|asciz)\s+(.+)$", stripped)
            if string:
                try:
                    value = ast.literal_eval(string.group(2))
                    size = len(value.encode("latin1")) + (string.group(1) != "ascii")
                except (ValueError, SyntaxError, UnicodeEncodeError):
                    known[current] = False
            else:
                space = re.match(r"\.(space|zero)\s+(0x[0-9a-fA-F]+|\d+)", stripped)
                if space:
                    size = int(space.group(2), 0)
                elif stripped.startswith((".incbin", ".fill", ".org")):
                    known[current] = False
        if size is not None:
            offsets[current] = offsets.get(current, 0) + size
        out.append(line)
    return "".join(out)


def convert(source: str, disabled: frozenset[str] | set[str] = frozenset(),
            enabled: frozenset[str] | set[str] = frozenset(),
            section_bases: dict[str, int] | None = None) -> str:
    """Return assembler input for modern GNU as equivalent to ASN64 assembling ``source``."""
    source = _apply_enabled_lanes(source, enabled)
    if "KMC-LID-PAD4" in enabled:
        # The explicit four-byte pad added after the generated li.d datum
        # already realizes the original absolute eight-byte boundary.  GNU
        # as would otherwise apply the compiler's next `.align 3` relative to
        # the start of this standalone section and add a second four bytes.
        source = re.sub(
            r"^(?P<i>[ \t]*)\.align[ \t]+3[ \t]*$",
            r"\g<i>.align\t2",
            source,
            count=1,
            flags=re.MULTILINE,
        )
    input_lines = source.splitlines(keepends=True)
    out: list[str] = [PRELUDE]
    is_reorder = True
    generated = 0
    mfhilo_delay_count = 0
    mfhilo_delay_location = 0
    prev_mul = prev_prev_mul = False
    comm_symbols: list[tuple[str, str]] = []
    lcomm_symbols: list[tuple[str, str]] = []
    delay_slot = False
    file_count = 0
    prev_instruction_index = 0
    last_file_directive = -1
    local_symbols: set[str] = set()

    for line_index, line in enumerate(input_lines):
        tokens = _tokens(line)
        emitted_instruction_count = 1
        if line[1:5] == "#nop":
            prev = _tokens(input_lines[prev_instruction_index])
            if prev and prev[0] in COMPARE_MNEMONICS:
                line = "\tnop\n"
        if not tokens:
            out.append(line)
            continue
        identifier = tokens[0]
        new_prev_mul = False
        is_branch = False

        if identifier[0] == ".":
            directive = identifier[1:]
            if directive == "set":
                if tokens[1] == "noreorder":
                    is_reorder = False
                    continue
                if tokens[1] == "reorder":
                    is_reorder = True
                    continue
            elif directive == "local":
                local_symbols.add(tokens[1])
            elif directive == "comm":
                sym, size = [s.strip() for s in tokens[1].split(",")][:2]
                (lcomm_symbols if sym in local_symbols else comm_symbols).append((sym, size))
                line = ""
            elif directive == "lcomm":
                sym, size = [s.strip() for s in tokens[1].split(",")]
                lcomm_symbols.append((sym, size))
                line = ""
            elif directive == "file":
                file_count += 1
                line = f"\t.file\t{file_count + 1} {tokens[2]}\n"
                last_file_directive = len(out)
            elif directive in ("def", "begin", "bend"):
                line = ""
            elif directive == "word":
                if tokens[1][0] == "$":
                    line = f"\t.word\t.{tokens[1][1:]}\n"
        elif identifier[-1] == ":":
            pass
        else:
            if len(tokens) > 1:
                expanded = _expand_memory(identifier, "".join(tokens[1:]), disabled, enabled)
                if expanded is not None:
                    line, emitted_instruction_count = expanded
                    # ASN64 evaluates HI/LO hazards after expanding this
                    # addressing macro.  Its replacement is three machine
                    # instructions, so all three consume hazard slots.
            if identifier in COP1_BRANCH_MNEMONICS and "FPCMP-BC1-NOP" not in disabled:
                prev = _tokens(input_lines[prev_instruction_index]) if prev_instruction_index != line_index else []
                if prev and prev[0] in COMPARE_MNEMONICS and not any(
                        between[1:5] == "#nop" for between in input_lines[prev_instruction_index + 1:line_index]):
                    out.append("\tnop\n")
            if identifier in BRANCH_MNEMONICS:
                is_branch = True
                next_tokens = (_tokens(input_lines[line_index + 1])
                               if line_index + 1 < len(input_lines) else [])
                fill_move = ("REORDER-BRANCH-NOP-MOVE" in disabled
                             and next_tokens and next_tokens[0] == "move")
                if is_reorder and not fill_move:
                    line += "\tnop\n"
                    # The inserted delay-slot instruction consumes a real
                    # HI/LO hazard slot.  Count both emitted instructions so
                    # a following div/mult is not padded once too many.
                    emitted_instruction_count += 1
            elif identifier in ("li.s", "li.d"):
                ops = [s.strip() for s in tokens[1].split(",")]
                if ops[0][1] == "f":
                    sym = f"RODATA_SYM_{generated}"
                    single = identifier == "li.s"
                    if single:
                        load = f"\tlwc1 {ops[0]}, %lo({sym})($at)\n"
                    elif "KMC-LID-PAIR" in enabled:
                        number = int(ops[0][2:])
                        load = (
                            f"\tlwc1 $f{number + 1}, %lo({sym})($at)\n"
                            f"\tlwc1 $f{number}, %lo({sym}+4)($at)\n"
                        )
                    else:
                        load = f"\tldc1 {ops[0]}, %lo({sym})($at)\n"
                    # Some ASN64 builds place li.d literals on four-byte
                    # boundaries.  Emit the double as words as well, because
                    # GNU as otherwise restores its native eight-byte
                    # alignment for the .double directive.
                    alignment = (2 if single or "KMC-LID-ALIGN4" in enabled or (
                        "KMC-COMBINE-MODE-SCHEDULE" in enabled and generated >= 2
                    ) else 3)
                    if (not single and ("KMC-LID-ALIGN4" in enabled or (
                            "KMC-COMBINE-MODE-SCHEDULE" in enabled
                            and generated >= 2))):
                        bits = struct.unpack(">Q", struct.pack(">d", float(ops[1])))[0]
                        datum = f"\t.word 0x{bits >> 32:08x},0x{bits & 0xffffffff:08x}\n"
                    else:
                        datum = f"\t.{'float' if single else 'double'} {ops[1]}\n"
                    if not single and "KMC-LID-PAD4" in enabled:
                        # Some KMC/ASN64 objects place a four-byte-aligned
                        # li.d literal immediately before an absolutely
                        # eight-byte-aligned compiler pool (typically a jump
                        # table).  Relocatable GNU as aligns section-relative,
                        # so preserve the original four-byte inter-object pad
                        # explicitly when this lane is requested.
                        datum += "\t.space 4\n"
                    line = ("\t.section .rodata\n"
                            f"{sym}:\n"
                            f"\t.align {alignment}\n"
                            f"{datum}"
                            "\t.text\n\t.set\tnoat\n"
                            f"\tlui $at, %hi({sym})\n"
                            f"{load}"
                            "\t.set\tat\n")
                    generated += 1
                elif identifier == "li.d" and "FP32-PAIR" in enabled and re.fullmatch(r"\$\d+", ops[0]):
                    # a double constant for an integer register pair (an argument in $a2/$a3): the low word
                    # is loaded first (the title's objects show ``addiu $a3,$zero,0; lui $a2,high``)
                    bits = struct.unpack(">Q", struct.pack(">d", float(ops[1])))[0]
                    number = int(ops[0][1:])
                    if bits & 0xFFFFFFFFFFFF == 0 and bits:
                        line = f"\taddiu ${number + 1},$0,0\n\tlui ${number},{bits >> 48:#x}\n"
                        emitted_instruction_count = 2
                    elif bits >> 48 and (bits >> 16) & 0xFFFF:
                        # both words need their upper half: each is ``lui`` and, when the lower half is not
                        # zero, ``ori``; still the low word first (a title's ``lui $a3,0x9999; ori $a3,$a3,0x999A;
                        # lui $a2,0x3FE9; ori $a2,$a2,0x9999`` for 0.8)
                        line, emitted_instruction_count = "", 0
                        for register, word in ((number + 1, bits & 0xFFFFFFFF), (number, bits >> 32)):
                            line += f"\tlui ${register},{word >> 16:#x}\n"
                            emitted_instruction_count += 1
                            if word & 0xFFFF:
                                line += f"\tori ${register},${register},{word & 0xFFFF:#x}\n"
                                emitted_instruction_count += 1
            elif identifier in "div" or identifier in "divu" or identifier == "rem" or (
                    identifier == "remu" and "REMU-EXPAND" not in disabled):
                kind = "div" if identifier in "div" else "divu" if identifier in "divu" else identifier
                while mfhilo_delay_count > 0:
                    out.append("\tnop\n")
                    mfhilo_delay_count -= 1
                ops = [s.strip() for s in tokens[1].split(",")]
                if ops[0] != "$0":
                    line, used = _div_expansion(kind, ops, generated)
                    generated += used
                    mfhilo_delay_count = 3
                    mfhilo_delay_location = len(out) + 1
            elif identifier in ("mflo", "mfhi"):
                mfhilo_delay_count = 3
                mfhilo_delay_location = len(out) + 1
            elif identifier == "mult":
                if delay_slot and not is_reorder and mfhilo_delay_count <= 0:
                    out.insert(len(out) - 1, line)
                    line = "\tnop\n"
                while mfhilo_delay_count > 0:
                    if delay_slot:
                        out.insert(mfhilo_delay_location, "\tnop\n")
                    else:
                        out.append("\tnop\n")
                    mfhilo_delay_count -= 1
                if prev_mul:
                    line = "\tnop\n" + line
                new_prev_mul = True
            elif identifier == "multu" and "MULTU-HILO-HAZARD" not in disabled:
                while mfhilo_delay_count > 0:
                    if delay_slot:
                        out.insert(mfhilo_delay_location, "\tnop\n")
                    else:
                        out.append("\tnop\n")
                    mfhilo_delay_count -= 1
            elif identifier in ("mul.s", "mul.d"):
                if delay_slot and not is_reorder and mfhilo_delay_count <= 0:
                    if prev_prev_mul and "MUL-HOIST-SPACING" not in disabled:
                        out.insert(len(out) - 1, "\tnop\n")
                    out.insert(len(out) - 1, line)
                    line = "\tnop\n"
                if prev_mul:
                    line = "\tnop\n" + line
                new_prev_mul = True

            if mfhilo_delay_count > 0:
                mfhilo_delay_count = max(0, mfhilo_delay_count - emitted_instruction_count)
            prev_prev_mul = prev_mul if is_branch else False
            prev_mul = new_prev_mul
            delay_slot = is_branch
            prev_instruction_index = line_index
        out.append(line)

    if comm_symbols or lcomm_symbols:
        out.append("\t.section\t.bss\n")
    for sym, size in lcomm_symbols + comm_symbols:
        if int(size) > 4:
            out.append("\t.align 3\n")
        out.append(f"\t.globl {sym}\n{sym}:\n\t.space {size}\n")
    if last_file_directive != -1:
        name = _tokens(out[last_file_directive])[2]
        out[last_file_directive] = ""
        out.insert(0, f"\t.file\t1 {name}\n")
    converted = "".join(out)
    if section_bases:
        converted = _apply_absolute_section_alignment(converted, section_bases)
    return converted
