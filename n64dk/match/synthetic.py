"""Seeded synthetic programs with recorded ground truth (round 12 product item 9).

Retail ROMs give the kit no ground truth: where a function starts, which unit it came from, what
its types are is what the kit is trying to find.  This module writes programs whose answers are
known by construction, builds them with any recipe through the kit's build path, links them with
a generated script (main segment plus one overlay at another address) and records the truth from
the linked image:

* ``generate(seed, ...)``: a deterministic program of C units (and an assembly unit holding the
  entry routine) covering ``FEATURES``: structures, unions, bitfields, arrays, pointers, volatile
  and memory-mapped I/O, switches dense enough for jump tables, loops, recursion, float and int
  conversions, varargs, K&R definitions, statics, BSS, tentative (COMMON) definitions, small data,
  calls between units, an overlay, function pointers, hand-written assembly;
* ``build_program(program, lane, flags, cfg, work)``: compiles each unit (``pipeline.build``, or
  an injected build), links (``default_link``, or an injected one), and writes ``rom.z64`` plus
  ``manifest.json``: functions (start, size, unit, segment, how they are reached, signature,
  features), data (address, region), layout and digests;
* the manifest's ``class`` is ``synthetic``.  ``assert_synthetic`` refuses anything else, and
  nothing here writes to a project, a case store or an evidence store: synthetic truth is a
  measuring instrument, never evidence about a retail title.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from string import Template

SCHEMA = "n64dk.synthetic/1"
CLASS = "synthetic"
VERSION = 1
FEATURES = ("struct", "union", "bitfield", "array", "pointer", "volatile", "mmio", "switch", "loop", "recursion",
            "float", "conversion", "varargs", "knr", "static", "bss", "common", "small-data", "multi-unit",
            "overlay", "function-pointer", "asm")
MAIN_VRAM, MAIN_ROM, OVERLAY_VRAM = 0x80000400, 0x1000, 0x80400000
ROM_MINIMUM = 0x101000                      # n64dk.rom.Rom refuses anything shorter


@dataclass(frozen=True)
class FunctionTruth:
    name: str
    unit: str
    segment: str
    reached_by: str                         # entry | call | pointer | static-call
    signature: str
    features: tuple = ()
    static: bool = False


@dataclass(frozen=True)
class DataTruth:
    name: str
    unit: str
    ctype: str
    kind: str                               # data | bss | tentative | small | table
    static: bool = False


@dataclass
class Unit:
    name: str
    language: str                           # c | asm
    segment: str                            # main | overlay
    source: str
    functions: list = field(default_factory=list)
    data: list = field(default_factory=list)
    features: list = field(default_factory=list)


@dataclass
class Program:
    seed: int
    units: list
    features: list

    def as_dict(self) -> dict:
        return {"seed": self.seed, "features": list(self.features),
                "units": [{"name": u.name, "language": u.language, "segment": u.segment,
                           "source_sha256": hashlib.sha256(u.source.encode()).hexdigest(),
                           "functions": [asdict(f) for f in u.functions], "data": [asdict(d) for d in u.data],
                           "features": list(u.features)} for u in self.units]}


@dataclass
class Piece:
    feature: str
    code: str
    decls: list
    functions: list                         # (name, signature, static, reached_by)
    data: list = field(default_factory=list)            # (name, ctype, kind, static)
    call: str | None = None
    returns: str = "int"                    # int | float | void
    simple: bool = False                    # an int f(int) another unit or a table may call
    shared: str = ""                        # code for another unit (a second tentative definition)


TYPES = """typedef struct SynVec { int x; short y; char tag; float w; } SynVec;
typedef union SynWord { int i; float f; unsigned char b[4]; } SynWord;
typedef struct SynBits { unsigned kind : 3; unsigned flag : 1; int delta : 5; unsigned rest : 23; } SynBits;
typedef struct SynNode { int value; struct SynNode *next; } SynNode;
#define SYN_IO(addr) (*(volatile unsigned int *)(0xA4600000u | (addr)))
extern SynVec syn_vec;
extern SynWord syn_word;
extern SynBits syn_bits;
extern SynNode syn_nodes[4];
extern unsigned char syn_bytes[16];
extern volatile int syn_sink;
extern volatile float syn_fsink;
"""
GLOBALS = """SynVec syn_vec;
SynWord syn_word;
SynBits syn_bits;
SynNode syn_nodes[4];
unsigned char syn_bytes[16];
volatile int syn_sink;
volatile float syn_fsink;
"""
GLOBAL_DATA = (("syn_vec", "SynVec", "tentative"), ("syn_word", "SynWord", "tentative"),
               ("syn_bits", "SynBits", "tentative"), ("syn_nodes", "SynNode[4]", "tentative"),
               ("syn_bytes", "unsigned char[16]", "tentative"), ("syn_sink", "volatile int", "tentative"),
               ("syn_fsink", "volatile float", "tentative"))


def _t(text: str, **values) -> str:
    return Template(text).substitute(**values)


def _simple(name: str, body: str, feature: str, rng: random.Random, **extra) -> Piece:
    return Piece(feature, body, [f"int {name}(int v);"], [(name, "int (int)", False, "call")],
                 call=f"{name}({rng.randint(1, 9)})", simple=True, **extra)


def _struct(rng, k):
    name = f"syn_struct_{k}"
    code = _t("""int $n(SynVec *v, int d)
{
    v->x += d;
    v->y = (short)(v->x >> $s);
    v->tag = (char)(v->y + $c);
    return v->x + v->tag;
}
""", n=name, s=rng.randint(1, 5), c=rng.randint(1, 99))
    return Piece("struct", code, [f"int {name}(SynVec *v, int d);"], [(name, "int (SynVec *, int)", False, "call")],
                 call=f"{name}(&syn_vec, {rng.randint(1, 50)})")


def _union(rng, k):
    name = f"syn_union_{k}"
    code = _t("""int $n(SynWord *w, int i)
{
    w->i = i * $c;
    return w->b[$a] + (int)w->b[$b];
}
""", n=name, c=rng.randint(3, 999), a=rng.randint(0, 3), b=rng.randint(0, 3))
    return Piece("union", code, [f"int {name}(SynWord *w, int i);"], [(name, "int (SynWord *, int)", False, "call")],
                 call=f"{name}(&syn_word, {rng.randint(1, 50)})")


def _bitfield(rng, k):
    name = f"syn_bits_{k}"
    code = _t("""int $n(SynBits *b, int k)
{
    b->kind = k & 7;
    b->delta = k - $c;
    b->flag = !b->flag;
    return b->kind + b->delta + (int)b->rest;
}
""", n=name, c=rng.randint(1, 15))
    return Piece("bitfield", code, [f"int {name}(SynBits *b, int k);"], [(name, "int (SynBits *, int)", False, "call")],
                 call=f"{name}(&syn_bits, {rng.randint(1, 50)})")


def _array(rng, k):
    name, table = f"syn_array_{k}", f"syn_grid_{k}"
    rows = ", ".join("{" + ", ".join(str(rng.randint(0, 99)) for _ in range(4)) + "}" for _ in range(4))
    code = _t("""int $t[4][4] = { $rows };
int $n(int v)
{
    int j, s = 0;
    for (j = 0; j < 4; j++)
        s += $t[v & 3][j] * $c;
    return s;
}
""", n=name, t=table, rows=rows, c=rng.randint(2, 9))
    return _simple(name, code, "array", rng, data=[(table, "int[4][4]", "data", False)])


def _pointer(rng, k):
    name = f"syn_list_{k}"
    code = _t("""int $n(SynNode *p)
{
    int s = $c;
    while (p != 0) {
        s += p->value;
        p = p->next;
    }
    return s;
}
""", n=name, c=rng.randint(0, 9))
    return Piece("pointer", code, [f"int {name}(SynNode *p);"], [(name, "int (SynNode *)", False, "call")],
                 call=f"{name}(&syn_nodes[0])")


def _volatile(rng, k):
    name, flag = f"syn_volatile_{k}", f"syn_flag_{k}"
    code = _t("""volatile int $f;
int $n(int v)
{
    $f = v;
    $f = $f + $c;
    return $f;
}
""", n=name, f=flag, c=rng.randint(1, 99))
    return _simple(name, code, "volatile", rng, data=[(flag, "volatile int", "tentative", False)])


def _mmio(rng, k):
    name = f"syn_mmio_{k}"
    code = _t("""unsigned int $n(unsigned int value)
{
    while (SYN_IO(0x$s) & 3)
        ;
    SYN_IO(0x$w) = value;
    return SYN_IO(0x$r);
}
""", n=name, s=f"{rng.choice((0x10, 0x14, 0x18)):02X}", w=f"{rng.choice((0x04, 0x08)):02X}",
           r=f"{rng.choice((0x0C, 0x1C)):02X}")
    return Piece("mmio", code, [f"unsigned int {name}(unsigned int value);"],
                 [(name, "unsigned int (unsigned int)", False, "call")], call=f"(int){name}({rng.randint(1, 9)}u)")


def _switch(rng, k):
    name = f"syn_switch_{k}"
    forms = ("op * $m", "op + $m", "$m - op", "op ^ $m", "(op << 2) + $m", "op * op + $m", "$m", "op - $m")
    cases = "\n".join(f"    case {index}: return {_t(form, m=rng.randint(1, 200))};"
                      for index, form in enumerate(rng.sample(forms, len(forms))))
    code = f"int {name}(int op)\n{{\n    switch (op) {{\n{cases}\n    default: return -1;\n    }}\n}}\n"
    return _simple(name, code, "switch", rng)


def _loop(rng, k):
    name = f"syn_loop_{k}"
    code = _t("""int $n(int v)
{
    int s = $c;
    do {
        s = s * 3 + v;
        v--;
    } while (v > 0);
    return s;
}
""", n=name, c=rng.randint(0, 9))
    return _simple(name, code, "loop", rng)


def _recursion(rng, k):
    name = f"syn_recurse_{k}"
    code = _t("""int $n(int v)
{
    if (v < 2)
        return v;
    return $n(v - 1) + $n(v - 2);
}
""", n=name)
    return _simple(name, code, "recursion", rng)


def _float(rng, k):
    name = f"syn_float_{k}"
    code = _t("""float $n(float a, float b)
{
    float t = a * b + $c;
    if (t > b)
        t = t - a;
    return t / 2.0f;
}
""", n=name, c=f"{rng.randint(1, 9)}.5f")
    return Piece("float", code, [f"float {name}(float a, float b);"], [(name, "float (float, float)", False, "call")],
                 call=f"{name}(1.5f, {rng.randint(1, 9)}.0f)", returns="float")


def _conversion(rng, k):
    name = f"syn_convert_{k}"
    code = _t("""int $n(double d, float f, int i)
{
    float g = (float)i;
    double e = d + (double)f;
    return (int)e + (int)(g * 0.5f) + (int)(unsigned char)i + (int)(short)$c;
}
""", n=name, c=rng.randint(1, 30000))
    return Piece("conversion", code, [f"int {name}(double d, float f, int i);"],
                 [(name, "int (double, float, int)", False, "call")], call=f"{name}(2.5, 1.5f, {rng.randint(1, 99)})")


def _varargs(rng, k):
    name = f"syn_varargs_{k}"
    code = _t("""int $n(int count, ...)
{
    int *p = &count + 1;
    int s = $c;
    while (count-- > 0)
        s += *p++;
    return s;
}
""", n=name, c=rng.randint(0, 9))
    return Piece("varargs", code, [f"int {name}(int count, ...);"], [(name, "int (int, ...)", False, "call")],
                 call=f"{name}(3, {rng.randint(1, 9)}, {rng.randint(1, 9)}, {rng.randint(1, 9)})")


def _knr(rng, k):
    name = f"syn_knr_{k}"
    code = _t("""int $n(c, s, f)
    char c;
    short s;
    float f;
{
    return c + s * $m + (int)f;
}
""", n=name, m=rng.randint(2, 9))
    return Piece("knr", code, [f"int {name}();"], [(name, "int (char, short, float) K&R", False, "call")],
                 call=f"{name}({rng.randint(1, 9)}, {rng.randint(1, 9)}, 3.0f)")


def _static(rng, k):
    name, helper, value = f"syn_static_{k}", f"syn_helper_{k}", f"syn_counter_{k}"
    code = _t("""static int $v = $c;
static int $h(int x)
{
    return x * $v + 1;
}
int $n(int v)
{
    $v += v;
    return $h(v) + $h(v + 1);
}
""", n=name, h=helper, v=value, c=rng.randint(1, 99))
    piece = _simple(name, code, "static", rng, data=[(value, "int", "data", True)])
    piece.functions.append((helper, "int (int)", True, "static-call"))
    return piece


def _bss(rng, k):
    name, buffer = f"syn_bss_{k}", f"syn_buffer_{k}"
    size = rng.choice((16, 32, 64))
    code = _t("""static int $b[$z];
int $n(int v)
{
    $b[v & $m] += v;
    return $b[(v + 1) & $m];
}
""", n=name, b=buffer, z=size, m=size - 1)
    return _simple(name, code, "bss", rng, data=[(buffer, f"int[{size}]", "bss", True)])


def _common(rng, k):
    name, value = f"syn_common_{k}", f"syn_shared_{k}"
    code = _t("""int $v;
int $n(int v)
{
    $v += v;
    return $v;
}
""", n=name, v=value)
    return _simple(name, code, "common", rng, data=[(value, "int", "tentative", False)],
                   shared=f"int {value};\n")


def _small(rng, k):
    name, word, half = f"syn_small_{k}", f"syn_word_{k}", f"syn_half_{k}"
    code = _t("""int $w = $c;
short $h;
int $n(int v)
{
    $h = (short)v;
    return $w + $h;
}
""", n=name, w=word, h=half, c=rng.randint(1, 99))
    return _simple(name, code, "small-data", rng, data=[(word, "int", "small", False), (half, "short", "small", False)])


def _multi(rng, k):
    name = f"syn_cross_{k}"
    code = _t("""int $n(int v)
{
    return $$OTHER(v + $c) + 1;
}
""", n=name, c=rng.randint(1, 9))
    return _simple(name, code, "multi-unit", rng)


def _overlay(rng, k):
    name = f"syn_overlay_{k}"
    code = _t("""int $n(int v)
{
    int i, s = 0;
    for (i = 0; i < v; i++)
        s += i ^ $c;
    return s;
}
""", n=name, c=rng.randint(1, 255))
    return _simple(name, code, "overlay", rng)


TEMPLATES = {"struct": _struct, "union": _union, "bitfield": _bitfield, "array": _array, "pointer": _pointer,
             "volatile": _volatile, "mmio": _mmio, "switch": _switch, "loop": _loop, "recursion": _recursion,
             "float": _float, "conversion": _conversion, "varargs": _varargs, "knr": _knr, "static": _static,
             "bss": _bss, "common": _common, "small-data": _small, "multi-unit": _multi, "overlay": _overlay}
BASIC = ("struct", "union", "bitfield", "array", "pointer", "volatile", "switch", "loop", "recursion", "float",
         "conversion", "static", "bss")

ENTRY = """\t.set\tnoreorder
\t.text
\t.globl\t_start
\t.ent\t_start
_start:
\tlui\t$$29, 0x8040
\tjal\tsyn_main
\taddiu\t$$29, $$29, -16
1:\tb\t1b
\tnop
\t.end\t_start
"""
ASM_FILL = """
\t.globl\tasm_fill
\t.ent\tasm_fill
asm_fill:
\tbeq\t$$5, $$0, 2f
\tnop
1:\tsb\t$$6, 0($$4)
\taddiu\t$$5, $$5, -1
\tbne\t$$5, $$0, 1b
\taddiu\t$$4, $$4, 1
2:\tjr\t$$31
\tnop
\t.end\tasm_fill
"""


def generate(seed: int, features=FEATURES, units: int = 3, extra: int = 6) -> Program:
    """A deterministic program for ``seed``: one function per requested feature, ``extra`` more
    from the basic templates, spread over ``units`` C units (an ``overlay`` feature adds unit
    ``ovl`` in the overlay segment), unit ``u0`` holding the shared globals and ``syn_main``, and
    the assembly unit ``entry`` holding ``_start`` (and ``asm_fill`` with the ``asm`` feature)."""
    rng = random.Random(seed)
    wanted = [feature for feature in FEATURES if feature in set(features)]
    count = max(1, units, 2 if "multi-unit" in wanted else 1)
    names = [f"u{index}" for index in range(count)]
    picks = [feature for feature in wanted if feature in TEMPLATES] + [rng.choice(BASIC) for _ in range(extra)]
    rng.shuffle(picks)
    placed: dict[str, list[Piece]] = {name: [] for name in names}
    if "overlay" in wanted:
        placed["ovl"] = []
    shared: dict[str, list[str]] = {name: [] for name in placed}
    for index, feature in enumerate(picks):
        piece = TEMPLATES[feature](rng, index)
        unit = "ovl" if feature == "overlay" else names[index % len(names)]
        placed[unit].append(piece)
        if piece.shared:
            shared[names[(names.index(unit) + 1) % len(names)] if unit in names else names[0]].append(piece.shared)
    called: set[str] = set()
    for unit, pieces in placed.items():                 # calls into another unit
        for piece in pieces:
            if "$OTHER" not in piece.code:
                continue
            others = [(u, p) for u, ps in placed.items() for p in ps if u not in (unit, "ovl") and p.simple
                      and "$OTHER" not in p.code]
            if not others:
                helper = _loop(rng, 1000 + len(picks))
                target_unit = next(name for name in names if name != unit)
                placed[target_unit].append(helper)
                others = [(target_unit, helper)]
            _other_unit, other = others[rng.randrange(len(others))]
            piece.code = piece.code.replace("$OTHER", other.functions[0][0])
            called.add(other.functions[0][0])
    simple = [p for u, ps in placed.items() if u != "ovl" for p in ps
              if p.simple and p.feature != "multi-unit" and p.functions[0][0] not in called]
    table = []
    if "function-pointer" in wanted and simple:
        for piece in rng.sample(simple, min(2, len(simple))):
            name, signature, static, _how = piece.functions[0]
            piece.functions[0] = (name, signature, static, "pointer")
            piece.call = None
            table.append(name)
    decls = TYPES + "".join(line + "\n" for ps in placed.values() for p in ps for line in p.decls)
    if "asm" in wanted:
        decls += "void asm_fill(unsigned char *dst, int count, int value);\n"
    if table:
        decls += f"extern int (*syn_table[{len(table)}])(int);\n"
    decls += "void syn_main(void);\n"
    calls = []
    for ps in placed.values():
        for p in ps:
            if p.call is None:
                continue
            calls.append(f"    syn_fsink += {p.call};" if p.returns == "float" else f"    syn_sink += {p.call};")
    if "asm" in wanted:
        calls.append("    asm_fill(syn_bytes, 8, 0x55);")
    body = "void syn_main(void)\n{\n    int i;\n" + "\n".join(calls) + "\n"
    if table:
        body += f"    for (i = 0; i < {len(table)}; i++)\n        syn_sink += syn_table[i](i);\n"
    else:
        body += "    i = 0;\n    syn_sink += i;\n"
    body += "}\n"
    out_units = []
    entry = _t(ENTRY) + (_t(ASM_FILL) if "asm" in wanted else "")
    entry_functions = [FunctionTruth("_start", "entry", "main", "entry", "asm entry routine", ("asm",))]
    if "asm" in wanted:
        entry_functions.append(FunctionTruth("asm_fill", "entry", "main", "call",
                                             "void (unsigned char *, int, int) asm", ("asm",)))
    out_units.append(Unit("entry", "asm", "main", f"# n64dk synthetic unit entry: seed {seed}; class synthetic\n"
                          + entry, entry_functions, [], ["asm"]))
    for unit, pieces in placed.items():
        segment = "overlay" if unit == "ovl" else "main"
        code = "".join(piece.code + "\n" for piece in pieces) + "".join(shared.get(unit, []))
        functions = [FunctionTruth(name, unit, segment, how, signature, (piece.feature,), static)
                     for piece in pieces for name, signature, static, how in piece.functions]
        data = [DataTruth(name, unit, ctype, kind, static) for piece in pieces for name, ctype, kind, static in piece.data]
        features_here = sorted({piece.feature for piece in pieces})
        if unit == "u0":
            code = GLOBALS + (f"int (*syn_table[{len(table)}])(int) = {{ {', '.join(table)} }};\n" if table else "") \
                + code + body
            functions.append(FunctionTruth("syn_main", "u0", "main", "call", "void (void)", ("main",)))
            data += [DataTruth(name, "u0", ctype, kind) for name, ctype, kind in GLOBAL_DATA]
            if table:
                data.append(DataTruth("syn_table", "u0", f"int (*[{len(table)}])(int)", "table"))
                features_here = sorted(set(features_here) | {"function-pointer"})
        source = (f"/* n64dk synthetic unit {unit}: seed {seed}, generator {VERSION}; class synthetic */\n"
                  + decls + "\n" + code)
        out_units.append(Unit(unit, "c", segment, source, functions, data, features_here))
    covered = sorted({f for u in out_units for f in u.features} | ({"multi-unit"} if len(names) > 1 else set()))
    return Program(seed, out_units, covered)


# ---------------------------------------------------------------------------------------------
# linking and ground truth

DISCARD = ("*(.reginfo) *(.MIPS.abiflags) *(.mdebug*) *(.pdr) *(.comment) *(.options) *(.gptab*) *(.note*) "
           "*(.debug*) *(.gnu.attributes)")


def linker_script(objects: list[tuple[str, str]], entry: str = "entry.o") -> str:
    """A GNU ld script: segment ``main`` at ``MAIN_VRAM`` (ROM ``MAIN_ROM``; ``entry`` first, then
    every other main object's text, read-only data, data, small data, then its BSS) and, when an
    object is in the ``overlay`` segment, segment ``ovl`` at ``OVERLAY_VRAM`` loaded right after
    ``main`` in ROM.  Layout symbols (``main_TEXT_END``, ``main_BSS_START`` ...) bound each region."""
    overlay = [name for name, segment in objects if segment == "overlay"]
    exclude = f"EXCLUDE_FILE({' '.join(overlay)}) " if overlay else ""
    lines = ["OUTPUT_ARCH(mips)", "ENTRY(_start)", "SECTIONS", "{",
             f"  .main 0x{MAIN_VRAM:08X} : AT(0x{MAIN_ROM:X})", "  {", "    main_TEXT_START = .;",
             f"    {entry}(.text)", f"    *({exclude}.text .text.*)", "    main_TEXT_END = .;",
             f"    *({exclude}.rodata .rodata.* .rdata .lit4 .lit8)", f"    *({exclude}.data .data.*)",
             "    . = ALIGN(16);", "    main_SDATA_START = .;", "    _gp = . + 0x7FF0;",
             f"    *({exclude}.sdata .sdata.*)", "    main_SDATA_END = .;", "    . = ALIGN(16);",
             "    main_DATA_END = .;", "  }", "  .main.bss (NOLOAD) :", "  {", "    main_BSS_START = .;",
             "    main_SBSS_START = .;", f"    *({exclude}.sbss .scommon)", "    main_SBSS_END = .;",
             f"    *({exclude}.bss .bss.* COMMON)", "    main_BSS_END = .;", "  }"]
    if overlay:
        lines += [f"  .ovl 0x{OVERLAY_VRAM:08X} : AT(0x{MAIN_ROM:X} + SIZEOF(.main))", "  {", "    ovl_TEXT_START = .;",
                  *[f"    {name}(.text .text.*)" for name in overlay], "    ovl_TEXT_END = .;",
                  *[f"    {name}(.rodata .rodata.* .rdata .lit4 .lit8 .data .data.* .sdata .sdata.*)" for name in overlay],
                  "    ovl_DATA_END = .;", "  }", "  .ovl.bss (NOLOAD) :", "  {", "    ovl_BSS_START = .;",
                  *[f"    {name}(.sbss .scommon .bss .bss.* COMMON)" for name in overlay], "    ovl_BSS_END = .;",
                  "  }"]
    lines += [f"  /DISCARD/ : {{ {DISCARD} }}", "}", ""]
    return "\n".join(lines)


def make_rom(main: bytes, overlay: bytes, entry: int, title: str = "N64DK SYNTHETIC") -> bytes:
    """A big-endian ROM: header (entry point at 8), an empty IPL3 area, ``main`` at ``MAIN_ROM``,
    ``overlay`` right after it, zero padding to at least ``ROM_MINIMUM`` bytes."""
    header = struct.pack(">IIIIII", 0x80371240, 0x0000000F, entry, 0x00001444, 0, 0) + bytes(8)
    header += title.encode("ascii")[:20].ljust(20, b" ") + bytes(7) + b"N" + b"SY" + b"E" + b"\0"
    rom = header.ljust(MAIN_ROM, b"\0") + main + overlay
    rom += bytes((-len(rom)) % 4)
    return rom.ljust(ROM_MINIMUM, b"\0")


def truth_from_elf(program: Program, elf) -> dict:
    """Layout, functions and data of ``program`` read from its linked image: every start is a
    symbol value; a size is the symbol's size, else the distance to the next known function or to
    the end of the segment's text."""
    symbols: dict[str, object] = {}
    for sym in elf.symbols:
        if sym.name and sym.name not in symbols and 0 < sym.section < len(elf.sections):
            symbols[sym.name] = sym

    def value(name: str):
        sym = symbols.get(name)
        return sym.value if sym is not None else None

    layout = {"entry": value("_start")}
    for segment in ("main", "ovl"):
        keys = ("TEXT_START", "TEXT_END", "DATA_END", "SDATA_START", "SDATA_END", "SBSS_START", "SBSS_END",
                "BSS_START", "BSS_END")
        found = {key.lower(): value(f"{segment}_{key}") for key in keys}
        section = elf.section(f".{segment}")
        if section is not None and found["text_start"] is not None:
            found["size"] = len(section.data)
            found["vram"] = MAIN_VRAM if segment == "main" else OVERLAY_VRAM
            layout[segment] = found
    starts: dict[str, list[int]] = {}
    present = []
    for unit in program.units:
        for function in unit.functions:
            segment = "main" if function.segment == "main" else "ovl"
            start = value(function.name)
            present.append((function, segment, start))
            if start is not None:
                starts.setdefault(segment, []).append(start)
    functions = []
    for function, segment, start in present:
        row = {**asdict(function), "features": list(function.features), "present": start is not None}
        if start is not None:
            sym = symbols[function.name]
            later = [s for s in starts.get(segment, []) if s > start]
            text_end = (layout.get(segment) or {}).get("text_end") or start
            row["vram"] = start
            row["size"] = sym.size or ((min(later) if later else text_end) - start)
            row["size_source"] = "symbol" if sym.size else "next start"
        functions.append(row)
    data = []
    for unit in program.units:
        for item in unit.data:
            address = value(item.name)
            data.append({**asdict(item), "vram": address, "region": region(layout, address)})
    return {"layout": layout, "functions": functions, "data": data}


def region(layout: dict, address: int | None) -> str:
    if address is None:
        return "absent"
    for segment in ("main", "ovl"):
        bounds = layout.get(segment)
        if not bounds:
            continue

        def inside(lo, hi):
            return bounds.get(lo) is not None and bounds.get(hi) is not None and bounds[lo] <= address < bounds[hi]
        prefix = "" if segment == "main" else "ovl-"
        for name, lo, hi in (("text", "text_start", "text_end"), ("sdata", "sdata_start", "sdata_end"),
                             ("data", "text_end", "data_end"), ("sbss", "sbss_start", "sbss_end"),
                             ("bss", "bss_start", "bss_end")):
            if inside(lo, hi):
                return prefix + name
    return "outside"


def assert_synthetic(manifest: dict) -> dict:
    """``manifest`` when it is a synthetic corpus manifest; ValueError otherwise (retail evidence
    and synthetic truth are never mixed)."""
    if not isinstance(manifest, dict) or manifest.get("class") != CLASS or manifest.get("schema") != SCHEMA:
        raise ValueError("not a synthetic corpus manifest")
    return manifest


def default_link(cfg, folder: Path, script: str, objects: list[str], output: str, timeout: float = 120.0) -> Path:
    """GNU ld (the binutils the kit already uses for source donors) in ``folder``."""
    from .. import safety
    from .sourcedonor import tool
    linker, env = tool(cfg, "ld")
    argv = [linker, "-EB", "-T", script, "-Map", "link.map", "-o", output, "--unresolved-symbols=ignore-all",
            "--no-warn-mismatch", "--no-check-sections", *objects]
    done = safety.run_tool(argv, kind="compile", timeout=timeout, cwd=str(folder), env={**os.environ, **env},
                           capture_output=True, text=True)
    (Path(folder) / "link.log").write_text((done.stdout or "") + (done.stderr or ""))
    path = Path(folder) / output
    if done.returncode != 0 and not path.is_file():
        raise RuntimeError("link failed: " + (done.stderr or "")[-400:])
    return path


def build_program(program: Program, lane: str, flags, cfg, work, *, build=None, link=None,
                  timeout: float = 120.0) -> dict:
    """Compile every unit (C with ``lane`` + ``flags``; the assembly unit with the ``as`` lane),
    link, write ``rom.z64`` and ``manifest.json`` under ``work`` and return the manifest.  A unit
    that does not build or a failed link gives a manifest with ``status`` and ``errors``."""
    from ..elf import Elf
    from .pipeline import BuildError, split_flags
    if build is None:
        from .pipeline import build
    link = link or default_link
    work = Path(work)
    sources, objects = work / "src", work / "link"
    sources.mkdir(parents=True, exist_ok=True)
    objects.mkdir(parents=True, exist_ok=True)
    recipe = " ".join([lane, *flags])
    manifest = {"schema": SCHEMA, "class": CLASS, "evidence": "synthetic ground truth: never retail evidence",
                "generator": {"version": VERSION, "seed": program.seed, "features": list(program.features)},
                "recipe": recipe, "lane": lane, "flags": list(flags), "program": program.as_dict(), "errors": []}
    built = []
    for unit in program.units:
        path = sources / f"{unit.name}.{'s' if unit.language == 'asm' else 'c'}"
        path.write_text(unit.source)
        pipeline, unit_flags = ("as", []) if unit.language == "asm" else (lane, list(flags))
        try:
            obj = Path(build(path, split_flags(unit_flags, pipeline), cfg, work / "build" / unit.name))
        except (BuildError, OSError, RuntimeError) as error:
            manifest["errors"].append({"unit": unit.name, "stage": "build", "error": str(error)[-400:]})
            continue
        target = objects / f"{unit.name}.o"
        shutil.copyfile(obj, target)
        built.append((unit, target))
    if manifest["errors"]:
        manifest["status"] = "build-failed"
        return _write(work, manifest)
    script = linker_script([(target.name, unit.segment) for unit, target in built])
    (objects / "link.ld").write_text(script)
    try:
        elf_path = Path(link(cfg, objects, "link.ld", [target.name for _unit, target in built], "program.elf", timeout))
        elf = Elf(elf_path.read_bytes())
    except Exception as error:                     # noqa: BLE001 - a link failure is a corpus record
        manifest["errors"].append({"unit": None, "stage": "link", "error": str(error)[-400:]})
        manifest["status"] = "link-failed"
        return _write(work, manifest)
    truth = truth_from_elf(program, elf)
    main = elf.section(".main")
    overlay = elf.section(".ovl")
    entry = truth["layout"].get("entry") or MAIN_VRAM
    rom = make_rom(main.data if main is not None else b"", overlay.data if overlay is not None else b"", entry)
    (work / "rom.z64").write_bytes(rom)
    if overlay is not None and "ovl" in truth["layout"]:
        truth["layout"]["ovl"]["rom"] = MAIN_ROM + len(main.data if main is not None else b"")
    if "main" in truth["layout"]:
        truth["layout"]["main"]["rom"] = MAIN_ROM
    manifest.update(truth)
    manifest["status"] = "built"
    manifest["elf_sha256"] = hashlib.sha256(elf.data).hexdigest()
    manifest["rom"] = {"name": "rom.z64", "sha256": hashlib.sha256(rom).hexdigest(), "size": len(rom)}
    manifest["absent_functions"] = [row["name"] for row in truth["functions"] if not row["present"]]
    return _write(work, manifest)


def _write(work: Path, manifest: dict) -> dict:
    (Path(work) / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    return manifest


def discovery_truth(manifest: dict) -> list[tuple[str, int, int]]:
    """``(name, vram, size)`` of the main segment's present functions: the reference for
    ``n64dk.analysis.functions.compare``."""
    assert_synthetic(manifest)
    return sorted(((row["name"], row["vram"], row["size"]) for row in manifest.get("functions", [])
                   if row.get("present") and row["segment"] == "main"), key=lambda item: item[1])
