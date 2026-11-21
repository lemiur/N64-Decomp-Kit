"""Historical C dialect of each recipe, read from object code (round 12 addendum, item B).

A draft is C, and how a compiler of the 1990s reads C is not how a modern one does: a call to
an unprototyped function promotes ``float`` to ``double``, an undeclared function returns
``int``, plain ``char`` may be unsigned, an old-style definition receives its narrow parameters
promoted, a tentative definition may become a ``COMMON`` symbol, type-based aliasing may or may
not let a load be reused.  Each such fact changes the bytes a draft compiles to, so the kit
measures them per recipe instead of assuming them.

``PROBES`` is a corpus of small translation units.  Each holds a pair (or a few) of functions
that differ only in the construct being asked about, and states the question and how the answer
is read from the object.  ``classify(probe, elf)`` reads the answer (pure: an ``n64dk.elf.Elf``
in, facts out; ``None`` for a probe that did not build).  ``matrix(rows)`` turns the readings of
every recipe into a recipe x fact table, ``facts_for(recipe)`` reads a stored table back.

``bench/dialect/run.py`` builds the corpus with every installed recipe, one compile at a time.
Reporting only: no fact here changes a build, a score or a verdict.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .passprobe import FP, GPR, SP, V0, WIDE, Insn, ObjectView


@dataclass(frozen=True)
class Probe:
    name: str
    facts: tuple[str, ...]
    question: str
    reading: str
    source: str


PROBES: tuple[Probe, ...] = (
    Probe("call_promotion", ("unprototyped_float_arg", "prototyped_narrow_arg"),
          "Is a float passed to an unprototyped callee promoted to double, and does a call through a prototype "
          "with char/short parameters narrow an int argument at the call site?",
          "cvt.d.s in call_k (and not in call_p) = promoted to double; sll/sra 24 and 16 (or andi 0xff/0xffff) in "
          "call_p and not in call_k = the caller narrows",
          "extern void take_p(char c, short s, float f);\n"
          "extern void take_k();\n"
          "void call_p(int i, int j, float f) { take_p(i, j, f); }\n"
          "void call_k(int i, int j, float f) { take_k(i, j, f); }\n"),
    Probe("implicit_decl", ("implicit_int", "narrow_return"),
          "Is an undeclared function taken to return int, and does a caller re-extend a short return value?",
          "cvt.s.w in use_undecl_f = the result is read as an int and converted; an extension after the call in "
          "use_decl = the caller re-extends",
          "short ret_s(void);\n"
          "int use_decl(void) { return ret_s() + 1; }\n"
          "int use_undecl(void) { return ret_u() + 1; }\n"
          "float use_undecl_f(void) { return ret_f() * 2.0f; }\n"),
    Probe("char_sign", ("plain_char",),
          "Is plain char signed or unsigned?",
          "the byte load of plain_char: lb = signed, lbu without a 24-bit sign extension = unsigned (signed_char "
          "and unsigned_char are the controls)",
          "int plain_char(char *p) { char c = *p; return c; }\n"
          "int signed_char(signed char *p) { signed char c = *p; return c; }\n"
          "int unsigned_char(unsigned char *p) { unsigned char c = *p; return c; }\n"),
    Probe("enum_size", ("enum_size", "enum_size_large", "enum_in_struct"),
          "How wide is an enum, and how is it laid out inside a structure?",
          "the constants returned by size_small/size_big; the byte offset enum_offset loads (8 = 4-byte enum, "
          "4 = 2-byte, 2 = 1-byte)",
          "enum Small { SA, SB, SC };\n"
          "enum Big { BA = 0x12345 };\n"
          "struct WithEnum { char a; enum Small e; char b; };\n"
          "int size_small(void) { return sizeof(enum Small); }\n"
          "int size_big(void) { return sizeof(enum Big); }\n"
          "int enum_offset(struct WithEnum *s) { return s->b; }\n"),
    Probe("bitfields", ("bitfield_order", "plain_int_bitfield"),
          "Are bitfields allocated from the most significant bit, and is a plain int bitfield signed?",
          "get_a: a right shift by (load width - 3) = most significant first, an andi 7 without shift = least "
          "significant first; get_c: sra = signed (get_d, declared signed, is the control)",
          "struct BF { unsigned a : 3; unsigned b : 5; int c : 4; signed int d : 4; unsigned e : 16; };\n"
          "int get_a(struct BF *p) { return p->a; }\n"
          "int get_c(struct BF *p) { return p->c; }\n"
          "int get_d(struct BF *p) { return p->d; }\n"),
    Probe("knr_params", ("knr_narrow_params", "proto_narrow_params", "knr_float_param"),
          "Does an old-style definition narrow its char/short parameters on entry, does a prototyped one, and "
          "does an old-style float parameter arrive as a double?",
          "extensions (sll/sra, andi, or a narrow store to the frame) in knr_narrow and proto_narrow; cvt.s.d in "
          "knr_float",
          "int knr_narrow(c, s) char c; short s; { return c + s; }\n"
          "int proto_narrow(char c, short s) { return c + s; }\n"
          "float knr_float(f) float f; { return f * f; }\n"
          "float proto_float(float f) { return f * f; }\n"),
    Probe("tentative", ("tentative_definition", "zero_initialized", "static_tentative"),
          "Where do a tentative definition, an explicitly zero-initialised definition and a static tentative "
          "definition go?",
          "the symbol's section index: COMMON (SHN_COMMON), SCOMMON, or a section name (.bss, .sbss, .data)",
          "int tent_def;\n"
          "int zero_init = 0;\n"
          "int one_init = 1;\n"
          "static int stat_tent;\n"
          "int use_tent(void) { stat_tent++; return tent_def + zero_init + one_init + stat_tent; }\n"),
    Probe("volatile", ("volatile_loads", "nonvolatile_reuse", "volatile_in_delay_slot"),
          "Does volatile keep both loads, is a plain global loaded once, and may a volatile store fill a call's "
          "delay slot?",
          "lw count in vol_twice and nonvol_twice; whether the global store sits in the jal delay slot of "
          "vol_store_call and nonvol_store_call",
          "extern volatile int vreg;\n"
          "extern int nreg;\n"
          "extern void tick(void);\n"
          "int vol_twice(void) { return vreg + vreg; }\n"
          "int nonvol_twice(void) { return nreg + nreg; }\n"
          "void vol_store_call(int a) { vreg = a; tick(); }\n"
          "void nonvol_store_call(int a) { nreg = a; tick(); }\n"),
    Probe("aliasing", ("type_based_aliasing",),
          "Is a load reused across a store through a pointer of another type (type-based alias analysis)?",
          "loads of *p (lw at offset 0, not from the frame) in each function: 1 = reused, 2 = reloaded; the char "
          "and int versions must reload",
          "int alias_char(int *p, char *q) { int a = *p; *q = 0; return a + *p; }\n"
          "int alias_short(int *p, short *q) { int a = *p; *q = 0; return a + *p; }\n"
          "int alias_float(int *p, float *q) { int a = *p; *q = 0.0f; return a + *p; }\n"
          "int alias_int(int *p, int *q) { int a = *p; *q = 0; return a + *p; }\n"),
    Probe("float_precision", ("float_arithmetic", "double_constant"),
          "Is float arithmetic done in single precision, and is a double constant honoured?",
          "fexpr: mul.s/add.s = single, cvt.d.s or mul.d/add.d = double (K&R excess precision); fconst: mul.d or "
          "cvt.d.s = double",
          "float fexpr(float a, float b, float c) { return a * b + c; }\n"
          "float fconst(float a) { return a * 2.0; }\n"),
    Probe("long_long", ("ll_shift", "ll_multiply", "ll_divide", "ll_add", "ll_runtime"),
          "Are long long operations inline or calls into a runtime, and which runtime?",
          "the jal relocations of each function (__ashldi3/__muldi3/__divdi3: libgcc; __ll_*: IDO); 64-bit "
          "instructions without a call = inline 64-bit",
          "long long ll_shl(long long a, int n) { return a << n; }\n"
          "long long ll_mul(long long a, long long b) { return a * b; }\n"
          "long long ll_div(long long a, long long b) { return a / b; }\n"
          "long long ll_add(long long a, long long b) { return a + b; }\n"),
    Probe("register_kw", ("register_keyword",),
          "Does the register keyword change the code?",
          "no_register and with_register differ only in the keyword: identical words = no effect",
          "extern int sink(int);\n"
          "int no_register(int *a, int n) { int i, s; s = 0; for (i = 0; i < n; i++) s += a[i]; return sink(s); }\n"
          "int with_register(int *a, int n) { register int i, s; s = 0; for (i = 0; i < n; i++) s += a[i]; "
          "return sink(s); }\n"),
    Probe("inline_plain", ("inline_keyword",),
          "Is `inline` accepted, and is the function inlined?",
          "a build failure = rejected; a jal in use_inline = accepted, not inlined; none = inlined",
          "static inline int sq(int x) { return x * x + 1; }\n"
          "int use_inline(int a) { return sq(a) + sq(a + 1); }\n"),
    Probe("inline_gnu", ("inline_gnu_keyword",),
          "Is `__inline__` accepted, and is the function inlined?",
          "as inline_plain",
          "static __inline__ int sq(int x) { return x * x + 1; }\n"
          "int use_inline(int a) { return sq(a) + sq(a + 1); }\n"),
    Probe("inline_sgi", ("inline_sgi_keyword",),
          "Is `__inline` accepted, and is the function inlined?",
          "as inline_plain",
          "static __inline int sq(int x) { return x * x + 1; }\n"
          "int use_inline(int a) { return sq(a) + sq(a + 1); }\n"),
)
BY_NAME = {probe.name: probe for probe in PROBES}
FACTS = {fact: probe.name for probe in PROBES for fact in probe.facts}


# ---------------------------------------------------------------------------------------------
# readings

FRAME = (SP, FP)


def _count(insns: list[Insn], *names: str) -> int:
    return sum(1 for insn in insns if insn.name in names)


def _extensions(insns: list[Insn]) -> int:
    """Sign or zero extensions of an 8- or 16-bit value: sll/sra (or srl) pairs by 16 or 24, andi
    0xff/0xffff."""
    lefts = sum(1 for insn in insns if insn.name == "sll" and insn.word and insn.shamt in (16, 24))
    rights = sum(1 for insn in insns if insn.name in ("sra", "srl") and insn.shamt in (16, 24))
    masks = sum(1 for insn in insns if insn.name == "andi" and insn.uimm in (0xFF, 0xFFFF))
    return min(lefts, rights) + masks


def _narrow_stores(insns: list[Insn]) -> int:
    return sum(1 for insn in insns if insn.name in ("sb", "sh") and insn.base in FRAME)


def _memory_loads(insns: list[Insn], name: str = "lw", offset: int | None = None) -> int:
    return sum(1 for insn in insns if insn.name == name and insn.base not in FRAME
               and (offset is None or insn.imm == offset))


def _constant(insns: list[Insn]) -> int | None:
    """The last constant put in $v0 (addiu/ori from $zero)."""
    value = None
    for insn in insns:
        if insn.name in ("addiu", "ori") and insn.operands == ((GPR, 0), (GPR, V0)):
            value = insn.imm if insn.name == "addiu" else insn.uimm
    return value


def _store_in_call_slot(insns: list[Insn]) -> bool:
    return any(insn.name == "jal" and k + 1 < len(insns) and insns[k + 1].name == "sw"
               and insns[k + 1].base not in FRAME for k, insn in enumerate(insns))


def _call_promotion(view: ObjectView) -> dict:
    p, k = view.insns("call_p"), view.insns("call_k")
    if p is None or k is None:
        return {}
    cvt_p, cvt_k = _count(p, "cvt.d.s"), _count(k, "cvt.d.s")
    if cvt_k and not cvt_p:
        promotion = ("double", True)
    elif cvt_k:
        promotion = ("double-even-with-prototype", True)
    elif not cvt_p:
        promotion = ("float", True)
    else:
        promotion = ("inconclusive", False)
    ext_p, ext_k = _extensions(p), _extensions(k)
    narrow = (("caller-narrows", True) if ext_p > ext_k else ("caller-passes-int", True) if ext_p == ext_k == 0
              else ("inconclusive", False))
    return {"unprototyped_float_arg": (promotion[0], [f"cvt.d.s: call_k {cvt_k}, call_p {cvt_p}"], promotion[1]),
            "prototyped_narrow_arg": (narrow[0], [f"narrowing sequences: call_p {ext_p}, call_k {ext_k}"],
                                      narrow[1])}


def _implicit_decl(view: ObjectView) -> dict:
    out = {}
    as_float = view.insns("use_undecl_f")
    if as_float is not None:
        conversions = _count(as_float, "cvt.s.w")
        out["implicit_int"] = ("int" if conversions else "not-int",
                               [f"cvt.s.w in use_undecl_f: {conversions}"], True)
    declared = view.insns("use_decl")
    if declared is not None:
        call = next((k for k, insn in enumerate(declared) if insn.name == "jal"), None)
        after = declared[call + 2:] if call is not None else declared
        again = _extensions(after) + sum(1 for insn in after if insn.name in ("lh", "lhu") and insn.base in FRAME)
        out["narrow_return"] = ("caller-reextends" if again else "caller-trusts-callee",
                                [f"extensions after the call to ret_s: {again}"], call is not None)
    return out


def _byte_sign(insns: list[Insn] | None) -> str | None:
    if insns is None:
        return None
    loads = [insn.name for insn in insns if insn.name in ("lb", "lbu")]
    if not loads:
        return None
    if loads[-1] == "lb":
        return "signed"
    extended = any(insn.name == "sra" and insn.shamt == 24 for insn in insns)
    return "signed" if extended else "unsigned"


def _char_sign(view: ObjectView) -> dict:
    plain = _byte_sign(view.insns("plain_char"))
    signed, unsigned = _byte_sign(view.insns("signed_char")), _byte_sign(view.insns("unsigned_char"))
    if plain is None:
        return {}
    controls = signed == "signed" and unsigned == "unsigned"
    return {"plain_char": (plain, [f"plain_char reads {plain}; controls: signed char {signed}, unsigned char "
                                   f"{unsigned}"], controls)}


def _enum_size(view: ObjectView) -> dict:
    out = {}
    for function, fact in (("size_small", "enum_size"), ("size_big", "enum_size_large")):
        insns = view.insns(function)
        value = _constant(insns) if insns is not None else None
        if value is not None:
            out[fact] = (str(value), [f"{function} returns {value}"], True)
    insns = view.insns("enum_offset")
    if insns is not None:
        offset = next((insn.imm for insn in insns if insn.name in ("lb", "lbu") and insn.base not in FRAME), None)
        width = {8: "4", 4: "2", 2: "1"}.get(offset) if offset is not None else None
        if width is not None:
            out["enum_in_struct"] = (width, [f"the char after the enum is read at offset {offset}"], True)
    return out


def _bitfields(view: ObjectView) -> dict:
    out = {}
    first = view.insns("get_a")
    if first is not None:
        load = next((insn for insn in first if insn.name in ("lw", "lh", "lhu", "lb", "lbu")
                     and insn.base not in FRAME), None)
        width = {"lw": 32, "lh": 16, "lhu": 16, "lb": 8, "lbu": 8}.get(load.name) if load is not None else None
        shifted = width is not None and any(insn.name in ("srl", "sra") and insn.shamt == width - 3 for insn in first)
        masked = any(insn.name == "andi" and insn.uimm == 7 for insn in first)
        if shifted:
            out["bitfield_order"] = ("msb-first", [f"{load.name} then a right shift by {width - 3}"], True)
        elif masked and not any(insn.name in ("srl", "sra") for insn in first):
            out["bitfield_order"] = ("lsb-first", ["andi 7 without a shift"], True)
        else:
            out["bitfield_order"] = ("unreadable", ["no recognised extraction of the first field"], False)
    plain, control, unsigned = view.insns("get_c"), view.insns("get_d"), first
    if plain is not None:
        signed = _count(plain, "sra") > 0
        controls = (control is not None and _count(control, "sra") > 0
                    and (unsigned is None or _count(unsigned, "sra") == 0))
        out["plain_int_bitfield"] = ("signed" if signed else "unsigned",
                                     [f"sra in get_c: {_count(plain, 'sra')}"], controls)
    return out


def _knr_params(view: ObjectView) -> dict:
    out = {}
    for function, fact, narrows, trusts in (("knr_narrow", "knr_narrow_params", "callee-narrows", "trusts-promotion"),
                                            ("proto_narrow", "proto_narrow_params", "callee-narrows", "trusts-caller")):
        insns = view.insns(function)
        if insns is not None:
            found = _extensions(insns) + _narrow_stores(insns)
            out[fact] = (narrows if found else trusts, [f"narrowing in {function}: {found}"], True)
    insns = view.insns("knr_float")
    if insns is not None:
        conversions = _count(insns, "cvt.s.d")
        out["knr_float_param"] = ("arrives-as-double" if conversions else "arrives-as-float",
                                  [f"cvt.s.d in knr_float: {conversions}"], True)
    return out


def _tentative(view: ObjectView) -> dict:
    out = {}
    for symbol, fact in (("tent_def", "tentative_definition"), ("zero_init", "zero_initialized"),
                         ("stat_tent", "static_tentative")):
        place = view.place(symbol)
        out[fact] = (place or "no-symbol", [f"{symbol}: {place or 'no symbol in the object'}"], place is not None)
    return out


def _volatile(view: ObjectView) -> dict:
    out = {}
    twice, plain = view.insns("vol_twice"), view.insns("nonvol_twice")
    if twice is not None:
        loads = _memory_loads(twice)
        out["volatile_loads"] = ("kept" if loads >= 2 else "merged", [f"lw in vol_twice: {loads}"], True)
    if plain is not None:
        loads = _memory_loads(plain)
        out["nonvolatile_reuse"] = ("reused" if loads == 1 else "reloaded", [f"lw in nonvol_twice: {loads}"], True)
    stored, plain_stored = view.insns("vol_store_call"), view.insns("nonvol_store_call")
    if stored is not None and plain_stored is not None:
        vol, non = _store_in_call_slot(stored), _store_in_call_slot(plain_stored)
        answer = ("allowed", True) if vol else ("kept-out", True) if non else ("no-slot-filling", False)
        out["volatile_in_delay_slot"] = (answer[0], [f"store in the jal slot: volatile {vol}, plain {non}"],
                                         answer[1])
    return out


def _aliasing(view: ObjectView) -> dict:
    loads = {}
    for function in ("alias_char", "alias_short", "alias_float", "alias_int"):
        insns = view.insns(function)
        if insns is None:
            return {}
        loads[function] = _memory_loads(insns, "lw", 0)
    evidence = [", ".join(f"{name} {count}" for name, count in loads.items()) + " load(s) of *p"]
    if loads["alias_char"] < 2 or loads["alias_int"] < 2:
        return {"type_based_aliasing": ("inconclusive", evidence, False)}
    if loads["alias_short"] < 2 or loads["alias_float"] < 2:
        return {"type_based_aliasing": ("yes", evidence, True)}
    return {"type_based_aliasing": ("no", evidence, True)}


def _float_precision(view: ObjectView) -> dict:
    out = {}
    expression = view.insns("fexpr")
    if expression is not None:
        double = _count(expression, "cvt.d.s", "mul.d", "add.d")
        single = _count(expression, "mul.s", "add.s")
        answer = "double" if double else "single" if single else "unreadable"
        out["float_arithmetic"] = (answer, [f"double operations {double}, single {single}"], answer != "unreadable")
    constant = view.insns("fconst")
    if constant is not None:
        double = _count(constant, "cvt.d.s", "mul.d")
        out["double_constant"] = ("double" if double else "single", [f"double operations in fconst: {double}"], True)
    return out


def _long_long(view: ObjectView) -> dict:
    out = {}
    called: set[str] = set()
    for function, fact in (("ll_shl", "ll_shift"), ("ll_mul", "ll_multiply"), ("ll_div", "ll_divide"),
                           ("ll_add", "ll_add")):
        insns = view.insns(function)
        if insns is None:
            continue
        calls = sorted(set(view.calls(function)))
        called |= set(calls)
        if calls:
            answer = "call:" + ",".join(calls)
        elif any(insn.name in WIDE for insn in insns):
            answer = "inline-64bit"
        else:
            answer = "inline-32bit"
        out[fact] = (answer, [f"{function}: {answer}"], True)
    if out:
        if any(name.startswith("__ll_") for name in called):
            runtime = "ido-ll"
        elif called & {"__ashldi3", "__muldi3", "__divdi3", "__lshrdi3", "__ashrdi3", "__moddi3"}:
            runtime = "libgcc"
        elif called:
            runtime = "other:" + ",".join(sorted(called))
        else:
            runtime = "inline"
        out["ll_runtime"] = (runtime, [f"callees: {sorted(called) or 'none'}"], True)
    return out


def _register_kw(view: ObjectView) -> dict:
    plain, keyword = view.words("no_register"), view.words("with_register")
    if plain is None or keyword is None:
        return {}
    frame = [sum(1 for insn in view.insns(name) if insn.memory and insn.base in FRAME)
             for name in ("no_register", "with_register")]
    return {"register_keyword": ("no-effect" if plain == keyword else "changes-code",
                                 [f"frame accesses: without {frame[0]}, with {frame[1]}"], True)}


def _inline(fact: str):
    def read(view: ObjectView) -> dict:
        insns = view.insns("use_inline")
        if insns is None:
            return {}
        calls = _count(insns, "jal")
        return {fact: ("inlined" if not calls else "accepted-not-inlined", [f"jal in use_inline: {calls}"], True)}
    return read


READERS = {"call_promotion": _call_promotion, "implicit_decl": _implicit_decl, "char_sign": _char_sign,
           "enum_size": _enum_size, "bitfields": _bitfields, "knr_params": _knr_params, "tentative": _tentative,
           "volatile": _volatile, "aliasing": _aliasing, "float_precision": _float_precision,
           "long_long": _long_long, "register_kw": _register_kw, "inline_plain": _inline("inline_keyword"),
           "inline_gnu": _inline("inline_gnu_keyword"), "inline_sgi": _inline("inline_sgi_keyword")}


def classify(probe: Probe | str, elf) -> dict[str, dict]:
    """fact -> {"answer", "evidence", "conclusive"} for one probe's object (``elf`` None: the probe
    did not build; for the inline probes that is the answer "rejected")."""
    probe = BY_NAME[probe] if isinstance(probe, str) else probe
    if elf is None:
        answer = "rejected" if probe.name.startswith("inline_") else "build-failed"
        return {fact: {"answer": answer, "evidence": ["the probe did not build"],
                       "conclusive": answer == "rejected"} for fact in probe.facts}
    try:
        found = READERS[probe.name](ObjectView(elf))
    except Exception as error:                     # noqa: BLE001 - an unreadable object is a reading
        found = {fact: ("unreadable", [f"{type(error).__name__}: {error}"[:160]], False) for fact in probe.facts}
    out = {}
    for fact in probe.facts:
        answer, evidence, conclusive = found.get(fact, ("unreadable", ["the function was not found"], False))
        out[fact] = {"answer": answer, "evidence": list(evidence), "conclusive": bool(conclusive)}
    return out


def matrix(rows: list[dict]) -> dict:
    """Recipe x fact table from probe readings ``[{"recipe", "probe", "built", "facts", "error"}]``.
    An inline probe's refusal counts only when the recipe built its other probes (the compiler
    works); ``differing`` lists the facts on which conclusive readings disagree between recipes."""
    recipes: dict[str, dict] = {}
    for row in rows:
        entry = recipes.setdefault(row["recipe"], {"facts": {}, "built": 0, "failed": 0, "errors": {}})
        if row.get("built"):
            entry["built"] += 1
        else:
            entry["failed"] += 1
            if row.get("error"):
                entry["errors"][row["probe"]] = str(row["error"])[:200]
        for fact, value in (row.get("facts") or {}).items():
            entry["facts"][fact] = dict(value)
    for entry in recipes.values():
        working = entry["built"] > 0
        for value in entry["facts"].values():
            if value.get("answer") == "rejected" and not working:
                value["conclusive"] = False
    facts = sorted({fact for entry in recipes.values() for fact in entry["facts"]})
    table = {recipe: {fact: entry["facts"].get(fact, {}).get("answer", "-") for fact in facts}
             for recipe, entry in sorted(recipes.items())}
    differing = {}
    for fact in facts:
        answers = {recipe: entry["facts"][fact]["answer"] for recipe, entry in recipes.items()
                   if fact in entry["facts"] and entry["facts"][fact].get("conclusive")}
        if len(set(answers.values())) > 1:
            differing[fact] = dict(sorted(answers.items()))
    return {"questions": {fact: BY_NAME[FACTS[fact]].question for fact in facts if fact in FACTS},
            "recipes": recipes, "table": table, "differing": differing}


def markdown(result: dict) -> str:
    """The matrix as a Markdown table (inconclusive readings marked with ``?``)."""
    table = result.get("table") or {}
    facts = sorted({fact for row in table.values() for fact in row})
    lines = ["| fact | " + " | ".join(table) + " |", "|---" * (len(table) + 1) + "|"]
    for fact in facts:
        cells = []
        for recipe in table:
            value = (result["recipes"][recipe]["facts"].get(fact) or {})
            answer = table[recipe].get(fact, "-")
            cells.append(answer + ("" if value.get("conclusive", True) or answer == "-" else " ?"))
        lines.append(f"| {fact} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _level(recipe: str) -> str | None:
    return next((part for part in recipe.split()[1:] if part.startswith(("-O", "-g"))), None)


def facts_for(recipe: str, path: str | Path | None = None) -> dict[str, str]:
    """The stored dialect facts of ``recipe`` (``lane flags``): the exact recipe, else the same lane
    at the same level, else the lane.  The table is ``path``, ``$N64DK_DIALECT_MATRIX`` or
    ``data/dialect_matrix.json`` beside this module; {} when there is none."""
    location = Path(path or os.environ.get("N64DK_DIALECT_MATRIX") or Path(__file__).parent / "data"
                    / "dialect_matrix.json")
    if not location.is_file():
        return {}
    table = (json.loads(location.read_text()) or {}).get("table") or {}
    if recipe in table:
        return dict(table[recipe])
    parts = recipe.split()
    if not parts:
        return {}
    same_lane = [key for key in sorted(table) if key.split() and key.split()[0] == parts[0]]
    level = _level(recipe)
    for key in same_lane:
        if _level(key) == level:
            return dict(table[key])
    return dict(table[same_lane[0]]) if same_lane else {}
