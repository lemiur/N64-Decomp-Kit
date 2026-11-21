"""Round trips for the two unoptimised-code translators: compile a small function without
optimisation, hand its machine code to the translator, compile what it writes, compare bytes."""
import atexit
import shutil
import tempfile
import unittest
from pathlib import Path

from n64dk.match import redraft, redraft_gcc_debug, redraft_ido_debug

TYPES = "typedef signed char s8; typedef unsigned char u8; typedef short s16; typedef unsigned short u16;\n" \
        "typedef int s32; typedef unsigned int u32; typedef float f32; typedef double f64;\n"
EXTERNALS = "extern s32 D_80100000; extern s32 D_80100010[]; extern u8 D_80100100[]; extern s16 D_80100200;\n" \
            "s32 func_80002000(s32, s32); void func_80002100(void);\n" \
            "struct S { s32 a; s16 b; u8 c; s8 d; s32 e[4]; struct S *next; };\n"
PROGRAMS = {
    "arith": "s32 f(s32 a, s32 b) {\n    s32 x;\n    s32 y;\n\n    x = a + b;\n    y = x * 2;\n    if (y > 10) {\n"
             "        y = y - a;\n    }\n    return y;\n}\n",
    "loop": "s32 f(s32 n) {\n    s32 i;\n    s32 t;\n\n    t = 0;\n    for (i = 0; i < n; i++) {\n        t += i;\n    }\n"
            "    return t;\n}\n",
    "bytes": "void f(u8 *p, s32 n) {\n    s32 i;\n\n    for (i = 0; i < n; i++) {\n        p[i] = 0;\n    }\n}\n",
    "dead": "void f(s32 a) {\n    s32 unused;\n\n    unused = a + 3;\n}\n",
    "call": "s32 f(s32 a) {\n    s32 r;\n\n    r = func_80002000(a, 3);\n    func_80002100();\n    return r + 1;\n}\n",
    "global": "void f(s32 a) {\n    D_80100000 = a;\n    D_80100010[2] = D_80100000 + 1;\n    D_80100200 = 5;\n}\n",
    "index": "s32 f(s32 i) {\n    return D_80100010[i] + D_80100100[i];\n}\n",
    "field": "s32 f(struct S *p) {\n    p->a = 1;\n    p->b = 2;\n    p->c = 3;\n    return p->e[1] + p->d;\n}\n",
    "while": "s32 f(struct S *p) {\n    s32 n;\n\n    n = 0;\n    while (p != 0) {\n        n++;\n        p = p->next;\n    }\n"
             "    return n;\n}\n",
    "dowhile": "s32 f(s32 n) {\n    s32 t;\n\n    t = 0;\n    do {\n        t += n;\n        n--;\n    } while (n > 0);\n"
               "    return t;\n}\n",
    "ifelse": "s32 f(s32 a, s32 b) {\n    s32 r;\n\n    if (a < b) {\n        r = a;\n    } else if (a == b) {\n        r = 0;\n"
              "    } else {\n        r = b;\n    }\n    return r;\n}\n",
    "logic": "s32 f(s32 a, s32 b) {\n    if (a > 0 && b > 0) {\n        return 1;\n    }\n    if (a < 0 || b < 0) {\n"
             "        return 2;\n    }\n    return 0;\n}\n",
    "narrow": "s32 f(u8 a, s16 b) {\n    u8 x;\n    s16 y;\n\n    x = a + 1;\n    y = b - x;\n    return y;\n}\n",
    "float": "f32 f(f32 a, f32 b) {\n    f32 r;\n\n    r = a * b + 1.5f;\n    if (r < a) {\n        r = a;\n    }\n    return r;\n}\n",
    "fcall": "f32 func_80002200(f32, s32);\nf32 f(f32 a, s32 n) {\n    return func_80002200(a * 2.0f, n) - 0.25f;\n}\n",
    "fconv": "f32 f(s32 n, f32 x) {\n    s32 k;\n\n    k = x;\n    return n + k * 0.5f;\n}\n",
    "fcmp": "s32 f(f32 a, f32 b) {\n    if (a > b) {\n        return 1;\n    }\n    if (a <= b) {\n        return 2;\n    }\n"
            "    if (a == b) {\n        return 3;\n    }\n    return 0;\n}\n",
    "fglobal": "extern f32 D_80100300; extern f32 D_80100310[];\nvoid f(s32 i) {\n    D_80100300 = D_80100310[i] / 3.0f;\n}\n",
    "switch3": "s32 f(s32 a) {\n    s32 r;\n\n    r = 0;\n    switch (a) {\n    case 1:\n        r = 10;\n        break;\n"
               "    case 2:\n        r = 20;\n    case 5:\n        r += 3;\n        break;\n    default:\n        r = -1;\n    }\n"
               "    return r;\n}\n",
    "gfx": "typedef struct { s32 w0; s32 w1; } G;\nextern G *D_80100400;\nvoid f(void) {\n    G *g;\n\n"
           "    g = D_80100400++;\n    g->w0 = 0x06000000;\n    g->w1 = 5;\n    g = D_80100400++;\n    g->w0 = 7;\n}\n",
    "manyargs": "void f(s16 a, s16 b, s16 c, s16 d, s16 e, u8 g) {\n    D_80100200 = a + e + g;\n}\n",
    "midinc": "struct R { s32 a; s32 n; u8 *q; };\nvoid f(u8 *out, s32 count, struct R *r) {\n    s32 i;\n\n"
              "    for (i = 0; i < count; i++) {\n        out[i] = r->q[r->n++];\n    }\n}\n",
    "logicval": "s32 f(s32 a, s32 b) {\n    s32 r;\n\n    r = a < -5 || a > 5;\n    return r && b != 3;\n}\n",
    "ternary": "s32 f(s32 a, s32 b) {\n    return func_80002000(a, (a & 4) ? 0x1234 : b) + 1;\n}\n",
    "array": "s32 f(void) {\n    s32 buf[4];\n    s32 i;\n\n    for (i = 0; i < 4; i++) {\n        buf[i] = i;\n    }\n"
             "    return buf[2];\n}\n",
    "switch": "s32 f(s32 a) {\n    switch (a) {\n    case 1:\n        return 10;\n    case 2:\n        return 20;\n    }\n"
              "    return 0;\n}\n",
    "address": "s32 f(void) {\n    s32 v;\n\n    v = 4;\n    return func_80002000((s32) &v, v);\n}\n",
    "unsigned": "u32 f(u32 a, u32 b) {\n    if (a < b) {\n        return b / a;\n    }\n    return a >> 3;\n}\n",
}
# Forms of unoptimised GCC established on Wheel of Fortune (USA), 2026-11-18: each was settled with a compiler
# probe and is checked here by the same round trip.  GCC only (the IDO translator has its own forms).
GCC_FORMS = {
    "indexed_float": "extern f32 D_80100600[];\nf32 f(s32 i) {\n    return D_80100600[i];\n}\n",
    "indexed_offset": "s32 f(s32 i) {\n    D_80100010[i + 3] = D_80100100[i + 1];\n    return D_80100010[i - 1];\n}\n",
    "indexed_twice": "s32 f(s32 i, s32 j) {\n    while (i < j) {\n        if (D_80100010[i] == D_80100010[j]) {\n"
                     "            break;\n        }\n        i++;\n    }\n    while (j > 0) {\n        j--;\n    }\n    return i;\n}\n",
    "struct_argument": "struct B { s32 w[5]; };\nextern struct B D_80100700;\nvoid func_80002200(struct B);\n"
                       "void f(void) {\n    func_80002200(D_80100700);\n}\n",
    "register": "s32 f(register s32 *p, s32 n) {\n    register s32 t;\n\n    t = func_80002000(p[1], n);\n    p++;\n"
                "    return t + *p + func_80002000(0, 0);\n}\n",
    "struct_array": "struct E { s32 a; s16 b; u8 c; u8 d; s32 e; };\nextern struct E D_80100500[];\n"
                    "s32 f(s32 i) {\n    D_80100500[i].a = 1;\n    return D_80100500[i].e + D_80100500[i].b;\n}\n",
    "struct_return": "struct R { s32 w[3]; };\nstruct R f(s32 a) {\n    struct R r;\n\n    r.w[1] = func_80002000(a, 1);\n"
                     "    if (a) {\n        r.w[0] = 0;\n        return r;\n    }\n    r.w[0] = 2;\n    return r;\n}\n",
    "struct_copy": "struct R { s32 w[3]; };\nvoid f(struct R *d, struct R *s) {\n    *d = *s;\n}\n",
    "divide_power": "s32 f(s32 x) {\n    return x / 2 + x / 8;\n}\n",
    "divide_constant": "s32 f(s32 x) {\n    return x / 3 + x % 10;\n}\n",
    "remainder_test": "s32 f(s32 x) {\n    if (x % 2) {\n        return 1;\n    }\n    return 0;\n}\n",
    "post_decrement": "struct P { s32 pad[4]; s32 n; };\nvoid f(struct P *p) {\n    p->n--;\n    p->n++;\n}\n",
    "plus_equals": "struct P { s32 pad[4]; s32 n; };\nvoid f(struct P *p, s32 y) {\n    p->n += y + 1;\n}\n",
    "or_constant": "void f(s32 a, s32 b, s32 *w) {\n"
                   "    *w = ((((a + 0x66) & 0x3FF) << 14) | 0xE4000000) | (((b + 0x66) & 0x3FF) << 2);\n}\n",
    "member_array_address": "struct G { u8 pad[0x1A40]; struct { u8 b[64]; } m[4]; };\n"
                            "s32 f(struct G *p, s32 i) {\n    return (s32) &p->m[i];\n}\n",
    "member_scalar_array": "struct A { char pad[0x264]; s32 arr[8]; };\nvoid f(struct A *p, s32 i) {\n"
                           "    p->arr[i] = func_80002000(1, 2);\n}\n",
    "shift_pair": "s32 f(s32 x) {\n    s32 y;\n\n    y = (x << 24) >> 24;\n    return y;\n}\n",
    "unsigned_switch": "s32 f(u32 x) {\n    switch (x) {\n    case 0:\n        return 5;\n    case 1:\n        return 6;\n"
                       "    case 2:\n        return 7;\n    case 5:\n        return 9;\n    }\n    return 0;\n}\n",
    "while_and": "extern s32 D_80100600; extern u32 D_80100604;\nvoid f(void) {\n"
                 "    while (D_80100600 && D_80100604 < 100) {\n    }\n    func_80002100();\n}\n",
    "and_else": "s32 f(s32 a, s32 b) {\n    s32 r;\n\n    if (a == 0 && b == 0) {\n        r = 1;\n    } else {\n"
                "        r = func_80002000(a, b);\n    }\n    return r;\n}\n",
    "or_condition": "s32 f(s32 a, s32 b) {\n    s32 r;\n\n    r = 0;\n    if (a == 1 || b == 2) {\n"
                    "        r = func_80002000(a, b);\n    }\n    return r;\n}\n",
    "call_before_condition": "s32 f(s32 a, s32 b) {\n    s32 r;\n\n    r = 0;\n    func_80002100();\n"
                             "    if (a == 1 || b == 2) {\n        r = 3;\n    }\n    return r;\n}\n",
    "chained_assignment": "struct Q { s32 a; s32 b; s32 c; };\nvoid f(struct Q *p) {\n    p->a = p->b = p->c = 0;\n}\n",
    "assignment_condition": "extern s32 D_80100700[];\ns32 f(s32 i) {\n"
                            "    if ((D_80100700[i] = func_80002000(i, 0)) != 0) {\n        return 1;\n    }\n    return 0;\n}\n",
    "two_dimensions": "extern s32 D_80100800[][8];\ns32 f(s32 i, s32 j) {\n    return D_80100800[i][j];\n}\n",
    "float_in_integer_register": "void func_80002300(s32, s32, s32, f32);\nvoid f(f32 x) {\n"
                                 "    func_80002300(1, 2, 3, x);\n}\n",
    "comparison_order": "s32 f(s32 a, s32 b) {\n    if (a == b - 2) {\n        return 1;\n    }\n"
                        "    if (D_80100010[a] > b) {\n        return 2;\n    }\n    return 0;\n}\n",
    "scratch_register": "s32 f(s32 *a, s32 n) {\n    register s32 *p;\n    register s32 t;\n\n    p = a;\n    t = 0;\n"
                        "    while (n > 0) {\n        t += *p;\n        p++;\n        n--;\n    }\n    return t;\n}\n",
    # control flow and what the same functions needed (agent a3)
    "product_of_three": "s32 f(s32 a, s32 b, s32 c) {\n    return a * b * c;\n}\n",
    "empty_else": "s32 f(s32 a) {\n    s32 r;\n\n    r = 0;\n    if (a) {\n        r = func_80002000(a, 1);\n    } else {\n    }\n    return r;\n}\n",
    "narrow_bit_test": "struct F { u16 flags; u8 b; u8 c; s32 x; };\nextern struct F D_80100900[];\ns32 f(s32 i) {\n    if (D_80100900[i].flags & 4) {\n        return 1;\n    }\n    if (D_80100900[i].b & 1) {\n        return 2;\n    }\n    return 0;\n}\n",
    "converted_call_condition": "s32 f(s32 a) {\n    s32 x;\n\n    if ((x = (s32) func_80002000(a, 0)) != 0) {\n        return x;\n    }\n    return 0;\n}\n",
    "source_labels": "s32 f(s32 a) {\nagain:\n    switch (func_80002000(a, 0)) {\n    case 0:\n        goto next;\n    case -1:\n        goto fail;\n    default:\n        goto again;\n    }\nnext:\n    switch (func_80002000(a, 1)) {\n    case 1:\n        goto last;\n    case 0:\n        goto next;\n    case -1:\n        goto fail;\n    default:\n        goto again;\n    }\nlast:\n    return func_80002000(a, 2) + 256;\nfail:\n    return -1;\n}\n",
    "goto_past_switch": "s32 f(s32 a, s32 n) {\n    s32 i;\n\n    switch (a) {\n    case 1:\n        func_80002100();\n        goto done;\n    case 2:\n        func_80002100();\n        break;\n    default:\n        if (a > 100) {\n            goto done;\n        }\n        break;\n    }\n    for (i = 0; i < n; i++) {\n        if (func_80002000(i, 0) != 0) {\n            goto done;\n        }\n    }\n    func_80002100();\ndone:\n    return n;\n}\n",
    "or_of_ands": "s32 f(s32 a, s32 b) {\n    s32 r;\n\n    r = 0;\n    if ((a == 1 && b == 2 && r == 0) || (a == 3 && b == 4 && r == 0)) {\n        r = func_80002000(a, b);\n    }\n    return r;\n}\n",
    "or_of_ands_constants": "s32 f(s32 a, s32 b) {\n    s32 r;\n\n    r = 0;\n    if ((a == 1 && b == 2 && r == 5) || (a == 3 && b == 4 && a + b == 7)) {\n        r = func_80002000(a, b);\n    }\n    return r;\n}\n",
    "long_chained_assignment": "void f(s32 *q) {\n    s32 *p;\n\n    func_80002100();\n    p = q;\n    p[0] = p[1] = p[2] = p[3] = p[4] = p[5] = p[6] = p[7] = p[8] = p[9] = p[10] = p[11] = p[12] = p[13] = p[14] = p[15] = p[16] = p[17] = p[18] = p[19] = p[20] = p[21] = p[22] = p[23] = p[24] = p[25] = p[26] = p[27] = p[28] = p[29] = p[30] = p[31] = 0;\n}\n",
    "plus_equals_difference": "s32 f(s32 x, s32 y) {\n    x += 1 - (x & 1);\n    y += 4 - x;\n    return x + y;\n}\n",
    "indexed_member_increment": "struct E { s32 a; s16 b; u8 c; u8 d; s32 e; };\nextern struct E D_80100A00[];\nvoid f(s32 i) {\n    D_80100A00[i].e++;\n    D_80100A00[i].a--;\n}\n",
    "register_in_first_loop": "void f(void) {\n    s32 i;\n\n    for (i = 0; i < 8; i++) {\n        register s32 r;\n\n"
                              "        r = func_80002000(i, 1);\n        D_80100010[i] = r + func_80002000(i, 2);\n    }\n}\n",
    # a register variable of an inner block: its register is a temporary before and after the block
    "scratch_block": "void f(s16 *a) {\n    register s16 *p;\n    s32 i;\n    s32 d;\n\n    p = a;\n"
                     "    for (i = 7; i >= 0; i--) {\n        register s32 *q = (s32 *) p;\n\n        d = p[0];\n"
                     "        if (q[1] == 0) {\n            p += 8;\n            continue;\n        }\n"
                     "        p[1] = d + 1;\n        p += 8;\n    }\n}\n",
    # a register variable assigned and read inside one basic block (no copy before the two stores)
    "scratch_value": "void f(s32 *a, s32 c) {\n    register s32 *p;\n\n    p = a;\n    if (c) {\n"
                     "        register s32 v = (c & 0xffff) | (c << 16);\n\n        p[0] = v;\n        p[1] = v;\n"
                     "    }\n    p[2] = 0;\n}\n",
    "negative_multiplier": "s32 f(s32 x, s32 z) {\n    s32 a;\n\n    a = x * -10703;\n    return z + x * -15137 + a;\n}\n",
    "negated_product": "s32 f(s32 x) {\n    return -(x * 7);\n}\n",
    "case_range": "s32 f(s32 a) {\n    s32 r;\n\n    switch (a) {\n    case 1:\n    case 2:\n        r = 2;\n        break;\n"
                  "    case 6:\n        r = 0;\n        break;\n    default:\n        r = 1;\n    }\n    return r;\n}\n",
    "case_range_jump": "s32 f(u8 c) {\n    s32 r;\n\n    r = 0;\n    switch ((u8) (c >> 4)) {\n    case 12:\n    case 13:\n"
                       "        r = 1;\n        break;\n    case 14:\n        r = 2;\n        break;\n    case 15:\n"
                       "        r = 3;\n        break;\n    }\n    return r;\n}\n",
    "byte_and_test": "s32 f(u8 *p, s32 i) {\n    if (p[i] & 0x80) {\n        return 1;\n    }\n    return 0;\n}\n",
    # 64-bit integers travel as register pairs: a result in $v0:$v1, an argument in $a0:$a1 or $a2:$a3
    "wide_call": "typedef unsigned long long W;\nW func_80002300(void); W func_80002400(W, W); extern W D_80100600;\n"
                 "void f(void) {\n    s32 a;\n\n    a = func_80002300() & 0xFFF;\n"
                 "    D_80100000 = func_80002400(func_80002300(), D_80100600) + a;\n}\n",
    "wide_store": "typedef unsigned long long W;\nextern W D_80100600;\nvoid f(void) {\n    D_80100600 = 0;\n"
                  "    D_80100000 = 1;\n    D_80100600++;\n    D_80100000 = 2;\n    D_80100600 += 1;\n}\n",
    "if_then_while": "s32 f(s32 a, s32 *p, s32 i) {\n    if (a != 3) {\n        while (p[i] != 0) {\n            i = p[i];\n"
                     "        }\n    }\n    return i;\n}\n",
    # floating point under -mfp32 (a double is a pair of float registers and moves a word at a time)
    "double_pair": "extern f64 D_80100900;\nf64 f(f64 a) {\n    f64 y;\n\n    y = a * D_80100900;\n"
                   "    D_80100900 = y;\n    return y;\n}\n",
    "double_compare": "extern f64 D_80100900;\ns32 f(f32 x) {\n    if (x > D_80100900) {\n        return 1;\n    }\n"
                      "    return 0;\n}\n",
    "double_in_integer_registers": "s32 func_80002400();\nvoid f(s32 a, f64 y, f32 x) {\n    func_80002400(a, y);\n"
                                   "    func_80002400(a, 2, x);\n    func_80002400(a, 2, 2.5);\n}\n",
    "double_on_the_stack": "s32 func_80002400();\nvoid f(s32 a, f64 y, f32 x) {\n    func_80002400(a, a, a, y, x);\n}\n",
    "double_result": "f64 func_80002600();\nf64 f(f64 a) {\n    f64 y;\n\n    y = func_80002600(a);\n    return y;\n}\n",
    "conditional_ending_a_conditional":
        "s32 f(s32 x, s32 k) {\n    s32 y;\n\n    y = x - 128;\n"
        "    y = (y >= 0) ? ((y < 128) ? (y + k) : 127) : ((y >= -128) ? (y - k) : -128);\n"
        "    return y;\n}\n",
    "double_constant_whole_words": "s32 func_80002400();\nvoid f(s32 a) {\n    func_80002400(a, 2, 0.8);\n"
                                   "    func_80002400(a, 2, 1234.5678);\n}\n",
    "division_inside_a_conditional": "void f(s32 *p, s32 w, s32 k) {\n    p[1] = ((w / 4) > 0) ? (k | (w / 4)) : (k | 1);\n}\n",
    "double_zero": "void f(f64 *p) {\n    f64 y;\n\n    y = 0.0;\n    *p = 0.0;\n    *p = y;\n}\n",
    "float_operand_order": "extern f32 D_80100910; extern f32 D_80100914;\nvoid f(s32 i) {\n"
                           "    D_80100914 = D_80100910 + (f32)i;\n    D_80100914 = (f32)i + D_80100910;\n}\n",
    "float_compare_memory": "struct F { s32 pad[3]; f32 m; };\ns32 f(struct F *p, s32 n) {\n    f32 x;\n\n    x = n;\n"
                            "    if (x < p->m) {\n        return 1;\n    }\n    return 0;\n}\n",
    "float_chain": "extern f32 D_80100910; extern f32 D_80100914;\nvoid f(void) {\n    f32 x;\n    f32 z;\n\n"
                   "    x = z = 0.0f;\n    D_80100910 = D_80100914 = 0.0f;\n}\n",
    "float_stack_arguments": "extern f32 D_80100910;\nvoid func_80002500(s32, s32, s32, s32, f32, f32);\n"
                             "void f(f32 x) {\n    func_80002500(1, 2, 3, 4, x, x + D_80100910);\n}\n",
    "float_constant_arguments": "void func_80002700(s32, f32, f32, f32, f32, f32);\nvoid f(s32 a, f32 x) {\n"
                                "    func_80002700(a, 33.0f, x, 0.0f, x, 0.0f);\n"
                                "    func_80002700(a, x, x, x, x, x);\n}\n",
    "conditional_value": "s32 f(s32 x) {\n    s32 y;\n\n    y = (x < -256) ? -256 : (x > 255) ? 255 : x;\n"
                         "    return y;\n}\n",
    "conditional_call": "f64 func_80002600();\nextern f64 D_80100900; extern f64 D_80100908;\nf64 f(s32 i) {\n"
                        "    f64 y;\n\n    y = (i == 0) ? func_80002600(D_80100900) : D_80100908;\n    return y;\n}\n",
    "address_kept_across_call": "f64 func_80002600();\nextern f64 D_80100B00[][8];\nvoid f(s32 i, s32 j, f64 y) {\n"
                                "    D_80100B00[i][j] = y * ({ func_80002600(y); });\n}\n",
    "byte_grid_read": "extern u8 D_80100A00[][21];\ns32 f(s32 i, s32 j) {\n    return D_80100A00[i][j];\n}\n",
    # expression spellings (agent a5, same title, same day)
    "pre_increment_operand": "s32 f(s32 a) {\n    s32 x;\n\n    x = a;\n    if (++x >= 5) {\n        return 1;\n    }\n"
                             "    return x;\n}\n",
    "void_pointer_increment": "s32 f(s32 a) {\n    void *p;\n    s32 n;\n\n    n = 0;\n"
                              "    p = (void *) &D_80100010[a];\n    while (p != 0) {\n        p = (void *) func_80002000((s32) p, a);\n"
                              "        if (p != 0) {\n            n++;\n            p++;\n        }\n    }\n    return n;\n}\n",
    "element_pointer_increment": "void f(s32 a, s32 n) {\n    s32 *p;\n    s32 i;\n\n    p = (s32 *) a;\n"
                                 "    for (i = 0; i < n; i += 4) {\n        *p = *p;\n        p++;\n    }\n}\n",
    "narrow_unsigned_divide": "s32 f(s32 a) {\n    u16 h;\n\n    h = a;\n    if (a < h / 4) {\n        return a + h / 2;\n    }\n"
                              "    return (h / 4) * 4;\n}\n",
    "if_zero": "s32 f(s32 a, s32 n) {\n    s32 i;\n\n    if (a != 1) {\n        return 1;\n    }\n    if (0) {\n"
               "        for (i = 0; i < n; i++) {\n            func_80002100();\n        }\n    }\n    return a + n;\n}\n",
    "constant_moved": "s32 f(struct S *p, struct S *q, s32 a, s32 b) {\n    s32 r;\n\n    r = p->a - 1 - q->a;\n"
                      "    r += p->a - (q->a + 1);\n    r += (a + 1) + b;\n    r += a + (b + 1);\n    return r;\n}\n",
    "compare_then_return": "void f(s32 a) {\n    if (D_80100000 < 318) {\n        return;\n    }\n    D_80100000 = a;\n}\n",
    "times_minus_one": "s32 f(s32 x) {\n    if (x < 0) {\n        x = x * -1;\n    }\n    x = -x;\n    return x;\n}\n",
    "local_pointer_sum": "s32 f(s32 n) {\n    u8 buf[20];\n\n    buf[n] = 45;\n    return func_80002000((s32) (buf + (n + 1)), 0);\n}\n",
    "unsigned_mask": "void f(struct S *p, u16 a, u16 b) {\n    p->b = (a - b + 96) & 0xFFF0u;\n}\n",
    "pointer_difference": "struct G8 { s32 w[2]; };\ns32 f(struct G8 *p, struct G8 *q) {\n"
                          "    return (p - q) * sizeof(struct G8);\n}\n",
    "equality_value": "void f(s32 a, s32 b) {\n    D_80100010[0] = a == b;\n    D_80100010[1] = a != b;\n"
                      "    D_80100010[2] = a == 5;\n    D_80100010[3] = a != 0;\n}\n",
    "byte_grid": "extern u8 D_80100900[][21];\ns32 f(s32 i, s32 j) {\n    if (D_80100900[i][j] == 43) {\n"
                 "        return func_80002000((s32) &D_80100900[i][j], 0);\n    }\n    return 0;\n}\n",
    "post_increment_index": "extern s32 D_80100A00;\nvoid f(s32 v) {\n    D_80100010[D_80100A00++] = v;\n}\n",
    "do_while_and": "s32 f(s32 a, s32 b, s32 n) {\n    do {\n        a += b;\n    } while (a >= 0 && a < n);\n"
                    "    return -1;\n}\n",
    "unsigned_byte_constant": "void f(s32 i) {\n    *(u8 *) (i + (s32) &D_80100000) = 255;\n}\n",
    "narrow_comparison_order": "extern u8 D_80100900[][21];\ns32 f(s32 i, s32 j, s32 v) {\n    u8 c;\n\n    c = v;\n"
                               "    if (c == D_80100900[i][j]) {\n        return 1;\n    }\n    return 0;\n}\n",
    "modulo_minus_one": "s32 f(struct S *p) {\n    return (func_80002000(0, 0) % -1) * p->a;\n}\n",
    "stack_argument_narrow_read": "void f(struct S *p, s32 a, s32 b, s32 c, s32 d) {\n    p->b = c;\n    p->b = d;\n}\n",
    "unsigned_divide_constant": "u32 f(u32 x) {\n    return x / 3 + x / 7 + x / 14 + x % 3;\n}\n",
    "structure_return_forwarded": "struct R { s32 w[3]; };\nstruct R func_80002400(s32);\nstruct R f(s32 a) {\n    struct R r;\n\n"
                                  "    if (a) {\n        r.w[0] = 0;\n        return r;\n    }\n    return func_80002400(a);\n}\n",
    "times_negative_constant": "s32 f(s32 x, s32 y) {\n    s32 r;\n\n    r = x * -0x29CF;\n    return y + r * -0x3B21;\n}\n",
    "factor_moved": "s32 f(s32 a, s32 b, s32 i, s32 *p) {\n    s32 r;\n\n    r = (a * 4) * b;\n    *p = a * (b * 3);\n"
                    "    r += (D_80100010[i] * 4) * D_80100000;\n    r += D_80100000 * (D_80100010[i] * 4);\n"
                    "    return func_80002000((a * 4) * b, r) * (b * 3);\n}\n",
    "factor_moved_narrow": "void f(struct S *p, s32 i) {\n    p->b = (D_80100010[i] * 4) * D_80100000 + D_80100200;\n"
                           "    p->b = D_80100000 * (D_80100010[i] * 4) + D_80100200;\n}\n",
    "structure_array_increment": "struct E { s32 a; s16 b; u8 c; u8 d; s32 e; };\nextern struct E D_80100500[];\n"
                                 "void f(s32 i) {\n    D_80100500[i].e++;\n}\n",
    # addresses and arrays
    "nested_member_constant": "struct E { s32 x; struct { u8 b[0xB000]; } a; };\nextern struct E D_80100900[];\n"
                              "s32 f(s32 i) {\n    return (s32) &D_80100900[i].a + 0xAC00;\n}\n",
    "nested_member_element": "struct E2 { s32 x; struct { u8 pad[0xAC00]; struct { u8 b[0x2000]; } m[4]; } a; };\n"
                             "extern struct E2 D_80120000[];\n"
                             "s32 f(s32 i, s32 j) {\n    return (s32) &D_80120000[i].a.m[j];\n}\n",
    "constant_between_products": "void f(s32 a, s32 b, s32 c, s16 *p) {\n    p[a * 3584 + 0x800 + b * 48 + c] = 1;\n}\n",
    "post_increment_times": "s32 f(s32 *p, s32 i) {\n    s32 x;\n\n    x = p[i++];\n    return x + i;\n}\n",
    "address_first_sum": "struct B40 { u8 b[40]; };\nextern struct B40 D_80100A00[][2];\n"
                         "s32 f(s32 i, s32 j) {\n    return (s32) &D_80100A00[i][j];\n}\n",
    "constant_across_sum": "s32 f(s32 x, s32 y) {\n    s32 r;\n\n    r = x - (65 - y);\n    x = x + 1 + y;\n"
                           "    return r + x;\n}\n",
    "constant_across_difference": "s32 f(s32 a, s32 b) {\n    s32 x;\n    s32 y;\n\n    x = (a + 7) - b;\n"
                                  "    y = a - (b - 7);\n    return x + y;\n}\n",
    "byte_rows": "extern u8 D_80100B00[][21];\nvoid f(s32 i, s32 j, u8 c) {\n    D_80100B00[i][j] = c;\n}\n",
    "member_byte_rows": "struct H { u8 pad[0x20]; u8 m[4][8]; };\n"
                        "void f(struct H *p, s32 i, s32 j, u8 x) {\n    p->m[i][j] = x;\n}\n",
    "call_in_operand": "s32 f(s32 i, s32 j) {\n    return D_80100010[i] + func_80002000(i, 0) * j;\n}\n",
    "structure_grid": "struct E3 { s32 a; s32 b; u8 pad[32]; };\nextern struct E3 D_80100C00[][7];\n"
                      "void f(s32 i, s32 j) {\n    D_80100C00[i][j].b = 1;\n}\n",
    "stepped_pointer_store": "typedef struct { s32 w0; s32 w1; } G;\nextern G *D_80100400;\nvoid f(s32 a) {\n    G *g;\n\n"
                             "    D_80100400 = (G *) (a + 0x4D18);\n    g = D_80100400++;\n    g->w0 = 1;\n}\n",
    "pointer_step_statement": "void f(s32 *a, s32 n) {\n    s32 *p;\n\n    p = a;\n    p++;\n    p += n;\n    *p = 0;\n}\n",
    "pointer_difference_far": "typedef struct { s32 w0; s32 w1; } G;\ns32 f(G *a, s32 b) {\n    s32 n;\n\n"
                          "    n = a - (G *) (b + 0x4CC0);\n    return n + (a - (G *) b) * sizeof(G);\n}\n",
    "far_displacement": "struct F { u8 pad[0x1E478]; s32 v; };\ns32 f(struct F *p) {\n    return p->v;\n}\n",
    "unaligned_copy": "struct T { s16 w[3]; };\nextern struct T D_80100D00[];\nextern struct T D_80100E00;\n"
                      "void f(s32 i) {\n    D_80100E00 = D_80100D00[i];\n}\n",
    "element_decrement": "struct S4 { s32 m0; f32 m4; f32 m8; s32 mC; };\nextern struct S4 D_80100F00[];\n"
                         "extern s32 D_80101000[][16];\n"
                         "void f(s32 i, s32 j) {\n    D_80100F00[i].mC--;\n    D_80101000[i][j]++;\n}\n",
    "member_both_signs": "struct P6 { u16 b; s8 x; s8 y; u8 e; };\nextern struct P6 D_80101100[];\n"
                         "s32 f(s32 i) {\n    if (D_80101100[i].x > 40) {\n        return (u8) D_80101100[i].x;\n    }\n"
                         "    return 0;\n}\n",
}
VRAM = 0x80001000


def _compiler(pipeline: str, flags: list[str], key: str):
    try:
        from n64dk.elf import Elf
        from n64dk.match import compare as cmp
        from n64dk.match.pipeline import build, split_flags
        from n64dk.match.symbols import SymbolMap
        from n64dk.match.toolchain import load_config
    except Exception as error:                                # pragma: no cover
        return None, str(error)
    project = Path(tempfile.mkdtemp(prefix="n64dk-debug-test-"))
    atexit.register(shutil.rmtree, project, True)
    cfg = load_config(project, None, None)
    if cfg.toolchain(key) is None:
        return None, f"{key} toolchain not configured (N64DK_TOOLCHAIN)"
    spec, counter = split_flags(flags, pipeline), [0]

    def compile_text(text: str) -> bytes:
        counter[0] += 1
        folder = project / f"c{counter[0]}"
        folder.mkdir()
        (folder / "f.c").write_text(text)
        elf = Elf.load(build(folder / "f.c", spec, cfg, folder))
        # placed at VRAM as a linker would; external names carry their own addresses
        code = cmp.relocate(elf, ".text", {".text": VRAM}, SymbolMap())
        last = max(at for at in range(0, len(code), 4) if code[at:at + 4] == b"\x03\xe0\x00\x08")
        return code[:last + 8]         # up to the last return and its delay slot (a debug build's unreachable ones included): no padding
    return compile_text, None


class RoundTrip:
    module = None
    pipeline, flags, key = "", [], ""
    expected = ()

    @classmethod
    def setUpClass(cls):
        found, reason = _compiler(cls.pipeline, cls.flags, cls.key)
        if found is None:
            raise unittest.SkipTest(reason)
        cls.compile = staticmethod(found)

    def round_trip(self, program: str, programs=None) -> bool:
        target = self.compile(TYPES + EXTERNALS + (programs or PROGRAMS)[program])
        found = self.module.variants(TYPES, "f", code=target, vram=VRAM, pipeline=self.pipeline,
                                     flags=tuple(self.flags), family=None)
        for _label, text in found:
            try:
                if self.compile(text) == target:
                    return True
            except Exception:
                continue
        return False

    def test_small_functions_come_back_byte_identical(self):
        results = {program: self.round_trip(program) for program in PROGRAMS}
        self.assertTrue({p for p, ok in results.items() if ok} >= set(self.expected), results)

    def test_optimised_code_is_not_this_generators_business(self):
        optimised, reason = _compiler(self.pipeline, ["-O2" if f in ("-O0", "-g") else f for f in self.flags], self.key)
        code = optimised(TYPES + EXTERNALS + PROGRAMS["arith"])
        self.assertEqual(self.module.variants(TYPES, "f", code=code, vram=VRAM, pipeline=self.pipeline,
                                              flags=("-O2",), family=None), [])


class GccDebugTest(RoundTrip, unittest.TestCase):
    module = redraft_gcc_debug
    pipeline, flags, key = "kmc+as", ["-O0", "-G0", "-mips3", "-mgp32", "-mfp32"], "kmc"
    expected = tuple(name for name in PROGRAMS if name not in ("logicval", "ternary"))   # known gaps: || and ?: as values


class LiteralPoolTest(unittest.TestCase):
    """Constants back to back with one the compiler made are literals (the round trip above cannot show it:
    an unlinked object has no read-only data to read)."""

    def pool(self, own, accesses):
        function = object.__new__(redraft_gcc_debug.Function)
        function.literal_addrs = set(own)
        function.globals = {address: {"acc": list(acc), "addr": taken} for address, (acc, taken) in accesses.items()}
        return function._literal_pool()

    def test_doubles_that_touch_the_conversion_constant_are_literals(self):
        doubles = {0x80088588: ([(8, "d")], False), 0x80088590: ([(8, "d"), (8, "d")], False),
                   0x800885A0: ([(8, "d")], False)}
        self.assertEqual(self.pool({0x80088598}, doubles), {0x80088588, 0x80088590, 0x800885A0})

    def test_a_gap_an_integer_access_or_a_taken_address_ends_the_pool(self):
        self.assertEqual(self.pool({0x80088598}, {0x800885A8: ([(8, "f")], False)}), set())          # not adjacent
        self.assertEqual(self.pool({0x80088598}, {0x800885A0: ([(8, "f"), (4, "s")], False)}), set())  # read as a word too
        self.assertEqual(self.pool({0x80088598}, {0x800885A0: ([(8, "f")], True)}), set())           # address taken
        self.assertEqual(self.pool(set(), {0x800885A0: ([(8, "f")], False)}), set())                 # nothing of the compiler's


class GccFormsTest(RoundTrip, unittest.TestCase):
    """The forms above on the lane of the title they were established on (KMC GCC with the SN assembler: the
    two assemblers expand ``lh sym+off($reg)`` differently, and the translator reads the SN expansion)."""
    module = redraft_gcc_debug
    pipeline, flags, key = "kmc+asn64", ["-O0", "-G0", "-mips3", "-mgp32", "-mfp32"], "kmc"
    expected = ()
    FORM_GAPS = ()                         # forms of GCC_FORMS not yet reproduced (none is allowed to be silent)

    def test_small_functions_come_back_byte_identical(self):
        self.skipTest("covered on the kmc+as lane")

    def test_optimised_code_is_not_this_generators_business(self):
        self.skipTest("covered on the kmc+as lane")

    def test_forms_established_on_a_title_come_back_byte_identical(self):
        results = {form: self.round_trip(form, GCC_FORMS) for form in GCC_FORMS}
        failed = sorted(form for form, ok in results.items() if not ok)
        self.assertEqual(failed, sorted(self.FORM_GAPS), results)


class IdoDebugTest(RoundTrip, unittest.TestCase):
    module = redraft_ido_debug
    pipeline, flags, key = "ido5.3", ["-g", "-mips2", "-non_shared", "-G", "0", "-Xcpluscomm", "-Wab,-r4300_mul"], "ido53"
    expected = tuple(name for name in PROGRAMS if name not in ("index",))         # known gap: operand order of two array loads


class RegistrationTest(unittest.TestCase):
    def test_both_translators_are_registered_and_decline_what_they_do_not_know(self):
        self.assertIn("redraft_gcc_debug", redraft.GENERATORS)
        self.assertIn("redraft_ido_debug", redraft.GENERATORS)
        junk = bytes(range(64))
        for module in (redraft_gcc_debug, redraft_ido_debug):
            self.assertEqual(module.variants("", "f", code=junk, vram=VRAM, pipeline="x", flags=()), [])
            self.assertEqual(module.variants("", "f", code=b"", vram=VRAM, pipeline="x", flags=()), [])


if __name__ == "__main__":
    unittest.main()
