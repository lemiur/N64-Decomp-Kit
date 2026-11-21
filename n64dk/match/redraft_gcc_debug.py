"""GCC 2.x without optimisation (-O0): write the function from its machine code.

Unoptimised GCC code is a literal transcript of the source: a frame pointer, every argument stored
to the frame on entry, every local in a stack slot laid out in declaration order, every statement
compiled on its own with its values in scratch registers, every jump kept as written.  So instead
of repairing a general decompiler's rendering, this generator reads the function back statement by
statement:

* the prologue gives the frame size, the saved registers and the parameters (how each is homed
  says how wide it is);
* inside one statement the registers hold expression trees; a store, a call whose value is not
  used, a branch and a jump each end a statement;
* the widths and signs of the loads and stores give the type of every local, parameter and global;
* locals are declared in ascending frame order with fillers, which is how GCC lays the frame out,
  so the frame of the result is the frame of the target;
* the jump skeleton is mapped back to ``if``/``else``, loops and ``goto`` (GCC -O0 keeps one jump
  per source construct, so a faithful reading compiles back to the same skeleton).

Everything is computed from the bytes.  The decompiler draft is used for one thing only: the
typedef prelude the kit compiles every candidate with.  The situation is recognised from the code
(``addu $fp,$sp,$zero`` in the prologue), not from the recipe, so an unoptimised function inside
an optimised title is translated too.  Anything not understood makes the generator decline.
"""
from __future__ import annotations

import re

from .gccdebug_mips import (A0, AT, BRANCHES, FP, LOADS, RA, REG, SP, STORES, V0, V1, decode)

MAX_INSNS = 16000                                    # the largest function met so far has 4,310 (Wheel of Fortune)


def _signed_magic(divisor: int) -> tuple[int, int, bool]:
    """(multiplier as 32 bits, shift, add the dividend) GCC uses for a signed 32-bit division by a constant
    (expmed.c ``choose_multiplier`` with precision 31: the high half of the product, shifted, less the sign)."""
    lgup = (divisor - 1).bit_length()
    low = (1 << (32 + lgup)) // divisor
    high = ((1 << (32 + lgup)) + (1 << (32 + lgup - 31))) // divisor
    shift = lgup
    while shift > 0 and low // 2 < high // 2:
        low, high, shift = low // 2, high // 2, shift - 1
    return high & 0xFFFFFFFF, shift, high >= 1 << 31


SIGNED_MAGIC: dict[tuple[int, int, bool], int] = {}
for _divisor in range(3, 1 << 13):
    if _divisor & (_divisor - 1):
        SIGNED_MAGIC.setdefault(_signed_magic(_divisor), _divisor)


def _unsigned_magic(divisor: int) -> tuple[int, int, int, bool]:
    """(multiplier as 32 bits, shift before, shift after, add the dividend) GCC uses for an unsigned 32-bit
    division by a constant (expmed.c ``choose_multiplier`` with precision 32; an even divisor whose multiplier
    needs 33 bits is shifted down first; an odd one takes the form ``t + ((x - t) >> 1)``)."""
    def choose(value: int, precision: int) -> tuple[int, int]:
        lgup = (value - 1).bit_length()
        low = (1 << (32 + lgup)) // value
        high = ((1 << (32 + lgup)) + (1 << (32 + lgup - precision))) // value
        shift = lgup
        while shift > 0 and low // 2 < high // 2:
            low, high, shift = low // 2, high // 2, shift - 1
        return high, shift

    multiplier, after = choose(divisor, 32)
    before = 0
    if multiplier >= 1 << 32 and divisor % 2 == 0:
        before = (divisor & -divisor).bit_length() - 1
        multiplier, after = choose(divisor >> before, 32 - before)
    if multiplier >= 1 << 32:
        return multiplier & 0xFFFFFFFF, 0, after - 1, True
    return multiplier, before, after, False


UNSIGNED_MAGIC: dict[tuple[int, int, int, bool], int] = {}
for _divisor in range(3, 1 << 13):
    if _divisor & (_divisor - 1):
        UNSIGNED_MAGIC.setdefault(_unsigned_magic(_divisor), _divisor)


def single_struct_blocked(proto) -> bool:
    """A structure argument among float arguments is not modelled."""
    return "f32" in proto or "f64" in proto


class Decline(Exception):
    """This function is not (yet) something the translator reads; the draft stands."""


class Promote(Exception):
    """A scratch register written at ``addr`` holds a ``register`` variable that lives inside one block:
    the function is read again with that write as an assignment."""

    def __init__(self, addr: int, reg: int):
        super().__init__(f"register variable in ${REG[reg]}")
        self.key = (addr, reg)


# ----------------------------------------------------------------------------------------------
# expression trees


class Node:
    """A value in a register.  ``uses`` counts readers, ``seq`` orders creation."""
    __slots__ = ("k", "a", "uses", "stmt", "seq")

    def __init__(self, k, *a):
        self.k, self.a, self.uses, self.stmt, self.seq = k, a, 0, None, 0

    def __repr__(self):
        return f"{self.k}{self.a!r}"


def strip(node: Node) -> Node:
    while node.k == "cp":
        node = node.a[0]
    return node


def plain_variable(node) -> bool:
    """A full-width local or global read directly: no code until the instruction that uses it."""
    node = strip(node)
    return node.k == "ld" and node.a[0] == 4 and node.a[2][0] in ("local", "global", "regvar")


def work_seq(node):
    """``first_seq`` without the calls.  GCC expands the calls of an expression before anything else in it
    (``preexpand_calls``), so where a call stands says nothing about the order of the operands; what else an
    operand does still shows it.  None for an operand that is nothing but a call."""
    best = None
    stack = [node]
    while stack:
        part = stack.pop()
        if isinstance(part, Node):
            if part.k == "call":
                continue
            if part.seq and (best is None or part.seq < best):
                best = part.seq
            stack.extend(part.a)
        elif isinstance(part, (list, tuple)):
            stack.extend(part)
    return best


def has_call(node) -> bool:
    """Whether a call is among the operands of ``node``.  GCC expands the calls of an expression before
    anything else in it (``preexpand_calls``), so creation order says nothing about source order then."""
    stack = [node]
    while stack:
        part = stack.pop()
        if isinstance(part, Node):
            if part.k == "call":
                return True
            stack.extend(part.a)
        elif isinstance(part, (tuple, list)):
            stack.extend(part)
    return False


def work_before(b, a) -> bool:
    """Whether operand ``b`` was begun before operand ``a``, calls aside (measured: ``X[i] + f() * j`` makes the
    call, forms the address of ``X[i]``, multiplies, loads, and adds as ``product + X[i]``; an operand that is
    only a call keeps the instruction's order)."""
    first, second = work_seq(b), work_seq(a)
    return first is not None and second is not None and first < second


def plain_float(node) -> bool:
    """A ``float`` or ``double`` local or global read directly."""
    node = strip(node)
    return node.k == "ld" and node.a[1] == "f" and node.a[2][0] in ("local", "global")


def first_seq(node) -> int:
    """When the first instruction that contributed to ``node`` ran (creation order of its leaves)."""
    best = node.seq if isinstance(node, Node) and node.seq else 1 << 30
    stack = list(node.a) if isinstance(node, Node) else []
    while stack:
        part = stack.pop()
        if isinstance(part, Node):
            if part.seq and part.seq < best:
                best = part.seq
            stack.extend(part.a)
        elif isinstance(part, (list, tuple)):
            stack.extend(part)
    return best


def float_text(bits: int) -> str:
    """The shortest C literal that is exactly this single-precision value."""
    import struct
    value = struct.unpack(">f", struct.pack(">I", bits & 0xFFFFFFFF))[0]
    if value != value or value in (float("inf"), float("-inf")):
        raise Decline("float constant that has no literal")
    for digits in range(1, 18):
        text = f"{value:.{digits}g}"
        if struct.pack(">f", float(text)) == struct.pack(">f", value):
            break
    if "e" not in text and "." not in text:
        text += ".0"
    return text + "f"


def double_text(bits: int) -> str:
    """The shortest C literal that is exactly this double-precision value."""
    import struct
    value = struct.unpack(">d", struct.pack(">Q", bits & 0xFFFFFFFFFFFFFFFF))[0]
    if value != value or value in (float("inf"), float("-inf")):
        raise Decline("float constant that has no literal")
    text = repr(value)
    if "e" not in text and "." not in text:
        text += ".0"
    return text


def const_of(node: Node):
    node = strip(node)
    if node.k == "const":
        return node.a[0]
    if node.k == "hi":
        return _s32(node.a[0] << 16)
    return None


def _s32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & 0x80000000 else value


INT_TYPES = {(1, True): "s8", (1, False): "u8", (2, True): "s16", (2, False): "u16", (4, True): "s32",
             (4, False): "u32"}
NEGATE = {"==": "!=", "!=": "==", "<": ">=", ">=": "<", "<=": ">", ">": "<="}
FLIP = {"==": "==", "!=": "!=", "<": ">", ">": "<", "<=": ">=", ">=": "<="}


class Stmt:
    __slots__ = ("k", "a", "end")

    def __init__(self, k, *a):
        self.k, self.a, self.end = k, a, None                # ``end``: index of the first instruction after it

    def __repr__(self):
        return f"S:{self.k}{self.a!r}"


# ----------------------------------------------------------------------------------------------
# the function


SCRATCH = frozenset((2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25))
# a branch on the selector's sign in a ``switch`` tree: (its values where taken, its values where not)
SIGN_TESTS = {"bgtz": ((1, None), (None, 0)), "blez": ((None, 0), (1, None)),
              "bltz": ((None, -1), (0, None)), "bgez": ((0, None), (None, -1))}
SCRATCH_KEY = 1000                                   # ``("regvar", SCRATCH_KEY + n)``: the n-th scratch variable
# A value some instruction computed (not a copy, a load or a call's result).  GCC copies such a value before
# it stores it twice (``a = b = x + 1``: ``addu $2,$3,1; move $3,$2; sw; sw``), so one that is read by two
# statements with no copy is a variable in a register.
COMPUTED = frozenset(("bin", "un", "mul", "cast"))
_THREE = ("addu", "subu", "and", "or", "xor", "nor", "slt", "sltu", "sllv", "srlv", "srav")
_IMMEDIATE = ("addiu", "ori", "andi", "xori", "slti", "sltiu")


def reads_writes(x) -> tuple[tuple, tuple]:
    """(integer registers read, integer registers written) by one instruction."""
    op = x.op
    if op in LOADS:
        return (x.rs,), (x.rt,)
    if op in STORES:
        return (x.rs, x.rt), ()
    if op == "lui":
        return (), (x.rt,)
    if op in _IMMEDIATE:
        return (x.rs,), (x.rt,)
    if op in _THREE:
        return (x.rs, x.rt), (x.rd,)
    if op in ("sll", "srl", "sra"):
        return (x.rt,), (x.rd,)
    if op in ("mult", "multu", "div", "divu") and x.fmt is None:
        return (x.rs, x.rt), ()
    if op in ("mflo", "mfhi"):
        return (), (x.rd,)
    if op in ("beq", "bne"):
        return (x.rs, x.rt), ()
    if op in ("blez", "bgtz", "bltz", "bgez"):
        return (x.rs,), ()
    if op == "jal":
        return (), tuple(SCRATCH)
    if op == "jalr":
        return (x.rs,), tuple(SCRATCH)
    if op == "mtc1":
        return (x.rt,), ()
    if op == "mfc1":
        return (), (x.rt,)
    if op in ("lwc1", "ldc1", "swc1", "sdc1"):
        return (x.rs,), ()
    return (), ()


def float_reads_writes(x) -> tuple[tuple, tuple]:
    """(float registers read, float registers written) by one instruction (a double names its even register)."""
    op = x.op
    if op in ("lwc1", "ldc1"):
        return (), (x.ft,)
    if op in ("swc1", "sdc1"):
        return (x.ft,), ()
    if op == "mtc1":
        return (), (x.fs,)
    if op == "mfc1":
        return (x.fs,), ()
    if x.fmt is None:
        return (), ()
    if op in ("add", "sub", "mul", "div"):
        return (x.fs, x.ft), (x.fd,)
    if op in ("c.eq", "c.lt", "c.le"):
        return (x.fs, x.ft), ()
    return (x.fs,), (x.fd,)


class Function:
    def __init__(self, code: bytes, vram: int, name: str, read=None, scratch: bool = False, forced=frozenset()):
        self.name, self.vram = name, vram
        self.scratch = scratch                               # read scratch registers that outlive a block as variables
        self.forced = forced                                 # (address, register): writes known to be assignments
        self.instances: list[dict] = []                      # the scratch variables: register, extent, name
        self.instance_at: dict[tuple[int, int], int] = {}    # (instruction address, register) -> its variable
        self.list_start: dict[int, int] = {}                 # id of a statement list -> its first instruction
        self.reg_def: dict[int, int] = {}                    # register -> address of the instruction that wrote it
        self.var_read: set[tuple[int, int]] = set()          # (instruction address, register) read as a variable
        self.var_def: set[tuple[int, int]] = set()           # (instruction address, register) assigned as one
        self.scratch_names: dict[int, str] = {}
        self.now = None                                      # the instruction being executed
        self.read = read                                     # read(address) -> word | None: the image's data
        self.tables: dict[int, dict] = {}                    # index of a table dispatch -> its cases
        insns = decode(code, vram)
        if insns is None or len(insns) < 6 or len(insns) > MAX_INSNS:
            raise Decline("not decodable")
        while len(insns) > 2 and insns[-1].op == "nop":
            insns.pop()                                      # alignment after the return
        self.all = insns
        self._frame()
        self._double_pairs()
        self._labels()
        self._unsigned_conversions()
        if scratch:
            self._scratch_variables()
        self.vars: dict[int, dict] = {}                      # frame offset -> local variable
        self.globals: dict[int, dict] = {}                   # address -> how it is used
        self.fields: set[tuple[int, str]] = set()            # (offset, type) accessor structs
        self.indexed: dict[int, dict] = {}                   # address of ``sym+off($index)`` -> scale, accesses
        self.value_and: set[int] = set()                     # ids of ``x & (2^k - 1)`` computed as a value
        self.inplace_products: set[int] = set()              # ids of ``x * C`` whose last step was done in place
        self.wide_calls: set[int] = set()                    # ids of calls whose value is 64 bits ($v0:$v1)
        self.divisions: set[int] = set()                     # ids of calls that are a 64-bit division (libgcc)
        self.remainders: set[int] = set()                    # ... and a 64-bit remainder
        self.saved_temp: dict[int, Node] = {}                # call-saved register -> a value kept across a call
        self.saved_temps_seen: set[int] = set()              # call-saved registers that held such a value
        self.keep: list = []                                 # keeps those nodes alive (ids stay unique)
        self.direct_products: set[int] = set()               # ids of products that are the whole value of a word store
        self.unsigned_masks: set[int] = set()                # ids of ``x & mask`` done as unsigned in a narrow store
        self.narrow_div: set[int] = set()                    # ids of ``u16 >> k`` made in a lower register than its operand
        self.member_arrays: set[tuple[int, int]] = set()     # (offset, element size) of ``&p->m[i]``
        self.pointer_fields: set[tuple[int, int]] = set()    # (offset, step) of ``p->q++`` on a pointer member
        self.scalar_arrays: set[tuple[int, str]] = set()     # (offset, type) of ``p->arr[i]``
        self.grids: dict[int, tuple[str, int]] = {}          # address -> (type, columns) of ``SYM[i][j]``
        self.forwarded: set[int] = set()                     # ids of calls given this function's result pointer
        self.loop_tests: set[int] = set()                    # ids of ``if (!(a && b)) break`` that are a loop's own test
        self.weak_grids: set[int] = set()                    # those whose element type only a store has named
        self.struct_arrays: set[tuple] = set()               # (offset, size, member offset, type) of ``p->m[i].v``
        self.void_fields: set[int] = set()                   # offsets of members assigned an address as a pointer
        self.element_sizes: set[int] = set()                 # element structures named in pointer arithmetic
        self.nested: dict[int, tuple[int, int, int]] = {}    # symbol address -> (base, element size, lead) of ``&S[i].a``
        self.member_grids: set[tuple[int, int, str]] = set() # (offset, row size, type) of ``p->m[i][j]``
        self.inserted: dict[int, int] = {}                   # id of a sum of two products -> the constant between them
        self.pointer_locals: set[int] = set()                # frame offsets of locals declared ``void *``
        self.callees: dict[object, dict] = {}
        self.returns_value = False
        self.return_type = "s32"
        self.label_used: set[int] = set()
        self.anchors: dict[int, int] = {}                    # address of an array of structures -> element size
        self.hard_labels: set[int] = set()                   # labels of the source that the code proves (a no-op before them)
        self.not_loops: set[int] = set()                     # targets of backward jumps that are not the top of a loop
        self.float_calls: set[int] = set()
        self.double_calls: set[int] = set()                  # calls whose value is read as a ``double``
        self.li_zero: set[int] = set()
        self.regvar_reads: dict[int, int] = {}               # saved register -> times read as a variable
        self.regvar_writes: dict[int, int] = {}
        self.unexpanded: set[int] = set()                    # calls the compiler did not expand ahead of their statement
        self.inside_unexpanded = False
        self.flex_args: list[tuple[int, int, Node]] = []     # (callee, position, argument) still of either kind
        self.callee_protos: dict[int, list[str]] = {}        # callee address -> the parameter types its calls show
        self.prototypes: dict[int, list[str]] = {}
        self.seq = 0
        self.barrier = 0                                     # sequence number at the last assignment
        self.int_fp = self._integer_fp_accesses()
        # GCC spills the high half of a product it is about to shift into a stack temporary of its own
        # (``mfhi; sw t($fp); lw t($fp)``): not a variable, and not part of the declared locals
        self.temp_at: dict[int, int] = {}                    # address of such a store or load -> frame offset
        for i in range(len(self.ins) - 2):
            high, put = self.ins[i:i + 2]
            if high.op == "mfhi" and put.op == "sw" and put.rs == FP and put.rt == high.rd:
                for get in self.ins[i + 2:i + 6]:
                    if get.op == "lw" and get.rs == FP and get.imm == put.imm:
                        self.temp_at[put.addr] = self.temp_at[get.addr] = put.imm
                        break
                    if get.op in BRANCHES or get.op in ("j", "jal", "jalr") or get.rs == FP:
                        break
        self.temp_value: dict[int, Node] = {}
        self.chains: dict[int, tuple[int, list]] = {}        # index -> (index after, [(pointer local, offset)])
        self._spilled_chains()
        self.block_sizes: set[int] = set()
        self.loose_blocks: set[tuple[str, int]] = set()      # ("u" | "b", bytes) of structures without word alignment
        self.struct_locals: dict[int, int] = {}              # frame offset -> size of a structure copied whole
        for i in range(len(self.ins)):
            run = self._copy_run(i, len(self.ins))
            if run is None:
                continue
            for back in self.ins[max(0, i - 2):i]:
                if back.op == "addiu" and back.rs == FP and back.rt in (run[1], run[2]):
                    if self.struct_locals.setdefault(back.imm, 4 * run[0]) != 4 * run[0]:
                        raise Decline("one local copied as two structure sizes")
        self._collect_frame_accesses()
        self.body = self._block(0, len(self.ins), None)

    # -- prologue and epilogue ----------------------------------------------------------------

    def _frame(self):
        ins = self.all
        # A function declared with ``...`` stores its four argument registers in the caller's argument area
        # before it opens its frame (probe: ``sw $a0,0($sp)`` … ``sw $a3,12($sp)``, then the usual prologue).
        self.variadic = len(ins) > 5 and all(
            x.op == "sw" and x.rs == SP and x.rt == 4 + k and x.imm == 4 * k for k, x in enumerate(ins[:4]))
        if self.variadic:
            ins = self.all = ins[4:]
        first = ins[0]
        if not (first.op == "addiu" and first.rs == SP and first.rt == SP and first.imm < 0):
            raise Decline("no frame")
        self.size = -first.imm
        index, saved = 1, {}
        fsaved = {}
        while index < len(ins) and ins[index].rs == SP and (
                (ins[index].op == "sw" and (ins[index].rt in (RA, FP) or 16 <= ins[index].rt <= 23))
                or (ins[index].op == "sdc1" and 20 <= ins[index].ft <= 30)):
            if ins[index].op == "sdc1":                       # a float kept across a call
                fsaved[ins[index].ft] = ins[index].imm
            else:
                saved[ins[index].rt] = ins[index].imm
            index += 1
        if index >= len(ins) or not (ins[index].op == "addu" and ins[index].rd == FP and ins[index].rs == SP
                                     and ins[index].rt == 0) or FP not in saved:
            raise Decline("no frame pointer")
        index += 1
        self.saved = saved
        # GCC -O0 uses a call-saved register for one thing only: a variable declared ``register`` (or a
        # value it must keep across a call, which reads back as one).  Each is a variable without a slot.
        self.regvars: dict[int, dict] = {reg: {"reg": reg, "name": f"reg_{REG[reg]}"}
                                         for reg in sorted(saved) if 16 <= reg <= 23}
        # epilogue, from the end
        end = len(ins) - 1
        self.fsaved = fsaved
        # A function that ends in ``for (;;) ;`` never reaches its epilogue.  The compiler writes one all the
        # same, but a catalogue that ends functions at the last transfer of control lists it apart (as a
        # 24-byte function of its own): the code given then stops at the jump to itself.
        endless = ins[end].op == "j" and ins[end].target == ins[end].addr
        if endless:
            end = len(ins)
        else:
            if not (ins[end].op == "jr" and ins[end].rs == RA):
                raise Decline("no return")
            end -= 1
            if not (ins[end].op == "addiu" and ins[end].rs == SP and ins[end].rt == SP and ins[end].imm == self.size):
                raise Decline("epilogue")
            end -= 1
            restored, frestored = {}, {}
            while end > index and ins[end].op in ("lw", "ldc1") and ins[end].rs == SP:
                (restored if ins[end].op == "lw" else frestored)[ins[end].rt] = ins[end].imm
                end -= 1
            if frestored != fsaved or restored != saved or not (ins[end].op == "addu" and ins[end].rd == SP
                                                                and ins[end].rs == FP and ins[end].rt == 0):
                raise Decline("epilogue")
        epilogue = end
        # A function that returns a structure keeps the hidden result pointer ($a0) in a saved register
        # and hands it back in $v0 at the common exit, with no jump after it (``return p;`` has one).
        self.sret = None
        back = ins[end - 1]
        if not endless and index < end - 1 and back.op == "addu" and back.rd == V0 and back.rt == 0 and back.rs in self.regvars \
                and ins[index].op == "addu" and ins[index].rd == back.rs and ins[index].rs == A0 and ins[index].rt == 0:
            self.sret = back.rs
            del self.regvars[back.rs]
            epilogue = end - 1
            index += 1
        # parameters: how each argument register is put away
        self.params: list[dict] = [{"kind": "hidden"}] if self.sret is not None else []
        pending: dict[int, int] = {}
        while index < epilogue:
            x = ins[index]
            number = len(self.params)
            if x.op == "sw" and x.rs == FP and 4 <= x.rt <= 7 and x.rt - 4 == number and \
                    x.imm == self.size + 4 * number:
                self.params.append({"kind": "int", "off": x.imm, "size": 4})
            elif x.op == "addu" and x.rt == 0 and 4 <= x.rs <= 7 and x.rs - 4 == number and x.rd in self.regvars \
                    and "param" not in self.regvars[x.rd]:
                # ``register`` parameter: moved from its argument register into the saved register it lives in
                self.regvars[x.rd].update(param=number, name=f"arg{number}")
                self.params.append({"kind": "register", "reg": x.rd})
            elif x.op == "addu" and x.rt == 0 and 4 <= x.rs <= 7 and x.rs - 4 == number and x.rd not in pending \
                    and x.rd not in (SP, FP, RA) and x.rd != x.rs:
                pending[x.rd] = number
                self.params.append({"kind": "moved", "reg": x.rd})
            elif x.op == "lw" and x.rs == FP and number >= 4 and x.imm == self.size + 4 * number \
                    and x.rt not in pending and x.rt not in (SP, FP, RA) and self._narrow_store_follows(ins, index, epilogue):
                # a narrow parameter passed on the stack: loaded, then stored narrow like the others
                pending[x.rt] = number
                self.params.append({"kind": "moved", "reg": x.rt})
            elif x.op == "sw" and x.rs == FP and x.rt == 6 and number == 1 and x.imm == self.size + 8 \
                    and index + 1 < epilogue and ins[index + 1].op == "sw" and ins[index + 1].rs == FP \
                    and ins[index + 1].rt == 7 and ins[index + 1].imm == x.imm + 4:
                # ``(int, double)``: the double is aligned to the second pair of argument registers
                self.params.append({"kind": "pad"})
                self.params.append({"kind": "double", "off": x.imm, "size": 8, "type": "f64"})
                self.params.append({"kind": "pad"})
                index += 1
            elif x.op == "swc1" and x.rs == FP and x.ft in (13, 15) and index + 1 < epilogue \
                    and ins[index + 1].op == "swc1" and ins[index + 1].rs == FP and ins[index + 1].ft == x.ft - 1 \
                    and ins[index + 1].imm == x.imm + 4 and x.imm >= self.size + 4 * number \
                    and (x.imm - self.size) % 8 == 0 and x.imm - self.size <= 4 * number + 4 \
                    and all(p["kind"] in ("float", "double", "pad") for p in self.params) \
                    and x.ft == 13 + 2 * sum(p["kind"] in ("float", "double") for p in self.params):
                # a leading ``double`` parameter arrives in $f12/$f13 or $f14/$f15 (the odd register is the
                # high word) and takes two aligned argument words
                while self.size + 4 * len(self.params) < x.imm:
                    self.params.append({"kind": "pad"})
                self.params.append({"kind": "double", "off": x.imm, "size": 8, "type": "f64"})
                self.params.append({"kind": "pad"})
                index += 1
            elif x.op == "swc1" and x.rs == FP and x.imm == self.size + 4 * number \
                    and all(p["kind"] in ("float", "double", "pad") for p in self.params) \
                    and x.ft == 12 + 2 * sum(p["kind"] in ("float", "double") for p in self.params) and x.ft <= 14:
                # leading ``float`` parameters arrive in $f12/$f14
                self.params.append({"kind": "float", "off": x.imm, "size": 4, "type": "f32"})
            elif x.op in ("sh", "sb") and x.rs == FP and x.rt in pending:
                param = self.params[pending.pop(x.rt)]
                param.update(kind="narrow", off=x.imm, size=STORES[x.op], signed=None)
            else:
                break
            index += 1
        if pending:
            raise Decline("register parameter or structure return")
        self.body_start = index
        self.ins = ins[index:epilogue]
        self.ret_addr = ins[epilogue].addr if epilogue < len(ins) else ins[-1].addr + 8
        calls = [x for x in self.ins if x.op in ("jal", "jalr")]
        stack = [x.imm for x in self.ins if x.op in ("sw", "swc1", "sdc1", "sb", "sh") and x.rs == SP]
        self.args_size = 0
        if calls:
            top = max([16] + [off + (8 if False else 4) for off in stack])
            self.args_size = (top + 7) & ~7
        elif stack:
            raise Decline("stack store without a call")
        self.save_base = min(saved.values())
        top = max(list(saved.values()) + [off + 4 for off in fsaved.values()]) + 4
        self.var_size = top - 4 * len(saved) - 8 * len(fsaved) - self.args_size
        if fsaved:                                            # doubles are saved above the integer registers, aligned
            self.var_size = self.save_base - self.args_size
        if self.var_size < 0 or self.var_size % 8 or self.args_size + self.var_size != self.save_base:
            raise Decline("frame layout")

    def _double_pairs(self):
        """Under ``-mfp32`` a ``double`` lives in two float registers (the odd one holds the high word) and
        is loaded and stored one word at a time, high word first: ``lwc1 $f3, A; lwc1 $f2, A+4``.  A single
        ``float`` never uses an odd register, so the pair is recognised from the instructions alone."""
        self.pair_first: dict[int, object] = {}             # address of the first instruction -> the second
        self.pair_second: set[int] = set()
        self.zero_pairs: dict[int, object] = {}             # ``sw $zero, A; sw $zero, A+4``: a double set to 0.0
        ins, zero_second = self.ins, set()
        self.skip: set[int] = set()                          # second halves already executed with the first
        for i in range(len(ins) - 1):
            x, y = ins[i], ins[i + 1]
            if x.addr in self.pair_second:
                continue
            if x.op in ("lwc1", "swc1") and y.op == x.op and x.ft % 2 == 1 and y.ft == x.ft - 1 \
                    and y.rs == x.rs and y.imm == x.imm + 4:
                self.pair_first[x.addr] = y
                self.pair_second.add(y.addr)
            elif x.op == "mfc1" and y.op == "mfc1" and x.fs % 2 == 0 and y.fs == x.fs + 1 and y.rt == x.rt - 1 \
                    and x.rt % 2 == 1:
                self.pair_first[x.addr] = y                   # a double moved to an integer register pair
                self.pair_second.add(y.addr)
            elif x.op == "sw" and y.op == "sw" and x.rt == 0 and y.rt == 0 and y.rs == x.rs \
                    and y.imm == x.imm + 4 and x.addr not in zero_second:
                self.zero_pairs[x.addr] = y
                zero_second.add(y.addr)

    def _unsigned_conversions(self):
        """``(double)u`` on an unsigned integer: converted as signed, then 2^32 is added when the sign bit
        was set (``cvt.d.w; lw; bgez over; l.d 4294967296.0; add.d``; measured).  The branch and its label
        belong to the conversion, not to the statement structure."""
        self.unsigned_fix: set[int] = set()
        # addresses of constants the compiler itself puts in read-only data (2^32 here, the 1.0 of ``x++``)
        self.literal_addrs: set[int] = set()
        self.literal_span = None
        self.literal_pool: set[int] = set()
        ins = self.ins
        for i in range(len(ins) - 4):
            x = ins[i]
            if x.op != "bgez" or self.index_of.get(x.target) != i + 5 or self.targets.get(i + 5) != [i]:
                continue
            lui, high, low, add = ins[i + 1:i + 5]
            if lui.op == "lui" and lui.rt == AT and high.op == low.op == "lwc1" and high.rs == low.rs == AT \
                    and high.ft == low.ft + 1 and low.imm == high.imm + 4 and add.op == "add" and add.fmt == "d" \
                    and add.fd == add.fs and add.ft == low.ft and self.read is not None:
                address = ((lui.uimm << 16) + high.imm) & 0xFFFFFFFF
                if (self.read(address), self.read(address + 4)) == (0x41F00000, 0):
                    self.unsigned_fix.add(i)
                    self.literal_addrs.add(address)
                    del self.targets[i + 5]

    def _wide_comparisons(self) -> dict[int, int]:
        """``X > K`` on an unsigned 64-bit value in a register pair, as the compiler lays it out (probe):
        ``bnez high, T; bnez high, F; sltiu t, low, K + 1; beqz t, T; li t, K; bne low, t, F; j F`` with T the
        instruction after it.  The whole run is one test, left standing as its last conditional branch
        (taken when the comparison is false); the rest of it is taken out of the instruction list before
        anything is indexed.  Returns {address of the run's first instruction: address of the branch kept}."""
        ins, aliases, k = self.ins, {}, 0                    # delay slots are not in the list
        while k + 7 < len(ins):
            a, b, test, c, load, d, jump = ins[k:k + 7]
            true = ins[k + 7].addr
            found = (a.op == b.op == d.op == "bne" and a.rt == b.rt == 0 and a.rs == b.rs and c.op == "beq"
                     and c.rt == 0 and test.op == "sltiu" and c.rs == test.rt and load.op == "addiu" and load.rs == 0
                     and d.rt == load.rt and d.rs == test.rs and jump.op == "j" and test.imm == load.imm + 1
                     and a.target == c.target == true and b.target == d.target == jump.target)
            inside = {x.addr for x in (b, test, c, load, d, jump)} | {true}
            if found and not any(x.target in inside for x in ins if x not in (a, c)):
                self.wide_tests[d.addr] = (a.rs, d.rs, load.imm)
                aliases[a.addr] = d.addr
                ins[k:k + 7] = [d]
            k += 1
        return aliases

    def _literal_pool(self) -> set[int]:
        """Addresses of floating constants that lie back to back with one of the compiler's own.

        The constants of a function go to read-only data one after another.  A global read only as a
        ``float`` or ``double`` that touches a constant the compiler made itself (the 2^32 of an unsigned
        conversion), directly or through others like it, is therefore a literal of this function and not a
        variable (measured: ``a * 65536.0 * 65536.0 + (f64) (u32) b`` then ``/ 90000.0`` lays out the four
        in that order).  Written as a literal the unit owns those bytes, and the strict comparison then
        checks them against the image; written as a variable nothing checked them."""
        if not self.literal_addrs:
            return set()
        sizes = {address: 8 for address in self.literal_addrs}
        for address, info in self.globals.items():
            acc = set(info.get("acc") or ())
            if len(acc) == 1 and not info.get("addr") and next(iter(acc))[1] in ("f", "d"):
                sizes.setdefault(address, next(iter(acc))[0])
        pool = set(self.literal_addrs)
        grew = True
        while grew:
            grew = False
            for address, size in sizes.items():
                if address not in pool and (address + size in pool or any(p + sizes[p] == address for p in pool)):
                    pool.add(address)
                    grew = True
        return pool - self.literal_addrs

    def _integer_fp_accesses(self) -> set[int]:
        """Addresses of ``lwc1``/``swc1`` that move an *integer*: a conversion reads its integer
        operand straight from memory into a float register (``lwc1``, ``cvt.s.w``) and writes its
        integer result straight back (``trunc.w.s``, ``swc1``)."""
        found, ins = set(), self.ins
        for i, x in enumerate(ins):
            if x.op == "lwc1":
                for later in ins[i + 1:i + 8]:
                    if later.op in BRANCHES or later.op in ("j", "jal", "jalr", "jr"):
                        break
                    if later.fmt is not None and later.fs == x.ft:
                        if later.op in ("cvt.s", "cvt.d") and later.fmt == "w":
                            found.add(x.addr)
                        break
                    if later.op in ("lwc1", "ldc1", "mtc1") and (later.ft if later.op != "mtc1" else later.fs) == x.ft:
                        break
            elif x.op == "swc1":
                for earlier in reversed(ins[max(0, i - 8):i]):
                    if earlier.fmt is not None and earlier.fd == x.ft:
                        if earlier.op == "trunc.w":
                            found.add(x.addr)
                        break
                    if earlier.op in ("lwc1", "ldc1") and earlier.ft == x.ft or earlier.op == "mtc1" and earlier.fs == x.ft:
                        break
        return found

    def _labels(self):
        self.wide_tests: dict[int, tuple[int, int, int]] = {}
        aliases = self._wide_comparisons()
        self.index_of = {x.addr: i for i, x in enumerate(self.ins)}
        self.index_of[self.ret_addr] = len(self.ins)
        for gone, kept in aliases.items():
            self.index_of[gone] = self.index_of[kept]
        # A label on a transfer whose delay slot the assembler filled stood, in the source, in
        # front of the instruction that was moved into the slot (an assembler never moves an
        # instruction across a label).  In logical order that instruction comes first, so the
        # label names it.
        moved_slots = set()
        for i in range(1, len(self.ins)):
            if self.ins[i - 1].addr > self.ins[i].addr:
                self.index_of[self.ins[i].addr] = i - 1
                moved_slots.add(self.ins[i - 1].addr)
        self.targets: dict[int, list[int]] = {}             # target index -> indices of the jumps to it
        for i, x in enumerate(self.ins):
            if x.op in BRANCHES or x.op == "j":
                if x.target not in self.index_of:
                    raise Decline("jump out of the function")
                if x.target in moved_slots:                   # into a slot: the transfer itself is skipped
                    raise Decline("jump into a filled delay slot")
                self.targets.setdefault(self.index_of[x.target], []).append(i)
            elif x.op == "jr":
                table = self._jump_table(i)
                if table is None:
                    raise Decline("jump table")
                self.tables[table["start"]] = table
                for place in set(table["cases"]) | {table["default"]}:
                    self.targets.setdefault(place, []).append(i)

    def _scratch_variables(self):
        """``register`` variables that live in scratch registers (a function without calls keeps them in
        $v0, $v1, ...): unoptimised code never carries an expression's value over a label or a branch, so a
        scratch register that is live from one block into another holds a variable there.  One register
        can be a variable in one stretch and a temporary in another, so each write and read is classified
        where it stands."""
        ins = self.ins
        count = len(ins)
        glued = set()                    # ``x / 2^k``: a branch around the rounding, inside one expression
        for i in range(count - 2):
            x, fix, shift = ins[i], ins[i + 1], ins[i + 2]
            if x.op == "bgez" and self.index_of.get(x.target) == i + 2 and fix.op == "addiu" \
                    and fix.rs == fix.rt == x.rs and shift.op == "sra" and shift.rt == x.rs:
                glued.update((i + 1, i + 2))
        # The decision tree of a ``switch`` keeps the selector in its register from test to test: one
        # expression, though it branches.  Its tests are read as one block that leads to every case.
        tree, inside, i = {}, set(), 0                     # index of a test or jump inside a tree -> its target
        while i < count:
            x = ins[i]
            ranged = x.op in ("slti", "sltiu") and i + 1 < count and ins[i + 1].op in ("beq", "bne") \
                and ins[i + 1].rs == x.rt and ins[i + 1].rt == 0
            if (x.op == "beq" and x.rs != 0) or ranged:
                before = ins[i - 1] if i else None
                known = {before.rt: before.imm} if before is not None and before.op == "addiu" and before.rs == 0 \
                    else {}
                found = self._dispatch(i, count, known)
                if found is not None and all(i <= j <= found["jump"] for k in range(i + 1, found["jump"] + 1)
                                             for j in self.targets.get(k, ())):
                    for k in range(i, found["jump"] + 1):
                        if ins[k].op in BRANCHES or ins[k].op == "j":
                            tree[k] = self.index_of[ins[k].target]
                    inside.update(range(i + 1, found["jump"] + 1))
                    i = found["jump"] + 1
                    continue
            i += 1
        # likewise the dispatch through a jump table: bounds test, scaled index, load, jump
        for table in self.tables.values():
            inside.update(range(table["start"] + 1, table["jr"] + 1))
            tree[table["start"] + 1] = table["default"]
        leaders = {0}
        for i, x in enumerate(ins):
            if x.op in BRANCHES or x.op == "j":
                leaders.add(i + 1)
                target = self.index_of.get(x.target)
                if target is not None and target < count:
                    leaders.add(target)
        leaders = sorted(k for k in leaders if k < count and k not in glued and k not in inside)
        blocks = [(start, leaders[n + 1] if n + 1 < len(leaders) else count) for n, start in enumerate(leaders)]
        block_at = {start: n for n, (start, _end) in enumerate(blocks)}
        uses, defs, succ = [], [], []
        for start, end in blocks:
            use, define = set(), set()
            for x in ins[start:end]:
                reads, writes = reads_writes(x)
                use.update(r for r in reads if r in SCRATCH and r not in define)
                define.update(w for w in writes if w in SCRATCH)
            uses.append(use)
            defs.append(define)
            last, following = ins[end - 1], [tree[k] for k in range(start, end - 1) if k in tree]
            if last.op == "jr":
                following += [place for table in self.tables.values() if table["jr"] == end - 1
                              for place in table["cases"]]
            if last.op != "j" and end < count and end - 1 not in glued:
                following.append(end)
            if (last.op in BRANCHES or last.op == "j") and end - 1 not in glued:
                target = self.index_of.get(last.target)
                if target is not None and target < count:
                    following.append(target)
            succ.append([block_at[k] for k in following if k in block_at])
        live_in = [set() for _ in blocks]
        live_out = [set() for _ in blocks]
        changed = True
        while changed:
            changed = False
            for n in range(len(blocks) - 1, -1, -1):
                out = set().union(*(live_in[m] for m in succ[n])) if succ[n] else set()
                into = uses[n] | (out - defs[n])
                if out != live_out[n] or into != live_in[n]:
                    live_out[n], live_in[n], changed = out, into, True
        for n, (start, end) in enumerate(blocks):
            holding = set(live_in[n])
            last_def = {}
            for i in range(start, end):
                for w in reads_writes(ins[i])[1]:
                    last_def[w] = i
            for i in range(start, end):
                x = ins[i]
                reads, writes = reads_writes(x)
                for r in reads:
                    if r in holding:
                        self.var_read.add((x.addr, r))
                for w in writes:
                    if w not in SCRATCH:
                        continue
                    if x.op not in ("jal", "jalr") and ((w in live_out[n] and last_def.get(w) == i)
                                                        or (x.addr, w) in self.forced):
                        self.var_def.add((x.addr, w))
                        holding.add(w)
                    else:
                        holding.discard(w)
        # The register both arms of a conditional expression leave their value in is alive across blocks
        # without being a variable: the value of ``c ? a : b``, read at the join.
        for i, x in enumerate(ins):
            if x.op not in BRANCHES:
                continue
            try:
                shape = self._conditional_shape(i, count)
            except (AttributeError, KeyError, IndexError):
                break
            if shape is None or shape[2] != "i" or shape[3] not in SCRATCH or not self._join_reads(shape):
                continue
            _target, after, _kind, reg = shape
            for k in range(i + 1, after):
                self.var_def.discard((ins[k].addr, reg))
                self.var_read.discard((ins[k].addr, reg))
            for y in ins[after:after + 12]:
                self.var_read.discard((y.addr, reg))
                if reg in reads_writes(y)[1]:
                    break
        self._scratch_instances()

    def _spilled_chains(self) -> None:
        """``p[0] = p[1] = ... = p[31] = 0``: a chained assignment so long that the addresses do not fit the
        scratch registers.  Every address is formed first, left to right (``lw r,p($fp); addiu r2,r,4k``),
        then zero is stored through them right to left.  The compiler takes every free register for them,
        the call-saved ones too (which it then saves, though no ``register`` variable lives there), and
        spills what still does not fit to stack slots of its own behind the locals (``sw r,t($fp)`` ...
        ``lw r,t($fp)``).  Such a run is read as one statement; its spill slots are not variables and its
        saved registers are not ``register`` variables."""
        ins, n = self.ins, len(self.ins)
        inside: set[int] = set()
        i = 0
        while i < n:
            first = ins[i]
            if not (first.op == "lw" and first.rs == FP):
                i += 1
                continue
            regs: dict[int, tuple] = {}                      # register -> (order of its pointer load, local, offset)
            slots: dict[int, tuple] = {}                     # spill slot -> the same
            spills, stores, loads, wide, k = [], [], 0, False, i
            while k < n and not (k > i and k in self.targets):
                x = ins[k]
                if x.op == "lw" and x.rs == FP and x.imm in slots:
                    regs[x.rt] = slots[x.imm]
                    spills.append(k)
                elif x.op == "lw" and x.rs == FP and not stores:
                    regs[x.rt] = (loads, x.imm, 0)
                    loads += 1
                elif x.op == "sw" and x.rs == FP and x.rt in regs and not stores:
                    slots[x.imm] = regs[x.rt]
                    spills.append(k)
                elif x.op == "addiu" and x.rs in regs and x.rs != FP and x.rt not in (0, FP, SP) and not stores:
                    order, local, off = regs[x.rs]
                    regs[x.rt] = (order, local, off + x.imm)
                elif x.op == "sw" and x.rt == 0 and x.imm == 0 and x.rs in regs:
                    stores.append(regs[x.rs])
                else:
                    break
                if x.op != "sw" and x.rt >= 16:
                    wide = True
                k += 1
            orders = [order for order, _local, _off in stores]
            if len(stores) >= 3 and len(stores) == loads and orders == list(range(loads - 1, -1, -1)) \
                    and all(off % 4 == 0 and off >= 0 for _order, _local, off in stores) \
                    and (slots or wide) and not any(x.rs == FP and x.imm in slots for x in ins[:i] + ins[k:]):
                self.chains[i] = (k, [(local, off) for _order, local, off in reversed(stores)])
                for at in spills:
                    self.temp_at[ins[at].addr] = ins[at].imm
                inside.update(range(i, k))
                i = k
            else:
                i += 1
        if self.chains:
            # a call-saved register that only such a run touches holds no variable
            for reg in [r for r in self.regvars if "param" not in self.regvars[r]]:
                if not any(reg in reads or reg in writes
                           for at, x in enumerate(ins) if at not in inside for reads, writes in [reads_writes(x)]):
                    del self.regvars[reg]

    def _chain_statement(self, i: int) -> int:
        """Emit the chained assignment found at ``i``; returns the index after it."""
        after, places = self.chains[i]
        self.flush_call()
        # written as elements (``p[k]``): the address of each is formed and kept, as the code does; a member
        # access (``((struct T *)p)->m``) takes fewer spill slots and gives another frame (measured)
        value = self.node("const", 0)
        for local, off in reversed(places[1:]):
            value = self.node("asg", ("pidx", local, off // 4), value)
        self.emit(Stmt("assign", ("pidx", places[0][0], places[0][1] // 4), value, 4))
        return after

    def _scratch_instances(self):
        """One variable per stretch in which a scratch register is never a temporary.  GCC's unoptimised
        allocator gives every value, variable or temporary, the lowest register that is free for its whole
        life, longest life first, and a ``register`` variable lives from its first assignment to the end of
        the block that declares it.  So the same register is a temporary before and after, and the block
        is found from the temporaries: one that takes a higher register although it is dead before this
        register is next written shows the variable was still alive there (``extent``)."""
        ins = self.ins
        count = len(ins)
        rank = {reg: n for n, reg in enumerate(sorted(SCRATCH))}
        open_now: dict[int, dict] = {}
        counts: dict[int, int] = {}

        def touch(reg, i, addr):
            inst = open_now.get(reg)
            if inst is None:
                counts[reg] = counts.get(reg, 0) + 1
                name = f"reg_{REG[reg]}" + (f"_{counts[reg]}" if counts[reg] > 1 else "")
                inst = open_now[reg] = {"reg": reg, "first": i, "last": i, "upto": count, "name": name,
                                        "key": SCRATCH_KEY + len(self.instances)}
                self.instances.append(inst)
            inst["last"] = i
            self.instance_at[(addr, reg)] = inst["key"]

        for i, x in enumerate(ins):
            reads, writes = reads_writes(x)
            for r in reads:
                if (x.addr, r) in self.var_read:
                    touch(r, i, x.addr)
            for w in writes:
                if w not in SCRATCH:
                    continue
                if (x.addr, w) in self.var_def:
                    touch(w, i, x.addr)
                elif w in open_now:
                    open_now.pop(w)["upto"] = i
        for inst in self.instances:
            reg, upto = inst["reg"], inst["upto"]
            inst["extent"] = inst["last"]
            for i in range(inst["last"] + 1, upto):
                x = ins[i]
                for w in reads_writes(x)[1]:
                    if w not in SCRATCH or rank[w] <= rank[reg] or (x.addr, w) in self.var_def:
                        continue
                    # A temporary in a higher register that is dead before this register is written again
                    # (a value read by the very instruction that writes another is in its way: measured,
                    # ``lw $4,x; addu $3,$4,-1`` with $3 free).
                    dead = True
                    for j in range(i + 1, min(upto + 1, count)):
                        later_reads, later_writes = reads_writes(ins[j])
                        if w in later_reads and j >= upto:
                            dead = False
                        if w in later_writes:
                            break
                    if dead:
                        inst["extent"] = i
            # Backwards: a variable declared without a value is alive from the head of its block (measured:
            # ``register int t; if (c / 4 > 0) t = ...`` keeps the condition's temporaries out of $2).
            # The register is free once its last temporary before the variable has been read.
            free, k = 0, inst["first"] - 1
            while k >= 0:
                if reg in reads_writes(ins[k])[0] and (ins[k].addr, reg) not in self.var_read:
                    free = k + 1
                    break
                if reg in reads_writes(ins[k])[1]:
                    free = k + 1
                    break
                k -= 1
            inst["free"], inst["start"] = free, inst["first"]
            for i in range(inst["first"] - 1, free - 1, -1):
                x = ins[i]
                if any(w in SCRATCH and rank[w] > rank[reg] and (x.addr, w) not in self.var_def
                       for w in reads_writes(x)[1]):
                    inst["start"] = i
        for inst in self.instances:
            self.scratch_names.setdefault(inst["reg"], inst["name"])

    def regvar_name(self, key: int) -> str:
        if key in self.regvars:
            return self.regvars[key]["name"]
        return self.instances[key - SCRATCH_KEY]["name"]

    def _variable_key(self, reg: int):
        """The scratch variable the current instruction reads or writes in ``reg``."""
        if not self.scratch or self.now is None:
            return None
        return self.instance_at.get((self.now.addr, reg))

    def _is_variable_read(self, reg: int) -> bool:
        return self.scratch and self.now is not None and (self.now.addr, reg) in self.var_read

    @staticmethod
    def _narrow_store_follows(ins, index: int, epilogue: int) -> bool:
        """Whether the word loaded at ``index`` is stored narrow in what is left of the prologue.  With
        several narrow parameters the compiler loads the stack ones first and stores them all afterwards
        (``lw $a2; lw $a3; sh $v0; sh $v1; ... sh $a2; sb $a3``), so the store need not come next: only
        moves of argument registers, further such loads and narrow stores may stand in between."""
        reg = ins[index].rt
        for k in range(index + 1, min(epilogue, index + 12)):
            x = ins[k]
            if x.op in ("sh", "sb") and x.rs == FP:
                if x.rt == reg:
                    return True
                continue
            if (x.op == "lw" and x.rs == FP and x.rt != reg) or (x.op == "addu" and x.rt == 0 and x.rd != reg):
                continue
            return False
        return False

    def _jump_table(self, i: int):
        """The ``switch`` dispatched through a table by the ``jr`` at ``i``: bounds test, scaled index,
        table address, load, jump.  The table is read from the image; every entry is a place in this
        function.  When the lowest case is 0 nothing is subtracted, and the selector is evaluated a second
        time for the index: its last load is repeated after the bounds test (``lw; sltiu; beq; lw; move;
        sll``), from the address the first evaluation left in its register."""
        ins = self.ins
        if self.read is None or i < 7:
            return None
        scale, high, low, add, load, jump = ins[i - 5:i + 1]
        if not (scale.op == "sll" and scale.sa == 2
                and high.op == "lui" and low.op == "addiu" and low.rs == low.rt == high.rt
                and add.op == "addu" and {add.rs, add.rt} == {scale.rd, low.rt}
                and load.op == "lw" and load.imm == 0 and load.rs == add.rd and load.rt == jump.rs):
            return None
        at, again = i - 6, None
        if ins[at].op == "addu" and ins[at].rt == 0 and ins[at].rd == scale.rt and ins[at].rs != 0:
            # the second evaluation: loads only, back to the branch of the bounds test
            at -= 1
            while at > 1 and i - at < 12 and (ins[at].op in LOADS or ins[at].op in ("lui", "addu")):
                at -= 1
            again = (at + 1, i - 5)
            if again[0] >= i - 6:
                return None
        branch, test = ins[at], ins[at - 1]
        if not (test.op == "sltiu" and test.imm > 0 and branch.op == "beq" and branch.rs == test.rt and branch.rt == 0
                and (again is not None or scale.rt == test.rs)):
            return None
        address = ((high.uimm << 16) + low.imm) & 0xFFFFFFFF
        cases = []
        for k in range(test.imm):
            word = self.read(address + 4 * k)
            if word is None or word not in self.index_of:
                return None
            cases.append(self.index_of[word])
        default = self.index_of.get(branch.target)
        if default is None:
            return None
        return {"start": at - 1, "jr": i, "cases": cases, "default": default, "reg": test.rs, "address": address,
                "again": again, "index": scale.rt}

    # -- frame accesses -----------------------------------------------------------------------

    def _collect_frame_accesses(self):
        """Every $fp-relative access: which slots exist and how wide they are."""
        direct: dict[int, set] = {}
        self.addr_taken: set[int] = set()
        for x in self.ins:
            if x.rs != FP or x.addr in self.temp_at:
                continue
            if x.op in LOADS:
                width, signed = LOADS[x.op]
                direct.setdefault(x.imm, set()).add((width, "s" if signed else "u"))
            elif x.op in STORES:
                direct.setdefault(x.imm, set()).add((STORES[x.op], "w"))
            elif x.op in ("lwc1", "swc1") and x.addr in self.pair_second:
                continue
            elif x.op in ("lwc1", "swc1") and x.addr in self.pair_first:
                direct.setdefault(x.imm, set()).add((8, "d"))
            elif x.op in ("lwc1", "swc1"):
                integer = x.addr in self.int_fp
                direct.setdefault(x.imm, set()).add((4, ("s" if x.op == "lwc1" else "w") if integer else "f"))
            elif x.op in ("ldc1", "sdc1"):
                direct.setdefault(x.imm, set()).add((8, "d"))
            elif x.op == "addiu":
                self.addr_taken.add(x.imm)
            elif x.op == "addu" and x.rt not in (0, FP):
                continue                                     # an index into a local array (below)
            else:
                raise Decline(f"frame pointer used by {x.op}")
        for off, size in sorted(self.struct_locals.items()):
            if not (self.args_size <= off and off + size <= self.save_base):
                raise Decline("structure copy outside the locals")
            # a structure local: its words are members, and GCC rounds the slot to eight bytes
            self.vars[off] = {"off": off, "size": (size + 7) & ~7, "type": "s32", "array": size // 4, "elem": 4,
                              "struct": size, "member": ".w", "decl": f"struct n64dk_r{size} sp{off:X}"}
            self.block_sizes.add(size)
        for param in self.params:
            if param["kind"] == "narrow":
                kinds = direct.get(param["off"], set())
                signed = any(k == (param["size"], "s") for k in kinds)
                self.vars[param["off"]] = {"off": param["off"], "size": param["size"], "param": True,
                                           "type": INT_TYPES[(param["size"], signed)]}
        # Local arrays.  ``a[i]`` on a local array adds the scaled index to the frame pointer and
        # accesses the sum at the array's own frame offset.  The array reaches as far as accesses
        # of its element width go: a neighbouring scalar of that width is spelled as an element,
        # which is the same instruction and the same frame.
        indexed: dict[int, tuple[int, str]] = {}
        for i, x in enumerate(self.ins):
            if x.op == "addu" and FP in (x.rs, x.rt) and x.rt != 0 and x.rd not in (FP, SP) and x.rs != x.rt:
                base, lead = x.rd, 0
                for later in self.ins[i + 1:i + 5]:
                    if later.op == "addiu" and later.rs == base and not lead and later.rt not in (FP, SP):
                        base, lead = later.rt, later.imm        # the element's address made whole (``A[n++] = v``)
                        continue
                    if later.rs == base and (later.op in LOADS or later.op in STORES or later.op in ("lwc1", "swc1")):
                        if later.op in LOADS:
                            access = (LOADS[later.op][0], "s" if LOADS[later.op][1] else "u")
                        elif later.op in STORES:
                            access = (STORES[later.op], "w")
                        else:
                            access = (4, "f")
                        if indexed.setdefault(lead + later.imm, access)[0] != access[0]:
                            raise Decline("local array with two element widths")
                        if access[1] != "w":
                            indexed[lead + later.imm] = access
                        break
        # A slot whose word is used as a pointer straight away (the base of the next access, or stepped and
        # stored back) is a pointer variable beside the array, not one more element of it: as an element it
        # could not be the plain local the pointer rules need, and either spelling is the same frame.
        pointers = set()
        for i, x in enumerate(self.ins[:-1]):
            if x.op == "lw" and x.rs == FP:
                after = self.ins[i + 1]
                if after.rs == x.rt and (after.op in LOADS or after.op in STORES):
                    pointers.add(x.imm)
                elif after.op == "addiu" and after.rs == x.rt and any(
                        y.op == "sw" and y.rs == FP and y.imm == x.imm for y in self.ins[i + 2:i + 4]):
                    pointers.add(x.imm)
        for off, (width, kind) in sorted(indexed.items()):
            if not (self.args_size <= off < self.save_base) or off % width:
                raise Decline("indexed frame access outside the locals")
            end = off + width
            while end + width <= self.save_base and end not in indexed and end not in pointers and \
                    all(k[0] == width for k in direct.get(end, ())) and \
                    not any(end < other < end + width for other in direct):
                end += width
            kinds = {k[1] for o in range(off, end, width) for k in direct.get(o, ())} | {kind}
            ctype = "f32" if "f" in kinds else INT_TYPES[(width, "s" in kinds or width == 4)]
            self.vars[off] = {"off": off, "size": end - off, "type": ctype, "array": (end - off) // width,
                              "elem": width, "decl": f"{ctype} sp{off:X}[{(end - off) // width}]"}
        # widest first: a narrower access inside a wider slot is a conversion, not a variable
        for width in (8, 4, 2, 1):
            for off in sorted(direct):
                kinds = [k for k in direct[off] if k[0] == width]
                if not kinds or off in self.vars:
                    continue
                holder = self._holder(off, width)
                if holder is not None:
                    continue
                if off >= self.size:
                    index = (off - self.size) // 4
                    if (off - self.size) % 8 == 0 and width == 8:
                        # a ``double`` that arrived in $a2/$a3 or on the stack: two argument words
                        while len(self.params) <= index + 1:
                            number = len(self.params)
                            self.params.append({"kind": "int" if number >= 4 else "unused",
                                                "off": self.size + 4 * number, "size": 4})
                        if self.params[index]["kind"] == "double":
                            continue
                        if any(self.params[k]["kind"] not in ("int", "unused") or "type" in self.params[k]
                               for k in (index, index + 1)):
                            raise Decline("odd access to an incoming argument")
                        self.params[index] = {"kind": "double", "off": off, "size": 8, "type": "f64"}
                        self.params[index + 1] = {"kind": "pad"}
                        if index % 2 == 0 and index and self.params[index - 1]["kind"] == "unused":
                            self.params[index - 1] = {"kind": "pad"}     # the word skipped to align it
                        continue
                    if (off - self.size) % 4 == 0 and width == 4:
                        while len(self.params) <= index:
                            number = len(self.params)
                            self.params.append({"kind": "int" if number >= 4 else "unused",
                                                "off": self.size + 4 * number, "size": 4})
                        if any(k[1] == "f" for k in kinds):
                            # a ``float`` that arrived in an integer register or on the stack
                            self.params[index]["type"] = "f32"
                        continue
                    if index >= 4 and width < 4 and off % 4 == 4 - width:
                        while len(self.params) <= index:
                            number = len(self.params)
                            self.params.append({"kind": "int" if number >= 4 else "unused",
                                                "off": self.size + 4 * number, "size": 4})
                        # It is an ``int`` read through its narrow end (a cast, a narrowing store): a
                        # parameter declared narrow is copied to a slot of its own on entry, on the stack
                        # as in a register (probe, prototype and old-style definitions alike).
                        continue
                    raise Decline("odd access to an incoming argument")
                if not (self.args_size <= off and off + width <= self.save_base):
                    raise Decline("frame access outside the locals")
                if any(k[1] == "d" for k in kinds):
                    ctype = "f64"
                elif any(k[1] == "f" for k in kinds):
                    ctype = "f32"
                else:
                    ctype = INT_TYPES[(width, any(k[1] == "s" for k in kinds) or width == 4)]
                self.vars[off] = {"off": off, "size": width, "type": ctype}
        for off in sorted(self.addr_taken):
            if off >= self.size:
                index = (off - self.size) // 4
                if (off - self.size) % 4 or index >= len(self.params):
                    raise Decline("address of an incoming argument")
                continue
            if off in self.vars or self._holder(off, 1) is not None:
                continue
            if not (self.args_size <= off < self.save_base):
                raise Decline("frame address outside the locals")
            self.vars[off] = {"off": off, "size": 0, "type": None, "aggregate": True}

    def _holder(self, off: int, width: int):
        """The already known wider slot that contains [off, off+width), if any."""
        for start, var in self.vars.items():
            if var.get("aggregate"):
                continue
            if start <= off and off + width <= start + var["size"] and (start, var["size"]) != (off, width):
                return var
        if off >= self.size:
            index = (off - self.size) // 4
            if index < len(self.params) and self.params[index]["kind"] in ("int", "unused") and width < 4:
                return self.params[index]
        return None

    # -- symbolic execution -------------------------------------------------------------------

    def node(self, k, *a) -> Node:
        n = Node(k, *a)
        self.seq += 1
        n.seq = self.seq
        return n

    def _reset(self):
        self.regs: dict[int, Node] = {}
        self.fregs: dict[int, Node] = {}
        self.fcc = None
        self.hilo = None
        self.stack_args: dict[int, Node] = {}
        self.stack_copy: dict[int, Node] = {}                # words of a structure argument copied to the stack
        self.stack_block = None                              # (offset, bytes, source) of a block copied there
        self.struct_args: set[int] = set()                   # sizes of structures passed by value
        self.stack_wide: set[int] = set()                    # offsets of the stack arguments that are doubles
        self.pending_call = None

    def usef(self, reg: int, double: bool = False) -> Node:
        n = self.fregs.get(reg)
        if n is None:
            raise Decline(f"read of $f{reg} with no known value")
        for part in ([strip(n.a[1]), strip(n.a[2])] if n.k == "cond" else [n]):
            if part.k == "call":
                self.float_calls.add(id(part))                # its value is taken from $f0
                if double:
                    self.double_calls.add(id(part))           # ... and from $f1: the callee returns a double
                self.keep.append(part)
        used = self.consume(n)
        if used is not n:
            self.fregs[reg] = used                            # ``a = b = c = 0.0f``: the chain goes on
        return used

    def is_float(self, n: Node) -> bool:
        n = strip(n)
        if n.k == "asg":
            return self.is_float(n.a[1])
        if n.k == "cond":
            return self.is_float(n.a[1]) or self.is_float(n.a[2])
        return n.k in ("fconst", "dconst", "fbin", "fun", "cvtf") or (n.k == "ld" and n.a[1] == "f") \
            or (n.k == "call" and id(n) in self.float_calls)

    def float_type(self, n: Node) -> str:
        n = strip(n)
        if n.k == "ld":
            return "f32" if n.a[0] == 4 else "f64"
        if n.k == "cvtf":
            return n.a[0]
        if n.k in ("fbin", "fun", "asg"):
            return self.float_type(n.a[1])
        if n.k == "cond":
            return self.float_type(n.a[1] if self.is_float(n.a[1]) and strip(n.a[1]).k != "call" else n.a[2])
        if n.k == "dconst" or (n.k == "call" and id(n) in self.double_calls):
            return "f64"
        return "f32"

    def use(self, reg: int) -> Node:
        if reg == 0:
            return self.node("const", 0)
        if reg == self.sret:
            return self.node("sret")
        if reg in self.saved_temp:
            return self.consume(self.saved_temp[reg])
        if reg in self.regvars or self._is_variable_read(reg):
            key = reg if reg in self.regvars else self._variable_key(reg)
            n = self.node("ld", 4, True, ("regvar", key))   # the variable itself: no instruction reads it
            n.uses += 1
            self.regvar_reads[reg] = self.regvar_reads.get(reg, 0) + 1
            return n
        n = self.regs.get(reg)
        if n is None:
            raise Decline(f"read of ${REG[reg]} with no known value")
        return self.consume(n, reg)

    def consume(self, n: Node, reg=None) -> Node:
        if n.seq <= self.barrier:
            n = self._stale(n, reg)
        n.uses += 1
        if n.k == "low":                                      # the call was used: by its second word
            n.a[0].uses += 1
            self.wide_calls.add(id(n.a[0]))
        return n

    def _wide_value(self, high: Node, low: Node, stored: bool = True):
        """The 64-bit value whose two words are ``high`` and ``low`` (big-endian: the high word first), or
        None.  Measured: ``D = f(g(), K)`` with 64-bit values is ``jal g; move $4,$2; move $5,$3; lw $6,K;
        lw $7,K+4; jal f; sw $3,D``; ``G = 0`` is ``move $2,$0; move $3,$0; sw $2,G; sw $3,G+4``; ``G++`` is
        ``lw $6,G; lw $7,G+4; addu $5,$7,1; sltu $2,$5,1; addu $4,$6,$2; move $2,$4; move $3,$5; sw; sw``
        (``G += 1`` the same without the two moves)."""
        top, bottom = strip(high), strip(low)
        if bottom.k == "low" and bottom.a[0] is top:
            self.wide_calls.add(id(top))
            return top
        if top.k == "ld" and bottom.k == "ld" and top.a[0] == bottom.a[0] == 4 and top.a[2][0] == "global" \
                and bottom.a[2] == ("global", top.a[2][1] + 4):
            return self._wide_load(top.a[2][1])
        if stored and high.k == "cp" and low.k == "cp" and const_of(top) == 0 and const_of(bottom) == 0:
            return self.node("const", 0)                      # as arguments two zeros are ``f(0, 0)``
        if top.k == "mh" and bottom.k == "bin" and bottom.a[0] == "*" and not top.a[2]:
            # both halves of a signed 32 x 32 multiplication: ``(s64) x * K`` (probe: ``mult; mfhi; mflo``)
            for value, constant in ((bottom.a[1], bottom.a[2]), (bottom.a[2], bottom.a[1])):
                if strip(value) is strip(top.a[0]) and const_of(constant) is not None \
                        and const_of(constant) & 0xFFFFFFFF == top.a[1]:
                    return self.node("wmul", value, const_of(constant))
        if top.k == "bin" and top.a[0] == ">>" and const_of(top.a[2]) == 31 and bottom.k == "ld" \
                and strip(top.a[1]).k == "ld" and strip(top.a[1]).a[2] == bottom.a[2] and bottom.a[0] == 4:
            return bottom                                     # a signed word widened: its sign in the high word
        if top.k == "bin" and top.a[0] == "-" and bottom.k == "bin" and bottom.a[0] == "-":
            # ``X - Y`` on two words each (probe): ``sltu c, xl, yl; subu l, xl, yl; subu h, xh, yh; subu h, h, c``
            carry, inner = strip(top.a[2]), strip(top.a[1])
            if carry.k == "bin" and carry.a[0] == "<u" and inner.k == "bin" and inner.a[0] == "-" \
                    and strip(carry.a[1]) is strip(bottom.a[1]) and strip(carry.a[2]) is strip(bottom.a[2]):
                left = self._wide_value(inner.a[1], bottom.a[1], stored=False)
                right = self._wide_value(inner.a[2], bottom.a[2], stored=False)
                if left is not None and right is not None:
                    return self.node("bin", "-", left, right)
        if top.k == "bin" and top.a[0] == "+" and bottom.k == "bin" and bottom.a[0] == "+":
            amount, carry = const_of(bottom.a[2]), strip(top.a[2])
            old_high, old_low = strip(top.a[1]), strip(bottom.a[1])
            if amount is not None and amount > 0 and carry.k == "bin" and carry.a[0] == "<u" \
                    and strip(carry.a[1]) is bottom and const_of(carry.a[2]) == amount:
                old = self._wide_value(old_high, old_low)
                if old is not None and old.k == "ld":
                    made = self.node("bin", "+", old, self.node("const", amount))
                    return self.node("cp", made) if high.k == "cp" and low.k == "cp" else made
        return None

    def _wide_load(self, addr: int) -> Node:
        """A 64-bit global read as two words: one read of a ``u64``."""
        for word in (addr, addr + 4):
            info = self.globals.get(word)
            if info is not None and (4, "s") in info["acc"]:
                info["acc"].remove((4, "s"))
                if not info["acc"] and not info["addr"] and word != addr:
                    del self.globals[word]
        self.note_global(addr, (8, "q"))
        self.globals[addr]["acc"].sort(key=lambda kind: kind != (8, "q"))
        return self.node("ld", 8, "q", ("global", addr))

    def _wide_store(self, x, width: int) -> bool:
        """The second word of a 64-bit store to a global: both words were ready before the first was
        stored.  The two stores become one assignment."""
        last = self.out[-1] if self.out else None
        base, low = self.regs.get(x.rs), self.regs.get(x.rt)
        if width != 4 or last is None or last.k != "assign" or last.a[2] != 4 or base is None or base.k != "hi" \
                or low is None or low.seq > self.barrier:
            return False
        addr = (((base.a[0] << 16) + x.imm) & 0xFFFFFFFF) - 4
        if last.a[0] != ("global", addr):
            return False
        value = self._wide_value(last.a[1], low)
        if value is None:
            return False
        base.uses += 1
        low.uses += 1
        self.out.pop()
        info = self.globals.get(addr)
        if info is not None and (4, "w") in info["acc"]:
            info["acc"].remove((4, "w"))
        self.note_global(addr, (8, "q"))
        self.globals[addr]["acc"].sort(key=lambda kind: kind != (8, "q"))
        stmt = Stmt("assign", ("global", addr), value, 8)
        value.stmt = stmt
        self.emit(stmt)
        return True

    def _stale(self, n: Node, reg) -> Node:
        """A value made before the last statement and read again: ``a = b = c`` or ``a = b++``."""
        last = self.out[-1] if self.out else None
        if last is None or last.k != "assign":
            self._variable_in_block(n, reg, None)
            raise Decline("value used across a statement")
        loc, value = last.a[0], last.a[1]
        if n.stmt is last:
            self._variable_in_block(n, reg, ("const",))
            # the value just stored is used again: the assignment is an operand
            self.out.pop()
            merged = self.node("asg", loc, value)
        else:
            inner = strip(value)
            bare = strip(n)
            old = strip(inner.a[1]) if inner.k == "bin" and inner.a[0] == "+" and const_of(inner.a[2]) is not None \
                and value.k != "cp" else None
            if old is not None and old.k == "ld" and old.a[2] == loc and bare is not old \
                    and getattr(self, "deferred", None) is None:
                # Another value from before the store is still in use: the store was the side
                # effect of a post-increment in the middle of an expression (``a[i] = p->q[p->n++]``)
                # and its old value comes later.  The statement becomes that operand; values made
                # before it are current again.
                self.out.pop()
                post = self.node("postinc", loc, const_of(inner.a[2]), old.a[0])
                for where in (self.regs, self.fregs):
                    for key, held in list(where.items()):
                        if strip(held) is old:
                            where[key] = post
                self.barrier = getattr(self, "before_barrier", 0)
                post.seq = max(post.seq, self.barrier + 1)
                self.deferred = post
                return n
            if not (bare.k == "ld" and bare.a[2] == loc and inner.k == "bin" and inner.a[0] == "+"
                    and strip(inner.a[1]) is bare and const_of(inner.a[2]) is not None and value.k != "cp"):
                self._variable_in_block(n, reg, None)
                raise Decline("value used across a statement")
            self.out.pop()
            merged = self.node("postinc", loc, const_of(inner.a[2]), bare.a[0])
        if reg is not None:
            self.regs[reg] = merged
        return merged

    def _variable_in_block(self, n: Node, reg, more=()):
        """A value read by a later statement than the one that computed it.  GCC copies a computed value
        before it stores it twice (``a = b = x + 1``: ``addu $2,$3,1; move $3,$2; sw; sw``) and loads a
        constant again (``a = b = 5``: ``li; sw; li; sw``), so with neither it is a ``register`` variable
        assigned in this block (``v = x + 1; a = v; b = v;``: ``addu $3,$4,1; sw; sw``).  ``more`` None:
        the value is not the one just stored, so nothing but a variable carries it this far."""
        if reg is None or reg not in SCRATCH or reg not in self.reg_def or self.regs.get(reg) is not n:
            return
        if more is None or n.k in COMPUTED or (n.k in more and n.a[0] != 0):
            if self.scratch:
                raise Promote(self.reg_def[reg], reg)
            raise Decline(f"read of ${REG[reg]} with no known value in its statement")

    def _call_value_awaited(self) -> bool:
        """Whether the value of the call just made is still to be read ($v0 or $v1 read before anything
        else writes it, in the instructions that follow without a transfer)."""
        call = self.pending_call
        at = self.index_of.get(self.now.addr) if self.now is not None else None
        if call is None or call.uses or at is None or at >= len(self.ins) or self.ins[at] is not self.now:
            return False
        for x in self.ins[at + 1:at + 12]:
            if x.op in BRANCHES or x.op in ("j", "jal", "jalr", "jr"):
                return False
            reads, writes = reads_writes(x)
            if V0 in reads or V1 in reads:
                return True
            if V0 in writes:
                return False
        return False

    def _is_udivdi3(self, addr: int) -> bool:
        """libgcc2's ``__udivdi3``: nothing but ``return __udivmoddi4 (n, d, (UDItype *) 0)``, a call with a
        zero fifth argument word.  The compiler calls it for ``/`` on unsigned 64-bit values."""
        from .runtimehelpers import is_udivdi3
        return is_udivdi3(self.read, addr)

    def setreg(self, reg: int, n: Node):
        if reg == 0:
            return
        if reg in self.regvars and self._call_value_awaited():
            # Written between a call and the use of its value: no statement ends there.  The compiler keeps a
            # value it needs after a further call in a call-saved register (measured: ``p->q->m = f() / K``
            # is ``jal f; lw $4,p; lw $17,q($4); move $4,$2; ...; jal __udivdi3; sw $3,m($17)``).
            self.saved_temp[reg] = n
            self.saved_temps_seen.add(reg)
            return
        if reg in self.regvars or (self.scratch and self.now is not None and (self.now.addr, reg) in self.var_def):
            # the instruction that finishes the value writes the variable's register: an assignment
            n.uses += 1
            self.flush_call()
            self.regvar_writes[reg] = self.regvar_writes.get(reg, 0) + 1
            stmt = Stmt("assign", ("regvar", reg if reg in self.regvars else self._variable_key(reg)), n, 4)
            n.stmt = stmt
            self.emit(stmt)
            return
        self.regs[reg] = n
        if self.now is not None:
            self.reg_def[reg] = self.now.addr

    def emit(self, stmt: Stmt):
        self.saved_temp = {}
        pending = getattr(self, "deferred", None)
        if pending is not None:
            self.deferred = None
            if pending.uses == 0:
                raise Decline("a stored increment whose old value is never used")
        self.out.append(stmt)
        if stmt.k in ("assign", "copy"):
            self.before_barrier = self.barrier
            self.barrier = self.seq

    def flush_call(self):
        """A call whose value nobody read is a statement of its own."""
        call = self.pending_call
        if call is not None and id(call) in self.forwarded:
            raise Decline("result pointer handed to a call that is not returned")
        if call is not None and call.uses == 0:
            self.pending_call = None
            for reg in [r for r, n in self.regs.items() if n is call or (n.k == "low" and n.a[0] is call)]:
                del self.regs[reg]
            for reg in [r for r, n in self.fregs.items() if n is call]:
                del self.fregs[reg]
            self.emit(Stmt("expr", call))
        self.pending_call = None

    def location(self, base: int, off: int):
        """The lvalue ``off(base)`` names."""
        if base == FP:
            return ("local", off)
        if base == SP:
            raise Decline("stack pointer access")
        if base in self.regvars or self._is_variable_read(base):
            return ("deref", self.use(base), off)
        n = self.regs.get(base)
        if n is None:
            raise Decline(f"address in ${REG[base]} with no known value")
        if n.k == "hi":
            n.uses += 1
            return ("global", ((n.a[0] << 16) + off) & 0xFFFFFFFF)
        if n.k == "gidx":
            n.uses += 1
            return ("gidx", ((n.a[0] << 16) + off) & 0xFFFFFFFF, n.a[1])
        if n.k == "lidx":
            n.uses += 1
            return ("lidx", off, n.a[0])
        if n.k == "lelem":
            # the element's address made whole first (``A[n++] = v``: the index has a side effect, so the
            # address is in a register before the value is computed; measured on a title)
            n.uses += 1
            return ("lidx", n.a[0] + off, n.a[1])
        if n.k == "far":
            n.uses += 1
            return ("deref", n.a[0], (n.a[1] << 16) + off)
        place = strip(n)
        if place.k == "bin" and place.a[0] == "+":
            # ``&SYM[i]`` formed as an address and accessed at a member's offset: the array starts at SYM
            for symbol, index in ((strip(place.a[1]), strip(place.a[2])), (strip(place.a[2]), strip(place.a[1]))):
                if symbol.k == "sym" and index.k == "mul" and index.a[1] >= 8 and 0 <= off < index.a[1]:
                    self.anchors.setdefault(symbol.a[0], index.a[1])
        return ("deref", self.use(base), off)

    def step(self, x):
        """Execute one straight-line instruction."""
        self.now = x
        op = x.op
        if x.addr in self.temp_at:                            # through the compiler's own temporary
            if op == "sw":
                self.temp_value[x.imm] = self.use(x.rt)
            else:
                self.setreg(x.rt, self.temp_value.pop(x.imm))
            return
        if op in LOADS and x.rs not in (FP, SP) and x.rs not in self.regvars and self.out \
                and self.out[-1].k == "assign":
            last, base = self.out[-1], self.regs.get(x.rs)
            if base is not None and base.seq <= self.barrier and last.a[0][0] == "deref" \
                    and strip(last.a[0][1]) is strip(base) and last.a[0][2] == x.imm and last.a[2] == LOADS[op][0]:
                # ``if ((p->x = f()) != 0)``: the stored value is read back from where it was put
                self.out.pop()
                self.barrier = getattr(self, "before_barrier", 0)
                merged = self.node("asg", last.a[0], last.a[1])
                self.setreg(x.rt, merged)
                return
        if op in LOADS:
            width, signed = LOADS[op]
            loc = self.location(x.rs, x.imm)
            if loc[0] == "global":
                self.note_global(loc[1], (width, "s" if signed else "u"))
            self.note_indexed(loc, (width, "s" if signed else "u"))
            self.setreg(x.rt, self.node("ld", width, signed, loc))
        elif op in STORES:
            width = STORES[op]
            held = self.regs.get(x.rs)
            if held is not None and held.k == "spad":        # a word of a structure argument's tail
                if width != 4:
                    raise Decline("narrow store into a structure argument")
                held.uses += 1
                self.stack_copy[held.a[0] + x.imm] = self.use(x.rt)
                return
            if x.rs == SP:
                self.stack_args[x.imm] = self.use(x.rt)
                if width != 4:
                    raise Decline("narrow stack argument")
                return
            if x.addr in self.skip:
                return
            if x.addr in self.zero_pairs and x.rs == FP and self._float_place(("local", x.imm)) == "f64":
                # ``y = 0.0`` on a double is two integer stores of zero (measured)
                self.flush_call()
                self.skip.add(self.zero_pairs[x.addr].addr)
                value = self.node("dconst", 0)
                value.uses += 1
                stmt = Stmt("assign", ("local", x.imm), value, 8)
                value.stmt = stmt
                self.emit(stmt)
                return
            if x.addr in self.zero_pairs and x.rs not in (FP, SP) and x.rs not in self.regvars \
                    and self.regs.get(x.rs) is not None and self.regs[x.rs].k not in ("hi", "gidx", "lidx") \
                    and self.regs[x.rs].seq > self.barrier:
                # ``*p = 0.0`` on a double: both words through one loaded address (two integer members set
                # to zero load the pointer twice; measured)
                loc = self.location(x.rs, x.imm)
                self.flush_call()
                self.skip.add(self.zero_pairs[x.addr].addr)
                value = self.node("dconst", 0)
                value.uses += 1
                stmt = Stmt("assign", loc, value, 8)
                value.stmt = stmt
                self.emit(stmt)
                return
            base, last = self.regs.get(x.rs), (self.out[-1] if self.out else None)
            if x.rs not in (FP, SP) and x.rs not in self.regvars and base is not None and base.seq <= self.barrier \
                    and base.k not in ("hi", "gidx", "lidx") and last is not None and last.k == "assign" \
                    and last.a[2] == width:
                held, stored = self.regs.get(x.rt), strip(last.a[1])
                while stored.k == "asg":                     # the innermost value of a longer chain
                    stored = strip(stored.a[1])
                if (x.rt == 0 and const_of(stored) == 0) or \
                        (held is not None and strip(held) is stored) or \
                        (held is not None and held.stmt is last):
                    # ``p->a = p->b = v``: both addresses were formed before the inner store
                    base.uses += 1
                    self.out.pop()
                    merged = self.node("asg", last.a[0], last.a[1])
                    stmt = Stmt("assign", ("deref", base, x.imm), merged, width)
                    merged.stmt = stmt
                    self.emit(stmt)
                    return
            if self._wide_store(x, width):
                return
            # the address is formed before the value is read
            loc = self.location(x.rs, x.imm)
            if loc[0] == "global":
                self.note_global(loc[1], (width, "w"))
            self.note_indexed(loc, (width, "w"))
            value = self.use(x.rt)
            self.flush_call()
            stmt = Stmt("assign", loc, value, width)
            value.stmt = stmt
            self.emit(stmt)
        elif op == "lui":
            self.setreg(x.rt, self.node("hi", x.uimm))
        elif op == "addiu":
            if x.rs == 0:
                made = self.node("const", x.imm)
                if x.imm == 0:
                    self.li_zero.add(id(made))                # ``li`` of zero (plain zero is ``move $r,$zero``):
                    self.keep.append(made)                    # the low word of a double constant
                self.setreg(x.rt, made)
            elif x.rs == FP:
                self.setreg(x.rt, self.node("lad", x.imm))
            elif x.rs == SP:
                # the address of the outgoing argument area: a structure passed by value has the part that
                # does not fit the four argument registers copied there word by word
                if x.imm < 16:
                    raise Decline("stack pointer arithmetic")
                self.setreg(x.rt, self.node("spad", x.imm))
            elif self.regs.get(x.rs) is not None and self.regs[x.rs].k == "lidx":
                indexed = self.regs[x.rs]
                indexed.uses += 1
                self.setreg(x.rt, self.node("lelem", x.imm, indexed.a[0]))
            else:
                src = self.regs.get(x.rs)
                if src is not None and src.k == "hi" and x.rt == x.rs:
                    src.uses += 1
                    self.note_global(((src.a[0] << 16) + x.imm) & 0xFFFFFFFF, "addr")
                    self.setreg(x.rt, self.node("sym", ((src.a[0] << 16) + x.imm) & 0xFFFFFFFF))
                else:
                    self.setreg(x.rt, self.node("bin", "+", self.use(x.rs), self.node("const", x.imm)))
        elif op == "ori":
            src = self.regs.get(x.rs)
            if x.rs == 0:
                self.setreg(x.rt, self.node("const", x.uimm))
            elif src is not None and src.k == "hi" and x.rt == x.rs:
                src.uses += 1
                self.setreg(x.rt, self.node("const", _s32((src.a[0] << 16) | x.uimm)))
            else:
                self.setreg(x.rt, self.node("bin", "|", self.use(x.rs), self.node("const", x.uimm)))
        elif op in ("andi", "xori"):
            src = self.use(x.rs)
            if op == "andi" and id(src) in self.narrow_div and x.uimm == (1 << 8 * strip(src.a[1]).a[0]) - 1:
                # ``x / 2^k`` on an unsigned narrow variable: the operand is loaded into the second free
                # register and the quotient made in the first (probe: ``x >> k`` has them the other way)
                self.setreg(x.rt, self.node("bin", "/", src.a[1], self.node("const", 1 << const_of(src.a[2]))))
                return
            made = self.node("bin", {"andi": "&", "xori": "^"}[op], src, self.node("const", x.uimm))
            if op == "andi" and x.rt > x.rs and x.uimm & (x.uimm + 1) == 0 and x.uimm:
                self.value_and.add(id(made))
                self.keep.append(made)
            self.setreg(x.rt, made)
        elif op in ("slti", "sltiu"):
            self.setreg(x.rt, self.node("bin", "<" if op == "slti" else "<u", self.use(x.rs),
                                        self.node("const", x.imm)))
        elif op == "addu":
            if x.rt == 0:
                self.setreg(x.rd, self.node("cp", self.use(x.rs)))
                return
            left = self.regs.get(x.rs)
            if left is not None and left.k == "hi" and x.rd == x.rs and (x.rs == AT or x.rt not in (AT, FP, SP)):
                # a store builds the element's address in $at; a load builds it in the register it loads
                # into (``lui $v1,hi; addu $v1,$v1,$v0; lw $v1,lo($v1)``), and the low half may be negative
                left.uses += 1
                self.setreg(x.rd, self.node("gidx", left.a[0], self.use(x.rt)))
                return
            right = self.regs.get(x.rt)
            if right is not None and right.k == "hi" and x.rd == x.rt == AT and x.rs not in (AT, FP, SP):
                # ``lw big($reg)``: the assembler builds a displacement that does not fit sixteen bits
                # in $at (``lui $at,hi; addu $at,$reg,$at``; a symbol's index is added the other way round)
                right.uses += 1
                self.setreg(x.rd, self.node("far", self.use(x.rs), right.a[0]))
                return
            if FP in (x.rs, x.rt):                           # the scaled index of a local array
                self.setreg(x.rd, self.node("lidx", self.use(x.rt if x.rs == FP else x.rs)))
                return
            a, b = self.use(x.rs), self.use(x.rt)
            self.setreg(x.rd, self._product_step(x, a, b, self._add(a, b)))
        elif op == "subu":
            if x.rs == 0:
                value = self.use(x.rt)
                if value.k == "cp":
                    # ``x * -1``: the operand goes through a ``move`` like any multiplication by a constant
                    # (probe: ``-x`` and ``0 - x`` negate the loaded value directly)
                    self.setreg(x.rd, self.node("mul", value.a[0], -1))
                elif value.k == "mul" and id(value) not in self.inplace_products:
                    # ``x * -C``: the steps of ``x * C`` with the last one in a register of its own, then
                    # the negation.  ``-(x * C)`` finishes the product in the accumulator (measured:
                    # ``subu $2,$4,$2; negu $3,$2`` against ``subu $3,$3,$2; negu $2,$3``).
                    self.setreg(x.rd, self.node("mul", value.a[0], -value.a[1]))
                else:
                    self.setreg(x.rd, self.node("un", "-", value))
                return
            a, b = self.use(x.rs), self.use(x.rt)
            self.setreg(x.rd, self._product_step(x, a, b, self._sub(a, b)))
        elif op in ("and", "or", "xor", "slt", "sltu"):
            a, b = self.use(x.rs), self.use(x.rt)
            self.setreg(x.rd, self.node("bin", {"and": "&", "or": "|", "xor": "^", "slt": "<", "sltu": "<u"}[op],
                                        a, b))
        elif op == "nor":
            if x.rs == 0 or x.rt == 0:
                self.setreg(x.rd, self.node("un", "~", self.use(x.rt if x.rs == 0 else x.rs)))
            else:
                raise Decline("nor")
        elif op in ("sll", "srl", "sra"):
            a = self.use(x.rt)
            if op == "sll" and a.k == "postinc" and x.rd != x.rt:
                # ``A[n++]``: the old value scaled into another register, then the array's address
                # (probe: written as a shift the index stays in its register and addresses ``A($reg)``)
                made = self.node("mul", a, 1 << x.sa)
            else:
                made = self._shift(op, a, x.sa)
            if op == "srl" and x.rd < x.rt and made.k == "bin" and made.a[0] == ">>u" and a.k == "ld" \
                    and a.a[0] in (1, 2) and a.a[1] is False:
                self.narrow_div.add(id(made))
                self.keep.append(made)
            self.setreg(x.rd, made)
        elif op in ("sllv", "srlv", "srav"):
            a, b = self.use(x.rt), self.use(x.rs)
            self.setreg(x.rd, self.node("bin", {"sllv": "<<", "srlv": ">>u", "srav": ">>"}[op], a, b))
        elif op in ("mult", "multu"):
            self.hilo = (op, self.use(x.rs), self.use(x.rt))
        elif op in ("div", "divu") and x.fmt is None:
            self.hilo = (op, self.use(x.rs), self.use(x.rt))
        elif op in ("mflo", "mfhi"):
            if self.hilo is None:
                raise Decline(op)
            kind = self.hilo[0]
            if kind == "mult" and op == "mfhi" and (const_of(self.hilo[1]) is None) != (const_of(self.hilo[2]) is None):
                # the high half of ``x * magic``: a signed division by a constant in the making
                value, magic = (self.hilo[1], self.hilo[2]) if const_of(self.hilo[2]) is not None \
                    else (self.hilo[2], self.hilo[1])
                self.setreg(x.rd, self.node("mh", value, const_of(magic) & 0xFFFFFFFF, False, 0))
                return
            if kind == "multu" and op == "mfhi" and (const_of(self.hilo[1]) is None) != (const_of(self.hilo[2]) is None):
                # the high half of an unsigned ``x * magic``: an unsigned division by a constant in the making
                value, magic = (self.hilo[1], self.hilo[2]) if const_of(self.hilo[2]) is not None \
                    else (self.hilo[2], self.hilo[1])
                self.setreg(x.rd, self.node("mhu", value, const_of(magic) & 0xFFFFFFFF, False))
                return
            if kind in ("mult", "multu") and op == "mflo":
                self.setreg(x.rd, self.node("bin", "*", self.hilo[1], self.hilo[2]))
            elif kind in ("div", "divu"):
                sign = "" if kind == "div" else "u"
                self.setreg(x.rd, self.node("bin", ("/" if op == "mflo" else "%") + sign, self.hilo[1], self.hilo[2]))
            else:
                raise Decline(op)
        elif op in ("lwc1", "ldc1"):
            if x.addr in self.pair_second:
                return                                        # the low word of the double just loaded
            pair = x.addr in self.pair_first
            width = 8 if pair or op == "ldc1" else 4
            if pair:
                loc = self.location(x.rs, x.imm)
                if loc[0] == "global":
                    self.note_global(loc[1], (8, "d"))
                self.note_indexed(loc, (8, "d"))
                self.fregs[x.ft - 1] = self.node("ld", 8, "f", loc)
                self.fregs.pop(x.ft, None)
                return
            loc = self.location(x.rs, x.imm)
            if x.addr in self.int_fp:                         # an integer on its way to a conversion
                if loc[0] == "global":
                    self.note_global(loc[1], (4, "s"))
                self.fregs[x.ft] = self.node("ibits", self.node("ld", 4, True, loc))
                return
            if loc[0] == "global":
                self.note_global(loc[1], (width, "f" if width == 4 else "d"))
            self.note_indexed(loc, (width, "f" if width == 4 else "d"))
            self.fregs[x.ft] = self.node("ld", width, "f", loc)
        elif op in ("swc1", "sdc1"):
            if x.addr in self.pair_second:
                return                                        # the low word of the double just stored
            pair = x.addr in self.pair_first
            width = 8 if pair or op == "sdc1" else 4
            source = x.ft - 1 if pair else x.ft
            if x.rs == SP:
                # an argument passed on the stack as a float (a prototype is in scope) or as a double
                value = self.usef(source, width == 8)
                if not self.is_float(value) or (self.float_type(value) == "f64") != (width == 8):
                    raise Decline("float stack argument")
                self.stack_args[x.imm] = value
                if width == 8:
                    self.stack_wide.add(x.imm)
                return
            loc = self.location(x.rs, x.imm)
            if loc[0] == "global":
                self.note_global(loc[1], (width, "f" if width == 4 else "d"))
            self.note_indexed(loc, (width, "f" if width == 4 else "d"))
            value = self.usef(source, width == 8)
            if x.addr in self.int_fp and value.k == "ftrunc":
                value = self.node("cast", "s32", value.a[0])    # the integer result of a conversion
                if loc[0] == "global":
                    self.globals[loc[1]]["acc"][-1] = (4, "w")
            elif not self.is_float(value):
                raise Decline("integer bits stored as a float")
            elif (self.float_type(value) == "f64") != (width == 8):
                raise Decline("float stored with the other width")
            self.flush_call()
            stmt = Stmt("assign", loc, value, width)
            value.stmt = stmt
            self.emit(stmt)
        elif op == "mtc1":
            src = self.regs.get(x.rt)
            low = self.fregs.get(x.fs - 1)
            if x.rt == 0 and x.fs % 2 == 1 and low is not None and low.k == "fconst" and low.a[0] == 0 \
                    and low.uses == 0:
                self.fregs[x.fs - 1] = self.node("dconst", 0)   # both halves cleared: the double 0.0
            elif x.rt == 0:
                self.fregs[x.fs] = self.node("fconst", 0)
            elif src is not None and src.k in ("hi", "const"):
                src.uses += 1
                bits = (src.a[0] << 16) if src.k == "hi" else src.a[0]
                self.fregs[x.fs] = self.node("fconst", bits & 0xFFFFFFFF)
            else:
                self.fregs[x.fs] = self.node("ibits", self.use(x.rt))
        elif op == "mfc1" and x.addr in self.pair_second:
            return
        elif op == "mfc1" and x.addr in self.pair_first:
            # a ``double`` argument that travels in an integer register pair (the callee has no prototype,
            # or the argument follows an integer one)
            src = self.usef(x.fs, True)
            if not self.is_float(src) or self.float_type(src) != "f64":
                raise Decline("float moved to an integer register")
            self.setreg(x.rt - 1, self.node("dbits", src))
            self.setreg(x.rt, self.node("dlow", src))
        elif op == "mfc1":
            src = self.usef(x.fs)
            if src.k != "ftrunc":
                # a ``float`` argument after an integer one travels in the integer argument register
                if 4 <= x.rt <= 7 and self.is_float(src) and self.float_type(src) == "f32":
                    self.setreg(x.rt, self.node("fbits", src))
                    return
                raise Decline("float moved to an integer register")
            self.setreg(x.rt, self.node("cast", "s32", src.a[0]))
        elif x.fmt is not None and op in ("add", "sub", "mul", "div"):
            wide = x.fmt == "d"
            a, b = self.usef(x.fs, wide), self.usef(x.ft, wide)
            if not (self.is_float(a) and self.is_float(b)):
                raise Decline("float arithmetic on integer bits")
            self.fregs[x.fd] = self.node("fbin", {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op], a, b)
        elif x.fmt is not None and op == "neg":
            self.fregs[x.fd] = self.node("fun", "-", self.usef(x.fs, x.fmt == "d"))
        elif x.fmt is not None and op == "mov":
            self.fregs[x.fd] = self.node("cp", self.usef(x.fs, x.fmt == "d"))
        elif op in ("cvt.s", "cvt.d"):
            src, to = self.usef(x.fs, x.fmt == "d"), ("f32" if op == "cvt.s" else "f64")
            if x.fmt == "w":
                if src.k != "ibits":
                    raise Decline("conversion of a truncated value")
                self.fregs[x.fd] = self.node("cvtf", to, src.a[0])
            else:
                self.fregs[x.fd] = self.node("cvtf", to, src)
        elif op == "trunc.w":
            self.fregs[x.fd] = self.node("ftrunc", self.usef(x.fs, x.fmt == "d"))
        elif op in ("c.eq", "c.lt", "c.le"):
            wide = x.fmt == "d"
            self.fcc = ({"c.eq": "==", "c.lt": "<", "c.le": "<="}[op], self.usef(x.fs, wide), self.usef(x.ft, wide))
        else:
            raise Decline(f"instruction {op}")

    def _same(self, a: Node, b: Node) -> bool:
        return strip(a) is strip(b)

    def _unsigned_quotient(self, high: Node, after: int):
        """``x / d`` (unsigned) for the high half ``high`` shifted right by ``after``, or None."""
        value, magic, add = high.a
        inner = strip(value)
        if not add and inner.k == "bin" and inner.a[0] == ">>u" and const_of(inner.a[2]) is not None:
            divisor = UNSIGNED_MAGIC.get((magic, const_of(inner.a[2]), after, False))
            if divisor is not None:                            # an even divisor: the dividend was shifted first
                return self.node("bin", "/u", inner.a[1], self.node("const", divisor))
        divisor = UNSIGNED_MAGIC.get((magic, 0, after, add))
        if divisor is None:
            return None
        return self.node("bin", "/u", value, self.node("const", divisor))

    def _alike(self, a, b) -> bool:
        """The same expression read twice (two evaluations of one operand)."""
        if isinstance(a, Node) and isinstance(b, Node):
            a, b = strip(a), strip(b)
            return a.k == b.k and len(a.a) == len(b.a) and all(self._alike(x, y) for x, y in zip(a.a, b.a))
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            return len(a) == len(b) and all(self._alike(x, y) for x, y in zip(a, b))
        return a == b

    def _add(self, a: Node, b: Node) -> Node:
        for high, other in ((a, b), (b, a)):
            half = strip(other)
            if high.k == "mhu" and not high.a[2] and half.k == "bin" and half.a[0] == ">>u" \
                    and const_of(half.a[2]) == 1:
                rest = strip(half.a[1])                        # t + ((x - t) >> 1): a multiplier of 33 bits
                if rest.k == "bin" and rest.a[0] == "-" and strip(rest.a[2]) is high \
                        and self._same(rest.a[1], high.a[0]):
                    return self.node("mhu", high.a[0], high.a[1], True)
        for high, other in ((a, b), (b, a)):
            if high.k == "mh" and not high.a[2] and high.a[3] == 0 and self._same(high.a[0], other):
                return self.node("mh", high.a[0], high.a[1], True, 0)      # a multiplier of 32 bits: + x
        return self._add_plain(a, b)

    def _product_step(self, x, a: Node, b: Node, made: Node) -> Node:
        """Note a step of a multiplication by a constant that was done in the accumulator's own register."""
        if made.k == "mul":
            accumulator = x.rs if a.k == "mul" else x.rt if b.k == "mul" else None
            if accumulator is not None and x.rd == accumulator:
                self.inplace_products.add(id(made))
                self.keep.append(made)
        return made

    def _add_plain(self, a: Node, b: Node) -> Node:
        # ``z + x * -C`` keeps the negation and the addition; ``z + -(x * C)`` and ``z - x * C`` are both
        # compiled as a subtraction, so a negated product under an addition is the negative constant
        for which in (a, b):
            if which.k == "un" and which.a[0] == "-" and which.a[1].k == "mul":
                product = self.node("mul", which.a[1].a[0], -which.a[1].a[1])
                product.seq = which.seq
                if which is a:
                    a = product
                else:
                    b = product
        # a multiplication by a constant is expanded into shifts, additions and subtractions
        if a.k == "mul" and self._same(a.a[0], b):
            return self.node("mul", a.a[0], a.a[1] + 1)
        if b.k == "mul" and self._same(b.a[0], a):
            return self.node("mul", b.a[0], b.a[1] + 1)
        if a.k == "mul" and b.k == "mul" and self._same(a.a[0], b.a[0]):
            return self.node("mul", a.a[0], a.a[1] + b.a[1])
        return self.node("bin", "+", a, b)

    def _sub(self, a: Node, b: Node) -> Node:
        if b.k == "cp" and self._same(a, b) and strip(a).k not in ("const", "hi"):
            # ``x % -1``: the quotient by one is a copy of the value, which is then subtracted from it
            # (probe; ``x % 1`` folds to 0 before any code is made)
            return self.node("bin", "%", a, self.node("const", -1))
        sign = strip(b)
        if a.k == "mh" and sign.k == "bin" and sign.a[0] == ">>" and const_of(sign.a[2]) == 31 \
                and self._same(sign.a[1], a.a[0]):
            divisor = SIGNED_MAGIC.get((a.a[1], a.a[3], a.a[2]))
            if divisor is None:
                raise Decline("division by a constant not recognised")
            return self.node("bin", "/", a.a[0], self.node("const", divisor))
        quotient = strip(b.a[0]) if b.k == "mul" else None
        if quotient is not None and quotient.k == "mhu":
            quotient = self._unsigned_quotient(quotient, 0) or quotient   # a quotient that needed no shift
        if quotient is not None and quotient.k == "bin" and quotient.a[0] in ("/", "/u") \
                and const_of(quotient.a[2]) == b.a[1] and self._same(quotient.a[1], a):
            return self.node("bin", "%" + quotient.a[0][1:], a, self.node("const", b.a[1]))   # x - (x / d) * d
        return self._sub_plain(a, b)

    def _sub_plain(self, a: Node, b: Node) -> Node:
        if a.k == "mul" and self._same(a.a[0], b):
            return self.node("mul", a.a[0], a.a[1] - 1)
        if b.k == "mul" and self._same(b.a[0], a):
            return self.node("mul", b.a[0], 1 - b.a[1])
        if a.k == "mul" and b.k == "mul" and self._same(a.a[0], b.a[0]):
            return self.node("mul", a.a[0], a.a[1] - b.a[1])
        return self.node("bin", "-", a, b)

    def _shift(self, op: str, a: Node, amount: int) -> Node:
        if op == "sll":
            if a.k == "cp":
                return self.node("mul", a.a[0], 1 << amount)
            if a.k == "mul":
                return self.node("mul", a.a[0], a.a[1] << amount)
            return self.node("bin", "<<", a, self.node("const", amount))
        if op == "sra" and a.k == "bin" and a.a[0] == "<<" and const_of(a.a[2]) == amount and amount in (16, 24):
            return self.node("cast", "s16" if amount == 16 else "s8", a.a[1])
        if op == "sra" and a.k == "cp" and a.a[0].k == "bin" and a.a[0].a[0] == "-":
            # the difference of two pointers to elements of 2^k bytes: an exact division, made with a
            # ``move`` and one shift (probe: an integer ``>>`` has no ``move``, ``/`` rounds first)
            return self.node("pdiff", a.a[0].a[1], a.a[0].a[2], 1 << amount)
        if op == "sra" and a.k == "mh" and a.a[3] == 0:
            return self.node("mh", a.a[0], a.a[1], a.a[2], amount)
        if op == "sra" and amount == 1 and a.k == "bin" and a.a[0] == "+":
            # ``x / 2`` on a signed value: the sign bit is added before the shift
            for value, fix in ((a.a[1], a.a[2]), (a.a[2], a.a[1])):
                fix = strip(fix)
                if fix.k == "bin" and fix.a[0] == ">>u" and const_of(fix.a[2]) == 31:
                    sign = strip(fix.a[1])
                    if sign.k == "bin" and sign.a[0] == ">>" and const_of(sign.a[2]) == 31 \
                            and strip(sign.a[1]) is strip(value):
                        return self.node("bin", "/", value, self.node("const", 2))
        if op == "srl" and a.k == "bin" and a.a[0] == "<<" and const_of(a.a[2]) == amount and amount in (16, 24):
            return self.node("cast", "u16" if amount == 16 else "u8", a.a[1])
        if op == "srl" and a.k == "mhu":
            quotient = self._unsigned_quotient(a, amount)
            if quotient is None:
                raise Decline("unsigned division by a constant not recognised")
            return quotient
        return self.node("bin", ">>" if op == "sra" else ">>u", a, self.node("const", amount))

    # -- calls --------------------------------------------------------------------------------

    def call(self, x):
        self.now = x
        self.flush_call()
        args, proto = [], []
        slot = 0                                              # argument words taken so far
        for reg in (12, 14):                                  # leading float arguments
            n = self.fregs.get(reg)
            if n is None or n.uses or n.seq <= self.barrier or not self.is_float(n):
                break
            n.uses += 1
            args.append(n)
            proto.append(self.float_type(n))
            slot = ((slot + 1) & ~1) + 2 if proto[-1] == "f64" else slot + 1
        forwarded = False
        struct_end = 16
        if self.stack_copy:
            raise Decline("structure argument of another shape")
        if self.stack_block is not None:
            # ``f(S)`` with ``S`` a structure of more than sixteen bytes held in a global: its first four
            # words are loaded into $a0-$a3 and the rest copied to the stack from 16($sp) (measured).
            # Read before the registers are taken as separate arguments (two of them would pass for one
            # 64-bit value).
            offset, tail_size, source = self.stack_block
            source = strip(source)
            held = [self.regs.get(r) for r in (4, 5, 6, 7)]

            def word_address(value):
                value = strip(value) if value is not None else None
                if value is None or value.k != "ld" or value.a[0] != 4:
                    return None
                loc = value.a[2]
                if loc[0] == "global":
                    return loc[1]
                if loc[0] == "deref" and strip(loc[1]).k == "sym":
                    return (strip(loc[1]).a[0] + loc[2]) & 0xFFFFFFFF
                return None

            if offset != 16 or slot or args or source.k != "sym" \
                    or any(n is None or n.uses or n.seq <= self.barrier for n in held):
                raise Decline("structure argument of another shape")
            base = (source.a[0] - 16) & 0xFFFFFFFF
            if [word_address(n) for n in held] != [(base + 4 * i) & 0xFFFFFFFF for i in range(4)]:
                raise Decline("structure argument of another shape")
            for n in held:
                n.uses += 1
            size = 16 + tail_size
            self.struct_args.add(size)
            self.note_global(base, "addr")
            args.append(self.node("structarg", base, size))
            proto.append("s32")
            struct_end = 16 + tail_size
            slot = 4
            self.stack_block = None
        # integer arguments follow in the registers the float ones did not shadow (a double two)
        reg = 4 + slot
        while reg < 8:
            n = self.regs.get(reg)
            fresh = n is not None and not n.uses and n.seq > self.barrier
            if reg == 5 and not fresh and self._double_in_a2() is not None:
                reg = 6                                       # a double is aligned to the second register pair
                continue
            if not fresh:
                break
            if reg == 6 and self._double_in_a2() is not None:
                value = self._double_in_a2()
                n.uses += 1
                self.regs[7].uses += 1
                args.append(value)
                proto.append("f64")
                reg = 8
                continue
            low = self.regs.get(reg + 1) if reg in (4, 6) else None
            if low is not None and not low.uses and low.seq > self.barrier:
                wide = self._wide_value(n, low, False)        # one 64-bit integer argument in an aligned pair
                if wide is not None:
                    n.uses += 1
                    low.uses += 1
                    args.append(wide)
                    proto.append("u64")
                    reg += 2
                    continue
            n.uses += 1
            was = reg
            reg += 1
            if was == 4 and not args and strip(n).k == "sret":
                # ``return f(...);`` where ``f`` returns the same structure: the result pointer is handed on
                forwarded = True
                continue
            if n.k == "fbits":                                # prototyped ``float`` parameter in an integer register
                args.append(n.a[0])
                proto.append("f32")
                continue
            if n.k == "ld" and n.a[0] == 4 and n.a[1] is True and self._float_place(n.a[2]) == "f32":
                # a ``float`` variable passed to a prototyped ``float`` parameter in an integer register is
                # loaded straight into it (``lw $a1, x``; measured)
                args.append(self.node("ld", 4, "f", n.a[2]))
                proto.append("f32")
                continue
            args.append(n)
            # a constant or a word loaded from memory may as well be a ``float`` of a prototyped callee
            # (``lui $a2, 0x4204`` is 33.0f; decided when every call of the callee has been read)
            proto.append("?" if self._flexible(n) else "s32")
        if self.stack_args:
            expected = struct_end
            if not (reg == 8 or (reg == 7 and 16 in self.stack_wide)):
                raise Decline("stack arguments")
            for off in sorted(self.stack_args):
                wide = off in self.stack_wide
                if wide:
                    expected = (expected + 7) & ~7
                if off != expected:
                    raise Decline("stack arguments")
                expected += 8 if wide else 4
                value = self.stack_args[off]
                args.append(value)
                proto.append(self.float_type(value) if self.is_float(value) else
                             "?" if self._flexible(value) else "s32")
            self.stack_args = {}
            self.stack_wide = set()
        single = "f32" in proto
        if x.op == "jal":
            target = self.node("func", x.target)
        else:
            target = self.use(x.rs)
        call = self.node("call", target, tuple(args))
        if x.op == "jal" and proto == ["u64", "u64"] and len(args) == 2 and self._is_udivdi3(x.target):
            self.divisions.add(id(call))                      # ``a / b`` on unsigned 64-bit values
            self.wide_calls.add(id(call))
            self.keep.append(call)
        if x.op == "jal" and proto == ["u64", "u64"] and len(args) == 2 and self.read is not None:
            from .runtimehelpers import is_umoddi3
            if is_umoddi3(self.read, x.target):
                self.remainders.add(id(call))                 # ``a % b``
                self.wide_calls.add(id(call))
                self.keep.append(call)
        if forwarded:
            self.forwarded.add(id(call))
            self.keep.append(call)
        if x.op == "jal":
            self.flex_args += [(x.target, k, arg) for k, arg in enumerate(args) if k < len(proto) and proto[k] == "?"]
        if single:
            self.prototypes[id(call)] = proto
            if x.op == "jal":
                known = self.callee_protos.get(x.target)
                if known is None:
                    self.callee_protos[x.target] = list(proto)
                elif len(known) != len(proto) or any(a != b and "?" not in (a, b) for a, b in zip(known, proto)):
                    raise Decline(f"one callee, two argument lists ({' '.join(known)} / {' '.join(proto)})")
                else:
                    self.callee_protos[x.target] = [b if a == "?" else a for a, b in zip(known, proto)]
        for reg in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25):
            self.regs.pop(reg, None)
        self.fregs = {reg: n for reg, n in self.fregs.items() if reg in self.fsaved}
        self.fregs[0] = call
        self.regs[V0] = call
        self.regs[V1] = self.node("low", call)               # the second word of a 64-bit result
        self.pending_call = call

    @staticmethod
    def _flexible(n: Node) -> bool:
        return const_of(n) is not None or (n.k == "ld" and n.a[0] == 4 and n.a[1] is True
                                           and n.a[2][0] in ("global", "deref", "gidx"))

    def _float_place(self, loc):
        """The float type of a frame slot (``f32``/``f64``) when the location is one that holds a float."""
        if loc[0] != "local":
            return None
        off = loc[1]
        if off >= self.size:
            index, rest = divmod(off - self.size, 4)
            if rest or index >= len(self.params):
                return None
            kind = self.params[index].get("type")
            return kind if kind in ("f32", "f64") else None
        var = self.vars.get(off)
        if var is not None and not var.get("array") and var.get("type") in ("f32", "f64"):
            return var["type"]
        return None

    def _double_in_a2(self):
        """The ``double`` that $a2/$a3 carry to a call, if they do: moved there from a float register pair
        (``mfc1 $a3,$f0; mfc1 $a2,$f1``) or loaded a word at a time from a double variable
        (``lw $a2, y; lw $a3, y+4``; measured)."""
        high, low = self.regs.get(6), self.regs.get(7)
        if high is None or low is None or high.uses or low.uses or min(high.seq, low.seq) <= self.barrier:
            return None
        if high.k == "dbits":
            return high.a[0] if low.k == "dlow" and low.a[0] is high.a[0] else None
        if high.k == "hi" and id(low) in self.li_zero and low.seq + 1 == high.seq:
            # a double constant: ``li.d $a2, c`` is ``addiu $a3,$zero,0; lui $a2, high`` when only the top
            # sixteen bits are set
            return self.node("dconst", high.a[0] << 48)
        top, bottom = const_of(high), const_of(low)
        if top is not None and bottom is not None and low.seq < high.seq and (top & 0xFFFFFFFF) >> 16 \
                and (bottom & 0xFFFFFFFF) >> 16 and id(low) not in self.li_zero:
            # two whole words, the one for $a3 made first: a double constant (two integer arguments are
            # loaded $a2 first; measured on a title, 0.8 as ``lui/ori $a3`` then ``lui/ori $a2``)
            return self.node("dconst", (top & 0xFFFFFFFF) << 32 | (bottom & 0xFFFFFFFF))
        if not (high.k == "ld" and low.k == "ld" and high.a[0] == low.a[0] == 4):
            return None
        first, second = high.a[2], low.a[2]
        if first[0] == "local" and second == ("local", first[1] + 4) and self._float_place(first) == "f64":
            return self.node("ld", 8, "f", first)
        if first[0] == "deref" and second[0] == "deref" and second[2] == first[2] + 4 \
                and strip(first[1]) is strip(second[1]) and low.seq == high.seq + 1:
            return self.node("ld", 8, "f", first)
        return None

    # -- control flow -------------------------------------------------------------------------

    def condition(self, x):
        """The condition under which the branch ``x`` is taken, as (operator, left, right)."""
        self.now = x
        if x.addr in self.wide_tests:
            high, low, limit = self.wide_tests[x.addr]
            value = self._wide_value(self.use(high), self.use(low), stored=False)
            if value is None:
                raise Decline("64-bit comparison of a value not recognised")
            return "<=", value, self.node("const", limit)
        if x.op in ("beq", "bne"):
            a, b = self.use(x.rs), self.use(x.rt)
            taken = "==" if x.op == "beq" else "!="
            # a comparison materialised with slt and then tested against zero
            if const_of(b) == 0 and x.rt == 0 and a.k == "bin" and a.a[0] in ("<", "<u"):
                cond = (a.a[0], a.a[1], a.a[2])
                return cond if taken == "!=" else self._negated(cond)
            return (taken, a, b)
        if x.op in ("blez", "bgtz", "bltz", "bgez"):
            return ({"blez": "<=", "bgtz": ">", "bltz": "<", "bgez": ">="}[x.op], self.use(x.rs),
                    self.node("const", 0))
        if x.op in ("bc1t", "bc1f"):
            if self.fcc is None:
                raise Decline("float branch without a comparison")
            op, a, b = self.fcc
            self.fcc = None
            cond = ("f" + op, a, b)
            return cond if x.op == "bc1t" else self._negated(cond)
        raise Decline(f"branch {x.op}")

    @staticmethod
    def _negated(cond):
        op = cond[0]
        if op.startswith("!f"):                               # not(a < b) is not (a >= b) for floats
            return (op[1:], cond[1], cond[2])
        if op.startswith("f"):
            return ("!" + op, cond[1], cond[2])
        if op.endswith("u"):
            return (NEGATE[op[:-1]] + "u", cond[1], cond[2])
        return (NEGATE[op], cond[1], cond[2])

    def _block(self, start: int, end: int, loop) -> list:
        """Statements of the instructions [start, end).  ``braced``: the block is written inside braces of
        its own (a loop's body, the parts of an ``if``), unlike the function's body and a case's code."""
        saved_out = getattr(self, "out", None)
        braced, self.braces = getattr(self, "braces", False), False
        self.out = out = []
        self.list_start[id(out)] = start
        self._reset()
        loop_from = None                                     # where a loop statement was announced

        def mark(at):
            for done in reversed(out):
                if done.end is not None:
                    break
                done.end = at

        labelled = None                                      # where a loop or ``switch`` just ended: its exit label
        try:
            i, joined = start, None
            while i < end:
                mark(i)
                if i in self.targets and i != joined:
                    self.flush_call()
                    self._reset()
                    out.append(Stmt("label", i))
                    out[-1].end = i                           # a label holds no instruction
                closing = self._loop_at(i, end)
                if closing is not None and not (loop is not None and loop[0] == i):
                    self.flush_call()
                    keep = self._snapshot()
                    try:
                        body = self._braced(i, closing, (i, closing))
                        if len(self._back_jumps(i, end)) > 1 and not self._test_first(body):
                            # Several jumps back to one place.  A loop has one, its closing jump; ``continue``
                            # goes there too only in a ``while``, whose test stands at the top.  Without a
                            # test this is a label of the source and its ``goto``s.
                            raise Decline("jumps back to a label")
                    except Decline:
                        # not a loop: the jumps back are ``goto``s to a label standing here
                        self._restore(keep)
                        self.not_loops.add(i)
                        closing = None
                if closing is not None and not (loop is not None and loop[0] == i):
                    init = None
                    if loop_from is not None:
                        # the statements since the announcement are the loop's initialiser
                        chunk, loop_from = out[loop_from:], None
                        del out[len(out) - len(chunk):]
                        out.extend(part for part in chunk if part.k == "label")   # the loop's own top label
                        init = [part for part in chunk if part.k != "label"]
                        if any(part.k not in ("assign", "expr") for part in init):
                            raise Decline("loop initialiser")
                    out.append(Stmt("loop", body, i, closing, init))
                    self._reset()
                    i = labelled = closing + 1
                    continue
                if loop_from is not None and i > 0 and self.ins[i - 1].op == "nop" and len(out) == loop_from + 1 \
                        and out[-1].k == "label" and out[-1].a[0] == i:
                    # The no-op announced no loop: a label of the source stands here, directly after a
                    # label of the compiler's (the end of an ``if``, a loop or a ``switch``, the top of a
                    # loop's body), and GCC separates the two with a no-op.
                    self.label_used.add(i)
                    self.hard_labels.add(i)
                    loop_from = None
                x = self.ins[i]
                if i in self.chains:
                    i = self._chain_statement(i)
                    self._reset()
                    continue
                if i in self.tables:
                    i = labelled = self._switch_table(i, end)
                    self._reset()
                    continue
                if i in self.unsigned_fix:
                    i = self._unsigned_fix(i)
                    continue
                if x.op == "bgez" and i + 2 < end and self.index_of.get(x.target) == i + 2:
                    fix, shift = self.ins[i + 1], self.ins[i + 2]
                    if fix.op == "addiu" and fix.rs == fix.rt == x.rs and shift.op == "sra" and shift.rt == x.rs \
                            and shift.sa > 1 and fix.imm == (1 << shift.sa) - 1 \
                            and self.targets.get(i + 2) == [i]:
                        value = self.use(x.rs)
                        self.setreg(shift.rd, self.node("bin", "/", value, self.node("const", 1 << shift.sa)))
                        i += 3
                        continue
                if x.op == "lwl":
                    loose = self._unaligned_run(i, end)
                    if loose is not None:
                        i = self._unaligned_copy(loose)
                        continue
                run = self._copy_run(i, end) if x.op == "lw" else None
                if run is not None:
                    i = self._block_copy(i, run)
                    if self.out and self.out[-1].k == "return":
                        self._reset()
                    continue
                if x.op in BRANCHES:
                    shape = self._conditional_shape(i, end)
                    if shape is not None and shape[2] == "i" and shape[3] in self.regvars:
                        shape = None            # both arms assign a register variable: statements, not a value
                    if shape is not None and self._join_reads(shape):
                        i = self._conditional(i, shape)
                        joined = i
                        continue
                if x.op == "beq" or (x.op in ("slti", "sltiu") and x.rs in self.regs and i + 1 < end
                                     and self.ins[i + 1].op in ("beq", "bne") and self.ins[i + 1].rs == x.rt):
                    # a dispatch tree opens with an equality test, or with a range test when it is large
                    after = self._switch(i, end)
                    if after is not None:
                        i = labelled = after
                        self._reset()
                        continue
                if x.op in BRANCHES:
                    i = self._if(i, end, loop)
                    self._reset()
                elif x.op == "j":
                    target = self.index_of[x.target]
                    call = self.pending_call
                    if call is not None and id(call) in self.forwarded:
                        if call.uses or target != len(self.ins):
                            raise Decline("result pointer handed to a call that is not returned")
                        self.pending_call = None
                        call.uses += 1
                        self.returned_calls = True
                        self.emit(Stmt("return", call))
                        self._reset()
                        i += 1
                        continue
                    self.flush_call()
                    if self._never_block(i, target, end, loop):
                        # ``if (0) { ... }``: a jump over code nothing else reaches.  A ``goto`` to a label
                        # behind the block compiles to the same jump, but a user label that follows another
                        # label (the block's own last one) gets a no-op in front of it; this has none (probe).
                        body = self._block_inner(i + 1, target, loop)
                        self.emit(Stmt("if", ("never",), body, None))
                        self._reset()
                        i = target
                        continue
                    self._jump(i, target, end, loop)
                    self._reset()
                    i += 1
                elif x.op in ("jal", "jalr"):
                    self.call(x)
                    i += 1
                elif x.op == "nop":
                    if self._hazard_nop(i):
                        i += 1                                # the delay a coprocessor move or compare needs
                        continue
                    if braced and i == start:
                        # The opening brace of a block leaves a note behind the label that stands there
                        # (the top of a loop, the else label), so nothing written first inside braces
                        # gets a no-op (measured: ``for (;;) { L: ...``, ``else { L: ...``).
                        raise Decline("no-op at the start of a block")
                    if loop_from is None and (i == 0 or i in self.targets or i == labelled):
                        # Without optimisation the compiler puts a no-op where a ``for``/``while``/
                        # ``do`` statement begins directly after a label or at the very start of
                        # the function.  What follows up to the loop is the ``for`` initialiser.
                        self.flush_call()
                        loop_from = len(out)
                        i += 1
                        continue
                    raise Decline("explicit nop")
                else:
                    self.step(x)
                    i += 1
            self.flush_call()
            mark(max(i, end))
            if loop_from is not None:
                raise Decline("explicit nop")
        finally:
            self.out = saved_out
        return out

    def _never_block(self, i: int, target: int, end: int, loop) -> bool:
        """Whether the forward jump at ``i`` skips a block that only it guards and whose end label is shared
        with a label of the block itself (the case where ``goto`` and ``if (0)`` differ)."""
        if not (i + 1 < target <= end and target < len(self.ins)) or i + 1 in self.targets:
            return False
        if loop is not None and target in (loop[1], loop[1] + 1):
            return False
        if self.ins[target - 1].op == "nop":                  # an explicit no-op: the label was a user label
            return False
        sources = self.targets.get(target, [])
        if len(sources) < 2 or not all(i <= j < target for j in sources):
            return False
        for place in range(i + 2, target):
            if any(not (i < j < target) for j in self.targets.get(place, ())):
                return False
        return True

    def _hazard_nop(self, i: int) -> bool:
        """A no-op the assembler put in, not the compiler: the delay after a coprocessor move or compare,
        or the distance HI/LO need (a multiplication or division may not start within two instructions of
        ``mflo``/``mfhi``: ``mflo; lw; nop; mult`` is ``a * b * c`` with ``c`` in memory)."""
        ins = self.ins
        if i > 0 and ins[i - 1].op in ("mtc1", "mfc1", "c.eq", "c.lt", "c.le"):
            return True
        after = i + 2 if i + 1 < len(ins) and ins[i + 1].op == "nop" and ins[i - 1].op in ("mflo", "mfhi") else i + 1
        return after < len(ins) and ins[after].op in ("mult", "multu", "div", "divu") and ins[after].fmt is None \
            and any(i - back >= 0 and ins[i - back].op in ("mflo", "mfhi") for back in (1, 2))

    # -- conditional expressions ----------------------------------------------------------------

    def _power_division(self, i: int, end: int):
        """The ``sra`` of ``x / 2^k`` on a signed value when ``i`` is its rounding branch (``bgez x; addiu x,
        x, 2^k - 1; sra``), else None.  The branch belongs to the expression, wherever the expression stands."""
        x = self.ins[i]
        if x.op != "bgez" or i + 2 >= end or self.index_of.get(x.target) != i + 2:
            return None
        fix, shift = self.ins[i + 1], self.ins[i + 2]
        if fix.op == "addiu" and fix.rs == fix.rt == x.rs and shift.op == "sra" and shift.rt == x.rs \
                and shift.sa > 1 and fix.imm == (1 << shift.sa) - 1 and self.targets.get(i + 2) == [i]:
            return shift
        return None

    def _conditional_shape(self, i: int, end: int, tails: tuple = ()):
        """``c ? a : b`` as a value: a branch over the first arm, the arm (no store, no statement), a jump to
        the join, the second arm, and both arms leave their value in one register (measured: ``jumpifnot c``,
        ``a``, ``j``, ``b``).  Returns (else index, join index, kind, register) or None."""
        x = self.ins[i]
        target = self.index_of.get(x.target) if x.op in BRANCHES else None
        if target is None or target < i + 3 or target >= end or self.ins[target - 1].op != "j":
            return None
        after = self.index_of.get(self.ins[target - 1].target)
        if after is None or not (target < after <= end) or after >= len(self.ins):
            return None
        if self.targets.get(target) != [i] or target - 1 not in (self.targets.get(after) or ()):
            return None
        # A conditional that is the last thing in the second arm of another has no join of its own: its jump
        # goes to the enclosing join (``a ? x : (b ? y : z)``; measured on a title's clamp).  So the join may
        # also be reached by the jumps of the conditionals this one ends (``tails``, only when it ends where
        # they do) and by those of a conditional that ends its own second arm (checked by reading that arm).
        for other in self.targets.get(after) or ():
            if other != target - 1 and not (target < other < after) and not (other in tails and after == end):
                return None
        inherited = tails if after == end else ()
        first, second = self._arm_shape(i + 1, target - 1), self._arm_shape(target, after, inherited + (target - 1,))
        if first is None or second is None:
            return None
        if first[0] == "call" and second[0] == "call":
            return target, after, "call", None
        if first[0] == "call":
            first = second
        elif second[0] == "call":
            second = first
        if first != second or first not in (("i", V0), ("f", 0)) and "call" in (first[0], second[0]):
            return None
        return target, after, first[0], first[1]

    def _arm_shape(self, start: int, stop: int, tails: tuple = ()):
        """(kind, register) the instructions [start, stop) leave a value in, when they are one expression."""
        k, result = start, None
        while k < stop:
            x = self.ins[k]
            if (k in self.targets and k != start) or k in self.tables:
                return None
            if k in self.unsigned_fix:
                if k + 5 > stop:
                    return None
                result, k = ("f", self.ins[k + 4].fd), k + 5
                continue
            if x.op in BRANCHES and self._power_division(k, stop) is not None:
                result, k = ("i", self._power_division(k, stop).rd), k + 3
                continue
            if x.op in BRANCHES:
                inner = self._conditional_shape(k, stop, tails)
                if inner is None or inner[2] == "call":
                    return None
                result, k = (inner[2], inner[3]), inner[1]
                continue
            if x.op in ("j", "jr", "break"):
                return None
            if x.op in ("jal", "jalr"):
                result = ("call", None)
            elif x.op in STORES or x.op in ("swc1", "sdc1"):
                if x.rs != SP:
                    return None
            elif x.op != "nop":
                written, fwritten = reads_writes(x)[1], float_reads_writes(x)[1]
                if fwritten:
                    reg = fwritten[-1]
                    result = ("f", reg - 1 if x.addr in self.pair_first else reg)
                elif written:
                    result = ("i", written[-1])
            k += 1
        return result

    def _join_reads(self, shape) -> bool:
        """Whether the code at the join reads the register the two arms wrote before anything writes it."""
        _target, after, kind, reg = shape
        candidates = [("i", V0), ("f", 0)] if kind == "call" else [(kind, reg)]
        for x in self.ins[after:after + 12]:
            reads, writes = reads_writes(x)
            freads, fwrites = float_reads_writes(x)
            if x.addr in self.pair_first and x.op in ("swc1", "mfc1"):
                freads = (x.ft - 1,) if x.op == "swc1" else (x.fs,)
            for what, number in candidates:
                if number in (reads if what == "i" else freads):
                    return True
            if x.op in ("jal", "jalr"):
                return any(number in ((4, 5, 6, 7) if what == "i" else (12, 14)) for what, number in candidates)
            if x.op in BRANCHES or x.op in ("j", "jr"):
                return False
            candidates = [(what, number) for what, number in candidates
                          if number not in (writes if what == "i" else fwrites)]
            if not candidates:
                return False
        return False

    def _conditional(self, i: int, shape, tails: tuple = ()) -> int:
        """Execute the conditional expression at ``i``; its value is in the register after the join."""
        target, after, kind, reg = shape
        cond = self._negated(self.condition(self.ins[i]))
        self.flush_call()
        base = (dict(self.regs), dict(self.fregs), self.hilo, dict(self.stack_args), set(self.stack_wide))
        mark, arms, states = len(self.out), [], []
        for start, stop, ends in ((i + 1, target - 1, ()), (target, after, tails + (target - 1,))):
            self.regs, self.fregs, self.hilo = dict(base[0]), dict(base[1]), base[2]
            self.stack_args, self.stack_wide, self.fcc = dict(base[3]), set(base[4]), None
            k = start
            while k < stop:
                x = self.ins[k]
                divided = self._power_division(k, stop) if x.op in BRANCHES else None
                inner = self._conditional_shape(k, stop, ends) if x.op in BRANCHES and divided is None else None
                if k in self.unsigned_fix:
                    k = self._unsigned_fix(k)
                elif divided is not None:
                    self.setreg(divided.rd, self.node("bin", "/", self.use(x.rs), self.node("const", 1 << divided.sa)))
                    k += 3
                elif inner is not None:
                    k = self._conditional(k, inner, ends)
                elif x.op in ("jal", "jalr"):
                    self.call(x)
                    k += 1
                elif x.op == "nop":
                    k += 1
                else:
                    self.step(x)
                    k += 1
            if len(self.out) != mark or self.stack_args:
                raise Decline("statement inside a conditional expression")
            arms.append((self.regs.get(V0), self.fregs.get(0), self.pending_call))
            states.append((self.regs, self.fregs))
            self.pending_call = None
        # which register: the one both arms wrote; two calls leave it to the reader at the join
        values = []
        for (regs, fregs), (v0, f0, call) in zip(states, arms):
            if kind == "call":
                values.append(call)
            else:
                values.append((regs if kind == "i" else fregs).get(reg))
        if any(v is None or v.uses or v.seq <= self.barrier for v in values):
            raise Decline("conditional expression without a value in each arm")
        for value in values:
            value.uses += 1
        merged = self.node("cond", cond, values[0], values[1])
        self.regs = {r: n for r, n in base[0].items() if all(state[0].get(r) is n for state in states)}
        self.fregs = {r: n for r, n in base[1].items() if all(state[1].get(r) is n for state in states)}
        self.hilo, self.stack_args, self.stack_wide, self.fcc = base[2], dict(base[3]), set(base[4]), None
        if kind == "call":
            self.regs[V0] = self.fregs[0] = merged
        elif kind == "i":
            self.regs[reg] = merged
        else:
            self.fregs[reg] = merged
        return after

    def _unsigned_fix(self, i: int) -> int:
        x = self.ins[i]
        value, test = self.fregs.get(self.ins[i + 4].fd), self.regs.get(x.rs)
        source = strip(value.a[1]) if value is not None and value.k == "cvtf" else None
        if source is None or value.a[0] != "f64" or value.uses or source.k != "ld" or test is None \
                or test.k != "ld" or test.uses or test.a[2] != source.a[2] or test.a[0] != 4:
            raise Decline("unsigned conversion not recognised")
        test.uses += 1
        self.fregs[self.ins[i + 4].fd] = self.node("cvtf", "f64", self.node("cast", "u32", value.a[1]))
        return i + 5

    def _copy_run(self, i: int, end: int):
        """A structure copied whole: ``n`` words loaded from one address into ``n`` registers, then stored
        in the same order at another (GCC's inline block move).  Returns (n, source reg, destination reg)."""
        ins = self.ins
        first = ins[i]
        if first.op != "lw" or first.imm != 0 or first.rs in (FP, SP, 0):
            return None
        source, regs, k = first.rs, [], i
        while k < end and ins[k].op == "lw" and ins[k].rs == source and ins[k].imm == 4 * len(regs) \
                and ins[k].rt not in regs and ins[k].rt != source:
            regs.append(ins[k].rt)
            k += 1
        count = len(regs)
        if count < 2 or k + count > end:
            return None
        dest = ins[k].rs
        if dest in (FP, SP, 0, source) or dest in regs:
            return None
        for j in range(count):
            x = ins[k + j]
            if not (x.op == "sw" and x.rs == dest and x.imm == 4 * j and x.rt == regs[j]):
                return None
        return count, source, dest

    def _unaligned_run(self, i: int, end: int):
        """A structure without word alignment copied whole (``struct { s16 w[3]; }``, ``struct { u8 b[7]; }``):
        each full word with ``lwl``/``lwr``, the rest with one ``lh`` or with ``lb``s, then the stores in the
        same order (measured).  Returns (bytes, "u" | "b", source reg, destination reg, registers, index after)."""
        ins = self.ins
        source, regs, k, off, kind = ins[i].rs, [], i, 0, "u"
        if source in (FP, SP, 0):
            return None
        while k + 1 < end and ins[k].op == "lwl" and ins[k + 1].op == "lwr" and ins[k].rs == ins[k + 1].rs == source \
                and ins[k].imm == off and ins[k + 1].imm == off + 3 and ins[k].rt == ins[k + 1].rt:
            regs.append((ins[k].rt, 4))
            k, off = k + 2, off + 4
        if not regs:
            return None
        if k < end and ins[k].op == "lh" and ins[k].rs == source and ins[k].imm == off:
            regs.append((ins[k].rt, 2))
            k, off = k + 1, off + 2
        else:
            while k < end and ins[k].op == "lb" and ins[k].rs == source and ins[k].imm == off:
                regs.append((ins[k].rt, 1))
                k, off, kind = k + 1, off + 1, "b"
        names = [reg for reg, _width in regs]
        if k >= end or len(set(names)) != len(names) or source in names:
            return None
        dest, at = ins[k].rs, 0
        if dest in (FP, SP, 0, source) or dest in names:
            return None
        for reg, width in regs:
            need = (("swl", at), ("swr", at + 3)) if width == 4 else ((("sh" if width == 2 else "sb"), at),)
            for op, where in need:
                if k >= end or not (ins[k].op == op and ins[k].rs == dest and ins[k].imm == where and ins[k].rt == reg):
                    return None
                k += 1
            at += width
        return off, kind, source, dest, names, k

    def _unaligned_copy(self, run) -> int:
        size, kind, source, dest, names, after = run
        src, dst = self.use(source), self.use(dest)
        self.flush_call()
        self.loose_blocks.add((kind, size))
        for reg in names:
            self.regs.pop(reg, None)
        self.emit(Stmt("copy", dst, src, size, kind))
        return after

    def _block_copy(self, i: int, run) -> int:
        """Execute the copy run at ``i``; returns the index after it."""
        count, source, dest = run
        size = 4 * count
        src, dst = self.use(source), self.use(dest)
        self.flush_call()
        self.block_sizes.add(size)
        after = i + 2 * count
        for x in self.ins[i:i + count]:
            self.regs.pop(x.rt, None)
        if strip(dst).k == "spad":
            # the tail of a structure passed by value, copied into the outgoing argument area
            self.stack_block = (strip(dst).a[0], size, src)
            return after
        if strip(dst).k == "sret":
            # ``return r;`` in a function that returns a structure: the copy, then the jump to the exit
            if after >= len(self.ins) or self.ins[after].op != "j" or \
                    self.index_of.get(self.ins[after].target) != len(self.ins):
                raise Decline("structure stored through the result pointer without a return")
            kind = f"struct n64dk_r{size}"
            if self.returns_value and self.return_type != kind:
                raise Decline("two return types")
            self.returns_value, self.return_type = True, kind
            self.emit(Stmt("return", self.node("blk", size, src)))
            return after + 1
        self.emit(Stmt("copy", dst, src, size))
        return after

    def block_operand(self, addr: Node, size: int, kind: str = "r") -> str:
        """A whole structure as an operand: the structure local itself, or what a pointer points at."""
        inner = strip(addr)
        if kind != "r":
            if inner.k in ("lad", "sret"):
                raise Decline("unaligned structure copy of a local")
            if inner.k == "sym":
                return f"(*(struct n64dk_{kind}{size} *)&{self.symbol(inner.a[0])})"
            return f"(*(struct n64dk_{kind}{size} *){self.expr(addr)})"
        if inner.k == "lad":
            var = self.vars.get(inner.a[0])
            if var is not None and var.get("struct") == size:
                return self.var_name(var)
            raise Decline("structure copy of a local that is not that structure")
        if inner.k == "sret":
            raise Decline("result pointer used as a value")
        return f"(*(struct n64dk_r{size} *){self.expr(addr)})"

    def _braced(self, start, end, loop):
        self.braces = True
        return self._block_inner(start, end, loop)

    def _block_inner(self, start, end, loop):
        regs = (self.regs, self.fregs, self.hilo, self.stack_args, self.pending_call)
        body = self._block(start, end, loop)
        self.regs, self.fregs, self.hilo, self.stack_args, self.pending_call = regs
        return body

    def _back_jumps(self, i: int, end: int) -> list[int]:
        return [j for j in self.targets.get(i, ()) if i <= j < end and self.ins[j].op == "j"]

    def _loop_at(self, i: int, end: int):
        """The index of the closing backward jump of a loop that starts at ``i``."""
        if i in self.not_loops:
            return None
        back = self._back_jumps(i, end)
        return max(back) if back else None

    @staticmethod
    def _test_first(body: list) -> bool:
        """Whether a loop's body opens with its exit test (``while (c)``, ``for (;c;)``)."""
        first = next((stmt for stmt in body if stmt.k != "label"), None)
        return first is not None and first.k == "if" and first.a[2] is None and len(first.a[1]) == 1 \
            and first.a[1][0].k == "break"

    def _snapshot(self):
        return (set(self.label_used), self.returns_value, self.return_type, self.barrier, set(self.hard_labels),
                set(self.not_loops))

    def _restore(self, keep) -> None:
        self.label_used, self.returns_value, self.return_type, self.barrier = set(keep[0]), keep[1], keep[2], keep[3]
        self.hard_labels, self.not_loops = set(keep[4]), set(keep[5])

    def _switch_ends(self, k: int, places: list, end: int) -> list[int]:
        """Where a ``switch`` whose dispatch ends at ``k`` may end, most likely first.  ``break`` is a jump to
        the end, so the end is one of the places the cases jump forward to; a ``goto`` out of the switch or
        a ``continue`` of the loop around it is another.  A no-op in front of such a place says a label of
        the source stands there behind a label of the compiler's: that one is a ``goto``, and the compiler's
        label (the no-op) can be the end."""
        ins, found = self.ins, []
        for j in range(k + 1, places[-1]):
            if ins[j].op == "j":
                target = self.index_of.get(ins[j].target)
                if target is not None and places[-1] < target <= end and target != len(ins):
                    if ins[target - 1].op == "nop" and not self._hazard_nop(target - 1) and target - 1 >= places[-1]:
                        target -= 1
                    if target not in found:
                        found.append(target)
        found.sort(reverse=True)
        if places[-1] not in found:
            found.append(places[-1])
        return found

    def _switch_sections(self, i: int, k: int, tests: list, default: int, places: list, end: int):
        """(sections, end) of the ``switch`` whose dispatch is [i, k]: the first end that reads without a
        contradiction.  A ``goto`` to the end itself is one: a label of the source there would stand behind
        the switch's own exit label, with a no-op between them."""
        first = None
        for last in self._switch_ends(k, places, end):
            keep = self._snapshot()
            try:
                bounds = [place for place in places if place < last] + [last]
                sections = []
                for lo, nxt in zip(bounds, bounds[1:]):
                    labels = [value for value, place in tests if place == lo] + ([None] if default == lo else [])
                    sections.append((labels, self._block_inner(lo, nxt, (i, last - 1))))
                if any(place == last for _value, place in tests):
                    sections.append(([value for value, place in tests if place == last], []))
                if last in self.label_used:
                    raise Decline("goto to the end of a switch")
            except Decline as why:
                self._restore(keep)
                first = first or why
                continue
            return sections, last
        raise first

    def _jump(self, i: int, target: int, end: int, loop):
        value = None
        if target == len(self.ins):
            n = self.regs.get(V0)
            f = self.fregs.get(0)
            if f is not None and f.uses == 0 and f.seq > self.barrier and f.k != "call" and self.is_float(f) \
                    and (n is None or n.uses or n.seq < f.seq):
                f.uses += 1
                value = f
                kind = self.float_type(f)
                if self.returns_value and self.return_type != kind:
                    raise Decline("two return types")
                self.returns_value, self.return_type = True, kind
            elif n is not None and n.uses == 0 and n.seq > self.barrier and n.k != "call":
                if self.returns_value and self.return_type != "s32":
                    raise Decline("two return types")
                n.uses += 1
                value = n
                self.returns_value = True
            self.emit(Stmt("return", value))
            return
        if loop is not None and target == loop[1] + 1:
            self.emit(Stmt("break"))
            return
        if loop is not None and target == loop[1]:
            self.emit(Stmt("continue"))
            return
        self.label_used.add(target)
        self.emit(Stmt("goto", target))

    def _dispatch(self, i: int, end: int, consts: dict):
        """The decision tree of a small ``switch`` that starts at ``i``, read without executing anything:
        equality tests that lead to the cases, range tests (``slti``) that lead to another part of the
        tree or, for a run of consecutive values with one body (``case 1: case 2:``), to that body, and
        jumps to the default.  It ends where the first case's code begins.  ``consts`` are the constants
        already in registers.  Returns its parts, or None."""
        ins = self.ins
        selector = ins[i].rs
        if selector == 0:
            return None
        # how far the tree can reach: up to the first case an equality test names, or the first
        # instruction that is no part of a tree
        limit, k = end, i
        while k < limit:
            x = ins[k]
            if x.op == "beq" and x.rs == selector:
                target = self.index_of.get(x.target)
                if target is None or target <= k:
                    return None
                limit = min(limit, target)
            elif not ((x.op == "addiu" and x.rs == 0 and x.rt not in (0, selector)) or x.op == "j"
                      or (x.op in ("slti", "sltiu") and x.rs == selector and x.rt not in (0, selector))
                      or (x.op in ("beq", "bne") and x.rt == 0 and x.rs != selector)
                      or (x.op in SIGN_TESTS and x.rs == selector and k > i)):
                break
            k += 1
        # the tree ends with a jump; the longest reading that holds together is the tree
        for stop in range(min(limit, k), i, -1):
            if ins[stop - 1].op == "j":
                found = self._tree(i, stop, end, consts)
                if found is not None:
                    return found
        return None

    def _tree(self, i: int, stop: int, end: int, consts: dict):
        """The decision tree in [i, stop): every path through it with the selector's values on that path."""
        ins = self.ins
        selector = ins[i].rs
        consts, ranges = dict(consts), {}
        unsigned_tests = False
        low, high = None, None                               # the selector's values on the path being read
        arrivals, pending, jumps = [], {}, set()             # (place, low, high); inner label -> its values
        single = set()                                       # values an equality test takes out of every range
        reachable = True

        def narrow(bounds, lo, hi):
            a, b = bounds
            return (lo if a is None or (lo is not None and lo > a) else a,
                    hi if b is None or (hi is not None and hi < b) else b)

        for k in range(i, stop):
            x = ins[k]
            if k in pending:
                if reachable:
                    return None
                low, high = pending.pop(k)
                reachable = True
            elif not reachable:
                return None
            if x.op == "addiu" and x.rs == 0 and x.rt not in (0, selector):
                consts[x.rt] = x.imm
                ranges.pop(x.rt, None)
                continue
            if x.op in ("slti", "sltiu") and x.rs == selector and x.rt not in (0, selector):
                unsigned_tests = unsigned_tests or x.op == "sltiu"
                ranges[x.rt] = x.imm
                consts.pop(x.rt, None)
                continue
            target = self.index_of.get(x.target) if x.op in BRANCHES or x.op == "j" else None
            if target is None or target <= k:
                return None
            if x.op == "j":
                if target < stop:
                    return None
                arrivals.append((target, low, high))
                jumps.add(target)
                reachable = False
                continue
            if x.op == "beq" and x.rs == selector and (x.rt == 0 or consts.get(x.rt) is not None):
                value = 0 if x.rt == 0 else consts[x.rt]
                taken, other = (value, value), (None, None)   # the other values stay as they are
                single.add(value)
            elif x.op in ("beq", "bne") and x.rt == 0 and x.rs in ranges:
                bound = ranges[x.rs]                          # the register holds ``selector < bound``
                below, above = (None, bound - 1), (bound, None)
                taken, other = (below, above) if x.op == "bne" else (above, below)
            elif x.op in SIGN_TESTS and x.rs == selector and k > i:
                # the tree splits at a case value of 0: the selector's sign is tested directly
                taken, other = SIGN_TESTS[x.op]
            else:
                return None
            reach = narrow((low, high), *taken)
            if target < stop:
                if target in pending:
                    return None
                pending[target] = reach                       # another part of the tree
            else:
                arrivals.append((target, *reach))
            low, high = narrow((low, high), *other)
        # the default is where the values outside every case arrive; a jump also leads to a case whose
        # values the tests before it have already pinned down
        if unsigned_tests:                                   # an unsigned selector below 1 is 0
            arrivals = [(place, 0 if lo is None else lo, hi) for place, lo, hi in arrivals]
        open_ended = {place for place, lo, hi in arrivals if lo is None or hi is None}
        if pending or reachable or len(open_ended) != 1 or not open_ended <= jumps:
            return None
        default = open_ended.pop()
        tests = []
        for place, lo, hi in arrivals:
            if place == default:
                continue
            if lo is None or hi is None or hi < lo or hi - lo > 255:
                return None
            tests += [(value, place) for value in range(lo, hi + 1) if lo == hi or value not in single]
        if not tests:
            return None
        if len(tests) == 1 and not (i + 1 < end and ins[i + 1].op == "j"):
            return None
        places = sorted({target for _value, target in tests} | {default})
        if places[0] != stop or places[-1] > end or len({v for v, _t in tests}) != len(tests):
            return None
        return {"selector": selector, "tests": tests, "default": default, "places": places, "jump": stop - 1,
                "unsigned": unsigned_tests}

    def _switch(self, i: int, end: int):
        """A small ``switch``: the selector stays in one register while it is compared with each
        case value in turn (``beq`` to the case's code), a jump goes to the default or past the
        end, and the bodies follow in source order with ``break`` as a jump to the end.  Returns
        the index after the statement, or None when the instructions at ``i`` are not this."""
        ins, first = self.ins, self.ins[i]
        known = self.regs.get(first.rt)
        tree = self._dispatch(i, end, {first.rt: const_of(known)} if first.rt and known is not None else {})
        if tree is None:
            return None
        selector, tests, default, places, k = (tree[key] for key in ("selector", "tests", "default", "places", "jump"))
        unsigned_tests = tree["unsigned"]
        held = self.regs.get(selector)
        if held is not None and held.k == "bin" and held.a[0] in ("<", "<u"):
            # the "selector" is a comparison's result: ``if (x < N) return;`` (probe: the same test and jump;
            # a ``switch`` on the comparison loads its operand into the other register)
            return None
        chosen = self.use(selector)                          # before the flush: ``switch (f())`` reads the call
        self.flush_call()
        if unsigned_tests:                                    # range tests with sltiu: an unsigned selector
            chosen = self.node("cast", "u32", chosen)
        elif self._unsigned_typed(chosen) and any(ins[j].op == "slti" for j in range(i, k)):
            chosen = self.node("cast", "s32", chosen)         # signed range tests on a value written unsigned
        sections, last = self._switch_sections(i, k, tests, default, places, end)
        self.out.append(Stmt("switch", chosen, sections))
        return last

    def _switch_table(self, i: int, end: int) -> int:
        """A ``switch`` compiled to a jump table: the selector (less the lowest case) indexes the table.
        The bodies follow the dispatch in source order, as for the compared form above."""
        table, ins = self.tables[i], self.ins
        chosen = self.use(table["reg"])
        inner = strip(chosen)
        lowest, selector = 0, chosen
        if table.get("again") is not None:
            # lowest case 0: the selector is evaluated again for the index, and must read the same
            for at in range(*table["again"]):
                self.step(ins[at])
            if not self._alike(self.use(table["index"]), chosen):
                raise Decline("jump table index is not its selector")
        elif inner.k == "bin" and inner.a[0] == "+" and const_of(inner.a[2]) is not None:
            selector, lowest = inner.a[1], -const_of(inner.a[2])
            selector.uses += 1
        self.flush_call()
        default, k = table["default"], table["jr"]
        tests = [(lowest + number, place) for number, place in enumerate(table["cases"]) if place != default]
        places = sorted({place for _value, place in tests} | {default})
        if not tests or places[0] != k + 1 or places[-1] > end or k >= end:
            raise Decline("jump table layout")
        sections, last = self._switch_sections(i, k, tests, default, places, end)
        self.out.append(Stmt("switch", selector, sections))
        return last

    def _chain(self, i: int, end: int, loop, avoid=()):
        """A condition made of several tests (``&&``, ``||``): a run of pure computation and branches that
        leaves by two exits.  Without optimisation GCC emits one conditional branch per test, to the label
        the whole condition's truth or falsehood leads to, and an explicit jump where both labels were
        given.  Two shapes: a loop's exit test (``while (a && b)``: the false exit is the jump out of the
        loop, the true exit the instruction after it) and an ``if`` (the true exit is the first instruction
        of the body, the false exit the end of it).  Returns (shape, tree, items, T, F) or None; nothing
        is executed here."""
        ins, items, k = self.ins, [], i
        found = None
        unread_call = False              # a call whose value no later instruction of the test has read yet
        while k < min(end, i + 96):
            x = ins[k]
            if x.op in BRANCHES or x.op == "j":
                target = self.index_of.get(x.target)
                if target is None or target <= k:
                    return None
                if x.op == "j" and not (items and items[-1][0] == k - 1):
                    return None        # a jump in a chain is the second half of a test (branch, then jump)
                if unread_call and V0 not in reads_writes(x)[0]:
                    return None        # a call the test does not read is a statement: not one condition
                unread_call = False
                items.append((k, "br" if x.op in BRANCHES else "j", target))
            elif x.op in STORES or x.op in ("swc1", "sdc1", "jr", "break", "jalr") or k in self.tables:
                return None
            elif x.op == "jal":
                if unread_call:
                    return None
                unread_call = True
            elif unread_call and V0 in reads_writes(x)[0]:
                unread_call = False
            k += 1
            if len(items) >= 2 and items[-1][0] == k - 1 and any(t == k for _at, _kind, t in items) and k < end \
                    and k not in avoid:
                found = k
                break
        if found is None or sum(kind == "br" for _at, kind, _t in items) < 2:
            return None
        # a test that something outside the condition jumps to is the top of a loop, not a further operand
        # (``if (a) { while (b) ... }``: the back jump lands on the second test)
        if any(not i <= source < found for k in range(i + 1, found + 1) for source in self.targets.get(k, ())):
            return None
        here = ins[found]
        if loop is not None and here.op == "j" and self.index_of.get(here.target) == loop[1] + 1:
            shape, true_exit, false_exit = "loop", found + 1, found
        else:
            outside = {t for _at, _kind, t in items if t > found}
            if len(outside) != 1:
                return None
            shape, true_exit, false_exit = "if", found, outside.pop()
            if false_exit > end:
                return None
        if any(t not in (true_exit, false_exit) and not (i < t < found) for _at, _kind, t in items):
            return None
        memo: dict[int, object] = {}

        def value(at: int):
            if at == true_exit:
                return True
            if at == false_exit:
                return False
            if at in memo:
                return memo[at]
            item = next((it for it in items if it[0] >= at), None)
            if item is None:
                raise Decline("condition chain runs off its end")
            where, kind, target = item
            if kind == "j":
                result = value(target)
            else:
                taken, other = value(target), value(where + 1)
                yes, no = ("leaf", where), ("nleaf", where)
                if taken is True:
                    result = yes if other is False else (None if other is True else ("||", yes, other))
                elif taken is False:
                    result = no if other is True else (None if other is False else ("&&", no, other))
                elif other is True:
                    result = ("||", no, taken)       # ``bne false; j true``: the last test of ``(.. && c) || rest``
                elif other is False:
                    result = ("&&", yes, taken)      # ``beq more; j false``: the last test of ``(.. || c) && rest``
                elif isinstance(other, tuple) and other[0] == "||" and other[2] is taken:
                    result = ("||", ("&&", no, other[1]), taken)       # (a && b) || rest
                elif isinstance(other, tuple) and other[0] == "&&" and other[2] is taken:
                    result = ("&&", ("||", yes, other[1]), taken)      # (a || b) && rest
                else:
                    result = None
                if result is None:
                    raise Decline("condition chain is not && / ||")
            memo[at] = result
            return result

        try:
            tree = value(i)
        except Decline:
            return None
        if not isinstance(tree, tuple):
            return None
        return shape, tree, items, true_exit, false_exit, found

    def _chain_condition(self, i: int, chain):
        """Execute the tests of a chain in order and return its condition with the real comparisons."""
        _shape, tree, items, _true, _false, found = chain
        leaves: dict[int, tuple] = {}
        mark = None
        for k in range(i, found):
            x = self.ins[k]
            if x.op in BRANCHES:
                leaves[k] = self.condition(x)
                if mark is None:
                    # a call made before the condition and not read by it is a statement of its own,
                    # as for a single test (``f(); if (a || b)``)
                    self.flush_call()
                    mark = len(self.out)
                elif self.pending_call is not None and self.pending_call.uses == 0:
                    raise Decline("call statement inside a condition")
                self._reset()
            elif x.op == "j":
                self._reset()
            elif x.op == "jal":
                self.call(x)
            elif x.op == "nop":
                if self._hazard_nop(k):
                    continue
                raise Decline("explicit nop")
            else:
                self.step(x)
        if mark is None or len(self.out) != mark:
            raise Decline("statement inside a condition")

        def build(node):
            if node[0] == "leaf":
                return leaves[node[1]]
            if node[0] == "nleaf":
                return self._negated(leaves[node[1]])
            return (node[0], build(node[1]), build(node[2]))
        return build(tree)

    def _if(self, i: int, end: int, loop) -> int:
        """The statement the branch at ``i`` opens; returns the index after it.  A compound condition can
        end in more than one place (``(a && b && c) || (d && e && f)`` reads up to the ``||`` as a condition
        of its own, whose block then does not read): the next end is tried when the first one declines."""
        avoid, first = set(), None
        while True:
            chain = self._chain(i, end, loop, avoid)
            if chain is None or chain[0] == "loop":
                try:
                    return self._if_at(i, end, loop, chain)
                except Decline as why:
                    raise (first or why) from None
            keep = self._snapshot()
            state = (list(self.out), dict(self.regs), dict(self.fregs), self.fcc, self.hilo, dict(self.stack_args),
                     self.pending_call, getattr(self, "before_barrier", 0), getattr(self, "deferred", None))
            try:
                return self._if_at(i, end, loop, chain)
            except Decline as why:
                first = first or why
                self._restore(keep)
                self.out[:] = state[0]
                self.regs, self.fregs, self.fcc, self.hilo, self.stack_args = state[1:6]
                self.pending_call, self.before_barrier, self.deferred = state[6:]
                avoid.add(chain[5])

    def _if_at(self, i: int, end: int, loop, chain) -> int:
        if chain is not None:
            cond = self._chain_condition(i, chain)
            self._reset()
            if chain[0] == "loop":
                # the loop's exit test: ``for (;;) { if (!(a && b)) break; ... }``, read back as ``while``
                leave = Stmt("if", ("not", cond), [Stmt("break")], None)
                if chain[2][-1][1] == "j" and chain[2][-1][2] == chain[4]:
                    # the last test is followed by an explicit jump to the false label: the condition of a
                    # loop statement (``while (a && b)``, ``do ... while (a && b)``), not of an ``if``
                    # (``if (a || b) break;`` ends with a jump to the true label instead)
                    self.loop_tests.add(id(leave))
                self.out.append(leave)
                return chain[5] + 1
            body, target = chain[3], chain[4]
        else:
            x = self.ins[i]
            target = self.index_of[x.target]
            if target <= i or target > end:
                raise Decline("branch out of its block")
            taken = self.condition(x)
            self.flush_call()
            cond = self._negated(taken)
            body = i + 1
            # ``if (a && b) ... else ...``: every test branches to the else label, which lies outside the
            # then part, so a further test there (pure computation, then a branch to that same label) is
            # part of the condition and not an ``if`` of its own.
            while target - 1 > body and self.ins[target - 1].op == "j" and \
                    target < (self.index_of.get(self.ins[target - 1].target) or 0) <= end:
                k = body
                while k < target - 1 and self.ins[k].op not in BRANCHES and k not in self.targets \
                        and k not in self.tables and self.ins[k].op not in STORES \
                        and self.ins[k].op not in ("j", "jal", "jalr", "jr", "swc1", "sdc1", "nop", "break"):
                    k += 1
                test = self.ins[k]
                if k >= target - 1 or test.op not in BRANCHES or self.index_of.get(test.target) != target \
                        or k in self.targets or k in self.tables:
                    break
                self._reset()
                mark = len(self.out)
                for at in range(body, k):
                    self.step(self.ins[at])
                more = self.condition(test)
                if len(self.out) != mark:
                    raise Decline("statement inside a condition")
                cond = ("&&", cond, self._negated(more))
                body = k + 1
        last = self.ins[target - 1]
        if target - 1 > body and last.op == "j":
            after = self.index_of[last.target]
            # a jump to the function's end is a ``return`` (its value may sit in the jump's delay
            # slot), not the jump over an else block: both spell the same instructions, and only
            # the first keeps the value
            if target < after <= end and after != len(self.ins):
                # if/else: the jump over the else block ends the then block
                keep = self._snapshot()
                try:
                    then = self._braced(body, target - 1, loop)
                    other = self._braced(target, after, loop)
                except Decline:
                    self._restore(keep)
                else:
                    self.out.append(Stmt("if", cond, then, other))
                    return after
            if after == target:
                # ``if (c) { ... } else { }``: the jump over an empty else block goes to the next instruction
                # (it can also be the ``break`` that ends a switch at the end of the block: then the block
                # does not read without it)
                keep = self._snapshot()
                try:
                    then = self._braced(body, target - 1, loop)
                except Decline:
                    self._restore(keep)
                else:
                    self.out.append(Stmt("if", cond, then, []))
                    return target
        then = self._braced(body, target, loop)
        self.out.append(Stmt("if", cond, then, None))
        return target

    # -- rendering ----------------------------------------------------------------------------

    def symbol(self, addr: int, call: bool = False) -> str:
        known = self.names.get(addr)
        if known:
            return known
        return f"func_{addr:08X}" if call else f"D_{addr:08X}"

    def note_global(self, addr: int, kind):
        self.globals.setdefault(addr, {"acc": [], "addr": False})
        if kind == "addr":
            self.globals[addr]["addr"] = True
        else:
            self.globals[addr]["acc"].append(kind)

    def note_indexed(self, loc, kind):
        """``sym+off($index)``: GCC addresses a member at a non-zero offset of an array of structures
        this way (``S[i].m``; a plain ``A[i]`` or a member at offset 0 loads the address first).  The
        scale of the index is the size of the structure."""
        if loc[0] != "gidx":
            return
        index = strip(loc[2])
        scale, cols = (index.a[1] if index.k == "mul" else 1), None
        pair = self.grid_index(index)
        if pair is not None:
            # ``S[i][j].m``: j * element size, then i * row size, added in that order (measured)
            scale, cols = pair[1].a[1], pair[0].a[1] // pair[1].a[1]
        info = self.indexed.setdefault(loc[1], {"scale": scale, "acc": [], "cols": cols})
        if info["scale"] != scale or info["cols"] != cols:
            raise Decline("one indexed global, two element sizes")
        info["acc"].append(kind)

    @staticmethod
    def grid_index(index: Node):
        """(row product, element product) of the index of ``S[i][j].m``, or None."""
        index = strip(index)
        if index.k != "bin" or index.a[0] != "+":
            return None
        inner, outer = strip(index.a[1]), strip(index.a[2])
        if inner.k == "mul" and outer.k == "mul" and outer.a[1] > inner.a[1] > 0 and outer.a[1] % inner.a[1] == 0:
            return outer, inner
        return None

    def _indexed_layout(self):
        """Group the indexed addresses into arrays: (base, structure size) -> {offset: type}."""
        self.member_of: dict[int, tuple[int, int]] = {}      # address -> (base, offset)
        self.arrays: dict[int, dict] = {}                    # base -> {"size": S, "fields": {offset: type}}
        sizes = {"s8": 1, "u8": 1, "s16": 2, "u16": 2, "s32": 4, "u32": 4, "f32": 4, "f64": 8}
        current = None
        for addr in sorted(self.indexed):
            info = self.indexed[addr]
            scale, acc = info["scale"], info["acc"]
            width = acc[0][0]
            if any(a[0] != width for a in acc):
                raise Decline("indexed global accessed with two widths")
            kinds = {a[1] for a in acc}
            if kinds & {"f", "d"}:
                if kinds - {"f", "d"}:
                    raise Decline("indexed global accessed as float and integer")
                ctype = "f32" if width == 4 else "f64"
            else:
                ctype = INT_TYPES[(width, "u" not in kinds)]
            align = min(width, 4)
            if current is not None and self.arrays[current]["size"] == scale and \
                    self.arrays[current]["cols"] == info["cols"] and \
                    addr - current + width <= scale and (addr - current) % align == 0 and \
                    all(off + sizes[t] <= addr - current for off, t in self.arrays[current]["fields"].items()):
                self.arrays[current]["fields"][addr - current] = ctype
                self.member_of[addr] = (current, addr - current)
                continue
            if scale % align or scale < width:
                raise Decline("indexed global whose element is not a structure size")
            # A member of an array element is addressed ``sym+off($index)`` at every offset, 0 included
            # (measured with the compiler: ``S[i].a``, a one-member structure, ``M[i][2]``).
            base, first = addr, 0
            for anchor, size in self.anchors.items():
                if size == scale and anchor <= addr and addr - anchor + width <= size:
                    base, first = anchor, addr - anchor    # the code names the array's start elsewhere
            current = base
            self.arrays[base] = {"size": scale, "fields": {first: ctype}, "cols": info["cols"]}
            self.member_of[addr] = (base, first)

    def global_type(self, addr: int) -> str:
        info = self.globals.get(addr) or {"acc": []}
        acc = info["acc"]
        if not acc:
            return "s32"
        width = acc[0][0]
        if (8, "q") in acc:
            return "u64"
        if acc[0][1] in ("f", "d"):
            return "f32" if acc[0][1] == "f" else "f64"
        same = [a for a in acc if a[0] == width]
        loads = [a[1] for a in same if a[1] in ("s", "u")]
        signed = (loads[0] == "s") if loads else True
        return INT_TYPES[(width, signed)]

    def frame_var(self, off: int, width: int):
        """(text of the variable at this frame slot, its type, exact) for an access of ``width``."""
        if off >= self.size:
            index, rest = divmod(off - self.size, 4)
            if index >= len(self.params):
                raise Decline("unknown incoming argument")
            param = self.params[index]
            if param.get("kind") == "stack-narrow" and (param["noff"], param["nsize"]) == (off, width):
                return f"arg{index}", param["type"], True
            if param.get("kind") == "stack-narrow":
                raise Decline("mixed access to a stack argument")
            if param.get("kind") == "pad":
                raise Decline("access to half of a double argument")
            if param.get("kind") == "double":
                if rest == 0 and width == 8:
                    return f"arg{index}", "f64", True
                raise Decline("access to half of a double argument")
            if rest == 0 and width == 4:
                return f"arg{index}", param.get("type", "s32"), True
            if rest == 4 - width:
                return f"arg{index}", "s32", False
            raise Decline("odd access to an incoming argument")
        for array in self.vars.values():
            if array.get("array") and array["off"] <= off < array["off"] + array["size"]:
                if width != array["elem"] or (off - array["off"]) % width:
                    raise Decline("access across the elements of a local array")
                if array.get("struct") and off - array["off"] >= array["struct"]:
                    raise Decline("access past a structure local")
                return (f"{self.var_name(array)}{array.get('member', '')}[{(off - array['off']) // width}]",
                        array["type"], True)
        var = self.vars.get(off)
        if var is not None and not var.get("aggregate") and var["size"] == width:
            return self.var_name(var), var["type"], True
        holder = self._holder(off, width)
        if holder is not None and "type" in holder and off == holder["off"] + holder["size"] - width:
            return self.var_name(holder), holder["type"], False
        raise Decline("access inside a local")

    def var_name(self, var) -> str:
        if var.get("param"):
            for index, param in enumerate(self.params):
                if param.get("kind") == "narrow" and param["off"] == var["off"]:
                    return f"arg{index}"
        return f"sp{var['off']:X}"

    def field(self, off: int, ctype: str) -> str:
        if off < 0:
            raise Decline("negative displacement")
        size = {"s8": 1, "u8": 1, "s16": 2, "u16": 2, "s32": 4, "u32": 4, "f32": 4, "f64": 8}[ctype]
        if off % min(size, 4):
            raise Decline("unaligned field")
        self.fields.add((off, ctype))
        return f"struct gd_{off:X}_{ctype}"

    def lvalue(self, loc, width: int, signed, store: bool = False) -> tuple[str, str]:
        """(C text, C type) of a location accessed with this width."""
        kind = loc[0]
        if signed == "f":
            want = "f32" if width == 4 else "f64"
        elif width == 8:
            want = "u64"
        else:
            want = INT_TYPES[(width, True if signed is None else signed)]
        stepped = getattr(self, "stepped", {})
        if kind == "local" and loc[1] in self.pointer_locals and width == 4 and not store:
            return f"((s32){self.frame_var(loc[1], 4)[0]})", "s32"
        if (kind, loc[1]) in stepped:                         # declared as a pointer: its value is an address
            if store or width != 4:
                raise Decline("store to a stepped pointer")
            name = self.frame_var(loc[1], 4)[0] if kind == "local" else self.symbol(loc[1])
            return f"((s32){name})", "s32"
        if kind == "regvar":
            if width != 4:
                raise Decline("narrow access to a register variable")
            return self.regvar_name(loc[1]), "s32"
        if kind == "pidx":                                    # an element through a pointer held in a local
            return f"((s32 *){self.frame_var(loc[1], 4)[0]})[{loc[2]}]", "s32"
        if kind == "gidx":
            if not hasattr(self, "member_of"):
                self._indexed_layout()
            base, off = self.member_of[loc[1]]
            ctype = self.arrays[base]["fields"][off]
            index = strip(loc[2])
            if self.arrays[base]["cols"]:
                outer, inner = self.grid_index(index)
                text = (f"{self.symbol(base)}[{self.unparen(self.expr(outer.a[0]))}]"
                        f"[{self.unparen(self.expr(inner.a[0]))}].m{off:X}")
            else:
                if index.k == "mul":
                    index = index.a[0]
                text = f"{self.symbol(base)}[{self.unparen(self.expr(index))}].m{off:X}"
            if signed is not None and not store and ctype in ("s8", "u8", "s16", "u16") and ctype != want:
                return f"(({want}){text})", want             # a member read with both signs: this read is the other
            return text, ctype
        if kind == "local":
            text, ctype, exact = self.frame_var(loc[1], width)
            if exact:
                if signed is not None and ctype in ("s8", "u8", "s16", "u16") and ctype != want:
                    return f"(({want}){text})", want
                return text, ctype
            if store:
                raise Decline("narrow store into a wider local")
            return f"(({want}){text})", want
        if kind == "global" and loc[1] in getattr(self, "arrays", {}) and 0 in self.arrays[loc[1]]["fields"]:
            # read directly as well as by index: the first element's first member
            name = f"{self.symbol(loc[1])}{'[0]' if self.arrays[loc[1]]['cols'] else ''}[0].m0"
            ctype = self.arrays[loc[1]]["fields"][0]
            if ctype == want or (signed is None and ctype[1:] == want[1:]):
                return name, ctype
            return f"(*({want} *)&{name})", want
        if kind == "global":
            addr = loc[1]
            name = self.symbol(addr)
            ctype = self.global_type(addr)
            if ctype == want or (signed is None and ctype[1:] == want[1:]):
                return name, ctype
            return f"(*({want} *)&{name})", want
        if kind == "deref":
            base, off = loc[1], loc[2]
            member = self.member_element(base, off, width, want, weak=store and signed is None)
            if member is not None:
                return member, want
            inner = self.between_products(base) if off == 0 else None
            if inner is None:
                inner = self.expr(base)
            if off == 0:
                return f"(*({want} *){inner})", want
            return f"(({self.field(off, want)} *){inner})->v", want
        if kind == "lidx":
            array = self.vars.get(loc[1])
            if array is None or not array.get("array") or array["elem"] != width:
                raise Decline("indexed access that is not a local array")
            index = strip(loc[2])
            if width > 1:
                if not (index.k == "mul" and index.a[1] == width):
                    raise Decline("local array index is not scaled by its element")
                index = index.a[0]
            return f"{self.var_name(array)}[{self.unparen(self.expr(index))}]", array["type"]
        raise Decline(f"location {kind}")

    def expr(self, n: Node) -> str:
        k = n.k
        if k == "cp":
            return self.expr(n.a[0])
        if k == "const":
            return self.number(n.a[0])
        if k == "hi":
            return self.number(_s32(n.a[0] << 16))
        if k == "structarg":
            return f"(*(struct n64dk_arg{n.a[1]} *)&{self.symbol(n.a[0])})"
        if k == "sym":
            return f"((s32)&{self.symbol(n.a[0])})"
        if k == "lad":
            off = n.a[0]
            if off >= self.size:
                index, rest = divmod(off - self.size, 4)
                if rest or index >= len(self.params) or self.params[index].get("kind") == "stack-narrow":
                    raise Decline("address inside an incoming argument")
                return f"((s32)&arg{index})"
            var = self.vars.get(off)
            if var is None:
                raise Decline("address inside a local")
            return f"((s32)&{self.var_name(var)})"
        if k == "mh":
            raise Decline("half of a division by a constant")
        if k == "mhu":
            quotient = self._unsigned_quotient(n, 0)
            if quotient is None:
                raise Decline("half of an unsigned division by a constant")
            return self.expr(quotient)
        if k == "fbits":
            raise Decline("float bits used as an integer")
        if k == "blk":
            return self.block_operand(n.a[1], n.a[0])
        if k == "sret":
            raise Decline("result pointer used as a value")
        if k == "ld":
            span = self.literal_span
            if n.a[2][0] == "global" and (
                    (n.a[1] == "f" and span is not None and span[0] < n.a[2][1] < span[1])
                    or (n.a[1] in ("f", "d") and n.a[2][1] in self.literal_pool)):
                # a constant that lies between two of the compiler's own: a literal of the source as well
                # (the constants of a function are laid out in the order they are used)
                words = [self.read(n.a[2][1] + 4 * k) for k in range(n.a[0] // 4)]
                if None in words:
                    raise Decline("constant outside the image")
                return float_text(words[0]) if n.a[0] == 4 else double_text(words[0] << 32 | words[1])
            return self.lvalue(n.a[2], n.a[0], n.a[1])[0]
        if k == "bin":
            op, a, b = n.a
            if op == "+" and id(n) in self.inserted:
                return f"(({self.expr(a)} + {self.number(self.inserted[id(n)])}) + {self.expr(b)})"
            if op == "+":
                nested = self.nested_member(a, b)
                if nested is not None:
                    return nested
            # The compiler works on the left operand first, and it may finish the right one before
            # it loads the left one's value (``A[i] + B[i]``: address of A, value of B, value of A).
            # The operand whose first instruction came first is the left one, whatever order the
            # final instruction names them in.
            # That only shows for an operand that produces code when the compiler first meets it
            # (a computed value, an element or field whose address is formed, a narrow load that is
            # widened).  A plain variable is loaded when the operation itself is emitted, wherever
            # it stood in the source: with one of those the instruction's own order is the source's.
            moved = self._constant_moved(op, a, b) or self._factor_moved(n, op, a, b)
            if moved is not None:
                return moved
            if op == "+" and const_of(a) is None and const_of(b) is None:
                # ``y + x * -K``: as ``y + -(x * K)`` it would be folded into a subtraction (probe)
                parts = []
                for part in (a, b):
                    inner = strip(part)
                    negated = inner.k == "un" and inner.a[0] == "-" and inner.a[1].k == "mul" and inner.a[1].a[1] > 1
                    parts.append(f"({self.expr(inner.a[1].a[0])} * {self.number(-inner.a[1].a[1])})" if negated
                                 else None)
                if (parts[0] is None) != (parts[1] is None):
                    return f"({parts[0] or self.expr(a)} + {parts[1] or self.expr(b)})"
            if op == "+" and a.k == "lad" and const_of(b) is None and first_seq(b) < a.seq \
                    and strip(b).k not in ("mul", "sym", "lad"):
                # the address of a local named first in the addition although the other operand was computed
                # first: pointer arithmetic on the local (probe; as integers the address is formed first)
                return f"((s32)((u8 *){self.expr(a)[6:-1]} + {self.expr(b)}))"
            if op in ("+", "*", "&", "|", "^") and const_of(a) is None and const_of(b) is None \
                    and not plain_variable(a) and not plain_variable(b) and work_before(b, a):
                a, b = b, a
            inner = strip(a)
            if op == "+" and inner.k == "bin" and inner.a[0] == "+" and strip(inner.a[1]).k == "mul" \
                    and const_of(inner.a[2]) is not None and const_of(inner.a[2]) > 0 and const_of(b) is None \
                    and first_seq(inner) < first_seq(b):
                index, offset = strip(inner.a[1]), const_of(inner.a[2])
                self.member_arrays.add((offset, index.a[1]))
                return (f"((s32)&((struct n64dk_m{offset:X}_{index.a[1]:X} *){self.expr(b)})"
                        f"->m[{self.unparen(self.expr(index.a[0]))}])")
            right = strip(b)
            if op == "+" and right.k == "bin" and right.a[0] == "+" and strip(right.a[2]).k == "sym" \
                    and const_of(a) is None and const_of(right.a[1]) is None and first_seq(a) < first_seq(right):
                # ``P + (Q + &SYM)``: the address stood first, ``(&SYM + P) + Q``; the compiler takes the
                # constant address out of the left sum and adds it to what follows (measured: ``&SYM[i][j]``)
                return f"(({self.expr(right.a[2])} + {self.expr(a)}) + {self.expr(right.a[1])})"
            if op == "|" and right.k == "bin" and right.a[0] == "|" and const_of(right.a[2]) is not None \
                    and const_of(right.a[1]) is None and const_of(a) is None:
                # ``(X | C) | Y``: the compiler moves the constant to the later operand (measured)
                return f"(({self.expr(a)} | {self.expr(right.a[2])}) | {self.expr(right.a[1])})"
            value = const_of(b)
            if op == "+" and value is not None and value < 0 and strip(b).k == "const":
                return f"({self.expr(a)} - {self.number(-value)})"
            if op == "<u":
                # ``a == b`` and ``a != b`` as values (probe): ``xor; sltiu 1`` and ``xor; sltu $zero``, with
                # ``xori`` for a small constant (``== 0`` included) and nothing at all for ``!= 0``
                test = "==" if const_of(b) == 1 and strip(b).k == "const" else \
                    "!=" if const_of(a) == 0 and strip(a).k == "const" else None
                differ = (a if test == "==" else b) if test else None
                if test and differ.k == "bin" and differ.a[0] == "^":
                    if const_of(differ.a[2]) is not None:
                        return f"({self.expr(differ.a[1])} {test} {self.expr(differ.a[2])})"
                    return f"({self.cond((test, differ.a[1], differ.a[2]))})"
                if test == "!=" and const_of(b) is None:
                    return f"({self.expr(b)} != 0)"
            if op in ("<u", ">>u", "/u", "%u"):
                return f"((u32){self.expr(a)} {op[:-1]} {self.unsigned(b)})"
            if id(n) in self.unsigned_masks:
                return f"({self.expr(a)} & {self.expr(b)}u)"
            return f"({self.expr(a)} {op} {self.expr(b)})"
        if k == "fconst":
            return float_text(n.a[0])
        if k == "dconst":
            return double_text(n.a[0])
        if k == "cond":
            test, first, second = n.a
            left = strip(test[1]) if isinstance(test[1], Node) else None
            value = strip(first)
            same = left is not None and left.k == "ld" and value.k == "ld" and value.a[2] == left.a[2]
            if not test[0].startswith(("f", "!f")) and const_of(first) is None and (const_of(second) is not None or same):
                # The compiler itself turns ``c ? K : a`` into ``!c ? a : K`` (a constant first arm, or a second
                # arm that is the compared operand), and ``a < K ? a : K - 1`` written directly becomes a
                # minimum: such a value is written the way that is turned into what the code shows (measured)
                return f"(({self.cond(self._negated(test))}) ? {self.expr(second)} : {self.expr(first)})"
            return f"(({self.cond(test)}) ? {self.expr(first)} : {self.expr(second)})"
        if k in ("dbits", "dlow"):
            raise Decline("double bits used as an integer")
        if k == "fbin":
            op, a, b = n.a
            if op == "+" and strip(a) is strip(b):
                return f"({self.expr(a)} * 2.0f)"              # a doubling is compiled as an addition
            # as for integers: a variable is loaded when the operation is emitted, so with one of those
            # the instruction's own order is the source's (``F + (f32)i`` is ``add.s $f0,$f2,$f0``; measured)
            if op in ("+", "*") and first_seq(b) < first_seq(a) and not plain_float(a) and not plain_float(b):
                a, b = b, a
            return f"({self.expr(a)} {op} {self.expr(b)})"
        if k == "fun":
            return f"({n.a[0]}{self.expr(n.a[1])})"
        if k == "cvtf":
            return f"(({n.a[0]}){self.expr(n.a[1])})"
        if k in ("ibits", "ftrunc"):
            raise Decline("half of a conversion")
        if k == "pdiff":
            # the two pointers: what the subtraction's operands were before the compiler moved a constant
            # across it (``_constant_moved``): ``(A + c) - B`` as computed was written ``A - (B - c)``
            left, right, size = n.a
            lin, rin = strip(left), strip(right)
            self.element_sizes.add(size)
            kind = f"(struct n64dk_e{size} *)"
            first, second = self.expr(left), self.expr(right)
            if const_of(right) is None and lin.k == "bin" and lin.a[0] == "+" and strip(lin.a[2]).k == "const" \
                    and const_of(lin.a[1]) is None:
                amount = const_of(lin.a[2])
                first = self.expr(lin.a[1])
                second = f"({second} {'+' if amount < 0 else '-'} {self.number(abs(amount))})"
            elif const_of(left) is None and rin.k == "bin" and rin.a[0] == "+" and strip(rin.a[2]).k == "const" \
                    and const_of(rin.a[1]) is None:
                amount = const_of(rin.a[2])
                first = f"({first} {'+' if amount < 0 else '-'} {self.number(abs(amount))})"
                second = self.expr(rin.a[1])
            return f"({kind}{first} - {kind}{second})"
        if k == "mul":
            if n.a[0].k == "pdiff" and n.a[1] % n.a[0].a[2] == 0:
                # times the element size the division would fold away; as ``sizeof`` (unsigned) it stays (probe)
                return f"((s32)((u32){self.expr(n.a[0])} * {self.number(n.a[1])}))"
            return f"({self.expr(n.a[0])} * {self.number(n.a[1])})"
        if k == "low":
            return f"((s32){self.expr(n.a[0])})"              # a 64-bit result narrowed: its second word
        if k == "un":
            return f"({n.a[0]}{self.expr(n.a[1])})"
        if k == "cast":
            if n.a[0] in ("s8", "s16", "u8", "u16") and plain_variable(n.a[1]) and strip(n.a[1]).a[2][0] != "regvar":
                # a real cast of a variable in memory loads its narrow end; these are the written shifts
                amount = 24 if n.a[0] in ("s8", "u8") else 16
                if n.a[0][0] == "s":
                    return f"(({self.expr(n.a[1])} << {amount}) >> {amount})"
                return f"((u32)({self.expr(n.a[1])} << {amount}) >> {amount})"
            return f"(({n.a[0]}){self.expr(n.a[1])})"
        if k == "call" and id(n) in self.divisions:
            return f"({self.expr(n.a[1][0])} / {self.expr(n.a[1][1])})"
        if k == "call" and id(n) in self.remainders:
            return f"({self.expr(n.a[1][0])} % {self.expr(n.a[1][1])})"
        if k == "wmul":
            return f"((s64){self.expr(n.a[0])} * {self.number(n.a[1])})"
        if k == "call" and id(n) in self.unexpanded and not self.inside_unexpanded:
            self.inside_unexpanded = True                      # (its own arguments are inside the braces already)
            try:
                return f"({{ {self.expr(n)}; }})"
            finally:
                self.inside_unexpanded = False
        if k == "call":
            target, args = n.a
            if target.k == "func":
                name = self.symbol(target.a[0], call=True)
                info = self.callees.setdefault(name, {"ret": "s32"})
                if id(n) in self.float_calls:
                    ret = "f64" if id(n) in self.double_calls else "f32"
                    if info.get("ret", "s32") not in ("s32", ret):
                        raise Decline("one callee, two return types")
                    info["ret"] = ret
                if id(n) in self.wide_calls:
                    info["ret"] = "u64"
                if id(n) in self.forwarded:
                    if not (self.returns_value and self.return_type.startswith("struct ")):
                        raise Decline("structure returned from a call, of unknown size")
                    info["ret"] = self.return_type
                proto = self.callee_protos.get(target.a[0])
                if proto is not None:
                    info["proto"] = ["s32" if kind == "?" else kind for kind in proto]
                    return f"{name}({', '.join(self.argument(arg, k, proto) for k, arg in enumerate(args))})"
                if id(n) in self.prototypes:
                    if info.get("proto") not in (None, self.prototypes[id(n)]):
                        raise Decline("one callee, two argument lists")
                    info["proto"] = self.prototypes[id(n)]
            else:
                name = f"((s32 (*)()){self.expr(target)})"
            return f"{name}({', '.join(self.expr(arg) for arg in args)})"
        if k == "asg":
            loc, value = n.a
            if value.k == "cp" and strip(value).k == "call":
                # The call's value was copied to a register, stored from there and used from there.  An
                # assignment whose right side is the bare call is read back from memory instead; any
                # conversion around the call (even to its own type) gives the copy.
                text = self.lvalue(loc, 4, None, store=True)[0]
                return f"({text} = (s32){self.expr(value)})"
            return f"({self.assignment(loc, value, None, used=True)})"
        if k == "postinc":
            loc, delta, width = n.a
            if getattr(self, "stepped", {}).get((loc[0], loc[1])) == abs(delta):
                name = self.frame_var(loc[1], 4)[0] if loc[0] == "local" else self.symbol(loc[1])
                return f"((s32){name}{'++' if delta > 0 else '--'})"
            text, ctype = self.lvalue(loc, width, None, store=True)
            if delta in (1, -1):
                return f"({text}{'++' if delta > 0 else '--'})"
            raise Decline("post-increment of a pointer")
        raise Decline(f"expression {k}")

    def _constant_moved(self, op: str, a: Node, b: Node):
        """A sum or difference of two values and a constant.  The compiler's constant folder moves the
        constant to the other operand before any code is made (probes, in the instruction's operand order):
        ``(X + C) + Y`` is compiled as ``X + (Y + C)`` and the reverse, ``X - C - Y`` as ``X - (Y + C)`` and
        the reverse.  So the text is the form the instructions do *not* show."""
        if op not in ("+", "-"):
            return None

        def summed(n: Node):
            if n.k == "bin" and n.a[0] == "+" and strip(n.a[2]).k == "const" and const_of(n.a[1]) is None \
                    and strip(n.a[1]).k not in ("sym", "lad", "sret") and self.pointer_value(n) is None:
                return n.a[1], const_of(n.a[2])
            return None

        def plus(text: str, amount: int) -> str:
            return f"({text} {'+' if amount >= 0 else '-'} {self.number(abs(amount))})"

        left, right = summed(a), summed(b)
        if (left is None) == (right is None) or const_of(a) is not None or const_of(b) is not None:
            return None
        if right is not None:
            if strip(a).k in ("sym", "lad", "sret"):
                return None
            value, amount = right                              # X op (Y + C)  <-  (X op' C) op Y
            return f"({plus(self.expr(a), amount if op == '+' else -amount)} {op} {self.expr(value)})"
        if strip(b).k in ("mul", "sym", "lad", "sret") or strip(left[0]).k == "mul":
            return None                                        # ``&p->m[i]`` and its relatives: read elsewhere
        value, amount = left                                   # (X + C) op Y  <-  X op (Y op' C)
        return f"({self.expr(value)} {op} {plus(self.expr(b), amount if op == '+' else -amount)})"

    def argument(self, arg: Node, position: int, proto: list[str]) -> str:
        """An argument of a prototyped callee: a constant or a memory word in a ``float`` position is one."""
        bare = strip(arg)
        if position < len(proto) and proto[position] == "f32":
            if const_of(bare) is not None:
                return float_text(const_of(bare) & 0xFFFFFFFF)
            if bare.k == "ld" and bare.a[0] == 4 and bare.a[1] != "f":
                return self.lvalue(bare.a[2], 4, "f")[0]
        return self.expr(arg)

    def _factor_moved(self, n: Node, op: str, a: Node, b: Node):
        """A product of two values and a constant.  As with sums, the constant is moved to the other operand
        before any code is made (probes, in the instruction's operand order): ``(X * C) * Y`` is compiled as
        ``X * (Y * C)`` and the reverse.  That order shows only when the product is the whole value of a word
        assignment; as an operand, an argument, a returned or a narrowed value the scaled operand is always
        named first, and the two spellings differ only in whose first instruction came first."""
        if op != "*" or (a.k == "mul") == (b.k == "mul") or const_of(a) is not None or const_of(b) is not None:
            return None
        scaled, other = (a, b) if a.k == "mul" else (b, a)
        if scaled.a[1] in (0, 1, -1):
            return None
        value, factor = self.expr(scaled.a[0]), self.number(scaled.a[1])
        if id(n) not in self.direct_products:
            if scaled is not a:
                return None
            if first_seq(other) < first_seq(scaled):
                return f"(({self.expr(other)} * {factor}) * {value})"
            return f"({value} * ({self.expr(other)} * {factor}))"
        if scaled is b:
            return f"(({self.expr(other)} * {factor}) * {value})"       # mult X, Y*C  <-  (X * C) * Y
        return f"({value} * ({self.expr(other)} * {factor}))"           # mult X*C, Y  <-  X * (Y * C)

    def unsigned(self, n: Node) -> str:
        value = const_of(n)
        if value is not None:
            return self.number(value)
        return f"(u32){self.expr(n)}"

    @staticmethod
    def number(value: int) -> str:
        if -4096 < value < 4096:
            return str(value)
        if value < 0:
            return f"-0x{-value:X}" if value > -0x8000 else f"0x{value & 0xFFFFFFFF:08X}"
        return f"0x{value:X}"

    def between_products(self, base: Node):
        """An address ``((... (a * m + b * n) ...) * w + p) + K`` whose two products were computed in the order
        their variables were loaded, both loads first.  In an address GCC collects the terms before it computes
        them, puts the later product first (``a * m + b * n``: both loads, then ``b * n``, then ``a * m``) and
        moves every constant to the end.  The products keep their order only when a constant stood between them:
        ``a * m + c + b * n`` is folded to ``a * m + (b * n + c)``, whose constant is then carried outwards and
        added last as ``c * w`` (measured).  So the trailing constant is written back where it stood."""
        n = strip(base)
        if n.k != "bin" or n.a[0] != "+" or strip(n.a[2]).k != "const":
            return None
        total, scale, part = const_of(n.a[2]), 1, strip(n.a[1])
        for _depth in range(8):
            if part.k == "mul":
                scale, part = scale * part.a[1], strip(part.a[0])
                continue
            if part.k != "bin" or part.a[0] != "+":
                return None
            left, right = strip(part.a[1]), strip(part.a[2])
            if left.k == "mul" and right.k == "mul" and plain_variable(left.a[0]) and plain_variable(right.a[0]) \
                    and strip(left.a[0]).seq < strip(right.a[0]).seq < left.seq < right.seq:
                if scale == 0 or total % scale or total <= 0:
                    return None
                self.inserted[id(part)] = total // scale
                self.keep.append(part)
                return self.expr(n.a[1])
            part = left
        return None

    def nested_element(self, value: Node):
        """``&S[i].a`` as text, for ``i * size + &SYM``: the element's member ``a`` stands ``lead`` bytes into the
        element, so ``SYM`` is the address of the first element's ``a`` and the array begins ``lead`` bytes lower."""
        x = strip(value)
        if x.k != "bin" or x.a[0] != "+":
            return None
        index, symbol = strip(x.a[1]), strip(x.a[2])
        if index.k != "mul" or symbol.k != "sym" or index.a[1] <= 4:
            return None
        addr, size = symbol.a[0], index.a[1]
        known = self.nested.get(addr)
        if known is None:
            taken = set(self.globals) | set(self.indexed) | {entry[0] for entry in self.nested.values()}
            lead = 4
            while lead < size and addr - lead in taken:
                lead += 4
            if lead >= size:
                return None
            known = self.nested[addr] = (addr - lead, size, lead)
        if known[1] != size:
            raise Decline("one symbol, two element sizes")
        return f"&{self.symbol(known[0])}[{self.unparen(self.expr(index.a[0]))}].a"

    def nested_member(self, a: Node, b: Node):
        """``(i * size + &SYM) + C`` and ``(i * size + &SYM) + (j * n + C)`` with the constant added on its own.
        Written as arithmetic the compiler folds ``C`` into the symbol (``la SYM+C``).  It stays apart when the
        left part is the address of a member that is itself not first in its structure: ``&S[i].a`` is folded
        to ``i * size + (&S + lead)`` and that inner sum is not marked constant, so nothing more is merged
        into it (measured: ``(s32)&S[i].a + C``, ``&S[i].a.m[j]``)."""
        amount, inner = const_of(b), strip(b)
        if amount is not None and inner.k == "const":
            element = self.nested_element(a)
            if element is None:
                return None
            return f"((s32){element} {'-' if amount < 0 else '+'} {self.number(abs(amount))})"
        if inner.k == "bin" and inner.a[0] == "+" and strip(inner.a[1]).k == "mul" and strip(inner.a[2]).k == "const" \
                and const_of(inner.a[2]) > 0 and first_seq(a) < first_seq(b):
            element = self.nested_element(a)
            if element is None:
                return None
            index, offset = strip(inner.a[1]), const_of(inner.a[2])
            self.member_arrays.add((offset, index.a[1]))
            return (f"((s32)&((struct n64dk_m{offset:X}_{index.a[1]:X} *){element})"
                    f"->m[{self.unparen(self.expr(index.a[0]))}])")
        return None

    def member_element(self, base: Node, off: int, width: int, want: str, weak: bool = False):
        """An element of an array inside the structure a pointer variable points at, when the pointer was
        loaded before the index (the order GCC uses for the access itself; measured):
        ``p->arr[i]`` (scalar elements): index * width, + p, + offset of the array, access at 0;
        ``p->m[i].v`` (structure elements): p + index * size, access at the member's offset."""
        n = strip(base)
        if n.k != "bin" or n.a[0] != "+":
            return None
        # ``SYM[i][j]``: the row (i * row size + &SYM) first, then j * width added in front of it (measured)
        column, row = strip(n.a[1]), strip(n.a[2])
        if off == 0 and column.k == "mul" and column.a[1] == width and row.k == "bin" and row.a[0] == "+":
            outer, symbol = strip(row.a[1]), strip(row.a[2])
            if outer.k == "mul" and symbol.k == "sym" and outer.a[1] % width == 0 and outer.a[1] > width:
                # the symbol itself is declared as the two-dimensional array (a cast to one compiles otherwise)
                shape = (want, outer.a[1] // width)
                if self.grids.setdefault(symbol.a[0], shape) != shape or \
                        (self.globals.get(symbol.a[0]) or {}).get("acc"):
                    raise Decline("one global, two array shapes")
                return (f"{self.symbol(symbol.a[0])}"
                        f"[{self.unparen(self.expr(outer.a[0]))}][{self.unparen(self.expr(column.a[0]))}]")
        if off == 0 and width == 1 and column.k == "bin" and column.a[0] == "+" and const_of(n.a[2]) is None \
                and row.k not in ("mul", "sym", "lad"):
            # the same with byte elements: nothing scales the column, and it is added after the row (probe)
            outer, symbol = strip(column.a[1]), strip(column.a[2])
            if outer.k == "mul" and symbol.k == "sym" and outer.a[1] > 1 and first_seq(column) < first_seq(n.a[2]):
                # a store does not say whether the bytes are signed: the first load decides
                shape, known = (want, outer.a[1]), self.grids.get(symbol.a[0])
                if known is not None and known != shape and known[1] == shape[1] and known[0] in ("s8", "u8") \
                        and (weak or symbol.a[0] in self.weak_grids):
                    if not weak:
                        self.grids[symbol.a[0]] = shape
                        self.weak_grids.discard(symbol.a[0])
                    known = shape
                if known not in (None, shape) or (self.globals.get(symbol.a[0]) or {}).get("acc"):
                    return None                                # used another way as well: leave it as arithmetic
                if known is None:
                    self.grids[symbol.a[0]] = shape
                    if weak:
                        self.weak_grids.add(symbol.a[0])
                return (f"{self.symbol(symbol.a[0])}"
                        f"[{self.unparen(self.expr(outer.a[0]))}][{self.unparen(self.expr(n.a[2]))}]")
        if off == 0 and width == 1 and const_of(n.a[2]) is None:
            # ``p->m[i][j]`` on rows of bytes: p loaded first, i * row size, + p, + offset of the member, then + j
            lead, line = 0, strip(n.a[1])
            if line.k == "bin" and line.a[0] == "+" and strip(line.a[2]).k == "const" and const_of(line.a[2]) > 0:
                lead, line = const_of(line.a[2]), strip(line.a[1])
            if line.k == "bin" and line.a[0] == "+":
                outer, pointer = strip(line.a[1]), strip(line.a[2])
                if outer.k == "mul" and outer.a[1] > 1 and pointer.k == "ld" and pointer.a[0] == 4 \
                        and pointer.seq < first_seq(outer) and strip(n.a[1]).seq < first_seq(n.a[2]):
                    self.member_grids.add((lead, outer.a[1], want))
                    return (f"((struct n64dk_h{lead:X}_{outer.a[1]:X}_{want} *){self.expr(line.a[2])})"
                            f"->m[{self.unparen(self.expr(outer.a[0]))}][{self.unparen(self.expr(n.a[2]))}]")
        amount = const_of(n.a[2])
        if off == 0 and amount is not None and amount > 0:
            inner = strip(n.a[1])
            if inner.k == "bin" and inner.a[0] == "+":
                index, pointer = strip(inner.a[1]), strip(inner.a[2])
                if index.k == "mul" and index.a[1] == width and pointer.k == "ld" and pointer.a[0] == 4 \
                        and pointer.seq < first_seq(index) and amount % min(width, 4) == 0:
                    self.scalar_arrays.add((amount, want))
                    return (f"((struct n64dk_f{amount:X}_{want} *){self.expr(inner.a[2])})"
                            f"->m[{self.unparen(self.expr(index.a[0]))}]")
            return None
        pointer, index = strip(n.a[1]), strip(n.a[2])
        if index.k == "mul" and index.a[1] > width and pointer.k == "ld" and pointer.a[0] == 4 \
                and pointer.seq < first_seq(index) and off >= 0:
            size = index.a[1]
            inside = off % size
            if inside + width > size or inside % min(width, 4) or size % 4:
                return None
            self.struct_arrays.add((off - inside, size, inside, want))
            return (f"((struct n64dk_g{off - inside:X}_{size:X}_{inside:X}_{want} *){self.expr(n.a[1])})"
                    f"->m[{self.unparen(self.expr(index.a[0]))}].v")
        return None

    def pointer_value(self, value: Node):
        """The text of ``value`` as a pointer expression when it is one: ``&p->m[i]`` or ``&SYM[i]`` whose
        final addition names the pointer first.  GCC adds in that order only while the type is still a
        pointer (assigned to one, passed as one); converted to an integer the index comes first."""
        n = strip(value)
        if n.k != "bin" or n.a[0] != "+":
            return None
        base, rest = strip(n.a[1]), strip(n.a[2])
        if rest.k == "bin" and rest.a[0] == "+" and strip(rest.a[1]).k == "mul" and (const_of(rest.a[2]) or 0) > 0 \
                and const_of(base) is None and base.k != "sym" and first_seq(rest) < first_seq(base):
            index, offset = strip(rest.a[1]), const_of(rest.a[2])
            self.member_arrays.add((offset, index.a[1]))
            return (f"&((struct n64dk_m{offset:X}_{index.a[1]:X} *){self.expr(n.a[1])})"
                    f"->m[{self.unparen(self.expr(index.a[0]))}]")
        if base.k == "sym" and rest.k == "mul":
            self.element_sizes.add(rest.a[1])
            return (f"&((struct n64dk_e{rest.a[1]} *)&{self.symbol(base.a[0])})"
                    f"[{self.unparen(self.expr(rest.a[0]))}]")
        return None

    @staticmethod
    def same_place(x, y) -> bool:
        """Two locations that are one lvalue: the same variable, or the same offset from the same pointer
        loaded twice (``p->n--`` loads ``p`` for the read and again for the store; ``p->q->n++`` likewise,
        one level further)."""
        if x == y:
            return True
        if x[0] != "deref" or y[0] != "deref" or x[2] != y[2]:
            return False
        a, b = strip(x[1]), strip(y[1])
        if not (a.k == "ld" and b.k == "ld" and a.a[0] == b.a[0] == 4):
            return False
        if a.a[2][0] in ("local", "global", "regvar"):
            return a.a[2] == b.a[2]
        return Function.same_place(a.a[2], b.a[2])

    def assignment(self, loc, value: Node, width, used: bool = False) -> str:
        """The assignment as C text.  ``used``: its value is an operand (``if (++x >= 5)``); the stored
        value is then the one read, so an increment is the prefix form (probe: ``x++ >= 5`` compares the
        old value and has no ``move``; ``++x``, ``x += 1`` and ``x = x + 1`` as operands are one sequence)."""
        if width is None:
            width = value.a[0] if value.k == "ld" else 4
            if self.is_float(value):
                width = 8 if self.float_type(value) == "f64" else 4

        def stepped(text: str, up: bool) -> str:
            sign = "++" if up else "--"
            return sign + text if used else text + sign

        if width == 4 and loc[0] in ("deref", "local"):
            pointer = self.pointer_value(value)
            if pointer is not None and loc[0] == "deref":
                self.void_fields.add(loc[2])
                return f"((struct n64dk_v{loc[2]:X} *){self.expr(loc[1])})->v = (void *){pointer}"
            if pointer is not None and loc[1] in self.pointer_locals:
                return f"{self.frame_var(loc[1], 4)[0]} = (void *){pointer}"
        inner = strip(value)
        if width == 4 and (loc[0], loc[1]) in getattr(self, "stepped", {}) and loc[0] in ("local", "global"):
            name = self.frame_var(loc[1], 4)[0] if loc[0] == "local" else self.symbol(loc[1])
            if abs(self.pointer_step(loc, value)) == self.stepped[(loc[0], loc[1])]:
                return stepped(name, self.pointer_step(loc, value) > 0)
            return f"{name} = (void *){self.expr(value)}"
        if width == 4 and loc[0] == "local" and loc[1] in self.pointer_locals:
            if value.k == "cp" and inner.k == "bin" and inner.a[0] == "+" and const_of(inner.a[2]) in (1, -1) \
                    and strip(inner.a[1]).k == "ld" and strip(inner.a[1]).a[2] == loc:
                # ``p++`` on a ``void *`` steps by one and has the ``move`` of an increment (probe)
                return stepped(self.frame_var(loc[1], 4)[0], const_of(inner.a[2]) > 0)
            return f"{self.frame_var(loc[1], 4)[0]} = (void *){self.expr(value)}"
        if value.k == "cp" and inner.k == "bin" and inner.a[0] == "+" and loc[0] == "deref" and width == 4 \
                and abs(const_of(inner.a[2]) or 0) > 1 and strip(inner.a[1]).k == "ld" \
                and self.same_place(strip(inner.a[1]).a[2], loc):
            # ``p->q++`` on a pointer member: stepped by the size of what it points at
            step = abs(const_of(inner.a[2]))
            self.pointer_fields.add((loc[2], step))
            self.element_sizes.add(step)
            return stepped(f"((struct n64dk_p{loc[2]:X}_{step:X} *){self.expr(loc[1])})->v", const_of(inner.a[2]) > 0)
        if width == 4 and value.k == "bin" and value.a[0] == "*" and not used:
            self.direct_products.add(id(value))
            self.keep.append(value)
        sign = "f" if self.is_float(value) else None
        if width in (1, 2) and loc[0] in ("deref", "global") and inner.k == "const" \
                and inner.a[0] >= 1 << (8 * width - 1):
            sign = False        # ``*(u8 *)p = 255`` loads 255; into a signed byte the constant is -1 (probe)
        text, ctype = self.lvalue(loc, width, sign, store=True)
        if width in (1, 2):
            # The operations of a value stored narrow are done in the narrow type.  A signed mask with its top
            # bit set is then a negative constant (``li -16; and``); ``andi`` with such a mask is the unsigned
            # one (probe: ``p->h = x & 0xFFF0`` against ``& 0xFFF0u``).
            stack = [value]
            while stack:
                part = stack.pop()
                if part.k == "cp":
                    stack.append(part.a[0])
                elif part.k == "bin" and part.a[0] in ("+", "-", "&", "|", "^"):
                    mask = part.a[2].a[0] if part.a[0] == "&" and part.a[2].k == "const" else None
                    if mask is not None and (1 << (8 * width - 1)) <= mask < (1 << 8 * width):
                        self.unsigned_masks.add(id(part))
                        self.keep.append(part)
                    stack += [part.a[1], part.a[2]]
        if value.k == "cp" and inner.k == "fbin" and inner.a[0] in ("+", "-"):
            # ``x++`` on a float or double: one is added and the sum is moved before it is stored (measured)
            old, one = strip(inner.a[1]), strip(inner.a[2])
            if old.k == "ld" and self.same_place(old.a[2], loc) and one.k == "ld" and one.a[2][0] == "global" \
                    and self.read is not None:
                bits = [self.read(one.a[2][1] + 4 * k) for k in range(width // 4)]
                if bits == ([0x3F800000] if width == 4 else [0x3FF00000, 0]):
                    self.literal_addrs.add(one.a[2][1])
                    return stepped(text, inner.a[0] == "+")
        if width == 8 and sign is None and value.k != "cp" and inner.k == "bin" and inner.a[0] == "+" \
                and strip(inner.a[1]).k == "ld" and self.same_place(strip(inner.a[1]).a[2], loc):
            return f"{text} += {self.expr(inner.a[2])}"       # 64 bits: ``G += 1`` (``G++`` copies the sum)
        if inner.k == "bin" and inner.a[0] == "+" and const_of(inner.a[2]) is None and width == 4:
            left = strip(inner.a[1])
            if left.k == "bin" and left.a[0] == "+" and const_of(left.a[2]) is not None \
                    and strip(left.a[1]).k == "ld" and self.same_place(strip(left.a[1]).a[2], loc):
                # ``x += y + c``: the constant is added to the old value first (measured)
                amount = const_of(left.a[2])
                sign = "+" if amount >= 0 else "-"
                return f"{text} += {self.expr(inner.a[2])} {sign} {self.number(abs(amount))}"
        if inner.k == "bin" and inner.a[0] == "-" and const_of(inner.a[2]) is None and width == 4:
            left = strip(inner.a[1])
            if left.k == "bin" and left.a[0] == "+" and (const_of(left.a[2]) or 0) > 0 \
                    and strip(left.a[1]).k == "ld" and self.same_place(strip(left.a[1]).a[2], loc):
                # ``x += c - y``: the constant is added to the old value first, then ``y`` taken off (measured;
                # ``x = x + c - y`` and ``x = (x + c) - y`` are folded to ``x - (y - c)`` instead)
                return f"{text} += {self.number(const_of(left.a[2]))} - {self.expr(inner.a[2])}"
        if value.k == "cp" and inner.k == "bin" and inner.a[0] == "+" and const_of(inner.a[2]) in (1, -1):
            source = strip(inner.a[1])
            if source.k == "ld" and self.same_place(source.a[2], loc):
                return stepped(text, const_of(inner.a[2]) > 0)
            place = strip(loc[1]) if loc[0] == "deref" else None
            if source.k == "ld" and source.a[2][0] == "gidx" and source.a[0] == width and place is not None \
                    and place.k == "bin" and place.a[0] == "+":
                # ``SYM[i].m++`` as a statement: the element's address is formed for the store (and the index
                # once more before it, unused), the old value is read through ``SYM+m($index)`` (measured;
                # ``SYM[i].m += 1`` reads and stores through the symbol and has no copy)
                for symbol, index in ((strip(place.a[1]), place.a[2]), (strip(place.a[2]), place.a[1])):
                    if symbol.k == "sym" and (symbol.a[0] + loc[2]) & 0xFFFFFFFF == source.a[2][1] \
                            and self._alike(index, source.a[2][2]):
                        element = self.lvalue(source.a[2], width, None, store=True)[0]   # the member itself, no cast
                        return stepped(element, const_of(inner.a[2]) > 0)
            if source.k == "ld" and source.a[0] == width and source.a[2][0] == "deref" and loc[0] == "deref" \
                    and source.a[2][2] == loc[2] and self._alike(source.a[2][1], loc[1]):
                # ``G[i][j]++`` as a statement: the element's address is formed once for nothing, once for the
                # store and once more for the load, and the new value is copied (measured)
                return stepped(text, const_of(inner.a[2]) > 0)
        if inner.k == "ld" and inner.a[2][0] == "local" and inner.a[0] == width and width < 4:
            # a narrowing copy reads the low end of the wider variable directly
            name, _type, exact = self.frame_var(inner.a[2][1], width)
            if not exact:
                return f"{text} = {name}"
        return f"{text} = {self.unparen(self.expr(value))}"

    @staticmethod
    def unparen(text: str) -> str:
        if text.startswith("(") and text.endswith(")"):
            depth = 0
            for index, char in enumerate(text):
                depth += char == "("
                depth -= char == ")"
                if depth == 0 and index < len(text) - 1:
                    return text
            return text[1:-1]
        return text

    def cond(self, cond) -> str:
        if cond[0] == "never":
            return "0"
        if cond[0] == "not":
            return f"!({self.cond(cond[1])})"
        if cond[0] in ("&&", "||"):
            return f"({self.cond(cond[1])}) {cond[0]} ({self.cond(cond[2])})"
        op, a, b = cond
        if op.startswith(("f", "!f")):
            negated, op = op.startswith("!"), op.lstrip("!")[1:]
            left, right = strip(a), strip(b)
            if left.k == "ld" and right.k == "ld":
                # two values in memory are loaded when the comparison is emitted, left operand first
                # (``x < p->m``: the pointer, then ``x``, then the member; measured)
                flipped = right.seq < left.seq
            else:
                flipped = first_seq(b) < first_seq(a)
            if op in ("<", "<=") and flipped:
                # ``a > b`` is compiled as ``b < a`` with ``a`` still evaluated first
                text = f"{self.expr(b)} {'>' if op == '<' else '>='} {self.expr(a)}"
            else:
                text = f"{self.expr(a)} {op} {self.expr(b)}"
            return f"!({text})" if negated else text
        if op.endswith("u"):
            if const_of(a) is None and const_of(b) is None and self._right_came_first(a, b):
                return f"(u32){self.expr(b)} {FLIP[op[:-1]]} {self.unsigned(a)}"
            return f"(u32){self.expr(a)} {op[:-1]} {self.unsigned(b)}"
        if op in ("==", "!=") and const_of(b) == 0 and id(strip(a)) in self.value_and \
                and const_of(strip(a).a[2]) == 0xFF and self._byte_and(strip(a).a[1]):
            # ``c & 0x80`` on an unsigned byte is computed as a byte and widened for the test
            # (measured: ``lbu; andi 0x80; andi 0xff; beq``)
            return f"{self.expr(strip(a).a[1])} {op} 0"
        if op in ("==", "!=") and const_of(b) == 0 and id(strip(a)) in self.value_and:
            inner = strip(a)                                   # ``x % 8 != 0``: the remainder tested, not the bits
            bits = strip(inner.a[1])
            if const_of(inner.a[2]) in (0xFF, 0xFFFF) and bits.k == "bin" and bits.a[0] == "&" \
                    and const_of(bits.a[2]) is not None and strip(bits.a[1]).k == "ld" \
                    and strip(bits.a[1]).a[:2] == ({0xFF: 1, 0xFFFF: 2}[const_of(inner.a[2])], False):
                # ``if (s.flags & 4)`` on an unsigned narrow member: the bits, then truncated to its width
                return f"{self.expr(bits)} {op} 0"
            return f"({self.expr(inner.a[1])} % {self.number(const_of(inner.a[2]) + 1)}) {op} 0"
        if const_of(a) is None and const_of(b) is None and self._right_came_first(a, b):
            a, b, op = b, a, FLIP[op]                          # ``a > b`` is compiled as ``b < a``, a first
        return f"{self.expr(a)} {op} {self.expr(b)}"

    @staticmethod
    def _byte_and(n: Node) -> bool:
        """``b & C`` with ``b`` an unsigned byte read from memory and ``C`` a byte constant."""
        n = strip(n)
        if n.k != "bin" or n.a[0] != "&" or const_of(n.a[2]) is None or not 0 <= const_of(n.a[2]) <= 0xFF:
            return False
        byte = strip(n.a[1])
        return byte.k == "ld" and byte.a[0] == 1 and byte.a[1] is False

    @staticmethod
    def _unsigned_typed(n: Node) -> bool:
        """Whether the C written for ``n`` has an unsigned type (it carries a ``(u32)`` operand)."""
        n = strip(n)
        if n.k == "cast":
            return n.a[0] == "u32"
        if n.k != "bin":
            return False
        if n.a[0] in (">>u", "/u", "%u"):
            return True
        return n.a[0] in ("&", "|", "^", "+", "-", "*") and (Function._unsigned_typed(n.a[1])
                                                             or Function._unsigned_typed(n.a[2]))

    @staticmethod
    def _right_came_first(a: Node, b: Node) -> bool:
        """Whether the comparison's second operand stood first in the source.  A value in memory is loaded
        when the comparison is emitted, left operand first, so two loads show the order directly
        (``G[i].m > x``: the element, then ``x``).  A computed value is finished earlier whichever side it
        is on, so a load on the left with a computed right operand says nothing: the instruction's order
        stands (``x == y - 2``)."""
        left, right = strip(a), strip(b)
        loads = [n.k == "ld" and n.a[0] == 4 for n in (left, right)]
        if all(loads):
            return right.seq < left.seq
        if left.k == right.k == "ld" and left.a[0] == right.a[0] and left.a[1] == right.a[1] \
                and left.a[1] in (True, False):
            # two narrow values of one type are compared in that type: both loaded at the comparison, in
            # source order, after both addresses were formed (probe: ``c == G[i][j]`` against ``G[i][j] == c``)
            return right.seq < left.seq
        if loads[0] and not loads[1]:
            return False
        return first_seq(b) < first_seq(a)

    def simple(self, stmt) -> str:
        """An assignment or expression statement as it stands in a ``for`` header."""
        if stmt.k == "assign":
            return self.assignment(stmt.a[0], stmt.a[1], stmt.a[2])
        if stmt.k == "expr":
            return self.unparen(self.expr(stmt.a[0]))
        raise Decline("statement in a loop header")

    def statements(self, stmts: list, depth: int) -> list[str]:
        pad = "    " * depth
        lines = []
        for index, stmt in enumerate(stmts):
            k = stmt.k
            if k == "assign":
                lines.append(f"{pad}{self.assignment(stmt.a[0], stmt.a[1], stmt.a[2])};")
            elif k == "expr":
                lines.append(f"{pad}{self.unparen(self.expr(stmt.a[0]))};")
            elif k == "if":
                lines.append(f"{pad}if ({self.cond(stmt.a[0])}) {{")
                lines += self.statements(stmt.a[1], depth + 1)
                if stmt.a[2] is not None:
                    lines.append(f"{pad}}} else {{")
                    lines += self.statements(stmt.a[2], depth + 1)
                lines.append(f"{pad}}}")
            elif k == "loop":
                head, inner = getattr(self, "loop_head", "for (;;)"), stmt.a[0]
                init = stmt.a[3] if len(stmt.a) > 3 else None
                init_text = ", ".join(self.simple(part) for part in init) if init else ""
                skip = 0
                while skip < len(inner) and inner[skip].k == "label" and inner[skip].a[0] not in self.label_used:
                    skip += 1                                # a label nothing in the text jumps to
                inner = inner[skip:] if head != "for (;;)" else inner
                first = inner[0] if inner else None
                # leaving the last statement of a ``void`` function is the same as returning from it
                last = depth == 1 and all(later.k == "label" and later.a[0] not in self.label_used
                                          for later in stmts[index + 1:])
                leaves = first is not None and first.k == "if" and first.a[2] is None and len(first.a[1]) == 1 and (
                    first.a[1][0].k == "break" or (last and first.a[1][0].k == "return" and first.a[1][0].a[0] is None))
                tail = len(inner)
                while tail and inner[tail - 1].k == "label" and inner[tail - 1].a[0] not in self.label_used:
                    tail -= 1
                closing = inner[tail - 1] if tail > 1 else None
                if closing is not None and not init and closing.k == "if" and closing.a[2] is None \
                        and len(closing.a[1]) == 1 and closing.a[1][0].k == "break" and id(closing) in self.loop_tests:
                    # ``do { ... } while (a && b);``: the last test jumps to the loop's top with an explicit jump
                    # to the false label before it (probe: ``if (!(a && b)) break;`` is turned into
                    # ``!a || !b`` by the parser and branches the other way)
                    lines.append(f"{pad}do {{")
                    lines += self.statements(inner[:tail - 1], depth + 1)
                    lines.append(f"{pad}}} while ({self.cond(closing.a[0][1])});")
                    continue
                if head != "for (;;)" and leaves:
                    # ``for (;;) { if (c) break; ... }`` read as the loop it was
                    test = self.cond(first.a[0][1]) if first.a[0][0] == "not" else f"!({self.cond(first.a[0])})"
                    inner = inner[1:]
                    if init is None:
                        lines.append(f"{pad}while ({test}) {{")
                    else:
                        step, tail = "", len(inner)
                        while tail and inner[tail - 1].k == "label" and inner[tail - 1].a[0] not in self.label_used:
                            tail -= 1
                        if head == "while" and tail and inner[tail - 1].k in ("assign", "expr"):
                            # the last statement as the step: the continue label then stands in
                            # front of it instead of in front of the jump back
                            step, inner = self.simple(inner[tail - 1]), inner[:tail - 1]
                        lines.append(f"{pad}for ({init_text}; {test}; {step}) {{")
                else:
                    lines.append(f"{pad}for ({init_text};;) {{")
                lines += self.statements(inner, depth + 1)
                lines.append(f"{pad}}}")
            elif k == "switch":
                lines.append(f"{pad}switch ({self.unparen(self.expr(stmt.a[0]))}) {{")
                for labels, inner in stmt.a[1]:
                    for value in labels:
                        lines.append(f"{pad}default:" if value is None else f"{pad}case {self.number(value)}:")
                    lines += self.statements(inner, depth + 1)
                lines.append(f"{pad}}}")
            elif k == "label":
                # a label that heads a loop is recorded both in front of the loop and as its first statement:
                # written once (in front, where a jump to the loop's top lands)
                written = self.__dict__.setdefault("_labels_written", set())
                if stmt.a[0] in self.label_used and stmt.a[0] not in written:
                    written.add(stmt.a[0])
                    lines.append(f"L{stmt.a[0]}: ;")
            elif k == "goto":
                lines.append(f"{pad}goto L{stmt.a[0]};")
            elif k in ("break", "continue"):
                lines.append(f"{pad}{k};")
            elif k == "scope":
                lines.append(f"{pad}{{")
                lines.append(f"{pad}    register s32 {stmt.a[0]};")
                lines += self.statements(stmt.a[1], depth + 1)
                lines.append(f"{pad}}}")
            elif k == "copy":
                kind = stmt.a[3] if len(stmt.a) > 3 else "r"
                lines.append(f"{pad}{self.block_operand(stmt.a[0], stmt.a[2], kind)} = "
                             f"{self.block_operand(stmt.a[1], stmt.a[2], kind)};")
            elif k == "return":
                value = stmt.a[0]
                lines.append(f"{pad}return {self.unparen(self.expr(value))};" if value is not None
                             else f"{pad}return;")
            else:
                raise Decline(f"statement {k}")
        return lines

    def locals_text(self) -> list[str]:
        """Declarations in ascending frame order, with fillers: GCC gives the first declared local
        the lowest slot, aligns each to its size and rounds aggregates to eight bytes."""
        lines, cursor, pads = [], self.args_size, 0
        end = self.args_size + self.var_size

        def fill(until):
            nonlocal cursor, pads
            while cursor < until:
                gap = until - cursor
                if cursor % 8 == 0 and gap >= 8:
                    lines.append(f"    s32 pad{pads}[{2 * (gap // 8)}];")
                    cursor += 8 * (gap // 8)
                elif cursor % 4 == 0 and gap >= 4:
                    lines.append(f"    s32 pad{pads};")
                    cursor += 4
                elif cursor % 2 == 0 and gap >= 2:
                    lines.append(f"    s16 pad{pads};")
                    cursor += 2
                else:
                    lines.append(f"    s8 pad{pads};")
                    cursor += 1
                pads += 1

        ordered = sorted(self.vars.values(), key=lambda v: v["off"])
        seen_local = False
        for position, var in enumerate(ordered):
            off = var["off"]
            following = ordered[position + 1]["off"] if position + 1 < len(ordered) else end
            if var.get("aggregate"):
                room = following - off
                if off % 8 == 0 and room >= 8:
                    var["size"], var["decl"] = 8 * (room // 8), f"s32 {self.var_name(var)}[{2 * (room // 8)}]"
                elif off % 4 == 0 and room >= 4:
                    var["size"], var["decl"] = 4, f"s32 {self.var_name(var)}"
                elif off % 2 == 0 and room >= 2:
                    var["size"], var["decl"] = 2, f"s16 {self.var_name(var)}"
                else:
                    var["size"], var["decl"] = 1, f"s8 {self.var_name(var)}"
            size = var["size"]
            align = 8 if var.get("aggregate") and size >= 8 else min(size, 8)
            if var.get("param"):
                if seen_local or (cursor + align - 1) // align * align != off:
                    raise Decline("parameter slot is not first in the frame")
                cursor = off + size
                continue
            seen_local = True
            if off < cursor:
                raise Decline("overlapping locals")
            if (cursor + align - 1) // align * align != off:
                fill(off)
            if (cursor + align - 1) // align * align != off:
                raise Decline("local cannot be placed")
            lines.append(f"    {var.get('decl') or var['type'] + ' ' + self.var_name(var)};")
            cursor = off + size
        temps = sorted(set(self.temp_at.values()))
        if temps:
            # The temporary is the 64-bit product's slot: eight bytes, aligned to eight, the word that is
            # stored and reloaded in its second half.  It follows the declared locals, which stop before it.
            slot = temps[0] - 4
            if slot % 8 or cursor > slot or any(off in self.vars for off in temps):
                raise Decline("compiler temporary inside the locals")
            if (cursor + 7) // 8 * 8 != slot:
                fill(slot - 7)
            if (cursor + 7) // 8 * 8 != slot:
                raise Decline("compiler temporary cannot be placed")
            return lines
        if cursor > end:
            raise Decline("locals exceed the frame")
        if (cursor - self.args_size + 7) // 8 * 8 < self.var_size:
            fill(end - 7 if (end - 7) > cursor else cursor + 1)
            if (cursor - self.args_size + 7) // 8 * 8 != self.var_size:
                fill(end)
        return lines

    def _stepped_pointers(self) -> None:
        """``x = p++`` with a step of several bytes: ``p`` is a pointer to elements of that size.
        Only a typed pointer's ``++`` reads the old value and stores the step in one statement
        (a display-list or buffer pointer advanced by a macro), so the variable is declared as one."""
        self.stepped: dict[tuple, int] = {}
        stack = list(self.body)
        while stack:
            part = stack.pop()
            if isinstance(part, (list, tuple)):
                stack.extend(part)
            elif isinstance(part, (Stmt, Node)):
                if isinstance(part, Node) and part.k == "postinc" and abs(part.a[1]) != 1 and part.a[2] == 4 \
                        and part.a[0][0] in ("global", "local"):
                    key = (part.a[0][0], part.a[0][1])
                    if self.stepped.setdefault(key, abs(part.a[1])) != abs(part.a[1]):
                        raise Decline("one pointer stepped by two sizes")
                if isinstance(part, Stmt) and part.k == "assign" and part.a[2] == 4 \
                        and part.a[0][0] in ("global", "local") and self.pointer_step(part.a[0], part.a[1]):
                    key, size = (part.a[0][0], part.a[0][1]), abs(self.pointer_step(part.a[0], part.a[1]))
                    if self.stepped.setdefault(key, size) != size:
                        raise Decline("one pointer stepped by two sizes")
                stack.extend(part.a)
        for (kind, where), size in self.stepped.items():
            if kind == "local":
                var = self.vars.get(where)
                if var is None or var.get("array") or var.get("aggregate") or var["size"] != 4 or where >= self.size:
                    raise Decline("stepped pointer that is not a plain local")
                var["decl"] = f"struct n64dk_e{size} *{self.var_name(var)}"

    @staticmethod
    def pointer_step(loc, value: Node) -> int:
        """The step of ``p++;`` on a pointer variable to elements of several bytes: the old value plus the
        element size goes through a ``move`` before it is stored (probe: ``p += 1``, ``p = p + 1`` and
        integer arithmetic have no ``move``; an integer's ``x++`` steps by one).  0 when it is not that."""
        inner = strip(value)
        if value.k == "cp" and inner.k == "bin" and inner.a[0] == "+" and abs(const_of(inner.a[2]) or 0) > 1 \
                and strip(inner.a[2]).k == "const" and strip(inner.a[1]).k == "ld" and strip(inner.a[1]).a[2] == loc \
                and strip(inner.a[1]).a[0] == 4:
            return const_of(inner.a[2])
        return 0

    def _pointer_locals(self) -> None:
        """Plain four-byte locals assigned ``&p->m[i]`` / ``&SYM[i]`` in pointer order are pointers."""
        stack = list(self.body)
        while stack:
            part = stack.pop()
            if isinstance(part, (list, tuple)):
                stack.extend(part)
            elif isinstance(part, Stmt):
                if part.k == "assign" and part.a[0][0] == "local" and part.a[2] == 4:
                    n = strip(part.a[1])
                    if n.k == "bin" and n.a[0] == "+":
                        base, rest = strip(n.a[1]), strip(n.a[2])
                        member = rest.k == "bin" and rest.a[0] == "+" and strip(rest.a[1]).k == "mul" \
                            and (const_of(rest.a[2]) or 0) > 0 and const_of(base) is None and base.k != "sym" \
                            and first_seq(rest) < first_seq(base)
                        if member or (base.k == "sym" and rest.k == "mul"):
                            var = self.vars.get(part.a[0][1])
                            if var is not None and var["size"] == 4 and not var.get("array") \
                                    and not var.get("aggregate") and not var.get("param") \
                                    and ("local", part.a[0][1]) not in self.stepped:
                                self.pointer_locals.add(part.a[0][1])
                                var["decl"] = f"void *{self.var_name(var)}"
                stack.extend(part.a)

    @staticmethod
    def _sublists(stmt) -> list:
        if stmt.k == "if":
            return [stmt.a[1]] + ([stmt.a[2]] if stmt.a[2] is not None else [])
        if stmt.k == "loop":
            return [stmt.a[0]]
        if stmt.k == "switch":
            return [inner for _labels, inner in stmt.a[1]]
        if stmt.k == "scope":
            return [stmt.a[1]]
        return []

    def _enclosing(self, stmts: list, low: int, first: int, last: int):
        """(list, a, b, low): the innermost statement list whose statements a..b hold the instructions
        first..last, and the index of the list's first instruction; or None."""
        a = b = None
        for position, stmt in enumerate(stmts):
            if stmt.end is None:
                continue
            if a is None and first < stmt.end:
                a = position
            if last < stmt.end:
                b = position
                break
        if a is None or b is None or first < low:
            return None
        if a == b:
            before = [s.end for s in stmts[:a] if s.end is not None]
            start = before[-1] if before else low
            for inner in self._sublists(stmts[a]):
                ends = [s.end for s in inner if s.end is not None]
                inner_low = self.list_start.get(id(inner), start)
                if ends and inner_low <= first and last < ends[-1]:
                    found = self._enclosing(inner, inner_low, first, last)
                    if found is not None:
                        return found
        return stmts, a, b, low

    def _scopes(self) -> list[str]:
        """Give every scratch variable the block it lives in: ``{ register s32 v; ... }`` around the
        statements that show it alive, widened to the whole of the statement list they stand in where
        the register is no temporary there (the block the source most likely had).  A block of its own
        costs no instruction.  The allocator serves the longest life first, so of two variables alive
        together the one in the lower register gets the outer block, or the earlier declaration.
        Returns the names declared at the head of the function (those alive throughout it)."""
        rank = {reg: n for n, reg in enumerate(sorted(SCRATCH))}
        outer = []
        order = sorted(self.instances, key=lambda inst: (inst["start"] - inst["extent"], rank[inst["reg"]]))
        for inst in order:
            found = self._enclosing(self.body, 0, inst["start"], inst["extent"])
            if found is None:
                outer.append(inst)
                continue
            stmts, a, b, low = found
            ends = [stmt.end for stmt in stmts if stmt.end is not None]
            if ends and ends[-1] <= inst["upto"]:
                b = len(stmts) - 1                             # to the end of the list: no temporary there
            if low >= inst["free"]:
                a = 0                                          # and from its head
            if stmts is self.body and a == 0 and all(later.k == "label" for later in stmts[b + 1:]):
                outer.append(inst)
                continue
            block = Stmt("scope", inst["name"], stmts[a:b + 1])
            block.end = stmts[b].end
            stmts[a:b + 1] = [block]
        return [inst["name"] for inst in sorted(outer, key=lambda inst: rank[inst["reg"]])]

    def _sink_labels(self, stmts: list) -> None:
        """A label written directly after an ``if`` statement meets the compiler's own end-of-if label, and
        GCC puts a no-op between two labels (``emit_nop`` before a user label).  The same address is
        reached by writing the label as the last thing inside the ``if``: no label stands before it there."""
        position = 0
        while position < len(stmts):
            stmt = stmts[position]
            if stmt.k == "label" and stmt.a[0] in self.label_used and stmt.a[0] not in self.hard_labels:
                before = position - 1
                while before >= 0 and stmts[before].k == "label" and stmts[before].a[0] not in self.label_used:
                    before -= 1
                if before >= 0 and stmts[before].k == "if":
                    block = stmts[before].a[2] if stmts[before].a[2] is not None else stmts[before].a[1]
                    while block and block[-1].k == "if":
                        last = block[-1]
                        block = last.a[2] if last.a[2] is not None else last.a[1]
                    if not block or block[-1].k not in ("loop", "switch"):
                        block.append(stmts.pop(position))
                        continue
            position += 1
        for stmt in stmts:
            if stmt.k == "if":
                self._sink_labels(stmt.a[1])
                if stmt.a[2] is not None:
                    self._sink_labels(stmt.a[2])
            elif stmt.k == "loop":
                self._sink_labels(stmt.a[0])
            elif stmt.k == "switch":
                for _labels, inner in stmt.a[1]:
                    self._sink_labels(inner)

    def _starts_with_noop_loop(self) -> bool:
        """Whether the function's first instruction is the no-op in front of a loop that is its first
        statement and holds every use of the ``register`` variables."""
        if not self.ins or self.ins[0].op != "nop":
            return False
        opening = [stmt for stmt in self.body if not (stmt.k == "label" and stmt.a[0] not in self.label_used)]
        if not opening or opening[0].k != "loop":
            return False
        stack = list(opening[1:]) + list((opening[0].a[3] if len(opening[0].a) > 3 else None) or ())   # the rest, the initialiser
        while stack:
            part = stack.pop()
            if isinstance(part, tuple) and len(part) == 2 and part[0] == "regvar":
                return False
            if isinstance(part, (list, tuple)):
                stack.extend(part)
            elif isinstance(part, (Stmt, Node)):
                stack.extend(part.a)
        return True

    def _address_temporaries(self):
        """A saved register that only ever holds the address of the very next store, across a call in the
        stored value, is not a variable: it is the compiler's own temporary for ``A[i][j] = x * f(y)`` when
        the call was not expanded ahead of the statement.  GCC expands the calls of an assignment to a
        complex place first (``preexpand_calls``) unless the call is to one of its built-ins (``sqrt``, ``sin``,
        ``cos``) or stands inside a statement expression; the callee has no name here, so the call is written
        ``({ f(y); })``, which gives the same instructions (measured both ways)."""
        found: dict[int, list] = {}

        def walk(stmts):
            for at in range(len(stmts) - 1):
                first, second = stmts[at], stmts[at + 1]
                if first.k == "assign" and first.a[0][0] == "regvar" and second.k == "assign" \
                        and second.a[0][0] == "deref":
                    base = strip(second.a[0][1])
                    if base.k == "ld" and base.a[2] == first.a[0] and has_call(second.a[1]):
                        found.setdefault(first.a[0][1], []).append((stmts, first, second))
            for stmt in stmts:
                if stmt.k == "if":
                    walk(stmt.a[1])
                    if stmt.a[2] is not None:
                        walk(stmt.a[2])
                elif stmt.k == "loop":
                    walk(stmt.a[0])
                elif stmt.k == "switch":
                    for _labels, inner in stmt.a[1]:
                        walk(inner)

        walk(self.body)
        for reg, pairs in found.items():
            var = self.regvars.get(reg)
            if var is None or "param" in var or self.regvar_writes.get(reg) != len(pairs) \
                    or self.regvar_reads.get(reg) != len(pairs):
                continue
            for stmts, first, second in pairs:
                second.a = (("deref", first.a[1], second.a[0][2]),) + tuple(second.a[1:])
                stmts.remove(first)
                stack = [second.a[1]]
                while stack:
                    part = stack.pop()
                    if isinstance(part, Node):
                        if part.k == "call":
                            self.unexpanded.add(id(part))
                        stack.extend(part.a)
                    elif isinstance(part, (tuple, list)):
                        stack.extend(part)
            del self.regvars[reg]

    def render(self, prelude: str, names: dict[int, str]) -> str:
        self.names = names
        self._address_temporaries()
        for reg in self.saved_temps_seen:                     # held only values kept across a call: no variable
            if reg in self.regvars and "param" not in self.regvars[reg] and not self.regvar_reads.get(reg) \
                    and not self.regvar_writes.get(reg):
                del self.regvars[reg]
        self._stepped_pointers()
        self._pointer_locals()
        self._sink_labels(self.body)
        for target, position, arg in self.flex_args:
            # a word loaded from a global into a ``float`` parameter's register: the global is a float
            proto = self.callee_protos.get(target)
            if proto is not None and position < len(proto) and proto[position] == "f32" and arg.k == "ld" \
                    and arg.a[2][0] in ("gidx", "global"):
                where = self.indexed if arg.a[2][0] == "gidx" else self.globals
                acc = where[arg.a[2][1]]["acc"]
                if (4, "s") in acc:
                    acc[acc.index((4, "s"))] = (4, "f")
        self._indexed_layout()
        outer = self._scopes()
        self._labels_written = set()
        body = self.statements(self.body, 1)
        if len(self.literal_addrs) > 1:
            self.literal_span = (min(self.literal_addrs), max(self.literal_addrs))
        self.literal_pool = self._literal_pool()
        if self.literal_span is not None or self.literal_pool:
            self._labels_written = set()
            body = self.statements(self.body, 1)
        local_lines = self.locals_text()
        params = []
        for index, param in enumerate(self.params):
            kind = param["kind"]
            if kind in ("int", "unused", "float", "double"):
                params.append(f"{param.get('type', 's32')} arg{index}")
            elif kind == "pad":
                continue
            elif kind == "narrow":
                params.append(f"{self.vars[param['off']]['type']} arg{index}")
            elif kind == "stack-narrow":
                params.append(f"{param['type']} arg{index}")
            elif kind == "register":
                params.append(f"register s32 arg{index}")
            elif kind == "hidden":
                continue
            else:
                raise Decline("parameter kind")
        out = [prelude.rstrip("\n"), ""]
        for size in sorted(self.block_sizes):
            out.append(f"struct n64dk_r{size} {{ s32 w[{size // 4}]; }};")
        for kind, size in sorted(self.loose_blocks):
            out.append(f"struct n64dk_{kind}{size} {{ s16 w[{size // 2}]; }};" if kind == "u"
                       else f"struct n64dk_{kind}{size} {{ u8 w[{size}]; }};")
        for offset, size in sorted(self.member_arrays):
            out.append(f"struct n64dk_m{offset:X}_{size:X} {{ u8 pad[0x{offset:X}]; struct {{ u8 b[0x{size:X}]; }} m[1]; }};")
        for size in sorted(self.struct_args):
            out.append(f"struct n64dk_arg{size} {{ s32 w[{size // 4}]; }};")
        for off, ctype in sorted(self.fields):
            pad = f"u8 pad[0x{off:X}]; " if off else ""
            out.append(f"struct gd_{off:X}_{ctype} {{ {pad}{ctype} v; }};")
        for size in sorted(set(self.stepped.values()) | self.element_sizes):
            out.append(f"struct n64dk_e{size} {{ u8 b[{size}]; }};")
        for offset, step in sorted(self.pointer_fields):
            pad = f"u8 pad[0x{offset:X}]; " if offset else ""
            out.append(f"struct n64dk_p{offset:X}_{step:X} {{ {pad}struct n64dk_e{step} *v; }};")
        for offset, ctype in sorted(self.scalar_arrays):
            out.append(f"struct n64dk_f{offset:X}_{ctype} {{ u8 pad[0x{offset:X}]; {ctype} m[1]; }};")
        sizes_of = {"s8": 1, "u8": 1, "s16": 2, "u16": 2, "s32": 4, "u32": 4, "f32": 4, "f64": 8}
        for offset, size, inside, ctype in sorted(self.struct_arrays):
            lead = f"u8 pad[0x{offset:X}]; " if offset else ""
            before = f"u8 a[0x{inside:X}]; " if inside else ""
            rest = size - inside - sizes_of[ctype]
            after = f" u8 z[0x{rest:X}];" if rest else ""
            out.append(f"struct n64dk_g{offset:X}_{size:X}_{inside:X}_{ctype} {{ {lead}struct {{ {before}{ctype} v;{after} }} m[1]; }};")
        for offset, row, ctype in sorted(self.member_grids):
            lead = f"u8 pad[0x{offset:X}]; " if offset else ""
            out.append(f"struct n64dk_h{offset:X}_{row:X}_{ctype} {{ {lead}{ctype} m[1][0x{row:X}]; }};")
        for offset in sorted(self.void_fields):
            pad = f"u8 pad[0x{offset:X}]; " if offset else ""
            out.append(f"struct n64dk_v{offset:X} {{ {pad}void *v; }};")
        if self.indexed and not hasattr(self, "member_of"):
            self._indexed_layout()
        sizes = {"s8": 1, "u8": 1, "s16": 2, "u16": 2, "s32": 4, "u32": 4, "f32": 4, "f64": 8}
        for base, array in sorted(getattr(self, "arrays", {}).items()):
            members, cursor = [], 0
            for off, ctype in sorted(array["fields"].items()):
                if off > cursor:
                    members.append(f"u8 pad{cursor:X}[0x{off - cursor:X}];")
                members.append(f"{ctype} m{off:X};")
                cursor = off + sizes[ctype]
            if array["size"] > cursor:
                members.append(f"u8 pad{cursor:X}[0x{array['size'] - cursor:X}];")
            out.append(f"struct n64dk_a{base:08X} {{ {' '.join(members)} }};")
            shape = f"[{array['cols']}]" if array["cols"] else ""
            out.append(f"extern struct n64dk_a{base:08X} {self.symbol(base)}[]{shape};")
        for addr, (base, size, lead) in sorted(self.nested.items()):
            out.append(f"struct n64dk_o{base:08X} {{ u8 pad[0x{lead:X}]; struct {{ u8 b[0x{size - lead:X}]; }} a; }};")
            out.append(f"extern struct n64dk_o{base:08X} {self.symbol(base)}[];")
        for addr in sorted(self.globals):
            name = self.symbol(addr)
            if name == self.name or name in self.callees or addr in getattr(self, "arrays", {}):
                continue
            if addr in self.grids:
                out.append(f"extern {self.grids[addr][0]} {name}[][{self.grids[addr][1]}];")
                continue
            if ("global", addr) in self.stepped:
                out.append(f"extern struct n64dk_e{self.stepped[('global', addr)]} *{name};")
                continue
            out.append(f"extern {self.global_type(addr)} {name};")
        for name, info in sorted(self.callees.items()):
            if name != self.name:
                # a ``float`` passed as a float (not widened to double) means a prototype was in scope
                out.append(f"{info['ret']} {name}({', '.join(info.get('proto') or [])});")
        ret = self.return_type if self.returns_value else "void"
        out.append("")
        if self.variadic:
            if not params:
                raise Decline("variadic function without a named parameter")
            params.append("...")
        out.append(f"{ret} {self.name}({', '.join(params) or 'void'}) {{")
        register_lines = [f"    register s32 {var['name']};" for reg, var in sorted(self.regvars.items())
                          if "param" not in var]
        register_lines += [f"    register s32 {name};" for name in outer]
        if register_lines and self._starts_with_noop_loop():
            # The function opens with the no-op of a loop that is its first statement.  A ``register``
            # declaration in front of that loop takes the no-op away (measured), so the variables were
            # declared further in: in the loop's own block, where a declaration changes no instruction.
            at = next(k for k, line in enumerate(body) if line.rstrip().endswith("{"))
            body[at + 1:at + 1] = ["    " + line for line in register_lines]
            register_lines = []
        local_lines = register_lines + local_lines
        out += local_lines
        if local_lines:
            out.append("")
        out += body
        out.append("}")
        return "\n".join(out) + "\n"


_NAMED = re.compile(r"\b(?:func|D|jtbl)_([0-9A-Fa-f]{8})\b")


def prelude_of(source: str) -> str:
    """The typedef/macro head the kit compiles every candidate with."""
    cut = source.rfind("#endif")
    if "N64DK_M2C_PRELUDE" in source and cut >= 0:
        return source[:cut + len("#endif")]
    from .autodraft import PRELUDE
    return PRELUDE


def names_in(source: str) -> dict[int, str]:
    """Address-bearing names the draft uses: the same address must get the same name."""
    names = {}
    for found in _NAMED.finditer(source):
        names.setdefault(int(found.group(1), 16), found.group(0))
    return names


def is_unoptimised_gcc(code: bytes) -> bool:
    """``addu $fp,$sp,$zero`` among the first instructions: GCC's frame pointer set-up."""
    head = code[:64]
    return any(head[i:i + 4] == b"\x03\xa0\xf0\x21" for i in range(0, len(head) - 3, 4))


# A loop is read as ``for (;;) { if (c) break; ... }``.  That has the instructions of the loop that
# was written and one label more: ``for (;;)`` puts a continue label in front of the jump back, a
# ``while (!c)`` has it at the top.  An assembler that fills delay slots never moves an instruction
# across a label, so only the second lets the last instruction of the body into the back jump's
# slot; the lane that fills no slots builds the same bytes from both.
LOOP_HEADS = ("for (;;)", "while", "for-nostep")


def translate(source: str, name: str, code: bytes, vram: int, loop_head: str = LOOP_HEADS[0], read=None) -> str:
    try:
        function = Function(code, vram, name, read)
        function.loop_head = loop_head
        return function.render(prelude_of(source), names_in(source))
    except Decline as first:
        if "with no known value" not in str(first):
            raise
        # a register read that nothing in its block wrote: ``register`` variables in scratch registers
        forced: set[tuple[int, int]] = set()
        while True:
            try:
                function = Function(code, vram, name, read, scratch=True, forced=frozenset(forced))
                function.loop_head = loop_head
                return function.render(prelude_of(source), names_in(source))
            except Promote as found:
                if found.key in forced or len(forced) >= 256:
                    raise first from None
                forced.add(found.key)                         # one more variable that lives inside a block
            except Decline:
                raise first from None


def decline_reason(name: str, code: bytes, vram: int, read=None) -> str | None:
    """Why this translator does not read the function (addresses and counts left out, so equal reasons are
    equal texts), or None when it reads it or the code is not unoptimised GCC.  The run keeps this with the
    case: it is the next thing to teach the translator, where "far-off draft" says nothing."""
    if not is_unoptimised_gcc(code):
        return None
    try:
        translate("", name, code, vram, LOOP_HEADS[0], read)
        return None
    except Decline as reason:
        return re.sub(r"0x[0-9A-Fa-f]+|\b\d+\b", "N", str(reason))[:120]
    except (KeyError, IndexError, ValueError, TypeError, AttributeError, RecursionError) as error:
        return f"internal: {type(error).__name__}"


def variants(source: str, name: str, *, code: bytes, vram: int, pipeline: str, flags: tuple[str, ...] = (),
             family: str | None = None, read=None) -> list[tuple[str, str]]:
    """[(label, complete C translation unit)]; [] when this is not unoptimised GCC code."""
    if not is_unoptimised_gcc(code):
        return []
    out, seen = [], set()
    for head in LOOP_HEADS:
        try:
            text = translate(source, name, code, vram, head, read)
        except Decline:
            return out
        except (KeyError, IndexError, ValueError, TypeError, AttributeError, RecursionError):
            return out
        if text not in seen:
            seen.add(text)
            out.append(("gcc-O0-translated" + ("" if head == LOOP_HEADS[0] else "+" + head), text))
    return out
