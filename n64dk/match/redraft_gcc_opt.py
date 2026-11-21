"""Rewrites of a decompiler draft for code built by GCC 2.x with optimisation (-O1/-O2).

An optimising GCC keeps nothing of the source's layout in the frame, so what separates a draft
from the original is not where the locals live but which *source form* was written: the
decompiler renders the instructions it saw, and several of its renderings are not the C that
produces those instructions.  Every rule here is decided by the target's machine code and
rewrites the draft's syntax tree; nothing is searched for.

Rules (each is described where it is implemented):

``bytes``    pointer arithmetic the decompiler wrote in bytes on a pointer whose type has a
             larger element (``&D_80089A6C + (i * 0x4C)`` with ``D`` an ``s32``): C scales it,
             the target does not.  The arithmetic is made byte-wise; a dereference takes the
             width of the load or store the target performs on that address.
``symbols``  a constant the target builds with ``lui``/``addiu``.  GCC builds an integer with
             ``lui``/``ori`` and only an address with ``lui``/``addiu``, so the constant is the
             address of a symbol the linker placed (a ROM offset, a segment address).
``passed``   a call for which the target sets a higher argument register and leaves a lower one
             exactly as the function received it: the lower one is passed through.  The
             decompiler drops it; the call gets it back and the function its parameter.
"""
from __future__ import annotations

import copy
import re
import threading

try:                                                        # the parser is optional for the kit
    from pycparser import c_ast, c_generator, c_parser
except Exception:                                           # pragma: no cover
    c_ast = c_generator = c_parser = None

from ..analysis import flow as flowmod

MAX_VARIANTS = 10
_TYPE_ARGUMENT = {"M2C_FIELD": 1, "M2C_BITWISE": 0}         # macros that take a type name as an argument
_LOCK = threading.Lock()
_PARSER = None
_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
_LOADS = {"lb": "s8", "lbu": "u8", "lh": "s16", "lhu": "u16", "lw": "s32", "lwu": "u32", "lwc1": "f32",
          "ldc1": "f64", "ld": "s64"}
_STORES = {"sb": 1, "sh": 2, "sw": 4, "swc1": 4, "sdc1": 8, "sd": 8}
_SIZES = {"s8": 1, "u8": 1, "s16": 2, "u16": 2, "s32": 4, "u32": 4, "f32": 4, "f64": 8, "s64": 8, "u64": 8}


# --------------------------------------------------------------------------------- situation
def _optimised_gcc(pipeline: str, flags: tuple[str, ...], family: str | None, insns) -> bool:
    """GCC family, and the code is not an unoptimised build (which addresses its frame through
    ``$fp`` set from ``$sp`` in the prologue)."""
    if family == "ido" or (family is None and pipeline.startswith("ido")):
        return False
    for insn in insns[:12]:
        word = insn.word
        # addu/or/daddu $fp, $sp, $zero
        if word >> 26 == 0 and (word >> 11) & 31 == 30 and (word & 0x3F) in (0x21, 0x25, 0x2D) and \
                {(word >> 21) & 31, (word >> 16) & 31} == {29, 0}:
            return False
    if family is None and not pipeline.startswith(("kmc", "sn", "gcc", "custom")):
        return False
    return True


# --------------------------------------------------------------------------------- the draft
def _blank_comments(text: str) -> str:
    return _COMMENT.sub(lambda found: re.sub(r"[^\n]", " ", found.group(0)), text)


def _split_arguments(text: str) -> list[tuple[int, int]]:
    """Spans of the top-level comma separated parts of ``text``."""
    out, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            out.append((start, index))
            start = index + 1
    out.append((start, len(text)))
    return out


def _wrap_type_arguments(text: str) -> str | None:
    """``M2C_FIELD(e, T, off)`` -> ``M2C_FIELD(e, (T)0, off)`` so that a C parser accepts it."""
    found = [m for m in re.finditer(r"\b(M2C_FIELD|M2C_BITWISE)\s*\(", text)]
    for match in reversed(found):                           # inner calls first
        depth, position = 1, match.end()
        while position < len(text) and depth:
            depth += (text[position] == "(") - (text[position] == ")")
            position += 1
        if depth:
            return None
        inside = text[match.end():position - 1]
        spans = _split_arguments(inside)
        index = _TYPE_ARGUMENT[match.group(1)]
        if index >= len(spans):
            return None
        a, b = spans[index]
        inside = f"{inside[:a]} ({inside[a:b].strip()})0{inside[b:]}"
        text = text[:match.end()] + inside + text[position - 1:]
    return text


class _Generator(c_generator.CGenerator if c_generator else object):
    """Prints the type-taking macros back in their macro form."""

    def visit_FuncCall(self, n):
        name = n.name.name if isinstance(n.name, c_ast.ID) else None
        if name in _TYPE_ARGUMENT and n.args is not None:
            parts = []
            for index, argument in enumerate(n.args.exprs):
                if index == _TYPE_ARGUMENT[name] and isinstance(argument, c_ast.Cast):
                    parts.append(self._generate_type(argument.to_type, emit_declname=False))
                else:
                    parts.append(self._visit_expr(argument))
            return f"{name}({', '.join(parts)})"
        return super().visit_FuncCall(n)


def _parse(text: str):
    global _PARSER
    with _LOCK:
        if _PARSER is None:
            _PARSER = c_parser.CParser()
        return _PARSER.parse(text, "<draft>")


def _type(names: str, pointers: int = 0):
    """A type node for ``names`` (space separated identifier words) with ``pointers`` stars."""
    node = c_ast.TypeDecl(None, [], None, c_ast.IdentifierType(names.split()))
    for _ in range(pointers):
        node = c_ast.PtrDecl([], node)
    return node


def _typename(node) -> "c_ast.Typename":
    return c_ast.Typename(None, [], None, _undeclared(node))


def _undeclared(node):
    """A copy of a declaration's type without the declared name (for a cast)."""
    node = copy.deepcopy(node)
    cursor = node
    while cursor is not None:
        if isinstance(cursor, c_ast.TypeDecl):
            cursor.declname = None
            break
        cursor = getattr(cursor, "type", None)
    return node


def _cast(type_node, expression):
    return c_ast.Cast(_typename(type_node), expression)


def _number(node) -> int | None:
    """Value of an integer constant expression made of a literal and unary minus."""
    if isinstance(node, c_ast.UnaryOp) and node.op == "-":
        inner = _number(node.expr)
        return None if inner is None else -inner
    if isinstance(node, c_ast.Constant) and node.type in ("int", "long int", "unsigned int", "unsigned long int",
                                                          "long long int", "unsigned long long int"):
        try:
            return int(node.value.rstrip("uUlL"), 0)
        except ValueError:
            return None
    return None


def _function_span(source: str, name: str) -> tuple[int, int]:
    """(start of the definition's first line, offset of its closing brace)."""
    blank = _blank_comments(source)
    found = re.search(rf"(?m)^[A-Za-z_][\w \t\*]*?\b{re.escape(name)}\s*\([^;{{}}]*\)\s*\{{", blank)
    if found is None:
        raise ValueError("function not found")
    depth, position = 0, found.end() - 1
    while position < len(blank):
        depth += (blank[position] == "{") - (blank[position] == "}")
        if depth == 0:
            return found.start(), position
        position += 1
    raise ValueError("function not closed")


class Draft:
    """A draft split into the text before the function, the function's syntax tree and the rest."""

    def __init__(self, source: str, name: str):
        if c_parser is None:
            raise ValueError("no parser")
        start, close = _function_span(source, name)
        self.name = name
        self.head, self.tail = source[:start], source[close + 1:]
        function = source[start:close + 1]
        # everything up to the last preprocessor line is kept verbatim; typedefs in it are read
        lines = self.head.split("\n")
        last = max((i for i, line in enumerate(lines) if line.lstrip().startswith("#")), default=-1)
        prelude, declarations = "\n".join(lines[:last + 1]), "\n".join(lines[last + 1:])
        typedefs = " ".join(re.findall(r"\btypedef\b[^;{}#]*;", _blank_comments(prelude)))
        text = _wrap_type_arguments(_blank_comments(declarations + "\n" + function))
        if text is None or "#" in text:
            raise ValueError("not parseable")
        self.ast = _parse(typedefs + "\n" + text)
        self.typedefs: dict[str, object] = {}
        self.structs: dict[str, object] = {}
        self.globals: dict[str, object] = {}
        self.function = None
        for item in self.ast.ext:
            self._collect_structs(item)
            if isinstance(item, c_ast.Typedef):
                self.typedefs[item.name] = item.type
            elif isinstance(item, c_ast.FuncDef):
                if item.decl.name == name:
                    self.function = item
                self.globals[item.decl.name] = item.decl.type
            elif isinstance(item, c_ast.Decl) and item.name:
                self.globals.setdefault(item.name, item.type)
        if self.function is None or self.function.body is None:
            raise ValueError("function not parsed")
        self.added: list[str] = []                           # declarations to place before the function
        self.dropped: set[str] = set()                       # global declarations replaced by ``added``
        self.locals: dict[str, object] = {}
        self._collect_locals()

    # -- declarations
    def _collect_structs(self, node) -> None:
        if isinstance(node, (c_ast.Struct, c_ast.Union)) and node.decls is not None and node.name:
            self.structs.setdefault(node.name, node)
        for _name, child in node.children():
            self._collect_structs(child)

    def _collect_locals(self) -> None:
        self.locals = {}
        arguments = self.function.decl.type.args
        for parameter in (arguments.params if arguments else []):
            if isinstance(parameter, c_ast.Decl) and parameter.name:
                self.locals[parameter.name] = parameter.type

        def walk(node):
            if isinstance(node, c_ast.Decl) and node.name and not isinstance(node.type, c_ast.FuncDecl):
                self.locals.setdefault(node.name, node.type)
            for _name, child in node.children():
                walk(child)
        walk(self.function.body)

    def parameters(self) -> list:
        arguments = self.function.decl.type.args
        out = []
        for parameter in (arguments.params if arguments else []):
            if isinstance(parameter, c_ast.Typename) or (isinstance(parameter, c_ast.Decl) and not parameter.name):
                continue                                     # (void)
            out.append(parameter)
        return out

    # -- types
    def resolve(self, node, depth: int = 0):
        while depth < 30 and isinstance(node, c_ast.TypeDecl) and isinstance(node.type, c_ast.IdentifierType) \
                and len(node.type.names) == 1 and node.type.names[0] in self.typedefs:
            node, depth = self.typedefs[node.type.names[0]], depth + 1
        if isinstance(node, c_ast.Typename):
            return self.resolve(node.type, depth + 1)
        return node

    def size_of(self, node, depth: int = 0) -> int | None:
        node = self.resolve(node)
        if node is None or depth > 12:
            return None
        if isinstance(node, c_ast.PtrDecl):
            return 4
        if isinstance(node, c_ast.ArrayDecl):
            count, element = _number(node.dim) if node.dim is not None else None, self.size_of(node.type, depth + 1)
            return None if count is None or element is None else count * element
        if isinstance(node, c_ast.TypeDecl):
            inner = node.type
            if isinstance(inner, c_ast.IdentifierType):
                names = inner.names
                if "void" in names or "char" in names:
                    return 1
                if "short" in names:
                    return 2
                if "double" in names or names.count("long") >= 2:
                    return 8
                return 4
            if isinstance(inner, c_ast.Enum):
                return 4
            if isinstance(inner, (c_ast.Struct, c_ast.Union)):
                layout = self.layout(inner, depth + 1)
                return None if layout is None else layout[0]
        return None

    def align_of(self, node, depth: int = 0) -> int | None:
        node = self.resolve(node)
        if isinstance(node, c_ast.ArrayDecl):
            return self.align_of(node.type, depth + 1)
        if isinstance(node, c_ast.TypeDecl) and isinstance(node.type, (c_ast.Struct, c_ast.Union)):
            layout = self.layout(node.type, depth + 1)
            return None if layout is None else layout[1]
        size = self.size_of(node, depth + 1)
        return None if size is None else min(size, 8)

    def layout(self, struct, depth: int = 0):
        """(size, alignment, {field: (offset, type)}) of a struct or union, or None."""
        if struct.decls is None:
            struct = self.structs.get(struct.name)
            if struct is None:
                return None
        offset, alignment, fields = 0, 1, {}
        union = isinstance(struct, c_ast.Union)
        extent = 0
        for field in struct.decls:
            if not isinstance(field, c_ast.Decl) or field.bitsize is not None:
                return None
            size, align = self.size_of(field.type, depth + 1), self.align_of(field.type, depth + 1)
            if size is None or not align:
                return None
            if not union:
                offset = (offset + align - 1) // align * align
            fields[field.name] = (0 if union else offset, field.type)
            extent = max(extent, (0 if union else offset) + size)
            if not union:
                offset += size
            alignment = max(alignment, align)
        return (extent + alignment - 1) // alignment * alignment, alignment, fields

    def pointee(self, node):
        """The element type when ``node`` is a pointer or an array, else None."""
        node = self.resolve(node)
        if isinstance(node, (c_ast.PtrDecl, c_ast.ArrayDecl)):
            return node.type
        return None

    def is_float(self, node) -> bool:
        node = self.resolve(node)
        return isinstance(node, c_ast.TypeDecl) and isinstance(node.type, c_ast.IdentifierType) and \
            bool({"float", "double"} & set(node.type.names))

    def type_of(self, node, depth: int = 0):
        """The C type of an expression as a declaration type node, or None when unknown."""
        if depth > 60:
            return None
        if isinstance(node, c_ast.ID):
            return self.locals.get(node.name, self.globals.get(node.name))
        if isinstance(node, c_ast.Constant):
            if node.type in ("float", "double", "long double"):
                return _type("float" if node.value.rstrip().endswith(("f", "F")) else "double")
            if node.type == "string":
                return _type("char", 1)
            return _type("int")
        if isinstance(node, c_ast.Cast):
            return node.to_type.type
        if isinstance(node, c_ast.UnaryOp):
            inner = self.type_of(node.expr, depth + 1)
            if node.op == "&":
                return None if inner is None else c_ast.PtrDecl([], inner)
            if node.op == "*":
                return self.pointee(inner)
            if node.op in ("!", "sizeof"):
                return _type("int")
            return inner
        if isinstance(node, c_ast.BinaryOp):
            if node.op in ("==", "!=", "<", ">", "<=", ">=", "&&", "||"):
                return _type("int")
            left, right = self.type_of(node.left, depth + 1), self.type_of(node.right, depth + 1)
            if node.op in ("+", "-"):
                if self.pointee(left) is not None and self.pointee(right) is not None:
                    return _type("int")
                if self.pointee(left) is not None:
                    return c_ast.PtrDecl([], self.pointee(left))
                if self.pointee(right) is not None and node.op == "+":
                    return c_ast.PtrDecl([], self.pointee(right))
            if node.op in ("<<", ">>"):
                return left
            for side in (left, right):
                if side is not None and self.is_float(side):
                    return side
            return left if left is not None else right
        if isinstance(node, c_ast.Assignment):
            return self.type_of(node.lvalue, depth + 1)
        if isinstance(node, c_ast.TernaryOp):
            found = self.type_of(node.iftrue, depth + 1)
            return found if found is not None else self.type_of(node.iffalse, depth + 1)
        if isinstance(node, c_ast.FuncCall):
            name = node.name.name if isinstance(node.name, c_ast.ID) else None
            if name in _TYPE_ARGUMENT and node.args is not None:
                index = _TYPE_ARGUMENT[name]
                if index < len(node.args.exprs) and isinstance(node.args.exprs[index], c_ast.Cast):
                    stated = node.args.exprs[index].to_type.type
                    return self.pointee(stated) if name == "M2C_FIELD" else stated
                return None
            callee = self.resolve(self.type_of(node.name, depth + 1))
            if isinstance(callee, c_ast.PtrDecl):
                callee = self.resolve(callee.type)
            return callee.type if isinstance(callee, c_ast.FuncDecl) else None
        if isinstance(node, c_ast.ArrayRef):
            return self.pointee(self.type_of(node.name, depth + 1))
        if isinstance(node, c_ast.StructRef):
            base = self.resolve(self.type_of(node.name, depth + 1))
            if node.type == "->":
                base = self.resolve(self.pointee(base))
            if isinstance(base, c_ast.TypeDecl) and isinstance(base.type, (c_ast.Struct, c_ast.Union)):
                layout = self.layout(base.type)
                if layout is not None and node.field.name in layout[2]:
                    return layout[2][node.field.name][1]
            return None
        if isinstance(node, c_ast.ExprList) and node.exprs:
            return self.type_of(node.exprs[-1], depth + 1)
        return None

    # -- output
    def text(self) -> str:
        head = self.head
        if self.dropped:
            kept = []
            for line in head.split("\n"):
                found = re.match(r"^(?:extern\s+)?[\w \*]+?\b(\w+)\s*(?:\([^{]*\)|\[[^\]]*\])?;", line)
                if found and found.group(1) in self.dropped and "{" not in line:
                    continue
                kept.append(line)
            head = "\n".join(kept)
        if self.added:
            head = head.rstrip("\n") + "\n" + "\n".join(dict.fromkeys(self.added)) + "\n\n"
        return head + _Generator().visit(self.function) + self.tail


def _replace(root, old, new) -> bool:
    """Replace the node ``old`` (by identity) below ``root`` with ``new``."""
    for node in _walk(root):
        for slot in node.__slots__:
            if slot in ("coord", "__weakref__"):
                continue
            value = getattr(node, slot, None)
            if value is old:
                setattr(node, slot, new)
                return True
            if isinstance(value, list):
                for index, item in enumerate(value):
                    if item is old:
                        value[index] = new
                        return True
    return False


def _walk(node):
    yield node
    for _name, child in node.children():
        yield from _walk(child)


# --------------------------------------------------------------------------------- machine code
INCOMING, SET, STALE, MIXED = 0, 1, 2, 3


def _argument_writes(insn, state: list) -> None:
    """Record what ``insn`` writes of the argument registers ($a0-$a3 -> 0-3, $f12 -> 4, $f14 -> 5)."""
    for register in insn.defs:
        if 4 <= register <= 7:
            state[register - 4] = SET
    word = insn.word
    opcode = word >> 26
    written = None
    if opcode in (0x31, 0x35):                               # lwc1, ldc1
        written = (word >> 16) & 31
    elif opcode == 0x11:
        kind = (word >> 21) & 31
        if kind in (4, 5):                                   # mtc1, dmtc1
            written = (word >> 11) & 31
        elif kind in (16, 17, 20, 21) and (word & 0x3F) < 0x30:
            written = (word >> 6) & 31
    if written in (12, 13):
        state[4] = SET
    elif written in (14, 15):
        state[5] = SET


class Code:
    """Facts read from the target's instructions."""

    def __init__(self, code: bytes, vram: int):
        self.vram = vram
        self._code = code
        self._sites = None
        self.insns = flowmod.decode(code, vram)
        self.addresses: dict[int, int] = {}                  # value built by lui/addiu -> times
        self.integers: dict[int, int] = {}                   # value built by lui/ori -> times
        self.accesses: dict[int, set[str]] = {}              # absolute address -> load/store mnemonics
        self.frame_accesses = 0                              # loads from the frame that restore no register
        self.events: list[tuple[int, str, str, int]] = []    # (instruction, mnemonic, "at"|"off", address|offset)
        self._pairs()
        self._frame()

    def _pairs(self) -> None:
        upper: dict[int, int] = {}                           # register -> value of a pending lui
        indexed: dict[int, int] = {}                         # register -> lui value after ``addu r, r, x``
        for insn in self.insns:
            word = insn.word
            opcode, rs, rt, immediate = word >> 26, (word >> 21) & 31, (word >> 16) & 31, word & 0xFFFF
            signed = immediate - 0x10000 if immediate & 0x8000 else immediate
            if opcode == 0x0F:                               # lui
                upper[rt] = immediate << 16
                indexed.pop(rt, None)
                continue
            if opcode == 0x09 and rs == rt and rs in upper:  # addiu r, r, lo
                value = (upper.pop(rs) + signed) & 0xFFFFFFFF
                self.addresses[value] = self.addresses.get(value, 0) + 1
                continue
            if opcode == 0x0D and rs == rt and rs in upper:  # ori r, r, lo
                value = (upper.pop(rs) | immediate) & 0xFFFFFFFF
                self.integers[value] = self.integers.get(value, 0) + 1
                continue
            if insn.op in _LOADS or insn.op in _STORES:
                base = upper.get(rs, indexed.get(rs))
                if base is not None:
                    self.accesses.setdefault((base + signed) & 0xFFFFFFFF, set()).add(insn.op)
                    self.events.append((insn.index, insn.op, "at", (base + signed) & 0xFFFFFFFF))
                elif rs != 29:
                    self.events.append((insn.index, insn.op, "off", signed))
            if opcode == 0 and (word & 0x7FF) == 0x21:       # addu rd, rs, rt
                rd = (word >> 11) & 31
                for a, b in ((rs, rt), (rt, rs)):
                    if rd == a and a in upper and b not in upper:
                        indexed[rd] = upper.pop(a)
                        break
                else:
                    upper.pop(rd, None)
                    indexed.pop(rd, None)
                continue
            for register in insn.defs:
                if insn.op in _LOADS and register == rs:
                    pass
                upper.pop(register, None)
                indexed.pop(register, None)

    def _frame(self) -> None:
        """Count reads of the function's own frame other than the epilogue's restores: a local
        that lives in memory (its address was taken) or an argument that arrived on the stack."""
        saved = set()
        for insn in self.insns:
            word = insn.word
            opcode, base, register, offset = word >> 26, (word >> 21) & 31, (word >> 16) & 31, word & 0xFFFF
            if base != 29 or offset & 0x8000:
                continue
            if opcode in (0x2B, 0x3F) and (register in (31, 30) or 16 <= register <= 23):   # sw/sd of a saved register
                saved.add((register, offset))
            elif insn.op in _LOADS and not (opcode in (0x23, 0x37) and (register, offset) in saved):
                self.frame_accesses += 1

    def call_sites(self) -> list[tuple[int | None, tuple[int, ...]]]:
        """For every call in address order: (target address or None, the state of ``$a0-$a3`` and
        of ``$f12``/``$f14`` when control leaves for the callee).

        A state says where the register's value comes from: ``INCOMING`` (untouched since the
        function was entered, so it still holds the function's own argument), ``SET`` (written
        since the previous call), ``STALE`` (a call returned since it was last written) or
        ``MIXED`` (it depends on the path)."""
        if self._sites is not None:
            return self._sites
        flow = flowmod.analyse(bytes(self._code), self.vram)
        insns, blocks = flow.insns, flow.blocks
        entry: list[tuple[int, ...] | None] = [None] * len(blocks)
        found: dict[int, tuple[int, ...]] = {}
        if blocks:
            entry[0] = (INCOMING,) * 6
        work = [0] if blocks else []
        while work:
            number = work.pop()
            block, state = blocks[number], list(entry[number])
            index = block.first
            while index <= block.last:
                insn = insns[index]
                if insn.kind == "call":
                    if index + 1 <= block.last:
                        _argument_writes(insns[index + 1], state)
                    found[index] = tuple(state)
                    state = [STALE] * 6
                    index += 2
                    continue
                _argument_writes(insn, state)
                index += 1
            for successor in block.succs:
                merged = tuple(state) if entry[successor] is None else tuple(
                    a if a == b else MIXED for a, b in zip(entry[successor], state))
                if merged != entry[successor]:
                    entry[successor] = merged
                    work.append(successor)
        self._sites = [(insn.target, found.get(insn.index, (MIXED,) * 6)) for insn in insns if insn.kind == "call"]
        return self._sites

    def access_type(self, address: int, declared_size: int | None, declared_unsigned: bool) -> str | None:
        """The type the target reads or writes at ``address`` with, when it uses one width."""
        operations = self.accesses.get(address)
        if not operations:
            return None
        loads = {_LOADS[o] for o in operations if o in _LOADS}
        widths = {_SIZES[t] for t in loads} | {_STORES[o] for o in operations if o in _STORES}
        if len(widths) != 1 or len(loads) > 1:
            return None
        if loads:
            return loads.pop()
        width = widths.pop()
        if any(o in ("swc1", "sdc1") for o in operations):
            return "f32" if width == 4 else "f64"
        if declared_size == width:
            return None
        return {1: "u8" if declared_unsigned else "s8", 2: "u16" if declared_unsigned else "s16", 4: "s32",
                8: "s64"}[width]


# --------------------------------------------------------------------------------- rule: bytes
def _symbol_address(node) -> int | None:
    """Address stated by the name of the global a pointer expression starts from."""
    if isinstance(node, c_ast.UnaryOp) and node.op == "&":
        node = node.expr
    if isinstance(node, c_ast.ID):
        found = re.fullmatch(r"D_([0-9A-Fa-f]{8})", node.name)
        if found:
            return int(found.group(1), 16)
    return None


def byte_arithmetic(draft: Draft, code: Code) -> int:
    """Make additions to a pointer with a wide element byte-wise, as the decompiler meant them.

    The decompiler writes every address computation in bytes.  When the operand happens to have a
    pointer type whose element is wider than a byte (``&D`` of an ``s32``, a parameter it typed
    ``s32 *`` because it is dereferenced), C multiplies the offset by the element size and the
    build computes another address.  ``p + n`` becomes ``(T *) ((s8 *) p + n)``; ``p += n`` with a
    constant multiple of the element size becomes ``p += n / size``.  A dereference of the sum of
    a global's address takes the type of the access the target makes at that address."""
    changed = 0
    body = draft.function.body
    byte = _type("s8", 1)

    def wide(expression):
        kind = draft.type_of(expression)
        element = draft.pointee(kind)
        if element is None:
            return None
        size = draft.size_of(element)
        if size is None or size <= 1:
            return None
        return c_ast.PtrDecl([], element), size

    nodes = list(_walk(body))
    parents = {}
    for node in nodes:
        for _name, child in node.children():
            parents[id(child)] = node
    for node in nodes:
        if isinstance(node, c_ast.Assignment) and node.op in ("+=", "-="):
            found = wide(node.lvalue)
            if found is None or draft.pointee(draft.type_of(node.rvalue)) is not None:
                continue
            kind, size = found
            amount = _number(node.rvalue)
            if amount is not None and amount % size == 0 and amount != 0:
                value = abs(amount) // size
                node.rvalue = c_ast.Constant("int", str(value) if value < 10 else hex(value).upper().replace("X", "x"))
                if amount < 0:
                    node.op = "-=" if node.op == "+=" else "+="
            else:
                node.rvalue = _cast(kind, c_ast.BinaryOp(node.op[0], _cast(byte, copy.deepcopy(node.lvalue)),
                                                         node.rvalue))
                node.op = "="
            changed += 1
        elif isinstance(node, c_ast.BinaryOp) and node.op in ("+", "-"):
            left, right = wide(node.left), wide(node.right)
            if left is not None and draft.pointee(draft.type_of(node.right)) is None:
                side, (kind, _size) = "left", left
            elif right is not None and node.op == "+" and draft.pointee(draft.type_of(node.left)) is None:
                side, (kind, _size) = "right", right
            else:
                continue
            pointer = getattr(node, side)
            address = _symbol_address(pointer)
            # ``n + (s8 *) p`` would be reordered to pointer-first by the compiler; an integer
            # addition keeps the operands in the order the target adds them
            setattr(node, side, _cast(byte if side == "left" else _type("s32"), pointer))
            parent = parents.get(id(node))
            result = kind
            if address is not None and isinstance(parent, c_ast.UnaryOp) and parent.op == "*":
                # the target's own access to this address decides the width
                element = draft.resolve(kind.type)
                names = element.type.names if isinstance(element, c_ast.TypeDecl) and \
                    isinstance(element.type, c_ast.IdentifierType) else []
                stated = code.access_type(address, draft.size_of(kind.type), "unsigned" in names)
                if stated is not None and (draft.size_of(kind.type) != _SIZES[stated] or stated in ("f32", "f64")
                                           or (stated.startswith("u") != ("unsigned" in names)
                                               and _SIZES[stated] < 4)):
                    result = _type(stated, 1)
            replacement = _cast(result, copy.copy(node))
            if parent is not None and _replace(parent, node, replacement):
                # the copy now stands in the tree: later visits must see it, not ``node``
                parents[id(replacement)] = parent
                parents[id(replacement.expr)] = replacement
                for _name, child in replacement.expr.children():
                    parents[id(child)] = replacement.expr
                changed += 1
            else:
                setattr(node, side, pointer)
    return changed


# --------------------------------------------------------------------------------- rule: symbols
def symbol_addresses(draft: Draft, code: Code) -> int:
    """Turn a constant the target builds with ``lui``/``addiu`` into the address of a symbol.

    GCC loads an integer constant with ``lui``/``ori`` and an address with ``lui``/``addiu``
    (the assembler's expansion of ``la``).  A draft that shows such a value as a number therefore
    cannot match: the source named something the linker placed there (a ROM offset, a segment
    address).  The number becomes ``(s32) &D_<address>``, declared here; the name states its own
    address, which is what lets the verifier accept it."""
    wanted = {value for value in code.addresses if value not in code.integers}
    if not wanted:
        return 0
    changed = 0
    nodes = list(_walk(draft.function.body))
    parents = {}
    for node in nodes:
        for _name, child in node.children():
            parents[id(child)] = node
    for node in nodes:
        if not isinstance(node, c_ast.Constant):
            continue
        value = _number(node)
        if value is None:
            continue
        parent = parents.get(id(node))
        target = node
        if isinstance(parent, c_ast.UnaryOp) and parent.op == "-":
            value, target = -value, parent
            parent = parents.get(id(parent))
        value &= 0xFFFFFFFF
        if value not in wanted or value < 0x10000 or parent is None:
            continue
        symbol = f"D_{value:08X}"
        if symbol in draft.locals:
            continue
        replacement = _cast(_type("s32"), c_ast.UnaryOp("&", c_ast.ID(symbol)))
        if _replace(parent, target, replacement):
            if symbol not in draft.globals:
                draft.added.append(f"extern M2C_UNK {symbol};")
                draft.globals[symbol] = _type("M2C_UNK")
            changed += 1
    return changed


# --------------------------------------------------------------------------------- rule: step

def shared_step(draft: Draft, code: Code) -> int:
    """``if (c) { i += 1; } else { A; i += 1; B; }`` becomes ``if (!c) { A; B; }  i += 1;``.

    A statement after an ``if`` (a loop's step) is copied by the compiler into both paths when it can fill
    a delay slot there, and a decompiler writes what it sees: the step once in each branch.  Compiled again,
    the two copies are two statements and the branches come out differently (measured on a title: two
    functions of 284 and 292 bytes matched once the step stood after the ``if``).  Only a branch that is
    nothing but the step is folded, so nothing else changes order."""
    generator = _Generator()
    changed = 0
    for node in list(_walk(draft.function.body)):
        items = getattr(node, "block_items", None)
        if not isinstance(node, c_ast.Compound) or not items:
            continue
        for position, statement in enumerate(list(items)):
            if not isinstance(statement, c_ast.If) or statement.iffalse is None:
                continue
            for lone, other, negate in ((statement.iftrue, statement.iffalse, True),
                                        (statement.iffalse, statement.iftrue, False)):
                alone = lone.block_items if isinstance(lone, c_ast.Compound) else [lone]
                rest = other.block_items if isinstance(other, c_ast.Compound) else None
                if not alone or len(alone) != 1 or not rest or len(rest) < 2 \
                        or not isinstance(alone[0], (c_ast.Assignment, c_ast.UnaryOp)):
                    continue
                step = generator.visit(alone[0])
                same = [k for k, item in enumerate(rest) if isinstance(item, type(alone[0])) and generator.visit(item) == step]
                if len(same) != 1:
                    continue
                del rest[same[0]]
                condition = statement.cond
                if negate:
                    condition = condition.expr if isinstance(condition, c_ast.UnaryOp) and condition.op == "!" \
                        else c_ast.UnaryOp("!", condition)
                statement.cond, statement.iftrue, statement.iffalse = condition, other, None
                items.insert(items.index(statement) + 1, alone[0])
                changed += 1
                break
    return changed


# --------------------------------------------------------------------------------- rule: reuse

_SAVED_TEMP = re.compile(r"(?:temp|var)_(s[0-7])_\d+")


def reused_variables(draft: Draft, code: Code) -> int:
    """Two temporaries a decompiler named after one call-saved register, used one after the other, are one
    variable of the source.

    A decompiler makes a new temporary for each value a register holds.  The compiler put both values in
    that register because the program kept them in one variable; declared as two, the second one gets a
    scratch register (measured on a title: a 344-byte function was two instructions away until the value
    read before a swap and the swap's temporary were one variable).  Only temporaries whose every use comes
    before the other's first, with no loop around both, are joined; the variable takes the wider type."""
    body = draft.function.body
    order: dict[str, list[int]] = {}
    loops: dict[str, set[int]] = {}

    def visit(node, inside: tuple[int, ...]) -> None:
        if isinstance(node, (c_ast.For, c_ast.While, c_ast.DoWhile)):
            inside = inside + (id(node),)
        if isinstance(node, c_ast.ID) and _SAVED_TEMP.fullmatch(node.name):
            order.setdefault(node.name, []).append(len(order_all))
            loops.setdefault(node.name, set()).update(inside)
        order_all.append(node)
        for _name, child in node.children():
            visit(child, inside)

    order_all: list = []
    visit(body, ())
    declared = {item.name: item for item in (body.block_items or [])
                if isinstance(item, c_ast.Decl) and item.name in order and item.init is None}
    changed = 0
    names = sorted(declared, key=lambda name: order[name][0])
    gone: set[str] = set()
    for first in names:
        for second in names:
            if first == second or first in gone or second in gone \
                    or _SAVED_TEMP.fullmatch(first).group(1) != _SAVED_TEMP.fullmatch(second).group(1) \
                    or order[first][-1] >= order[second][0] or loops[first] & loops[second]:
                continue
            between = [name for name in names if name not in (first, second) and name not in gone
                       and _SAVED_TEMP.fullmatch(name).group(1) == _SAVED_TEMP.fullmatch(first).group(1)
                       and order[first][-1] < order[name][0] < order[second][0]]
            if between:
                continue                                     # the register held something else in between
            sizes = [_SIZES.get(" ".join(getattr(getattr(declared[name].type, "type", None), "names", []) or []), 0)
                     for name in (first, second)]
            if 0 in sizes:
                continue                                     # a pointer or a structure: left alone
            if sizes[1] > sizes[0]:
                declared[first].type = declared[second].type
                declared[first].type.declname = first
            for node in order_all:
                if isinstance(node, c_ast.ID) and node.name == second:
                    node.name = first
            body.block_items.remove(declared[second])
            order[first] += order[second]
            loops[first] |= loops[second]
            gone.add(second)
            changed += 1
    return changed


# --------------------------------------------------------------------------------- rule: stride

def strided_arrays(draft: Draft, code: Code) -> int:
    """``*(D_x + (i * K))`` on an array of scalars becomes ``D_x[i].v`` on an array of K-sized structures.

    A decompiler that does not know an array's element writes the access as arithmetic on its first
    member's type.  The value is the same, but the compiler treats the multiplication as the program's own
    and keeps a temporary for it (measured on a title: a 344-byte function had a frame eight bytes too
    large until the array was one of 40-byte structures).  Only an array whose every use in the function
    has this form with one K is changed."""
    parents = _parents(draft.function.body)
    uses: dict[str, list] = {}
    bad: set[str] = set()
    for node in _walk(draft.function.body):
        if not isinstance(node, c_ast.ID) or not _ADDRESS_NAMED.fullmatch(node.name) or node.name in draft.locals:
            continue
        declared = draft.globals.get(node.name)
        if not isinstance(declared, c_ast.ArrayDecl) or not isinstance(declared.type, c_ast.TypeDecl):
            continue
        size = _SIZES.get(" ".join(getattr(declared.type.type, "names", []) or []))
        add = parents.get(id(node))
        star = parents.get(id(add)) if add is not None else None
        index = stride = None
        if size and isinstance(add, c_ast.BinaryOp) and add.op == "+" and add.left is node \
                and isinstance(star, c_ast.UnaryOp) and star.op == "*" \
                and isinstance(add.right, c_ast.BinaryOp) and add.right.op == "*":
            for value, other in ((add.right.right, add.right.left), (add.right.left, add.right.right)):
                number = _number(value)
                if number is not None and number > 1:
                    index, stride = other, number * size
                    break
        if index is None:
            bad.add(node.name)
        else:
            uses.setdefault(node.name, []).append((star, index, stride, size))
    changed = 0
    for symbol, places in uses.items():
        found = re.search(rf"^extern\s+([^;{{}}()\n]*?)\b{symbol}\s*\[\s*\]\s*;", draft.head, re.M)
        if symbol in bad or found is None or len({place[2] for place in places}) != 1:
            continue
        stride, size = places[0][2], places[0][3]
        for star, index, _stride, _size in places:
            new = c_ast.StructRef(c_ast.ArrayRef(c_ast.ID(symbol), index), ".", c_ast.ID("v"))
            if not _replace(parents.get(id(star)), star, new):
                return -1
        draft.dropped.add(symbol)
        draft.added.append(f"struct n64dk_{symbol} {{ {found.group(1).strip()} v; char pad[{stride - size}]; }};")
        draft.added.append(f"extern struct n64dk_{symbol} {symbol}[];")
        changed += 1
    return changed


# --------------------------------------------------------------------------------- rule: signs

def _sign_tests(code: Code) -> set[int]:
    """Widths (8, 16) the target tests the sign of: ``sll r, x, 24|16`` and then ``bgez``/``bltz`` on it."""
    found = set()
    words = [insn.word for insn in code.insns]
    for first, second in zip(words, words[1:]):
        if first >> 26 == 0 and first & 0x3F == 0 and (first >> 6) & 31 in (24, 16) and second >> 26 == 1 \
                and (second >> 16) & 31 in (0, 1, 2, 3) and (second >> 21) & 31 == (first >> 11) & 31:
            found.add(32 - ((first >> 6) & 31))
    return found


def sign_tests(draft: Draft, code: Code) -> int:
    """``x & 0x80`` as a condition becomes ``(s8) x < 0`` where the target tests a sign.

    GCC tests the top bit of a byte two ways: ``andi``/``beqz`` for the mask a program wrote, and a shift to
    the word's sign with ``bgez``/``bltz`` for a signed comparison with zero.  A decompiler renders both as
    the mask.  Where the target has the shift form the comparison is written (measured on a title: one
    function of 404 bytes was this one test away)."""
    widths = _sign_tests(code)
    if not widths:
        return 0
    parents = _parents(draft.function.body)
    changed = 0
    for node in list(_walk(draft.function.body)):
        if not isinstance(node, c_ast.BinaryOp) or node.op != "&":
            continue
        mask = _number(node.right)
        width = {0x80: 8, 0x8000: 16}.get(mask)
        if width not in widths:
            continue
        parent = parents.get(id(node))
        test = c_ast.BinaryOp("<", _cast(_type("s8" if width == 8 else "s16"), node.left), c_ast.Constant("int", "0"))
        if isinstance(parent, c_ast.BinaryOp) and parent.op in ("==", "!=") and _number(parent.right) == 0 \
                and parent.left is node:
            grand = parents.get(id(parent))
            if parent.op == "==":
                test.op = ">="
            if grand is not None and _replace(grand, parent, test):
                changed += 1
        elif isinstance(parent, (c_ast.If, c_ast.While, c_ast.DoWhile, c_ast.TernaryOp)) and parent.cond is node:
            parent.cond = test
            changed += 1
        elif isinstance(parent, c_ast.BinaryOp) and parent.op in ("&&", "||") and _replace(parent, node, test):
            changed += 1
        elif isinstance(parent, c_ast.UnaryOp) and parent.op == "!" and _replace(parent, node, test):
            changed += 1
    return changed


# --------------------------------------------------------------------------------- rule: parameters

def positional_parameters(source: str, name: str) -> str | None:
    """The draft with the parameters a decompiler left out put back, or None when none is missing.

    A decompiler names a parameter after the register or stack slot it arrives in (``arg2`` is ``$a2``) and
    lists only the ones the body reads.  ``f(s32 arg2, s32 *arg3)`` then compiles with the two values in
    ``$a0``/``$a1``: the compiler places parameters by position.  Each missing position before the last
    one named becomes an unused ``s32`` parameter (measured on a title: ``*arg3 = arg2`` is two instructions
    that differ in one register until the two leading parameters are declared)."""
    found = re.search(rf"^([^\n;{{}}()]*\b{re.escape(name)}\s*)\(([^(){{}};]*)\)(\s*\{{)", source, re.M)
    if found is None:
        return None
    pieces = [piece.strip() for piece in found.group(2).split(",")]
    numbered = {}
    for piece in pieces:
        hit = re.search(r"\barg(\d+)\s*(?:\[[^\]]*\])?$", piece)
        if hit is None:
            return None                                      # a parameter not named by position (or ``void``)
        numbered[int(hit.group(1))] = piece
    if len(numbered) != len(pieces) or sorted(numbered) == list(range(len(pieces))) or max(numbered) > 15:
        return None
    full = ", ".join(numbered.get(k, f"s32 arg{k}") for k in range(max(numbered) + 1))
    return source[:found.start(2)] + full + source[found.end(2):]


# --------------------------------------------------------------------------------- rule: addressed

_ADDRESS_NAMED = re.compile(r"D_[0-9A-Fa-f]{8}")


def addressed_globals(draft: Draft, code: Code) -> int:
    """A scalar global whose address the target keeps in a register is read as element 0 of an array.

    GCC 2.x reads and writes a scalar global through its address split in two (``lui``; ``lw %lo(sym)``).
    It forms the whole address first (``lui``/``addiu``) and goes through the register when the object is
    an aggregate: an array, a structure.  A draft that declares such an object as a scalar cannot give that
    code (measured on a title: ``if (S != x) { ...; S = x; }`` matches as ``S[0]``, not as ``S``).  The
    declaration becomes ``extern T D_x[];`` and each use ``D_x[0]``.  A symbol the draft only takes the
    address of (``&D_x``) is left alone, since that already explains the register; one it also reads or
    writes directly is changed in both places (``&D_x`` becomes ``D_x``), because a scalar is read by its
    own address pair even when its address is in a register (measured on a title: a 136-byte function)."""
    wanted = {value for value in code.addresses if value not in code.integers}
    if not wanted:
        return 0
    parents = _parents(draft.function.body)
    uses: dict[str, list] = {}
    taken: dict[str, list] = {}
    for node in _walk(draft.function.body):
        if not isinstance(node, c_ast.ID) or not _ADDRESS_NAMED.fullmatch(node.name) \
                or int(node.name[2:], 16) not in wanted or node.name in draft.locals:
            continue
        if not isinstance(draft.globals.get(node.name), (c_ast.TypeDecl, c_ast.PtrDecl)):
            continue                                         # an array or a function already, or not declared
        parent = parents.get(id(node))
        if isinstance(parent, c_ast.UnaryOp) and parent.op == "&":
            taken.setdefault(node.name, []).append(parent)
        elif parent is not None:
            uses.setdefault(node.name, []).append((parent, node))
    changed = 0
    for symbol, places in uses.items():
        found = re.search(rf"^(extern\s+[^;{{}}()\n]*?\b{symbol})\s*;", draft.head, re.M)
        if found is None or symbol in draft.dropped:         # another rule gave it a new declaration already
            continue
        for parent, node in places:
            if not _replace(parent, node, c_ast.ArrayRef(c_ast.ID(symbol), c_ast.Constant("int", "0"))):
                return -1
        for address in taken.get(symbol, []):                # ``&D_x`` beside direct uses: the array itself
            if not _replace(parents.get(id(address)), address, c_ast.ID(symbol)):
                return -1
        draft.dropped.add(symbol)
        draft.added.append(f"{found.group(1)}[];")
        changed += 1
    return changed


# --------------------------------------------------------------------------------- rule: grouped
GROUP_SPAN = 0x200             # how far past the base a member of the same object is looked for


def grouped_globals(draft: Draft, code: Code) -> int:
    """Scalar globals the target reaches as offsets from one address become members of one structure.

    The target forms an object's address once (``lui``/``addiu``) and reads its members through that
    register; a decompiler that sees only addresses names each member as a global of its own, and the
    compiler then addresses each one by itself (measured on a title: a 352-byte function matched once
    eight globals were one structure).  A group is a base whose address the target forms and the scalar
    globals within reach above it that the draft uses; the structure holds them at their distances, padded
    between.  The base itself need not be one of them: a draft may name only members and write the object's
    address as ``&member - distance``.  Nearness is the only evidence that they belong together, so this is
    one variant to try."""
    wanted = {value for value in code.addresses if value not in code.integers}
    if not wanted:
        return 0
    declared: dict[str, tuple[str, int]] = {}
    for found in re.finditer(r"^extern\s+([\w ]+?\s*\**)\s*\b(D_[0-9A-Fa-f]{8})\s*;", draft.head, re.M):
        kind = " ".join(found.group(1).split())
        size = 4 if kind.endswith("*") else _SIZES.get(kind)
        if size:
            declared[found.group(2)] = (kind, size)
    parents = _parents(draft.function.body)
    used: dict[str, list] = {}
    for node in _walk(draft.function.body):
        if isinstance(node, c_ast.ID) and node.name in declared and node.name not in draft.locals:
            used.setdefault(node.name, []).append(node)
    changed = 0
    taken: set[str] = set()
    for base in sorted(wanted):
        name = f"D_{base:08X}"
        members = sorted((int(other[2:], 16) - base, other) for other in used
                         if other not in taken and 0 <= int(other[2:], 16) - base < GROUP_SPAN
                         and (other.upper() == name.upper() or int(other[2:], 16) not in wanted))
        if len(members) < 2:
            continue
        fields, end = [], 0
        for offset, other in members:
            kind, size = declared[other]
            if offset < end or offset % size:
                fields = None
                break
            if offset > end:
                fields.append(f"char pad_{end:X}[{offset - end}];")
            fields.append(f"{kind} f_{offset:X};")
            end = offset + size
        if not fields:
            continue
        root = members[0][1] if members[0][0] == 0 else name
        for offset, other in members:
            for node in used[other]:
                # ``&member - distance``: the decompiler's way of writing the object's own address
                parent = parents.get(id(node))
                outer = parents.get(id(parent)) if parent is not None else None
                if isinstance(parent, c_ast.UnaryOp) and parent.op == "&" and isinstance(outer, c_ast.BinaryOp) \
                        and outer.op == "-" and outer.left is parent and _number(outer.right) == offset:
                    whole = _cast(c_ast.PtrDecl([], _type(declared[other][0].rstrip("* ").strip() or "s32",
                                                          declared[other][0].count("*"))),
                                  c_ast.UnaryOp("&", c_ast.ID(root)))
                    if not _replace(parents.get(id(outer)), outer, whole):
                        return -1
                    continue
                if not _replace(parent, node, c_ast.StructRef(c_ast.ID(root), ".", c_ast.ID(f"f_{offset:X}"))):
                    return -1
            draft.dropped.add(other)
            taken.add(other)
        draft.added.append(f"struct n64dk_{root} {{ {' '.join(fields)} }};")
        draft.added.append(f"extern struct n64dk_{root} {root};")
        changed += 1
    return changed


# --------------------------------------------------------------------------------- rule: inner

def _inner_offsets(code: Code) -> list[int]:
    """Constants K of ``addiu $sX, $sY|$aY, K``: a call-saved register made to point K bytes into an object."""
    found = set()
    for insn in code.insns:
        word = insn.word
        rs, rt, immediate = (word >> 21) & 31, (word >> 16) & 31, word & 0xFFFF
        if word >> 26 == 0x09 and 16 <= rt <= 23 and rs != rt and (16 <= rs <= 23 or 4 <= rs <= 7) \
                and 0 < immediate < 0x8000:
            found.add(immediate)
    return sorted(found)


def inner_pointers(draft: Draft, code: Code) -> int:
    """Members of a parameter's object past K are reached through a pointer to K, where the target keeps one.

    A program that works on a structure inside a larger one takes its address once (``sub = &p->sub``) and
    the compiler keeps that address in a register of its own (``addiu $s1, $s2, 0x74``); a draft that
    reaches every member from the outer pointer gives one register and larger offsets (measured on a title:
    a 412-byte function, and the pattern stands in a dozen more of the same family).  For each such K, a
    local pointer is made and the accesses at K and beyond go through it."""
    offsets = _inner_offsets(code)
    if not offsets:
        return 0
    parents = _parents(draft.function.body)
    changed = 0
    for parameter in draft.parameters():
        name = parameter.name
        target = draft.pointee(parameter.type)
        struct = draft.resolve(target).type if target is not None and isinstance(draft.resolve(target), c_ast.TypeDecl) else None
        layout = draft.layout(struct) if isinstance(struct, c_ast.Struct) else None
        refs = []                                            # (node, offset, type of a pointer to the member)
        for node in _walk(draft.function.body):
            if isinstance(node, c_ast.StructRef) and node.type == "->" and isinstance(node.name, c_ast.ID) \
                    and node.name.name == name and layout is not None and node.field.name in layout[2]:
                offset, kind = layout[2][node.field.name]
                refs.append((node, offset, c_ast.PtrDecl([], _undeclared(kind))))
            else:
                call = _field_call(node)
                if call is not None and isinstance(call[0], c_ast.ID) and call[0].name == name:
                    refs.append((node, call[2], call[1]))
        for inner in reversed(offsets):
            chosen = [ref for ref in refs if inner <= ref[1] < inner + 0x100]
            if len(chosen) < 2 or any(isinstance(draft.resolve(kind.type), c_ast.ArrayDecl) for _n, _o, kind in chosen):
                continue
            local = f"n64dk_inner_{inner:X}"
            for node, offset, kind in chosen:
                new = c_ast.FuncCall(c_ast.ID("M2C_FIELD"), c_ast.ExprList([
                    c_ast.ID(local), c_ast.Cast(_typename(kind), c_ast.Constant("int", "0")),
                    c_ast.Constant("int", hex(offset - inner))]))
                if not _replace(parents.get(id(node)), node, new):
                    return -1
            declaration = _parse(f"typedef signed char s8; void g(void) {{ s8 *{local} = 0; }}").ext[1].body.block_items[0]
            declaration.init = c_ast.BinaryOp("+", _cast(_type("s8", 1), c_ast.ID(name)), c_ast.Constant("int", hex(inner)))
            items = draft.function.body.block_items
            position = next((i for i, item in enumerate(items) if not isinstance(item, c_ast.Decl)), len(items))
            items.insert(position, declaration)
            refs = [ref for ref in refs if ref not in chosen]
            changed += 1
    return changed


# --------------------------------------------------------------------------------- rule: tails

_NEGATED = {"==": "!=", "!=": "==", "<": ">=", ">=": "<", ">": "<=", "<=": ">"}


def _negated(condition):
    if isinstance(condition, c_ast.UnaryOp) and condition.op == "!":
        return condition.expr
    if isinstance(condition, c_ast.BinaryOp) and condition.op in _NEGATED:
        return c_ast.BinaryOp(_NEGATED[condition.op], condition.left, condition.right)
    return c_ast.UnaryOp("!", condition)


def branch_tails(draft: Draft, code: Code) -> int:
    """``if (c) { v = A; } else { v = B; } use(v);`` becomes ``if (c) { use(A); } else { use(B); }``.

    The compiler joins the equal ends of branches, and a decompiler reads the joined code: one use fed by
    a temporary that each path set.  Written that way the values meet before the use and the compiler
    joins more than the target did (measured on a title: calls each followed by the load of the next
    argument, or each with its own argument set up, where the draft gave one call with a chosen argument).
    The use is written at the end of every path through the ``if``, with the value that path gave the
    temporary (the one it had before the ``if`` where a path sets none).  Only a temporary read once, in
    the statement after the ``if``, and set nowhere but at statement level on those paths is folded back."""
    def reads(node, name):
        return [inner for inner in _walk(node) if isinstance(inner, c_ast.ID) and inner.name == name]

    def sets(item, name):
        return isinstance(item, c_ast.Assignment) and item.op == "=" and isinstance(item.lvalue, c_ast.ID) \
            and item.lvalue.name == name

    def tail(use, name, value):
        copied = copy.deepcopy(use)
        read = reads(copied, name)[0]
        return copied if _replace(copied, read, copy.deepcopy(value)) else None

    def walk(items, name, use, current, counted):
        out = []
        for index, item in enumerate(items):
            if sets(item, name) and not reads(item.rvalue, name):
                current = item.rvalue
                counted[0] += 1
                continue
            if isinstance(item, c_ast.If) and index == len(items) - 1 and not reads(item.cond, name):
                forked = fork(item, name, use, current, counted)
                return None if forked is None else out + [forked]
            if reads(item, name):
                return None
            out.append(item)
        last = tail(use, name, current) if current is not None else None
        return None if last is None else out + [last]

    def fork(statement, name, use, current, counted):
        branches = []
        for branch in (statement.iftrue, statement.iffalse):
            items = [] if branch is None else (branch.block_items or [] if isinstance(branch, c_ast.Compound) else [branch])
            walked = walk(items, name, use, current, counted)
            if walked is None:
                return None
            branches.append(c_ast.Compound(walked))
        nested = [bool(branch.block_items) and isinstance(branch.block_items[-1], c_ast.If) for branch in branches]
        if nested[0] and not nested[1]:                      # an ``else if`` chain: the plain path first
            return c_ast.If(_negated(statement.cond), branches[1], branches[0])
        return c_ast.If(statement.cond, branches[0], branches[1])

    changed = 0
    counts: dict[str, int] = {}
    for node in _walk(draft.function.body):
        if isinstance(node, c_ast.ID):
            counts[node.name] = counts.get(node.name, 0) + 1
    for node in list(_walk(draft.function.body)):
        items = getattr(node, "block_items", None)
        if not isinstance(node, c_ast.Compound) or not items:
            continue
        for statement in list(items):
            position = items.index(statement) if statement in items else -1
            if position < 0 or position + 1 >= len(items) or not isinstance(statement, c_ast.If):
                continue
            use = items[position + 1]
            if not isinstance(use, (c_ast.FuncCall, c_ast.Assignment)):
                continue
            for name in sorted({inner.name for inner in _walk(use) if isinstance(inner, c_ast.ID)} & set(draft.locals)):
                if len(reads(use, name)) != 1 or (isinstance(use, c_ast.Assignment) and use.lvalue is reads(use, name)[0]):
                    continue
                before = items[position - 1] if position and sets(items[position - 1], name) else None
                counted = [1 if before is not None else 0]
                forked = fork(statement, name, use, before.rvalue if before is not None else None, counted)
                if forked is None or counted[0] < 2 or counts.get(name) != counted[0] + 1:
                    continue                                 # set or read somewhere else as well
                items[position] = forked
                items.remove(use)
                if before is not None:
                    items.remove(before)
                changed += 1
                break
    return changed


# --------------------------------------------------------------------------------- rule: returns

def swapped_returns(draft: Draft, code: Code) -> int:
    """``if (c) { A; return x; } B; return y;`` becomes ``if (!c) { B; return y; } A; return x;``.

    Both say the same, and the compiler lays the code out as written: the block under the ``if`` out of
    line, the rest straight on.  A decompiler picks one of the two from the branch it happens to read
    first (measured on a title: a 240-byte function matched with the two swapped).  The last such ``if``
    of the function body is swapped; which of the two the program had is for the comparison to say."""
    items = draft.function.body.block_items or []
    for position in range(len(items) - 1, -1, -1):
        statement = items[position]
        if not isinstance(statement, c_ast.If) or statement.iffalse is not None \
                or not isinstance(statement.iftrue, c_ast.Compound) or not statement.iftrue.block_items:
            continue
        inner, rest = statement.iftrue.block_items, items[position + 1:]
        if not isinstance(inner[-1], c_ast.Return) or not rest or not isinstance(rest[-1], c_ast.Return) \
                or any(isinstance(item, (c_ast.Decl, c_ast.Label)) for item in inner + rest):
            return 0
        items[position:] = [c_ast.If(_negated(statement.cond), c_ast.Compound(rest), None)] + inner
        return 1
    return 0


# --------------------------------------------------------------------------------- rule: assoc

def constant_last(draft: Draft, code: Code) -> int:
    """``a + K + b`` becomes ``a + (b + K)``.

    The compiler adds in the order the source groups, with a constant moved to the end of its own group:
    ``a + (b + K)`` comes out as ``a + K`` then ``+ b``, which is what a decompiler reads back and writes
    as ``a + K + b``; compiled again that is ``b + K`` then ``+ a`` (measured on a title).  Sums whose
    middle term is a constant are regrouped so that the round trip gives the target's order."""
    changed = 0
    parents = _parents(draft.function.body)
    for node in list(_walk(draft.function.body)):
        if isinstance(node, c_ast.BinaryOp) and node.op == "+" and isinstance(node.left, c_ast.BinaryOp) \
                and node.left.op == "+" and _number(node.left.right) is not None and _number(node.right) is None \
                and _number(node.left.left) is None:
            new = c_ast.BinaryOp("+", node.left.left, c_ast.BinaryOp("+", node.right, node.left.right))
            if not _replace(parents.get(id(node)), node, new):
                return -1
            changed += 1
    return changed


# --------------------------------------------------------------------------------- rule: shifts

def split_shifts(draft: Draft, code: Code) -> int:
    """``v = (s32) (a << K) >> L;`` becomes ``v = a << K; v >>= L;``.

    Where a function computes the same shifted value into the same variable a second time, the second
    computation written as one expression goes through a scratch register; written as two statements on
    the variable, both shifts happen in the variable's own register (measured on a title: a 180-byte
    function was one instruction pair away until the repeated pair was split, and splitting the first one
    as well broke it).  Only the repeats of a 32-bit signed local's statement-level assignment are split."""
    def word(kind) -> bool:
        """A signed 32-bit integer type, by its own name or through the prelude's typedefs."""
        if not isinstance(kind, c_ast.TypeDecl):
            return False
        if " ".join(getattr(kind.type, "names", [])) in ("s32", "int", "signed int"):
            return True
        resolved = draft.resolve(kind)
        return isinstance(resolved, c_ast.TypeDecl) and " ".join(getattr(resolved.type, "names", [])) in ("int", "signed int")

    changed = 0
    generator, seen = _Generator(), set()
    for node in list(_walk(draft.function.body)):
        items = getattr(node, "block_items", None)
        if not isinstance(node, c_ast.Compound) or not items:
            continue
        for statement in list(items):
            if not (isinstance(statement, c_ast.Assignment) and statement.op == "=" and isinstance(statement.lvalue, c_ast.ID)
                    and isinstance(statement.rvalue, c_ast.BinaryOp) and statement.rvalue.op == ">>"
                    and _number(statement.rvalue.right) is not None):
                continue
            inner = statement.rvalue.left
            if isinstance(inner, c_ast.Cast):
                if not word(inner.to_type.type):
                    continue
                inner = inner.expr
            if not (isinstance(inner, c_ast.BinaryOp) and inner.op == "<<" and _number(inner.right) is not None
                    and word(draft.locals.get(statement.lvalue.name))):
                continue
            if any(isinstance(part, c_ast.ID) and part.name == statement.lvalue.name for part in _walk(inner)):
                continue
            text = generator.visit(statement)
            if text not in seen:                             # the first computation stays one expression
                seen.add(text)
                continue
            position = items.index(statement)
            items[position:position + 1] = [
                c_ast.Assignment("=", c_ast.ID(statement.lvalue.name), inner),
                c_ast.Assignment(">>=", c_ast.ID(statement.lvalue.name), statement.rvalue.right)]
            changed += 1
    return changed


# --------------------------------------------------------------------------------- rule: scan

def scanning_loops(draft: Draft, code: Code) -> int:
    """``do { p += 1; } while (*p != 0);`` becomes ``while (*p++ != 0) { }``.

    A scan for the end of a string compiles to a load, a branch back and the increment in the branch's
    delay slot, where it also runs on the way out.  A decompiler that reads the increment as the loop's
    body writes a loop that steps before it tests and stops one short (measured on a title: two such loops
    in a 508-byte function).  The two forms are not the same program, so this is a variant for the
    comparison to judge; only a loop whose whole body is the step of the pointer it tests is rewritten."""
    changed = 0
    parents = _parents(draft.function.body)
    for node in list(_walk(draft.function.body)):
        if not isinstance(node, c_ast.DoWhile) or not isinstance(node.cond, c_ast.BinaryOp) or node.cond.op != "!=" \
                or _number(node.cond.right) != 0:
            continue
        tested = node.cond.left
        while isinstance(tested, c_ast.Cast):
            tested = tested.expr
        body = node.stmt.block_items if isinstance(node.stmt, c_ast.Compound) else [node.stmt]
        if not (isinstance(tested, c_ast.UnaryOp) and tested.op == "*" and isinstance(tested.expr, c_ast.ID)
                and body and len(body) == 1):
            continue
        name, step = tested.expr.name, body[0]
        stepped = (isinstance(step, c_ast.Assignment) and step.op == "+=" and isinstance(step.lvalue, c_ast.ID)
                   and step.lvalue.name == name and _number(step.rvalue) == 1) or \
                  (isinstance(step, c_ast.UnaryOp) and step.op in ("p++", "++") and isinstance(step.expr, c_ast.ID)
                   and step.expr.name == name)
        if not stepped:
            continue
        loop = c_ast.While(c_ast.BinaryOp("!=", c_ast.UnaryOp("*", c_ast.UnaryOp("p++", c_ast.ID(name))),
                                          c_ast.Constant("int", "0")), c_ast.Compound([]))
        if not _replace(parents.get(id(node)), node, loop):
            return -1
        changed += 1
    return changed


# --------------------------------------------------------------------------------- rule: block

def block_copies(draft: Draft, code: Code) -> int:
    """Byte-by-byte assignments from consecutive constant bytes to an object's first bytes become one copy.

    ``strcpy(buffer, "-")`` with a constant string is compiled as a copy of its bytes: every byte loaded,
    then every byte stored through the buffer's address.  A decompiler writes one assignment per byte, and
    those compile to a load and a store each (measured on a title: a 508-byte function).  A run of
    statements ``G.a = D_x; G.b = D_x+1; ...`` over the first members of one global, from address-named
    bytes in order, is written as one structure copy of that many bytes."""
    items_of = [node for node in _walk(draft.function.body) if isinstance(node, c_ast.Compound) and node.block_items]
    changed = 0
    for compound in items_of:
        items = compound.block_items
        index = 0
        while index < len(items):
            run = []
            for statement in items[index:]:
                source = statement.rvalue if isinstance(statement, c_ast.Assignment) and statement.op == "=" else None
                while isinstance(source, c_ast.Cast):
                    source = source.expr
                target = statement.lvalue if source is not None else None
                if not (isinstance(source, c_ast.ID) and _ADDRESS_NAMED.fullmatch(source.name)
                        and isinstance(target, c_ast.StructRef) and target.type == "." and isinstance(target.name, c_ast.ID)):
                    break
                owner = draft.resolve(draft.globals.get(target.name.name))
                layout = draft.layout(owner.type) if isinstance(owner, c_ast.TypeDecl) and isinstance(owner.type, c_ast.Struct) else None
                place = layout[2].get(target.field.name) if layout else None
                if place is None or place[0] != len(run) or draft.size_of(place[1]) != 1 \
                        or (run and (target.name.name != run[0][1] or int(source.name[2:], 16) != run[0][2] + len(run))):
                    break
                run.append((statement, target.name.name, int(source.name[2:], 16), source.name))
            if len(run) < 2:
                index += 1
                continue
            size, owner, first = len(run), run[0][1], run[0][3]
            tag = f"n64dk_bytes{size}"
            copy_ = _parse(f"struct {tag} {{ char c[{size}]; }}; void g(void) {{ *(struct {tag} *) &{owner} = "
                           f"*(struct {tag} *) &{first}; }}").ext[1].body.block_items[0]
            items[index:index + size] = [copy_]
            draft.added.append(f"struct {tag} {{ char c[{size}]; }};")
            changed += 1
            index += 1
    return changed


# --------------------------------------------------------------------------------- rule: forever

def endless_loops(draft: Draft, code: Code) -> int:
    """``L: if (c) { body; goto L; }`` becomes ``while (1) { if (!c) break; body; }``.

    A loop written ``for (;;) { if (done) break; step; }`` keeps its test at the top and jumps back from
    the bottom, and the compiler still treats it as a loop (it moves constants out of it).  A decompiler
    shows the test and a ``goto``; compiled from that, nothing is moved out and the registers differ
    (measured on a title: a 464-byte function).  Only a label whose one ``goto`` ends the ``if`` it marks
    is rewritten."""
    gotos: dict[str, int] = {}
    for node in _walk(draft.function.body):
        if isinstance(node, c_ast.Goto):
            gotos[node.name] = gotos.get(node.name, 0) + 1
    changed = 0
    for node in list(_walk(draft.function.body)):
        items = getattr(node, "block_items", None)
        if not isinstance(node, c_ast.Compound) or not items:
            continue
        for position, statement in enumerate(list(items)):
            test = statement.stmt if isinstance(statement, c_ast.Label) else None
            body = test.iftrue.block_items if isinstance(test, c_ast.If) and test.iffalse is None \
                and isinstance(test.iftrue, c_ast.Compound) else None
            if not body or not isinstance(body[-1], c_ast.Goto) or body[-1].name != statement.name \
                    or gotos.get(statement.name) != 1:
                continue
            loop = c_ast.While(c_ast.Constant("int", "1"), c_ast.Compound(
                [c_ast.If(_negated(test.cond), c_ast.Compound([c_ast.Break()]), None)] + body[:-1]))
            items[items.index(statement)] = loop
            changed += 1
    return changed


# --------------------------------------------------------------------------------- rule: sides
_MIRRORED = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}


def comparison_sides(draft: Draft, code: Code) -> int:
    """``a->x < a->y`` becomes ``a->y > a->x`` where the target loads ``y`` first.

    The compiler loads a comparison's operands in the order the source names them; a decompiler writes
    the comparison in the order of the instruction that tests it, which may be the other way round
    (measured on a title: a 284-byte function differed only in which of two bytes was loaded first).
    Where the target loads two offsets one right after the other, a comparison of members at those
    offsets written in the opposite order is turned round: each comparison by its own pair of loads when
    the draft's comparisons and the target's pairs line up one to one, else only where the target uses
    one order throughout."""
    loads = [(index, offset) for index, op, kind, offset in code.events if kind == "off" and op in _LOADS]
    pairs = {(a[1], b[1]) for a, b in zip(loads, loads[1:]) if 0 < b[0] - a[0] <= 2 and a[1] != b[1]}

    def offset_of(node):
        while isinstance(node, c_ast.Cast):
            node = node.expr
        call = _field_call(node)
        return None if call is None else call[2]

    found = []
    for node in _walk(draft.function.body):
        if isinstance(node, c_ast.BinaryOp) and node.op in _MIRRORED:
            left, right = offset_of(node.left), offset_of(node.right)
            if left is not None and right is not None and left != right:
                found.append((node, left, right))
    # the comparisons in the order written against the target's load pairs in the order executed: when
    # the two lists line up one to one, each comparison takes the order of its own pair
    kinds = {frozenset((left, right)) for _node, left, right in found}
    ordered = [(a[1], b[1]) for a, b in zip(loads, loads[1:]) if 0 < b[0] - a[0] <= 2 and frozenset((a[1], b[1])) in kinds]
    aligned = len(ordered) == len(found) and all(frozenset(pair) == frozenset((left, right))
                                                 for pair, (_node, left, right) in zip(ordered, found))
    changed = 0
    for position, (node, left, right) in enumerate(found):
        wanted = ordered[position] if aligned else ((right, left) if (right, left) in pairs and (left, right) not in pairs
                                                    else (left, right))
        if wanted == (right, left):
            node.left, node.right, node.op = node.right, node.left, _MIRRORED[node.op]
            changed += 1
    return changed


# --------------------------------------------------------------------------------- rule: ints
_NARROW = {"s8": (24, True), "s16": (16, True), "u8": (0xFF, False), "u16": (0xFFFF, False)}


def widened_locals(draft: Draft, code: Code) -> int:
    """Narrow locals become ``s32`` where the target never narrows a value to that width.

    A local declared ``s8`` makes the compiler sign-extend whatever is assigned to it; the load that fed
    it had already extended the byte, so the target has no such shifts and the draft has two instructions
    too many (measured on a title: a 284-byte function).  The target tells: with no ``sll``/``sra`` pair by
    24 (or 16) anywhere, no value was narrowed to a signed byte (or halfword) in a register, and with no
    ``andi`` by 0xFF (or 0xFFFF) none to an unsigned one; the locals of that type are then plain words."""
    present = set()
    words = [insn.word for insn in code.insns]
    for word in words:
        if word >> 26 == 0 and (word & 0x3F) == 0x03 and ((word >> 6) & 31) in (16, 24):      # sra by 16 or 24
            present.add((((word >> 6) & 31), True))
        if word >> 26 == 0x0C and (word & 0xFFFF) in (0xFF, 0xFFFF):                            # andi
            present.add(((word & 0xFFFF), False))
    arguments = {parameter.name for parameter in draft.parameters()}
    changed = 0
    for item in draft.function.body.block_items or []:
        if not isinstance(item, c_ast.Decl) or item.name in arguments or not isinstance(item.type, c_ast.TypeDecl) \
                or not isinstance(item.type.type, c_ast.IdentifierType):
            continue
        kind = " ".join(item.type.type.names)
        if kind in _NARROW and _NARROW[kind] not in present:
            item.type.type.names[:] = ["s32"]
            changed += 1
    return changed


# --------------------------------------------------------------------------------- rule: members

def member_globals(draft: Draft, code: Code) -> int:
    """A scalar global becomes the one member of a structure at the same address.

    The compiler takes a load from a plain scalar global to be independent of stores made through
    structure pointers and moves it above them; a load from a member of a global structure it leaves where
    the source put it.  A draft that declares a member of some larger object as a scalar of its own
    therefore gets its loads earlier than the target has them, and other registers with that (measured on
    a title: two functions of 316 and 496 bytes matched once one such global was a member).  The address
    and the type stay the same; only what the compiler may assume changes.  Which globals are members of
    something is not known, so all the scalars the function uses directly are changed together, as one
    variant."""
    declared: dict[str, str] = {}
    for found in re.finditer(r"^extern\s+(s8|u8|s16|u16|s32|u32)\s+(D_[0-9A-Fa-f]{8})\s*;", draft.head, re.M):
        declared[found.group(2)] = found.group(1)
    parents = _parents(draft.function.body)
    uses: dict[str, list] = {}
    skipped: set[str] = set()
    for node in _walk(draft.function.body):
        if isinstance(node, c_ast.ID) and node.name in declared and node.name not in draft.locals:
            parent = parents.get(id(node))
            if isinstance(parent, c_ast.UnaryOp) and parent.op == "&":
                skipped.add(node.name)                       # its address is used as such: left alone
            else:
                uses.setdefault(node.name, []).append((parent, node))
    changed = 0
    for name, places in uses.items():
        if name in skipped or name in draft.dropped:
            continue
        for parent, node in places:
            if not _replace(parent, node, c_ast.StructRef(c_ast.ID(name), ".", c_ast.ID("v"))):
                return -1
        draft.dropped.add(name)
        draft.added.append(f"extern struct {{ {declared[name]} v; }} {name};")
        changed += 1
    return changed


# --------------------------------------------------------------------------------- rule: cases

def switch_from_zero(draft: Draft, code: Code) -> int:
    """A ``switch`` whose table the target indexes from zero gets a ``case 0`` that does nothing.

    The compiler subtracts the lowest case before it tests the range, unless the lowest case is 0.  A
    decompiler lists only the cases that do something; where the program also had cases that only
    ``break`` below them, the target tests ``value < highest + 1`` with nothing subtracted, and the draft
    compiles to a subtraction and a shorter table (measured on a title: a 356-byte function).  The target
    tells: a ``sltiu`` by the highest case plus one, and none by the span of the listed cases."""
    bounds = {insn.word & 0xFFFF for insn in code.insns if insn.word >> 26 == 0x0B}        # sltiu immediates
    changed = 0
    for node in _walk(draft.function.body):
        if not isinstance(node, c_ast.Switch) or not isinstance(node.stmt, c_ast.Compound):
            continue
        values = []
        for item in node.stmt.block_items or []:
            cursor = item
            while isinstance(cursor, c_ast.Case):
                values.append(_number(cursor.expr))
                cursor = cursor.stmts[0] if cursor.stmts else None
        if not values or None in values or min(values) <= 0:
            continue
        low, high = min(values), max(values)
        if high + 1 in bounds and high - low + 1 not in bounds:
            node.stmt.block_items.insert(0, c_ast.Case(c_ast.Constant("int", "0"), [c_ast.Break()]))
            changed += 1
    return changed


# --------------------------------------------------------------------------------- rule: narrowed

def narrowed_results(draft: Draft, code: Code) -> int:
    """``s16 v = f();`` becomes ``s32 v = (s16) f();`` with ``f`` returning a word, where the target
    narrows right after the call.

    A function that returns ``int`` whose result the program keeps as a 16-bit value is narrowed once,
    where it is assigned (``sll``/``sra`` by 16 right after the ``jal``), and used as a word from then on.
    A decompiler gives the callee the narrow return type and the local the narrow type; compiled, nothing
    is narrowed at the call and every later use is narrowed instead (measured on a title: a 560-byte
    function).  Changed only when the target has such a pair within three instructions of a call."""
    words = [insn.word for insn in code.insns]
    after_call = False
    for index, word in enumerate(words):
        if word >> 26 == 3:                                  # jal
            window = words[index + 2:index + 5]
            if any(w >> 26 == 0 and (w & 0x3F) == 0 and ((w >> 6) & 31) == 16 and ((w >> 16) & 31) == 2 for w in window) \
                    and any(w >> 26 == 0 and (w & 0x3F) == 3 and ((w >> 6) & 31) == 16 for w in window):
                after_call = True
                break
    if not after_call:
        return 0
    changed = 0
    declared = {item.name: item for item in (draft.function.body.block_items or []) if isinstance(item, c_ast.Decl)}
    for node in list(_walk(draft.function.body)):
        if not (isinstance(node, c_ast.Assignment) and node.op == "=" and isinstance(node.lvalue, c_ast.ID)
                and isinstance(node.rvalue, c_ast.FuncCall) and isinstance(node.rvalue.name, c_ast.ID)):
            continue
        local = declared.get(node.lvalue.name)
        if local is None or not isinstance(local.type, c_ast.TypeDecl) or not isinstance(local.type.type, c_ast.IdentifierType) \
                or local.type.type.names != ["s16"]:
            continue
        callee = node.rvalue.name.name
        found = re.search(rf"^s16(\s+{re.escape(callee)}\s*\([^;{{}}]*\)\s*;[^\n]*)$", draft.head, re.M)
        if found is None:
            continue
        local.type.type.names[:] = ["s32"]
        node.rvalue = _cast(_type("s16"), node.rvalue)
        draft.dropped.add(callee)
        draft.added.append("s32" + found.group(1))
        changed += 1
    return changed


# --------------------------------------------------------------------------------- rule: chain
CHAIN_CASES = 3                # a switch of up to this many cases may have been an if/else chain


def case_chains(draft: Draft, code: Code) -> int:
    """A small ``switch`` without fall-through becomes ``if (x == a) { } else if (x == b) { }``.

    A decompiler writes a run of equality tests on one value as a ``switch`` ("irregular": no table).
    The compiler orders a switch's tests its own way (a binary search, the default first), so the round
    trip does not give the tests back in the order the program had them (measured on a title: the tail of
    a 560-byte function).  A switch of at most three cases, each ending in ``break`` or a bare ``return``,
    with no ``default`` that does anything, is written as the chain in case order."""
    changed = 0
    parents = _parents(draft.function.body)
    for node in list(_walk(draft.function.body)):
        if not isinstance(node, c_ast.Switch) or not isinstance(node.stmt, c_ast.Compound):
            continue
        arms = []
        for item in node.stmt.block_items or []:
            if isinstance(item, c_ast.Default):
                if any(not isinstance(inner, c_ast.Break) for inner in item.stmts or []):
                    arms = None
                    break
                continue
            if not isinstance(item, c_ast.Case) or not item.stmts or isinstance(item.stmts[0], c_ast.Case) \
                    or not (isinstance(item.stmts[-1], c_ast.Break)
                            or (isinstance(item.stmts[-1], c_ast.Return) and item.stmts[-1].expr is None)):
                arms = None
                break
            body = item.stmts[:-1]
            if any(isinstance(inner, (c_ast.Break, c_ast.Case, c_ast.Label)) for part in body for inner in _walk(part)):
                arms = None
                break
            arms.append((item.expr, body))
        if not arms or len(arms) > CHAIN_CASES or len(arms) < 2:
            continue
        chain = None
        for value, body in reversed(arms):
            chain = c_ast.If(c_ast.BinaryOp("==", copy.deepcopy(node.cond), value), c_ast.Compound(body), chain)
        if not _replace(parents.get(id(node)), node, chain):
            return -1
        changed += 1
    return changed


# --------------------------------------------------------------------------------- rule: literal
_FRAME_NAMED = re.compile(r"sp([0-9A-F]+)")


def copied_literals(draft: Draft, code: Code) -> int:
    """Word copies from a constant into consecutive frame slots become one initialised character array.

    ``char name[16] = "...";`` is compiled as unaligned word loads from the literal and stores to the frame,
    and a decompiler writes each word as a local of its own (``sp10 = M2C_UNALIGNED32(...)``, ``sp14 = ...``).
    As separate words the compiler loads them aligned (measured on a title: two functions of 288 and 320
    bytes).  The array is written as a structure of that many characters copied from the constant the
    draft already names by its address, which compiles to the same copy and leaves nothing unplaced."""
    items = draft.function.body.block_items or []
    run = []                                                  # (statement, frame offset, source offset, symbol)
    for statement in items:
        found = None
        if isinstance(statement, c_ast.Assignment) and statement.op == "=" and isinstance(statement.lvalue, c_ast.ID) \
                and _FRAME_NAMED.fullmatch(statement.lvalue.name) and isinstance(statement.rvalue, c_ast.FuncCall) \
                and isinstance(statement.rvalue.name, c_ast.ID) and statement.rvalue.name.name == "M2C_UNALIGNED32" \
                and statement.rvalue.args is not None and len(statement.rvalue.args.exprs) == 1:
            call = _field_call(statement.rvalue.args.exprs[0])
            if call is not None and isinstance(call[0], c_ast.UnaryOp) and call[0].op == "&" and isinstance(call[0].expr, c_ast.ID):
                found = (statement, int(statement.lvalue.name[2:], 16), call[2], call[0].expr.name)
        if found is not None and (not run or (found[3] == run[-1][3] and found[1] == run[-1][1] + 4 and found[2] == run[-1][2] + 4)):
            run.append(found)
        elif run:
            break
    if len(run) < 2 or run[0][2] != 0:
        return 0
    names = {statement.lvalue.name for statement, *_rest in run}
    first = run[0][0].lvalue.name
    copies = {id(statement) for statement, *_rest in run}
    for node in _walk(draft.function.body):
        if isinstance(node, c_ast.ID) and node.name in names - {first}:
            if not any(node is statement.lvalue for statement, *_rest in run):
                return 0                                     # a later word is used by itself: not one array
    size = 4 * len(run)
    symbol, tag = run[0][3], f"n64dk_text{4 * len(run)}"
    # the literal is named by its address, as the draft named it: every relocation can then be verified,
    # which a placeholder string in a section of its own could not be
    declaration = _parse(f"struct {tag} {{ char c[{size}]; }}; void g(void) {{ struct {tag} {first} = {symbol}; }}"
                         ).ext[1].body.block_items[0]
    draft.dropped.add(symbol)
    draft.added.append(f"struct {tag} {{ char c[{size}]; }};")
    draft.added.append(f"extern struct {tag} {symbol};")
    kept = [item for item in items if id(item) not in copies and not (isinstance(item, c_ast.Decl) and item.name in names)]
    # frame padding added for the draft stood in for the words that were dead stores there; the array holds them
    freed = 2 * ((size - 4) // 8)
    for item in list(kept):
        if isinstance(item, c_ast.Decl) and item.name == "pad" and isinstance(item.type, c_ast.ArrayDecl):
            words = _number(item.type.dim)
            if words is not None and words <= freed:
                kept.remove(item)
            elif words is not None:
                item.type.dim = c_ast.Constant("int", str(words - freed))
    position = next((i for i, item in enumerate(kept) if not isinstance(item, c_ast.Decl)), len(kept))
    kept.insert(position, declaration)
    draft.function.body.block_items = kept
    draft._collect_locals()
    return 1


# --------------------------------------------------------------------------------- rule: joined

def joined_stores(draft: Draft, code: Code) -> int:
    """``if (c) { G = A; ... } else { G = B; ... }`` becomes one store after the ``if`` when the target has one.

    Where both branches end by storing to the same global, the compiler keeps the value in a register and
    stores once where the branches meet; a decompiler shows a store in each branch, wherever the value
    was ready.  With two stores in the source the compiler emits two (measured on a title: two functions).
    Each store is moved to the end of its branch, where the compiler joins the two into one; this moves it
    past whatever the branch calls, so it is offered only when the target stores to the address once."""
    stores: dict[int, int] = {}
    for _index, op, kind, address in code.events:
        if kind == "at" and op in _STORES:
            stores[address] = stores.get(address, 0) + 1
    changed = 0
    for node in list(_walk(draft.function.body)):
        items = getattr(node, "block_items", None)
        if not isinstance(node, c_ast.Compound) or not items:
            continue
        for statement in list(items):
            if not isinstance(statement, c_ast.If) or not isinstance(statement.iftrue, c_ast.Compound) \
                    or not isinstance(statement.iffalse, c_ast.Compound):
                continue
            sets = []
            for branch in (statement.iftrue, statement.iffalse):
                found: dict[str, list] = {}
                for item in branch.block_items or []:
                    if isinstance(item, c_ast.Assignment) and item.op == "=" and isinstance(item.lvalue, c_ast.ID) \
                            and _ADDRESS_NAMED.fullmatch(item.lvalue.name) and item.lvalue.name not in draft.locals:
                        found.setdefault(item.lvalue.name, []).append(item)
                sets.append(found)
            for name in sorted(set(sets[0]) & set(sets[1])):
                declared = draft.globals.get(name)
                if len(sets[0][name]) != 1 or len(sets[1][name]) != 1 or stores.get(int(name[2:], 16)) != 1 \
                        or not isinstance(declared, (c_ast.TypeDecl, c_ast.PtrDecl)):
                    continue
                reads = [inner for inner in _walk(statement) if isinstance(inner, c_ast.ID) and inner.name == name]
                if len(reads) != 2:
                    continue                                 # the branches read it back: the store cannot move
                moved = 0
                for branch, found in zip((statement.iftrue, statement.iffalse), sets):
                    assignment = found[name][0]
                    if branch.block_items[-1] is not assignment:
                        branch.block_items.remove(assignment)
                        branch.block_items.append(assignment)
                        moved += 1
                changed += 1 if moved else 0
    return changed


# --------------------------------------------------------------------------------- rule: widths
_LOAD_FOR_TYPE = {"s8": "lb", "u8": "lbu", "s16": "lh", "u16": "lhu", "s32": "lw", "u32": "lw"}


def access_widths(draft: Draft, code: Code) -> int:
    """An offset access whose type the target never uses at that offset takes the type the target does use.

    A decompiler types a member from one use and writes that type at every access; where it guessed wrong
    the draft loads with the other extension or stores the wrong width (measured on a title: ``lh`` for the
    target's ``lhu`` at one offset, a word store for a byte store at another).  The target's instructions
    say which is right when every access it makes at that offset, through any register, agrees: then a
    read whose type would load differently, or a plain store whose width differs, is retyped."""
    loads: dict[int, set[str]] = {}
    stores: dict[int, set[int]] = {}
    for _index, op, kind, offset in code.events:
        if kind != "off":
            continue
        if op in _LOADS and op in _LOAD_FOR_TYPE.values():
            loads.setdefault(offset, set()).add(op)
        elif op in _STORES:
            stores.setdefault(offset, set()).add(_STORES[op])
    parents = _parents(draft.function.body)
    changed = 0
    for node in list(_walk(draft.function.body)):
        call = _field_call(node)
        offset = pointer = None
        if call is not None:
            pointer, offset = call[1], call[2]
        elif isinstance(node, c_ast.UnaryOp) and node.op == "*" and isinstance(node.expr, c_ast.Cast) \
                and isinstance(parents.get(id(node)), c_ast.Assignment) and parents[id(node)].lvalue is node \
                and not any(isinstance(part, c_ast.ID) and _ADDRESS_NAMED.fullmatch(part.name) for part in _walk(node.expr)):
            # a store through a computed pointer; one through a global's address is not an offset access,
            # and a read this way has no register to tie it to an offset
            pointer, offset = node.expr.to_type.type, 0
        if not isinstance(pointer, c_ast.PtrDecl) or not isinstance(pointer.type, c_ast.TypeDecl) \
                or not isinstance(pointer.type.type, c_ast.IdentifierType):
            continue
        names = pointer.type.type.names
        kind = " ".join(names)
        if kind not in _LOAD_FOR_TYPE:
            continue
        if call is not None and any(isinstance(part, c_ast.ID) and _ADDRESS_NAMED.fullmatch(part.name)
                                    for part in _walk(call[0])):
            continue                                         # reached from a global's address: not an offset access
        parent = parents.get(id(node))
        written = isinstance(parent, c_ast.Assignment) and parent.lvalue is node
        if written and parent.op == "=":
            widths = stores.get(offset, set())
            if len(widths) == 1 and _SIZES[kind] not in widths and next(iter(widths)) in (1, 2, 4):
                names[:] = [f"{kind[0]}{8 * next(iter(widths))}"]
                changed += 1
        elif not written and not (isinstance(parent, c_ast.UnaryOp) and parent.op == "&"):
            seen = loads.get(offset, set())
            if len(seen) == 1 and _LOAD_FOR_TYPE[kind] not in seen:
                names[:] = [_LOADS[next(iter(seen))]]
                changed += 1
    return changed


# --------------------------------------------------------------------------------- rule: passed
_MACRO_CALL = re.compile(r"M2C_\w+|MULTU?_HI")


def _calls(draft: Draft) -> list:
    """The function's calls in evaluation order (the arguments of a call before the call)."""
    out = []

    def walk(node):
        for _name, child in node.children():
            walk(child)
        if isinstance(node, c_ast.FuncCall):
            name = node.name.name if isinstance(node.name, c_ast.ID) else None
            if name is None or not _MACRO_CALL.fullmatch(name):
                out.append(node)
    walk(draft.function.body)
    return out


def _pair_calls(draft: Draft, code: Code) -> list[tuple[object, tuple[int, ...]]]:
    """(call in the draft, argument register states at the target's call) for the calls that can
    be told apart: a callee named by its address is paired by that address, the others in order."""
    sites, calls = code.call_sites(), _calls(draft)
    by_target: dict[int | None, list] = {}
    for target, state in sites:
        by_target.setdefault(target, []).append(state)
    by_name: dict[str | None, list] = {}
    for call in calls:
        name = call.name.name if isinstance(call.name, c_ast.ID) and call.name.name not in draft.locals else None
        by_name.setdefault(name, []).append(call)
    pairs, used = [], set()
    for name, group in by_name.items():
        found = re.fullmatch(r"func_([0-9A-Fa-f]{8})", name or "")
        if found:
            states = by_target.get(int(found.group(1), 16), [])
            used.add(int(found.group(1), 16))
            if len(states) == len(group):
                pairs += list(zip(group, states))
    indirect = by_name.get(None, [])
    if indirect and len(indirect) == len(by_target.get(None, [])):
        pairs += list(zip(indirect, by_target[None]))
    named = [call for call in calls if isinstance(call.name, c_ast.ID) and call.name.name not in draft.locals
             and not re.fullmatch(r"func_[0-9A-Fa-f]{8}", call.name.name)]
    rest = [(target, state) for target, state in sites if target is not None and target not in used]
    if named and len(named) == len(rest):
        seen: dict[str, int] = {}
        if all(seen.setdefault(call.name.name, target) == target for call, (target, _state) in zip(named, rest)):
            pairs += [(call, state) for call, (_target, state) in zip(named, rest)]
    return pairs


def passed_arguments(draft: Draft, code: Code) -> int:
    """Give a call the arguments the function passes on unchanged.

    When the target sets ``$a1`` for a call and has not touched ``$a0`` since the function was
    entered, the callee receives the function's own first argument.  The decompiler, which does
    not know the callee, lists only the registers it saw written.  The call gets the missing
    arguments back, the function the parameters they come from, and a prototype that says
    otherwise is replaced by an unprototyped declaration."""
    own = draft.parameters()
    if any(draft.is_float(parameter.type) or (draft.size_of(parameter.type) or 4) > 4 for parameter in own):
        return 0                                             # argument registers shift: not handled
    # the decompiler names a parameter after its argument slot: arg0-arg3 are $a0-$a3
    slots: dict[int, object] = {}
    for parameter in own:
        found = re.fullmatch(r"arg(\d+)", parameter.name or "")
        if not found:
            return 0
        slots[int(found.group(1))] = parameter
    changed, wanted = 0, set()
    for call, state in _pair_calls(draft, code):
        registers, floats = state[:4], state[4:]
        if SET in floats or MIXED in registers:
            continue
        written = [index for index in range(4) if registers[index] == SET]
        arguments = list(call.args.exprs) if call.args is not None else []
        if any(draft.is_float(draft.type_of(argument)) for argument in arguments):
            continue
        if len(arguments) < len(written):
            continue
        if len(arguments) > len(written):                    # the rest travel on the stack
            top = 3
            if any(registers[index] not in (SET, INCOMING) for index in range(4)):
                continue
        elif written:
            top = written[-1]
        else:
            continue
        missing = [index for index in range(top + 1) if registers[index] == INCOMING]
        if not missing or any(registers[index] == STALE for index in range(top + 1)):
            continue
        rebuilt, supply = [], iter(arguments)
        for index in range(top + 1):
            rebuilt.append(c_ast.ID(f"arg{index}") if registers[index] == INCOMING else next(supply))
        rebuilt += list(supply)
        callee = call.name.name if isinstance(call.name, c_ast.ID) else None
        declared = draft.resolve(draft.globals.get(callee)) if callee else None
        if isinstance(declared, c_ast.FuncDecl) and declared.args is not None:
            # a prototype would reject the longer list: keep its return type only
            draft.dropped.add(callee)
            bare = c_ast.Decl(callee, [], [], [], [], c_ast.FuncDecl(None, _declared(declared.type, callee)), None, None)
            draft.added.append(_Generator().visit(bare) + ";")
            draft.globals[callee] = bare.type
        call.args = c_ast.ExprList(rebuilt)
        wanted.update(missing)
        changed += 1
    if changed and wanted - set(slots):
        for index in wanted - set(slots):
            slots[index] = c_ast.Decl(f"arg{index}", [], [], [], [], c_ast.TypeDecl(f"arg{index}", [], None,
                                      c_ast.IdentifierType(["s32"])), None, None)
        if sorted(slots) != list(range(len(slots))) and min(set(slots) - wanted, default=4) < 4:
            return -1                                        # a register argument in between is unknown
        for index in range(min(4, max(slots) + 1)):          # stack arguments need every register slot
            slots.setdefault(index, c_ast.Decl(f"arg{index}", [], [], [], [], c_ast.TypeDecl(
                f"arg{index}", [], None, c_ast.IdentifierType(["s32"])), None, None))
        draft.function.decl.type.args = c_ast.ParamList([slots[index] for index in sorted(slots)])
        draft.dropped.add(draft.name)                        # an earlier prototype of the function itself
        draft._collect_locals()
    return changed


def _declared(node, name: str):
    """A copy of a return type declaring ``name``."""
    node = copy.deepcopy(node)
    cursor = node
    while cursor is not None:
        if isinstance(cursor, c_ast.TypeDecl):
            cursor.declname = name
            break
        cursor = getattr(cursor, "type", None)
    return node


# --------------------------------------------------------------------------------- rule: order
_LOAD_OF = {("s", 1): "lb", ("u", 1): "lbu", ("s", 2): "lh", ("u", 2): "lhu"}
_STORE_OF = {1: "sb", 2: "sh", 4: "sw", 8: "sd"}


def _memory_events(draft: Draft, statement) -> tuple[list, set, set, bool] | None:
    """What an expression statement does: ([(mnemonic, "at"|"off", address|offset)] of its loads and
    stores, locals read, locals written, whether it has an effect that cannot be placed)."""
    events, reads, writes = [], set(), set()
    opaque = False

    def access(kind, where, key, store: bool):
        nonlocal opaque
        resolved = draft.resolve(kind)
        size = draft.size_of(kind)
        if size is None or isinstance(resolved, (c_ast.ArrayDecl, c_ast.FuncDecl)) or (
                isinstance(resolved, c_ast.TypeDecl) and isinstance(resolved.type, (c_ast.Struct, c_ast.Union))):
            opaque = True
            return
        if draft.is_float(kind):
            name = {4: "swc1" if store else "lwc1", 8: "sdc1" if store else "ldc1"}.get(size)
        elif store:
            name = _STORE_OF.get(size)
        elif size == 4:
            name = "lw"
        else:
            names = resolved.type.names if isinstance(resolved, c_ast.TypeDecl) and \
                isinstance(resolved.type, c_ast.IdentifierType) else []
            name = _LOAD_OF.get(("u" if "unsigned" in names else "s", size))
        if name is None:
            opaque = True
            return
        events.append((name, where, key))

    def visit(node, store=False, both=False):
        nonlocal opaque
        if isinstance(node, c_ast.Assignment):
            visit(node.rvalue)
            visit(node.lvalue, store=True, both=node.op != "=")
            return
        if isinstance(node, c_ast.UnaryOp) and node.op in ("p++", "p--", "++", "--"):
            visit(node.expr, store=True, both=True)
            return
        if isinstance(node, c_ast.ID):
            if node.name in draft.locals:
                (writes if store else reads).add(node.name)
                if both:
                    reads.add(node.name)
                return
            kind = draft.globals.get(node.name)
            found = re.fullmatch(r"D_([0-9A-Fa-f]{8})", node.name)
            resolved = draft.resolve(kind)
            if isinstance(resolved, (c_ast.FuncDecl, c_ast.ArrayDecl)):
                return                                       # an address, not an access
            if kind is None or not found:
                opaque = True
                return
            if both:
                access(kind, "at", int(found.group(1), 16), False)
            access(kind, "at", int(found.group(1), 16), store)
            return
        if isinstance(node, c_ast.UnaryOp) and node.op == "&":
            inner = node.expr
            if isinstance(inner, c_ast.ID):
                if inner.name in draft.locals:
                    opaque = True                            # a local in memory
                return
            visit(inner.name if isinstance(inner, (c_ast.StructRef, c_ast.ArrayRef)) else inner)
            return
        call = _field_call(node)
        if call is not None:
            visit(call[0])
            if both:
                access(draft.pointee(call[1]), "off", call[2], False)
            access(draft.pointee(call[1]), "off", call[2], store)
            return
        if isinstance(node, c_ast.StructRef):
            base = draft.resolve(draft.type_of(node.name))
            if node.type == "->":
                base = draft.resolve(draft.pointee(base))
                visit(node.name)
            elif not (isinstance(node.name, c_ast.ID) and node.name.name not in draft.locals):
                opaque = True
                return
            layout = draft.layout(base.type) if isinstance(base, c_ast.TypeDecl) and isinstance(
                base.type, (c_ast.Struct, c_ast.Union)) else None
            if layout is None or node.field.name not in layout[2]:
                opaque = True
                return
            offset, kind = layout[2][node.field.name]
            where = "off"
            if node.type == ".":
                found = re.fullmatch(r"D_([0-9A-Fa-f]{8})", node.name.name)
                if not found:
                    opaque = True
                    return
                where, offset = "at", int(found.group(1), 16) + offset
            if both:
                access(kind, where, offset, False)
            access(kind, where, offset, store)
            return
        if isinstance(node, c_ast.UnaryOp) and node.op == "*":
            visit(node.expr)
            kind = draft.pointee(draft.type_of(node.expr))
            if kind is None:
                opaque = True
                return
            if both:
                access(kind, "off", 0, False)
            access(kind, "off", 0, store)
            return
        if isinstance(node, (c_ast.FuncCall, c_ast.ArrayRef)):
            opaque = True
            return
        if isinstance(node, c_ast.Cast):
            visit(node.expr, store, both)
            return
        if store and not isinstance(node, c_ast.Cast):
            opaque = True
            return
        for _name, child in node.children():
            visit(child)

    visit(statement)
    return events, reads, writes, opaque


def statement_order(draft: Draft, code: Code) -> int:
    """Put consecutive assignments in the order in which the target starts them.

    A decompiler lists statements in the order of their last instruction (the store).  GCC's
    scheduler keeps the order in which statements begin but moves a store of a loaded value away
    from its load, so ``a->x = b->y; a->z = 0;`` is emitted as load, store z, store x, and the draft
    has the two statements the other way round, which builds to a different order.  Each load and
    store of a run of assignments is located in the target by its width and offset (only when
    that is unique in the function); the run is then sorted by where each statement starts."""
    positions: dict[tuple, list[int]] = {}
    for index, name, where, key in code.events:
        positions.setdefault((name, where, key), []).append(index)
    changed = 0
    for node in list(_walk(draft.function.body)):
        if not isinstance(node, c_ast.Compound) or not node.block_items:
            continue
        items = node.block_items
        start = 0
        while start < len(items):
            run = []
            cursor = start
            while cursor < len(items) and isinstance(items[cursor], c_ast.Assignment):
                found = _memory_events(draft, items[cursor])
                if found is None or found[3]:
                    break
                first = None
                for event in found[0]:
                    where = positions.get(event, [])
                    if len(where) != 1:
                        first = None
                        break
                    first = where[0] if first is None else min(first, where[0])
                if first is None:
                    break
                run.append((first, cursor, found[1], found[2]))
                cursor += 1
            if len(run) > 1:
                ordered = sorted(run, key=lambda row: (row[0], row[1]))
                if [row[1] for row in ordered] != [row[1] for row in run]:
                    rank = {row[1]: place for place, row in enumerate(ordered)}
                    safe = True
                    for i, a in enumerate(run):
                        for b in run[i + 1:]:
                            if rank[a[1]] > rank[b[1]] and (a[3] & (b[2] | b[3]) or b[3] & a[2]):
                                safe = False                 # one needs a local the other sets
                    if safe:
                        items[start:start + len(run)] = [items[row[1]] for row in ordered]
                        changed += 1
            start = cursor + 1                               # the statement that ended the run is a barrier
    return changed


# --------------------------------------------------------------------------------- rule: fields
def _parents(root) -> dict:
    out = {}
    for node in _walk(root):
        for _name, child in node.children():
            out[id(child)] = node
    return out


def _field_call(node):
    """(base, pointer type, offset) of an ``M2C_FIELD(base, T *, offset)`` with a constant offset."""
    if not (isinstance(node, c_ast.FuncCall) and isinstance(node.name, c_ast.ID) and node.name.name == "M2C_FIELD"
            and node.args is not None and len(node.args.exprs) == 3 and isinstance(node.args.exprs[1], c_ast.Cast)):
        return None
    offset = _number(node.args.exprs[2])
    if offset is None or offset < 0:
        return None
    return node.args.exprs[0], node.args.exprs[1].to_type.type, offset


def struct_fields(draft: Draft, code: Code) -> int:
    """Turn the decompiler's offset accesses into members of structures.

    ``M2C_FIELD(p, s32 *, 0x20)`` is ``*(s32 *) ((s8 *) p + 0x20)``: to GCC 2.x a scalar reached
    through a pointer, which may alias any other scalar, so a load after a store to a global stays
    after it.  ``p->unk_20`` is a member of an aggregate at a varying address, which GCC 2.x
    assumes cannot be a scalar variable at a fixed address; its scheduler then moves the access
    across such stores, and the instruction order of the build changes.  Which of the two the
    source had is decided by building both.  Each pointer the function reaches members through
    (a parameter, a local, a global pointer, the address of a global, a pointer member) gets a
    structure laid out from the offsets and widths of its accesses."""
    body = draft.function.body
    parents = _parents(body)
    own = {parameter.name for parameter in draft.parameters()}

    def key_of(base, depth=0):
        if depth > 6:
            return None
        if isinstance(base, c_ast.ID):
            if base.name in draft.locals:
                kind = draft.resolve(draft.locals[base.name])
                if isinstance(kind, c_ast.PtrDecl) or (isinstance(kind, c_ast.TypeDecl) and isinstance(
                        kind.type, c_ast.IdentifierType) and draft.size_of(kind) == 4 and not draft.is_float(kind)):
                    return ("var", base.name)
                return None
            kind = draft.resolve(draft.globals.get(base.name))
            if isinstance(kind, c_ast.PtrDecl) or (isinstance(kind, c_ast.TypeDecl) and isinstance(
                    kind.type, c_ast.IdentifierType) and draft.size_of(kind) == 4 and not draft.is_float(kind)):
                return ("global", base.name)
            return None
        if isinstance(base, c_ast.UnaryOp) and base.op == "&" and isinstance(base.expr, c_ast.ID) \
                and base.expr.name not in draft.locals:
            kind = draft.resolve(draft.globals.get(base.expr.name))
            if isinstance(kind, c_ast.TypeDecl) and isinstance(kind.type, c_ast.IdentifierType):
                return ("object", base.expr.name)
            return None
        inner = _field_call(base)
        if inner is not None:
            parent = key_of(inner[0], depth + 1)
            if parent is not None and draft.size_of(draft.pointee(inner[1])) == 4 \
                    and not draft.is_float(draft.pointee(inner[1])):
                return ("member", parent, inner[2])
        return None

    # every access: key -> [(offset, member type, stores, loads, node, kind)]
    found: dict[tuple, list] = {}
    unusable: set[tuple] = set()
    for node in _walk(body):
        call = _field_call(node)
        if call is not None:
            key = key_of(call[0])
            member = draft.pointee(call[1])
            if key is None or member is None:
                continue
            parent = parents.get(id(node))
            stores = isinstance(parent, c_ast.Assignment) and parent.lvalue is node
            loads = not stores or parent.op != "="
            found.setdefault(key, []).append((call[2], member, stores, loads, node, "field"))
        elif isinstance(node, c_ast.UnaryOp) and node.op == "*" and isinstance(node.expr, c_ast.ID) \
                and node.expr.name in draft.locals:
            member = draft.pointee(draft.locals[node.expr.name])
            if member is None or isinstance(draft.resolve(member), c_ast.FuncDecl):
                continue
            parent = parents.get(id(node))
            stores = isinstance(parent, c_ast.Assignment) and parent.lvalue is node
            found.setdefault(("var", node.expr.name), []).append(
                (0, member, stores, not stores or parent.op != "=", node, "deref"))
    if not found:
        return 0
    # a variable used in a way a structure pointer cannot be: leave it
    for node in _walk(body):
        if isinstance(node, (c_ast.ArrayRef, c_ast.StructRef)) and isinstance(node.name, c_ast.ID):
            unusable.add(("var", node.name.name))
            unusable.add(("global", node.name.name))
            unusable.add(("object", node.name.name))
        if isinstance(node, c_ast.UnaryOp) and node.op in ("p++", "p--", "++", "--") and isinstance(node.expr, c_ast.ID):
            unusable.add(("var", node.expr.name))
        if isinstance(node, c_ast.ID) and ("object", node.name) in found:
            parent = parents.get(id(node))
            if not (isinstance(parent, c_ast.UnaryOp) and parent.op == "&"):
                unusable.add(("object", node.name))
        if isinstance(node, c_ast.FuncCall) and isinstance(node.name, c_ast.ID):
            unusable.add(("var", node.name.name))           # called through: a function pointer
    for key in found:
        if key[0] == "var" and all(row[5] == "deref" for row in found[key]):
            unusable.add(key)                                # a plain typed pointer: nothing to gain

    def text_of(kind) -> str:
        return _Generator()._generate_type(kind, emit_declname=False)

    layouts: dict[tuple, dict[int, object]] = {}
    for key, rows in found.items():
        if key in unusable:
            continue
        members: dict[int, tuple] = {}
        valid = True
        for offset, member, stores, loads, _node, _kind in rows:
            size = draft.size_of(member)
            if size is None or size not in (1, 2, 4, 8) or offset % min(size, 4) or \
                    isinstance(draft.resolve(member), (c_ast.ArrayDecl, c_ast.FuncDecl)):
                valid = False
                break
            if offset not in members:
                members[offset] = (member, size, loads)
                continue
            current, held, loaded = members[offset]
            if held != size:
                valid = False
                break
            if text_of(current) == text_of(member):
                members[offset] = (current, held, loaded or loads)
                continue
            pointer_now, pointer_new = draft.pointee(current) is not None, draft.pointee(member) is not None
            if draft.is_float(current) != draft.is_float(member):
                valid = False
                break
            if size == 4 and pointer_now != pointer_new:
                members[offset] = (current if pointer_now else member, held, loaded or loads)
            elif loads and loaded and size < 4:
                valid = False                                # read both signed and unsigned
                break
            elif loads and not loaded:
                members[offset] = (member, held, True)
        ordered = sorted(members)
        if not valid or any(a + members[a][1] > b for a, b in zip(ordered, ordered[1:])):
            continue
        layouts[key] = {offset: members[offset][0] for offset in ordered}
    # a member reached through needs its parent
    for key in sorted(layouts, key=lambda k: -len(repr(k))):
        cursor = key
        while cursor[0] == "member":
            if cursor[1] not in layouts or cursor[2] not in layouts[cursor[1]]:
                layouts.pop(key, None)
                break
            cursor = cursor[1]
    if not layouts:
        return 0
    taken = set(draft.structs)
    tags: dict[tuple, str] = {}
    for key in layouts:
        stem = "Fields_" + re.sub(r"\W", "_", "_".join(str(part) if not isinstance(part, tuple) else "m"
                                                        for part in _flatten(key)))
        tag, number = stem, 1
        while tag in taken:
            number += 1
            tag = f"{stem}_{number}"
        taken.add(tag)
        tags[key] = tag

    def pointer_to(tag: str, name=None):
        return c_ast.PtrDecl([], c_ast.TypeDecl(name, [], None, c_ast.Struct(tag, None)))

    for key, tag in tags.items():
        if key[0] == "member" and key[1] in layouts:
            layouts[key[1]][key[2]] = pointer_to(tag)
    for key, tag in tags.items():
        declarations, cursor = [], 0
        for offset in sorted(layouts[key]):
            if offset > cursor:
                declarations.append(c_ast.Decl(f"pad_{cursor:X}", [], [], [], [], c_ast.ArrayDecl(
                    c_ast.TypeDecl(f"pad_{cursor:X}", [], None, c_ast.IdentifierType(["char"])),
                    c_ast.Constant("int", hex(offset - cursor)), []), None, None))
            member = _declared(layouts[key][offset], f"unk_{offset:X}")
            declarations.append(c_ast.Decl(f"unk_{offset:X}", [], [], [], [], member, None, None))
            cursor = offset + (draft.size_of(layouts[key][offset]) or 4)
        struct = c_ast.Struct(tag, declarations)
        draft.structs[tag] = struct
        draft.added.append(_Generator().visit(c_ast.Decl(None, [], [], [], [], struct, None, None)) + ";")
    # the declarations of the things the members are reached through
    retyped = False
    for key, tag in tags.items():
        if key[0] == "var":
            for node in list(_walk(draft.function.decl)) + list(_walk(body)):
                if isinstance(node, c_ast.Decl) and node.name == key[1] and not isinstance(node.type, c_ast.FuncDecl):
                    node.type = pointer_to(tag, key[1])
            retyped = retyped or key[1] in own
        elif key[0] == "global":
            draft.dropped.add(key[1])
            draft.added.append(f"extern struct {tag} *{key[1]};")
            draft.globals[key[1]] = pointer_to(tag, key[1])
        elif key[0] == "object":
            draft.dropped.add(key[1])
            draft.added.append(f"extern struct {tag} {key[1]};")
            draft.globals[key[1]] = c_ast.TypeDecl(key[1], [], None, c_ast.Struct(tag, None))
    if retyped:
        draft.dropped.add(draft.name)
    draft._collect_locals()
    # the accesses themselves
    changed = 0
    for key, rows in found.items():
        if key not in tags:
            continue
        for offset, _member, _stores, _loads, node, kind in rows:
            base = node.expr if kind == "deref" else node.args.exprs[0]
            if key[0] == "object":
                replacement = c_ast.StructRef(base.expr, ".", c_ast.ID(f"unk_{offset:X}"))
            else:
                replacement = c_ast.StructRef(base, "->", c_ast.ID(f"unk_{offset:X}"))
            if _replace(body, node, replacement):
                changed += 1
    return changed


def _flatten(key):
    for part in key:
        if isinstance(part, tuple):
            yield from _flatten(part)
        else:
            yield part if not isinstance(part, int) else f"{part:X}"


# --------------------------------------------------------------------------------- the generator
def variants(source: str, name: str, *, code: bytes, vram: int, pipeline: str,
             flags: tuple[str, ...] = (), family: str | None = None) -> list[tuple[str, str]]:
    """[(label, complete C translation unit)]; [] when this is not an optimised GCC function."""
    if c_parser is None or not code or len(code) % 4:
        return []
    try:
        facts = Code(code, vram)
        if not _optimised_gcc(pipeline, tuple(flags), family, facts.insns):
            return []
        out: list[tuple[str, str]] = []
        seen = {source}
        placed = positional_parameters(source, name)
        if placed is not None:
            # every other rewrite starts from the draft with its parameters in their places
            source = placed
            seen.add(source)
            out.append(("parameters", source))

        def offer(label: str, draft: Draft) -> None:
            text = draft.text()
            if text not in seen:
                seen.add(text)
                out.append((label, text))

        fixes = (("passed", passed_arguments), ("bytes", byte_arithmetic), ("symbols", symbol_addresses))
        typed = fixes + (("widths", access_widths), ("sides", comparison_sides), ("cases", switch_from_zero),
                         ("narrowed", narrowed_results))

        def build(rules):
            draft, applied = Draft(source, name), []
            for label, rule in rules:
                count = rule(draft, facts)
                if count < 0:                                # the rule left the tree half changed
                    return None, []
                if count:
                    applied.append(label)
            return draft, applied

        # 1. what the decompiler got wrong, corrected together
        draft, applied = build(fixes)
        if applied:
            offer("+".join(applied), draft)
        draft, applied = build(typed)
        if "widths" in applied or "sides" in applied or "cases" in applied or "narrowed" in applied:
            offer("+".join(applied), draft)
        # not part of the base: a word in place of a halfword index changes the address arithmetic of
        # functions that were right (measured: one function lost with it in the base, one gained)
        draft, applied = build(typed + (("ints", widened_locals),))
        if "ints" in applied:
            offer("+".join(applied), draft)
        draft, applied = build(typed + (("chain", case_chains),))
        if "chain" in applied:
            offer("+".join(applied), draft)
        draft, applied = build(typed + (("members", member_globals),))
        if "members" in applied:
            offer("+".join(applied), draft)
        draft, applied = build(typed + (("addressed", addressed_globals),))
        if "addressed" in applied:
            offer("+".join(applied), draft)
        draft, applied = build(typed + (("signs", sign_tests),))
        if "signs" in applied:
            offer("+".join(applied), draft)
        draft, applied = build(typed + (("step", shared_step),))
        if "step" in applied:
            offer("+".join(applied), draft)
        draft, applied = build(typed + (("reuse", reused_variables), ("stride", strided_arrays)))
        if "reuse" in applied or "stride" in applied:
            offer("+".join(applied), draft)
        draft, applied = build(typed + (("grouped", grouped_globals),))
        if "grouped" in applied:
            offer("+".join(applied), draft)
        draft, applied = build(typed + (("inner", inner_pointers), ("tails", branch_tails)))
        if "inner" in applied or "tails" in applied:
            offer("+".join(applied), draft)
        draft, applied = build(typed + (("literal", copied_literals),))
        if "literal" in applied:
            offer("+".join(applied), draft)
        draft, applied = build(typed + (("returns", swapped_returns),))
        if "returns" in applied:
            offer("+".join(applied), draft)
        draft, applied = build(typed + (("scan", scanning_loops), ("block", block_copies), ("forever", endless_loops)))
        if "scan" in applied or "block" in applied or "forever" in applied:
            offer("+".join(applied), draft)
        draft, applied = build(typed + (("shifts", split_shifts),))
        if "shifts" in applied:
            offer("+".join(applied), draft)
        draft, applied = build(typed + (("assoc", constant_last),))
        if "assoc" in applied:
            offer("+".join(applied), draft)
        draft, applied = build(typed + (("joined", joined_stores),))
        if "joined" in applied:
            offer("+".join(applied), draft)
        shape = (("step", shared_step), ("reuse", reused_variables), ("stride", strided_arrays),
                 ("grouped", grouped_globals), ("inner", inner_pointers), ("tails", branch_tails),
                 ("literal", copied_literals), ("joined", joined_stores), ("addressed", addressed_globals),
                 ("returns", swapped_returns), ("assoc", constant_last), ("shifts", split_shifts),
                 ("scan", scanning_loops), ("block", block_copies), ("forever", endless_loops),
                 ("ints", widened_locals), ("chain", case_chains))
        draft, applied = build(typed + shape)
        if len([label for label, _rule in shape if label in applied]) > 1:
            out.insert(1 if placed is not None else 0, ("+".join(applied), draft.text()))   # together, and first
            seen.add(draft.text())
        draft, applied = build(typed + (("order", statement_order),))
        if "order" in applied:
            offer("+".join(applied), draft)
        # 2. the same with members of structures in place of offset accesses.  The two forms
        #    differ only in how accesses are ordered around scalars at fixed addresses, so
        #    without a global or a local in memory there is nothing to gain
        if facts.accesses or facts.frame_accesses:
            draft, applied = build((("fields", struct_fields),) + fixes)
            if "fields" in applied:
                offer("+".join(applied), draft)
        return out[:MAX_VARIANTS]
    except Exception:                                        # a rewrite is optional; the draft stands
        return []
