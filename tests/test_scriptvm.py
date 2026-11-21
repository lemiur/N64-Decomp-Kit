"""Script-VM detection: dispatch loops (byte and halfword opcodes), handler summaries, bytecode listing,
CFG and byte ownership, on hand-assembled interpreters."""
import struct
import unittest

from n64dk.analysis import loaders_generic as lg
from n64dk.analysis import scriptvm

ZERO, AT, V0, A0, A1 = 0, 1, 2, 4, 5
T0, T1, T2, T3, T4, T9 = 8, 9, 10, 11, 12, 25
S0, S1, S2, RA = 16, 17, 18, 31


def R(fn, rs=0, rt=0, rd=0, sa=0):
    return (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | fn


def I(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def addu(rd, rs, rt): return R(0x21, rs, rt, rd)
def or_(rd, rs, rt): return R(0x25, rs, rt, rd)
def move(rd, rs): return or_(rd, rs, ZERO)
def sll(rd, rt, sa): return R(0x00, 0, rt, rd, sa)
def jr(rs): return R(0x08, rs)
def jalr(rs): return R(0x09, rs, 0, RA)
def addiu(rt, rs, imm): return I(9, rs, rt, imm)
def sltiu(rt, rs, imm): return I(11, rs, rt, imm)
def lui(rt, imm): return I(15, 0, rt, imm)
def ori(rt, rs, imm): return I(13, rs, rt, imm)
def lb(rt, off, base): return I(32, base, rt, off)
def lbu(rt, off, base): return I(36, base, rt, off)
def lhu(rt, off, base): return I(37, base, rt, off)
def lw(rt, off, base): return I(35, base, rt, off)
def sw(rt, off, base): return I(43, base, rt, off)
def jal(addr): return (3 << 26) | ((addr >> 2) & 0x3FFFFFF)


NOP, JR_RA = 0, jr(RA)


def br(op, rs, rt, label):
    return lambda pc, labels: I(op, rs, rt, (labels[label] - pc - 4) >> 2)


def j(label):
    return lambda pc, labels: (2 << 26) | ((labels[label] >> 2) & 0x3FFFFFF)


def assemble(vram, items):
    labels, pc = {}, vram
    for item in items:
        if isinstance(item, tuple):
            labels[item[1]] = pc
        else:
            pc += 4
    words, pc = [], vram
    for item in items:
        if isinstance(item, tuple):
            continue
        words.append(item(pc, labels) if callable(item) else item)
        pc += 4
    return struct.pack(f">{len(words)}I", *words), labels


def image(size, parts):
    data = bytearray(size)
    for offset, blob in parts:
        data[offset:offset + len(blob)] = blob
    return bytes(data)


def reader(data, base):
    def read_word(a):
        o = a - base
        return struct.unpack_from(">I", data, o)[0] if 0 <= o <= len(data) - 4 and not o & 3 else None
    return read_word


# A byte-code interpreter with a jump table inside the function.  IP in $s0 (from $a0), a call
# stack in $s1, an accumulator in $s2.  Opcodes: 0 halt, 1 add imm8, 2 jump rel16, 3 call rel16,
# 4 return, 5 branch-if-zero rel8 (relative targets count from the byte after the opcode).
BYTE_VM = 0x80100000
BYTE_ITEMS = [
    move(S0, A0),
    ("label", "loop"), lbu(T0, 0, S0), addiu(S0, S0, 1), sltiu(AT, T0, 6), br(4, AT, ZERO, "halt"),
    sll(T0, T0, 2), lui(AT, 0x8010), addu(AT, AT, T0), lw(T1, 0x200, AT), jr(T1), NOP,
    ("label", "halt"), JR_RA, NOP,
    ("label", "add"), lbu(T2, 0, S0), addu(S2, S2, T2), j("loop"), addiu(S0, S0, 1),
    ("label", "jump"), lb(T2, 0, S0), lbu(T3, 1, S0), sll(T2, T2, 8), or_(T2, T2, T3), j("loop"),
    addu(S0, S0, T2),
    ("label", "call"), lb(T2, 0, S0), lbu(T3, 1, S0), sll(T2, T2, 8), or_(T2, T2, T3), addiu(T4, S0, 2),
    sw(T4, 0, S1), addiu(S1, S1, 4), j("loop"), addu(S0, S0, T2),
    ("label", "ret"), addiu(S1, S1, -4), lw(S0, 0, S1), j("loop"), NOP,
    ("label", "bz"), lb(T2, 0, S0), br(5, S2, ZERO, "skip"), addiu(S0, S0, 1), addu(S0, S0, T2),
    ("label", "skip"), j("loop"), NOP,
]
BYTE_CODE, BYTE_LABELS = assemble(BYTE_VM, BYTE_ITEMS)
BYTE_TABLE = struct.pack(">6I", *(BYTE_LABELS[n] for n in ("halt", "add", "jump", "call", "ret", "bz")))
#            add 5 | call +6  | jump +7  | junk | add 7 | ret | junk | bz -15 | halt
SCRIPT = bytes([1, 5, 3, 0, 6, 2, 0, 7, 0xFF, 1, 7, 4, 0xFF, 5, 0xF1, 0])
BYTE_IMAGE = image(0x400, [(0, BYTE_CODE), (0x200, BYTE_TABLE), (0x300, SCRIPT)])
S = BYTE_VM + 0x300

# A halfword-opcode interpreter calling handler functions through a table; the IP goes to the
# handler in $a0 and comes back in $v0 (0 ends the loop).
HALF_VM = 0x80200000
HALF_ITEMS = [
    move(S0, A0),
    ("label", "loop"), lhu(T0, 0, S0), addiu(S0, S0, 2), sll(T0, T0, 2), lui(T1, 0x8020), addu(T1, T1, T0),
    lw(T9, 0x200, T1), move(A0, S0), jalr(T9), NOP, move(S0, V0), br(5, S0, ZERO, "loop"), NOP, JR_RA, NOP,
]
HALF_CODE = assemble(HALF_VM, HALF_ITEMS)[0]
HANDLERS = [(0x100, [lhu(T0, 0, A0), JR_RA, addiu(V0, A0, 2)]),     # 0: one halfword operand
            (0x110, [JR_RA, addiu(V0, A0, 4)]),                      # 1: skips four bytes
            (0x120, [lw(V0, 0, A0), JR_RA, NOP]),                    # 2: jump to an absolute address
            (0x130, [JR_RA, or_(V0, ZERO, ZERO)])]                   # 3: halt (null IP)
HALF_TABLE = struct.pack(">4I", *(HALF_VM + off for off, _code in HANDLERS))
HALF_SCRIPT = bytes([0, 0, 0x12, 0x34, 0, 1, 0xAA, 0xAA, 0xAA, 0xAA, 0, 2, 0x80, 0x20, 0x03, 0x14,
                     0xFF, 0xFF, 0xFF, 0xFF, 0, 3])
HALF_IMAGE = image(0x400, [(0, HALF_CODE), (0x200, HALF_TABLE), (0x300, HALF_SCRIPT)] +
                   [(off, assemble(HALF_VM + off, code)[0]) for off, code in HANDLERS])


def byte_vm():
    found = scriptvm.find_interpreters(BYTE_CODE, BYTE_VM, reader(BYTE_IMAGE, BYTE_VM), (BYTE_VM, BYTE_VM + 0x400))
    assert len(found) == 1, found
    return found[0]


def half_vm():
    found = scriptvm.find_interpreters(HALF_CODE, HALF_VM, reader(HALF_IMAGE, HALF_VM), (HALF_VM, HALF_VM + 0x400))
    assert len(found) == 1, found
    return found[0]


class DetectionTests(unittest.TestCase):
    def test_byte_opcode_jump_table(self):
        vm = byte_vm()
        self.assertEqual((vm.flavour, vm.width, vm.ip_reg, vm.ip_arg), ("jump-table", 1, S0, 0))
        self.assertEqual((vm.table, len(vm.entries), vm.bound), (BYTE_VM + 0x200, 6, 6))
        self.assertEqual((vm.opcode_offset, vm.dispatch_advance, vm.delta), (0, 1, 1))
        kinds = [vm.handlers[a].kind for a in vm.entries]
        self.assertEqual(kinds, ["halt", "plain", "jump", "call", "return", "branch"])
        add, jump, call, bz = (vm.handlers[vm.entries[i]] for i in (1, 2, 3, 5))
        self.assertEqual((add.advance, add.operands), (1, [[0, 1, False]]))
        self.assertEqual(jump.target, {"mode": "rel", "operand": [0, 2, True], "bias": 0})
        self.assertEqual(call.advance, 2)
        self.assertEqual((bz.advance, bz.target["bias"]), (1, 1))

    def test_halfword_opcode_call_table(self):
        vm = half_vm()
        self.assertEqual((vm.flavour, vm.width, vm.ip_update, vm.ip_arg), ("call-table", 2, "v0", 0))
        self.assertEqual((vm.dispatch_advance, vm.delta, len(vm.entries)), (2, 2, 4))
        kinds = [vm.handlers[a].kind for a in vm.entries]
        self.assertEqual(kinds, ["plain", "plain", "jump", "halt"])
        self.assertEqual([vm.handlers[a].advance for a in vm.entries[:2]], [2, 4])
        self.assertEqual(vm.handlers[vm.entries[2]].target["mode"], "abs")

    def test_no_interpreter_in_a_one_shot_switch_or_a_copy(self):
        items = [lbu(T0, 0, A0), addiu(A0, A0, 1), sll(T0, T0, 2), lui(AT, 0x8030), addu(AT, AT, T0),
                 lw(T1, 0x100, AT), jr(T1), NOP, ("label", "c0"), JR_RA, addiu(V0, ZERO, 1),
                 ("label", "c1"), JR_RA, addiu(V0, ZERO, 2)]
        code, labels = assemble(0x80300000, items)
        data = image(0x200, [(0, code), (0x100, struct.pack(">2I", labels["c0"], labels["c1"]))])
        self.assertEqual(scriptvm.find_interpreters(code, 0x80300000, reader(data, 0x80300000)), [])
        copy = assemble(0x80300000, [("label", "l"), lbu(T0, 0, A1), addiu(A1, A1, 1), I(40, A0, T0, 0),
                                     br(5, A1, T1, "l"), addiu(A0, A0, 1), JR_RA, NOP])[0]
        self.assertEqual(scriptvm.find_interpreters(copy, 0x80300000), [])


class BytecodeTests(unittest.TestCase):
    def test_listing_cfg_and_ownership(self):
        out = scriptvm.disassemble(byte_vm(), BYTE_IMAGE, BYTE_VM, S)
        listing = [(int(i["at"], 16) - S, i["kind"], i["length"]) for i in out["instructions"]]
        self.assertEqual(listing, [(0x0, "plain", 2), (0x2, "call", 3), (0x5, "jump", 3), (0x9, "plain", 2),
                                   (0xB, "return", 1), (0xD, "branch", 2), (0xF, "halt", 1)])
        self.assertEqual(out["calls"], [{"site": lg.hx(S + 2), "target": lg.hx(S + 9), "returns_to": lg.hx(S + 5)}])
        self.assertEqual(out["owned_bytes"], 14)
        self.assertEqual(out["owned"], [[lg.hx(S), lg.hx(S + 8)], [lg.hx(S + 9), lg.hx(S + 0xC)],
                                        [lg.hx(S + 0xD), lg.hx(S + 0x10)]])
        blocks = [(int(b["start"], 16) - S, b["instructions"], [int(x, 16) - S for x in b["succs"]])
                  for b in out["blocks"]]
        self.assertEqual(blocks, [(0x0, 2, [0x9, 0x5]), (0x5, 1, [0xD]), (0x9, 2, []), (0xD, 1, [0xF, 0x0]),
                                  (0xF, 1, [])])
        self.assertEqual(out["conflicts"], [])
        self.assertTrue(out["complete"])

    def test_halfword_script(self):
        start = HALF_VM + 0x300
        out = scriptvm.disassemble(half_vm(), HALF_IMAGE, HALF_VM, start)
        self.assertEqual([(int(i["at"], 16) - start, i["kind"]) for i in out["instructions"]],
                         [(0x0, "plain"), (0x4, "plain"), (0xA, "jump"), (0x14, "halt")])
        self.assertEqual(out["instructions"][0]["operands"], [{"at": 0, "value": 0x1234}])
        self.assertEqual(out["owned_bytes"], 18)
        self.assertEqual(len(out["blocks"]), 2)

    def test_unknown_opcode_stops_the_walk(self):
        out = scriptvm.disassemble(byte_vm(), BYTE_IMAGE, BYTE_VM, S + 8)      # 0xFF: outside the table
        self.assertEqual(out["instructions"], [])
        self.assertFalse(out["complete"])
        self.assertIn("no handler", out["stopped"][0]["why"])

    def test_script_start_from_a_constant_argument(self):
        caller = assemble(0x80100100, [lui(A0, 0x8010), ori(A0, A0, 0x0300), jal(BYTE_VM), NOP, JR_RA, NOP])[0]
        calls = lg.call_constants(lg.decode_range(caller, 0x80100100))
        self.assertEqual(scriptvm.script_starts(byte_vm(), calls), [S])


if __name__ == "__main__":
    unittest.main()
