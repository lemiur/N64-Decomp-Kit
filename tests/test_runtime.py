"""Runtime model (n64dk.analysis.runtime) on a hand-assembled image with libultra-like stubs."""
import json
import struct
import unittest

from n64dk.analysis import runtime

DELTA, SIZE = 0x80001000, 0x2000
AT, V0, A0, A1, A2, A3, T6, T7, T8, T9, SP, RA = 1, 2, 4, 5, 6, 7, 14, 15, 24, 25, 29, 31
BOOT, ENTRY, WRAP, CALLER, ENTRY2 = 0x80001000, 0x80001200, 0x80001400, 0x80001500, 0x80001600
CREATE, START, QCREATE, SETEVENT, RECV, PIDMA, SPLOAD = (0x80002000 + 0x10 * k for k in range(7))
Q, QBUF, THREAD, TASK, MB, STACKTOP = 0x80002800, 0x80002820, 0x80002900, 0x80002A00, 0x80002A80, 0x80002C00
NAMES = {CREATE: "osCreateThread", START: "osStartThread", QCREATE: "osCreateMesgQueue",
         SETEVENT: "osSetEventMesg", RECV: "osRecvMesg", PIDMA: "osPiStartDma", SPLOAD: "osSpTaskLoad"}


def I(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def R(rs, rt, rd, sa, funct):
    return (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | funct


def lui(rt, imm): return I(15, 0, rt, imm)
def addiu(rt, rs, imm): return I(9, rs, rt, imm)
def lw(rt, off, base): return I(35, base, rt, off)
def sw(rt, off, base): return I(43, base, rt, off)
def move(rd, rs): return R(rs, 0, rd, 0, 0x25)
def jal(target): return (3 << 26) | ((target >> 2) & 0x3FFFFFF)


JR_RA = R(RA, 0, 0, 0, 8)

BOOT_CODE = [
    addiu(SP, SP, -0x28), sw(RA, 0x24, SP),
    lui(A0, 0x8000), addiu(A0, A0, 0x2800), lui(A1, 0x8000), addiu(A1, A1, 0x2820),
    jal(QCREATE), addiu(A2, 0, 8),                                   # osCreateMesgQueue(Q, buf, 8)
    addiu(A0, 0, 5), lui(A1, 0x8000), addiu(A1, A1, 0x2800),
    jal(SETEVENT), move(A2, 0),                                      # osSetEventMesg(OS_EVENT_SI, Q, 0)
    lui(T6, 0x8000), addiu(T6, T6, 0x2C00), sw(T6, 0x10, SP),        # stack top
    addiu(T7, 0, 10), sw(T7, 0x14, SP),                              # priority
    lui(A0, 0x8000), addiu(A0, A0, 0x2900), addiu(A1, 0, 3),
    lui(A2, 0x8000), addiu(A2, A2, 0x1200),
    jal(CREATE), move(A3, 0),                                        # osCreateThread(&t, 3, ENTRY, 0, top, 10)
    lui(A0, 0x8000), jal(START), addiu(A0, A0, 0x2900),
    lw(RA, 0x24, SP), JR_RA, addiu(SP, SP, 0x28)]
ENTRY_CODE = [
    addiu(SP, SP, -0x28), sw(RA, 0x24, SP),
    lui(A0, 0x8000), addiu(A0, A0, 0x2800), addiu(A1, SP, 0x20),
    jal(RECV), addiu(A2, 0, 1),                                      # osRecvMesg(Q, &msg, BLOCK)
    lui(T6, 0x8000), addiu(T6, T6, 0x4000), sw(T6, 0x10, SP),
    addiu(T7, 0, 0x100), sw(T7, 0x14, SP),
    lui(T8, 0x8000), addiu(T8, T8, 0x2800), sw(T8, 0x18, SP),
    lui(A0, 0x8000), addiu(A0, A0, 0x2A80), move(A1, 0), move(A2, 0), lui(A3, 0x0010),
    jal(PIDMA), 0,                                                   # osPiStartDma(mb, 0, READ, 0x100000, ...)
    addiu(T9, 0, 1), lui(AT, 0x8000), sw(T9, 0x2A00, AT),            # task.type = M_GFXTASK
    lui(A0, 0x8000), jal(SPLOAD), addiu(A0, A0, 0x2A00),
    lw(RA, 0x24, SP), JR_RA, addiu(SP, SP, 0x28)]
WRAP_CODE = [addiu(SP, SP, -0x20), sw(RA, 0x1C, SP), jal(CREATE), 0, lw(RA, 0x1C, SP), JR_RA, addiu(SP, SP, 0x20)]
CALLER_CODE = [addiu(SP, SP, -0x20), sw(RA, 0x1C, SP), lui(A2, 0x8000), jal(WRAP), addiu(A2, A2, 0x1600),
               lw(RA, 0x1C, SP), JR_RA, addiu(SP, SP, 0x20)]
# the stubs do the stores that corroborate a pattern recognition (OSThread.id/context.pc,
# OSMesgQueue.msgCount/msg); each fits the 0x10 bytes between stubs
STUBS = {START: [JR_RA, lw(V0, 0, A0)],
         CREATE: [sw(A1, 0x14, A0), JR_RA, sw(A2, 0x11C, A0), 0],
         QCREATE: [sw(A2, 0x10, A0), JR_RA, sw(A1, 0x14, A0), 0]}


def image():
    data = bytearray(SIZE)
    rows = []
    pieces = [("boot", BOOT, BOOT_CODE), ("entry", ENTRY, ENTRY_CODE), ("wrap", WRAP, WRAP_CODE),
              ("caller", CALLER, CALLER_CODE), ("entry2", ENTRY2, [JR_RA, 0])]
    pieces += [(NAMES[a], a, STUBS.get(a, [JR_RA, 0])) for a in sorted(NAMES)]
    for name, address, words in pieces:
        data[address - DELTA:address - DELTA + 4 * len(words)] = b"".join(struct.pack(">I", w) for w in words)
        rows.append((name, address, 4 * len(words)))
    return bytes(data), rows


def model(names=NAMES):
    data, rows = image()
    return runtime.analyse_images([("main", data, DELTA, rows)], dict(names))


class RuntimeTest(unittest.TestCase):
    def test_threads_direct_and_through_a_wrapper(self):
        m = model()
        threads = {t["entry"]: t for t in m.threads}
        self.assertEqual(set(threads), {ENTRY, ENTRY2})
        direct = threads[ENTRY]
        self.assertEqual((direct["id"], direct["pri"], direct["stack"], direct["thread"], direct["confidence"]),
                         (3, 10, "0x80002C00", "0x80002900", 1.0))
        self.assertIn("forwarded parameter", " ".join(threads[ENTRY2]["provenance"]))
        self.assertEqual(len(m.entrypoints()), 2)

    def test_queue_event_dma_and_task_graphs(self):
        m = model()
        q = m.queues[Q]
        self.assertEqual(q["count"], 8)
        self.assertEqual(q["events"], ["OS_EVENT_SI"])
        self.assertEqual([r["function"] for r in q["receivers"]], ["0x80001200"])
        self.assertEqual([r.get("implicit") for r in q["senders"]], ["DMA completion"])
        [dma] = m.dma
        self.assertEqual((dma["device"], dma["ram"], dma["size"], dma["queue"], dma["direction"]),
                         ("0x00100000", "0x80004000", 0x100, "0x80002800", 0))
        self.assertEqual(m.tasks[TASK]["type"], ["M_GFXTASK"])

    def test_ownership_and_variants(self):
        m = model()
        self.assertIn(ENTRY, m.owners[RECV])
        self.assertNotIn(BOOT, m.owners)
        flagged = {v["name"] for v in m.variants}
        self.assertIn("osCreateThread", flagged)                 # the stub does not read $a3
        self.assertNotIn("osStartThread", flagged)               # reads $a0 like the stock function
        json.dumps(m.as_dict())

    def test_call_pattern_without_sdk_names(self):
        m = model(names={})
        self.assertEqual(m.api[CREATE], ("osCreateThread?", "thread-create", "pattern"))
        self.assertEqual(m.api[QCREATE], ("osCreateMesgQueue?", "queue-create", "pattern"))
        self.assertEqual({t["entry"] for t in m.threads}, {ENTRY, ENTRY2})
        self.assertTrue(all(t["confidence"] <= 0.5 for t in m.threads))
        self.assertTrue(all((t["confidence_label"], t["provenance_kind"]) == ("low", "pattern") for t in m.threads))
        self.assertEqual(m.queues[Q]["created"][0]["confidence_label"], "low")
        self.assertIn("recognised by call pattern only", {v["why"] for v in m.variants})
        self.assertEqual({c["address"]: c["accepted"] for c in m.pattern_candidates},
                         {"0x80002000": True, "0x80002020": True})

    def test_sdk_named_results_are_high(self):
        m = model()
        self.assertEqual(m.pattern_candidates, [])
        self.assertTrue(all((t["confidence_label"], t["provenance_kind"]) == ("high", "sdk-name") for t in m.threads))
        recognised = m.as_dict()["api_recognised"]["0x80002000"]
        self.assertEqual((recognised["confidence"], recognised["provenance"]), ("high", "sdk-name"))

    def test_load_names(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sdk_symbols.txt"
            path.write_text("osCreateThread = 0x800079B0; // type:func\n// comment\nbad line\n")
            self.assertEqual(runtime.load_names(path), {0x800079B0: "osCreateThread"})


# --------------------------------------------------------------------------- pattern fallback
# A title whose SDK matcher named no osCreateThread: the real create (-O0 style, arguments spilled
# and reloaded), an engine process creator called with the same argument shape from 4 sites, and a
# wrapper that forwards to the real create.
MAIN2, TRUE0, LOOK, ALLOC, WRAP2 = 0x80001000, 0x80002000, 0x80002100, 0x80002200, 0x80002300
E1, E2, E3, E4 = 0x80001800, 0x80001840, 0x80001880, 0x800018C0

MAIN2_CODE = [addiu(SP, SP, -0x28), sw(RA, 0x24, SP)]
for _e in (E1, E2, E3, E1):                                          # process_create(0, 0x10, e, ?, 0)
    MAIN2_CODE += [sw(0, 0x10, SP), lui(A2, 0x8000), addiu(A2, A2, _e & 0xFFFF), addiu(A1, 0, 0x10),
                   jal(LOOK), move(A0, 0)]
for _k in range(2):                                                  # osCreateThread(&a, 1, E4, 0, top, 10) twice
    MAIN2_CODE += [lui(T6, 0x8000), addiu(T6, T6, 0x2C00), sw(T6, 0x10, SP), addiu(T7, 0, 10), sw(T7, 0x14, SP),
                   lui(A0, 0x8000), addiu(A0, A0, 0x2900), addiu(A1, 0, 1), lui(A2, 0x8000),
                   addiu(A2, A2, E4 & 0xFFFF), jal(TRUE0), move(A3, 0)]
MAIN2_CODE += [lui(T6, 0x8000), addiu(T6, T6, 0x2C00), sw(T6, 0x10, SP),   # wrapper(&b, 2, E3, 0, top)
               lui(A0, 0x8000), addiu(A0, A0, 0x2980), addiu(A1, 0, 2), lui(A2, 0x8000),
               addiu(A2, A2, E3 & 0xFFFF), jal(WRAP2), move(A3, 0),
               lw(RA, 0x24, SP), JR_RA, addiu(SP, SP, 0x28)]
TRUE0_CODE = [addiu(SP, SP, -0x18), sw(A0, 0x18, SP), sw(A1, 0x1C, SP), sw(A2, 0x20, SP),
              lw(T6, 0x18, SP), lw(T7, 0x20, SP), sw(T7, 0x11C, T6),       # t->context.pc = entry
              lw(T8, 0x1C, SP), sw(T8, 0x14, T6),                          # t->id = id
              JR_RA, addiu(SP, SP, 0x18)]
LOOK_CODE = [addiu(SP, SP, -0x20), sw(RA, 0x1C, SP),
             sw(A2, 0x8, A0),                                              # entry into its $a0, not at 0x11C
             sw(A2, 0x18, SP), jal(ALLOC), 0,
             lw(T6, 0x18, SP), sw(T6, 0x11C, V0),                          # 0x11C, but of a fresh object
             lw(RA, 0x1C, SP), JR_RA, addiu(SP, SP, 0x20)]
WRAP2_CODE = [addiu(SP, SP, -0x20), sw(RA, 0x1C, SP), jal(TRUE0), 0, lw(RA, 0x1C, SP), JR_RA, addiu(SP, SP, 0x20)]


def fallback_model(names=None):
    data = bytearray(SIZE)
    rows = []
    pieces = [("main", MAIN2, MAIN2_CODE), ("create", TRUE0, TRUE0_CODE), ("process", LOOK, LOOK_CODE),
              ("alloc", ALLOC, [JR_RA, 0]), ("wrapper", WRAP2, WRAP2_CODE)]
    pieces += [(f"e{k}", e, [JR_RA, 0]) for k, e in enumerate((E1, E2, E3, E4))]
    for name, address, words in pieces:
        data[address - DELTA:address - DELTA + 4 * len(words)] = b"".join(struct.pack(">I", w) for w in words)
        rows.append((name, address, 4 * len(words)))
    return runtime.analyse_images([("main", bytes(data), DELTA, rows)], dict(names or {}))


class PatternFallbackTest(unittest.TestCase):
    def test_body_stores_are_recorded_through_stack_spills(self):
        m = fallback_model()
        self.assertIn((4, 0x11C, 6), m.functions[TRUE0].param_stores)
        self.assertIn((4, 0x14, 5), m.functions[TRUE0].param_stores)
        self.assertEqual(m.functions[LOOK].param_stores, {(4, 0x8, 6)})   # the fresh object is not a parameter
        self.assertEqual(m.functions[WRAP2].param_stores, set())

    def test_true_create_found_look_alikes_rejected(self):
        m = fallback_model()
        self.assertEqual(m.api, {TRUE0: ("osCreateThread?", "thread-create", "pattern")})
        candidates = {c["address"]: c for c in m.pattern_candidates}
        self.assertEqual({a: c["accepted"] for a, c in candidates.items()},
                         {"0x80002000": True, "0x80002100": False, "0x80002300": False})
        look = candidates["0x80002100"]
        self.assertEqual((look["call_sites"], look["shape_matches"], look["distinct_entries"]), (4, 4, 3))
        self.assertIn("0x11C", look["rejected_because"])
        self.assertIn("OSThread.id", " ".join(candidates["0x80002000"]["corroboration"]))
        self.assertTrue(all((c["confidence"], c["provenance"]) == ("low", "pattern") for c in candidates.values()))

    def test_threads_are_distinct_tuples_not_call_sites(self):
        m = fallback_model()
        threads = {t["entry"]: t for t in m.threads}
        self.assertEqual(set(threads), {E4, E3})                     # none of E1, E2 (process entries)
        direct = threads[E4]
        self.assertEqual(len(direct["sites"]), 2)
        self.assertEqual((direct["id"], direct["pri"], direct["stack"], direct["thread"]),
                         (1, 10, "0x80002C00", "0x80002900"))
        self.assertEqual((threads[E3]["id"], threads[E3]["thread"]), (2, "0x80002980"))
        self.assertIn("forwarded parameter", " ".join(threads[E3]["provenance"]))
        self.assertTrue(all((t["confidence_label"], t["provenance_kind"]) == ("low", "pattern") for t in m.threads))
        out = m.as_dict()
        self.assertEqual((out["thread_count"], out["thread_entries"], out["thread_create_sites"]), (2, 2, 3))
        self.assertEqual((out["api_recognised"]["0x80002000"]["confidence"],
                          out["api_recognised"]["0x80002000"]["provenance"]), ("low", "pattern"))
        self.assertEqual({e["confidence_label"] for e in m.entrypoints()}, {"low"})
        json.dumps(out)

    def test_sdk_name_bypasses_the_fallback(self):
        m = fallback_model({TRUE0: "osCreateThread"})
        self.assertEqual(m.pattern_candidates, [])
        self.assertEqual({t["entry"] for t in m.threads}, {E4, E3})
        self.assertTrue(all(t["confidence_label"] == "high" for t in m.threads))

    def test_uncorroborated_queue_leaf_rejected(self):
        data, rows = image()
        stubbed = bytearray(data)
        stubbed[QCREATE - DELTA:QCREATE - DELTA + 16] = struct.pack(">4I", JR_RA, 0, 0, 0)
        m = runtime.analyse_images([("main", bytes(stubbed), DELTA, rows)], {})
        self.assertNotIn(QCREATE, m.api)
        [queue] = [c for c in m.pattern_candidates if c["kind"] == "queue-create"]
        self.assertFalse(queue["accepted"])
        self.assertIn("msgCount", queue["rejected_because"])


if __name__ == "__main__":
    unittest.main()
