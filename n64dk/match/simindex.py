"""Semantic function-similarity index: one function against a corpus, at several levels at once.

``similar`` answers "which matched function has nearly the same opcodes"; ``lineage`` answers
"which function is the same machine code linked elsewhere".  Neither says *how* two functions are
related when they are neither, and that is the useful question for donor search across titles,
regions and compilers.  Each function is described once (``describe``) at these levels:

``exact``      SHA-1 of the bytes (trailing alignment ``nop`` words dropped);
``masked``     the same over ``lineage.masked_words``: relocated fields blanked;
``renamed``    the masked words with every register field renumbered by first use ($zero, $gp,
               $sp and $ra keep their numbers): the same code under another register allocation;
``semantic``   ``ir.fingerprint`` (symbolic addresses) when the function lifts: its digest and
               its effect/operation/memory features; absent, never guessed, when it does not;
``cfg``        the IR skeleton when lifted, plus branch, loop, switch and call counts always;
``const``      immediates that are not relocations, and ``lui`` halves that are not addresses;
``str``        text the function addresses through ``%hi``/``%lo`` pairs, when a data lookup
               is given (printable, NUL-terminated, 4+ characters);
``mem``        load/store shapes: width, base kind (sp, gp, absolute, pointer) and, for
               pointer bases, the field offset;
``loop``       back edges, indirect jumps (switches), indirect calls;
``neighbours`` callees and callers by key (the caller supplies keys that mean the same across
               titles, for instance masked hashes: ``index_image`` does this).

``Index`` keeps compact entries, an inverted index over hashed features (postings capped: a
feature in too many functions keeps its document frequency but stops nominating candidates)
and streams to and from JSON lines.  ``query`` returns, for each candidate, separate metrics
(structural, semantic, call-neighbourhood, constants/strings, corpus rarity), a confidence label,
a match class and provenance.  Classes: ``exact``, ``relocation-only``, ``register-renamed``,
``same-source-other-compiler`` (semantics agree, instructions do not), ``modified``,
``related-helper``, ``likely-library`` (the hash or digest is carried by ``library_titles`` or
more titles).  Functions under ``SMALL`` instructions are penalised explicitly: they are generic.

A similarity is a lead for donors, names, prototypes and recipes, never evidence of a match:
every result carries ``proof: False``, and only a rebuild verified against the target's bytes
counts.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..analysis import lineage
from . import similar

SMALL = 8                    # instructions: below this a function is generic, whatever matches
LIBRARY_TITLES = lineage.LIBRARY_TITLES
MAX_POSTING = 400            # entries per feature posting; beyond it the feature is generic
MAX_LIFT = 1500              # words: larger functions are not lifted (time), the rest still indexed
LIFT_VALUES_PER_WORD = 256   # SSA-value budget of one lift pass: this per word plus LIFT_VALUES_BASE;
LIFT_VALUES_BASE = 4096      # a lift beyond it is abandoned (entry not lifted), bounding its memory
LIFT_STATS: Counter = Counter()   # lift outcomes in this process (diagnostics; never part of a result)
MAX_TOKENS = 1024            # instruction shape tokens kept per entry
CAPS = {"gram": 96, "sem": 96, "cfg": 32, "const": 48, "str": 16, "mem": 48, "loop": 16, "neigh": 64}
KINDS = tuple(CAPS)
REPEAT = 4                   # a multiset is kept as a set of (token, occurrence) up to this many
STRUCT_FLOOR = 0.5           # structural similarity below this is reported as 0.0 (banded edit distance)
LONG = 256                   # instructions: longer pairs pass a shape-sketch gate before the edit distance
SEM_HIGH, STRUCT_HIGH = 0.85, 0.8
MODIFIED = 0.6
RELATED = 0.5
CONFIDENCE = (("high", 0.8), ("medium", 0.6), ("low", 0.4))
FIXED_GPR = frozenset((0, 28, 29, 31))      # $zero $gp $sp $ra: ABI roles, never renamed
GRAM = 4

_LOADS = {0x20: 1, 0x21: 2, 0x23: 4, 0x24: 1, 0x25: 2, 0x27: 4, 0x37: 8, 0x31: 4, 0x35: 8, 0x22: 4, 0x26: 4,
          0x1A: 8, 0x1B: 8}
_STORES = {0x28: 1, 0x29: 2, 0x2B: 4, 0x3F: 8, 0x39: 4, 0x3D: 8, 0x2A: 4, 0x2E: 4, 0x2C: 8, 0x2D: 8}
_FPR_MEM = {0x31, 0x35, 0x39, 0x3D}
_CONST_OPS = {0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x18, 0x19}
_BRANCH_OPS = {0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17}
_REGIMM_BRANCH = {0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13}


def _h(text: str) -> int:
    """Stable 32-bit feature hash (Python's ``hash`` is salted per process)."""
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=4).digest(), "big")


def _sha(payload: bytes) -> str:
    return hashlib.sha1(payload).hexdigest()[:16]


def _s16(value: int) -> int:
    return value - 0x10000 if value & 0x8000 else value


def _bottom(values, cap: int) -> tuple[int, ...]:
    """A consistent bottom-k sketch: the ``cap`` smallest hashes (Jaccard-preserving)."""
    return tuple(sorted(set(values))[:cap])


def _multiset(kind: str, counts: Counter) -> list[int]:
    return [_h(f"{kind}:{token}#{n}") for token, count in counts.items() for n in range(1, min(count, REPEAT) + 1)]


def _jaccard(a, b) -> float | None:
    a, b = set(a), set(b)
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


def _words(code: bytes) -> list[int]:
    words = list(struct.unpack(f">{len(code) // 4}I", code[:len(code) & ~3]))
    while words and words[-1] == 0:                       # alignment padding is not code
        words.pop()
    return words


def _shape(word: int) -> int:
    """Instruction kind without registers or immediates (the structural alphabet)."""
    if word == 0:
        return 1
    op = word >> 26
    if op == 0:
        return 64 + (word & 63)
    if op == 1:
        return 128 + ((word >> 16) & 31)
    if op == 0x11:
        return 256 + ((word >> 21) & 31) * 64 + (word & 63)
    return op


def _renamed(masked: list[int]) -> bytes:
    """Masked words with register fields renumbered by first use (GPR and FPR apart)."""
    gmap: dict[int, int] = {}
    fmap: dict[int, int] = {}

    def g(r: int) -> int:
        return r if r in FIXED_GPR else gmap.setdefault(r, 32 + len(gmap))

    def f(r: int) -> int:
        return fmap.setdefault(r, len(fmap))

    out = []
    for w in masked:
        op, rs, rt, rd, fd = w >> 26, (w >> 21) & 31, (w >> 16) & 31, (w >> 11) & 31, (w >> 6) & 31
        if op == 0:
            base, regs = w & ~(0x7FFF << 11), (g(rs), g(rt), g(rd))
        elif op == 1:
            base, regs = w & ~(31 << 21), (g(rs),)
        elif op in (2, 3):
            base, regs = w, ()
        elif op == 0x10:
            base, regs = w & ~(31 << 16), (g(rt),)
        elif op == 0x11:
            if rs in (0, 1, 4, 5):                        # mfc1/dmfc1/mtc1/dmtc1
                base, regs = w & ~(0x3FF << 11), (g(rt), f(rd))
            elif rs in (2, 6):                            # cfc1/ctc1: fs is a control register
                base, regs = w & ~(31 << 16), (g(rt),)
            elif rs == 8:                                 # bc1t/bc1f
                base, regs = w, ()
            else:                                         # arithmetic: ft, fs, fd
                base, regs = w & ~(0x7FFF << 6), (f(rt), f(rd), f(fd))
        elif op in _FPR_MEM:
            base, regs = w & ~(0x3FF << 16), (g(rs), f(rt))
        elif op == 0x0F:
            base, regs = w & ~(31 << 16), (g(rt),)
        else:
            base, regs = w & ~(0x3FF << 16), (g(rs), g(rt))
        out.append(struct.pack(">I", base) + bytes(regs))
    return b"".join(out)


def _ascii(raw: bytes | None) -> str | None:
    if not raw:
        return None
    end = raw.find(b"\0")
    if end < 4:
        return None
    text = raw[:end]
    if all(32 <= c < 127 or c in (9, 10, 13) for c in text):
        return text.decode("ascii")[:48]
    return None


@dataclass(slots=True)
class Entry:
    """Compact derived features only: strings, ints, tuples of ints (never IR or code bytes)."""
    key: str
    title: str
    image: str
    name: str
    vram: int
    n: int                                   # instructions (padding dropped)
    exact: str
    masked: str
    renamed: str
    semantic: str = ""                       # IR digest when the lift is complete
    lifted: bool = False
    tokens: tuple = ()                       # instruction shapes, at most MAX_TOKENS
    feats: dict = field(default_factory=dict)   # kind -> tuple of feature hashes
    callees: int = 0

    def as_dict(self) -> dict:
        return {"key": self.key, "title": self.title, "image": self.image, "name": self.name, "vram": self.vram,
                "n": self.n, "exact": self.exact, "masked": self.masked, "renamed": self.renamed,
                "semantic": self.semantic, "lifted": self.lifted, "tokens": list(self.tokens),
                "feats": {k: list(v) for k, v in sorted(self.feats.items())}, "callees": self.callees}

    @classmethod
    def from_dict(cls, d: dict) -> "Entry":
        return cls(d["key"], d["title"], d.get("image", ""), d["name"], d["vram"], d["n"], d["exact"], d["masked"],
                   d["renamed"], d.get("semantic", ""), d.get("lifted", False), tuple(d.get("tokens") or ()),
                   {k: tuple(v) for k, v in (d.get("feats") or {}).items()}, d.get("callees", 0))


def _semantic_features(fp) -> tuple[str, list[int], list[int]]:
    """The compact part of an ``ir.Fingerprint`` an entry keeps: (digest, or "" when incomplete;
    ``sem`` feature hashes; extra ``cfg`` feature hashes).  Nothing of the IR survives it."""
    semantic = fp.digest[:16] if fp.complete else ""
    sem = _multiset("e", fp.effects) + _multiset("o", fp.ops) + _multiset("im", fp.memory)
    nodes = Counter((kind, len(t), any(x <= n for x in t)) for n, (kind, t) in enumerate(fp.cfg))
    cfg = _multiset("n", Counter({repr(k): v for k, v in nodes.items()})) + [_h(f"c:ir:{fp.cfg!r}")]
    return semantic, sem, cfg


def _semantic(words: list[int], vram: int) -> tuple[str, list[int], list[int]] | None:
    """``_semantic_features`` of the function's ``ir.fingerprint``, or None when it does not lift
    (any failure, the SSA-value budget included).  The IR and the fingerprint die here: an entry
    never holds either.  ``LIFT_STATS`` counts the outcomes (diagnostics only)."""
    if not words or len(words) > MAX_LIFT:
        LIFT_STATS["not-lifted:too-large" if words else "not-lifted:empty"] += 1
        return None
    try:
        from . import ir
        fn = ir.lift(words, vram, max_values=LIFT_VALUES_PER_WORD * len(words) + LIFT_VALUES_BASE)
        fp = ir.fingerprint(fn, symbolic_addresses=True)
        del fn
        out = _semantic_features(fp)
    except Exception as error:                           # noqa: BLE001 - absent, never guessed
        LIFT_STATS[f"failed:{type(error).__name__}"] += 1
        return None
    LIFT_STATS["lifted"] += 1
    return out


def describe(title: str, name: str, vram: int, code: bytes, image: str = "", data_at=None, call_key=None,
             callers=(), lift: bool = True) -> Entry:
    """One function's index entry.

    ``data_at(address, length) -> bytes | None`` reads the title's data (for strings);
    ``call_key(address) -> str | None`` names a callee so that it compares across titles;
    ``callers`` are the callers' keys."""
    words = _words(code)
    raw = b"".join(struct.pack(">I", w) for w in words)
    masked = lineage.masked_words(raw)
    feats: dict[str, list[int]] = {kind: [] for kind in KINDS}
    shapes = [_shape(w) for w in words]
    feats["gram"] = [_h("g:" + ",".join(map(str, shapes[i:i + GRAM]))) for i in range(max(1, len(shapes) - GRAM + 1))]

    consts: Counter = Counter()
    memory: Counter = Counter()
    strings: Counter = Counter()
    loops: Counter = Counter()
    callees: Counter = Counter()
    hi: dict[int, int] = {}
    branches = calls = 0
    for i, w in enumerate(words):
        op, rs, rt, imm = w >> 26, (w >> 21) & 31, (w >> 16) & 31, w & 0xFFFF
        fn = w & 63
        address = None
        if op == 0x0F:
            hi[rt] = imm << 16
            if not 0x8000 <= imm < 0x8100:               # not an address in KSEG0 RDRAM: a constant
                consts[f"hi:{imm:x}"] += 1
            continue
        if op in _CONST_OPS and rs in hi:
            address = (hi[rs] + _s16(imm)) & 0xFFFFFFFF if op != 0x0D else hi[rs] | imm
        elif op in _CONST_OPS and 29 not in (rs, rt):
            consts[f"k:{_s16(imm) if op in (0x08, 0x09, 0x0A, 0x18, 0x19) else imm}"] += 1
        if op in _LOADS or op in _STORES:
            width = _LOADS.get(op) or _STORES.get(op)
            kind = "ld" if op in _LOADS else "st"
            if rs == 29:
                base = "sp"
            elif rs == 28:
                base = "gp"
            elif rs in hi:
                base = "abs"
                address = (hi[rs] + _s16(imm)) & 0xFFFFFFFF
            else:
                base = f"p{_s16(imm)}"
            memory[f"{kind}{width}{'f' if op in _FPR_MEM else ''}:{base}"] += 1
        if address is not None and data_at is not None:
            text = _ascii(data_at(address, 128))
            if text is not None:
                strings[text] += 1
        if op in _BRANCH_OPS or (op == 1 and rt in _REGIMM_BRANCH) or (op == 0x11 and rs == 8):
            branches += 1
            if _s16(imm) < 0:
                loops["back-edge"] += 1
        elif op == 3:
            calls += 1
            target = ((vram + 4 * i + 4) & 0xF0000000) | ((w & 0x3FFFFFF) << 2)
            key = call_key(target) if call_key is not None else None
            if key:
                callees[key] += 1
        elif op == 0 and fn == 8 and rs != 31:
            loops["indirect-jump"] += 1
        elif op == 0 and fn == 9:
            loops["indirect-call"] += 1
        written = lineage._written(w)              # its destination no longer holds a %hi half
        if written is not None:
            hi.pop(written, None)
    feats["const"] = _multiset("k", consts) + [_h(f"s:{t}") for t in strings]
    feats["str"] = [_h(f"s:{t}") for t in strings]
    feats["mem"] = _multiset("m", memory)
    feats["loop"] = _multiset("l", loops)
    feats["cfg"] = [_h(f"c:branches:{min(branches, 32).bit_length()}"), _h(f"c:calls:{min(calls, 32).bit_length()}"),
                    _h(f"c:loops:{min(loops['back-edge'], 8)}"), _h(f"c:switch:{min(loops['indirect-jump'], 4)}")]
    neigh = [_h(f"callee:{k}") for k in callees] + [_h(f"caller:{k}") for k in callers if k]
    feats["neigh"] = neigh
    semantic, lifted = "", False
    lifted_features = _semantic(words, vram) if lift else None
    if lifted_features is not None:
        lifted = True
        semantic, feats["sem"], extra_cfg = lifted_features
        feats["cfg"] += extra_cfg
    return Entry(key=f"{title}:{image}:{name}" if image else f"{title}:{name}", title=title, image=image, name=name,
                 vram=vram, n=len(words), exact=_sha(raw), masked=_sha(b"".join(struct.pack(">I", w) for w in masked)),
                 renamed=_sha(_renamed(masked)), semantic=semantic, lifted=lifted, tokens=tuple(shapes[:MAX_TOKENS]),
                 feats={kind: _bottom(values, CAPS[kind]) for kind, values in feats.items() if values},
                 callees=sum(callees.values()))


def _label(score: float) -> str:
    for name, floor in CONFIDENCE:
        if score >= floor:
            return name
    return "none"


_RANK = {"high": 3, "medium": 2, "low": 1, "none": 0}


class Index:
    """Streaming similarity index (see the module docstring)."""

    def __init__(self, library_titles: int = LIBRARY_TITLES, max_posting: int = MAX_POSTING):
        self.library_titles, self.max_posting = library_titles, max_posting
        self.entries: list[Entry] = []
        self.by_key: dict[str, int] = {}
        self.levels: dict[str, dict[str, list[int]]] = {"exact": {}, "masked": {}, "renamed": {}, "semantic": {}}
        self.carriers: dict[tuple[str, str], set] = {}      # (level, hash) -> titles, capped
        self.postings: dict[int, list[int] | None] = {}
        self.df: Counter = Counter()

    def __len__(self) -> int:
        return len(self.entries)

    # ---- building ---------------------------------------------------------------------------
    def add(self, entry: Entry) -> bool:
        if entry.key in self.by_key:
            return False
        number = len(self.entries)
        self.entries.append(entry)
        self.by_key[entry.key] = number
        for level in ("exact", "masked", "renamed", "semantic"):
            value = getattr(entry, level)
            if not value:
                continue
            ids = self.levels[level].setdefault(value, [])
            if len(ids) < self.max_posting:
                ids.append(number)
            if level in ("masked", "semantic"):
                titles = self.carriers.setdefault((level, value), set())
                if len(titles) <= self.library_titles:
                    titles.add(entry.title)
        for feature in {f for kind, values in entry.feats.items() for f in values}:
            self.df[feature] += 1
            posting = self.postings.get(feature, [])
            if posting is None:
                continue
            if len(posting) >= self.max_posting:
                self.postings[feature] = None            # generic: frequency kept, no nominations
            else:
                posting.append(number)
                self.postings[feature] = posting
        return True

    def save(self, path) -> None:
        path = Path(path)
        with path.open("w") as handle:
            handle.write(json.dumps({"format": "n64dk-simindex", "version": 1, "library_titles": self.library_titles,
                                     "max_posting": self.max_posting, "entries": len(self.entries)}) + "\n")
            for entry in self.entries:
                handle.write(json.dumps(entry.as_dict(), separators=(",", ":")) + "\n")

    @classmethod
    def load(cls, path) -> "Index":
        with Path(path).open() as handle:
            header = json.loads(handle.readline())
            if header.get("format") != "n64dk-simindex":
                raise ValueError(f"{path}: not a similarity index")
            index = cls(header.get("library_titles", LIBRARY_TITLES), header.get("max_posting", MAX_POSTING))
            for line in handle:
                if line.strip():
                    index.add(Entry.from_dict(json.loads(line)))
        return index

    # ---- measuring --------------------------------------------------------------------------
    def idf(self, feature: int) -> float:
        return math.log((len(self.entries) + 1) / (self.df.get(feature, 0) + 1)) + 0.05

    def titles_carrying(self, level: str, value: str) -> int:
        return len(self.carriers.get((level, value), ())) if value else 0

    @staticmethod
    def structural(q: Entry, c: Entry, level: str | None) -> tuple[float, str]:
        """Banded edit-distance ratio over instruction shapes (``similar.similarity``; 0.0 below
        ``STRUCT_FLOOR``).  Long pairs whose shape 4-gram sketches barely overlap are not worth the
        quadratic cost: their sketch Jaccard is reported instead, and the method says so."""
        if level:
            return 1.0, level
        if max(len(q.tokens), len(c.tokens)) > LONG:
            grams = _jaccard(q.feats.get("gram", ()), c.feats.get("gram", ())) or 0.0
            if grams < 0.3:
                return round(grams, 4), "gram-sketch"
        return round(similar.similarity(q.tokens, c.tokens, STRUCT_FLOOR), 4), "edit-distance"

    def compare(self, q: Entry, c: Entry) -> dict:
        """Separate metrics, confidence, class and provenance for candidate ``c`` of query ``q``."""
        if q.exact == c.exact:
            level = "exact"
        elif q.masked == c.masked:
            level = "relocation-only"
        elif q.renamed == c.renamed:
            level = "register-renamed"
        else:
            level = None
        structural, method = self.structural(q, c, level)
        same_semantics = bool(q.semantic) and q.semantic == c.semantic
        semantic = 1.0 if same_semantics else _jaccard(q.feats.get("sem", ()), c.feats.get("sem", ())) \
            if q.lifted and c.lifted else None
        cfg = _jaccard(q.feats.get("cfg", ()) + q.feats.get("loop", ()), c.feats.get("cfg", ()) + c.feats.get("loop", ()))
        memory = _jaccard(q.feats.get("mem", ()), c.feats.get("mem", ()))
        constants = _jaccard(q.feats.get("const", ()), c.feats.get("const", ()))
        strings = _jaccard(q.feats.get("str", ()), c.feats.get("str", ()))
        neighbours = _jaccard(q.feats.get("neigh", ()), c.feats.get("neigh", ())) \
            if q.feats.get("neigh") and c.feats.get("neigh") else None
        mine = {f for kind, values in q.feats.items() if kind != "gram" for f in values}
        theirs = {f for kind, values in c.feats.items() if kind != "gram" for f in values}
        total = sum(self.idf(f) for f in sorted(mine))
        rarity = sum(self.idf(f) for f in sorted(mine & theirs)) / total if total else 0.0
        parts = {"structural": (structural, 0.3), "semantic": (semantic, 0.3), "cfg": (cfg, 0.1),
                 "memory": (memory, 0.1), "constants": (constants, 0.1), "neighbours": (neighbours, 0.1)}
        weight = sum(w for value, w in parts.values() if value is not None)
        score = sum(value * w for value, w in parts.values() if value is not None) / weight if weight else 0.0
        small = min(q.n, c.n) < SMALL
        if level:
            score = 1.0
        if small:
            score *= min(q.n, c.n) / SMALL                  # generic code: explicitly discounted
        confidence = "high" if level and not small else _label(score * (0.6 + 0.4 * rarity))
        if small and _RANK[confidence] > _RANK["low"]:
            confidence = "low"
        carriers = max(self.titles_carrying("masked", c.masked) if level else 0,
                       self.titles_carrying("semantic", c.semantic) if same_semantics else 0)
        library = carriers >= self.library_titles
        if level:
            kind = level
        elif same_semantics or (semantic is not None and semantic >= SEM_HIGH and structural < STRUCT_HIGH):
            kind = "same-source-other-compiler"
        elif structural >= MODIFIED or (semantic or 0.0) >= MODIFIED:
            kind = "modified"
        elif (neighbours or 0.0) >= RELATED or (constants or 0.0) >= RELATED or (strings or 0.0) > 0:
            kind = "related-helper"
        else:
            kind = "unrelated"
        if library and kind != "unrelated":
            kind = "likely-library"
        r = lambda v: None if v is None else round(v, 4)          # noqa: E731
        return {"candidate": c.key, "title": c.title, "image": c.image, "name": c.name, "vram": f"0x{c.vram:08X}",
                "class": kind, "level": level or ("semantic" if same_semantics else None), "confidence": confidence,
                "score": round(score, 4),
                "metrics": {"structural": structural, "structural_method": method, "semantic": r(semantic),
                            "cfg": r(cfg), "memory": r(memory), "constants": r(constants), "strings": r(strings), "neighbours": r(neighbours),
                            "rarity": round(rarity, 4)},
                "small": small, "library": library, "titles_carrying": carriers,
                "provenance": {"query": q.key, "instructions": [q.n, c.n], "lifted": [q.lifted, c.lifted],
                               "shared_features": {kind: len(set(q.feats.get(kind, ())) & set(c.feats.get(kind, ())))
                                                   for kind in KINDS if q.feats.get(kind) and c.feats.get(kind)},
                               "index_entries": len(self.entries)},
                "proof": False}

    def candidates(self, q: Entry, shortlist: int = 24, skip=None) -> list[int]:
        """Entry numbers worth comparing: every level match, then the best by shared rare features."""
        skip = skip or (lambda e: False)
        chosen: list[int] = []
        seen: set[int] = set()
        for level in ("exact", "masked", "renamed", "semantic"):
            value = getattr(q, level)
            for number in self.levels[level].get(value, ()) if value else ():
                if number not in seen and not skip(self.entries[number]):
                    chosen.append(number)
                    seen.add(number)
        weight: Counter = Counter()
        for feature in sorted({f for values in q.feats.values() for f in values}):
            posting = self.postings.get(feature)
            if not posting:
                continue
            value = self.idf(feature)
            for number in posting:
                weight[number] += value
        budget = len(chosen) + shortlist
        for number, _w in sorted(weight.items(), key=lambda kv: (-kv[1], kv[0])):
            if len(chosen) >= budget:
                break
            if number not in seen and not skip(self.entries[number]):
                chosen.append(number)
                seen.add(number)
        return chosen

    def query(self, q: Entry, limit: int = 5, shortlist: int = 24, exclude_titles=(), only_titles=None,
              min_confidence: str = "none", include_unrelated: bool = False) -> list[dict]:
        """The ``limit`` best candidates for ``q`` (never ``q`` itself), best first."""
        excluded, only = set(exclude_titles), set(only_titles) if only_titles else None

        def skip(e: Entry) -> bool:
            return e.key == q.key or e.title in excluded or (only is not None and e.title not in only)

        rows = [self.compare(q, self.entries[n]) for n in self.candidates(q, shortlist, skip)]
        rows = [row for row in rows if _RANK[row["confidence"]] >= _RANK[min_confidence]
                and (include_unrelated or row["class"] != "unrelated")]
        order = {"exact": 0, "relocation-only": 1, "register-renamed": 2}
        rows.sort(key=lambda row: (order.get(row["level"] or "", 3), -row["score"], row["candidate"]))
        return rows[:limit]


def index_image(index: Index, title: str, image: str, data: bytes, delta: int, rows, lift: bool = True,
                min_size: int = 8) -> int:
    """Add every catalogued function ``(name, vram, size)`` of one image, with call neighbourhoods
    keyed by masked hash (comparable across titles) and strings read from the image.  Returns the
    number of entries added.  One image in memory at a time; the code is not kept."""
    def code_of(vram, size):
        offset = vram - delta
        return data[offset:offset + size] if size >= min_size and 0 <= offset and offset + size <= len(data) else b""

    def data_at(address, length):
        offset = address - delta
        return data[offset:offset + length] if 0 <= offset < len(data) else None

    masked_of: dict[int, str] = {}
    callers_of: dict[int, set] = {}
    for _name, vram, size in rows:
        code = code_of(vram, size)
        if code:
            words = _words(code)
            masked_of[vram] = _sha(b"".join(struct.pack(">I", w) for w in lineage.masked_words(
                b"".join(struct.pack(">I", w) for w in words))))
            for i, w in enumerate(words):
                if w >> 26 == 3:
                    target = ((vram + 4 * i + 4) & 0xF0000000) | ((w & 0x3FFFFFF) << 2)
                    callers_of.setdefault(target, set()).add(vram)
    added = 0
    for name, vram, size in rows:
        code = code_of(vram, size)
        if not code:
            continue
        callers = sorted(masked_of[c] for c in callers_of.get(vram, ()) if c in masked_of)
        added += index.add(describe(title, name, vram, code, image=image, data_at=data_at,
                                    call_key=masked_of.get, callers=callers, lift=lift))
    return added
