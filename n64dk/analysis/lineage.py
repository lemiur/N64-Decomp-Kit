"""Cross-revision, cross-region and cross-title lineage from function hashes (round 12, item H).

Two hashes per catalogued function:

``exact``   SHA-1 of the function's bytes.
``masked``  the same with every field a linker would patch blanked: ``jal``/``j`` targets, ``lui``
            immediates, and the 16-bit immediate of an I-type instruction whose base register a
            ``lui`` in the same function set (the ``%lo`` half).  Branch offsets, stack offsets and
            ordinary constants stay.  Two functions with equal masked hashes are the same machine
            code linked at different addresses: the same source and recipe, or very nearly.

A shared hash is evidence of shared code, never of a match: lineage proposes donors, names,
prototypes and recipes for a target, which are then rebuilt and verified against the target's own
bytes.  ``graph`` classifies each shared hash by how many titles carry it: hashes in many titles
are libraries (libultra, libc, the compiler's runtime), hashes in a few are engine or studio
lineage, and the report keeps the two apart so that a gain from a close revision is visible as
such.
"""
from __future__ import annotations

import hashlib
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass

LUI, J, JAL = 0x0F, 0x02, 0x03
# I-type opcodes whose immediate is a %lo half when the base register came from a lui
LO_USERS = {0x09, 0x0D, 0x08, 0x20, 0x21, 0x23, 0x24, 0x25, 0x27, 0x28, 0x29, 0x2B, 0x31, 0x35, 0x37,
            0x39, 0x3D, 0x3F, 0x22, 0x26, 0x2A, 0x2E}
LIBRARY_TITLES = 5          # a masked hash in this many titles or more is a library, not lineage
MIN_BYTES = 32              # shorter functions (stubs, returns) say nothing about lineage


@dataclass(frozen=True)
class FunctionHash:
    title: str
    name: str
    vram: int
    size: int
    exact: str
    masked: str


def masked_words(code: bytes) -> list[int]:
    """The function's words with relocated fields blanked (see the module docstring)."""
    out, from_lui = [], set()
    for i in range(0, len(code) - len(code) % 4, 4):
        word = struct.unpack_from(">I", code, i)[0]
        op, rs, rt = word >> 26, (word >> 21) & 31, (word >> 16) & 31
        if op in (J, JAL):
            word &= 0xFC000000
        elif op == LUI:
            word &= 0xFFFF0000
            from_lui.add(rt)
        elif op in LO_USERS and rs in from_lui:
            word &= 0xFFFF0000
            if op in (0x09, 0x0D) and rt != rs:      # addiu/ori into another register: still an address
                from_lui.add(rt)
        else:
            dest = _written(word)
            if dest is not None:
                from_lui.discard(dest)
        out.append(word)
    return out


def _written(word: int) -> int | None:
    """The general register an instruction writes (approximate: R-type rd, I-type rt for ALU and
    loads), None when none or unknown."""
    op = word >> 26
    if op == 0:
        return (word >> 11) & 31 or None
    if op in (0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x20, 0x21, 0x23, 0x24, 0x25, 0x27, 0x18, 0x19,
              0x37, 0x1A, 0x1B, 0x22, 0x26):
        return (word >> 16) & 31 or None
    return None


def hash_function(title: str, name: str, vram: int, code: bytes) -> FunctionHash:
    masked = b"".join(struct.pack(">I", w) for w in masked_words(code))
    return FunctionHash(title, name, vram, len(code), hashlib.sha1(code).hexdigest(),
                        hashlib.sha1(masked).hexdigest())


def hashes(title: str, image: bytes, delta: int, rows) -> list[FunctionHash]:
    """Hashes of every catalogued function ``(name, vram, size)`` that lies inside ``image``."""
    out = []
    for name, vram, size in rows:
        offset = vram - delta
        if size >= 4 and 0 <= offset and offset + size <= len(image):
            out.append(hash_function(title, name, vram, image[offset:offset + size]))
    return out


def graph(all_hashes: list[FunctionHash], library_titles: int = LIBRARY_TITLES,
          min_bytes: int = MIN_BYTES) -> dict:
    """Title-to-title lineage edges from shared masked hashes, libraries kept apart.

    Returns ``{"titles": {title: {...}}, "edges": [{a, b, shared, shared_bytes, exact_shared,
    class}], "library_hashes": n}``; ``class`` is ``lineage`` (hashes carried by fewer than
    ``library_titles`` titles) or ``library``.  A function is counted once per title pair even when
    a title carries the same code twice."""
    by_hash: dict[str, dict[str, FunctionHash]] = defaultdict(dict)
    exact_titles: dict[str, set[str]] = defaultdict(set)
    totals: Counter = Counter()
    for h in all_hashes:
        totals[h.title] += 1
        if h.size >= min_bytes:
            by_hash[h.masked].setdefault(h.title, h)
            exact_titles[h.exact].add(h.title)
    pairs: dict[tuple[str, str, str], list] = defaultdict(lambda: [0, 0, 0])
    library = 0
    for masked, holders in by_hash.items():
        if len(holders) < 2:
            continue
        kind = "library" if len(holders) >= library_titles else "lineage"
        library += kind == "library"
        titles = sorted(holders)
        for i, a in enumerate(titles):
            for b in titles[i + 1:]:
                entry = pairs[(a, b, kind)]
                entry[0] += 1
                entry[1] += holders[a].size
                entry[2] += holders[a].exact == holders[b].exact
    edges = [{"a": a, "b": b, "class": kind, "shared": n, "shared_bytes": nbytes, "exact_shared": exact}
             for (a, b, kind), (n, nbytes, exact) in pairs.items()]
    edges.sort(key=lambda e: (e["class"] != "lineage", -e["shared"], e["a"], e["b"]))
    titles = {}
    for title, count in sorted(totals.items()):
        shared = {e["b"] if e["a"] == title else e["a"]: e["shared"] for e in edges
                  if title in (e["a"], e["b"]) and e["class"] == "lineage"}
        titles[title] = {"functions": count, "lineage_partners": dict(sorted(shared.items(), key=lambda kv: -kv[1]))}
    return {"titles": titles, "edges": edges, "library_hashes": library,
            "parameters": {"library_titles": library_titles, "min_bytes": min_bytes}}


def proposals(target: str, all_hashes: list[FunctionHash], library_titles: int = LIBRARY_TITLES,
              min_bytes: int = MIN_BYTES) -> list[dict]:
    """For each function of ``target``, the other titles' functions with the same masked hash: the
    candidates a lineage transfer may propose (names, sources, recipes), with the relationship
    class.  Proposals only: each must be rebuilt and verified against the target."""
    by_hash: dict[str, list[FunctionHash]] = defaultdict(list)
    for h in all_hashes:
        if h.size >= min_bytes:
            by_hash[h.masked].append(h)
    out = []
    for h in all_hashes:
        if h.title != target or h.size < min_bytes:
            continue
        others = [o for o in by_hash[h.masked] if o.title != target]
        if not others:
            continue
        carriers = {o.title for o in by_hash[h.masked]}
        out.append({"function": h.name, "vram": h.vram, "size": h.size,
                    "class": "library" if len(carriers) >= library_titles else "lineage",
                    "same_bytes": sorted({o.title for o in others if o.exact == h.exact}),
                    "sources": [{"title": o.title, "name": o.name, "vram": o.vram} for o in others][:8]})
    return out
