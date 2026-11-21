"""Name, symbol and semantic-role recovery with separate fields (round 12, item L).

A function's names are kept apart by what they are, never merged into one "name":

``canonical``  ``func_XXXXXXXX``: the address name, always present, the only name strict
               matching, caches and receipts use.
``original``   a name the title's own development records carry (debug records, SN/PsyQ objects,
               ECOFF/STABS): the developers' name, with the record's provenance.
``sdk``        a library name from an SDK signature (libultra, libc): original in the library's
               sources, attributed to this address by a signature match.
``donor``      a name proposed by an exact source donor or by lineage from another title: the other
               program's name for the same code.
``semantic``   a generated description of a role (MMIO subsystem, referenced message string,
               callgraph role), always prefixed ``sem_``: never presented as an original name.
``user``       a name the operator supplied.

``display`` picks one for people to read (user, original, sdk, donor, semantic, canonical, in that
order) and states which field it came from.  The table is written beside symbols, never into the
symbol maps a verification reads: renaming cannot change what a strict comparison or a cache key
sees (``canonical`` is the identity).
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field

FIELDS = ("user", "original", "sdk", "donor", "semantic", "canonical")
# KSEG1 hardware register blocks (upper halves of 0xA4xx_xxxx / 0xBFC0_xxxx)
MMIO = {0xA404: "sp", 0xA408: "sp", 0xA410: "dp", 0xA420: "dps", 0xA430: "mi", 0xA440: "vi", 0xA450: "ai",
        0xA460: "pi", 0xA470: "ri", 0xA480: "si", 0xBFC0: "pif"}
IDENT = re.compile(rb"^([A-Za-z_][A-Za-z0-9_]{3,40})(?:\(\))?\s*[:(]")


@dataclass
class Names:
    vram: int
    canonical: str
    original: dict | None = None
    sdk: dict | None = None
    donor: dict | None = None
    semantic: dict | None = None
    user: dict | None = None
    roles: list[str] = field(default_factory=list)

    def display(self) -> tuple[str, str]:
        for name in FIELDS[:-1]:
            value = getattr(self, name)
            if value:
                return value["name"], name
        return self.canonical, "canonical"

    def as_row(self) -> dict:
        shown, source = self.display()
        row = {"vram": f"0x{self.vram:08X}", "canonical": self.canonical, "display": shown, "display_from": source,
               "roles": self.roles}
        for name in FIELDS[:-1]:
            row[name] = getattr(self, name)
        return row


def canonical(vram: int) -> str:
    return f"func_{vram:08X}"


def mmio_roles(code: bytes) -> list[str]:
    """Hardware blocks a function addresses through ``lui`` of a KSEG1 register page."""
    roles = set()
    for i in range(0, len(code) - 3, 4):
        word = struct.unpack_from(">I", code, i)[0]
        if word >> 26 == 0x0F:
            role = MMIO.get(word & 0xFFFF)
            if role:
                roles.add(role)
    return sorted(roles)


def referenced_strings(code: bytes, read_string) -> list[str]:
    """Strings a function addresses with a ``lui``/``addiu`` pair; ``read_string(address)`` returns
    the NUL-terminated printable text there, or None."""
    out, his = [], {}
    for i in range(0, len(code) - 3, 4):
        word = struct.unpack_from(">I", code, i)[0]
        op, rs, rt, imm = word >> 26, (word >> 21) & 31, (word >> 16) & 31, word & 0xFFFF
        if op == 0x0F:
            his[rt] = imm << 16
        elif op == 0x09 and rs in his:
            address = (his[rs] + (imm - 0x10000 if imm & 0x8000 else imm)) & 0xFFFFFFFF
            text = read_string(address)
            if text:
                out.append(text)
    return out


def semantic_name(vram: int, roles: list[str], strings: list[str]) -> dict | None:
    """A generated role name: a message string that starts with an identifier (``"osPiStartDma:
    bad length"``) names the function that prints it, provisionally; otherwise one dominant
    hardware block.  Always ``sem_``-prefixed."""
    for text in strings:
        m = IDENT.match(text.encode("latin-1", "replace"))
        if m:
            return {"name": f"sem_{m.group(1).decode()}", "source": "message string", "evidence": text[:80],
                    "confidence": "low"}
    if len(roles) == 1:
        return {"name": f"sem_{roles[0]}_{vram:08X}", "source": "mmio", "evidence": roles[0], "confidence": "low"}
    return None


def build(rows, image: bytes | None = None, delta: int = 0, sdk: dict[int, str] | None = None,
          original: dict[int, tuple[str, str]] | None = None, donor: dict[int, tuple[str, str]] | None = None,
          user: dict[int, str] | None = None, read_string=None) -> list[Names]:
    """The name table for catalogue ``rows`` ``(name, vram, size)``.  ``sdk``: address -> library
    name; ``original``/``donor``: address -> (name, provenance); ``user``: address -> name;
    ``image``/``delta``/``read_string``: the code for semantic roles."""
    out = []
    for _name, vram, size in rows:
        entry = Names(vram, canonical(vram))
        if user and vram in user:
            entry.user = {"name": user[vram], "source": "user"}
        if original and vram in original:
            entry.original = {"name": original[vram][0], "source": original[vram][1], "confidence": "high"}
        if sdk and vram in sdk:
            entry.sdk = {"name": sdk[vram], "source": "sdk signature", "confidence": "high"}
        if donor and vram in donor:
            entry.donor = {"name": donor[vram][0], "source": donor[vram][1], "confidence": "medium"}
        if image is not None:
            offset = vram - delta
            if 0 <= offset and offset + size <= len(image):
                code = image[offset:offset + size]
                entry.roles = mmio_roles(code)
                strings = referenced_strings(code, read_string) if read_string else []
                entry.semantic = semantic_name(vram, entry.roles, strings)
        out.append(entry)
    return out


def string_reader(image: bytes, delta: int, minimum: int = 4):
    """``read_string`` over one image: printable NUL-terminated text of at least ``minimum`` chars."""
    def read(address: int) -> str | None:
        offset = address - delta
        if not 0 <= offset < len(image):
            return None
        end = image.find(b"\x00", offset, offset + 200)
        if end < 0 or end - offset < minimum:
            return None
        raw = image[offset:end]
        if all(32 <= b < 127 or b in (9, 10) for b in raw):
            return raw.decode()
        return None
    return read


def write_tsv(table: list[Names]) -> str:
    """NAMES.tsv: one row per function, every field in its own column."""
    head = "vram\tcanonical\tdisplay\tdisplay_from\toriginal\tsdk\tdonor\tsemantic\tuser\troles"
    lines = [head]
    for n in table:
        r = n.as_row()

        def cell(value):
            return "" if not value else f"{value['name']} [{value['source']}]"
        lines.append("\t".join([r["vram"], r["canonical"], r["display"], r["display_from"], cell(r["original"]),
                                cell(r["sdk"]), cell(r["donor"]), cell(r["semantic"]), cell(r["user"]),
                                ",".join(r["roles"])]))
    return "\n".join(lines) + "\n"
