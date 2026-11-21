"""What kind of unit each catalogued function is, so reports never mix them (round 12).

=================  ==========================================================================
``boot``           the program's entry routine (the header entry point) and its handoff code
``rsp``            RSP microcode text inside a CPU image (the survey's ``rsp_in_text`` ranges):
                   not CPU code, never in a C denominator
``handwritten``    the catalogue's ``asm`` kind (no C frame; register use no compiler makes)
``sdk``            named by the SDK fingerprint (``sdk_symbols.txt``): library code
``game``           everything else
=================  ==========================================================================

The classes come from the project's own survey (its entry, catalogue comments, RSP ranges and SDK
symbol file); nothing here reads the ROM.  ``boot`` wins over ``handwritten`` (an entry routine is
both), ``rsp`` over everything.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

CLASSES = ("game", "sdk", "handwritten", "boot", "rsp")
_ROW = re.compile(r"vram\s*=\s*(0x[0-9A-Fa-f]+).*#\s*(\w+)")


def _ranges(values) -> list[tuple[int, int]]:
    out = []
    for text in values or ():
        lo, _, hi = str(text).partition("..")
        try:
            out.append((int(lo, 16), int(hi, 16)))
        except ValueError:
            continue
    return out


def load(project: Path) -> dict:
    """The survey evidence: entry, RSP ranges, catalogue kinds, SDK names (empty when absent)."""
    survey = Path(project) / "survey"
    evidence = {"entry": None, "rsp": [], "kinds": {}, "sdk": set()}
    layout = survey / "layout.json"
    if layout.is_file():
        try:
            evidence["entry"] = int(json.loads(layout.read_text()).get("entry") or "0", 16) or None
        except (ValueError, TypeError):
            pass
    report = survey / "report.json"
    if report.is_file():
        try:
            evidence["rsp"] = _ranges(json.loads(report.read_text()).get("rsp_in_text"))
        except ValueError:
            pass
    for catalogue in survey.glob("functions*.toml"):
        for line in catalogue.read_text(errors="replace").splitlines():
            match = _ROW.search(line)
            if match:
                evidence["kinds"][int(match.group(1), 16)] = match.group(2)
    sdk = survey / "sdk_symbols.txt"
    if sdk.is_file():
        for line in sdk.read_text(errors="replace").splitlines():
            name, _, rest = line.partition("=")
            if rest.strip().startswith("0x"):
                try:
                    evidence["sdk"].add(int(rest.strip().split(";")[0], 16))
                except ValueError:
                    continue
    return evidence


def classify(vram: int, size: int, evidence: dict) -> str:
    end = vram + max(size, 4)
    if any(lo < end and vram < hi for lo, hi in evidence["rsp"]):
        return "rsp"
    if evidence["entry"] is not None and vram == evidence["entry"]:
        return "boot"
    if evidence["kinds"].get(vram) == "asm":
        return "handwritten"
    if vram in evidence["sdk"]:
        return "sdk"
    return "game"
