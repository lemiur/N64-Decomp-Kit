"""Full-cartridge relink: every image a generated project can link, assembled into a whole ROM and
compared with the user's ROM, with each byte's provenance kept apart.

The cartridge is assembled from parts, never copied whole:

- ``linked``: an image linked from the project's units by :func:`n64dk.match.relink.relink` (its
  own report says how many of those bytes came from source and how many are raw stand-ins cut
  from the ROM);
- ``repacked``: a compressed code segment rebuilt as ``encoder(relinked payload)``, counted only
  when the relinked payload is identical to the decoded segment and the encoder's output equals
  the stored stream byte for byte (``codecboundary`` exact repack);
- ``preserved-compressed``: a compressed segment whose stored stream is kept as a declared
  extraction input (the payload differs, or no encoder reproduces the stream);
- a segment the loader copies uncompressed is ``linked`` at its ROM offset, once the image is
  checked to be exactly that ROM slice;
- ``preserved``: every other byte (header, IPL3, assets, fill), a declared extraction input.

Separate metrics, never pooled: ROM hash match (with the preserved inputs it needed), bytes per
provenance class, bytes linked from source (strict C and assembly as the image relinks report
them), compressed payload reconstruction vs exact repack.  Original translation-unit
reconstruction is not measured here.  The assembled ROM is written only with ``keep_rom``
(it holds ROM bytes); nothing is written into the project tree.  Grants no credit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import relink as rl

CLASSES = ("linked", "repacked", "preserved-compressed", "preserved")


def _hex(value) -> int:
    return int(str(value), 16) if isinstance(value, str) else int(value)


def code_segments(init_project: Path) -> list[dict]:
    """The survey's compressed or copied code segments (name, rom, rom_end, codec)."""
    path = Path(init_project) / "survey" / "layout.json"
    try:
        layout = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    return [{"name": item["name"], "rom": _hex(item["rom"]), "rom_end": _hex(item["rom_end"]),
             "codec": item.get("codec")} for item in layout.get("code_segments") or [] if item.get("name")]


def assemble(rom: bytes, parts: list[dict]) -> tuple[bytes, list[str]]:
    """The cartridge from ``parts`` (``{"rom", "data", "cls", "owner"}``, disjoint) and the user's ROM
    for every byte no part covers (``preserved``).  Returns the bytes and each byte's class index."""
    out = bytearray(rom)                      # every byte starts as a preserved extraction input ...
    owner = ["preserved"] * len(rom)
    taken = bytearray(len(rom))
    for part in sorted(parts, key=lambda p: p["rom"]):
        start, data = part["rom"], part["data"]
        if start < 0 or start + len(data) > len(rom):
            raise rl.RelinkError(f"{part['owner']}: 0x{start:X}+0x{len(data):X} lies outside the ROM")
        if any(taken[start:start + len(data)]):
            raise rl.RelinkError(f"{part['owner']}: overlaps another part at ROM 0x{start:X}")
        out[start:start + len(data)] = data   # ... unless a part built from the project replaces it
        taken[start:start + len(data)] = b"\1" * len(data)
        owner[start:start + len(data)] = [part["owner"]] * len(data)
    return bytes(out), owner


def relink_cartridge(project_tree, init_project, build, *, rom: bytes, images: dict, linker: str = "kit",
                     assumed: str = "raw", cfg=None, rom_sha1: str | None = None, any_rom: bool = False,
                     keep_rom: bool = False, relink_fn=rl.relink, encoders=None) -> dict:
    """``images``: image name -> ``(data, delta, is_rom)`` from the project's image registry."""
    from ..assets import codecboundary as cb
    build = Path(build)
    build.mkdir(parents=True, exist_ok=True)
    segments = {item["name"]: item for item in code_segments(init_project)}
    parts, per_image, linked_source = [], {}, {"text": 0, "data": 0}
    for name, (data, delta, is_rom) in sorted(images.items()):
        report = relink_fn(project_tree, name, build / name, data=data, delta=delta, is_rom=is_rom, cfg=cfg,
                           assumed=assumed, linker=linker, rom_sha1=rom_sha1, any_rom=any_rom, keep_image=True)
        image_bin = build / name / "relinked.bin"
        linked = image_bin.read_bytes() if image_bin.is_file() else None
        row = {"outcome": report.get("outcome"), "differing_bytes": report.get("differing_bytes"),
               "coverage": report.get("coverage"), "source_text_bytes": report.get("source_text_bytes", 0),
               "source_data_bytes": report.get("source_data_bytes", 0)}
        per_image[name] = row
        if linked is None:
            row["placed"] = "not placed: the relink kept no image"
            continue
        if is_rom:
            start = _hex(report["extent"]["rom_offset"])
            parts.append({"rom": start, "data": linked, "cls": "linked", "owner": f"linked:{name}"})
            row["placed"] = f"ROM 0x{start:X}"
            linked_source["text"] += row["source_text_bytes"]
            linked_source["data"] += row["source_data_bytes"]
            continue
        segment = segments.get(name)
        if segment is None:
            row["placed"] = "not placed: the survey records no ROM range for this image"
            continue
        stored = rom[segment["rom"]:segment["rom_end"]]
        if not segment.get("codec"):
            # a segment the loader copies uncompressed: its image is a slice of the ROM, so the
            # relinked bytes go back at the segment's ROM offset (checked: the image is that slice)
            if len(data) > len(stored) or stored[:len(data)] != data or len(linked) != len(data):
                row["placed"] = "not placed: the image is not the ROM slice the survey records"
                continue
            parts.append({"rom": segment["rom"], "data": linked, "cls": "linked", "owner": f"linked:{name}"})
            row["placed"] = f"ROM 0x{segment['rom']:X} (copied, uncompressed)"
            linked_source["text"] += row["source_text_bytes"]
            linked_source["data"] += row["source_data_bytes"]
            continue
        found = cb.classify_slice(stored, segment["codec"], rom=segment["rom"], expected=data, reference="declared",
                                  **({"encoders": encoders} if encoders is not None else {}))
        row["codec"] = {"codec": segment["codec"], "class": found.cls, "reason": found.reason,
                        "payload_reconstructed": found.payload_reconstructed, "stored_size": len(stored)}
        payload_identical = report.get("outcome") == "identical" and linked == data
        repack = None
        if payload_identical and found.cls == cb.EXACT and found.encoder:
            # rebuild the stream from the relinked payload (not from the ROM) with the encoder that matched
            for label, encode in (encoders if encoders is not None else cb.default_encoders(segment["codec"])):
                if label == found.encoder:
                    candidate = encode(linked)
                    repack = candidate if candidate == stored else None
                    break
        if repack is not None:
            parts.append({"rom": segment["rom"], "data": repack, "cls": "repacked", "owner": f"repacked:{name}"})
            row["placed"] = f"repacked at ROM 0x{segment['rom']:X} ({found.encoder})"
            linked_source["text"] += row["source_text_bytes"]
            linked_source["data"] += row["source_data_bytes"]
        else:
            parts.append({"rom": segment["rom"], "data": stored, "cls": "preserved-compressed",
                          "owner": f"preserved-compressed:{name}"})
            row["placed"] = ("preserved: " + ("the relinked payload differs from the decoded segment"
                                              if not payload_identical else f"no exact repack ({found.reason})"))
    cartridge, owner = assemble(rom, parts)
    by_class = {cls: 0 for cls in CLASSES}
    for item in owner:
        by_class[item.split(":")[0]] += 1
    first = next((i for i in range(len(rom)) if cartridge[i] != rom[i]), None)
    differing = sum(1 for a, b in zip(cartridge, rom) if a != b)
    report = {
        "schema": "n64dk-cartridge-relink/1",
        "credit": "a measurement: grants no credit and changes no run, receipt or evidence store",
        "rom_bytes": len(rom),
        "rom_sha1_match": hashlib.sha1(cartridge).hexdigest() == hashlib.sha1(rom).hexdigest(),
        "differing_bytes": differing,
        "first_difference": None if first is None else {"rom_offset": f"0x{first:X}", "owner": owner[first]},
        "bytes_by_provenance": by_class,
        "preserved_inputs_needed": by_class["preserved"] + by_class["preserved-compressed"],
        "linked_from_source_bytes": linked_source,
        "compressed_payload_reconstructed": sum((row.get("codec") or {}).get("payload_reconstructed", 0)
                                                for row in per_image.values()),
        "compressed_exactly_repacked_bytes": by_class["repacked"],
        "translation_units_reconstructed": "not measured (original unit boundaries are not part of this check)",
        "images": per_image,
    }
    if keep_rom:
        (build / "cartridge.z64").write_bytes(cartridge)
        report["rom_bytes_kept_in"] = ["cartridge.z64 (holds ROM bytes; never package it)"]
    (build / "CARTRIDGE.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return report
