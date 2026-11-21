"""Logical image names and address resolution.

A project declares its images in ``n64dk.toml``::

    [images.resident]
    path = "build/main.z64"            # linear image or ROM
    vram_delta = 0x7FFFF400            # vram - file offset; omit for a ROM with a boot segment
    catalogue = "config/functions.toml"
    catalogue_shift = 0                # catalogue address - link address
    manifests = ["config/matching_units.tsv"]
    raw_segments = [[0x1000, 0x1000, 0x100000, "copy"]]   # rom offset, image offset, size, transform
    aliases = ["main"]
    extent = [0x80000400, 0x80064A00]  # optional: the VRAM range this image owns (default: catalogue span)

``resolve`` fails closed: an unknown name is never treated as a file name, an alias shared by
several images is an error.  ``explain`` reports every coordinate system for one address and
refuses to pick an image when several map the same VRAM and none was named.
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


class ImageError(LookupError):
    pass


def hx(value: int) -> str:
    return f"0x{value:08X}"


@dataclass
class ImageSpec:
    name: str
    path: Path
    vram_delta: int | None = None
    catalogue: Path | None = None
    catalogue_shift: int = 0
    manifests: list[Path] = field(default_factory=list)
    raw_segments: list[tuple[int, int, int, str]] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    extent: tuple[int, int] | None = None
    delta_source: str = "declared"
    _data: bytes | None = None
    _rows: list | None = None

    def load(self) -> bytes:
        if self._data is None:
            self._data = Path(self.path).read_bytes()
            if self.vram_delta is None:
                from ..rom import Rom
                from .engine import Target
                target = Target.from_rom(Rom.load(self.path))
                self._data, self.vram_delta, self.delta_source = target.rom, target.delta, "boot segment of the ROM"
        return self._data

    @property
    def delta(self) -> int:
        if self.vram_delta is None:
            self.load()
        return self.vram_delta

    def rows(self) -> list[tuple[str, int, int]]:
        """Catalogue rows in link (image) space."""
        if self._rows is None:
            from ..analysis.boundaries import read_catalogue
            self._rows = [(n, v - self.catalogue_shift, s) for n, v, s in read_catalogue(self.catalogue)] \
                if self.catalogue else []
        return self._rows

    def span(self) -> tuple[int, int]:
        if self.extent:
            return self.extent
        rows = self.rows()
        if rows:
            return rows[0][1], max(v + s for _n, v, s in rows)
        size = Path(self.path).stat().st_size
        return self.delta, self.delta + size

    def rom_of(self, offset: int):
        for rom, image_off, size, transform in self.raw_segments:
            if image_off <= offset < image_off + size:
                if transform == "copy":
                    return rom + offset - image_off, f"segment rom {hx(rom)} -> image {hx(image_off)} (copy)"
                return None, f"not representable: compressed segment ({transform}) rom {hx(rom)}"
        if not self.raw_segments and self.delta_source != "declared":
            return offset, "the image is the ROM"
        return None, "not representable: no raw segment covers this offset"


def _int(value) -> int:
    return value if isinstance(value, int) else int(str(value), 0)


def _catalogue_delta(path: Path, shift: int) -> int | None:
    """Infer a linear image delta from an explicitly mapped catalogue section.

    Small extracted images used by projects and tests are not complete N64 ROMs, so boot-header
    discovery is unavailable.  A section carrying both ``vram`` and ``rom`` is nevertheless an
    independent, exact mapping.  Refuse to infer when mapped sections disagree.
    """
    if not path.is_file():
        return None
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return None
    deltas = set()
    sections = data.get("section", []) if isinstance(data, dict) else []
    for section in sections if isinstance(sections, list) else []:
        if isinstance(section, dict) and "vram" in section and "rom" in section:
            deltas.add(_int(section["vram"]) - shift - _int(section["rom"]))
    return next(iter(deltas)) if len(deltas) == 1 else None


def load_registry(project) -> dict[str, ImageSpec]:
    project = Path(project)
    config = project / "n64dk.toml" if project.is_dir() else project
    root = config.parent
    data = tomllib.loads(config.read_text()) if config.exists() else {}
    out = {}
    for name, table in (data.get("images") or {}).items():
        if isinstance(table, str):                         # short form: name = "path"
            table = {"path": table}
        # one schema for every consumer: `path` (or its synonym `rom`) names the image file
        image_path = table.get("composite") or table.get("path") or table.get("raw") or table.get("rom")
        if not image_path:
            raise ImageError(f"[images.{name}] has no path")
        manifests = []
        for pattern in table.get("manifests", []):
            hits = sorted(root.glob(pattern)) if any(c in str(pattern) for c in "*?[") else [root / pattern]
            manifests += hits
        catalogue = root / table["catalogue"] if table.get("catalogue") else None
        shift = _int(table.get("catalogue_shift", 0))
        declared_delta = _int(table["vram_delta"]) if "vram_delta" in table else None
        mapped_delta = _catalogue_delta(catalogue, shift) if catalogue and declared_delta is None else None
        out[name] = ImageSpec(
            name, root / image_path,
            declared_delta if declared_delta is not None else mapped_delta,
            catalogue,
            shift,
            manifests,
            [(_int(a), _int(b), _int(c), str(d)) for a, b, c, d in table.get("raw_segments", [])],
            list(table.get("aliases", [])),
            tuple(_int(x) for x in table["extent"]) if table.get("extent") else None,
            "declared" if declared_delta is not None else
            ("catalogue section map" if mapped_delta is not None else "declared"))
    return out


def resolve(project, name: str) -> ImageSpec:
    registry = load_registry(project)
    if name in registry:
        spec = registry[name]
        if not spec.path.is_file():
            raise ImageError(f"configured image is missing: {spec.path.resolve()}")
        return spec
    normalized = name.replace("_", "").replace("-", "").lower()
    hits = [spec for key, spec in registry.items()
            if name in spec.aliases or key.replace("_", "").replace("-", "").lower() == normalized]
    if len(hits) == 1:
        if not hits[0].path.is_file():
            raise ImageError(f"configured image is missing: {hits[0].path.resolve()}")
        return hits[0]
    if len(hits) > 1:
        raise ImageError(f"image alias {name!r} is ambiguous: " + ", ".join(sorted(s.name for s in hits)))
    known = sorted([*registry, *(f"{a} (alias of {s.name})" for s in registry.values() for a in s.aliases)])
    raise ImageError(f"unknown image {name!r}; known images: " + (", ".join(known) or "none declared in n64dk.toml"))


def function_id(image: str, vram: int, size: int) -> str:
    return f"{image}:{hx(vram)}:0x{size:X}"


_ID = re.compile(r"^([^:\s]+):(0x[0-9A-Fa-f]+):(0x[0-9A-Fa-f]+)$")


def parse_function_id(text: str) -> tuple[str, int, int]:
    m = _ID.match(text.strip())
    if not m:
        raise ValueError(f"not a function id IMAGE:VRAM:SIZE: {text!r}")
    return m.group(1), int(m.group(2), 16), int(m.group(3), 16)


def _manifest_rows(spec: ImageSpec):
    from .byteledger import read_manifest
    rows = []
    for path in spec.manifests:
        rows += read_manifest(path, spec.delta, "c")
    return rows


def _in_image(spec: ImageSpec, link: int) -> dict:
    """What this image says about one link-space address."""
    offset = link - spec.delta
    size = Path(spec.path).stat().st_size if spec._data is None else len(spec._data)
    rom, rom_why = spec.rom_of(offset) if 0 <= offset < size else (None, "outside the image file")
    rows = [r for r in spec.rows() if r[1] <= link < r[1] + r[2]]
    out = {"image": spec.name, "link_vram": hx(link), "image_offset": hx(offset) if 0 <= offset < size else None,
           "catalogue_vram": hx(link + spec.catalogue_shift) if spec.catalogue else None,
           "raw_rom": hx(rom) if rom is not None else rom_why,
           "catalogue_rows": [{"name": n, "link_vram": hx(v), "catalogue_vram": hx(v + spec.catalogue_shift),
                               "size": f"0x{s:X}", "id": function_id(spec.name, v, s)} for n, v, s in rows],
           "manifest_rows": [{"source": m.source, "vram": hx(m.vram), "size": f"0x{m.size:X}", "origin": m.origin}
                             for m in _manifest_rows(spec) if m.vram <= link < m.end]}
    return out


def explain(project, image: str | None = None, vram: int | None = None, offset: int | None = None,
            rom: int | None = None, space: str = "auto") -> dict:
    """Every coordinate of one address.  ``space`` says what ``vram`` is: ``link``, ``catalogue``
    or ``auto`` (prefer the catalogue reading when a catalogue row starts exactly there, then a
    manifest row starting at the link reading, then whichever reading lands in a catalogue row)."""
    registry = load_registry(project)
    if sum(x is not None for x in (vram, offset, rom)) != 1:
        raise ValueError("give exactly one of vram, offset, rom")
    if image is None:
        if vram is None:
            raise ImageError("an image offset or ROM offset needs --image")
        hits = []
        for spec in registry.values():
            for label, link in (("link", vram), ("catalogue", vram - spec.catalogue_shift)):
                if label == "catalogue" and not spec.catalogue_shift:
                    continue
                lo, hi = spec.span()
                if lo <= link < hi:
                    hits.append({**_in_image(spec, link), "reading": label})
        if len(hits) == 1:
            return {"query": hx(vram), "chosen": hits[0]["image"], "why": "only one image maps this VRAM",
                    "images": hits}
        return {"query": hx(vram), "chosen": None,
                "why": (f"{len(hits)} images map this VRAM; name one with --image (refusing to pick)" if hits
                        else "no declared image maps this VRAM"),
                "images": hits}
    spec = resolve(project, image)
    if offset is not None:
        link, why = offset + spec.delta, "image offset + vram_delta"
    elif rom is not None:
        link = None
        for r, image_off, size, transform in spec.raw_segments:
            if r <= rom < r + size:
                if transform != "copy":
                    raise ImageError(f"ROM {hx(rom)} is inside a compressed segment ({transform}): no linear mapping")
                link = rom - r + image_off + spec.delta
        if link is None and not spec.raw_segments and spec.delta_source != "declared":
            link = rom + spec.delta
        if link is None:
            raise ImageError(f"ROM {hx(rom)} is not covered by any raw segment of {spec.name}")
        why = "raw segment mapping"
    else:
        as_link, as_cat = vram, vram - spec.catalogue_shift
        rows = spec.rows()
        if space == "link" or not spec.catalogue_shift:
            link, why = as_link, "VRAM taken as a link address" + ("" if spec.catalogue_shift else " (no catalogue shift)")
        elif space == "catalogue":
            link, why = as_cat, f"VRAM taken as a catalogue address (catalogue_shift {spec.catalogue_shift:#x})"
        elif any(v == as_cat for _n, v, _s in rows):
            link, why = as_cat, (f"a catalogue row starts at {hx(vram)} in catalogue space; link address = "
                                 f"{hx(vram)} - {spec.catalogue_shift:#x}")
        elif any(m.vram == as_link for m in _manifest_rows(spec)):
            link, why = as_link, "a manifest row starts at this link address"
        elif any(v <= as_cat < v + s for _n, v, s in rows):
            link, why = as_cat, "the catalogue reading falls inside a catalogue row"
        else:
            link, why = as_link, "VRAM taken as a link address (no catalogue row at the catalogue reading)"
    result = _in_image(spec, link)
    result.update({"query": hx(vram) if vram is not None else (hx(offset) if offset is not None else hx(rom)),
                   "chosen": spec.name, "why": why, "vram_delta": hx(spec.delta),
                   "vram_delta_source": spec.delta_source, "catalogue_shift": spec.catalogue_shift})
    others = []
    for other in registry.values():
        if other.name == spec.name:
            continue
        lo, hi = other.span()
        if lo <= link < hi:
            others.append(_in_image(other, link))
    result["other_images_mapping_this_vram"] = others
    return result


RUN_CATALOGUES = "discovery/catalogues.json"


def note_run_catalogue(run: Path, image: str, catalogue: Path) -> None:
    """Record that a run resolved ``image`` against ``catalogue`` (repaired or corrected, inside the run
    directory) instead of the project's."""
    import json
    import os
    path = Path(run) / RUN_CATALOGUES
    held = json.loads(path.read_text()) if path.is_file() else {}
    held[image] = os.path.relpath(Path(catalogue).resolve(), Path(run).resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(held, indent=1, sort_keys=True) + "\n")


def with_run_catalogue(spec: "ImageSpec", run: Path) -> "ImageSpec":
    """``spec`` with the catalogue a run recorded for it (``note_run_catalogue``); ``spec`` itself when the
    run used the project's catalogue or the recorded file is gone or lies outside the run directory."""
    import dataclasses
    import json
    path = Path(run) / RUN_CATALOGUES
    try:
        relative = json.loads(path.read_text()).get(spec.name)
    except (OSError, ValueError):
        return spec
    if not relative:
        return spec
    found = (Path(run) / relative).resolve()
    if not found.is_file() or Path(run).resolve() not in found.parents:
        return spec
    return dataclasses.replace(spec, catalogue=found, _rows=None)
