"""``n64dk relink-image``: link one image of a decompile-project tree with every piece pinned, and compare.

See :mod:`n64dk.match.relink`.  Opt-in and read-only towards the generated project, the run and
the ``decompile-init`` project: ROM bytes (the raw blobs, the linked image) are written only under
``--build``.  A measurement: it grants no credit.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from ..cli import command


def _extent(text: str | None):
    if not text:
        return None
    first, _sep, second = text.partition(":")
    return int(first, 16), int(second, 16)


def _refuse(reason) -> int:
    sys.stderr.write(f"relink-image refused: {reason}\n")
    return 2


@command("relink-image", "Link every unit of one image of a decompile-project tree at its own address, raw blobs "
                         "cut from your ROM for the rest, and compare the image byte for byte (opt-in; ROM bytes "
                         "only under --build; no credit)")
def relink_image(args):
    from ..match import images as imagemod
    from ..match import projectgen as pg
    from ..match import relink as rl
    from ..match.toolchain import load_config
    try:
        registry = imagemod.load_registry(args.project)
    except (OSError, ValueError, LookupError) as error:
        return _refuse(f"{args.project}: {error}")
    spec = registry.get(args.image)
    if spec is None:
        return _refuse(f"image {args.image!r} is not configured in {args.project} "
                       f"(images: {', '.join(sorted(registry))})")
    if spec.catalogue_shift:
        return _refuse(f"image {args.image} has a catalogue_shift: catalogue and link addresses differ, which a "
                       "pinned relink does not support yet")
    try:
        data, delta = spec.load(), spec.delta
        extent = _extent(args.extent)
    except (OSError, ValueError, LookupError) as error:
        return _refuse(error)
    is_rom = Path(spec.path).suffix.lower() in pg.ROM_SUFFIXES
    rom = next((item for item in registry.values() if Path(item.path).suffix.lower() in pg.ROM_SUFFIXES), None)
    try:
        rom_sha1 = hashlib.sha1(Path(rom.path).read_bytes()).hexdigest() if rom is not None else None
    except OSError:
        rom_sha1 = None
    cfg = load_config(args.project, None, args.toolchain_dir)
    try:
        return rl.relink(args.source, args.image, args.build, data=data, delta=delta, is_rom=is_rom, extent=extent,
                         cfg=cfg, assumed=args.assumed, linker=args.linker, rom_sha1=rom_sha1, any_rom=args.any_rom,
                         limit=args.limit, keep_image=args.keep_image)
    except rl.RelinkError as error:
        return _refuse(error)


def _args(p):
    p.add_argument("source", help="a decompile-project tree (read only)")
    p.add_argument("--project", required=True,
                   help="the decompile-init project of your ROM (read only): the ROM, the image files, the toolchains")
    p.add_argument("--image", default="main", help="the image to relink (default: main)")
    p.add_argument("--build", required=True,
                   help="build directory: objects, raw blobs cut from your ROM, link.ld, the linked image and "
                        "RELINK.json; never inside the project tree, the kit or a git work tree")
    p.add_argument("--extent", metavar="VRAM_START:VRAM_END",
                   help="hex VRAM range to compare (default: config/layout.json for main, the whole image file "
                        "otherwise)")
    p.add_argument("--assumed", choices=("raw", "link"), default="raw",
                   help="assumed units (addresses read from the target): raw blobs (default) or linked (then no "
                        "level is claimed)")
    p.add_argument("--linker", choices=("gnu-ld", "kit"), default="gnu-ld",
                   help="gnu-ld: mips-linux-gnu-ld on the generated script (default); kit: the kit's own relocator "
                        "(a cross-check)")
    p.add_argument("--toolchain-dir", help="toolchain root (default: the project's configuration, N64DK_TOOLCHAIN)")
    p.add_argument("--any-rom", action="store_true", help="do not require the ROM SHA-1 the project records")
    p.add_argument("--keep-image", action="store_true",
                   help="keep the linked image (relinked.elf, relinked.bin) in --build; it holds the raw blobs, so by "
                        "default only raw/raw.o, the extraction input, keeps ROM bytes")
    p.add_argument("--limit", type=int, default=50, help="items listed per category in the report")


relink_image.arguments = _args


@command("relink-cartridge", "Link every image of a generated project into a whole cartridge and compare it with "
         "the ROM; bytes kept apart by provenance (linked, repacked, preserved).  Grants no credit")
def relink_cartridge(args):
    """For each image the project registers: relink it (``relink-image``); place ROM-resident images
    at their ROM offset; rebuild a compressed code segment as encoder(relinked payload) only when
    that equals the stored stream exactly, otherwise keep the stored stream as a preserved input;
    every other byte is a preserved extraction input.  Writes <build>/CARTRIDGE.json."""
    from ..match import cartridge as cart
    from ..match import images as imagemod
    from ..match import projectgen as pg
    from ..match import relink as rl
    from ..match.toolchain import load_config
    try:
        registry = imagemod.load_registry(args.project)
    except (OSError, ValueError, LookupError) as error:
        return _refuse(f"{args.project}: {error}")
    images, rom_spec = {}, None
    for name, spec in sorted(registry.items()):
        if spec.catalogue_shift:
            continue
        is_rom = Path(spec.path).suffix.lower() in pg.ROM_SUFFIXES
        try:
            images[name] = (spec.load(), spec.delta, is_rom)
        except (OSError, ValueError, LookupError) as error:
            return _refuse(error)
        if is_rom and rom_spec is None:
            rom_spec = spec
    if rom_spec is None:
        return _refuse("the project registers no ROM image")
    rom = images[rom_spec.name][0]
    cfg = load_config(args.project, None, args.toolchain_dir)
    try:
        return cart.relink_cartridge(args.source, args.project, args.build, rom=rom, images=images, cfg=cfg,
                                     linker=args.linker, assumed=args.assumed, any_rom=args.any_rom,
                                     rom_sha1=hashlib.sha1(Path(rom_spec.path).read_bytes()).hexdigest(),
                                     keep_rom=args.keep_rom)
    except rl.RelinkError as error:
        return _refuse(error)


def _cartridge_args(p):
    p.add_argument("source", help="a decompile-project tree (read only)")
    p.add_argument("--project", required=True, help="the init project (its image registry and survey layout)")
    p.add_argument("--build", required=True, help="a fresh directory outside the project for objects and reports")
    p.add_argument("--assumed", choices=("raw", "link"), default="raw")
    p.add_argument("--linker", choices=("gnu-ld", "kit"), default="gnu-ld")
    p.add_argument("--toolchain-dir")
    p.add_argument("--any-rom", action="store_true")
    p.add_argument("--keep-rom", action="store_true",
                   help="write the assembled cartridge (it holds ROM bytes: never package it)")


relink_cartridge.arguments = _cartridge_args
