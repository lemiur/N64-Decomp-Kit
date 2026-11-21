"""Interoperability with the formats other N64 decompilation tools keep their facts in.

* ``splat``    — splat-style YAML segment configurations, read into the kit's image and catalogue
  model (``[images.<name>]`` tables and catalogue TOML) and written back from it.  The kit has no
  YAML dependency: ``splat.load_yaml`` reads the subset of YAML that splat configurations use (and
  JSON-converted configurations are accepted as they are).
* ``symaddrs`` — ``symbol_addrs.txt`` (``name = 0xADDR; // type:func size:0x..``) read and written
  with every attribute, free comment text and non-symbol line preserved.
* ``maps``     — GNU ld ``-Map`` output and column-style map listings (address / symbol / object)
  read into symbol and object-order facts: link layout evidence.
* ``objdiff``  — an ``objdiff.json`` project view for trees that hold per-unit objects.

Everything imported is a claim made by another tool's configuration, never a verified fact: each
imported record carries ``PROVENANCE`` and nothing here sets or implies an exact (matched) status.
Nothing exported contains ROM bytes: only names, addresses, sizes and paths.
"""
from __future__ import annotations

PROVENANCE = "imported: not proof"
EXPORT_PROVENANCE = "exported by n64dk: kit catalogue boundaries and names, not proof"
