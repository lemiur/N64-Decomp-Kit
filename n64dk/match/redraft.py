"""Recipe-specific rewrites of a decompiler draft.

A general decompiler writes one rendering of a function.  Which rendering a compiler turns back
into the same bytes depends on the compiler and its settings: an unoptimised build keeps every
local in a stack slot in an order the declarations decide, an optimising one does not.  Each
generator registered here knows one such situation.  It receives the draft *and the target's
machine code*, so what it writes is computed from the bytes (the frame layout, the homed
arguments, the widths of the accesses), not searched for.

A generator is ``variants(source, name, *, code, vram, pipeline, flags, family[, read]) ->
[(label, text)]`` (``read(address)`` returns a word of the image, for data such as jump tables).  It returns nothing when the situation is not its own.  A generator must
never raise for a draft it does not understand: the portfolio it adds to already holds the
unmodified draft, so declining costs nothing and a wrong rewrite costs one build.
"""
from __future__ import annotations

import importlib

# module name -> why it exists; each module exposes ``variants`` with the signature above
GENERATORS: dict[str, str] = {
    "redraft_ido_debug": "IDO built without optimisation (-g / -O0): locals, arguments and frame as IDO lays them out",
    "redraft_gcc_debug": "GCC built with -O0: locals, arguments and frame as GCC 2.x lays them out",
    "redraft_gcc_opt": "GCC -O1/-O2 near misses: source forms GCC 2.x distinguishes that a decompiler does not",
    "redraft_ido_opt": "IDO -O1/-O2 near misses: source forms IDO distinguishes that a decompiler does not",
    "redraft_elements": "pointer sums written in bytes, said in elements (a byte cast changes IDO's registers)",
}
MAX_VARIANTS = 10               # per generator and draft: each one costs a build for every unmatched function


def _load(module: str):
    try:
        return importlib.import_module(f"{__package__}.{module}")
    except ImportError:
        return None


def variants(source: str, name: str, *, code: bytes, vram: int, pipeline: str, flags: tuple[str, ...] = (),
             family: str | None = None, generators=None, read=None) -> list[tuple[str, str]]:
    """Every registered generator's rewrites of ``source`` (only those that differ from it)."""
    out, seen = [], {source}
    for module in (generators if generators is not None else GENERATORS):
        found = _load(module) if isinstance(module, str) else module
        if found is None or not hasattr(found, "variants"):
            continue
        try:
            context = dict(code=code, vram=vram, pipeline=pipeline, flags=tuple(flags), family=family)
            try:
                # ``read(address) -> word | None`` reads the image (a jump table lives in its data)
                rows = list(found.variants(source, name, read=read, **context))[:MAX_VARIANTS]
            except TypeError:                               # a generator that takes no reader
                rows = list(found.variants(source, name, **context))[:MAX_VARIANTS]
        except Exception:                                    # a rewrite is optional; the draft stands
            continue
        for label, text in rows:
            if isinstance(text, str) and text and text not in seen:
                seen.add(text)
                out.append((str(label), text))
    return out
