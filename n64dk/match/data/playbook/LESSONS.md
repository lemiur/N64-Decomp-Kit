# General lessons (title-independent; the project's own lessons file adds to these)

- What is left after the unattended run is the hard remainder: expect to write from `dis` by hand.
- Try every lane of the recipe's compiler for every function; use the project's own recipe for a covered unit
  first (some carry extra flags).
- Catalogue boundaries are often wrong: check `boundary` before spending builds on a function without a
  prologue or one that falls into its neighbour.
- Types decide instructions: read the extension and compare instructions and choose variable types from them;
  splitting `x = (a << 8) | b` into `x = a; x <<= 8; x |= b;` changes the code.
- Display-list code matches when written with the SDK's `Gfx` type and macros as the SDK expands them, one
  block-scoped command pointer per command.
- Before typing a global, check its type in the project's matched sources (`neighbours`): an `lh` of an s32
  global means `x >> 16`.
- A jump table or literal in another unit's rodata needs `--section-base=.rodata:ADDR` (exact prints the base).
- A file whose `try` says MATCH and whose only `exact` complaint is a symbol name is fixed by the name or a
  `--symbol` pin, not by changing the C.
- Twins (identical code at other addresses, relocations masked): match one, then write the twin with the twin's
  names and run `exact` on it too.
- Record what you ruled out: the next attempt reads it and must not repeat it.
