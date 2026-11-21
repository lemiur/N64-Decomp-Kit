# Matching brief (n64dk model stage)

Goal: C that the project's compiler turns into exactly the bytes of ONE function of the title, written in the
project's own style so it can be imported. Only the kit's strict verifier decides: `./kit exact` printing `EXACT`
is the one acceptance. `try` MATCH alone is not a result; nothing you say about a file is believed.

## Tools (run them from this folder; every call is logged and counted against your budget)

    ./kit dis                       disassembly with the project's names (`; NAME (address)` notes)
    ./kit draft draft.c             a decompiler draft plus normalised forms draft.formN.c (may fail)
    ./kit try a.c [b.c ...]         MATCH, or the differing hunks (- target, + yours), on every lane
    ./kit exact a.c [b.c ...]       the strict verifier: EXACT, EXACT-ASSISTED or NOT EXACT and why
    ./kit info                      the unit and the recipe(s) (compiler lane + flags)
    ./kit regdump a.c [lgLsjSrc]    GCC pass dumps (.lreg local / .greg global allocation, ...) and pseudo priorities
    ./kit neighbours                matched sources nearest this function, their recipes, struct definitions,
                                    declarations of the symbols the target uses
    ./kit boundary                  the true start and size when the catalogue is wrong

Options for try/exact/draft/regdump: `--flags='...'` replaces the compiler flags, `--pipe LANE` picks the lane
(e.g. kmc+builtin, kmc+asn64, ido5.3, ido7.1), `--size 0x..` verifies a repaired range (true start = this
function's address), `--show N` / `--lines N` for more hunks or lines. Put variants of one function in ONE call.

## Method

1. `info`, `dis`, `boundary`, `neighbours`; then `draft` and `try` the draft unchanged to see the distance.
2. Write natural C by hand from the disassembly; the draft is a hint, not a starting point to patch. The kit's
   own drafts usually start far away; what works is plain source, iterated with `try`, one change per build.
3. No invented temporaries or locals the original would not have, no gotos unless the shape needs them.
   Decompiler locals (`temp_v0`, `var_a1`) are noise: remove them.
4. Read types from the instructions: sign/zero extensions, masks, compare forms (see the compiler notes).
5. Check constants: every immediate, float literal, struct offset and array stride; jump tables and literals
   are compared byte for byte by `exact`.
6. Once `try` says MATCH, run `exact`. ASSUMED / CONFLICT / SECTION lines mean NOT EXACT: use the names `dis`
   prints (the project's name for an address it already names), pin a symbol with `--flags='--symbol=NAME:0xADDR ...'`
   when the project's catalogue name differs from the real address, record the rodata base `exact` prints.
7. If two tries do not move a difference, change hypothesis (struct vs array, types, loop form, statement order)
   instead of repeating variations.

## Rules

- One function per file, `#include` only the project's shared headers; declare externs, prototypes and structs
  in the file (`unk_XX` members by offset where the meaning is unknown), as the matched neighbours do.
- Ordinary C only. Inline assembly is never plain C credit; a macro header holding inline assembly makes the
  result EXACT-ASSISTED (recorded as an assisted class, never counted as C).
- Use only flags the project's recipes use; record any extra flag.
- Write only in this folder. No network. Do not run compilers or the kit any other way than `./kit`.

## What to leave behind (the stage reads these files)

- `final.c`: the file `./kit exact` called EXACT (or your best near miss if you ran out of budget).
- `LESSON.md`: ONE line: the general cause and the source change that fixed it (or what you ruled out),
  phrased so it helps on another function of another title: no addresses.
- `NOTES.md` (optional): what you tried and what remained, for the next attempt.
