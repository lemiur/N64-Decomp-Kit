# Techniques per diagnosis category (the stage includes the function's category)

Read the near miss and its notes first; change one thing per `try`, then `exact`. Every lane, every time.

## reg-swap
Same instructions, registers differ.
- Use `regdump` to see which two pseudos trade places, then change declaration order or first-assignment order.
- Prologue copy order wrong: take the parameter as `void *arg0` and assign a typed local (`Obj *obj = arg0;`).
- Saved registers rotated among constant-initialised locals: narrow them (s32 -> s16).
- a0/a1 swap around a table: declare it as a plain 2D array instead of a struct of arrays (or vice versa).
- Pointer walk -> index (`p[i]`), or the element address assigned inside the body (`o = &D[i];`).
- A call repeated per case: write it in each case instead of one shared tail.
- `p = f(); p += g(x) * k;` vs `p = f() + g(x) * k;`: one expression lets x die before p is born.
- A u8 argument in the wrong argument register: declare it u32 and cast at every use.
- Reuse one variable for successive cursors or loop counters when the target reuses one register.

## schedule
Order of independent instructions, delay slots.
- Permute the source order of assignments (loads follow source order; stores to distinct fields reorder).
- Scalar globals passing struct stores: make the globals one struct/array; or write the member store as a
  non-struct access (`*(s32 *)p = 90;`) to keep the order.
- A store the target keeps first in a join block: write it in each branch.
- Redundant `andi` after a byte/halfword load: try `-fno-expensive-optimizations` (record the flag).
- `x = D + x` vs `x += D`; assembler lanes differ in delay-slot filling: try both.

## merge-tail
Cross-jumping merges or fails to merge tails.
- A merged test from two ifs with the same ending: write two separate ifs.
- The target keeps tails apart: give each case its own temporaries; the target merges: write the identical tail
  literally in each branch and `return x;` per case.
- A shared constant store at the end: write it once after the switch, for every case and the default.
- Block order follows source order: cases in jump-table target order; descending when the fall-through body is
  the highest case.

## hoist
Invariants hoisted or not; reload vs reuse.
- The target reloads a global where yours reuses it: a store or a call sits between the reads in source, or the
  reads go through different expressions; a store to a sibling field between two reads forces the reload.
- An address kept in a saved register across calls: a block-local pointer (`u16 *p = &D_x;`) or a pair of
  globals declared as one array.
- A mask hoisted out of a loop where the target re-applies it: the field is a bitfield.
- A constant folded into the wrong operand: a cast stops reassociation (`(s16)(g + 20)`); an index temp
  (`j = n + 1`) keeps a sum unfolded.

## frame
Stack size and slots.
- Unexplained +8: a top-tested `for`/`while` loop, a struct copy through a slot, a u32 index compared as (u16)
  (declare it u16), or an unused local array the original had.
- A vector on the stack 8 bytes short: `s32 vec[2]` -> `s32 vec[3]`; a jump table from 0 with a bigger frame:
  add the empty `case 0: break;`.
- A parameter spilled to a stack slot: the copy `T *p = arg0;` was assigned after a call; spill-slot order is the
  declaration order of the spilled locals.

## type
Extension, mask or compare forms differ.
- Choose every variable's type from the target's extension instructions (see the compiler notes); signed vs
  unsigned changes compares, shifts, division sequences and switch test order.
- A redundant extension of a just-loaded value: the global is `volatile`, or the value went through a
  `static inline` helper with s32 parameters.
- Callee arity: an argument register the caller never sets means the callee takes fewer arguments (decompilers
  guess extra ones).

## unwritten
No draft, or nothing close.
- `dis`, `neighbours` for struct layouts and the declarations of the symbols used; write the function by hand
  section by section, fixing the first differing hunk each build.
- Callees whose catalogue name differs from the real address: use the real-address name and pin it with
  `--flags='--symbol=func_X:0xX <the recipe flags>'`.

## boundary
The catalogue boundary is wrong (no prologue, a call without saving `$ra`, a tail jump or branches leaving the
range, or the function falls into its neighbour).
- Run `boundary`; write the WHOLE true function and verify it at the true start with `--size` (name it after the
  true start). Never claim a piece on its own. Multi-entry functions are verified from the first entry.
- A leading instruction that is not C-expressible (a dead computation the compiler would remove) usually belongs
  to the previous function: check the boundary before anything else.

## assisted
Classes that verify only with help and are never credited as plain C.
- Inline-assembly macros (e.g. a fixed-point multiply in fixed registers): a macro header in your folder makes
  the result EXACT-ASSISTED; the stage records it as assisted.
- Hand-written assembly (cache, COP0, exception code) stays assembly with its provenance; do not imitate it in C.

## other
- Read the whole diff once more; check the recipe (`info`), the lane, the optimisation level and the boundary.
