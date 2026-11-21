# Compiler notes (keyed by compiler family; the stage includes the section of the recipe's family)

## gcc

KMC GCC 2.7.2 (and other GCC 2.x for MIPS). Applies to every lane of the family (builtin assembler, the
period assembler `asn64`, gas): the assembler lane decides whether the slot after the last `jr $ra` and some
load delays are filled and whether `lui $at; addu; lw sym($at)` table loads appear (only the asn64 lane makes
those). Try both lanes for every function; a unit already covered by an assembly file may default to one lane.

- Register priority rule (local and global allocation): priority = floor_log2(refs) * refs / live_length, refs
  weighted by loop depth, ties to the lower pseudo number (the earlier declared / first used). Real loops (`for`,
  `while`, `do`) weigh references; goto loops do not. `regdump` shows the competing pseudos.
- Narrow types come from extension instructions: `sll/sra 16` = s16, `sll/sra 24` = s8, `andi 0xFF` = u8,
  `andi 0xFFFF` = u16. A callee result kept narrow (`sll/sra` right after `jal`) means the prototype returns that
  type; a parameter narrowed at one call and passed raw to another is an s32 going to an s16 prototype.
  A flag copied to another register right before `== 1` at a join is an `s16` flag.
- Loop forms compile differently: `for`, `while`, `do {} while`, `while (1) { if (!c) break; }`. A test at the
  bottom with a jump into it is `while`/`for`; a test copied to the top (folded entry test) is a top-tested loop.
- An unexplained 8-byte stack frame with nothing stored comes from a top-tested `for`/`while` loop (the folded
  entry test leaves a `use`), or from a struct copy through a slot, or a u32 index compared as (u16); KMC frames
  are often 8 bytes larger than the sum of the locals. N*16 extra bytes: N inlined calls of a static helper.
- Cross-jumping: identical tails are merged after register allocation. Two `if`s with the same ending become
  one test: write two ifs, not a ternary or a combined condition. A call repeated per branch (argument set up in
  each predecessor) was written per branch; each copy still counts toward the saved-register choice.
  Give each switch case its own temporaries when the target keeps tails apart. Block order follows source
  order: write cases in the order of their code (jump-table targets).
- Struct-member globals (MEM_IN_STRUCT_P): stores to scalar globals may pass stores through struct pointers in
  scheduling; when the target keeps source order, the globals are members of one struct or one array
  (`extern struct S D_x; D_x.a = ...`, `extern s32 D_x[4];`). Neighbouring globals loaded/stored strictly in order
  are one aggregate; a global whose address sits in a saved register across calls is an array element or is
  taken by a block-local pointer.
- Chained stores `a = b = x` store b first. `x += y` vs `x = y + x` swaps operands. A constant assigned in one
  block and tested in another is not folded. One statement per arithmetic step (`v = f(); v -= 3; v /= 4;`)
  gives fresh pseudos; one expression lets a value die earlier and share a register.
- Loads follow source order while stores to distinct fields reorder: permute the source order of assignments to
  move loads. A store the target keeps first in a join block was written in each branch.
- A parameter copied into a local (`T *p = arg0;`) changes where its saved-register copy is made; the position
  of that copy relative to the first call decides whether reorg puts it in the delay slot.
- Pointer walks vs indexing: a strength-reduced induction pointer comes from `a[i]` indexing; `p++` keeps a
  different base. 2D indexing `D[i][j]` vs a pointer walk decides whether constants are hoisted.
- `-fno-expensive-optimizations` shows as a redundant `andi` after `lbu/lhu` and odd scheduling; some projects
  build whole files with it. `-O0` code has `$fp` frames and every local on the stack.
- Fixed-point multiply `mult; mfhi $t6; mflo $t7; sll 16; srl 16; or` in FIXED registers is an inline-assembly
  macro in the original (assisted class). In other registers it is plain C:
  `union { s64 v; struct { u32 hi, lo; } w; } p; p.v = (s64)a * b; r = (p.w.hi << 16) | (p.w.lo >> 16);`

## ido

IDO 5.3 / 7.1 (libultra and IDO-built titles).
- Local-free sources: IDO register allocation follows expression shape more than declarations; sources written
  without extra locals (operands read in place) match more often. Try the version siblings (5.3 vs 7.1) and the
  optimisation levels (-O1/-O2/-O3, `-g` for unoptimised); libraries mix levels per file.
- `$at` appears in IDO output (assembler expansion of compare-and-branch and indexed addressing): it is not a sign
  of hand-written code there.
- Public SDK sources (libultra) are the best draft when the function is a library routine: compile them with
  the library's recipe before writing anything.
