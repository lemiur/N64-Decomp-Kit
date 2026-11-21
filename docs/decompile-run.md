# `decompile-run`: one unattended pass over a title

`decompile-run` takes a project that declares its images and runs every automatic stage the kit
has, in order, with a checkpoint after each. It writes only below `--out`; the project tree is
read, never changed. Nothing in it is specific to a game, a compiler or an address layout.

Madden 64 users can copy the reviewed raw-overlay declarations from
[`madden-images.example.toml`](madden-images.example.toml). Overlay 1 and Overlay 4 require
catalogue-to-link shifts; an unattended run should not guess a delta from their composite images.
The proof and acceptance details for this release are in
[`round5-integration.md`](round5-integration.md).

<!-- example: fixture=project requires=project,compiler,source id=decompile-run-donors -->
```sh
n64dk --format json decompile-run --project . --all-images --out out/run \
    --donors donors.toml --max-size 0x400 --budget 100
```

`--effort quick|normal|deep|exhaustive` chooses how much work follows this pass; the default
`normal` is the single pass described here. Deep and exhaustive add cause-driven rounds until a
fixed point or a budget, `--resume RUN` continues a run and `decompile-status RUN` shows its
rounds: see [solver.md](solver.md).  `exhaustive` is the kitchen sink: nothing is skipped to save
builds (the recipe gate and search economics are off) and every opt-in technique is on (region and
unit recipe retries, regional calibration).  `decompile-everything ROM... --out DIR` runs init, an
exhaustive run, export and status for each title.

The export's `FUNCTIONS.tsv` has a `unit_class` column (`boot`, `rsp`, `handwritten`, `sdk`,
`game`) and `COVERAGE.md` counts the classes apart: no figure combines boot code, RSP microcode,
hand-written routines and compiled C.

## Starting from a ROM alone

A title nobody has worked on has no catalogue, no symbol list and no known compiler.
`decompile-init` derives all three from the ROM and writes a project `decompile-run` can use:

<!-- example: fixture=rom requires=rom,compiler,source id=decompile-run-rom-alone -->
```sh
n64dk decompile-init game.z64 --out work/game
n64dk decompile-run --project work/game --all-images --out work/game/run --allow-project-output \
    --donors donors.toml
```

- **Code and boundaries** come from the survey of the ROM (the boot segment, or a packed
  program's unpacked form). The ROM is linked, not copied, and never written.
- **Library names** are the SDK matcher's confident identifications (`survey/sdk_symbols.txt`).
- **The recipe** starts from the compiler family the SDK fingerprint attributes to the game code.
  That is only a guess, so the project sets `[run] calibrate = 96`: before the sweep,
  `decompile-run` resolves a spread sample of small functions with every available recipe tried
  on each (other compilers, `-O1`, `-O3`, unoptimised) and sweeps with the recipe most of the
  byte-identical ones were built with. Fewer than three matches keeps the configured recipe, and
  the report says so. `--calibrate N` sets the sample for any project.
- **Code outside the boot segment** is looked for in three ways, each of which makes an image of
  its own (`[images.seg_<ROM offset>]`):
  - a ROM range a recognised loader reads raw, placed where its own calls agree;
  - a compressed stream that decodes to code (zlib, gzip, header-less deflate, MIO0, Yay0, Yaz0),
    unpacked to `survey/seg_<ROM offset>.bin` and placed the same way (`--no-compressed` skips it);
  - any other ROM range with the density of compiled code, placed by one vote over the whole ROM:
    a linked program keeps one distance between a function's ROM offset and its address, so every
    (call target, prologue offset) pair votes for that distance. Overlays that share addresses
    have different distances and are told apart (`--no-region-scan` skips it).
- **Identical functions are settled automatically.** The project sets `[run] declare_assumed = true`:
  a function that is byte-identical but leans on an undeclared data address is verified again
  with `D_<address>` declared at that address. Only names that state their own address are
  declared; the verdict is still the strict verifier's.
- **The whole ROM is mapped, not only its code.** `survey/layout.json` lists the header, the boot
  code (IPL3, with the CIC), the main program's `.text`, `.data`, `.rodata` and `.bss` (the
  last has no ROM bytes), RSP microcode and its data found inside the program image, the
  identified microcode and SDK versions, every further code segment, and the bytes each
  classifier accounted for. `survey/extract/` holds each piece that has bytes as its own file.
  Ranges nothing explains are counted as unknown.
- **Compilers.** `n64dk compilers` lists every compiler known to have built N64 retail code
  (seventeen, the set the public decomp.me service keeps for the platform), which are installed
  under the toolchain root and which the kit can drive. `n64dk compilers --fetch` downloads the
  missing ones from that public archive into the toolchain root; every layer is checked against
  the SHA-256 its manifest states before it is unpacked, and nothing is executed. Four have
  built-in pipelines (IDO 5.3, IDO 7.1, KMC GCC 2.7.2, SN GCC 2.8.1). Nine more (IDO 5.2,
  IDO 6.0, MIPSpro 7.4.4, three SN builds of GCC 2.7.2, the rebuilt SN GCC 2.7.2, GCC 2.8.1,
  EGCS 1.1.2) are driven as configured lanes: `decompile-init` writes a `[pipelines.<lane>]`
  table for each one that is installed, and calibration and the per-function alternatives then
  try them like any other recipe. The SN kits need `wibo` (it runs their Windows executables)
  and the rebuilt SN GCC needs a MIPS GNU assembler; a lane whose helper is missing is left out
  and `n64dk compilers` says why. A title built with a compiler that is not installed cannot be
  matched; check this before reading a low result.
- **A refiner can be recorded at the start.** `decompile-init --refine-command CMD` writes
  `[run.refine] command`, so the one `decompile-run` that follows also asks it about every case
  the mechanical pass leaves.
- **What is covered** is stated in `survey/blind.json`: every image and its code range, and under
  `code_not_placed` the ranges that look like code but whose calls support no load address
  (they are not attempted). A stream in a compression format the kit does not decode is not seen.


**Getting into the program.** Two things the survey has to get right before anything else:

- *Where the image is.* The boot code copies the first megabyte to the address in the header,
  except under the boot chips that move it (6103 by 1 MiB, 6106 by 2 MiB): the image is placed
  where the code actually runs.
- *Where the code starts.* The entry routine usually calls the program; some jump to it through
  a register, with data lying between the routine and the code.  The address it hands over to
  is a function start in its own right.

Titles whose boot segment only unpacks the real program with a compressor the kit does not know
are still surveyed as a handful of functions.

## Stages

| stage | what it does | output |
|---|---|---|
| discovery | audits every image against call, branch, pointer, prologue, datamap and placement evidence; by default repairs evidence-backed omissions in a run-local catalogue without modifying the project | `discovery.json`, `discovery/`, repaired `functions.toml` plus `*.unrepaired.toml` |
| boundaries | reconciles catalogue starts with call targets, prologues, padding and fragment chains; proposes reversible operations only | `<image>/boundary_operations.json` |
| donors | looks every function up, relocation-masked, in the donor index; a hit is rebuilt from the donor's source with this title's symbols and compared | `<image>/donors/`, `stage_donors.json` |
| resolve | for each remaining function: declarations recovered from the machine code, a portfolio of drafts (plain, with callee/own prototypes, loop-normal, type-narrowed), compile repair, recipe selection, one search over all transformation families, twin reuse | `state/` (case store) |
| refine | optional: an external author answers for the cases the ladder left (see below) | `state/` |
| declare | optional (`[run] declare_assumed`, `--declare-assumed`): declares the self-describing data addresses (`D_<address>`) that byte-identical cases lean on and verifies those cases again | `<image>/declared_symbols.txt` |
| queue | exports what is finished | `queue/queue.tsv`, `queue/<image>/*.c` |
| ledger | one owner class per byte, with violations | `<image>/ownership.tsv` |

**Running again.** A second run over the same output directory never loses a result. An `exact`
case is returned as stored; an `assumed` case is only verified again (it becomes `exact` when its
addresses have been declared since); any other case continues from its stored best source under
the recipe that source was built with, and the stored case stays when the new attempt ends
further from the target.

The discovery policy is `--discovery repair` by default. `--discovery audit` reports without
repairing and `--discovery off` restores the previous behavior. A report that remains
`incomplete` after repair warns that every percentage is over an incomplete catalogue.
`toolchain-regions` and each run report separately describe stretches that fit known recipes,
are uncertain, or differ in a compiler/assembler component; these classifications guide and
explain work but do not grant matching credit.

**What the resolve stage does beyond drafting.**

- *Padding.* A catalogue interval that runs to the next function includes the zero words that
  align the next object. No C produces them, so the interval that is resolved ends at the code
  (`resolve-hard --keep-padding` restores the old behaviour); the summary counts what was trimmed.
- *Drafts × recipes.* When no draft is exact under the configured recipe, the best drafts are
  built under the sibling compilers with the same flags and under the other optimisation levels
  of each (`-O1`, `-O3`). By default, the broad retry is gated by unoptimised-code evidence in
  the function or its catalogue neighbourhood; `--no-recipe-gate` restores the earlier retry
  for every far case.
- *Frame fit.* When the candidate's stack frame differs from the target's, every stack-relative
  instruction differs and the score says nothing about the rest. The difference is measured, so
  the repair is computed: temporaries that only name a sub-expression are substituted, and the
  bytes still missing are reserved with an unused local.

**Parallelism and machine safety.** `--jobs` is the total number of parallel builds. The default
planner caps it at four, reserves 2 GiB of currently available memory, and includes swap usage and
the limiting reason in the `resource-plan` event. `N64DK_MAX_JOBS` lowers the process-wide cap.
Only `N64DK_ALLOW_OVERSUBSCRIBE=1` permits a requested value above it; that is an explicit unsafe
override. The resolve stage can split functions among worker processes (`--workers`), but the
resource plan bounds the compile concurrency handed to it. Batch runs process titles sequentially.

Every compiler/assembler subprocess has a timeout and its own process group. Timeout or interrupt
terminates the group, so abandoned compiler children cannot continue consuming memory or I/O.
Object-cache identities cover the source, recursively included headers, pipeline, flags,
normalizer, environment and tool binaries.

When m2c is importable, drafting uses a bounded fork-server pool by default. The server imports m2c
once but forks a fresh, timeout-controlled child for every draft, and is covered by the same
resource tokens, process-group cleanup and parent-death handling as other tools.
`N64DK_M2C_SERVER=0` selects the command-line path for diagnosis or comparison.

**Where the search is not tried.** The variant search runs only when the best draft is within
`--search-ceiling` of the target (default 40, `[run] search_ceiling`, 0 to always search). Over
1,712 searches on an IDO title and a KMC GCC title, none that started above 40 reached an exact
match; they took 37% of the builds. Such a function needs a better draft, not more variants, and
its case says so.

The run can be interrupted and restarted: finished stages are read from their checkpoint
(`--redo` recomputes them) and finished cases from the case store. `--skip-stage NAME` omits a
stage; `--limit N` and `--max-size` bound a trial run.

**Search economics** (`--search-economics`, off by default) keeps, per stratum (the image, the
recipe a function is built under, whether the toolchain classifier called its code unknown, and
how far its first draft is), what the variant search has cost and returned.  A stratum that has had 24 searches and 600 builds without making a function exact or
bringing one within ten differences is not searched any more, except every eighth function as a
probe; its cases record `iterate-skipped` with the stratum's numbers, and
`state/economics_<image>.json` ranks the strata by yield per build.  Nothing is skipped on a
classification of the target alone.

**Nothing left out silently.**  Every function a resolve pass leaves out (`--max-size`,
`--limit`, sampling, `--only-names`) is listed with the reason in `state/scope_<image>.json`,
and the export shows the reason beside `not attempted`.

**Measured experiments that are not defaults.** `--region-recipes` retries unresolved functions
under a recipe inferred for an address region (3 exact results in 126 measured retries).
`--unit-recipes` uses a nearby unit's recipe (0/165). `--skip-unknown-search` trusts the
unknown-toolchain classifier enough to suppress variant search; without that switch the
classification is advisory. These switches remain available for controlled experiments, but a
normal unattended run does not spend work or discard search based on those weak measurements.

## Small data: `-G`, `$gp` and GP-relative relocations

Code built with a nonzero `-G` keeps small globals (`.sdata`, `.sbss`) and float literals
(`.lit4`, `.lit8`) within 32 KiB of `_gp` and addresses them as `offset($gp)`.  Such code cannot
match without the program's `_gp`, and its objects carry `R_MIPS_GPREL16` and
`R_MIPS_LITERAL` relocations.

* **Stage 0b** (`--small-data auto`, the default) recovers `_gp` from the program's own code: every
  `lui $gp` / `addiu $gp` (or `ori`) pair in catalogued functions forms a value; all setting sites
  must agree, the value must be nonzero, and the evidence must be strong (two or more sites, or one
  site and at least 16 GP-relative uses).  Otherwise nothing is declared.  The value is written to
  `small_data/gp_symbols.txt` in the run directory and declared to the resolver; a project that
  declares `_gp` itself keeps its value.  The report's `small_data` entry says what was found.
* **Relocation.**  GP-relative fields are relocated against the declared `_gp`
  (`A + S + GP0 - GP` for a local symbol, `A + S - GP` for an external one, with the object's
  `.reginfo` GP0).  A value outside the signed 16-bit range is an error, never truncated, and
  without `_gp` the object does not relocate at all.  `.sdata`, `.lit4` and `.lit8` are compared
  byte for byte like `.rodata` and `.data`; sections and symbols inferred through `_gp` from the
  retail code are assumptions, as with `%hi`/`%lo`.
* **Lanes.**  The established lanes still assemble at `-G0`.  The small-data lanes give the
  assembler the compiler's `-G`: `kmc+asg` (KMC `as` 2.6: `.comm` symbols and float literals
  through `$gp`) and `kmc+gas` (period GNU `as` 2.9.x, `[toolchains] gas29` or found under the
  toolchain root: also symbols the compiler declares `.extern name, size`).  They are offered only
  when `_gp` is declared and the function's own code addresses through `$gp`.
* **Drafts** name GP-relative accesses `%gp_rel(sym)($gp)`, so the decompiler sees the global.
* `--small-data off` skips all of it.

## A set of titles: `decompile-batch`

```sh
n64dk decompile-batch a.z64 b.z64 c.z64 --out DIR [--donors donors.toml] [--refine-command CMD]
```

runs `decompile-init` and `decompile-run` for each ROM, one at a time, with one library
(`DIR/library`, see below). With the default `--passes 2` every title is visited a second time
after all have run once, so early titles also receive what later ones matched; the second visit
recomputes the donor stage and continues the stored cases. Projects are `DIR/<title>/`, runs
`DIR/<title>.run/`, and `DIR/batch.md` / `batch.json` hold the per-title and total numbers after
every title. A title that fails is reported and the batch goes on.

## A library of finished runs

`decompile-run --library DIR` makes every title a donor for the ones that follow.

- **At the end of a run** the strictly exact functions are described as donor entries in
  `<out>/donor.toml` (one per image: the sources and per-pipeline manifests under `<out>/queue`,
  the image, its catalogue and its symbol files) and that description is copied to
  `DIR/<ROM digest>.toml`. `donor.toml` is written whether or not `--library` is given, so a
  finished run can also be named in a `--donors` file by hand.
- **At the start of a run** every other title in `DIR` is added to the donors. A title is told
  apart by the content of its ROM and is never its own donor. An entry whose image no longer
  exists is skipped.
- **Nothing transplanted is trusted.** The donor stage compiles each donor unit for the receiving
  image and compares bytes and relocations, as it does for a hand-written donor project.

What this is for: code titles share that no SDK signature names (a studio's engine, a middleware
library, the same library built by the same compiler), and functions a refiner was paid to
match. Matched once, they are not matched again.

## Drafts that do not compile

A decompiler's draft is not always valid C for the target compiler. `n64dk/match/autodraft.py`
repairs the diagnostics that have one mechanical answer (a missing declaration, a call through a
function pointer loaded from a table written without parentheses, a pointer compared with an
integer, arithmetic on a `void *`, a built-in function called with other arguments, and so on),
up to twelve repairs per draft. A draft that still does not build is kept with the case
(`source_digest` in its `baseline` stage) together with the compiler's output, and the refine
stage shows both to the refiner.

## The refine stage: an external author for the hard cases

Drafting and mechanical search stop paying off beyond the small functions: on the finished
projects used for validation they settle near 40% of a title's functions, and a search budget
twenty times larger changed nothing. What the remaining cases need is authorship (the type a
value really has, the idiom the programmer used). The refine stage hands each of them to a
command you configure and judges the answer with the same oracle as everything else.

```toml
[run.refine]
command = "my-refiner --flags"   # reads the prompt on stdin, prints C on stdout
rounds = 3                       # answers per function, each after seeing the previous result
max_cases = 200                  # nearest first; 0 = all
max_score = 0                    # only cases at most this far from the target; 0 = all
statuses = ["near", "open", "does-not-build"]
max_size = 0x400
jobs = 1                         # calls at once (model work: one at a time unless you raise it)
known_source = []                # folders of source the author must not be shown (blind evaluation)
timeout = 600
```

or `--refine-command "..."` on `decompile-run` and `resolve-hard`
(`--refine-rounds`, `--refine-max-cases`, `--refine-max-score`, `--refine-jobs`,
`resolve-hard --refine-names a,b` to ask about named functions only, `--refine-known-source DIR`).

Answers are cached by the digest of the settings and the prompt (`state/refine_cache/`): a run
started again asks nothing it has asked before.  Every refine stage records its cost (calls,
cached answers, seconds, prompt and answer bytes) and the pass reports the totals.  With
`--refine-known-source DIR` (an imported tree's `split/` folder) a prompt holding a line of that
source is not sent, and an answer whose body lines are mostly that source is marked
`known_source`: the author remembered published code, which is not a blind recovery.

The configured command is executable code and runs with the invoking user's OS permissions. Use
only a trusted local wrapper. It starts in an empty temporary working directory, but this is not a
security sandbox. Candidate output is size-bounded, forbidden assembly escapes and external
preprocessor directives are rejected, and accepted C must still pass the normal strict verifier.
The prompt can contain target disassembly and candidate source; configuring a network-backed
command authorizes sending that material to its provider.

- **Protocol.** One prompt per round on stdin: the rules, the target assembly, the declarations
  recovered for that function, the best C so far, what the compiler made of it (classified
  instruction differences, or the diagnostic when it did not build), the diagnosed causes, and
  the earlier answers of this conversation with their results. The answer is one fenced C block
  holding a complete translation unit. The kit adds its typedef prelude.
- **Nothing is believed.** An answer is built under the case's recipe, then under the configured
  default and its siblings. It replaces the case's best only when it scores better, and then goes
  through the normal ladder (search, strict verification, classification). `exact` still means
  the strict verifier's verdict; an answer that is byte-identical but leans on an undeclared
  address is `assumed`, like any other. A wrong, empty or slow answer costs builds.
- **Any author.** The command can be a headless coding assistant, a local model or a script.
  For example a wrapper that passes stdin to an assistant's non-interactive mode and prints its
  reply. The prompt contains disassembly of the image and candidate sources: whoever configures
  the command decides where that text is sent.
- **Worked examples.** Each prompt carries up to `examples` (default 3) functions of the same
  image that already compile to their exact bytes under the same recipe, as (assembly, source)
  pairs: those sharing callees or data with the target first, then its nearest neighbours.
  Functions the refiner settles join the pool, so later cases learn from earlier ones.
  `examples = 0` turns this off.
- **Resumable.** A case already refined by the same command and settings is not asked again.
  Changing the refiner configuration gives it a new identity; `resolve-hard --force` reruns it
  even when the identity is unchanged.

What it can do depends entirely on the author. Two measurements exist, on a 173-function sample
of a finished IDO title treated as unknown. With an author that returns the project's real source
(a check of the stage, not of any model) 151 of 173 end byte-identical, against 47 without the
stage. In a pilot of seven hard cases answered by a language model through this interface, all
seven became byte-identical: five on the first answer, one on the second, one on the third.
No unattended whole-title run with a real model has been performed; the oracle-source plumbing
check and seven hand-selected cases do not establish a recovery rate, latency, cost, or privacy
profile for any provider.

## The SDK as a donor: `sdk-donor`

```sh
n64dk sdk-donor --source ULTRALIB --out DIR [--versions F,H,I,J,K,L] [--flavours libgultra_rom,libultra_rom,libgultra]
```

builds a source tree of the SDK library once per version and flavour and describes each build as
a donor (`DIR/donors.toml`, one image, catalogue and manifest set per build). The C units are
compiled with the recipe the library's own makefile gives each of them. The hand-written units
(`.s`: the exception handler, interrupt masking, cache and TLB routines, `bcopy`, `bzero`) are
preprocessed once and assembled: with the GNU assembler for the GCC builds, through IDO's own
assembler for the IDO builds (its objects list local and global symbols mixed; the table is put
in ELF order before the link). Labels inside a hand-written routine are not functions and are
left out of the catalogue. The addresses the boot code leaves in low memory (`osTvType`,
`osRomBase`, ...) are the same on every console and are defined in the link.

The donor stage treats an assembly unit like any other: found by its relocation-masked bytes,
rebuilt for the receiving image, compared.

**Several donor units for one place.** A function of the SDK is in every build of it that was
indexed, in the projects that contain the SDK, and in the library of finished runs.  Up to eight
*distinct* units are built for one place: the same source file under the same compiler and flags
(macro definitions aside) with identical donor code counts once.  Units of decompiled projects
and of the SDK are tried before units of the library, so a library that grows with every title
does not push them out.  (With four attempts and no order between the kinds, a title visited a
second time, after the library had grown, lost nine functions it had the first time.)

A place is a unit start *and* the size compared.  Units of different sizes that begin at one
address are tried separately, and of the rows that overlap the largest exact one is kept: an
empty function at the head of a unit is also some donor's 4-byte unit, and it must not stand in
for the unit that begins there.  A unit of several functions is also built when some of its
functions differ from the donor's only in `%lo` halves the masking missed; the strict comparison
decides.  Every catalogued function that lies wholly inside a verified unit's compared bytes is
recovered by it, whether or not it matched a donor function by itself (a blind catalogue cuts
functions short and lists their `jr ra; nop` tails apart).

## Donor units that await symbols

A donor unit can be byte-identical and still rest on an address nothing independent confirms.
Three cases are settled by evidence the address derivation never looked at:

- a **section of the unit itself** (`.rodata`, `.data`) whose compiled contents, at least eight
  bytes and not just zeros, are found in the image at the derived address;
- a **function** the unit calls that was itself found, identical, at that address and nowhere
  else: as a function hit, as a function of a unit found whole (its whole masked code stands
  there), or as a function of a unit already verified exact (the units still waiting are then
  compared again, for as long as that settles something). This holds also when the unit is the
  only one that calls it;
- a **fixed address** (low memory below `0x80000400`, the hardware registers) that is the same
  in the donor.

A fourth is settled by the units around it: **sections lying end to end**.  Each unit's `.data`
or `.bss` address is read from that unit's own code; that the sections of several units abut at
the sizes and alignments they were compiled with is something no single reading implies.  For
`.data` the rule is the one read-only data already had (at least two pieces, 16 bytes and 8
non-zero bytes over the chain, every piece's contents equal to the image, zero bytes between
pieces).  `.bss` holds no bytes, so the abutting addresses are all the evidence: at least two
pieces and 16 bytes, and only in the order of the units' code, as a linker lays them out.  A unit
in an accepted chain is compared again with that base given and is exact only if the strict
comparison is, with nothing left assumed.  Such a base has the pin origin `section-chain`
(apart from `section-content`), the report lists the chains and the units resting on them
(`section_chains`), and `verify-run` builds every unit the run lists and computes the chains
again: a unit whose section is in none of them fails its replay.

What is left (a `.bss` section nothing abuts, a data symbol no other unit placed) ends
`identical-awaiting-symbols`: the functions are not attempted again, are counted with the
`assumed` cases, and `<image>/donors/awaiting_symbols.txt` lists the declarations that would
settle them. Code that is identical at several addresses with a different symbol each time (a
getter, a setter), or whose fixed-address symbol is a different address in the target, is the
shape of a function and not the donor's function; it goes to the resolve stage as before.

A target function that is, masked, several functions of one donor at once (two getters, one
routine written for two devices) does not vote for an address at the sites where those functions
refer to different things: at most one reading is right, and the wrong ones used to contradict
the functions that really use those symbols.  `<image>/donors/hits.json` lists every hit with
its status, the start its unit would have and the notes.

## Hand-written code

Part of every image was never C: the entry stub, the exception and interrupt handlers, cache
and TLB routines. A function with a system instruction (`mfc0`, `mtc0`, `eret`, `cache`, the TLB
instructions), or with other strong evidence of hand-written code, is not drafted. Its assembly
source is written (calls, branches and address pairs as names; coprocessor registers by number
with their names beside them), assembled with GNU as (`-march=vr4300 -mabi=o64`), relocated and
compared. When the bytes are the target's, the case ends `assembly` and its source is exported
to `queue/<image>/<name>.s` and listed in `queue/assembly.tsv`. Otherwise it stays
`probable-handwritten`.

`assembly` is reported apart from what was matched from C: `decompile-batch` gives `identical`
(C) and `accounted` (C plus assembly). It is never offered for a function a compiler could
have produced, so it cannot stand in for decompilation.

## Code no known compiler explains

`n64dk/match/idioms.py` holds, for each compiler and level proven so far, how often its functions
show each encoding and convention (which register an address is built in, how a move is spelled,
whether arguments are stored on entry).  A stretch of an image that keeps doing something every
known class all but never does was built by a compiler or assembler the kit does not have.  One
and the same trait must rule every class out: where each class is ruled out by something
different, the stretch is only the seam between two known compilers.  A jump out of a function is
not counted (it is what a wrongly split function looks like).

Such functions are still drafted and tried under the recipes, but the variant search is not run on
them, and the run's report lists the stretches (`resolve.unknown_toolchain`) with the trait that
gave them away.  On the title that showed the case, 786 of 1,545 functions lie in such stretches;
they had taken three quarters of the title's builds and the search had matched none of them.
`n64dk recipe-prior --project P` shows the same map without running anything.

The remedy is a compiler, not a draft: `n64dk compilers` lists what is installed, and a lane for
a further compiler is a `[pipelines.<name>]` table.  A lane may list further usual settings
(`flag_sets`); the ones declared by `decompile-init` carry the optimisation levels and instruction
sets that published decompilations of real titles were built with.

**A configured recipe is checked against the code.** A project states its recipe (`[run]
pipeline`, `flags`) and, unless calibration is asked for, the run takes it on trust.  When the
marks put most of the likelihood of at least four tenths of the image's functions on another
compiler family or another optimisation level, the run calibrates anyway and records why
(`recipe_doubt` in the image's report).  A project that had been run at `-O2` for three rounds
turned out to be `-O1` this way.

## A title somebody has already decompiled

Source that is known to give a title's bytes is the shortest way to them, and the donor stage
already rebuilds and verifies foreign units.  What it lacked was the build: which files are
units, where the headers are, which compiler and flags each unit takes, which configuration
macros are set.  Every source tree says that in its own build files.  `source-donor` does not
read them; it measures:

    n64dk source-donor --source TREE --project PROJECT --run RUN --out DIR

- units are the C files with a function body (generated directories are left out);
- include directories come from the `#include` lines: a name not beside the including file is
  looked up among the tree's files by its path suffix;
- candidate recipes are the ones the title's own run matched functions under, plus the first of
  them at the other usual levels; each unit takes the recipe under which most of the title's
  functions are found in it (masked signatures, the donor stage's comparison);
- configuration macros are the names the tree tests in `#if` lines and does not define
  unconditionally.  Each is measured on up to 32 units that can see it and kept when more of the
  title is found with it.  The measurement runs in rounds, so a macro that only counts once
  another is set is found too, and of two that exclude each other (two releases) the better one
  is kept.

The result is a `donors.toml` for `decompile-run --donors`.  Nothing in it is trusted: each
function still counts only when its unit is rebuilt for the title and verified.

`decompile-batch --source-tree DIR` (repeatable) does this unattended.  Nobody says which tree
belongs to which title: after a title's first run every tree is measured against it, one that
gives it at least eight functions becomes a donor of that title, and the title runs again.  The
batch report lists every tree against every title with what was found.

What is deliberately not taken from a tree:

- **Assembly files.**  In most trees they are the title's own disassembly, and a unit built from
  them compares equal to the title without being a decompilation.
- **Routines a C unit pulls in through the assembler** (the macros trees in progress use for
  functions that are only disassembled).  Only a symbol whose name is followed by a parameter
  list and a body in the unit, or in a file it includes, is catalogued; the report gives the
  number left out (`symbols_not_written_in_c`).

A macro the tree compares with something (`#if RELEASE >= RELEASE_C`) is tried at each value it
is compared with.  Macros are measured first on the units with the most functions still unfound.

**Generated files.**  `--prepare-command SHELL` runs the tree's own generator in a copy (the tree
is never written), e.g. for SM64 the `textconv` steps that make `text_strings.h`.  A file still
missing gets a stand-in: `0` where it is included inside an initialiser, nothing elsewhere.  A
function that needs the real content fails to compile or compiles to other bytes, and the
comparison refuses it.

**Preprocessing** is done as C89 (`-std=gnu89`, no `__STDC_VERSION__`), as the IDO and KMC GCC
compilers see it: headers that test the version take their old branch.

**Whole units as well.**  `--units both` writes a second donor (`<name>_units`) made of the
tree's own units beside the one-function units: a function that only matches when its unit is
compiled together (shared literals, static data, the unit's layout) is taken through it, with
the whole unit standing in the title.  On Dr. Mario 64 this verified 4 functions the split could
not; on SM64 none.

**Every function accounted for.**  `source-donor-account --project P --run RUN --donor-dir DIR
[--donor a,b] [--out rows.json]` gives each function of the title one cause: `strict`,
`awaiting-symbols`, `other-donor`, `refused` (with the stage's reason), `near`,
`tree-function-differs` (placed by source order between two found functions of its unit, with
the word difference), `tree-function-not-compiled`, `tree-function-assembly` or
`no-tree-function`.  Measured (round 10): SM64 3,205 strict + 891 awaiting of 4,511 functions;
Dr. Mario 64 732 strict + 50 awaiting of 1,430.

## Every function, every byte

`decompile-export` writes a finished run as a tree to commit, and leaves nothing out:

- a function matched from C is its C source;
- a hand-written function is its assembly source;
- **every other function is written as assembly that assembles back to its bytes**
  (`asm_not_decompiled/`).  That is a disassembly, not a decompilation; the table
  (`FUNCTIONS.tsv`, column `reproduced_by`) and the counts keep the three apart.  A function
  with several jump tables gets each table in a section of its own.
- a function matched from C counts as matched only when every [match] code rule passes in its
  source.  The `rules` column says so (`pass`, `MATCH-PENDING: <rules>`, `ORIGINAL-ASM`,
  `unchecked (donor)`), the README gives *matched under the code rules* as the headline next to
  *bytes identical*, `RULES.tsv` lists every finding, `DECLARATIONS.tsv` every `extern` by name,
  and `AGENTS.md` is the project profile ([code-rules.md](code-rules.md));
- `COVERAGE.md` gives the functions by what reproduces them and **every byte of the ROM by
  class**, from the survey's ledger: its rows cover the ROM from the first byte to the last, and
  the report states the bytes no row covers and the bytes two rows claim (both must be 0).
  Content the survey has a row for and could not identify is the class `unknown`; its largest
  ranges are listed with what was measured about them.

Functions above 0x400 bytes are drafted and built under their recipe but not put through the
other recipes or the variant search: they are 5% of functions and 37% of the code, the
translators for unoptimised builds match some outright, and searching them costs the most.

## Where every function ended

`decompile-batch` closes its report with one table: for each title, the functions matched from C,
the hand-written ones kept as assembly, and the unmatched ones by the kind of thing that would
have to change (a compiler the catalogue lacks; the catalogue itself; a draft that does not
compile; a draft within ten differences; a draft far off, small and larger functions apart).

## What "finished" means

A case is `exact` only when the engine's strict verifier accepts it: owned text and data equal,
every relocation resolved through an independently declared symbol. Those are in `queue.tsv`.

A case that is byte-identical but refers to an address nobody declared is `assumed`. It is listed
in `queue/assumed.tsv`; `queue/symbols.assumed.txt` holds the declarations that would settle
them, one `NAME = 0xADDRESS;` per symbol with the functions that use it. Declaring a symbol is a
decision about the project, so the run does not make it: review the file, append it to the
project's symbol file, and run again. Those cases are then verified strictly like any other.

Everything else is in the case store with a status (`near`, `open`, `does-not-build`,
`needs-tu`, `blocked-boundary`, `probable-handwritten`, `needs-source`) and a ranked cause
(`assembly` is the finished state of hand-written code, above);
`hard-queue --state <out>/state` prints them grouped by cause for whoever works on them next.

`report.md` and `report.json` summarise the run; `events.jsonl` is the stream of what happened.
