# N64 Decomp Kit

`n64dk` is a game-agnostic command-line and Python toolkit for analysing Nintendo 64 ROMs,
recovering program structure and assets, identifying SDK code, and reproducing functions with
historical N64 toolchains.

Everything is derived from a retail ROM image plus optional project inputs (symbol lists,
manifests, sources); nothing is hard-coded to one game.

* **Boot and program image**: header, CIC and checksum; the entry routine (stack, BSS, hand-over);
  an R4300 boot emulator that follows loaders (PI DMA, decompression) and maps the program's
  RAM image back to its ROM sources.
* **Code**: function boundaries, jump tables and boundary audits; a discovery-completeness gate
  that finds evidence-backed code outside the catalogue; libultra/libaudio identification by
  masked signatures; regional compiler, assembler and microcode fingerprints.
* **Data**: cross-references, `.data`/`.rodata` split and ownership, typed data units.
* **Assets**: audio (libaudio banks, sequences, MIDI, libmus, VADPCM to WAV), graphics (F3D,
  F3DEX, F3DEX2, S2DEX display lists, textures to PNG, blob pools), compression (Yay0, Yaz0,
  MIO0, LZARI, deflate, with encoders) and loader tracing.
* **Matching**: build a C source with the original toolchain (KMC GCC, IDO 5.3/7.1, SN GCC),
  compare it relocated against the ROM with classified diffs, sweep pipelines and flags,
  re-verify whole manifests, diagnose the residual mechanism, explain GCC 2.7.x allocator
  dumps, enforce provenance, and use permutation only for the appropriate residual stage.
* **Code rules**: identical bytes alone are not a match. A mechanical checker reports the
  fakematches (unused locals, `volatile` carriers, empty blocks, offset casts, scattered
  declarations) that make a byte-identical function `MATCH-PENDING`, and the open naming work.
* **Autonomous tranches**: build interval-correct progress inventories, rank complete remaining
  functions, collect one-call context bundles, and distribute persistent matching sessions
  across functions while stopping exact winners independently.
* **Survey**: one command that runs everything and classifies every ROM byte in one interval
  ledger.

## Install

Python 3.11 or newer. The only required dependency is `rabbitizer`.

```bash
git clone https://github.com/TheGh0stShip/N64-Decomp-Kit.git
cd N64-Decomp-Kit
python3 -m venv .venv
.venv/bin/pip install -e .
```

First-draft lifting is optional. Install it with `.venv/bin/pip install -e '.[draft]'`;
ordinary analysis and matching do not require m2c.

Historical compiler binaries are not bundled. `n64dk compilers` reports the supported catalogue,
installed lanes, missing helpers and checksums; `n64dk compilers --fetch --toolchain-dir DIR`
can fetch checksum-pinned public packages. Review the reported sources and licensing before use.

ROM images are never part of the repository. Pass a path on the command line; tests and
fixtures find ROMs through `roms.toml` (ignored by git) or the `N64DK_ROMS` environment
variable (see [Tests](#tests)).

## Quick start

```bash
n64dk info baserom.z64
```

```bash
n64dk survey baserom.z64 --out out/survey
```

`survey` writes `report.md` (a summary with ranked next steps, then every analysis),
`report.json`, `ledger.tsv` (every ROM byte with a class, category and evidence),
`functions.toml`, `symbol_addrs.txt` and `main_segment.bin` (the program as loaded in RAM,
unpacked when the game packs it). Games whose boot loader unpacks the program with an
unknown codec are emulated, and the game's own routine is used as the codec. Add
`--require-complete` to fail while any byte is still `unknown`.

The published [N64 retail build survey](docs/research/N64_BUILD_VERSIONS.md) applies this evidence
pipeline to 66 ROMs and reports detected libultra release ranges, compiler-family shares, graphics
microcode strings and function counts. Machine-readable [CSV](docs/research/N64_BUILD_VERSIONS.csv)
and [JSON](docs/research/N64_BUILD_VERSIONS.json) accompany the report. Compiler results are family
classifications rather than exact historical version claims; ambiguous SDK ranges remain explicit.

For a ROM-only unattended decompilation pass:

```bash
n64dk decompile-init baserom.z64 --out work/title
n64dk decompile-run --project work/title --all-images --out work/title/.n64dk/run
n64dk decompile-export --project work/title --run work/title/.n64dk/run --out out/title
```

`decompile-init` discovers loaded images, functions, symbols, compiler evidence and a complete ROM
ledger, then audits and repairs incomplete catalogues from call, pointer and prologue evidence.
`decompile-run` repeats that discovery gate before trying verified donors, compiler-specific
draft generators, strict rebuilds, bounded source search and an optional external refiner. It
records regional toolchain evidence and a durable per-function attempt ledger so interrupted runs
resume without repeating settled work. `decompile-export` keeps four categories
separate: strictly verified C, C awaiting independent symbol declarations, verified assembly, and
generated assembly for code not decompiled. Generated assembly and raw-word fallbacks provide byte
coverage but never count as C decompilation.

The export reports two numbers for C: *bytes identical* and *matched under the code rules*. The
second is the headline: a byte-identical function whose source carries fake code, an offset cast
or a declaration outside its header is `MATCH-PENDING` and is not counted. See
[docs/code-rules.md](docs/code-rules.md).

The run planner is deliberately conservative: at most four build jobs by default, with 2 GiB held
in reserve and observed swap reported in the `resource-plan` event. Set `N64DK_MAX_JOBS` to lower
the cap. `N64DK_ALLOW_OVERSUBSCRIBE=1` is the explicit unsafe override; unattended collection runs
should remain sequential.

Every command accepts `--format json` before the command name for machine-readable output:

```bash
n64dk --format json sdk baserom.z64
```

For a persistent source-matching campaign, configure a project adapter and run:

```bash
n64dk --format json match-job-run path/to/PROJECT.toml \
  --unit UNIT --state-root .n64dk/match-jobs --budget 500 --verify
```

The job builds candidates in temporary workspaces, records failures and lineage, compares
all adapter-owned sections, and never edits project source. See
[Autonomous matching jobs](docs/autonomous-matching.md) for adapter requirements, resume,
export, and proof boundaries.

## Commands

| Command | What it does |
|---|---|
| `decompile-init ROM --out PROJECT` | Build a logical-image project, catalogue, symbols, compiler lanes and byte ledger from a bare ROM |
| `decompile-run --project PROJECT --all-images --out DIR` | Run the checkpointed donor/draft/recipe/search/strict-verification pipeline with bounded resources |
| `discovery-audit --project PROJECT [--repair]` | Audit catalogue completeness from independent call, branch, pointer, prologue and image-placement evidence; optionally write repaired catalogues outside the project |
| `toolchain-regions --project PROJECT --out DIR` | Reconstruct ranked compiler/assembler regions, component mismatches and open-set unknown-toolchain dossiers |
| `decompile-run ... --effort quick\|normal\|deep\|exhaustive`, `--resume DIR`; `decompile-status DIR` | Effort modes, cause-driven later rounds to a fixed point under time/build/model-call budgets, resume, read-only round table and per-function hard-queue dossiers (default `normal` is the single pass above) |
| `decompile-batch ROM... --out DIR` | Process titles sequentially and share only independently verified donor results |
| `decompile-everything ROM... --out DIR` | Everything for a set of ROMs: init, run at the kitchen-sink effort (`exhaustive`), library sharing, export and status per title, one title at a time, resumable |
| `decompile-run ... --small-data auto\|off` | Small data (`-G`): the program's `_gp` from the code that sets `$gp`, GP-relative relocations, and the small-data assembler lanes `kmc+asg`/`kmc+gas` for functions that address through `$gp` (default `auto`; titles that never set `$gp` are unchanged) |
| `discovery-audit ... --overlays` | Also infer per-overlay bases inside segments that hold several separately linked overlays (a report; never applied to a catalogue) |
| `decompile-export --project PROJECT --run DIR --out DIR` | Export every function with honest C/assembly/unresolved provenance and full ROM-byte coverage |
| `source-donor TREE --project PROJECT` | Measure an existing C source tree and split verified translation units into reusable per-function donors |
| `sdk-donor TREE --out DIR` | Build version/flavour-specific libultra donor descriptions |
| `compilers` | Audit or fetch the checksum-pinned catalogue of known retail N64 compilers and usable lanes |
| `compiler-components` / `compiler-probes` | Toolchain components and recipes with provenance and digests; probe every recipe for determinism, equivalence and stage attribution |
| `dev-records-scan` / `dev-records-decode` | Find and decode SN/PsyQ object records and SN debug streams left in a ROM, fail-closed, as evidence |
| `training-corpus-build` / `-audit` / `-stats` / `-explore` / `-evaluate` / `-export` | Exact draft/source pairs from imported trees; held-out evaluation of drafting conditions and learned rules |
| `jobs` | List heavy jobs on the machine, cancel one with its process group, prune stale leases |
| `info ROM` | Header, CIC, checksum, main segment, stack, BSS, hand-over, staged boot |
| `disasm ROM VRAM SIZE` | Disassemble a range of the main segment |
| `survey ROM --out DIR [--image auto\|static\|emulated] [--require-complete]` | Everything below over the program image; report and ROM ledger |
| `boot-emu ROM [--max-steps N] [--after N]` | Emulate the entry code; DMAs, handoffs, thread dispatches, RAM segments with ROM sources |
| `functions ROM [--out DIR] [--truth FILE] [--audit FILE]` | Function catalogue, jump tables, `-O0`/hand-written detection; precision and recall against a reference |
| `datamap ROM [--matched TSV] [--symbols FILE] [--emit-units DIR] [--ledger TSV]` | Data cross-references, section split, ownership, initialized-data ledger, typed data units |
| `sdk ROM [--truth FILE]` | Name SDK library functions; SDK version, microcode, audio library, compiler verdict |
| `sdk-build-db RECIPE.toml [--out PATH]` | Rebuild the signature database from library sources and matched objects |
| `audio ROM [--out DIR] [--wav] [--midi]` | Audio inventory with byte-exact intervals; WAV and sequence export |
| `gfx ROM [--range A:B] [--ucode F] [--mode blobs\|scan\|walk] [--decompress] [--png --out DIR]` | Find and walk display lists; vertices, textures, PNG export |
| `assets-scan ROM [--out DIR] [--extract] [--verify]` | Compressed streams (with round-trip checks) and loader-referenced ROM ranges |
| `match SRC --rom ROM [--addr VRAM] [--sweep]` | Build one source and compare it relocated against the ROM; sweep pipelines and flags |
| `match-batch MANIFEST... --project DIR` | Rebuild and verify every unit of matching manifests in parallel |
| `match-permute SRC --rom ROM --addr VRAM` | Permutation search scored by the relocated diff |
| `match-iterate SRC --rom ROM --addr VRAM --size N` | Run one preflighted, session-persistent beam search over types, signedness, safe lifetimes, declaration order and bounded expression forms |
| `match-session-start SRC --rom ROM ...` | Start a persistent standalone source-matching session |
| `match-session-try SESSION_ID` | Apply in-memory edits and compile a batch of source/flag variants |
| `match-session-status SESSION_ID` | Show the best result and nondominated residual frontier |
| `match-session-export SESSION_ID OUTPUT` | Export a retained candidate only when explicitly requested |
| `match-inventory --project DIR [--verify]` | Reconcile catalogues and manifest intervals; optionally rebuild and cache strict current-source receipts |
| `context-bundle --project DIR --image IMAGE --function NAME` | Collect target code, callers, callees, globals and access patterns |
| `quickwin-sweep --project DIR --all-images --limit N --jobs N` | Create/resume a persistent ranked cross-function tranche |
| `quickwin-sweep-status ID [--status active]` | Filter tranche rows and inspect current versus historical best results |
| `tranche-run --project DIR --all-images --limit N --jobs N` | Run cached preflight, per-function recipes, durable matching, bounded exploration and winner queuing |
| `source-scaffold FUNCTION --image FILE` | Emit compilable ordinary C and a separate evidence receipt for a source-less interval |
| `m2c-draft IMAGE --addr A --size N --vram-delta D` | Emit symbolized assembly and, when m2c is installed, a first-draft C function without modifying project source |
| `draft-sweep --rom IMAGE --catalogue FILE --out DIR` | Preflight source-less intervals, lift them with m2c, apply bounded compile repairs and rank exact/near drafts |
| `fragment-audit IMAGE --catalogue FILE --vram-delta D` | Identify probable interior catalogue starts and produce advisory merge proposals |
| `asm-score TARGET CANDIDATE` | Compute asm-differ/decomp-permuter-style penalties alongside byte-exact matching evidence |
| `match-hints REPORT.json` | Convert aligned residual instructions into compiler-aware source hypotheses |
| `similar-queue --set NAME,IMAGE,DELTA,CATALOGUE[,MANIFEST...]` | Rank unmatched functions by reusable matched siblings and estimated difficulty |
| `structure-layout-infer` / `prototype-solve` / `tu-reconstruct` | Aggregate shared types, solve ABI declarations and prepare whole-unit reconstruction |
| `match-session-explore ID --rounds N --jobs N` | Run bounded residual-guided source experiments over the frontier |
| `match-session-explain ID --from A --to B` | Explain residual and source changes between two candidates |
| `investigate FUNCTION --image IMAGE --all` | Build one verified dossier from layout, code, source, recipes, history and corpus evidence |
| `image-map-explain --project DIR --image NAME --function NAME` | Explain raw/composite/catalogue/linked mappings, byte digests, transforms and ownership collisions |
| `corpus-index PROJECT...` | Incrementally index local N64 assembly by exact and normalized signatures |
| `corpus-query --image IMAGE --addr A --size N --vram-delta D` | Find exact, relocation-only, register-renamed and opcode-shape analogues |
| `corpus-prototype SYMBOL` | Infer prototypes from indexed declarations, call arguments, return use and width evidence |
| `result-index --roots PATH...` / `result-query` | Mine historical JSON, logs, sessions and candidates while invalidating stale targets |
| `reproduce RESULT_ID` | Independently replay a digest-pinned source, toolchain, target and environment bundle |
| `residual-clusters --max-score N` | Group typed, address-independent residuals across related functions |
| `script-index PROJECT...` / `script-query` | Inventory scattered sweeps, audits, transforms, scorers and build wrappers |
| `operation-schema` / `automation-serve` | Expose typed operations over newline-delimited JSON for local controllers |
| `doctor --profile matching` | Validate ROM, host tools, original toolchains, paths, storage and optional research capabilities |
| `compiler-capabilities` | Inspect or selectively invalidate per-toolchain, stage and option capability evidence |
| `prototype-matrix` / `register-lifetime-explore` / `match-matrix` | Run bounded ABI, call-use, narrowing, value-lifetime and declarative experiment families |
| `control-flow-explore` / `clone-family-transfer` | Classify branch targets, test loop-exit forms and transfer sibling CFGs |
| `family-session-start` / `tu-session-start` | Preserve cross-function and real translation-unit experiment context |
| `frontier-synthesize` / `compiler-explain` / `residual-attribution` | Synthesize complementary candidates and explain one candidate or a causal pairwise compiler/residual difference |
| `code-query` / `asm-query` / `negatives-query` | Search source, manifests, historical failures and structural assembly without shell pipelines |
| `boundary-audit` / `boundary-correct` | Audit CFG/Splat/ownership evidence and prepare denominator-safe corrections |
| `entry-audit` / `abi-solve` / `structure-context` | Produce CFG/liveness/frame, prototype, and cross-function structure-layout evidence |
| `manifest-verify` | Rebuild manifest units and issue strict `exact`, `assumed`, or `mismatch` verdicts |
| `resolve-case` / `resolve-hard` / `hard-queue` | Persist digest-pinned hard cases and run the bounded boundary/draft/recipe/diagnose/iterate/proof ladder |
| `ownership-ledger` / `progress-reconcile` | Partition authoritative byte ownership and explain numerator/denominator changes without granting unverified credit |
| `exact-reconcile` / `workspace-audit` / `workspace-clean` | Find reproducible unintegrated winners and audit or remove only known generated artifacts |
| `address-explain` / `gate-policy` | Explain logical/catalogue/linked coordinates and inspect fail-closed transition-gate policy |
| `integrate-winner` / `integration-audit` / `gate-delta` | Verify exact winners twice and prepare reviewable integration/gate transactions |
| `project-gate` | Run project-defined gates with compact image and failure summaries |
| `scheduler-plan` / `scheduler-status` | Persist separate discovery, exploitation, recovery, boundary, family and verification lanes |
| `workbench-snapshot` / `self-test` | Produce one-call function context and run canonical test discovery |
| `match-session-prepare-winner SESSION_ID DIR` | Independently recompile an exact winner and prepare an auditable export queue |
| `match-job-run PROJECT.toml --unit UNIT` | Create or resume a bounded, persistent candidate/compile/diagnose search job |
| `match-job-status JOB_ID` | Inspect job state, best result, failures and candidate lineage |
| `match-job-export JOB_ID OUTPUT` | Export a candidate and non-proof-grade evidence without editing project source |
| `match-diagnose COMPILED TARGET` | Rank boundary/toolchain/semantic/type/regalloc/reload/scheduling causes |
| `gcc27-regalloc FILE.lreg [--global-dump FILE.greg]` | Explain GCC 2.7.x pseudo priorities, ties and hard-register assignments |
| `toolchain-fingerprint --profile P --file ROLE=PATH...` | Content-hash the real compiler/assembler/linker identity |
| `provenance-audit SOURCE...` | Reject assembly escapes, fake code (the [match] code rules), unproven assembly and non-proof-grade credit |
| `code-rules SOURCE... [--project DIR] [--strict] [--declarations]` | Check sources against the code rules: [match] findings (`MATCH-PENDING`), exceptions, `ORIGINAL-ASM`, and advisory open work |
| `research-audit SOURCES.tsv FINDINGS.tsv` | Reject unindexed, search-only or low-quality evidence before automation |

`match-iterate` never edits its input. It validates the project image and strict function entry,
compiles variants concurrently, deduplicates equivalent objects, and stores the complete run under
`.n64dk/match-iterate/` plus the standard session registry. Type, cast, field-signedness and
variable-splitting edits are explicitly marked as requiring semantic review; pure lifetime moves
and declaration reordering use conservative dependency guards. A compact invocation is:

```sh
n64dk --format json match-iterate candidate.c --project . --rom overlay_composite.z64 \
  --addr 0x80000000 --size 0x100 --jobs 16 --budget 400 --summary-only --out best.c
```

Use `--families lifetime,order,expression` to restrict the run to behavior-preserving forms.
`m2c-draft` and preflighted `draft-sweep` reject explicit raw/composite deltas that disagree with
the authoritative project mapping.

Exploratory `compare_object` exactness is intentionally permissive and is not proof. Credit,
winner queues, inventory numerators, and integration require a current `strict_compare` verdict
of `exact`; `assumed` results remain research evidence only. See
[Round 4 integration and adversarial verification](docs/round4-integration.md), the durable
[goal and invariants](docs/durable-goal.md), and the paste-ready
[Madden operator handoff](docs/MADDEN_ROUND4_HANDOFF.md).

Region-recipe retry, nearest-unit recipe retry, and search suppression based on an
unknown-toolchain classifier are retained as experiments but are off by default: their measured
returns were 3/126, 0/165, and not sufficiently validated, respectively. Enable them explicitly
with `--region-recipes`, `--unit-recipes`, or `--skip-unknown-search`; ordinary runs still use
manifest-proven per-unit recipes and report unknown-toolchain evidence without treating it as fact.

Ordinary runs do enable two measured optimizations. The recipe gate avoids a broad recipe retry
when neither the function nor its neighbours show unoptimised-code evidence; use
`--no-recipe-gate` for the previous exhaustive retry. When m2c is installed, an isolated fork
server amortizes Python import cost while retaining a fresh child per draft; set
`N64DK_M2C_SERVER=0` to use one command-line process per draft.

Run `n64dk COMMAND --help` for every option.

### Round 12 closeout: what ships in decompile-run

`decompile-run` runs these steps for each image, in this order. Each line gives the switch and the default.

| Step | What it does | Switch | Default |
|---|---|---|---|
| Machine guard | One heavy n64dk job at a time; refuses under the memory and disk floors | `--wait-for-lock`, `--force-resources` | refuse when busy; floors enforced |
| Blind run | Refuses (exit 3) before any stage when a forbidden tree is reachable from the configuration, project or environment; checks again at the end; refuses a run directory that held non-blind state; receipts in `solver/blind_*.json` | `--blind-forbid TREE` (repeatable), `--blind-walk-limit N`, `--blind-shared-library MANIFEST@sha256:HEX` | off; walk limit 50,000 (a walk that stops short refuses the run, so large toolchain trees need a higher limit) |
| Run modes | `blind`: the title's corpus trees forbidden, self-donors and other revisions refused, title-defined names refused in the project's metadata; `research`: only the title's trees forbidden; `source-tree`: allowed and recorded. Recorded in `solver/run_mode.json`; a run directory is reused only by a mode that ranks at or above every earlier session | `--mode blind\|research\|source-tree`, `--corpus-manifest`, `--title` | off (no mode) |
| Run-directory provenance | Every run records in `solver/blind_provenance.json` whether it was blind | — | always |
| Discovery gate | Audits each catalogue against call, pointer and prologue evidence; repair adds evidence-backed functions to a copy in the run directory, never the project | `--discovery repair\|audit\|off`, `--skip-stage discovery` | `repair` |
| Small data (`-G`) | Recovers `_gp` from the code that sets `$gp`; writes `small_data/gp_symbols.txt`; offers the `kmc+asg` and `kmc+gas` lanes to functions that address through `$gp`; reports a disagreeing project `_gp` (the project's value is used) | `--small-data auto\|off` | `auto` (titles that never set `$gp` are unchanged) |
| Boundaries | Boundary operations report (`boundary_operations.json`); applies nothing | `--skip-stage boundaries` | on |
| Donors | Transplants verified donor units from source trees, SDK donors and a library of earlier runs. An address that only the unit's own retail bytes support is never declared: the unit waits for an independent declaration | `--donors TOML`, `--library DIR`, `--donor`, `--exclude-donor`, `--skip-stage donors`, `--skip-stage rodata-settlement` | runs only when donors or a library are given |
| Calibration | Measures the title's recipe on a sample of its functions; when none is asked for and the code's marks contradict the configured recipe on ≥ 40% of the image, samples 96 | `--calibrate N`, `--skip-stage calibrate`, `--regional-calibration` / `--no-regional-calibration` | `[run] calibrate`, else 0 (recipe doubt still applies); regional off except `--effort exhaustive` |
| Resolve | Per function: m2c drafts with context, the draft portfolio, recipe selection, drafts × recipes, frame fit, twins, hand-written assembly, variant search, strict verification | `--budget`, `--search-ceiling`, `--no-recipe-gate`, `--region-recipes`, `--unit-recipes`, `--search-economics`, `--skip-unknown-search`, `--max-size`, `--limit`, `--only-names` | budget 200, search ceiling 40, recipe gate on, region/unit recipes off, economics off |
| Assembler lanes | `kmc+atload` when the function loads through `$at` (evidence-gated); `kmc+atmacro` for `$at` branch comparisons | `--skip-stage at-loads`; `N64DK_AT_MACROS=1` | atload on; atmacro off |
| Variant search | Stops at once from a perfect start (no variant could be adopted); the e-graph family is opt-in | `--skip-stage search`, `N64DK_EGRAPH=1` | on; e-graph off |
| Declare | Declares self-describing data addresses that byte-identical cases wait for, and verifies those cases again | `--declare-assumed`, `--skip-stage declare` | `[run] declare_assumed`, else off |
| Later rounds | Fixed-point rounds of search, recipe sweep, declare and refine | `--effort quick\|normal\|deep\|exhaustive`, `--rounds`, `--schedule`, `--budget-time`, `--budget-builds`, `--budget-model-calls` | `normal`: one round |
| Refiner | An external author for unresolved cases | `--refine-command`, `--skip-stage refine` / `round-refine` | none configured: skipped |
| Strict queue, toolchain regions, dossiers | Exports strictly verified cases; per-region recipe evidence; a primary cause per unresolved case | `--strict-only` | on |
| Data settlement | One ownership model over ROM, image and runtime spaces (`.bss`/`.sbss` own no ROM bytes); declared and inferred `_gp` kept apart; waiting data references grouped and placed from independent evidence only; `replay` re-verifies the cases whose every need is established | `--data-stage off\|report\|replay` | off |
| Unresolved dossier | For every unsettled function: reason, eligible next techniques, measured cost (or "unmeasured"), the evidence that would justify a retry, and a technique ledger | — | always (report-only); `UNRESOLVED.md` in the export |
| Byte ledger | `ownership.tsv` from the project's manifests over the catalogue | `--skip-stage ledger` | on |
| Library registration | Registers this run's strictly exact functions as donors for other titles | `--library DIR` | off |
| Proof receipts | A replayable receipt per settled result: generated C, assembly and donor units. Replay binds the offered recipe's pins and section bases; a self-derived pin origin never makes a strict proof | `--receipts` | off (or run `issue-receipts` later) |
| Ablation | 20 subsystems can be switched off for a measurement; the run records what it disabled in `report.json` `ablation`, and refuses a filled case store or a resume under other switches | `--skip-stage NAME` or `N64DK_DISABLE_<NAME>=1` | everything on |
| Resume | Continues from the case store and `rounds.jsonl`; refuses to add `--blind-forbid` or change switches | `--resume RUN_DIR` | — |
| Resources | Worker processes and parallel builds | `--workers`, `--jobs`, `--worker-memory-mb`, `--memory-reserve-mb` | automatic; at most 4 builds |

Custom lanes always build with `TMPDIR` set to their private build directory. Opt-in experiments pending their
measurements: `N64DK_LAYOUT_INFER=1` (array and struct-field declarations for optimised drafts) and
`N64DK_DECLARE_MEMO=1` (skip re-verifying unchanged assumed cases in later declare passes).

Not part of `decompile-run` (separate commands or libraries, none grants credit): reason codes (`decompile-export`),
`verify-run` / `verify-export`, `relink-image` and `relink-cartridge` (link one image or the whole ROM from a
generated project and compare it byte for byte, keeping linked, repacked and preserved bytes apart), `decompile-project`, `link-layout`, `discovery-audit --overlays` and
`--indirect`, the behaviour oracle, translation validation, the similarity index, the data-object solver
(`datasolve`), RSP/IPL3 identification (`rspid`), the RSP disassembler/assembler (`rspasm`), codec-boundary classification
(`codecboundary`), calibration, the dashboard and the review commands.

## Checking, reviewing and packaging results

These commands never grant match credit; strict byte identity stays the only credit.

- `issue-receipts RUN`, `verify-receipt`, `verify-run RUN`, `verify-export EXPORT`: proof receipts.
  Each settled result is rebuilt in a fresh directory and verified strictly. The receipt records
  the source, recipe, toolchain, declarations and comparison digests. Replaying a receipt rebuilds
  and compares again. A tampered receipt is rejected.
- `decompile-dashboard RUN --out DIR`: a self-contained, read-only HTML page of a run. It holds no
  ROM bytes and no machine paths.
- `review-queue`, `review-show`, `review-decide`: review assumed, near and donor items. Accepting
  an item as exact requires `--verify`.
- `link-layout` and `decompile-project`: a hypothesis of the original link order, and a
  maintainable project generated from an export. Generating twice gives identical output.
- `blind-check`: checks, at the moment it runs, that no forbidden source tree is reachable from a run's configuration, project or environment (paths, symlinks, byte copies); it cannot see a path computed inside a process, a network fetch or knowledge copied by hand. For enforcement during a run use `decompile-run --blind-forbid` or `--mode blind`.
- `migrate --check`, `release-info`: schema and version compatibility of runs, stores and caches.
- `doctor --host`: what is installed and what is missing.

Worked, executable examples are in [docs/workflows.md](docs/workflows.md). Every option of every
command is in [docs/cli-reference.md](docs/cli-reference.md), which is generated from the parser
and checked by the test suite.

## Library use

```python
from n64dk import image, survey
from n64dk.analysis import functions
from n64dk.rom import Rom

rom = Rom.load("baserom.z64")
program = image.build(rom)                  # emulates the boot only when a loader runs
catalogue = functions.discover(rom, program=program)
report = survey.run(rom)["report"]
```

## Documentation

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): module map and conventions
* [docs/workbench.md](docs/workbench.md): ownership, bounded exploration, search, integration and scheduling
* [docs/boot.md](docs/boot.md): entry-routine analysis and boot emulation
* [docs/survey.md](docs/survey.md): program image, survey and ROM ledger
* [docs/functions.md](docs/functions.md), [docs/datamap.md](docs/datamap.md),
  [docs/sdk.md](docs/sdk.md): code and data analysis
* [docs/audio.md](docs/audio.md), [docs/graphics.md](docs/graphics.md),
  [docs/compression.md](docs/compression.md): assets
* [docs/matching.md](docs/matching.md): toolchains, pipelines and matching
* [docs/code-rules.md](docs/code-rules.md): what counts as matched: the [match] rules, the
  checker, the project profile and how credit, search and export apply them
* [docs/autonomous-matching.md](docs/autonomous-matching.md): persistent matching jobs,
  usage, evidence, verification boundary and limitations
* [docs/autonomous-tranches.md](docs/autonomous-tranches.md): interval inventories, context
  bundles, cross-function scheduling, compact session output and exact-winner preparation
* [docs/safety.md](docs/safety.md): machine lock, worker tokens, memory and disk floors, leases
* [docs/toolchain-components.md](docs/toolchain-components.md): component graph, recipes, probes
* [docs/dev-records.md](docs/dev-records.md): development records left in ROMs
* [docs/decompile-run.md](docs/decompile-run.md): bare-ROM initialization, unattended stages,
  resource limits, donors, refinement, batch operation and export semantics
* [docs/solver.md](docs/solver.md): effort modes, primary causes and their techniques, the
  fixed-point rule, budgets, resume, `decompile-status`, what is observational and what is not
* [docs/claude-integration-audit.md](docs/claude-integration-audit.md): recovered-lab integration
  ledger, accepted/rejected experiments, verification gates and remaining external limitations
* [docs/tooling-research.md](docs/tooling-research.md): upstream workflow concepts and the
  first-class operations that replace repeated searches, extraction and comparison calls
* [docs/AUTOMATION_HANDOFF.md](docs/AUTOMATION_HANDOFF.md): operating and reporting
  instructions for automated matching workers
* [docs/BRIEF.md](docs/BRIEF.md): rules for adding an analysis module

Each document ends with measured results on the fixture ROMs and known limitations.

## Matching toolchains

The original compilers are not included. `match` looks them up from `--toolchain-dir`,
`[toolchains]` in `n64dk.toml`, the `N64DK_TOOLCHAIN` environment variable,
`<project>/.toolchain` or `./.toolchain`; see [docs/matching.md](docs/matching.md) for the
expected layout. SN GCC additionally needs `wibo`.

Original compiler binaries, SDK files, and ROM images are not distributed by this project.
Users must provide legally obtained copies. Generated job state and local toolchain paths are
ignored by Git.

## Project adapters

Persistent campaigns use a title-specific adapter so the core never hard-codes a game's
layout or build system. A `PROJECT.toml` selects a Python factory:

```toml
[project]
adapter = "project_adapter:create"
python_path = "."
```

The factory implements `n64dk.project.ProjectAdapter`: unit discovery, immutable input and
toolchain files, owned target spans, workspace preparation, the authentic build command, and
built-span extraction. Candidate source must be staged at its project-relative path inside
the temporary workspace. See [Autonomous matching jobs](docs/autonomous-matching.md) for the
complete execution contract.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -t .
```

Evidence-driven matching commands include `match-diagnose`, `gcc27-regalloc`,
`toolchain-fingerprint`, `provenance-audit`, `research-audit`,
`workbench-diagnose`, `transform-list`, and `transform-apply`.  Transformations
are applied only to a separate candidate file from a digest-pinned JSON proposal;
the command also writes a mechanical inverse proposal.

Tests that need a ROM skip when it is not configured. To enable them, create `roms.toml` in
the repository root (it is ignored by git):

```toml
[roms]
battletanx = "/path/to/battletanx.z64"
mk64 = "/path/to/mk64.z64"

[truth]                     # optional ground truth from decompilation projects
battletanx = {path = "/path/to/battletanx-project", rev = "2e6bc1c"}   # pinned to a commit
battletanx_ga = "/path/to/global-assault-project"                     # live working tree
```

A pinned entry (`rev`) makes the tests read the project's tracked files as of that commit:
`n64dk.fixtures` extracts them once with `git archive` into `build/truth/` and links the
project's git-ignored files (toolchains, generated headers, base ROMs) to the live ones, so
uncommitted edits in the project cannot break the tests. The project itself is only read.
Without `rev`, the live working tree is used. `N64DK_TRUTH_<NAME>` and
`N64DK_TRUTH_<NAME>_REV` override the file.

The known fixture images and their SHA-1 hashes are listed in `n64dk/fixtures.py`; a ROM whose
hash does not match is rejected. `N64DK_ROMS=name=path:name=path` works as well. Set
`N64DK_SLOW=1` to include the longest emulation test.

## Licence

GPL-3.0-or-later (see [COPYING](COPYING)). Parts are adapted from the BattleTanx and
BattleTanx: Global Assault decompilation projects (GPL-3.0); the modules concerned say so in
their docstrings.
