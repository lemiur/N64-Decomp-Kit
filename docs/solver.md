# The fixed-point solver behind `decompile-run --effort`

`decompile-run` makes one pass per image: calibrate, sweep the resolution ladder, settle the
byte-identical cases that wait for a declaration. With an effort mode above `normal`, that pass
becomes round 1 of a loop. Each later round gives every unresolved case one primary cause, picks
a technique that can act on that cause, runs it, and records what happened. The loop stops when
a round adds nothing.

The code is in `n64dk/match/solver.py`. It builds nothing itself: `decompile-run` passes it
callables bound to `resolve-hard` and `declare_assumed`. The tests in `tests/test_solver.py`
drive it with fakes.

```sh
n64dk decompile-run --project P --all-images --out RUN                     # normal: the single pass
n64dk decompile-run --project P --all-images --out RUN --effort deep --budget-time 6h --budget-builds 200000
n64dk decompile-run --resume RUN --budget-time 2h                           # continue, finished rounds kept
n64dk decompile-status RUN                                                  # read-only round table and queue
```

## Effort modes

Each mode is one row of `solver.MODES`. A mode only fills options the command line left unset;
an explicit option such as `--budget 50` always wins. `[run]` switches in `n64dk.toml`
(`region_recipes`, `search_economics`, `declare_assumed`) can still turn a technique on.

| | quick | normal (default) | deep | exhaustive |
|---|---|---|---|---|
| builds per function (`--budget`) | 60 | 200 | 300 | 400 |
| size window (`--max-size`) | 0x200 | none | none | none |
| search ceiling | 20 | `[run]` or 40 | `[run]` or 40 | `[run]` or 40 |
| drafts x recipes when no draft is exact | 4 x 4 | 4 x 8 | 4 x 8 | 4 x 12 |
| region-recipe retry in round 1 (3/126) | off | off | off | on |
| nearest-unit retry (0/165) | off | off | off | off |
| search economics (-2.7% builds, no outcome change) | on | off | off | off |
| refiner, when one is configured | never | round 1 | round 1 and later rounds | round 1 and later rounds |
| rounds | 1 | 1 | up to 4 | up to 8 |
| scheduling | observe | observe | ranking orders work | ranking orders work |
| later-round techniques | (predicted only) | (predicted only) | search, recipe-sweep, refine | + draft-options |
| cascade after each round | none | none | declare, neighbours, recipes | declare, neighbours, recipes |
| every function not attempted has a specific reason | recorded | recorded | recorded | required; a gap is a report warning |

**`normal` is today's `decompile-run`.** `solver.LEGACY_DEFAULTS` holds the option defaults the
command had before modes existed. `normal` fills exactly those values. In normal mode the stage
plan is the old one (`boundaries`, `donors`, `calibrate`, `resolve`, `declare`, `queue`, `ledger`,
each under its old condition), and nothing runs after round 1. The tests check three things:

- the policy's fields equal `LEGACY_DEFAULTS`;
- the stage plan for parsed default arguments equals the old plan;
- the full `resolve-hard` option namespace built for the default equals the one built from the
  old literal defaults.

An end-to-end test runs `decompile-run` against a fake `resolve-hard` and checks that it is
called once, with the old options.

The new outputs (`rounds.jsonl`, `solver/`) are read-only observations. They build nothing.

**Bounds.** Every mode keeps a per-function budget. A technique varies one thing at a time: more
recipes for the best drafts, or more search on the best source. No mode forms a product of
recipes x flags x transforms. `check_policy` refuses any mode whose drafts x recipes product
exceeds 128.

**Functions over 0x400 bytes.** In every mode, including exhaustive, these are drafted and built
under their recipe only. Attempting more on them measured 0 exact in 964 builds. Exhaustive does
not hide this limit. Each such function's coverage entry says what was withheld and why.

## Primary causes and their techniques

`primary_cause(case)` maps each unresolved case to one cause from the review's table. It uses the
case's status, its top residual cause (`causes.TAXONOMY`), its score against the search ceiling,
its size, and whether it lies in an unknown-toolchain stretch reported by `resolve-hard`. The
first rule that matches wins.

| primary cause | from | techniques (only these) | when none applies |
|---|---|---|---|
| no discovered image | coverage: the range is outside every image | none | the survey has to place it first |
| uncertain boundary | `blocked-boundary`, `boundary.*`, `abi.nonstandard`, `tu.context` | none | boundary operations are proposed, never applied unattended |
| handwritten | `probable-handwritten`, `assembly`, `assembly.handwritten` | none | kept as assembly when it reassembles; never C |
| no buildable draft | `needs-source`, `does-not-build`, no score | refine, draft-options, recipe-sweep | |
| link/address mismatch | `assumed`, `relocation.target` | declare (cascade), search | only `D_<address>` names are declared unattended |
| unknown toolchain | in an unknown-toolchain stretch | none | identifying an unknown compiler does not reproduce it |
| large far-off | score above the ceiling, or larger than 0x400 | recipe-sweep, refine (not above 0x400) | |
| recipe uncertain | `compiler.family`, `assembler.variant`, `schedule.load_hazard` | recipe-sweep, search | |
| wrong load width | `type.width`, `type.signedness` | search, refine | |
| global offsets differ | `global.declaration`, `type.volatile` | search, refine | |
| register-only residual | `register.*` | search, refine | |
| branch topology | `control.*`, `schedule.delay_slot`, `semantic.unknown` | search, refine | |
| wrong stack frame | `frame.size` | search, refine | |

Each technique is a `resolve-hard` call over a named set of stored cases (`technique_options`):

- **search**: continue the variant search from the case's best source. Families already closed
  for that source are skipped. Not applicable above the search ceiling, above 0x400, or when every
  family is closed.
- **recipe-sweep**: draft again (`fresh`) and try the best drafts under `recipe_sweep` alternative
  recipes. The stored case stands if the new attempt ends further away.
- **draft-options**: draft again with alternative m2c renderings (`--no-andor,--gotos-only`). This
  measured 6/639, so it is offered only in exhaustive mode.
- **refine**: ask the configured refiner only (`skip_sweep`). Every answer goes through the
  ladder and the strict verifier. A case already asked is not asked again.
- **declare**: `declare_assumed`. It declares only names that state their own address, then the
  strict verifier decides.

In every technique other than refine, the refiner is switched off.

## Ranking and negative facts

`Observations` (`RUN/solver/observations.json`) keeps one cell per (cause, recipe family, size
bucket, technique): attempts, exact results, improvements, builds and model calls.

**P(exact).** Each cell has a prior worth at most 10 observations. Measured priors are used where
they exist:

- register-only residuals with search: 17/127;
- alternative renderings: 6/639;
- far-off recipe attempts: 0/630.

Elsewhere the prior is a stated assumption. The prior is then updated with this run's own
observations: those for the cause in general, plus those for the exact cell, which count twice.

**Expected cost.** This run's mean builds per attempt of the technique. Before the first attempt,
a mode default is used. One model call is weighed as 10 builds; that is a convention, not a
measurement.

**Ranking.** Techniques are ranked by P(exact) / expected cost.

**Negative facts:**

- **No-change.** The technique ran on this case's current best source and changed nothing. It is
  not repeated until the case gets new evidence (a caller or callee became exact) or a new best
  source.
- **Plateau.** The technique has had 12 or more attempts on a cause this run without a single
  gain. It is set aside for that cause, but only when the ranking orders work.

**What is observational and what is active:**

- In `observe` mode (quick and normal by default), every decision is still computed and written:
  the next action and its P(exact) in each dossier, and in `decompile-status`. Batches run in the
  technique table's order and cases by size. Plateaus are recorded, not acted on.
- In `order` mode (deep and exhaustive), the ranking orders batches and the cases inside them.
  Because budgets cut a round short, order decides what is reached first.
- Ranking never decides credit. A case becomes exact only through the strict verifier inside
  `resolve`. A test runs the same fake title with opposite rankings and gets the same exact set
  and the same credit.
- Per-case no-change facts are active in every mode. Repeating an identical attempt on the same
  source would only spend builds.

## The fixed-point loop and convergence

For each image (`solve_image`):

1. **Round 1** is the sweep: today's calibrate, resolve and configured declare, unchanged.
2. **Later rounds** plan one technique per unresolved case. Batches are grouped by technique and
   run while the budgets allow.
3. **Cascade** (deep and exhaustive), after every round:
   - **declare**: settlement of assumed cases that have self-describing addresses and have not
     been tried on their current source;
   - **neighbours**: every unresolved caller or callee of a function that became exact this round
     gets new evidence, which lifts its no-change facts;
   - **recipes**: the recipe confidence update, a tally of the recipes that produced exact
     functions, recorded with the round and in the dossiers.
4. **Stop** when the first of these holds:
   - **fixed point**: a round after the first adds no new exact function, no new declaration, no
     new evidence (status, score or top cause changed, or a neighbour mark) and no new negative
     fact. A no-change attempt is itself new knowledge, so the loop goes on until every applicable
     technique has been tried on every case's current source. It then stops with "no unresolved
     case has an applicable technique left";
   - **round limit** of the mode (`--rounds` overrides it);
   - **budget**: `--budget-time` (seconds, or `30m`, `2h`), `--budget-builds` (all images
     together), `--budget-model-calls` (refiner calls). A budget is checked before each batch,
     and a batch that starts runs to its end, so one batch is the most a budget is overrun by.
     A build budget also cuts a batch down to what is expected to fit. Round 1 can only be
     stopped before it starts. A model-call budget also caps the refiner's `max_cases`.

Each completed round is one line of `RUN/rounds.jsonl`. A line holds:

- the session, image, round and mode;
- the sieve (`sweep`, or the techniques run);
- per technique: cases, builds, model calls, exact results, no-change results and the predicted
  exact count;
- the cascade results and any batches skipped with their reason;
- gains by category: new exact C, change in awaiting-declaration and hand-written-assembly counts,
  new declarations, evidence and negative facts;
- totals and budget use.

Each stop is a `stop` line with its reason.

**How builds are counted.** Builds are measured from the stored cases before and after each
batch. A rerun that ended further away keeps the old case and adds to its build count. One that
ended as near or nearer replaces it, and its count starts at that attempt. `resolve-hard`'s twin
pass is not counted.

## Credit stays strict

The categories are kept apart in the rounds, in the report and in `decompile-status`:

- exact C (status `exact` with a strict `exact` verdict);
- byte-identical awaiting declarations (`assumed`);
- hand-written code kept as assembly (`assembly`);
- unresolved.

Generated assembly, data, opaque and unknown bytes are not case-store categories. They stay with
`decompile-export` (`reproduced_by`) and the ledger, as before.

## Resume and status

`decompile-run --resume RUN` reads `RUN/solver/invocation.json`, the options the run was started
with, and continues the latest session of `rounds.jsonl`:

- a round recorded as complete is not redone;
- round 1's results are reloaded from `RUN/solver/round1_<image>.json`;
- an image that stopped at a fixed point is left alone;
- one that stopped on a budget or the round limit goes on (raise `--rounds` for the latter).

On resume, budgets, `--rounds`, `--schedule`, worker and memory options and lock options come
from the new command line. Everything else comes from the saved invocation. Budgets apply to each
invocation.

`--effort` given on resume switches the mode for the rounds still to come. The new mode's rounds,
scheduling, techniques and cascade then apply. The options the run started with (budget per
function, size window, recipes) stay as they were. This continues a finished `normal` run as
`deep` from the same round 1, which is the comparison to measure.

Two things are redone after an interruption:

- an interrupted round is run again, and the cases it already settled return at once;
- in-memory state of a round is not kept between invocations. Concretely, the new-declaration
  count of the first cascade after a resume counts every declared name as new.

A plain re-run without `--resume` starts a new session, as `decompile-run` always did.

`decompile-status RUN [--image NAME] [--format json]` prints:

- the round table (`Round N [image]: search +k exact (...)`);
- the stops;
- the credit categories;
- the remaining queue grouped by primary cause, with the next action or the stop reason, and
  example functions;
- when the run has a `report.json` (written when an invocation finishes, so it describes the last
  finished one): the discovery verdict and its denominator warning, the `small_data` entry (`_gp`
  value, confidence, reason, the project's own `_gp` and any disagreement) and per-image totals
  (functions, resolution outcomes, exact bytes, strict queue, ledger byte classes, coverage).
  `--format json` adds the keys `report`, `discovery`, `small_data` and `image_totals` (null or
  empty without a report); the earlier keys are unchanged.

It opens the case store with `CaseStore(readonly=True)`: immutable when no write-ahead log is
present, `mode=ro` otherwise. It never writes. A test checks that no file, including `-wal` and
`-shm`, is created or touched.

## Outputs

| file | content |
|---|---|
| `RUN/rounds.jsonl` | one line per completed round and per stop |
| `RUN/solver/invocation.json` | the options of the invocation that started the run |
| `RUN/solver/observations.json` | ranking cells, per-case facts (technique, round, outcome, builds, prediction), evidence marks, plateaus |
| `RUN/solver/round1_<image>.json` | round 1's calibrate/resolve/declare results (for resume) |
| `RUN/solver/coverage_<image>.json` | every catalogued function: its case and status, or the reason it was not attempted |
| `RUN/solver/hard_queue_<image>.json` | the dossier of every function still unresolved |

`coverage_<image>.json` accounts for every function, including the ones `resolve-hard` drops
without a record: functions covered by a project manifest, recovered by a donor, outside the
image's bytes, or below the minimum size. Its other sources are the scope file (`--max-size`,
`--limit`, `--only-names`) and the over-0x400 limit. Anything no rule explains is marked
`unexplained`; exhaustive mode turns a non-zero count into a report warning.

Each dossier in `hard_queue_<image>.json` holds:

- image, vram, size and status;
- boundary status, from the case and from `boundary_operations.json`;
- best recipe, other recipes tried, and the image's confirmed recipes;
- best score and residual causes;
- primary cause;
- every ladder stage with its builds and outcome, and every solver attempt with round, outcome,
  builds, calls and prediction;
- build counts;
- related exact functions (callers or callees, catalogue neighbours);
- next action with its ranking, the excluded actions and why, and why automation stopped.

The `hard-queue` command is kept as it was: it groups by the residual taxonomy (`top_cause`) with
the old tractability table. It is the grouped summary in the report. The dossier is per function
and is written by the run.

## Limitations

- **None of this has been measured on a title.** The priors that are not measured are stated
  assumptions. Whether later rounds pay for their builds is the open question; the measurements
  to run are listed in the round-10 solver report.
- **The neighbours cascade has no new information channel.** It re-opens callers and callees of
  new exact functions for another attempt, but the drafts do not yet read a neighbour's exact
  source or prototype. Prototype propagation measured 0/85, so this step's prior is low, and it
  should be measured before it is kept on.
- **`search` on a case re-runs the ladder from its best source.** That includes a baseline build
  and, for a far draft or the wrong compiler family, the baseline recipe sweep, before the
  unclosed families are searched.
- **"No discovered image"** is only seen through coverage (rows outside the image's bytes). Code
  the survey did not place at all is in `survey/blind.json`, not in any catalogue.
- **Model availability in `decompile-status`** is taken from the saved `--refine-command` only. A
  refiner configured in `[run.refine]` alone is not seen there.
