#!/bin/sh
# tools/smoke_test.sh: a five-minute smoke test of an installed kit.  It changes nothing outside a
# temporary directory it removes, needs no ROM and no project, and every step runs under a time limit.
#
#   python3.11 -m venv /path/to/fresh-venv
#   /path/to/fresh-venv/bin/pip install -e /path/to/kit          # rabbitizer comes with it
#   PYTHON=/path/to/fresh-venv/bin/python sh tools/smoke_test.sh [--toolchain-dir DIR]
#
# Steps (each PASS, FAIL or SKIP; the script exits 1 when any step FAILs, 2 on a usage error):
#   import      the package and its command table import
#   doctor      n64dk doctor --host --out <tmp>/out (FAIL = a required check failed; the JSON is kept
#               in the log and its required failures are printed)
#   synthetic   the synthetic unit tests that need no ROM, toolchain or network: media detection,
#               interoperability formats, the doctor (n64dk/fixtures.py holds no synthetic fixtures:
#               it locates configured ROMs and ground truth, so it is not run here)
#   match       one synthetic function compiled by the first installed KMC GCC 2.7.2 / IDO 5.3 /
#               IDO 7.1 and compared with its known two-instruction code; SKIP without a compiler
#               (its compiler-slot and usage state goes to N64DK_STATE_DIR=<tmp>/state, not the
#               shared state directory)
#
# Settings: PYTHON (default python3), SMOKE_TIMEOUT_STEP seconds per step (default 70; four steps
# stay inside five minutes), N64DK_TOOLCHAIN (or --toolchain-dir) for the match step.
set -u

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || exit 2
PY=${PYTHON:-python3}
STEP_LIMIT=${SMOKE_TIMEOUT_STEP:-70}
TOOLCHAIN=${N64DK_TOOLCHAIN:-}
while [ $# -gt 0 ]; do
    case $1 in
        --toolchain-dir) [ $# -ge 2 ] || { echo "smoke_test: --toolchain-dir needs a directory" >&2; exit 2; }
                         TOOLCHAIN=$2; shift 2 ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "smoke_test: unknown argument $1" >&2; exit 2 ;;
    esac
done
command -v "$PY" >/dev/null 2>&1 || { echo "smoke_test: no Python interpreter ($PY)" >&2; exit 2; }

WORK=$(mktemp -d "${TMPDIR:-/tmp}/n64dk-smoke.XXXXXX") || exit 2
trap 'rm -rf "$WORK"' EXIT
trap 'exit 130' INT TERM
PYTHONDONTWRITEBYTECODE=1
export PYTHONDONTWRITEBYTECODE

if command -v timeout >/dev/null 2>&1; then
    LIMIT="timeout -k 5 $STEP_LIMIT"
else
    LIMIT=
    echo "smoke_test: 'timeout' not found; steps run without a time limit"
fi

RESULTS=
FAILED=0
record() {        # name status note
    RESULTS="$RESULTS
  $2  $1${3:+  ($3)}"
    [ "$2" = FAIL ] && FAILED=1
    echo "[$2] $1${3:+: $3}"
}

# --- import -----------------------------------------------------------------------------------
cd "$ROOT" || exit 2
if $LIMIT "$PY" -c "import n64dk.cli as c; c.load_commands(); assert 'doctor' in c.COMMANDS" \
        >"$WORK/import.log" 2>&1; then
    record import PASS
else
    record import FAIL "see output below"
    tail -n 20 "$WORK/import.log"
fi

# --- doctor -----------------------------------------------------------------------------------
$LIMIT "$PY" -m n64dk --format json doctor --host --out "$WORK/out" \
    ${TOOLCHAIN:+--toolchain-dir "$TOOLCHAIN"} >"$WORK/doctor.json" 2>"$WORK/doctor.err"
status=$?
case $status in
    0) record doctor PASS ;;
    1) record doctor FAIL "required checks failed"
       grep -A1 '"required_failures"' "$WORK/doctor.json" | head -n 20 ;;
    124) record doctor FAIL "timed out after ${STEP_LIMIT} s" ;;
    *) record doctor FAIL "exit $status"; tail -n 20 "$WORK/doctor.err" ;;
esac

# --- synthetic unit tests ---------------------------------------------------------------------
if $LIMIT "$PY" -m unittest tests.test_media tests.test_interop tests.test_doctor >"$WORK/tests.log" 2>&1; then
    record synthetic PASS "$(tail -n 3 "$WORK/tests.log" | grep -m1 '^Ran' || true)"
else
    status=$?
    record synthetic FAIL "$( [ $status = 124 ] && echo "timed out" || echo "exit $status")"
    tail -n 30 "$WORK/tests.log"
fi

# --- synthetic function match -----------------------------------------------------------------
N64DK_STATE_DIR="$WORK/state" $LIMIT "$PY" -m n64dk.commands.doctor smoke-match --work "$WORK/match" \
    ${TOOLCHAIN:+--toolchain-dir "$TOOLCHAIN"} >"$WORK/match.json" 2>"$WORK/match.err"
status=$?
case $status in
    0) record match PASS "$(grep -m1 '"pipeline"' "$WORK/match.json" | tr -d ' ,')" ;;
    2) record match SKIP "$(grep -m1 '"reason"' "$WORK/match.json" | sed 's/^ *"reason": *//; s/"$//; s/^"//')" ;;
    124) record match FAIL "timed out after ${STEP_LIMIT} s" ;;
    *) record match FAIL "exit $status"; cat "$WORK/match.json"; tail -n 20 "$WORK/match.err" ;;
esac

echo "smoke_test summary:$RESULTS"
exit $FAILED
