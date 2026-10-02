#!/bin/bash
# CMA-ES tuning job on Ada (guest account: see CLAUDE.md rules).
# usage: sbatch --job-name=<name> tune/slurm/tune.sbatch <run-name> <hours> <map-base> <opp1> [opp2 ...]
#SBATCH -A research
#SBATCH --qos=low
#SBATCH -p u22
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --mem=24G
#SBATCH --time=14:00:00
#SBATCH -o /home2/$USER/botbattle-saiyam/logs/%x-%j.out
set -euo pipefail
RUN="$1"; HOURS="$2"; MAPBASE="$3"; shift 3
H=/home2/$USER/botbattle-saiyam
S=/scratch/$USER/botbattle-saiyam
[ -n "$S" ] && [ -n "$H" ] || exit 1
V="$S/${VENV_NAME:-venv-$SLURM_JOB_ID}"
mkdir -p "$S/bin" "$S/cache" "$S/tmp" "$S/runs"
export PIP_CACHE_DIR="$S/cache/pip" XDG_CACHE_HOME="$S/cache" UV_CACHE_DIR="$S/cache/uv" \
       UV_PYTHON_INSTALL_DIR="$S/uvpy" JAX_COMPILATION_CACHE_DIR="$S/cache/jax" \
       MPLCONFIGDIR="$S/cache/mpl" TMPDIR="$S/tmp"
if [ ! -x "$S/bin/uv" ]; then
  curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz \
    | tar -xz -C "$S/bin" --strip-components=1
fi
if [ ! -x "$V/bin/python" ]; then
  "$S/bin/uv" venv -q -p 3.12 "$V"
  "$S/bin/uv" pip install -q --python "$V/bin/python" cma numpy
fi
if ! "$V/bin/python" -c "import torch" 2>/dev/null; then
  "$S/bin/uv" pip install -q --python "$V/bin/python" torch --index-url https://download.pytorch.org/whl/cpu
fi
# fresh copy of the code for this run
CODE="$S/code-$RUN"
mkdir -p "$CODE"
cp -r "$H/code/." "$CODE/"
cd "$CODE"
if [ "${WITH_EXT:-0}" = "1" ]; then
  bash tune/slurm/setup_ext.sh "$V/bin/python" "$S/bin/uv"
fi
mkdir -p "$H/results/$RUN"
( while true; do sleep 300; cp -f runs/"$RUN"/*.json* "$H/results/$RUN/" 2>/dev/null || true; done ) &
SYNC=$!
START_ARG=""
[ -f "runs/start_$RUN.json" ] && START_ARG="--start runs/start_$RUN.json"
"$V/bin/python" tune/cma_tune.py --name "$RUN" --bot "${TUNE_BOT:-bots/versions/tune_base2.py}" $START_ARG \
  --sigma "${SIGMA:-0.15}" --opps "$@" --games "${GAMES:-12}" --popsize "${POP:-16}" --workers "${SLURM_CPUS_PER_TASK:-10}" \
  --hours "$HOURS" --map-base "$MAPBASE" ${EXTRA_ARGS:-} || true
cp -f runs/"$RUN"/*.json* "$H/results/$RUN/" 2>/dev/null || true
kill $SYNC || true
