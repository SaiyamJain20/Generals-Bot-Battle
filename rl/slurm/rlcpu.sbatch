#!/bin/bash
# RL track CPU job (guest account: follow CLAUDE.md). Runs one python command from the staged RL tree.
# usage: sbatch --job-name=bb-rl-<tag> -c <N> --mem=<M> rl/slurm/rlcpu.sbatch <tag> <run-subdir> <python args...>
#   <run-subdir> is the output dir (relative to the run tree) that gets synced to ~/botbattle-saiyam/rl/results/<tag>/
#SBATCH -A irel
#SBATCH --qos=normal
#SBATCH -p u22
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH --time=07:00:00
#SBATCH -o /home2/$USER/botbattle-saiyam/logs/%x-%j.out
set -euo pipefail
TAG="$1"; OUTSUB="$2"; shift 2
H=/home2/$USER/botbattle-saiyam
S=/scratch/$USER/botbattle-saiyam
[ -n "$S" ] && [ -n "$H" ] && [ -n "$TAG" ] && [ -n "$OUTSUB" ] || exit 1
mkdir -p "$S/bin" "$S/cache" "$S/tmp" "$H/rl/results/$TAG"
export PIP_CACHE_DIR="$S/cache/pip" XDG_CACHE_HOME="$S/cache" UV_CACHE_DIR="$S/cache/uv" \
       UV_PYTHON_INSTALL_DIR="$S/uvpy" JAX_COMPILATION_CACHE_DIR="$S/cache/jax" \
       MPLCONFIGDIR="$S/cache/mpl" TMPDIR="$S/tmp" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
if [ ! -x "$S/bin/uv" ]; then
  curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz \
    | tar -xz -C "$S/bin" --strip-components=1
fi
V="$S/rl-venv-$SLURM_JOB_ID"
"$S/bin/uv" venv -q -p 3.12 "$V"
"$S/bin/uv" pip install -q --python "$V/bin/python" numpy safetensors
"$S/bin/uv" pip install -q --python "$V/bin/python" torch --index-url https://download.pytorch.org/whl/cpu
W="$S/rl-run-$TAG"
mkdir -p "$W"
cp -r "$H/rl/code/." "$W/"
cd "$W"
[ -e .venv312 ] || ln -s "$V" .venv312   # ext_* wrappers call ../../.venv312/bin/python (dir link keeps pyvenv.cfg)
[ -f data/maps.jsonl ] || gunzip -k data/maps.jsonl.gz
[ -f data/maps_fresh.jsonl ] || [ ! -f data/maps_fresh.jsonl.gz ] || gunzip -k data/maps_fresh.jsonl.gz
mkdir -p "$OUTSUB"
( while true; do sleep 300; cp -f "$OUTSUB"/*.jsonl "$OUTSUB"/*.json "$OUTSUB"/last.pt "$H/rl/results/$TAG/" 2>/dev/null || true; done ) &
SYNC=$!
PYTHONPATH="$W" "$V/bin/python" "$@" || true
cp -f "$OUTSUB"/*.jsonl "$OUTSUB"/*.json "$OUTSUB"/*.pt "$H/rl/results/$TAG/" 2>/dev/null || true
kill $SYNC || true
echo done
