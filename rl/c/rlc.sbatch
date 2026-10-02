#!/bin/bash
# Track C (GRPO/PPO over the context modulator) on Ada. Guest account: follow CLAUDE.md.
# usage: sbatch --job-name=bb-rlc-<tag> [-A irel --qos=normal -c 36 --mem=72G] rl/c/rlc.sbatch <tag> <grpo.py args...>
# Code = the RL track's staging (~/botbattle-saiyam/rl/code, read only) + Track C overlay (~/botbattle-saiyam/rlc/code).
#SBATCH -p u22
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --mem=28G
#SBATCH --time=08:00:00
#SBATCH -o /home2/$USER/botbattle-saiyam/logs/%x-%j.out
set -euo pipefail
TAG="$1"; shift
H=/home2/$USER/botbattle-saiyam
S=/scratch/$USER/botbattle-saiyam
[ -n "$S" ] && [ -n "$H" ] && [ -n "$TAG" ] || exit 1
mkdir -p "$S/bin" "$S/cache" "$S/tmp" "$H/rlc/results/$TAG"
export PIP_CACHE_DIR="$S/cache/pip" XDG_CACHE_HOME="$S/cache" UV_CACHE_DIR="$S/cache/uv" \
       UV_PYTHON_INSTALL_DIR="$S/uvpy" JAX_COMPILATION_CACHE_DIR="$S/cache/jax" \
       MPLCONFIGDIR="$S/cache/mpl" TMPDIR="$S/tmp" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
if [ ! -x "$S/bin/uv" ]; then
  curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz \
    | tar -xz -C "$S/bin" --strip-components=1
fi
V="$S/rlc-venv-$SLURM_JOB_ID"
"$S/bin/uv" venv -q -p 3.12 "$V"
"$S/bin/uv" pip install -q --python "$V/bin/python" numpy safetensors
"$S/bin/uv" pip install -q --python "$V/bin/python" torch --index-url https://download.pytorch.org/whl/cpu
W="$S/rlc-run-$TAG"
mkdir -p "$W"
cp -r "$H/rl/code/." "$W/"
cp -r "$H/rlc/code/." "$W/"
mkdir -p "$W/data/bc"
[ -f "$W/data/bc/ResBot_128x10.pt" ] || cp "$H/code/data/bc/ResBot_128x10.pt" "$W/data/bc/" 2>/dev/null || true
cd "$W"
[ -e .venv312 ] || ln -s "$V" .venv312
[ -f data/maps.jsonl ] || gunzip -k data/maps.jsonl.gz
[ -f data/maps_fresh.jsonl ] || [ ! -f data/maps_fresh.jsonl.gz ] || gunzip -k data/maps_fresh.jsonl.gz
OUT="rl/c/runs/$TAG"
mkdir -p "$OUT"
( while true; do sleep 300; cp -f "$OUT"/*.jsonl "$OUT"/*.json "$H/rlc/results/$TAG/" 2>/dev/null || true; done ) &
SYNC=$!
PYTHONPATH="$W" "$V/bin/python" rl/c/grpo.py --out "$OUT" "$@" || true
cp -f "$OUT"/*.jsonl "$OUT"/*.json "$H/rlc/results/$TAG/" 2>/dev/null || true
kill $SYNC || true
echo done
