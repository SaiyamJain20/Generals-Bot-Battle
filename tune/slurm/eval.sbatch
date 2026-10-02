#!/bin/bash
# Generic evaluation job on Ada: runs tools/abpool.py with the given arguments.
# usage: sbatch --job-name=bb-eval-<tag> tune/slurm/eval.sbatch <tag> <abpool args...>
#SBATCH -A irel
#SBATCH --qos=normal
#SBATCH -p u22
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH -o /home2/$USER/botbattle-saiyam/logs/%x-%j.out
set -euo pipefail
TAG="$1"; shift
H=/home2/$USER/botbattle-saiyam
S=/scratch/$USER/botbattle-saiyam
[ -n "$S" ] && [ -n "$H" ] || exit 1
V="$S/${VENV_NAME:-venv-$SLURM_JOB_ID}"
mkdir -p "$S/bin" "$S/cache" "$S/tmp" "$H/results/eval"
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
CODE="$S/code-eval-$TAG"
mkdir -p "$CODE"
cp -r "$H/code/." "$CODE/"
cd "$CODE"
if [ "${WITH_EXT:-0}" = "1" ]; then
  bash tune/slurm/setup_ext.sh "$V/bin/python" "$S/bin/uv"
fi
TOOL="${EVAL_TOOL:-tools/abpool.py}"
"$V/bin/python" "$TOOL" "$@" | tee "$H/results/eval/$TAG.txt"
