#!/bin/bash
# RL track: train small "student" BC policies (deployable sizes) on the ResBot shards that already sit on
# gnode006's node-local scratch (created by our job bb-bc2). Guest-account rules: see CLAUDE.md.
# usage: sbatch --job-name=bb-rl-student --nodelist=gnode006 rl/slurm/student.sbatch <tag> <cfg> [<cfg> ...]
#   cfg = CHxBLOCKS, e.g. 8x2 12x1 16x1
#SBATCH -A irel
#SBATCH --qos=normal
#SBATCH -p u22
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH -o /home2/%u/botbattle-saiyam/logs/%x-%j.out
set -euo pipefail
TAG="$1"; shift
H=/home2/$USER/botbattle-saiyam
S=/scratch/$USER/botbattle-saiyam
[ -n "$S" ] && [ -n "$H" ] && [ -n "$TAG" ] || exit 1
export PIP_CACHE_DIR="$S/cache/pip" XDG_CACHE_HOME="$S/cache" UV_CACHE_DIR="$S/cache/uv" \
       UV_PYTHON_INSTALL_DIR="$S/uvpy" JAX_COMPILATION_CACHE_DIR="$S/cache/jax" \
       MPLCONFIGDIR="$S/cache/mpl" TMPDIR="$S/tmp"
mkdir -p "$S/tmp" "$H/rl/results/student"
PY="$S/venv-gpu126/bin/python"
[ -x "$PY" ] || { echo "no GPU venv on this node"; exit 2; }
[ -d "$S/code-bc/data/bc/ResBot" ] || { echo "no ResBot shards on this node"; exit 2; }
W="$S/rl-student-$TAG"
mkdir -p "$W/data" "$W/learn" "$W/out"
cp "$H/rl/code/learn/bc_train.py" "$W/learn/"
[ -e "$W/data/bc" ] || ln -s "$S/code-bc/data/bc" "$W/data/bc"
cd "$W"
"$PY" -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pids=()
for CFG in "$@"; do
  CH="${CFG%x*}"; BL="${CFG#*x}"
  "$PY" learn/bc_train.py --name ResBot --epochs 6 --bs 512 --lr 3e-3 --ch "$CH" --blocks "$BL" \
      --out "out/ResBot_${CFG}.pt" > "out/ResBot_${CFG}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || true; done
cp -f out/*.pt out/*.log "$H/rl/results/student/"
echo done
