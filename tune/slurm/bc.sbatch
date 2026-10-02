#!/bin/bash
# GPU pipeline on Ada: fetch replays -> infer actions -> BC datasets -> train clones.
# usage: sbatch --job-name=bb-bc tune/slurm/bc.sbatch
#SBATCH -A irel
#SBATCH --qos=normal
#SBATCH -p u22
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH -o /home2/$USER/botbattle-saiyam/logs/%x-%j.out
set -euo pipefail
H=/home2/$USER/botbattle-saiyam
S=/scratch/$USER/botbattle-saiyam
[ -n "$S" ] && [ -n "$H" ] || exit 1
mkdir -p "$S/bin" "$S/cache" "$S/tmp" "$H/results/bc"
export PIP_CACHE_DIR="$S/cache/pip" XDG_CACHE_HOME="$S/cache" UV_CACHE_DIR="$S/cache/uv" \
       UV_PYTHON_INSTALL_DIR="$S/uvpy" JAX_COMPILATION_CACHE_DIR="$S/cache/jax" \
       MPLCONFIGDIR="$S/cache/mpl" TMPDIR="$S/tmp"
if [ ! -x "$S/bin/uv" ]; then
  curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz \
    | tar -xz -C "$S/bin" --strip-components=1
fi
# Ada's driver is CUDA 12.8: use the cu126 torch build (the default PyPI wheel needs a newer driver)
if [ ! -x "$S/venv-gpu126/bin/python" ]; then
  "$S/bin/uv" venv -q -p 3.12 "$S/venv-gpu126"
  "$S/bin/uv" pip install -q --python "$S/venv-gpu126/bin/python" numpy
  "$S/bin/uv" pip install -q --python "$S/venv-gpu126/bin/python" torch --index-url https://download.pytorch.org/whl/cu126
fi
PY="$S/venv-gpu126/bin/python"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
CODE="$S/code-bc"
mkdir -p "$CODE"
cp -r "$H/code/." "$CODE/"
cd "$CODE"
mkdir -p data/replays data/bc
"$PY" -c "import torch,sys; ok=torch.cuda.is_available(); print('CUDA OK' if ok else 'NO CUDA'); sys.exit(0 if ok else 3)"
PLAYERS="ResBot nanomena Kubic"
[ -f data/replays/meta.jsonl ] && [ "$(wc -l < data/replays/meta.jsonl)" -gt 8000 ] || \
  "$PY" learn/fetch_replays.py --events marathon --players $PLAYERS --workers 16
[ -f data/acts.jsonl ] && [ "$(wc -l < data/acts.jsonl)" -gt 8000 ] || \
  "$PY" learn/replay_parse.py --players $PLAYERS --max 9000 --workers 8 --out data/acts.jsonl
for P in $PLAYERS; do
  [ -f "data/bc/$P/act_000.npy" ] || \
    "$PY" learn/bc_dataset.py --players "$P" --max-games 3300 --sample 0.25 --name "$P" --workers 8
done
for P in $PLAYERS; do
  "$PY" learn/bc_train.py --name "$P" --epochs 6 --bs 512 --lr 2e-3 --ch 96 --blocks 8 --out "data/bc/${P}_96x8.pt"
  cp -f "data/bc/${P}_96x8.pt" "$H/results/bc/"
done
"$PY" learn/bc_train.py --name ResBot --epochs 8 --bs 512 --lr 2e-3 --ch 128 --blocks 10 --out data/bc/ResBot_128x10.pt
cp -f data/bc/ResBot_128x10.pt "$H/results/bc/"
echo done
