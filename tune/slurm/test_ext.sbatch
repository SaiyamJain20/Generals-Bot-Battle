#!/bin/bash
#SBATCH -A irel
#SBATCH --qos=normal
#SBATCH -p u22
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH -o /home2/$USER/botbattle-saiyam/logs/%x-%j.out
set -euo pipefail
H=/home2/$USER/botbattle-saiyam
S=/scratch/$USER/botbattle-saiyam
mkdir -p "$S/bin" "$S/cache" "$S/tmp"
export PIP_CACHE_DIR="$S/cache/pip" XDG_CACHE_HOME="$S/cache" UV_CACHE_DIR="$S/cache/uv" \
       UV_PYTHON_INSTALL_DIR="$S/uvpy" JAX_COMPILATION_CACHE_DIR="$S/cache/jax" MPLCONFIGDIR="$S/cache/mpl" TMPDIR="$S/tmp"
which git g++ || true
if [ ! -x "$S/bin/uv" ]; then
  curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz | tar -xz -C "$S/bin" --strip-components=1
fi
[ -x "$S/venv/bin/python" ] || { "$S/bin/uv" venv -q -p 3.12 "$S/venv"; "$S/bin/uv" pip install -q --python "$S/venv/bin/python" cma numpy; }
CODE="$S/code-exttest"; mkdir -p "$CODE"; cp -r "$H/code/." "$CODE/"; cd "$CODE"
bash tune/slurm/setup_ext.sh "$S/venv/bin/python" "$S/bin/uv"
"$S/venv/bin/python" tools/ab4.py bots/opp/ext_sentinel10.py bots/opp/hunter.py 4 2
"$S/venv/bin/python" tools/ab4.py bots/opp/ext_bca.py bots/opp/hunter.py 2 1
