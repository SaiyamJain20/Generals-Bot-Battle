#!/bin/bash
# Fetch the strongest external sparring bots into $CODE/vendor/ext on an Ada node (sparse, pinned commits)
# and add their Python deps to the venv. Usage (inside a job, from $CODE): bash tune/slurm/setup_ext.sh <venv-python> <uv>
set -euo pipefail
PY="$1"; UV="$2"
mkdir -p vendor/ext
clone() {  # url dir commit paths...
  local url="$1" dir="$2" commit="$3"; shift 3
  if [ ! -d "vendor/ext/$dir/.git" ]; then
    git clone -q --filter=blob:none --no-checkout "$url" "vendor/ext/$dir"
    git -C "vendor/ext/$dir" sparse-checkout set "$@"
    git -C "vendor/ext/$dir" checkout -q "$commit"
  fi
}
clone https://github.com/relh/generals-bots relh_generals-bots 04f81887b186ff5a4343a58b52d32a23746d82ae \
  competition/agents/sentinel_python generals
clone https://github.com/blake-ar/generals-bots blake-ar_generals-bots df93b034f4da2ade4e90234ef83cc9afceb7423d \
  competition/agents/conv_1313 competition/agents/smoke_1260_baseline
[ -e .venv312 ] || ln -s "$(dirname "$(dirname "$PY")")" .venv312
"$UV" pip install -q --python "$PY" "jax==0.11.0" threadpoolctl numpy
"$PY" -c "import jax; print('jax', jax.__version__)"
