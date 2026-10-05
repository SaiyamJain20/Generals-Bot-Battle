#!/bin/bash
# One-shot development setup.
#   ./setup.sh                 # venv + deps + pinned engine + 20,000 training maps + 2,000 held-out maps
#   MAPS=2000 FRESH=200 ./setup.sh   # smaller map pools (faster)
#   LEARN=1 ./setup.sh         # also install torch for BC / RL experiments
set -euo pipefail
cd "$(dirname "$0")"
PYBIN="${PYBIN:-python3.12}"
MAPS="${MAPS:-20000}"
FRESH="${FRESH:-2000}"
ENGINE_COMMIT=13db8f69a422380ea184d2f4ca262a38866c5fc6
if [ ! -x .venv312/bin/python ]; then
  "$PYBIN" -m venv .venv312 2>/dev/null || virtualenv -q -p "$(command -v "$PYBIN")" .venv312
fi
.venv312/bin/python -m pip install -q --upgrade pip
.venv312/bin/python -m pip install -q -r requirements.txt
[ "${LEARN:-0}" = "1" ] && .venv312/bin/python -m pip install -q -r requirements-learn.txt
if [ ! -d vendor/generals-bots/.git ]; then
  mkdir -p vendor
  git clone -q https://github.com/strakam/generals-bots.git vendor/generals-bots
fi
git -C vendor/generals-bots checkout -q "$ENGINE_COMMIT"
export PYTHONPATH="vendor/generals-bots:."
mkdir -p data
[ -s data/maps.jsonl ] || .venv312/bin/python sim/make_maps.py --start 0 --count "$MAPS" --out data/maps.jsonl
[ -s data/maps_fresh.jsonl ] || .venv312/bin/python sim/make_maps.py --start 900000 --count "$FRESH" --out data/maps_fresh.jsonl
echo "setup done. Next:  export PYTHONPATH=vendor/generals-bots:.  &&  .venv312/bin/python -m pytest -q tests/"
