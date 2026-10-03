#!/bin/bash
# Final submission build + verification in one command.
#   tools/final_build.sh <participant_id> "<bot name>"
# 1) bakes runs/final_params.json (F2) into bots/participant.py -> submission/<participant_id>.py
#    (fills the header, DEBUG=False, runs tools/check_submission.py)
# 2) official organizer evaluator: validate + one full official-sandbox game vs the kit's starter bot
# 3) prints size and SHA-256 (keep it as the upload receipt)
set -euo pipefail
ID="${1:?usage: tools/final_build.sh <participant_id> \"<bot name>\"}"
NAME="${2:?usage: tools/final_build.sh <participant_id> \"<bot name>\"}"
[[ "$ID" =~ ^[A-Za-z0-9_-]{1,64}$ ]] || { echo "participant ID must be letters/digits/_/- (<=64)"; exit 1; }
cd "$(dirname "$0")/.."
PY=.venv312/bin/python
export PYTHONPATH=vendor/generals-bots:.
OUT="submission/$ID.py"
"$PY" tools/build_submission.py --params runs/final_params.json --id "$ID" --name "$NAME" --out "$OUT"
EVAL=.claude/worktrees/rl/rl/c/evaluator/evaluator/evaluate.py
if [ -f "$EVAL" ] && docker image inspect codebot-python:1 >/dev/null 2>&1; then
  "$PY" "$EVAL" validate "$OUT"
  RUN=".scratch/official_final_$(date +%H%M%S)"
  "$PY" "$EVAL" match "$OUT" .claude/worktrees/rl/rl/c/evaluator/evaluator/examples/starter.py \
      --seed 2026 --cpu 12 --out "$RUN" | tail -8
  echo "official game written to $RUN"
else
  echo "WARNING: official evaluator or image not found; skipped official validate/match"
fi
echo "----"
ls -l "$OUT"
sha256sum "$OUT"
