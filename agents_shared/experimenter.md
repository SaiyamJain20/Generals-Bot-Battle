# experimenter — A/B of algo-study §4 ports

Owner: experimenter agent. Status: IN PROGRESS.

## Setup
- Baseline `bots/versions/exp_base.py` = `bots/participant.py` (snapshot taken 06:10, md5 117b59a2…, which already
  contains the coordinator's fog_model/fog_tracks code with both flags OFF) + `runs/local3_best.json` via
  `tools/make_variant.py`. NOTE: participant.py was being edited while I started; the first snapshot (06:08) was
  mid-edit and broken (AttributeError prevT -> every move PASS under DEBUG=False), so it was discarded.
  Always smoke-test a variant with DEBUG=True before A/B (arena swallows exceptions into PASS).
- Each `bots/versions/exp_<idea>.py` = exp_base.py + one change behind a `x_<idea>` PARAMS flag (default ON).
  The change is applied by a patch script (string edits) so the same edit can be applied to participant.py.
- Runner: same as `tools/abpool.py` (run_match, map_offset 600, 60 games per opponent, 4 workers), plus per-game
  dump for a paired (same map, same side) difference. Opponents: exp_base, rusher, zoo_sniper, t1c for the exp;
  rusher, zoo_sniper, t1c for the baseline.
- Pooled CI = 1.96*sqrt(p(1-p)/n). Paired CI = 1.96*sd(diff)/sqrt(n) over the 180 common games (rusher, sniper, t1c).

## Results
(see table at the bottom; newest experiments appended)
