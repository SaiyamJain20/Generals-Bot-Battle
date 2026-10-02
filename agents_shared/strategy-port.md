# strategy-port — porting the best readable bots' strategies into our bot

Owner: strategy-port agent. Status: IN PROGRESS.
Base: `bots/versions/sp_base.py` = `bots/participant.py` + `runs/ada2/best.json` (made with tools/make_variant.py).
Variants: `bots/versions/sp_<idea>.py` (sp_base + one change behind a PARAMS flag, flag ON).
A/B: `tools/abmulti.py 40 3 --bots ... --opps bots/opp/ext_sentinel10.py bots/versions/t1c.py bots/opp/zoo_sniper.py --offset 3000`
Raw outputs: `runs/sp/*.txt`.

## Log (newest at bottom)
