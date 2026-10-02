# zoo-builder

Status: resumed; bots written one at a time (results below).
Rules: 2 workers max, one batch at a time, run tools/capacity.sh first.

Build: scratchpad core.py (shared helper G) + pol_<name>.py concatenated -> bots/opp/zoo_<name>.py (single file each).
| bot | vs hunter (16g) | t1c vs bot (16g, t1c score) | random forfeits |
| zoo_expander_plus | 0.78 (9W 7D 0L) | 0.875 | 0 |
(after core fix: general pushes out via stuck_step)
| zoo_flash | 0.94 (15W 0D 1L) | t1c 0.875 (t1c lost 2 of 16... t1c score 0.875) | 0 |
| zoo_castler | 0.59 | t1c 0.81 | 0 |
| zoo_expander_plus | 0.97 | t1c 0.875 | 0 |
(t1c column = score of t1c in t1c-vs-bot, 16 games)
| zoo_gatherer | 0.81 | t1c 0.81 | 0 |
| zoo_sniper | 0.75 | (see below) | |
| zoo_sniper | 0.75 | t1c 0.625 | 0 |
| zoo_turtle_dt | 0.75 | t1c 0.875 | 0 |
| zoo_mixed (picks style by map seed, loads sibling zoo_*.py) | 0.81 | t1c 0.875 | n/a |
All <8 ms/move max. Sources: scratchpad zoo/core.py + pol_*.py (generated into bots/opp/zoo_*.py).
