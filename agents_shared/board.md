05:32 | bot-scout | started research on local sparring-partner bots
05:32 | porter | started: wrapping external bots as bots/opp/ext_*.py (notes in agents_shared/porter.md)
05:44 | zoo-builder | resumed after crash; writing bots/opp/zoo_*.py one at a time
05:45 | bot-scout | wrote part1 of agents_shared/bot-scout.md: Amin PPO numpy weights (LOW effort), relh Sentinel JAX heuristic, quant-eagle (no weights)
05:45 | training-research | wrote Recommendations section in agents_shared/training-research.md (pool 55/30/15, pfsp 1.0, promotion w/ re-eval, holdout set)
05:46 | training-research | DONE: full answers in agents_shared/training-research.md; questions.md updated (Q8-11 added)
05:46 | zoo-builder | zoo_expander_plus done (0.78 vs hunter)
05:49 | bot-scout | part2 written: Marathon standings (public JSON), blake-ar=bca conv_1313 weights (25MB), hv-nguyeen heuristic, juraj C++; no code for ResBot/nanomena/Kubic/FreeLunch
05:50 | bot-scout | DONE: agents_shared/bot-scout.md complete (priority: Amin PPO numpy > blake-ar/bca conv_1313 JAX > relh Sentinel > Juraj C++ > hv-nguyeen)
05:51 | zoo-builder | zoo_flash, zoo_castler, zoo_expander_plus done; next gatherer/sniper/turtle_dt/mixed
05:52 | algo-study | started: reading participant.py + EklipZ/relh/juraj/RL bots; notes in agents_shared/algo-study.md
05:54 | strategy-research | agents_shared/strategy-research.md done: 20 ranked ideas, top: race calculator, castle-build strike window, 2-ply local search, deathtouch solver, fog heat map, opp profiles
05:55 | zoo-builder | all 7 zoo bots done: bots/opp/zoo_{flash,castler,gatherer,sniper,turtle_dt,expander_plus,mixed}.py; results in zoo-builder.md
05:56 | porter | ext_amin.py (Amin PPO CNN numpy, 12 ms/move) working; ext_bca.py (bca transformer JAX, 64 ms/move, 550 MB) runs; testing
05:58 | replay-analyst | findings in agents_shared/replay-analyst.md (tools/replay_an/*): castles at dist exactly 7 from gen price 35, t~123/167/212; kills = single 70-100 stack run, 3-5x gen army; opening not discriminating
05:59 | algo-study | algo-study.md §0 context + §1.1 fog gather-queue stack estimate, §1.2 fog-aware tracks written
06:00 | porter | bots/opp/ext_stdio.py generic stdio adapter; ext_superbot/ext_juraj35/ext_juraj34 (Juraj C++, ~1 ms/move) working; superbot 0.95 vs hunter, t1c 0.85 vs it
06:01 | algo-study | §1.3 threat-path multi-root defense gather, §1.4 chokes, §1.5 sweep-DP kill/threat written
06:03 | algo-study | §1.6 land-per-move expansion knapsack, §1.7 cycle timing, §1.8 general prediction bounds, §1.9 all-in, §1.10 prune written
06:03 | porter | +ext_hvn (hv-nguyeen heuristic, 1.00 vs hunter), ext_doomstack, ext_mybot9 (stdlib, in-process via ext_starterkit.py); sentinel wrappers ready
06:04 | algo-study | §2 relh/juraj (chase-kill, choke block, approach-threat FSM, castle rules, all-in pricing) + §3 RL findings written
06:05 | algo-study | DONE: agents_shared/algo-study.md complete; ranked port list in §4 (chase-kill, castle gating, capture-rate leaks, enemy move classifier/fog stack, fog tracks, drain veto, choke defense, sweep-DP kill)
06:07 | porter | +ext_boss (bca NumPy heuristic), ext_humanexe (bca JAX port of EklipZ Human.exe), ext_relhexp; bca 0.94 vs hunter
06:10 | experimenter | started: A/B of algo-study §4 ports vs exp_base (= participant.py + runs/local3_best.json); notes in agents_shared/experimenter.md
06:3x | vast-research | section 0-1 (framing + 25 ranked ideas) written to agents_shared/vast-research.md
06:4x | vast-research | DONE: agents_shared/vast-research.md (25 ideas, test plans, sources); top: stealth routing, archetype gate matrix, launch-value, CMA mean averaging, SPRT, catastrophe gate
06:13 | porter | STRONG: ext_bca and ext_sentinel10 both score 0.50 vs t1c (16/20 games); juraj34 0.20, superbot/hvn 0.15
16:48 | strategy-port | started: studying Sentinel v10 / bca heuristics, porting top-3 ideas as sp_*.py (notes in agents_shared/strategy-port.md)
16:49 | rl-research | DONE agents_shared/rl-research.md: Jev not found anywhere (candidates AverageJoe/ResBot/quant-eagle); rec = ES over low-dim context modulator of heuristic weights
16:50 | auditor | agents_shared/auditor.md done: top risks = (1) THREAT_TREES 141KB replay-trained blob goes LIVE if candidate has learned_threat_w>0 (ada3/5c/6c/local7/7b do) -> remove unless clear gain, else disclose; (2) placeholders only WARN in check_submission; (3) header lacks constants/training disclosure, proposed text in file; checks+build OK
