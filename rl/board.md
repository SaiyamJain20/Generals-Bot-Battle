19:18 | coordinator | RL track started (worktree rl-track); agents: rl-survey, rl-deploy, rl-env
19:19 | rl-survey | started: reading vendor RL bots + generals-bots env, then web survey
19:19 | rl-deploy | started: benchmarking pure-Python forward-pass techniques
19:19 | rl-env | started: reading engine/features/JAX env; JAX competition mode has build_castles+deathtouch+fog natively
19:22 | rl-survey | interim: 'Jev'=TypeSafe's LLM classifier trained w/ proprietary RLCD (calibrated decisions) - not a game RL method. Prior generals RL: zero v3 (BC+PPO 36h H100), AverageJoe (15M-param transformer, 4xH200x4d), quant-eagle (10M ViT, 4x5090x16h, #10 ladder, numpy-allowed deploy). All 10-100x our compute and >>1MiB stdlib budget.
20:46 | rl-deploy | pyfeat.py done, exact on 600 positions (20-game parity running); starting bigint-lane pyinfer
21:40 | rl-survey | report done: rl/agents/rl-survey.md (14 claims graded; rec: heuristic core + guardrails + ES/BC-scorer hybrid, no scratch RL)
20:49 | rl-train | rl/ppo_cpu.py works end-to-end (spawn workers, league, PFSP, PPO+KL-to-BC, pipelined rollouts); first smoke: ~250-340 turns/s/worker under heavy laptop load, worker RSS ~510MB; now measuring + drift check
20:54 | rl-train | DONE: rl/ppo_cpu.py + rl/agents/rl-train.md (launch cmd, throughput, hparams; lr1e-6 one update -> 99.95% argmax agree, KL 1e-6)
20:55 | rl-hier | started; bot_ro.py hook generated (rl/hier/gen_bot_ro.py)
20:56 | rl-deploy | pyfeat exact on 21110 pos (20 games); pyinfer bigint-lane 12x1: ~8ms p50/11ms p99 infer on 1 core, 99.9% argmax agreement vs torch; pystudent.py (30KB) builds & passes check_submission
20:58 | rl-hier | bot_ro.py + ro_ppo.py written; unit checks (a)-(d) pass; starting smoke run
21:05 | rl-deploy | DONE: pystudent.py 29.8KB, same-move 99.83% vs torch, act p50 11.6/p99 20.7ms (loaded core); 8x2 ~9ms, 16x1 ~15ms p50
21:08 | bot-finder | ksolmann wrapper works (3M plain transformer, 4/4 vs hunter); testing vs tune_base12g
21:11 | rl-hier | fixed t_end discount per coordinator; tests pass; sanity run (14 it x 32 games, 3 workers) running
21:24 | bot-finder | ksolmann beats tune_base12g 8-2 (n=10) and hunter 10-0; ronit_graph beats hunter 9-0-1, loses 0-10 to ours; threads capped
21:40 | rl-hier | done: rl/hier/* ; tests pass; sanity 14 it: argmax 0.467+-0.064 vs tune_base12g (30 pairs fresh), 0.75 vs t1c; ~0.8 games/s on 3 workers. Report rl/agents/rl-hier.md
22:31 | track-C (main session) | launched GRPO over the context modulator on Ada: bb-rlc-c1 (36 CPU, W8 K4 G24) + bb-rlc-c2 (10 CPU, W16 K4 G8); code rl/c/; checkpoints will appear in ~/botbattle-saiyam/rlc/results/<tag>/policy_it*.json (full PARAMS for rl/c/participant_c.py)
22:36 | track-C | C3 (MLP-12 GRPO) running on the laptop, 9 workers, log rl/c/runs/c3/out.log
01:42 | track-C | OFFICIAL evaluator: image codebot-python:1 built (6b00c2e0d815); submission/participant_id.py (F2) passes validate and beats starter (capture t252), first response 75 ms, max 5.6 ms; rl/c/official_parity.py: sim observe+step == official adapter on 3 full games (1564 obs identical)
01:57 | track-C | paired eval fresh 1550-1574 (300 g each): F2 0.670, C1-it299 0.688, C3-it339 (MLP) 0.690 (t1c .52->.75), C2 0.672 (replaced by ES run e1). C3b = C3 resumed on laptop; C1 continues on Ada
04:20 | rl-coord | Track A interim vs F2 (200 g paired, maps_fresh 1550+): roA2 i2 0.497 -> i3 0.568+-0.035; roF2 i2 0.477 -> i3 0.537+-0.035; 0 errs, 2.4 ms/turn. Final verdict on untouched 1850-1949 after runs end (~04:20 / ~05:00).
05:40 | rl-coord | Track A finalist #1 = rl/final/ro_a2_avg.py (RO-PPO roA2, mean of weights it 395-460, base tune_base12g+RO). Head-to-head vs F2, selection slices (200 g each): maps 1250+ 0.590, maps 1350+ 0.580 (single snapshot it427: 0.557 / 0.570; 0.568 on 1550+). 0 errs, ~3.4 ms/turn. Weak-spot matrix (KSolmann, 3 clones, t1c, c_a2es3, zoo_mixed, rusher) running on maps 1350-1369 vs F2.
