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
