# porter: external sparring bots wrapped as bots/opp/ext_<name>.py

All external code lives in vendor/ext/ (gitignored). LOCAL TESTING ONLY, never in the submission.
Scores below are from tools/ab4.py (score = first-named bot's points/games, draws 0.5), 3 workers unless noted.

## Found leads (beyond bot-scout.md)
relh docs (vendor/ext/relh_generals-bots/docs/agent-development/opponent-coverage.md, v10-opponent-coverage.md) list more
competition-native opponents: Juraj V3.4 C++ + V3.5 C++ (Klincent/generals-bots@e50123ce, competition/agents/juraj_cpp, juraj_v35_cpp;
J35 beat Sentinel v2 14-2), Juraj final-superbot C++ (Klincent@704ce13d submissions/final-superbot.zip, newer J35 with muster/doomguard),
doomstack_rusher Python (Klincent@2260b6f1), my_bot..my_bot9 stdlib Python (mrinmoy2developer/gio-competition-tooling@f624c741, PR 138
upstream; castle prox penalty 10 instead of 14 -> some invalid builds), Amin archives main7_iter79/main7_iter308.
Extracted copies: vendor/ext/juraj/{juraj_v34,juraj_v35,doomstack}, vendor/ext/juraj/final-superbot.zip.

## 1. ext_amin (+ ext_amin_iter79, ext_amin_iter308)  WORKING
- Source: https://github.com/Amin-Debabeche/generals-bots @ 34506fe2d684f163ada3a0cc3401d192436bfc37, competition/agents/my_bot
  (HEAD weights.npz == bot_archive/my_bot_main8_iter160.zip, sha1 c2cd058b). Licence MIT (engine fork).
- Self-play PPO 5-layer CNN, pure NumPy, greedy masked argmax, memory features; castle-aware.
- Wrapper calls original numpy_infer.py (loaded by path); PARAMS["ckpt"] in {main8_iter160 (default), main7_iter79, main7_iter308, head}.
- IMPORTANT: forces BLAS to 1 thread (threadpoolctl) - with 16 OpenBLAS threads on the busy laptop it took 144 ms/move; 1 thread = 12 ms.
- Per move: mean 12 ms, max 13 ms, first 29 ms.
- Results: vs random 10/10 (1.00); vs hunter 0.875 (17W 1D 2L); t1c vs ext_amin 0.90 (ext_amin scores 0.10). 0 forfeits.

## Generic stdio adapter: bots/opp/ext_stdio.py
- StdioBot(cmd, cwd, env).act / make_act(...): one subprocess per game, handshake "pid H W", frame per turn, one reply line.
  Restarts at turn 0 (or when turn goes backwards); kills processes of finished games via a process-wide registry
  and at exit. Child stderr -> /dev/null (EXT_STDIO_STDERR=<file> to keep it). Dead child / reply timeout raises
  (visible forfeit in the arena). Overhead ~0.1 ms per move.

## 2. ext_superbot, ext_juraj35, ext_juraj34 (Juraj C++ family)  WORKING
- Sources (MIT): https://github.com/Klincent/generals-bots
  - superbot: @704ce13d891af22361cb76b9389da92a36b9903c submissions/final-superbot.zip (author's final submission; J35 lineage +
    rear collection, muster, doomguard, live castle price 14, anti-cycle)
  - juraj35: @e50123cee7d924f0d643acd372a5300971f93917 competition/agents/juraj_v35_cpp (deterministic)
  - juraj34: same commit, competition/agents/juraj_cpp (RNG seeded from entropy; JURAJ_RNG_SEED env pins it; left unset)
- Build: `bash vendor/ext/juraj/<superbot|juraj_v35|juraj_v34>/build.sh` (g++ -O2 -std=c++17, ~10 s). Binaries already built.
- Per move ~1 ms (mean 0.8-1.3 ms, max ~10 ms, first ~20 ms).
- Results (2 workers): superbot: random 1.00, hunter 0.95 (19-0-1), t1c vs superbot 0.85 (superbot 0.15).
  juraj35: random 1.00, hunter 0.95 (19-0-1).
  juraj35: t1c vs juraj35 1.00 (juraj35 0.00). juraj34: random 1.00, hunter 0.975 (19-1-0), t1c vs juraj34 0.80 (juraj34 0.20).
  => superbot and juraj34 are the strongest of the family vs t1c; juraj35 is weak vs t1c despite beating hunter.

## 3. ext_bca (Marathon #5 "bca", conv_1313 transformer PPO)  WORKING (results pending)
- Source: https://github.com/blake-ar/generals-bots @ df93b034f4da2ade4e90234ef83cc9afceb7423d competition/agents/conv_1313
  (MIT engine-fork licence; weights.npz 25 MB bf16 EMA iter 1313). Needs jax (installed in .venv312), CPU only.
- In-process, bot.py loaded by path (module `bot` name clash); jitted forward + params cached process-wide;
  persistent JAX compile cache at conv_1313/.jax_cache. Greedy argmax (original default).
- Per move: mean 64 ms, max 104 ms, first 5.2 s (compile; must stay < 10 s arena first-move limit). RSS ~550 MB. 1 worker only.
- Results: random 6/6 (1.00).

## 4. ext_hvn (hv-nguyeen heuristic Controller)  WORKING
- Source: https://github.com/hv-nguyeen/Generals-RL-bot @ cc259256a787a11cb2e0fea04590f604c351f950, bot/ (no licence file).
  NumPy heuristic; config configs/v18.json (identical bytes to v9, v13..v17; only HEAD code is public). Their ladder: v13 heuristic
  ~1737, v18 ~1587 Elo; their neural bot (1849, rank 21/86) has NO public weights so the heuristic plays.
- stdio subprocess (`python -m bot.main`, BOT_CONFIG) via ext_stdio. Per move mean 6 ms, p99 25 ms, max 32 ms, first ~0.5 s.
- Results (2 workers): random 1.00, hunter 1.00 (20-0-0).

## 5. ext_doomstack, ext_mybot9 (stdlib Python, in-process via bots/opp/ext_starterkit.py)
- ext_starterkit.make_act(agent.py path): generic adapter for starter-kit Agent(player_id,H,W).act(obs dataclass);
  loads by path, fresh Agent per game, silences print/sys.stderr of the original.
- doomstack: https://github.com/Klincent/generals-bots @ 2260b6f19d51a14d7c68770677f22d04dfd88022 competition/agents/doomstack_rusher
  (MIT), STRESS_MODE unset (default doomstack). No castles, no defence. <1 ms/move.
- mybot9: https://github.com/mrinmoy2developer/gio-competition-tooling @ f624c741ad5084be63fc17bacd3961599b8dcc82 competition/agents/my_bot9
  (MIT). PARAMS fix_price=True patches PROXIMITY_PENALTY 10->14 (original targets older castle price; False = unchanged). <1 ms/move.
- (06:07) bca: hunter 0.94 (14W 2D 0L, 1 worker). hvn: t1c vs hvn 0.85 (hvn 0.15). mybot9: random 1.00, hunter 0.60, t1c 0.95 (mybot9 0.05).
  doomstack: random 1.00, hunter 0.15 (weak, no castles), t1c 1.00.

## 6. ext_boss (bca's NumPy Boss heuristic)  WORKING
- https://github.com/blake-ar/generals-bots @ df93b034 competition/agents/boss/agent.py (MIT). In-process starter-kit adapter. ~10 ms/move.

## 7. ext_humanexe (bca's JAX competition-rules port of EklipZ Human.exe)  WORKING
- https://github.com/blake-ar/generals-bots @ df93b034 generals/agents/human_exe_agent.py (MIT; original idea EklipZgit/generals-bot MIT).
  This covers lead #4 (classic Human.exe) without porting EklipZ's 119 MB live-API code.
- Single file, two modes: imported -> spawns itself as a stdio server (blake-ar `generals` package on PYTHONPATH) via ext_stdio.
  PARAMS["agent"] can pick blake-ar's other JAX heuristics (boss, castle_economist, deathtouch_clock, draw_grinder, fog_scout, raider).
- Adapter check: the JAX BossAgent through this adapter replays the NumPy ext_boss game move-for-move (map 302: same 628-turn win,
  83 land, 4 castles), so the observation conversion is right.
- Per move ~12 ms, first ~5 s (JIT; cache in vendor/ext/blake-ar_generals-bots/.jax_cache_porter, pre-warmed for 16 shapes).
- (06:10) sentinel (v2): random 1.00, hunter 0.95 (19-0-1), t1c vs sentinel 0.925 (sentinel 0.075). Per move mean ~11 ms, max 27 ms,
  first 1.9-2.8 s. sentinel10: random 1.00, hunter 0.95; mean ~20 ms, max 34 ms, first ~4.6 s.

## Unexplored lead (needs user approval)
- mortid0/generals-bots, branch Marathon, commit 78128d6db6cc41ab2a94ee855349be42589fcc8f:
  competition/submissions/a9_20260812/atlas-negative-candidate.zip = self-contained trained C++ submission ("A9", my_bot_cpp/,
  weights.bin), downloaded to vendor/ext/mortid0_a9/ (zip sha256 matches the repo's SHA256SUMS). Building/running it was
  blocked by the auto-mode permission classifier, so it is NOT wrapped. If approved: bash my_bot_cpp/build.sh, then an
  ext_stdio wrapper like ext_juraj35.py. (Its lg45_cpp dir is a random-weight feasibility runtime: skip.)
- Other upstream forks (dylantirandaz, neurion-ai, thomasarmstrong98, Frankily, inbannable, LinuxFan2718, MaximHirschmann,
  QuarKUS7, OnGlacier, tyfleming): only starter expanders on default branches.

## How to run / caveats (all ext_ bots)
- Use like any opponent: `.venv312/bin/python tools/ab4.py bots/versions/t1c.py bots/opp/ext_<name>.py 20 3`.
- Requirements: vendor/ext/ clones (gitignored) at the commits above; C++ binaries built (juraj family); jax in .venv312
  (bca, sentinel*, humanexe); threadpoolctl (amin, boss: BLAS forced to 1 thread).
- Subprocess bots (juraj*, superbot, sentinel*, hvn, humanexe) run one child per game per worker; children of finished
  games are killed when the next game starts and at exit (checked: no orphans after batches).
- JAX bots: first move 2-5 s (JIT, persistent caches pre-warmed). The arena's 10 s first-move limit applies; on a very
  loaded box a cold cache could approach it. bca is ~65 ms/move and ~550 MB; run it with 1 worker.
- Rules caveats: all are competition-ruleset native (castle building, deathtouch 800, cap 1200) except: mybot9 written for
  an older castle price (patched to 14 by default); doomstack never builds; the learned bots (amin, bca) are greedy argmax
  (deterministic), as in their submissions.
- (06:15) **bca: t1c vs bca 0.50 (8W 0D 8L)** => bca scores 0.50 vs t1c, the strongest sparring partner so far.
  **sentinel10: t1c vs sentinel10 0.50 (10-0-10)** => v10 is far stronger vs t1c than default v2 (0.075). Per move ~17-24 ms.
  relhexp: random 1.00, hunter 1.00, t1c vs relhexp 1.00 (relhexp 0.00); per move mean 1.5 ms but p99 up to 195 ms, max 201 ms.
- ext_bca1260: bca's earlier checkpoint (competition/agents/smoke_1260_baseline, Smoke1260Agent), same wrapper (ext_bca.py
  PARAMS["model"]). ~38 ms/move when the box is quieter. Lost to conv_1313 on map 303.
