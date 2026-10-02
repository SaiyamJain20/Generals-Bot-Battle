# rl-research (web research + reading only; no compute, no Ada)
[V]=verified from fetched source, [I]=inference, [U]=unverified/from memory

## 1. Who is "Jev"?  NOT FOUND (be honest)
Searched (web + gh): "Jev generals.io bot", "Jev RL", generals.bot leaderboard, GitHub repos/users, Straka paper, quant-eagle README,
Marathon standings in bot-scout.md. No generals-related "Jev" exists in any source. The only "Jev" hit on GitHub is cany7/Jev-Comparative-Evaluation
(an LLM reranker; unrelated). Marathon top-10 names have no Jev (ResBot, nanomena, Kubic, FreeLunch, bca, Chig, shy, sosynec666, hiems, thor).
Candidates for what the suggester meant (ranked), all RL models in this space:
 1. AverageJoe (Matej Straka, Lisy, Schmid) - the famous RL model; #1 human ladder. "Joe"/"Jev" mix-up plausible. [V]
 2. ResBot (Marathon #1, 90.7%) - almost certainly an RL/BC net ("Res"=ResNet?) but no public code or write-up. Possibly a person/Discord handle "Jev". [U]
 3. quant-eagle "Generals-Zero" (ViT, Q-boosted PPO). [V]
 -> ASK the suggester for a link/handle; or check generals.bot Discord. Do not spend more time.

## 2. Public RL entrants (ruleset-relevant)
- AverageJoe / Straka et al. arXiv 2606.23348 [V]: ViT 15.4M params (7 layers, d=448, 8 heads, 3x3 patches, 2 history tokens), per-cell HxWx9 action grid,
  PPO gamma=1, GAE .9, SPARSE +-1 reward only (shaping hurt), no BC, no pool (self-play vs frozen self), top-25% advantage filtering, EMA weights tau=.999 (+30 Elo),
  spawn-distance curriculum, 512 envs x 512 steps, JAX sim 50M fps/H200, 4 days x 4 H200 (260B steps in the 7-day variant). Earlier paper 2507.06825: BC pretrain + self-play, 36 h on 1 H100.
- quant-eagle [V]: 10M ViT, Q-boosted advantages (factored rank-8 Q-critic), privileged critic, backward curriculum from mid-game positions, scripted-expander league,
  EMA planes, top-25% filter; 16-24 h on 4-8 RTX5090 => ladder #10; greedy argmax; CPU torch twin <150 ms. No weights.
- bca conv_1313 (Marathon #5): transformer PPO + replay BC (bot-scout). Amin: CNN PPO with shaped rewards, rank ~105/116 (weak). Mattz: heuristics -> BC -> PPO + targeted castle examples, 6th Sprint, weak castle defence.
- Takeaway: all winners need 10^9+ env steps on big GPUs + ViTs; none are deployable in pure Python at 45 ms. The 1080Ti/2080Ti on Ada (11 GB, no bf16) cannot reproduce them in 5 h.

## 3. Method/framework survey 2024-26 (what matters for us)
- Frameworks: PureJaxRL / purejaxrl-style single-jit PPO (fits strakam JAX env), JaxMARL, Gymnax, Brax; PufferLib 3.0 (C envs, 1M+ sps/core, good for custom CPU sims - needs C env, not our Python sim);
  CleanRL (single-file PPO, easy to hack for a Python-sim worker pool); Sample Factory / EnvPool (async CPU rollouts; EnvPool needs C++ env); SB3/TorchRL (slow Python overhead, irrelevant at ~1 game/s/core).
  Our sim is ~1-2 s/game => ~10 games/s on 16 CPUs => ~40 games/s peak?? no: ~8-16 games/s; 5 h => ~150-300k games, ~10^8 decision steps. Fine for ES / small PPO, orders too small for from-scratch deep RL.
- Methods: PQN (parallel Q-learning, no replay/target, 2024) = simple value-based option; SimBa/BRO (bigger nets + norm scale well, but need lots of data); DreamerV3/MuZero/Gumbel (need big compute/search; no);
  OpenAI-ES / NES / CMA-ES / evosax / EvoJAX (evosax 2024-25 has OpenES, SNES, sep-CMA; scales to 1e3-1e5 params - we have a CMA-ES already); PBT/PFSP leagues (AlphaStar; already in tune/cma_tune.py via --pfsp);
  top-advantage filtering + EMA (Straka); distillation of nets to small MLP/linear/tree (DAgger; Viper decision-tree distillation) is the only route to deploy a net in stdlib.
- Key lessons for tiny policies [I]: (1) RL on a high-level discrete option choice has far shorter horizon + denser credit than cell-level PPO; (2) ES handles sparse win/loss and 3-5% win-rate noise best with
  antithetic sampling + common random numbers + rank-shaping; (3) with <=500 params, sep-CMA/OpenES beat PPO at this sample budget; (4) policy EMA / mean-averaging reduces noise (we already do mean avg).

## 4. Options for a 5-hour RL session
Baseline fact: current bot = heuristic, ~60 params, CMA-ES over self-play+pool. Eval noise: 80 games => SE ~0.056 (~40 Elo).
(a) ES over a small modulator net (state features -> multiplicative/additive adjustments of the ~60 option weights/thresholds)
   Gain: +10-30 Elo if features expose opponent type (aggression, castle count, army ratio, turn phase); mostly captures what archetype-gating already does by hand. Risk: LOW-MED (dimension 60 -> ~500 params with a linear 8-feature x 60 map
   = 480 params is too many for 5 h of noisy fitness; use LOW-RANK or 4-6 modulator channels). Deploy: trivial pure Python. 5 h: yes. Fallback = zero-init (identical to current bot), so can't regress by construction.
(b) PPO on option-selection with Python sim: Gain uncertain (0 to +30), Risk HIGH: needs wrapping participant.py's option scoring as a policy, ~1 s/game python sim, sparse reward, ~1e8 steps max; PPO variance + engineering (rollout workers, buffers, bugs) eats the 5 h; typically ends below the CMA-tuned baseline. Not recommended as primary.
(c) DAgger/imitation of top-bot replays at option level: replays have NO actions publicly (Marathon JSON replays = blob pointer; bot-scout) and option labels must be inferred from raw moves; learn/ already has BC for ResBot-BC/nanomena clones (bc_train.py) so labels exist only for clones. Gain: low (imitation caps at teacher, and our options are not the teacher's). Risk MED. Useful only as a PRIOR for (a): init modulator from clone-queried states. Skip as primary.
(d) Train a neural sparring partner on GPU in JAX (strakam env, PPO + Straka tricks: sparse reward, top-25% filter, EMA, spawn curriculum): not deployable; but a stronger/different opponent finds exploits and checks robustness. 5 h on one 2080 Ti ~ maybe 1-3B steps (vs 260B) => a bot at the level of Amin-ish/BC-ish, probably NOT stronger than our pool (ext_bca/sentinel/ResBot-BC). Gain for us: low. Risk: MED-HIGH (JAX/GPU env setup on old CUDA, 11 GB, Ada rules). Only worth it as a background job on the idle GPU with low effort: BC-init from learn/ datasets + PPO fine-tune, to add diversity. Treat as optional.

### Recommendation: (a), implemented as sep-CMA/ES over a LOW-DIMENSIONAL context modulator, with antithetic sampling and mean-averaging, tuned on the existing pool/holdout infra. Plus (d) optional background.

## 5. 5-hour plan (8 steps)  -- see final message (same text)
1. 0:00-0:30 Define context features from existing state (all already computed in participant.py: turn/800, my/opp land+army ratio, enemy_gather_ratio, classify_enemy_move aggression EMA,
   #castles mine/theirs, fog_risk, distance to enemy general estimate, own-general threat flag). Pick <=8, scale to ~[-1,1].
2. 0:30-1:15 Add modulator: w_k' = w_k * exp(s * (A_k . f)) for K=6-8 chosen high-leverage params groups (capture bias, garrison need, castle price/gating, scout, home-fill, cycle-gather, launch threshold, risk). A is KxF (<=64 numbers, init 0); store in the same json as params so cma_tune can optimise [params_scale + A]. Zero-init => bit-identical to the current best (test with parity run, 0 extra ms: ~64 mult-adds per move).
3. 1:15-1:45 Smoke on laptop (3 procs max): ab4 vs hunter 20 games to verify no crash/time regression; then launch on Ada: 16 workers, popsize ~24, antithetic pairs, CRN seeds, pool = anchors 55/league 30/self 15 (per training-research.md), PFSP p=1.
4. 1:45-4:15 Run ~2.5 h of generations (~25-35 gens at ~5 min each); start from the current best mean (x0), sigma 0.05 on A and 0.03 on scales; L2 penalty on |A| (lambda so that |A|<~1) to keep it from overfitting to noise; track pool AND holdout (ResBot-BC argmax, nanomena clone, bca, sentinel).
5. In parallel, spare GPU/CPU: optional (d) 2080 Ti JAX self-play run (strakam env, BC-init) only if setup < 30 min; otherwise skip.
6. 4:15-4:45 Re-evaluate top-3 checkpoints + mean + current best, 300+ fresh-seed games each vs full pool + holdout; also run an A-ablation: modulator on vs A=0 (must show >=+0.03 and holdout not down).
7. 4:45-5:00 Ship only if SPRT/600-game match passes (>=0.53 vs shipped, zero timeouts, <45 ms p99 on build_submission/check_submission). Else ship the best A=0 params (no loss).
8. Log: params json + A + git commit + manifest; fall back instantly by zeroing A.
