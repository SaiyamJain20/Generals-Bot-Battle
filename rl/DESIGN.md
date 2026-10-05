# RL design for the Code Bot entry (RL track)

Status: design v1, 2 Oct ~21:15 IST. Deadline 3 Oct 12:00 IST. The heuristic bot (main checkout) remains
the safe submission; this track only replaces it if it beats it under the gates in section 7.

## 0. Official facts (evaluator.zip README + organizer answers relayed by the user, 3 Oct ~01:45)
- Deadline: 3 Oct 2026, 2:00 PM IST.
- Format: a league where each pair plays 2 games with seats swapped (W/D/L = 1/0.5/0). The top 4 then play
  4-game semifinals and a 6-game final; tied playoffs get extra pairs.
- Runtime: 1 core (Sapphire Rapids), shared by both bots, taking turns and paused between decisions.
  150 ms wall per call including observation transport; 10 s for the first call including import.
  2 GiB RAM, at most 32 processes/threads.
- Forfeits: an invalid action code, exception or timeout forfeits; a legal-format illegal move is a pass.
- File: one .py of at most 1 MiB with no separate model files, so any learned weights are embedded
  ("checkpoint size depends on your training" within that cap).
- Open-source bots may not be submitted (the organizers have a list); taking inspiration is allowed.
  External bots here are local sparring partners only.
- The evaluator engine is byte-identical to our vendored pin, and boards are exact-size (no padding).

## 1. Grounded starting facts (measured or sourced; see rl/agents/rl-survey.md)
- Deployment wall: stdlib Python, <= 1 MiB, 150 ms/move on one shared CPU.
  - One dense 8->8 3x3 conv layer over 21x21 is ~80 ms in naive pure Python (rl-survey), so dense CNN
    policies must be tiny. rl-deploy is measuring the optimised ceiling.
- Size/strength curve (our measurements, 30 games per opponent, maps 700+):

  | policy | params | val top-1 | vs heuristic tune_base12g | vs t1c | vs expander |
  |---|---|---|---|---|---|
  | BC 96x8 clone, argmax | 1.36 M | 0.64 | 0.30 | 0.53 | 0.92 |
  | BC 96x8 clone, temp 0.3 (older eval) | 1.36 M | 0.64 | ~0.63 | ~0.6 | - |
  | BC student 12x1, argmax | 6.8 k | 0.674 | 0.17 | 0.23 | 0.67 |
  | BC student 8x2, argmax | 5.4 k | 0.669 | 0.20 | 0.22 | 0.70 |

  Top-1 accuracy does not track strength: deployable-size pure imitation is far weaker than the heuristic.

  Follow-up (Ada GPU students trained on the full ResBot shards, temperature 0.3, 60 games per opponent, maps 700+):

  | student | params | pure-Python time | vs tune_base12g | vs t1c | vs zoo_flash | vs expander |
  |---|---|---|---|---|---|---|
  | 12x1 | 6.8 k | 11.6 ms p50 / 20.7 p99 (measured) | 0.167 | 0.10 | 0.50 | 0.78 |
  | 16x2 | 14.7 k | ~28 ms (est.) | 0.167 | 0.20 | 0.57 | 0.90 |
  | 24x2 | 28.6 k | ~58 ms (est., too slow) | 0.30 | 0.25 | 0.54 | 0.87 |
  | 32x3 | 65.7 k | too slow | 0.267 | 0.40 | 0.65 | 0.93 |

  - Track B PPO (12x1 init, Ada job 3165), argmax vs tune_base12g over 48 games: 0.208 at iteration 1 (BC init),
    0.083 at iteration 21 and 0.083 at iteration 41. RL fine-tuning of the tiny net is not improving it as
    configured.
  - Conclusion: a fully neural policy that fits the deployment budget plays at ~0.17 vs the heuristic.
    Track A is the path to an RL bot that beats it.
- RLVR / GRPO / DAPO / RLCR / RLCD are LLM post-training methods.
  - What transfers: an outcome reward checked by the exact engine (already "verifiable"), group-relative
    baselines from replaying the same map from both seats (paired seats / common random numbers),
    a KL penalty to a reference policy, and a calibrated win-probability value head.
  - Prior generals RL agents (zero, AverageJoe, quant-eagle) used 10-100x our compute and networks far
    beyond the 1 MiB stdlib budget; AverageJoe used plain PPO without a league.

## 2. Two tracks
- **Track A (main): Residual Option Policy (RO-PPO).** RL learns which of the heuristic's candidate options to
  execute each turn. Its prior is exactly the heuristic; deterministic tactical guardrails stay outside it.
  - Cheap in pure Python: a small MLP over ~40 features x <= 8 candidates.
  - Starts at heuristic strength, so every improvement is relative to the real baseline.
- **Track B (research): neural micro-policy (BC -> PPO).** A tiny conv policy initialised from the ResBot BC
  student and fine-tuned by PPO in self-play plus league.
  - Kept only if it closes the gap; expected to lose to Track A within this deadline.

## 3. Track A policy
- Decision point: every turn where `macro()` builds its option list
  `options = [(score, action, label)]` (labels capture, garrison, build, scout, home, launch, cyc_attack, cyc_castle).
  - Guardrails stay deterministic and run before it:
    win_now, dt_guard, urgent_defense, intercept, opening (t < 50), endgame (t >= fortress_turn), try_kill.
  - Turns without a choice (0 or 1 option) are not decisions.
- Logits for candidate i: `z_i = alpha * s_i + u_label(i) . phi(s) + v . psi_i`.
  - s_i is the heuristic score after the modulator.
  - phi(s): ~24 global features. They include turn phase, log army/land ratios, my army, need_g / A[g],
    threat_eta, the largest tracked threat / A[g], enemy castles, egen known, belief distance, cycle mode,
    turns since the last launch and land-bonus phase t%50.
  - psi_i: ~8 option features: source army, target distance, path length, own/enemy/neutral target,
    whether it moves the general, gather budget used.
  - Initialisation: alpha = alpha0 (e.g. 4), u = v = 0. Then pi_ref = softmax(alpha0 * s) almost always
    picks the heuristic's argmax, and the trained policy starts there.
- Deployment: argmax of z (deterministic) or temperature 0.1 sampling with a seeded RNG. A few hundred
  multiply-adds per turn.
- Critic V(s) (training only): MLP on phi(s) giving a win-probability logit (Bernoulli, BCE) or an
  HL-Gauss histogram over [-1, 1] (21 bins).

## 4. Reward (must be correct)
- Terminal outcome from the exact engine (sim/engine.py, which matches the pinned engine; tests/test_parity.py):
  win +1, loss -1, draw at 1200 = r_draw (default -0.1, configurable). A draw in a knockout is at best a
  replay, so slightly negative pushes toward decisive play without dominating.
- Potential-based shaping (policy-invariant, Ng, Harada & Russell 1999).
  - On the transition from decision k to k+1, spanning tau_k env turns:
    `F_k = gamma^tau_k * Phi(s_{k+1}) - Phi(s_k)`, with `Phi(terminal) = 0`.
  - `Phi(s) = beta * (0.5 * tanh(log(my_army/opp_army)) + 0.5 * tanh(log(my_land/opp_land)))`.
    These four counts are exact in every observation (always given), so there is no fog leakage.
  - beta starts at 0.2 and is annealed to 0 by mid-run, so the final objective is pure win/loss.
- Semi-MDP discounting: decisions are irregular in time, so each transition is discounted by
  gamma^tau_k (gamma = 0.998 per turn).
  - `delta_k = R_k + gamma^tau_k * V(s_{k+1}) - V(s_k)`.
  - GAE: `A_k = delta_k + (gamma^tau_k * lambda) * A_{k+1}`, with lambda = 0.95.
  - R_k is the outcome on the final transition (0 otherwise) plus F_k.
- Variance reduction (the transferable part of GRPO): every map is played from both seats against the
  same opponent. The paired-seat mean outcome is logged as a group baseline, and the critic is trained on both.

## 5. Optimisation
- PPO with clip 0.2, 3-4 epochs per batch, minibatches of 4096 decisions, Adam at lr 3e-4 (2e-4 for the critic).
- Loss: entropy bonus 0.003, plus `beta_KL * KL(pi || pi_ref)` with beta_KL = 0.02, decayed to 0.005.
  pi_ref is the heuristic softmax, as in RLHF / RLVR's KL-to-reference.
- Policy learning rates: alpha is frozen for the first iterations, then trained.
- League (PFSP): each game draws its opponent with weight proportional to (1 - winrate)^2 over a running
  window. Pool:
  - frozen self snapshots (every N updates, last 6);
  - the base heuristic tune_base12g, plus the newest heuristic from the main checkout when it is final;
  - t1c and c_a2es3;
  - zoo_* bots;
  - external public bots (ext_sentinel10, ext_hvn, ext_juraj35, ext_boss, ext_superbot, ...), local only;
  - BC clones (bc_resbot96, bc_resbot128, bc_nanomena96, bc_kubic96);
  - plus any new public bots found (bot-finder).
- Maps: training uses data/maps.jsonl offsets 100000+ (wrapping is modulo 20k, so offsets 10000-19999).
  Evaluation uses data/maps_fresh.jsonl 1800-1999, which no decision has used.

## 6. Engineering
- Rollouts: multiprocessing workers on our exact simulator. Each worker plays full games and records, per
  decision: phi, psi for each candidate, chosen index, logp under the behaviour policy, logp_ref, tau,
  and the shaping potentials.
- The learner is torch CPU (tiny networks). It writes new weights to JSON; workers reload them each
  iteration. The bot's inference is the same pure-Python function in rollouts and in deployment, so
  features and inference are identical in both.
- Throughput target: about 2-4 games/s on 12 laptop cores (the heuristic is 5-10 ms/move), i.e. 7-14k games
  or ~3-5M decisions per hour. That is enough for a policy this size. Ada CPUs would be 5-8x more (needs user approval).

## 7. Evaluation and gates (paired seats, zero forfeits)
- Map slices (data/maps_fresh.jsonl).
  - Already used for decisions (do not use for final claims): 0-49, 400-449, 500-549, 1000-1049, 1100-1129,
    1200-1229, 1500-1549, 1700-1749 (F2 selection) and 1800-1849 (F2 confirmation).
  - Interim RL checkpoint evals: 1550-1699.
  - Final comparison: 1850-1949.
  - Last unbiased check: 1950-1999 (touch once).
- Baseline is F2, the main session's final heuristic: bots/versions/F2.py.
- Gate 1: >= 0.53 vs tune_base12g over >= 400 games, and not worse than -0.03 vs any of t1c, zoo_mixed, rusher.
- Gate 2: the full opponent matrix (heuristics, zoo, BC clones, external bots) >= the heuristic's matrix on
  pooled score and on worst-case opponent.
- Gate 3: subprocess timing at 150 ms with a 2x slowdown, p99 < 60 ms, and tools/check_submission.py passes.
