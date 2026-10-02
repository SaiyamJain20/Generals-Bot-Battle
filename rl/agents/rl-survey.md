# rl-survey report (2 Oct 2026, ~21:30 IST). Research + reading only; one 80 ms micro-benchmark.
Tags: [V] verified from a fetched primary source; [L] verified from local files/data; [I] my inference; [U] unverified.

## 0. Bottom line
- No public result shows a stdlib-only, <=1 MiB, 150 ms policy reaching top-ladder strength through RL. Every strong public RL generals bot
  uses a large net and 10-100x our compute (claims 2, 13).
- RLVR/GRPO/DAPO/RLCR/RLCD are LLM post-training techniques. Only group-relative baselines and a calibrated win-probability head
  transfer, and both are already covered by paired-seed evaluation / ordinary PPO critics (sec 2).
- The premise "no heuristic bot will win" is NOT established. Comparable constrained competitions were mostly won by rule-based bots, and the
  generals.bot Marathon top-4 are of unknown type (sec 5).
- Best use of ~15 h: keep the heuristic as deployed core + guardrail, add a learned part that cannot regress it by construction
  (zero-init modulator tuned by ES, or a learned scorer re-ranking the heuristic's own top-K candidates), gated by head-to-head tests (sec 4).

## 1. Claims verification (rl/agents/claims_to_verify.md)
1. PARTLY. Repo says "10M+ steps/second", vmap, `mode="competition"` with build-your-own castles, deathtouch at 800, 1200 cap, fog
   [V github.com/strakam/generals-bots; L vendor README l.17-45]. "1.4M" is not in their benchmark. Their table (vendor/generals-bots/paper/benchmarks/README.md):
   1 H200 = 45.0M fps at 65,536 envs (21k at 1 env); 128-core CPU = 735k fps at 65,536 envs (9.6k at 1 env); paper quotes 50.7M on H200
   [V arxiv.org/html/2606.23348v1]. Auto-reset pool is real (core/env.py). Us: Ada 2080 Ti probably 1-10M fps [I, unmeasured, needs JAX CUDA on old drivers];
   on the 2-core laptop budget about 20-50k fps [I].
2. TRUE. 15.35M params, 7 layers, d=448, 8 heads, FFN 1344, 3x3 patches, 2 temporal tokens over a 512-step window; PPO, sparse +-1, HL-Gauss
   (128 bins, [-1,1], sigma .04), GAE .9, gamma 1, top-25% advantage filter, EMA .999, 4x H200 for 4 days; 81.5% (815/1000) first ranked games, #1
   [V arxiv.org/html/2606.23348v1, github.com/strakam/AverageJoe]. No BC, no league, shaping "destabilized late training". Us: cheap transferable
   tricks are sparse reward, top-25% filter, EMA, spawn-distance curriculum. The model is nowhere near 1 MiB stdlib.
3. PARTLY. Cross-entropy/HL-Gauss value learning beats MSE and scales (Atari, chess, Q-transformers) [V proceedings.mlr.press/v235/farebrother24a.html];
   HL-Gauss originates in Imani & White 2018 for supervised regression [V arxiv.org/abs/2402.13425]. "More stable" is supported; "better calibrated"
   and "avoids capacity loss" are not the headline claims [I]. AverageJoe uses it but I found no HL-Gauss-vs-MSE ablation. Us: free training-time add-on
   if PPO runs; not a reason to run PPO.
4. PARTLY. Advantage collapse (all group rewards equal gives zero advantage) is TRUE [V arxiv.org/html/2605.21125]; DAPO dynamic sampling resamples
   until groups are mixed [V arxiv.org/pdf/2503.14476]. GRPO drops the value net so memory falls, but "~50%" is a loose number [I]. "Dense multi-check
   verifier rewards" is LLM-specific. Us: in a game the analogue is picking opponents near 50% (PFSP).
5. PARTLY. RLVR = RL on rule-checkable rewards (math/code LLMs); "implicit reasoning" is an LLM claim with no game evidence. RLCR reward is
   1[correct] - (q - 1[correct])^2 (Brier); it fixes calibration (ECE 0.37 to 0.03 on HotpotQA) with no accuracy loss, it does not raise accuracy
   [V arxiv.org/html/2507.16806v2]. Us: neither adds anything over outcome-reward RL; see sec 2.
6. PARTLY. My benchmark: ONE dense 8->8 3x3 conv layer on 21x21 in naive CPython = 80 ms (about 254k MACs) [L this session]. Our dense 8x2 student (about 4 dense convs)
   is roughly 300 ms naive, maybe 60-100 ms hand-optimised [I]. Depthwise-separable at C=8 is about 60k MACs/layer, so 10-20 ms for 2 layers is plausible [I].
   rl-deploy's measurements supersede mine. Us: judge students with the exact deployed arithmetic; dense 8x2 is borderline.
7. PARTLY / doubtful as stated. m2cgen exists and emits pure Python [U github.com/BayesWitnesses/m2cgen]. A full depth-12 tree has up to 4095 internal nodes
   (about 40-60 bytes each, 160-250 KB per full tree), so 50 full trees would be about 10 MB, not 800 KiB; 800 KiB fits only sparse trees [I]. "<1 ms" holds
   per feature vector (about 600 comparisons) but a move scorer runs per candidate (tens to hundreds): 5-50 ms [I]. Us: trees are a viable stdlib
   scorer; budget per candidate.
8. PARTLY. gc.disable() is standard and removes pauses; flat lists/bytearrays and big-int bitboards are plausible for BFS/flood fill but unsourced and
   unmeasured [U]. Jitter matters: a late/malformed reply is a pass + fault, 50 faults forfeit [V www.generals.bot/docs]; our README is stricter (timeout = forfeit).
9. PARTLY. Spawn-distance curriculum TRUE (starts at 4 cells, widens) [V 2606.23348]. The "87% historical checkpoints / 13% self+exploiters" split has NO
   source; AverageJoe states no league (frozen-self self-play) [V]. quant-eagle used a scripted-expander league [L agents_shared/rl-research.md]. Treat 87/13 as invented.
10. TRUE. yilundu/generals_a3c: supervised replay policy + A3C vs bundled opponent [V github.com/yilundu/generals_a3c]. TySayers/generals-io-bot (AlphaGenerals):
    behavioural cloning + augmentation with CNNs [V github.com/TySayers/generals-io-bot]. Old, pre-competition ruleset, no strength numbers.
11. PARTLY. stdin/stdout protocol TRUE [V www.generals.bot/docs]. An engine-illegal move is a no-op (game.py: "such a move is invalid and never executes")
    [L vendor/generals-bots/generals/core/game.py:417]. A late or malformed LINE is pass + fault (50 faults = forfeit) [V docs].
12. TRUE at huge scale, misleading for us. AverageJoe: entropy bonus + sparse +-1 + no shaping [V 2606.23348], but about 260B env steps [L rl-research.md]. The
    earlier paper used potential-based shaping (0.3 land, 0.3 army, 0.4 castle) and says it significantly improved robustness [V arxiv.org/html/2507.06825v1];
    Mattz had to teach castles by hand because their payoff is delayed [V eventwaves.substack.com/p/trying-to-win-an-ai-bot-competition]. At 1e8-1e10 steps, initialise from BC/heuristic.
13. TRUE (details). 2507.06825: BC on 16,320 games of >=70-star players (3 h, 1 H100), then self-play RL 36 h on 1 H100 with opponent pool N=3; U-Net torso
    (Perolat et al. 2022 architecture), shaping + sparse +-1; top 0.003% of the human 1v1 ladder (top 25 players); 54.82% vs Human.exe over 529 games; env about 3,500 fps
    on 12 CPU cores [V arxiv.org/html/2507.06825v1]. Parameter count not stated. Us: the only published BC->PPO recipe; 36 h H100 is about 50x our budget, and
    that agent (zero v3) later lost 0-20 to AverageJoe, which beat Human.exe 20-0 [V 2606.23348].
14. See sec 5. Mixed evidence; and the Marathon was a round-robin, not a knockout.
Correction to earlier note: quant-eagle's repo title says 16 h on 4x RTX 5090, its README body says about 24 h on 8x 5090; deploy limit there is 50 MB with numpy/torch
[V github.com/quant-eagle/generals-competition-rl-bot]. Ladder #10, no weights released.

## 2. RLVR / GRPO / DAPO / RLOO / RLCR / RLCD: what transfers
- RLVR: RL with engine-checkable reward. For a game the engine already is the verifier; there is nothing extra. Reward-hacking via partial verifiers
  does not apply (outcome is exact). Verdict: no benefit.
- GRPO: no critic; baseline = group mean of G samples of the SAME prompt. DAPO: dynamic sampling, clip-higher, token-level loss. Dr.GRPO: drops length
  normalisation. RLOO: leave-one-out baseline.
  * Transfers: group-relative baseline = replay the same map/seed G times, advantage = outcome minus group mean. This removes map luck (18-21 boards,
    random spawns). It is the same idea as our paired-seat A/Bs and as ES with common random numbers. Cost: G x games per update.
  * Does not transfer: token-level losses, length bias, KL to a frozen LLM reference (the analogue is KL-to-BC, sec 3), reasoning emergence.
    Saving a value net matters for 7B models, not for a 6k-parameter policy. With a terminal-only reward over 400-500 turns, a critic + GAE
    (what AverageJoe does) gives better credit assignment than a single group mean.
- RLCR: correctness minus Brier(confidence). Analogue = train a win-probability head with a proper scoring rule (Brier / log loss / HL-Gauss). Possible
  use: choose aggressive vs safe options by calibrated v(s). It does not improve the policy by itself; a PPO critic gives the same thing.
- RLCD ("Jev", TypeSafe): per the earlier board note, proprietary calibrated-decision method for an LLM classifier, not a game method (not re-verified by me).
  No "Jev" appears in any Marathon standing [L agents_shared/rl-research.md].
- Net: keep paired-start/common-random-number evaluation and a calibrated value feature. Drop the rest.

## 3. Methods that fit
Constraint: deploy net must be tiny (about 5-30k params, stdlib); training may use Ada GPUs (JAX env) or ~100 CPUs (python sim, about 1-2 s/game
[L rl-research.md]). About 15 h remain.
- BC warm start + PPO/APPO with KL-to-BC: only published path (2507.06825). KL(pi||pi_BC) stops degenerate self-play drift. But needs a working JAX/GPU
  env on old Ada, feature re-implementation, and 1e8-1e9 steps; our students score 0.35-0.40 vs the tuned pool [L rl/students/eval_s1.txt], i.e. weaker
  than the heuristic, and a KL leash to a weaker teacher caps the result. Stretch goal only, and only with a better teacher (sec 4.1).
- League / PFSP: PFSP picks opponents near 50% win rate, which fixes claim-4 degeneracy. AverageJoe reached #1 with no league, so not necessary at
  large scale; at small scale diversity helps [I]. Use as the opponent sampler (heuristic + BC clones + A9 + zoo; fixed anchors; tune/cma_tune.py --pfsp exists).
- R-NaD (DeepNash) / NFSP: principled for imperfect information, shown on Stratego with huge compute [V arxiv.org/abs/2206.15378]. AverageJoe reached
  superhuman in a fogged game with plain PPO. Skip.
- Policy distillation + quantisation: the only way a big teacher reaches stdlib. Soft-label KL to teacher logits, DAgger on student-visited states,
  int/float lists, test with the exact deployed arithmetic. Top-1 accuracy (0.67) is not strength; error compounds without DAgger.
- HL-Gauss value head: see claim 3; add if PPO runs.
- AlphaZero/MuZero: infeasible. Fog needs belief states/determinisation; our python sim costs about 3 ms/turn [I], so 150 ms is about 30 simulations including net
  cost; net inference alone is 10-80 ms (claim 6); training needs 1e9+ positions. A cheap exact 2-ply tactical search around contacts and the general is the
  useful alternative (agents_shared/strategy-research.md idea #1).
- ES / CMA over a small modulator: handles sparse win/loss, 100 numbers to ship, zero-init = incumbent. Noise: 80 games = SE 0.056 [L rl-research.md].
- Composition: do not stack learners. Pick ONE (ES modulator or BC/DAgger scorer); both reuse the same pool and gate.

## 4. Heuristic as base: concrete evaluation
Facts: heuristic beats all public/heuristic bots 0.75-1.0 except ResBot-BC clones (about 0.4) [task brief]. Real ResBot won the Marathon (84.8-90.7%) [L bot-scout.md]; we have no measured number against real ResBot.
1. BC from heuristic self-play: unlimited, exactly labelled data; DAgger possible with the heuristic as perfect teacher. Caps at the heuristic, so it is a starting point for
   step 3/6, not a product. Risk low, gain about 0 alone.
2. Heuristic as league opponent + gate: already our best gate; stops an RL net that beats RL nets but loses to heuristics. Rule: >=0.53 on >=600 paired-seat games, zero faults
   (rl-research.md). Do regardless.
3. KL/residual policy: pi = softmax(log pi_h + delta), delta zero-init with an L2 penalty. Starts at heuristic strength, cannot regress if gated, cheap to deploy.
   Needs the heuristic to expose per-move scores (it chooses among "options"). Gain small but real (est. +10-30 Elo [I]).
4. RL over high-level options: horizon 50-100 decisions instead of 500, denser credit; fits Mattz's lesson that castle timing needs guidance [V substack]. Needs an options interface
   and fast sim; PPO engineering risk high; ES on option modulators is the cheap form.
5. Heuristic as guardrail/fallback: lethal checks, dt_guard, block/chase, and an act()-level time-budget fallback. Third-party A9 does this (model + tactical resolver + budget
   guard dropping to a legal heuristic at 60% of 150 ms) [L vendor/ext/mortid0_a9/my_bot_cpp/agent.hpp]; hv-nguyeen's bot falls back to its heuristic without weights [L its README].
   Removes the forfeit tail, which dominates in a knockout. Do regardless.
6. Heuristic + learned move scorer: heuristic generates K candidates, a tiny tree-set/net re-ranks (heuristic score as a feature). Net sees 5-20 candidates, not 441x4x2; stdlib trees fit.
   Gain uncertain (0 to +5 pts [I]); supervision from ResBot replays is bounded by clones that only reach 0.4 vs us, so it must add information the heuristic lacks.
Ranking (gain x safety): A = 5+2 (necessary floor); B = 3/4 via zero-init ES modulator (highest gain per risk); C = 6 only if the gate passes.
Not recommended: a pure RL/BC net replacing the heuristic (students 0.35-0.40, 96ch x 8 clones about 0.4, about 1/50 of public training budgets, CPython arithmetic 10-100x slower than numpy/torch deploys).

## 5. Evidence on "no heuristic bot will win, it needs RL"
- Marathon (1 Sep): round-robin, 22 bots, every pair x 10 maps x both sides (20 games/pair), top 10 replayed. Final: 1 ResBot 90.7% (Elo 515), 2 nanomena 85.0%, 3 Kubic 76.3%,
  4 FreeLunch 67.8%, 5 bca 42.6%, 6 Chig, 7 shy, 8 sosynec666, 9 hiems, 10 thor [V www.generals.bot/assets/marathon-2026-09-01.json, re-fetched; ResBot 95.7% in qualifier, 84.8% in deep round].
  Sprint (8 Aug) top-3: ResBot, Kubic, nanomena [V www.generals.bot/assets/sprint-2026-08-08.json]. Only bca (transformer PPO + replay BC) is known neural; no public code or write-up
  for ResBot, nanomena, Kubic, FreeLunch [L bot-scout.md]. So "top bots are RL" is UNVERIFIED. The results pages themselves did not render for me.
- Pro-RL: AverageJoe beat the top two humans 199-70 and Human.exe (heuristic) 20-0 [V 2606.23348]; zero v3 beat Human.exe 54.8% [V 2507.06825]; Mattz: rules "performed poorly",
  PPO best, 6th in Sprint [V substack]. This is RL with big nets and big compute; not evidence about our deploy limits.
- Pro-heuristic: Human.exe was the heuristic SOTA before RL [V 2507.06825]. Rule-based winners in Halite, Kore 2022 (top-3 with forward simulation), Lux S2; Lux S1 won by deep RL;
  Battlecode winners hand-engineered [search summaries of kaggle.com/writeups/zoli800/winning-kaggles-reinforcement-learning-competitio and Battlecode pages: [U] detail, the Kaggle page
  would not load; corroborated by agents_shared/strategy-research.md]. Halite III's ML-bot winner cloned top rule-based bots.
- Format [I]: in single elimination, per-match win prob p compounds (p=0.8 over 4 rounds = 41%), so zero faults/timeouts matter more than a few Elo points. Verify the actual bracket
  and games per match (README says knockout; Marathon was not).
- Verdict: premise not established. Top bots are probably learned or learned+tactics, but a faithful copy is impossible under stdlib/150 ms. The practical comparator is "heuristic + learned prior".

## 6. Recommendation
1. Ship the heuristic with guardrails and a hard time-budget fallback in act().
2. Gate every change on the heuristic-inclusive pool: >=0.53 over the shipped bot, >=600 paired-seat games, zero faults.
3. ONE learner: zero-init ES/CMA low-rank context modulator over heuristic weights (primary); optional DAgger-distilled top-K re-ranker. PPO only as a side run (BC init,
   KL-to-BC, sparse reward, top-25% filter, EMA, PFSP pool) on Ada if the user approves.
4. Drop RLVR/GRPO/DAPO/RLCR/R-NaD/AlphaZero; keep paired-start CRN evaluation and a calibrated value feature.
5. Open: bracket format and games/match; measured heuristic vs real ResBot; Kaggle per-competition detail.
