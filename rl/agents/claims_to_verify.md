# Claims from two externally generated research reports (user-supplied, NOT trusted)

Verify each against primary sources (papers, repos, docs, our own code). Mark it TRUE / PARTLY / FALSE /
UNVERIFIED, give the source, and say what it means for our bot.

1. generals-bots (strakam, JAX) runs 1.4M to 10M+ env steps/s via jax.vmap/jit with an auto-reset pool,
   and has a `mode="competition"` with castles, deathtouch and the 1200-turn cap.
2. AverageJoe = 15.35M-parameter policy-value transformer (7 layers, 8 heads, FFN 1344, 3x3 patch tokens,
   temporal tokens from 512-step windows of army/land counts), 81.5% win rate on the public ladder,
   trained with PPO + HL-Gauss value loss, top-k advantage filtering and EMA weights.
3. HL-Gauss categorical value loss (Imani & White 2018; Farebrother et al. 2024 "Stop Regressing")
   beats MSE value regression: more stable and better calibrated, and avoids capacity loss.
4. GRPO drops the critic (~50% less training memory) but suffers "advantage collapse" when all G group
   rewards are equal. Claimed fixes: dense multi-check verifier rewards, or PPO+GAE with a critic.
5. RLVR = RL with verifiable (rule-checked) rewards; a game engine is a natural verifier. Claims: it
   develops implicit reasoning; its main risk is reward hacking through partial verifiers. Also RLCR
   (RL with calibration rewards). Does either add anything beyond ordinary outcome-reward RL for a game policy?
6. A depthwise-separable micro-CNN with < 5,000 parameters runs in 10-20 ms per move in pure CPython
   (with flattened weights and sum()/zip loops) on a 21x21 board.
7. Distil the teacher into decision trees and transpile them to pure Python with m2cgen: < 1 ms
   inference; 50 trees of depth 12 is about 800 KiB.
8. gc.disable(), flat bytearray boards and big-int bitboards remove latency jitter and speed up spatial ops.
9. Spawn-distance curriculum (start generals close and widen); a league with ~87% historical checkpoints
   and ~13% self/exploiters.
10. Prior bots: TySayers/generals-io-bot (behaviour-cloning CNN on replays), yilundu/generals_a3c (A3C).
11. The engine talks to bots over stdin/stdout; an illegal move is a silent pass.
12. Self-play with an entropy bonus learns from sparse win/loss rewards without reward shaping.
13. arXiv 2507.06825 ("Artificial Generals Intelligence", Straka & Schmid): what they actually trained
    (BC from human replays, then PPO self-play? network size? compute: ~36 h on one H100?), and how strong it got.
14. The user's tournament note: knockout brackets, "no heuristic algo-based bot will win, it requires RL".
    Give the evidence on both sides (e.g. who won comparable constrained competitions, and the generals.bot
    Marathon/Sprint top bots: neural or heuristic?).
