# rl-train report: rl/ppo_cpu.py (CPU PPO fine-tune from BC)

Status: pipeline works end-to-end locally (2 workers, spawn). Not run at scale, not tuned; no Ada access used.

## What it is
`rl/ppo_cpu.py`: learner + N spawn-workers. Workers play whole games on sim/engine.py with the deployment features
(learn/bc_features.py Tracker/legal_mask). Learning policy sampled from masked logits at T=1. League: `self` (latest, both seats,
2 trajectories/game), `snap:*` (frozen past policies; `snap:bc` = BC init always kept), heuristic/zoo/expander/hunter bots
(fresh module per game; default list = tune_base12g, t1c, expander, hunter, zoo_*), optional `--bc-clones`.
Opponent choice: category mass self-w/snap-w/bot-w (0.2/0.15/0.65); inside snaps and bots, PFSP weight (1-EMAscore)^1.5+0.1.
Seats alternate; maps random from data/maps.jsonl[map-offset : +map-count] (default 1000..16000; eval maps 0.. are disjoint).
Reward: +1/-1 at end, `--draw-reward` (default -0.2) on draw/truncation; optional potential shaping `--shape` (default 0.1, annealed to
`--shape-final`) on phi = .5 tanh(log land ratio) + .3 tanh(log army ratio) + .2 min(own castles,4)/4, in the policy-invariant form
gamma*phi' - phi (phi=0 after the last step).
Learner: GAE (gamma .998, lam .95), clip .2, entropy bonus (annealed), value MSE on v=tanh(l/2)=2*sigmoid(l)-1 from the BC win-logit head,
KL(pi||pi_BC) penalty (full masked distribution, coef annealed), grad clip, early-stop on approx-KL, `--vf-warmup` iters with pg off
(value fit only). BN is kept in eval mode (running stats frozen) in rollouts and learner. Linear annealing is driven by
max(iter/iters, elapsed/time-limit-h). Checkpoints `ckpt_<it>.pt` / `last.pt` in {"state","ch","blocks"}; `train_state.pt` for `--resume`.
Pipelining (`--pipeline 1`, default): workers generate batch i+1 with pre-update weights while the learner updates on batch i
(1-update-stale behaviour policy; PPO ratio uses stored behaviour logp). `--pipeline 0` = fully synchronous.
Eval: every `--eval-every` iters (and at start unless `--no-eval0`): argmax policy vs `--eval-bot` (tune_base12g) on fixed maps, both seats.
Log `<out>/log.jsonl` per iter: steps, worker_steps_per_s, per-opponent winrate/n, pg/vf/ent, kl_bc, agree_bc (argmax agreement with BC on a
2048-step probe), kl_drift/agree_prev (same probe before vs after the update), approx_kl, clipfrac, update_s, per-kind us/step
(nn/feat/env/opp), worker max RSS, eval.

## Measurements (laptop, load average ~18 on 16 cores from other sessions => pessimistic; 2 workers, ResBot_12x1, 6.9k params)
- Per worker: 250-340 game turns/s (turn = both seats observed + featurised + stepped).
  Policy-vs-bot turn: nn forward ~1.7 ms (batch 1), feat (2x observe + tracker + mask) ~1.1 ms, env.step 0.06 ms, opponent act ~0.65-0.9 ms avg
  (tune_base12g 1.3 ms, t1c 1.0 ms early game; zoo/simple bots cheaper). Self-play turn: nn 2.1 + feat 1.5 ms but 2 samples/turn.
  Snapshot-opponent turn: nn 2.9 + feat 1.6 ms. Heuristic opponents are cheap; torch forward and feature code dominate.
- Worker RSS ~510 MB each (torch import dominates). Learner RSS grows with batch (~8 KB/step: 85k steps ~0.7 GB, ~2 GB peak with concat copies).
- Learner update (2 threads, loaded): ~1 ms/sample/epoch (fwd+bwd+BC fwd); 14k samples, 1 epoch: ~15 s. The learner, not rollouts, is the bottleneck.
- Games vs bots last ~300-700 turns.
- Drift check (one update from BC, 1 epoch, first update): lr 1e-6 -> argmax agreement with BC 99.95%, KL 1e-6;
  lr 1e-5 -> 98.9-99.4%, KL 2.5e-4; lr 3e-5 -> 96.3%, KL 2e-3 (2nd update: 98.6% vs previous). BC is not destroyed by one update.

## Extrapolation to 64 Ada cores x 6 h (assumes an Ada core ~ laptop core; unverified)
Split 40 workers + 24 learner threads. Workers: 40 x ~200 turns/s ~ 8k turns/s ~ 9k samples/s. Learner: assumed ~0.2-0.3 ms/sample/epoch at 24 threads
(thread scaling of tiny convs is sublinear: measure) => ~4-5k samples/s at 1 epoch => learner-bound, ~90-110M samples (~150-200k trajectories) in 6 h,
~1000 updates of ~85k samples (128 games/iter, ~17-20 s/iter). If update_s >> rollout_wait_s in the log, lower workers / raise learner threads;
if rollout_wait_s is large, the reverse. Memory request >= 40 GB (40 x 0.5 GB workers + learner).

## Launch (64-CPU node; coordinator sets Ada cache env vars from CLAUDE.md first)
```
PYTHONPATH=vendor/generals-bots:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python rl/ppo_cpu.py --init rl/students/ResBot_12x1.pt --out runs/ppo_12x1 \
  --workers 40 --learner-threads 24 --games-per-iter 128 --iters 100000 --time-limit-h 5.75 \
  --epochs 1 --mb 2048 --lr 2e-5 --lr-final 1e-5 --kl-coef 0.2 --kl-final 0.05 --ent 0.002 --ent-final 0.0005 \
  --draw-reward -0.2 --shape 0.1 --shape-final 0 --gamma 0.998 --lam 0.95 --vf-warmup 5 \
  --self-w 0.2 --snap-w 0.15 --bot-w 0.65 --snap-every 20 --max-snaps 8 \
  --ckpt-every 20 --eval-every 20 --eval-games 48 --target-kl 0.03
```
Staged tree needs data/maps.jsonl, bots/, sim/, learn/, vendor/generals-bots, rl/students; Python 3.12 with numpy + CPU torch. `--resume` continues.
Watch in log.jsonl: eval score vs tune_base12g (baseline at iter 1), kl_bc (keep < ~0.05), per-opponent winrate, early_stop frequency.

## Caveats
- Hyperparameters untuned; unexercised at 40 workers; first 5 iters are value warmup (pg off).
- Value head reuses BC win-logit via tanh(l/2); no HL-Gauss.
- Draw/truncation treated as terminal (time is in the features), no bootstrapping.
