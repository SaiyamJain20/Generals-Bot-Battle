# Generals Competition — Research & Planning Guide

A single reference for the team and for coding agents working on our generals.io bot for the **Bot-Battle**. It covers what the challenge is, how the engine really behaves, where to find code and prior art, which algorithms and papers are relevant, and the build plan.

**Legend used throughout**

| Mark | Meaning |
|---|---|
| ✅ | Verified directly from the engine source or the official competition docs |
| 📚 | From general knowledge (papers, courses). Well known, but double-check exact titles, years and links before citing |
| ❓ | Open question, still to confirm |

---

## Table of contents

1. [Competition at a glance](#1-competition-at-a-glance)
2. [Game rules (competition ruleset)](#2-game-rules-competition-ruleset)
3. [Engine-verified mechanics (read this carefully)](#3-engine-verified-mechanics-read-this-carefully)
4. [Bot interface, sandbox and constraints](#4-bot-interface-sandbox-and-constraints)
5. [The core challenges](#5-the-core-challenges)
6. [Key numbers and economics](#6-key-numbers-and-economics)
7. [Prior art and code to study](#7-prior-art-and-code-to-study)
8. [Approaches and algorithms](#8-approaches-and-algorithms)
9. [RL concepts primer and glossary](#9-rl-concepts-primer-and-glossary)
10. [Reading list (papers)](#10-reading-list-papers)
11. [Courses, blogs and tools](#11-courses-blogs-and-tools)
12. [Build plan](#12-build-plan)
13. [Open questions](#13-open-questions)
14. [Notes for coding agents](#14-notes-for-coding-agents)

---

## 1. Competition at a glance

| Item | Detail |
|---|---|
| Organizers | Equilibre Technologies with ÚFAL (Charles University) |
| Goal | Build a bot that wins 1v1 generals.io games under a custom ruleset |
| Site | https://www.generals.bot (rules at `/rules`) |
| Engine | https://github.com/strakam/generals-bots (JAX simulator, same rules as graded matches) |
| Ranking | Elo, recomputed after every batch. Provisional `?` until most of the field is played |
| Matchmaking | New submissions play similar-skill opponents within minutes, plus a round robin every 4 hours |
| Checkpoints | **Sprint** (Aug 8, one week in) and **Marathon** (end of August); top of the board goes to a head-to-head tournament. ❓ Confirm the current schedule on the site |
| Fair play | One account per person or team. No sharing code, weights or strategies between accounts. Sandbox scanned. Don't try to deanonymize or interfere with other bots |
| Our language | Python (CPython 3.12.10 in the sandbox) |

---

## 2. Game rules (competition ruleset)

### Board ✅
- Rectangular, each side independently **18–21** (324–441 tiles). Read H and W from the handshake, never hard-code.
- Tile types: plains, mountains (impassable, permanent), castles (built by players), generals.
- Mountains cover about 20–24% of the board (66–105 tiles).
- **No castles at the start.** Every castle is built by a player.
- Generals are at least **17 BFS steps** apart over plain tiles, and a path between them always exists.
- Every map is freshly random. No fixed seeds.

### Turns and actions ✅
Each turn, each player sends one line: `kind row col dir split`.

| Action | Format | Notes |
|---|---|---|
| Move | `0 r c d s` | `d`: 0 up, 1 down, 2 left, 3 right. `s=0` sends all but 1; `s=1` sends half (floor) |
| Pass | `1 0 0 0 0` | |
| Build castle | `2 r c 0 0` | Replaces your move that turn |

Invalid moves and builds are silent passes, with no fault. Invalid means: source not yours, off board, into a mountain, or source has ≤ 1 army.

### Castles ✅
- Build on any **plain cell you own** (not a general, not an existing castle).
- **Price = 35 + Σ max(0, 14 − 2·d)** over each of *your* structures (your general and your castles), where d is Manhattan distance. Structures 7+ tiles away add nothing. Enemy structures never affect your price.
- The price is paid from the army on that cell; the remainder stays. Building with exactly the price leaves the castle at **0 army**.
- Production starts immediately: +1 every other turn.
- Captured castles produce for the capturer and count toward the capturer's surcharge.

| Distance to nearest own structure | 1 | 2 | 3 | 4 | 5 | 6 | ≥7 |
|---|---|---|---|---|---|---|---|
| Price (single structure nearby) | 47 | 45 | 43 | 41 | 39 | 37 | 35 |

Surcharges stack. Example: distance 2 from both your general and a castle costs 35 + 10 + 10 = 55.

### Army growth ✅
- Generals and castles: +1 **every other turn** (on even turns).
- Every owned cell: +1 **every 50 turns**.
- Growth is applied **after** all moves and fights in a turn.

### Combat ✅
- Moving onto an enemy (or neutral) cell subtracts armies. The attacker takes the cell only with **strictly more** army. On a tie, the defender keeps it.
- Moving between your own cells merges armies.
- Capturing the enemy general wins instantly.

### Visibility ✅
Fog of war. You see only the 3×3 area around each tile you own (including diagonals). Scouted tiles fade back to fog when you leave.

### Game end ✅
- **Win:** capture the enemy general.
- **Deathtouch (turn ≥ 800):** any valid move that executes onto the enemy general wins, regardless of its army. The only defense is capturing the attack's source cell that same turn from a third tile.
- **Draw:** no winner by turn 1200. A hard draw, worth half points. Passive play is punished on Elo.

---

## 3. Engine-verified mechanics (read this carefully)

These details come from reading `generals/core/game.py`, `generals/modifiers/build_castles.py` and `generals/modifiers/deathtouch.py`. None of them are spelled out on the rules page, and several matter a lot for tactics. ✅

### 3.1 Turn resolution order
1. **Builds resolve first**, before any move (for both players). Build actions are then treated as passes for the rest of the turn.
2. **Moves resolve sequentially, not simultaneously**, in a priority order (`_determine_move_order`):
   1. **Defensive moves first.** A move whose destination is a tile the mover already owns.
   2. Then ordinary attacks.
   3. **Moves onto any general tile last**, including a friendly merge onto your own general.
   4. Within the same category: **larger moving army first**.
   5. Remaining ties: player order, reversed on odd turns.
   6. **Chase rule:** if the enemy's move enters the tile your move starts from, your move waits until theirs has resolved. The exception is a head-on swap (two moves into each other's source tiles).
   7. Passes last.
3. Time advances, then growth is applied. On the turn a game is decided, no growth is paid.

### 3.2 Army is read at execution time
A move's army is computed when that move executes, from the board as it stands at that moment. If your source was hit earlier in the order, you send what's left. If it was captured, your move becomes an invalid pass.

### 3.3 Tactical consequences
- **Reinforcing always beats an incoming attack in the same turn.** Moving army onto your own threatened tile is a defensive move, so it lands before the enemy's attack.
- **You can't dodge by moving the target away.** If the enemy attacks tile X and you try to move out of X, the chase rule makes you wait, so their attack lands first.
- **Attacks on generals resolve last**, after everything else that turn.

### 3.4 Growth timing
- Your general starts with **1 army**.
- Structures grow when `time % 2 == 0`.
- All owned cells grow when `time % 50 == 0`.

### 3.5 Fog encoding
- Visible area: the 3×3 around every owned tile.
- In fog, **mountains and castles both show as type 5** (structure-in-fog). We have to remember what we saw earlier to tell an enemy castle from a mountain.
- The enemy general in fog shows as plain fog (type 0).
- Army in fog reads as 0, and owner in fog reads as 0.
- `opp_land` and `opp_army` totals are **always** given, even through fog.

### 3.6 General capture
- The capturer wins and receives all the loser's tiles with army halved (rounded up). Irrelevant for a 1v1 win, but it shows up in simulations.
- **Mutual capture on the same turn is a draw**, at any turn.

### 3.7 Deathtouch details (turn ≥ 800)
- A "touch" is a move that is valid at its own slot in the resolution order and whose destination is the enemy general. The source must have at least 2 army so at least one unit moves.
- Defense: capture the attacker's source from a third tile. That capture is an ordinary attack and resolves before the touch, which is an attack on a general and so goes last. If the source still has ≥ 2 army afterwards, the touch executes and the attacker wins.
- Counter-attacking the source from your general also resolves before the touch, but only helps if it strips the source.
- Both players touching on the same turn is a draw.
- A castle built next to the enemy general is not a touch.

### 3.8 Useful engine tests to read
`tests/test_move_order.py`, `tests/test_deathtouch.py`, `tests/test_build_castles.py`, `tests/test_mutual_capture.py`, `tests/test_rules_coverage.py`. These are the ground truth for corner cases and good templates for our own tests.

---

## 4. Bot interface, sandbox and constraints

### 4.1 Protocol ✅
```
startup:     "player_id H W"
each turn:   "turn my_land my_army opp_land opp_army"
             H lines of W ints   # type:  0 fog, 1 plain, 2 mountain, 3 castle, 4 general, 5 structure-in-fog
             H lines of W ints   # owner: 0 neutral/unknown, 1 me, 2 opponent (always perspective-relative)
             H lines of W ints   # army
reply:       one line "kind row col dir split", then FLUSH
game over:   stdin EOF (no winner line)
```

Minimal skeleton:
```python
import sys
_, H, W = map(int, sys.stdin.readline().split())
while True:
    line = sys.stdin.readline()
    if not line:
        break  # EOF: game over
    turn, my_land, my_army, opp_land, opp_army = map(int, line.split())
    grids = [[list(map(int, sys.stdin.readline().split())) for _ in range(H)] for _ in range(3)]
    types, owner, army = grids
    sys.stdout.write("1 0 0 0 0\n")
    sys.stdout.flush()
```

### 4.2 Time and faults ✅
- **150 ms** wall-clock per move. The first move gets 10 s.
- A late, missing or malformed reply becomes a pass plus one fault. **50 faults forfeits.** A crash or exit forfeits immediately.
- Stale replies are discarded, never used on a later turn.
- Always write exactly one line per observation.

### 4.3 Hardware ✅
One dedicated CPU core per bot, a hard **2 GB** memory cap (exceeding it is a crash and a forfeit), no GPU, no network ever (including during `build.sh`).

### 4.4 Submission ✅
A zip containing `run.sh` (required) and optionally `build.sh`, run once offline at intake.

| Limit | Value |
|---|---|
| Zip size | ≤ 50 MB |
| Unpacked | ≤ 512 MB, ≤ 10,000 files |
| Writable | `$HOME` (use it for on-disk JIT caches) |

Python `run.sh`:
```bash
#!/usr/bin/env bash
exec python -u main.py
```

### 4.5 Pre-installed Python libraries ✅
numpy 2.4.6, scipy 1.18.0, pandas 3.0.5, scikit-learn 1.9.0, **torch 2.13.0 (CPU)**, **jax 0.11.0**, numba 0.66.0, networkx 3.6.1, safetensors 0.8.0, gymnasium 1.3.0.

Not installed: TensorFlow, CUDA builds, torchvision/torchaudio, RL frameworks, **Equinox**, Flax, Optax. Extra pure-Python dependencies must be vendored as manylinux CPython 3.12 wheels in `./wheels/` and installed in `build.sh` with `pip install --no-index --find-links=./wheels -r requirements.txt`.

Startup costs: `import torch` takes about 2.3 s and the full stack about 3 s of the 10 s first-move budget. JIT caches from `build.sh` don't survive in memory, so persist them to disk inside the tree.

### 4.6 Local testing ✅ / ❓
```bash
python competition/matchup.py botA/run.sh botB/run.sh --mode competition [--gui]
```
❓ The `competition/` folder (`matchup.py`, `expander_python`, `expander_cpp`, `expander_rust`) is **not** on the public `master` branch as of our check. Get it from the dashboard, another branch, or Discord. Otherwise we build our own runner on `GeneralsEnv` with the competition preset (castle building, deathtouch at 800, fog, truncation 1200, 18–21 boards padded to 21×21).

**Training vs. graded difference:** the training environment pads every board to **21×21 with mountains**; graded matches use the exact 18–21 rectangle.

### 4.7 Common pitfalls ✅
Forgetting to flush; not exiting on EOF; hard-coding the board size; slow imports on the first move; any exception in the main loop (wrap everything and fall back to a safe pass).

---

## 5. The core challenges

**Long horizon and sparse reward.** Games last up to 1200 turns and the only true outcome is win, loss or draw at the end. For RL this makes credit assignment hard. For heuristics it means early-game tempo errors only show up hundreds of turns later.

**Partial observability.** Fog hides the enemy general, their army and their castles. Castles and mountains look identical in fog. A good bot keeps a memory of the whole map (last seen type, owner, army and time for every cell) and infers the rest. Useful inference signals: the enemy's total land and army every turn, the 17-step minimum distance between generals, the direction enemy tiles appear from, and army jumps that reveal castle builds (sudden ~35+ drops) or fights out of sight.

**A large, mostly invalid action space.** Each turn offers roughly H·W·4·2 moves plus H·W builds plus a pass, around 4,000 options. Almost all are invalid or pointless in any given state.

**One action per turn.** Early on army is the bottleneck; later the single move per turn is. Army spread over many tiles is hard to use, so *gathering* (merging armies along a tree toward one tile) is a core skill, as in human generals.io play.

**Castle economics.** With no neutral cities, the whole mid-game economy depends on *when*, *where* and *how many* castles to build, balanced against exposure (a fresh castle can sit at 0 army and fall to a single unit).

**The deathtouch endgame.** From turn 800 any adjacent tile with ≥ 2 army is a lethal threat. Games shift to hunting the enemy general and keeping your own general's surroundings clear.

**Draw avoidance.** A draw is half a point, so the bot has to force a result before 1200.

**Tight runtime.** 150 ms on one CPU core in Python. Neural network inference and any search must fit, with margin, every single turn.

**Non-stationary opposition.** The field changes as people resubmit. The bot must be robust, not just tuned to beat one opponent.

---

## 6. Key numbers and economics

| Quantity | Value | Notes |
|---|---|---|
| General or castle production | 0.5 army/turn | +1 on even turns |
| Tile production | 0.02 army/turn | +1 per 50 turns |
| One castle ≈ | 25 tiles | in production |
| Castle payback (price 35) | ~70 turns | ignoring defense and gathering costs |
| Castle payback (price 47) | ~94 turns | adjacent to a structure |
| Min general distance | 17 BFS steps | at 1 tile per turn, ≥ 17 turns for a direct walk |
| Deathtouch start | turn 800 | |
| Hard draw | turn 1200 | |

Expansion early costs 1 army per tile, and with only 0.5 army per turn income the opening is army-limited. Simulating opening build orders (expand vs. gather vs. first castle timing) in the local engine is a cheap way to find a strong opening.

---

## 7. Prior art and code to study

### 7.1 The engine: `strakam/generals-bots` ✅
https://github.com/strakam/generals-bots

| File | Why read it |
|---|---|
| `generals/core/game.py` | State, move validation, resolution order, growth, observations, fog |
| `generals/modifiers/build_castles.py` | Castle pricing (`build_cost_grid`) and build resolution |
| `generals/modifiers/deathtouch.py` | Exact deathtouch and mutual-capture semantics |
| `generals/core/env.py` | `GeneralsEnv`, including the competition preset |
| `generals/core/grid.py` | Map generation (sizes, mountain density, general distance) |
| `generals/agents/*.py` | Simple baseline agents (random, expander, harvester, hunter) |
| `tests/` | Ground truth for corner cases |
| `examples/_experimental/ppo/` | An experimental PPO setup |

Associated paper ✅: **Straka & Schmid (2025), "Artificial Generals Intelligence: Mastering Generals.io with Reinforcement Learning,"** arXiv:2507.06825. Read this first.

### 7.2 Average Joe: `strakam/AverageJoe` ✅
https://github.com/strakam/AverageJoe

Its README describes it as the first superhuman generals.io bot, trained from scratch with self-play RL: 81.5% wins in its first 1,000 ranked games and #1 on the generals.io 1v1 ladder. It's written by the engine author and built on JAX + Equinox, about 3,400 lines.

| Component | What it does | File |
|---|---|---|
| Policy-value transformer | Board as 3×3 patches, plus temporal tokens for a short army and land history. Pre-norm self-attention. Per-cell move logits, and a distributional (HL-Gauss) value head | `networks/transformer.py` |
| Self-play PPO | One network plays both sides. GAE, top-k advantage filtering, EMA weights for evaluation | `train/ppo.py`, `train/rollout_selfplay.py` |
| "Magnet" prior | A lightweight heuristic action distribution (favoring expansion and captures) used as a KL anchor | `train/magnet.py` |
| Reward | Win/lose reward | `train/rewards.py` |
| Released config | `L_7d_gae90`: depth 7, embed 448, 8 heads, patch 3, bf16, 512 envs × 512 steps, EMA decay 0.999 | `configs/custom/L_7d_gae90.yaml` |
| Curriculum | Generals distance grows in stages (2–6 → 4–8 → 6–13 → 11–17 → 17–28), advancing at 60% eval win rate | same config |

**How we use it**
- It's a proven recipe on this exact engine, but trained on the **standard** ruleset (neutral cities, no building, no deathtouch, 2048-turn cap, 17–23 boards). The weights won't transfer directly. Castle building adds a new action type, and the endgame is different.
- ⚠️ **The repo has no LICENSE file**, which means default copyright. The competition also requires every submission to be the account's own work. **Ask on the competition Discord before reusing its code or weights.** Studying its design and re-implementing it ourselves is the safe default.
- Expect many competitors to follow this recipe. Our edge must come from ruleset-specific adaptation (Section 12, Phase 3).

### 7.3 Other generals.io bots ✅ (repos exist; quality varies)
- `harrischristiansen/generals-bot`: a long-standing heuristic client.
- EklipZ bot: a well-known strong heuristic bot on the original ladder; a copy exists at `69hhhh/generals-bot-eklipz`. Useful for gather, expand and defense ideas.
- Others: search GitHub for "generals.io bot". These are written against the live generals.io API, so treat them as idea sources, not drop-in code.

### 7.4 Related game-AI competitions 📚
Kaggle's **Lux AI** (Seasons 1–3), **Halite**, and **Kore** are grid-based resource and territory games with bot submissions. Their top-solution writeups on Kaggle's discussion forums are some of the most practical material available, including U-Net policies, imitation of top replays, and self-play PPO/IMPALA.

---

## 8. Approaches and algorithms

### 8.1 Heuristic (rule-based) bot
Hand-written logic: map memory, BFS distances, opening expansion, gather trees, castle-building rules, threat detection, and a general-hunting attack.
- **Pros:** fast to build, explainable, reliable, cheap at runtime. Gets us on the leaderboard and gives us a benchmark and a teacher.
- **Cons:** limited ceiling; hard to hand-tune dozens of interacting parameters.

### 8.2 Parameter tuning without RL
Treat heuristic parameters (castle timing, gather sizes, attack thresholds) as a black-box function of win rate from local self-play. Optimize them with **Bayesian optimization (Optuna)** or **CMA-ES (pycma)**. Cheap, no GPU needed, often large gains.

### 8.3 Search
- **Shallow tactical search:** enumerate our candidate moves against predicted enemy replies for 1–3 turns, using the exact resolution order. Ideal for defense, capture races and deathtouch.
- **MCTS**, as in AlphaZero: learned policy and value networks guide a tree search. Fog and simultaneous moves complicate it. Variants include *determinization* (sample the hidden state, then search) and *Information Set MCTS*.
- **Simultaneous-move search:** both players act each turn, so plain minimax isn't exact. Simultaneous-move MCTS or decoupled UCT handles it.
- **Gumbel MuZero-style planning** works with very few simulations, which suits our 150 ms budget.

### 8.4 Reinforcement learning (self-play PPO)
Train a policy and value network by playing against itself in the fast JAX environment on GPU. This is the proven path for this game (Average Joe). The key ingredients are in Section 9.

### 8.5 Imitation learning (behavior cloning)
Supervised learning to copy a teacher: our heuristic bot, or later our own strong RL checkpoints. It gives RL a competent start instead of random play. **DAgger** fixes the drift problem (the student reaching states the teacher never visited). **Kickstarting** adds a decaying "match the teacher" loss during RL.

### 8.6 Hybrid / hierarchical
The network chooses high-level goals (expand, gather to X, build a castle at Y, attack toward Z) and scripts execute them. This shrinks the action space and horizon. Alternatively, a flat network policy runs with a **hand-written safety layer** that overrides it for guaranteed tactics (deathtouch wins and blocks, immediate general captures, obvious defenses).

### 8.7 Game-theoretic training
With hidden information, the goal is a policy that is hard to exploit, not one that beats a single opponent.
- **Fictitious play / NFSP:** respond to the *average* of past opponents.
- **PSRO / league training:** keep a population and train new policies against a mix of it.
- **Regularized dynamics** (R-NaD in DeepNash, Magnetic Mirror Descent): PPO-like updates with a regularizer that pulls toward a reference policy and helps convergence. Average Joe's "magnet" prior is in this spirit.
- **CFR:** the foundation of poker AI; too large for this game directly, but good background.

### 8.8 Auxiliary prediction
Extra outputs (or a separate model) trained to predict hidden facts: the enemy general's location, enemy army in fog, enemy castle locations. Labels are free in simulation. These improve representations and can directly drive the heuristic's hunting logic.

### 8.9 Recommended combination for us
1. Heuristic bot, tuned (8.1, 8.2), with tactical search for defense and deathtouch (8.3).
2. Self-play PPO following the Average Joe recipe, adapted to this ruleset (8.4), optionally started from behavior cloning (8.5).
3. League of past checkpoints plus the heuristic bot as opponents (8.7).
4. Safety layer on top at match time (8.6).

---

## 9. RL concepts primer and glossary

**MDP (Markov Decision Process).** An agent observes a state *s*, takes an action *a*, gets a reward *r*, and moves to a new state. The *policy* π(a|s) chooses actions. The *return* is the sum of future rewards, usually discounted by γ so nearer rewards count more.

**Value functions.** V(s) is the expected return from state s under the policy. Q(s, a) is the expected return from taking a in s, then following the policy.

**POMDP.** An MDP where the agent sees only part of the state. Fog makes this game a POMDP. The agent acts on a *belief*: its memory plus inference about hidden parts. In practice: memory channels in the input (last-seen values, turns since seen) or a recurrent or history component in the network.

**Policy gradient.** Directly adjust the policy's parameters to make actions with high return more likely. REINFORCE is the simplest version.

**Actor-critic.** An *actor* (the policy) plus a *critic* (a value estimate). The critic judges whether an action turned out better or worse than expected. That difference is the **advantage** A(s, a).

**GAE (Generalized Advantage Estimation).** Blends short- and long-horizon advantage estimates with a parameter λ, trading off bias and variance.

**PPO (Proximal Policy Optimization).** Actor-critic with a clipped objective that stops each update from moving the policy too far. Stable and simple; the default choice for game self-play.

**Action masking.** Set invalid actions' logits to −∞ before the softmax, so the agent never samples them and gradients aren't wasted on them.

**Reward shaping.** Small intermediate rewards (land, army, castles) to fix sparse rewards. Agents exploit badly designed shaping, so keep it small, anneal it away, or use potential-based shaping. Average Joe uses win/lose reward only.

**Self-play.** Training against copies of yourself. **League training** adds a pool of past versions and specialist exploiters so the agent doesn't forget how to beat old strategies or fall into cycles.

**Curriculum.** Start with easier game settings and make them harder as the agent improves. Average Joe starts with generals close together.

**Distributional value / HL-Gauss.** Predict a distribution over returns (as classification over bins) instead of a single number. Often more stable than regression.

**EMA weights.** An exponential moving average of the network weights, used for evaluation and deployment. Smoother and usually stronger than the latest checkpoint.

**Behavior cloning.** Supervised learning of a teacher's actions. **DAgger** iteratively relabels the student's own states with the teacher's actions.

**MCTS.** Build a search tree by simulating games, balancing trying promising moves against exploring new ones (UCT or PUCT). AlphaZero guides it with policy and value networks.

**Nash equilibrium.** A strategy pair where neither player can gain by changing alone. In two-player zero-sum games, playing an equilibrium guarantees you can't be exploited.

**Elo.** Rating from pairwise results; the competition's ranking metric. Use it locally too to measure progress between our versions.

---

## 10. Reading list (papers)

Entries marked ✅ were verified directly; entries marked 📚 are from general knowledge, so double-check titles, years and authors before citing.

### Must read first

| Paper | Why |
|---|---|
| ✅ Straka & Schmid, 2025. *Artificial Generals Intelligence: Mastering Generals.io with Reinforcement Learning.* arXiv:2507.06825 | The engine paper, by the organizer. Our environment |
| 📚 Schulman et al., 2017. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347 | PPO |
| 📚 Huang et al., 2022. *The 37 Implementation Details of Proximal Policy Optimization.* ICLR Blog Track | What makes PPO work in practice |
| 📚 Vinyals et al., 2019. *Grandmaster level in StarCraft II using multi-agent reinforcement learning.* Nature | AlphaStar: imitation, then RL, then league |
| 📚 Perolat et al., 2022. *Mastering the game of Stratego with model-free multiagent reinforcement learning.* Science | DeepNash: imperfect information via self-play (R-NaD) |

### Core RL

| Paper | Why |
|---|---|
| 📚 Schulman et al., 2015. *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* arXiv:1506.02438 | GAE |
| 📚 Andrychowicz et al., 2020. *What Matters in On-Policy Reinforcement Learning? A Large-Scale Empirical Study.* | Which PPO settings matter |
| 📚 Huang & Ontañón, 2020. *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms.* | Action masking |
| 📚 Farebrother et al., 2024. *Stop Regressing: Training Value Functions via Classification for Scalable Deep RL.* | HL-Gauss value head |
| 📚 Espeholt et al., 2018. *IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures.* | V-trace, an alternative to PPO |

### Grid and RTS games

| Paper | Why |
|---|---|
| 📚 Huang et al., 2021. *Gym-μRTS: Toward Affordable Full Game Real-time Strategy Games Research with Deep Reinforcement Learning.* | Per-cell actions on a grid |
| 📚 Han et al., 2019. *Grid-Wise Control for Multi-Agent Reinforcement Learning in Video Game AI.* ICML | GridNet: one action per cell from a convolutional net |
| 📚 Berner et al., 2019. *Dota 2 with Large Scale Deep Reinforcement Learning.* arXiv:1912.06680 | OpenAI Five: PPO over long games |
| 📚 Ronneberger et al., 2015. *U-Net.* / Dosovitskiy et al., 2020. *An Image is Worth 16x16 Words (ViT).* | Board encoders: convolutional vs. patch transformer |

### Search and planning

| Paper | Why |
|---|---|
| 📚 Silver et al., 2018. *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play.* Science | AlphaZero |
| 📚 Schrittwieser et al., 2020. *Mastering Atari, Go, chess and shogi by planning with a learned model.* Nature | MuZero |
| 📚 Danihelka et al., 2022. *Policy improvement by planning with Gumbel.* ICLR | Search with very few simulations |
| 📚 Cowling, Powley & Whitehouse, 2012. *Information Set Monte Carlo Tree Search.* | MCTS under hidden information |
| 📚 Lanctot et al., 2013. *Monte Carlo Tree Search in Simultaneous Move Games with Applications to Goofspiel.* | Simultaneous moves |

### Imperfect information and game theory

| Paper | Why |
|---|---|
| 📚 Moravčík, Schmid et al., 2017. *DeepStack: Expert-level artificial intelligence in heads-up no-limit poker.* Science | Search plus learning in hidden-info games |
| 📚 Schmid et al., 2023. *Student of Games: A unified learning algorithm for both perfect and imperfect information games.* Science Advances | Unifies AlphaZero-style and poker-style methods |
| 📚 Brown et al., 2020. *Combining Deep Reinforcement Learning and Search for Imperfect-Information Games.* NeurIPS | ReBeL |
| 📚 Heinrich & Silver, 2016. *Deep Reinforcement Learning from Self-Play in Imperfect-Information Games.* | NFSP |
| 📚 Lanctot et al., 2017. *A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning.* NeurIPS | PSRO, the theory behind leagues |
| 📚 Sokota et al., 2023. *A Unified Approach to Reinforcement Learning, Quantal Response Equilibria, and Two-Player Zero-Sum Games.* ICLR | Magnetic Mirror Descent |
| 📚 Zinkevich et al., 2007. *Regret Minimization in Games with Incomplete Information.* NeurIPS | CFR (background) |

### Imitation and bootstrapping

| Paper | Why |
|---|---|
| 📚 Ross, Gordon & Bagnell, 2011. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.* | DAgger |
| 📚 Schmitt et al., 2018. *Kickstarting Deep Reinforcement Learning.* | Teacher-guided RL |

---

## 11. Courses, blogs and tools

### Learning RL 📚
| Resource | Link | Notes |
|---|---|---|
| Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) | http://incompleteideas.net/book/the-book-2nd.html | Free textbook. Chapters 1–6 and 13 |
| OpenAI Spinning Up | https://spinningup.openai.com | Clear intro to policy gradients and PPO |
| Hugging Face Deep RL Course | https://huggingface.co/learn/deep-rl-course | Free and hands-on; includes PPO and self-play |
| David Silver's RL lectures (UCL/DeepMind) | https://www.davidsilver.uk/teaching/ | Classic lecture series (also on YouTube) |
| Lilian Weng, "Policy Gradient Algorithms" | https://lilianweng.github.io | Excellent overview blog post |
| "The 37 Implementation Details of PPO" | https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ | Must-read before writing PPO |

### Code and libraries ✅ (repos confirmed to exist)
| Tool | Link | Use |
|---|---|---|
| generals-bots | https://github.com/strakam/generals-bots | Engine and JAX training environment |
| AverageJoe | https://github.com/strakam/AverageJoe | Reference RL pipeline (license caveat above) |
| CleanRL | https://github.com/vwxyzjn/cleanrl | Readable single-file PPO implementations |
| PureJaxRL | https://github.com/luchris429/purejaxrl | Fully-jitted PPO in JAX, the pattern for GPU training on our env |
| Equinox | https://github.com/patrick-kidger/equinox | JAX neural network library (used by AverageJoe) |
| mctx | https://github.com/google-deepmind/mctx | JAX MCTS, including Gumbel MuZero |
| OpenSpiel | https://github.com/google-deepmind/open_spiel | Reference game-theory algorithms (CFR, PSRO, NFSP) |
| Optuna | https://github.com/optuna/optuna | Bayesian optimization for parameter tuning |
| pycma | https://github.com/CMA-ES/pycma | CMA-ES for parameter tuning |

Also useful: Weights & Biases (experiment tracking); Kaggle Lux AI discussion forums (winning writeups).

---

## 12. Build plan

Each phase has concrete deliverables so coding agents can work in parallel and we can measure progress.

### Phase 0: Infrastructure (blocks everything else)
**Deliverables**
- `engine/`: a pinned copy of generals-bots, and a wrapper building `GeneralsEnv` with the competition preset.
- `runner/match.py`: runs two stdio bots (via their `run.sh`) against each other with exact competition timing (150 ms, 10 s first move, fault counting), and saves replays.
- `runner/batch.py`: runs N games in parallel across seeds, swapping sides, and reports win, loss and draw rates, average game length and faults.
- `runner/elo.py`: an Elo table over all our bot versions.
- `bots/common/`: an observation parser, persistent map memory, BFS utilities, action encoding, a move-legality check, and a timing guard that always replies before a deadline.
- Unit tests reproducing the verified mechanics in Section 3 (move order, chase rule, build pricing, deathtouch defense).

**Done when:** the pass bot and an expander bot play 100 headless games locally with 0 faults, and replays can be viewed.

### Phase 1: Heuristic bot v1
**Modules**
1. **Memory:** last-seen type, owner, army and time per cell. Castle vs. mountain disambiguation. Enemy-general candidate set, narrowed using the 17-step distance, the direction of enemy tiles, and the land and army totals.
2. **Opening:** an expansion schedule optimized in simulation (which tiles and when, when to start gathering).
3. **Gathering:** tree-based gathering of army toward a target.
4. **Economy:** castle-building rules (when affordable, at distance ≥ 7 from own structures where safe, not in enemy vision, protecting fresh 0-army castles).
5. **Defense:** threat detection around the general and castles; reinforcement uses the defensive-first move order.
6. **Attack:** scouting, general hunting, capturing enemy castles.
7. **Endgame:** from about turn 750, prioritize locating the general and a deathtouch attack. From 800, keep the general's neighborhood clear of enemy tiles, and use the chase-capture defense.
8. **Draw avoidance:** increasing aggression as turn 1200 approaches.

**Done when:** it beats the expander baseline over 95% of the time locally and is submitted to the leaderboard.

### Phase 2: Tuning and tactics
- Expose all heuristic parameters in a config; tune with Optuna or CMA-ES using `runner/batch.py`.
- Add a 1–3 turn tactical search using an exact engine copy for defense, capture races and deathtouch.

**Done when:** the tuned bot beats v1 by a clear Elo margin over at least 200 games.

### Phase 3: RL pipeline (GPU)
- **Environment:** batched JAX competition environment (castle building, deathtouch, fog, 1200 turns, 18–21 boards padded to 21×21).
- **Observation encoding:** current grids plus memory channels (last-seen army and owner, turns since seen, known enemy castles, general candidates), plus scalars (turn, turns until 800, turns until 1200, land and army totals and their recent history).
- **Action space:** per-cell {4 directions × 2 splits} plus per-cell build plus a global pass, with full masking. Builds are masked by the live `build_cost_grid`.
- **Network:** start with Average Joe's patch transformer design (re-implemented), and optionally compare against a U-Net baseline.
- **Algorithm:** self-play PPO with GAE, EMA weights, a distributional value head, and a generals-distance curriculum.
- **Optional warm start:** behavior cloning from the Phase 2 heuristic bot.
- **Opponent pool:** past checkpoints plus heuristic bots, sampled with weight toward opponents we lose to.

**Done when:** an RL checkpoint beats the tuned heuristic bot over 60% of the time locally.

### Phase 4: Ruleset-specific edge
- The deathtouch phase is learned by the network via turn features, and verified with targeted test positions.
- An auxiliary head predicting the enemy general's location (free labels from simulation).
- An anti-draw term (for example a small penalty for draws, tuned so it doesn't cause reckless play).
- Analysis of castle-building behavior against the economics in Section 6.

### Phase 5: Deployment
- Export weights (safetensors) and run inference in the sandbox's JAX or torch (CPU). Vendor any missing pure-Python dependency as a wheel.
- **Target ≤ 60 ms per move** on one core, benchmarked against the sandbox's library versions. Warm up or compile on the first move within 10 s; persist caches to `$HOME` or the bot tree.
- A **safety layer** overriding the network: an immediate winning move (capture or deathtouch), a forced deathtouch block, and a timeout fallback to the heuristic.
- Memory check well under 2 GB.

### Phase 6: League and iteration
- Continuous training with a growing league; regular submissions; compare local Elo with leaderboard results; study replays of our losses.

---

## 13. Open questions

| # | Question | How to resolve |
|---|---|---|
| 1 | Where is the `competition/` folder (`matchup.py`, starter bots)? | Dashboard or Discord; otherwise build our own runner |
| 2 | May we reuse AverageJoe code or weights, given no license and the "own work" rule? | Ask organizers on Discord before using it |
| 3 | Current competition schedule and prize checkpoints | Check https://www.generals.bot |
| 4 | Does the stdio observation encode anything beyond what `game.py`'s `_observe` produces (for example, how our own castles vs. generals appear)? | Run the official runner and dump observations |
| 5 | Exact inference latency of our network on the sandbox CPU | Benchmark after first submission; log timings to stderr if allowed |
| 6 | Is logging to stderr allowed or visible in match logs? | Check docs or Discord |

---

## 14. Notes for coding agents

- **The engine is the source of truth.** When in doubt about a rule, read `generals/core/game.py` and the modifiers, or write a test against the engine. Don't assume standard generals.io behavior. Resolution order and fog encoding differ from what most people expect.
- **Never block the reply.** Every bot must reply within 150 ms. Wrap the main loop in try/except, keep a deadline guard, and fall back to a cheap heuristic move or a pass.
- **Flush after every write**, and exit cleanly on EOF.
- **Read H and W from the handshake.** Handle every size from 18 to 21 on each side.
- **Keep bots self-contained** under `bots/<name>/` with their own `run.sh`, so each can be zipped and submitted directly.
- **Measure everything with `runner/batch.py`.** No change is "better" without a win-rate result over enough games (at least 200 for small differences), with sides swapped.
- **Keep the test suite green**, especially the mechanics tests from Section 3.
- **Respect fair-play rules.** No code or weights from other competitors; no attempts to probe the sandbox or other bots.
