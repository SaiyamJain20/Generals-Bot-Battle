# Bot-Battle: our Code Bot entry

A single-file, standard-library-only Python bot for the **Code Bot** hackathon. The game is the
generals.bot competition ruleset: fog of war, castle building, deathtouch from turn 800, and a draw at turn 1200.

The bot is a hand-written strategy whose numbers were tuned by self-play. It is wrapped in a small learned
"context modulator", tuned by evolution strategies, the black-box RL method we used for the 5-hour RL session.
Everything it needs is inside one file of about 92 KB.

> **TL;DR**
> - The submission is built from `bots/participant.py` plus `runs/final_params.json`
>   (a copy is at `agents_shared/final_params.json`). This parameter set is called **F2**.
> - On maps never used for any decision, it scores **0.752 and 0.754** (two independent slices of 1,000 games).
>   - This morning's best scored 0.665.
>   - The same bot without the RL modulator scores 0.71 / 0.69.
> - Against the strongest opponents we have, the behaviour-cloned top Marathon bots, it now wins **55–69 %** of games.
>   That figure was 32–48 % this morning.
> - To produce the file you upload, run `tools/build_submission.py` with your participant ID and bot name
>   (see [How to build the submission](#1-how-to-build-the-submission)).
> - It passes the **official evaluator** (`evaluate.py validate`) and plays in the official Docker sandbox
>   with a worst move of about 5.6 ms (limit 150 ms).
> - An overnight RL deep-dive (PPO over options, GRPO, ES, CMA-ES) did not produce a bot that beats F2 under a
>   fair test; see [7.5](#75-the-overnight-rl-deep-dive-23-oct-did-proper-rl-beat-f2).

---

## Contents
1. [How to build the submission](#1-how-to-build-the-submission)
2. [The game in two minutes (and the engine facts that matter)](#2-the-game-in-two-minutes)
3. [How the bot thinks](#3-how-the-bot-thinks)
4. [How it was tuned (including the RL session)](#4-how-it-was-tuned)
5. [Who it was tested against](#5-who-it-was-tested-against)
6. [What we tried: kept and rejected](#6-what-we-tried-kept-and-rejected)
7. [Final results](#7-final-results)
8. [Rule compliance](#8-rule-compliance)
9. [Known weaknesses and next ideas](#9-known-weaknesses-and-next-ideas)
10. [Repository map](#10-repository-map)
11. [Working on the Ada cluster](#11-working-on-the-ada-cluster)
12. [Still open](#12-still-open)

---

## 1. How to build the submission

**Quickest path, one command.** It builds the file, runs our checks, runs the official organizer evaluator's
`validate`, plays one official-sandbox game, and prints the SHA-256 to keep as your receipt:

```bash
cd ~/Desktop/Bot-Battle && tools/final_build.sh <your_participant_id> "<your bot name>"
#  -> submission/<your_participant_id>.py  (upload this file)
```

The same steps by hand:

```bash
cd ~/Desktop/Bot-Battle
export PYTHONPATH=vendor/generals-bots:.

# 1) bake the final parameters into one file, fill the header, run all checks
.venv312/bin/python tools/build_submission.py \
    --params runs/final_params.json \
    --id <your_participant_id> --name "<your bot name>"
#  -> writes submission/<your_participant_id>.py and prints its SHA-256

# 2) (optional) re-run the checks by hand
.venv312/bin/python tools/check_submission.py submission/<your_participant_id>.py --games 4
```

`build_submission.py` does four things:
1. Writes the tuned values straight into the `PARAMS` literal, so there is no runtime update.
2. Forces `DEBUG = False`.
3. Fills `[PARTICIPANT_ID]` and `[BOT_NAME]` in the header.
4. Runs `check_submission.py`. That script checks:
   - UTF-8 and ≤ 1 MiB;
   - standard-library imports only;
   - no `print`, `open`, `exec`, network or threads;
   - a top-level `act()`;
   - the placeholders are filled;
   - a few full games with no forfeits.

**Before uploading, please check two things:**
- **The header's AI-assistance paragraph.** It says the participant reviewed the code and can explain it. Make
  sure that is true, or edit the sentence.
- **The SHA-256.** Keep the printed hash as your receipt.

Other useful commands:

```bash
# unit + parity tests (exact engine replica vs. the pinned JAX engine, tactics, robustness)
.venv312/bin/python -m pytest -q tests/

# play A vs B, both slots, many workers (in-process, fast)
.venv312/bin/python tools/abmulti.py 40 6 --bots bots/versions/F2.py --opps bots/opp/hunter.py bots/versions/t1c.py

# realistic timing: both bots in separate processes on ONE core, 150 ms limit, 3x CPU slowdown
.venv312/bin/python arena/subproc.py bots/versions/F2.py bots/versions/t1c.py --games 8 --slowdown 3

# catastrophe gate: every simple/zoo/public bot, fails on any forfeit or a score below 0.85
.venv312/bin/python tools/gate.py bots/versions/F2.py --games 30 --workers 6
```

---

## 2. The game in two minutes

**Each turn,** each player makes one action:
- **move** an army to a neighbouring cell, sending either *all but one* or *half*;
- **build** a castle;
- **pass**.

**Combat** subtracts armies. A tie keeps the cell for the defender.

**Growth:**
- Generals and castles grow +1 every other turn.
- Every owned cell gets +1 every 50 turns (the "land bonus").

**Winning:**
- You win by capturing the enemy general.
- From turn 800, *any* move onto the enemy general wins ("deathtouch").
- At turn 1200 the game is a draw.

**Limits (forfeit if broken):** 150 ms per move (10 s for the first move, import included), one shared CPU,
2 GiB of memory, stdlib Python 3.12, one file of at most 1 MiB. A timeout, an exception or a malformed return
forfeits the game.

### Engine facts we verified at the pinned commit (`strakam/generals-bots@13db8f69`)
These are things a casual reading of the rules gets wrong. They drive many of the bot's decisions.

| Fact | Why it matters |
|---|---|
| Moves resolve in this order: **chasing** (moving onto the cell the enemy is leaving) → **reinforcing** (moving onto your own cell) → smaller army first → player 0. | Reinforcing a cell beats an attack on it in the same turn. You cannot dodge a chase. |
| Deathtouch can only be stopped by a **chase from a third tile** onto the attacker's cell, leaving it with ≤ 1 army. Countering from your own general fails. | This is the basis of the deathtouch guard (section 3). |
| Every mountain is visible from turn 0. **A cell that later turns "structure in fog" is an enemy castle**, even deep in fog. | We know where every enemy castle is. |
| `opp_army` and `opp_land` are exact every turn. | We can work out every enemy castle build and its exact price. |
| Castle price = 35 + Σ max(0, 14 − 2d) over the builder's own general and castles. | The price pins the enemy general to a ring of known radius around the new castle. |
| Map generator: generals are ≥ 17 walking steps apart, and the "room within 7 steps" counts differ by ≤ 5. | This rules out most cells as the enemy general's spawn at turn 0. |
| Observed turn T: structures grow after odd T; the land bonus comes after T = 49, 99, … | Timing of builds, strikes and captures. |

`sim/engine.py` is a pure-Python replica of these rules. `tests/test_parity.py` checks it against the pinned
JAX engine, so all of our local games follow the real rules.

---

## 3. How the bot thinks

All of this lives in `bots/participant.py`, about 2,400 lines. Every turn runs the same pipeline:

```
observation ──► parse + memory ──► opponent accounting ──► enemy-general belief
                                                                 │
           ┌─────────────────────────────────────────────────────┘
           ▼
   win_now ─► deathtouch guard ─► urgent defence ─► intercept ─► opening (T<50) ─► macro options
                                                                                     │
                                         one-turn exact safety veto ◄────────────────┘
                                                     │
                                          sanitise ─► [kind, row, col, dir, split]
```

### 3.1 Memory and exact opponent accounting
- The bot keeps a turn-0 snapshot of the mountains, and for every cell the last type, owner, army and turn seen.
- From the change in `opp_army` / `opp_land`, it computes every turn whether the enemy built a castle and exactly
  what it paid.
- It matches that build to the new "structure in fog" cell.
- From the price it derives a ring of possible general locations. A remainder of r means distance (14 − r) / 2.
- This was validated against public replays, with 28 of 28 builds matched.

### 3.2 Where is the enemy general?
1. **Hard filters at turn 0:** the generator's spawn rules (walking distance ≥ 17; room-within-7 within ±5).
2. **Updates during the game:**
   - cells seen without a general are ruled out;
   - castle-price rings narrow the area;
   - so do places where enemy armies come out of the fog.
3. **Ranking:** a small learned spawn prior ranks what is left. It is a 10-weight logistic regression (`PRIOR_W`,
   `PRIOR_B`), trained on maps from the pinned engine's own generator.

### 3.3 Options every turn
The "macro" step scores a handful of options and plays the best:
- **capture:** take a neutral or enemy cell. Timed toward the land bonus, with an early-expansion bonus until
  turn 100.
- **garrison:** gather army onto the general. The size is set by visible threats, tracked stacks moving in from the
  fog, and part of the hidden enemy army. The gather budget is the arrival time of the binding threat.
- **army cycle:**
  - gather a stack along value-per-move gather trees;
  - launch half of it at the most likely general, an enemy castle or a nearby threat;
  - route it stealthily, avoiding cells the enemy can probably see.
- **kill:** the "sweep" kill check walks the exact path to the enemy general. Our own cells on the way join the
  stack, and every enemy cell must be beaten. It strikes when the stack arrives with more than the general's army
  plus a margin.
- **build** a castle, **scout**, or **home fill** (own every cell near the general for early warning).
- **intercept:** chase-kill an attacker next to our general or castles.

### 3.4 The context modulator (the part tuned by RL)
Each option weight above is multiplied by `exp(w · f)`, clipped to e^±1.5.
- **Features** (8, all in about [−1, 1]):
  - game phase;
  - our/their army ratio;
  - land ratio;
  - number of enemy castles;
  - the enemy's gather style;
  - whether the enemy general is known;
  - the biggest tracked threat relative to our general;
  - a bias.
- **Weights:** 7 option groups × 8 features = 56, all called `m_<group>_<j>`.

This is how the bot adapts to opponent type and game situation. What the RL session learned is in section 4.3.

### 3.5 Safety layers
- **One-turn exact resolver:** before any move goes out, it is simulated against every enemy move that could hit
  our general, using the engine's exact move order. Moves that lose the general are vetoed.
- **Deathtouch guard (`dt_guard`, from turn 790):**
  1. an enemy stack next to our general is chased from a third tile with at least equal army;
  2. a stack two steps away is killed, or the neighbour it would take is reinforced (reinforcing resolves before
     the attack).
  - This was added after a real loss at turn 865 that we were winning 1,253 to 895 army.
- **Endgame fortress:** from turn 696 the bot owns every neighbour of its general and clears enemy cells within
  2; from 760 it stages a deathtouch strike.
- **Shell:**
  - `act()` never raises; any internal error returns a valid pass;
  - every value goes through `int()` and a range clamp;
  - observation keys are read through aliases, and both nested and flat grids are accepted;
  - typical moves take 2–8 ms.

---

## 4. How it was tuned

### 4.1 CMA-ES self-play (classic parameter tuning)
- `tune/cma_tune.py` tunes about 100 numbers: option weights, garrison fractions, launch sizes, castle timing,
  and so on.
- Candidates are compared on **paired maps**: the same maps, both slots.
- **Opponent pool:**
  - fixed opponents, mostly the behaviour-cloned top bots;
  - PFSP weighting: opponents we lose to count more;
  - self-play against the current mean;
  - a small league of past snapshots.
- **Output:** the average of the last 6 means (`mean_avg.json`). It is less noisy than the single best candidate.
- **Where it ran:** on Ada (24–40 CPUs per job) and on the laptop.

### 4.2 Map-overfitting check
- **Setup:** we generated 2,000 *fresh* maps from the pinned generator (`data/maps_fresh.jsonl`). Tuning never uses
  them.
- **Result:** the tuned bot scored 0.670 on fresh maps and 0.653 on tuning maps. The bot learned strategy, not
  maps.
- **Final selection:** made on fresh-map slices that no earlier decision had used.

### 4.3 The RL session: evolution strategies on the modulator
- **Method:** black-box RL (ES, run with CMA-ES) over the 56 modulator weights plus one kill-check setting.
  - Starting point: the tuned bot.
  - Pool: clone-heavy, plus self-play and league snapshots.
  - Runs: es2 → es3 → es4, about 3.5 hours of Ada time.
  - es2 started from the wrong base, and es3 was superseded once the sweep kill check was adopted.
  - es4 ran 9 generations, then stalled for its last hour; its averaged mean is the final modulator.
- **Why ES rather than PPO etc.:** a deep network could not run in 150 ms of pure Python. The reports in
  `agents_shared/rl-research.md` and `agents_shared/vast-research.md` found that ES over a low-dimensional policy
  is the best RL we can actually deploy here.
  - We also looked for "Jev's" RL model and found nothing public under that name.
  - The candidates are AverageJoe (a 15M-parameter net, far too big here), ResBot and quant-eagle.
- **Key weights learned in es4 (`runs/es4/mean_avg.json`):**

| Group | Learned weight | Effect |
|---|---|---|
| launch | bias +0.51, +0.26 × enemy castles | ~1.7× more launches, more again vs castle builders |
| garrison | bias −0.48, +0.45 × army ratio | thinner garrison, thicker when ahead |
| kill | bias −0.24 | strikes with a thinner margin |
| capture | bias −0.35, −0.41 × enemy castles | fewer small captures |
| scout / home | +0.46…+0.58 × threat / general known | more scouting and home-fill under threat |

In short: **more aggressive, especially against economy builders.** That is exactly what beats the cloned top bots,
which out-grow us if the game goes long. The cost is a little robustness against hunter-style bots (section 7).

---

## 5. Who it was tested against

| Group | Bots | Notes |
|---|---|---|
| **Behaviour clones of the top Marathon bots** | `bc_resbot96`, `bc_resbot128`, `bc_nanomena96`, `bc_kubic96` | Our strongest opponents. They are ResNet policies (96–128 channels) trained on the Ada GPU from public Marathon replays (up to 3.3k games each, from the 15k we downloaded), with actions inferred from the replays by our exact engine. They need torch, so they are local-only. |
| Public bots from GitHub | Sentinel / Sentinel v10 (relh), bca (blake-ar, **an RL conv-net**), hvn (hv-nguyeen), juraj34/35, superbot, boss, Human.exe, A9 (C++), doomstack, amin, mybot9 | Wrappers in `bots/opp/ext_*.py`, code in `vendor/ext/` (gitignored). |
| Hand-written zoo | rusher, hunter, hunter_castles, turtle, zoo_flash / castler / gatherer / sniper / turtle_dt / expander_plus / mixed | Each one probes a style: rush, snipe castles, turtle for deathtouch, and so on. |
| Our older versions | t1c, c_a2es3, v1, … | Stop regressions; t1c is a good "medium heuristic" stand-in. |

---

## 6. What we tried: kept and rejected

Every change was measured with paired-map A/B tests, usually 400–1,000 games. The full log, with every number,
is in `agents_shared/ROADMAP.md`.

**Kept**

| Change | Effect |
|---|---|
| Early expansion until turn 100 | 0.738 → 0.766 |
| Stealth routing of strikes (`stealth_w` 0.5) | 0.738 → 0.782 |
| Chase-kill interceptor | 0.738 → 0.753 |
| Sweep kill check (exact path walk, 10 candidate stacks; idea from Sentinel) | 0.580 → 0.620 |
| Deathtouch guard | fixes a real class of turn-800+ losses; unit-tested |
| ES-tuned context modulator (es4) | 0.71 → 0.75 on untouched maps |

**Rejected** (each tested, then dropped or left off)

| Idea | What happened |
|---|---|
| Earlier / more castles (start at turn 120–160, castles from the general, castles on the stack) | 0.43 → 0.31 vs clones. Our castles get drained and taken; the tuner keeps castles late. |
| Keep half the army on castles | 0.53 → 0.48. Replays show top bots drain castles fully too. |
| Attack only when ahead | 0.56 → 0.42 / 0.09. Launches are what win games. |
| Bonus-timed enemy captures (the replay rule) | 0.54 → 0.44–0.49 |
| Spread small stacks after each land bonus | 0.555 → 0.458 |
| Longer early expansion (to turn 130 / 160) | 0.645 → 0.615 / 0.557 |
| Kill-front gathering, stealth kill paths, kill-margin changes | negative or no effect |
| Defence escalation (urgent garrison when a big threat is far but known) | neutral (+0.016 / −0.013). It fixed one game but not the average; left off. |
| A learned threat model (gradient-boosted, trained on replays) | no gain; removed from the file (it was also a 141 KB audit risk) |

**A lesson from all of this:** the tuned bot sits at a strong local optimum. Moves taken away from attack cycles and
spent on "economy" kept losing, even though the clones out-grow us economically. That is why the gains came from
better strikes (sweep), fewer blunders (deathtouch guard) and the RL-tuned aggression.

---

## 7. Final results

### 7.1 Untouched fresh maps (100 games per opponent, both slots)
"Pooled" is the average over all 10 opponents (1,000 games, about ±0.027 at 95 %).

| Bot | Slice | Pooled | ResBot-96 | ResBot-128 | nanomena | Kubic | t1c | sniper | flash | mixed | rusher | hunter |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| this morning's best (c_a2es3) | 1700 | 0.665 | .36 | .32 | .48 | .41 | .72 | .78 | .92 | .87 | .80 | .99 |
| base, no modulator (F0) | 1700 | 0.713 | .45 | .39 | .57 | .45 | .69 | .89 | .94 | .93 | .84 | .99 |
| **final (F2)** | 1700 | **0.752** | **.56** | **.58** | **.66** | **.66** | .57 | .92 | .90 | .94 | .78 | .96 |
| base, no modulator (F0) | 1800 | 0.692 | .46 | .34 | .46 | .54 | .68 | .84 | .91 | .90 | .84 | .96 |
| **final (F2)** | 1800 | **0.754** | **.64** | **.55** | **.69** | **.64** | .57 | .89 | .89 | .92 | .86 | .90 |

### 7.2 Public bots (laptop; 40 games each, bca and boss 100; "pooled" is the 40-game run over all 7)

| Bot | Sentinel v10 | Sentinel | hvn | juraj35 | superbot | boss | bca (RL) | pooled |
|---|---|---|---|---|---|---|---|---|
| base (F0) | .83 | .83 | .68 | .89 | .93 | .93 | .53 | 0.816 |
| **final (F2)** | .85 | .95 | .78 | .88 | 1.00 | .90 | .44 | 0.821 |

### 7.3 Safety checks on the final bot
| Check | Result |
|---|---|
| Catastrophe gate: 20 simple / zoo / public bots × 30 games (`tools/gate.py`) | **0 forfeits in 600 games.** Worst move 36 ms (laptop under full load). Lowest scores: zoo_gatherer 0.83, hvn / flash / mixed 0.87; all others ≥ 0.93. Gate average 0.953 (base: 0.944). |
| Realistic runner, 1 shared core, 150 ms limit, 3× CPU slowdown, vs t1c (8 games) | 0 timeouts; worst move **56.6 ms**, worst first move **529 ms** (limit 10 s) |
| Same, 2× slowdown, vs the ResBot clone (4 games) | 0 timeouts; worst move 22 ms, first move ≤ 513 ms |
| Unit tests (`pytest tests/`) | engine parity with the pinned JAX engine, rules, accounting, tactics (incl. 2 deathtouch-guard tests), robustness: all pass |
| Built file (`tools/build_submission.py` → `check_submission.py`) | 92.5 KB, stdlib imports only, no banned calls, header filled, 4 games without forfeit |
| **Official organizer evaluator** (`evaluate.py validate` + `match` in the `codebot-python:1` Docker sandbox; official 150 ms / 1200-turn limits; 1 CPU; no network) | validate passes. **16 official games, both seats, 0 faults.** Results: starter 3/3, t1c 2–2, own copy 0–1, and 8/8 vs hunter, rusher, expander, zoo flash / mixed / castler, and deathtouch-turtle ×2. Worst move **7.3 ms**, first move ≤ 75 ms. |

### 7.4 Why F2 and not the safer base
- F2 is the best on both untouched slices (+0.04 and +0.06).
- It has the best worst case: its lowest is 0.44 vs bca, against the base's 0.34 vs the ResBot-128 clone.
- It is the only version that beats every strong clone.
- **The price:** it is weaker against t1c (0.57 vs 0.69) and slightly weaker against Hunter and the bca RL bot.
- We also tested blends:
  - half-strength modulator: 0.725;
  - modulator without its garrison cut: 0.739.
  - Both landed in between, so there was no free lunch.

### 7.5 The overnight RL deep-dive (2–3 Oct): did proper RL beat F2?
Short answer: **no, not under a fair test.** F2 stays the submission. Two sessions ran RL tracks in parallel.
Every result below is a paired, deterministic evaluation on fresh maps that no training or earlier choice
had touched.

**The constraint that shapes everything.**
- The organizers confirmed one core (Sapphire Rapids), Python 3.12 and the standard library only, with no
  packages, model files or separate assets.
- A pure neural policy small enough for 150 ms of pure Python (a 12×1 conv student, 30 KB, 11.6 ms per move)
  plays at only **0.17** against our heuristic, and PPO fine-tuning made it worse (0.21 → 0.08–0.19).
- So the RL here learns the *decision layer* on top of the tested heuristic.

**What was tried.** All of these are deployable as numbers inside the single file:

| Track | Method | Scale |
|---|---|---|
| A (other session) | **RO-PPO**: a learned residual policy rescores the heuristic's candidate options every turn (24 global + 8 option features, 16-unit hidden layer). PPO + GAE, potential-based shaping annealed to 0, semi-MDP discounting, KL to the heuristic, PFSP league with self snapshots. Weights averaged over late iterations. | ~75k–90k games per run on Ada |
| C1 | **GRPO/PPO** over the context modulator: Gaussian exploration in logit space, group-relative advantages on the engine outcome (the "verifiable reward"), PPO-clip, KL anchor to F2, league incl. KSolmann | 696 iterations, ~67k games |
| C3/C3b | Same as C1, with a 12-unit MLP added to the modulator | ~1,000 iterations (laptop) |
| E1 | OpenAI-ES (antithetic, common random numbers) over the extended modulator (+ build group + 3 opponent-style features) | 58 generations |
| es5 | Rerun of the CMA-ES that produced F2, now with a stall watchdog (es4 had hung after generation 8) | 7.5 h without a stall |

**Correctness checks:**
- The learner gradients match finite differences.
- The bot and the learner compute the identical policy.
- With zero new weights, every candidate plays move-for-move like F2.
- Our simulator matches the **official evaluator** observation-for-observation: 1,564 of 1,564 observations
  identical over 3 full games.

**Selection** (fresh maps 1650–1699, 13 opponents × 100 games each):
- (a) = 8 heuristic opponents; (b) = 4 neural BC clones + KSolmann.

| Bot | (a) heuristic | (b) neural |
|---|---|---|
| F2 | 0.771 | 0.556 |
| C1 | 0.781 | 0.581 |
| C3b | 0.778 | 0.543 |
| es5 | 0.764 | 0.580 |
| E1 | 0.742 | 0.560 |
| RO `ro_a2_avg` | 0.824 | 0.504 |

**Joint final** (fresh maps 1850–1949, untouched until this point).

The rule, chosen by the participant:
- **decide on stdlib-feasible heuristic opponents**, because real entries must also run in pure Python at 150 ms;
- use the neural clones and KSolmann only as a guardrail: no worse than −0.05 vs F2.

| Bot | In-house heuristic bots (13 × 100 games) | Independent public heuristic bots (6 × 40 games) | Neural guardrail (5 × 100 games) | vs F2 head-to-head |
|---|---|---|---|---|
| **F2** | 0.804 | **0.908** | **0.562** | – |
| RO `ro_a2_p2avg` | **0.844** | 0.900 | 0.479 (fails: −0.083) | 0.64 |
| RO `ro_a2_avg` | 0.845 | 0.846 | 0.475 (fails: −0.087) | 0.58 |
| C1 | 0.802 | 0.890 | 0.569 | 0.485 |

**The lesson.** RO-PPO learned a real gain against the bot family it trained with: F2, t1c and the zoo.
- It beat F2 head-to-head 0.64.
- That gain **did not transfer** to independent public bots, and it cost 0.08 against strong learned play.
- The modulator RL runs (GRPO, ES, CMA-ES) all landed within ±0.02 of F2.
- So F2, the earlier ES-tuned version, remains the most robust bot we have.

**KSolmann's transformer.**
- It's a 3M-parameter AverageJoe-style model, trained with JAX PPO on GPUs, and the strongest bot we found.
- F2 scores **0.27–0.37** against it (100-game slices).
- It cannot be entered under the stdlib, 150 ms rule.

All RL code is in the worktree `.claude/worktrees/rl` (branch `rl-track`):
- Track A: `rl/hier/`, `rl/final/`;
- Track C: `rl/c/`;
- the design notes: `rl/DESIGN.md`;
- the full log: `rl/board.md`.

---

## 8. Rule compliance

Checked by `tools/check_submission.py`, and by a separate audit in `agents_shared/auditor.md`.

**No copied open-source bot code.** The organizers said open bots are not allowed. We ran a verbatim-copy
audit (`tools/copy_audit.py submission/<id>.py vendor/ext`) over all 1,104 third-party source files we downloaded (14 repos).
- Method: comments and whitespace removed, and every shared run of at least 25 consecutive tokens reported.
- Result: the longest overlap is 31 tokens, the engine-defined direction table
  `DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))`.
- Nothing else is shared. The ideas taken from other bots are credited in the header, and the code was
  written from scratch.

**The file itself:**
- One UTF-8 `.py` file of about 92 KB.
- Imports: `collections`, `gc`, `heapq`, `math`, `time`. Nothing else.
- No file, network, thread or subprocess use; no `print`; no `eval`/`exec`.

**What it reads:**
- Only the observation dictionary and its own memory, which the rules allow.
- No replay feeds, hidden board state or other files.

**What is embedded:**
- Only numbers produced during the event by our own scripts: `PARAMS`, including the 56 modulator weights, and
  the 10-weight spawn prior.
- No third-party code or weights.
- The neural clones and public bots were **sparring partners only**, and they are not in the file.

**The header:**
- It discloses the strategy, the embedded constants and how they were made.
- It names the reused ideas: the pinned engine (MIT), EklipZ, Sentinel and juraj ideas, and the Straka & Schmid
  paper.
- It describes the AI assistance (Claude Code).
- It notes that no AI is called at game time.

**Timing:** typical moves take 2–8 ms in-process. Even under a 3× CPU slowdown on one shared core, moves are far
below 150 ms (section 7.3).

---

## 9. Known weaknesses and next ideas

**Weaknesses**
- **Economy against the strongest bots.** By turn 300 the clones hold about 1.5–2× our land, and they keep castles.
  We win by striking first; long games favour them.
- **Merged attacks.** The garrison need is the *maximum* over single threats. Two stacks that merge next to the
  general can beat it.
- **Hunter-style aggression.** The final bot's thinner garrison costs a few percent against hunters (0.90–0.96).
- **Slot asymmetry against the clones.** We score about 0.50 as player 0 and 0.35–0.38 as player 1. It looks like a
  property of the clones (we are symmetric against t1c), but it is not fully understood.

**Ideas we did not get to**
- Add a "build" group and a distance-to-enemy feature to the modulator, then run a longer ES with more games per
  candidate. The es4 run was cut short by a tuner hang; a watchdog is now in place.
- A merge-aware threat estimate.
- 2-ply tactical search near the general.
- A learned "should I launch now?" value from the replays.

---

## 10. Repository map

| Path | What it is |
|---|---|
| `bots/participant.py` | **The bot** (source of the submission; placeholders in the header) |
| `bots/versions/` | Frozen candidates and A/B variants. `F2.py` = the final parameters applied to `participant.py`. |
| `bots/opp/` | Sparring partners: zoo, public bots (`ext_*`), behaviour clones (`bc_*`) |
| `sim/engine.py` | Exact pure-Python replica of the pinned engine (rules, fog observation) |
| `sim/make_maps.py` | Map pools from the pinned generator (`data/maps.jsonl`, `data/maps_fresh.jsonl`) |
| `arena/run.py`, `arena/subproc.py` | Fast in-process match runner / realistic subprocess runner (1 core, 150 ms) |
| `tune/cma_tune.py`, `tune/slurm/*.sbatch` | CMA-ES / ES tuner and the Ada job scripts |
| `learn/` | Replay download and parsing, action inference, BC dataset and training, spawn-prior training |
| `tools/` | A/B tools (`abmulti`, `abpool`), gate, submission build/check, loss and economy analysis, map viewer |
| `tests/` | Engine parity, rules, accounting, tactics (incl. deathtouch guard), robustness |
| `agents_shared/` | Research and experiment reports from the helper agents; `ROADMAP.md` holds the full experiment log |
| `runs/` (gitignored) | Tuning outputs; `runs/final_params.json` = the chosen parameter set |
| `OPEN_QUESTIONS.md` | Questions still waiting for answers |
| `ada_manifest.txt` | Every path and job we created on Ada |

Local environment: `.venv312/bin/python` with `PYTHONPATH=vendor/generals-bots:.`. The pinned engine lives in
`vendor/generals-bots`.

---

## 11. Working on the Ada cluster

We use a **guest account**, and the rules are in `CLAUDE.md`. In short:
- Touch only what we created.
- Never install globally.
- Keep caches in our scratch directory.
- Log every path and job in `ada_manifest.txt`.

How our setup works:
- **Code:** rsync it into `~/botbattle-saiyam/code`. Keep that directory under 200 MB and 2,000 files.
- **Jobs:** each job copies the code to the node-local `/scratch/$USER/botbattle-saiyam/` and builds its
  own `uv` venv there.
- **Results:** jobs copy their summaries back to `~/botbattle-saiyam/results/`.
- **Typical eval:**
  ```bash
  MAPS_FILE=data/maps_fresh.jsonl EVAL_TOOL=tools/abmulti.py sbatch --export=ALL --exclude=gnode007 \
      -A irel --qos=normal -c 36 --mem=72G --time=04:00:00 --job-name=bb-eval-x \
      code/tune/slurm/eval.sbatch <tag> 100 0 --bots <bots...> --opps <opps...> --offset <map offset>
  ```
- **Quirks:**
  - gnode007 hangs, so exclude it;
  - nodes give at most 38 CPUs;
  - the `irel` account has a group CPU limit;
  - fresh-map offsets wrap modulo 2,000.
- **Kept on purpose:** the Ada setup was *not* cleaned up, because more experiments are planned.

---

## 12. Still open
See `OPEN_QUESTIONS.md`. The ones that matter for submitting:
- **Participant ID and bot name.** They are needed for the file name `participant_id.py` and the header. Pass them
  to `build_submission.py`.
- ~~The event's exact adapter / starter kit.~~ **Resolved:**
  - The organizers' evaluator (`evaluator.zip`) is unpacked in `.claude/worktrees/rl/rl/c/evaluator/` and its
    Docker image `codebot-python:1` is built locally.
  - Our simulator reproduces its observations exactly (`rl/c/official_parity.py`).
  - The submission passes `evaluate.py validate`, and its official-sandbox games are listed in 7.3.
  - The real deadline from the kit's README is **3 Oct 2026, 2:00 PM IST**.
