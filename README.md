# Generals Bot Battle

A single-file, **standard-library-only Python 3.12** bot for **Code Bot** (Fuzzy Dynamics, 2–3 October 2026), a
20-hour bot-programming hackathon. It is played on the [generals.bot](https://www.generals.bot/rules) competition
ruleset: fog of war, castle building, "deathtouch" from turn 800, and a draw at turn 1200.

**Result: reached the knockout stage (top 16); eliminated in the round of 16.**

The bot is a hand-written strategy. Every rule of the engine is simulated exactly inside it, and its ~100 numbers
were tuned by self-play. On top of that sits a small learned **context modulator** (56 weights) trained with
evolution strategies. An overnight deep-dive into "proper" RL (PPO over options, GRPO, ES, CMA-ES) did not beat it
under a fair test. All of that work is in this repo too.

> **At a glance**
> - **Final bot:** [`submission/saiyam_f2.py`](submission/saiyam_f2.py), a 92.7 KB file whose only imports are
>   `collections`, `gc`, `heapq`, `math` and `time`.
> - **Held-out maps:** it scores **0.752 / 0.754** on maps never used for any decision (two slices of 1,000 games
>   vs 10 opponents).
> - **Strongest opponents:** 0.55–0.69 against neural clones of the top generals.bot ladder bots, and 0.91 pooled
>   against 6 independent public heuristic bots.
> - **Official evaluator:** 223 games in the organizers' Docker sandbox with 0 faults. The worst move took 13.6 ms
>   (limit 150 ms).

---

## Contents
1. [Quick start](#1-quick-start)
2. [The competition and the game](#2-the-competition-and-the-game)
3. [How the bot works](#3-how-the-bot-works)
4. [How it was tuned](#4-how-it-was-tuned)
5. [Who it was tested against](#5-who-it-was-tested-against)
6. [Experiments: what we kept and what we dropped](#6-experiments-what-we-kept-and-what-we-dropped)
7. [Results](#7-results)
8. [The RL deep-dive: did "proper" RL beat it?](#8-the-rl-deep-dive-did-proper-rl-beat-it)
9. [Tournament outcome and lessons](#9-tournament-outcome-and-lessons)
10. [Rule compliance](#10-rule-compliance)
11. [Repository layout](#11-repository-layout)
12. [Running experiments on a SLURM cluster](#12-running-experiments-on-a-slurm-cluster)
13. [Credits](#13-credits)

---

## 1. Quick start

### 1.1 Just use the bot (no setup)
[`submission/saiyam_f2.py`](submission/saiyam_f2.py) is self-contained. It exposes `act(observation)` and returns
`[kind, row, col, direction, split]`:

```python
import importlib.util
spec = importlib.util.spec_from_file_location("bot", "submission/saiyam_f2.py")
bot = importlib.util.module_from_spec(spec); spec.loader.exec_module(bot)

action = bot.act(observation)   # observation: the event's dict (section 2.2); module globals keep game memory
```

One module instance plays one game, because it keeps memory in module globals. Load a fresh copy for each game.

> This published file differs from the file actually submitted only in its header's participant ID.
> The code and parameters are identical.

### 1.2 Development setup
Requirements: Linux or macOS, Python 3.12 and git. Docker is needed only for the organizers' evaluator.

```bash
git clone https://github.com/SaiyamJain20/Generals-Bot-Battle.git && cd Generals-Bot-Battle
./setup.sh                        # venv + deps + pinned engine + map pools  (MAPS=2000 FRESH=200 ./setup.sh is faster)
export PYTHONPATH=vendor/generals-bots:.
.venv312/bin/python -m pytest -q tests/
```

`setup.sh` does four things:
1. Creates `.venv312` and installs `requirements.txt`.
2. Clones the pinned engine [`strakam/generals-bots@13db8f69`](https://github.com/strakam/generals-bots/tree/13db8f69a422380ea184d2f4ca262a38866c5fc6)
   into `vendor/generals-bots`.
3. Generates the training maps (`data/maps.jsonl`, seeds 0–19,999).
4. Generates the held-out maps (`data/maps_fresh.jsonl`, seeds 900,000+) with the engine's own generator.

To also install torch for behaviour cloning and the RL learners, run `LEARN=1 ./setup.sh`.

**Not included in the repo,** because they're large, third-party or derived:
- **Public bots** (`vendor/ext/`). Their wrappers are `bots/opp/ext_*.py`. `tune/slurm/setup_ext.sh` fetches
  Sentinel and bca; the full source list is in `agents_shared/bot-scout.md`.
- **Neural clone weights** (`data/bc/*.pt`). The `bots/opp/bc_*.py` wrappers need them; train them with
  `learn/` (section 5).
- **Downloaded replays.**
- **The organizers' evaluator kit.**

Everything else, including the simulator, tuner, tests and all bot versions, works out of the box.

### 1.3 Everyday commands
All commands assume `export PYTHONPATH=vendor/generals-bots:.` and use `.venv312/bin/python`.

| Task | Command |
|---|---|
| Play A vs B, both seats, paired maps | `python tools/abmulti.py 40 6 --bots bots/versions/F2.py --opps bots/opp/hunter.py bots/versions/t1c.py` |
| Same, on held-out maps | `ARENA_MAPS=data/maps_fresh.jsonl python tools/abmulti.py 40 6 --bots ... --opps ... --offset 100` |
| Realistic timing: separate processes on one core, 150 ms limit, CPU slowdown | `python arena/subproc.py bots/versions/F2.py bots/versions/t1c.py --games 8 --slowdown 3` |
| Catastrophe gate: fails on any forfeit or a score < 0.85 | `python tools/gate.py bots/versions/F2.py --games 30 --workers 6` |
| Trace one game (army and land over time) | `python tools/trace.py bots/versions/F2.py bots/opp/hunter.py 0` |
| Tune parameters (CMA-ES) | `python tune/cma_tune.py --name myrun --bot bots/participant.py --opps bots/opp/hunter.py:4 ...` |
| Build a submission from the final params | `tools/final_build.sh <participant_id> "<bot name>"` |

`tools/final_build.sh` does the following:
1. Bakes `agents_shared/final_params.json` into `bots/participant.py`, forces `DEBUG = False` and fills the header.
2. Runs `tools/check_submission.py`: size, stdlib-only imports, no banned calls, and a few full games.
3. If the organizers' evaluator is unzipped at `./evaluator/` and its Docker image is built, runs its `validate`
   plus one official sandbox game.
4. Prints the SHA-256.

---

## 2. The competition and the game

### 2.1 Event rules
- **Submission:** one UTF-8 Python 3.12 file of at most 1 MiB, standard library only. No packages, model files,
  assets, network or GPU.
- **Interface:** the file exposes `act(observation)` and returns five signed 32-bit ints.
- **Limits:**
  - 150 ms per move (10 s for the first move, import included);
  - one shared CPU (Sapphire Rapids);
  - 2 GiB of memory.
- **Forfeits:** a timeout, an exception or a malformed return forfeits the game immediately.
- **Open-source bots:** not allowed as entries. Ideas from them may be used with attribution.
- **Format:** a league (every pair plays two games, seats swapped), then knockouts.

### 2.2 The game in two minutes
- **Your turn:** **move** an army to a neighbouring cell (sending all but one, or half), **build** a castle, or
  **pass**.
- **Combat** subtracts armies. A tie keeps the cell for the defender.
- **Growth:** generals and castles grow +1 every other turn; every owned cell gets +1 every 50 turns (the "land
  bonus").
- **Winning:** capture the enemy general. From turn 800, *any* move onto it wins ("deathtouch"). Turn 1200 is a
  draw.
- **Observation:**
  - `turn`, `height`, `width`, `player_id`;
  - exact totals: `my_land`, `my_army`, `opp_land`, `opp_army`;
  - grids `type` (0 fog, 1 plain, 2 mountain, 3 castle, 4 general, 5 structure in fog), `owner` (0, 1 = you,
    2 = opponent) and `army`.

### 2.3 Engine facts that matter (verified at the pinned commit)
| Fact | Why it matters |
|---|---|
| Move order: **chasing** (moving onto the cell the enemy is leaving) → **reinforcing** → smaller army → player 0 | A reinforcement beats an attack in the same turn, and a chase can't be dodged |
| Deathtouch can only be stopped by a **chase from a third tile** onto the attacker | This is the basis of the bot's deathtouch guard |
| Mountains are visible from turn 0, so a cell that later turns "structure in fog" **is an enemy castle** | Every enemy castle is located, even deep in fog |
| `opp_army` and `opp_land` are exact every turn | Every enemy castle build and its exact price can be worked out |
| Castle price = 35 + Σ max(0, 14 − 2d) over the builder's general and castles | The price pins the enemy general to a ring around the new castle |
| Generator: generals are ≥ 17 steps apart, and their "room within 7 steps" counts differ by ≤ 5 | Most cells are ruled out as the enemy spawn at turn 0 |
| Structures grow after odd turns; the land bonus comes after turns 49, 99, … | Timing for builds, strikes and captures |

[`sim/engine.py`](sim/engine.py) is an exact pure-Python replica of the engine. It's checked against the JAX engine
by `tests/test_parity.py`, and against the organizers' evaluator by `rl/c/official_parity.py` (1,564 of 1,564
observations identical).

---

## 3. How the bot works

Everything is in [`bots/participant.py`](bots/participant.py), about 2,400 lines. Each turn:

```
observation ─► parse + memory ─► exact opponent accounting ─► enemy-general belief
                                                                     │
      ┌──────────────────────────────────────────────────────────────┘
      ▼
  win now? ─► deathtouch guard ─► urgent defence ─► intercept ─► opening (turn < 50) ─► scored macro options
                                                                                            │
                                            one-turn exact safety veto ◄────────────────────┘
                                                        │
                                             sanitise ─► [kind, row, col, dir, split]
```

### 3.1 Memory and exact opponent accounting
- It keeps a turn-0 mountain snapshot, plus the last type, owner, army and turn seen for every cell.
- From each turn's change in `opp_army` and `opp_land`, it detects every enemy castle build and its exact price.
- It matches that build to the new "structure in fog" cell.
- The price gives a ring of possible enemy-general locations. This was validated on public replays: 28 of 28
  builds matched.

### 3.2 Where is the enemy general?
1. **Turn 0:** filter candidates with the generator's spawn rules.
2. **Each turn:** prune cells seen without a general, cells outside the castle-price rings, and cells
   inconsistent with where enemy armies leave the fog.
3. **Rank what's left** with a 10-weight logistic spawn prior (`PRIOR_W`, `PRIOR_B`), trained on the engine's own
   map generator.

### 3.3 Options scored every turn
- **capture:** take neutral or enemy land, timed around the land bonus, with an early-expansion bonus until
  turn 100.
- **garrison:** gather army onto the general, sized to visible threats, stacks tracked moving through fog, and
  part of the hidden enemy army.
- **army cycle:** gather a stack along value-per-move trees, then launch it at the likely general or an enemy
  castle. The route is stealthy, avoiding cells the enemy probably sees.
- **kill (the "sweep" check):** walk the exact path to the enemy general, adding our own cells and beating every
  enemy one. Strike when the stack arrives with more than the general's army plus a margin.
- **build, scout, home fill, intercept:** build castles, scout, own all cells near home, and chase-kill
  attackers next to the general or castles.

### 3.4 The context modulator (the learned part)
Every option weight is multiplied by `exp(clip(w·f, ±1.5))`.
- **Features** (8, each in [−1, 1]):
  - game phase;
  - army ratio;
  - land ratio;
  - number of enemy castles;
  - enemy gather style;
  - whether the enemy general is known;
  - the largest tracked threat;
  - a bias.
- **Weights:** 7 option groups × 8 features = 56, named `m_<group>_<j>`, trained by evolution strategies
  (section 4.3).

### 3.5 Safety layers
- **Exact one-turn veto:** each candidate move is resolved against every enemy move that could hit our general,
  using the engine's move order. Losing moves are rejected.
- **Deathtouch guard** (from turn 790):
  - an adjacent enemy stack is chased from a third tile;
  - a stack two steps away is killed, or the neighbour it would take is reinforced.
  - This was added after a real loss at turn 865 in a game we led 1,253 to 895 in army.
- **Endgame fortress:** from turn 696 the bot holds every neighbour of its general; from turn 760 it stages its
  own deathtouch strike.
- **Shell:**
  - `act()` never raises; any internal error returns a valid pass;
  - every output is cast to `int` and clamped;
  - observation keys are read through aliases;
  - typical moves take 2–8 ms.

---

## 4. How it was tuned

### 4.1 CMA-ES self-play
[`tune/cma_tune.py`](tune/cma_tune.py) tunes about 100 numbers: option weights, garrison fractions, launch sizes,
castle timing and so on.
- **Fair comparisons:** candidates play paired maps (the same maps, both seats).
- **Opponent pool:**
  - fixed opponents, mostly the neural clones;
  - PFSP weighting (opponents we lose to count more);
  - self-play against the current mean;
  - a league of past snapshots.
- **Output:** `mean_avg.json`, the average of the last 6 means. It's less noisy than the single best candidate.

### 4.2 Overfitting check
We generated 2,000 held-out maps that tuning never sees. The tuned bot scored 0.670 on them and 0.653 on the
tuning maps, so it learned strategy rather than maps. Every final decision was made on held-out slices no earlier
decision had used.

### 4.3 Evolution strategies on the modulator
- **Method:** black-box RL with CMA-ES over the 56 modulator weights, starting from the tuned bot, against a
  clone-heavy pool with self-play and league snapshots.
- **Why ES:** a deep network can't fit 150 ms of pure Python, so ES over a low-dimensional policy was the best RL
  we could actually deploy.
- **What it learned** (`agents_shared/final_params.json`):

| Group | Learned weight | Effect |
|---|---|---|
| launch | bias +0.51, +0.26 × enemy castles | about 1.7× more launches, more again against castle builders |
| garrison | bias −0.48, +0.45 × army ratio | thinner garrison, thicker when ahead |
| kill | bias −0.24 | strikes with a thinner margin |
| capture | bias −0.35, −0.41 × enemy castles | fewer small captures |
| scout / home | +0.46…+0.58 × threat / general known | more scouting and home fill under threat |

In short: **more aggressive, especially against economy builders.** That's worth +0.04–0.06 win rate on held-out
maps (section 7).

---

## 5. Who it was tested against

| Group | Bots | Notes |
|---|---|---|
| Neural clones of top ladder bots | `bc_resbot96`, `bc_resbot128`, `bc_nanomena96`, `bc_kubic96` | ResNet policies (96–128 channels) trained on a GPU from public generals.bot replays, with actions inferred by our exact engine (`learn/`). The strongest opponents we had, apart from KSolmann. |
| KSolmann's transformer | `ext_ksolmann` | A 3M-parameter AverageJoe-style model trained with PPO for this ruleset. The strongest bot we found; it can't run under the stdlib / 150 ms rule. |
| Public bots from GitHub | Sentinel and Sentinel v10, bca (an RL conv-net), hvn, juraj34/35, superbot, boss, Human.exe, A9, doomstack, amin, mybot9 | Wrappers in `bots/opp/ext_*.py`; sources in `agents_shared/bot-scout.md` |
| Hand-written "zoo" | rusher, hunter, turtle, zoo_flash / castler / gatherer / sniper / turtle_dt / expander_plus / mixed | Each one probes a single style |
| Our older versions | t1c, c_a2es3, F0, … | Regression checks |

---

## 6. Experiments: what we kept and what we dropped
Every change was measured with paired-map A/B tests, usually 400–1,000 games. The full log is in
[`agents_shared/ROADMAP.md`](agents_shared/ROADMAP.md).

**Kept**

| Change | Effect |
|---|---|
| Early expansion until turn 100 | 0.738 → 0.766 |
| Stealth routing of strikes (`stealth_w` 0.5) | 0.738 → 0.782 |
| Chase-kill interceptor | 0.738 → 0.753 |
| Sweep kill check (exact path walk, 10 candidate stacks; idea from Sentinel) | 0.580 → 0.620 |
| Deathtouch guard | fixes a real class of turn-800+ losses (unit-tested) |
| ES-tuned context modulator | 0.71 → 0.75 on held-out maps |

**Dropped**

| Idea | What happened |
|---|---|
| Earlier or more castles | 0.43 → 0.31 vs clones: our castles get drained and taken |
| Keep half the army on castles | 0.53 → 0.48 |
| Attack only when ahead | 0.56 → 0.42 / 0.09: launches are what win games |
| Bonus-timed enemy captures; spreading after each land bonus | 0.54 → 0.44–0.49; 0.555 → 0.458 |
| Longer early expansion (to turn 130 / 160) | 0.645 → 0.615 / 0.557 |
| Kill-front gathering, stealth kill paths, kill-margin changes | negative or no effect |
| A learned threat model (gradient boosting on replays) | no gain; removed |

**Lesson:** the tuned bot sits at a strong local optimum. Turns moved from attack cycles into "economy" kept
losing, even though the strongest opponents out-grow us economically. The gains came from better strikes, fewer
blunders and RL-tuned aggression.

---

## 7. Results

### 7.1 Held-out maps (100 games per opponent, both seats)
"Pooled" is the average over 10 opponents: 1,000 games, about ±0.027 at 95 %.

| Bot | Map slice | Pooled | ResBot-96 | ResBot-128 | nanomena | Kubic | t1c | sniper | flash | mixed | rusher | hunter |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| earlier best (c_a2es3) | 1700 | 0.665 | .36 | .32 | .48 | .41 | .72 | .78 | .92 | .87 | .80 | .99 |
| without modulator (F0) | 1700 | 0.713 | .45 | .39 | .57 | .45 | .69 | .89 | .94 | .93 | .84 | .99 |
| **final (F2)** | 1700 | **0.752** | **.56** | **.58** | **.66** | **.66** | .57 | .92 | .90 | .94 | .78 | .96 |
| without modulator (F0) | 1800 | 0.692 | .46 | .34 | .46 | .54 | .68 | .84 | .91 | .90 | .84 | .96 |
| **final (F2)** | 1800 | **0.754** | **.64** | **.55** | **.69** | **.64** | .57 | .89 | .89 | .92 | .86 | .90 |

### 7.2 Public bots (40 games each; bca and boss 100)

| Bot | Sentinel v10 | Sentinel | hvn | juraj35 | superbot | boss | bca (RL) | pooled |
|---|---|---|---|---|---|---|---|---|
| F0 | .83 | .83 | .68 | .89 | .93 | .93 | .53 | 0.816 |
| **F2** | .85 | .95 | .78 | .88 | 1.00 | .90 | .44 | 0.821 |

Against KSolmann's transformer, F2 scores 0.27–0.37 (100-game slices).

### 7.3 Safety checks on the final file

| Check | Result |
|---|---|
| **Organizers' evaluator**, 1 hour: the official Docker sandbox at official limits, a full `tournament` run plus a long-game loop | **223 games, 0 faults.** Worst move 13.6 ms; first move ≤ 80 ms. Two games went past turn 800: a deathtouch win at 801 and a simultaneous-capture draw at 829. |
| Catastrophe gate: 20 simple, zoo and public bots × 30 games | 0 forfeits in 600 games. The lowest score was 0.83 (zoo_gatherer). |
| Subprocess runner: 1 shared core, 150 ms, 3× CPU slowdown | 0 timeouts. Worst move 56.6 ms; worst first move 529 ms. |
| Unit tests (`pytest tests/`) | engine parity, rules, accounting, tactics (incl. the deathtouch guard), robustness: all pass |

### 7.4 Why F2 and not the base bot (F0)
- **For F2:**
  - best on both held-out slices (+0.04 and +0.06);
  - best worst case (0.44 vs bca, against F0's 0.34);
  - the only version that beats every neural clone.
- **Against F2:**
  - weaker vs t1c (0.57 vs 0.69);
  - F0 beat F2 head-to-head 0.585 in the official-sandbox run (59 games).
- **Blends:** a half-strength modulator (0.725) and one without its garrison cut (0.739) both landed in between.

---

## 8. The RL deep-dive: did "proper" RL beat it?
Short answer: **no, not under a fair test.** Two Claude Code sessions ran RL tracks in parallel overnight. Every
result below is a paired, deterministic evaluation on held-out maps.

**The constraint that shapes everything.**
- A pure neural policy small enough for 150 ms of pure Python plays at only **0.17** against the heuristic. That
  was a 12×1 conv student: 30 KB, 11.6 ms per move, using big-integer "lane" convolutions.
- PPO fine-tuning made it worse (0.21 → 0.08–0.19).
- So RL was used to learn the **decision layer** on top of the heuristic, which is deployable as plain numbers.

| Track | Method | Code |
|---|---|---|
| A | **RO-PPO**: a residual policy rescores the heuristic's candidate options each turn. It has 24 global + 8 option features and a 16-unit hidden layer. Trained with PPO + GAE, potential-based shaping annealed to 0, semi-MDP discounting, KL to the heuristic, and a PFSP league. About 75k–90k games per run. | `rl/hier/`, `rl/final/` |
| B | Behaviour-cloned conv "students" compiled to pure Python, then PPO | `rl/students/`, `rl/py*.py` |
| C1 / C3 | **GRPO/PPO** over the modulator: Gaussian logit exploration, group-relative advantages on the engine's outcome, PPO-clip, KL anchor to F2. C3 adds a 12-unit MLP. | `rl/c/grpo.py` |
| E1 | OpenAI-ES (antithetic, common random numbers) over an extended modulator | `rl/c/es.py` |
| es5 | Rerun of the CMA-ES that produced F2 | `tune/cma_tune.py` |

**Correctness checks:**
- The learner gradients match finite differences.
- The bot and the learner compute the identical policy.
- With zero new weights, every candidate plays move-for-move like F2.

**Joint final** (held-out maps, untouched until then).
- The decision rule: decide on **stdlib-feasible heuristic opponents**, because real entries must also run in
  pure Python.
- Neural opponents are only a guardrail: a candidate may be no more than 0.05 worse than F2 against them.

| Bot | 13 in-house heuristic bots | 6 independent public bots | Neural guardrail (5 bots) | vs F2 head-to-head |
|---|---|---|---|---|
| **F2** | 0.804 | **0.908** | **0.562** | – |
| RO-PPO (`ro_a2_p2avg`) | **0.844** | 0.900 | 0.479 (fails, −0.083) | 0.64 |
| RO-PPO (`ro_a2_avg`) | 0.845 | 0.846 | 0.475 (fails, −0.087) | 0.58 |
| GRPO (C1) | 0.802 | 0.890 | 0.569 | 0.485 |

**The lesson.**
- RO-PPO learned a real edge against the bot family it trained with, beating F2 0.64 head-to-head.
- That edge **did not transfer** to independent bots, and it cost 0.08 against strong learned play.
- The modulator runs (GRPO, ES, CMA-ES) all landed within ±0.02 of F2.

Design notes are in [`rl/DESIGN.md`](rl/DESIGN.md) and the timestamped log is in [`rl/board.md`](rl/board.md).

A side finding: the CMA-ES tuner hung twice inside `numpy.linalg.eigh`, which pycma calls in a process that keeps
forking pool workers. A `faulthandler` watchdog pinned it down. The fix is single-threaded BLAS (`OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`).

---

## 9. Tournament outcome and lessons
The bot made the **knockout stage (top 16)** and was eliminated in the **round of 16**. We don't have the
knockout games' logs. These are the takeaways from our own data:
- **Robustness beats head-to-head gains.** Several candidates beat our own bots but lost ground against
  independent ones. Evaluate on opponents you didn't train against before adopting anything.
- **Economy is the open weakness.** By turn 300 the strongest opponents hold 1.5–2× our land and keep their
  castles. The bot wins by striking first, and long games favour them.
- **Merged attacks.** The garrison need is the *maximum* over single threats, so two stacks merging next to the
  general can beat it.
- **Knockout series are short.** A 4–6-game series between near-equal bots is close to a coin flip, and small
  matchup weaknesses decide it.

Ideas we didn't get to:
- a merge-aware threat estimate;
- 2-ply tactical search near the general;
- a learned "launch now?" value;
- longer ES runs with more games per candidate.

---

## 10. Rule compliance
- **Single file, stdlib only:** imports `collections`, `gc`, `heapq`, `math` and `time`. No file, network,
  thread or subprocess use, no `print`, no `eval`/`exec` (checked by `tools/check_submission.py`).
- **Inputs:** it reads only the observation dictionary and its own memory.
- **Embedded numbers:** only parameters produced by our own scripts (`PARAMS`, the modulator and the spawn prior).
  No third-party code or weights.
- **No copied bot code:** `tools/copy_audit.py` compared the file with 1,104 third-party source files. The longest
  shared token run is the engine-defined direction table `DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))`.
- **Disclosure:** the file header lists the strategy, the borrowed ideas (with credit) and the AI assistance.
  The code was written with Claude Code; no AI is called at game time.

---

## 11. Repository layout

| Path | What it is |
|---|---|
| `submission/saiyam_f2.py` | **The final bot**, a self-contained file |
| `bots/participant.py` | Bot source with tunable `PARAMS` (the submission = this + `agents_shared/final_params.json`) |
| `bots/versions/` | Frozen candidates and A/B variants (`F2.py` = final, `F0.py` = without the modulator) |
| `bots/opp/` | Sparring partners: zoo, public-bot wrappers (`ext_*`), neural clones (`bc_*`) |
| `sim/` | Exact engine replica (`engine.py`) and map generator export (`make_maps.py`) |
| `arena/` | Fast in-process runner (`run.py`) and realistic one-core subprocess runner (`subproc.py`) |
| `tune/` | CMA-ES / ES tuner and SLURM job scripts |
| `learn/` | Replay download and parsing, action inference, BC datasets and training, spawn-prior training |
| `rl/` | The RL deep-dive (section 8) |
| `tools/` | A/B tools, gate, submission build and check, copy audit, loss and economy analysis, viewers |
| `tests/` | Engine parity, rules, accounting, tactics, robustness |
| `agents_shared/` | Research and experiment reports from the helper agents; `ROADMAP.md` is the experiment log |
| `data/` | Small replay-analysis outputs (map pools are generated by `setup.sh`) |
| `CLAUDE.md`, `OPEN_QUESTIONS.md`, `generals_bot_research_guide.md` | Working notes from the event (agent instructions, open questions, initial research guide) |

---

## 12. Running experiments on a SLURM cluster
Heavy tuning and evaluation ran on a SLURM GPU/CPU cluster. The scripts in `tune/slurm/` and `rl/slurm/` follow
this pattern:
1. Rsync the code to a small staging directory in `$HOME`.
2. Each job copies it to node-local `/scratch/$USER/...`.
3. The job builds its own `uv` virtualenv there, with every cache pointed at scratch.
4. It runs, then copies its summaries back.

The paths, account and QoS flags are specific to that cluster, so adjust them before use. Example evaluation job:

```bash
MAPS_FILE=data/maps_fresh.jsonl EVAL_TOOL=tools/abmulti.py sbatch --export=ALL -c 36 --mem=72G --time=04:00:00 \
    code/tune/slurm/eval.sbatch <tag> 100 0 --bots <bots...> --opps <opps...> --offset <map offset>
```

Use single-threaded BLAS in every job (section 8), and note that held-out map offsets wrap modulo the pool size.

---

## 13. Credits
- **Engine:** [`strakam/generals-bots`](https://github.com/strakam/generals-bots) (MIT), pinned at `13db8f69`, and
  the generals.bot competition rules.
- **Ideas** (no code copied):
  - [EklipZ's generals-bot](https://github.com/EklipZgit/generals-bot): gather pruning, back-tracing;
  - relh's Sentinel and juraj's bots: the chase-kill interceptor and the "sweep" kill check;
  - Straka & Schmid, [arXiv 2507.06825](https://arxiv.org/abs/2507.06825);
  - statistics from the public generals.bot Marathon replays.
- **Sparring partners only** (not included): the public bots listed in `agents_shared/bot-scout.md`, and the
  public generals.bot replays used to train the neural clones.
- **Built with** [Claude Code](https://claude.com/claude-code). It ran several parallel agents for research,
  experiments and the RL tracks; their reports are in `agents_shared/` and `rl/agents/`.
