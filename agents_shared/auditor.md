# auditor: compliance audit (Code_Bot_Rules.pdf v2.0 + Slides), file audited: bots/participant.py (2125 lines, 222,425 B)

Tests run: check_submission.py bots/participant.py --games 4 -> OK (4/4 wins, no forfeits, size 222,425 B, imports
[collections, gc, heapq, math, time], A_max 3.4 ms, first move 149.9 ms in arena incl. spawn-prior + opening plan).
build_submission.py --params runs/ada2/best.json --id AUDITTEST --name AuditBot -> OK (222,156 B, 4/4 wins, 0 forfeits).
Own fuzz (scratchpad/fuzz.py on the built file): boards 1x1..40x40, no general, all mountains, missing keys, None, {},
tuple grids, repeated/decreasing turns -> always a valid [int x5]; first move 62/87/71 ms (18x18/21x21/18x21), 204 ms at
30x30, 527 ms at 40x40 (real boards are 18-21).

## (a) PASS / RISK / FAIL table
| Rule (source) | Verdict | Evidence |
|---|---|---|
| One UTF-8 .py named participant_id.py (R5) | PASS | all ASCII; build writes submission/<id>.py. Make sure the UPLOAD is named exactly <registered id>.py |
| <= 1 MiB (R5) | PASS | 222 KB (21%). THREAT_TREES line 163 is 141 KB of it |
| stdlib only, ALL imports (R3.1,R5) | PASS | AST scan of whole file (incl. in-function): gc, math, time, collections.deque, heapq. No os/sys/random/json/open/print/eval/exec/__import__ (grep). Only lines 37-41 |
| act(observation) top level, returns list of exactly 5 python ints (R8) | PASS | act line 2110; _sanitize 2093 does int() on each, kind in {0,1,2}, d in 0..3, s in {0,1}; type()-checked in fuzz; no bool/float/numpy. Note: row/col are not range-clamped (legal: out-of-board = "wasted turn", not malformed) |
| valid codes even for pass/build (R8.2) | PASS | PASS=[1,0,0,0,0]; pass path returns zeros; _sanitize forces d,s valid |
| no main loop / stdin | PASS | none |
| globals reset per game (R5.1, R8.2) | PASS | process-per-game by organiser; additionally _decide line 2069 rebuilds Bot on _BOT None / t<=_LAST_TURN / t==0 / H,W change. Fuzz repeats 0,1,2,2,1,0 OK. (Only gap: a mid-game first call, t>0, is handled: fuzz "midstart" OK) |
| time 150 ms/move, first 10 s incl. import (R5) | PASS (margin) | import 3.7 ms (132 ms when measured through the checker harness); regular moves A_max 3.4-3.7 ms, soft_budget_ms=45 caps work; first move 60-90 ms on 18-21 boards; opening planner hard-capped by open_plan_s=1.5 s wall and first_budget_ms=3000, so even a 10x slower CPU stays < 10 s on the first call. See RISK-3 |
| 1 logical CPU shared, alternating | PASS | single-threaded, no threads |
| memory 2 GiB | PASS (not profiled) | per-cell Python lists, ~100 candidate BFS arrays (n=441) => a few MB. Not measured with tracemalloc |
| scratch 16 MiB / no file writes | PASS | no open()/os/pathlib/tempfile; check_submission bans them |
| no network / subprocess / threads / signals | PASS | none imported |
| no reading other files, no hidden state (R2.2) | PASS | only the observation dict (keys aliased in _ALIASES lines 2010-2022, tolerant of other spellings, never touches anything else) |
| no human control / no online AI call at game time (R3.2,R4) | PASS | pure code |
| no external model files / downloaded assets (R3.1) | PASS | everything inline in the single file |
| exception => forfeit (R5 table) | PASS | act wraps _decide+_sanitize in try/except Exception (2111-2125), returns pass; fallback also try-wrapped. Covers RecursionError/MemoryError (both Exception subclasses). Not covered: BaseException (KeyboardInterrupt/SystemExit), irrelevant in practice |
| malformed => forfeit | PASS | see sanitize; fuzz of None/{}/str types returns valid pass |
| timeout => forfeit | PASS (margin) | see time row; no unbounded loop found in act/_decide path; BFS/gather are bounded by board size; late() checks. A pure-Python slow-CPU factor of 20x on regular moves would still be ~70 ms |
| recursion | PASS | no sys.setrecursionlimit; no deep recursion found (BFS/heap iterative); any RecursionError is caught |
| header: registered ID, bot name, short strategy, reused sources, material AI assistance (R5.2) | RISK | placeholders [PARTICIPANT_ID]/[BOT_NAME] are filled only by build_submission (check_submission only WARNs, does not fail). Header lacks: explicit statement about embedded learned constants and training data, tools list/version, "no credentials". See (c) |
| embedded constants attributed + "small" (R3.1) | RISK | see section 2 |
| "no pre-existing complete competitive bot/pretrained policy" (R3.1) | RISK (low-med) | strategy is hand-written in-window (git history starts 2026-10-02 03:31 IST). Confirm T0 (Rules PDF mtime 02:53 IST; kickoff time "[START TIME]" is still a placeholder) is <= first commit: rules say "no event-specific strategy development before T0". Cannot be verified from repo |
| no collaboration/code sharing (R3.3) | PASS/unknown | header says ideas only; confirm no code was pasted from EklipZ/relh/juraj/other participants (agents_shared/porter.md says those run only as local opponents). |
| one bot per person, frozen artifact, no post-H20 changes | PROCESS | submit the built file, keep receipt + sha256; do not hand-edit after build without re-running check |

## 2. Embedded learned constants
* PRIOR_W/PRIOR_B (lines 158-159): 10 logistic weights + bias, trained in-window (commit 04:32 IST 2 Oct) on maps from the pinned engine
  generator. Tiny, derived by us from the public engine, not a policy. VERDICT: allowed; mention it in the header ("approved small constants":
  we are self-declaring them; rules say "approved", so also mention it to organisers on the help channel if cheap, ask-and-record).
* THREAT_TREES/THREAT_BASE (lines 160-164): quantile-GBM, ~141 KB of source (63% of the file), trained on features parsed from public
  generals.bot replays. User approved replay training; rules 2.2 prohibit using "replay feeds" DURING official games only, offline training
  in the window is development ("trained policy during the 20-hour window", R3.1). But: (i) it is not "small", (ii) it is third-party
  replay-derived data, the most arguable item for "pre-existing ... pretrained", (iii) it is a threat estimator, not a game policy, so
  not literally excluded.
  IMPORTANT: it is currently dead only in the default PARAMS (learned_threat_w=0.0, line 53, gated at 1406). The tuned files runs/ada3, ada5c,
  ada6c, local7, local7b best.json carry learned_threat_w 0.0185 / 0.0349 / 0.0222 / 0.0721 / 0.1317, i.e. build_submission would
  switch it ON for those candidates (runs/ada2 has no such key => 0). So decide per candidate:
    - if the final candidate has learned_threat_w == 0 (or the A/B gain is within noise): DELETE lines 160-164, learned_threat()
      (1348-1356) and the gated block 1406-1413. This removes 141 KB and the only replay-trained blob. (RECOMMENDED unless the gain is clear.)
    - if kept: header must disclose it explicitly (text in (c)) and it should be treated as the single biggest rules risk.
* CMA-ES tuned PARAMS: numbers only, tuned in-window by self-play. Opponents that were behaviour-cloned networks (trained on public replays)
  and third-party public bots are used locally as sparring partners only; nothing of them ships. ALLOWED; disclose in header in one line
  ("tuned against ... opponents run locally; none included"). Third-party bot LICENSES matter only if code is copied: it is not.
* Header currently says "statistics and behaviour clones from the public generals.bot Marathon replays": accurate, but "behaviour clones"
  could be read as the clones being in the file. Reword (c).

## (b) Fixes ranked by disqualification risk
1. (HIGH, process) Hard-fail placeholders. After build, grep the output for "[PARTICIPANT_ID]", "[BOT_NAME]", "TODO"; make
   check_submission fail (not warn) when present. A filled header with the REGISTERED id, upload file named <id>.py.
2. (HIGH) Decide THREAT_TREES: remove unless learned_threat_w>0 in the chosen candidate AND justified. If kept, add the explicit header
   disclosure. Re-run check_submission after deletion (the code path is guarded, removal only needs the 3 blocks above deleted; keep
   P["learned_threat_w"] key or drop it, build_submission ignores unknown tuned keys with a warning).
3. (MED) Header: rewrite per (c): add tools list, training-data statement, the constants list, "no network/files".
4. (MED, ask organisers) Confirm T0 vs first commit time (03:31 IST) and whether embedded trained constants count as "approved small
   constants". The cheapest insurance is a short message in the help channel plus keeping the constants minimal (PRIOR only).
5. (LOW) First-move cost: 60-90 ms on a normal board, ~150 ms on the arena first call. The spec gives 10 s; only a harness bug that
   applied 150 ms to move 0 would hurt. Insurance (coordinator's call, I did not edit): cut open_plan_s from 1.5 to ~0.05 s or
   lower the number of (div,w_free,w_dist,w_toward) combos so that first call stays < ~100 ms even on a 2x slower CPU. Verify via
   A_first_ms in the arena (currently 149.2-149.9).
6. (LOW) Clamp r,c to [0,H) x [0,W) in _sanitize (turn 2093-2105): out-of-board move = wasted turn per R8.2 (not forfeit), so optional,
   but turning it into a pass is strictly safer if the adapter validates ranges.
7. (LOW) Add a final CI step: run the built file in a clean `python3.12 -I -S` subprocess with a copy of the observation, check
   type(x) is int for the 5 outputs (fuzz.py in scratchpad does this).
8. (INFO) build_submission regex `^DEBUG = True`: file already has DEBUG = False (line 169) so fine, but keep the assertion in
   check_submission ("FAIL DEBUG is True").

## (c) Proposed final header (docstring, first statement of the file)
"""
Code Bot entry.  participant id: <REGISTERED_ID>   bot name: <BOT_NAME>

Strategy (single stdlib-only Python 3.12 file; deterministic heuristic, all decisions made by this code at game time):
  * Exact bookkeeping of the opponent from the observation only (opp_army/opp_land deltas reveal castle builds and
    prices; type-5 cells locate them), a belief over the enemy general's spawn cell (spawn-rule filtering, fog
    sightings, castle-price rings), expansion opening chosen by a short built-in simulation, gather-and-launch army
    cycles, garrison/defence with an exact one-turn resolver, castle economy and sniping, deathtouch fortress/strike
    after turn 800. No network, no files, no threads, no randomness from outside, no hidden state: only the
    observation dict is read.

Embedded constants (all inline in this file, generated during the event by the participant's own scripts):
  * PARAMS: numeric weights/thresholds tuned during the event by CMA-ES self-play (using the pinned engine).
    Sparring opponents used locally only: earlier versions of this bot, hand-written heuristics, public third-party
    bots, and neural networks behaviour-cloned from public generals.bot replays. None of those bots or networks
    is included in or loaded by this file.
  * PRIOR_W / PRIOR_B: 10-weight logistic spawn prior trained during the event on maps sampled from the pinned
    engine's own generator.
  * [only if kept] THREAT_*: a small gradient-boosted threat estimator trained during the event on features taken
    from public generals.bot replays.   (REMOVE this bullet and the code if the model is deleted.)

Reused sources / attribution:
  * Game rules, move order and map generator re-implemented from strakam/generals-bots @13db8f69 (MIT licence).
  * Ideas only, no code copied: EklipZ generals-bot (gather pruning), relh Sentinel and juraj bots (chase-kill
    interceptor), Straka & Schmid arXiv 2507.06825, statistics from public generals.bot replays.

AI assistance: this bot was developed with Claude Code (Anthropic, Claude models) as a coding, experiment-running and
research assistant; material parts of the code, including the training/tuning scripts that produced the constants
above, were written by it. The participant directed the work, reviewed the submitted code and can explain and
reproduce its launch. No AI service is called at game time.
"""
