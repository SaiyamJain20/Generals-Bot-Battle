"""Differential test: sim/engine.py must reproduce the pinned JAX engine exactly.

Runs games with randomized, interaction-heavy policies (expansion, attacks
toward the enemy general, builds, splits, invalid actions) through both the
JAX transition used by competition/matchup.py and our pure-Python step, and
compares the full state after every turn plus the fogged observations.

    PYTHONPATH=vendor/generals-bots:. .venv312/bin/python tests/test_parity.py --games 30
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, deque

import jax.numpy as jnp
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "generals-bots"))
sys.path.insert(0, os.path.join(ROOT, "vendor", "generals-bots", "competition"))

from generals import GeneralsEnv  # noqa: E402
from generals.core import game  # noqa: E402
from generals.core.game import create_initial_state  # noqa: E402
from generals.modifiers import build_castles as _bc  # noqa: E402
from generals.modifiers import deathtouch as _dt  # noqa: E402
import protocol  # noqa: E402

from sim import engine as E  # noqa: E402


def jax_transition(state, actions):
    state, actions = _bc.apply_build_actions(state, actions)
    return _dt.step(state, actions, 800)


def bfs_from(s, src):
    H, W = s.H, s.W
    dist = [-1] * (H * W)
    dist[src] = 0
    dq = deque([src])
    while dq:
        i = dq.popleft()
        r, c = divmod(i, W)
        for dr, dc in E.DIRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                j = rr * W + cc
                if dist[j] < 0 and not s.mountain[j]:
                    dist[j] = dist[i] + 1
                    dq.append(j)
    return dist


def policy(s, p, rng, to_enemy):
    """Randomized interaction-heavy policy on the full state."""
    H, W = s.H, s.W
    x = rng.random()
    if x < 0.03:
        return [1, 0, 0, 0, 0]
    if x < 0.06:  # junk / invalid action
        return [rng.choice([0, 2]), rng.randrange(-1, H + 1), rng.randrange(-1, W + 1),
                rng.randrange(4), rng.randrange(2)]
    own = [i for i in range(H * W) if s.owner[i] == p and s.army[i] > 1]
    if not own:
        return [1, 0, 0, 0, 0]
    if x < 0.16:  # try to build on a rich own plain cell
        cands = [i for i in own if not s.general[i] and not s.castle[i]]
        if cands:
            i = max(cands, key=lambda j: s.army[j])
            return [2, i // W, i % W, 0, 0]
    if x < 0.55:  # step the biggest stack toward the enemy general
        i = max(own, key=lambda j: s.army[j] + rng.random())
        r, c = divmod(i, W)
        best = None
        for d, (dr, dc) in enumerate(E.DIRS):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                j = rr * W + cc
                if to_enemy[j] >= 0 and (best is None or to_enemy[j] < best[0]):
                    best = (to_enemy[j], d)
        if best is not None:
            return [0, r, c, best[1], 1 if rng.random() < 0.2 else 0]
    i = rng.choice(own)
    return [0, i // W, i % W, rng.randrange(4), 1 if rng.random() < 0.25 else 0]


def compare(js, s, info):
    arm = np.asarray(js.armies).reshape(-1).tolist()
    own = np.asarray(js.ownership)
    owner = [-1] * (s.H * s.W)
    for p in (0, 1):
        for i, v in enumerate(own[p].reshape(-1).tolist()):
            if v:
                owner[i] = p
    cas = np.asarray(js.castles).reshape(-1).tolist()
    gen = np.asarray(js.generals).reshape(-1).tolist()
    errs = []
    if arm != s.army:
        errs.append("army")
    if owner != s.owner:
        errs.append("owner")
    if [bool(v) for v in cas] != s.castle:
        errs.append("castle")
    if [bool(v) for v in gen] != s.general:
        errs.append("general")
    if int(js.time) != s.time:
        errs.append(f"time {int(js.time)} vs {s.time}")
    return errs


def obs_frame_sim(s, p):
    o = E.observe(s, p)
    lines = [f"{o['turn']} {o['my_land']} {o['my_army']} {o['opp_land']} {o['opp_army']}"]
    for g in (o["type"], o["owner"], o["army"]):
        for row in g:
            lines.append(" ".join(str(v) for v in row))
    return "\n".join(lines) + "\n"


def run_game(grid, seed, start_time=0, max_turns=1200, check_obs_every=7):
    rng = random.Random(seed)
    js = create_initial_state(jnp.asarray(grid, dtype=jnp.int32))
    s = E.from_grid(grid)
    if start_time:
        js = js._replace(time=jnp.int32(start_time))
        s.time = start_time
    to_enemy = [bfs_from(s, s.gpos[1]), bfs_from(s, s.gpos[0])]
    stats = Counter()
    while s.time < max_turns:
        if s.time % check_obs_every == 0:
            for p in (0, 1):
                fj = protocol.encode_observation(game.get_observation(js, p))
                if fj != obs_frame_sim(s, p):
                    return False, f"obs mismatch t={s.time} p={p}", stats
        acts = [policy(s, p, rng, to_enemy[p]) for p in (0, 1)]
        pre = s.copy()
        for p in (0, 1):
            if acts[p][0] == 2 and E.build_valid(pre, p, acts[p]):
                stats["build"] += 1
        a2 = [list(a) if a[0] != 2 else [1, 0, 0, 0, 0] for a in acts]
        if a2[0][0] == 0 and a2[1][0] == 0:
            for p in (0, 1):
                q = 1 - p
                dr, dc = E.DIRS[a2[p][3]]
                if (a2[p][1] + dr, a2[p][2] + dc) == (a2[q][1], a2[q][2]):
                    stats["chase"] += 1
        js, info = jax_transition(js, jnp.asarray(acts, dtype=jnp.int32))
        E.step(s, acts)
        jdone = bool(info.is_done)
        jwin = int(info.winner)
        if s.done and not jdone and s.winner == -1 and s.time >= E.TRUNCATION:
            errs = compare(js, s, info)  # truncation is the runner's job in JAX
            if errs:
                return False, f"state mismatch at truncation: {errs}", stats
            stats["truncated"] += 1
            break
        if jdone or s.done:
            if jdone != s.done or jwin != s.winner:
                return False, f"terminal mismatch t={s.time} jax=({jdone},{jwin}) sim=({s.done},{s.winner})", stats
            stats["win" if jwin >= 0 else "draw"] += 1
            if s.time > 800:
                stats["late_end"] += 1
            break
        errs = compare(js, s, info)
        if errs:
            return False, f"state mismatch t={s.time}: {errs} acts={acts}", stats
        stats["steps"] += 1
    else:
        stats["truncated"] += 1
    return True, "", stats


def load_grids(n):
    path = os.path.join(ROOT, "data", "maps.jsonl")
    grids = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    grids.append(json.loads(line)["grid"])
                except Exception:
                    break
                if len(grids) >= n:
                    break
    if len(grids) < n:
        sys.path.insert(0, os.path.join(ROOT, "sim"))
        from make_maps import board_grid
        env = GeneralsEnv(mode="competition")
        for seed in range(10_000, 10_000 + n - len(grids)):
            grids.append(board_grid(env, seed)[2])
    return grids


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=12)
    args = ap.parse_args(argv)
    grids = load_grids(args.games)
    total = Counter()
    for k, grid in enumerate(grids):
        start = 0 if k % 3 == 0 else (650 if k % 3 == 1 else 780)
        ok, msg, st = run_game(grid, seed=k, start_time=start)
        total.update(st)
        print(f"game {k} start={start} ok={ok} {msg} {dict(st)}", flush=True)
        if not ok:
            return 1
    print("TOTAL", dict(total))
    return 0


def test_parity_small():
    assert main(["--games", "3"]) == 0


if __name__ == "__main__":
    sys.exit(main())
