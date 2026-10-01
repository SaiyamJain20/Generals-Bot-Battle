"""Export competition boards from the pinned JAX generator to JSONL.

Each line: {"seed": s, "H": h, "W": w, "grid": [[...], ...]} where grid uses
the engine's numeric encoding after strip_neutral_castles: -2 mountain,
0 empty, 1/2 general of player 0/1. Identical to competition/matchup.make_board.

    python sim/make_maps.py --start 0 --count 20000 --out data/maps.jsonl
"""
import argparse
import json
import sys
import time

import jax.numpy as jnp
import jax.random as jrandom

from generals import GeneralsEnv
from generals.core.grid import generate_grid
from generals.modifiers import build_castles as _bc


def board_grid(env, seed):
    key = jrandom.PRNGKey(seed)
    kd, kg = jrandom.split(key)
    lo, hi = env.min_grid_size, env.max_grid_size
    h = int(jrandom.randint(kd, (), lo, hi + 1))
    w = int(jrandom.randint(jrandom.fold_in(kd, 1), (), lo, hi + 1))
    grid = generate_grid(
        kg, grid_dims=(h, w),
        mountain_density_range=env.mountain_density_range,
        num_castles_range=env.num_castles_range,
        min_generals_distance=env.min_generals_distance,
        castle_val_range=env.castle_val_range,
    )[:h, :w]
    grid = _bc.strip_neutral_castles(grid)
    return h, w, [[int(x) for x in row] for row in grid.astype(jnp.int32).tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--out", default="data/maps.jsonl")
    args = ap.parse_args()
    env = GeneralsEnv(mode="competition")
    t0 = time.time()
    with open(args.out, "a") as f:
        for k, seed in enumerate(range(args.start, args.start + args.count)):
            h, w, grid = board_grid(env, seed)
            f.write(json.dumps({"seed": seed, "H": h, "W": w, "grid": grid}) + "\n")
            if (k + 1) % 200 == 0:
                f.flush()
                print(f"[maps] {k + 1}/{args.count} {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
