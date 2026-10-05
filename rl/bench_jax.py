"""Pinned JAX engine, COMPETITION rules (build_castles + deathtouch@800 + fog obs for both players),
vmapped over N games inside lax.scan, on CPU.

    taskset -c 2 env JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
        PYTHONPATH=vendor/generals-bots:. .venv312/bin/python rl/bench_jax.py --envs 1 16 64 256

Boards come from data/maps.jsonl padded to 21x21 with mountains (vmap needs one shape).
Actions are uniform random (10% pass, 5% build), as in the vendor paper benchmark.
Variants: step = transition only; env = transition + get_observation(p) for p in 0,1;
env_feat = env + a JAX port of the fog-memory part of the features (elementwise only).
"""
import argparse
import functools
import json
import os
import time

import jax
import jax.numpy as jnp
import jax.random as jr

from generals.core import game
from generals.core.game import create_initial_state
from generals.modifiers import build_castles as BC
from generals.modifiers import deathtouch as DT

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
S = 21


def boards(n):
    out = []
    with open(os.path.join(ROOT, "data", "maps.jsonl")) as f:
        for line in f:
            m = json.loads(line)
            g = [[-2] * S for _ in range(S)]
            for r in range(m["H"]):
                for c in range(m["W"]):
                    g[r][c] = m["grid"][r][c]
            out.append(create_initial_state(jnp.array(g, jnp.int32)))
            if len(out) >= n:
                break
    return jax.tree.map(lambda *xs: jnp.stack(xs), *out)


def transition(state, actions):
    state, actions = BC.apply_build_actions(state, actions)
    return DT.step(state, actions, 800)


def rand_actions(key, n):
    k = jr.split(key, 6)
    kind = jnp.where(jr.uniform(k[0], (n, 2)) < 0.1, 1, 0)
    kind = jnp.where(jr.uniform(k[5], (n, 2)) < 0.05, 2, kind)
    return jnp.stack([kind, jr.randint(k[1], (n, 2), 0, S), jr.randint(k[2], (n, 2), 0, S),
                      jr.randint(k[3], (n, 2), 0, 4), jr.randint(k[4], (n, 2), 0, 2)], -1).astype(jnp.int32)


def mem_update(mem, ob):
    """Fog memory (castles/mountains seen, last-seen time, last enemy army) - elementwise."""
    mnt, cas, last_seen, last_enemy = mem
    vis = ~ob.fog_cells & ~ob.structures_in_fog
    mnt = mnt | ob.mountains | ob.structures_in_fog & ~cas
    cas = cas | ob.castles
    last_seen = jnp.where(vis, ob.timestep, last_seen)
    last_enemy = jnp.where(vis, jnp.where(ob.opponent_cells, ob.armies, 0), last_enemy)
    planes = jnp.stack([ob.owned_cells, ob.opponent_cells, ob.neutral_cells, vis, mnt, cas,
                        ob.generals]).astype(jnp.float32)
    planes = jnp.concatenate([planes, jnp.log1p(ob.armies.astype(jnp.float32))[None],
                              jnp.log1p(last_enemy.astype(jnp.float32))[None],
                              jnp.minimum(ob.timestep - last_seen, 255)[None].astype(jnp.float32)])
    return (mnt, cas, last_seen, last_enemy), planes


def make(n, variant):
    vtrans = jax.vmap(transition)
    vobs = jax.vmap(game.get_observation, in_axes=(0, None))
    vmem = jax.vmap(mem_update)

    @functools.partial(jax.jit, static_argnums=1)
    def run(carry, steps):
        def body(c, _):
            s, key, mem = c
            key, k = jr.split(key)
            s, info = vtrans(s, rand_actions(k, n))
            acc = info.winner.sum()
            if variant != "step":
                o0, o1 = vobs(s, 0), vobs(s, 1)
                acc = acc + o0.armies.sum() + o1.armies.sum()
                if variant == "env_feat":
                    m0, p0 = vmem(mem[0], o0)
                    m1, p1 = vmem(mem[1], o1)
                    mem = (m0, m1)
                    acc = acc + p0.sum() + p1.sum()
            return (s, key, mem), acc
        c, r = jax.lax.scan(body, carry, None, length=steps)
        return c, r.sum()
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, nargs="+", default=[1, 16, 64, 256])
    ap.add_argument("--seconds", type=float, default=6)
    ap.add_argument("--variants", nargs="+", default=["step", "env", "env_feat"])
    a = ap.parse_args()
    print("# jax", jax.__version__, jax.devices()[0].platform)
    for n in a.envs:
        st = boards(n)
        z = jnp.zeros((n, S, S), bool)
        zi = jnp.zeros((n, S, S), jnp.int32)
        mem1 = (z, z, zi - 1, zi)
        mem = (mem1, mem1)
        steps = max(20, min(200, 20000 // n))
        for v in a.variants:
            fn = make(n, v)
            t0 = time.perf_counter()
            out = fn((st, jr.PRNGKey(0), mem), steps)
            jax.block_until_ready(out)
            comp = time.perf_counter() - t0
            frames, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < a.seconds:
                out = fn((st, jr.PRNGKey(frames), mem), steps)
                jax.block_until_ready(out)
                frames += n * steps
            fps = frames / (time.perf_counter() - t0)
            print(json.dumps({"envs": n, "variant": v, "frames_per_s": round(fps), "compile_s": round(comp, 1)}),
                  flush=True)


if __name__ == "__main__":
    main()
