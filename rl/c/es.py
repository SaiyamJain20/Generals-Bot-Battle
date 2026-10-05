"""Track C, variant E: evolution strategies (OpenAI-ES, antithetic, centred ranks) over the context
modulator of participant_c.py. Each candidate keeps ONE perturbation for whole games (parameter-space
noise), which is what made es4 work; GRPO's per-window noise had too little signal per game.

Generation:
  * N antithetic pairs theta +- sigma * eps_i, plus the current mean (incumbent, for logging).
  * Common random numbers: every candidate plays the same T tasks (opponent, map, seat).
  * Fitness = mean over tasks of R = outcome (+1/-1, draw = -draw) + margin * tanh(0.5 log army ratio).
  * Gradient  g = sum_i (u(F_i+) - u(F_i-)) eps_i / (2 N sigma) with centred ranks u in [-0.5, 0.5],
    minus decay * (theta - theta0); Adam ascent.
Opponents: PFSP over a broad fixed pool + snapshots of the mean every --league-every generations.

    python rl/c/es.py --out rl/c/runs/e1 --hours 6.5 --workers 36 --pairs 12 --tasks 40
"""
import argparse
import faulthandler
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import worker as Wk  # noqa: E402
import grpo as G  # noqa: E402  (Policy, params helpers, Adam, snapshots)

POOL = [
    "bots/versions/F2.py", "rl/c/opp/F0.py", "bots/versions/t1c.py", "bots/versions/c_a2es3.py",
    "bots/opp/bc_resbot96.py", "bots/opp/bc_resbot128.py", "bots/opp/bc_nanomena96.py", "bots/opp/bc_kubic96.py",
    "bots/opp/ext_ksolmann.py",
    "bots/opp/zoo_flash.py", "bots/opp/zoo_mixed.py", "bots/opp/zoo_sniper.py", "bots/opp/zoo_castler.py",
    "bots/opp/zoo_gatherer.py", "bots/opp/zoo_turtle_dt.py", "bots/opp/rusher.py", "bots/opp/hunter.py",
]


def centred_ranks(x):
    x = np.asarray(x, dtype=float)
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r / max(1, len(x) - 1) - 0.5


def _run(t):
    try:
        return Wk.play_c(t)
    except Exception as e:
        return {"outcome": None, "error": f"{type(e).__name__}: {e}", "opp": t.get("opp")}


def reward(r, args):
    o = r["outcome"]
    if o == 0.0:
        o = -args.draw
    return o + args.margin * math.tanh(0.5 * r.get("army_ratio", 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="bots/versions/F2.py")
    ap.add_argument("--start-json", default=None, help="optional params JSON (e.g. a GRPO checkpoint) to start from")
    ap.add_argument("--hours", type=float, default=6.5)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--tasks", type=int, default=40)
    ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--decay", type=float, default=0.002)
    ap.add_argument("--mlp", type=int, default=0)
    ap.add_argument("--draw", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--league-every", type=int, default=6)
    ap.add_argument("--league-max", type=int, default=3)
    ap.add_argument("--map-base", type=int, default=5000)
    ap.add_argument("--gen-timeout", type=float, default=2400)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    nrs = np.random.RandomState(args.seed)
    base = G.load_params(os.path.join(ROOT, args.start))
    if args.start_json:
        base.update(json.load(open(args.start_json)))
    for g in G.GROUPS:
        for j in range(G.NF):
            base.setdefault("m_%s_%d" % (g, j), 0.0)
    pol = G.Policy(G.w_from_params(base), H=args.mlp, seed=args.seed)
    if args.mlp and base.get("mh_n") == args.mlp:  # resume an MLP checkpoint
        for k in range(args.mlp):
            pol.c[k] = base.get("mc_%d" % k, 0.0)
            for j in range(G.NF):
                pol.U[k, j] = base.get("mu_%d_%d" % (k, j), 0.0)
            for gi, g in enumerate(G.GROUPS):
                pol.V[gi, k] = base.get("mv_%s_%d" % (g, k), 0.0)
    theta = pol.pack()
    theta0 = theta.copy()
    opt = G.Adam(theta.shape, args.lr)
    pool_paths = [os.path.join(ROOT, p) for p in POOL if os.path.exists(os.path.join(ROOT, p))]
    score = {p: 0.5 for p in pool_paths}
    league = []
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    pool = Pool(workers, maxtasksperchild=20)
    t_end = time.time() + 3600 * args.hours
    gen = 0
    map_ptr = args.map_base
    logf = open(os.path.join(args.out, "log.jsonl"), "a")
    json.dump({"args": vars(args), "pool": pool_paths}, open(os.path.join(args.out, "config.json"), "w"))
    hist = []
    while time.time() < t_end:
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(args.gen_timeout + 600, repeat=True)
        opps = pool_paths + league
        wts = [(1.0 - score.get(o, 0.5) + 0.15) ** 2 for o in opps]
        task_specs = []
        for k in range(args.tasks):
            o = rng.choice(opps) if rng.random() < 0.25 else rng.choices(opps, weights=wts)[0]
            task_specs.append((o, map_ptr + k // 2, k % 2))
        map_ptr += args.tasks // 2 + 1
        eps = nrs.randn(args.pairs, theta.size)
        cands = []
        for i in range(args.pairs):
            for sgn in (1.0, -1.0):
                th = theta + sgn * args.sigma * eps[i]
                pol.unpack(th)
                cands.append(pol.params(base))
        pol.unpack(theta)
        cands.append(pol.params(base))           # incumbent (mean) on the same tasks
        tasks, index = [], []
        for ci, p in enumerate(cands):
            for ti, (o, mi, seat) in enumerate(task_specs):
                tasks.append(dict(bot=G.PC, opp=o, map_idx=mi, seat=seat, params=p, sigma=0, W=8, seed=0))
                index.append((ci, ti))
        t0 = time.time()
        try:
            res = pool.map_async(_run, tasks, chunksize=1).get(timeout=args.gen_timeout)
        except Exception as e:
            print(json.dumps({"gen": gen, "warning": f"pool failure {type(e).__name__}; rebuilding"}), flush=True)
            pool.terminate()
            pool = Pool(workers, maxtasksperchild=20)
            continue
        # tasks with any failed game are dropped for everyone (keeps the comparison paired)
        bad = {ti for (ci, ti), r in zip(index, res) if r.get("outcome") is None}
        fit = np.zeros(len(cands))
        cnt = np.zeros(len(cands))
        per_opp = {}
        for (ci, ti), r in zip(index, res):
            if ti in bad:
                continue
            fit[ci] += reward(r, args)
            cnt[ci] += 1
            if ci == len(cands) - 1:
                per_opp.setdefault(os.path.basename(task_specs[ti][0]), []).append(0.5 * (r["outcome"] + 1))
        fit = fit / np.maximum(cnt, 1)
        # opponent difficulty (EMA) from all candidates' games
        for (ci, ti), r in zip(index, res):
            if ti in bad:
                continue
            o = task_specs[ti][0]
            score[o] = 0.97 * score.get(o, 0.5) + 0.03 * 0.5 * (r["outcome"] + 1)
        u = centred_ranks(fit[:-1])
        up, um = u[0::2], u[1::2]
        grad = ((up - um)[:, None] * eps).sum(0) / (2 * args.pairs * args.sigma)
        grad -= args.decay * (theta - theta0) / max(1e-6, args.sigma)
        theta = opt.step(theta, grad)
        pol.unpack(theta)
        inc_win = sum(sum(v) for v in per_opp.values()) / max(1, sum(len(v) for v in per_opp.values()))
        hist.append(inc_win)
        rec = {"gen": gen, "sec": round(time.time() - t0, 1), "games": len(tasks), "bad_tasks": len(bad),
               "inc_fit": round(float(fit[-1]), 4), "best_fit": round(float(fit[:-1].max()), 4),
               "mean_fit": round(float(fit[:-1].mean()), 4), "inc_win": round(inc_win, 3),
               "inc_win_avg5": round(sum(hist[-5:]) / len(hist[-5:]), 3),
               "drift": round(float(np.abs(theta - theta0).mean()), 4), "gnorm": round(float(np.linalg.norm(grad)), 3),
               "per_opp": {k: round(sum(v) / len(v), 2) for k, v in per_opp.items()}}
        print(json.dumps(rec), flush=True)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        params = pol.params(base)
        json.dump(params, open(os.path.join(args.out, "policy_params.json"), "w"), indent=0)
        json.dump(params, open(os.path.join(args.out, f"policy_g{gen}.json"), "w"), indent=0)
        if args.league_every and gen % args.league_every == args.league_every - 1:
            path = G.write_snapshot(os.path.join(args.out, f"snap_g{gen}.py"), params)
            league.append(path)
            score[path] = 0.5
            if len(league) > args.league_max:
                score.pop(league.pop(0), None)
        gen += 1
    pool.close()


if __name__ == "__main__":
    main()
