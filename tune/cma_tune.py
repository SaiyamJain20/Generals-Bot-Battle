"""CMA-ES tuning of participant PARAMS against a fixed opponent pool.

Each generation uses a fresh block of maps shared by every candidate (common
random numbers); each candidate plays `--games` games (side-swapped pairs)
against every opponent. Fitness = mean score (win 1, draw .5) plus a small
tiebreak on game length for wins. Results append to runs/<name>/log.jsonl and
the best params so far are written to runs/<name>/best.json.

    python tune/cma_tune.py --name t1 --games 16 --popsize 16 --workers 15 \
        --opps bots/opp/hunter.py bots/opp/rusher.py bots/versions/v1.py
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import cma

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import arena.run as R  # noqa: E402

# name: (lo, hi, kind)
SPACE = {
    "castle_start": (40, 400, "int"),
    "castle_every": (20, 120, "int"),
    "castle_stop": (400, 800, "int"),
    "castle_safe_dist": (2, 8, "int"),
    "castle_price_w": (0.3, 3.0, "float"),
    "castle_safety_w": (0.0, 2.0, "float"),
    "castle_gather_budget": (8, 45, "int"),
    "attack_min_army": (20, 200, "int"),
    "gather_budget": (4, 30, "int"),
    "min_stack": (3, 30, "int"),
    "launch_max": (10, 80, "int"),
    "w_launch": (1.0, 6.0, "float"),
    "w_cycle": (0.3, 3.0, "float"),
    "w_scout": (0.2, 2.5, "float"),
    "w_home_fill": (0.5, 5.0, "float"),
    "home_r": (2, 7, "int"),
    "garrison_frac_hidden": (0.0, 1.0, "float"),
    "hidden_stack_frac": (0.2, 0.9, "float"),
    "garrison_cap_frac": (0.1, 0.8, "float"),
    "track_threat_dist": (4, 20, "int"),
    "v_neutral": (0.3, 2.5, "float"),
    "v_enemy": (0.5, 5.0, "float"),
    "v_kill": (0.0, 0.3, "float"),
    "bonus_mult": (1.0, 4.0, "float"),
    "bonus_window": (0, 30, "int"),
    "small_frac": (0.0, 0.15, "float"),
    "scout_max_frac": (0.02, 0.4, "float"),
    "belief_enemy_w": (0.0, 2.0, "float"),
    "intercept_dist": (1, 8, "int"),
    "kill_margin": (0, 8, "int"),
    "open_div": (1, 3, "int"),
    "threat_vis_range": (6, 18, "int"),
    "threat_decay": (0.3, 1.5, "float"),
    "w_build": (3.0, 10.0, "float"),
    "w_garrison": (2.0, 10.0, "float"),
    "feed_min": (2, 20, "int"),
    "v_near_home": (0.0, 5.0, "float"),
    "fortress_turn": (650, 790, "int"),
    "learned_threat_w": (0.0, 1.2, "float"),
    "early_expand_until": (0, 130, "int"),
    "early_expand_bonus": (0.0, 10.0, "float"),
    "ring_w": (0.0, 1.0, "float"),
    "ring_r": (1, 3, "int"),
    "castle_front_w": (0.0, 3.0, "float"),
    "stealth_w": (0.0, 4.0, "float"),
    "lead_w": (0.0, 1.5, "float"),
    "fog_tracks": (0, 1, "int"),
}


def decode(x, base):
    p = dict(base)
    for (k, (lo, hi, kind)), v in zip(SPACE.items(), x):
        v = min(1.0, max(0.0, float(v)))
        val = lo + v * (hi - lo)
        p[k] = int(round(val)) if kind == "int" else round(float(val), 4)
    return p


def encode(p):
    out = []
    for k, (lo, hi, kind) in SPACE.items():
        v = p.get(k, (lo + hi) / 2)
        out.append(min(1.0, max(0.0, (v - lo) / (hi - lo))))
    return out


def _game(args):
    bot, opp, map_idx, swap, params = args
    r = R._task((bot, opp, map_idx, swap, {"limit_ms": None, "params": (params, None)}))
    s = r["a_score"]
    # small tiebreak: faster wins are better, slow losses better than fast losses
    t = r["turns"] / 1200.0
    bonus = 0.02 * (1 - t) if s == 1.0 else (0.02 * t if s == 0.0 else 0.0)
    return s, bonus, r["reason"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--bot", default="bots/participant.py")
    ap.add_argument("--opps", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=16, help="per opponent per candidate (even)")
    ap.add_argument("--popsize", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--gens", type=int, default=1000)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--map-base", type=int, default=1000)
    ap.add_argument("--start", default=None, help="json file with starting params")
    ap.add_argument("--self-play", action="store_true", help="add the current CMA mean as an opponent")
    ap.add_argument("--league-every", type=int, default=0, help="snapshot best into the pool every K gens")
    ap.add_argument("--league-max", type=int, default=3)
    ap.add_argument("--pfsp", type=float, default=0.0, help="weight opponents by (1-score)^p (0 = mean)")
    ap.add_argument("--mean-avg", type=int, default=8, help="also write the average of the last K CMA means")
    args = ap.parse_args()

    out = os.path.join(ROOT, "runs", args.name)
    os.makedirs(out, exist_ok=True)
    mod = R.load_bot(os.path.join(ROOT, args.bot))
    base = dict(mod.PARAMS)
    if args.start:
        base.update(json.load(open(args.start)))
    x0 = encode(base)
    es = cma.CMAEvolutionStrategy(x0, args.sigma, {"popsize": args.popsize, "bounds": [0, 1],
                                                   "seed": 1234, "verbose": -9})
    best = (-1.0, base)
    t_end = time.time() + 3600 * args.hours
    pool = Pool(args.workers, maxtasksperchild=200)
    gen = 0
    league = []
    means_hist = []
    bot_src = open(os.path.join(ROOT, args.bot)).read()

    def write_variant(path, params):
        k = bot_src.index("\nDIRS = ")
        code = bot_src[:k] + "\nPARAMS.update(" + repr(params) + ")\n" + bot_src[k:]
        with open(path, "w") as f:
            f.write(code)
        return os.path.relpath(path, ROOT)

    while gen < args.gens and time.time() < t_end:
        xs = es.ask()
        cands = [decode(x, base) for x in xs]
        opps = list(args.opps)
        if args.self_play:
            opps.append(write_variant(os.path.join(out, "self.py"), decode(es.mean, base)))
        opps += league
        # include the incumbent so drift is visible
        cands_eval = cands + [best[1]]
        map0 = args.map_base + gen * args.games
        tasks, index = [], []
        for ci, p in enumerate(cands_eval):
            for oi, opp in enumerate(opps):
                for g in range(args.games):
                    pair, swap = divmod(g, 2)
                    tasks.append((args.bot, opp, map0 + pair, bool(swap), p))
                    index.append((ci, oi))
        t0 = time.time()
        res = pool.map(_game, tasks, chunksize=4)
        per = [[0.0] * len(opps) for _ in cands_eval]
        cnt = [[0] * len(opps) for _ in cands_eval]
        for (ci, oi), (sc, bonus, _) in zip(index, res):
            per[ci][oi] += sc + bonus
            cnt[ci][oi] += 1
        per = [[per[c][o] / max(1, cnt[c][o]) for o in range(len(opps))] for c in range(len(cands_eval))]
        opp_mean = [sum(per[c][o] for c in range(len(cands))) / len(cands) for o in range(len(opps))]
        wts = [(1.0 - min(1.0, m) + 0.05) ** args.pfsp for m in opp_mean]
        fit = [sum(w * x for w, x in zip(wts, per[c])) / sum(wts) for c in range(len(cands_eval))]
        es.tell(xs, [-f for f in fit[:-1]])
        gi = max(range(len(cands)), key=lambda i: fit[i])
        inc = fit[-1]
        # promote only if it beats the incumbent on the same maps
        if fit[gi] > inc and fit[gi] > best[0] * 0.0:
            best = (fit[gi], cands[gi])
            json.dump(best[1], open(os.path.join(out, "best.json"), "w"), indent=1)
        mean_p = decode(es.mean, base)
        json.dump(mean_p, open(os.path.join(out, "mean.json"), "w"), indent=1)
        means_hist.append([float(v) for v in es.mean])
        k = min(len(means_hist), args.mean_avg)
        avg = [sum(m[j] for m in means_hist[-k:]) / k for j in range(len(means_hist[-1]))]
        json.dump(decode(avg, base), open(os.path.join(out, "mean_avg.json"), "w"), indent=1)
        if args.league_every and gen % args.league_every == args.league_every - 1:
            league.append(write_variant(os.path.join(out, f"league_g{gen}.py"), cands[gi]))
            league = league[-args.league_max:]
        rec = {"gen": gen, "best_fit": round(fit[gi], 4), "incumbent_fit": round(inc, 4),
               "mean_fit": round(sum(fit[:-1]) / len(cands), 4), "sec": round(time.time() - t0, 1),
               "sigma": round(es.sigma, 4),
               "opp_mean": {os.path.basename(o): round(m, 3) for o, m in zip(opps, opp_mean)},
               "best": cands[gi]}
        with open(os.path.join(out, "log.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k != "best"}), flush=True)
        gen += 1
    pool.close()


if __name__ == "__main__":
    main()
