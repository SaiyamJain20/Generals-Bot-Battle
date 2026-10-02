"""CPU PPO fine-tuning of the behaviour-cloned policy (learner + N rollout workers).

    PYTHONPATH=vendor/generals-bots:. OMP_NUM_THREADS=1 .venv312/bin/python rl/ppo_cpu.py \
        --init rl/students/ResBot_12x1.pt --out runs/ppo1 --workers 60 --games-per-iter 240

Rollouts use the exact stdlib simulator (sim/engine.py) and the deployment feature code
(learn/bc_features.py). Workers play complete games: the learning policy (sampled from masked
logits, temperature 1) vs an opponent from a league (latest self, frozen snapshots incl. the BC
init, heuristic/zoo bots as fresh modules per game, optional BC clones), both seats.
Learner: PPO + GAE, KL penalty to the frozen BC policy, BN kept in eval mode (running stats frozen).
Value head = win-probability logit l of BC; value used for GAE is v = 2*sigmoid(l)-1 = tanh(l/2).

Outputs in --out: log.jsonl, ckpt_<it>.pt ({"state","ch","blocks"}), last.pt, snaps/, cur.pt, train_state.pt.
"""
import argparse
import glob
import gc
import importlib.util
import json
import math
import multiprocessing as mp
import os
import random
import resource
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "learn"))

import torch  # noqa: E402
from sim import engine as E  # noqa: E402
import bc_features as F  # noqa: E402
from bc_train import Net  # noqa: E402

NACT = F.NACT
DEFAULT_BOTS = (["bots/versions/tune_base12g.py", "bots/versions/t1c.py",
                 "bots/opp/expander.py", "bots/opp/hunter.py"]
                + sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "bots/opp/zoo_*.py"))))

# --------------------------------------------------------------------------- worker side
_W = {"nets": {}, "cnt": 0}


def _worker_init():
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"


def _load_net(path):
    nets = _W["nets"]
    ent = nets.get(path)
    mt = os.stat(path).st_mtime_ns
    if ent is not None and ent[0] == mt:
        return ent[1]
    ck = torch.load(path, map_location="cpu")
    net = Net(ch=ck["ch"], blocks=ck["blocks"])
    net.load_state_dict(ck["state"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    if len(nets) > 12:
        for k in list(nets)[:6]:
            if k != path and "cur.pt" not in k:
                del nets[k]
    nets[path] = (mt, net)
    return net


def _load_bot(path):
    _W["cnt"] += 1
    spec = importlib.util.spec_from_file_location(f"bot_{os.getpid()}_{_W['cnt']}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _valid(a):
    if not isinstance(a, (list, tuple)) or len(a) != 5:
        return False
    for v in a:
        if type(v) is not int or not (-2**31 <= v < 2**31):
            return False
    return a[0] in (0, 1, 2) and 0 <= a[3] <= 3 and a[4] in (0, 1)


def _phi(scal, O, T):
    """Shaping potential in [~-0.9, 0.9] (own perspective): land ratio, army ratio, own castles."""
    nc = int(((T == 3) & (O == 1)).sum())
    return 0.5 * math.tanh(float(scal[4])) + 0.3 * math.tanh(float(scal[3])) + 0.2 * min(nc, 4) / 4.0


def play_task(task):
    """Play one game. Returns dict with per-learner-seat trajectories and timing."""
    t_start = time.perf_counter()
    rng = np.random.default_rng(task["seed"])
    m = json.loads(task["map_line"])
    s = E.from_grid(m["grid"])
    H, W = s.H, s.W
    kind = task["kind"]
    greedy = task.get("greedy", False)
    max_turns = task.get("max_turns", E.TRUNCATION)
    learner_net = _load_net(task["policy_path"])
    # who is controlled how
    ctrl = [None, None]   # "L" learner net, "N" other net, "B" bot module
    if kind == "self":
        ctrl = ["L", "L"]
        learn_seats = [0, 1]
    else:
        ls = task["seat"]
        learn_seats = [ls]
        ctrl[ls] = "L"
        ctrl[1 - ls] = "N" if kind == "snap" else "B"
    other_net = _load_net(task["opp_path"]) if kind == "snap" else None
    bot = None
    try:
        if kind in ("bot", "eval"):
            bot = _load_bot(os.path.join(ROOT, task["opp_path"]))
    except Exception:
        bot = None
        return {"task": task, "error": "import", "traj": {}, "winner": -1}
    trk = [F.Tracker(H, W), F.Tracker(H, W)]
    rec = {p: {"planes": [], "scal": [], "mask": [], "act": [], "logp": [], "val": [], "phi": []}
           for p in learn_seats} if not greedy else {}
    tm = {"nn": 0.0, "feat": 0.0, "env": 0.0, "opp": 0.0, "steps": 0}
    forfeit = -1
    pc = time.perf_counter
    while not s.done and s.time < max_turns:
        t0 = pc()
        obs = [E.observe(s, 0), E.observe(s, 1)]
        feats = {}
        for p in (0, 1):
            if ctrl[p] in ("L", "N"):
                planes, scal, (O, A, T) = trk[p].update(obs[p])
                mask = F.legal_mask(O, A, T, H, W, planes)
                feats[p] = (planes, scal, mask, _phi(scal, O, T) if p in rec else 0.0)
        t1 = pc()
        tm["feat"] += t1 - t0
        acts = [None, None]
        for tag, net in (("L", learner_net), ("N", other_net)):
            ps = [p for p in (0, 1) if ctrl[p] == tag]
            if not ps:
                continue
            tn = pc()
            with torch.inference_mode():
                pl = torch.from_numpy(np.stack([feats[p][0] for p in ps]))
                sc = torch.from_numpy(np.stack([feats[p][1] for p in ps]))
                lg, vl = net(pl, sc)
                mk = torch.from_numpy(np.stack([feats[p][2] for p in ps]))
                lg = lg.masked_fill(~mk, -1e9)
                lp = torch.log_softmax(lg, 1)
                pr = lp.exp().numpy().astype(np.float64)
                val = torch.tanh(vl / 2).numpy()
                lpn = lp.numpy()
            for j, p in enumerate(ps):
                if greedy and tag == "L":
                    idx = int(pr[j].argmax())
                else:
                    c = np.cumsum(pr[j])
                    idx = min(int(np.searchsorted(c, rng.random() * c[-1])), NACT - 1)
                    while not feats[p][2][idx]:      # numerical safety
                        idx = int(rng.choice(np.flatnonzero(feats[p][2])))
                acts[p] = F.index_action(idx)
                if p in rec:
                    r = rec[p]
                    r["planes"].append(feats[p][0])
                    r["scal"].append(feats[p][1])
                    r["mask"].append(np.packbits(feats[p][2]))
                    r["act"].append(idx)
                    r["logp"].append(float(lpn[j, idx]))
                    r["val"].append(float(val[j]))
                    r["phi"].append(feats[p][3])
            tm["nn"] += pc() - tn
        for p in (0, 1):
            if ctrl[p] == "B":
                to = pc()
                try:
                    a = bot.act(obs[p])
                    if not _valid(a):
                        raise ValueError("malformed")
                    acts[p] = [int(v) for v in a]
                except Exception:
                    forfeit = p
                tm["opp"] += pc() - to
        if forfeit >= 0:
            break
        t2 = pc()
        E.step(s, acts)
        tm["env"] += pc() - t2
        tm["steps"] += 1
    try:
        gc.unfreeze()
    except Exception:
        pass
    winner = (1 - forfeit) if forfeit >= 0 else s.winner
    out = {"task": {k: v for k, v in task.items() if k != "map_line"}, "winner": winner,
           "turns": s.time, "tm": tm, "wall": time.perf_counter() - t_start,
           "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
           "final": [s.land(0), s.land(1)], "traj": {}}
    for p, r in rec.items():
        if not r["act"]:
            continue
        out["traj"][p] = {
            "planes": np.stack(r["planes"]), "scal": np.stack(r["scal"]).astype(np.float32),
            "mask": np.stack(r["mask"]), "act": np.asarray(r["act"], np.int16),
            "logp": np.asarray(r["logp"], np.float32), "val": np.asarray(r["val"], np.float32),
            "phi": np.asarray(r["phi"], np.float32)}
    return out


# --------------------------------------------------------------------------- learner side
def save_ckpt(net, path, ch, blocks):
    tmp = path + ".tmp"
    torch.save({"state": {k: v.detach().clone() for k, v in net.state_dict().items()},
                "ch": ch, "blocks": blocks}, tmp)
    os.replace(tmp, path)


def set_eval(net):
    net.eval()   # BN running stats frozen for both rollout and update (consistent logp)
    return net


def gae(rew, val, gamma, lam):
    n = len(rew)
    adv = np.zeros(n, np.float32)
    last = 0.0
    for t in range(n - 1, -1, -1):
        nv = val[t + 1] if t + 1 < n else 0.0     # terminal after last step
        d = rew[t] + gamma * nv - val[t]
        last = d + gamma * lam * last
        adv[t] = last
    return adv, adv + val


def traj_rewards(tr, winner, seat, args, shape_coef):
    n = len(tr["act"])
    rew = np.zeros(n, np.float32)
    if shape_coef > 0:
        phi = tr["phi"]
        nxt = np.append(phi[1:], 0.0)
        rew += shape_coef * (args.gamma * nxt - phi)
    if winner < 0:
        rew[-1] += args.draw_reward
    else:
        rew[-1] += 1.0 if winner == seat else -1.0
    return rew


def sample_league(rng, stats, snaps, args):
    """Return (kind, opp_id, opp_path). stats: opp_id -> EMA score of learner."""
    cats, ws = [], []
    cats.append(("self", None)); ws.append(args.self_w)
    if snaps:
        pw = np.array([(1 - stats.get(k, 0.5)) ** args.pfsp_p + args.pfsp_eps for k, _ in snaps])
        pw /= pw.sum()
        for (k, pth), w in zip(snaps, pw):
            cats.append(("snap", (k, pth))); ws.append(args.snap_w * w)
    bots = args.bot_list
    if bots:
        pw = np.array([(1 - stats.get(b, 0.5)) ** args.pfsp_p + args.pfsp_eps for b in bots])
        pw /= pw.sum()
        for b, w in zip(bots, pw):
            cats.append(("bot", (b, b))); ws.append(args.bot_w * w)
    ws = np.array(ws) / sum(ws)
    i = rng.choice(len(cats), p=ws)
    k, v = cats[i]
    if k == "self":
        return "self", "self", None
    return k, v[0], v[1]


def update(net, bc_net, batch, args, opt, coefs):
    N = len(batch["act"])
    adv = batch["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    idx_all = np.arange(N)
    stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl_bc": 0.0, "approx_kl": 0.0, "clipfrac": 0.0, "n": 0}
    stopped = False
    t0 = time.time()
    for ep in range(args.epochs):
        np.random.shuffle(idx_all)
        for b in range(0, N, args.mb):
            ii = np.sort(idx_all[b:b + args.mb])
            planes = torch.from_numpy(batch["planes"][ii])
            scal = torch.from_numpy(batch["scal"][ii])
            mask = torch.from_numpy(np.unpackbits(batch["mask"][ii], axis=1, count=NACT).astype(bool))
            act = torch.from_numpy(batch["act"][ii].astype(np.int64))
            old = torch.from_numpy(batch["logp"][ii])
            a = torch.from_numpy(adv[ii])
            ret = torch.from_numpy(batch["ret"][ii])
            lg, vl = net(planes, scal)
            lg = lg.masked_fill(~mask, -1e9)
            lp_all = torch.log_softmax(lg, 1)
            lp = lp_all.gather(1, act[:, None])[:, 0]
            ratio = torch.exp(lp - old)
            pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a).mean()
            pr = lp_all.exp()
            zero = torch.zeros(())
            ent = -torch.where(mask, pr * lp_all, zero).sum(1).mean()
            with torch.no_grad():
                bl, _ = bc_net(planes, scal)
                bl = bl.masked_fill(~mask, -1e9)
                bc_lp = torch.log_softmax(bl, 1)
            kl = torch.where(mask, pr * (lp_all - bc_lp), zero).sum(1).mean()
            v = torch.tanh(vl / 2)
            vf = 0.5 * ((v - ret) ** 2).mean()
            loss = (coefs["pg"] * pg + args.vf_coef * vf - coefs["ent"] * ent + coefs["kl"] * kl)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad)
            opt.step()
            with torch.no_grad():
                ak = float((old - lp).mean())
                stats["pg"] += float(pg); stats["vf"] += float(vf); stats["ent"] += float(ent)
                stats["kl_bc"] += float(kl); stats["approx_kl"] += ak
                stats["clipfrac"] += float(((ratio - 1).abs() > args.clip).float().mean())
                stats["n"] += 1
            if args.target_kl > 0 and ak > 1.5 * args.target_kl:
                stopped = True
                break
        if stopped:
            break
    n = max(1, stats.pop("n"))
    out = {k: v / n for k, v in stats.items()}
    out["early_stop"] = stopped
    out["update_s"] = time.time() - t0
    return out


def run_eval(pool, args, policy_path, it, maps):
    tasks = []
    n = args.eval_games
    for g in range(n):
        tasks.append({"kind": "eval", "seat": g % 2, "map_line": maps[(args.eval_map_offset + g // 2) % len(maps)],
                      "opp_path": args.eval_bot, "policy_path": policy_path, "seed": g, "greedy": True,
                      "max_turns": args.max_turns})
    res = pool.map(play_task, tasks, chunksize=1)
    wins = sum(1 for r in res if r["winner"] == r["task"]["seat"])
    draws = sum(1 for r in res if r["winner"] < 0)
    return {"n": n, "wins": wins, "draws": draws, "score": (wins + 0.5 * draws) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="rl/students/ResBot_12x1.pt")
    ap.add_argument("--out", default="runs/ppo")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--learner-threads", type=int, default=2)
    ap.add_argument("--games-per-iter", type=int, default=16)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--time-limit-h", type=float, default=0.0, help="stop (and anneal) by wall clock; 0=off")
    ap.add_argument("--max-turns", type=int, default=1200)
    ap.add_argument("--map-offset", type=int, default=1000, help="first map index for training maps")
    ap.add_argument("--map-count", type=int, default=15000)
    ap.add_argument("--eval-map-offset", type=int, default=0, help="fixed eval maps (disjoint from training)")
    ap.add_argument("--seed", type=int, default=0)
    # league
    ap.add_argument("--bots", default="", help="comma list of bot paths (default: heuristics+zoo+expander+hunter)")
    ap.add_argument("--bc-clones", default="", help="comma list of extra torch BC clone bots (slow)")
    ap.add_argument("--self-w", type=float, default=0.2)
    ap.add_argument("--snap-w", type=float, default=0.15)
    ap.add_argument("--bot-w", type=float, default=0.65)
    ap.add_argument("--pfsp-p", type=float, default=1.5)
    ap.add_argument("--pfsp-eps", type=float, default=0.1)
    ap.add_argument("--snap-every", type=int, default=10)
    ap.add_argument("--max-snaps", type=int, default=8)
    ap.add_argument("--ema", type=float, default=0.2)
    # reward
    ap.add_argument("--draw-reward", type=float, default=-0.2)
    ap.add_argument("--shape", type=float, default=0.1, help="potential shaping coef (annealed to 0)")
    ap.add_argument("--shape-final", type=float, default=0.0)
    # PPO
    ap.add_argument("--gamma", type=float, default=0.998)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr-final", type=float, default=-1, help="linear anneal target (<0: = lr)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--mb", type=int, default=2048)
    ap.add_argument("--vf-coef", type=float, default=1.0)
    ap.add_argument("--ent", type=float, default=0.002)
    ap.add_argument("--ent-final", type=float, default=0.0005)
    ap.add_argument("--kl-coef", type=float, default=0.1, help="KL(pi||pi_BC) penalty coef")
    ap.add_argument("--kl-final", type=float, default=0.01)
    ap.add_argument("--target-kl", type=float, default=0.03, help="early stop epochs when approx kl > 1.5x (0=off)")
    ap.add_argument("--max-grad", type=float, default=0.5)
    ap.add_argument("--vf-warmup", type=int, default=3, help="iters with policy gradient off (value fit only)")
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-games", type=int, default=24)
    ap.add_argument("--eval-bot", default="bots/versions/tune_base12g.py")
    ap.add_argument("--pipeline", type=int, default=1,
                    help="1: workers generate batch i+1 (1-update-stale policy) while the learner updates on batch i")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-eval0", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.learner_threads)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "snaps"), exist_ok=True)
    args.bot_list = [b for b in args.bots.split(",") if b] if args.bots else list(DEFAULT_BOTS)
    args.bot_list += [b for b in args.bc_clones.split(",") if b]
    for b in args.bot_list:
        assert os.path.exists(os.path.join(ROOT, b)), b
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    ck = torch.load(args.init, map_location="cpu")
    ch, blocks = ck["ch"], ck["blocks"]
    net = Net(ch=ch, blocks=blocks); net.load_state_dict(ck["state"]); set_eval(net)
    bc_net = Net(ch=ch, blocks=blocks); bc_net.load_state_dict(ck["state"]); set_eval(bc_net)
    for p in bc_net.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-6)
    cur_path = os.path.abspath(os.path.join(args.out, "cur.pt"))
    bc_path = os.path.abspath(os.path.join(args.out, "snaps", "bc.pt"))
    save_ckpt(bc_net, bc_path, ch, blocks)
    snaps = [("snap:bc", bc_path)]
    start_it = 0
    t_prev_elapsed = 0.0
    ts_path = os.path.join(args.out, "train_state.pt")
    ema = {}
    if args.resume and os.path.exists(ts_path):
        ts = torch.load(ts_path, map_location="cpu")
        net.load_state_dict(ts["net"]); opt.load_state_dict(ts["opt"])
        start_it = ts["it"]; ema = ts["ema"]; t_prev_elapsed = ts.get("elapsed", 0.0)
        snaps = [(k, p) for k, p in ts["snaps"] if os.path.exists(p)]
        print(f"resumed at iter {start_it}", flush=True)

    with open(os.path.join(ROOT, "data", "maps.jsonl")) as f:
        maps = f.readlines()
    nmaps = len(maps)
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.workers, initializer=_worker_init)
    logf = open(os.path.join(args.out, "log.jsonl"), "a")
    t_begin = time.time() - t_prev_elapsed
    total_steps = 0

    def progress(it):
        p = (it - start_it) / max(1, args.iters - start_it) if args.iters else 0.0
        if args.time_limit_h > 0:
            p = max(p, (time.time() - t_begin) / (args.time_limit_h * 3600))
        return min(1.0, p)

    def lerp(a, b, p):
        return a + (b - a) * p

    pending = None

    def launch():
        save_ckpt(net, cur_path, ch, blocks)
        tasks = []
        for g in range(args.games_per_iter):
            kind, oid, opath = sample_league(rng, ema, snaps, args)
            mi = args.map_offset + int(rng.integers(0, args.map_count))
            tasks.append({"kind": kind, "opp_id": oid, "opp_path": opath, "seat": g % 2,
                          "map_line": maps[mi % nmaps], "map_idx": mi % nmaps,
                          "policy_path": cur_path, "seed": int(rng.integers(1 << 30)),
                          "max_turns": args.max_turns})
        return tasks, pool.map_async(play_task, tasks, chunksize=1)

    for it in range(start_it, args.iters):
        prog = progress(it)
        if prog >= 1.0 and it > start_it:
            break
        t_it = time.time()
        if pending is None:                      # first iteration (or sync mode): launch with current weights
            pending = launch()
        t_r = time.time()
        tasks, results = pending[0], pending[1].get()
        rollout_wait_s = time.time() - t_r
        pending = None
        if args.eval_every and it > start_it and it % args.eval_every == 0 or (it == start_it and not args.no_eval0 and args.eval_every):
            save_ckpt(net, cur_path, ch, blocks)
            eval_rec = run_eval(pool, args, cur_path, it, maps)
        else:
            eval_rec = None
        if args.pipeline:                        # generate the next batch (weights before this update) during the update
            pending = launch()
        rollout_s = max(rollout_wait_s, 1e-3)
        shape = lerp(args.shape, args.shape_final, prog)
        cols = {k: [] for k in ("planes", "scal", "mask", "act", "logp", "adv", "ret")}
        per_opp = {}
        tm_tot = {"nn": 0.0, "feat": 0.0, "env": 0.0, "opp": 0.0, "steps": 0}
        tm_by = {}
        rss = 0.0
        nsteps = 0
        rew_sum = 0.0
        ntraj = 0
        for r in results:
            if r.get("error"):
                continue
            t = r["task"]
            rss = max(rss, r["rss_mb"])
            for k in tm_tot:
                tm_tot[k] += r["tm"][k]
            d = tm_by.setdefault(t["kind"], {"nn": 0.0, "feat": 0.0, "env": 0.0, "opp": 0.0, "steps": 0, "games": 0})
            for k in ("nn", "feat", "env", "opp", "steps"):
                d[k] += r["tm"][k]
            d["games"] += 1
            for seat, tr in r["traj"].items():
                rew = traj_rewards(tr, r["winner"], seat, args, shape)
                adv, ret = gae(rew, tr["val"], args.gamma, args.lam)
                for k, v in (("planes", tr["planes"]), ("scal", tr["scal"]), ("mask", tr["mask"]),
                             ("act", tr["act"]), ("logp", tr["logp"]), ("adv", adv), ("ret", ret)):
                    cols[k].append(v)
                nsteps += len(adv)
                rew_sum += float(rew.sum()); ntraj += 1
            if t["kind"] != "self":
                sc = 0.5 if r["winner"] < 0 else (1.0 if r["winner"] == t["seat"] else 0.0)
                po = per_opp.setdefault(t["opp_id"], [])
                po.append(sc)
        for k, v in per_opp.items():
            m_ = float(np.mean(v))
            ema[k] = m_ if k not in ema else (1 - args.ema) * ema[k] + args.ema * m_
        if not cols["act"]:
            print("no trajectories", flush=True)
            continue
        batch = {k: np.concatenate(v) for k, v in cols.items()}
        batch["ret"] = np.clip(batch["ret"], -2, 2)
        total_steps += nsteps
        # ---- update
        coefs = {"pg": 0.0 if it - start_it < args.vf_warmup else 1.0,
                 "ent": lerp(args.ent, args.ent_final, prog),
                 "kl": lerp(args.kl_coef, args.kl_final, prog)}
        lr = lerp(args.lr, args.lr if args.lr_final < 0 else args.lr_final, prog)
        for g_ in opt.param_groups:
            g_["lr"] = lr
        # drift probe: logits before update on a fixed subsample
        nprobe = min(2048, len(batch["act"]))
        pi_ = np.sort(np.random.choice(len(batch["act"]), nprobe, replace=False))
        pr_planes = torch.from_numpy(batch["planes"][pi_]); pr_scal = torch.from_numpy(batch["scal"][pi_])
        pr_mask = torch.from_numpy(np.unpackbits(batch["mask"][pi_], axis=1, count=NACT).astype(bool))

        def probe():
            with torch.no_grad():
                l_, _ = net(pr_planes, pr_scal)
                return torch.log_softmax(l_.masked_fill(~pr_mask, -1e9), 1)

        lp_before = probe()
        up = update(net, bc_net, batch, args, opt, coefs)
        lp_after = probe()
        with torch.no_grad():
            zero = torch.zeros(())
            kl_drift = float(torch.where(pr_mask, lp_after.exp() * (lp_after - lp_before), zero).sum(1).mean())
            agree = float((lp_after.argmax(1) == lp_before.argmax(1)).float().mean())
            lp_bc = torch.log_softmax(bc_net(pr_planes, pr_scal)[0].masked_fill(~pr_mask, -1e9), 1)
            kl_bc = float(torch.where(pr_mask, lp_after.exp() * (lp_after - lp_bc), zero).sum(1).mean())
            agree_bc = float((lp_after.argmax(1) == lp_bc.argmax(1)).float().mean())
        # ---- league bookkeeping
        if (it + 1) % args.snap_every == 0:
            sp = os.path.abspath(os.path.join(args.out, "snaps", f"snap_{it + 1}.pt"))
            save_ckpt(net, sp, ch, blocks)
            snaps.append((f"snap:{it + 1}", sp))
            while len(snaps) > args.max_snaps:
                old = snaps.pop(1)     # keep bc (index 0)
                try:
                    os.remove(old[1])
                except OSError:
                    pass
        if (it + 1) % args.ckpt_every == 0:
            save_ckpt(net, os.path.join(args.out, f"ckpt_{it + 1}.pt"), ch, blocks)
        save_ckpt(net, os.path.join(args.out, "last.pt"), ch, blocks)
        rec = {"it": it + 1, "t": round(time.time() - t_begin, 1), "games": len(tasks), "steps": nsteps,
               "rollout_wait_s": round(rollout_wait_s, 1), "worker_steps_per_s": round(sum(r["tm"]["steps"] for r in results if not r.get("error")) / max(1e-9, sum(r["wall"] for r in results if not r.get("error"))), 1),
               "winrate": {k: round(float(np.mean(v)), 3) for k, v in per_opp.items()},
               "n_opp": {k: len(v) for k, v in per_opp.items()},
               "mean_traj_return": round(rew_sum / max(1, ntraj), 4),
               "pg": round(up["pg"], 5), "vf": round(up["vf"], 4), "ent": round(up["ent"], 3),
               "kl_bc": round(kl_bc, 5), "agree_bc": round(agree_bc, 4),
               "kl_drift": round(kl_drift, 6), "agree_prev": round(agree, 4),
               "approx_kl": round(up["approx_kl"], 5), "clipfrac": round(up["clipfrac"], 3),
               "early_stop": up["early_stop"], "update_s": round(up["update_s"], 1),
               "coefs": {k: round(v, 5) for k, v in coefs.items()}, "lr": lr, "shape": round(shape, 4),
               "rss_worker_mb_max": round(rss, 0),
               "tm_us_per_step": {k: round(v / max(1, tm_tot["steps"]) * 1e6, 0) for k, v in tm_tot.items() if k != "steps"},
               "tm_by_kind": {k: {"games": d["games"], "steps": d["steps"],
                                  **{c: round(d[c] / max(1, d["steps"]) * 1e6, 0) for c in ("nn", "feat", "env", "opp")}}
                              for k, d in tm_by.items()},
               "iter_s": round(time.time() - t_it, 1)}
        if eval_rec is not None:
            rec["eval"] = eval_rec
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        print(json.dumps(rec), flush=True)
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "it": it + 1, "ema": ema,
                    "snaps": snaps, "elapsed": time.time() - t_begin}, ts_path + ".tmp")
        os.replace(ts_path + ".tmp", ts_path)
        gc.collect()
    if pending is not None:
        pending[1].wait()
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
