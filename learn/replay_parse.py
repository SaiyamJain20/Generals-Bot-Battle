"""Infer both players' actions from full-information replays using sim/engine.py.

For every tick t we search action pairs (a0, a1) such that engine.step from
tick t reproduces tick t+1 exactly (armies + owners). Candidates come from the
cells that differ from a "both pass" baseline. Castle builds are inferred the
same way (the replay does not record them). On a failed tick we resync from
the replay and count a failure.

    python learn/replay_parse.py --players ResBot nanomena --max 3000 --out data/acts.jsonl
"""
import argparse
import gzip
import json
import os
import sys
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from sim import engine as E  # noqa: E402

PASS = (1, 0, 0, 0, 0)
RDIR = os.path.join(ROOT, "data", "replays")


def load_replay(path):
    d = json.loads(gzip.open(path).read())
    H, W = d["dims"]["rows"], d["dims"]["cols"]
    mountains = [r * W + c for r, c in d["mountains"]]
    gpos = [r * W + c for r, c in d["generals"]]
    return d, H, W, mountains, gpos


def flat_tick(tk):
    army = [v for row in tk["armies"] for v in row]
    owner = [v for row in tk["owners"] for v in row]
    return army, owner


def matches(s, army, owner):
    return s.army == army and s.owner == owner


def candidates(s, p, changed):
    W = s.W
    out = []
    srcs = [i for i in changed if s.owner[i] == p and s.army[i] >= 2]
    srcs.sort(key=lambda i: -s.army[i])
    for i in srcs:
        r, c = divmod(i, W)
        for d, (dr, dc) in enumerate(E.DIRS):
            rr, cc = r + dr, c + dc
            if 0 <= rr < s.H and 0 <= cc < W and not s.mountain[rr * W + cc]:
                out.append((0, r, c, d, 0))
                if s.army[i] >= 3:
                    out.append((0, r, c, d, 1))
    for i in changed:
        if s.owner[i] == p and not s.castle[i] and not s.general[i] and s.army[i] >= 35:
            out.append((2, i // W, i % W, 0, 0))
    out.append(PASS)
    return out


def parse_game(path):
    d, H, W, mountains, gpos = load_replay(path)
    s = E.State(H, W, mountains, gpos)
    ticks = d["ticks"]
    acts, fails = [], 0
    for t in range(len(ticks) - 1):
        army1, owner1 = flat_tick(ticks[t + 1])
        base = s.copy()
        E.step(base, [PASS, PASS])
        changed = [i for i in range(H * W) if base.army[i] != army1[i] or base.owner[i] != owner1[i]]
        found = None
        if not changed and not base.done:
            found = (PASS, PASS)
        else:
            c0 = candidates(s, 0, changed)
            c1 = candidates(s, 1, changed)
            last = t == len(ticks) - 2
            for a0 in c0:
                for a1 in c1:
                    x = s.copy()
                    E.step(x, [a0, a1])
                    if matches(x, army1, owner1) or (last and x.done and x.winner == d.get("winner", -2)):
                        found = (a0, a1)
                        break
                if found:
                    break
        if found is None:
            fails += 1
            acts.append(None)
            # resync from the replay; castles that grew like structures stay unknown
            s.army, s.owner = army1[:], owner1[:]
            s.time += 1
            continue
        acts.append([list(found[0]), list(found[1])])
        E.step(s, [found[0], found[1]])
    return {"file": os.path.basename(path), "players": d["players"], "winner": d.get("winner"),
            "H": H, "W": W, "ticks": len(ticks), "fails": fails, "acts": acts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", nargs="*", default=["ResBot"])
    ap.add_argument("--max", type=int, default=100)
    ap.add_argument("--stages", nargs="*", default=["final", "deep", "qualifier"])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default="data/acts.jsonl")
    args = ap.parse_args()
    meta = [json.loads(l) for l in open(os.path.join(RDIR, "meta.jsonl"))]
    sel = [m for m in meta if m["event"] == "marathon" and m["stage"] in args.stages
           and (m["p0_name"] in args.players or m["p1_name"] in args.players)
           and not m.get("forfeit") and not m.get("suspect")]
    sel.sort(key=lambda m: ({"final": 0, "deep": 1, "qualifier": 2}[m["stage"]], m["file"]))
    sel = sel[: args.max]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["file"] for l in open(args.out)}
    paths = [os.path.join(RDIR, m["file"]) for m in sel if m["file"] not in done]
    print(f"{len(paths)} games to parse", file=sys.stderr)
    tot_fail = tot_ticks = 0
    with Pool(args.workers) as pool, open(args.out, "a") as f:
        for k, r in enumerate(pool.imap_unordered(parse_game, paths, chunksize=2)):
            f.write(json.dumps(r) + "\n")
            tot_fail += r["fails"]
            tot_ticks += r["ticks"]
            if (k + 1) % 20 == 0:
                f.flush()
                print(f"[parse] {k + 1}/{len(paths)} fail rate {tot_fail / max(1, tot_ticks):.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
