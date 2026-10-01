"""For lost games: when did the killing stack first become visible, and how big was our general?"""
import json
import sys
from multiprocessing import Pool
sys.path.insert(0, ".")
from sim import engine as E
from tools.trace import load

MAPS = None


def one(args):
    a, b, map_idx, swap = args
    global MAPS
    if MAPS is None:
        MAPS = [json.loads(l) for l in open("data/maps.jsonl")][:400]
    s = E.from_grid(MAPS[map_idx]["grid"])
    paths = (b, a) if swap else (a, b)
    bots = [load(paths[0], "P0%d" % map_idx), load(paths[1], "P1%d" % map_idx)]
    me = 1 if swap else 0
    g = s.gpos[me]
    hist = []
    while not s.done:
        vis = E.visibility(s, me)
        en = [(s.army[i], i) for i in range(len(s.army)) if s.owner[i] == 1 - me]
        ea, ei = max(en) if en else (0, -1)
        acts = [bots[p].act(E.observe(s, p)) for p in (0, 1)]
        bot = bots[me]._BOT
        hist.append((s.time, s.army[g], ea, bot.dist_g[ei] if ei >= 0 else -1, vis[ei] if ei >= 0 else False,
                     bot.need_g, s.total_army(me), s.total_army(1 - me), s.land(me), acts[me][0]))
        E.step(s, acts)
    if s.winner == me or s.winner < 0:
        return None
    # find first turn the killer stack was visible continuously before death
    last = hist[-1]
    k = len(hist) - 1
    while k > 0 and hist[k - 1][4]:
        k -= 1
    return {"map": map_idx, "swap": swap, "death": s.time, "first_vis": hist[k][0], "vis_dist": hist[k][3],
            "stack_at_vis": hist[k][2], "gen_at_vis": hist[k][1], "need_at_vis": hist[k][5],
            "my_army": last[6], "opp_army": last[7], "my_land": last[8],
            "gen_hist": [h[1] for h in hist[-12:]], "stack_hist": [h[2] for h in hist[-12:]]}


if __name__ == "__main__":
    a, b, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    tasks = [(a, b, i // 2, i % 2 == 1) for i in range(n)]
    with Pool(int(sys.argv[4]) if len(sys.argv) > 4 else 14) as pool:
        for r in pool.imap_unordered(one, tasks):
            if r:
                print(r)
