"""Replay one game and dump the last N turns before our general dies."""
import json
import sys
sys.path.insert(0, ".")
from sim import engine as E
from tools.trace import load


def run(a, b, map_idx, swap, last=12):
    m = [json.loads(l) for _, l in zip(range(map_idx + 1), open("data/maps.jsonl"))][map_idx]
    s = E.from_grid(m["grid"])
    paths = (b, a) if swap else (a, b)
    bots = [load(paths[0], "P0"), load(paths[1], "P1")]
    me = 1 if swap else 0
    hist = []
    while not s.done:
        acts = [bots[p].act(E.observe(s, p)) for p in (0, 1)]
        bot = bots[me]._BOT
        g = s.gpos[me]
        en = sorted(((s.army[i], i, bot.dist_g[i]) for i in range(len(s.army)) if s.owner[i] == 1 - me), reverse=True)[:2]
        hist.append((s.time, s.army[g], bot.need_g, getattr(bot, "threat_eta", None), en, acts[me],
                     None if not bot.cyc else (bot.cyc["mode"], bot.cyc["purpose"]), [tr[1:] for tr in bot.tracks],
                     s.total_army(me), s.total_army(1 - me)))
        E.step(s, acts)
    print("winner", s.winner, "me", me, "turn", s.time)
    for h in hist[-last:]:
        print(h)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1", int(sys.argv[5]) if len(sys.argv) > 5 else 12)
