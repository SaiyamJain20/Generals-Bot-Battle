"""Catastrophe gate for a submission candidate.

Plays the candidate against every simple / zoo / external bot and fails if
any score is below the threshold or if the candidate ever forfeits
(exception, malformed return, timeout over --limit-ms).

    python tools/gate.py bots/versions/cand.py --games 20 --workers 4 [--threshold 0.85] [--limit-ms 150]
"""
import argparse
import os
import sys

sys.path.insert(0, ".")
import arena.run as R  # noqa: E402

SIMPLE = ["bots/opp/pass_bot.py", "bots/opp/random_bot.py", "bots/opp/expander.py", "bots/opp/hunter.py",
          "bots/opp/hunter_castles.py"]
ZOO = ["bots/opp/zoo_flash.py", "bots/opp/zoo_castler.py", "bots/opp/zoo_gatherer.py", "bots/opp/zoo_sniper.py",
       "bots/opp/zoo_turtle_dt.py", "bots/opp/zoo_expander_plus.py", "bots/opp/zoo_mixed.py"]
EXT = ["bots/opp/ext_superbot.py", "bots/opp/ext_juraj34.py", "bots/opp/ext_hvn.py", "bots/opp/ext_mybot9.py",
       "bots/opp/ext_doomstack.py", "bots/opp/ext_amin.py", "bots/opp/ext_boss.py", "bots/opp/ext_sentinel.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bot")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--limit-ms", type=float, default=None)
    ap.add_argument("--no-ext", action="store_true")
    args = ap.parse_args()
    pool = SIMPLE + ZOO + ([] if args.no_ext else [p for p in EXT if os.path.exists(p)])
    failed = []
    worst_ms = 0.0
    for opp in pool:
        res = R.run_match(args.bot, opp, args.games, workers=args.workers, limit_ms=args.limit_ms,
                          quiet=True, map_offset=4000)
        sm = R.summarize(res)
        ours = [r for r in res if r["reason"].startswith("forfeit") and r["a_score"] == 0.0]
        worst_ms = max(worst_ms, sm.get("A_max_ms", 0))
        flag = ""
        if sm["score"] < args.threshold:
            flag += " LOW"
        if ours:
            flag += f" OUR_FORFEITS={len(ours)} ({ours[0]['reason']}: {(ours[0]['error'] or '')[-200:]})"
        if flag:
            failed.append(opp)
        print(f"{opp:40s} score {sm['score']:.3f}  W{sm['W']} D{sm['D']} L{sm['L']}  max {sm['A_max_ms']:.0f} ms{flag}",
              flush=True)
    print(f"worst move time {worst_ms:.0f} ms")
    print("GATE PASSED" if not failed else f"GATE FAILED: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
