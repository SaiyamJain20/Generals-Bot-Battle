"""Download public generals.bot tournament replays (full-information, gzipped JSON).

Index files list every match with a `replay_gz` URL. We save the raw .json.gz
untouched plus one metadata line per match, finals first.

    python learn/fetch_replays.py [--max N] [--workers 16]
"""
import argparse
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

INDEXES = {
    "marathon": "https://www.generals.bot/assets/marathon-2026-09-01.json",
    "sprint": "https://www.generals.bot/assets/sprint-2026-08-08.json",
}
STAGE_PRIORITY = {"final": 0, "deep": 1, "qualifier": 2}
ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "replays")


def fetch(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def matches_of(index):
    for v in index.values():
        if isinstance(v, list) and v and isinstance(v[0], dict) and "replay_gz" in v[0]:
            return v
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--players", nargs="*", default=None, help="only matches involving these bots")
    ap.add_argument("--events", nargs="*", default=None, help="subset of: marathon sprint")
    args = ap.parse_args()

    os.makedirs(ROOT, exist_ok=True)
    jobs = []
    for event, url in INDEXES.items():
        if args.events and event not in args.events:
            continue
        try:
            index = json.loads(fetch(url))
        except Exception as e:  # keep going with the other index
            print(f"[fetch] index {event} failed: {e}", file=sys.stderr)
            continue
        with open(os.path.join(ROOT, f"{event}_index.json"), "w") as f:
            json.dump(index, f)
        for m in matches_of(index):
            if not m.get("replay_gz"):
                continue
            if args.players and m.get("p0_name") not in args.players and m.get("p1_name") not in args.players:
                continue
            m = dict(m, event=event)
            base = m["replay_gz"].rsplit("/", 1)[-1]  # "<seed>-<a_side>.json.gz"
            m["file"] = f"{event}_{m['pair'].replace('|', '_')}_{base}"
            jobs.append(m)
    jobs.sort(key=lambda m: (STAGE_PRIORITY.get(m.get("stage"), 9), m["event"] != "marathon"))
    if args.max:
        jobs = jobs[: args.max]

    meta_path = os.path.join(ROOT, "meta.jsonl")
    done = set()
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            done = {json.loads(line)["file"] for line in f if line.strip()}
    todo = [m for m in jobs if m["file"] not in done]
    print(f"[fetch] {len(jobs)} matches, {len(todo)} to download", file=sys.stderr)

    def one(m):
        path = os.path.join(ROOT, m["file"])
        if not os.path.exists(path):
            data = fetch(m["replay_gz"])
            with open(path + ".part", "wb") as f:
                f.write(data)
            os.replace(path + ".part", path)
        return m

    n_ok = n_err = 0
    with open(meta_path, "a") as meta, ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(one, m) for m in todo]
        for fut in as_completed(futs):
            try:
                m = fut.result()
                meta.write(json.dumps({k: m.get(k) for k in (
                    "file", "event", "stage", "p0_name", "p1_name", "winner", "turns",
                    "seed", "a_side", "faults", "forfeit", "suspect")}) + "\n")
                n_ok += 1
            except Exception as e:
                n_err += 1
                print(f"[fetch] error: {e}", file=sys.stderr)
            if (n_ok + n_err) % 500 == 0:
                meta.flush()
                print(f"[fetch] {n_ok} ok, {n_err} errors", file=sys.stderr)
    print(f"[fetch] done: {n_ok} ok, {n_err} errors", file=sys.stderr)


if __name__ == "__main__":
    main()
