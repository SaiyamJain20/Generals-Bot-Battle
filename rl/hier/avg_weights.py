"""Average RO policy weights over a window of snapshots (Polyak/EMA-style evaluation weights).

    python rl/hier/avg_weights.py OUT.json W1.json W2.json ...

All snapshots come from one run (same init, same shapes), so a plain element-wise mean of the
parameters is meaningful. alpha0 (the reference temperature) is kept from the first file.
"""
import json
import sys


def mean_tree(xs):
    if isinstance(xs[0], list):
        return [mean_tree([x[i] for x in xs]) for i in range(len(xs[0]))]
    return sum(float(x) for x in xs) / len(xs)


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    ws = [json.load(open(p)) for p in paths]
    keys = ws[0].keys()
    avg = {}
    for k in keys:
        if k == "alpha0":
            avg[k] = ws[0][k]
        else:
            avg[k] = mean_tree([w[k] for w in ws])
    json.dump(avg, open(out, "w"))
    print(f"averaged {len(ws)} snapshots -> {out}; alpha {avg['alpha']:.3f}")


if __name__ == "__main__":
    main()
