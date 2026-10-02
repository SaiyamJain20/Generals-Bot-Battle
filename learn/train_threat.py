"""Train the learned threat model (quantile GBM) and export it as pure-Python trees.

Target: log1p(max enemy army within BFS 6 of our general over the next 12 turns),
predicted at the q-quantile so the garrison covers most futures.

    python learn/train_threat.py --q 0.8 --trees 80 --depth 5 --out learn/threat_model.py
"""
import argparse
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def export(model):
    trees = []
    for preds in model._predictors:
        nodes = preds[0].nodes
        flat = []
        for nd in nodes:
            flat.append((int(nd["feature_idx"]), float(nd["num_threshold"]),
                         int(nd["left"]), int(nd["right"]), round(float(nd["value"]), 5), int(nd["is_leaf"])))
        trees.append(flat)
    return float(model._baseline_prediction.ravel()[0]), trees


def predict(base, trees, x):
    s = base
    for t in trees:
        i = 0
        while not t[i][5]:
            f, thr, l, r, _, _ = t[i]
            i = l if x[f] <= thr else r
        s += t[i][4]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/threat.jsonl")
    ap.add_argument("--q", type=float, default=0.8)
    ap.add_argument("--trees", type=int, default=80)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--out", default="learn/threat_model.py")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.data)]
    X = np.array([r["f"] for r in rows])
    y = np.array([r["y"] for r in rows])
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    tr, te = idx[: int(0.8 * len(y))], idx[int(0.8 * len(y)):]
    m = HistGradientBoostingRegressor(loss="quantile", quantile=args.q, max_iter=args.trees,
                                      max_depth=args.depth, learning_rate=0.1, min_samples_leaf=50)
    m.fit(X[tr], y[tr])
    p = m.predict(X[te])
    cover = float((y[te] <= p + 1e-9).mean())
    mae_army = float(np.mean(np.abs(np.expm1(p) - np.expm1(y[te]))))
    base, trees = export(m)
    p2 = np.array([predict(base, trees, x) for x in X[te][:500]])
    err = float(np.max(np.abs(p2 - p[:500])))
    print(f"rows {len(y)} q {args.q} coverage {cover:.3f} MAE(army) {mae_army:.1f} export max err {err:.2e}")
    code = "THREAT_BASE = %r\nTHREAT_TREES = %r\n" % (round(base, 5), trees)
    open(args.out, "w").write(code)
    print(f"wrote {args.out} ({len(code)} bytes, {sum(len(t) for t in trees)} nodes)")


if __name__ == "__main__":
    main()
