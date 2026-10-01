"""Train the spawn prior: conditional-logit-like logistic regression over candidates.

Prints held-out top-1/top-3 accuracy and mean rank vs uniform, and the weights
to paste into PRIOR_W / PRIOR_B in bots/participant.py.
"""
import collections
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "data/prior.jsonl")]
rows = [r for r in rows if r["y"] >= 0]
groups = collections.defaultdict(list)
for r in rows:
    groups[r["g"]].append(r)
keys = sorted(groups)
rng = np.random.default_rng(0)
rng.shuffle(keys)
cut = int(0.8 * len(keys))
train_k, test_k = keys[:cut], keys[cut:]


def mat(ks):
    X, y, g = [], [], []
    for k in ks:
        for r in groups[k]:
            X.append(r["f"])
            y.append(r["y"])
            g.append(k)
    return np.array(X), np.array(y), g


Xtr, ytr, _ = mat(train_k)
Xte, yte, gte = mat(test_k)
clf = LogisticRegression(C=1.0, max_iter=2000)
clf.fit(Xtr, ytr)
logit = Xte @ clf.coef_[0] + clf.intercept_[0]
by = collections.defaultdict(list)
for s, y, g in zip(logit, yte, gte):
    by[g].append((s, y))
top1 = top3 = 0
ranks, uni = [], []
for g, lst in by.items():
    order = sorted(lst, key=lambda x: -x[0])
    r = [i for i, (_, y) in enumerate(order) if y == 1][0] + 1
    ranks.append(r)
    uni.append((len(lst) + 1) / 2)
    top1 += r == 1
    top3 += r <= 3
n = len(by)
print(f"groups {n} top1 {top1/n:.3f} top3 {top3/n:.3f} mean rank {np.mean(ranks):.2f} uniform mean rank {np.mean(uni):.2f}")
print("PRIOR_W =", [round(float(w), 4) for w in clf.coef_[0]])
print("PRIOR_B =", round(float(clf.intercept_[0]), 4))
