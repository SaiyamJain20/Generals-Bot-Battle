"""Phase-in-50-tick-cycle profile of enemy captures / neutral captures / gather moves (T50-300)."""
import json, collections, os
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BOTS = ['ResBot', 'nanomena', 'Kubic', 'bca', 'Chig']
recs = collections.defaultdict(list)
for line in open(os.path.join(ROOT, 'data/castle_study_recs2.jsonl')):
    R = json.loads(line); recs[R['name']].append(R)
for lo, hi in ((50, 150), (150, 200)):
    print(f'== T{lo}-{hi}: per game per 50-cycle, by phase t%50 in 5-tick bins: enemy caps | neutral caps | own-dst moves amt>=2 (gather) | mean amt of own-dst moves')
    for b in BOTS:
        G = [R for R in recs[b] if R['T'] > hi + 4]
        E = np.zeros(10); N = np.zeros(10); Gm = np.zeros(10); Ga = [[] for _ in range(10)]
        for R in G:
            for t, amt, isg, dg, typ, cap, sa, outw in R['mv']:
                if not (lo <= t < hi) or amt < 0: continue
                k = (t % 50) // 5
                if cap and typ == 'enemy': E[k] += 1
                if cap and typ == 'neutral': N[k] += 1
                if typ == 'own' and amt >= 2: Gm[k] += 1; Ga[k].append(amt)
        nc = len(G) * (hi - lo) / 50
        print(f'{b:9s} E ' + ' '.join(f'{x / nc:4.1f}' for x in E) + ' | N ' + ' '.join(f'{x / nc:4.1f}' for x in N) + ' | G ' + ' '.join(f'{x / nc:3.1f}' for x in Gm))
