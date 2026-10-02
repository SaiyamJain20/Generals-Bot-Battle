"""castle-study part 2: opening move sources, army distribution snapshots, drains under threat.

    .venv312/bin/python agents_shared/castle_study/study2.py [workers=4]
"""
import gzip, json, sys, os, collections
import numpy as np
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CAPS = {'ResBot': 300, 'nanomena': 300, 'Kubic': 300, 'bca': 300, 'Chig': 300}
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
SNAP = (75, 99, 125, 149, 175, 199, 249, 299)


def nb4(m):
    x = np.zeros_like(m)
    x[1:, :] |= m[:-1, :]; x[:-1, :] |= m[1:, :]; x[:, 1:] |= m[:, :-1]; x[:, :-1] |= m[:, 1:]
    return x


def proc(arg):
    line, players = arg
    g = json.loads(line)
    d = json.load(gzip.open(os.path.join(ROOT, 'data/replays', g['file'])))
    A = np.array([t['armies'] for t in d['ticks']], dtype=np.int32)
    O = np.array([t['owners'] for t in d['ticks']], dtype=np.int8)
    T, H, W = A.shape
    acts = g['acts']; names = g['players']
    gens = [tuple(x) for x in d['generals']]
    rr, cc = np.mgrid[0:H, 0:W]
    out = []
    for p in players:
        q = 1 - p
        gp = gens[p]
        DG = np.abs(rr - gp[0]) + np.abs(cc - gp[1])
        R = {'name': names[p], 'opp': names[q], 'T': T}
        # per-move log T40-200: (t, amt, src_is_gen, src_dg, dst_type, captured, src_army, src_adj_enemy)
        mv = []
        for t in range(40, min(200, T - 1)):
            a = acts[t][p]
            if a[0] != 0:
                mv.append((t, -1 if a[0] == 1 else -2, 0, 0, 'x', 0, 0, 0))
                continue
            r, c = a[1], a[2]; dr, dc = DIRS[a[3]]; r2, c2 = r + dr, c + dc
            if not (0 <= r2 < H and 0 <= c2 < W) or O[t, r, c] != p:
                continue
            sa = int(A[t, r, c]); amt = sa // 2 if a[4] else sa - 1
            o2 = int(O[t, r2, c2])
            typ = 'own' if o2 == p else ('enemy' if o2 == q else 'neutral')
            cap = int(o2 != p and O[t + 1, r2, c2] == p)
            mv.append((t, amt, int((r, c) == gp), int(DG[r, c]), typ, cap, sa, int(DG[r2, c2] > DG[r, c])))
        R['mv'] = mv
        # army distribution snapshots
        sn = {}
        for t in SNAP:
            if t >= T - 4:
                continue
            own = O[t] == p
            en = O[t] == q
            a = A[t] * own
            tot = int(a.sum())
            front = own & nb4(en)                  # own cells touching enemy land
            near = own & (nb4(nb4(nb4(en))) | en)  # within 3 steps of enemy land
            cells = np.sort(a[own])[::-1]
            sn[t] = {'tot': tot, 'land': int(own.sum()), 'max': int(cells[0]) if len(cells) else 0,
                     'top3': int(cells[:3].sum()), 'n6': int((a >= 6).sum()), 'n16': int((a >= 16).sum()), 'n41': int((a >= 41).sum()),
                     'ones': int(((a == 1) & own).sum()), 'front': int(front.sum()), 'front_army': int(a[front].sum()),
                     'near_army': int(a[near].sum()), 'gen': int(A[t][gp]), 'frontier_len': int((en & nb4(own)).sum()),
                     'h': [int(((a == 1) & own).sum()), int((a == 2).sum()), int(((a >= 3) & (a <= 5)).sum()), int(((a >= 6) & (a <= 15)).sum()), int((a >= 16).sum())],
                     'ha': [int(a[(a >= 3) & (a <= 5)].sum()), int(a[(a >= 6) & (a <= 15)].sum()), int(a[a >= 16].sum())],
                     'fh': [int(((a == 1) & front).sum()), int(((a == 2) & front).sum()), int(((a >= 3) & front).sum())],
                     'efh': [int(((A[t] == 1) & en & nb4(own)).sum()), int(((A[t] >= 2) & en & nb4(own)).sum())]}
        R['snap'] = sn
        out.append(R)
    return out


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    outp = os.path.join(ROOT, 'data/castle_study_recs2.jsonl')
    cnt = collections.Counter(); jobs = []
    for line in open(os.path.join(ROOT, 'data/acts.jsonl')):
        head = line[:400]
        i = head.find('"players"'); j = head.find(']', i)
        pl = json.loads(head[i + 11:j + 1])
        if '"fails": 0,' not in head:
            continue
        sel = [p for p in (0, 1) if pl[p] in CAPS and cnt[pl[p]] < CAPS[pl[p]]]
        if not sel:
            if all(cnt[k] >= v for k, v in CAPS.items()):
                break
            continue
        for p in sel:
            cnt[pl[p]] += 1
        jobs.append((line, sel))
    print(len(jobs), dict(cnt), file=sys.stderr, flush=True)
    with Pool(nw) as pool, open(outp, 'w') as f:
        for recs in pool.imap_unordered(proc, jobs, chunksize=4):
            for R in recs:
                f.write(json.dumps(R) + '\n')


if __name__ == '__main__':
    main()
