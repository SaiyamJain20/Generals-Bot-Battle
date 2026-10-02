"""castle-study: per player-game castle keeping/loss + land-war stats from data/acts.jsonl + replays.

    .venv312/bin/python agents_shared/castle_study/study.py [workers=4] [out=agents_shared/castle_study/recs.jsonl]
"""
import gzip, json, sys, os, collections
import numpy as np
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CAPS = {'ResBot': 500, 'nanomena': 500, 'Kubic': 500, 'FreeLunch': 250, 'bca': 400, 'Chig': 400}
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
WINS = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 400), (400, 600)]
CHK = (50, 75, 100, 150, 200, 250, 300, 400)


def win_of(t):
    for k, (a, b) in enumerate(WINS):
        if a <= t < b:
            return k
    return -1


def dil(m):
    x = m.copy()
    x[1:, :] |= m[:-1, :]; x[:-1, :] |= m[1:, :]
    y = x.copy()
    y[:, 1:] |= x[:, :-1]; y[:, :-1] |= x[:, 1:]
    return y


def proc(arg):
    line, players = arg
    g = json.loads(line)
    d = json.load(gzip.open(os.path.join(ROOT, 'data/replays', g['file'])))
    A = np.array([t['armies'] for t in d['ticks']], dtype=np.int32)
    O = np.array([t['owners'] for t in d['ticks']], dtype=np.int8)
    T, H, W = A.shape
    acts = g['acts']
    names = g['players']; win = g['winner']
    gens = [tuple(x) for x in d['generals']]
    rr, cc = np.mgrid[0:H, 0:W]
    end_by_cap = win in (0, 1) and T < 1200
    out = []
    for p in players:
        q = 1 - p
        gp, gq = gens[p], gens[q]
        DG = np.abs(rr - gp[0]) + np.abs(cc - gp[1])
        DQ = np.abs(rr - gq[0]) + np.abs(cc - gq[1])
        R = {'file': g['file'], 'name': names[p], 'opp': names[q], 'won': int(win == p), 'T': T, 'H': H, 'W': W}
        land = (O == p).sum((1, 2)); eland = (O == q).sum((1, 2))
        army = (A * (O == p)).sum((1, 2)); earmy = (A * (O == q)).sum((1, 2))
        R['land'] = {t: [int(land[t]), int(eland[t]), int(army[t]), int(earmy[t])] for t in CHK if t < T}
        tend = T - 4 if end_by_cap else T - 1   # ignore the final elimination flip
        # ---------------- castles ----------------
        builds = [(t, ac[p][1], ac[p][2]) for t, ac in enumerate(acts) if ac[p][0] == 2]
        cas = []
        drains = []
        ticks = []   # per castle-tick (age, castle army, thr3, thr5, thr8 visible, thr5 full, lost_within20)
        for k, (t0, r, c) in enumerate(builds):
            if t0 >= tend:
                continue
            Dm = np.abs(rr - r) + np.abs(cc - c)
            col = O[t0 + 1:, r, c]
            bad = np.nonzero(col != p)[0]
            tl = int(t0 + 1 + bad[0]) if len(bad) else None
            if tl is not None and tl >= tend:
                tl = None   # lost in final flip -> treat as kept
            stop = tl if tl is not None else tend
            traj = {}
            for kk in (1, 5, 10, 20, 40, 80):
                t = t0 + kk
                traj[kk] = int(A[t, r, c]) if t < stop else (-1 if (tl is not None and t >= tl) else None)
            C = {'t0': t0, 'dg': int(DG[r, c]), 'dq': int(DQ[r, c]), 'a0': int(A[t0, r, c]), 'traj': traj, 'tl': tl, 'k': k}
            em0 = (O[t0] == q)
            C['de'] = int(Dm[em0].min()) if em0.any() else 99
            thr_hist = []
            last_out = None
            for t in range(t0 + 1, stop):
                vis = dil(O[t] == p)
                E = (O[t] == q)
                Ev = E & vis
                ca = int(A[t, r, c])
                th = []
                for Rd in (3, 5, 8):
                    m = Ev & (Dm <= Rd)
                    th.append(int(A[t][m].max()) if m.any() else 0)
                m = E & (Dm <= 5)
                thf = int(A[t][m].max()) if m.any() else 0
                lost20 = int(tl is not None and tl - t <= 20)
                thr_hist.append(th[1])
                if t - t0 <= 200:
                    ticks.append((t - t0, ca, th[0], th[1], th[2], thf, lost20, t))
                a = acts[t][p]
                if a[0] == 0 and a[1] == r and a[2] == c:
                    amt = ca // 2 if a[4] else ca - 1
                    if amt > 0:
                        dr, dc = DIRS[a[3]]
                        r2, c2 = r + dr, c + dc
                        dst = 'x'
                        if 0 <= r2 < H and 0 <= c2 < W:
                            o2 = O[t, r2, c2]
                            dst = 'own' if o2 == p else ('enemy' if o2 == q else 'neutral')
                        lost_in = (tl - t) if tl is not None else None
                        drains.append({'k': k, 't': t, 'age': t - t0, 'ca': ca, 'amt': amt, 'rem': ca - amt, 'split': a[4], 'dst': dst,
                                       'th3': th[0], 'th5': th[1], 'th8': th[2], 'thf': thf, 'lost_in': lost_in,
                                       'army': int(army[t]), 'earmy': int(earmy[t])})
                        last_out = (t, ca - amt)
            C['nout'] = sum(1 for x in drains if x['k'] == k)
            if tl is not None:
                t = tl - 1
                aq = acts[t][q]
                info = {'ca_pre': int(A[t, r, c]), 'thr5_pre': thr_hist[-1] if thr_hist else 0}
                if aq[0] == 0:
                    sr, sc = aq[1], aq[2]
                    dr, dc = DIRS[aq[3]]
                    if (sr + dr, sc + dc) == (r, c) and O[t, sr, sc] == q:
                        sa = int(A[t, sr, sc]); amt = sa // 2 if aq[4] else sa - 1
                        info.update(att_src=sa, att_amt=amt)
                ap = acts[t][p]
                info['p_moved_out'] = int(ap[0] == 0 and ap[1] == r and ap[2] == c)
                own = (O[t] == p).copy(); own[r, c] = False
                amt = info.get('att_amt', int(A[t, r, c]) + 1)
                big = own & (A[t] >= amt)
                info['d_own_big'] = int(Dm[big].min()) if big.any() else 99
                idx = np.unravel_index(np.argmax(np.where(own, A[t], -1)), A[t].shape)
                info['own_max'] = int(A[t][idx]); info['own_max_d'] = int(Dm[idx])
                info['own_r3'] = int((A[t] * own * (Dm <= 3)).sum())
                info['army'] = int(army[t]); info['earmy'] = int(earmy[t])
                # warning time: ticks before loss since a visible enemy stack >= 0.8*attack amount was within 5
                warn = 0
                for j in range(len(thr_hist) - 1, -1, -1):
                    if thr_hist[j] >= 0.8 * amt:
                        warn += 1
                    else:
                        break
                info['warn'] = warn
                info['last_out_dt'] = (tl - last_out[0]) if last_out else None
                info['last_out_rem'] = last_out[1] if last_out else None
                col2 = O[tl:, r, c]
                back = np.nonzero(col2 == p)[0]
                info['recap'] = int(back[0]) if len(back) else None
                info['post'] = [int(A[tl + j, r, c]) for j in range(0, 6) if tl + j < T and O[tl + j, r, c] == q]
                if info['recap'] is not None:
                    tr = tl + info['recap'] - 1
                    ar = acts[tr][p]
                    if ar[0] == 0 and O[tr, ar[1], ar[2]] == p:
                        sa = int(A[tr, ar[1], ar[2]])
                        info['recap_amt'] = sa // 2 if ar[4] else sa - 1
                        info['recap_ca'] = int(A[tr, r, c])
                        info['recap_d_src_gen'] = int(DG[ar[1], ar[2]])
                info['age'] = tl - t0
                C['loss'] = info
            cas.append(C)
        R['castles'] = cas
        R['drains'] = drains
        R['cticks'] = ticks
        # ---------------- land war ----------------
        cap_e = []  # enemy captures: t, amt, srcarmy, tgtarmy, dg, dq
        fail_e = collections.Counter()
        neut = collections.Counter()   # (win, amt bucket)
        nmoves = collections.Counter(); nsplit = collections.Counter(); npass = collections.Counter()
        chains = []
        cur = None
        prev_dst = None; prev_t = -9
        for t in range(min(len(acts), tend)):
            a = acts[t][p]
            w = win_of(t)
            if a[0] != 0:
                npass[w] += a[0] == 1
                if cur:
                    chains.append(cur); cur = None
                prev_dst = None
                continue
            r, c = a[1], a[2]
            dr, dc = DIRS[a[3]]
            r2, c2 = r + dr, c + dc
            if not (0 <= r2 < H and 0 <= c2 < W) or O[t, r, c] != p:
                continue
            sa = int(A[t, r, c]); amt = sa // 2 if a[4] else sa - 1
            if amt <= 0:
                continue
            nmoves[w] += 1; nsplit[w] += a[4]
            o2 = int(O[t, r2, c2]); post = int(O[t + 1, r2, c2])
            ecap = o2 == q and post == p
            ncap = o2 == -1 and post == p
            if ecap:
                cap_e.append((t, amt, sa, int(A[t, r2, c2]), int(DG[r2, c2]), int(DQ[r2, c2])))
            elif o2 == q:
                fail_e[w] += 1
            if ncap:
                b = 1 if amt == 1 else (2 if amt <= 3 else 3)
                neut[(w, b)] += 1
            cont = cur is not None and prev_dst == (r, c) and prev_t == t - 1
            if not cont:
                if cur:
                    chains.append(cur)
                cur = {'t0': t, 'm0': amt, 'n': 0, 'ec': 0, 'nc': 0, 'dq0': int(DQ[r, c]), 'dg0': int(DG[r, c])}
            cur['n'] += 1; cur['ec'] += ecap; cur['nc'] += ncap
            prev_dst = (r2, c2); prev_t = t
        if cur:
            chains.append(cur)
        R['cap_e'] = cap_e
        R['fail_e'] = dict(fail_e)
        R['neut'] = {f'{k[0]}_{k[1]}': v for k, v in neut.items()}
        R['nmoves'] = dict(nmoves); R['nsplit'] = dict(nsplit); R['npass'] = dict(npass)
        R['chains'] = [[x['t0'], x['m0'], x['n'], x['ec'], x['nc'], x['dq0'], x['dg0']] for x in chains if x['ec'] > 0 or x['n'] >= 3]
        # losses + recapture near general
        lost = collections.Counter()
        rec = []
        for t in range(tend):
            fl = (O[t] == p) & (O[t + 1] == q)
            if not fl.any():
                continue
            lost[win_of(t)] += int(fl.sum())
            for (r, c) in zip(*np.nonzero(fl & (DG <= 5))):
                col = O[t + 1:tend + 1, r, c]
                back = np.nonzero(col == p)[0]
                rec.append((t, int(DG[r, c]), int(back[0]) + 1 if len(back) else None))
        R['lost'] = dict(lost)
        R['recap'] = rec
        # opening potential at t=50/75/100: own non-general cells with army>=2 and a neutral neighbour
        pot = {}
        for t in (50, 51, 75, 100, 101):
            if t >= T:
                continue
            own = O[t] == p
            nb = np.zeros_like(own)
            ne = O[t] == -1
            nb[1:, :] |= ne[:-1, :]; nb[:-1, :] |= ne[1:, :]; nb[:, 1:] |= ne[:, :-1]; nb[:, :-1] |= ne[:, 1:]
            g2 = own & (A[t] >= 2); g2[gp] = False
            pot[t] = [int((g2 & nb).sum()), int(g2.sum()), int(A[t][gp])]
        R['pot'] = pot
        out.append(R)
    return out


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    outp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'data/castle_study_recs.jsonl')
    cnt = collections.Counter()
    jobs = []
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
    n = 0
    with Pool(nw) as pool, open(outp, 'w') as f:
        for recs in pool.imap_unordered(proc, jobs, chunksize=4):
            for R in recs:
                f.write(json.dumps(R) + '\n')
            n += 1
            if n % 200 == 0:
                print(n, file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
