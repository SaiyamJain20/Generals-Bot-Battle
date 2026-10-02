"""castle-study part 2: castle-loss extras, land war (Q3), opening/economy (Q4)."""
import json, sys, collections, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BOTS = ['ResBot', 'nanomena', 'Kubic', 'FreeLunch', 'bca', 'Chig']
TOP = {'ResBot', 'nanomena', 'Kubic', 'FreeLunch'}
WINS = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 400)]
opp_filter = sys.argv[1] if len(sys.argv) > 1 else 'all'
recs = collections.defaultdict(list)
for line in open(os.path.join(ROOT, 'data/castle_study_recs.jsonl')):
    R = json.loads(line)
    if opp_filter == 'top' and R['opp'] not in TOP: continue
    if opp_filter == 'weak' and R['opp'] in TOP: continue
    R.pop('cticks', None)
    recs[R['name']].append(R)


def q(x, ps=(25, 50, 75), f='.0f'):
    x = [v for v in x if v is not None]
    if not x: return '-'
    return '/'.join(format(np.percentile(x, p), f) for p in ps)


def fr(x):
    return f'{np.mean(x):.2f}' if len(x) else '-'


def wi(t):
    for k, (a, b) in enumerate(WINS):
        if a <= t < b: return k
    return -1


def alive(R, k):
    return R['T'] - 4 >= WINS[k][1]


print(f'opp filter={opp_filter}', {b: len(recs[b]) for b in BOTS})
print('\n== Q2b castle loss aftermath: enemy army on castle at loss tick / +5 (while enemy-owned); recapture move amount; recap src dist to own general')
for b in BOTS:
    L = [c['loss'] for R in recs[b] for c in R['castles'] if c.get('loss')]
    p0 = [l['post'][0] for l in L if l.get('post')]
    p5 = [l['post'][-1] for l in L if l.get('post')]
    print(f'{b:10s} castle army after capture {q(p0)} (last enemy-owned obs {q(p5)}); recap tick {q([l["recap"] for l in L])}; recap amt {q([l.get("recap_amt") for l in L])} vs castle {q([l.get("recap_ca") for l in L])}; recap src d_gen {q([l.get("recap_d_src_gen") for l in L])}')
print('   P(castle lost before game end | position at build): by d_enemy_cell at build, by dist to enemy general, by castle index')
for b in BOTS:
    C = [c for R in recs[b] for c in R['castles']]
    s = []
    for lo, hi in ((0, 3), (4, 6), (7, 9), (10, 99)):
        v = [c.get('loss') is not None for c in C if lo <= c['de'] <= hi]
        s.append(f'de{lo}-{hi}:{fr(v)}[{len(v)}]')
    for lo, hi in ((0, 12), (13, 16), (17, 20), (21, 99)):
        v = [c.get('loss') is not None for c in C if lo <= c['dq'] <= hi]
        s.append(f'dq{lo}-{hi}:{fr(v)}[{len(v)}]')
    for k in (0, 1, 2, 3):
        v = [c.get('loss') is not None for c in C if (c['k'] == k if k < 3 else c['k'] >= 3)]
        s.append(f'#{k + 1}{"+" if k == 3 else ""}:{fr(v)}')
    print(f'{b:10s}', ' '.join(s))

print('\n== Q3 enemy-tile captures per 50 ticks (median; mean) | own tiles lost to enemy per 50t | failed attacks per 50t')
for b in BOTS:
    s = []
    for k, (a, e) in enumerate(WINS[1:], 1):
        G = [R for R in recs[b] if alive(R, k)]
        f = 50 / (e - a)
        ec = [sum(1 for x in R['cap_e'] if wi(x[0]) == k) * f for R in G]
        lo = [R['lost'].get(str(k), 0) * f for R in G]
        fa = [R['fail_e'].get(str(k), 0) * f for R in G]
        s.append(f'T{a}-{e}: cap {np.median(ec):.0f};{np.mean(ec):.1f} lost {np.median(lo):.0f};{np.mean(lo):.1f} fail {np.mean(fa):.1f}')
    print(f'{b:10s}', ' | '.join(s))
print('   capturing stack (moved amount) for enemy captures, T50-300: frac 2-3 / 4-6 / 7-15 / 16-40 / 41+;  target army 1 / 2 / 3+;  dist from own gen q25/50/75; from enemy gen')
for b in BOTS:
    for (a, e) in ((50, 150), (150, 300)):
        X = [x for R in recs[b] for x in R['cap_e'] if a <= x[0] < e]
        am = np.array([x[1] for x in X]); tg = np.array([x[3] for x in X])
        bk = [((am >= 2) & (am <= 3)).mean(), ((am >= 4) & (am <= 6)).mean(), ((am >= 7) & (am <= 15)).mean(), ((am >= 16) & (am <= 40)).mean(), (am >= 41).mean()]
        print(f'{b:10s} T{a}-{e} n={len(X)} amt ' + '/'.join(f'{v:.2f}' for v in bk) +
              f'  tgt {np.mean(tg == 1):.2f}/{np.mean(tg == 2):.2f}/{np.mean(tg >= 3):.2f}  dg {q([x[4] for x in X])}  dq {q([x[5] for x in X])}')
print('\n== Q3 chains (same stack moving on consecutive ticks) with >=1 enemy capture, per 50 ticks (mean per game) by first-move amount m0 and window; ec per chain (q25/50/75)')
CB = [(2, 5), (6, 15), (16, 40), (41, 99999)]
for b in BOTS:
    for k, (a, e) in enumerate(WINS[1:], 1):
        if e > 300: break
        G = [R for R in recs[b] if alive(R, k)]
        f = 50 / (e - a)
        s = []
        for lo, hi in CB:
            ch = [[c for c in R['chains'] if wi(c[0]) == k and c[3] > 0 and lo <= c[1] <= hi] for R in G]
            rate = np.mean([len(x) for x in ch]) * f
            allc = [c for x in ch for c in x]
            s.append(f'm0 {lo}-{hi}: {rate:.1f}/50t ec {q([c[3] for c in allc])} len {q([c[2] for c in allc])}')
        tot = np.mean([sum(c[3] for c in R['chains'] if wi(c[0]) == k and c[3] > 0) for R in G]) * f
        print(f'{b:10s} T{a}-{e} | ' + ' | '.join(s) + f' | ec in chains {tot:.1f}/50t')
print('   raid share: frac of enemy captures T50-200 made by chains with m0<=5 / m0 6-15 / m0>=16')
for b in BOTS:
    tot = collections.Counter()
    for R in recs[b]:
        for c in R['chains']:
            if 50 <= c[0] < 200 and c[3] > 0:
                key = 'small' if c[1] <= 5 else ('mid' if c[1] <= 15 else 'big')
                tot[key] += c[3]
    n = sum(tot.values())
    print(f'{b:10s} small {tot["small"] / n:.2f} mid {tot["mid"] / n:.2f} big {tot["big"] / n:.2f}')

print('\n== Q3 recapture of own tiles taken by enemy within dist<=5 of own general: dt q25/50/75; P(<=2), P(<=5), P(<=10), P(never)')
for b in BOTS:
    for (a, e) in ((0, 200), (200, 400), (400, 2000)):
        X = [x for R in recs[b] for x in R['recap'] if a <= x[0] < e]
        dt = [x[2] for x in X]
        print(f'{b:10s} T{a}-{e} n={len(X)} ({len(X) / max(1, len(recs[b])):.1f}/game) dt {q(dt)}  <=2 {fr([d is not None and d <= 2 for d in dt])} <=5 {fr([d is not None and d <= 5 for d in dt])} <=10 {fr([d is not None and d <= 10 for d in dt])} never {fr([d is None for d in dt])}')

print('\n== Q4 neutral captures per 50t: total (frac by moved amt 1 / 2-3 / 4+) | moves | splits | passes')
for b in BOTS:
    s = []
    for k, (a, e) in enumerate(WINS[:5]):
        G = [R for R in recs[b] if alive(R, k)]
        f = 50 / (e - a)
        n1 = np.mean([R['neut'].get(f'{k}_1', 0) for R in G]) * f
        n2 = np.mean([R['neut'].get(f'{k}_2', 0) for R in G]) * f
        n3 = np.mean([R['neut'].get(f'{k}_3', 0) for R in G]) * f
        mv = np.mean([R['nmoves'].get(str(k), 0) for R in G]) * f
        sp = np.mean([R['nsplit'].get(str(k), 0) for R in G]) * f
        ps = np.mean([R['npass'].get(str(k), 0) for R in G]) * f
        tot = n1 + n2 + n3
        s.append(f'T{a}-{e}: N {tot:.1f} ({n1 / tot:.2f}/{n2 / tot:.2f}/{n3 / tot:.2f}) mv {mv:.0f} sp {sp:.1f} ps {ps:.1f}')
    print(f'{b:10s}', ' | '.join(s))
print('   untapped frontier: own non-general cells with army>=2 adjacent to neutral (median) at t50/51/75/100/101; cells army>=2; general army')
for b in BOTS:
    s = []
    for t in ('50', '51', '75', '100', '101'):
        v = [R['pot'][t] for R in recs[b] if t in R['pot']]
        s.append(f't{t}: ' + '/'.join(f'{np.median([x[i] for x in v]):.0f}' for i in range(3)))
    print(f'{b:10s}', '  '.join(s))
