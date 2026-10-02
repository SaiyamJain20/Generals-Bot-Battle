"""Tables for castle-study from data/castle_study_recs.jsonl (produced by study.py)."""
import json, sys, collections, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BOTS = ['ResBot', 'nanomena', 'Kubic', 'FreeLunch', 'bca', 'Chig']
TOP = {'ResBot', 'nanomena', 'Kubic', 'FreeLunch'}
WINS = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 400), (400, 600)]
opp_filter = sys.argv[1] if len(sys.argv) > 1 else 'all'   # all | top | weak
recs = collections.defaultdict(list)
for line in open(os.path.join(ROOT, 'data/castle_study_recs.jsonl')):
    R = json.loads(line)
    if opp_filter == 'top' and R['opp'] not in TOP: continue
    if opp_filter == 'weak' and R['opp'] in TOP: continue
    recs[R['name']].append(R)


def q(x, ps=(25, 50, 75), f='.0f'):
    x = [v for v in x if v is not None]
    if not x: return '-'
    return '/'.join(format(np.percentile(x, p), f) for p in ps)


def fr(x):
    return f'{np.mean(x):.2f}' if len(x) else '-'


print(f'opp filter={opp_filter}  player-games:', {b: len(recs[b]) for b in BOTS})
print('\n== land/army checkpoints (median own land / opp land / own army / opp army)')
for b in BOTS:
    s = []
    for t in ('50', '75', '100', '150', '200', '300'):
        v = [R['land'][t] for R in recs[b] if t in R['land']]
        s.append(f"t{t}:" + '/'.join(f'{np.median([x[i] for x in v]):.0f}' for i in range(4)))
    print(f'{b:10s}', '  '.join(s))

print('\n== Q1 castles: per castle army at build+k (median; IQR) for castles still owned; P(lost by k)')
for b in BOTS:
    C = [c for R in recs[b] for c in R['castles']]
    s = [f'n={len(C)}']
    for k in ('1', '5', '10', '20', '40', '80'):
        v = [c['traj'][k] for c in C if c['traj'][k] is not None and c['traj'][k] >= 0]
        lost = [c['traj'][k] == -1 for c in C if c['traj'][k] is not None]
        s.append(f'+{k}:{q(v)} L{fr(lost)}')
    print(f'{b:10s}', ' '.join(s))
print('   lost overall (excl final flip), median age at loss, recapture <=10 / <=30 / ever')
for b in BOTS:
    C = [c for R in recs[b] for c in R['castles']]
    L = [c['loss'] for c in C if c.get('loss')]
    rc = [l['recap'] for l in L]
    print(f'{b:10s} lost {len(L)}/{len(C)}={len(L)/max(1,len(C)):.2f} age {q([l["age"] for l in L])} '
          f'recap<=10 {fr([r is not None and r <= 10 for r in rc])} <=30 {fr([r is not None and r <= 30 for r in rc])} ever {fr([r is not None for r in rc])}')

print('\n== Q1 move-outs from own castles (all castles, life<=end)')
for b in BOTS:
    D = [d for R in recs[b] for d in R['drains']]
    C = [c for R in recs[b] for c in R['castles']]
    life = sum(((c['tl'] or (R['T'] - 4)) - c['t0']) for R in recs[b] for c in R['castles'])
    full = [d for d in D if d['rem'] <= 1]
    half = [d for d in D if d['split']]
    print(f'{b:10s} moveouts/castle/100t {100*len(D)/max(1,life):.1f}; full(rem<=1) {len(full)/max(1,len(D)):.2f} split {len(half)/max(1,len(D)):.2f}; '
          f'army moved out {q([d["amt"] for d in D])}; castle army at moveout {q([d["ca"] for d in D])}; dst own {fr([d["dst"]=="own" for d in D])} enemy {fr([d["dst"]=="enemy" for d in D])}')
print('   threat (max VISIBLE enemy cell army within r of castle) at FULL drains: median r3/r5/r8; frac th5>=10, >=20, th5>=castle army')
for b in BOTS:
    D = [d for R in recs[b] for d in R['drains'] if d['rem'] <= 1]
    print(f'{b:10s} th3 {q([d["th3"] for d in D])} th5 {q([d["th5"] for d in D])} th8 {q([d["th8"] for d in D])} '
          f'th5>=10 {fr([d["th5"]>=10 for d in D])} th5>=20 {fr([d["th5"]>=20 for d in D])} th5>=ca {fr([d["th5"]>=d["ca"] for d in D])} th3>=10 {fr([d["th3"]>=10 for d in D])}')
print('   P(castle lost within 10 / 20 ticks after a FULL drain) by visible threat th5 bucket   [n]')
bk = [(0, 2), (3, 9), (10, 19), (20, 39), (40, 9999)]
for b in BOTS:
    D = [d for R in recs[b] for d in R['drains'] if d['rem'] <= 1]
    s = []
    for lo, hi in bk:
        v = [d for d in D if lo <= d['th5'] <= hi]
        s.append(f'th5 {lo}-{hi}: {fr([d["lost_in"] is not None and d["lost_in"]<=10 for d in v])}/{fr([d["lost_in"] is not None and d["lost_in"]<=20 for d in v])} [{len(v)}]')
    print(f'{b:10s}', ' | '.join(s))
print('   same, bucket by th3 (enemy within 3)')
for b in BOTS:
    D = [d for R in recs[b] for d in R['drains'] if d['rem'] <= 1]
    s = []
    for lo, hi in bk:
        v = [d for d in D if lo <= d['th3'] <= hi]
        s.append(f'th3 {lo}-{hi}: {fr([d["lost_in"] is not None and d["lost_in"]<=10 for d in v])}/{fr([d["lost_in"] is not None and d["lost_in"]<=20 for d in v])} [{len(v)}]')
    print(f'{b:10s}', ' | '.join(s))

print('\n== Q1 castle-ticks: castle army vs visible enemy within 5 (th5); P(lost within 20 ticks)')
for b in BOTS:
    X = np.array([t for R in recs[b] for t in R['cticks']], dtype=np.int64)
    if not len(X): continue
    age, ca, t3, t5, t8, tf, l20 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5], X[:, 6]
    s = [f'ticks {len(X)} castle army med {np.median(ca):.0f} (IQR {np.percentile(ca,25):.0f}-{np.percentile(ca,75):.0f}); base P(l20) {l20.mean():.3f}']
    print(f'{b:10s}', s[0])
    for lo, hi in bk:
        m = (t5 >= lo) & (t5 <= hi)
        if m.sum() < 30: continue
        cov = ca[m] >= t5[m]
        print(f'   th5 {lo:>2}-{hi:<4} frac {m.mean():.3f} castle army med {np.median(ca[m]):.0f}  covered(ca>=th5) {cov.mean():.2f}  '
              f'P(l20|covered) {l20[m][cov].mean() if cov.any() else float("nan"):.3f}  P(l20|not) {l20[m][~cov].mean() if (~cov).any() else float("nan"):.3f}')
    # ratio of castle army to threat when threat big
    m = t5 >= 10
    if m.sum():
        r = ca[m] / t5[m]
        print(f'   when th5>=10: castle/threat ratio q25/50/75 {np.percentile(r,25):.2f}/{np.median(r):.2f}/{np.percentile(r,75):.2f}; ratio>=1 {np.mean(r>=1):.2f}')

print('\n== Q2 castle losses: attacker moved amount, castle army before, attacker/castle, owner nearest stack>=attacker, owner max stack dist, warning ticks, last drain dt/rem')
for b in BOTS:
    L = [c['loss'] for R in recs[b] for c in R['castles'] if c.get('loss')]
    A_ = [l for l in L if 'att_amt' in l]
    print(f'{b:10s} n={len(L)} att {q([l["att_amt"] for l in A_])} ca_pre {q([l["ca_pre"] for l in L])} ratio {q([l["att_amt"]/max(1,l["ca_pre"]) for l in A_],(25,50,75),'.1f')} '
          f'd_own_big {q([l["d_own_big"] for l in L])} (none:{fr([l["d_own_big"]==99 for l in L])}) own_max {q([l["own_max"] for l in L])}@d{q([l["own_max_d"] for l in L])} '
          f'warn {q([l["warn"] for l in L])} lastout dt {q([l["last_out_dt"] for l in L])} rem {q([l["last_out_rem"] for l in L])} drained<=20t {fr([l["last_out_dt"] is not None and l["last_out_dt"]<=20 for l in L])} '
          f'p_moved_out_sametick {fr([l["p_moved_out"] for l in L])} armyratio {q([l["army"]/max(1,l["earmy"]) for l in L],(25,50,75),'.2f')}')
    rat = [l["att_amt"] / max(1, l["ca_pre"]) for l in A_]
    print(f'           attacker >= 2x castle {fr([r>=2 for r in rat])}  >=4x {fr([r>=4 for r in rat])}; castle army pre <=5 {fr([l["ca_pre"]<=5 for l in L])}  <=10 {fr([l["ca_pre"]<=10 for l in L])}')
