"""castle-study part 3 tables (data/castle_study_recs2.jsonl + recs.jsonl drains)."""
import json, collections, os
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BOTS = ['ResBot', 'nanomena', 'Kubic', 'bca', 'Chig']
recs = collections.defaultdict(list)
for line in open(os.path.join(ROOT, 'data/castle_study_recs2.jsonl')):
    R = json.loads(line); recs[R['name']].append(R)
SUB = [(40, 50), (50, 56), (56, 63), (63, 75), (75, 100), (100, 125), (125, 150), (150, 200)]
print('== moves per game in sub-windows: pass | gen-src moves (mean amt) | amt1->neutral | amt1->other | amt2-5 own/neu/ene | amt>=6 own/neu/ene | captures neutral/enemy')
for b in BOTS:
    print(b)
    for a, e in SUB:
        G = [R for R in recs[b] if R['T'] > e + 4]
        c = collections.Counter(); ga = []
        for R in G:
            for m in R['mv']:
                t, amt, isg, dg, typ, cap, sa, outw = m
                if not (a <= t < e): continue
                if amt < 0: c['pass'] += 1; continue
                if isg: c['gen'] += 1; ga.append(amt)
                if amt == 1: c['a1_' + ('neu' if typ == 'neutral' else 'oth')] += 1
                elif amt <= 5: c['a25_' + typ] += 1
                else: c['a6_' + typ] += 1
                if cap: c['cap_' + typ] += 1
        n = len(G)
        f = lambda k: c[k] / n
        print(f'  T{a}-{e} pass {f("pass"):.1f} | gen {f("gen"):.1f} ({np.mean(ga) if ga else 0:.1f}) | a1>neu {f("a1_neu"):.1f} a1>oth {f("a1_oth"):.1f} | a2-5 {f("a25_own"):.1f}/{f("a25_neutral"):.1f}/{f("a25_enemy"):.1f} | a6+ {f("a6_own"):.1f}/{f("a6_neutral"):.1f}/{f("a6_enemy"):.1f} | cap N {f("cap_neutral"):.1f} E {f("cap_enemy"):.1f}')
print('\n== army distribution snapshots (median): land | tot army | max stack | top3/tot | #cells>=6 | >=16 | >=41 | 1-army cells/land | front cells | front army/tot | army within 3 of enemy/tot | gen | enemy border cells')
for b in BOTS:
    print(b)
    for t in ('75', '100', '150', '200', '250', '300'):
        S = [R['snap'][t] for R in recs[b] if t in R['snap']]
        md = lambda k: np.median([s[k] for s in S])
        r = lambda k, k2: np.median([s[k] / max(1, s[k2]) for s in S])
        print(f'  t{t}: land {md("land"):.0f} army {md("tot"):.0f} max {md("max"):.0f} top3 {r("top3","tot"):.2f} n6 {md("n6"):.0f} n16 {md("n16"):.0f} n41 {md("n41"):.0f} ones {r("ones","land"):.2f} front {md("front"):.0f} frontA {r("front_army","tot"):.2f} near3A {r("near_army","tot"):.2f} gen {md("gen"):.0f} border {md("frontier_len"):.0f}')
print('\n== castle full drains under threat (th3>=10): dst own/enemy/neutral, amt moved')
R1 = collections.defaultdict(list)
for line in open(os.path.join(ROOT, 'data/castle_study_recs.jsonl')):
    i = line.find('"drains": ')
    j = line.find('"cticks"')
    name = json.loads(line[:line.find(', "opp"')] + '}')['name'] if False else None
    R = json.loads(line)
    for d in R['drains']:
        R1[R['name']].append(d)
for b in BOTS:
    D = [d for d in R1[b] if d['rem'] <= 1 and d['th3'] >= 10]
    D0 = [d for d in R1[b] if d['rem'] <= 1 and d['th3'] < 10]
    print(f'{b:10s} n={len(D)} own {np.mean([d["dst"]=="own" for d in D]):.2f} enemy {np.mean([d["dst"]=="enemy" for d in D]):.2f} neu {np.mean([d["dst"]=="neutral" for d in D]):.2f}; amt med {np.median([d["amt"] for d in D]):.0f}; threat/castle med {np.median([d["th3"]/max(1,d["ca"]) for d in D]):.2f} | no-threat drains: castle army med {np.median([d["ca"] for d in D0]):.0f} age-gap')
