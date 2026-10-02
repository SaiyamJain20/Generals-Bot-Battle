import json,collections,numpy as np
R=[json.loads(l) for l in open('data/replay_an.jsonl')]
B=['ResBot','nanomena','Kubic','bca','Chig']
by={b:[r for r in R if r['name']==b] for b in B}
def q(x):
    x=np.array(x,float)
    return '%g [%g-%g]'%tuple(np.round(np.percentile(x,[50,25,75]),1)) if len(x) else '-'
print('games',{b:len(by[b]) for b in B}, 'winrate',{b:round(np.mean([r['won'] for r in by[b]]),2) for b in B})
print('median turns',{b:int(np.median([r['turns'] for r in by[b]])) for b in B})
print('== 1 ECONOMY (land,army,castles,gen army) median[IQR]; enemy land/army')
for t in (50,100,150,200,300,400,500,600):
    for b in B:
        xs=[r['eco'][str(t)] for r in by[b] if str(t) in r['eco']]
        if len(xs)<20: continue
        xs=np.array(xs)
        print(t,b,len(xs),'land',q(xs[:,0]),'army',q(xs[:,1]),'cast',q(xs[:,2]),'gen',q(xs[:,3]),'| opp land',q(xs[:,4]),'army',q(xs[:,5]))
print('== 2 CASTLES')
for b in B:
    cs=[r['castles'] for r in by[b]]
    print(b,'frac with>=1',round(np.mean([len(c)>=1 for c in cs]),2),'>=3',round(np.mean([len(c)>=3 for c in cs]),2),'mean n',round(np.mean([len(c) for c in cs]),2))
    for k in range(3):
        x=[c[k] for c in cs if len(c)>k]
        if len(x)<10:continue
        x=np.array(x)
        print('  castle',k+1,len(x),'turn',q(x[:,0]),'price',q(x[:,1]),'d_gen',q(x[:,2]),'d_enemy',q(x[:,3]),'army_in_cell',q(x[:,5]))
    al=[c for cc in cs for c in cc]
    lost=[c for c in al if c[4] is not None]
    print('  lost frac',round(len(lost)/max(1,len(al)),3),'lost turn',q([c[4] for c in lost]),'life',q([c[4]-c[0] for c in lost]), 'd_enemy lost',q([c[3] for c in lost]),'d_enemy kept',q([c[3] for c in al if c[4] is None]))
    print('  dgen dist hist',dict(sorted(collections.Counter(min(c[2],10) for c in al).items())))
print('== 3 STRIKES')
for b in B:
    S=[(r,s) for r in by[b] for s in r['strikes']]
    n=len(by[b])
    print(b,'strikes/game',round(len(S)/n,2),'games with >=1',round(np.mean([len(r['strikes'])>0 for r in by[b]]),2))
    if not S:continue
    ss=[s for r,s in S]
    print('  turn',q([s['s'] for s in ss]),'mv0',q([s['mv0'] for s in ss]),'maxmv',q([s['maxmv'] for s in ss]),'len',q([s['n'] for s in ss]),'d0',q([s['d0'] for s in ss]))
    print('  mv/ownarmy',q([s['maxmv']/s['own'] for s in ss]),'maxmv/enemy_gen',q([s['maxmv']/max(1,s['egen']) for s in ss]),'maxmv/enemy_army',q([s['maxmv']/max(1,s['en']) for s in ss]),'egen',q([s['egen'] for s in ss]))
    print('  capture general',round(np.mean([s['capg'] for s in ss]),3),'castle',round(np.mean([s['capc'] for s in ss]),3),'land flipped>=5 ',round(np.mean([s['flip']>=5 for s in ss]),3),'flip med',np.median([s['flip'] for s in ss]))
    w=[s for r,s in S if s['capg']]
    print('  genkill strikes: mv',q([s['maxmv'] for s in w]),'egen',q([s['egen'] for s in w]),'turn',q([s['s'] for s in w]),'n',len(w),'of games won',sum(r['won'] for r in by[b]))
    # success by ratio
    for lo,hi in ((0,.8),(.8,1.2),(1.2,2),(2,99)):
        z=[s for s in ss if lo<=s['maxmv']/max(1,s['egen'])<hi]
        if z:print('   mv/egen',lo,hi,len(z),'capg',round(np.mean([s['capg'] for s in z]),2),'flip>=5',round(np.mean([s['flip']>=5 for s in z]),2))
print('== 4 DEFENSE events (enemy stack>=X within 6 of general)')
for b in B:
    for X in (30,60):
        E=[e for r in by[b] for e in r['def'] if e['sz']>=X]
        if len(E)<10:continue
        tot=lambda k:sum(e[k] for e in E)
        den=sum(e['gather']+e['att']+e['pas']+e['oth'] for e in E)
        print(b,'X',X,'n',len(E),'per game',round(len(E)/len(by[b]),2),'gather',round(tot('gather')/den,2),'attack',round(tot('att')/den,2),'pass',round(tot('pas')/den,2),'other',round(tot('oth')/den,2),
          '| lost<=30',round(np.mean([e['lost'] for e in E]),3),'gen/stack',q([e['gen']/e['sz'] for e in E]),'gen+r3/stack',q([e['gar3']/e['sz'] for e in E]),'gen',q([e['gen'] for e in E]),'own/en',q([e['own']/max(1,e['en']) for e in E]),'gen/en',q([e['gen']/max(1,e['en']) for e in E]),'gen10-gen',q([e['gen10']-e['gen'] for e in E]),'dist',q([e['dist'] for e in E]))
        for lo,hi in ((0,.5),(.5,1),(1,2),(2,99)):
            z=[e for e in E if lo<=e['gar3']/e['sz']<hi]
            if len(z)>5: print('    gar3/stack',lo,hi,len(z),'lost',round(np.mean([e['lost'] for e in z]),3))
print('== 5 BONUS: move class fractions in 10 ticks after each bonus')
for b in B:
    for k in (50,100,200,300,400):
        c=collections.Counter()
        for r in by[b]:
            for kk,v in r['bonus'].get(str(k),{}).items(): c[kk]+=v
        tt=sum(c.values())
        if tt>200: print(b,k,{kk:round(v/tt,2) for kk,v in sorted(c.items())})
print('== 6 DECISIVE')
for b in B:
    ws=[r for r in by[b] if r['won']]; ls=[r for r in by[b] if not r['won']]
    print(b,'wins',len(ws),'win turns',q([r['turns'] for r in ws]),'frac win <800',round(np.mean([r['turns']<800 for r in ws]),3),'frac >=1200',round(np.mean([r['turns']>=1200 for r in by[b]]),3),'loss turns',q([r['turns'] for r in ls]))
    for off in (100,50,0):
        x=np.array([r['end'][str(off)] for r in ws if r['turns']<800])
        if len(x)>5: print('   T-%d winner:'%off,'land ratio',q(x[:,0]/np.maximum(1,x[:,3])),'army ratio',q(x[:,1]/np.maximum(1,x[:,4])),'gen',q(x[:,2]),'egen',q(x[:,5]),'win land',q(x[:,0]),'opp land',q(x[:,3]))
    x=np.array([r['end']['50'] for r in ls if r['turns']<800])
    if len(x)>5: print('   T-50 as loser: land ratio',q(x[:,0]/np.maximum(1,x[:,3])),'army ratio',q(x[:,1]/np.maximum(1,x[:,4])),'gen',q(x[:,2]),'egen',q(x[:,5]))
    for t in ('100','200','300'):
        a=[(r['lead'][t],r['won']) for r in by[b] if t in r['lead']]
        ld=[(x[0]>x[2],w) for x,w in a]
        la=[(x[1]>x[3],w) for x,w in a]
        print('   P(win|land lead@%s)'%t,round(np.mean([w for l,w in ld if l]),2),'n',sum(l for l,w in ld),' P(win|no lead)',round(np.mean([w for l,w in ld if not l]),2),'| army lead',round(np.mean([w for l,w in la if l]),2),round(np.mean([w for l,w in la if not l]),2))
print('== 7 MOVES by phase (per 100 ticks): pass, split, build fraction')
for b in B:
    for ph in (0,1,2,3,5):
        mv=collections.Counter()
        for r in by[b]:
            for k,v in r['mv'].items():
                kk,p=k.split('_')
                if int(p)==ph: mv[kk]+=v
        tt=mv['pass']+mv['move']+mv['build']
        if tt>1000: print(b,'ph',ph*100,'pass',round(mv['pass']/tt,3),'split/move',round(mv['split']/max(1,mv['move']),3),'build/1000',round(1000*mv['build']/tt,1))
