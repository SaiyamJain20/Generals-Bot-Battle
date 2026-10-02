import gzip,json,sys,collections
import numpy as np
FOCUS=['ResBot','nanomena','Kubic','bca','Chig']
CAP=int(sys.argv[1]) if len(sys.argv)>1 else 330
OUT=sys.argv[2] if len(sys.argv)>2 else 'data/replay_an.jsonl'
cnt=collections.Counter()
fo=open(OUT,'w')
def man(a,b):return abs(a[0]-b[0])+abs(a[1]-b[1])
def proc(g,line):
    d=json.load(gzip.open('data/replays/'+g['file']))
    T=len(d['ticks'])
    A=np.array([t['armies'] for t in d['ticks']],dtype=np.int32)
    O=np.array([t['owners'] for t in d['ticks']],dtype=np.int8)
    H,W=A.shape[1:]
    gens=[tuple(x) for x in d['generals']]
    acts=g['acts']; names=g['players']; win=g['winner']; tt=d['total_ticks']
    # castles from builds
    builds=[]  # (t,p,r,c)
    for t,(a0,a1) in enumerate(acts):
        for p,a in ((0,a0),(1,a1)):
            if a[0]==2: builds.append((t,p,a[1],a[2]))
    rr,cc=np.mgrid[0:H,0:W]
    recs=[]
    for p in (0,1):
        nm=names[p]
        if nm not in FOCUS or cnt[nm]>=CAP: continue
        cnt[nm]+=1
        q=1-p
        R={'game':g['file'],'name':nm,'opp':names[1-p],'p':p,'won':int(win==p),'turns':tt,'H':H,'W':W}
        land=(O==p).sum((1,2)); eland=(O==q).sum((1,2))
        army=(A*(O==p)).sum((1,2)); earmy=(A*(O==q)).sum((1,2))
        gp=gens[p]; gq=gens[q]
        garmy=A[:,gp[0],gp[1]]; egarmy=A[:,gq[0],gq[1]]
        # castles
        mine=[(t,r,c) for (t,pp,r,c) in builds if pp==p]
        theirs=[(t,r,c) for (t,pp,r,c) in builds if pp==q]
        def ncast(t,pl):
            return sum(1 for (tb,pp,r,c) in builds if pp==pl and tb<t and O[min(t,T-1),r,c]==pl)
        eco={}
        for t in range(50,601,50):
            if t<T:
                eco[t]=[int(land[t]),int(army[t]),ncast(t,p),int(garmy[t]),int(eland[t]),int(earmy[t]),int(egarmy[t]),ncast(t,q)]
        R['eco']=eco
        cs=[]
        for i,(t,r,c) in enumerate(mine):
            own=[gp]+[(r2,c2) for (t2,r2,c2) in mine[:i] if O[t,r2,c2]==p]
            price=35+sum(max(0,14-2*man((r,c),x)) for x in own)
            em=(O[t]==q)
            de=int((np.abs(rr-r)+np.abs(cc-c))[em].min()) if em.any() else -1
            lost=None
            col=O[t+1:,r,c]
            bad=np.nonzero(col!=p)[0]
            if len(bad): lost=int(t+1+bad[0])
            cs.append([t,price,man((r,c),gp),de,lost,int(A[t,r,c]),len(own)])
        R['castles']=cs
        R['ecastles']=[[t,man((r,c),gq)] for (t,r,c) in theirs]
        # move stats per phase
        mv=collections.Counter()
        strikes=[]; cur=None
        bon=collections.defaultdict(collections.Counter)
        dg=np.abs(rr-gq[0])+np.abs(cc-gq[1])
        for t,ac in enumerate(acts):
            a=ac[p]; ph=min(t//100,11)
            k=a[0]
            if k==1: mv[('pass',ph)]+=1
            elif k==2: mv[('build',ph)]+=1
            else:
                mv[('move',ph)]+=1
                if a[4]: mv[('split',ph)]+=1
        for t,ac in enumerate(acts):
            a=ac[p]
            cls=None
            if a[0]==1:cls='pass'
            elif a[0]==2:cls='build'
            elif a[0]==0:
                r,c,dr,sp=a[1],a[2],a[3],a[4]
                dd=((-1,0),(1,0),(0,-1),(0,1))[dr]
                r2,c2=r+dd[0],c+dd[1]
                if 0<=r2<H and 0<=c2<W:
                    ow=O[t,r2,c2]
                    cls='neutral' if ow==-1 else ('enemy' if ow==q else 'own')
                    if O[t,r2,c2]==-1 and A[t,r2,c2]>=0 and False: pass
                else: cls='bad'
                # strike detection
                if O[t,r,c]==p and 0<=r2<H and 0<=c2<W:
                    av=int(A[t,r,c]); moved=av//2 if sp else av-1
                    toward=dg[r2,c2]<dg[r,c]
                    if cur is not None and cur['pos']==(r,c) and cur['last']==t-1 and moved>=15:
                        cur['pos']=(r2,c2); cur['last']=t; cur['maxmv']=max(cur['maxmv'],moved); cur['n']+=1
                        cur['endmv']=moved
                    else:
                        if cur is not None: strikes.append(cur); cur=None
                        if moved>=30 and moved>=0.25*army[t] and toward:
                            cur={'s':t,'pos':(r2,c2),'last':t,'maxmv':moved,'n':1,'mv0':moved,'endmv':moved,
                                 'own':int(army[t]),'en':int(earmy[t]),'egen':int(egarmy[t]),'d0':int(dg[r,c]),'ef':int(eland[t]),'ef_land':int(land[t])}
            if t>=50 and (t-50)%50<10 and cls:
                bon[(t-50)//50*50+50][cls]+=1
        if cur is not None: strikes.append(cur)
        out=[]
        for s in strikes:
            e=s['last']; e1=min(e+1,T-1)
            flip=int(((O[s['s']]==q)&(O[e1]==p)).sum())
            capg=int(win==p and tt-e<=4)
            ec=[(t2,r,c) for (t2,r,c) in theirs if t2<s['s']]
            capc=int(any(O[s['s'],r,c]==q and O[e1,r,c]==p for (_,r,c) in ec))
            # defender general army change
            s.update(flip=flip,capg=capg,capc=capc,egen_end=int(egarmy[e1]),pos=None)
            s['pos']=None
            out.append(s)
        R['strikes']=out
        R['bonus']={str(k):dict(v) for k,v in bon.items()}
        R['mv']={f'{k[0]}_{k[1]}':v for k,v in mv.items()}
        # defense events
        ev=[]; lastt=-99; lastsz=0
        mask=(np.abs(rr-gp[0])+np.abs(cc-gp[1]))<=6
        for t in range(20,T-1):
            em=(O[t]==q)&mask
            if not em.any(): continue
            sz=int(A[t][em].max())
            if sz>=25 and (t-lastt>15 or sz>=1.5*lastsz):
                lastt=t; lastsz=sz
                idx=np.unravel_index(np.argmax(np.where(em,A[t],-1)),A[t].shape)
                dist=int(abs(idx[0]-gp[0])+abs(idx[1]-gp[1]))
                gather=att=pas=oth=0
                for t2 in range(t,min(t+10,T-1)):
                    a=acts[t2][p]
                    if a[0]==1: pas+=1;continue
                    if a[0]==2: oth+=1;continue
                    r,c=a[1],a[2]; dd=((-1,0),(1,0),(0,-1),(0,1))[a[3]]; r2,c2=r+dd[0],c+dd[1]
                    if not(0<=r2<H and 0<=c2<W): oth+=1;continue
                    if O[t2,r2,c2]==q: att+=1
                    elif abs(r2-gp[0])+abs(c2-gp[1])<abs(r-gp[0])+abs(c-gp[1]): gather+=1
                    else: oth+=1
                t10=min(t+10,T-1)
                gar3=int((A[t]*((O[t]==p)&((np.abs(rr-gp[0])+np.abs(cc-gp[1]))<=3))).sum())
                lost=int(win==q and tt-t<=30)
                ev.append(dict(t=t,sz=sz,dist=dist,gen=int(garmy[t]),gen10=int(garmy[t10]),gar3=gar3,own=int(army[t]),en=int(earmy[t]),
                    gather=gather,att=att,pas=pas,oth=oth,lost=lost,sz10=int(A[t10][(O[t10]==q)&mask].max()) if ((O[t10]==q)&mask).any() else 0,
                    ownland=int(land[t]),enland=int(eland[t])))
        R['def']=ev
        # decisive stats
        Rt={}
        for off in (0,50,100):
            t=max(0,tt-off)
            t=min(t,T-1)
            Rt[off]=[int(land[t]),int(army[t]),int(garmy[t]),int(eland[t]),int(earmy[t]),int(egarmy[t])]
        R['end']=Rt
        # early lead predictors
        R['lead']={str(t):[int(land[t]),int(army[t]),int(eland[t]),int(earmy[t])] for t in (100,200,300) if t<T}
        recs.append(R)
    return recs
n=0
for line in open('data/acts.jsonl'):
    if all(cnt[b]>=CAP for b in FOCUS): break
    # cheap filter
    g=None
    head=line[:300]
    if not any(f'"{b}"' in head and cnt[b]<CAP for b in FOCUS): continue
    g=json.loads(line)
    if g.get('fails',0)>0 or g['winner'] not in (0,1): continue
    for R in proc(g,line):
        fo.write(json.dumps(R)+'\n')
    n+=1
    if n%50==0: print(n,dict(cnt),flush=True); fo.flush()
print(dict(cnt))
