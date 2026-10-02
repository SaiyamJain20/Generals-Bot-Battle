import gzip,json,collections,numpy as np
FOCUS={'ResBot':0,'nanomena':0,'Kubic':0}
CAP=200
rows=[]
for line in open('data/acts.jsonl'):
    if all(v>=CAP for v in FOCUS.values()):break
    head=line[:300]
    if not any(f'"{b}"' in head and FOCUS[b]<CAP for b in FOCUS):continue
    g=json.loads(line)
    if g.get('fails',0) or g['winner'] not in (0,1):continue
    w=g['winner']; nm=g['players'][w]
    if nm not in FOCUS or FOCUS[nm]>=CAP or g['ticks']>=800:continue
    d=json.load(gzip.open('data/replays/'+g['file']))
    FOCUS[nm]+=1
    A=np.array([t['armies'] for t in d['ticks']]);O=np.array([t['owners'] for t in d['ticks']])
    H,W=A.shape[1:];gq=d['generals'][1-w];tt=d['total_ticks']
    rr,cc=np.mgrid[0:H,0:W];dg=np.abs(rr-gq[0])+np.abs(cc-gq[1])
    r={'name':nm,'tt':tt}
    for off in (1,3,6,10,20,40):
        t=max(0,min(tt-off,len(A)-1))
        win=(O[t]==w)
        r['st%d'%off]=int((A[t]*(win&(dg<=2))).max()) if (win&(dg<=2)).any() else 0
        r['st8_%d'%off]=int(A[t][win&(dg<=8)].max()) if (win&(dg<=8)).any() else 0
        r['eg%d'%off]=int(A[t][gq[0],gq[1]])
        r['ea3_%d'%off]=int((A[t]*((O[t]==1-w)&(dg<=3))).sum())
        r['eatot%d'%off]=int((A[t]*(O[t]==1-w)).sum())
        r['wtot%d'%off]=int((A[t]*(O[t]==w)).sum())
        r['dmin%d'%off]=int(dg[win].min())
    # first time winner has owned cell within dist<=3 of loser's general
    ts=[t for t in range(len(A)) if (O[t]==w)[dg<=3].any()]
    r['first_d3']=ts[0] if ts else -1
    r['dmin_hist']=[int(dg[O[t]==w].min()) for t in range(max(0,tt-60),tt,5)]
    # closest approach of any winner stack >=30
    rows.append(r)
json.dump(rows,open('data/replay_kill.json','w'))
def q(x):
    x=np.array(x,float);return '%g [%g-%g]'%tuple(np.round(np.percentile(x,[50,25,75]),1))
for nm in FOCUS:
    R=[r for r in rows if r['name']==nm]
    print(nm,len(R),'turn',q([r['tt'] for r in R]))
    for off in (1,3,6,10,20,40):
        print('  T-%d'%off,'stack(r<=2)',q([r['st%d'%off] for r in R]),'stack(r<=8)',q([r['st8_%d'%off] for r in R]),'egen',q([r['eg%d'%off] for r in R]),'enemy army r<=3',q([r['ea3_%d'%off] for r in R]),'enemy total',q([r['eatot%d'%off] for r in R]),'win total',q([r['wtot%d'%off] for r in R]),'dmin',q([r['dmin%d'%off] for r in R]))
    print('  first_d3 turn',q([r['first_d3'] for r in R]),'time from first d<=3 to kill',q([r['tt']-r['first_d3'] for r in R]))
    print('  stack(r<=2)@T-1 / egen',q([r['st1']/max(1,r['eg1']) for r in R]), 'frac stack>=3*egen',np.mean([r['st1']>=3*r['eg1'] for r in R]))
