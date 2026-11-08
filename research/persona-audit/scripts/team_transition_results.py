"""Join locked readings; report identification bounds, not sampling intervals."""
from collections import Counter
from team_transition_packet import *
from itertools import product
from statistics import mean
packet,key=build(); ratings=load(O/'ratings.json')
assert packet==load(O/'masked.json') and len(packet)==len(ratings)
rows=[]
for r,k,t in zip(ratings,key,packet,strict=True):
 assert r['id']==k['id']==t['id'] and r['sha256']==k['sha256'] and r['before']+' '+r['after'] in t['text']
 rows.append(dict(**k,**{s:r[s] for s in 'ABC'}))
result={}
for g in sorted({r['generator'] for r in rows}):
 rs=[r for r in rows if r['generator']==g]
 stats={}; contrasts={}
 for c in ('N0','NMULTI','DER2'):
  sub=[r for r in rs if r['condition']==c];assert len(sub)==5
  stats[c]={s:dict(Counter(r[s] for r in sub)) for s in 'ABC'}
 for c in ('NMULTI','N0'):
  pairs=[]
  for q in sorted({r['q_id'] for r in rs}):
   pair=[next(r for r in rs if r['q_id']==q and r['condition']==a) for a in (c,'DER2')]
   vs=[{'yes':[1],'no':[0],'uncertain':[0,1],'unassessable':[0,1]}[r['B']] for r in pair]
   ds=[y-x for x,y in product(*vs)]
   pairs.append(dict(q=q,ids=[r['id'] for r in pair],labels=[r['B'] for r in pair],bounds=[min(ds),max(ds)],point=ds[0] if len(ds)==1 else None))
  pts=[v['point'] for v in pairs if v['point'] is not None]
  contrasts[c]=dict(pairs=pairs,paired_point=mean(pts) if pts else None,complete=len(pts),positive=sum(v['bounds'][0]>0 for v in pairs),bounds=[mean(p['bounds'][i] for p in pairs) for i in (0,1)])
 result[g]=dict(stats=stats,contrasts=contrasts)
freeze(O/'results.json',dict(rows=rows,by_source=result))
for g,v in result.items():print(g,v['stats'],v['contrasts']['NMULTI'])
