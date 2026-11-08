"""Hash join two frozen rating rounds. Possible ranges are not confidence intervals."""
from collections import Counter
from itertools import product
from statistics import mean
import re
from team_strength_packet import *
vals={'yes':[1],'no':[0],'uncertain':[0,1],'unassessable':[0,1]}
def main():
 p,key=build(); assert load(D/'masked.json')==p
 old={r['sha256']:r for r in load(R/'analysis/team-transition-v01/ratings.json')}
 rows=[]
 for t,k,r in zip(p,key,load(D/'ratings.json'),strict=True):
  assert t['id']==k['id']==r['id'] and r['sha256']==digest(t['text']) and re.search(re.escape(r['before'])+r'\s+'+re.escape(r['after']),t['text'])
  prior=old.get(r['sha256']);assert (prior is None)==(k['condition']=='DER3')
  rows.append(dict(**k,B=r['B'],prior=prior['B'] if prior else None))
 result={}
 for g in ('Claude','GPT-4.1'):
  rs=[r for r in rows if r['generator']==g];pairs=[]
  for q in sorted({r['q_id'] for r in rs}):
   a,b=[next(r for r in rs if r['q_id']==q and r['condition']==c) for c in ['DER2','DER3']]
   pair={'q':q,'ids':[a['id'],b['id']],'old2':a['prior'],'fresh2':a['B'],'fresh3':b['B']}
   for k in ('old2','fresh2'):
    ds=[y-x for x,y in product(vals[pair[k]],vals[pair['fresh3']])]
    pair[k+'_bounds']=[min(ds),max(ds)]
   pairs.append(pair)
  bounds={k:[mean(p[k+'_bounds'][j] for p in pairs) for j in (0,1)] for k in ('old2','fresh2')}
  counts={c:dict(Counter(r['B'] for r in rs if r['condition']==c)) for c in ('NMULTI','DER2','DER3')}
  result[g]=dict(pairs=pairs,bounds=bounds,counts=counts,rerating_discrepancies=[r['id'] for r in rs if r['prior'] and r['B']!=r['prior']])
 freeze(D/'results.json',dict(by_source=result,rows=rows))
 for g,v in result.items():print(g,v['bounds'],v['counts'],v['rerating_discrepancies']);print([(x['q'],x['old2'],x['fresh3']) for x in v['pairs']])
if __name__=='__main__':main()
