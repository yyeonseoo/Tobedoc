"""Posthoc form descriptor, not a semantic or clinical score. No inference calls."""
import re
from team_transition_packet import *
from itertools import pairwise
from statistics import median
from collections import defaultdict
# Split at common visible clause/sentence separators. No POS tagger, stemming or synonym mapping.
stop=set('a an the and but or so nor as if than because although that this these those which who what when where while then there here it its my me mine i you your we our they their them he she his her him us to of on at in for by with from into out up off over is are was were be being been have has had do does did can could would should will not just very really like one all some more most get got'.split())
def tokens(s):return [w for w in re.findall(r"[a-z]+",s.casefold())]
def get(text):
 chunks=[s.strip() for s in re.split('[,;:.!?—–]+',text) if tokens(s)]
 hits=[]
 for a,b in pairwise(chunks):
  inter=(set(tokens(a)[-4:])&set(tokens(b)[:4]))-stop
  words=sorted(w for w in inter if len(w)>=3)
  if words:hits.append(dict(before=a,after=b,shared=words))
 n=len(chunks)-1
 return {'hits':hits,'boundaries':n,'rate':len(hits)/n if n else None}
packet,_=build();rows=load(O/'results.json')['rows']
groups=defaultdict(dict)
for r,p in zip(rows,packet,strict=True):
 assert r['id']==p['id']
 groups[(r['generator'],r['condition'])][r['q_id']]=get(p['text'])
summary={}
for g in sorted({r['generator'] for r in rows}):
 d=groups[(g,'DER2')];n=groups[(g,'NMULTI')]
 delta=[d[q]['rate']-n[q]['rate'] for q in d]
 summary[g]=dict(paired_delta_pp=[round(100*(d[q]['rate']-n[q]['rate']),3) for q in sorted(d)],median_delta_pp=round(100*median(delta),3),positive=sum(x>0 for x in delta),conditions={c:{'hits':sum(len(v['hits']) for v in groups[(g,c)].values()),'boundaries':sum(v['boundaries'] for v in groups[(g,c)].values())} for c in ('N0','NMULTI','DER2')})
freeze(O/'echo.json',dict(summary=summary,details={g+' / '+c:v for (g,c),v in groups.items()}))
print(summary)
