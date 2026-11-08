"""Deterministic boundary sampling from the frozen echo features; no model calls."""
import csv,re,json
from collections import defaultdict
from team_transition_packet import R,freeze,load,digest
D=R/'analysis/team-span-v01'
def build():
 F=load(R/'analysis/team-echo-controls-v01/features.json')
 with (R/'data/team-corpus-20261104/utterances.csv').open(encoding='utf-8-sig') as f: raw={digest(r['text']):r for r in csv.DictReader(f)}
 group=defaultdict(list)
 for f in F:
  c=f['condition'];g=f['generator'];q=f['question']
  if c not in ('NMULTI','PER2','DER2'):continue
  h=f['text_sha256'];r=raw[h];s=r['text']
  spans=[(m.start(),m.end(),m.group()) for m in re.finditer(r'[^,;:.!?—–]+',s) if re.search('[a-z]',m.group(),re.I)]
  hits={(x['before'],x['after']) for x in f['hits']}
  assert len(spans)-1==f['boundaries']
  for j,(a,b) in enumerate(zip(spans,spans[1:])):
   left=a[2].strip();right=b[2].strip()
   if min(len(re.findall('[a-z]+',p.lower())) for p in (left,right))<3:continue
   echo=(left,right) in hits;priority=digest(f'span-v1|{h}|{j}')
   row=dict(priority=priority,g=g,c=c,q=q,echo=echo,sha256=h,index=j)
   item=dict(question=r['question'],left=left,right=right,context=s[spans[max(0,j-2)][0]:spans[min(len(spans)-1,j+3)][1]].strip())
   group[g,c,echo].append((row,item))
 picks=[];frame=[]
 for key in sorted(group):
  vals=sorted(group[key],key=lambda x:x[0]['priority']);first={}
  for rec in vals:first.setdefault(rec[0]['q'],rec)
  sel=list(first.values())[:2]; picks.extend(sel)
  frame.append(dict(source=key[0],condition=key[1],echo=key[2],eligible=len(vals),questions=len(first),picked=len(sel)))
 picks.sort(key=lambda x:digest('mask:'+x[0]['priority']))
 return ([dict(id=f'S{j:02}',**p) for j,(k,p) in enumerate(picks,1)],
         [dict(id=f'S{j:02}',**k) for j,(k,p) in enumerate(picks,1)],frame)
if __name__=='__main__':freeze(D/'masked.json',build()[0])
