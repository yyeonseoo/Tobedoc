"""Matched comparisons of one frozen surface feature; offline, standard library only."""
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from statistics import median
import csv,json
from team_transition_packet import R,freeze,load
# This unmodified module also replays and verifies the previous 30 readings' echo file.
from team_transition_echo import get
D=R/'analysis/team-echo-controls-v01'
QS=('Q01','Q02','Q04','Q05','Q06')
CS=('N0','NMULTI','CIRC2','DER1','DER2','DER3','PER1','PER2','PER3')
def features():
 path=R/'data/team-corpus-20261104/utterances.csv'
 with path.open(encoding='utf-8-sig') as f: raw=list(csv.DictReader(f))
 assert len(raw)==228
 prior=load(R/'analysis/team-transition-v01/echo.json')['details']
 out={}; seen=0
 for r in raw:
  g,c,q=r['generator'],r['condition'],r['q_id']
  if q not in QS or c not in CS: continue
  key=(g,c,q); assert key not in out
  val=get(r['text'])
  if c in ('N0','NMULTI','DER2'):
   assert val==prior[f'{g} / {c}'][q];seen+=1
  for hit in val['hits']:
   assert hit['before'] in r['text'] and hit['after'] in r['text']
  out[key]=dict(generator=g,condition=c,question=q,text_sha256=sha256(r['text'].encode()).hexdigest(),**val)
 assert len(out)==90 and seen==30
 return out
def analyze(out):
 def rate(g,c,q):
  v=out[g,c,q]
  return Fraction(len(v['hits']),v['boundaries']) if v['boundaries'] else None
 result={}
 for g in ('Claude','GPT-4.1'):
  sums={};contrasts={};trends={}
  for c in CS:
   vs=[out[g,c,q] for q in QS]
   sums[c]=dict(hits=sum(len(v['hits']) for v in vs),boundaries=sum(v['boundaries'] for v in vs),missing=sum(v['rate'] is None for v in vs))
  for a,b in [('DER2','PER2'),('DER2','CIRC2'),('DER2','NMULTI'),('CIRC2','NMULTI'),('PER2','NMULTI'),('DER3','PER3')]:
   diffs={q:rate(g,a,q)-rate(g,b,q) for q in QS if rate(g,a,q) is not None and rate(g,b,q) is not None}
   m=median(diffs.values()) if diffs else None
   ex=min(diffs,key=lambda q:(abs(diffs[q]-m),q)) if diffs else None
   contrasts[a+'-'+b]=dict(pp={q:round(float(v)*100,3) for q,v in diffs.items()},median_pp=None if m is None else round(float(m)*100,3),win=sum(v>0 for v in diffs.values()),tie=sum(v==0 for v in diffs.values()),loss=sum(v<0 for v in diffs.values()),missing=len(QS)-len(diffs),representative_q=ex)
  for root in ('DER','PER'):
   tr=[tuple(rate(g,root+str(i),q) for i in (1,2,3)) for q in QS]
   valid=[(a,b,c) for a,b,c in tr if None not in (a,b,c)]
   trends[root]=dict(strict=sum(a<b<c for a,b,c in valid),nondecreasing=sum(a<=b<=c for a,b,c in valid),missing=len(QS)-len(valid),rates_pp={q:[round(float(v)*100,3) if v is not None else None for v in seq] for q,seq in zip(QS,tr)})
  result[g]=dict(counts=sums,contrasts=contrasts,trends=trends)
 return result
if __name__=='__main__':
 out=features();summary=analyze(out)
 freeze(D/'features.json',[out[k] for k in sorted(out)])
 freeze(D/'results.json',dict(metric_file_sha256=sha256((R/'scripts/team_transition_echo.py').read_bytes()).hexdigest(),corpus_sha256=sha256((R/'data/team-corpus-20261104/utterances.csv').read_bytes()).hexdigest(),by_source=summary))
 for g,v in summary.items():
  print(g,{k:(d['hits'],d['boundaries']) for k,d in v['counts'].items()})
  for k,d in v['contrasts'].items():print(k,d)
  for k,d in v['trends'].items():print(k,d['strict'],d['nondecreasing'])
