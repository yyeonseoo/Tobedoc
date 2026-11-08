"""Join frozen exploratory annotations to conditions; same-question/seed blocks only."""
from collections import Counter
from itertools import product
import re,json
from main_dev_packet import R,O,read,put,sha
from statistics import mean
from pathlib import Path
rub=read(R/'config/main-freeze-v01/cases.json')[0]
masked={r['id']:r for r in read(O/'masked.json')}
rates={r['id']:r for r in read(O/'discourse.json')};facts=read(O/'facts.json')
cal={x['text_sha256'] for x in read(R/'analysis/main-comparison-v01/blinded-inputs.json')}
rows=[];assert len(masked)==len(rates)==16 and len(facts)==64
links=read(O/"condition-link.json"); assert len({x["id"] for x in links})==len({x["request_id"] for x in links})==16
for link in links:
 i,k=link["id"],link["request_id"];s=masked[i];assert sha(s["text"].encode())==link["text_sha256"]==s["text_sha256"]
 r=rates[i];_,_,q,sd,c=k.split('-');t=s['text']
 assert r['text_sha256']==s['text_sha256'] and r['opening_quote'] in t
 assert r['left'] in t and (not r['right'] or r['left']+' '+r['right'] in t)
 assert r['B'] not in ['yes','uncertain'] or r['right']
 f=[v for v in facts if v['id']==i];assert [v['fact'] for v in f]==rub['facts'][q]+rub['facts']['guard']
 for v in f:assert v['rating']=='absent' or v['quote'] and v['quote'] in t
 words=len(re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|\d+(?:\.\d+)?",t))
 labels=[l for l in ['schizophrenia','derailment','thought process','verbal productivity','MFC','PSYCHE-SP'] if l.lower() in t.lower()]
 rows.append(dict(id=i,condition=c,block=q+'-'+sd,prior_calibration=s['text_sha256'] in cal,A=r['A'],B=r['B'],C=r['C'],words=words,leaks=labels,facts={scope:dict(Counter(v['rating'] for v in f if v['scope']==scope)) for scope in ('target','guard')}))
def counts(rs,key):return dict(Counter(r[key] for r in rs))
conds={}
for c in ('C1','C2','C3','C4'):
 rs=[r for r in rows if r['condition']==c];assert len(rs)==4
 conds[c]={k:counts(rs,k) for k in ('A','B','C')}
 conds[c]['facts']={s:dict(sum((Counter(r['facts'][s]) for r in rs),Counter())) for s in ('target','guard')}
 conds[c]['word_range']=[min(r['words'] for r in rs),max(r['words'] for r in rs)]
blocks=[]
for b in sorted({r['block'] for r in rows}):
 rs=sorted((r for r in rows if r['block']==b),key=lambda r:r['condition'])
 assert len(rs)==4 and [r['condition'] for r in rs]==list(conds)
 possible=product(*[{'yes':[1],'no':[0],'uncertain':[0,1],'unassessable':[0,1]}[r['B']] for r in rs])
 effects=[v[3]-v[2]-v[1]+v[0] for v in possible]
 blocks.append(dict(block=b,ids=[r['id'] for r in rs],B=[r['B'] for r in rs],point=effects[0] if len(effects)==1 else None,bounds=[min(effects),max(effects)]))
complete=[x['point'] for x in blocks if x['point'] is not None]
qc=[]
for j in range(1,5):
 raw=read(O/f'local-extract-{j}.json'); parsed=json.loads(raw['raw']['response']);
 for i,vs in parsed.items():
  for field,v in vs.items():
   if v is not None:qc.append((i,field,v in masked[i]['text']))
out=dict(conditions=conds,blocks=blocks,complete_blocks=len(complete),point=mean(complete) if complete else None,bounds=[mean(b['bounds'][j] for b in blocks) for j in (0,1)],development_cases=1,overlap=sum(r['prior_calibration'] for r in rows),in_word_range=sum(100<=r['words']<=140 for r in rows),label_leaks=sum(bool(r['leaks']) for r in rows),local_evidence={'returned_ids':sum(len(json.loads(read(O/f'local-extract-{j}.json')['raw']['response'])) for j in range(1,5)),'non_null':len(qc),'exact':sum(x[2] for x in qc)},rows=rows)
put(O/'results.json',out)
voice={q:{c:sum(bool(re.search(r'\bvoice\b',masked[r['id']]['text'],re.I)) for r in rows if r['block'].startswith(q) and r['condition']==c) for c in conds} for q in ['Q1','Q2']}
put(O/'voice-posthoc.json',dict(scope='posthoc descriptive only, case-insensitive whole word voice, two replies per cell',counts=voice))
print(json.dumps({k:v for k,v in out.items() if k!='rows'},ensure_ascii=False))
