"""Offline packet; reads metadata to select and mask, never prints the condition key by default."""
from pathlib import Path
from hashlib import sha256
import csv,json
R=Path(__file__).resolve().parents[1];O=R/'analysis/team-transition-v01'
def digest(text):return sha256(text.encode()).hexdigest()
def load(p):return json.loads(p.read_text())
def freeze(p,v):
 s=json.dumps(v,indent=2,ensure_ascii=False)+'\n'
 if p.exists() and p.read_text()!=s:raise ValueError(f'Frozen artifact mismatch: {p}')
 if not p.exists():p.write_text(s)
def build():
 with (R/'data/team-corpus-20261104/utterances.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 assert len(rows)==228
 old={v['text_sha256'] for v in load(R/'analysis/main-comparison-v01/blinded-inputs.json')}
 use=[r for r in rows if r['condition'] in ['DER2','NMULTI','N0'] and r['q_id']!='Q03']
 assert len(use)==len({(r['q_id'],r['generator'],r['condition']) for r in use})==30
 use.sort(key=lambda r:digest('team-transition-20261108:'+r['text']))
 packet=[];key=[]
 for j,r in enumerate(use,1):
  i=f'T{j:02}';h=digest(r['text']);assert h not in old
  packet.append(dict(id=i,question=r['question'],text=r['text'],sha256=h))
  key.append(dict(id=i,**{k:r[k] for k in ['generator','q_id','condition']},sha256=h))
 assert len({r['sha256'] for r in packet})==30
 return packet,key
if __name__=='__main__':freeze(O/'masked.json',build()[0])
