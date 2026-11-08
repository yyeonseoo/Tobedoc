"""Offline masked packet builder. Outputs only anonymous text and questions."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[1]; O=R/'analysis/main-dev-v01'
def read(p):return json.loads(p.read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def put(p,v):
    data=json.dumps(v,ensure_ascii=False,indent=2)+'\n'
    if p.exists() and p.read_text()!=data:raise ValueError(f'Frozen file differs: {p}')
    p.write_text(data)
def packets():
    cases=read(R/'config/main-freeze-v01/cases.json')[0]
    rows=[]
    for k,x in read(R/'analysis/main-freeze-v01/development-reuse.json').items():
        p=R/'analysis/psyche-thought-v01/responses'/f"{x['previous_id']}.json"
        assert sha(p.read_bytes())==x['response_file_sha256']
        t=read(p)['raw']['message']['content']; h=sha(t.encode())
        qi=int(k.split('-')[-3][1:])
        rows.append((sha(('dev-v1:'+h).encode()),k,dict(question=cases['questions'][qi-1],text=t,text_sha256=h)))
    assert len(set(r[0] for r in rows))==16
    return [(f'R{i:02}',k,v) for i,(_,k,v) in enumerate(sorted(rows),1)]
if __name__=='__main__':put(O/'masked.json',[dict(id=i,**v) for i,_,v in packets()])
