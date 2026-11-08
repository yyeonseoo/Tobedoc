"""Prepare qualitative consultation materials from locked readings. No new rating or model call."""
from collections import Counter
from hashlib import sha256
from itertools import combinations, product
import json
import re
from pathlib import Path

R = Path(__file__).resolve().parents[1]
SRC = R/'analysis/main-dev-v01'
OUT = R/'analysis/main-boundary-v01'
SHARE = R/'share/main-boundary-v01'

def read(name): return json.loads((SRC/name).read_text())
def h(s): return sha256(s.encode()).hexdigest()
def emit(path, text):
    if path.exists() and path.read_text()!=text: raise ValueError(f'Changed frozen output: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists(): path.write_text(text)

def build():
    masked={r['id']:r for r in read('masked.json')}
    rows={r['id']:r for r in read('results.json')['rows']}
    old={r['id']:r for r in read('discourse.json')}
    assert set(masked)==set(rows)==set(old) and len(rows)==16
    v=[]
    for i,r in rows.items():
        s=masked[i]; assert h(s['text'])==s['text_sha256']==old[i]['text_sha256'] and r['B']==old[i]['B']
        if re.search(r'\bvoice\b',s['text'],re.I): v.append(i)
    groups=[('yes','no'),('yes','uncertain'),('no','uncertain')]
    cand=[[tuple(sorted((i,j))) for i,j in combinations(sorted(v),2)
           if rows[i]['block']==rows[j]['block'] and {rows[i]['B'],rows[j]['B']}==set(g)] for g in groups]
    choices=[ps for ps in product(*cand) if len({i for p in ps for i in p})==6]
    # Prefer distinct matched blocks, then the lexicographically first pairs; descriptive selection only.
    best=min(choices,key=lambda ps:(-len({rows[i]['block'] for i,j in ps}),ps))
    ids=sorted([i for p in best for i in p],key=lambda i:h('consult-v1/'+masked[i]['text_sha256']))
    aliases={i:f'K{k:02}' for k,i in enumerate(ids,1)}
    audit={'purpose':'posthoc qualitative boundary discussion only; selected using existing labels, not an agreement sample',
           'counts_all':dict(Counter(rows[i]['B'] for i in rows)),
           'counts_voice':dict(Counter(rows[i]['B'] for i in v)), 'counts_without_voice':dict(Counter(rows[i]['B'] for i in rows if i not in v)),
           'voice_ids':sorted(v), 'candidate_pairs':{f'{a}/{b}':sorted(ps) for (a,b),ps in zip(groups,cand,strict=True)},
           'selected_pairs':best,'reviewer_id_lookup':aliases,
           'inputs_sha256':{name:sha256((SRC/name).read_bytes()).hexdigest() for name in ['masked.json','results.json','discourse.json']}}
    emit(OUT/'selection.json',json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    intro="""# 발화 연결 검토용 워크시트 · 개발 사례

각 문항은 질문과 가상 화자의 답변입니다. 각 답변을 따로 읽고, **각각 뜻이 통하는 인접한 두 구절 사이에 내용상 연결이 불분명한 곳이 있는지** 검토해 주세요. 질환·중증도나 화자의 의도는 평가하지 않습니다.

명확한 곳이 있으면 ‘있음’과 인접한 원문 두 구절·이유를 적습니다. 해석 가능한 전환이 모두 설명되면 ‘없음’, 연결된다는 해석과 끊긴다는 해석이 모두 가능하면 ‘불확실’과 두 해석을 남깁니다. 텍스트 자체로 관계를 평가할 수 없으면 ‘판정 불가’로 표시합니다. 질문 답변이 늦거나 소재가 바뀌었다는 사실만으로 ‘있음’으로 판정하지 않습니다. 한 문장 안에서도 인접한 두 절을 검토할 수 있습니다. 처음 읽은 판정은 별도 사본에 기록해 보존한 후 토론합니다.
"""
    result=intro
    for i in ids:
        s=masked[i];result+=f"\n## {aliases[i]}\n\n**질문**: {s['question']}\n\n{s['text']}\n\n판정:　　　앞 구절:　　　뒤 구절:　　　이유 또는 대안 해석:\n"
    emit(SHARE/'검토자용_원문6개.md',result)
    print(audit['counts_voice'],best,sum(map(len,cand)),aliases)
if __name__=='__main__': build()
