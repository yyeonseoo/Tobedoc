"""Offline only. Reconstruct literal requests, verify committed manifest, optionally show one."""
import argparse
from copy import deepcopy
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/'config/main-freeze-v01'
OUT=ROOT/'analysis/main-freeze-v01'
def load(p): return json.loads(p.read_text())
def dig(b): return sha256(b).hexdigest()
def fp(x): return dig(json.dumps(x,sort_keys=True,ensure_ascii=False).encode())
def build():
    shared=load(CFG/'shared.json'); cases=load(CFG/'cases.json')
    template=(CFG/'system-template.txt').read_text()
    assert template.count('{{data.mfc}}')==1
    runtime=shared['runtime_lock']; controls=shared['controls']; seeds=shared['seeds']
    assert [c['id'] for c in cases]==['D-bookshop','H01','H02','H03','H04']
    assert [c['id'] for c in controls]==[f'C{i}' for i in range(6)]
    combos=sorted(product(range(1,3),range(1,3)),key=lambda qs:dig(f'expert-v1|Q{qs[0]}|S{qs[1]}'.encode()))
    selected=dict(zip([c['id'] for c in cases[1:]],combos,strict=True))
    rows=[]; payloads={}
    for c in cases:
        base=deepcopy(shared['base_case'])
        if c['split']=='confirmation':
            base['MFC-Profile']['Identifying data'].update(Age=c['age'],Sex=c['sex'],Occupation=c['occupation'])
            base['MFC-Profile']['Developmental/social history']['Work history']=c['work_history']
            base['MFC-History']=c['history']
            base['MFC-Behavior']['General appearance/attitude/behavior']=c['appearance']
        for scope in ['Q1','Q2','guard']:
            facts=c['facts'][scope]
            assert len(facts)==len(set(facts)) and all(f in base['MFC-History'] for f in facts)
            if c['split']=='confirmation': assert len(facts)==(2 if scope=='guard' else 3)
        for qi,si in product(range(1,3),range(1,3)):
            normalized=set()
            for arm in controls:
                cond=arm['id'];case=deepcopy(base);beh=case['MFC-Behavior']
                beh['Verbal productivity']='Normal'
                if cond in ('C1','C2','C3','C4'):beh['Thought process']=arm['thought']
                else:beh.pop('Thought process',None)
                case['Experimental output constraint']=shared['length_instruction']
                if cond in ('C3','C4','C5'):
                    case['Experimental thought-process instruction']=arm['instruction'] if cond=='C5' else shared['instructions'][arm['thought']]
                system=template.replace('{{data.mfc}}',json.dumps(case,ensure_ascii=False,indent=2))
                payload={'model':runtime['model'],'stream':False,'think':runtime['think'],'keep_alive':'5m',
                         'messages':[{'role':'system','content':system},{'role':'user','content':c['questions'][qi-1]}],
                         'options':{**runtime['options'],'seed':seeds[si-1]}}
                assert '{{' not in system and sum(len(x['content'].encode()) for x in payload['messages'])+runtime['options']['num_predict']+2048<runtime['options']['num_ctx']
                # Compare the actual complete requests with only the two manipulation fields removed.
                common=deepcopy(case);common['MFC-Behavior'].pop('Thought process',None);common.pop('Experimental thought-process instruction',None)
                norm=deepcopy(payload);norm['messages'][0]['content']=template.replace('{{data.mfc}}',json.dumps(common,ensure_ascii=False,indent=2))
                normalized.add(fp(norm))
                key=f"{c['id']}-Q{qi}-S{si}-{cond}"
                rows.append(dict(id=key,case=c['id'],split=c['split'],condition=cond,question=qi,seed=seeds[si-1],request_sha256=fp(payload),
                                 expert_plan=selected.get(c['id'])==(qi,si)))
                payloads[key]=payload
            assert len(normalized)==1 and len(set(r['request_sha256'] for r in rows[-6:]))==6
    assert len(payloads)==len({r['request_sha256'] for r in rows})==120 and sum(r['expert_plan'] for r in rows)==24
    return rows,payloads

def legacy(rows):
    folder=ROOT/'analysis/psyche-thought-v01'; attest={}
    for r in rows:
        if r['split']!='development' or r['condition'] in ('C0','C5'):continue
        arm,thought={'C1':('hub_category','normal'),'C2':('hub_category','derailment'),'C3':('source_operationalized','normal'),'C4':('source_operationalized','derailment')}[r['condition']]
        q=['morning','work'][r['question']-1]
        ident=f"D-bookshop-{arm}-{thought}-{q}-{r['seed']}"
        f=folder/'responses'/f'{ident}.json'; item=load(f)
        assert item['id']==ident and item['status']=='completed' and item['request_sha256']==r['request_sha256']
        attest[r['id']]={'previous_id':ident,'response_file_sha256':dig(f.read_bytes())}
    assert len(attest)==16
    return attest

def frozen(path,obj,write):
    if path.exists():
        if load(path)!=obj: raise ValueError(f'Frozen artifact mismatch: {path}')
    elif write: path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    else: raise FileNotFoundError(path)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-lock',action='store_true');parser.add_argument('--check-local-archive',action='store_true')
    parser.add_argument('--show',metavar='REQUEST_ID');args=parser.parse_args()
    rows,payloads=build()
    files=[*sorted(CFG.glob('*')),ROOT/'share/main-comparison-v01/AI판독기준_v1.md',ROOT/'config/main-comparison-v01.json']
    attest=legacy(rows) if args.check_local_archive else None
    frozen(OUT/'manifest.json',{'input_sha256':{str(f.relative_to(ROOT)):dig(f.read_bytes()) for f in files},'rows':rows},args.write_lock)
    if attest is not None:frozen(OUT/'development-reuse.json',attest,args.write_lock)
    if args.show:print(json.dumps(payloads[args.show],ensure_ascii=False,indent=2))
    else:print('120 request hashes verified; 20 matched six-condition blocks; 24 consultation slots.')
