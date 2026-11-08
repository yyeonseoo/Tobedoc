"""Post-hoc exact MATTR decomposition using frozen lists; no network/model calls."""
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
import numpy as np
from analyze_team_corpus import tokens, paired_summary

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/inc3-mechanism-v01'
CFG=ROOT/'config/inc3-mechanism-v01.json'
METRICS=['mattr50','function_repeat_loss','nonlist_repeat_loss','function_share','nonlist_mattr25']
def measure(ts,lexicon):
    windows=[ts[i:i+50] for i in range(len(ts)-49)]
    f=[];c=[];t=[]
    for w in windows:
        fw=[x for x in w if x in lexicon];cw=[x for x in w if x not in lexicon]
        f.append((len(fw)-len(set(fw)))/50)
        c.append((len(cw)-len(set(cw)))/50)
        t.append(len(set(w))/50)
    nonlist=[x for x in ts if x not in lexicon]
    out={'mattr50':statistics.mean(t),'function_repeat_loss':statistics.mean(f),
        'nonlist_repeat_loss':statistics.mean(c),'function_share':sum(x in lexicon for x in ts)/len(ts),
        'nonlist_mattr25':statistics.mean(len(set(nonlist[i:i+25]))/25 for i in range(len(nonlist)-24)) if len(nonlist)>=25 else None,
        'tokens':len(ts),'nonlist_tokens':len(nonlist)}
    assert abs(out['mattr50']+out['function_repeat_loss']+out['nonlist_repeat_loss']-1)<1e-12
    return out
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(CFG.read_text());source=ROOT/cfg['input']
    conditions=['N0']+cfg['controls']+['INC1','INC2','INC3','PER3']
    lexicons={'extended':set((cfg['function_word_base']+' '+cfg['extended_additions']).split()),
              'base':set(cfg['function_word_base'].split())}
    source_rows=[r for r in csv.DictReader(source.open(encoding='utf-8-sig')) if r['condition'] in conditions]
    assert len(source_rows)==96
    rows=[];word_counts=[]
    for r in source_rows:
        ts=tokens(r['text'])
        meta={k:r[k] for k in ['generator','q_id','condition']}
        for name,lex in lexicons.items():
            for scope,seq in [('full',ts),('first100',ts[:100])]:
                rows.append({**meta,'lexicon':name,'scope':scope,**measure(seq,lex)})
        word_counts.append({**meta,'function_top5':Counter(t for t in ts if t in lexicons['extended']).most_common(5),
            'nonlist_top5':Counter(t for t in ts if t not in lexicons['extended']).most_common(5)})
    summary=[];contrasts=[]
    idx={(r['generator'],r['q_id'],r['condition'],r['lexicon'],r['scope']):r for r in rows}
    for g in ['Claude','GPT-4.1']:
        for lex in lexicons:
            for scope in ['full','first100']:
                for cond in conditions:
                    sub=[idx[g,f'Q0{q}',cond,lex,scope] for q in range(1,7)]
                    row={'generator':g,'condition':cond,'lexicon':lex,'scope':scope,'n':6}
                    for m in METRICS:
                        values=[r[m] for r in sub if r[m] is not None]
                        row[m+'_mean']=statistics.mean(values) if values else None
                        row[m+'_n']=len(values)
                        if cond!='N0':
                            qs=[f'Q0{q}' for q in range(1,7) if idx[g,f'Q0{q}',cond,lex,scope][m] is not None and idx[g,f'Q0{q}','N0',lex,scope][m] is not None]
                            delta=[idx[g,q,cond,lex,scope][m]-idx[g,q,'N0',lex,scope][m] for q in qs]
                            contrasts.append({'generator':g,'condition':cond,'lexicon':lex,'scope':scope,'metric':m,
                                'paired_questions':qs,**paired_summary(delta)})
                    summary.append(row)
    with (OUT/'features.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    save('condition-summary.json',summary);save('contrasts.json',contrasts);save('word-counts.json',word_counts)
    save('run-provenance.json',{'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'config_sha256':hashlib.sha256(CFG.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'shared_feature_code_sha256':hashlib.sha256((ROOT/'scripts/analyze_team_corpus.py').read_bytes()).hexdigest(),
        'numpy':np.__version__,'source_rows':96,'primary_pairs':6,'lexicon_sizes':{k:len(v) for k,v in lexicons.items()},
        'new_speech':0,'new_embeddings':0,'clinical_ratings':0})
    compact=[r for r in contrasts if r['condition']=='INC3' and r['lexicon']=='extended']
    print(json.dumps([{k:r[k] for k in ['generator','scope','metric','mean_delta','ci95_question_bootstrap','positive','negative','n_pairs']} for r in compact],indent=2))
if __name__=='__main__':main()
