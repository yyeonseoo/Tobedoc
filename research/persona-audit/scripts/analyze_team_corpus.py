"""Frozen first-pass feature analysis; no model calls, network or new speech.

Run with bundled Python (numpy, matplotlib). Outputs can be regenerated offline.
"""
import csv
import hashlib
import itertools
import json
import platform
import re
import statistics
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/team-corpus-v01'
CFG = ROOT / 'config/team-corpus-v01.json'
WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
GENERATORS = ['Claude','GPT-4.1']
CONDITIONS = ['N0','NMULTI','CIRC1','CIRC2'] + [t+str(l) for t in ['DER','TAN','INC','PER','CLANG'] for l in [1,2,3]]
METRICS = ['words','sentences','mattr50','adjacent_jaccard','repeated_bigram_fraction','repeated_trigram_fraction']
BOOT = np.array(list(itertools.product(range(6), repeat=6)))

def tokens(text):
    return [m.group().lower().replace('’', "'") for m in WORD.finditer(text)]

def split_sentences(text):
    return [tokens(s) for s in re.split(r'[.!?]+(?:\s+|$)',text) if tokens(s)]

def repeated_fraction(ts,n):
    grams = [tuple(ts[i:i+n]) for i in range(len(ts)-n+1)]
    return (len(grams)-len(set(grams)))/len(grams) if grams else None

def features(text):
    ts=tokens(text); ss=split_sentences(text)
    mattr=statistics.mean(len(set(ts[i:i+50]))/50 for i in range(len(ts)-49)) if len(ts)>=50 else None
    js=[len(set(a)&set(b))/len(set(a)|set(b)) for a,b in zip(ss,ss[1:])]
    return {'words':len(ts),'sentences':len(ss),'mattr50':mattr,
            'adjacent_jaccard':statistics.mean(js) if js else None,
            'repeated_bigram_fraction':repeated_fraction(ts,2),
            'repeated_trigram_fraction':repeated_fraction(ts,3)}

def prefix(text,n=100):
    matches=list(WORD.finditer(text))
    return text[:matches[n-1].end()] if len(matches)>=n else None

def ranks(xs):
    ordered=sorted(xs)
    return [statistics.mean(i+1 for i,y in enumerate(ordered) if y==x) for x in xs]

def spearman(xs,ys):
    a=np.array(ranks(xs)); b=np.array(ranks(ys))
    if np.std(a)==0 or np.std(b)==0: return None
    return float(np.corrcoef(a,b)[0,1])

def paired_summary(d):
    x=np.asarray(d,dtype=float)
    if not len(x):
        return {'n_pairs':0,'mean_delta':None,'ci95_question_bootstrap':None,
                'positive':0,'negative':0,'ties':0,'question_deltas':[],
                'leave_one_question_out_means':[]}
    indices=BOOT if len(x)==6 else np.array(list(itertools.product(range(len(x)),repeat=len(x))))
    boot=x[indices].mean(axis=1)
    lo,hi=np.quantile(boot,[.025,.975])
    return {'n_pairs':len(x),'mean_delta':float(x.mean()),'ci95_question_bootstrap':[float(lo),float(hi)],
            'positive':int(sum(x>1e-12)),'negative':int(sum(x < -1e-12)),
            'ties':int(sum(abs(x)<=1e-12)), 'question_deltas':list(map(float,x)),
            'leave_one_question_out_means':[float(np.delete(x,i).mean()) for i in range(len(x))] if len(x)>1 else []}

def signflip_p(d):
    observed=abs(statistics.mean(d))
    null=[abs(statistics.mean(x*s for x,s in zip(d,sign))) for sign in itertools.product([-1,1], repeat=len(d))]
    return sum(x>=observed-1e-12 for x in null)/len(null)

def write_json(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(CFG.read_text()); source=ROOT/cfg['input']
    raw=list(csv.DictReader(source.open(encoding='utf-8-sig')))
    key=lambda r:(r['generator'],r['q_id'],r['condition'])
    expected=set(itertools.product(GENERATORS,[f'Q0{i}' for i in range(1,7)],CONDITIONS))
    assert len(raw)==228 and set(map(key,raw))==expected
    assert len(set(map(key,raw)))==len(raw)
    assert all(r['text'].strip() for r in raw)
    assert len({r['persona'] for r in raw})==1
    for q in range(1,7):
        assert len({r['question'] for r in raw if r['q_id']==f'Q0{q}'})==1
    for r in raw:
        c=r['condition']; expected_type=re.sub('[123]$','',c) if c not in ['N0','NMULTI'] else c
        expected_level=int(c[-1]) if c not in ['N0','NMULTI'] else 0
        assert r['type']==expected_type and int(r['level'])==expected_level
    rows=[]; details=[]
    for r in raw:
        base={k:r[k] for k in ['generator','q_id','condition','type','level']}
        f=features(r['text']);short=prefix(r['text']); assert short is not None
        f100=features(short)
        row={**base,'row_id':f"{r['generator']}:{r['q_id']}:{r['condition']}",**f,
             **{k+'_100':v for k,v in f100.items()}}
        rows.append(row)
        tri=Counter(tuple(tokens(r['text'])[i:i+3]) for i in range(len(tokens(r['text']))-2))
        details.append({**base,'text':r['text'],'question':r['question'],'features':f,
                        'top_repeated_trigrams':[{'phrase':' '.join(t),'count':n} for t,n in tri.most_common(5) if n>1]})
    idx={key(r):r for r in rows}
    summary=[];contrasts=[];trends=[]
    for g in GENERATORS:
        for c in CONDITIONS:
            group=[r for r in rows if r['generator']==g and r['condition']==c]
            s={'generator':g,'condition':c,'n':len(group)}
            for m in METRICS:
                for suffix in ['', '_100']:
                    values=[r[m+suffix] for r in group if r[m+suffix] is not None]
                    s[m+suffix+'_n']=len(values)
                    s[m+suffix+'_mean']=statistics.mean(values) if values else None
                    s[m+suffix+'_min']=min(values) if values else None;s[m+suffix+'_max']=max(values) if values else None
            summary.append(s)
            if c=='N0':continue
            for m in METRICS:
                for suffix in ['', '_100']:
                    col=m+suffix
                    valid_q=[f'Q0{q}' for q in range(1,7) if idx[g,f'Q0{q}',c][col] is not None and idx[g,f'Q0{q}','N0'][col] is not None]
                    d=[idx[g,q,c][col]-idx[g,q,'N0'][col] for q in valid_q]
                    contrasts.append({'generator':g,'condition':c,'reference':'N0','metric':col,'paired_questions':valid_q,**paired_summary(d)})
        for t in cfg['target_types']:
            for m in METRICS:
                arrays=[[idx[g,f'Q0{q}',t+str(l)][m] for l in [1,2,3]] for q in range(1,7)]
                rhos=[spearman([1,2,3],a) if all(v is not None for v in a) else None for a in arrays]
                valid=[v for v in rhos if v is not None]
                trends.append({'generator':g,'type':t,'metric':m,'within_question_spearman':rhos,
                               'mean_rho':statistics.mean(valid) if valid else None,'defined_questions':len(valid),
                               'strictly_increasing_questions':sum(all(v is not None for v in a) and a[0]<a[1]<a[2] for a in arrays)})
    primary=[]
    for g in GENERATORS:
        x=next(c for c in contrasts if c['generator']==g and c['condition']=='PER3' and c['metric']==cfg['primary_feature'])
        primary.append({**x,'p_exact_sign_flip':signflip_p(x['question_deltas'])})
    running=0
    for i,p in enumerate(sorted(primary,key=lambda p:p['p_exact_sign_flip'])):
        running=max(running,min(1,(len(primary)-i)*p['p_exact_sign_flip']))
        p['p_holm_two_primary_tests']=running
    lengths=[r['words'] for r in rows]
    whitespace=[len(r['text'].split()) for r in raw]
    validation={'n_rows':len(raw),'unique_texts':len({r['text'] for r in raw}), 'unique_cells':len(idx),
        'complete_grid':True,'personas':1,'questions':6,'outputs_per_cell':1,
        'words_lexical':{'min':min(lengths),'median':statistics.median(lengths),'max':max(lengths),
            'within_150_170':sum(150<=n<=170 for n in lengths)},
        'words_whitespace':{'min':min(whitespace),'median':statistics.median(whitespace),'max':max(whitespace)},
        'within_150_170_by_generator':{g:sum(150<=r['words']<=170 for r in rows if r['generator']==g) for g in GENERATORS},
        'description_discrepancy':'CIRC1/CIRC2 level=1/2 in CSV despite general statement that controls have level 0',
        'claude_pilot_question_ids':None, 'all_texts_at_least_100_tokens':min(lengths)>=100}
    write_csv('features.csv',rows);write_csv('condition-summary.csv',summary)
    write_json('response-details.json',details);write_json('contrasts.json',contrasts)
    write_json('severity-trends.json',trends);write_json('validation.json',validation)
    write_json('results.json',{'validation':validation,'primary':primary,'condition_summary':summary,
        'limitations':cfg['limitations'],'new_utterances':0,'new_embeddings':0,'llm_ratings':0})
    write_json('run-provenance.json',{'python':platform.python_version(),'numpy':np.__version__,
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'config_sha256':hashlib.sha256(CFG.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'bootstrap':'all 46656 ordered resamples of six questions; no RNG',
        'sign_flip':'all 64 sign assignments; no RNG'})
    print(json.dumps({'validation':validation,'primary':primary},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
