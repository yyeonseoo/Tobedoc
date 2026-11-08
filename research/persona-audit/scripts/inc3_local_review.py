"""Six cached local linguistic-description batches; no synthetic speech or scoring."""
import csv
import hashlib
import json
from pathlib import Path
from team_corpus_local_notes import request

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/inc3-mechanism-v01/local-review'
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'config/inc3-mechanism-v01.json').read_text())
    source=ROOT/cfg['input']
    rows=[r for r in csv.DictReader(source.open(encoding='utf-8-sig')) if r['condition'] in ['N0','INC3']]
    mapping={}
    for r in rows:
        key=':'.join(r[k] for k in ['generator','q_id','condition'])
        oid=hashlib.sha256(('inc3-review-v01:'+key).encode()).hexdigest()[:12]
        mapping[oid]={**r,'row_id':key}
    save('mapping.json',mapping)
    if not (OUT/'metadata.json').exists():
        tags=request('tags')
        model=next(m for m in tags['models'] if m['name']=='qwen3:14b')
        save('metadata.json',{'model':model,'show':request('show',{'model':'qwen3:14b'}),
            'version':request('version'),'context_policy':'Fresh 4-text batch per question; no previous conversation',
            'role':cfg['qwen']['role'],'blinding':cfg['qwen']['blinding']})
    for q in sorted({r['q_id'] for r in rows}):
        filename=q+'-response.json'
        if (OUT/filename).exists():continue
        batch=[{'id':k,'text':v['text']} for k,v in sorted(mapping.items()) if v['q_id']==q]
        prompt=('Describe observable word use, not clinical severity. Do not infer model identity or diagnosis. '
            'For EACH of the 4 texts, give a short neutral observation about repetition of connecting/grammatical '
            'words versus concrete words and whether the text is sentence-like or list-like. '
            'Mention ambiguity when present. Use EXACT contiguous excerpts copied from each text as evidence, '
            'one or two excerpts, each 5-18 words. No ellipses, no paraphrases inside quotes. '
            'Return ONLY JSON: {"items":[{"id":"...","observation":"one brief English sentence",'
            '"quotes":["exact excerpt","exact excerpt"]}]}. No scores, no new patient speech. Texts:\n'
            +json.dumps(batch,ensure_ascii=False))
        settings=cfg['qwen']
        payload={'model':'qwen3:14b','prompt':prompt,'think':False,'stream':False,'format':'json',
            'options':{k:settings[k] for k in ['temperature','seed','num_ctx','num_predict']}}
        save(q+'-request.json',payload)
        response=request('generate',payload)
        # Preserve original response without large model context-token list.
        response.pop('context',None)
        save(filename,response)
        print(q,'done',response.get('done_reason'),'tokens',response.get('eval_count'),flush=True)
    audit=[]; usage={'calls':0,'prompt_tokens':0,'output_tokens':0}
    for file in sorted(OUT.glob('Q*-response.json')):
        response=json.loads(file.read_text());usage['calls']+=1
        usage['prompt_tokens']+=response.get('prompt_eval_count',0)
        usage['output_tokens']+=response.get('eval_count',0)
        try: items=json.loads(response['response'])['items']
        except (KeyError,ValueError):audit.append({'batch':file.stem,'parse_error':True});continue
        expected={k for k,v in mapping.items() if v['q_id']==file.name[:3]}
        seen=[item.get('id') for item in items]
        if len(seen)!=4 or set(seen)!=expected:
            audit.append({'batch':file.stem,'id_coverage_error':True,'returned_ids':seen})
        for item in items:
            key=item.get('id');r=mapping.get(key)
            if not r:continue
            quotes=item.get('quotes',[])
            audit.append({'row_id':r['row_id'],'observation':item.get('observation'),
                'quotes':[{'text':x,'exact_substring':isinstance(x,str) and bool(x) and x in r['text'],
                    'whitespace_words':len(x.split()) if isinstance(x,str) else None,
                    'length_instruction_met':isinstance(x,str) and 5<=len(x.split())<=18} for x in quotes],
                'quote_count_valid':1<=len(quotes)<=2,'batch_done_reason':response.get('done_reason')})
    save('audit.json',{'usage':usage,'items':audit,'clinical_ratings':0})
    print(json.dumps({'usage':usage,'items':len(audit),'invalid_quotes':sum(not q['exact_substring'] for a in audit for q in a.get('quotes',[]))}))
if __name__=='__main__':main()
