"""Qwen suggests verbatim fact evidence only; masked inputs, no transition labels."""
from pathlib import Path
import json
from team_corpus_local_notes import request
from main_dev_packet import O,R,read,put
def main():
    rows=read(O/'masked.json'); lock=read(R/'config/main-freeze-v01/shared.json')['runtime_lock']
    model=next(x for x in request('tags')['models'] if x['name']==lock['model'])
    assert model['digest']==lock['digest']
    for start in range(0,16,4):
        p=O/f'local-extract-{start//4+1}.json'
        if p.exists():continue
        items=[{k:r[k] for k in ('id','question','text')} for r in rows[start:start+4]]
        task='Extract evidence only from these four synthetic answers for two background facts: living with a sister, and working part time in a bookshop. For each answer return JSON keyed by ID with two short exact substrings (sister and job) or null. Include partial clues, do not infer status or generate dialogue. Input: '+json.dumps(items,ensure_ascii=False)
        payload={'model':model['name'],'prompt':task,'think':False,'format':'json','stream':False,'options':{'seed':20261105,'temperature':0,'num_ctx':4096,'num_predict':450}}
        raw=request('generate',payload);raw.pop('context',None)
        put(p,{'digest':model['digest'],'scope':'evidence suggestions only; one fresh context per batch','request':payload,'raw':raw})
if __name__=='__main__':main()
