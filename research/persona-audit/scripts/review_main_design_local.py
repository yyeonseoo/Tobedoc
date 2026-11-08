"""One cached local review of annotation instructions, without access to test texts/labels."""
import json
import hashlib
from pathlib import Path
from team_corpus_local_notes import request
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/main-comparison-v01/local-design-review'
def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'response.json').exists():print('Cached review retained; zero new calls.');return
    model=next(m for m in request('tags')['models'] if m['name']=='qwen3:14b')
    show=request('show',{'model':'qwen3:14b'})
    text=(ROOT/'share/main-comparison-v01/본실험_한장설계.md').read_text()
    guide=(ROOT/'share/main-comparison-v01/블라인드16개_검토.md').read_text().split('\n## B01')[0]
    prompt=('Check these research instructions only for ambiguity or missing practical steps. '
        'Do not rate any speech or propose new experiments, metrics, or models. Return JSON with '
        'at most 3 concrete issues, each keys location, issue, minimal_fix. '
        'Also give a short statement whether single-reviewer observations are confused with clinical gold. '
        'No invented paper citations. Korean preferred.\nDESIGN:\n'+text+'\nGUIDE:\n'+guide)
    payload={'model':'qwen3:14b','prompt':prompt,'think':False,'stream':False,'format':'json',
        'options':{'temperature':0,'seed':20261104,'num_ctx':8192,'num_predict':650}}
    save('metadata.json',{'model':model,'show':show,'version':request('version'),
        'scope':'Instruction clarity review only; no response text or hidden condition labels supplied',
        'context_policy':'single fresh request; no prior chat','prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
    save('request.json',payload)
    result=request('generate',payload);result.pop('context',None);save('response.json',result)
    print(json.dumps({k:result.get(k) for k in ['done_reason','prompt_eval_count','eval_count','response']},ensure_ascii=False))
if __name__=='__main__':main()
