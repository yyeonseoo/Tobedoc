"""One cached local Qwen documentation extraction, never a clinical judge."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/team-corpus-v01/local-assistance'
def request(route, payload=None):
    req = urllib.request.Request('http://127.0.0.1:11434/api/' + route,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type':'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=180) as res:
        return json.load(res)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'response.json').exists():
        print('Cached documentation notes retained; no call.'); return
    tags = request('tags')
    model = next(m for m in tags['models'] if m['name']=='qwen3:14b')
    show = request('show', {'model':'qwen3:14b'})
    metadata = {'model':model, 'version':request('version'), 'show':show,
                'role':'Documentation extraction only; no utterance generation or ratings',
                'context_policy':'One fresh request with entire description; no prior chat; num_ctx=8192'}
    prompt = ('Extract only explicitly stated facts from the source into JSON keys '
        'generation_differences, missing_metadata, caveats, possible_column_inconsistencies. '
        'Use short Korean strings. Do not infer which question IDs were pilot questions. '
        'Do not rate utterances or generate any patient speech. Source:\n' +
        (ROOT/'data/team-corpus-20261104/generation-description.txt').read_text())
    payload = {'model':'qwen3:14b','prompt':prompt,'think':False,'stream':False,'format':'json',
               'options':{'temperature':0,'seed':20261104,'num_ctx':8192,'num_predict':900}}
    metadata['prompt_sha256']=hashlib.sha256(prompt.encode()).hexdigest()
    for name,obj in [('metadata',metadata),('request',payload)]:
        (OUT/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    response=request('generate',payload)
    (OUT/'response.json').write_text(json.dumps(response,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:response.get(k) for k in ['done_reason','prompt_eval_count','eval_count','response']},ensure_ascii=False))
if __name__=='__main__': main()
