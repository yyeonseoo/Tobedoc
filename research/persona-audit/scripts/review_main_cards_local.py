"""Optional local editorial check; no interview response or rating is requested."""
import hashlib, json
from pathlib import Path
from urllib.request import ProxyHandler, build_opener, Request
R=Path(__file__).resolve().parents[1]
dst=R/'analysis/main-freeze-v01/qwen-editorial.json'
if dst.exists():
    print('Cached editorial review exists.'); raise SystemExit()
local=build_opener(ProxyHandler({}))
def api(endpoint,data=None):
    url='http://127.0.0.1:11434/api/'+endpoint
    return json.load(local.open(Request(url,data=json.dumps(data).encode() if data else None,headers={'Content-Type':'application/json'}),timeout=90))
cases=json.loads((R/'config/main-freeze-v01/cases.json').read_text())[1:]
lock=json.loads((R/'config/main-freeze-v01/shared.json').read_text())['runtime_lock']
model=next(m for m in api('tags')['models'] if m['name']==lock['model'])
assert model['digest']==lock['digest']
prompt='Editorial check only. For each synthetic card, flag a concrete internal contradiction, ambiguity in its scheduled job duties, or awkward wording that could change interpretation. If none, say none. Do not roleplay, generate sample speech, rate symptoms, or rewrite the cards. Return JSON with one brief issue or none for each ID.\n'+json.dumps([{k:c[k] for k in ['id','occupation','history','questions','facts']} for c in cases])
payload={'model':lock['model'],'prompt':prompt,'think':False,'stream':False,'format':'json','options':{'temperature':0,'seed':61,'num_ctx':4096,'num_predict':350}}
result=api('generate',payload);result.pop('context',None)
dst.write_text(json.dumps({'scope':'local editorial check of authored inputs only, not experimental inference; no chat history','digest':model['digest'],'request':payload,'response':result},ensure_ascii=False,indent=2)+'\n')
print(result.get('done_reason'),result['response'][:950])
