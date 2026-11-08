"""Local Qwen selects short literal snippets for the five-question reading sheet, after ratings are locked."""
from collections import Counter
import json
from team_strength_packet import *
from team_corpus_local_notes import request
def main():
 out=D/'local-snippets.json'; join=load(D/'results.json')['rows'];rated={r['id']:r for r in load(D/'ratings.json')}
 ids=[x['id'] for x in join if x['condition']=='DER3' and x['generator']=='GPT-4.1']
 pairs=[{k:rated[i][k] for k in ('id','before','after')} for i in ids];assert len(ids)==5
 if not out.exists():
  lock=load(R/'config/main-freeze-v01/shared.json')['runtime_lock'];m=next(x for x in request('tags')['models'] if x['name']==lock['model'])
  assert m['digest']==lock['digest'];show=request('show',{'model':m['name']})
  fields=['before','after'];row={'type':'object','properties':{k:{'type':'string'} for k in fields},'required':fields,'additionalProperties':False}
  schema={'type':'object','properties':{i:row for i in ids},'required':ids,'additionalProperties':False}
  prompt='For each field copy one short exact contiguous English phrase capturing its central action or entity from that same field. Preserve punctuation/case; never judge its meaning or create a new utterance. IDs and both fields required.\n'+json.dumps(pairs,ensure_ascii=False)
  req=dict(model=m['name'],prompt=prompt,think=False,stream=False,format=schema,options=dict(temperature=0,seed=20261108,num_ctx=4096,num_predict=420))
  raw=request('generate',req);raw.pop('context',None)
  freeze(out,dict(request=req,raw=raw,digest=m['digest'],quant=m['details']['quantization_level'],template_sha256=digest(show['template']),version=request('version'),context='single fresh context; no condition, model provenance or ratings supplied'))
 data=json.loads(load(out)['raw']['response']);audit=[]
 for r in pairs:
  for k in ('before','after'):
   q=data.get(r['id'],{}).get(k);audit.append(dict(id=r['id'],field=k,snippet=q,exact=bool(q and q in r[k])))
 freeze(D/'local-audit.json',audit);print(sum(r['exact'] for r in audit),'/',len(audit))
if __name__=='__main__':main()
