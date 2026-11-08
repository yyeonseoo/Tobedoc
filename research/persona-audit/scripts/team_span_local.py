"""Use Qwen only for literal snippets, with no semantic labels; batches are cached."""
import json
from team_corpus_local_notes import request
from team_span_packet import D,R,load,freeze,digest
def main():
 rows=load(D/'masked.json');lock=load(R/'config/main-freeze-v01/shared.json')['runtime_lock']
 batches=[rows[j:j+4] for j in range(0,len(rows),4)]
 if all((D/f'qwen-{i:02}.json').exists() for i,_ in enumerate(batches,1)):return
 m=next(x for x in request('tags')['models'] if x['name']==lock['model'])
 assert m['digest']==lock['digest']
 show=request('show',{'model':m['name']});opts={'temperature':0,'seed':20261108,'num_ctx':4096,'num_predict':540}
 freeze(D/'runtime.json',dict(name=m['name'],digest=m['digest'],quant=m['details']['quantization_level'],template_sha256=digest(show['template']),options=opts,version=request('version'),context='fresh per batch; no past answers or condition labels; no clinical decisions'))
 for i,b in enumerate(batches,1):
  path=D/f'qwen-{i:02}.json'
  if path.exists():continue
  schema={'type':'object','properties':{r['id']:{'type':'object','properties':{k:{'type':'string'} for k in ['left','right']},'required':['left','right'],'additionalProperties':False} for r in b},'required':[r['id'] for r in b],'additionalProperties':False}
  inp=[{k:r[k] for k in ['id','left','right']} for r in b]
  prompt='For each left and right fragment, copy a single short exact contiguous phrase naming a central entity or action from THAT side only. Return an empty string if none. Preserve spelling and case. Do not label relations or symptoms, paraphrase, or complete any speech.\n'+json.dumps(inp,ensure_ascii=False)
  req={'model':m['name'],'prompt':prompt,'stream':False,'think':False,'format':schema,'options':opts}
  result=request('generate',req);result.pop('context',None)
  freeze(path,dict(request=req,response=result))
if __name__=='__main__':main()
