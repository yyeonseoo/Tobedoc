"""Join locked readings to lexical strata and verify Qwen's independent extraction."""
from collections import Counter
from itertools import product
from team_span_packet import *
def main():
 packet,key,frame=build()
 assert packet==load(D/'masked.json');ratings=load(D/'ratings.json')
 assert [r['id'] for r in ratings]==[r['id'] for r in packet]
 joined=[dict(**k,**{z:r[z] for z in ('label','reason')}) for r,k in zip(ratings,key,strict=True)]
 assert all(r['label'] in ('E','L','W','U') for r in joined)
 for g,c,b in product(['Claude','GPT-4.1'],['NMULTI','PER2','DER2'],[False,True]):
  if not any(f['source']==g and f['condition']==c and f['echo']==b for f in frame):
   frame.append(dict(source=g,condition=c,echo=b,eligible=0,questions=0,picked=0))
 counts={};matched={}
 for g,c in product(['Claude','GPT-4.1'],['NMULTI','PER2','DER2']):
  rs=[r for r in joined if (r['g'],r['c'])==(g,c)]
  a,b=({r['q'] for r in rs if r['echo']==v} for v in [False,True])
  matched[g+'/'+c]=sorted(a&b)
  counts[g+'/'+c]={str(v):dict(Counter(r['label'] for r in rs if r['echo']==v)) for v in (False,True)}
 freeze(D/'joined.json',joined)
 freeze(D/'results.json',dict(frame=frame,by_source_condition=counts,matched_questions=matched,by_echo={str(v):dict(Counter(r['label'] for r in joined if r['echo']==v)) for v in [False,True]},unique_utterances=len({r['sha256'] for r in joined})))
 print(load(D/'results.json')['by_echo'],matched,'units',len({r['sha256'] for r in joined}))
def qc():
 packet=load(D/'masked.json');out=[]
 for j in range(0,len(packet),4):
  batch=load(D/f'qwen-{j//4+1:02}.json')
  parsed=json.loads(batch['response']['response'])
  assert batch['response']['done'] and batch['response']['done_reason']=='stop' and set(parsed)=={r['id'] for r in packet[j:j+4]}
  for r in packet[j:j+4]:
   for side in ('left','right'):
    v=parsed[r['id']][side];status='empty' if not v else 'exact' if v in r[side] else 'mismatch'
    out.append(dict(id=r['id'],side=side,extraction=v,status=status))
 freeze(D/'extraction-audit.json',out);print(dict(Counter(r['status'] for r in out)))
if __name__=='__main__':
 import sys
 main()
 if '--qwen-audit' in sys.argv:qc()
