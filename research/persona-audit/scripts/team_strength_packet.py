from team_transition_packet import R,digest,freeze,load
import csv
D=R/'analysis/team-strength-v01'
def build():
 with (R/'data/team-corpus-20261104/utterances.csv').open(encoding='utf-8-sig') as f:r=list(csv.DictReader(f))
 use=sorted((x for x in r if x['q_id']!='Q03' and x['condition'] in ('NMULTI','DER2','DER3')),key=lambda x:digest('strength-final-20261108:'+x['text']))
 assert len(use)==len({(x['q_id'],x['condition'],x['generator']) for x in use})==30
 old={v['text_sha256'] for v in load(R/'analysis/main-comparison-v01/blinded-inputs.json')}
 packet=[];key=[]
 for j,x in enumerate(use,1):
  h=digest(x['text']);assert h not in old
  i=f'M{j:02}';packet.append(dict(id=i,sha256=h,question=x['question'],text=x['text']))
  key.append(dict(id=i,sha256=h,**{k:x[k] for k in ('generator','q_id','condition')}))
 return packet,key
if __name__=='__main__':freeze(D/'masked.json',build()[0])
