"""Read-only source audit and blank human calibration packet; no model/network calls."""
import csv
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/main-comparison-v01'
SHARE=ROOT/'share/main-comparison-v01'
def digest(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads((ROOT/p).read_text())
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def cos(a,b):
    assert len(a)==len(b)==512 and all(math.isfinite(x) for x in a+b)
    return sum(x*y for x,y in zip(a,b))/math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))
def local(v):
    if len(v)<2:return None,None,None
    adj=mean(cos(a,b) for a,b in zip(v,v[1:]))
    allpairs=mean(cos(a,b) for a,b in combinations(v,2))
    return adj,allpairs,adj-allpairs
def main():
    OUT.mkdir(parents=True,exist_ok=True);SHARE.mkdir(parents=True,exist_ok=True)
    result=read('analysis/topic-local-separation-v01/results.json')
    details=read('analysis/topic-local-separation-v01/response-details.json')
    for p,h in result['provenance'].items():
        assert digest((ROOT/p).read_bytes())==h,p
    native=read('analysis/topic-trajectory-v01/native-output.json')
    cache={r['id']:r for r in native['records']}
    inputs=read('analysis/topic-trajectory-v01/native-input.json')
    assert all(digest(r['text'].encode())==r['id'] for r in inputs)
    assert set(cache)=={r['id'] for r in inputs}
    text_by_id={r['id']:r['text'] for r in inputs}
    for ident,record in cache.items():
        for s in record['sentences']:
            b=text_by_id[ident].encode('utf-16-le');start=s['utf16Start']*2
            assert b[start:start+s['utf16Length']*2].decode('utf-16-le')==s['text']
    byid={r['condition_id']:r for r in details['responses']}
    for r in details['responses']:
        vectors=[s['vector'] for s in cache[digest(r['response_text'].encode())]['sentences']]
        values=local(vectors)
        for k,val in zip(['adjacent_cos','all_pairs_cos','adjacency_excess'],values):
            assert (val is None and r[k] is None) or (val is not None and abs(val-r[k])<1e-12)
    wrong=0
    for r in details['insertion_controls']:
        src=byid[r['condition_id']]
        v=[s['vector'] for s in cache[digest(src['response_text'].encode())]['sentences']]
        for kind in ['requested','other']:
            new=v.copy();new.insert(r['insertion_position_zero_based'],cache[digest(r[kind]['text'].encode())]['vector'])
            for k,val in zip(['adjacent_cos','all_pairs_cos','adjacency_excess'],local(new)):
                assert abs(val-r[kind][k])<1e-12
        assert r['requested']['word_count']==r['other']['word_count']
        assert r['requested']['sentence_count']==r['other']['sentence_count']
        wrong+=r['other']['adjacent_cos']>r['requested']['adjacent_cos']
    assert wrong==20
    requests=read('analysis/psyche-thought-v01/requests.json')
    selected=[r for r in requests if r['split']=='development' and r['seed']==20261105]
    assert len(selected)==8
    records=[];sources={}
    for r in selected:
        p=ROOT/'analysis/psyche-thought-v01/responses'/(r['id']+'.json')
        raw=json.loads(p.read_text())
        assert raw['status']=='completed' and raw['raw']['done_reason']=='stop'
        # Match saved request fingerprint to the exact payload using the original convention.
        request_hash=digest(json.dumps(r['payload'],ensure_ascii=False,sort_keys=True).encode())
        assert raw['request_sha256']==request_hash
        text=raw['raw']['message']['content']
        records.append({'source_id':r['id'],'question':r['payload']['messages'][-1]['content'],
            'text':text,'source':'psyche-development','source_path':str(p.relative_to(ROOT))})
        sources[str(p.relative_to(ROOT))]=digest(p.read_bytes())
    team_path=ROOT/'data/team-corpus-20261104/utterances.csv'
    for r in csv.DictReader(team_path.open(encoding='utf-8-sig')):
        if r['q_id']=='Q03' and r['condition'] in ['N0','NMULTI','CIRC2','DER2']:
            records.append({'source_id':':'.join(r[k] for k in ['generator','q_id','condition']),
                'question':r['question'],'text':r['text'],'source':'team-corpus','source_path':str(team_path.relative_to(ROOT))})
    assert len(records)==16
    # IDs/order depend only on source keys, not observed scores or text quality.
    for r in records:r['opaque_key']=digest(('calibration-v01:'+r['source_id']).encode())
    records.sort(key=lambda r:r['opaque_key'])
    for i,r in enumerate(records,1):r['blind_id']=f'B{i:02d}';r['text_sha256']=digest(r['text'].encode())
    public=[{k:r[k] for k in ['blind_id','question','text','text_sha256']} for r in records]
    save(OUT/'blinding-key.json',records);save(OUT/'blinded-inputs.json',public)
    fields=['opening_answer','transition','before_quote','after_quote','transition_reason','within_clause_unintelligible','within_clause_quote','recognized_source','notes']
    blank=[{'blind_id':r['blind_id'],**{k:'' for k in fields}} for r in records]
    annotation_path=SHARE/'annotations.json'
    if not annotation_path.exists():save(annotation_path,blank)
    guide=[
        '# 기존 발화 16개 블라인드 검토',
        '\n목적: 본 실험 전에 평가 규칙을 점검합니다. 진단·TLC 중증도 채점이 아닙니다. 출처나 의도 조건을 맞히려 하지 마세요. 과거에 본 발화는 recognized_source에 표시합니다. 모델과 조건을 가려도 문체·내용으로 알아볼 수 있으므로 완전한 눈가림은 아닙니다.',
        '\n## 답변 방법',
        '\n각 B번호에 아래 항목을 적어 주세요. JSON 대신 이 파일에 메모하거나 채팅으로 B01~B04처럼 나눠 답해도 됩니다.',
        '\n- A 시작 답변: direct / partial / absent / uncertain. 질문에 답하는 방향으로 시작하는지 봅니다.',
        '- B 절·문장 사이 연결: yes / no / uncertain / unassessable. 앞뒤 생각이 각각 이해되지만 연결이 불명확하거나 엉뚱한 연상으로 이동하는 구간이 있는지 봅니다.',
        '- B=yes: 앞 구절과 뒤 구절을 각각 원문 그대로 복사하고, 왜 연결이 불명확한지 한 문장으로 적습니다. 점수에 맞추려고 정상적 시간 순서·인과·예시·상세 설명을 오류로 잡지 않습니다.',
        '- C 절·문장 내부 이해 곤란: yes / no / uncertain. 단어 연결 자체를 이해하기 어려우면 구절을 복사합니다. 이런 경우 B를 무조건 yes로 하지 말고, 앞뒤 의미를 판정할 수 없으면 unassessable로 둡니다.',
        '- 주제가 달라졌다는 사실만으로 B=yes가 아닙니다. 질문 답변성(A), 발화 내부 전환(B), 절 내부 이해(C)를 따로 판단합니다. 확신이 없으면 uncertain을 사용합니다.',
        '- 이 묶음은 기준 조정용이며, 여기서 본 발화는 미사용 검증 자료로 재사용하지 않습니다. 원래 프로필이 숨겨져 있으므로 이 묶음에서는 사실 보존을 평가하지 않습니다.',
        '\n예시 응답 형식: B01: A=partial, B=uncertain, C=no, 메모=두 생각의 연결을 판단하기 어려움. B=yes이면 앞/뒤 원문 구절과 이유를 추가합니다.']
    for r in public:
        guide += ['\n## '+r['blind_id'],'\n질문: '+r['question'],'\n'+r['text'],
                  '\nA= / B= / C= / 앞 구절= / 뒤 구절= / 이유= / C 근거= / 이전에 본 발화인지= / 메모=']
    packet=SHARE/'블라인드16개_검토.md'
    if not packet.exists():packet.write_text('\n'.join(guide)+'\n')
    held=[r for r in requests if r['split']=='holdout']
    assert len(held)==32
    assert not any((ROOT/'analysis/psyche-thought-v01/responses'/(r['id']+'.json')).exists() for r in held)
    sources[str(team_path.relative_to(ROOT))]=digest(team_path.read_bytes())
    for p in ['config/main-comparison-v01.json','config/psyche-thought-v01.json','analysis/psyche-thought-v01/requests.json',
              'scripts/NativeFeatures.swift','scripts/extract_automatic_features.py','scripts/topic_local_separation.py',
              'analysis/topic-local-separation-v01/results.json','analysis/topic-local-separation-v01/response-details.json',
              'analysis/topic-trajectory-v01/native-output.json','analysis/measurement-external-v03/results.json']:
        sources[p]=digest((ROOT/p).read_bytes())
    audit={'source_hashes':sources,'prior_provenance_files_checked':len(result['provenance']),
        'responses_local_formulas_recomputed':48,'matched_insertion_pairs_recomputed':48,'other_topic_increased_adjacent_cos':wrong,
        'external_nonclinical_spearman':read('analysis/measurement-external-v03/results.json')['correlations']['adjacent_cos']['summary_spearman'],
        'human_packet':{'rows':16,'psyche_existing':8,'team_existing':8,'completed_annotations':sum(bool(r.get('transition')) for r in read(str(annotation_path.relative_to(ROOT))))},
        'old_holdout_requests':32,'old_holdout_outputs':0,'new_generation':0,'new_embeddings':0,
        'review_packet_selection':'Existing development first seed, all arms/thought/questions (8) + team Q03 both generators N0/NMULTI/CIRC2/DER2 (8); chosen for calibration coverage, not random performance estimation',
        'metric_decisions':{'question_answer_cos':'actual question centroid vs answer sentence centroid; auxiliary association only',
            'topic_association':'mean maximum sentence-to-fixed-event-prototype cosine, not actual question similarity',
            'adjacent_cos':'secondary descriptor; fails monotonicity under some same-length other-event insertions',
            'adjacency_excess':'arrangement descriptor with reversal invariance; not clinical coherence',
            'raw_lexical_diversity':'composition confound monitor based on INC3 analysis, not target outcome'},
        'interpretation':'Single reviewer annotation calibration pending; not clinical validation or independent agreement.'}
    save(OUT/'evidence-audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ['source_hashes','metric_decisions','review_packet_selection']},ensure_ascii=False))
if __name__=='__main__':main()
