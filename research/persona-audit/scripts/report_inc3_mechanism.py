"""Build the narrow follow-up report from cached numerical and local-review outputs."""
import csv
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'analysis/inc3-mechanism-v01'
OUT=ROOT/'share/inc3-mechanism-v01'
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summaries=json.loads((SRC/'condition-summary.json').read_text())
    contrasts=json.loads((SRC/'contrasts.json').read_text())
    idx={(r['generator'],r['condition'],r['lexicon'],r['scope'],r['metric']):r for r in contrasts}
    def d(g,m,c='INC3',lex='extended',scope='full'):return idx[g,c,lex,scope,m]
    def fci(r):return f"{r['mean_delta']:+.4f} [{r['ci95_question_bootstrap'][0]:+.4f}, {r['ci95_question_bootstrap'][1]:+.4f}]"
    audit=json.loads((SRC/'local-review/audit.json').read_text())
    valid_items=[r for r in audit['items'] if 'row_id' in r]
    quotes=[q for r in valid_items for q in r['quotes']]
    invalid=sum(not q['exact_substring'] for q in quotes)
    invalid_length=sum(not q['length_instruction_met'] for q in quotes)
    protocol_errors=[r for r in audit['items'] if 'row_id' not in r]
    truncated=sum(r.get('batch_done_reason')!='stop' for r in valid_items)
    rows=list(csv.DictReader((ROOT/'data/team-corpus-20261104/utterances.csv').open(encoding='utf-8-sig')))
    raw={':'.join(r[k] for k in ['generator','q_id','condition']):r for r in rows}
    report=['# INC3 어휘 다양성 반대 방향의 원인 분해',
        '\n2026-11-04 · 로드맵 1단계 내 두 번째 탐색 · 새 발화 생성 0개',
        '\n**이 자료에서 반대 방향을 만든 주된 수치 성분은 기능어 목록에 속한 단어의 반복이었다.** Claude INC3에서는 소수의 기능어를 거듭 사용해 MATTR50이 낮아졌다. GPT-4.1 INC3에서는 기능어 비중이 크게 낮아져 해당 반복 성분이 줄었고, 그 결과 전체 MATTR50이 높아졌다. “GPT가 더 다양한 내용어를 생성했다”는 단순 해석은 지지되지 않는다. 기능어 목록 외 단어만 비교하면 두 자료 모두 변화가 작고 비슷했다.',
        '\n이것은 고정된 지표의 산술 분해와 삭제 민감도 분석이다. 인과적 생성 메커니즘, 내용어의 의미적 다양성, 문법 파괴 정도 또는 임상적 사고장애를 검증한 결과는 아니다.',
        '\n## 분석 범위와 고정 규칙',
        '\n기존 228개 중 N0, NMULTI, CIRC1, CIRC2, INC1, INC2, INC3, PER3의 96개를 사용했다(2생성 방식 × 6질문 × 8조건). 주 비교는 같은 질문 안의 INC3−N0 6쌍씩이다. 앞 단계 결과를 본 뒤 선택한 사후 설명 분석이며, 이번 분해 결과를 계산하기 전 규칙을 Git commit be91e8a에 고정했다. 별도 검증 표본은 없다.',
        '\n품사 분석기나 LLM 판정을 수치 계산에 쓰지 않았다. 명시적 기능어 목록과 축약형을 합친 목록을 주 분석에, 축약형 추가 전 목록을 민감도 분석에 썼다. 정확한 목록은 config/inc3-mechanism-v01.json에 있다. 이 목록은 대명사·전치사·관사·보조동사뿐 아니라 일부 부사·수량어도 포함하는 조작적 분류다. 동형어·have/do의 본동사 용법·축약형의 문맥별 역할을 구분하지 않으므로 목록 밖 단어를 모두 내용어로 단정하지 않는다.',
        '\n원래의 모든 연속 50단어 창을 그대로 유지하여 다음 항등식으로 분해했다.',
        '\n**MATTR50 = 1 − 기능어 목록 반복 손실 − 목록 외 반복 손실**',
        '\n각 손실 = (해당 범주의 총 토큰 수 − 해당 범주의 서로 다른 단어 수) / 50. 같은 창에서 계산한 뒤 창 전체를 평균한다. 따라서 두 손실의 변화에 마이너스를 붙이면 전체 MATTR 변화와 정확히 일치한다. 수치상 기여도를 특정할 수 있지만, 그 성분과 MATTR의 상관을 독립적인 발견으로 제시하지 않는다.',
        '\n## 주 결과',
        '\n아래 값은 INC3−N0의 질문별 차이를 평균한 값이다. 대괄호는 6개 질문의 모든 6^6 재표집에서 얻은 기술적 95% percentile 구간이다. 손실의 양수는 반복 증가, 즉 MATTR를 낮추는 방향이다.',
        '\n| 생성 방식 | 전체 MATTR50 차이 | 기능어 목록 반복 손실 차이 | 목록 외 반복 손실 차이 |',
        '|---|---:|---:|---:|']
    for g in ['Claude','GPT-4.1']:
        report.append('| '+g+' | '+' | '.join(fci(d(g,m)) for m in ['mattr50','function_repeat_loss','nonlist_repeat_loss'])+' |')
    report+=['\nClaude: −0.2380 = −0.2391 + 0.0011. 감소가 기능어 목록의 반복 손실 증가로 거의 전부 설명된다. 목록 외 반복 손실 변화는 거의 0이다. 기능어 반복 손실 증가는 6/6 질문에서 나타났다.',
        '\nGPT-4.1: +0.0575 = +0.0699 − 0.0124. 기능어 목록의 반복 손실 감소가 전체 다양성 증가를 만들었고 목록 외 반복 손실은 오히려 증가하여 이를 일부 상쇄했다. 기능어 반복 손실 감소는 5/6 질문에서 나타났다. 이는 “내용어 중복이 줄어서 전체 다양성이 올랐다”는 설명과 다르다.',
        '\n## 단어 비중과 삭제 민감도',
        '\n| 생성 방식 | 기능어 목록 비중 N0→INC3 | 비중 차이 | 목록 제거 후 MATTR25 차이 [95% 구간] |',
        '|---|---:|---:|---:|']
    for g in ['Claude','GPT-4.1']:
        ss={r['condition']:r for r in summaries if r['generator']==g and r['lexicon']=='extended' and r['scope']=='full'}
        report.append(f"| {g} | {100*ss['N0']['function_share_mean']:.2f}% → {100*ss['INC3']['function_share_mean']:.2f}% | {100*d(g,'function_share')['mean_delta']:+.2f}%p | {fci(d(g,'nonlist_mattr25'))} |")
    report+=['\nClaude에서도 기능어 비중 자체는 감소했다. 따라서 “기능어를 더 많이 넣어서 다양성이 낮아졌다”는 설명은 부정확하다. 양이 늘어난 것이 아니라 일부 기능어에 반복이 집중된 양상이다. GPT-4.1은 기능어 목록의 비중이 약 56%에서 25%로 줄어 문법적 연결 표현이 적은 단어 나열 쪽으로 이동했다.',
        '\n목록을 제거한 뒤 남은 단어의 연속 25단어 창을 비교하면 두 자료의 변화는 모두 약 +0.009였다. 두 구간 모두 0을 포함한다. 삭제 후 25단어 창과 원래 50단어 창은 서로 다른 측정이므로 “몇 %의 효과를 제거했다”는 비율을 계산하지 않았다. N0와 INC3의 목록 외 MATTR25 평균이 약 .976~.988로 상한 1에 가까워, 작은 차이는 천장효과의 영향을 받을 수 있다. 남은 단어들의 의미가 잘 연결되거나 두 자료의 내용어 다양성이 동등함을 입증하지 않는다.',
        '\n## 규칙과 길이를 바꿔도 남는 결과',
        '\n| 조건 | Claude 기능어 반복 손실 차이 | GPT 기능어 반복 손실 차이 | Claude 목록 외 MATTR25 차이 | GPT 목록 외 MATTR25 차이 |',
        '|---|---:|---:|---:|---:|']
    for label,lex,scope in [('주 목록·전체','extended','full'),('축약형 추가 전 목록·전체','base','full'),('주 목록·첫 100단어','extended','first100')]:
        vals=[d(g,m,lex=lex,scope=scope)['mean_delta'] for m in ['function_repeat_loss','nonlist_mattr25'] for g in ['Claude','GPT-4.1']]
        report.append('| '+label+' | '+' | '.join(f'{v:+.4f}' for v in vals)+' |')
    report+=['\n두 목록과 첫 100단어 모두에서 기능어 목록 반복 손실의 반대 방향이 유지됐다. 다만 두 목록은 대부분 겹치므로 독립적인 품사 태깅 검증은 아니다. 첫 100단어 제한 역시 생성 절차나 위치 효과를 제거하지 않는다.',
        '\n## 대조 조건과 특징 선택성',
        '\n| 생성 방식 | 조건 | 전체 MATTR50 차이 | 기능어 반복 손실 차이 | 목록 외 반복 손실 차이 | 목록 외 MATTR25 차이 |',
        '|---|---|---:|---:|---:|---:|']
    for g in ['Claude','GPT-4.1']:
        for c in ['NMULTI','CIRC1','CIRC2','INC1','INC2','PER3']:
            report.append('| '+g+' | '+c+' | '+' | '.join(f"{d(g,m,c=c)['mean_delta']:+.4f}" for m in ['mattr50','function_repeat_loss','nonlist_repeat_loss','nonlist_mattr25'])+' |')
    report+=['\nClaude의 CIRC2에서도 기능어 반복 손실이 늘고 전체 MATTR50이 감소한다. GPT-4.1의 INC 단계는 완전한 단조 추세가 아니다. 기능어 반복 성분을 새로운 사고장애 점수로 바꾸면 안 된다. 한편 PER3에서는 목록 제거 후에도 어휘 반복에 따른 다양성 감소가 크게 남는다. 따라서 기능어를 무조건 제거하는 것도 검증된 지표 개선안이 아니며, 유형에 따라 무엇을 측정하는지를 분리해야 한다.',
        '\n## 같은 질문의 원문과 단어 빈도',
        '\n앞 단계와 같은 Q01을 고정하여 예시로 사용했다. 결과가 가장 강한 질문을 새로 선택하지 않았다. 빈도는 전체 발화에서 대소문자와 아포스트로피를 정규화한 단어 빈도다.']
    counts=json.loads((SRC/'word-counts.json').read_text())
    for g in ['Claude','GPT-4.1']:
        for c in ['N0','INC3']:
            r=raw[g+':Q01:'+c]
            counts_row=next(x for x in counts if x['generator']==g and x['q_id']=='Q01' and x['condition']==c)
            first=r['text'].split('. ')[0]+'.'
            report.extend([f'\n**{g} · Q01 · {c}**','\n> '+first,
                '\n기능어 목록 상위 빈도: '+', '.join(f'{t} {n}회' for t,n in counts_row['function_top5'])+'.'])
    report+=['\nClaude Q01의 INC3에서는 the 39회·of 19회가 반복된다(N0: the 8회, of는 상위 5개 밖). GPT Q01의 INC3은 “toast blue football kettle wonder sky”처럼 다양한 단어가 이어지지만, 이 예시만으로 의미적 다양성이나 임상적 자연스러움을 판단하지 않는다.',
        '\n## Ollama 사용과 검증',
        f"\n로컬 qwen3:14b에 N0·INC3 24개를 질문별 4개씩, 총 {audit['usage']['calls']}회로 나눠 짧은 언어적 관찰과 원문 근거 구절을 요청했다. 입력 {audit['usage']['prompt_tokens']:,}·출력 {audit['usage']['output_tokens']:,}토큰을 로컬에서 처리했다. 생성기·조건·수치 결과를 숨기고 불투명 ID와 고정 해시 순서를 사용했다. 스타일을 보고 출처를 짐작할 가능성까지 차단한 것은 아니다.",
        f"\n반환된 {len(valid_items)}개 항목의 인용 {len(quotes)}개 중 원문 연속 부분문자열 검사를 통과하지 못한 인용은 {invalid}개, 5~18단어 길이 지시를 벗어난 인용은 {invalid_length}개, ID/파싱 오류는 {len(protocol_errors)}개, 정상 종료가 아닌 항목은 {truncated}개다. 정확한 인용이라는 사실이 해석의 정확성까지 보증하지는 않는다. Qwen은 특징 계산·임상 채점·정답 생성에 사용하지 않았다. 관찰 원문과 인용 검사는 analysis/inc3-mechanism-v01/local-review/audit.json에 보존했고, 설명용 보조 자료로만 취급했다.",
        '\nQwen은 일부 서로 다른 발화에 동일한 “frequent repetition” 문구를 적용했고 Claude Q03·Q04에는 “minimal connecting words”라고 서술했다. 그러나 제시한 원문에는 the/of의 반복이 뚜렷하여, 연결 의미와 기능어 빈도를 구별한 설명으로 충분하지 않았다. 본 보고서는 이 표현을 정답으로 채택하지 않았다. 로컬 모델은 원문 검토용 발췌·초벌 메모에 활용하고, 핵심 주장에는 실제 빈도와 정확한 분해 계산을 사용했다.',
        '\n## 보고서에 추가할 결론과 다음 경계',
        '\n“INC3 조건의 생성 절차별 MATTR 반대 방향을 고정된 단어 목록으로 분해한 결과, Claude에서는 목록 내 단어 반복 증가가 전체 감소를 설명했고 GPT-4.1에서는 목록 내 반복 감소가 전체 증가를 설명했다. 목록 외 단어의 고정 창 다양성 변화는 두 절차에서 작았다. 이 자료에서 전체 어휘 다양성은 지리멸렬의 공통 강도보다 생성된 문장의 어휘 구성에 민감했다.”',
        '\n이번 분석은 기존 결과의 설명력을 높였지만, 단일 인물·6질문·출력당 반복 없음·Claude의 평가 피드백·API/대화 생성 차이와 교락이라는 한계는 그대로다. bootstrap 구간은 질문 사이의 기술적 변동성만 표현한다. 새로운 유의성 검정을 추가하지 않았고 확증적 기전 발견이라고 쓰지 않는다.',
        '\n이번 한정 분석의 완료는 전체 연구 종료를 뜻하지 않는다. 사용자 제공 11월 4일 로드맵에 따라 현재는 1단계 탐색에 해당한다. 다음은 기존 의미 관련성·문장 연결 코드와 실패 대조를 점검하고 본 실험의 주 비교·평가 준거·대조·고정/조작 요소를 한 장으로 확정하는 2단계다. 이후 동일 조건 비교, 평가 가능한 실패가 있을 때의 수정, 미사용 조건 검증, 자문·팀 자료 연결을 진행한다. 논문 작성은 앞으로 약 한 달간 후속 연구를 수행한 뒤다. 새 생성은 지금 실행하지 않는다.',
        '\n## 재현',
        '\n저장소 루트에서 NumPy가 있는 Python으로 아래 두 명령을 실행한다. 로컬 언어모델 호출은 포함되지 않는다.',
        '\n    python3 scripts/analyze_inc3_mechanism.py\n    python3 scripts/report_inc3_mechanism.py\n    python3 scripts/verify_inc3_mechanism.py',
        '\n현재 Mac 검증 실행기는 /Users/ohs/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3이다. 모든 입력·설정·코드 해시는 run-provenance.json에 있으며, 기능어 목록·부분집합·누락값·원문 보존 규칙도 Git에서 추적한다. 임상 판정·새 발화·새 임베딩·패키지 설치·유료 API 호출은 모두 0회다.']
    (OUT/'INC3_어휘구성분해보고서.md').write_text('\n'.join(report)+'\n')
    notes=['# 로컬 Qwen 언어 관찰과 자동 인용 점검','\n임상 평가가 아닌 기계 생성 보조 메모다. 인용 일치는 해석의 타당성을 검증하지 않는다.']
    for item in sorted(valid_items,key=lambda r:r['row_id']):
        notes+=['\n## '+item['row_id'],'\n'+str(item['observation'])]
        for quote in item['quotes']:
            notes+=['\n> '+quote['text'],'\n원문 일치: '+str(quote['exact_substring'])]
    (OUT/'로컬언어관찰.md').write_text('\n'.join(notes)+'\n')
    print('Built mechanism report and evidence-linked local observations.')
if __name__=='__main__':main()
