"""Build Korean report, full condition appendix and vector figure from results."""
import hashlib
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'analysis/team-corpus-v01'
OUT=ROOT/'share/team-corpus-v01'
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    result=json.loads((SRC/'results.json').read_text())
    contrasts=json.loads((SRC/'contrasts.json').read_text())
    details=json.loads((SRC/'response-details.json').read_text())
    cfg=json.loads((ROOT/'config/team-corpus-v01.json').read_text())
    summary={(r['generator'],r['condition']):r for r in result['condition_summary']}
    lookup={(r['generator'],r['condition'],r['metric']):r for r in contrasts}
    def value(g,c,m):return summary[g,c][m+'_mean']
    # Static SVG uses paired-difference data directly, with labelled scales and CIs.
    panels=[('PER3 minus N0: repeated trigrams','repeated_trigram_fraction','PER3',100,-2,36,'Difference (percentage points)'),
            ('INC3 minus N0: lexical diversity','mattr50','INC3',1,-.32,.14,'Difference in MATTR50 (0-1 scale)'),
            ('PER3 minus N0: first 100 words','repeated_trigram_fraction_100','PER3',100,-2,36,'Difference (percentage points)')]
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="370" viewBox="0 0 1180 370">',
         '<rect width="1180" height="370" fill="white"/>',
         '<style>text{font-family:Arial,sans-serif;fill:#20262c} .small{font-size:11px} .label{font-size:12px}</style>',
         '<text x="24" y="26" font-size="18" font-weight="bold">Observed condition effects within each generation protocol</text>']
    for j,(title,metric,condition,scale,lo,hi,axis) in enumerate(panels):
        left=24+j*388; x0=left+80;width=276
        fx=lambda v:x0+(v-lo)/(hi-lo)*width
        svg.append(f'<text x="{left}" y="63" font-size="13" font-weight="bold">{html.escape(title)}</text>')
        svg.append(f'<line x1="{fx(0)}" y1="85" x2="{fx(0)}" y2="245" stroke="#aab2ba" stroke-dasharray="4 3"/>')
        ticks=[0,10,20,30] if scale==100 else [-.3,-.2,-.1,0,.1]
        for tick in ticks:
            x=fx(tick)
            svg.append(f'<line x1="{x}" y1="250" x2="{x}" y2="255" stroke="#59636d"/><text x="{x}" y="271" class="small" text-anchor="middle">{tick:g}</text>')
        for i,g in enumerate(['Claude','GPT-4.1']):
            y=115+92*i; obj=lookup[g,condition,metric]; v=obj['mean_delta']*scale
            a,b=[z*scale for z in obj['ci95_question_bootstrap']]
            color=['#155e75','#a04e23'][i]
            svg.append(f'<text x="{left}" y="{y+4}" class="label">{g}</text>')
            svg.append(f'<line x1="{fx(a)}" y1="{y}" x2="{fx(b)}" y2="{y}" stroke="{color}" stroke-width="3"/>')
            svg.append(f'<circle cx="{fx(v)}" cy="{y}" r="5" fill="{color}"/>')
            svg.append(f'<text x="{fx(v)}" y="{y+24}" class="small" text-anchor="middle">{v:+.3f} [{a:.3f}, {b:.3f}]</text>')
        svg.append(f'<text x="{x0+width/2}" y="296" class="label" text-anchor="middle">{html.escape(axis)}</text>')
    svg+=['<text x="24" y="329" class="small">Dots: mean paired difference over Q01-Q06. Bars: exact question-bootstrap percentile 95% interval (6^6 resamples).</text>',
          '<text x="24" y="349" class="small">Source: acquired 228-output corpus, analyzed 2026-11-04. One persona; one output per cell. No patient-level inference.</text>','</svg>']
    (OUT/'feature-contrasts.svg').write_text('\n'.join(svg))
    report=['# 확보한 228개 생성 발화의 특징 비교',
        '\n2026-11-04 · 기존 발화 재분석 1단계 · 새 발화 생성 0개',
        '\n보속 지시에서 반복 구절이 증가하는 방향은 두 생성 자료 모두에서 관찰됐다. 반면 지리멸렬 지시에서 어휘 다양성이 변하는 방향은 서로 반대였다. 따라서 **일부 표면 특징의 재현 가능성과 지표 해석의 생성 방식 의존성**을 함께 보여주는 결과다. 임상적 보속·지리멸렬의 검증이나 모델 우열을 뜻하지 않는다.',
        '\n## 자료 확보와 분석 단위',
        '\nClaude 114개와 GPT-4.1 114개, 총 228개 고유 텍스트를 확인했다. 한 가상 인물 × 질문 6개 × 조건 19개 × 생성 방식 2개의 모든 셀이 있으며 누락·중복 셀은 없다. 원문 CSV와 설명서는 원본 바이트 및 SHA-256을 보존했다. GitHub의 Claude 112개 증강 자료와는 별도 자료다.',
        '\nGPT-4.1은 API·temperature 0.9로 생성했다. Claude는 대화 안에서 작성했으며 시범 질문 2개의 평가 후 나머지 4개에서 1단계를 강화했다. 시범 질문 ID는 사용자가 현재 확인할 수 없다고 답했다. 정확한 Claude 모델 버전, 원래 프롬프트·대화 순서·세션 기록·seed 등은 제공되지 않았다. **생성기와 생성 절차가 교락되어 있으며 Claude 자료는 평가 피드백을 반영한 적응적 자료다.** 1단계를 독립 검증 자료로 취급하지 않고 나머지 단계도 독립 생성 반복으로 가정하지 않는다.',
        '\nCSV의 CIRC1·CIRC2 level은 1·2여서 설명서의 “대조 조건은 0”이라는 일반 설명과 다르다. 원값을 보존하고 조건 코드로 정상/대조군을 분리했다. CIRC는 이 연구에서 대조로 사용한 조건이며 임상적으로 정상이라고 재정의하지 않는다. level은 지시 강도이며 평가 점수가 아니다.',
        '\n## 고정한 분석과 해석 범위',
        '\n스키마와 설명서를 읽은 뒤 특징값을 보기 전에 분석 조건을 Git commit `b2c8b2d`에 기록했다. 사전등록은 아니다. 주 비교는 각 생성 방식 내 같은 질문의 PER3−N0 3-gram 반복률이고, 나머지 조건과 특징은 탐색적이다. 3-gram 반복률은 (전체 연속 3단어 구절 수−서로 다른 구절 수)/전체 구절 수다. 구절은 문장 경계를 가로지를 수 있다. 대소문자를 통일하고 축약형의 아포스트로피를 보존하며 하이픈은 단어 경계로 처리한다.',
        '\n보조 지표는 어휘 토큰 수, 마침표·물음표·느낌표 기준 문장 수, MATTR50(연속 50단어 창의 어휘 다양성 평균), 인접 문장의 단어 집합 Jaccard, 2-gram 반복률이다. 기능어를 제거하지 않았다. Jaccard는 단어 겹침이며 의미적 연결성이나 임상적 coherence 점수가 아니다. 문장 1개인 발화는 인접 점수가 정의되지 않으므로 0으로 대체하지 않고 해당 대응쌍을 제외하고 분모를 보고했다.',
        '\n질문별 차이를 먼저 계산하고, 질문 블록 6개를 복원추출하는 모든 6^6=46,656개 표본의 평균으로 percentile 95% 구간을 계산했다. 이는 이 여섯 질문 사이의 변동성 요약이며, 한 인물에 대한 자료를 환자 집단이나 독립 생성 시행으로 확대하는 구간이 아니다. 생성 배치 간 의존성도 해결하지 않는다. 주 비교 2개의 sign-flip 검정은 64개 부호 조합을 전수 계산했다. 무작위 실험이 아니므로 교환가능성을 가정한 탐색적 수치로만 제시한다.',
        '\n![조건별 대응 차이](feature-contrasts.svg)',
        '\n## 핵심 정량 결과',
        '\n| 생성 방식 | N0 반복률 | PER3 반복률 | 대응 차이, %p [질문 bootstrap 95% 구간] | 증가 질문 | 첫 100단어 차이, %p |',
        '|---|---:|---:|---:|---:|---:|']
    for g in ['Claude','GPT-4.1']:
        p=lookup[g,'PER3','repeated_trigram_fraction'];lo,hi=p['ci95_question_bootstrap']
        report.append(f"| {g} | {100*value(g,'N0','repeated_trigram_fraction'):.2f}% | {100*value(g,'PER3','repeated_trigram_fraction'):.2f}% | {100*p['mean_delta']:+.2f} [{100*lo:.2f}, {100*hi:.2f}] | {p['positive']}/6 | {100*lookup[g,'PER3','repeated_trigram_fraction_100']['mean_delta']:+.2f} |")
    report += ['\n두 자료의 PER3−N0는 모두 6/6 질문에서 양수였고, 질문 하나씩 제외한 평균도 모두 양수였다. 정확 sign-flip p는 각각 0.03125이고, 두 주 비교를 Holm 보정하면 각각 **0.0625**다. 따라서 보정된 5% 유의성을 충족했다고 쓰지 않는다. bootstrap 구간과 검정은 구성·가정이 달라 수치가 일치하지 않을 수 있다.',
        '\nNMULTI·CIRC1·CIRC2의 평균 3-gram 반복률은 '+', '.join(g+' '+ '·'.join(f"{100*value(g,c,'repeated_trigram_fraction'):.2f}%" for c in ['NMULTI','CIRC1','CIRC2']) for g in ['Claude','GPT-4.1'])+'였다. 다른 네 목표 유형(DER/TAN/INC/CLANG)의 각 조건 평균도 모두 1% 미만이었다. 이는 이 자료에서 이 지표가 보속형 문자 반복에 선택적으로 반응했다는 근거다. 정상 집단 특이도나 분류 성능을 추정한 결과는 아니다.',
        '\nPER1<PER2<PER3 반복률의 엄격한 순서가 성립한 질문은 Claude 5/6, GPT-4.1 1/6이었다. 질문 안의 3단계 Spearman을 평균하면 각각 0.917, 0.583이다. 세 점의 기술적 순서 요약이며 임상 중증도 타당도 상관이 아니다. Claude 1단계는 피드백 적응을 거쳤으므로 두 값의 차이를 모델의 본래 능력 차이로 해석하지 않는다.',
        '\n**지리멸렬 조건에서는 어휘 다양성의 방향이 반대였다.**',
        '\n| 생성 방식 | N0 MATTR50 | INC3 MATTR50 | 대응 차이 [질문 bootstrap 95% 구간] | 첫 100단어 차이 |',
        '|---|---:|---:|---:|---:|']
    for g in ['Claude','GPT-4.1']:
        p=lookup[g,'INC3','mattr50'];lo,hi=p['ci95_question_bootstrap']
        report.append(f"| {g} | {value(g,'N0','mattr50'):.3f} | {value(g,'INC3','mattr50'):.3f} | {p['mean_delta']:+.3f} [{lo:.3f}, {hi:.3f}] | {lookup[g,'INC3','mattr50_100']['mean_delta']:+.3f} |")
    report += ['\nClaude INC3의 감소는 6/6 질문, GPT-4.1 INC3의 증가는 5/6 질문에서 관찰됐고 첫 100단어에서는 각각 6/6이었다. 원문에서는 Claude가 “the/of” 같은 기능어를 되풀이하며 문법적 연결을 깨뜨리고 GPT-4.1은 여러 내용어를 나열하는 양상이 보인다. 이는 아래 Q01 예시에 근거한 해석 가설이며 별도로 검증한 언어학적 주석 결과는 아니다. 또한 CIRC2에서도 MATTR50은 Claude −0.051(6/6 감소), GPT-4.1 −0.019(5/6 감소)였다. 어휘 다양성 감소 하나를 사고장애의 공통 방향으로 사용할 수 없다는 구체적 반례다.',
        '\n인접 단어 겹침도 PER3에서 증가했다(대응 차이 Claude +0.183, 6쌍; GPT-4.1 +0.114, 5쌍). 반복되는 단어가 많으면 점수가 올라갈 수 있으므로 높은 겹침을 좋은 연결성으로 읽으면 안 된다. GPT-4.1의 PER3 한 발화는 구두점 기준 1문장이어서 빠졌다. 다른 조건에서도 유효 분모가 2~6으로 달라 모든 분모를 부록에 명시했다. 기존 임베딩 코사인 결과를 재계산한 것은 아니다.',
        '\n## 길이 점검',
        '\n공백 기준 길이는 설명서와 같은 123~178단어, 중앙값 155다. 분석용 어휘 토큰 기준은 123~180개, 중앙값 155이며 150~170개 범위는 168/228개(Claude 87/114, GPT-4.1 81/114)였다. 하이픈 처리 등에 따라 최대치가 달라지므로 원문 불일치로 보지 않는다. PER3는 N0보다 평균 Claude 17.0단어, GPT-4.1 3.7단어 짧았다. 두 자료 모두 첫 100개 어휘 토큰에서도 반복률 증가와 INC3 다양성의 반대 방향이 남았다. 이 분석은 단순 토큰 수 차이만으로 결과가 사라지는지 확인한 것으로, 위치 효과나 생성 절차 교락을 제거하지 않는다.',
        '\n## 원문 예시',
        '\n수치상 극단 사례를 고르지 않기 위해 같은 질문 Q01(“What did you do last weekend?”)의 고정된 조건 예시를 사용했다. 결과 탐색 후 정한 설명용 선정 규칙이며 독립 검증 표본은 아니다. 아래는 원문에서 연속으로 발췌한 문장이다. 전체 원문과 반복 구절 빈도는 `response-details.json` 및 별도 예시 파일에 있다.']
    examples=[]
    for g in ['Claude','GPT-4.1']:
        for c in ['N0','PER3','INC3']:
            r=next(x for x in details if x['generator']==g and x['condition']==c and x['q_id']=='Q01')
            # First period-delimited span is a continuous exact original-text excerpt.
            excerpt=r['text'].split('. ')[0]+('.' if '. ' in r['text'] else '')
            report.extend([f"\n**{g} · Q01 · {c}**",'\n> '+excerpt])
    for g in ['Claude','GPT-4.1']:
        for c in ['N0','NMULTI','CIRC2','PER1','PER2','PER3','INC3','CLANG3']:
            r=next(x for x in details if x['generator']==g and x['condition']==c and x['q_id']=='Q01')
            examples.extend([f"\n## {g} Q01 {c}", '\n'+r['text'],
                '\n반복 3-gram: '+json.dumps(r['top_repeated_trigrams'],ensure_ascii=False)])
    (OUT/'원문예시.md').write_text('# Q01 조건별 원문과 반복 구절\n\n같은 질문의 고정 조건을 발췌한 설명용 자료다.\n'+'\n'.join(examples)+'\n')
    report += ['\n## 보고서와 논문에 사용할 수 있는 주장',
        '\n“한 가상 인물의 228개 통제 합성 발화에서 보속 지시는 두 생성 절차 모두에서 문자 구절 반복 증가와 연관되었지만, 지리멸렬 지시의 어휘 다양성 변화는 절차에 따라 반대 방향이었다. 따라서 동일한 증상 지시를 받은 합성 자료라도 지표가 반응하는 언어적 구현은 다를 수 있으며, 조건별 대조 및 생성 절차 명시가 필요하다.”',
        '\n현재 성과는 기존 페르소나 연구의 길이·말투 혼동 문제를 팀의 유형별 발화 자료에 연결하는 **합성 발화 측정 연구의 결과 절 초안**이다. 실제 환자와의 유사성, 임상 척도 타당도, 모델 간 우열, 새로운 진단 도구의 정확도는 주장하지 않는다. 모델의 효과와 대화/API·피드백의 효과를 분리할 수 없으므로 “독립 모델 재현”보다는 “서로 다른 두 생성 절차에서의 방향 일치”라고 쓴다.',
        '\n설명서의 평가 상관 Claude ρ=.89, GPT-4.1 ρ=.61은 제공된 보고값이며 원평가표 없이 재계산하지 않았다. 기존 41% 국소 탐지 역시 참조 구간·원분석이 없어 미검증으로 남긴다. 생성 원문 미확보라는 장애는 해소됐지만 이 평가·위치 주석의 확보 여부는 별도다.',
        '\n## 이번 단계 종료와 다음 선택',
        '\n이번 단계는 고정한 특징 비교·대조·원문 예시·재현 검증으로 종료한다. 다음에는 새 생성보다 **INC3의 반대 방향이 기능어 반복과 내용어 나열에서 오는지**, 고정된 단어 분류 규칙 하나와 유형별 전체 원문 검토로 확인하는 것이 우선이다. 동일 자료에 대한 사후 메커니즘 분석으로 명시하고 새 임상 점수는 만들지 않는다. 평가표가 도착하면 측정값과 평가 항목을 질문·조건별로 연결할 수 있지만 그 도착을 다음 작업 전체의 관문으로 두지 않는다.',
        '\n## 재현과 계산 자원',
        '\n`config/team-corpus-v01.json`에 분석 조건, `analysis/team-corpus-v01/run-provenance.json`에 입력·설정·코드 해시와 실행 환경, `features.csv`에 발화별 특징, `contrasts.json`에 질문별 차이와 구간, `condition-summary.csv`에 전체 38조건 요약을 기록했다. 원본은 `data/team-corpus-20261104/`에 있다. 아래 명령은 저장소 루트에서 실행한다.',
        '\n```bash\npython3 scripts/analyze_team_corpus.py\npython3 scripts/report_team_corpus.py\npython3 -m unittest discover -s tests -p test_team_corpus.py -v\n```',
        '\n분석은 Python 3와 NumPy가 필요하다. 현재 Mac의 시스템 Python에는 NumPy가 없어 `/Users/ohs/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`로 검증했다. 패키지 설치는 하지 않았다. 전수 재표집으로 난수 seed가 필요 없다. 문장 1개인 사례의 결측 처리 규칙은 첫 실행 검증에서 보완했으며 0으로 대체하지 않는다.',
        '\nOllama qwen3:14b를 설명서 요약에 한 번 사용했다(입력 1,429·출력 180토큰). 모델 digest·양자화·template·설정·원요청·출력은 `local-assistance/`에 보존했다. 짧은 출력은 API/대화 생성 차이를 충분히 열거하지 못했고 열 불일치도 찾아내지 못해 원문 대조로 보완했다. Qwen 출력은 특징값·임상 판정·통계에 사용하지 않았다. 새 환자 발화·임베딩·LLM 평가·유료 API 호출·모델 다운로드는 모두 0회다.',
        '\n## 출처',
        '\n사용자 제공 CSV와 생성 설명서의 원본 사본 및 해시: `data/team-corpus-20261104/manifest.json`. 설명서의 보고값과 이번 재계산값을 구분했다. 배경 팀 저장소: [yyeonseoo/Tobedoc](https://github.com/yyeonseoo/Tobedoc/tree/58477f301a5b850b7fe0f8056f18bf9f01a10ea7). 이 분석은 해당 저장소의 원래 지표 파이프라인을 재현했다고 주장하지 않는다.']
    (OUT/'특징비교보고서.md').write_text('\n'.join(report)+'\n')
    appendix=['# 전체 조건별 특징 평균','\n모든 조건은 6발화. Jaccard 분모만 구두점 문장 수에 따라 달라진다. 반복률은 %이며 MATTR/Jaccard는 0~1이다.',
        '\n| 생성 방식 | 조건 | 단어 | 문장 | MATTR50 | Jaccard (n) | 2-gram % | 3-gram % | 100단어 3-gram % |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['condition_summary']:
        appendix.append(f"| {r['generator']} | {r['condition']} | {r['words_mean']:.1f} | {r['sentences_mean']:.1f} | {r['mattr50_mean']:.3f} | {r['adjacent_jaccard_mean']:.3f} ({r['adjacent_jaccard_n']}) | {100*r['repeated_bigram_fraction_mean']:.2f} | {100*r['repeated_trigram_fraction_mean']:.2f} | {100*r['repeated_trigram_fraction_100_mean']:.2f} |")
    (OUT/'전체조건표.md').write_text('\n'.join(appendix)+'\n')
    print('Built report, full examples, condition appendix and SVG figure.')
if __name__=='__main__':main()
