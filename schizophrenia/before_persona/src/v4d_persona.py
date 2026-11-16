"""v4d: 공개 '조현병' 페르소나 모델(underscore2/llama3-8b-schizophrenia) 답변에 v4c의 견고한 지표 적용 -> reports/v4/05_persona.md.

답변은 Colab(data/v4/persona_colab.ipynb)에서 생성한 data/v4/persona_answers.json.
- raw: 생성 그대로 / clean: 모델이 지어낸 다음 면담자 턴·지시문 이전의 첫 답변만
- GPT-4.1 블라인드 평가(clean), v4·v4b 정상(N0)·사고장애 2단계(DER2·TAN2) 평균과 비교
"""
import json
import os
import re

import numpy as np
import pandas as pd

from text import content, path
from v4_judge import rate_items
from v4_literature import metrics
from v4c_robust import style

DATA = os.path.dirname(path('processed'))
REP = os.path.join(DATA, '..', 'reports', 'v4')
ROBUST = ['elv_mean', 'end_sim', 'anchor_mean', 'stop_ratio', 'mean_wlen']
CUT = re.compile(r'\n\s*(?:Interviewer:|###|What’s wrong|What\'s wrong)|\n\n')


def clean(text):
    return CUT.split(text, maxsplit=1)[0].strip()


def main():
    from sentence_transformers import SentenceTransformer
    from text import CFG
    from v4_features import sbert_rows
    m = SentenceTransformer(CFG['features']['sbert_model'])
    d = json.load(open(os.path.join(DATA, 'v4', 'persona_answers.json'), encoding='utf-8'))
    p = pd.DataFrame(d['answers'])
    p['question'] = p.q.map(d['questions'])
    p['item_id'] = p.format + '_' + p.q + '_' + p.rep.astype(str)
    rows = []
    for ver in ['raw', 'clean']:
        for r in p.itertuples():
            t = r.text if ver == 'raw' else clean(r.text)
            n = len(t.split())
            row = dict(item_id=r.item_id, format=r.format, version=ver, n_words=n)
            if n >= 20:
                row.update({k: v for k, v in metrics(m, t, r.question).items() if k in ('elv_mean', 'end_sim')})
                row.update({k: v for k, v in style(t).items() if k in ('stop_ratio', 'mean_wlen')})
                if len(content(t)) >= 40:
                    row['anchor_mean'] = sbert_rows(m, [(content(t), content(r.question))]).anchor_mean.iloc[0]
            rows.append(row)
    f = pd.DataFrame(rows)

    # GPT 블라인드 평가 (clean, 20단어 이상)
    out = os.path.join(DATA, 'v4', 'persona_judge.csv')
    todo = p.assign(text=p.text.map(clean))[lambda x: x.text.str.split().str.len() >= 20]
    j = rate_items(todo, out)

    # 기준: v4(Claude 작성)·v4b(GPT 작성)의 정상과 2단계
    ref = []
    for name, sub in [('v4 Claude 작성', 'v4'), ('v4b GPT 작성', 'v4b')]:
        g = pd.read_csv(os.path.join(DATA, sub, 'features_all.csv'))
        # v4b는 주제 이탈이 생성기 기준에서 탈락해 탈선만 쓴다
        hi = [('탈선·이탈 2단계', ['DER2', 'TAN2']), ('탈선·이탈 3단계', ['DER3', 'TAN3'])] if sub == 'v4' else [('탈선 2단계', ['DER2']), ('탈선 3단계', ['DER3'])]
        for lab, conds in [('정상 N0', ['N0']), ('대조 NMULTI·CIRC', ['NMULTI', 'CIRC1', 'CIRC2'])] + hi:
            ref.append(dict(group=f'{name}: {lab}', n=int(g.cond.isin(conds).sum()), n_anchor=int(g.cond.isin(conds).sum()), **g[g.cond.isin(conds)][ROBUST].mean()))
    for (ver, fmt_), g in f.groupby(['version', 'format']):
        ref.append(dict(group=f'페르소나 {ver} ({fmt_} 형식)', n=int(g.elv_mean.notna().sum()), n_anchor=int(g.anchor_mean.notna().sum()), **g[ROBUST].mean()))
    ref = pd.DataFrame(ref)

    # 07의 구간 지표: 미묘한 생성 발화(v4f) 조건별 평균과 비교 (clean, 내용어 15개 이상)
    from v4f_local import Enc, TYPES, gen_items, local_metrics
    enc = Enc(m)
    lf = pd.DataFrame([dict(item_id=r.item_id, format=r.format, **local_metrics(enc, content(clean(r.text)), content(r.question)))
                       for r in p.itertuples() if len(content(clean(r.text))) >= 15]).merge(j[['item_id', 'overall_disorganization']], on='item_id', how='left')
    g4f = gen_items(enc)
    LOC = ['q_mean', 'q_min', 'q_dip', 'jump_max']
    loc = pd.DataFrame([dict(group=lab, n=len(s), 내용어_중앙값=s.n_win.median() + 4, **s[LOC].mean()) for lab, s in [
        ('v4f 정상 N0', g4f[g4f.cond == 'N0']), ('v4f 우회증', g4f[g4f.cond == 'CIRC']), ('v4f 국소 사고장애 5종', g4f[g4f.cond.isin(TYPES)]),
        *[(f'페르소나 clean ({k} 형식)', s) for k, s in lf.groupby('format')],
        ('페르소나 clean, 평가 1점 이상', lf[lf.overall_disorganization > 0])]])

    dist = j.overall_disorganization.value_counts().sort_index()
    md = f"""# v4d: 공개 '조현병' 페르소나 모델 답변에 지표 적용

재현: Colab `data/v4/persona_colab.ipynb` -> `data/v4/persona_answers.json`, `python src/v4d_persona.py`.

모델 `{d['model']}` (기본 `unsloth/llama-3-8b-bnb-4bit` + LoRA). 학습 데이터와 목적이 공개되지 않은 모델이다. 질문 6개 × 형식 2개(지시문, 대화) × 3회 = 답변 {len(p)}개.

- raw: 생성 그대로. 대화 형식은 모델이 면담자 질문을 지어내 대화를 이어 가거나 다른 글(칼럼, 면접 요령)로 넘어간 부분을 포함한다.
- clean: 처음 지어낸 면담자 턴·지시문·빈 줄 이전의 첫 답변만. 20단어 미만은 지표 계산에서 뺐다(질문 기준 거리는 40 내용어 이상만).

## 1. GPT-4.1 블라인드 평가 (clean, {len(j)}개)

전체 와해 점수 분포: {', '.join(f'{int(k)}점 {v}개' for k, v in dist.items())}. 항목별 평균:

{j[[c for c in j.columns if c != 'item_id']].mean().to_frame('평균').to_markdown(floatfmt='.2f')}

## 2. 견고한 지표로 본 위치 (v4c에서 두 생성 모델 모두 전체 와해와 관련된 지표)

방향: 사고장애가 심할수록 elv_mean·end_sim·stop_ratio는 낮고, anchor_mean·mean_wlen은 높다. anchor_mean은 내용어 40개 이상인 답변만 계산되어, 페르소나 행에서는 n_anchor개(가장 긴 답변들)의 평균이다.

{ref.to_markdown(index=False, floatfmt='.3f')}

## 3. 구간 지표(07)로 본 위치 (clean, 내용어 15개 이상 {len(lf)}개)

07에서 쓴 답변 안 구간 지표를 미묘한 생성 발화(v4f) 조건별 평균과 나란히 놓았다. 방향: 사고장애 쪽일수록 q_mean·q_min은 낮고 q_dip·jump_max는 높다.

{loc.to_markdown(index=False, floatfmt='.3f')}

## 4. 해석

- **평가자는 이 모델의 답변에서 사고장애를 거의 찾지 못했다**(전체 와해 0점이 {int(dist.get(0, 0))}/{len(j)}개). 답변은 일반 챗봇의 평범한 답이거나, 다른 질환(알코올 중독, 다발성 경화증 등)의 환자 역할이었다.
- **질문 기준 지표도 같은 판단을 냈다.** 정리본(clean)의 질문 유사도 평균과 답변 끝 유사도는 두 세트의 정상(N0) 수준이거나 그보다 높았다. 사고장애가 없는 답을 사고장애로 잘못 잡지 않았다.
- **원본(raw) 대화 형식에서는 답변 끝 유사도가 사고장애 2단계 수준으로 떨어졌다.** 모델이 면담자 질문을 지어내거나 칼럼 같은 다른 글로 넘어간 부분 때문이다. 질문 기준 지표는 "질문에서 벗어난 내용"을 재므로, 벗어난 이유가 사고장애인지 생성 오류인지는 가리지 못한다. 실제 면담 데이터에서도 면담자 턴 분리 같은 전처리가 결과를 좌우한다는 뜻이다.
- **구간 지표로 봐도 페르소나 답변에서 국소 이탈의 흔적은 보이지 않았다**(3절). 질문과 가장 먼 구간(q_min)도 v4f 정상 답변보다 질문에 가까웠다. 다만 페르소나 답변은 훨씬 짧아 창 수가 적고, 창이 적으면 최솟값은 기계적으로 높아진다. 또 07에서 이 지표들은 생성 발화에서도 정상과 국소 사고장애를 답변 단위로 가르지 못했다. 그래서 이 비교는 참고용이다.
- **문체 지표(불용어 비율, 평균 단어 길이)는 모델마다 기준점이 달랐다.** 대화 형식 답변은 불용어 비율이 두 세트의 어떤 조건보다 높았다. 이런 지표는 출처가 다른 텍스트끼리 절대값으로 비교하면 안 된다.
"""
    open(os.path.join(REP, '05_persona.md'), 'w', encoding='utf8').write(md)
    print(md)


if __name__ == '__main__':
    assert clean('I went out.\nInterviewer: And then?\nPatient: x') == 'I went out.'
    assert clean('Fine.\n\n### Hint:\nx') == 'Fine.'
    main()
