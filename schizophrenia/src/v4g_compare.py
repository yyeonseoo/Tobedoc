"""v4g: 생성 모델 효과와 인물·질문 효과 분리 -> reports/v4/08_generator.md. 계획: 00_plan.md 'v4g 계획'.

세 세트: CA = v4 (Claude, 인물 A, Q01-06), GA = v4g (GPT, 인물 A, Q01-06), GB = v4b (GPT, 인물 B, Q11-18).
평가는 모두 GPT-4.1 블라인드 평가. 지표 계산은 v4c_robust.featurize 그대로.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer

from text import CFG, path
from v4c_robust import RATE, cohen, featurize, load_v4b, rho_table
from v4_judge import load_items, rate_items

DATA = os.path.dirname(path('processed'))
REP = os.path.join(DATA, '..', 'reports', 'v4')
CALM = ['N0', 'NMULTI', 'CIRC1', 'CIRC2']
FOCUS = ['overall_disorganization', 'tangentiality', 'derailment', 'incoherence', 'perseveration', 'clanging']


def load_v4g():
    d = json.load(open(os.path.join(DATA, 'v4g', 'gen_gpt.json'), encoding='utf-8'))
    it = pd.DataFrame(d['items'])
    return it.assign(item_id=it.q + '_' + it.cond, question=it.q.map(d['questions']))


def gen_check(f):
    """의도한 단계와 전체 와해 ρ, 종류별 해당 항목 평균(단계 1~3)."""
    own = dict(DER='derailment', TAN='tangentiality', INC='incoherence', PER='perseveration', CLANG='clanging')
    d = f[f.cond.str[:-1].isin(own) | (f.cond == 'N0')].copy()
    d['level'] = np.where(d.cond == 'N0', 0, d.cond.str[-1].map(lambda x: int(x) if x.isdigit() else 0))
    rho = stats.spearmanr(d.level, d.overall_disorganization)[0]
    t = pd.DataFrame({k: d[d.cond.str.startswith(k) & (d.cond != 'N0')].groupby('level')[v].mean() for k, v in own.items()}).T
    t['순서 통과'] = (t[1] < t[2]) & (t[2] < t[3])
    return rho, t


def main():
    m = SentenceTransformer(CFG['features']['sbert_model'])
    ca, gb, ga = load_items(), load_v4b(), load_v4g()
    r = {'CA': pd.read_csv(os.path.join(DATA, 'v4', 'judge.csv')).drop(columns='model'),
         'GB': rate_items(gb, os.path.join(DATA, 'v4b', 'judge_gpt.csv')),
         'GA': rate_items(ga, os.path.join(DATA, 'v4g', 'judge_gpt.csv'))}
    F = {k: featurize(m, it).merge(r[k], on='item_id') for k, it in [('CA', ca), ('GB', gb), ('GA', ga)]}
    feats = [c for c in F['CA'].columns if c not in ['item_id', 'cond'] + RATE]
    calm = {k: f[f.cond.isin(CALM)] for k, f in F.items()}
    dis = lambda f, c: cohen(f[f.overall_disorganization >= 2][c], f[f.overall_disorganization == 0][c])

    gap = pd.DataFrame({c: dict(생성모델_d=cohen(calm['GA'][c], calm['CA'][c]), 인물질문_d=cohen(calm['GA'][c], calm['GB'][c]),
                                예전_세트차이_d=cohen(calm['GB'][c], calm['CA'][c]), 사고장애_d_GA=dis(F['GA'], c)) for c in feats}).T
    gap['더 큰 쪽'] = np.where(gap.생성모델_d.abs() > gap.인물질문_d.abs(), '생성 모델', '인물, 질문')
    n_gen = int((gap['더 큰 쪽'] == '생성 모델').sum())
    over = [c for c in feats if abs(gap.loc[c, '생성모델_d']) > abs(gap.loc[c, '사고장애_d_GA'])]

    # 같은 인물·질문 두 세트(CA, GA)에서 모두 작동하는 지표
    A, G = rho_table(F['CA'], feats), rho_table(F['GA'], feats)
    robust = {k: [c for c in feats if np.sign(A.loc[c, k]) == np.sign(G.loc[c, k]) and min(abs(A.loc[c, k]), abs(G.loc[c, k])) >= 0.3]
              for k in FOCUS}
    rho, chk = gen_check(F['GA'])
    key = ['elv_mean', 'anchor_mean', 'end_sim', 'stop_ratio', 'mean_wlen', 'comma_rate', 'coh1_mean', 'coh2_mean', 'drift_slope', 'semantic_drift_slope']
    fmt = lambda d, idx=True: d.to_markdown(index=idx, floatfmt='.2f')
    md = f"""# v4g: 생성 모델 효과와 인물, 질문 효과 분리

재현: `python src/v4g_generate.py`, `python src/v4g_compare.py`. 계획: [00_plan.md](00_plan.md) 'v4g 계획'.

04에서 두 세트(Claude 작성 v4, GPT 작성 v4b)의 차이를 생성 모델 효과로 보고했지만, 두 세트는 인물과 질문도 달랐다. v4와 같은 인물(34세 남성)과 같은 질문 6개로 GPT-4.1이 19개 조건을 다시 써서({len(F['GA'])}개) 둘을 갈랐다. 평가는 세 세트 모두 GPT-4.1 블라인드 평가다.

| 세트 | 작성 | 인물 | 질문 | 개수 |
| --- | --- | --- | --- | --- |
| CA (v4) | Claude | 34세 남성 | Q01-06 | {len(F['CA'])} |
| GA (v4g, 새로 생성) | GPT-4.1 | 34세 남성 | Q01-06 | {len(F['GA'])} |
| GB (v4b) | GPT-4.1 | 52세 여성 | Q11-18 | {len(F['GB'])} |

## 1. 생성 검증 (GA)

의도한 단계와 전체 와해 점수의 Spearman ρ = {rho:.2f}. 해당 항목 평균:

{fmt(chk)}

## 2. 사고장애 없는 발화에서의 차이 (Cohen's d)

생성모델_d = GA − CA (인물, 질문 같음). 인물질문_d = GA − GB (생성 모델 같음). 예전_세트차이_d = GB − CA (04에서 생성 모델 효과로 보고한 값). 사고장애_d_GA = GA 안에서 전체 와해 2 이상 대 0.

{fmt(gap.loc[key])}

전체 지표 {len(feats)}개 중 생성 모델 차이가 인물, 질문 차이보다 큰 지표: {n_gen}개.

## 3. 같은 인물, 같은 질문의 두 세트(CA, GA)에서 모두 작동하는 지표

두 세트에서 방향이 같고 둘 다 ρ 절댓값 0.3 이상:

{chr(10).join(f'- {k}: {", ".join(v) if v else "없음"}' for k, v in robust.items())}

## 4. 해석

- **생성 모델 효과는 실제로 있다.** 인물과 질문을 맞춰도 GPT 답변은 Claude 답변보다 질문에 붙어서 답하고(elv_mean d = {gap.loc['elv_mean', '생성모델_d']:.2f}), 기능어와 쉼표가 많고 단어가 짧았다(stop_ratio {gap.loc['stop_ratio', '생성모델_d']:.2f}, mean_wlen {gap.loc['mean_wlen', '생성모델_d']:.2f}). 이 지표들에서는 인물과 질문의 차이보다 생성 모델 차이가 컸다.
- **하지만 04의 "생성 모델 효과가 사고장애 효과보다 크다"는 과장이었다.** 같은 인물과 질문으로 만든 GA 안에서는 사고장애 효과(elv_mean d = {gap.loc['elv_mean', '사고장애_d_GA']:.2f})가 생성 모델 효과보다 컸다. 생성 모델 효과가 사고장애 효과보다 큰 지표는 {len(over)}개({', '.join(over)})로, 대부분 원래 사고장애를 잘 따르지 않던 지표다. 04의 비교는 사고장애 단계가 제대로 나뉘지 않은 v4b를 기준으로 해서 사고장애 효과가 작게 잡혔다.
- **인물과 질문을 맞추면 두 생성 모델에서 함께 작동하는 지표가 늘었다.** 특히 주제 이탈은 질문 기준 지표, 답변 끝 유사도, 문체 지표가 모두 두 세트에서 작동했다. 지리멸렬은 여전히 함께 작동하는 지표가 없다.
- GPT-4.1은 같은 지시문에서도 사고장애를 약하게 썼다(ρ = {rho:.2f}, 탈선 3단계 평균 {chk.loc['DER', 3]:.2f}점). 보속은 단계 순서가 나오지 않았다.

## 5. 전체 표

{fmt(gap.sort_values('생성모델_d', key=abs, ascending=False))}
"""
    open(os.path.join(REP, '08_generator.md'), 'w', encoding='utf8').write(md)
    F['GA'].to_csv(os.path.join(DATA, 'v4g', 'features_all.csv'), index=False)
    print(md)


if __name__ == '__main__':
    assert abs(cohen(pd.Series([1., 2]), pd.Series([1., 2]))) < 1e-9
    main()
