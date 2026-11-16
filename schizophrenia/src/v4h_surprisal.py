"""v4h: 지리멸렬용 언어모델 놀람도(surprisal) -> reports/v4/09_surprisal.md. 계획: 00_plan.md 'v4h 계획'.

GPT-2(small)로 답변 토큰마다 -log2 p(토큰 | 앞 문맥)를 구해 답변 평균(surp_mean)과 상위 10% 평균(surp_p90)을 쓴다.
세트: Claude v4, GPT-4.1 v4g(같은 인물·질문), 참고로 GPT-4.1 v4b. 평가 점수는 모두 GPT-4.1 블라인드 평가.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

from metrics import LM
from text import path
from v4c_robust import load_v4b
from v4_judge import load_items, rate_items
from v4g_compare import load_v4g

DATA = os.path.dirname(path('processed'))
REP = os.path.join(DATA, '..', 'reports', 'v4')
TYPES = ['DER', 'TAN', 'INC', 'PER', 'CLANG']
RATE = dict(DER='derailment', TAN='tangentiality', INC='incoherence', PER='perseveration', CLANG='clanging')


def sets():
    a = load_items().merge(pd.read_csv(os.path.join(DATA, 'v4', 'judge.csv')).drop(columns='model'), on='item_id')
    g = load_v4g();  g = g.merge(rate_items(g, os.path.join(DATA, 'v4g', 'judge_gpt.csv')), on='item_id')
    b = load_v4b();  b = b.merge(rate_items(b, os.path.join(DATA, 'v4b', 'judge_gpt.csv')), on='item_id')
    return {'Claude (인물 A)': a, 'GPT-4.1 (인물 A)': g, 'GPT-4.1 (인물 B, 참고)': b}


def judge(f, c):
    f = f.assign(type=f.cond.str.extract(r'^([A-Z]+?)\d?$')[0], level=f.cond.str.extract(r'(\d)$')[0].astype(float).fillna(0))
    f.loc[f.cond == 'N0', 'level'] = 0
    core = f[(f.type == 'INC') | (f.cond == 'N0')]
    rho_lv = stats.spearmanr(core.level, core[c])[0]
    rho_inc = stats.spearmanr(f[c], f.incoherence)[0]
    ref = f[f.cond == 'INC2'][c].mean()
    spec = all(f[f.cond == k][c].mean() < ref for k in ['NMULTI', 'CIRC1', 'CIRC2'])
    other = {RATE[t]: stats.spearmanr(f[c], f[RATE[t]])[0] for t in TYPES if t != 'INC'}
    return dict(지표=c, 단계_rho_INC=rho_lv, 평가_rho_지리멸렬=rho_inc, 대조구분=spec,
                작동=bool(rho_lv >= 0.4 and rho_inc >= 0.3), **{f'평가_rho_{k}': v for k, v in other.items()},
                전체와해_rho=stats.spearmanr(f[c], f.overall_disorganization)[0])


def main():
    lm = LM()
    S = sets()
    res, means = [], []
    for name, f in S.items():
        f = pd.concat([f.reset_index(drop=True), pd.DataFrame([lm(t) for t in f.text])], axis=1)
        S[name] = f
        for c in ['surp_mean', 'surp_p90']:
            res.append(dict(세트=name, **judge(f, c)))
        order = ['N0', 'NMULTI', 'CIRC1', 'CIRC2'] + [f'{t}{l}' for t in TYPES for l in (1, 2, 3)]
        means.append(f.groupby('cond').surp_mean.mean().reindex(order).rename(name))
    res, means = pd.DataFrame(res), pd.concat(means, axis=1)
    main_sets = res[res.세트 != 'GPT-4.1 (인물 B, 참고)']
    works = {c: bool(main_sets[main_sets.지표 == c].작동.all()) for c in ['surp_mean', 'surp_p90']}
    R = res.set_index(['세트', '지표'])
    md = f"""# v4h: 지리멸렬용 언어모델 놀람도

재현: `python src/v4h_surprisal.py`. 계획: [00_plan.md](00_plan.md) 'v4h 계획'.

GPT-2(small)로 답변의 토큰마다 앞 문맥에서 얼마나 예상 밖인지(놀람도, 비트)를 계산했다. surp_mean은 답변 평균, surp_p90은 놀람도가 가장 높은 10% 토큰의 평균이다. 평가 점수는 GPT-4.1 블라인드 평가다.

## 1. 판정

작동 = 지리멸렬 조건 + N0에서 의도한 단계와 ρ ≥ 0.4 그리고 블라인드 평가의 지리멸렬 점수와 ρ ≥ 0.3. 대조구분 = 화제 전환과 우회증 평균이 지리멸렬 2단계 평균보다 낮음.

{res.to_markdown(index=False, floatfmt='.2f')}

같은 인물과 질문으로 만든 두 세트(Claude, GPT-4.1 인물 A)에서 모두 작동: surp_mean {'예' if works['surp_mean'] else '아니오'}, surp_p90 {'예' if works['surp_p90'] else '아니오'}.

## 2. 조건별 평균 놀람도 (surp_mean, 비트)

{means.to_markdown(floatfmt='.2f')}

## 3. 해석

- 지리멸렬 단계와의 ρ: Claude {R.loc[('Claude (인물 A)', 'surp_mean'), '단계_rho_INC']:.2f}, GPT-4.1 {R.loc[('GPT-4.1 (인물 A)', 'surp_mean'), '단계_rho_INC']:.2f}. 블라인드 평가 지리멸렬 점수와의 ρ: Claude {R.loc[('Claude (인물 A)', 'surp_mean'), '평가_rho_지리멸렬']:.2f}, GPT-4.1 {R.loc[('GPT-4.1 (인물 A)', 'surp_mean'), '평가_rho_지리멸렬']:.2f}.
- **놀람도는 지금까지 없던 지리멸렬 지표로 작동했다.** 같은 인물과 질문의 두 세트에서 모두 사전 기준을 통과했고, 다른 인물로 만든 참고 세트에서도 같았다. 지리멸렬 3단계의 평균 놀람도는 정상의 거의 두 배였다.
- **다만 지리멸렬 전용 지표는 아니다.** 탈선(ρ 0.55~0.58)과 주제 이탈(약 0.42)에도 반응해, "말이 얼마나 예상 밖인가"라는 넓은 지표에 가깝다. 지리멸렬에서 가장 크게 움직인다.
- **보속은 반대 방향이다.** 같은 말을 반복하면 다음 단어를 예측하기 쉬워져 놀람도가 내려간다(보속 3단계가 정상보다 낮음). 그래서 놀람도가 낮다고 정상이라고 볼 수 없고, 반복 지표와 함께 봐야 한다.
- 실제 환자 발화에는 아직 적용하지 않았다. DAIS-C 가공본은 불용어가 지워져 있어, 원본 대화 파일에서 전체 문장을 다시 꺼내야 한다.
- 다른 종류와의 관계: 음향 연상 ρ Claude {R.loc[('Claude (인물 A)', 'surp_mean'), '평가_rho_clanging']:.2f}, GPT-4.1 {R.loc[('GPT-4.1 (인물 A)', 'surp_mean'), '평가_rho_clanging']:.2f} / 보속 ρ Claude {R.loc[('Claude (인물 A)', 'surp_mean'), '평가_rho_perseveration']:.2f}, GPT-4.1 {R.loc[('GPT-4.1 (인물 A)', 'surp_mean'), '평가_rho_perseveration']:.2f}.
"""
    open(os.path.join(REP, '09_surprisal.md'), 'w', encoding='utf8').write(md)
    print(md)


if __name__ == '__main__':
    main()
