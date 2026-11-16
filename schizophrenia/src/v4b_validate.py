"""v4b 가설 검증: GPT-4.1 작성 발화(조건 모르는 Claude 평가)에서 combo = z(elv_mean) + z(end_sim) 확인 -> reports/v4/03_v4b_validation.md.

계획: reports/v4/00_plan.md 'v4b 검증 계획'.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer

from text import CFG, path
from v4_literature import metrics

DATA = os.path.join(os.path.dirname(path('processed')), 'v4b')
REP = os.path.join(path('processed'), '..', '..', 'reports', 'v4')
TARGET = {'DER': 'derailment', 'TAN': 'tangentiality', 'INC': 'incoherence', 'PER': 'perseveration', 'CLANG': 'clanging'}
ORDER = ['N0', 'NMULTI', 'CIRC1', 'CIRC2'] + [f'{t}{l}' for t in ['DER', 'TAN', 'INC'] for l in (1, 2, 3)]


def split(cond):
    t = cond.rstrip('0123')
    return t, float(cond[len(t):]) if cond[len(t):] else 0.0


def check(f, c, sign):
    """계획서 기준: ① 핵심 조건 + N0에서 sign*ρ ≥ 0.4 ② 대조 3조건 평균이 DER2·TAN2 평균보다 예상 방향으로 낮은 쪽."""
    core = f[f.type.isin(['DER', 'TAN', 'INC']) | (f.cond == 'N0')]
    rho = stats.spearmanr(core.level, core[c])[0]
    by = {t: stats.spearmanr(*f[(f.type == t) | (f.cond == 'N0')][['level', c]].T.values)[0] for t in ['DER', 'TAN', 'INC']}
    ref = f[f.cond.isin(['DER2', 'TAN2'])][c].mean()
    spec = all(sign * (f[f.cond == k][c].mean() - ref) < 0 for k in ['NMULTI', 'CIRC1', 'CIRC2'])
    return dict(feature=c, rho=rho, rho_DER=by['DER'], rho_TAN=by['TAN'], rho_INC=by['INC'], spec=spec,
                passed=sign * rho >= 0.4 and spec)


def main():
    d = json.load(open(os.path.join(DATA, 'gen_gpt.json'), encoding='utf-8'))
    it = pd.DataFrame(d['items'])
    it['item_id'] = it.q + '_' + it.cond
    it['question'] = it.q.map(d['questions'])
    it[['type', 'level']] = it.cond.apply(lambda c: pd.Series(split(c)))

    # 1. 생성기 검증 (조건 모르는 Claude 평가)
    key = json.load(open(os.path.join(DATA, 'blind_key.json')))
    j = pd.read_csv(os.path.join(DATA, 'judge_claude.csv'))
    j['item_id'] = j.id.map(key)
    j = j.merge(it[['item_id', 'cond', 'type', 'level']], on='item_id')
    assert len(j) == len(it)
    jt = j[j.type.isin(TARGET)].copy()
    jt['score'] = [r[TARGET[r.type]] for _, r in jt.iterrows()]
    gen_tab = jt.groupby(['type', 'level']).score.mean().unstack()
    jd = j[j.type.isin(TARGET) | (j.cond == 'N0')]
    rho_gen = stats.spearmanr(jd.level, jd.overall_disorganization)[0]
    ctrl = j[j.cond.isin(['N0', 'NMULTI', 'CIRC1', 'CIRC2'])].groupby('cond')[['overall_disorganization', 'circumstantiality']].mean()
    # 생성기 기준: 해당 항목 1<2<3 단계 순서 (같은 값이면 탈락)
    ok = {t: bool(gen_tab.loc[t, 1.0] < gen_tab.loc[t, 2.0] < gen_tab.loc[t, 3.0]) for t in gen_tab.index}

    # 2. 지표
    m = SentenceTransformer(CFG['features']['sbert_model'])
    f = pd.concat([it[['item_id', 'cond', 'type', 'level']], pd.DataFrame([metrics(m, t, q) for t, q in zip(it.text, it.question)])], axis=1)
    z = lambda s: (s - s.mean()) / s.std()
    f['combo'] = z(f.elv_mean) + z(f.end_sim)   # 계획대로 새 발화 전체 기준으로 표준화한 뒤
    f = f[f.type.map(ok).fillna(True)]   # 생성기 기준에서 탈락한 사고장애 종류는 판정에서 뺀다
    res = pd.DataFrame([check(f, c, -1) for c in ['combo', 'elv_mean', 'end_sim']])
    means = f.groupby('cond')[['elv_mean', 'end_sim', 'combo']].mean().reindex([c for c in ORDER if c in set(f.cond)])
    ref = f[f.cond.isin(['DER2', 'TAN2'])].combo.mean()
    ref_lab = '·'.join(c for c in ['DER2', 'TAN2'] if c in set(f.cond))
    # 계획 외 보충: 의도한 단계 대신 블라인드 평가 점수와의 상관 (Elvevåg 2007의 검증 방식). 보속·음향 연상 제외, 생성기 탈락 종류 포함
    g = pd.concat([it[['item_id', 'cond']], pd.DataFrame([metrics(m, t, q) for t, q in zip(it.text, it.question)])], axis=1).merge(j.drop(columns='cond'), on='item_id')
    g['combo'] = z(g.elv_mean) + z(g.end_sim)
    g = g[~g.cond.str.startswith(('PER', 'CLANG'))]
    rat = ['overall_disorganization', 'tangentiality', 'derailment', 'incoherence', 'circumstantiality']
    sup = pd.DataFrame({c: {r: stats.spearmanr(g[c], g[r])[0] for r in rat} for c in ['combo', 'elv_mean', 'end_sim']}).T
    verdict = '재현됨' if bool(res.set_index('feature').loc['combo', 'passed']) else '재현되지 않음'
    md = f"""# v4b 가설 검증: 새 발화(GPT-4.1 작성, 조건 모르는 Claude 평가)

재현: `python src/v4b_generate.py`, (Claude 블라인드 평가 -> `data/v4b/judge_claude.csv`), `python src/v4b_validate.py`. 계획: [00_plan.md](00_plan.md) 'v4b 검증 계획'.

**결론: combo 가설 {verdict}.**

## 1. 생성기 검증 (발화 {len(it)}개 = 질문 8개 × 조건 19개)

해당 항목 점수 평균 (0~3):

{gen_tab.to_markdown(floatfmt='.2f')}

단계 순서(1 < 2 < 3) 통과: {', '.join(f'{t} {"통과" if v else "탈락"}' for t, v in ok.items())}. 의도한 단계와 전체 와해 점수의 Spearman ρ = {rho_gen:.2f}.

정상·대조 조건:

{ctrl.to_markdown(floatfmt='.2f')}

## 2. 가설 판정 (탈선·주제 이탈·지리멸렬 중 통과한 종류 + N0)

통과 = ρ ≤ −0.4 이고 NMULTI·CIRC1·CIRC2 평균이 모두 DER2·TAN2 평균보다 높음.

{res.to_markdown(index=False, floatfmt='.2f')}

조건별 평균 (특이도 기준값 = combo의 {ref_lab} 평균 {ref:.2f}. 주제 이탈이 생성기 기준에서 빠지면 탈선 2단계만이다):

{means.to_markdown(floatfmt='.3f')}

## 3. 계획 외 보충: 블라인드 평가 점수와의 상관

주제 이탈·지리멸렬이 생성기 기준에서 탈락해 판정이 탈선 하나로만 이루어졌다. 그래서 의도한 단계 대신 **조건 모르는 평가자의 점수**와 지표의 Spearman 상관을 봤다(보속·음향 연상 제외 {len(g)}개). Elvevåg(2007)가 기울기를 검증한 방식과 같다. 계획에 없던 분석이다.

{sup.to_markdown(floatfmt='.2f')}

## 4. 해석

- **사전 기준으로는 재현되지 않았다.** GPT가 주제 이탈 1단계를 너무 강하게, 지리멸렬 2·3단계를 모두 최고 수준으로 써서 두 종류가 생성기 기준에서 탈락했다. 남은 탈선만으로는 combo의 ρ가 {res.set_index('feature').loc['combo', 'rho']:.2f}로 기준(−0.4)에 못 미쳤다. 탈선 3단계가 2단계보다 질문에서 더 멀지 않았다.
- **질문 유사도 평균(elv_mean) 단독은 이번에 기준을 통과했다.** 하지만 우회증과의 차이가 매우 작아(CIRC2 {means.loc['CIRC2', 'elv_mean']:.3f}, DER2 {means.loc['DER2', 'elv_mean']:.3f}) 안정적이라고 보기 어렵다.
- **평가 점수 기준으로는 combo가 가장 낫다.** 주제 이탈({sup.loc['combo', 'tangentiality']:.2f}), 탈선({sup.loc['combo', 'derailment']:.2f}), 전체 와해({sup.loc['combo', 'overall_disorganization']:.2f})와 상관이 가장 컸다. 또 우회증과는 반대 방향({sup.loc['combo', 'circumstantiality']:+.2f})이었다. 답변 끝 유사도가 우회증 쪽으로 움직여(+{sup.loc['end_sim', 'circumstantiality']:.2f}) 우회증을 사고장애로 오인하지 않게 해 준다는 02의 아이디어와 맞다.
- **지리멸렬은 어떤 질문 기준 지표로도 잡히지 않는다**(상관 ≈ 0). 문장 수준이 깨지는 것은 질문과의 거리로는 드러나지 않는다. 02처럼 문장 응집성 계열이 필요하다.
- 정리: combo는 "탈선·주제 이탈을 우회증과 구분해 재는 지표"로서 가능성을 보였지만, 사전 기준의 재현은 실패했다. 다음 생성에서는 단계마다 예시 발화를 함께 주어 강도를 보정하고, 시범 생성으로 단계 순서를 먼저 확인한 뒤 본 생성을 해야 한다.
"""
    open(os.path.join(REP, '03_v4b_validation.md'), 'w', encoding='utf8').write(md)
    print(md)


if __name__ == '__main__':
    assert split('DER2') == ('DER', 2.0) and split('NMULTI') == ('NMULTI', 0.0) and split('CIRC1') == ('CIRC', 1.0)
    main()
