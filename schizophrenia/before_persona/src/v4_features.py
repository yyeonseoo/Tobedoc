"""v4 지표 계산과 판정 (reports/v4/00_plan.md) -> reports/v4/01_results.md, figures/.

생성 발화: 팀장님 TF-IDF 피처(원문 앞 100토큰), v3 SBERT 피처(앞 40 내용어), 질문 기준 거리.
DAIS-C: 생성 발화에서 민감도·특이도를 통과한 SBERT 피처만, 같은 사람 안에서 DT 표시 답변과 일반 답변 비교.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer

from text import CFG, chunks, content, path
from v2_features import slope
from v2_notebook_audit import geometry, toks
from v3_features import FEATS, features
from v4_judge import ITEMS, load_items

C = CFG['features']['chunk_words']
N = 40
TFIDF = ['adj_semantic_distance', 'semantic_drift_slope', 'max_semantic_drift']
ANCHOR = ['anchor_first', 'anchor_mean', 'anchor_slope']
DRIFT = TFIDF + ANCHOR + ['drift_slope', 'drift_mean', 'adj_mean', 'late_slope', 'jump_max', 'adj_sd', 'global_coh']
SIGN = {f: 1 for f in DRIFT} | {'global_coh': -1}   # 예상 방향 (계획서)
SPECIAL = {'PER': ['recurrence', 'sem_redund'], 'CLANG': ['clang_rate']}
REP = os.path.join(path('processed'), '..', '..', 'reports', 'v4')
COND_ORDER = ['N0', 'NMULTI', 'CIRC1', 'CIRC2'] + [f'{t}{l}' for t in ['DER', 'TAN', 'INC', 'PER', 'CLANG'] for l in (1, 2, 3)]


def sbert_rows(m, units):
    """units: (words, anchor_words) 목록. 앞 40 내용어 8조각 + 질문 기준 거리."""
    heads = [w[:N] for w, _ in units]
    texts = sorted({c for h in heads for c in chunks(h, C)} | {' '.join(a) for _, a in units if len(a) >= 2})
    emb = dict(zip(texts, m.encode(texts, normalize_embeddings=True, batch_size=256)))
    vocab = sorted({w for h in heads for w in h})
    we = dict(zip(vocab, m.encode(vocab, normalize_embeddings=True, batch_size=512)))
    rows = []
    for h, (_, a) in zip(heads, units):
        E = np.stack([emb[c] for c in chunks(h, C)])
        row = features(E, np.stack([we[w] for w in h]), h)
        if len(a) >= 2:
            da = 1 - E @ emb[' '.join(a)]
            row.update(anchor_first=da[0], anchor_mean=da.mean(), anchor_slope=slope(da))
        rows.append(row)
    return pd.DataFrame(rows)


def generated(m):
    it = load_items()
    it['words'] = it.text.map(content)
    assert it.words.str.len().min() >= N and it.text.map(lambda t: len(toks(t))).min() >= 100
    f = sbert_rows(m, list(zip(it.words, it.question.map(content))))
    t = pd.DataFrame([geometry(' '.join(toks(x))) for x in it.text])
    out = pd.concat([it[['item_id', 'q', 'cond']], f, t], axis=1)
    out['type'] = out.cond.str.extract(r'^([A-Z]+?)(\d?)$')[0]
    out['level'] = out.cond.str.extract(r'(\d)$')[0].astype(float).fillna(0)
    return out


def judge(f):
    """계획서의 판정 1(민감도), 2(특이도). 탈선·주제 이탈·지리멸렬 + N0(0단계)."""
    core = f[f.type.isin(['DER', 'TAN', 'INC']) | (f.cond == 'N0')]
    ctrl = f[f.cond.isin(['NMULTI', 'CIRC1', 'CIRC2'])].groupby('cond')
    ref = f[f.cond.isin(['DER2', 'TAN2'])]
    rows = []
    for c in DRIFT + ['recurrence', 'sem_redund', 'mattr', 'sem_density', 'rare_rate', 'clang_rate']:
        s = SIGN.get(c, 1)
        rho, p = stats.spearmanr(core.level, core[c], nan_policy='omit')
        by_type = {t: stats.spearmanr(*f[(f.type == t) | (f.cond == 'N0')][['level', c]].T.values)[0] for t in ['DER', 'TAN', 'INC']}
        spec = all(s * (ctrl[c].mean()[k] - ref[c].mean()) < 0 for k in ['NMULTI', 'CIRC1', 'CIRC2'])
        sens = c in DRIFT and s * rho >= 0.4
        rows.append(dict(feature=c, family='TF-IDF' if c in TFIDF else 'SBERT', rho=rho, p=p,
                         rho_DER=by_type['DER'], rho_TAN=by_type['TAN'], rho_INC=by_type['INC'],
                         spec=spec, passed=sens and spec))
    return pd.DataFrame(rows)


def special(f):
    rows = []
    for t, cols in SPECIAL.items():
        sub = f[(f.type == t) | (f.cond == 'N0')]
        for c in cols:
            rho, p = stats.spearmanr(sub.level, sub[c])
            rows.append(dict(condition=t, feature=c, rho=rho, p=p))
    return pd.DataFrame(rows)


def daisc(m, passed):
    d = pd.read_parquet(os.path.join(path('processed'), 'daisc_units.parquet'))
    d = d[d.n_words >= N].reset_index(drop=True)
    d['dt40'] = d.in_DT.map(lambda v: bool(np.any(v[:N])))   # 지표를 계산한 앞 40 내용어 안의 DT 표시
    f = pd.concat([d[['speaker_id', 'group', 'dt40']], sbert_rows(m, [(list(w), list(a)) for w, a in zip(d.words, d.anchor_words)])], axis=1)
    both = f.groupby('speaker_id').dt40.transform(lambda s: s.any() and not s.all())
    f = f[both]
    rows = []
    for c in passed:
        g = f.groupby(['speaker_id', 'dt40'])[c].mean().unstack().dropna()
        diff = SIGN.get(c, 1) * (g[True] - g[False])
        k = int((diff > 0).sum())
        rows.append(dict(feature=c, speakers=len(diff), dt_higher=k, sign_p=stats.binomtest(k, len(diff)).pvalue,
                         median_diff=float(diff.median())))
    return pd.DataFrame(rows), f


def heatmap(f, feats, out):
    z = f.groupby('cond')[feats].mean().loc[COND_ORDER]
    z = (z - f[feats].mean()) / f[feats].std()
    z = z * np.array([SIGN.get(c, 1) for c in feats])
    fig, ax = plt.subplots(figsize=(1 + 0.55 * len(feats), 7))
    im = ax.imshow(z.values, cmap='RdBu_r', vmin=-1.5, vmax=1.5, aspect='auto')
    ax.set_xticks(range(len(feats)), feats, rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(len(z)), z.index, fontsize=8)
    for y in [3.5, 6.5, 9.5, 12.5, 15.5]:
        ax.axhline(y, color='white', lw=2)
    fig.colorbar(im, ax=ax, shrink=0.6, label='z (sign-aligned)')
    ax.set_title('Feature by condition (z, expected direction = red)', fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    m = SentenceTransformer(CFG['features']['sbert_model'])
    f = generated(m)
    j = pd.read_csv(os.path.join(os.path.dirname(path('processed')), 'v4', 'judge.csv')).merge(f[['item_id', 'cond', 'type', 'level']], on='item_id')
    target = {'DER': 'derailment', 'TAN': 'tangentiality', 'INC': 'incoherence', 'PER': 'perseveration', 'CLANG': 'clanging'}
    jt = j[j.type.isin(target)].assign(score=lambda x: [r[target[r.type]] for _, r in x.iterrows()])
    jdis = j[j.type.isin(target) | (j.cond == 'N0')]
    res = judge(f)
    sp = special(f)
    passed = [c for c in res[res.passed].feature if c not in TFIDF]
    dres, _ = daisc(m, passed) if passed else (pd.DataFrame(), None)
    # 계획 외 탐색: 통과 지표가 없을 때, 민감도만 통과한 SBERT 지표로 같은 비교
    sens = [r.feature for r in res.itertuples() if r.feature in DRIFT and r.feature not in TFIDF and SIGN[r.feature] * r.rho >= 0.4]
    xres, xf = daisc(m, sens) if sens else (pd.DataFrame(), None)
    R = res.set_index('feature')
    pooled = f[f.cond.isin(['DER2', 'TAN2'])].anchor_mean.mean()
    means = f.groupby('cond')[['semantic_drift_slope', 'max_semantic_drift', 'drift_slope', 'anchor_mean', 'global_coh']].mean().loc[
        ['N0', 'NMULTI', 'CIRC1', 'CIRC2', 'DER1', 'DER2', 'DER3', 'TAN1', 'TAN2', 'TAN3', 'INC1', 'INC2', 'INC3']]
    os.makedirs(os.path.join(REP, 'figures'), exist_ok=True)
    heatmap(f, DRIFT + ['recurrence', 'sem_redund', 'clang_rate'], os.path.join(REP, 'figures', 'fig1_heatmap.png'))
    f.to_csv(os.path.join(os.path.dirname(path('processed')), 'v4', 'features.csv'), index=False)

    gen_tab = jt.groupby(['type', 'level']).score.mean().unstack().round(2)
    ctrl_tab = j[j.cond.isin(['N0', 'NMULTI', 'CIRC1', 'CIRC2'])].groupby('cond')[['overall_disorganization', 'circumstantiality']].mean().round(2)
    rho_gen = stats.spearmanr(jdis.level, jdis.overall_disorganization)[0]
    fmt = lambda d: d.to_markdown(index=False, floatfmt='.2f')
    md = f"""# v4 결과: 단계별 사고장애 발화로 지표 점검

재현: `python src/v4_judge.py --model gpt-4.1`, `python src/v4_features.py`. 계획: [00_plan.md](00_plan.md).

## 1. 생성기 검증 (GPT-4.1 블라인드 평가, 발화 {len(f)}개 = 질문 6개 × 조건 19개)

해당 항목 점수 평균 (0~3):

{gen_tab.to_markdown(floatfmt='.2f')}

정상·대조 조건:

{ctrl_tab.to_markdown(floatfmt='.2f')}

의도한 단계와 전체 와해 점수의 Spearman ρ = {rho_gen:.2f} (사고장애 5종 + N0).

## 2. 지표 판정 (탈선·주제 이탈·지리멸렬 + N0, n = {len(f[f.type.isin(['DER', 'TAN', 'INC']) | (f.cond == 'N0')])})

통과 = 예상 방향 ρ ≥ 0.4 이고 NMULTI·CIRC1·CIRC2가 모두 DER2·TAN2 평균보다 낮음(예상 방향 기준).

{fmt(res)}

TF-IDF 3개는 팀 노트북과 같이 40토큰 창(20토큰씩 이동)을 답변 전체에 적용하고, TF-IDF는 답변 하나의 창들로 학습한다.

보속·음향 연상 전용 지표 (해당 조건 + N0). 음향 연상은 생성기 검증의 단계 순서 기준(1 < 2 < 3)을 통과하지 못했다(2단계와 3단계가 모두 3점). 아래 음향 연상 행은 참고값이다:

{fmt(sp)}

![](figures/fig1_heatmap.png)

## 3. DAIS-C (같은 사람 안 비교, 예비)

통과한 SBERT 피처: {', '.join(passed) if passed else '없음'}. 앞 40 내용어 안에 DT 표시가 있는 답변과 없는 답변을 둘 다 가진 화자만 사용.

{fmt(dres) if len(dres) else '해당 없음 (계획대로 분석하지 않음)'}

**계획 외 탐색**: 민감도만 통과한 지표({', '.join(sens)})로 같은 비교. 조건을 만족하는 화자 {xf.speaker_id.nunique() if xf is not None else 0}명(DT 답변 {int(xf.dt40.sum()) if xf is not None else 0}개)뿐이라 판단 근거가 되지 못한다.

{fmt(xres) if len(xres) else '해당 없음'}

## 4. 해석

조건별 평균 (주요 지표):

{means.to_markdown(floatfmt='.3f')}

- **이탈 기울기는 사고장애의 정도를 따라가지 않는다.** SBERT 기울기는 오히려 반대 방향이다(ρ = {R.loc['drift_slope', 'rho']:.2f}). 와해된 발화는 처음부터 거리가 높아 "점점 멀어지는" 기울기가 생기지 않는다. 이 지표는 점점 멀어지는 정도를 재는데, 교과서적 탈선과 지리멸렬은 처음부터 멀다.
- **TF-IDF 기울기는 탈선에서만 예상 방향이었다**(ρ_DER = {R.loc['semantic_drift_slope', 'rho_DER']:.2f}, 전체 ρ = {R.loc['semantic_drift_slope', 'rho']:.2f}). 한 답변 안의 40토큰 창들은 겹치지 않는 부분끼리 같은 단어를 거의 공유하지 않아, 첫 창과의 거리가 정상 답변에서도 천장 근처다(max_semantic_drift N0 {means.loc['N0', 'max_semantic_drift']:.3f}). 그래서 이 지표는 의미보다 단어가 다시 나오는지에 좌우된다.
- **질문 기준 거리(anchor)는 단계를 잘 따라간다**(ρ ≈ 0.7). 하지만 우회증 2단계({means.loc['CIRC2', 'anchor_mean']:.3f})가 탈선 2단계와 주제 이탈 2단계의 평균({pooled:.3f})보다 커서 특이도에서 탈락했다. 앞 40 내용어만 보면 "결국 질문에 답하는 장황한 말"과 "벗어나는 말"을 가를 수 없다. 우회증은 끝에서 질문으로 돌아오는 것이 특징이라 답변 전체를 봐야 한다.
- **전체 응집성(global_coh)**도 단계를 따라가지만(ρ = {res.set_index('feature').loc['global_coh', 'rho']:.2f}) 같은 이유로 우회증과 겹친다.
- **TF-IDF 지표 3개는 의미가 아니라 단어 반복에 반응한다.** 같은 말을 반복하는 보속 조건에서는 크게 낮아지고, 새 단어를 계속 쓰는 음향 연상 조건에서는 높아진다(그림의 왼쪽 세 열).
- **보속·음향 연상 전용 지표**는 해당 조건을 잘 따라간다. 다만 생성 발화가 반복과 운율을 글자 그대로 넣었기 때문에 당연한 결과에 가깝다.
- 정리: 지금 지표 중 이탈 기울기는 와해된 언어의 지표로 쓰기 어렵다. 질문 기준 거리가 가장 유망하지만, 우회증과 구분하려면 답변 끝부분이 질문으로 돌아오는지를 함께 봐야 한다.

## 5. 한계

- 생성 발화는 Claude가 작성하고 GPT-4.1이 평가했다. 둘 다 LLM이라 교과서 정의에 맞는지까지만 확인되고, 실제 환자 발화와 비슷한지는 확인되지 않는다.
- 조건마다 질문 6개(발화 6개)라 수치의 변동이 크다.
- 1단계는 GPT가 거의 0점으로 평가해(0.2~0.3), 사실상 0~3단계 중 0·2·3단계만 구분된다.
- 음향 연상 2·3단계는 GPT 평가가 모두 3점으로 구분되지 않았다.
- SBERT 지표는 앞 40 내용어로 계산해, 답변에서 차지하는 비율이 조건마다 다르다(대부분 절반 정도, 음향 연상 3단계는 약 3분의 1).
"""
    open(os.path.join(REP, '01_results.md'), 'w', encoding='utf8').write(md)
    print(md)


if __name__ == '__main__':
    assert SIGN['global_coh'] == -1 and SIGN['drift_slope'] == 1
    main()
