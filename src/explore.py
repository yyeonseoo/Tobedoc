"""탐색 분석 E1, E2 (00_analysis_plan.md 변경 기록) -> reports/02_exploratory.md, reports/figures/fig4_exploratory.png.

E1: r/schizophrenia 안에서 본인 글 vs 가족·보호자 글 (게시판 장르를 맞춘 비교)
E2: DAIS-C 답변을 SBERT 주제 군집으로 나누고, 두 그룹이 모두 있는 군집 안에서 혼합효과모형
E3: Reddit 글을 SBERT 주제 군집으로 나누고, 두 그룹이 모두 있는 군집 안에서 주제 보정 회귀 (작성자 군집 강건 SE)
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from analyze import FEATS, FIG, INK, INK2, LABEL, SEED, effects, fmt, grouped_cv, hedges_g, speaker_auc
from text import CFG, path

N = CFG['features']['n_words']


def e1(f):
    r = f[(f.corpus == 'reddit') & (f.n == N) & f.source.isin(['self', 'other'])]
    mixed = r.groupby('speaker_id').source.nunique().gt(1)
    r = r[~r.speaker_id.isin(mixed[mixed].index)].assign(label=lambda x: (x.source == 'self').astype(int)).reset_index(drop=True)
    return r, effects(r, 2000), speaker_auc(r, grouped_cv(r)), int(mixed.sum())


def e2(f):
    d = f[(f.corpus == 'daisc') & (f.n == N)].reset_index(drop=True)
    u = pd.read_parquet(os.path.join(path('processed'), 'daisc_units.parquet')).set_index('unit_id')
    texts = [' '.join(u.loc[i, 'words']) for i in d.unit_id]
    E = SentenceTransformer(CFG['features']['sbert_model']).encode(texts, normalize_embeddings=True)
    sil = {k: silhouette_score(E, KMeans(k, n_init=10, random_state=SEED).fit_predict(E), metric='cosine') for k in range(4, 9)}
    k = max(sil, key=sil.get)
    d['topic'] = KMeans(k, n_init=10, random_state=SEED).fit_predict(E)
    # 군집 대표 단어: 군집 내 빈도 / 전체 빈도 (군집 안 3회 이상)
    allw = pd.Series([w for t in texts for w in t.split()]).value_counts()
    tops = {}
    for t in range(k):
        cw = pd.Series([w for x, tt in zip(texts, d.topic) if tt == t for w in x.split()]).value_counts()
        cw = cw[cw >= 3]
        tops[t] = ', '.join((cw / allw[cw.index]).sort_values(ascending=False).head(8).index)
    comp = d.groupby(['topic', 'group']).agg(answers=('unit_id', 'size'), speakers=('speaker_id', 'nunique')).unstack(fill_value=0)
    ok = [t for t in range(k) if comp.loc[t, ('speakers', 'CL')] >= 2 and comp.loc[t, ('speakers', 'CO')] >= 2]
    m = d[d.topic.isin(ok)]
    rows = []
    for c in FEATS:
        res = smf.mixedlm(f'{c} ~ label + C(topic) + C(sex) + C(start_question)', m, groups=m.speaker_id).fit(reml=True)
        lo, hi = res.conf_int().loc['label']
        sd = m[c].std()
        rows.append(dict(feature=LABEL[c], coef=res.params['label'], lo=lo, hi=hi, p=res.pvalues['label'],
                         std_coef=res.params['label'] / sd, std_lo=lo / sd, std_hi=hi / sd))
    return d, sil, k, tops, comp, ok, m, pd.DataFrame(rows)


def e3(f):
    r = f[(f.corpus == 'reddit') & (f.n == N) & (f.source != 'other')].reset_index(drop=True)
    u = pd.read_parquet(os.path.join(path('processed'), 'reddit_units.parquet')).set_index('unit_id')
    E = SentenceTransformer(CFG['features']['sbert_model']).encode(
        [' '.join(u.loc[i, 'words']) for i in r.unit_id], normalize_embeddings=True, batch_size=128)
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(E), min(3000, len(E)), replace=False)  # 실루엣은 표본 3,000개로 계산
    labs = {k: KMeans(k, n_init=5, random_state=SEED).fit_predict(E) for k in range(5, 21)}
    sil = {k: silhouette_score(E[sub], v[sub], metric='cosine') for k, v in labs.items()}
    k = max(sil, key=sil.get)
    r['topic'] = labs[k]
    cnt = r.groupby(['topic', 'label']).size().unstack(fill_value=0)
    ok = cnt[(cnt[0] >= 30) & (cnt[1] >= 30)].index.tolist()
    m = r[r.topic.isin(ok)].copy()
    rows, per = [], []
    for c in FEATS:
        m['z'] = (m[c] - m[c].mean()) / m[c].std()
        res = smf.ols('z ~ label + C(topic)', m).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(m.speaker_id)[0]})
        lo, hi = res.conf_int().loc['label']
        rows.append(dict(feature=LABEL[c], std_coef=res.params['label'], std_lo=lo, std_hi=hi, p=res.pvalues['label']))
        per.append({t: hedges_g(g[g.label == 1][c], g[g.label == 0][c]) for t, g in m.groupby('topic')})
    per = pd.DataFrame(per, index=[LABEL[c] for c in FEATS]).T
    per.insert(0, 'posts_SZ', cnt.loc[per.index, 1])
    per.insert(1, 'posts_CO', cnt.loc[per.index, 0])
    return sil, k, cnt, ok, m, pd.DataFrame(rows), per


def figure(ee1, mm, m3):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    y = np.arange(len(FEATS))[::-1]
    e = ee1.set_index('feature').loc[FEATS]
    axes[0].errorbar(e.g, y, xerr=[e.g - e.lo, e.hi - e.g], fmt='o', color='#4a3aa7', capsize=3)
    axes[0].set_title('E1 Reddit: 본인 글 - 가족·보호자 글 (g)', loc='left', color=INK)
    axes[0].set_xlabel("Hedges' g, 95% CI")
    axes[1].errorbar(mm.std_coef, y, xerr=[mm.std_coef - mm.std_lo, mm.std_hi - mm.std_coef], fmt='o', color='#8a8984', capsize=3)
    axes[1].set_title('E2 DAIS-C: 주제 군집 보정 CL - CO (표준화 계수)', loc='left', color=INK)
    axes[1].set_xlabel('그룹 계수 / 지표 표준편차, 95% CI')
    axes[2].errorbar(m3.std_coef, y, xerr=[m3.std_coef - m3.std_lo, m3.std_hi - m3.std_coef], fmt='o', color='#2a78d6', capsize=3)
    axes[2].set_title('E3 Reddit: 주제 군집 보정 조현병 - 대조 (표준화 계수)', loc='left', color=INK)
    axes[2].set_xlabel('표준화 계수, 95% CI (작성자 군집 강건)')
    axes[0].set_yticks(y, [LABEL[c] for c in FEATS])
    for ax in axes:
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig4_exploratory.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    f = pd.read_parquet(os.path.join(path('processed'), 'features.parquet'))
    r1, ee1, auc1, nmix = e1(f)
    d, sil, k, tops, comp, ok, m, mm = e2(f)
    sil3, k3, cnt3, ok3, m3, mm3, per3 = e3(f)
    figure(ee1, mm, mm3)
    comp.columns = [f'{a}_{b}' for a, b in comp.columns]
    comp['대표 단어'] = pd.Series(tops)
    md = f"""# 02. 탐색 분석 (계획 외, 00_analysis_plan.md 변경 기록 E1·E2)

재현: `python src/explore.py`. 본 분석 판정(01)은 바꾸지 않는다.

## E1: r/schizophrenia 안에서 본인 글 vs 가족·보호자 글
같은 게시판이라 장르와 주제가 비슷하다. 작성 주체는 규칙 기반 1차 분류라 예비 결과다 (팀 수작업 확인 전).
- 글 {len(r1)}개, 작성자 본인 {r1[r1.label == 1].speaker_id.nunique()}명 / 가족·보호자 {r1[r1.label == 0].speaker_id.nunique()}명 (두 범주 모두 쓴 작성자 {nmix}명 제외)
- g 양수 = 본인 글 쪽이 큼

{fmt(ee1).replace('n_case', 'n_self').replace('n_ctrl', 'n_other')}

6개 지표 로지스틱 회귀, 작성자 단위 AUC: {auc1[0]:.3f} (95% CI {auc1[1]:.3f} - {auc1[2]:.3f})

## E2: DAIS-C 주제 군집 안에서 비교
- 실루엣 점수 {', '.join(f'k={kk}: {v:.3f}' for kk, v in sil.items())} -> k = {k}
- 두 그룹 화자가 각각 2명 이상인 군집: {ok} -> 답변 {len(m)}개, 화자 {m.speaker_id.nunique()}명

{comp.to_markdown()}

혼합효과모형 지표 ~ 그룹 + 주제 군집 + 성별 + 시작 질문 + (1|화자). coef = CL - CO, std_coef = coef / 지표 표준편차:
{mm.to_markdown(index=False, floatfmt='.4f')}

DAIS-C는 답변이 적고 주제가 그룹과 거의 겹쳐 군집이 약하고(실루엣 약 0.1) 두 그룹이 함께 있는 군집이 적다. 이 결과는 참고용이다.

## E3: Reddit 주제 군집 보정
- 실루엣 점수(표본 3,000) 최댓값 k = {k3} ({sil3[k3]:.3f}). 범위 {min(sil3.values()):.3f} - {max(sil3.values()):.3f}
- 두 그룹 글이 각각 30개 이상인 군집 {len(ok3)}/{k3}개 -> 글 {len(m3)}개 (조현병 {int((m3.label == 1).sum())} / 대조 {int((m3.label == 0).sum())}), 작성자 {m3.speaker_id.nunique()}명

주제 보정 회귀 (표준화 지표 ~ 그룹 + 군집, 작성자 군집 강건 SE):
{mm3.to_markdown(index=False, floatfmt='.3f')}

군집별 그룹 차이 g (조현병 - 대조, 글 단위):
{per3.to_markdown(floatfmt='.2f')}

그림: `figures/fig4_exploratory.png`
"""
    open(os.path.join(path('reports'), '02_exploratory.md'), 'w', encoding='utf8').write(md)
    print(md)


if __name__ == '__main__':
    main()
