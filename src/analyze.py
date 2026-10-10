"""reports/00_analysis_plan.md의 분석 1-3과 민감도 분석 -> reports/01_results.md, reports/figures/*.png."""
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from text import CFG, path

warnings.filterwarnings('ignore')
SEED = CFG['seed']
N = CFG['features']['n_words']
PRIMARY = 'drift_slope'
FEATS = ['drift_slope', 'drift_mean', 'adj_mean', 'late_slope', 'recurrence', 'mattr']
LABEL = {'drift_slope': '이탈 기울기 (주)', 'drift_mean': '평균 이탈', 'adj_mean': '인접 조각 거리',
         'late_slope': '후반 기울기', 'recurrence': '어휘 재등장률', 'mattr': 'MATTR'}
FIG = os.path.join(path('reports'), 'figures')
COL = {1: '#2a78d6', 0: '#eb6834'}
INK, INK2, GRID, BG = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'figure.facecolor': BG,
                     'axes.facecolor': BG, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})


def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp * (1 - 3 / (4 * (len(a) + len(b)) - 9))


def boot_g(a, b, B):
    rng = np.random.default_rng(SEED)
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.percentile([hedges_g(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(B)], [2.5, 97.5])


def effects(f, B):
    """화자(작성자) 평균 -> 지표별 g, CI, Welch p, FDR."""
    s = f.groupby(['speaker_id', 'label'])[FEATS].mean().reset_index()
    rows = []
    for c in FEATS:
        a, b = s[s.label == 1][c].dropna(), s[s.label == 0][c].dropna()
        lo, hi = boot_g(a, b, B)
        rows.append(dict(feature=c, g=hedges_g(a, b), lo=lo, hi=hi, p=stats.ttest_ind(a, b, equal_var=False).pvalue,
                         n_case=len(a), n_ctrl=len(b)))
    e = pd.DataFrame(rows)
    e['q'] = multipletests(e.p, method='fdr_bh')[1]
    return e


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced'))


def speaker_auc(f, prob, B=2000):
    a = f[['speaker_id', 'label']].assign(p=prob).groupby(['speaker_id', 'label']).p.mean().reset_index()
    rng = np.random.default_rng(SEED)
    y, p = a.label.values, a.p.values
    bs = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(B)) if len(set(y[i])) == 2]
    return roc_auc_score(y, p), *np.percentile(bs, [2.5, 97.5])


def grouped_cv(f, est=model, reps=5, labels=None):
    X, y = f[FEATS].values, (f.label.values if labels is None else labels)
    pred = np.zeros(len(f))
    for r in range(reps):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED + r).split(X, y, f.speaker_id):
            pred[te] += est().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1] / reps
    return pred


def topic_match(f):
    vec = TfidfVectorizer(min_df=5, max_df=0.95, max_features=8000, token_pattern=r'\S+')
    X = vec.fit_transform(f.words.map(' '.join))
    Z = TruncatedSVD(40, random_state=SEED).fit_transform(X)
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    pos, neg = np.where(f.label.values == 1)[0], np.where(f.label.values == 0)[0]
    S = Z[pos] @ Z[neg].T
    best = S.max(1)
    cal = np.quantile(best, 0.25)
    free, keep = np.ones(len(neg), bool), []
    for i in np.argsort(best):  # 짝 찾기 어려운 글부터
        j = np.where(free)[0][np.argmax(S[i, free])] if free.any() else None
        if j is not None and S[i, j] >= cal:
            keep += [pos[i], neg[j]]
            free[j] = False
    return f.iloc[sorted(keep)], cal, len(keep) // 2


def tag_validation(ch, units):
    d = ch[(ch.corpus == 'daisc') & ch.dist_adj.notna()]
    out = {}
    for t in ['DT', 'TC']:
        s = d.groupby(['speaker_id', f'in_{t}']).dist_adj.mean().unstack().dropna()
        diff = s[True] - s[False]
        out[t] = dict(speakers=len(s), median_diff=diff.median(), n_pos=int((diff > 0).sum()),
                      p=stats.wilcoxon(diff).pvalue if len(diff) >= 5 else np.nan, per_speaker=s)
    # 화자 단위 DT 밀도(1,000 내용어당 DT 구간 수) vs 주 지표
    u = units.assign(spans=units.in_DT.map(lambda m: int(np.sum(np.diff(np.r_[0, np.asarray(m, int)]) == 1))))
    dens = u.groupby('speaker_id').apply(lambda g: g.spans.sum() / max(g.n_words.sum(), 1) * 1000, include_groups=False)
    return out, dens


def fig_effects(er, ed):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(FEATS))[::-1]
    for e, off, col, name in [(er, 0.15, '#4a3aa7', 'Reddit (작성자 단위)'), (ed, -0.15, '#8a8984', 'DAIS-C (화자 단위)')]:
        e = e.set_index('feature').loc[FEATS]
        ax.errorbar(e.g, y + off, xerr=[e.g - e.lo, e.hi - e.g], fmt='o', color=col, ecolor=col, capsize=3, label=name, markersize=6)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(y, [LABEL[c] for c in FEATS])
    ax.set_xlabel("Hedges' g (양수 = 조현병 쪽이 큼), 95% CI")
    ax.set_title(f'그룹 차이: Reddit 발견과 DAIS-C 검증 (앞 {N} 내용어)', loc='left', color=INK)
    ax.legend(frameon=False, loc='lower right')
    ax.grid(axis='y', visible=False)
    fig.savefig(os.path.join(FIG, 'fig1_effects.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_curves(ch, f):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    k = N // CFG['features']['chunk_words']
    for ax, corpus, title in [(axes[0], 'reddit', 'Reddit'), (axes[1], 'daisc', 'DAIS-C')]:
        ids = set(f[(f.corpus == corpus)].unit_id)
        c = ch[(ch.corpus == corpus) & ch.unit_id.isin(ids) & (ch.pos >= 1) & (ch.pos < k)]
        s = c.groupby(['label', 'speaker_id', 'pos']).dist_first.mean().reset_index()
        for lab, name in [(1, '조현병'), (0, '대조')]:
            g = s[s.label == lab].groupby('pos').dist_first.agg(['mean', 'std', 'count'])
            half = stats.t.ppf(0.975, g['count'] - 1) * g['std'] / np.sqrt(g['count'])
            x = (g.index + 1) * CFG['features']['chunk_words']
            ax.fill_between(x, g['mean'] - half, g['mean'] + half, color=COL[lab], alpha=0.15, linewidth=0)
            ax.plot(x, g['mean'], color=COL[lab], linewidth=2, label=f'{name} ({s[s.label == lab].speaker_id.nunique()}명)')
        ax.set_title(title, loc='left', color=INK)
        ax.set_xlabel('내용어 위치 (조각 끝)')
        ax.legend(frameon=False, loc='lower right')
    axes[0].set_ylabel('첫 조각과의 SBERT 거리')
    fig.suptitle('말한 분량에 따른 첫 조각과의 거리 (화자 평균의 그룹 평균, 95% CI)', x=0.01, ha='left', color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig2_drift_curves.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_tags(tv):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, t in zip(axes, ['DT', 'TC']):
        s = tv[t]['per_speaker']
        for _, r in s.iterrows():
            ax.plot([0, 1], [r[False], r[True]], color='#8a8984', linewidth=1, alpha=0.7)
        ax.scatter(np.zeros(len(s)), s[False], color='#8a8984', s=24, zorder=3)
        ax.scatter(np.ones(len(s)), s[True], color='#4a3aa7', s=24, zorder=3)
        ax.set_xticks([0, 1], [f'{t} 없는 조각', f'{t} 있는 조각'])
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(f'{t}: {len(s)}명 중 {tv[t]["n_pos"]}명에서 증가 (p={tv[t]["p"]:.2f})', loc='left', color=INK)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('인접 조각과의 SBERT 거리 (화자 평균)')
    fig.suptitle('같은 사람 안에서: 연구자 판단 태그 구간의 인접 거리', x=0.01, ha='left', color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig3_tag_validation.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)


def fmt(e):
    t = e[['feature', 'g', 'lo', 'hi', 'p', 'q', 'n_case', 'n_ctrl']].copy()
    t['feature'] = t.feature.map(LABEL)
    return t.to_markdown(index=False, floatfmt='.3f')


def main():
    os.makedirs(FIG, exist_ok=True)
    f = pd.read_parquet(os.path.join(path('processed'), 'features.parquet'))
    ch = pd.read_parquet(os.path.join(path('processed'), 'chunks.parquet'))
    du = pd.read_parquet(os.path.join(path('processed'), 'daisc_units.parquet'))
    ru = pd.read_parquet(os.path.join(path('processed'), 'reddit_units.parquet'))[['unit_id', 'words']]
    r_all = f[(f.corpus == 'reddit')]
    r = r_all[(r_all.n == N) & (r_all.source != 'other')].merge(ru, on='unit_id').reset_index(drop=True)
    d = f[(f.corpus == 'daisc') & (f.n == N)].reset_index(drop=True)

    # 분석 1: Reddit
    er = effects(r, 2000)
    pred = grouped_cv(r)
    auc = speaker_auc(r, pred)
    rng = np.random.default_rng(SEED)
    amap = r.groupby('speaker_id').label.first()
    perm = []
    for b in range(200):
        sh = dict(zip(amap.index, rng.permutation(amap.values)))
        lab = r.speaker_id.map(sh).values
        pp = grouped_cv(r, reps=1, labels=lab)
        a = pd.DataFrame({'s': r.speaker_id, 'y': lab, 'p': pp}).groupby('s').agg(y=('y', 'first'), p=('p', 'mean'))
        perm.append(roc_auc_score(a.y, a.p))
    perm_p = (1 + np.sum(np.array(perm) >= auc[0])) / 201
    auc_lgb = speaker_auc(r, grouped_cv(r, est=lambda: LGBMClassifier(random_state=SEED, verbose=-1), reps=1))
    rm, cal, npairs = topic_match(r)
    erm = effects(rm, 2000)
    auc_m = speaker_auc(rm, grouped_cv(rm))

    # 분석 2: DAIS-C
    ed = effects(d, 5000)
    conc = (np.sign(er.set_index('feature').g) == np.sign(ed.set_index('feature').g)).loc[FEATS]
    clf = model().fit(r[FEATS].values, r.label.values)
    auc_t = speaker_auc(d, clf.predict_proba(d[FEATS].values)[:, 1], B=5000)
    md = d[d.anchor_words_n > 0].copy()
    mixed = []
    for c in FEATS:
        res = smf.mixedlm(f'{c} ~ label + C(start_question) + C(sex) + anchor_words_n', md, groups=md.speaker_id).fit(reml=True)
        lo, hi = res.conf_int().loc['label']
        mixed.append(dict(feature=LABEL[c], coef=res.params['label'], lo=lo, hi=hi, p=res.pvalues['label'], sd=md[c].std()))
    mixed = pd.DataFrame(mixed)

    # 분석 3: 태그
    tv, dens = tag_validation(ch, du)
    sp = d.groupby('speaker_id')[PRIMARY].mean()
    rho = stats.spearmanr(dens.loc[sp.index], sp)

    # 민감도
    sens = []
    for n in CFG['features']['n_words_sensitivity']:
        for corpus, filt in [('reddit', lambda x: x.source != 'other'), ('daisc', lambda x: x.corpus == 'daisc')]:
            x = f[(f.corpus == corpus) & (f.n == n)]
            x = x[filt(x)]
            e = effects(x, 1000).set_index('feature').loc[PRIMARY]
            sens.append(dict(analysis=f'{corpus} 앞 {n} 내용어', g=e.g, lo=e.lo, hi=e.hi, n=f'{int(e.n_case)}/{int(e.n_ctrl)}'))
    rs = r_all[(r_all.n == N) & r_all.source.isin(['self', 'control'])]
    e = effects(rs, 1000).set_index('feature').loc[PRIMARY]
    sens.append(dict(analysis='reddit 본인 글만', g=e.g, lo=e.lo, hi=e.hi, n=f'{int(e.n_case)}/{int(e.n_ctrl)}'))
    sens = pd.DataFrame(sens)

    # 계획 외 탐색: 노트북 AUC(0.78)와의 차이 설명용. 글 길이만으로의 작성자 단위 AUC
    raw = pd.read_csv(path('reddit_csv'), usecols=['id', 'selftext']).rename(columns={'id': 'unit_id'})
    lens = r[['unit_id', 'speaker_id', 'label']].merge(raw, on='unit_id')
    lens = lens.assign(L=lens.selftext.fillna('').str.split().str.len()).groupby('speaker_id').agg(y=('label', 'first'), L=('L', 'mean'))
    auc_len = roc_auc_score(lens.y, -lens.L)

    fig_effects(er, ed)
    fig_curves(ch, pd.concat([r, d]))
    fig_tags(tv)

    md_txt = f"""# 01. 결과 (v2)

재현: `python src/analyze.py`. 계획은 `00_analysis_plan.md`. 모든 지표는 앞 {N} 내용어(5개씩 {N // 5}조각)에서 계산했다.

## 표본
- Reddit (가족·보호자 글 제외): 글 {len(r)}개, 작성자 조현병 {r[r.label == 1].speaker_id.nunique()}명 / 대조 {r[r.label == 0].speaker_id.nunique()}명
- DAIS-C: 답변 {len(d)}개, 화자 CL {d[d.label == 1].speaker_id.nunique()}명 / CO {d[d.label == 0].speaker_id.nunique()}명

## 분석 1: Reddit 발견 (작성자 단위)
{fmt(er)}

| 분류 | AUC | 95% CI |
|---|---|---|
| 로지스틱 (주) | {auc[0]:.3f} | {auc[1]:.3f} - {auc[2]:.3f} |
| LightGBM (기본값) | {auc_lgb[0]:.3f} | {auc_lgb[1]:.3f} - {auc_lgb[2]:.3f} |
| 로지스틱, 주제 매칭 표본 ({npairs}쌍, 유사도 기준 {cal:.3f}) | {auc_m[0]:.3f} | {auc_m[1]:.3f} - {auc_m[2]:.3f} |

순열 검정 (작성자 라벨 200회): 귀무 AUC 평균 {np.mean(perm):.3f}, p = {perm_p:.3f}

주제 매칭 표본의 효과 크기:
{fmt(erm)}

## 분석 2: DAIS-C 검증 (화자 단위)
{fmt(ed)}

- 방향 일치 (Reddit과 부호가 같은 지표): {int(conc.sum())}/{len(FEATS)} ({', '.join(LABEL[c] for c in FEATS if conc[c])})
- Reddit 모델을 그대로 적용한 화자 단위 AUC: {auc_t[0]:.3f} (95% CI {auc_t[1]:.3f} - {auc_t[2]:.3f})

혼합효과모형 (지표 ~ 그룹 + 시작 질문 + 성별 + 질문 내용어 수 + (1|화자), 기준점 있는 답변 {len(md)}개, 화자 {md.speaker_id.nunique()}명). coef는 CL - CO 차이, sd는 지표의 답변 단위 표준편차:
{mixed.to_markdown(index=False, floatfmt='.4f')}

## 분석 3: 태그로 지표 타당도 확인 (같은 사람 안에서)
| 태그 | 화자 | 태그 조각이 더 먼 화자 | 차이 중앙값 (인접 거리) | Wilcoxon p |
|---|---|---|---|---|
""" + '\n'.join(f"| {t} | {v['speakers']} | {v['n_pos']} | {v['median_diff']:+.4f} | {v['p']:.3f} |" for t, v in tv.items()) + f"""

화자 단위 DT 밀도(1,000 내용어당 구간 수)와 주 지표의 Spearman ρ = {rho.statistic:.3f} (p = {rho.pvalue:.3f}, {len(sp)}명)

## 민감도 분석 (주 지표 g)
{sens.to_markdown(index=False, floatfmt='.3f')}

## 탐색 (계획 외)
- 같은 Reddit 표본에서 본문 길이 하나만으로 구한 작성자 단위 AUC (짧을수록 조현병): {auc_len:.3f}.
  우리 지표는 앞 {N} 내용어로 길이를 맞췄기 때문에 이 차이를 쓰지 못한다. 길이를 맞추지 않은 노트북 지표의 AUC 0.78에는 길이 차이가 섞여 있을 수 있다.

그림: `figures/fig1_effects.png`, `fig2_drift_curves.png`, `fig3_tag_validation.png`
"""
    open(os.path.join(path('reports'), '01_results.md'), 'w', encoding='utf8').write(md_txt)
    print(md_txt)


if __name__ == '__main__':
    assert abs(hedges_g([1, 2, 3, 4], [1, 2, 3, 4])) < 1e-12
    main()
