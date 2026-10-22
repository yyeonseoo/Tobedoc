"""v3 분석 M1-M4와 참고 모델 R1 (reports/v3/00_plan.md) -> reports/v3/01_results.md, reports/v3/figures/*.png."""
import os
import re
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from text import CFG, path, words
from v2_analyze import BG, INK, INK2, boot_g, hedges_g
from v3_features import FEATS

warnings.filterwarnings('ignore')
SEED = CFG['seed']
OUT = path('reports_v3')
FIG = os.path.join(OUT, 'figures')
LABEL = {'drift_slope': '이탈 기울기', 'drift_mean': '평균 이탈', 'adj_mean': '인접 거리', 'late_slope': '후반 기울기',
         'jump_max': '최대 점프', 'adj_sd': '인접 거리 변동성', 'global_coh': '전역 응집도', 'sem_redund': '의미 되풀이',
         'recurrence': '어휘 재등장률', 'mattr': 'MATTR', 'sem_density': '의미 밀도', 'rare_rate': '희귀어 비율',
         'clang_rate': '각운 연결'}
CLAR = re.compile(r'what do you mean|did you say|\bsorry\b|pardon|\brepeat|what was that|not sure i follow|\byou mean\b|say that again|didnt catch|whats that')


def lr(C=1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced', C=C))


def oof(X, y, groups, est=lr, reps=5):
    pred = np.zeros(len(y))
    for r in range(reps):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED + r).split(X, y, groups):
            pred[te] += est().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1] / reps
    return pred


def by_speaker(df, p):
    return df[['speaker_id', 'label']].assign(p=p).groupby(['speaker_id', 'label']).p.mean().reset_index()


def auc_ci(a, B=2000):
    rng = np.random.default_rng(SEED)
    y, p = a.label.values, a.p.values
    bs = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(B)) if len(set(y[i])) == 2]
    return roc_auc_score(y, p), *np.percentile(bs, [2.5, 97.5])


def delta_auc(a_base, a_full, B=2000):
    m = a_base.merge(a_full, on=['speaker_id', 'label'], suffixes=('_b', '_f'))
    rng = np.random.default_rng(SEED)
    y, pb, pf = m.label.values, m.p_b.values, m.p_f.values
    d = roc_auc_score(y, pf) - roc_auc_score(y, pb)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            bs.append(roc_auc_score(y[i], pf[i]) - roc_auc_score(y[i], pb[i]))
    return d, *np.percentile(bs, [2.5, 97.5])


def g_table(s, B):
    rows = []
    for c in FEATS:
        a, b = s[s.label == 1][c].dropna(), s[s.label == 0][c].dropna()
        lo, hi = boot_g(a, b, B)
        rows.append(dict(feature=c, g=hedges_g(a, b), lo=lo, hi=hi, p=stats.ttest_ind(a, b, equal_var=False).pvalue))
    t = pd.DataFrame(rows)
    t['q'] = multipletests(t.p, method='fdr_bh')[1]
    return t.set_index('feature')


def m1(f):
    r = f[(f.corpus == 'reddit') & (f.variant == 'main') & (f.n == 40) & (f.source != 'other')].reset_index(drop=True)
    B = pd.concat([r[['log_len', 'title_q', 'title_len']], pd.get_dummies(r.topic, prefix='t').astype(float)], axis=1)
    y, g = r.label.values, r.speaker_id.values
    Xb, Xc = B.values, r[FEATS].values
    a_b, a_c, a_bc = (by_speaker(r, oof(X, y, g)) for X in (Xb, Xc, np.hstack([Xb, Xc])))
    res = dict(n_posts=len(r), n_case=r[r.label == 1].speaker_id.nunique(), n_ctrl=r[r.label == 0].speaker_id.nunique(),
               auc_b=auc_ci(a_b), auc_c=auc_ci(a_c), auc_bc=auc_ci(a_bc), delta=delta_auc(a_b, a_bc))
    res['auc_lgb'] = auc_ci(by_speaker(r, oof(np.hstack([Xb, Xc]), y, g, est=lambda: LGBMClassifier(random_state=SEED, verbose=-1), reps=1)))
    words_ = pd.read_parquet(os.path.join(path('processed'), 'reddit_units.parquet'))[['unit_id', 'words']]
    txt = r[['unit_id']].merge(words_, on='unit_id').words.map(' '.join)
    Xt = TfidfVectorizer(min_df=5, token_pattern=r'\S+').fit_transform(txt)
    pred = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xt, y, g):
        pred[te] = LogisticRegression(max_iter=3000, class_weight='balanced').fit(Xt[tr], y[tr]).predict_proba(Xt[te])[:, 1]
    res['auc_tfidf'] = auc_ci(by_speaker(r, pred))
    Xe = np.stack(r.emb_mean.values)
    res['auc_r1'] = auc_ci(by_speaker(r, oof(Xe, y, g, reps=1)))
    # 피처별 g: 원래 값과, 기준선 변수로 회귀한 잔차
    resid = r.copy()
    Bc = np.column_stack([np.ones(len(B)), B.values])
    for c in FEATS:
        beta, *_ = np.linalg.lstsq(Bc, r[c].values, rcond=None)
        resid[c] = r[c].values - Bc @ beta
    raw_g = g_table(r.groupby(['speaker_id', 'label'])[FEATS].mean().reset_index(), 2000)
    res_g = g_table(resid.groupby(['speaker_id', 'label'])[FEATS].mean().reset_index(), 2000)
    return r, res, raw_g, res_g


def m2(f):
    r = f[(f.corpus == 'reddit') & (f.variant == 'nofp') & (f.n == 40)]
    both = r.groupby('speaker_id').source.nunique().gt(1)
    r = r[~r.speaker_id.isin(both[both].index)].assign(label=lambda x: (x.source == 'self').astype(int)).reset_index(drop=True)
    t = g_table(r.groupby(['speaker_id', 'label'])[FEATS].mean().reset_index(), 2000)
    auc = auc_ci(by_speaker(r, oof(r[FEATS].values, r.label.values, r.speaker_id.values)))
    return r, t, auc


def m3(f, r1):
    d = f[(f.corpus == 'daisc') & (f.n == 40)].reset_index(drop=True)
    t = g_table(d.groupby(['speaker_id', 'label'])[FEATS].mean().reset_index(), 5000)
    clf = lr().fit(r1[FEATS].values, r1.label.values)
    tr = auc_ci(by_speaker(d, clf.predict_proba(d[FEATS].values)[:, 1]), 5000)
    emb = lr().fit(np.stack(r1.emb_mean.values), r1.label.values)
    tr_r1 = auc_ci(by_speaker(d, emb.predict_proba(np.stack(d.emb_mean.values))[:, 1]), 5000)
    md = d[d.anchor_words_n > 0]
    mixed = {}
    for c in FEATS:
        res = smf.mixedlm(f'{c} ~ label + C(start_question) + C(sex) + anchor_words_n', md, groups=md.speaker_id).fit(reml=True)
        lo, hi = res.conf_int().loc['label']
        sd = md[c].std()
        mixed[c] = dict(std_coef=res.params['label'] / sd, lo=lo / sd, hi=hi / sd, p=res.pvalues['label'])
    return d, t, tr, tr_r1, pd.DataFrame(mixed).T, md


def m4(f):
    d = f[(f.corpus == 'daisc') & (f.n == 30)].copy()
    d['clar'] = d.next_prompt.fillna('').map(lambda s: bool(CLAR.search(' '.join(words(s)))))
    both = d.groupby('speaker_id').clar.agg(lambda c: c.any() and not c.all())
    d = d[d.speaker_id.isin(both[both].index)]
    rows = {}
    for c in FEATS:
        s = d.groupby(['speaker_id', 'clar'])[c].mean().unstack()
        diff = s[True] - s[False]
        rows[c] = dict(speakers=len(diff), higher=int((diff > 0).sum()), median_diff=diff.median(),
                       std_diff=diff.median() / d[c].std(), p=stats.wilcoxon(diff).pvalue)
    t = pd.DataFrame(rows).T
    t['q'] = multipletests(t.p.astype(float), method='fdr_bh')[1]
    return d, t


def figures(res, res_g, dt, t4):
    os.makedirs(FIG, exist_ok=True)
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'figure.facecolor': BG,
                         'axes.facecolor': BG, 'axes.spines.top': False, 'axes.spines.right': False})
    # fig1: 모델별 Reddit AUC
    items = [('교란 기준선 (B)', res['auc_b'], '#8a8984'), ('임상 피처만 (C)', res['auc_c'], '#2a78d6'),
             ('B + C', res['auc_bc'], '#2a78d6'), ('B + C (LightGBM)', res['auc_lgb'], '#2a78d6'),
             ('참고: 문장 임베딩 (R1)', res['auc_r1'], '#c3c2b7'), ('참고: 단어 TF-IDF', res['auc_tfidf'], '#c3c2b7')]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(items))[::-1]
    for yy, (name, (a, lo, hi), col) in zip(y, items):
        ax.errorbar(a, yy, xerr=[[a - lo], [hi - a]], fmt='o', color=col, capsize=3, markersize=7)
        ax.text(hi + 0.01, yy, f'{a:.3f}', va='center', color=INK)
    ax.set_yticks(y, [i[0] for i in items])
    ax.axvline(0.5, color=INK2, linewidth=0.8, linestyle=':')
    ax.set_xlim(0.45, 1.02)
    ax.set_xlabel('Reddit 작성자 단위 AUC (95% CI)')
    d, lo, hi = res['delta']
    ax.set_title(f'임상 피처의 증가분 ΔAUC = {d:+.3f} (95% CI {lo:+.3f} ~ {hi:+.3f})', loc='left', color=INK)
    fig.savefig(os.path.join(FIG, 'fig1_auc.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)
    # fig2: 피처별 Reddit 잔차 g vs DAIS-C g
    fig, ax = plt.subplots(figsize=(8, 5.5))
    y = np.arange(len(FEATS))[::-1]
    for t, off, col, name in [(res_g, 0.15, '#4a3aa7', 'Reddit (교란 보정 잔차, 작성자)'), (dt, -0.15, '#8a8984', 'DAIS-C (화자)')]:
        t = t.loc[FEATS]
        ax.errorbar(t.g, y + off, xerr=[t.g - t.lo, t.hi - t.g], fmt='o', color=col, capsize=3, label=name)
    for yy, c in zip(y, FEATS):
        ax.text(1.62, yy, f"{int(t4.loc[c, 'higher'])}/{int(t4.loc[c, 'speakers'])}", va='center', ha='right', color=INK2, fontsize=9)
    ax.text(1.62, y[0] + 0.8, '되물음↑', ha='right', color=INK2, fontsize=9)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(y, [LABEL[c] for c in FEATS])
    ax.set_xlim(-1.6, 1.65)
    ax.set_xlabel("Hedges' g (양수 = 조현병 쪽이 큼), 95% CI")
    ax.set_title('피처별 효과: Reddit 교란 보정 vs DAIS-C (오른쪽: 되물은 답변에서 값이 큰 화자 수)', loc='left', color=INK, fontsize=10)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)
    fig.savefig(os.path.join(FIG, 'fig2_features.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)


def tbl(t, cols, fmt='.3f'):
    x = t[cols].copy()
    x.index = [LABEL[c] for c in x.index]
    return x.to_markdown(floatfmt=fmt)


def main():
    f = pd.read_parquet(os.path.join(path('processed'), 'v3_features.parquet'))
    r1, res, raw_g, res_g = m1(f)
    r2, t2, auc2 = m2(f)
    d3, t3, tr, tr_r1, mixed, md = m3(f, r1)
    d4, t4 = m4(f)
    sign = np.sign(res_g.g)
    promising = [c for c in FEATS if (res_g.loc[c, 'lo'] > 0 or res_g.loc[c, 'hi'] < 0)
                 and np.sign(t3.loc[c, 'g']) == sign[c] and np.sign(t4.loc[c, 'median_diff']) == sign[c]]
    figures(res, res_g, t3, t4)
    conc = int((np.sign(t3.g) == sign).sum())
    fa = lambda x: f'{x[0]:.3f} ({x[1]:.3f} - {x[2]:.3f})'
    g1 = raw_g[['g', 'lo', 'hi']].join(res_g[['g', 'lo', 'hi', 'q']], rsuffix='_resid')
    md_txt = f"""# 01. v3 결과

재현: `python src/v3_features.py` 다음 `python src/v3_model.py`. 계획은 `00_plan.md`. 피처는 앞 40 내용어(M4는 30)에서 계산했다.

## M1. Reddit: 교란 기준선 대비 임상 피처의 증가분
표본: 글 {res['n_posts']}개, 작성자 조현병 {res['n_case']}명 / 대조 {res['n_ctrl']}명 (가족·보호자 글 제외)

| 모델 | 작성자 단위 AUC (95% CI) |
|---|---|
| 교란 기준선 B (길이, 주제 군집, 제목 질문형, 제목 길이) | {fa(res['auc_b'])} |
| 임상 피처만 C (13개) | {fa(res['auc_c'])} |
| B + C (로지스틱) | {fa(res['auc_bc'])} |
| B + C (LightGBM 기본값) | {fa(res['auc_lgb'])} |
| 참고 R1: 문장 임베딩 평균 (384차원) | {fa(res['auc_r1'])} |
| 참고: 본문 단어 TF-IDF | {fa(res['auc_tfidf'])} |

**주 지표 ΔAUC = AUC(B+C) - AUC(B) = {res['delta'][0]:+.3f} (95% CI {res['delta'][1]:+.3f} ~ {res['delta'][2]:+.3f})**

피처별 효과 (작성자 평균. 잔차 = 기준선 변수로 회귀하고 남은 부분, q는 잔차 기준 FDR):
{tbl(g1, ['g', 'lo', 'hi', 'g_resid', 'lo_resid', 'hi_resid', 'q'])}

## M2. Reddit 같은 게시판 안: 본인 글 vs 가족·보호자 글 (1인칭 축약어 제거, 예비)
글 {len(r2)}개, 작성자 본인 {r2[r2.label == 1].speaker_id.nunique()}명 / 가족·보호자 {r2[r2.label == 0].speaker_id.nunique()}명. 임상 피처만의 AUC {fa(auc2)}. g 양수 = 본인 글이 큼.
{tbl(t2, ['g', 'lo', 'hi', 'q'])}

## M3. DAIS-C 검증
답변 {len(d3)}개, 화자 CL {d3[d3.label == 1].speaker_id.nunique()}명 / CO {d3[d3.label == 0].speaker_id.nunique()}명.
- Reddit 잔차 g와 방향이 같은 피처: {conc}/{len(FEATS)}
- Reddit 임상 피처 모델을 그대로 적용한 화자 단위 AUC: {fa(tr)}
- 참고 R1(문장 임베딩) 모델을 그대로 적용한 AUC: {fa(tr_r1)}

{tbl(t3, ['g', 'lo', 'hi', 'q'])}

혼합효과모형 (피처 ~ 그룹 + 시작 질문 + 성별 + 질문 내용어 수 + (1|화자), 답변 {len(md)}개, 화자 {md.speaker_id.nunique()}명). 계수는 피처 표준편차 단위:
{tbl(mixed, ['std_coef', 'lo', 'hi', 'p'])}

## M4. DAIS-C 같은 사람 안: 면담자가 되물은 답변 vs 나머지 (앞 30 내용어)
되물은 답변 {int(d4.clar.sum())}개, 나머지 {int((~d4.clar).sum())}개, 두 종류가 모두 있는 화자 {d4.speaker_id.nunique()}명.
{tbl(t4, ['speakers', 'higher', 'median_diff', 'std_diff', 'p', 'q'], '.4f')}

## 유망 피처 (사전 기준 세 가지 모두 충족)
{', '.join(LABEL[c] for c in promising) if promising else '없음'}

그림: `figures/fig1_auc.png`, `figures/fig2_features.png`
"""
    open(os.path.join(OUT, '01_results.md'), 'w', encoding='utf8').write(md_txt)
    print(md_txt)


if __name__ == '__main__':
    main()
