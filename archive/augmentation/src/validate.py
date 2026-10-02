"""Step 3: 실제 vs 합성 4축 검증 (fidelity / diversity / memorization / discriminability).

  python src/validate.py          -> reports/03_validation.md, reports/figures/*.png, reports/validation_metrics.json
  python src/validate.py pilot    -> reports/02_pilot_preview.md, reports/figures/pilot_*.png
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from preprocess import CFG, ROOT, clean, nlp, segment

V = CFG['validate']
SEED = CFG['seed']
TAG = sys.argv[1] if len(sys.argv) > 1 else ''
# 실행별 입출력 경로만 다르다 (평가 로직은 공통). TAG -> (합성 디렉터리, 리포트 파일, 제목, 산출물 접두어)
RUNS = {'': ('synthetic', '03_validation.md', '03. 검증 (Step 3)', ''),
        'pilot': ('synthetic', '02_pilot_preview.md', '02. 파일럿 검증 미리보기', 'pilot_'),
        'v2': ('synthetic/v2', '05_validation_v2.md', '05. 2차 검증', 'v2_'),
        'v2pilot': ('synthetic/v2', '05_pilot_preview_v2.md', '05. 2차 파일럿 검증 미리보기', 'v2pilot_'),
        'v3': ('synthetic/v3', '06_validation_v3.md', '06. 3차 검증', 'v3_'),
        'v3pilot': ('synthetic/v3', '06_pilot_preview_v3.md', '06. 3차 파일럿 검증 미리보기', 'v3pilot_')}
SYN_DIR, REPORT_NAME, TITLE, PRE = RUNS[TAG]
PILOT = TAG.endswith('pilot')
REP = os.path.join(ROOT, CFG['paths']['reports'])
FIG = os.path.join(REP, 'figures')
os.makedirs(FIG, exist_ok=True)
C_REAL, C_SYN = '#2a78d6', '#eb6834'
FEATS = ['n_sentences', 'n_tokens', 'mean_sent_len', 'mattr', 'noun', 'verb', 'pron', 'adv', 'filler_per100',
         'adj_sim', 'first_dist']
SINGLE_FILLERS = {f for f in V['fillers'] if ' ' not in f}
MULTI_FILLERS = [f for f in V['fillers'] if ' ' in f]


# ---------------------------------------------------------------- features
def mattr(words, w):
    if len(words) < w:
        return len(set(words)) / max(len(words), 1)
    return float(np.mean([len(set(words[i:i + w])) / w for i in range(len(words) - w + 1)]))


def surface(doc):
    toks = [t for t in doc if not (t.is_space or t.is_punct)]
    words = [t.lower_ for t in toks]
    n = len(toks)
    pos = pd.Series([t.pos_ for t in toks]).value_counts()
    low = ' ' + ' '.join(words) + ' '
    fill = sum(w in SINGLE_FILLERS for w in words) + sum(low.count(f' {m} ') for m in MULTI_FILLERS)
    sents = segment(doc)
    return dict(n_sentences=len(sents), n_tokens=n, mean_sent_len=n / max(len(sents), 1),
                mattr=mattr(words, V['mattr_window']),
                noun=(pos.get('NOUN', 0) + pos.get('PROPN', 0)) / n, verb=pos.get('VERB', 0) / n,
                pron=pos.get('PRON', 0) / n, adv=pos.get('ADV', 0) / n, filler_per100=100 * fill / n,
                sentences=sents)


def build(df, model):
    docs = list(nlp().pipe(df.text.tolist(), batch_size=64))
    f = pd.DataFrame([surface(d) for d in docs], index=df.index)
    flat = [s for ss in f.sentences for s in ss]
    E = model.encode(flat, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    adj, first, emb, i = [], [], [], 0
    for ss in f.sentences:
        e = E[i:i + len(ss)]
        i += len(ss)
        m = e.mean(0)
        emb.append(m / np.linalg.norm(m))
        if len(ss) < 2:
            adj.append(np.nan)
            first.append(np.nan)
        else:
            adj.append(float(np.mean(np.sum(e[:-1] * e[1:], 1))))
            first.append(float(np.mean(1 - e[1:] @ e[0])))
    f['adj_sim'], f['first_dist'] = adj, first
    return pd.concat([df, f.drop(columns='sentences')], axis=1), np.vstack(emb)


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / s if s > 0 else np.nan


def mwu(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    return stats.mannwhitneyu(a, b).pvalue if len(a) > 1 and len(b) > 1 else np.nan


# ---------------------------------------------------------------- n-gram helpers
def words(t):
    return clean(t).lower().split()


def ngrams(w, n):
    return [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]


def distinct(texts, n):
    g = [x for t in texts for x in ngrams(words(t), n)]
    return len(set(g)) / max(len(g), 1)


# ---------------------------------------------------------------- figures
def hist_grid(R, S, group, path):
    fig, axes = plt.subplots(3, 4, figsize=(13, 8.5))
    for ax, f in zip(axes.flat, FEATS):
        a, b = R[f].dropna(), S[f].dropna()
        lo, hi = np.nanpercentile(np.r_[a, b], [1, 99])
        bins = np.linspace(lo, hi, 25)
        ax.hist(a, bins, density=True, histtype='step', lw=2, color=C_REAL, label=f'real (n={len(a)})')
        ax.hist(b, bins, density=True, histtype='step', lw=2, color=C_SYN, label=f'synthetic (n={len(b)})')
        ax.set_title(f, fontsize=10)
        ax.tick_params(labelsize=8)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes.flat[-1].axis('off')
    h, l = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(h, l, loc='center', frameon=False)
    fig.suptitle(f'Fidelity — {group}: real vs synthetic (density)', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def two_hist(pairs, title, xlabel, path):
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.5 * len(pairs), 3.6), squeeze=False)
    for ax, (name, a, b) in zip(axes.flat, pairs):
        bins = np.linspace(min(np.min(a), np.min(b)), max(np.max(a), np.max(b)), 40)
        ax.hist(a, bins, density=True, histtype='step', lw=2, color=C_REAL, label='real')
        ax.hist(b, bins, density=True, histtype='step', lw=2, color=C_SYN, label='synthetic')
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def effect_plot(eff, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(eff))
    ax.axvline(0, color='#c8c7c2', lw=1)
    ax.scatter(eff.d_real, y, s=60, color=C_REAL, label='real', zorder=3, edgecolor='white', linewidth=2)
    ax.scatter(eff.d_syn, y, s=60, color=C_SYN, marker='D', label='synthetic', zorder=3, edgecolor='white', linewidth=2)
    for i, (a, b) in enumerate(zip(eff.d_real, eff.d_syn)):
        ax.plot([a, b], [i, i], color='#c8c7c2', lw=1, zorder=1)
    ax.set_yticks(y, eff.index)
    ax.set_xlabel("Cohen's d (CL − CO), response level")
    ax.legend(frameon=False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_title('Effect preservation: CL vs CO difference, real vs synthetic', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(SEED)
    corpus = pd.read_parquet(os.path.join(ROOT, CFG['paths']['corpus']))
    real = corpus[corpus.substantive][['speaker_id', 'group', 'text']].reset_index(drop=True)
    syn = pd.read_parquet(os.path.join(ROOT, SYN_DIR, 'synthetic_corpus.parquet'))
    syn = syn.rename(columns={'seed_speaker_id': 'speaker_id'})[['sample_id', 'speaker_id', 'group', 'temperature', 'round', 'text']]
    syn['text'] = syn.text.map(clean)
    if PILOT:  # 파일럿: 생성된 시드 화자만 실제 쪽도 동일하게 제한
        real = real[real.speaker_id.isin(syn.speaker_id.unique())].reset_index(drop=True)
    model = SentenceTransformer(V['sbert_model'], device='cpu')
    R, ER = build(real, model)
    S, ES = build(syn, model)
    M, out = {}, []
    pre = PRE

    # ============ 3a Fidelity
    out.append('## 3a. Fidelity (분포 충실도)\n')
    out.append('실제 = 30토큰 이상 참여자 턴, 합성 = 같은 전처리(clean→spaCy→병합)를 거친 합성 응답. '
               'KS 검정 p와 Wasserstein 거리(W; 피처 원 단위). 그룹별 분리.\n')
    rows = []
    for g in ['CL', 'CO']:
        r, s = R[R.group == g], S[S.group == g]
        for f in FEATS:
            a, b = r[f].dropna(), s[f].dropna()
            ks = stats.ks_2samp(a, b)
            rows.append(dict(group=g, feature=f, real_mean=a.mean(), syn_mean=b.mean(), real_sd=a.std(), syn_sd=b.std(),
                             KS=ks.statistic, KS_p=ks.pvalue, W=stats.wasserstein_distance(a, b),
                             W_over_realSD=stats.wasserstein_distance(a, b) / a.std()))
        hist_grid(r, s, g, os.path.join(FIG, f'{pre}fidelity_{g}.png'))
    fid = pd.DataFrame(rows)
    fid['KS_q'] = stats.false_discovery_control(fid.KS_p)
    M['fidelity'] = fid.to_dict('records')
    out.append(fid.round(3).to_markdown(index=False))
    out.append(f'\n`W_over_realSD` = W를 실제 분포 SD로 나눈 값(피처 간 비교용). KS_q = BH 보정.\n')
    out.append(f'![CL]({"figures/" + pre}fidelity_CL.png)\n![CO]({"figures/" + pre}fidelity_CO.png)\n')

    # 효과 보존
    out.append('### 효과 보존 검사 (CL − CO)\n')
    out.append("d = Cohen's d (CL−CO). 응답 단위 p = Mann-Whitney, q = BH 보정. "
               '화자 단위 d = 화자(합성은 시드 화자)별 평균으로 계산 — 응답이 화자에 내포된 비독립성 보정용.\n')
    eff = []
    for f in FEATS:
        rc, ro = R.loc[R.group == 'CL', f], R.loc[R.group == 'CO', f]
        sc, so = S.loc[S.group == 'CL', f], S.loc[S.group == 'CO', f]
        rs = R.groupby(['group', 'speaker_id'])[f].mean()
        ss = S.groupby(['group', 'speaker_id'])[f].mean()
        eff.append(dict(feature=f, d_real=cohen_d(rc, ro), p_real=mwu(rc, ro), d_syn=cohen_d(sc, so), p_syn=mwu(sc, so),
                        d_real_spk=cohen_d(rs['CL'], rs['CO']), p_real_spk=mwu(rs['CL'], rs['CO']),
                        d_syn_spk=cohen_d(ss['CL'], ss['CO']), p_syn_spk=mwu(ss['CL'], ss['CO'])))
    eff = pd.DataFrame(eff).set_index('feature')
    eff['q_real'] = stats.false_discovery_control(eff.p_real)
    eff['q_syn'] = stats.false_discovery_control(eff.p_syn)
    eff['direction_match'] = np.sign(eff.d_real) == np.sign(eff.d_syn)
    eff['d_ratio'] = eff.d_syn / eff.d_real

    def verdict(x):
        if x.q_real >= .05:
            return '실제에서 비유의'
        if not x.direction_match:
            return '**뒤집힘**' if x.q_syn < .05 else '**뒤집힘(비유의)**'
        if x.q_syn >= .05:
            return '**사라짐**'
        return '보존'
    eff['status'] = eff.apply(verdict, axis=1)
    M['effect'] = eff.reset_index().to_dict('records')
    cols = ['d_real', 'q_real', 'd_syn', 'q_syn', 'direction_match', 'd_ratio', 'd_real_spk', 'd_syn_spk', 'status']
    out.append(eff[cols].round(3).to_markdown())
    lost = eff[eff.status.str.contains('사라짐|뒤집힘')].index.tolist()
    out.append(f'\n실제에서 유의(q<.05)한 피처 {int((eff.q_real < .05).sum())}개 중 '
               f'보존 {int((eff.status == "보존").sum())}개. 사라지거나 뒤집힌 피처: {", ".join(lost) or "없음"}.\n')
    effect_plot(eff, os.path.join(FIG, f'{pre}effect.png'))
    out.append(f'![effect](figures/{pre}effect.png)\n')

    # 라운드별: 합성 − 시드 화자 실제 평균 (pilot 교정 효과 추적; 화자 차이를 빼기 위해 화자 평균 대비 편차로 비교)
    if S['round'].nunique() > 1:
        spk_mean = R.groupby('speaker_id')[FEATS].mean()
        dev = S[FEATS] - spk_mean.loc[S.speaker_id].to_numpy()
        rd = dev.groupby(S['round']).mean()[['mattr', 'filler_per100', 'mean_sent_len', 'adj_sim']]
        rd['n'] = S['round'].value_counts()
        M['round'] = rd.to_dict()
        out.append('### 생성 라운드별 편차 (합성 − 시드 화자의 실제 평균)\n')
        out.append('파일럿 이후 생성자 측 교정(비유창성 밀도 재현 강화)의 효과 추적용. 0에 가까울수록 시드 화자와 비슷하다.\n')
        out.append(rd.round(3).to_markdown() + '\n')

    # ============ 3b Diversity
    out.append('## 3b. Diversity (mode collapse 검사)\n')
    out.append('응답 임베딩 = 문장 SBERT 임베딩 평균(정규화) — MiniLM 256 wordpiece 절단 회피.\n')
    div, pairs = [], []
    for g in ['CL', 'CO']:
        er, es = ER[(R.group == g).to_numpy()], ES[(S.group == g).to_numpy()]
        pr = (er @ er.T)[np.triu_indices(len(er), 1)]
        ps = (es @ es.T)[np.triu_indices(len(es), 1)]
        pairs.append((g, pr, ps))
        tr, ts = R[R.group == g].text.tolist(), S[S.group == g].text.tolist()
        # distinct-n: 합성과 같은 응답 수의 실제 부분표본 200회 (크기 효과 통제)
        boot = {n: [distinct(list(rng.choice(tr, len(ts), replace=False)), n) for _ in range(200)] for n in (1, 2)}
        div.append(dict(group=g, pair_cos_real_mean=pr.mean(), pair_cos_syn_mean=ps.mean(),
                        pair_cos_real_p95=np.percentile(pr, 95), pair_cos_syn_p95=np.percentile(ps, 95),
                        KS_p=stats.ks_2samp(pr, ps).pvalue,
                        distinct1_syn=distinct(ts, 1), distinct1_real_matched=np.mean(boot[1]),
                        distinct1_real_95=f'{np.percentile(boot[1], 2.5):.3f}–{np.percentile(boot[1], 97.5):.3f}',
                        distinct2_syn=distinct(ts, 2), distinct2_real_matched=np.mean(boot[2]),
                        distinct2_real_95=f'{np.percentile(boot[2], 2.5):.3f}–{np.percentile(boot[2], 97.5):.3f}'))
    div = pd.DataFrame(div)
    M['diversity'] = div.to_dict('records')
    out.append(div.set_index("group").round(3).T.to_markdown())
    two_hist(pairs, 'Pairwise cosine similarity between responses (within group)', 'cosine',
             os.path.join(FIG, f'{pre}pairwise.png'))
    out.append(f'\n![pairwise](figures/{pre}pairwise.png)\n')

    # 중복/준중복
    def near_dups(E, texts, ids):
        sim = E @ E.T
        g5 = [set(ngrams(words(t), 5)) for t in texts]
        hits = []
        for i in range(len(E)):
            for j in range(i + 1, len(E)):
                jac = len(g5[i] & g5[j]) / max(len(g5[i] | g5[j]), 1)
                if sim[i, j] >= V['near_dup_cosine'] or jac >= .3:
                    hits.append((ids[i], ids[j], round(float(sim[i, j]), 3), round(jac, 3)))
        return hits
    nd_s = near_dups(ES, S.text.tolist(), S.sample_id.tolist())
    nd_r = near_dups(ER, R.text.tolist(), [f'{a}#{i}' for i, a in enumerate(R.speaker_id)])
    nps, npr = len(S) * (len(S) - 1) / 2, len(R) * (len(R) - 1) / 2
    M['near_dup'] = dict(syn_pairs=len(nd_s), syn_rate=len(nd_s) / nps, real_pairs=len(nd_r), real_rate=len(nd_r) / npr)
    out.append(f'**준중복** (cos ≥ {V["near_dup_cosine"]} 또는 5-gram Jaccard ≥ 0.3): 합성 {len(nd_s)}쌍 '
               f'({len(nd_s) / nps:.3%} of pairs), 실제 {len(nd_r)}쌍 ({len(nd_r) / npr:.3%}). 완전 중복 합성 텍스트: '
               f'{int(S.text.duplicated().sum())}개.\n')
    if nd_s:
        out.append(pd.DataFrame(nd_s, columns=['a', 'b', 'cos', 'jaccard5']).to_markdown(index=False) + '\n')

    # 시드 화자 수렴
    spk = sorted(R.speaker_id.unique())
    cent = {s: ER[(R.speaker_id == s).to_numpy()].sum(0) for s in spk}
    cnt = {s: int((R.speaker_id == s).sum()) for s in spk}
    grp = dict(zip(R.speaker_id, R.group))

    def conv(E, df, loo):
        res = []
        for e, (_, x) in zip(E, df.iterrows()):
            sims = {}
            for s in spk:
                c = cent[s] - (e if loo and s == x.speaker_id else 0)
                k = cnt[s] - (1 if loo and s == x.speaker_id else 0)
                if k > 0:
                    sims[s] = float(e @ (c / np.linalg.norm(c)))
            if x.speaker_id not in sims:
                continue
            others = [v for s, v in sims.items() if s != x.speaker_id and grp[s] == x.group]
            res.append(dict(own=sims[x.speaker_id], other=np.mean(others),
                            top1=max(sims, key=sims.get) == x.speaker_id))
        return pd.DataFrame(res)
    cr, cs = conv(ER, R, True), conv(ES, S, False)
    conv_t = pd.DataFrame({'real (leave-one-out)': [cr.own.mean(), cr.other.mean(), (cr.own - cr.other).mean(), cr.top1.mean()],
                           'synthetic → seed speaker': [cs.own.mean(), cs.other.mean(), (cs.own - cs.other).mean(), cs.top1.mean()]},
                          index=['cos to own speaker centroid', 'mean cos to other same-group speakers', 'gap (own − other)',
                                 'own speaker is nearest centroid (top-1 rate)'])
    M['convergence'] = conv_t.to_dict()
    out.append('**시드 화자 수렴**: 응답과 화자 중심(실질 응답 임베딩 평균) 간 유사도. 실제 쪽은 자기 응답을 뺀 중심(LOO)을 기준선으로 쓴다. '
               '합성의 gap·top-1이 실제 기준선보다 크게 높으면 시드 화자에게 과수렴한 것이다.\n')
    out.append(conv_t.round(3).to_markdown() + '\n')

    # ============ 3c Memorization
    out.append('## 3c. Memorization / Privacy (원문 유출 검사)\n')
    allw = {s: words(' '.join(d.text)) for s, d in corpus.groupby('speaker_id')}
    g5 = {s: set(ngrams(w, 5)) for s, w in allw.items()}
    all5 = set().union(*g5.values())

    def ov(w, ref):
        g = ngrams(w, 5)
        return sum(x in ref for x in g) / len(g) if g else 0.0

    def longest_run(w, ref):
        best = cur = 0
        for x in ngrams(w, 5):
            cur = cur + 1 if x in ref else 0
            best = max(best, cur)
        return best + 4 if best else 0
    S['ov_all'] = [ov(words(t), all5) for t in S.text]
    S['ov_seed'] = [ov(words(t), g5[s]) for t, s in zip(S.text, S.speaker_id)]
    S['ov_nonseed'] = [ov(words(t), set().union(*(g5[k] for k in g5 if k != s))) for t, s in zip(S.text, S.speaker_id)]
    S['longest_span'] = [longest_run(words(t), all5) for t in S.text]
    other5 = {s: set().union(*(g5[k] for k in g5 if k != s)) for s in R.speaker_id.unique()}
    R['ov_other'] = [ov(words(t), other5[s]) for t, s in zip(R.text, R.speaker_id)]
    R['longest_span_other'] = [longest_run(words(t), other5[s]) for t, s in zip(R.text, R.speaker_id)]
    tot = lambda texts, refs: sum(sum(x in r for x in ngrams(words(t), 5)) for t, r in zip(texts, refs)) / \
        max(sum(len(ngrams(words(t), 5)) for t in texts), 1)
    mem = pd.DataFrame({
        'synthetic → all real': [tot(S.text, [all5] * len(S)), S.ov_all.mean(), S.ov_all.max(), S.longest_span.median(), S.longest_span.max()],
        'synthetic → seed speaker only': [tot(S.text, [g5[s] for s in S.speaker_id]), S.ov_seed.mean(), S.ov_seed.max(), np.nan, np.nan],
        'synthetic → non-seed speakers': [np.nan, S.ov_nonseed.mean(), S.ov_nonseed.max(), np.nan, np.nan],
        'baseline: real → other speakers': [tot(R.text, [other5[s] for s in R.speaker_id]), R.ov_other.mean(), R.ov_other.max(),
                                            R.longest_span_other.median(), R.longest_span_other.max()],
    }, index=['pooled 5-gram overlap', 'per-sample mean', 'per-sample max', 'longest shared span (words, median)',
              'longest shared span (words, max)'])
    M['memorization'] = mem.to_dict()
    out.append('5-gram = 소문자 공백 단어 5연속. 기준선은 실제 응답이 **다른 화자** 발화와 공유하는 비율이다. 구어에는 '
               '"I don\'t know what to"류 상투구가 많으므로 0이 정상치가 아니다. longest span은 연속 5-gram 히트로 계산한 상한이다.\n')
    out.append(mem.round(4).to_markdown() + '\n')

    # 최근접 이웃 거리
    D_sr = 1 - ES @ ER.T
    nn_sr = D_sr.min(1)
    D_rr = 1 - ER @ ER.T
    np.fill_diagonal(D_rr, np.inf)
    nn_rr = D_rr.min(1)
    same = (R.speaker_id.to_numpy()[:, None] == R.speaker_id.to_numpy()[None, :])
    nn_rr_x = np.where(same, np.inf, D_rr).min(1)
    seedmask = S.speaker_id.to_numpy()[:, None] == R.speaker_id.to_numpy()[None, :]
    nn_sr_x = np.where(seedmask, np.inf, D_sr).min(1)
    p5 = np.percentile(nn_rr, 5)
    nn = pd.DataFrame({'median': [np.median(nn_sr), np.median(nn_rr), np.median(nn_sr_x), np.median(nn_rr_x)],
                       'p5': [np.percentile(nn_sr, 5), p5, np.percentile(nn_sr_x, 5), np.percentile(nn_rr_x, 5)]},
                      index=['synthetic → real', 'real → real (excl. self)', 'synthetic → real (excl. seed speaker)',
                             'real → real (excl. same speaker)'])
    p_closer = stats.mannwhitneyu(nn_sr, nn_rr, alternative='less').pvalue
    frac = float((nn_sr < p5).mean())
    M['nn'] = dict(table=nn.to_dict(), p_syn_closer=p_closer, frac_syn_below_real_p5=frac)
    out.append('**임베딩 최근접 이웃 코사인 거리** (작을수록 가까움)\n')
    out.append(nn.round(4).to_markdown())
    out.append(f'\n합성→실제가 실제→실제보다 가까운지 단측 Mann-Whitney p = {p_closer:.3g}; '
               f'실제→실제 5퍼센타일({p5:.3f})보다 가까운 합성 샘플 비율 = {frac:.1%} (기대치 ≈5%).\n')
    two_hist([('nearest-neighbour distance', nn_rr, nn_sr)], 'Nearest real neighbour: synthetic→real vs real→real',
             'cosine distance', os.path.join(FIG, f'{pre}nn.png'))
    out.append(f'![nn](figures/{pre}nn.png)\n')

    # 상위 10 육안 확인
    out.append('### overlap 상위 10개 샘플 (육안 확인용)\n')
    out.append('굵게 = 실제 코퍼스와 공유하는 5-gram에 속한 단어. 실제 측은 공유 5-gram이 가장 많은 실제 턴의 해당 구간(±12단어)이다. '
               '⚠ 이 절에는 원문 발췌가 포함되므로 리포트를 재배포하지 않는다.\n')
    turn_w = [(r.speaker_id, words(r.text)) for r in corpus.itertuples()]
    for x in S.sort_values('ov_all', ascending=False).head(10).itertuples():
        w = words(x.text)
        hit = np.zeros(len(w), bool)
        for i, g in enumerate(ngrams(w, 5)):
            if g in all5:
                hit[i:i + 5] = True
        sg = set(ngrams(w, 5))
        best = max(turn_w, key=lambda t: len(sg & set(ngrams(t[1], 5))))
        bw = best[1]
        bh = np.zeros(len(bw), bool)
        for i, g in enumerate(ngrams(bw, 5)):
            if g in sg:
                bh[i:i + 5] = True
        idx = np.where(bh)[0]
        lo, hi = (max(idx.min() - 12, 0), min(idx.max() + 13, len(bw))) if len(idx) else (0, 0)
        mark = lambda ws, h: ' '.join(f'**{a}**' if b else a for a, b in zip(ws, h))
        out.append(f'**{x.sample_id}** (overlap {x.ov_all:.1%}, seed {x.speaker_id}, 최장 공유 {x.longest_span}단어)\n')
        out.append(f'- 합성: {mark(w, hit)}\n- 실제 [{best[0]}{" = seed" if best[0] == x.speaker_id else ""}]: '
                   f'{"…" if lo else ""}{mark(bw[lo:hi], bh[lo:hi]) if len(idx) else "(공유 5-gram 없음)"}{"…" if hi < len(bw) else ""}\n')

    # ============ 3d Discriminability
    out.append('## 3d. Discriminability (탐지 가능성)\n')
    X = pd.concat([R.text, S.text]).tolist()
    y = np.r_[np.zeros(len(R)), np.ones(len(S))]
    cv = StratifiedKFold(V['cv_folds'], shuffle=True, random_state=SEED)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(vec, LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000))
    auc = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    Lx = np.log(pd.concat([R.n_tokens, S.n_tokens]).to_numpy())[:, None]
    auc_len = cross_val_score(LogisticRegression(class_weight='balanced'), Lx, y, cv=cv, scoring='roc_auc')
    per_g = {}
    for g in ['CL', 'CO']:
        m = np.r_[(R.group == g).to_numpy(), (S.group == g).to_numpy()]
        per_g[g] = cross_val_score(clf, list(np.array(X, object)[m]), y[m], cv=cv, scoring='roc_auc').mean()
    clf.fit(X, y)
    coef = clf[-1].coef_[0]
    vocab = np.array(clf[0].get_feature_names_out())
    o = np.argsort(coef)
    top_syn = ', '.join(f'`{vocab[i]}` ({coef[i]:+.2f})' for i in o[::-1][:20])
    top_real = ', '.join(f'`{vocab[i]}` ({coef[i]:+.2f})' for i in o[:20])
    M['discrim'] = dict(auc_mean=auc.mean(), auc_sd=auc.std(), auc_folds=auc.tolist(), auc_length_only=auc_len.mean(),
                        auc_CL=per_g['CL'], auc_CO=per_g['CO'])
    out.append(f'TF-IDF(단어 1–2gram, min_df=2) + 로지스틱(class_weight=balanced), {V["cv_folds"]}-fold 층화 CV. '
               f'실제 {len(R)} vs 합성 {len(S)}.\n')
    out.append(pd.DataFrame({'AUC': [f'{auc.mean():.3f} ± {auc.std():.3f}', f'{per_g["CL"]:.3f}', f'{per_g["CO"]:.3f}',
                                     f'{auc_len.mean():.3f}']},
                            index=['전체 (TF-IDF)', 'CL만', 'CO만', '기준선: log(토큰 수)만']).to_markdown())
    out.append(f'\n**합성 쪽으로 기우는 상위 피처**: {top_syn}\n\n**실제 쪽으로 기우는 상위 피처**: {top_real}\n')

    title = TITLE
    head = (f'# {title}\n\n재현: `python src/validate.py {TAG}` (seed {SEED}). 실제 응답 {len(R)}개 '
            f'(CL {int((R.group == "CL").sum())}/CO {int((R.group == "CO").sum())}), 합성 {len(S)}개 '
            f'(CL {int((S.group == "CL").sum())}/CO {int((S.group == "CO").sum())}).\n'
            + ('\n파일럿: 실제 쪽도 파일럿 시드 화자 4명으로 제한. 표본이 작아 검정력은 낮다. 이상 징후 확인 용도.\n' if PILOT else '') + '\n')
    name = REPORT_NAME
    open(os.path.join(REP, name), 'w', encoding='utf8').write(head + '\n'.join(out))
    json.dump(M, open(os.path.join(REP, f'{pre}validation_metrics.json'), 'w', encoding='utf8'), ensure_ascii=False,
              indent=1, default=lambda v: v.item() if hasattr(v, 'item') else str(v))
    S[['sample_id', 'speaker_id', 'group', 'temperature', 'ov_all', 'ov_seed', 'longest_span']].to_csv(
        os.path.join(ROOT, SYN_DIR, f'{pre}per_sample_overlap.csv'), index=False)
    print(f'wrote reports/{name}')


if __name__ == '__main__':
    assert abs(mattr(list('aab'), 50) - 2 / 3) < 1e-9 and mattr(['a', 'b'] * 50, 50) == 2 / 50
    assert ngrams('a b c d e f'.split(), 5) == [tuple('abcde'), tuple('bcdef')]
    main()
