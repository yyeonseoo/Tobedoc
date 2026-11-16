"""v4c 사후 탐색: Claude 작성(v4) · GPT 작성(v4b) 발화 모두에서 작동하는 지표 찾기 -> reports/v4/04_robustness.md, figures/fig2_robust.png.

- 평가 기준: 두 세트 모두 GPT-4.1 블라인드 평가(같은 평가자). v4b는 Claude 평가도 있어 평가자 일치도를 함께 본다.
- 지표: TF-IDF 3, SBERT 13(v3), 질문 기준 3(v2), 선행연구 7(v4_literature), 문체 6.
- 분석: A. 세트별 평가 점수와의 Spearman ρ, B. 생성 모델 효과 대 사고장애 효과, C. 한 세트로 학습 -> 다른 세트에서 예측.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from text import CFG, STOP, content, path, words
from v2_notebook_audit import geometry, toks
from v4_features import sbert_rows
from v4_judge import ITEMS, load_items, rate_items
from v4_literature import metrics, sentences

DATA = os.path.dirname(path('processed'))
REP = os.path.join(DATA, '..', 'reports', 'v4')
RATE = ITEMS + ['overall_disorganization']
STYLE = ['n_sent', 'ttr100', 'stop_ratio', 'mean_wlen', 'punct_rate', 'comma_rate']


def load_v4b():
    d = json.load(open(os.path.join(DATA, 'v4b', 'gen_gpt.json'), encoding='utf-8'))
    it = pd.DataFrame(d['items'])
    it['item_id'] = it.q + '_' + it.cond
    it['question'] = it.q.map(d['questions'])
    return it


def style(text):
    tok = toks(text)[:100]
    raw = text.split()[:100]
    head = ' '.join(raw)
    return dict(n_sent=len(sentences(head)), ttr100=len(set(tok)) / len(tok), stop_ratio=np.mean([w in STOP for w in tok]),
                mean_wlen=np.mean([len(w) for w in tok]), punct_rate=sum(c in '.!?' for c in head) / len(raw),
                comma_rate=head.count(',') / len(raw))


def featurize(m, it):
    s = sbert_rows(m, list(zip(it.text.map(content), it.question.map(content))))
    t = pd.DataFrame([geometry(' '.join(toks(x))) for x in it.text])
    lit = pd.DataFrame([metrics(m, x, q) for x, q in zip(it.text, it.question)]).drop(columns='n_sent')
    st = pd.DataFrame([style(x) for x in it.text])
    return pd.concat([it[['item_id', 'cond']].reset_index(drop=True), s, t, lit, st], axis=1)


def rho_table(f, feats):
    return pd.DataFrame({c: {r: stats.spearmanr(f[c], f[r], nan_policy='omit')[0] for r in RATE} for c in feats}).T


def cohen(a, b):
    a, b = a.dropna(), b.dropna()
    sd = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sd


def transfer(train, test, feats, target):
    """train, test는 호출 전에 모든 지표 묶음 기준으로 결측을 한 번에 지운 것(묶음마다 같은 항목으로 비교)."""
    tr, te = train, test
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(tr[feats], tr[target])
    return stats.spearmanr(model.predict(te[feats]), te[target])[0]


def main():
    m = SentenceTransformer(CFG['features']['sbert_model'])
    a = load_items()
    b = load_v4b()
    ra = pd.read_csv(os.path.join(DATA, 'v4', 'judge.csv')).drop(columns='model')
    rb = rate_items(b, os.path.join(DATA, 'v4b', 'judge_gpt.csv'))
    key = json.load(open(os.path.join(DATA, 'v4b', 'blind_key.json')))
    rc = pd.read_csv(os.path.join(DATA, 'v4b', 'judge_claude.csv')).assign(item_id=lambda x: x.id.map(key)).drop(columns='id')
    fa = featurize(m, a).merge(ra, on='item_id').assign(gen='Claude')
    fb = featurize(m, b).merge(rb, on='item_id').assign(gen='GPT')
    feats = [c for c in fa.columns if c not in ['item_id', 'cond', 'gen'] + RATE]

    # 평가자 일치도 (v4b: GPT vs Claude)
    agree = rb.merge(rc, on='item_id', suffixes=('_gpt', '_claude'))
    agree_tab = pd.Series({r: stats.spearmanr(agree[f'{r}_gpt'], agree[f'{r}_claude'])[0] for r in RATE}, name='ρ (GPT vs Claude)')

    # A. 세트별 상관
    A, B = rho_table(fa, feats), rho_table(fb, feats)
    focus = ['overall_disorganization', 'tangentiality', 'derailment', 'incoherence', 'perseveration', 'clanging', 'circumstantiality']
    rows = []
    for c in feats:
        r = dict(feature=c)
        for k in focus:
            r[f'{k[:4]}_C'], r[f'{k[:4]}_G'] = A.loc[c, k], B.loc[c, k]
        rows.append(r)
    both = pd.DataFrame(rows)
    robust = {k: [c for c in feats if np.sign(A.loc[c, k]) == np.sign(B.loc[c, k]) and min(abs(A.loc[c, k]), abs(B.loc[c, k])) >= 0.3]
              for k in focus}

    # B. 생성 모델 효과 (사고장애 없는 발화: N0·NMULTI·CIRC) 대 사고장애 효과 (전체 와해 ≥2 vs 0, 세트 안)
    calm = lambda f: f[f.cond.isin(['N0', 'NMULTI', 'CIRC1', 'CIRC2'])]
    dis = lambda f, c: cohen(f[f.overall_disorganization >= 2][c], f[f.overall_disorganization == 0][c])
    gap = pd.DataFrame({c: dict(gen_d=cohen(calm(fb)[c], calm(fa)[c]), dis_d_Claude=dis(fa, c), dis_d_GPT=dis(fb, c)) for c in feats}).T
    gap['ratio'] = gap.gen_d.abs() / gap[['dis_d_Claude', 'dis_d_GPT']].abs().min(axis=1)
    gap['dis_부호일치'] = np.sign(gap.dis_d_Claude) == np.sign(gap.dis_d_GPT)

    # C. 교차 예측 (Ridge, 세트 하나로 학습 -> 다른 세트에서 Spearman)
    sets = {'전체 지표': feats, '문체 제외': [c for c in feats if c not in STYLE],
            '질문 기준만': ['elv_mean', 'end_sim', 'anchor_first', 'anchor_mean'],
            '문체만': STYLE}
    fa_t, fb_t = fa.dropna(subset=feats), fb.dropna(subset=feats)
    trans = pd.DataFrame([dict(지표묶음=k, 목표=t, Claude_to_GPT=transfer(fa_t, fb_t, v, t), GPT_to_Claude=transfer(fb_t, fa_t, v, t))
                          for k, v in sets.items() for t in ['overall_disorganization', 'tangentiality', 'incoherence']])

    # 그림: 두 세트의 ρ 비교 (전체 와해, 주제 이탈, 지리멸렬)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, k in zip(axes, ['overall_disorganization', 'tangentiality', 'incoherence']):
        ax.scatter(A[k], B[k], s=12, color='#2a78d6')
        for c in feats:
            if max(abs(A.loc[c, k]), abs(B.loc[c, k])) >= 0.35:
                ax.annotate(c, (A.loc[c, k], B.loc[c, k]), fontsize=7)
        ax.axhline(0, color='#999', lw=0.5); ax.axvline(0, color='#999', lw=0.5)
        ax.plot([-1, 1], [-1, 1], color='#ccc', lw=0.5, ls='--')
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_xlabel('ρ in Claude-written (v4)'); ax.set_ylabel('ρ in GPT-written (v4b)'); ax.set_title(k, fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(REP, 'figures', 'fig2_robust.png'), dpi=150); plt.close(fig)

    fa.to_csv(os.path.join(DATA, 'v4', 'features_all.csv'), index=False)
    fb.to_csv(os.path.join(DATA, 'v4b', 'features_all.csv'), index=False)
    fmt = lambda d, idx=False: d.to_markdown(index=idx, floatfmt='.2f')
    md = f"""# v4c 사후 탐색: 두 생성 모델의 발화 모두에서 작동하는 지표

재현: `python src/v4c_robust.py`. Claude 작성 v4 {len(fa)}개, GPT 작성 v4b {len(fb)}개. 정답은 두 세트 모두 **GPT-4.1 블라인드 평가**(같은 평가자, 같은 기준표). 결과를 본 뒤의 탐색이다.

## 0. 평가자 일치도 (v4b, GPT vs Claude 평가)

{fmt(agree_tab.to_frame(), True)}

## A. 세트별 평가 점수와의 Spearman ρ (_C = Claude 작성, _G = GPT 작성)

두 세트에서 방향이 같고 둘 다 |ρ| ≥ 0.3인 지표:

{chr(10).join(f'- **{k}**: {", ".join(v) if v else "없음"}' for k, v in robust.items())}

{fmt(both)}

![](figures/fig2_robust.png)

## B. 세트 차이 대 사고장애 효과 (Cohen's d)

gen_d = 사고장애 없는 발화(N0·NMULTI·CIRC)에서 GPT 세트 − Claude 세트(생성 모델, 인물, 질문이 모두 다름). dis_d = 세트 안에서 전체 와해 ≥ 2 대 0. ratio = |gen_d| / (두 dis_d 중 작은 절댓값). ratio가 1보다 크면 같은 정상 발화끼리의 문체 차이가 사고장애로 인한 차이보다 크다.

{fmt(gap.sort_values('ratio'), True)}

dis_부호일치가 False인 지표는 두 세트에서 사고장애 효과의 방향이 반대라 ratio가 작아도 안정적이지 않다. Cohen's d는 표본 수로 가중한 합동 표준편차를 쓴다. 이 비교는 생성 모델, 인물, 질문이 함께 다른 두 세트를 비교한 것이라, 생성 모델 효과만 분리한 결과는 [08_generator.md](08_generator.md)에 있다.

## C. 교차 예측 (Ridge, 한 세트로 학습해 다른 세트의 평가 점수와 Spearman ρ)

모든 묶음에서 같은 항목으로 비교하도록, 지표 중 하나라도 결측인 발화(문장이 3개 미만이라 문장 응집성을 계산할 수 없는 발화)는 미리 뺐다(Claude {len(fa_t)}개, GPT {len(fb_t)}개).

{fmt(trans)}

## 해석

- **모든 사고장애 종류와 두 생성 모델에서 함께 작동하는 단일 지표는 없다.** 전체 와해에서 두 세트 모두 |ρ| ≥ 0.3인 지표는 질문 기준 지표 2개(elv_mean, anchor_mean)와 문체 지표 2개(stop_ratio, mean_wlen)뿐이다. 지리멸렬은 두 세트에서 함께 작동하는 지표가 없다.
- **사고장애 종류마다 맞는 지표가 다르다.** 질문 기준 지표는 주제 이탈·탈선, 반복 지표(mattr, 문장 응집성)는 보속, clang_rate는 음향 연상을 잡는다. 하나의 "이탈 점수"로 와해된 언어 전체를 재려는 설계 자체가 맞지 않는다.
- **질문 기준 지표는 세트에 따라 기준점이 움직인다.** 사고장애가 없는 발화끼리도 두 세트의 elv_mean 차이(d = {gap.loc['elv_mean', 'gen_d']:.2f})가 GPT 세트 안의 사고장애 차이(d = {gap.loc['elv_mean', 'dis_d_GPT']:.2f})보다 크다. 다만 두 세트는 생성 모델, 인물, 질문이 모두 달라 이 차이를 생성 모델 탓으로 돌릴 수 없다. 인물과 질문을 맞춘 비교([08_generator.md](08_generator.md))에서는 생성 모델 효과가 사고장애 효과보다 작았다. GPT 세트의 사고장애 효과가 작게 잡힌 것은 그 세트에서 사고장애 단계가 제대로 나뉘지 않은 탓도 있다. 출처가 다른 데이터를 하나의 기준값으로 비교하지 말고, 같은 사람, 같은 면담 안에서 비교하거나 데이터마다 정상 대조군으로 기준을 맞춰야 한다는 결론은 유지된다.
- **교차 예측은 중간 수준이다**(전체 지표 ρ {trans.iloc[0].Claude_to_GPT:.2f}, {trans.iloc[0].GPT_to_Claude:.2f}). 질문 기준 지표만으로는 GPT → Claude 방향이 더 잘 된다(전체 와해 {trans.iloc[6].GPT_to_Claude:.2f}, 주제 이탈 {trans.iloc[7].GPT_to_Claude:.2f}, 반대 방향 {trans.iloc[6].Claude_to_GPT:.2f}, {trans.iloc[7].Claude_to_GPT:.2f}). 지리멸렬은 어느 묶음으로도 잘 옮겨지지 않았고(최대 {trans[trans.목표 == 'incoherence'][['Claude_to_GPT', 'GPT_to_Claude']].max().max():.2f}), 의미 지표만으로는 0 근처다(문체 제외 {trans.iloc[5].Claude_to_GPT:.2f}, {trans.iloc[5].GPT_to_Claude:.2f}).
- **평가자 사이에서도 주제 이탈(ρ = {agree_tab['tangentiality']:.2f})과 우회증({agree_tab['circumstantiality']:.2f})은 일치도가 낮다.** 정답 자체가 흔들리는 항목이라, 이 두 항목에서 지표가 약하게 나오는 일부는 평가의 한계다.
"""
    open(os.path.join(REP, '04_robustness.md'), 'w', encoding='utf8').write(md)
    print(md)


if __name__ == '__main__':
    assert abs(cohen(pd.Series([1., 2, 3]), pd.Series([1., 2, 3]))) < 1e-9
    main()
