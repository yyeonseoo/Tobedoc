"""v4f: 국소 사고장애에 대한 구간 지표 판정 (생성 발화 + DAIS-C) -> reports/v4/07_local.md. 계획: 00_plan.md 'v4f 계획'.

지표(내용어 기준, 5 내용어 창): q_mean, q_min, q_dip = q_mean − q_min, jump_max(겹치지 않는 5 내용어 조각 사이 최대 거리).
"""
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer

from text import CFG, content, path

DATA = os.path.dirname(path('processed'))
REP = os.path.join(DATA, '..', 'reports', 'v4')
W = 5
SIGN = dict(q_mean=-1, q_min=-1, q_dip=1, jump_max=1)   # 사고장애일수록 예상되는 방향
TYPES = ['DER', 'TAN', 'INC', 'PER', 'CLANG']


class Enc:
    def __init__(self, m):
        self.m, self.cache = m, {}

    def __call__(self, texts):
        new = [t for t in set(texts) if t not in self.cache]
        if new:
            self.cache.update(zip(new, self.m.encode(new, normalize_embeddings=True, batch_size=256)))
        return np.stack([self.cache[t] for t in texts])


def local_metrics(enc, w, a):
    """w: 내용어 목록, a: 질문 내용어. 창 유사도와 조각 경계 거리, 각 최솟값/최댓값 위치(단어 범위)."""
    wins = [' '.join(w[i:i + W]) for i in range(len(w) - W + 1)]
    sim = enc(wins) @ enc([' '.join(a)])[0]
    ch = [' '.join(w[i:i + W]) for i in range(0, len(w) - W + 1, W)]
    E = enc(ch)
    jumps = 1 - np.sum(E[1:] * E[:-1], axis=1)
    i, j = int(np.argmin(sim)), int(np.argmax(jumps))
    return dict(q_mean=sim.mean(), q_min=sim.min(), q_dip=sim.mean() - sim.min(), jump_max=jumps.max(),
                min_span=(i, i + W), jump_span=(j * W, (j + 2) * W), n_win=len(wins), n_chunk_pairs=len(jumps))


def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def chance(n_words, span, width, step):
    """같은 답변에서 위치를 무작위로 골랐을 때 표시 구간과 겹칠 확률."""
    starts = range(0, n_words - width + 1, step)
    return np.mean([overlap((s, s + width), span) for s in starts])


def gen_items(enc):
    d = json.load(open(os.path.join(DATA, 'v4f', 'gen_local.json'), encoding='utf-8'))
    rows = []
    for it in d['items']:
        raw = it['text']
        m = re.search(r'\[\[(.+?)\]\]', raw, re.S)
        clean = raw.replace('[[', '').replace(']]', '')
        w = content(clean)
        span = None
        if m:
            s = len(content(raw[:m.start()]))
            span = (s, s + len(content(m.group(1))))
        f = local_metrics(enc, w, content(d['questions'][it['q']]))
        rows.append(dict(q=it['q'], rep=it['rep'], cond=it['cond'], span=span, n_words=len(w), clean=clean, **f))
    return pd.DataFrame(rows)


def blind(g):
    """조건 모르는 Claude 에이전트 평가(data/v4f/judge_claude.csv)를 붙이고, 평가자가 고른 구간의 내용어 위치를 구한다."""
    key = json.load(open(os.path.join(DATA, 'v4f', 'blind_key.json')))
    j = pd.read_csv(os.path.join(DATA, 'v4f', 'judge_claude.csv')).fillna({'passage': ''})
    j[['q', 'cond', 'rep']] = pd.DataFrame([[key[i]['q'], key[i]['cond'], key[i]['rep']] for i in j.id])
    g = g.merge(j, on=['q', 'cond', 'rep'], validate='1:1')
    spans = []
    for r in g.itertuples():
        i = r.clean.find(r.passage) if r.passage else -1
        s0 = len(content(r.clean[:i])) if i >= 0 else None
        spans.append((s0, s0 + len(content(r.passage))) if s0 is not None else None)
    return g.assign(rater_span=spans)


def paired(g):
    base = g[g.cond == 'N0'].set_index(['q', 'rep'])
    rows = []
    for cond in TYPES + ['ALL', 'CIRC']:
        # ALL: 같은 정상 답변이 다섯 종류에 반복해 쓰이지 않도록 질문·회차별로 다섯 종류를 평균한 뒤 비교
        sub = g[g.cond.isin(TYPES)].groupby(['q', 'rep'])[list(SIGN)].mean().reset_index() if cond == 'ALL' else g[g.cond == cond]
        for c, s in SIGN.items():
            diff = s * (sub.set_index(['q', 'rep'])[c] - base[c].reindex(sub.set_index(['q', 'rep']).index)).dropna()
            k = int((diff > 0).sum())
            rows.append(dict(cond=cond, feature=c, pairs=len(diff), expected=k, sign_p=stats.binomtest(k, len(diff)).pvalue))
    return pd.DataFrame(rows)


def localize(g, span_col, kind, width, step):
    loc = g[g.cond.isin(TYPES)]
    hit = np.array([overlap(r[kind], r[span_col]) for _, r in loc.iterrows()])
    ch = np.array([chance(r.n_words, r[span_col], width, step) for _, r in loc.iterrows()])
    return dict(n=len(loc), hit_rate=hit.mean(), chance=ch.mean(), p=stats.binomtest(int(hit.sum()), len(hit), ch.mean(), alternative='greater').pvalue)


def daisc(enc, feats):
    d = pd.read_parquet(os.path.join(path('processed'), 'daisc_units.parquet'))
    d = d[(d.n_words >= 15) & (d.anchor_words.map(len) >= 2)].reset_index(drop=True)
    assert all(len(a) == len(w) for a, w in zip(d.in_DT, d.words))
    rows = []
    for _, r in d.iterrows():
        f = local_metrics(enc, list(r.words), list(r.anchor_words))
        dt = np.flatnonzero(r.in_DT)
        rows.append(dict(speaker_id=r.speaker_id, dt=len(dt) > 0, span=(int(dt.min()), int(dt.max()) + 1) if len(dt) else None,
                         n_words=len(r.words), **f))
    f = pd.DataFrame(rows)
    res = []
    for c in feats:
        g = f[f.groupby('speaker_id').dt.transform(lambda s: s.any() and not s.all())]
        t = g.groupby(['speaker_id', 'dt'])[c].mean().unstack().dropna()
        diff = SIGN[c] * (t[True] - t[False])
        k = int((diff > 0).sum())
        res.append(dict(feature=c, speakers=len(diff), expected=k, sign_p=stats.binomtest(k, len(diff)).pvalue,
                        wilcoxon_p=stats.wilcoxon(diff).pvalue))
    loc = pd.DataFrame([dict(지표=k, **localize(f[f.dt].assign(cond='DER'), 'span', s, wd, st))
                        for k, s, wd, st in [('q_min 창', 'min_span', W, 1), ('jump_max 경계', 'jump_span', 2 * W, W)]])
    return pd.DataFrame(res), loc, f


def fig3(loc, by_type, dloc):
    """위치 맞히기: 질문과 가장 먼 창이 사고장애 구간과 겹친 비율(점)과 우연 수준(회색 막대)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    name = dict(TAN='주제 이탈', DER='탈선', CLANG='음향 연상', INC='지리멸렬', PER='보속')
    rows = [('생성 발화: 전체', loc.loc[0])] + [(f'  {name[r.cond]}', r) for r in by_type.itertuples()] + ([('DAIS-C 실제 면담', dloc.loc[0])] if len(dloc) else [])
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(rows) + 0.9))
    for y, (lab, r) in enumerate(rows[::-1]):
        ax.barh(y, r['chance'] if isinstance(r, pd.Series) else r.chance, height=0.5, color='#d9d9d9')
        h = r['hit_rate'] if isinstance(r, pd.Series) else r.hit_rate
        ax.plot(h, y, 'o', color='#2a6fb0', ms=8)
        ax.text(h + 0.015, y, f'{h:.0%}', va='center', fontsize=9, color='#333')
    ax.set_yticks(range(len(rows)), [l for l, _ in rows[::-1]])
    ax.set_xlim(0, 0.8); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.set_xlabel('질문과 가장 먼 구간이 사고장애 구간과 겹친 비율 (점), 우연 수준 (회색)')
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(REP, 'figures', 'fig3_localize.png'), dpi=150); plt.close(fig)


def main():
    enc = Enc(SentenceTransformer(CFG['features']['sbert_model']))
    g = blind(gen_items(enc))
    own = dict(DER='derailment', TAN='tangentiality', INC='incoherence', PER='perseveration', CLANG='clanging', CIRC='circumstantiality')
    found = g[g.rater_span.notna() & g.span.notna()]
    bt = pd.DataFrame([dict(조건=c, n=len(x), 전체와해_평균=x.overall_disorganization.mean(), 전체와해_1점이상=(x.overall_disorganization > 0).mean(),
                            해당항목_1점이상=(x[own[c]] > 0).mean() if c in own else np.nan,
                            탈선항목_1점이상=(x.derailment > 0).mean(), 평가자구간_있음=(x.passage != '').mean(),
                            표시구간과_겹침=np.mean([overlap(a, b) for a, b in zip(found[found.cond == c].rater_span, found[found.cond == c].span)]) if c in TYPES else np.nan)
                       for c, x in g.groupby('cond')]).set_index('조건').reindex(['N0', 'CIRC'] + TYPES)
    conf = g[g.cond.isin(TYPES) & (g.overall_disorganization > 0)]
    loc_conf = localize(conf, 'span', 'min_span', W, 1)
    pr = paired(g)
    loc = pd.DataFrame([dict(지표=k, **localize(g, 'span', s, wd, st))
                        for k, s, wd, st in [('q_min 창', 'min_span', W, 1), ('jump_max 경계', 'jump_span', 2 * W, W)]])
    by_type = pd.DataFrame([dict(cond=t, **localize(g[g.cond == t], 'span', 'min_span', W, 1)) for t in TYPES])
    # 생성 발화에서 통과: 사고장애 5종 합친 답변 수준 비교 p < .05 이고 예상 방향 과반, 또는 위치 맞히기 p < .05
    allp = pr[pr.cond == 'ALL'].set_index('feature')
    passed = [c for c in SIGN if allp.loc[c, 'sign_p'] < 0.05 and allp.loc[c, 'expected'] > allp.loc[c, 'pairs'] / 2]
    if loc.loc[0, 'p'] < 0.05 and 'q_min' not in passed:
        passed.append('q_min')
    if loc.loc[1, 'p'] < 0.05 and 'jump_max' not in passed:
        passed.append('jump_max')
    dres, dloc, df = daisc(enc, passed) if passed else (pd.DataFrame(), pd.DataFrame(), None)
    piv = pr.pivot(index='feature', columns='cond', values='expected').reindex(columns=TYPES + ['ALL', 'CIRC'])
    pivp = pr.pivot(index='feature', columns='cond', values='sign_p').reindex(columns=TYPES + ['ALL', 'CIRC'])
    fmt = lambda x: x.to_markdown(floatfmt='.3f')
    md = f"""# v4f: 미묘한 국소 사고장애와 구간 지표

재현: `python src/v4f_generate.py`, `python src/v4f_local.py`. 계획: [00_plan.md](00_plan.md) 'v4f 계획'.

GPT-4.1 작성 {len(g)}개 = 질문 8 × 조건 7(N0, CIRC, 국소 DER·TAN·INC·PER·CLANG) × 2회. 국소 조건은 답변 대부분이 정상이고 한두 문장(표시 구간 내용어 중앙값 {int(np.median([s[1] - s[0] for s in g.span.dropna()]))}개, 답변 내용어 중앙값 {int(g.n_words.median())}개)만 사고장애다. 지표는 내용어 기준 5개 창.

## 0. 생성 검증: 조건 모르는 평가자 (Claude 에이전트, 표시 괄호를 지운 112개를 섞어 평가)

평가 기준은 v4·v4b와 같은 TLC 요약이고, 가장 이상한 한두 문장을 그대로 옮겨 적게 했다. '표시구간과_겹침'은 평가자가 옮겨 적은 구간이 생성 때 표시한 구간과 겹친 비율이다.

{bt.to_markdown(floatfmt='.2f')}

- 정상과 우회증은 모두 0점이었고, 평가자가 이상한 구간을 고른 답변은 모두 표시 구간과 겹쳤다({len(found)}개). 표시 구간을 정답으로 써도 된다.
- 다만 미묘하게 쓴 만큼 탈선·주제 이탈은 절반만 전체 와해 1점 이상이었고, **국소 주제 이탈은 평가자가 '주제 이탈'이 아니라 '탈선'으로 봤다.** TLC에서 주제 이탈은 질문에 대한 답 전체가 비껴가는 것이라, 답변 중간 한두 문장의 이탈은 정의상 탈선에 가깝다.
- 평가자가 전체 와해 1점 이상으로 본 국소 답변 {len(conf)}개만으로 위치 맞히기를 다시 하면 q_min 창 적중 {loc_conf['hit_rate']:.0%}, 우연 {loc_conf['chance']:.0%} (p = {loc_conf['p']:.3f})로 결론이 같다.

## 1. 답변 수준: 같은 질문·회차의 정상(N0) 대비 예상 방향인 쌍의 수 (종류별 {int(pr[pr.cond == 'DER'].pairs.iloc[0])}쌍, ALL은 질문·회차별로 다섯 종류를 평균해 {int(pr[pr.cond == 'ALL'].pairs.iloc[0])}쌍)

{fmt(piv)}

부호 검정 p:

{fmt(pivp)}

CIRC 열은 우회증(대조)이 정상보다 '사고장애 쪽'으로 나온 쌍의 수다. 많을수록 우회증을 사고장애로 오인한다.

## 2. 위치: 지표가 고른 구간이 표시 구간과 겹치는 비율 (국소 사고장애 {int((g.cond.isin(TYPES)).sum())}개)

{loc.to_markdown(index=False, floatfmt='.3f')}

![위치 맞히기](figures/fig3_localize.png)

q_min 창의 종류별 적중:

{by_type.to_markdown(index=False, floatfmt='.3f')}

## 3. DAIS-C (생성 발화에서 통과한 지표: {', '.join(passed) if passed else '없음'})

{('같은 사람 안 DT 답변 대 일반 답변:' + chr(10) + chr(10) + dres.to_markdown(index=False, floatfmt='.3f') + chr(10) + chr(10) + f'DT 답변 안에서 위치 맞히기 (DT 답변 {int(df.dt.sum())}개, 표시 구간은 첫 DT 단어부터 마지막 DT 단어까지):' + chr(10) + chr(10) + dloc.to_markdown(index=False, floatfmt='.3f')) if passed else '해당 없음'}

## 4. 해석

- **사고장애가 미묘하고 일부 구간에만 있으면, 답변 단위 지표로는 정상 답변과 구분되지 않는다.** 다섯 종류를 평균해도 예상 방향인 쌍이 절반 안팎이었다(q_min {int(allp.loc['q_min', 'expected'])}/{int(allp.loc['q_min', 'pairs'])}, q_dip {int(allp.loc['q_dip', 'expected'])}/{int(allp.loc['q_dip', 'pairs'])}). 정상 답변에도 질문과 덜 관련된 구간이 늘 있어서, 짧은 이탈 하나는 그 변동에 묻힌다. DAIS-C에서 답변 단위 차이가 없었던 것(06)과 같은 현상이다.
- **위치는 어느 정도 맞힌다.** 답변 안에서 질문과 가장 먼 창(q_min)이 표시 구간과 겹친 비율은 {loc.loc[0, 'hit_rate']:.0%}로 우연 수준({loc.loc[0, 'chance']:.0%})보다 높았다. 특히 주제 이탈({by_type.set_index('cond').loc['TAN', 'hit_rate']:.0%})과 탈선({by_type.set_index('cond').loc['DER', 'hit_rate']:.0%})에서 높았고, 보속에서는 우연 수준 이하였다. "이 답변이 사고장애인가"는 못 가려도, "이상하다면 어디가 이상한가"는 일부 짚는다.
- **우회증을 사고장애로 오인하지는 않았다**(CIRC 열이 절반 안팎).
- **DAIS-C에서는 위치 맞히기도 우연 수준과 차이가 없었다**(표시 답변 30개). 실제 표시 구간은 첫 표시 단어부터 마지막 표시 단어까지로 넓게 잡혀 있고, 표시 기준도 알 수 없어 정답 자체가 거칠다. 또 DAIS-C의 질문은 직전 면담자 발화 전체라 생성 발화의 짧은 질문보다 길고 잡음이 많다.

"""
    fig3(loc, by_type, dloc)
    open(os.path.join(REP, '07_local.md'), 'w', encoding='utf8').write(md)
    print(md)


if __name__ == '__main__':
    assert overlap((0, 5), (4, 9)) and not overlap((0, 5), (5, 9))
    assert abs(chance(10, (0, 5), 5, 1) - 5 / 6) < 1e-9
    main()
