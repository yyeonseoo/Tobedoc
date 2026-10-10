"""주 분석 지표: 답변 전체를 10단어 조각으로 끝까지 나눠 질문과의 거리를 위치별로 구한다.

앞 4문장 방식(시행착오)은 답변의 약 절반만 보고(CL 53%, CO 68%) 그룹별 비율이 달라 폐기했다.
출력:
- data/processed/qa_chunks.parquet   조각 단위 (혼합효과모형 '거리 ~ 그룹 x 위치' 입력)
- data/processed/qa_features.parquet 답변 단위 지표
- reports/figures/fig5~fig8 *.png, reports/03_features.md
기술통계만 낸다. 그룹 검정은 분석 계획을 고정한 뒤 별도로 한다.
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer

from preprocess import CFG, ROOT
from summary_figures import BG, C, INK, INK2, legend, mattr, save

MAXW = 150  # 곡선 표시 범위 (단어). 그 뒤는 화자 수가 10명 안팎으로 줄어듦
LANG = re.compile(r'creativ|languag|\bwords?\b|speak|speech|writ|metaphor|phrase|express|communicat|poe|pun|spoonerism|rhym|talk about', re.I)
EXP = re.compile(r'experiment|\btask|\bitems?\b|\bblocks?\b|\brules?\b|\bpattern|\btest\b|\boptions?\b', re.I)


def topic(q):
    if LANG.search(q):
        return 'language'
    return 'experiment' if EXP.search(q) else 'other'


Q = CFG['qa']
W, MAXC = Q['chunk_words'], Q['max_chunks']
FILLERS = set(Q['fillers'])


def chunks(text):
    w = text.split()
    out = [' '.join(w[i:i + W]) for i in range(0, len(w), W)]
    if len(out) > 1 and len(out[-1].split()) < Q['min_last_chunk']:
        out = out[:-1]
    return out


def slope(y):
    return np.polyfit(np.arange(len(y)), y, 1)[0] if len(y) >= 2 else np.nan


def build():
    qa = pd.read_parquet(os.path.join(ROOT, 'data/processed/qa_units.parquet'))
    qa = qa[qa.anchor.notna()].copy()
    qa['chunks'] = qa.text.map(chunks)
    u = qa[qa.chunks.str.len() >= Q['min_chunks']].reset_index(drop=True)
    m = SentenceTransformer(Q['sbert_model'])
    Qe = m.encode(u.anchor.tolist(), normalize_embeddings=True)
    flat = [c for cs in u.chunks for c in cs[:MAXC]]
    Ef = m.encode(flat, normalize_embeddings=True, batch_size=128)
    rows, crow, k, cens = [], [], 0, []
    for i, r in u.iterrows():
        n = min(len(r.chunks), MAXC)
        E, k = Ef[k:k + n], k + n
        qd = 1 - E @ Qe[i]
        cen = E.mean(0) / np.linalg.norm(E.mean(0))
        cens.append(cen)
        meta = dict(speaker_id=r.speaker_id, group=r.group, answer_idx=r.answer_idx, sex=r.sex,
                    start_question=r.start_question, q_words=len(r.anchor.split()), n_words=len(r.text.split()),
                    n_chunks=len(r.chunks), used_chunks=n, topic=topic(r.anchor), tag_DT=r.tag_DT)
        crow += [dict(meta, pos=p + 1, words_end=(p + 1) * W, qdist=qd[p]) for p in range(n)]
        allw = r.text.lower().split()
        rows.append(dict(meta, q_slope=slope(qd), q_first=qd[0], q_mean=1 - cen @ Qe[i],
                         first_slope=slope(1 - E[1:] @ E[0]), adj=np.mean(1 - np.sum(E[1:] * E[:-1], axis=1)),
                         slope100=slope(qd[:10]) if n >= 10 else np.nan,
                         slope150=slope(qd[:15]) if n >= 15 else np.nan,
                         mattr=mattr(allw, 50), filler_rate=sum(w in FILLERS for w in allw) / len(allw)))
    f = pd.DataFrame(rows)
    # 상대 관련성: 자기 질문 거리 - 같은 화자의 다른 질문들과의 평균 거리 (음수일수록 자기 질문에 더 가까움)
    f['q_rel'] = [f.q_mean[i] - np.mean([1 - cens[i] @ Qe[j] for j in f.index[f.speaker_id == f.speaker_id[i]] if j != i])
                  if (f.speaker_id == f.speaker_id[i]).sum() > 1 else np.nan for i in f.index]
    ch = pd.DataFrame(crow)
    f.to_parquet(os.path.join(ROOT, 'data/processed/qa_features.parquet'), index=False)
    ch.to_parquet(os.path.join(ROOT, 'data/processed/qa_chunks.parquet'), index=False)
    return f, ch, qa


def fig_units(f):
    n = f.groupby(['group', 'speaker_id']).size().reset_index(name='n').sort_values('n')
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.barh(n.speaker_id, n.n, color=[C[g] for g in n.group], height=0.7)
    ax.set_xlabel(f'분석 답변 수 (30단어 이상, 질문 있음)')
    ax.set_title(f'화자별 분석 답변 수: CL {n[n.group == "CL"].n.sum()}개, CO {n[n.group == "CO"].n.sum()}개',
                 loc='left', color=INK)
    ax.grid(axis='y', visible=False)
    legend(ax, loc='lower right')
    save(fig, 'fig5_units_per_speaker.png')


def fig_features(f):
    labels = {'q_slope': '질문 기준 이탈 기울기 (답변 전체)', 'q_first': '질문↔첫 조각 거리',
              'q_mean': '질문↔답변 평균 거리', 'q_rel': '상대 관련성 (자기 - 다른 질문)',
              'first_slope': '답변 내 이탈 기울기 (첫 조각 기준)', 'adj': '인접 조각 거리',
              'mattr': '어휘 다양성 MATTR (답변 전체)', 'filler_rate': '더듬기 비율 (er/erm 등)',
              'n_words': '답변 길이 (단어)'}
    sp = f.groupby(['group', 'speaker_id'])[list(labels)].mean().reset_index()
    fig, axes = plt.subplots(3, 3, figsize=(10, 8.5))
    rng = np.random.default_rng(42)
    for ax, (c, lab) in zip(axes.flat, labels.items()):
        for k, g in enumerate(C):
            v = sp[sp.group == g][c].values
            ax.scatter(k + rng.uniform(-0.12, 0.12, len(v)), v, s=36, color=C[g], edgecolor=BG, linewidth=1.5, zorder=3)
            ax.hlines(np.median(v), k - 0.25, k + 0.25, color=INK, linewidth=2, zorder=4)
        ax.set_xticks([0, 1], ['CL', 'CO'])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(lab, loc='left', fontsize=9.5, color=INK)
        ax.grid(axis='x', visible=False)
    fig.suptitle(f'답변 단위 지표 (점 = 화자 평균, 가로선 = 그룹 중앙값, 화자 {sp.speaker_id.nunique()}명)',
                 x=0.01, ha='left', color=INK)
    fig.tight_layout()
    save(fig, 'fig6_features.png')
    return sp


def fig_curve(ch):
    # 화자별로 위치마다 평균을 낸 뒤 그룹 평균과 95% CI (화자가 단위). 화자 3명 미만인 위치는 그리지 않음
    sp = ch.groupby(['group', 'speaker_id', 'words_end']).qdist.mean().reset_index()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    for g in C:
        s = sp[sp.group == g].groupby('words_end').qdist.agg(['mean', 'std', 'count'])
        s = s[s['count'] >= 3]
        half = stats.t.ppf(0.975, s['count'] - 1) * s['std'] / np.sqrt(s['count'])
        ax.fill_between(s.index, s['mean'] - half, s['mean'] + half, color=C[g], alpha=0.15, linewidth=0)
        ax.plot(s.index, s['mean'], color=C[g], linewidth=2)
        ax2.plot(s.index, s['count'], color=C[g], linewidth=2)
    ax.set_ylabel('질문과의 거리 (1 - 코사인)')
    ax.set_title('말한 분량에 따른 질문과의 거리 (선 = 화자 평균의 그룹 평균, 띠 = 95% CI)', loc='left', color=INK)
    legend(ax, loc='lower right')
    ax2.set_ylabel('화자 수')
    ax2.set_xlabel('답변 시작부터 말한 단어 수 (10단어 조각의 끝 위치)')
    ax2.set_ylim(0, 15)
    save(fig, 'fig7_distance_curve.png')


def curve(ax, df, y, key, colors, labels, min_spk=3):
    """화자별 위치 평균 -> 그룹 평균과 95% CI. 화자가 min_spk 미만인 위치는 생략."""
    sp = df.groupby([key, 'speaker_id', 'words_end'])[y].mean().reset_index()
    for k, col in colors.items():
        s = sp[sp[key] == k].groupby('words_end')[y].agg(['mean', 'std', 'count'])
        s = s[s['count'] >= min_spk]
        half = stats.t.ppf(0.975, s['count'] - 1) * s['std'] / np.sqrt(s['count'])
        ax.fill_between(s.index, s['mean'] - half, s['mean'] + half, color=col, alpha=0.15, linewidth=0)
        ax.plot(s.index, s['mean'], color=col, linewidth=2, label=f'{labels[k]} ({sp[sp[key] == k].speaker_id.nunique()}명)')
    ax.legend(frameon=False, loc='lower right', fontsize=9)


def fig_checks(ch):
    """검정 전 확인: (a) 같은 주제 질문끼리 (b) 출발점(첫 30단어 평균) 맞춤 (c) CL에서 연구자 DT 표시 여부."""
    ch = ch[ch.words_end <= MAXW].copy()
    base = ch[ch.pos <= 3].groupby(['speaker_id', 'answer_idx']).qdist.mean().rename('base')
    ch = ch.join(base, on=['speaker_id', 'answer_idx'])
    ch['qdist_b'] = ch.qdist - ch.base
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    lab = {'CL': 'CL', 'CO': 'CO'}
    curve(axes[0], ch[ch.topic == 'language'], 'qdist', 'group', C, lab)
    axes[0].set_title('(a) 언어·창의성 질문에 대한 답변만', loc='left', color=INK)
    axes[0].set_ylabel('질문과의 거리')
    curve(axes[1], ch[ch.pos > 3], 'qdist_b', 'group', C, lab)
    axes[1].axhline(0, color=INK2, linewidth=0.8)
    axes[1].set_title('(b) 첫 30단어 평균 대비 이후 거리 변화', loc='left', color=INK)
    axes[1].set_ylabel('첫 30단어 평균 대비 거리 증가')
    cl = ch[ch.group == 'CL'].assign(dt=lambda d: np.where(d.tag_DT > 0, 'DT', 'none'))
    curve(axes[2], cl, 'qdist', 'dt', {'DT': '#4a3aa7', 'none': '#8a8984'}, {'DT': 'DT 표시 있음', 'none': 'DT 표시 없음'})
    axes[2].set_title('(c) CL: 연구자 DT(담화 추적) 표시 여부', loc='left', color=INK)
    axes[2].set_ylabel('질문과의 거리')
    for ax in axes:
        ax.set_xlabel('답변 시작부터 말한 단어 수')
    fig.tight_layout()
    save(fig, 'fig8_curve_checks.png')
    return ch


def dval(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)


def main():
    f, ch, qa = build()
    used = f.groupby('group').used_chunks.sum() * W
    total = qa.groupby('group').text.apply(lambda s: s.str.split().str.len().sum())
    fig_units(f)
    sp = fig_features(f)
    fig_curve(ch)
    chb = fig_checks(ch)
    cols = [c for c in sp.columns if c not in ('group', 'speaker_id')]
    tab = {c: dict(CL=sp[sp.group == 'CL'][c].median(), CO=sp[sp.group == 'CO'][c].median(),
                   d=dval(sp[sp.group == 'CL'][c], sp[sp.group == 'CO'][c]), 화자=f'{sp.speaker_id.nunique()}명') for c in cols}
    sp2 = f.groupby(['group', 'speaker_id'])[['slope100', 'slope150']].mean().reset_index()
    for c in ['slope100', 'slope150']:
        a, b = sp2[sp2.group == 'CL'][c].dropna(), sp2[sp2.group == 'CO'][c].dropna()
        tab[c] = dict(CL=a.median(), CO=b.median(), d=dval(a, b), 화자=f'{len(a)}/{len(b)}명')
    tab = pd.DataFrame(tab).T
    # 확인 수치: 같은 주제 앞 100단어 기울기, 30단어 이후 기울기, CL 화자 내 DT 비교
    lang = f[f.topic == 'language'].groupby(['group', 'speaker_id']).slope100.mean().dropna().reset_index()
    late = chb[chb.pos > 3].groupby(['group', 'speaker_id', 'answer_idx']).filter(lambda g: len(g) >= 3)
    late = late.groupby(['group', 'speaker_id', 'answer_idx']).apply(
        lambda g: np.polyfit(g.pos, g.qdist, 1)[0], include_groups=False).groupby(['group', 'speaker_id']).mean().reset_index(name='s')
    dt = f[f.group == 'CL'].assign(dt=f.tag_DT > 0).groupby(['speaker_id', 'dt']).q_slope.mean().unstack().dropna()
    topics = pd.crosstab(f.topic, f.group)
    checks = [
        ('(a) 언어·창의성 질문 답변만, 앞 100단어 기울기', f'CL {(lang.group == "CL").sum()}명 / CO {(lang.group == "CO").sum()}명',
         f'{dval(lang[lang.group == "CL"].slope100, lang[lang.group == "CO"].slope100):+.2f}'),
        ('(b) 30단어 이후 구간 기울기 (40-150단어)', f'CL {(late.group == "CL").sum()}명 / CO {(late.group == "CO").sum()}명',
         f'{dval(late[late.group == "CL"].s, late[late.group == "CO"].s):+.2f}'),
        ('(c) CL 화자 내: DT 있는 답변 - 없는 답변 기울기', f'{len(dt)}명',
         f'{int((dt[True] > dt[False]).sum())}명에서 DT 쪽이 큼, 중앙값 차 {(dt[True] - dt[False]).median():+.4f}'),
    ]
    md = f"""# 03. 답변 단위 지표와 검정 전 확인

재현: `python src/position_features.py` → `data/processed/qa_chunks.parquet`, `qa_features.parquet`, `reports/figures/fig5-8`.
기술통계만 담았다. 그룹 검정은 분석 계획을 고정한 뒤 한다.

## 설정
- 질문 있는 답변을 처음부터 끝까지 {W}단어 조각으로 나눈다. 마지막 조각이 {Q['min_last_chunk']}단어 미만이면 버린다.
- 조각 {Q['min_chunks']}개(약 30단어) 이상인 답변만 쓰고, 위치는 {MAXC}조각({MAXC * W}단어)까지만 쓴다.
- 임베딩: {Q['sbert_model']}. 거리 = 1 - 코사인.
- 시행착오: 처음에는 spaCy 문장 기준 앞 4문장을 썼다. 답변의 중앙값 53%(CL), 68%(CO)만 보고, 앞 4문장의 단어 수도 CL 60 대 CO 80으로 달라 폐기했다.

## 규모
- 답변 CL {int((f.group == 'CL').sum())}개 / CO {int((f.group == 'CO').sum())}개, 화자 CL {f[f.group == 'CL'].speaker_id.nunique()}명 / CO {f[f.group == 'CO'].speaker_id.nunique()}명
- 분석에 쓰인 단어 비율 (질문 있는 답변 전체 대비): CL {used['CL'] / total['CL']:.0%}, CO {used['CO'] / total['CO']:.0%}
- {MAXC * W}단어 상한에 걸린 답변: CL {(f[f.group == 'CL'].n_chunks > MAXC).mean():.0%}, CO {(f[f.group == 'CO'].n_chunks > MAXC).mean():.0%}

## 지표 (화자 평균의 그룹 중앙값, d = 화자 단위 Cohen's d)
{tab.to_markdown(floatfmt='.4f')}

## 질문 주제 (키워드 1차 분류, 답변 수)
{topics.to_markdown()}

## 검정 전 확인 (fig8)
| 확인 | 화자 | 결과 |
|---|---|---|
""" + '\n'.join(f'| {a} | {b} | {c} |' for a, b, c in checks) + '\n'
    open(os.path.join(ROOT, CFG['paths']['reports'], '03_features.md'), 'w', encoding='utf8').write(md)
    print(md)


if __name__ == '__main__':
    assert chunks(' '.join(['w'] * 23)) == [' '.join(['w'] * 10)] * 2  # 마지막 3단어 조각은 버림
    assert len(chunks(' '.join(['w'] * 26))) == 3
    assert abs(slope([0, 1, 2]) - 1) < 1e-9
    main()
