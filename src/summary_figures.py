"""정리 문서(reports/SUMMARY_2026-10-02.md)용 그림 fig1-4 (증강 결과, 데이터 개요)와 공용 스타일·mattr.

fig5-7과 답변 단위 지표는 position_features.py가 만든다.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocess import ROOT

FIG = os.path.join(ROOT, 'reports/figures')
os.makedirs(FIG, exist_ok=True)
# dataviz 기본 팔레트 슬롯 1·2 (인접쌍 CVD 검증 통과), 텍스트·격자는 중립색
C = {'CL': '#2a78d6', 'CO': '#eb6834'}
INK, INK2, GRID, BG = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'figure.facecolor': BG,
                     'axes.facecolor': BG, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=160, bbox_inches='tight')
    plt.close(fig)


def legend(ax, **kw):
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C[g]) for g in C], labels=['CL (조현병)', 'CO (대조군)'],
              frameon=False, **kw)


def fig_data(corpus):
    # fig1: 화자별 발화량
    spk = corpus.groupby(['group', 'speaker_id']).n_tokens.sum().reset_index().sort_values('n_tokens')
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(spk.speaker_id, spk.n_tokens, color=[C[g] for g in spk.group], height=0.7)
    ax.set_xlabel('참여자 발화 토큰 수')
    ax.set_title('화자별 발화량: 03EB14·21AN11 두 명이 CL 발화의 약 45%', loc='left', color=INK)
    ax.grid(axis='y', visible=False)
    legend(ax, loc='lower right')
    save(fig, 'fig3_speaker_volume.png')
    # fig2: 턴 길이 분포
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.logspace(0, np.log10(corpus.n_tokens.max() + 1), 40)
    for g in C:
        ax.hist(corpus[corpus.group == g].n_tokens, bins=bins, histtype='step', linewidth=2, color=C[g])
    ax.set_xscale('log')
    ax.axvline(30, color=INK2, linewidth=1, linestyle='--')
    ax.text(32, ax.get_ylim()[1] * 0.9, '30토큰', color=INK2)
    short = (corpus.n_tokens < 30).mean()
    ax.set_title(f'턴 길이 분포: {short:.0%}가 30토큰 미만(맞장구 위주)', loc='left', color=INK)
    ax.set_xlabel('턴 길이 (토큰, 로그 축)')
    ax.set_ylabel('턴 수')
    legend(ax, loc='upper right')
    save(fig, 'fig4_turn_length.png')


def fig_augmentation(aug):
    # fig3: LLM 합성 1-3차 판별 AUC (archive/augmentation/REPORT.md 표 값)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    auc = {'1차\n자가 생성': 0.940, '2차\n프로파일+필터': 0.932, '3차\nGPT-4o 샘플링': 0.999}
    ax.bar(list(auc), list(auc.values()), color='#8a8984', width=0.55)
    for i, v in enumerate(auc.values()):
        ax.text(i, v + 0.008, f'{v:.3f}', ha='center', color=INK)
    ax.hlines(0.85, -0.5, 2.35, color=INK, linewidth=1, linestyle='--')
    ax.text(2.4, 0.85, '불합격 경계 0.85', ha='left', va='center', color=INK2)
    ax.hlines(0.5, -0.5, 2.35, color=INK2, linewidth=0.8, linestyle=':')
    ax.text(2.4, 0.5, '구분 불가 0.5', ha='left', va='center', color=INK2)
    ax.set_ylim(0.45, 1.04)
    ax.set_xlim(-0.5, 3.3)
    ax.set_ylabel('실제 vs 합성 판별 AUC')
    ax.set_title('LLM 합성은 세 번 모두 실제와 쉽게 구분됨', loc='left', color=INK)
    ax.grid(axis='x', visible=False)
    save(fig, 'fig1_llm_auc.png')
    # fig4: 증강 방식별 화자 단위 CL-CO 효과 크기 (응답 단위)
    def d_(a, b):
        return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    r = aug[aug.unit == 'response']
    names = {'original': '원본', 'sent_shuffle': '문장 순서 섞기', 'eda_swap_del': '어절 swap+삭제 (EDA)'}
    ds = {}
    for c in names:
        sp = r[r.cond == c].groupby(['spk', 'group']).slope.mean().reset_index()
        ds[names[c]] = d_(sp[sp.group == 'CL'].slope, sp[sp.group == 'CO'].slope)
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    y = np.arange(len(ds))[::-1]
    ax.hlines(y, 0, list(ds.values()), color=GRID, linewidth=2)
    ax.scatter(list(ds.values()), y, s=80, color=['#0b0b0b', '#8a8984', '#8a8984'], zorder=3,
               edgecolor=BG, linewidth=2)
    for yy, v in zip(y, ds.values()):
        ax.text(v + 0.04, yy, f'd = {v:+.2f}', va='center', color=INK)
    ax.set_yticks(y, list(ds))
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_xlim(-0.2, 1.4)
    ax.set_xlabel("이탈 기울기의 CL-CO 차이 (화자 단위 Cohen's d)")
    ax.set_title('섞기는 신호를 지우고, EDA는 없던 차이를 키운다', loc='left', color=INK)
    ax.grid(axis='y', visible=False)
    save(fig, 'fig2_aug_effect.png')


def mattr(words, w=25):
    if len(words) <= w:
        return len(set(words)) / max(len(words), 1)
    return np.mean([len(set(words[i:i + w])) / w for i in range(len(words) - w + 1)])


def main():
    fig_data(pd.read_parquet(os.path.join(ROOT, 'data/processed/corpus.parquet')))
    fig_augmentation(pd.read_parquet(os.path.join(ROOT, 'data/processed/aug_impact.parquet')))


if __name__ == '__main__':
    assert abs(mattr(['a', 'b'] * 30, w=4) - 0.5) < 1e-9 and mattr(['a', 'b', 'c']) == 1.0
    main()
