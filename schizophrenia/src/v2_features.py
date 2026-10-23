"""두 코퍼스 공통 지표 (data/processed/features.parquet, chunks.parquet).

단위마다 내용어를 5개씩 조각내 SBERT로 임베딩한다. 앞 N 내용어(N = 40 주 분석, 30·60 민감도)만으로 지표를 구해 길이를 맞춘다.
- drift_slope : 첫 조각과의 거리를 조각 위치에 회귀한 기울기 (주 지표)
- drift_mean  : 첫 조각과의 평균 거리
- adj_mean    : 인접 조각 거리 평균
- late_slope  : 첫 3조각 평균을 기준점으로 한 그 뒤 조각들의 기울기 (출발부 효과 제외)
- recurrence  : 각 조각의 내용어 중 앞에서 이미 나온 단어 비율 (어휘 반복)
- mattr       : 내용어 MATTR
- anchor_*    : 기준점(Reddit 제목, DAIS-C 면담자 질문)과의 거리 (보조)
chunks.parquet: 조각 단위 거리. DAIS-C는 답변 전체를 저장하고 판단 태그 포함 여부를 표시한다 (태그 검증용).
"""
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from text import CFG, chunks, path

F = CFG['features']
C = F['chunk_words']
NS = [F['n_words']] + F['n_words_sensitivity']
NMIN, NMAX = min(NS), max(NS)
JUDGE = CFG['daisc']['judge_tags']


def slope(y, x=None):
    y = np.asarray(y, float)
    return np.polyfit(np.arange(len(y)) if x is None else x, y, 1)[0] if len(y) >= 2 else np.nan


def mattr(ws, w):
    if len(ws) <= w:
        return len(set(ws)) / max(len(ws), 1)
    return float(np.mean([len(set(ws[i:i + w])) / w for i in range(len(ws) - w + 1)]))


def recurrence(ws, k):
    return float(np.mean([len(set(ws[i * C:(i + 1) * C]) & set(ws[:i * C])) / C for i in range(1, k)]))


def unit_features(E, A, ws, n):
    k = n // C
    E = E[:k]
    d0 = 1 - E[1:] @ E[0]
    base = E[:F['late_from_chunk']].mean(0)
    base /= np.linalg.norm(base)
    late = 1 - E[F['late_from_chunk']:] @ base
    out = dict(n=n, drift_slope=slope(d0, np.arange(1, k)), drift_mean=d0.mean(),
               adj_mean=np.mean(1 - np.sum(E[1:] * E[:-1], axis=1)),
               late_slope=slope(late, np.arange(F['late_from_chunk'], k)),
               recurrence=recurrence(ws, k), mattr=mattr(ws[:n], F['mattr_window']))
    if A is not None:
        da = 1 - E @ A
        out.update(anchor_first=da[0], anchor_mean=da.mean(), anchor_slope=slope(da))
    return out


def load():
    r = pd.read_parquet(os.path.join(path('processed'), 'reddit_units.parquet')).assign(corpus='reddit')
    d = pd.read_parquet(os.path.join(path('processed'), 'daisc_units.parquet')).assign(corpus='daisc')
    d['label'] = (d.group == 'CL').astype(int)
    return pd.concat([r, d], ignore_index=True)


def main():
    u = load()
    u = u[u.n_words >= NMIN].reset_index(drop=True)
    # 임베딩할 조각: Reddit은 앞 NMAX 내용어, DAIS-C는 태그 검증을 위해 답변 전체
    u['chunk_text'] = [chunks(list(w) if c == 'daisc' else list(w)[:NMAX], C) for w, c in zip(u.words, u.corpus)]
    u['anchor_text'] = [' '.join(a) if len(a) >= 2 else None for a in u.anchor_words]
    texts = sorted({t for cs in u.chunk_text for t in cs} | set(u.anchor_text.dropna()))
    m = SentenceTransformer(F['sbert_model'])
    emb = dict(zip(texts, m.encode(texts, normalize_embeddings=True, batch_size=256, show_progress_bar=True)))
    meta_cols = ['corpus', 'unit_id', 'speaker_id', 'label', 'n_words', 'source', 'subreddit', 'group', 'sex',
                 'start_question', 'interview_mode', 'filler_rate']
    frows, crows = [], []
    for _, r in u.iterrows():
        E = np.stack([emb[t] for t in r.chunk_text])
        A = emb[r.anchor_text] if r.anchor_text else None
        meta = {c: r[c] for c in meta_cols if c in r and not (isinstance(r[c], float) and np.isnan(r[c]))}
        meta['anchor_words_n'] = len(r.anchor_words)
        for n in NS:
            if r.n_words >= n:
                frows.append(dict(meta, **unit_features(E, A, list(r.words), n)))
        tags = {t: [any(r[f'in_{t}'][i * C:(i + 1) * C]) for i in range(len(E))] for t in JUDGE} if r.corpus == 'daisc' else {}
        for i in range(len(E)):
            crows.append(dict(corpus=r.corpus, unit_id=r.unit_id, speaker_id=r.speaker_id, label=r.label, pos=i,
                              dist_first=1 - float(E[i] @ E[0]), dist_adj=1 - float(E[i] @ E[i - 1]) if i else np.nan,
                              dist_anchor=1 - float(E[i] @ A) if A is not None else np.nan,
                              **{f'in_{t}': v[i] for t, v in tags.items()}))
    f, ch = pd.DataFrame(frows), pd.DataFrame(crows)
    f.to_parquet(os.path.join(path('processed'), 'features.parquet'), index=False)
    ch.to_parquet(os.path.join(path('processed'), 'chunks.parquet'), index=False)
    print(f.groupby(['corpus', 'n', 'label']).agg(units=('unit_id', 'size'), speakers=('speaker_id', 'nunique')))


if __name__ == '__main__':
    assert recurrence(list('abcde') + list('abxyz'), 2) == 0.4
    assert abs(slope([0, 1, 2]) - 1) < 1e-9 and mattr(list('ab') * 30, 4) == 0.5
    main()
