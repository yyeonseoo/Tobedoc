"""v3 임상 개념 피처 13개 (reports/v3/00_plan.md) -> data/processed/v3_features.parquet.

단위마다 앞 N 내용어(N = 40 주 분석, 30 = M4용)를 5개씩 조각내 SBERT로 임베딩하고, 단어 하나씩도 임베딩한다(의미 밀도).
Reddit은 교란 기준선 변수(log_len, title_q, title_len, topic)도 함께 저장한다.
M2용으로 본인/가족·보호자 글은 1인칭 축약어(im, ive, id, ill)를 지운 변형(variant = nofp)도 계산한다.
R1용으로 N = 40 조각 벡터의 평균(384차원)을 emb_mean에 저장한다.
"""
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from wordfreq import zipf_frequency

from text import CFG, chunks, path
from v2_features import mattr, recurrence, slope

C = CFG['features']['chunk_words']
NS = [40, 30]
FIRST_PERSON = {'im', 'ive', 'id', 'ill'}
FEATS = ['drift_slope', 'drift_mean', 'adj_mean', 'late_slope', 'jump_max', 'adj_sd', 'global_coh', 'sem_redund',
         'recurrence', 'mattr', 'sem_density', 'rare_rate', 'clang_rate']


def clang(ws):
    pairs = [(a, b) for a, b in zip(ws, ws[1:]) if len(a) >= 3 and len(b) >= 3 and a != b]
    return float(np.mean([a[-3:] == b[-3:] for a, b in pairs])) if pairs else 0.0


def rare(ws):
    ws = [w for w in ws if not w.isdigit()]
    return float(np.mean([zipf_frequency(w, 'en') < 2.5 for w in ws])) if ws else 0.0


def features(E, W, ws):
    """E: 조각 벡터(k x d, 정규화), W: 단어 벡터(n x d, 정규화), ws: 단어 n개."""
    k = len(E)
    d0 = 1 - E[1:] @ E[0]
    adj = 1 - np.sum(E[1:] * E[:-1], axis=1)
    base = E[:3].mean(0)
    base /= np.linalg.norm(base)
    S = E @ E.T
    red = [S[i, :i - 1].max() for i in range(2, k)]
    return dict(drift_slope=slope(d0, np.arange(1, k)), drift_mean=d0.mean(), adj_mean=adj.mean(),
                late_slope=slope(1 - E[3:] @ base, np.arange(3, k)), jump_max=adj.max(), adj_sd=adj.std(),
                global_coh=S[np.triu_indices(k, 1)].mean(), sem_redund=float(np.mean(red)),
                recurrence=recurrence(ws, k), mattr=mattr(ws, CFG['features']['mattr_window']),
                sem_density=float(np.linalg.norm(W.mean(0))), rare_rate=rare(ws), clang_rate=clang(ws))


def load():
    r = pd.read_parquet(os.path.join(path('processed'), 'reddit_units.parquet')).assign(corpus='reddit')
    raw = pd.read_csv(path('reddit_csv'), usecols=['id', 'title']).rename(columns={'id': 'unit_id'})
    r = r.merge(raw, on='unit_id', how='left')
    r['title_q'] = r.title.fillna('').str.strip().str.endswith('?').astype(int)
    r['title_len'] = r.title.fillna('').str.split().str.len()
    r['log_len'] = np.log(r.n_words.clip(lower=1))
    d = pd.read_parquet(os.path.join(path('processed'), 'daisc_units.parquet')).assign(corpus='daisc')
    d['label'] = (d.group == 'CL').astype(int)
    nofp = r[r.source.isin(['self', 'other'])].copy()
    nofp['words'] = nofp.words.map(lambda ws: [w for w in ws if w not in FIRST_PERSON])
    nofp['n_words'] = nofp.words.str.len()
    u = pd.concat([r.assign(variant='main'), nofp.assign(variant='nofp'), d.assign(variant='main')], ignore_index=True)
    return u[u.n_words >= min(NS)].reset_index(drop=True)


def topics(m, r):
    """Reddit 40 내용어 이상 글의 본문 전체 임베딩 KMeans (그룹 라벨 미사용, v2 E3와 같은 방법)."""
    x = r[(r.variant == 'main') & (r.n_words >= 40)]
    E = m.encode([' '.join(w) for w in x.words], normalize_embeddings=True, batch_size=128)
    sub = np.random.default_rng(CFG['seed']).choice(len(E), min(3000, len(E)), replace=False)
    labs = {k: KMeans(k, n_init=5, random_state=CFG['seed']).fit_predict(E) for k in range(5, 21)}
    sil = {k: silhouette_score(E[sub], v[sub], metric='cosine') for k, v in labs.items()}
    k = max(sil, key=sil.get)
    print(f'topic: k = {k} (실루엣 {sil[k]:.3f})')
    return dict(zip(x.unit_id, labs[k]))


def main():
    u = load()
    m = SentenceTransformer(CFG['features']['sbert_model'])
    topic = topics(m, u[u.corpus == 'reddit'])
    heads = [list(w)[:max(NS)] for w in u.words]
    ch = sorted({c for h in heads for c in chunks(h, C)})
    vocab = sorted({w for h in heads for w in h})
    ce = dict(zip(ch, m.encode(ch, normalize_embeddings=True, batch_size=256)))
    we = dict(zip(vocab, m.encode(vocab, normalize_embeddings=True, batch_size=512)))
    keep = ['corpus', 'variant', 'unit_id', 'speaker_id', 'label', 'source', 'n_words', 'log_len', 'title_q',
            'title_len', 'group', 'sex', 'start_question', 'next_prompt']
    rows = []
    for (_, r), h in zip(u.iterrows(), heads):
        meta = {c: r[c] for c in keep if c in r and not (isinstance(r[c], float) and np.isnan(r[c]))}
        meta['anchor_words_n'] = len(r.anchor_words)
        meta['topic'] = topic.get(r.unit_id, -1) if r.corpus == 'reddit' else -1
        for n in NS:
            if r.n_words < n:
                continue
            ws = h[:n]
            E = np.stack([ce[c] for c in chunks(ws, C)])
            W = np.stack([we[w] for w in ws])
            row = dict(meta, n=n, **features(E, W, ws))
            if n == 40:
                row['emb_mean'] = E.mean(0).astype(np.float32)
            rows.append(row)
    f = pd.DataFrame(rows)
    f.to_parquet(os.path.join(path('processed'), 'v3_features.parquet'), index=False)
    print(f.groupby(['corpus', 'variant', 'n', 'label']).agg(units=('unit_id', 'size'), speakers=('speaker_id', 'nunique')))


if __name__ == '__main__':
    assert clang(['nation', 'station', 'cat']) == 0.5 and clang(['a', 'b']) == 0.0
    assert rare(['the', 'flibbertigibbet', '42']) == 0.5
    E = np.eye(8)
    out = features(E, np.ones((40, 3)) / np.sqrt(3), ['w'] * 40)
    assert abs(out['global_coh']) < 1e-9 and abs(out['sem_density'] - 1) < 1e-9
    main()
