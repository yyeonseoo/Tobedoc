"""3d 판별기(validate.py와 동일 설정)를 다시 학습해 지정 어휘의 계수 순위와 빈도를 조회한다 (진단용, 판정에 쓰지 않음).

  python src/feature_probe.py v2   -> reports/05b_feature_probe_v2.md
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from preprocess import CFG, ROOT, clean

TAG = sys.argv[1] if len(sys.argv) > 1 else 'v2'
SYN = {'': 'synthetic', 'v2': 'synthetic/v2'}[TAG]
PROBE = ['signal', 'network', 'connection', 'internet', 'wifi', 'line', 'phone', 'call', 'video', 'zoom', 'screen',
         'camera', 'hear', 'crackly', 'patchy', 'broke up', 'cut out', 'lost you', 'can you hear']

corpus = pd.read_parquet(os.path.join(ROOT, CFG['paths']['corpus']))
R = corpus[corpus.substantive].text.tolist()
S = pd.read_parquet(os.path.join(ROOT, SYN, 'synthetic_corpus.parquet')).text.map(clean).tolist()
clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                    LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000))
clf.fit(R + S, np.r_[np.zeros(len(R)), np.ones(len(S))])
vocab = clf[0].get_feature_names_out()
coef = clf[-1].coef_[0]
rank_syn = {vocab[i]: r + 1 for r, i in enumerate(np.argsort(-coef))}   # 1 = 합성 쪽으로 가장 강함
rank_real = {vocab[i]: r + 1 for r, i in enumerate(np.argsort(coef))}  # 1 = 실제 쪽으로 가장 강함


def per10k(texts, term):
    low = ' ' + ' '.join(' '.join(t.lower().split()) for t in texts) + ' '
    return 10000 * low.count(f' {term} ') / max(len(low.split()), 1)


rows = []
for t in PROBE:
    c = coef[list(vocab).index(t)] if t in rank_syn else np.nan
    rows.append(dict(term=t, in_vocab=t in rank_syn, coef=c,
                     rank_toward_syn=rank_syn.get(t), rank_toward_real=rank_real.get(t),
                     real_per10k=per10k(R, t), syn_per10k=per10k(S, t)))
tab = pd.DataFrame(rows)
top = 20
hit = tab[(tab.rank_toward_syn <= top) | (tab.rank_toward_real <= top)]
md = f"""# 05b. 판별기 피처 조회: 화상·전화 통화 어휘 (진단용)

재현: `python src/feature_probe.py {TAG}`. 3d와 같은 설정(TF-IDF 1–2gram min_df=2 + 로지스틱 balanced)으로 전체 데이터에 학습. 어휘 크기 {len(vocab)}.
순위 1 = 그 방향(합성 또는 실제)으로 계수가 가장 큰 피처.

{tab.round(3).to_markdown(index=False)}

**상위 {top} 피처에 든 통화 어휘: {', '.join(hit.term) if len(hit) else '없음'}.**
"""
open(os.path.join(ROOT, CFG['paths']['reports'], f'05b_feature_probe_{TAG or "v1"}.md'), 'w', encoding='utf8').write(md)
print(md)
