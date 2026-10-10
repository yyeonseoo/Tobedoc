"""선행 연구 기준으로 1·2·3차 합성 데이터 재채점 -> reports/07_prior_criteria_rescore.md (원문 미포함)

선행 기준 출처:
- Utility / privacy(min distance): arXiv 2411.17672 (DAIC-WOZ 우울증) — 실제/합성/실제+합성 학습 → 실제 테스트 성능, BERT 임베딩 최소 거리
- Fidelity(MMD) / privacy(표절 d<0.05) / diversity(TTR): arXiv 2604.27014 (스페인어 정신과 보고서)

Utility 설계: 과제 = 응답 단위 CL vs CO 분류 (선행의 PHQ-8 예측에 대응). 화자 단위 층화 그룹 5-fold x 10회 반복.
테스트는 항상 실제 응답(학습에 없는 화자)만. 테스트 화자를 시드로 한 합성 샘플은 학습에서 제외(누수 차단).
주의: 그룹-주제 교란 때문에 이 분류는 주제 분류를 일부 포함한다 (세 조건에 동일하게 작용).
"""
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

from preprocess import CFG, ROOT, clean

SEED = CFG['seed']
REPEATS = 10
corpus = pd.read_parquet(os.path.join(ROOT, CFG['paths']['corpus']))
real = corpus[corpus.substantive][['speaker_id', 'group', 'text']].reset_index(drop=True)
real['y'] = (real.group == 'CL').astype(int)
SYN = {}
for k, p in [('1차', 'synthetic'), ('2차', 'synthetic/v2'), ('3차', 'synthetic/v3')]:
    s = pd.read_parquet(os.path.join(ROOT, p, 'synthetic_corpus.parquet'))
    SYN[k] = pd.DataFrame({'speaker_id': s.seed_speaker_id, 'group': s.group, 'text': s.text.map(clean),
                           'y': (s.group == 'CL').astype(int)}).reset_index(drop=True)

model = SentenceTransformer(CFG['validate']['sbert_model'], device='cpu')
emb = lambda texts: model.encode(list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
ER = emb(real.text)
ES = {k: emb(s.text) for k, s in SYN.items()}


def make_clf(kind):
    if kind == 'tfidf':
        return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                             LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000))
    return LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)


def fit_predict(kind, Xtr, ytr, Xte):
    clf = make_clf(kind)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def utility():
    spk = real.groupby('speaker_id').y.first()
    rows = []
    for rep in range(REPEATS):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + rep)
        for kind in ['tfidf', 'sbert']:
            preds = {c: np.zeros(len(real)) for c in ['real'] + [f'{k}:{m}' for k in SYN for m in ('syn', 'real+syn')]}
            for tr, te in cv.split(real.text, real.y, groups=real.speaker_id):
                test_spk = set(real.speaker_id.iloc[te])
                Xr = real.text.iloc[tr] if kind == 'tfidf' else ER[tr]
                Xte = real.text.iloc[te] if kind == 'tfidf' else ER[te]
                preds['real'][te] = fit_predict(kind, Xr, real.y.iloc[tr], Xte)
                for k, s in SYN.items():
                    keep = ~s.speaker_id.isin(test_spk).to_numpy()
                    Xs = s.text[keep] if kind == 'tfidf' else ES[k][keep]
                    ys = s.y[keep]
                    preds[f'{k}:syn'][te] = fit_predict(kind, Xs, ys, Xte)
                    Xc = pd.concat([Xr, Xs]) if kind == 'tfidf' else np.vstack([Xr, Xs])
                    preds[f'{k}:real+syn'][te] = fit_predict(kind, Xc, np.r_[real.y.iloc[tr], ys], Xte)
            for c, p in preds.items():
                sp = pd.Series(p).groupby(real.speaker_id).mean()
                rows.append(dict(rep=rep, model=kind, cond=c, auc_resp=roc_auc_score(real.y, p),
                                 auc_spk=roc_auc_score(spk.loc[sp.index], sp)))
    return pd.DataFrame(rows)


def mmd_rbf(X, Y, gamma=None):
    Z = np.vstack([X, Y])
    d2 = np.maximum(2 - 2 * Z @ Z.T, 0)   # 정규화 임베딩: ||a-b||^2 = 2-2cos
    g = gamma or 1 / np.median(d2[np.triu_indices(len(Z), 1)])
    K = np.exp(-g * d2)
    n = len(X)
    return K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean()


def fidelity_privacy():
    rng = np.random.default_rng(SEED)
    base = []
    for _ in range(20):   # 기준선: 실제를 무작위로 반으로 나눴을 때의 MMD
        idx = rng.permutation(len(ER))
        base.append(mmd_rbf(ER[idx[:len(idx) // 2]], ER[idx[len(idx) // 2:]]))
    rr = 1 - ER @ ER.T
    np.fill_diagonal(rr, np.inf)
    rr_nn = rr.min(1)
    rows = [dict(set='실제 (기준선)', MMD=f'{np.mean(base):.4f} ± {np.std(base):.4f} (반분)', min_dist=rr_nn.min(),
                 avg_min_dist=rr_nn.mean(), plagiarism_d_lt_0_05='{:.1%}'.format((rr_nn < 0.05).mean()),
                 TTR=np.mean([len(set(t.lower().split())) / len(t.split()) for t in real.text]))]
    for k, s in SYN.items():
        nn = (1 - ES[k] @ ER.T).min(1)
        rows.append(dict(set=k, MMD=f'{mmd_rbf(ES[k], ER):.4f}', min_dist=nn.min(), avg_min_dist=nn.mean(),
                         plagiarism_d_lt_0_05='{:.1%}'.format((nn < 0.05).mean()),
                         TTR=np.mean([len(set(t.lower().split())) / len(t.split()) for t in s.text])))
    return pd.DataFrame(rows).set_index('set')


def main():
    u = utility()
    agg = u.groupby(['model', 'cond'])[['auc_resp', 'auc_spk']].agg(['mean', 'std']).round(3)
    base = u[u.cond == 'real'].set_index(['rep', 'model'])
    gain = []
    for c in sorted(set(u.cond) - {'real'}):
        x = u[u.cond == c].set_index(['rep', 'model'])
        for m in ['tfidf', 'sbert']:
            dlt = (x.xs(m, level='model').auc_resp - base.xs(m, level='model').auc_resp)
            gain.append(dict(cond=c, model=m, delta_auc_resp_mean=dlt.mean(), better_in_reps=f'{int((dlt > 0).sum())}/{REPEATS}'))
    fp = fidelity_privacy()
    md = f"""# 07. 선행 연구 기준으로 재채점 (1·2·3차)

재현: `python src/prior_criteria.py`. 원문 미포함.

선행 기준 출처:
- arXiv 2411.17672 (DAIC-WOZ): utility(실제/합성/실제+합성 학습 → 실제 테스트), privacy(BERT 임베딩 최소 거리)
- arXiv 2604.27014 (정신과 보고서): fidelity(MMD 등), diversity(TTR 등), privacy(최근접 거리 d<0.05 = 표절)

## 1. Utility — CL vs CO 응답 분류, 학습에 없는 화자의 실제 응답으로만 평가
설계: 화자 단위 층화 그룹 5-fold, 10회 반복. 테스트 화자를 시드로 한 합성은 학습에서 제외했다. 모델 두 종(TF-IDF 로지스틱, SBERT 임베딩 로지스틱).
AUC는 응답 단위(`auc_resp`)와 화자 단위(`auc_spk`, 화자별 예측 평균)로 보고한다. ⚠ 그룹-주제 교란 때문에 이 과제에는 주제 분류가 일부 섞여 있다.

{agg.to_markdown()}

**실제 단독 대비 변화 (응답 단위 AUC, 같은 분할에서 쌍 비교)**
{pd.DataFrame(gain).round(3).to_markdown(index=False)}

## 2. Fidelity·Privacy·Diversity — 선행 지표
- MMD: SBERT 임베딩, RBF 커널(중앙값 휴리스틱). 실제 기준선 = 실제를 반으로 나눈 두 묶음 사이 MMD (20회 평균).
- min_dist / avg_min_dist: 합성 → 실제 최근접 코사인 거리 (기준선: 실제 → 다른 실제, 자기 제외).
- 표절률: 최근접 거리 < 0.05 비율.
- TTR: 응답별 type-token ratio 평균 (길이에 민감).

{fp.round(4).to_markdown()}
"""
    open(os.path.join(ROOT, CFG['paths']['reports'], '07_prior_criteria_rescore.md'), 'w', encoding='utf8').write(md)
    print(md)


if __name__ == '__main__':
    X = np.eye(3)
    assert abs(mmd_rbf(X, X)) < 1e-12
    main()
