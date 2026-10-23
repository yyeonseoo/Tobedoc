"""E6: 팀 Colab 노트북(v1/colab_v5_1.ipynb)의 Reddit 작성자 단위 AUC 재현과 점검 -> reports/03_notebook_audit.md.

노트북의 전처리·지표·모델을 그대로 옮기고(아래 '노트북 그대로' 구간), 한 번에 하나씩 바꿔 AUC 변화를 본다.
"""
import os
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from text import path

SEED = 42

# ---------------- 노트북 그대로 (cell 5, 7, 10) ----------------
DIRECT_TERMS = [r"schizophren\w*", r"schizo\w*", r"psychosis", r"psychotic", r"antipsychotic\w*", r"neuroleptic\w*",
                r"haloperidol", r"clozapine", r"olanzapine", r"risperidone", r"quetiapine", r"aripiprazole",
                r"diagnos\w*", r"hallucinat\w*", r"delusion\w*", r"hearing voices?"]
DIRECT_RE = re.compile(r"\b(?:" + "|".join(DIRECT_TERMS) + r")\b", re.I)
TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z'-]*\b")
FIRST_PERSON = {'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'}
FUNCTION_WORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'because', 'while', 'although', 'to', 'of', 'in',
                  'on', 'at', 'for', 'from', 'with', 'without', 'by', 'as', 'is', 'am', 'are', 'was', 'were', 'be',
                  'been', 'being', 'do', 'does', 'did', 'have', 'has', 'had', 'can', 'could', 'will', 'would', 'shall',
                  'should', 'may', 'might', 'must', 'this', 'that', 'these', 'those', 'it', 'its', 'he', 'she', 'they',
                  'them', 'his', 'her', 'their', 'i', 'me', 'my', 'we', 'us', 'our', 'you', 'your'}
CONJUNCTIONS = {'and', 'or', 'but', 'because', 'although', 'though', 'while', 'whereas', 'however', 'therefore', 'so',
                'yet', 'if', 'unless', 'since'}
STRUCTURAL = ['adj_semantic_distance', 'semantic_drift_slope', 'max_semantic_drift', 'mattr50', 'avg_word_length',
              'first_person_ratio', 'function_word_ratio', 'conjunction_ratio']


def toks(text):
    return TOKEN_RE.findall(str(text).lower())


def mattr(tok, window=50):
    if not tok:
        return np.nan
    if len(tok) <= window:
        return len(set(tok)) / len(tok)
    return float(np.mean([len(set(tok[i:i + window])) / window for i in range(len(tok) - window + 1)]))


def geometry(text):
    tok = toks(text)
    ch = [' '.join(tok[i:i + 40]) for i in range(0, len(tok) - 40 + 1, 20)] if len(tok) >= 40 else []
    nan = dict(adj_semantic_distance=np.nan, semantic_drift_slope=np.nan, max_semantic_drift=np.nan)
    if len(ch) < 3:
        return nan
    try:
        X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]*\b").fit_transform(ch)
    except ValueError:
        return nan
    adj = 1 - np.asarray(X[:-1].multiply(X[1:]).sum(axis=1)).ravel()
    drift = 1 - np.asarray(X.multiply(X[0]).sum(axis=1)).ravel()
    return dict(adj_semantic_distance=adj.mean(), semantic_drift_slope=np.polyfit(np.arange(len(drift)), drift, 1)[0],
                max_semantic_drift=drift.max())


def extract(text):
    tok = toks(text)
    n = max(len(tok), 1)
    out = dict(mattr50=mattr(tok), avg_word_length=np.mean([len(x) for x in tok]) if tok else np.nan,
               first_person_ratio=sum(x in FIRST_PERSON for x in tok) / n,
               function_word_ratio=sum(x in FUNCTION_WORDS for x in tok) / n,
               conjunction_ratio=sum(x in CONJUNCTIONS for x in tok) / n)
    out.update(geometry(text))
    return out


def author_auc(df, cols):
    X, y = df[cols], df.label.values
    pred = np.full(len(df), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, df.author.astype(str)):
        pipe = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler()),
                         ('lr', LogisticRegression(max_iter=3000, class_weight='balanced', random_state=SEED))])
        pred[te] = pipe.fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    a = df[['author', 'label']].assign(p=pred).groupby(['author', 'label'], as_index=False).p.mean()
    return roc_auc_score(a.label, a.p)
# -----------------------------------------------------------------


def featurize(texts):
    return pd.DataFrame([extract(t) for t in texts])


def notebook_matching(df):
    """노트북 7A 그대로: 가린 텍스트 TF-IDF -> SVD 40 -> 유사도 하위 25% 제외 -> 짝 찾기 어려운 글부터 1:1 비복원 매칭."""
    vec = TfidfVectorizer(lowercase=True, stop_words='english', min_df=5, max_df=0.95, max_features=8000,
                          ngram_range=(1, 2), token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]*\b")
    Z = TruncatedSVD(n_components=40, random_state=SEED).fit_transform(vec.fit_transform(df.text_masked))
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    lab = df.label.to_numpy()
    pos, neg = np.where(lab == 1)[0], np.where(lab == 0)[0]
    S = Z[pos] @ Z[neg].T
    best = S.max(1)
    cal = max(0.20, float(np.quantile(best, 0.25)))
    avail, keep = np.ones(len(neg), bool), []
    for pi in np.argsort(best):
        cand = np.where(avail)[0]
        j = cand[int(np.argmax(S[pi, cand]))]
        if S[pi, j] >= cal:
            keep += [pos[pi], neg[j]]
            avail[j] = False
    return df.iloc[sorted(keep)].reset_index(drop=True)


def main():
    raw = pd.read_csv(path('reddit_csv'))
    for c in ['title', 'selftext', 'author', 'subreddit']:
        raw[c] = raw[c].fillna('').astype(str)
    raw['label'] = raw.is_schizophrenic.astype(int)
    raw['text_raw'] = (raw.title.str.strip() + '\n' + raw.selftext.str.strip()).str.strip()
    raw['word_count'] = raw.text_raw.str.findall(r"\b[\w'-]+\b").str.len()
    df = raw[raw.author.ne('') & raw.author.ne('[deleted]') & raw.word_count.between(80, 1200)].reset_index(drop=True)
    df['text_masked'] = df.text_raw.map(lambda t: DIRECT_RE.sub('MASKTERM', t))
    df['body_masked'] = df.selftext.map(lambda t: DIRECT_RE.sub('MASKTERM', t))
    df['text_removed'] = df.text_raw.map(lambda t: DIRECT_RE.sub(' ', t))
    df['body_removed'] = df.selftext.map(lambda t: DIRECT_RE.sub(' ', t))

    def cut(s, n=100):  # 노트북 10B와 같은 100 토큰
        return s.map(lambda t: ' '.join(toks(t)[:n]))

    def auc(d, txt, cols=STRUCTURAL):
        X = pd.concat([d[['author', 'label']].reset_index(drop=True), featurize(txt.reset_index(drop=True))], axis=1)
        return author_auc(X, cols)

    fa = pd.concat([df[['author', 'label', 'word_count']], featurize(df.text_masked)], axis=1)
    single = {c: author_auc(fa, [c]) for c in STRUCTURAL}
    t1 = [('노트북 그대로 (MASKTERM, 제목+본문, 길이 그대로)', author_auc(fa, STRUCTURAL)),
          ('진단어를 MASKTERM 대신 삭제만', auc(df, df.text_removed)),
          ('제목만 제외 (본문만)', auc(df, df.body_masked)),
          ('길이만 고정: 제목+본문 앞 100 토큰 (10B 방식을 전체 표본에)', auc(df, cut(df.text_masked))),
          ('참고: 글 길이(단어 수) 하나만', author_auc(fa, ['word_count']))]
    m = notebook_matching(df)
    t2 = [(f'노트북 10B 재현 (주제 매칭 {len(m) // 2}쌍, 제목+본문 앞 100 토큰)', auc(m, cut(m.text_masked))),
          ('10B에서 제목만 제외 (본문 앞 100 토큰)', auc(m, cut(m.body_masked)))]
    short_b = int(m.body_masked.map(lambda t: len(toks(t))).lt(100).sum())
    lb = df.body_removed.map(lambda t: len(toks(t))).ge(100)
    t3 = [('진단어 삭제', t1[1][1]),
          ('진단어 삭제 + 제목 제외', auc(df, df.body_removed)),
          (f'진단어 삭제 + 제목 제외 + 본문 100 토큰 이상인 글({int(lb.sum())}개)의 앞 100 토큰 (정확히 같은 길이)',
           auc(df[lb], cut(df.body_removed[lb])))]
    has_mask = df.text_masked.str.contains('MASKTERM')

    def tab(rows):
        return pd.DataFrame(rows, columns=['조건', '작성자 단위 AUC']).to_markdown(index=False, floatfmt='.3f')
    s = pd.Series(single, name='단독 AUC').to_frame()
    s['조현병 평균'] = fa[fa.label == 1][STRUCTURAL].mean()
    s['대조 평균'] = fa[fa.label == 0][STRUCTURAL].mean()
    md = f"""# 03. 팀 노트북 Reddit AUC 재현과 점검 (E6)

재현: `python src/v2_notebook_audit.py`. 팀 노트북(`v1/colab_v5_1.ipynb`, 최신본 `v1/colab_latest.ipynb`도 전처리·피처·모델 코드가 같음)의 전처리·지표·모델을 그대로 옮겼다 (80~1,200단어 글 {len(df)}개, 작성자 {df.author.nunique()}명). 모델은 구조 지표 8개 로지스틱 회귀, 작성자 단위 5-fold.

## 1. 노트북 주 분석에서 한 가지씩만 바꿨을 때 (전체 표본)
{tab(t1)}

## 2. 노트북 10B(주제 매칭 + 앞 100 토큰)에서 제목만 뺐을 때
{tab(t2)}

제목을 빼면 본문이 100 토큰보다 짧은 글이 {len(m)}개 중 {short_b}개 생긴다. 노트북 조건에서 제목만 바꾼 비교라 그대로 둔다.

## 3. 문제를 모두 정리했을 때 (전체 표본)
{tab(t3)}

## 지표 하나씩 단독 AUC (노트북 그대로)
{s.to_markdown(floatfmt='.3f')}

## 해석
- 본문(selftext)은 데이터셋 제작자가 NLTK 불용어를 지운 상태라, 1인칭·기능어·접속사 단어는 대부분 제목에서 세어진다. 본문에는 NLTK 목록에 없는 일부(would, could, us, mine, might 등)만 남아 있다.
- "MASKTERM"이 들어간 글: 조현병 {has_mask[df.label == 1].mean():.0%}, 대조 {has_mask[df.label == 0].mean():.0%}. 8글자 표시가 평균 단어 길이를 부풀린다.
- 주 분석에서 제목만 빼면 {t1[2][1] - t1[0][1]:+.3f}, 길이만 고정하면 {t1[3][1] - t1[0][1]:+.3f}다. 제목이 있으면 제목 문체가 신호를 주도해 길이 고정의 효과가 보이지 않는다.
- 주제와 길이를 이미 통제한 10B에서는 제목만 빼도 {t2[1][1] - t2[0][1]:+.3f}가 된다. 10B에서 성능이 유지된 부분의 상당 부분을 제목이 지탱했다.
- 세 가지를 모두 정리하면 노트북 AUC {t1[0][1]:.3f}가 {t3[2][1]:.3f}가 된다. 남는 부분은 본문 어휘 차이로, 게시판 장르 차이인지 환자 언어 특징인지는 이 데이터만으로 가를 수 없다.
"""
    open(os.path.join(path('reports_v2'), '03_notebook_audit.md'), 'w', encoding='utf8').write(md)
    print(md)


if __name__ == '__main__':
    main()
