"""E6: 팀 Colab 노트북(v1/colab_v5_1.ipynb)의 Reddit 작성자 단위 AUC 재현과 점검 -> reports/03_notebook_audit.md.

노트북의 전처리·지표·모델을 그대로 옮기고(아래 '노트북 그대로' 구간), 한 번에 하나씩 바꿔 AUC 변화를 본다.
"""
import os
import re

import numpy as np
import pandas as pd
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


def main():
    raw = pd.read_csv(path('reddit_csv'))
    for c in ['title', 'selftext', 'author', 'subreddit']:
        raw[c] = raw[c].fillna('').astype(str)
    raw['label'] = raw.is_schizophrenic.astype(int)
    raw['text_raw'] = (raw.title.str.strip() + '\n' + raw.selftext.str.strip()).str.strip()
    raw['word_count'] = raw.text_raw.str.findall(r"\b[\w'-]+\b").str.len()
    df = raw[raw.author.ne('') & raw.author.ne('[deleted]') & raw.word_count.between(80, 1200)].reset_index(drop=True)
    masked = df.text_raw.map(lambda t: DIRECT_RE.sub('MASKTERM', t))
    removed = df.text_raw.map(lambda t: DIRECT_RE.sub(' ', t))
    body80 = df.selftext.map(lambda t: ' '.join(DIRECT_RE.sub(' ', t).split()[:80]))
    title_body80 = (df.title + '\n' + body80)

    rows = []
    fa = pd.concat([df[['author', 'label', 'word_count']], featurize(masked)], axis=1)
    rows.append(('(a) 노트북 재현: MASKTERM 치환, 길이 그대로', author_auc(fa, STRUCTURAL)))
    single = {c: author_auc(fa, [c]) for c in STRUCTURAL}
    fc = pd.concat([df[['author', 'label']], featurize(removed)], axis=1)
    rows.append(('(c) 진단 단어를 치환 대신 삭제', author_auc(fc, STRUCTURAL)))
    rows.append(('(d) 글 길이(단어 수) 하나만', author_auc(fa, ['word_count'])))
    fe = pd.concat([df[['author', 'label']], featurize(title_body80)], axis=1)
    rows.append(('(e) 삭제 + 본문 앞 80단어로 길이 고정 (제목 포함)', author_auc(fe, STRUCTURAL)))
    ff = pd.concat([df[['author', 'label']], featurize(body80)], axis=1)
    rows.append(('(f) (e)에서 제목 제외 (본문만)', author_auc(ff, STRUCTURAL)))
    has_mask = masked.str.contains('MASKTERM')
    t = pd.DataFrame(rows, columns=['조건', '작성자 단위 AUC'])
    s = pd.Series(single, name='단독 AUC').to_frame()
    s['조현병 평균'] = fa[fa.label == 1][STRUCTURAL].mean()
    s['대조 평균'] = fa[fa.label == 0][STRUCTURAL].mean()
    md = f"""# 03. 팀 노트북 Reddit AUC 재현과 점검 (E6)

재현: `python src/audit_notebook.py`. 노트북(`v1/colab_v5_1.ipynb`)의 전처리·지표·모델을 그대로 옮겼다 (80~1,200단어 글 {len(df)}개, 작성자 {df.author.nunique()}명).

## 조건별 AUC (구조 지표 8개, 로지스틱 회귀, 작성자 단위 5-fold)
{t.to_markdown(index=False, floatfmt='.3f')}

## (b) 지표 하나씩 단독 AUC (조건 a)
{s.to_markdown(floatfmt='.3f')}

## 참고
- "MASKTERM"이 들어간 글: 조현병 {has_mask[df.label == 1].mean():.0%}, 대조 {has_mask[df.label == 0].mean():.0%}
- 본문(selftext)은 데이터셋 제작자가 NLTK 불용어를 지운 상태라, 1인칭·기능어·접속사 단어는 사실상 제목에서만 세어진다.
"""
    open(os.path.join(path('reports'), '03_notebook_audit.md'), 'w', encoding='utf8').write(md)
    print(md)


if __name__ == '__main__':
    main()
