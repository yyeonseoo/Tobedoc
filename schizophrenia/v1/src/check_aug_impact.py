"""증강 방식(문장 셔플, EDA swap/삭제, 윈도우)이 이탈 기울기와 CV에 주는 영향 실측.

결과: data/processed/aug_impact.parquet, 표준출력. 요약은 reports/SUMMARY_2026-10-02.md §3.
"""
import os, random, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import mannwhitneyu
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = random.Random(42)
df = pd.read_parquet(os.path.join(ROOT, "data/processed/corpus.parquet")).sort_values(["speaker_id", "turn_idx"])
model = SentenceTransformer("all-MiniLM-L6-v2")

def emb(sents):
    return model.encode(list(sents), normalize_embeddings=True, batch_size=64)

def feats(E, toks):
    d = 1 - E[1:] @ E[0]                      # 첫 문장 대비 거리 (i=1..n-1)
    adj = 1 - np.sum(E[1:] * E[:-1], axis=1)  # 인접 문장 거리
    slope = np.polyfit(np.arange(1, len(E)), d, 1)[0]
    ttr = len(set(toks)) / max(len(toks), 1)
    return dict(slope=slope, anchor=d.mean(), adj=adj.mean(), var=d.var(), ttr=ttr)

def eda_swap_delete(s, p=0.1):
    w = s.split()
    w = [x for x in w if rng.random() > p] or w[:1]
    for _ in range(max(1, int(p * len(w)))):
        i, j = rng.randrange(len(w)), rng.randrange(len(w)); w[i], w[j] = w[j], w[i]
    return " ".join(w)

def unit_rows(units, label):
    out = []
    for spk, grp, sents in units:
        if len(sents) < 4: continue
        toks = " ".join(sents).lower().split()
        out.append(dict(spk=spk, group=grp, cond="original", **feats(emb(sents), toks)))
        sh = sents[:]; rng.shuffle(sh)
        out.append(dict(spk=spk, group=grp, cond="sent_shuffle", **feats(emb(sh), toks)))
        ed = [eda_swap_delete(s) for s in sents]
        out.append(dict(spk=spk, group=grp, cond="eda_swap_del", **feats(emb(ed), " ".join(ed).lower().split())))
    r = pd.DataFrame(out); r["unit"] = label; return r

stream = {s: (g.group.iloc[0], [x for l in g.sentences for x in l]) for s, g in df.groupby("speaker_id")}
# A) 세션 단위(화자 전체 문장열)
A = [(s, g, ss) for s, (g, ss) in stream.items()]
# B) 응답 단위(>=30토큰 턴)
B = [(r.speaker_id, r.group, list(r.sentences)) for r in df[df.substantive].itertuples()]
# C) 슬라이딩 윈도우 W=10문장, stride 5 (화자 문장열 위, 순서 유지)
W, S = 10, 5
C = [(s, g, ss[i:i + W]) for s, (g, ss) in stream.items() for i in range(0, len(ss) - W + 1, S)]

res = pd.concat([unit_rows(A, "session"), unit_rows(B, "response"), unit_rows(C, "window10")])
res.to_parquet(os.path.join(ROOT, "data/processed/aug_impact.parquet"))

def d_(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)

print("units:", {k: v for k, v in res[res.cond == "original"].groupby("unit").size().items()})
print("windows per speaker by group:", res[(res.unit == "window10") & (res.cond == "original")].groupby(["group", "spk"]).size().groupby("group").describe()[["min", "50%", "max"]].to_dict("index"))
rows = []
for (u, c), g in res.groupby(["unit", "cond"]):
    sp = g.groupby(["spk", "group"]).slope.mean().reset_index()   # 화자 단위 집계
    cl, co = sp[sp.group == "CL"].slope, sp[sp.group == "CO"].slope
    rows.append(dict(unit=u, cond=c, n_units=len(g), slope_mean=g.slope.mean(), slope_sd=g.slope.std(),
                     spk_d_CLvsCO=d_(cl, co), spk_MWU_p=mannwhitneyu(cl, co).pvalue))
print(pd.DataFrame(rows).round(4).to_string())

# 원본 vs 셔플: 같은 단위에서 기울기 상관
for u in ["response", "window10", "session"]:
    o = res[(res.unit == u) & (res.cond == "original")].slope.values
    sh = res[(res.unit == u) & (res.cond == "sent_shuffle")].slope.values
    print(u, "orig mean %.4f  shuffle mean %.4f  corr %.3f  orig>shuffle %.2f" % (o.mean(), sh.mean(), np.corrcoef(o, sh)[0, 1], (o > sh).mean()))

# 누수 실험: 윈도우 단위 피처로 CL/CO 분류, 레코드 단위 vs 화자 단위 CV
X_cols = ["slope", "anchor", "adj", "var", "ttr"]
w = res[(res.unit == "window10") & (res.cond == "original")].reset_index(drop=True)
X, y, grp = w[X_cols].values, (w.group == "CL").astype(int).values, w.spk.values
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
for name, cv, kw in [("record-wise StratifiedKFold", StratifiedKFold(5, shuffle=True, random_state=42), {}),
                     ("speaker-wise StratifiedGroupKFold", StratifiedGroupKFold(5, shuffle=True, random_state=42), {"groups": grp})]:
    s = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", **kw)
    print("%-36s AUC %.3f ± %.3f" % (name, s.mean(), s.std()))
# 화자 ID만으로 맞히는 정도(누수 상한 감각): 피처 + 화자 원핫을 레코드 단위로
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(300, random_state=42)
for name, cv, kw in [("RF record-wise", StratifiedKFold(5, shuffle=True, random_state=42), {}),
                     ("RF speaker-wise", StratifiedGroupKFold(5, shuffle=True, random_state=42), {"groups": grp})]:
    s = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc", **kw)
    print("%-36s AUC %.3f ± %.3f" % (name, s.mean(), s.std()))

# filler(er/erm 등) 제거가 응답 단위 기울기에 주는 영향 (제거하지 않기로 한 근거)
import re
F = re.compile(r"\b(er|erm|mm|mhm|ah|oh)\b", re.I)
def defill(s):
    s = re.sub(r"\b(\w+)( \1\b)+", r"\1", F.sub(" ", s))  # filler 제거 + 즉시 반복 축약
    return " ".join(s.split()) or "_"
fr = []
for spk, grp, sents in B:
    if len(sents) < 4: continue
    a, b = emb(sents), emb([defill(x) for x in sents])
    sl = lambda E: np.polyfit(np.arange(1, len(E)), 1 - E[1:] @ E[0], 1)[0]
    fr.append(dict(spk=spk, group=grp, raw=sl(a), clean=sl(b)))
fr = pd.DataFrame(fr)
sp = fr.groupby(["spk", "group"]).mean(numeric_only=True).reset_index()
for c in ["raw", "clean"]:
    cl, co = sp[sp.group == "CL"][c], sp[sp.group == "CO"][c]
    print(f"filler {c:5s} d {d_(cl, co):+.2f}  MWU p {mannwhitneyu(cl, co).pvalue:.3f}")
print("corr(raw, clean) = %.3f" % np.corrcoef(fr.raw, fr.clean)[0, 1])
