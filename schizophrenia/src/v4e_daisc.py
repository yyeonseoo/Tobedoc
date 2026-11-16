"""v4e: 생성 발화(v4 Claude, v4b GPT)에서 찾은 지표를 실제 조현병 면담(DAIS-C)에 적용 -> reports/v4/06_daisc.md.

DAIS-C 가공본은 불용어가 지워진 내용어 목록이라, 지표를 내용어 기준으로 다시 정의하고 생성 발화에도 같은 정의로 계산해
방향(예상 부호)을 먼저 정한 뒤 DAIS-C에 적용한다.
- q_mean: 내용어 5개 창(1개씩 이동)과 면담자 질문(내용어)의 SBERT 유사도 평균
- q_end: 마지막 10 내용어와 질문의 유사도
- sem_density: 앞 15 내용어의 단어 벡터 평균의 크기 (길이 고정)
- combo: z(q_mean) + z(q_end) (코퍼스 안에서 표준화)
분석: ① 같은 사람 안 DT 표시 답변 대 일반 답변 ② 같은 사람 안 되물음 받은 답변 대 아닌 답변 ③ 환자군 대 대조군(화자 평균, 참고)
길이 보정: 지표를 log(내용어 수)에 회귀한 잔차로도 본다.
"""
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer

from text import CFG, content, path, words
from v4_judge import load_items

DATA = os.path.dirname(path('processed'))
REP = os.path.join(DATA, '..', 'reports', 'v4')
MIN, WIN, END, DENS = 15, 5, 10, 15
FEATS = ['q_mean', 'q_end', 'combo', 'sem_density']
# 되물음: 맞장구("I know what you mean")와 사과("sorry to hear")는 빼고 실제로 다시 묻는 표현만
CLAR = re.compile(r'(what|how) do you mean|what you mean by|did you say|say that again|pardon|didnt catch|can you repeat|what was that|not sure i follow')
# 축약형 잔여물과 말버릇: words()가 아포스트로피를 지워 불용어 목록을 빠져나간 것. DAIS-C에 특히 많다
FILL = set('im ive id ill youre youve theyre weve hes shes thats whats theres dont didnt doesnt isnt wasnt arent werent cant couldnt wouldnt shouldnt wont havent hasnt hadnt lets like yeah yes know mean kind sort really just well okay ok um uh er erm oh gonna wanna got get'.split())


def clean(w):
    return [x for x in w if x not in FILL]


def compute(m, units):
    """units: (내용어 목록, 질문 내용어 목록)."""
    wins = [[' '.join(w[i:i + WIN]) for i in range(len(w) - WIN + 1)] for w, _ in units]
    texts = sorted({t for ws in wins for t in ws} | {' '.join(w[-END:]) for w, _ in units} | {' '.join(a) for _, a in units})
    emb = dict(zip(texts, m.encode(texts, normalize_embeddings=True, batch_size=256)))
    vocab = sorted({x for w, _ in units for x in w[:DENS]})
    we = dict(zip(vocab, m.encode(vocab, normalize_embeddings=True, batch_size=512)))
    rows = []
    for (w, a), ws in zip(units, wins):
        q = emb[' '.join(a)]
        rows.append(dict(q_mean=float(np.mean([emb[t] @ q for t in ws])), q_end=float(emb[' '.join(w[-END:])] @ q),
                         sem_density=float(np.linalg.norm(np.mean([we[x] for x in w[:DENS]], axis=0))), n_words=len(w)))
    f = pd.DataFrame(rows)
    z = lambda s: (s - s.mean()) / s.std()
    f['combo'] = z(f.q_mean) + z(f.q_end)
    return f


def generated_signs(m):
    """생성 발화 두 세트에서 GPT-4.1 전체 와해 점수와의 ρ. 두 세트 부호가 같으면 그 부호를 예상 방향으로 쓴다."""
    out = {}
    a = load_items().merge(pd.read_csv(os.path.join(DATA, 'v4', 'judge.csv')), on='item_id')
    d = json.load(open(os.path.join(DATA, 'v4b', 'gen_gpt.json'), encoding='utf-8'))
    b = pd.DataFrame(d['items']).assign(item_id=lambda x: x.q + '_' + x.cond, question=lambda x: x.q.map(d['questions']))
    b = b.merge(pd.read_csv(os.path.join(DATA, 'v4b', 'judge_gpt.csv')), on='item_id')
    for name, s in [('v4 Claude', a), ('v4b GPT', b)]:
        f = compute(m, [(clean(content(t)), clean(content(q))) for t, q in zip(s.text, s.question)])
        out[name] = {c: stats.spearmanr(f[c], s.overall_disorganization)[0] for c in FEATS}
    t = pd.DataFrame(out)
    t['sign'] = np.where(np.sign(t.iloc[:, 0]) == np.sign(t.iloc[:, 1]), np.sign(t.iloc[:, 0]), 0)
    return t


def within(f, flag, feats, sign):
    rows = []
    for c in feats:
        for adj in ['', '_adj']:
            g = f[f.groupby('speaker_id')[flag].transform(lambda s: s.any() and not s.all())]
            t = g.groupby(['speaker_id', flag])[c + adj].mean().unstack().dropna()
            diff = sign[c] * (t[True] - t[False])  # 양수 = 예상 방향(사고장애 쪽이 더 '와해'됨)
            k = int((diff > 0).sum())
            rows.append(dict(feature=c + (' (길이 보정)' if adj else ''), speakers=len(diff), expected=k,
                             sign_p=stats.binomtest(k, len(diff)).pvalue if len(diff) else np.nan,
                             wilcoxon_p=stats.wilcoxon(diff).pvalue if len(diff) >= 5 else np.nan))
    return pd.DataFrame(rows)


def main():
    m = SentenceTransformer(CFG['features']['sbert_model'])
    signs = generated_signs(m)
    feats = [c for c in FEATS if signs.loc[c, 'sign'] != 0]
    # 와해 점수와의 상관 부호: −1이면 값이 낮을수록 와해. diff = sign*(DT − 일반) > 0 이 예상 방향
    sign = {c: signs.loc[c, 'sign'] for c in feats}

    d = pd.read_parquet(os.path.join(path('processed'), 'daisc_units.parquet'))
    d['w'], d['a'] = d.words.map(lambda x: clean(list(x))), d.anchor_words.map(lambda x: clean(list(x)))
    d = d[(d.w.map(len) >= MIN) & (d.a.map(len) >= 2)].reset_index(drop=True)
    f = pd.concat([d[['speaker_id', 'group']], compute(m, list(zip(d.w, d.a)))], axis=1)
    f['short_q'] = d.a.map(len) <= 5
    f['dt'] = d.in_DT.map(lambda v: bool(np.any(v)))
    f['clar'] = d.next_prompt.fillna('').map(lambda s: bool(CLAR.search(' '.join(words(s)))))
    for c in feats:
        b = np.polyfit(np.log(f.n_words), f[c], 1)
        f[c + '_adj'] = f[c] - np.polyval(b, np.log(f.n_words))
    dt, cl = within(f, 'dt', feats, sign), within(f, 'clar', feats, sign)
    dt_short = within(f[f.short_q], 'dt', feats, sign)
    sp = f.groupby(['speaker_id', 'group'])[feats + ['n_words']].mean().reset_index()
    grp = pd.DataFrame([dict(feature=c, CL=sp[sp.group == 'CL'][c].mean(), CO=sp[sp.group == 'CO'][c].mean(),
                             mwu_p=stats.mannwhitneyu(sp[sp.group == 'CL'][c], sp[sp.group == 'CO'][c]).pvalue) for c in feats + ['n_words']])
    lens = f.groupby('dt').n_words.median()
    D = dt.set_index('feature')
    md = f"""# v4e: 생성 발화에서 찾은 지표를 실제 조현병 면담(DAIS-C)에 적용

재현: `python src/v4e_daisc.py`. 결과를 본 뒤 정한 탐색이다.

DAIS-C 가공본은 내용어만 남아 있어 지표를 내용어 기준으로 다시 정의했다(내용어 5개 창과 질문의 유사도 평균 q_mean, 마지막 10 내용어와 질문의 유사도 q_end, 앞 15 내용어의 의미 밀도 sem_density, combo = z(q_mean) + z(q_end)). 축약형 잔여물(im, dont, thats 등)과 말버릇(like, yeah, know, mean 등)은 불용어 목록을 빠져나가 내용어로 남으므로, 생성 발화와 DAIS-C 모두에서 뺐다(DAIS-C에서 특히 많다). 정리 후 내용어 {MIN}개 이상이고 질문 내용어가 2개 이상인 답변 {len(f)}개({f.speaker_id.nunique()}명)를 썼다.

생성 발화의 질문은 내용어 2~3개짜리 한 문장이지만, DAIS-C의 '질문'은 직전 면담자 발화 전체라 더 길고(내용어 중앙값 {int(d.a.map(len).median())}개) 맞장구나 끊긴 말이 섞여 있다. 그래서 질문 내용어가 5개 이하인 답변만으로도 같은 비교를 했다(2절 아래).

## 1. 같은 정의로 생성 발화에서 본 방향 (GPT-4.1 전체 와해 점수와의 ρ)

{signs.to_markdown(floatfmt='.2f')}

두 세트에서 부호가 같은 지표만 DAIS-C에 적용했다: {', '.join(feats)}.

## 2. 같은 사람 안: DT 표시 답변 대 일반 답변

DT는 DAIS-C 연구진이 자신들의 사고장애 분석 틀에서 담화 추적(discourse tracking) 특징으로 표시한 부분이다(코퍼스 Readme). 임상적인 사고장애 판정이 아니고, 표시 절차와 평가자 일치도는 공개되지 않았다. 참여 환자는 항정신병약을 꾸준히 복용 중이고 연구 참여 동의 능력이 있다고 판단된 사람으로 제한되어, 급성기 환자는 포함되기 어렵다.

DT 답변 {int(f.dt.sum())}개, 일반 {int((~f.dt).sum())}개. 내용어 수 중앙값: DT {lens.get(True, np.nan):.0f}, 일반 {lens.get(False, np.nan):.0f}. expected = DT 답변이 예상 방향(더 와해)인 화자 수.

{dt.to_markdown(index=False, floatfmt='.3f')}

질문 내용어 5개 이하인 답변만 ({int(f.short_q.sum())}개, DT {int((f.short_q & f.dt).sum())}개):

{dt_short.to_markdown(index=False, floatfmt='.3f')}

## 3. 같은 사람 안: 면담자가 되물은 답변 대 아닌 답변

"what do you mean", "did you say", "pardon"처럼 실제로 다시 묻는 경우만 셌다. "I know what you mean" 같은 맞장구와 "sorry to hear" 같은 말은 뺐다. 되물음 답변은 {int(f.clar.sum())}개, 두 종류 답변을 모두 가진 화자는 {cl.speakers.max() if len(cl) else 0}명이라 검정할 수 없다.

## 4. 환자군 대 대조군 (화자 평균, 녹음 방식·성별 교란으로 참고용)

{grp.to_markdown(index=False, floatfmt='.3f')}

## 5. 해석

- **생성 발화에서 찾은 질문 기준 지표는 실제 환자 면담에서 작동하지 않았다.** 같은 사람 안에서 DT 표시 답변과 일반 답변을 비교하면 예상 방향인 화자가 절반 정도였다(q_mean {int(D.loc['q_mean', 'expected'])}/{int(D.loc['q_mean', 'speakers'])}). 집단 기준도 차이가 없었다. 질문이 짧은 답변만 보면 {int(dt_short.set_index('feature').loc['q_mean', 'expected'])}/{int(dt_short.set_index('feature').loc['q_mean', 'speakers'])}명이 예상 방향이었지만 표시 답변이 {int((f.short_q & f.dt).sum())}개뿐이라 판단할 수 없다. 질문 정의 차이가 결과를 가렸을 가능성은 남는다.
- **의미 밀도는 차이가 없었다**({int(D.loc['sem_density', 'expected'])}/{int(D.loc['sem_density', 'speakers'])}). 말버릇 단어를 빼기 전에는 표시 답변의 의미 밀도가 더 높게 나왔는데(13명 중 11명), like, yeah 같은 단어가 반복되어 생긴 차이였다.
- 가능한 이유:
  1. **실제 사고장애는 교과서적 사고장애보다 훨씬 약하고 부분적이다.** 생성 발화의 2~3단계는 답변 대부분이 질문에서 벗어나지만, DAIS-C의 표시 답변은 대부분 질문과 관련된 말 속에 일부 구간만 이상하다.
  2. **DT는 사고장애 판정이 아니다.** 연구진이 담화 추적 특징으로 표시한 부분이라, 지표가 잡으려는 사고장애와 같은 대상이라는 보장이 없다. 표시 절차와 일치도도 공개되지 않았다.
  3. **수가 적다.** 표시 답변 {int(f.dt.sum())}개, 비교 가능한 화자 {int(D.loc['q_mean', 'speakers'])}명이다.
- 정리: 생성 발화에서 작동한 지표가 실제 환자 발화로 그대로 옮겨진다는 근거는 없다.
"""
    open(os.path.join(REP, '06_daisc.md'), 'w', encoding='utf8').write(md)
    print(md)


if __name__ == '__main__':
    main()
