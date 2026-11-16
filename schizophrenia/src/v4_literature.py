"""v4 사후 탐색: 선행연구 지표를 단계별 사고장애 발화에 적용 -> reports/v4/02_literature_metrics.md.

v4 결과를 본 뒤 정한 분석이다(계획서 변경 기록 참고). 판정 기준은 01과 같다.
- Elvevåg 2007: 8단어 창(1단어씩 이동)과 질문의 유사도 -> 위치에 대한 기울기(elv_slope), 평균(elv_mean)
- Bedi 2015 / Corcoran 2018: 이웃 문장 유사도 평균·최솟값·표준편차(coh1_mean, coh1_min, coh1_sd)
- Parola 2023: 한 문장 건너뛴 유사도(coh2_mean)
- 우리 제안: 답변 끝 15 내용어와 질문의 유사도(end_sim, 답변 전체 사용)
임베딩은 문헌의 LSA 대신 SBERT. 답변은 앞 100단어(문장부호 유지)로 자른다(end_sim 제외).
"""
import os
import re

import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer

from text import CFG, content, path, words
from v4_judge import load_items

W, CUT, END = 8, 100, 15
# 예상 방향: 사고장애가 심할수록 질문·이웃 문장과의 유사도가 낮다(-), 유사도의 흔들림은 크다(+)
SIGN = dict(elv_slope=-1, elv_mean=-1, coh1_mean=-1, coh1_min=-1, coh1_sd=1, coh2_mean=-1, end_sim=-1)
REP = os.path.join(path('processed'), '..', '..', 'reports', 'v4')


def sentences(text):
    return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if len(words(s)) >= 3]


def cos_next(E, k):
    return np.sum(E[:-k] * E[k:], axis=1) if len(E) > k else np.array([np.nan])


def metrics(m, text, question):
    head = ' '.join(text.split()[:CUT])
    q = m.encode(question, normalize_embeddings=True)
    tok = words(head)
    wins = [' '.join(tok[i:i + W]) for i in range(len(tok) - W + 1)]
    sim = m.encode(wins, normalize_embeddings=True, batch_size=256) @ q
    sents = sentences(text)
    E = m.encode(sents, normalize_embeddings=True) if len(sents) >= 3 else np.zeros((0, 1))
    c1, c2 = cos_next(E, 1), cos_next(E, 2)
    end = ' '.join(content(text)[-END:])
    return dict(elv_slope=np.polyfit(np.arange(len(sim)), sim, 1)[0] * 100, elv_mean=sim.mean(),
                coh1_mean=np.nanmean(c1), coh1_min=np.nanmin(c1), coh1_sd=np.nanstd(c1), coh2_mean=np.nanmean(c2),
                n_sent=len(sents), end_sim=float(m.encode(end, normalize_embeddings=True) @ q))


def main():
    m = SentenceTransformer(CFG['features']['sbert_model'])
    it = load_items()
    f = pd.concat([it[['item_id', 'cond']], pd.DataFrame([metrics(m, t, q) for t, q in zip(it.text, it.question)])], axis=1)
    f['type'] = f.cond.str.extract(r'^([A-Z]+?)\d?$')[0]
    f['level'] = f.cond.str.extract(r'(\d)$')[0].astype(float).fillna(0)
    core = f[f.type.isin(['DER', 'TAN', 'INC']) | (f.cond == 'N0')]
    ctrl = f[f.cond.isin(['NMULTI', 'CIRC1', 'CIRC2'])].groupby('cond')
    ref = f[f.cond.isin(['DER2', 'TAN2'])]
    rows = []
    for c, s in SIGN.items():
        rho = stats.spearmanr(core.level, core[c], nan_policy='omit')[0]
        by = {t: stats.spearmanr(*f[(f.type == t) | (f.cond == 'N0')][['level', c]].dropna().T.values)[0] for t in ['DER', 'TAN', 'INC']}
        spec = all(s * (ctrl[c].mean()[k] - ref[c].mean()) < 0 for k in ['NMULTI', 'CIRC1', 'CIRC2'])
        rows.append(dict(feature=c, rho=rho, rho_DER=by['DER'], rho_TAN=by['TAN'], rho_INC=by['INC'],
                         spec=spec, passed=s * rho >= 0.4 and spec))
    res = pd.DataFrame(rows)
    # 결과를 보고 만든 조합(가설): 평균적으로 질문에서 먼 정도 + 끝에서 돌아오지 않는 정도
    z = lambda s: (s - s.mean()) / s.std()
    f['combo'] = z(f.elv_mean) + z(f.end_sim)
    core = f[f.type.isin(['DER', 'TAN', 'INC']) | (f.cond == 'N0')]
    combo_rho = stats.spearmanr(core.level, core.combo)[0]
    combo_by = {t: stats.spearmanr(*f[(f.type == t) | (f.cond == 'N0')][['level', 'combo']].T.values)[0] for t in ['DER', 'TAN', 'INC']}
    order = ['N0', 'NMULTI', 'CIRC1', 'CIRC2'] + [f'{t}{l}' for t in ['DER', 'TAN', 'INC'] for l in (1, 2, 3)]
    means = f.groupby('cond')[list(SIGN) + ['combo', 'n_sent']].mean().loc[order]
    means['n_coh'] = f.groupby('cond').coh1_mean.count().loc[order]
    R = res.set_index('feature')
    pooled = ref.elv_mean.mean()
    md = f"""# v4 사후 탐색: 선행연구 지표 (01 결과를 본 뒤 정한 분석)

재현: `python src/v4_literature.py`. 발화와 판정 기준은 [01_results.md](01_results.md)와 같다. 임베딩은 SBERT(all-MiniLM-L6-v2). 질문 유사도(elv)는 앞 100단어, 문장 응집성(coh)과 답변 끝 유사도(end_sim)는 답변 전체로 계산한다. 문장 응집성은 3단어 이상 문장이 3개 이상인 답변만 계산되며, 조건별 개수는 n_coh 열이다(조건마다 6개).

| 지표 | 출처 | 계산 | 예상 방향 |
|---|---|---|---|
| elv_slope, elv_mean | Elvevåg 2007 | 8단어 창(1단어씩 이동)과 질문의 유사도의 위치 기울기(×100), 평균 | − |
| coh1_mean, coh1_min, coh1_sd | Bedi 2015, Corcoran 2018 | 이웃 문장 유사도의 평균, 최솟값, 표준편차 | −, −, + |
| coh2_mean | Parola 2023 | 한 문장 건너뛴 문장 유사도의 평균 | − |
| end_sim | 우리 제안 | 답변 끝 15 내용어와 질문의 유사도 | − |

## 판정 (탈선·주제 이탈·지리멸렬 + N0, n = {len(core)})

{res.to_markdown(index=False, floatfmt='.2f')}

## 조건별 평균

{means.to_markdown(floatfmt='.3f')}

## 해석

- **Elvevåg 방식의 질문 유사도 평균(elv_mean)이 사고장애 단계를 가장 잘 따라간다**(ρ = {res.set_index('feature').loc['elv_mean', 'rho']:.2f}). 01의 질문 기준 거리와 같은 결과다. 그러나 우회증 2단계({means.loc['CIRC2', 'elv_mean']:.3f})가 탈선 2단계와 주제 이탈 2단계의 평균({pooled:.3f})보다 낮아 특이도에서 탈락했다.
- **Elvevåg의 기울기(elv_slope)는 반대 방향이다.** 와해된 답변은 처음부터 질문과 멀어서 "점점 멀어지는" 기울기가 생기지 않는다. 01에서 이탈 기울기가 실패한 것과 같은 이유다. 원 논문도 기울기를 사람의 주제 이탈 평가와의 상관(r = 0.44)으로만 보였다.
- **문장 응집성(Bedi, Corcoran, Parola)은 지리멸렬에서 가장 강하게 작동한다**(ρ_INC coh1_mean {R.loc['coh1_mean', 'rho_INC']:.2f}, coh2_mean {R.loc['coh2_mean', 'rho_INC']:.2f}; 탈선 {R.loc['coh1_mean', 'rho_DER']:.2f}, 주제 이탈 {R.loc['coh1_mean', 'rho_TAN']:.2f}). 다만 문장 수가 문장부호에 좌우되어, 심한 조건일수록 문장이 3개 미만이라 계산되지 않는 답변이 있다(n_coh 열, 탈선 3단계 {int(means.loc['DER3', 'n_coh'])}/6, 우회증 2단계 {int(means.loc['CIRC2', 'n_coh'])}/6). 이 칸의 평균은 계산된 답변만의 값이다. 문장부호가 없는 전사(DAIS-C 가공본)에는 그대로 쓸 수 없다.
- **답변 끝 유사도(end_sim)는 특이도를 통과한 유일한 지표다.** 우회증 답변은 끝에서 질문으로 돌아와 높게 나온다. 다만 단독으로는 민감도가 낮다(ρ = {res.set_index('feature').loc['end_sim', 'rho']:.2f}).

## 가설: 두 지표의 조합 (이 데이터로 만든 것이라 검증이 아님)

질문 유사도 평균(elv_mean)과 답변 끝 유사도(end_sim)를 표준화해 더했다. ρ = {combo_rho:.2f} (DER {combo_by['DER']:.2f}, TAN {combo_by['TAN']:.2f}, INC {combo_by['INC']:.2f}). 대조 조건 평균은 NMULTI {means.loc['NMULTI', 'combo']:.2f}, CIRC1 {means.loc['CIRC1', 'combo']:.2f}, CIRC2 {means.loc['CIRC2', 'combo']:.2f}이고, DER2·TAN2 평균은 {(means.loc['DER2', 'combo'] + means.loc['TAN2', 'combo']) / 2:.2f}이다. 결과를 본 뒤 같은 데이터로 만든 조합이라 과적합 가능성이 크다. **새로 생성한 발화(다른 모델 작성)에서 확인해야 한다.**
"""
    open(os.path.join(REP, '02_literature_metrics.md'), 'w', encoding='utf8').write(md)
    print(md)


if __name__ == '__main__':
    assert sentences('Hi there you. Ok. Then we went home!') == ['Hi there you.', 'Then we went home!']
    main()
