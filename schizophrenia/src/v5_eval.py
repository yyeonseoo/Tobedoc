"""v5: 페르소나 모델 v1 평가 -> reports/v5/01_persona_v1.md. 계획: reports/v5/00_plan.md.

python src/v5_eval.py --rate   # GPT-4.1이 Claude 면담 평가 + Claude 평가용 블라인드 파일(GPT 면담) 생성
python src/v5_eval.py          # 지표 계산 + 판정 + 보고서
"""
import glob
import json
import os
import random
import sys

import numpy as np
import pandas as pd
from scipy import stats

from metrics import PROFILE, Cached, flags
from persona_model import OUT, PROFILES, Budget
from v4_judge import RUBRIC as TLC_RUBRIC, rate_items

REP = os.path.join(OUT, '..', '..', 'reports', 'v5')
ITEMS = ['derailment', 'tangentiality', 'incoherence', 'perseveration', 'clanging', 'circumstantiality', 'poverty_of_speech']
RATE = dict(DER='derailment', INC='incoherence', POV='poverty_of_speech', PER='perseveration')
RUBRIC = TLC_RUBRIC.replace(
    '- circumstantiality:', '- poverty_of_speech: restriction in the amount of spontaneous speech; replies are brief and unelaborated.\n- circumstantiality:').replace(
    '"circumstantiality":n,', '"circumstantiality":n,"poverty_of_speech":n,')
SINGLE = ['DER', 'INC', 'POV']
NCLAUDE = 4  # Claude 평가 에이전트 수


def load():
    rows = []
    for gen, files in [('GPT-4.1', ['gen_gpt.json']), ('Claude', [f'gen_claude_{p}.json' for p in 'ABC'])]:
        for f in files:
            for it in json.load(open(os.path.join(OUT, f), encoding='utf-8'))['interviews']:
                for k, t in enumerate(it['turns']):
                    rows.append(dict(gen=gen, persona=it['persona'], profile=it['profile'], rep=it['rep'], turn=k + 1,
                                     q=t['q'], question=t['question'], text=t['text']))
    d = pd.DataFrame(rows)
    d['iid'] = d.gen + '_' + d.persona + '_' + d.profile + '_' + d.rep.astype(str)
    d['tid'] = d.iid + '_' + d.q
    return d


def rate():
    budget = Budget(os.path.join(OUT, 'spent.json'))
    d = load()
    rate_items(d[d.gen == 'Claude'], os.path.join(OUT, 'judge_gpt_on_claude.csv'), key='tid', rubric=RUBRIC, budget=budget)
    g = d[d.gen == 'GPT-4.1'].to_dict('records')
    random.Random(5).shuffle(g)
    key = {}
    for k in range(NCLAUDE):
        with open(os.path.join(OUT, f'blind_{k + 1}.jsonl'), 'w', encoding='utf-8') as f:
            for j, r in list(enumerate(g))[k::NCLAUDE]:
                key[f'R{j:03d}'] = r['tid']
                f.write(json.dumps(dict(id=f'R{j:03d}', question=r['question'], text=r['text']), ensure_ascii=False) + '\n')
    json.dump(key, open(os.path.join(OUT, 'blind_key.json'), 'w'))
    open(os.path.join(OUT, 'rubric.txt'), 'w', encoding='utf-8').write(RUBRIC)


def ratings():
    key = json.load(open(os.path.join(OUT, 'blind_key.json')))
    c = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, 'judge_claude_on_gpt_*.csv')))])
    c['tid'] = c.id.map(key)
    return pd.concat([pd.read_csv(os.path.join(OUT, 'judge_gpt_on_claude.csv')), c.drop(columns='id')], ignore_index=True)


def mw(a, b):
    return stats.mannwhitneyu(a, b, alternative='greater').pvalue


def main():
    d = load().merge(ratings(), on='tid', validate='1:1')
    sc = Cached()
    d = pd.concat([d.reset_index(drop=True), pd.DataFrame([sc(t, q) for t, q in zip(d.text, d.question)])], axis=1)
    sc.save()
    z = pd.concat([flags(g, g.profile == 'NORM') for _, g in d.groupby('gen')]).loc[d.index]
    d = pd.concat([d, z.add_prefix('z_')], axis=1)
    iv = d.groupby(['gen', 'profile', 'iid'])[ITEMS + [f'z_{t}' for t in PROFILE]].mean().reset_index()
    res = []
    for gen in ['Claude', 'GPT-4.1']:
        n = iv[(iv.gen == gen) & (iv.profile == 'NORM')]
        for p in SINGLE:
            it, item = iv[(iv.gen == gen) & (iv.profile == p)], RATE[p]
            turns = d[(d.gen == gen) & (d.profile == p)]
            r = dict(생성모델=gen, 설정=p, 면담수=len(it), 평가_평균=it[item].mean(), 평가_NORM=n[item].mean(), 발현_p=mw(it[item], n[item]),
                     지표_z=it[f'z_{p}'].mean(), 지표_p=mw(it[f'z_{p}'], n[f'z_{p}']), 지속_1점이상_턴비율=(turns[item] >= 1).mean())
            r['작동'] = bool(r['발현_p'] < .05 and r['평가_평균'] >= 1 and r['지표_p'] < .05 and r['지속_1점이상_턴비율'] >= .75)
            res.append(r)
    res = pd.DataFrame(res)
    prof_means = iv.groupby(['gen', 'profile'])[ITEMS].mean().round(2)
    flag_means = iv.groupby(['gen', 'profile'])[[f'z_{t}' for t in ['DER', 'INC', 'PER', 'POV']]].mean().round(2)
    norm_fp = d[d.profile == 'NORM'].groupby('gen').apply(lambda g: (g[ITEMS].max(axis=1) >= 1).mean())
    words = d.groupby(['gen', 'profile']).n_words.median()
    T = lambda x, i=False: x.to_markdown(index=i, floatfmt='.2f')
    md = f"""# v5: 페르소나 모델 v1 평가

재현: `python src/persona_model.py`(GPT 생성), `python src/v5_eval.py --rate`, `python src/v5_eval.py`. 계획: [00_plan.md](00_plan.md). 모델: [src/persona_model.py](../../src/persona_model.py).

인물 3명 × 증상 설정 5개 × 반복 2 = 면담 30개 × 8턴, 생성 모델마다 240턴. 평가는 교차 블라인드 평가(Claude가 쓴 것은 GPT-4.1, GPT-4.1이 쓴 것은 Claude 에이전트)이고 턴 단위로 섞어서 했다. 지표 표시(z)는 같은 생성 모델의 NORM 턴 기준이다.

## 1. 판정 (단일 증상 설정)

발현 = 면담 평균 평가 점수가 NORM보다 높음(단측 p < .05)이고 평균 1점 이상. 지표 = 면담 평균 표시가 NORM보다 높음(단측 p < .05). 지속 = 의도한 항목이 1점 이상인 턴 비율 ≥ 75%.

{T(res)}

## 2. 설정별 평균 평가 점수 (면담 평균의 평균, 0~3)

{T(prof_means, True)}

## 3. 설정별 지표 표시 (NORM 기준 z, 양수 = 그 유형 쪽)

{T(flag_means, True)}

답변 단어 수 중앙값:

{T(words.unstack(), True)}

NORM 면담에서 어느 항목이든 1점 이상인 턴 비율(오탐): {', '.join(f'{k} {v:.0%}' for k, v in norm_fp.items())}.

## 4. 해석

- **단일 증상 설정 6개(생성 모델 2 × 설정 3) 중 {int(res.작동.sum())}개가 사전 기준을 모두 통과했다.** 통과하지 못한 Claude의 탈선과 말의 빈곤도 NORM과의 차이, 지표, 지속은 모두 통과했고 평균 평가 점수만 1.0에 조금 못 미쳤다(0.96, 0.98). 설정한 증상은 면담 8턴 내내 유지됐고(1점 이상 턴 92~100%), NORM 면담에서는 어느 항목도 1점 이상으로 평가되지 않았다.
- **지표 세트는 면담에서도 의도한 증상 쪽을 가리켰다.** 탈선 설정은 질문 유사도, 지리멸렬 설정은 놀람도, 말의 빈곤 설정은 단어 수 표시가 NORM보다 뚜렷하게 컸다(모두 p < .01).
- **증상이 하나만 나오지는 않았다.** 탈선 설정은 우회증으로도 평가됐고(약 1점), 지리멸렬 설정은 탈선으로도 평가됐다. 생성 모델이 정의를 섞어서 썼거나, 평가자가 비슷한 증상을 함께 표시한 것이다.
- **놀람도는 아주 짧은 답에서 오작동한다.** 말의 빈곤 설정(답변 중앙값 4~15단어)에서 놀람도 표시가 크게 올랐다(z 4~5). 짧은 답은 문맥이 없어 단어마다 예측이 어렵기 때문이다. 지리멸렬 판정에는 길이 조건(예: 20단어 이상)을 붙여야 한다.
- **혼합 설정(탈선 1 + 보속 1 + 말의 빈곤 1)은 평가자가 거의 알아보지 못했다**(각 항목 평균 0.33점 이하). 1단계 증상 세 개를 섞으면 너무 약하다. Claude는 답을 짧게 써서 단어 수와 반복 지표가 움직였지만, GPT-4.1은 말의 빈곤을 거의 반영하지 않았다(답변 중앙값 134단어).
- **생성 모델 차이**: Claude는 모든 설정에서 답이 짧았고(정상 71단어 대 GPT 118단어), GPT-4.1은 증상을 더 강하게 썼다(평가 점수가 더 높음). 지표는 같은 생성 모델의 NORM 기준으로 비교해야 한다.
- **한계**: Claude 면담은 에이전트가 면담 전체를 한 번에 쓴 것이라 실제 여러 턴 호출(GPT-4.1)과 다르다. 평가는 AI가 했고, 인물의 사실 일관성은 확인하지 않았다.
"""
    open(os.path.join(REP, '01_persona_v1.md'), 'w', encoding='utf8').write(md)
    d.drop(columns=['text', 'question']).to_csv(os.path.join(OUT, 'turn_scores.csv'), index=False)
    print(md)


if __name__ == '__main__':
    assert '"poverty_of_speech":n' in RUBRIC and '- poverty_of_speech:' in RUBRIC
    rate() if '--rate' in sys.argv else main()
