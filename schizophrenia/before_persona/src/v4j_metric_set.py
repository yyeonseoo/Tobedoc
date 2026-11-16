"""v4j: 고정 지표 세트(metrics.py) 확인 + 말 빈곤 지표 판정 -> reports/v4/11_metric_set.md. 계획: 00_plan.md 'v4j 계획'."""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from metrics import PROFILE, Cached, flags
from text import path
from v4_judge import load_items, rate_items
from v4g_compare import load_v4g

DATA = os.path.dirname(path('processed'))
REP = os.path.join(DATA, '..', 'reports', 'v4')
TYPES = ['DER', 'TAN', 'INC', 'PER', 'CLANG']
RATE = dict(DER='derailment', TAN='tangentiality', INC='incoherence', PER='perseveration', CLANG='clanging',
            POV='poverty_of_speech', POVC='poverty_of_content')
CALM = ['NMULTI', 'CIRC1', 'CIRC2']
BASE = ['N0', 'NMULTI', 'CIRC1', 'CIRC2', 'PER1', 'PER2', 'PER3']
INTERP = '''
## 4. 해석

- **지표 7개 중 6개가 두 생성 모델 모두에서 사전 기준을 통과했다.** 사고장애 5종과 말의 빈곤이다. 내용 빈곤은 Claude 세트에서 실패하고 GPT-4.1 세트에서 기준선을 겨우 넘어, 작동한다고 보기 어렵다. metrics.py 계산은 기존 스크립트 값과 같았다.
- **모든 평가 점수는 교차 평가다.** 쓴 모델과 다른 모델이 조건을 모른 채 평가했다(Claude가 쓴 것은 GPT-4.1, GPT-4.1이 쓴 것은 Claude). 자기 평가와 비교하면, 사고장애 5종의 GPT-4.1 세트는 교차 평가 ρ가 자기 평가와 비슷하거나 더 높았고(보속 0.38 → 0.61), 말 빈곤 Claude 세트는 자기 평가가 더 높았다(말의 빈곤 0.74 → 0.59). 자기 평가가 결과를 일관되게 부풀리지는 않았지만, 교차 평가가 더 공정한 기준이다.
- **신뢰구간(부트스트랩 95%).** 모든 구간이 0을 넘었다. 다만 유형당 답변이 24개라 구간이 넓다. 구간 하한까지 기준을 넘지 못한 것은 Claude 세트 보속의 평가 ρ(하한 0.16), GPT-4.1 탈선의 단계 ρ(하한 0.30), 내용 빈곤의 평가 ρ(두 세트)다. 페르소나 모델 평가에서는 조건마다 답변을 더 많이 만들어(예: 반복 3회) 구간을 좁혀야 한다.
- **지표 하나가 한 유형만 가리키지는 않는다(2절).** 질문 유사도는 우회증에서도 켜지고(탈선 대조구분 실패), 놀람도는 탈선과 음향 연상에서도 오르고, 어휘 다양도는 탈선에서도 떨어졌다. 그래서 유형 판정은 지표 하나가 아니라 여러 표시를 함께 보고 해야 한다. 예를 들어 놀람도와 운율이 함께 높으면 음향 연상, 놀람도만 높으면 지리멸렬이다.
- **말의 빈곤은 생성 방식에 따라 크게 달랐다.** Claude는 1단계도 평균 14단어, 3단계는 한두 단어로 과장되게 썼고, GPT-4.1은 1~3단계 모두 20단어 안팎으로 단계 차이가 작았다. 실제 환자의 말의 빈곤 정도와는 둘 다 다를 수 있다.
- **내용 빈곤(info_rate)이 실패한 이유.** 두 평가자 모두 지리멸렬과 보속 답변도 내용 빈곤으로 높게 평가했다. 보속 답변은 info_rate도 낮아 문제가 없지만, 지리멸렬 답변은 서로 다른 단어가 많아 info_rate가 오히려 높다. 정상, 대조, 보속, 말 빈곤만 보면 두 세트 모두 ρ 0.73 이상이다. info_rate는 "단어 종류가 적은 빈 말"만 잡고, "단어는 많지만 뜻이 통하지 않는 말"은 놀람도가 맡아야 한다. 내용 빈곤과 보속도 info_rate로는 구분되지 않아 반복 지표(mattr)를 함께 봐야 한다.
'''


def split(cond):
    t = cond.str.extract(r'^([A-Z]+?)(\d?)$')
    return t[0], t[1].replace('', '0').astype(int)


def boot(x, y, n=2000, seed=0):
    """Spearman ρ의 부트스트랩 95% 구간(행 단위 복원 추출)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    rng = np.random.default_rng(seed)
    r = [stats.spearmanr(x[i], y[i])[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    return np.nanpercentile(r, [2.5, 97.5])


def judge(f, t):
    c, s = PROFILE[t]
    core = f[(f.type == t) | (f.cond == 'N0')]
    m = f.groupby('cond')[c].mean()
    lo, hi = sorted(s * boot(core.level, core[c]))
    plo, phi = sorted(s * boot(f[c], f[RATE[t]]))
    return dict(유형=t, 지표=c, n=len(core), 단계_rho=s * stats.spearmanr(core.level, core[c])[0], 단계_95CI=f'{lo:.2f}~{hi:.2f}',
                평가_rho=s * stats.spearmanr(f[c], f[RATE[t]])[0], 평가_95CI=f'{plo:.2f}~{phi:.2f}',
                대조구분=bool(all(s * m[k] < s * m[f'{t}2'] for k in CALM if k in m)),
                **({'평가_rho_비교조건만': s * (lambda g: stats.spearmanr(g[c], g[RATE[t]])[0])(f[f.cond.isin(BASE) | f.type.isin(['POV', 'POVC'])])}
                   if t in ('POV', 'POVC') else {}),
                **({'자기평가_rho': s * (lambda g: stats.spearmanr(g[c], g['self_' + RATE[t]])[0])(f.dropna(subset=['self_' + RATE[t]]))}
                   if 'self_' + RATE[t] in f else {}))


def keyed(csv, key):
    """블라인드 id 평가 -> item_id 평가."""
    k = json.load(open(os.path.join(DATA, 'v4j', key)))
    j = pd.concat([pd.read_csv(os.path.join(DATA, 'v4j', c)) for c in csv])
    return j.assign(item_id=[k[i]['q'] + '_' + k[i]['cond'] for i in j.id]).drop(columns='id')


def poverty():
    """-> {세트 이름: 말 빈곤 36개 + 같은 생성 모델의 기존 114개}. 주 평가 = 다른 모델의 교차 평가, self_* = 자기 평가(참고)."""
    out = {}
    POV = ['poverty_of_speech', 'poverty_of_content']
    self_ = lambda j: j.rename(columns={c: 'self_' + c for c in POV})
    for name, gen, base, cross, own in [
            ('Claude 말 빈곤', 'gen_claude.json', load_items, lambda: pd.read_csv(os.path.join(DATA, 'v4j', 'judge_x_gpt.csv')),
             lambda: keyed(['judge_claude.csv'], 'blind_key.json')),
            ('GPT-4.1 말 빈곤', 'gen_gpt.json', load_v4g, lambda: keyed(['judge_x1.csv', 'judge_x2.csv'], 'blind_x_key.json'),
             lambda: pd.read_csv(os.path.join(DATA, 'v4j', 'judge_poverty.csv')))]:
        d = json.load(open(os.path.join(DATA, 'v4j', gen), encoding='utf-8'))
        new = pd.DataFrame(d['items']).assign(question=lambda x: x.q.map(d['questions']))
        it = pd.concat([base()[['q', 'cond', 'question', 'text']], new], ignore_index=True)
        it['item_id'] = it.q + '_' + it.cond
        out[name] = it.merge(cross(), on='item_id', validate='1:1').merge(self_(own()), on='item_id', how='left')
    return out


def main():
    sc = Cached()
    S = {'Claude': load_items().merge(pd.read_csv(os.path.join(DATA, 'v4', 'judge.csv')).drop(columns='model'), on='item_id'),
         # GPT-4.1 세트: 주 평가 = Claude 교차 평가, self_* = GPT-4.1 자기 평가(참고)
         'GPT-4.1': (lambda g: g.merge(keyed(['judge_t1.csv', 'judge_t2.csv'], 'blind_t_key.json'), on='item_id', validate='1:1').merge(
             rate_items(g, os.path.join(DATA, 'v4g', 'judge_gpt.csv')).rename(columns=lambda c: c if c == 'item_id' else 'self_' + c), on='item_id'))(load_v4g())}
    pv = poverty()
    S.update(pv)
    pmeans = {}
    res, same, heat = [], [], {}
    for name, it in S.items():
        f = pd.concat([it.reset_index(drop=True), pd.DataFrame([sc(t, q) for t, q in zip(it.text, it.question)])], axis=1)
        f['type'], f['level'] = split(f.cond)
        old = {'Claude': 'v4', 'GPT-4.1': 'v4g'}.get(name)
        if old:  # 기존 계산과 같은지
            ref = pd.read_csv(os.path.join(DATA, old, 'features_all.csv')).set_index('item_id')
            for c in ['elv_mean', 'mattr', 'clang_rate']:
                same.append(dict(세트=name, 지표=c, 최대차이=float((f.set_index('item_id')[c] - ref[c].reindex(f.item_id)).abs().max())))
        for t in (['POV', 'POVC'] if name in pv else TYPES):
            res.append(dict(세트=name, **judge(f, t)))
        if name not in pv:
            z = flags(f, f.cond == 'N0').assign(cond=f.cond)
            heat[name] = z[z.cond.str.match(r'^[A-Z]+[23]$|^(N0|NMULTI|CIRC\d)$')].assign(
                cond=lambda x: x.cond.str.replace(r'[23]$', '2~3', regex=True)).groupby('cond')[TYPES].mean()
        else:
            pmeans[name] = f[f.cond.isin(BASE) | f.type.isin(['POV', 'POVC'])].groupby('cond')[['n_words', 'info_rate', 'poverty_of_speech', 'poverty_of_content']].mean()
    sc.save()
    res = pd.DataFrame(res)
    res['작동'] = (res.단계_rho >= 0.4) & (res.평가_rho >= 0.3)
    res['하한도_통과'] = res.단계_95CI.str.split('~').str[0].astype(float).ge(0.4) & res.평가_95CI.str.split('~').str[0].astype(float).ge(0.3)
    T = lambda x, i=False: x.to_markdown(index=i, floatfmt='.2f')
    pov_txt = """## 3. 말 빈곤 지표

두 세트 모두 말 빈곤 36개와 같은 생성 모델의 기존 114개(정상, 대조, 사고장애 5종)를 섞어, 쓴 모델과 다른 모델이 조건을 모른 채 평가했다(Claude가 쓴 세트는 GPT-4.1, GPT-4.1이 쓴 세트는 Claude 에이전트). 표의 평가 점수는 이 교차 평가다. 자기평가_rho는 쓴 모델이 직접 평가한 점수와의 ρ로 참고용이다(Claude 자기 평가는 78개만). 평가_rho_비교조건만은 정상, 대조, 보속, 말 빈곤만으로 계산한 값이다.

""" + '\n\n'.join(f'**{k}**\n\n{T(v, True)}' for k, v in pmeans.items()) + '\n' if pv else ''
    md = f"""# v4j: 고정 지표 세트

재현: `python src/v4j_metric_set.py`. 지표 계산: [src/metrics.py](../../src/metrics.py). 계획: [00_plan.md](00_plan.md) 'v4j 계획'.

단계_rho와 평가_rho는 예상 방향이면 양수가 되도록 부호를 맞췄다. 작동 = 단계_rho ≥ 0.4 그리고 평가_rho ≥ 0.3 (사전 기준). 95CI는 답변을 복원 추출한 부트스트랩 2000회의 95% 구간이고, 하한도_통과는 구간 하한까지 기준을 넘은 경우다(사후에 더한 보수적 판정). 대조구분 = 화제 전환과 우회증이 그 유형 2단계보다 덜 그 유형 쪽으로 나옴.

## 1. 유형별 담당 지표

{T(res)}

기존 계산과의 차이(0이면 같음): {', '.join(f"{r.세트} {r.지표} {r.최대차이:.1e}" for r in pd.DataFrame(same).itertuples())}

## 2. 어떤 조건에서 어떤 표시가 켜지나

정상 답변 기준 z 점수(양수 = 그 유형 쪽). 행은 생성 조건(2~3단계 합침), 열은 지표가 가리키는 유형이다. 대각선이 크고 나머지가 작을수록 유형을 구분한다. DER과 TAN은 같은 지표라 열이 같다.

""" + '\n\n'.join(f'**{k}**\n\n{T(v, True)}' for k, v in heat.items()) + '\n\n' + pov_txt + INTERP
    open(os.path.join(REP, '11_metric_set.md'), 'w', encoding='utf8').write(md)
    print(md)


if __name__ == '__main__':
    t, l = split(pd.Series(['N0', 'DER2', 'NMULTI', 'POVC3']))
    assert list(t) == ['N', 'DER', 'NMULTI', 'POVC'] and list(l) == [0, 2, 0, 3]
    main()
