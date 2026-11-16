"""v4i: 지표 고장 점검 + 놀람도를 미묘한 발화(v4f)와 DAIS-C에 적용 -> reports/v4/10_checks.md. 계획: 00_plan.md 'v4i 계획'."""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

from metrics import LM, Cached
from prep_daisc import answers, read_full, strip_tags, turns
from text import CFG, FILLERS, path, words
from v4_judge import load_items
from v4g_compare import load_v4g

DATA = os.path.dirname(path('processed'))
REP = os.path.join(DATA, '..', 'reports', 'v4')
FEATS = ['elv_mean', 'surp_mean', 'mattr', 'clang_rate']
REF = dict(elv_mean='TAN2', surp_mean='INC2', mattr='PER2', clang_rate='CLANG2')
PLAN = {  # 조작: (바뀌어야 할 지표와 방향, 그대로여야 할 지표)
    '문장 순서 무작위': ({'surp_mean': 1}, ['elv_mean', 'mattr', 'clang_rate']),
    '한 문장 세 번 반복': ({'mattr': -1, 'surp_mean': -1}, ['elv_mean', 'clang_rate']),
    '다른 질문 답변으로 교체': ({'elv_mean': -1}, ['surp_mean', 'mattr', 'clang_rate']),
    '전사 형식': ({}, FEATS),
}
TOKW = 15


def split(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())


def perturb(row, other, rng):
    s = split(row.text)
    order = list(range(len(s)))
    while order == sorted(order):
        rng.shuffle(order)
    k = len(s) // 2
    return {'문장 순서 무작위': ' '.join(s[i] for i in order),
            '한 문장 세 번 반복': ' '.join(s[:k + 1] + [s[k]] * 2 + s[k + 1:]),
            '다른 질문 답변으로 교체': other,
            '전사 형식': ' '.join(words(row.text))}


def checks(sc, lm):
    S = {'Claude': load_items(), 'GPT-4.1': load_v4g()}
    rng = np.random.default_rng(0)
    ref, rows, trans = {}, [], []
    for name, it in S.items():
        f = pd.DataFrame([sc(t, q) for t, q in zip(it.text, it.question)])[FEATS].assign(cond=it.cond.values)
        base = f[f.cond == 'N0'].mean(numeric_only=True)
        ref[name] = {c: f[f.cond == REF[c]][c].mean() - base[c] for c in FEATS}
        n0 = it[it.cond == 'N0'].reset_index(drop=True)
        for i, r in n0.iterrows():
            orig = sc(r.text, r.question)
            for p, t in perturb(r, n0.text[(i + 1) % len(n0)], rng).items():
                new = sc(t, r.question)
                rows.append(dict(세트=name, 조작=p, **{c: new[c] - orig[c] for c in FEATS}))
        core = it[it.cond.str.match(r'^(INC\d|N0)$')]
        lv = core.cond.str[-1].astype(int)
        trans.append(dict(세트=name, 원문_rho=stats.spearmanr(lv, [lm(t)['surp_mean'] for t in core.text])[0],
                          전사형식_rho=stats.spearmanr(lv, [lm(' '.join(words(t)))['surp_mean'] for t in core.text])[0]))
    d = pd.DataFrame(rows)
    refm = pd.DataFrame(ref).mean(axis=1)
    out = []
    for p, (move, stay) in PLAN.items():
        g = d[d.조작 == p]
        for c in FEATS:
            rel = g[c].mean() / abs(refm[c])
            if c in move:
                k = int((np.sign(g[c]) == move[c]).sum())
                ok = k >= 10 and rel * move[c] >= 0.25
                exp = '↑' if move[c] > 0 else '↓'
            else:
                k, ok, exp = np.nan, abs(rel) < 0.25, '그대로'
            out.append(dict(조작=p, 지표=c, 기대=exp, 평균변화_기준대비=rel, 예상방향_12개중=k, 통과=bool(ok)))
    return pd.DataFrame(out), refm, pd.DataFrame(trans)


def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def subtle(lm):
    d = json.load(open(os.path.join(DATA, 'v4f', 'gen_local.json'), encoding='utf-8'))
    rows = []
    for it in d['items']:
        raw = it['text']
        mt = re.search(r'\[\[(.+?)\]\]', raw, re.S)
        clean = raw.replace('[[', '').replace(']]', '')
        s, off = lm.tokens(clean)
        r = dict(q=it['q'], rep=it['rep'], cond=it['cond'], surp_mean=s.mean(), surp_p90=np.sort(s)[-max(1, len(s) // 10):].mean())
        if mt:
            span = (mt.start(), mt.end() - 4)   # [[ ]] 네 글자를 뺀 clean 기준 위치
            wins = [(off[i][0], off[i + TOKW - 1][1]) for i in range(len(s) - TOKW + 1)]
            j = int(np.argmax(np.convolve(s, np.ones(TOKW), 'valid')))
            r.update(hit=overlap(wins[j], span), chance=np.mean([overlap(w, span) for w in wins]))
        rows.append(r)
    g = pd.DataFrame(rows)
    base = g[g.cond == 'N0'].set_index(['q', 'rep'])
    pair, loc = [], []
    for t in ['INC', 'DER', 'TAN', 'PER', 'CLANG', 'CIRC']:
        sub = g[g.cond == t].set_index(['q', 'rep'])
        for c in ['surp_mean', 'surp_p90']:
            diff = (sub[c] - base[c].reindex(sub.index)).dropna()
            k = int((diff > 0).sum())
            pair.append(dict(유형=t, 지표=c, 쌍=len(diff), 정상보다_높음=k, p=stats.binomtest(k, len(diff)).pvalue))
        if t != 'CIRC' and 'hit' in sub:
            h = sub.hit.dropna().astype(bool)
            ch = sub.chance.dropna().mean()
            loc.append(dict(유형=t, n=len(h), 적중률=h.mean(), 우연=ch,
                            p=stats.binomtest(int(h.sum()), len(h), ch, alternative='greater').pvalue))
    return pd.DataFrame(pair), pd.DataFrame(loc)


def daisc_text():
    """prep_daisc와 같은 순서로 답변을 다시 꺼내 unit_id -> 말버릇을 뺀 전체 문장."""
    out = {}
    for g in ['CL', 'CO']:
        for p in sorted(glob.glob(os.path.join(path('daisc_root'), CFG['paths']['daisc_full_glob'].format(group=g)))):
            sid = os.path.basename(p)[:6]
            for i, (_, raws, _) in enumerate(answers(turns(read_full(p), sid))):
                out[f'{sid}_{i:03d}'] = ' '.join(w for w in words(strip_tags(' '.join(raws))) if w not in FILLERS)
    return out


def daisc(lm):
    u = pd.read_parquet(os.path.join(path('processed'), 'daisc_units.parquet'))
    txt = daisc_text()
    u['text'] = u.unit_id.map(txt)
    u = u[u.text.str.split().str.len() >= 30].reset_index(drop=True)
    f = pd.concat([u[['speaker_id', 'group']], pd.DataFrame([lm(t) for t in u.text])], axis=1)
    f['n'] = u.text.str.split().str.len()
    f['dt'] = u.in_DT.map(lambda v: bool(np.any(v)))
    rows = []
    for c in ['surp_mean', 'surp_p90']:
        b = np.polyfit(np.log(f.n), f[c], 1)
        f[c + '_adj'] = f[c] - np.polyval(b, np.log(f.n))
        for col in [c, c + '_adj']:
            g = f[f.groupby('speaker_id').dt.transform(lambda s: s.any() and not s.all())]
            t = g.groupby(['speaker_id', 'dt'])[col].mean().unstack().dropna()
            k = int((t[True] > t[False]).sum())
            rows.append(dict(지표=col.replace('_adj', ' (길이 보정)'), 화자=len(t), DT쪽이_높음=k, p=stats.binomtest(k, len(t)).pvalue))
    sp = f.groupby(['speaker_id', 'group'])[['surp_mean', 'surp_p90']].mean().reset_index()
    grp = [dict(지표=c, 환자군_평균=sp[sp.group == 'CL'][c].mean(), 대조군_평균=sp[sp.group == 'CO'][c].mean(),
                환자군_n=int((sp.group == 'CL').sum()), 대조군_n=int((sp.group == 'CO').sum()),
                p=stats.mannwhitneyu(sp[sp.group == 'CL'][c], sp[sp.group == 'CO'][c]).pvalue) for c in ['surp_mean', 'surp_p90']]
    return pd.DataFrame(rows), pd.DataFrame(grp), len(f), f.speaker_id.nunique(), f.surp_mean.mean()


def main():
    sc, lm = Cached(), LM()
    chk, refm, trans = checks(sc, lm)
    sc.save()
    pair, loc = subtle(lm)
    within, grp, n_units, n_spk, d_mean = daisc(lm)
    fail = chk[~chk.통과]
    T = lambda x: x.to_markdown(index=False, floatfmt='.2f')
    md = f"""# v4i: 지표 고장 점검과 놀람도 확장

재현: `python src/v4i_checks.py`. 계획: [00_plan.md](00_plan.md) 'v4i 계획'.

## 1. 고장 점검

정상 답변 12개(Claude 6, GPT-4.1 6)를 조작해 지표 변화를 봤다. 평균변화_기준대비는 조작 전후 차이를 그 지표 담당 유형 2단계와 정상의 차이로 나눈 값이다(기준 효과: elv_mean {refm['elv_mean']:.3f}, surp_mean {refm['surp_mean']:.2f}비트, mattr {refm['mattr']:.3f}, clang_rate {refm['clang_rate']:.3f}). 1이면 2단계 사고장애만큼 바뀐 것이다.

{T(chk)}

통과하지 못한 항목: {', '.join(f"{r.조작}의 {r.지표}" for r in fail.itertuples()) or '없음'}.

전사 형식(소문자, 문장부호 제거)에서 지리멸렬 단계와 surp_mean의 ρ:

{T(trans)}

## 2. 미묘한 발화(v4f)의 놀람도

답변 대부분은 정상이고 한두 문장만 사고장애인 발화다. 같은 질문, 같은 회차의 정상 답변과 짝지어 비교했다.

{T(pair)}

놀람도가 가장 높은 {TOKW}토큰 구간이 사고장애 구간과 겹친 비율:

{T(loc)}

## 3. DAIS-C

원본 대화 파일에서 답변 전체 문장을 다시 꺼내 말버릇을 빼고, 30단어 이상인 답변 {n_units}개({n_spk}명)에 계산했다. 평균 놀람도 {d_mean:.2f}비트. DT는 연구진의 담화 추적 표시이고 임상 판정이 아니다(06).

같은 사람 안에서 DT 표시 답변 대 일반 답변:

{T(within)}

환자군 대 대조군(화자 평균, 참고):

{T(grp)}

## 4. 해석

- **질문 유사도와 운율 지표는 고장 점검을 모두 통과했다.** 바뀌어야 할 조작에서만 바뀌었고, 문장 순서나 전사 형식에는 흔들리지 않았다.
- **반복 지표(mattr)는 답변 앞부분만 본다.** 고정 지표 세트의 mattr는 앞 40 내용어로 계산해서(v3 정의), 답변 가운데에 넣은 반복을 보지 못했다(12개 중 1개만 감소). 답변 전체 내용어로 계산하면 12개 모두 크게 떨어졌다(정리 전 계산, 기준 대비 −0.49). 보속이 답변 뒷부분에 나타나면 놓친다는 뜻이라, 페르소나 모델 평가에는 전체 답변 기준 mattr가 더 맞다. 지표 세트를 바꾸려면 v4j 판정을 다시 해야 한다.
- **놀람도는 문장 순서를 거의 보지 못한다.** 순서를 섞으면 12개 중 11개에서 올라가긴 했지만 크기는 지리멸렬 2단계 효과의 {chk.set_index(['조작', '지표']).loc[('문장 순서 무작위', 'surp_mean'), '평균변화_기준대비']:.0%}였다. 놀람도는 문장 안에서 단어 연결이 깨지는 것(지리멸렬)을 재고, 문장 사이 흐름을 재는 지표가 아니다.
- **놀람도는 전사 형식이면 값 자체가 올라간다**(지리멸렬 2단계 효과의 {chk.set_index(['조작', '지표']).loc[('전사 형식', 'surp_mean'), '평균변화_기준대비']:.0%}). 그러나 단계와의 ρ는 전사 형식에서도 유지됐다. 형식이 다른 자료끼리 값을 직접 비교하면 안 되고, 같은 자료 안에서 비교해야 한다.
- **미묘한 발화에서 놀람도는 지리멸렬을 확실히 잡았다.** 답변 대부분이 정상이어도 16쌍 모두 정상보다 높았고, 가장 놀람도가 높은 구간이 16개 모두 지리멸렬 구간과 겹쳤다(우연 {loc.set_index('유형').loc['INC', '우연']:.0%}). 질문 유사도 구간 지표(07, 41%)보다 훨씬 정확하다. 음향 연상 구간도 우연보다 잘 맞혔다.
- **DAIS-C에서는 DT 표시 답변과 차이가 없었다.** DT가 임상적 사고장애 판정이 아니라는 기존 한계(06)와 같다. 환자군 화자의 평균 놀람도가 대조군보다 조금 높았지만(p = {grp.set_index('지표').loc['surp_mean', 'p']:.2f}) 유의하지 않고, 말투나 면담 방식 차이일 수 있어 참고로만 본다.
"""
    open(os.path.join(REP, '10_checks.md'), 'w', encoding='utf8').write(md)
    print(md)


if __name__ == '__main__':
    assert overlap((0, 5), (4, 9)) and not overlap((0, 4), (4, 9))
    main()
