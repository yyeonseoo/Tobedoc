"""Step A (2차): 실제 화자별 비유창성 프로파일 -> data/processed/speaker_profiles.parquet (+ reports/04a_speaker_profiles.md)

단위: 실질 응답(>=30토큰) 턴. 토큰 = 소문자 공백 단어 (전사 규약 그대로; validate.py의 spaCy 토큰과는 별개 정의).
response_profile()은 post_filter.py에서 합성 샘플에도 같은 방식으로 쓴다.
"""
import os

import numpy as np
import pandas as pd

from preprocess import CFG, ROOT

FILLERS_OTHER = ['you know', 'i mean', 'like']   # 다의어 'like'도 전부 센다 (실제·합성 동일 정의)
NON_REP = {'er', 'erm', 'mm', 'mhm', 'ah', 'oh', 'um', 'uh'}   # 채움말 연속은 반복으로 세지 않음
# 1-2글자 정상어 화이트리스트 (corpus 실측 빈도로 구성). 그 밖의 1-2글자 알파벳 토큰 = 절단어(자기수정·미완성 흔적)
SHORT_OK = set('i a it of to er so in is my or do me on if be as at no mm up we oh go an he ah ok by am us em tv '
               'eh ay da y ye ya uh um hm ow aw'.split())


def mattr(words, w=CFG['validate']['mattr_window']):
    if len(words) < w:
        return len(set(words)) / max(len(words), 1)
    return float(np.mean([len(set(words[i:i + w])) / w for i in range(len(words) - w + 1)]))


def response_profile(text, sentences):
    w = text.lower().split()
    n = max(len(w), 1)
    low = ' ' + ' '.join(w) + ' '
    rep = sum(1 for i in range(1, len(w)) if w[i] not in NON_REP and (w[i] == w[i - 1] or (i >= 2 and w[i] == w[i - 2])))
    sl = [len(s.split()) for s in sentences]
    return dict(n_words=len(w), er_per100=100 * w.count('er') / n, erm_per100=100 * w.count('erm') / n,
                other_filler_per100=100 * sum(low.count(f' {f} ') for f in FILLERS_OTHER) / n,
                filler_per100=100 * (w.count('er') + w.count('erm') + sum(low.count(f' {f} ') for f in FILLERS_OTHER)) / n,
                rep_per100=100 * rep / n,
                frag_per100=100 * sum(1 for x in w if x.isalpha() and len(x) <= 2 and x not in SHORT_OK) / n,
                sent_len_mean=float(np.mean(sl)), sent_len_sd=float(np.std(sl)) if len(sl) > 1 else 0.0,
                mattr=mattr(w))


def real_responses():
    c = pd.read_parquet(os.path.join(ROOT, CFG['paths']['corpus']))
    c = c[c.substantive].reset_index(drop=True)
    prof = pd.DataFrame([response_profile(t, s) for t, s in zip(c.text, c.sentences)])
    return pd.concat([c[['speaker_id', 'group', 'turn_idx']], prof], axis=1)


METRICS = ['er_per100', 'erm_per100', 'other_filler_per100', 'filler_per100', 'rep_per100', 'frag_per100',
           'sent_len_mean', 'sent_len_sd', 'mattr']


def main():
    r = real_responses()
    c = pd.read_parquet(os.path.join(ROOT, CFG['paths']['corpus']))
    c = c[c.substantive].reset_index(drop=True)
    rows = []
    for (g, s), d in r.groupby(['group', 'speaker_id']):
        # 비율 지표: 응답 단위 값을 단어 수로 가중 평균. 문장 길이: 화자의 모든 실질 응답 문장을 모아 평균·SD.
        sl = [len(x.split()) for ss in c.loc[d.index, 'sentences'] for x in ss]
        rows.append(dict(group=g, speaker_id=s, n_responses=len(d), n_words=int(d.n_words.sum()),
                         **{m: float(np.average(d[m], weights=d.n_words)) for m in METRICS[:6]},
                         sent_len_mean=float(np.mean(sl)), sent_len_sd=float(np.std(sl)), mattr=float(d.mattr.mean())))
    prof = pd.DataFrame(rows)
    prof.to_parquet(os.path.join(ROOT, 'data/processed/speaker_profiles.parquet'), index=False)
    r.to_parquet(os.path.join(ROOT, 'data/processed/response_profiles.parquet'), index=False)

    resp = r.groupby('group')[METRICS + ['n_words']].quantile([.05, .5, .95]).unstack().T.round(2)
    md = f"""# 04a. 화자별 비유창성 프로파일 (2차 Step A)

재현: `python src/profile_speakers.py` → `data/processed/speaker_profiles.parquet`(화자), `response_profiles.parquet`(응답)

정의 (단위: 실질 응답 ≥30토큰, 토큰 = 소문자 공백 단어):
- `er_per100`, `erm_per100`: 100단어당 빈도.
- `other_filler_per100`: you know / i mean / like (다의어 like 포함).
- `filler_per100`: 위 셋의 합.
- `rep_per100`: 즉시 반복. 같은 단어가 직전 또는 2칸 전에 다시 나오는 경우를 센다 (채움말 제외).
- `frag_per100`: 화이트리스트에 없는 1–2글자 토큰. 전사 규약상 절단어(w, th, f 등)로, 자기수정·미완성 흔적이다.
- `sent_len_mean`, `sent_len_sd`: spaCy 분절 문장의 길이 (단어).
- `mattr`: 응답별 MATTR(창 50)의 평균.
- 화자 값은 응답들을 단어 수로 가중 평균한 것이다 (문장 길이와 MATTR는 제외).

## 그룹별 화자 분포 (화자 단위, CL {int((prof.group == 'CL').sum())}명 / CO {int((prof.group == 'CO').sum())}명)
{prof.groupby('group')[METRICS].agg(['mean', 'std', 'min', 'max']).T.round(2).to_markdown()}

## 그룹별 응답 단위 분포 (5 / 50 / 95 백분위) — 사후 필터 범위의 근거
{resp.to_markdown()}

## 화자별 프로파일
{prof.round(2).to_markdown(index=False)}
"""
    open(os.path.join(ROOT, CFG['paths']['reports'], '04a_speaker_profiles.md'), 'w', encoding='utf8').write(md)
    print(md[:3000])


if __name__ == '__main__':
    p = response_profile('i i went to the the w shop er erm you know like', ['i i went to the the w shop', 'er erm you know like'])
    assert p['rep_per100'] == 100 * 2 / 13 and p["frag_per100"] == 100 / 13 and p["er_per100"] == 100 / 13, p
    main()
