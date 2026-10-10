"""Step 1: speaker-only tagged 파일 -> 참여자 턴 단위 corpus.parquet (+ reports/01_preprocess.md).

clean()/segment()는 build_qa_units.py에서 재사용한다. (과거 합성 증강 코드도 사용 — archive/augmentation/)
"""
import glob
import os
import re

import pandas as pd
import spacy
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = yaml.safe_load(open(os.path.join(ROOT, 'config.yaml'), encoding='utf8'))
P = CFG['paths']
_nlp = None


def nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(CFG['preprocess']['spacy_model'])
    return _nlp


def clean(text):
    # 빈 태그(<Lh> </Lh>, <InAu> </InAu>, <An> </An>, <Noi=...> </Noi>)와 범위 태그(<Gr> 등)의 태그만 제거.
    # 닫는 '>'가 빠진 깨진 태그(<Noi=... </Noi)도 다음 '<' 전까지 제거된다.
    return re.sub(r'\s+', ' ', re.sub(r'<[^<>]*>?', ' ', text)).strip()


def n_tokens(span):
    return sum(1 for t in span if not (t.is_space or t.is_punct))


def segment(doc):
    """spaCy 문장 분할 후 min_sentence_tokens 미만 문장은 앞 문장(없으면 뒤 문장)에 병합."""
    k = CFG['preprocess']['min_sentence_tokens']
    sents = [s.text.strip() for s in doc.sents if n_tokens(s) > 0]
    toks = [n_tokens(s) for s in doc.sents if n_tokens(s) > 0]
    out, out_t = [], []
    for s, t in zip(sents, toks):
        if out and (t < k or out_t[-1] < k):  # 현재가 짧거나, 직전이 (맨 앞이라) 짧게 남아 있으면 병합
            out[-1] += ' ' + s
            out_t[-1] += t
        else:
            out.append(s)
            out_t.append(t)
    return out


def process(texts):
    """clean된 텍스트 리스트 -> (문장 리스트, 토큰 수) 리스트."""
    return [(segment(d), n_tokens(d)) for d in nlp().pipe(texts, batch_size=64)]


def load_turns():
    rows = []
    for g in ['CL', 'CO']:
        for p in sorted(glob.glob(os.path.join(ROOT, P['raw_root'], P['speaker_glob'].format(group=g)))):
            sid = os.path.basename(p)[:6]
            raw = open(p, encoding='utf8', errors='replace').read()
            for i, t in enumerate(re.findall(rf'<{sid}>(.*?)</{sid}>', raw, re.S)):
                rows.append(dict(speaker_id=sid, group=g, turn_idx=i, text=clean(t)))
    df = pd.DataFrame(rows)
    return df[df.text != ''].reset_index(drop=True)


def main():
    df = load_turns()
    res = process(df.text.tolist())
    df['sentences'] = [r[0] for r in res]
    df['n_sentences'] = df.sentences.str.len()
    df['n_tokens'] = [r[1] for r in res]
    df['substantive'] = df.n_tokens >= CFG['preprocess']['min_response_tokens']
    out = os.path.join(ROOT, P['corpus'])
    df.to_parquet(out, index=False)

    spk = df.groupby(['group', 'speaker_id']).agg(
        turns=('text', 'size'), sentences=('n_sentences', 'sum'), tokens=('n_tokens', 'sum'),
        substantive_turns=('substantive', 'sum'),
        sent_len_mean=('n_tokens', lambda s: s.sum() / df.loc[s.index, 'n_sentences'].sum()),
    ).round(1)
    sub = df[df.substantive]
    grp = sub.groupby('group').agg(responses=('text', 'size'), tok_median=('n_tokens', 'median'),
                                   tok_mean=('n_tokens', 'mean'), sent_median=('n_sentences', 'median')).round(1)
    allsl = [len(s.split()) for ss in df.sentences for s in ss]
    md = f"""# 01. 전처리 (Step 1)

재현: `python src/preprocess.py` → `{P['corpus']}`

## 처리 내용
- 원천: tagged speaker-only 파일 (근거는 00_exploration.md §3). 면담자·가족·임상의 발화는 원천에 없음.
- 태그 제거: 빈 태그(`<InAu>`, `<An>`, `<Lh>`, `<Noi=…>` 등)는 통째로, 범위 태그(`<Gr>`, `<WS>`, `<TC>`, `<DT>`, `<FL>`)는 태그만 지우고 내용 유지. 깨진 태그 1건(`<Noi=… </Noi`)도 처리.
- spaCy `{CFG['preprocess']['spacy_model']}` 문장 분할 → {CFG['preprocess']['min_sentence_tokens']}토큰 미만 문장은 직전 문장(첫 문장이면 다음 문장)에 병합.
- 토큰 = spaCy 토큰 중 공백·구두점 제외 (`I'm` → 2토큰).
- 저장 단위: **참여자 턴 1행** (speaker_id, group, turn_idx, text, sentences, n_sentences, n_tokens, substantive). 화자 단위 텍스트는 speaker_id로 묶으면 복원됨. `substantive` = {CFG['preprocess']['min_response_tokens']}토큰 이상 (비교·생성 단위 "응답").

## 전체
- 턴 {len(df)}개 (빈 턴 제거 후), 화자 {df.speaker_id.nunique()}명 (CL {df[df.group=='CL'].speaker_id.nunique()}, CO {df[df.group=='CO'].speaker_id.nunique()}), 토큰 {df.n_tokens.sum():,}, 문장 {df.n_sentences.sum():,}
- 문장 길이(공백 단어) 중앙값 {pd.Series(allsl).median():.0f}, 90% {pd.Series(allsl).quantile(.9):.0f}, 최대 {max(allsl)} — 구두점이 없어 parser 추정 경계임. 긴 꼬리는 경계를 못 찾은 연속 발화.

## 실질 응답(≥{CFG['preprocess']['min_response_tokens']}토큰) 그룹 요약
{grp.to_markdown()}

## 화자별 통계
{spk.to_markdown()}
"""
    open(os.path.join(ROOT, P['reports'], '01_preprocess.md'), 'w', encoding='utf8').write(md)
    print(md)


if __name__ == '__main__':
    # 최소 자기검사: 태그 제거와 병합 규칙
    assert clean('a <Lh> </Lh> b <Gr> c d </Gr> <Noi=x y? </Noi> e') == 'a b c d e'
    segs = segment(nlp()('yeah. I think that is right. ok. so we went there and it was fine. no.'))
    assert all(len(s.replace('.', ' ').split()) >= 3 for s in segs), segs
    assert segs[0].startswith('yeah') and segs[-1].endswith('no.'), segs
    main()
