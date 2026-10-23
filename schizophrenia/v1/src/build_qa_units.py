"""본 연구 전처리: Interactional(FULL) 파일 -> 질문-답변 단위 qa_units.parquet (+ reports/02_qa_units.md).

증강을 하지 않고, 면담자 질문을 기준점으로 참여자 답변을 자른다 (근거: 대화 기록, reports/DATA_AUGMENTATION_FINAL_REPORT.md).
- 면담자(또는 가족·임상의) 턴이 맞장구면 답변이 이어진 것으로 보고 참여자 턴을 합친다. 질문이면 기준점을 갱신한다.
- 태그는 텍스트에서만 지우고(clean) 개수는 답변별로 따로 남긴다.
- 출력은 답변 전문을 보관한다. 조각 분할과 지표는 position_features.py에서 한다.
"""
import glob
import html
import os
import re
import zipfile

import pandas as pd
from striprtf.striprtf import rtf_to_text

from preprocess import CFG, ROOT, clean, n_tokens, nlp, segment

Q = CFG['qa']
RAW = os.path.join(ROOT, CFG['paths']['raw_root'])
BACK = set(Q['backchannel'])
OUT = os.path.join(ROOT, 'data/processed/qa_units.parquet')


def read_full(p):
    if p.endswith('.docx'):
        x = zipfile.ZipFile(p).read('word/document.xml').decode('utf8')
        return html.unescape(re.sub(r'<[^>]+>', '', re.sub(r'</w:p>', '\n', x)))
    return rtf_to_text(open(p, encoding='latin-1').read())


def turns(text, sid):
    """(role, raw) 순서열. role: 'P'(참여자) / 'O'(면담자·가족·임상의)."""
    return [('P' if tag == sid else 'O', body)
            for tag, body in re.findall(rf'<({sid}|INT|FAM|DOC)>(.*?)</\1>', text, re.S)]


def is_prompt(raw):
    return sum(w not in BACK for w in re.findall(r"[a-z']+", clean(raw).lower())) >= Q['prompt_min_words']


def answers(seq):
    """질문 기준 답변 묶기. 반환: dict(anchor, raw_turns, n_turns)."""
    out, anchor, cur = [], None, []
    for role, raw in seq:
        if role == 'P':
            cur.append(raw)
        elif '<Mis>' in raw or is_prompt(raw):  # <Mis> = 면담자 발화 전사 누락(23EB14 전체): 경계는 맞고 기준점 텍스트만 없음
            if cur:
                out.append(dict(anchor=anchor, raw_turns=cur))
            anchor, cur = (None if '<Mis>' in raw else clean(raw)), []
        # 맞장구면 cur 유지 -> 다음 참여자 턴이 같은 답변에 붙는다
    if cur:
        out.append(dict(anchor=anchor, raw_turns=cur))
    return out


def metadata():
    m = pd.read_excel(os.path.join(RAW, Q['metadata'])).dropna(subset=['group'])
    m['speaker_id'] = m['Participant ID'].str.strip('<>')
    return m.set_index('speaker_id')[['sex', 'initial question type', 'interview mode']].rename(
        columns={'initial question type': 'start_question', 'interview mode': 'interview_mode'})


def main():
    rows = []
    for g in ['CL', 'CO']:
        for p in sorted(glob.glob(os.path.join(RAW, Q['full_glob'].format(group=g)))):
            sid = os.path.basename(p)[:6]
            for i, a in enumerate(answers(turns(read_full(p), sid))):
                raw = ' '.join(a['raw_turns'])
                r = dict(speaker_id=sid, group=g, answer_idx=i, anchor=a['anchor'], n_turns=len(a['raw_turns']),
                         turn_texts=[clean(t) for t in a['raw_turns']])
                r.update({f'tag_{t}': len(re.findall(rf'<{t}>', raw)) for t in Q['tags']})
                rows.append(r)
    df = pd.DataFrame(rows)
    # 문장 분할은 턴별로 한 뒤 이어 붙인다 (턴 경계 = 자연 경계)
    flat = [t for ts in df.turn_texts for t in ts if t]
    seg = dict(zip(flat, [(segment(d), n_tokens(d)) for d in nlp().pipe(flat, batch_size=64)]))
    df['sentences'] = [[s for t in ts if t for s in seg[t][0]] for ts in df.turn_texts]
    df['n_tokens'] = [sum(seg[t][1] for t in ts if t) for ts in df.turn_texts]
    df['text'] = [' '.join(ts) for ts in df.turn_texts]
    df['n_sentences'] = df.sentences.str.len()
    df = df[df.n_tokens > 0].drop(columns='turn_texts').join(metadata(), on='speaker_id')
    df.to_parquet(OUT, index=False)
    report(df)


def report(df):
    w = df.text.str.split().str.len()
    a = df[df.anchor.notna()].assign(w=w)
    dist = a.groupby('group').w.describe(percentiles=[.25, .5, .75])[['count', '25%', '50%', '75%', 'max']].round(0)
    spk = a[a.w >= 30].groupby(['group', 'speaker_id']).agg(
        answers=('w', 'size'), words_median=('w', 'median'), merged=('n_turns', lambda s: int((s > 1).sum())))
    md = f"""# 02. 질문-답변 단위 구축 (본 연구 전처리)

재현: `python src/build_qa_units.py` → `data/processed/qa_units.parquet`. 지표 계산은 `03_features.md`.

## 처리 규칙
- 원천: Interactional `*-FULL.docx/rtf`. 참여자 단어 수가 speaker-only 파일과 28명 전원 일치함을 확인.
- 면담자·가족·임상의 턴에서 맞장구어(config `qa.backchannel`)를 뺀 단어가 {Q['prompt_min_words']}개 이상이면 질문 → 기준점(anchor) 갱신.
  미만이면 맞장구 → 앞뒤 참여자 턴을 한 답변으로 합침.
- 태그: 텍스트에서는 제거(SBERT 입력용), 답변별 개수는 `tag_*` 열로 보관. 더듬기(er/erm)는 유지.
  - InAu: 녹음 방식(CL 전화·대면 / CO 화상)과 교락 → 피처 사용 금지.
  - An: 익명화 흔적 → 피처 사용 금지.
  - Gr/WS/TC/DT: 연구자 판단 표시 → 피처가 아니라 자동 지표의 검증 기준.
- 메타데이터 결합: sex, start_question, interview_mode (공변량·교락 보고용).
- 답변은 자르지 않고 전문을 보관한다. 지표는 답변 전체를 10단어 조각으로 나눠 계산한다(03).

## 전체
- 답변 {len(df)}개 (맞장구로 합쳐진 답변 {(df.n_turns > 1).sum()}개), 화자 {df.speaker_id.nunique()}명
- 기준점 없는 답변: {df.anchor.isna().sum()}개 (첫 질문 이전, 또는 면담자 발화가 `<Mis>`로 누락된 23EB14). 질문 기준 지표에서 제외

## 질문 있는 답변의 길이 (단어)
{dist.to_markdown()}

## 화자별 (질문 있음, 30단어 이상)
{spk.to_markdown()}

> 독립 표본은 화자 수다. 단위 수가 늘어도 통계 검정의 n은 화자 수로 보고하고, 분류는 화자 단위 검증(LOSO)으로 한다.
"""
    open(os.path.join(ROOT, CFG['paths']['reports'], '02_qa_units.md'), 'w', encoding='utf8').write(md)
    print(md)


if __name__ == '__main__':
    # 자기검사: 맞장구는 답변을 잇고, 질문은 기준점을 갱신한다
    s = [('O', 'do you use language creatively'), ('P', 'yes I'), ('O', 'mm'), ('P', 'do'),
         ('O', 'can you tell me more'), ('P', 'well')]
    a = answers(s)
    assert [x['raw_turns'] for x in a] == [['yes I', 'do'], ['well']], a
    assert a[1]['anchor'] == 'can you tell me more'
    assert [x['anchor'] for x in answers([('O', '<Mis> </Mis>'), ('P', 'a'), ('O', '<Mis> </Mis>'), ('P', 'b')])] == [None, None]
    assert not is_prompt('<Lh> </Lh> yeah OK right') and is_prompt('and then what happened')
    main()
