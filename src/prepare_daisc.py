"""DAIS-C Interactional(FULL) 파일 -> 질문-답변 단위 (data/processed/daisc_units.parquet).

질문-답변 묶기 규칙은 v1(build_qa_units.py)과 같다.
- 면담자·가족·임상의 턴이 맞장구면 참여자 턴을 이어 붙이고, 질문이면 기준점을 갱신한다.
- <Mis>(면담자 발화 전사 누락, 23EB14)는 경계로 쓰고 기준점 텍스트는 비운다.
v2에서 추가: 내용어마다 연구자 판단 태그(Gr/WS/TC/DT) 안에 있었는지 표시해 같은 사람 안에서 비교할 수 있게 한다.
"""
import glob
import html
import os
import re
import zipfile

import pandas as pd
from striprtf.striprtf import rtf_to_text

from text import CFG, FILLERS, content, keep, path, words

D = CFG['daisc']
BACK = set(D['backchannel'])
JUDGE = D['judge_tags']
TAG = re.compile(r'(<[^<>]*>?)')  # 깨진 태그(<Noi=... </Noi)도 다음 '<' 전까지 태그로 처리


def strip_tags(s):
    return re.sub(r'\s+', ' ', TAG.sub(' ', s)).strip()


def read_full(p):
    if p.endswith('.docx'):
        x = zipfile.ZipFile(p).read('word/document.xml').decode('utf8')
        return html.unescape(re.sub(r'<[^>]+>', '', re.sub(r'</w:p>', '\n', x)))
    return rtf_to_text(open(p, encoding='latin-1').read())


def turns(text, sid):
    return [('P' if tag == sid else 'O', body)
            for tag, body in re.findall(rf'<({sid}|INT|FAM|DOC)>(.*?)</\1>', text, re.S)]


def is_prompt(raw):
    return sum(w not in BACK for w in words(strip_tags(raw))) >= D['prompt_min_words']


def answers(seq):
    """-> [(기준점 질문, 참여자 턴들, 답변 직후 면담자 질문)]. 직후 질문은 E5(되묻기)에 쓴다."""
    out, anchor, cur = [], None, []
    for role, raw in seq:
        if role == 'P':
            cur.append(raw)
        elif '<Mis>' in raw or is_prompt(raw):
            nxt = None if '<Mis>' in raw else strip_tags(raw)
            if cur:
                out.append((anchor, cur, nxt))
            anchor, cur = nxt, []
    if cur:
        out.append((anchor, cur, None))
    return out


def tagged_content(raw):
    """태그가 남은 원문 -> [(내용어, 그 단어를 감싼 판단 태그 집합)]."""
    active, out = set(), []
    for piece in TAG.split(raw):
        m = re.match(r'<(/?)(\w+)', piece)
        if m:
            if m.group(2) in JUDGE:
                (active.discard if m.group(1) else active.add)(m.group(2))
            continue
        out += [(w, frozenset(active)) for w in words(piece) if keep(w)]
    return out


def metadata():
    m = pd.read_excel(os.path.join(path('daisc_root'), CFG['paths']['daisc_metadata'])).dropna(subset=['group'])
    m['speaker_id'] = m['Participant ID'].str.strip('<>')
    return m.set_index('speaker_id')[['sex', 'initial question type', 'interview mode']].rename(
        columns={'initial question type': 'start_question', 'interview mode': 'interview_mode'})


def main():
    rows = []
    for g in ['CL', 'CO']:
        for p in sorted(glob.glob(os.path.join(path('daisc_root'), CFG['paths']['daisc_full_glob'].format(group=g)))):
            sid = os.path.basename(p)[:6]
            for i, (anchor, raws, nxt) in enumerate(answers(turns(read_full(p), sid))):
                raw = ' '.join(raws)
                tc = tagged_content(raw)
                allw = words(strip_tags(raw))
                rows.append(dict(
                    speaker_id=sid, group=g, unit_id=f'{sid}_{i:03d}', n_turns=len(raws),
                    anchor=anchor, anchor_words=content(anchor) if anchor else [], next_prompt=nxt,
                    words=[w for w, _ in tc], n_raw_words=len(allw),
                    filler_rate=sum(w in FILLERS for w in allw) / max(len(allw), 1),
                    **{f'in_{t}': [t in a for _, a in tc] for t in JUDGE}))
    df = pd.DataFrame(rows)
    df['n_words'] = df.words.str.len()
    df = df[df.n_raw_words > 0].join(metadata(), on='speaker_id')
    os.makedirs(path('processed'), exist_ok=True)
    df.to_parquet(os.path.join(path('processed'), 'daisc_units.parquet'), index=False)
    print(f'답변 {len(df)}개, 화자 {df.speaker_id.nunique()}명, 기준점 없음 {df.anchor.isna().sum()}개')
    print(df.groupby('group').n_words.describe(percentiles=[.5, .9]).round(0))


if __name__ == '__main__':
    assert [w for w, _ in tagged_content('I <DT> went home </DT> <Lh> </Lh> yesterday')] == ['went', 'home', 'yesterday']
    assert [sorted(a) for _, a in tagged_content('a <Gr> big <DT> red </DT> </Gr> car')] == [['Gr'], ['DT', 'Gr'], []]
    assert [x[0] for x in answers([('O', 'do you use language creatively'), ('P', 'yes'), ('O', 'mm'), ('P', 'I do'),
                                  ('O', '<Mis> </Mis>'), ('P', 'well')])] == ['do you use language creatively', None]
    assert [x[2] for x in answers([('O', 'how was it'), ('P', 'fine'), ('O', 'what do you mean by fine'), ('P', 'ok')])] == ['what do you mean by fine', None]
    main()
