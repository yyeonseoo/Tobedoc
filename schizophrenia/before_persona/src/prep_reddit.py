"""Reddit CSV -> 글 단위 (data/processed/reddit_units.parquet).

- 작성자 없음/[deleted] 제외, 두 라벨 모두에 글이 있는 작성자 제외.
- 본문(selftext)은 제작자가 이미 NLTK 불용어를 지운 상태다. 같은 content() 가공을 한 번 더 적용해 진단 직접 단어를 지운다.
- 제목은 원문이다. 기준점(anchor)으로 쓴다.
- 조현병 게시판 글의 작성 주체를 규칙으로 1차 분류한다: self(본인), other(가족·보호자 등), both, unknown.
  주 분석은 other를 제외하고, self만 쓴 분석은 민감도 분석이다.
"""
import os
import re

import pandas as pd

from text import CFG, content, path, words

REL = (r'son|daughter|brother|sister|mom|mum|mother|dad|father|husband|wife|boyfriend|girlfriend|bf|gf|partner|'
       r'friend|fiance|fiancee|uncle|aunt|cousin|grandma|grandmother|grandpa|grandfather|parents?|sibling|child|kid|'
       r'roommate|ex|loved one|family member|stepdad|stepmom|stepfather|stepmother')
COND = r'schizo\w*|psychos[ie]s|psychotic|diagnos\w*|hallucinat\w*|delusion\w*|voices|paranoi\w*|meds|medications?|hospitali\w*|episode'
# 제목(원문): 문장 형태로 찾는다
OTHER_T = re.compile(rf"\b(my|our)\s+(\w+\s+)?({REL})\b.{{0,60}}\b({COND})|\bcare ?giver|\bcaring for\b|\b({REL})\s+(has|have|was|is|got|with)\s+(been\s+)?({COND})", re.I)
SELF_T = re.compile(rf"\bi\s*('m|am|was|have|'ve|had|got)\s+(been\s+)?(just\s+)?(diagnosed|schizo\w*|psychotic)|\bmy\s+(diagnosis|meds|medication|psychiatrist|symptoms|hallucinations|voices|delusions|psychosis|episode)\b|\bi\s+(hear|heard|see|saw)\s+(voices|things)", re.I)
# 본문(불용어 제거됨): 관계어와 증상어가 3단어 안에 붙어 있는지, 본인 표현(im/ive + 진단)이 있는지 본다
OTHER_B = re.compile(rf"\b({REL})\b(\s+\S+){{0,3}}\s+({COND})\b|\bcare ?giver", re.I)
SELF_B = re.compile(r"\b(im|ive)\s+(\S+\s+)?(diagnosed|schizo\w*|psychotic|hearing voices)\b")


def source(title, body):
    o = bool(OTHER_T.search(title) or OTHER_B.search(body))
    s = bool(SELF_T.search(title) or SELF_B.search(body))
    return 'both' if o and s else 'other' if o else 'self' if s else 'unknown'


def main():
    d = pd.read_csv(path('reddit_csv'))
    n0 = len(d)
    d = d[d.author.notna() & ~d.author.isin(CFG['reddit']['exclude_authors'])]
    both = d.groupby('author').is_schizophrenic.nunique().gt(1)
    d = d[~d.author.isin(both[both].index)].copy()
    d['title'] = d.title.fillna('').astype(str)
    d['selftext'] = d.selftext.fillna('').astype(str)
    d['source'] = [source(t, ' '.join(words(b))) if y == 1 else 'control'
                   for t, b, y in zip(d.title, d.selftext, d.is_schizophrenic)]
    out = pd.DataFrame({
        'unit_id': d.id, 'speaker_id': d.author, 'label': d.is_schizophrenic.astype(int),
        'subreddit': d.subreddit, 'source': d.source,
        'anchor': d.title, 'anchor_words': d.title.map(content), 'words': d.selftext.map(content)})
    out['n_words'] = out.words.str.len()
    os.makedirs(path('processed'), exist_ok=True)
    out.to_parquet(os.path.join(path('processed'), 'reddit_units.parquet'), index=False)
    # 팀 확인용 표본: 조현병 게시판 40 내용어 이상 글 100개 (원문 포함이라 data/ 아래에만 저장)
    chk = d.assign(n=out.n_words.values)
    chk = chk[(chk.is_schizophrenic == 1) & (chk.n >= CFG['features']['n_words'])]
    chk.sample(100, random_state=CFG['seed'])[['id', 'source', 'title', 'selftext']].assign(human_label='').to_csv(
        os.path.join(path('processed'), 'reddit_source_check.csv'), index=False, encoding='utf-8-sig')
    print(f'원본 {n0}개 -> 작성자 정리 후 {len(out)}개 (두 라벨 작성자 {int(both.sum())}명 제외)')
    long = out[out.n_words >= CFG['features']['n_words']]
    print(f'{CFG["features"]["n_words"]} 내용어 이상:')
    print(long.groupby('source').agg(posts=('unit_id', 'size'), authors=('speaker_id', 'nunique')))


if __name__ == '__main__':
    assert source('My brother has schizophrenia, advice?', '') == 'other'
    assert source('I was diagnosed last year', '') == 'self'
    assert source('question', 'mom diagnosed paranoid schizophrenia years ago') == 'other'
    assert source('question', 'im diagnosed schizoaffective') == 'self'
    assert source('how do you cope', 'feel tired lately') == 'unknown'
    main()
