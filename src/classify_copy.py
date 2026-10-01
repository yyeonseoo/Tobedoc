"""seed-copy로 탈락한 시도의 공유 6-gram을 실제 코퍼스 화자 수로 분류 -> reports/04c_seed_copy_classes.md (n-gram 원문 미기록)

분류: 상투 = 실제 코퍼스에서 >=3명 화자가 사용 / 중간 = 2명 / 고유 = 시드 화자 1명만 사용.
필터 기준은 바꾸지 않는다 (진단용).
"""
import collections
import json
import os

import pandas as pd

from post_filter import COPY_N, V2, load_log, ngrams, seed_text, words
from preprocess import CFG, ROOT

corpus = pd.read_parquet(os.path.join(ROOT, CFG['paths']['corpus']))
spk_ng = {s: ngrams(words(' '.join(d.text)), COPY_N) for s, d in corpus.groupby('speaker_id')}
n_users = collections.Counter(g for s in spk_ng.values() for g in s)
prompts = {json.loads(l)['sample_id']: json.loads(l) for l in open(os.path.join(V2, 'prompts.jsonl'), encoding='utf8')}

rows, per_sample = [], []
for e in load_log():
    if e['passed'] or not any(r.startswith('seed_copy') for r in e['reasons']):
        continue
    text = open(os.path.join(V2, 'rejected', f"{e['sample_id']}__a{e['attempt']}.txt"), encoding='utf8').read()
    shared = ngrams(words(text), COPY_N) & ngrams(words(seed_text(prompts[e['sample_id']]['prompt'])), COPY_N)
    cls = []
    for g in shared:
        k = n_users[g]
        c = '상투(>=3명)' if k >= 3 else ('중간(2명)' if k == 2 else '고유(시드 화자만)')
        cls.append(c)
        rows.append(dict(sample_id=e['sample_id'], attempt=e['attempt'], group=e['group'], n_speakers=k, cls=c))
    other = [r for r in e['reasons'] if not r.startswith('seed_copy')]
    per_sample.append(dict(sample_id=e['sample_id'], attempt=e['attempt'], group=e['group'], n_shared=len(shared),
                           only_formulaic=all(c == '상투(>=3명)' for c in cls), any_unique=any(c.startswith('고유') for c in cls),
                           seed_copy_only_reason=not other))
df, ps = pd.DataFrame(rows), pd.DataFrame(per_sample)
by_cls = df.cls.value_counts()
tab = pd.DataFrame({'6-gram 수': by_cls, '비율': (by_cls / len(df)).map('{:.1%}'.format)})
grp = df.groupby(['group', 'cls']).size().unstack(fill_value=0)
md = f"""# 04c. seed-copy 탈락 6-gram 분류 (진단용, 필터 기준 변경 없음)

재현: `python src/classify_copy.py`. 대상: seed_copy 사유로 탈락한 시도 {len(ps)}건 (샘플 {ps.sample_id.nunique()}개).
공유 6-gram은 로그(3개만 기록)가 아니라 탈락 원문과 시드로부터 전부 다시 계산했다. 화자 수 = 실제 코퍼스 전체 발화 중 그 6-gram을 쓴 화자 수.
n-gram 원문은 원본 유래라 여기 적지 않는다.

## 6-gram 단위
{tab.to_markdown()}

그룹별:
{grp.to_markdown()}

## 탈락 시도 단위
- 공유 6-gram이 **모두 상투 표현**인 시도: {int(ps.only_formulaic.sum())}/{len(ps)} ({ps.only_formulaic.mean():.1%})
- 화자 고유 구절이 **1개 이상** 포함된 시도: {int(ps.any_unique.sum())}/{len(ps)} ({ps.any_unique.mean():.1%})
- 상투 표현만 공유했고 **다른 탈락 사유도 없는** 시도 (상투 표현 때문에만 탈락): {int((ps.only_formulaic & ps.seed_copy_only_reason).sum())}/{len(ps)}
- 시도당 공유 6-gram 수: 중앙값 {ps.n_shared.median():.0f}, 최대 {ps.n_shared.max()}
"""
open(os.path.join(ROOT, CFG['paths']['reports'], '04c_seed_copy_classes.md'), 'w', encoding='utf8').write(md)
print(md)
