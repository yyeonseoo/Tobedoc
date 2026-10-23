"""Step 0: 원본 구조 탐색 -> reports/00_exploration.md 용 통계 출력 (전처리 전, 원본 그대로)."""
import collections
import glob
import os
import re

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = yaml.safe_load(open(os.path.join(ROOT, 'config.yaml'), encoding='utf8'))
RAW = os.path.join(ROOT, cfg['paths']['raw_root'])


def turns(path):
    sid = os.path.basename(path)[:6]
    t = open(path, encoding='utf8', errors='replace').read()
    return sid, re.findall(rf'<{sid}>(.*?)</{sid}>', t, re.S)


rows, tags = [], collections.Counter()
for g in ['CL', 'CO']:
    for p in sorted(glob.glob(os.path.join(RAW, cfg['paths']['speaker_glob'].format(group=g)))):
        sid, ts = turns(p)
        for x in ts:
            tags.update(re.findall(r'<([A-Za-z]+)', x))
            rows.append(dict(speaker_id=sid, group=g, n_words=len(re.sub(r'<[^>]*>', ' ', x).split())))
df = pd.DataFrame(rows)
spk = df.groupby(['group', 'speaker_id']).agg(turns=('n_words', 'size'), words=('n_words', 'sum'),
                                              median_turn=('n_words', 'median'), max_turn=('n_words', 'max'),
                                              turns_ge30=('n_words', lambda s: (s >= 30).sum()))
print(spk.to_markdown())
print(df.groupby('group').n_words.describe().to_markdown())
print('turn length quantiles', df.n_words.quantile([.25, .5, .75, .9]).to_dict())
print('share of turns < 30 words', (df.n_words < 30).mean())
print('substantive (>=30w) by group', df[df.n_words >= 30].groupby('group').n_words.describe().to_markdown())
print(tags.most_common())
