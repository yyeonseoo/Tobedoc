"""고정 지표 세트 (reports/v4/00_plan.md 'v4j 계획' B). 페르소나 모델 평가는 이 파일로만 한다.

Scorer()(text, question) -> 유형별 담당 지표. Cached()는 같은 계산을 data/scores_cache.json에 저장해 재사용한다. flags(f, normal) -> 같은 자료의 정상 답변 기준 z 점수(양수 = 그 유형 쪽).
계산은 검증 때와 같다: elv_mean은 v4_literature, mattr·clang_rate는 v3 정의(앞 40 내용어), 놀람도는 v4h(GPT-2 small).
"""
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import v4_literature as lit
from text import CFG, ROOT, content, words
from v2_features import mattr
from v3_features import clang

HEAD = 40
# 유형: (담당 지표, 사고장애일수록 +1 높음 / −1 낮음)
PROFILE = {'DER': ('elv_mean', -1), 'TAN': ('elv_mean', -1), 'INC': ('surp_mean', 1), 'PER': ('mattr', -1),
           'CLANG': ('clang_rate', 1), 'POV': ('n_words', -1), 'POVC': ('info_rate', -1)}


class LM:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained('gpt2')
        self.lm = AutoModelForCausalLM.from_pretrained('gpt2').eval()

    def tokens(self, text):
        """-> 토큰별 놀람도(비트), 토큰 문자 위치."""
        enc = self.tok(text, return_offsets_mapping=True)
        ids, off = enc.input_ids[:1023], enc.offset_mapping[:1023]
        x = torch.tensor([[self.tok.bos_token_id] + ids])
        with torch.no_grad():
            logp = torch.log_softmax(self.lm(x).logits[0, :-1], -1)
        return (-logp[torch.arange(len(ids)), torch.tensor(ids)].numpy() / np.log(2)), off

    def __call__(self, text):
        s, _ = self.tokens(text)
        return dict(surp_mean=float(s.mean()), surp_p90=float(np.sort(s)[-max(1, len(s) // 10):].mean()))


class Scorer:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.m, self.lm = SentenceTransformer(CFG['features']['sbert_model']), LM()

    def __call__(self, text, question):
        w, tok = content(text), words(text)
        head = tok[:100]
        elv = lit.metrics(self.m, text, question)['elv_mean'] if len(tok) >= 20 else np.nan  # 창이 몇 개 안 나오는 짧은 답(말의 빈곤)은 계산하지 않음
        return dict(elv_mean=elv, **self.lm(text),
                    mattr=mattr(w[:HEAD], CFG['features']['mattr_window']), clang_rate=clang(w[:HEAD]),
                    n_words=len(tok), info_rate=len(set(content(' '.join(head)))) / max(len(head), 1))


def temp():
    """노트북 CPU 온도(°C). 읽지 못하면 0."""
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command', '(Get-CimInstance Win32_PerfFormattedData_Counters_ThermalZoneInformation | '
                              'Measure-Object HighPrecisionTemperature -Maximum).Maximum'], capture_output=True, text=True, timeout=30).stdout
        return float(out) / 10 - 273.15
    except (ValueError, subprocess.SubprocessError):
        return 0


class Cached:
    """점수 캐시(data/scores_cache.json). 다시 돌릴 때 무거운 계산을 건너뛴다. 95°C 이상이면 85°C까지 쉰다."""
    def __init__(self):
        self.p = os.path.join(ROOT, 'data', 'scores_cache.json')
        self.c = json.load(open(self.p, encoding='utf-8')) if os.path.exists(self.p) else {}
        self.sc, self.n = None, 0

    def __call__(self, text, question):
        k = hashlib.md5((question + '\n' + text).encode()).hexdigest()
        if k not in self.c:
            self.sc = self.sc or Scorer()
            self.n += 1
            if self.n % 10 == 0:
                if temp() >= 95:
                    print('95°C 이상, 쉬는 중', flush=True)
                    while temp() > 85:
                        time.sleep(30)
                json.dump(self.c, open(self.p, 'w', encoding='utf-8'))
            self.c[k] = {a: (None if b != b else float(b)) for a, b in self.sc(text, question).items()}
        return {a: (np.nan if b is None else b) for a, b in self.c[k].items()}

    def save(self):
        json.dump(self.c, open(self.p, 'w', encoding='utf-8'))


def flags(f, normal):
    """f: score() 결과 표, normal: 같은 자료의 정상 답변 표시(bool). -> 유형별 z(양수 = 그 유형 쪽)."""
    out = {}
    for t, (c, s) in PROFILE.items():
        mu, sd = f.loc[normal, c].mean(), f.loc[normal, c].std()
        out[t] = s * (f[c] - mu) / sd
    return pd.DataFrame(out)


if __name__ == '__main__':
    f = pd.DataFrame(dict(elv_mean=[.5, .6, .1], surp_mean=[4, 5, 9], mattr=[.9, .85, .4], clang_rate=[0, .02, .3],
                          n_words=[150, 160, 20], info_rate=[.4, .5, .2]))
    z = flags(f, pd.Series([True, True, False]))
    assert (z.iloc[2] > 0).all() and abs(z.iloc[:2].values.mean()) < 1e-9
    print('ok')
