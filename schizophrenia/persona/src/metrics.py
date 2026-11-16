"""고정 지표 세트. 페르소나 모델 평가는 이 파일로만 한다.

Scorer()(text, question) -> 유형별 담당 지표. Cached()는 계산 결과를 data/scores_cache.json에 저장해 다시 쓴다.
flags(f, normal) -> 같은 자료의 정상 답변 기준 z 점수(양수 = 그 유형 쪽).
계산은 검증(pre-persona 브랜치 before_persona/reports/v4/11_metric_set.md) 때와 같다.
"""
import hashlib
import json
import os
import re
import subprocess
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SBERT = 'all-MiniLM-L6-v2'
HEAD, MATTR_WIN = 40, 20  # 반복·운율은 앞 내용어 40개, MATTR 창 20
ELV_WIN, ELV_CUT = 8, 100  # 질문 유사도: 앞 100단어를 8단어 창으로
FILLERS = {'er', 'erm', 'mm', 'mhm', 'ah', 'oh', 'um', 'uh', 'hmm'}
DIRECT = re.compile(r'^(?:schizophren\w*|schizo\w*|psychosis|psychotic|antipsychotic\w*|neuroleptic\w*|haloperidol|clozapine|'
                    r'olanzapine|risperidone|quetiapine|aripiprazole|diagnos\w*|hallucinat\w*|delusion\w*)$')

# 유형: (담당 지표, 사고장애일수록 +1 높음 / -1 낮음)
PROFILE = {'DER': ('elv_mean', -1), 'TAN': ('elv_mean', -1), 'INC': ('surp_mean', 1), 'PER': ('mattr', -1),
           'CLANG': ('clang_rate', 1), 'POV': ('n_words', -1), 'POVC': ('info_rate', -1)}


def stopwords():
    from nltk.corpus import stopwords as sw
    try:
        return set(sw.words('english'))
    except LookupError:
        import nltk
        nltk.download('stopwords', quiet=True)
        return set(sw.words('english'))


STOP = stopwords()


def words(text):
    """소문자, 영숫자 외 제거(i'm -> im), 공백 분리."""
    return re.sub(r"[^a-z0-9\s]", "", str(text).lower()).split()


def content(text):
    """불용어, 더듬기, 진단 단어를 뺀 내용어."""
    return [w for w in words(text) if w not in STOP and w not in FILLERS and not DIRECT.match(w)]


def mattr(ws, w=MATTR_WIN):
    if len(ws) <= w:
        return len(set(ws)) / max(len(ws), 1)
    return float(np.mean([len(set(ws[i:i + w])) / w for i in range(len(ws) - w + 1)]))


def clang(ws):
    """이웃한 두 단어(3글자 이상, 서로 다름)의 끝 세 글자가 같은 비율."""
    pairs = [(a, b) for a, b in zip(ws, ws[1:]) if len(a) >= 3 and len(b) >= 3 and a != b]
    return float(np.mean([a[-3:] == b[-3:] for a, b in pairs])) if pairs else 0.0


def elv_mean(m, text, question):
    """Elvevåg 2007: 8단어 창과 질문의 SBERT 유사도 평균."""
    tok = words(' '.join(text.split()[:ELV_CUT]))
    wins = [' '.join(tok[i:i + ELV_WIN]) for i in range(len(tok) - ELV_WIN + 1)]
    q = m.encode(question, normalize_embeddings=True)
    return float((m.encode(wins, normalize_embeddings=True, batch_size=256) @ q).mean())


class LM:
    """GPT-2 small 놀람도."""
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained('gpt2')
        self.lm = AutoModelForCausalLM.from_pretrained('gpt2').eval()

    def tokens(self, text):
        """-> 토큰별 놀람도(비트), 토큰 문자 위치."""
        import torch
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
        self.m, self.lm = SentenceTransformer(SBERT), LM()

    def __call__(self, text, question):
        w, tok = content(text), words(text)
        head = tok[:100]
        elv = elv_mean(self.m, text, question) if len(tok) >= 20 else np.nan  # 20단어 미만은 창이 너무 적다
        return dict(elv_mean=elv, **self.lm(text), mattr=mattr(w[:HEAD]), clang_rate=clang(w[:HEAD]),
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
    """계산 결과를 저장해 다시 쓴다. 95°C 이상이면 85°C까지 쉰다."""
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
                self.save()
            self.c[k] = {a: (None if b != b else float(b)) for a, b in self.sc(text, question).items()}
        return {a: (np.nan if b is None else b) for a, b in self.c[k].items()}

    def save(self):
        json.dump(self.c, open(self.p, 'w', encoding='utf-8'))


def flags(f, normal):
    """f: 지표 표, normal: 정상 답변 표시(bool). -> 유형별 z(양수 = 그 유형 쪽)."""
    out = {}
    for t, (c, s) in PROFILE.items():
        mu, sd = f.loc[normal, c].mean(), f.loc[normal, c].std()
        out[t] = s * (f[c] - mu) / sd
    return pd.DataFrame(out)


if __name__ == '__main__':
    assert words("I'm  OK, really!") == ['im', 'ok', 'really']
    assert content("My son was diagnosed with schizophrenia erm last year") == ['son', 'last', 'year']
    assert clang('cat hat sat mat'.split()) == 0.0 and clang('fight light night'.split()) == 1.0
    f = pd.DataFrame(dict(elv_mean=[.5, .6, .1], surp_mean=[4, 5, 9], mattr=[.9, .85, .4], clang_rate=[0, .02, .3],
                          n_words=[150, 160, 20], info_rate=[.4, .5, .2]))
    z = flags(f, pd.Series([True, True, False]))
    assert (z.iloc[2] > 0).all() and abs(z.iloc[:2].values.mean()) < 1e-9
    print('ok')
