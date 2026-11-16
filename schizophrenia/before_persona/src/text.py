"""두 코퍼스 공통 텍스트 가공. Reddit 본문의 기존 가공(소문자, 문장부호 제거, NLTK 불용어 제거)과 같게 맞춘다."""
import os
import re

import yaml
from nltk.corpus import stopwords

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = yaml.safe_load(open(os.path.join(ROOT, 'config.yaml'), encoding='utf8'))
T = CFG['text']
try:
    STOP = set(stopwords.words('english'))
except LookupError:
    import nltk
    nltk.download('stopwords', quiet=True)
    STOP = set(stopwords.words('english'))
FILLERS = set(T['fillers'])
DIRECT = re.compile(r'^(?:' + '|'.join(T['direct_terms']) + r')$')


def path(key):
    return os.path.join(ROOT, CFG['paths'][key])


def words(text):
    """소문자 -> 영숫자 외 제거(아포스트로피 포함, i'm -> im) -> 공백 분리. Reddit 본문의 상태와 같다."""
    return re.sub(r"[^a-z0-9\s]", "", str(text).lower()).split()


def keep(w):
    """공통 지표에 남길 단어: 불용어, 더듬기, 진단 직접 단어가 아닌 것."""
    return w not in STOP and w not in FILLERS and not DIRECT.match(w)


def content(text):
    return [w for w in words(text) if keep(w)]


def chunks(ws, size):
    return [' '.join(ws[i:i + size]) for i in range(0, len(ws) - size + 1, size)]


if __name__ == '__main__':
    assert words("I'm  OK, really!") == ['im', 'ok', 'really']
    assert content("My son was diagnosed with schizophrenia erm last year") == ['son', 'last', 'year']
    assert chunks(list('abcdefghijk'), 5) == ['a b c d e', 'f g h i j']
    print('ok')
