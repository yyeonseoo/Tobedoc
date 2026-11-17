"""TLC 정의, 단계, 채점표와 GPT 채점.

정의와 단계 문구는 이전 단계(v4b, v4j) 생성 때와 같다. 채점표는 v4 채점표에 말의 빈곤을 더한 것.
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = os.path.join(ROOT, '..', '..', '.env')

LEVEL = {1: 'mild: it happens occasionally, and the answer is still mostly understandable',
         2: 'moderate: it happens frequently, and understanding is clearly impaired',
         3: 'severe: it dominates most of the answer, and it is hard to follow'}
DEF = {
    'DER': 'derailment: ideas slip off track onto obliquely related or unrelated ideas within the answer',
    'TAN': 'tangentiality: the reply is oblique or irrelevant to the question, not addressing it from the start or soon leaving it',
    'INC': 'incoherence (word salad): words and phrases are joined without meaningful grammatical or semantic connection within sentences',
    'PER': 'perseveration: persistent repetition of the same words, phrases or ideas',
    'CLANG': 'clanging: word choice is driven by sound (rhyme, alliteration) rather than meaning',
    'POV': ('poverty of speech: restriction in the amount of spontaneous speech, so replies are brief, '
            'unelaborated and add nothing beyond what is strictly asked'),
}



def symptom_lines(profile):
    """{'DER': 2} -> 지시문에 넣을 정의와 단계 줄."""
    return [f'- {DEF[t]}. Severity {lv} ({LEVEL[lv]}).' for t, lv in profile.items()]


ITEMS = ['derailment', 'tangentiality', 'incoherence', 'perseveration', 'clanging', 'circumstantiality', 'poverty_of_speech']
RUBRIC = """You are an experienced psychiatrist rating formal thought disorder in a transcribed interview answer, using the definitions of Andreasen's Scale for the Assessment of Thought, Language and Communication (TLC). Rate only what is in the text. Most answers from ordinary people score 0.

Items:
- derailment: ideas slip off track onto obliquely related or unrelated ideas within the answer.
- tangentiality: the reply to the question is oblique or irrelevant; it does not address the question from the start or soon leaves it.
- incoherence: speech is incomprehensible at the level of the sentence; words or phrases are joined without meaningful grammatical or semantic connection (word salad).
- perseveration: persistent repetition of the same words, phrases or ideas.
- clanging: word choice is driven by sound (rhyme, alliteration) rather than meaning.
- poverty_of_speech: restriction in the amount of spontaneous speech; replies are brief and unelaborated.
- circumstantiality: indirect, delayed reaching of the goal because of excessive, tedious detail, but the goal IS eventually reached.

Scale for each item: 0 = absent, 1 = mild (occasional, speech still mostly understandable), 2 = moderate (frequent, understanding is impaired), 3 = severe (most of the answer, hard to follow).
Also give overall_disorganization (0-3): how disorganized the speech is overall as evidence of formal thought disorder. Circumstantiality alone or normal topic changes should not raise this much.

Return JSON only: {"derailment":n,"tangentiality":n,"incoherence":n,"perseveration":n,"clanging":n,"circumstantiality":n,"poverty_of_speech":n,"overall_disorganization":n}"""

BUDGET = 2.5  # 달러, 한 번 돌릴 때 생성 + 채점 합계
PRICE_IN, PRICE_OUT = 2 / 1e6, 8 / 1e6  # GPT-4.1


def client():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ENV)
    return OpenAI()


class Budget:
    """토큰 사용량으로 비용을 누적해 파일에 적고, 상한을 넘으면 멈춘다."""
    def __init__(self, path):
        self.path = path
        self.spent = json.load(open(path))['spent'] if os.path.exists(path) else 0.0

    def add(self, usage):
        self.spent += usage.prompt_tokens * PRICE_IN + usage.completion_tokens * PRICE_OUT
        json.dump({'spent': self.spent}, open(self.path, 'w'))
        if self.spent > BUDGET:
            sys.exit(f'비용 상한 {BUDGET}달러 도달 ({self.spent:.2f}). 중단.')


def rate(c, question, text, rubric=RUBRIC, budget=None, model='gpt-4.1'):
    r = c.chat.completions.create(
        model=model, temperature=0, response_format={'type': 'json_object'},
        messages=[{'role': 'system', 'content': rubric},
                  {'role': 'user', 'content': f'Interviewer question: {question}\n\nAnswer:\n{text}'}])
    if budget:
        budget.add(r.usage)
    return {k: int(v) for k, v in json.loads(r.choices[0].message.content).items()}


def rate_items(it, out, key, rubric=RUBRIC, budget=None):
    """it(question, text, key 열)를 섞은 순서로 채점해 out(csv)에 쌓는다. 이미 채점한 것은 건너뛴다."""
    done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(columns=[key])
    todo = it[~it[key].isin(done[key])].sample(frac=1, random_state=42)
    rows = []
    if len(todo):
        c = client()
        try:
            for r in todo.itertuples():
                rows.append({key: getattr(r, key), **rate(c, r.question, r.text, rubric, budget)})
        finally:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_csv(out, index=False)
    return done
