"""v4 생성기 검증: 생성 발화를 섞고 조건을 가린 채 GPT에게 TLC 항목별 0~3점을 매기게 한다 -> data/v4/judge.csv.

이미 평가한 발화는 다시 부르지 않는다(캐시). 사용법: python src/v4_judge.py [--model gpt-4.1]
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'v4')
OUT = os.path.join(DATA, 'judge.csv')
ITEMS = ['derailment', 'tangentiality', 'incoherence', 'perseveration', 'clanging', 'circumstantiality']

RUBRIC = """You are an experienced psychiatrist rating formal thought disorder in a transcribed interview answer, using the definitions of Andreasen's Scale for the Assessment of Thought, Language and Communication (TLC). Rate only what is in the text. Most answers from ordinary people score 0.

Items:
- derailment: ideas slip off track onto obliquely related or unrelated ideas within the answer.
- tangentiality: the reply to the question is oblique or irrelevant; it does not address the question from the start or soon leaves it.
- incoherence: speech is incomprehensible at the level of the sentence; words or phrases are joined without meaningful grammatical or semantic connection (word salad).
- perseveration: persistent repetition of the same words, phrases or ideas.
- clanging: word choice is driven by sound (rhyme, alliteration) rather than meaning.
- circumstantiality: indirect, delayed reaching of the goal because of excessive, tedious detail, but the goal IS eventually reached.

Scale for each item: 0 = absent, 1 = mild (occasional, speech still mostly understandable), 2 = moderate (frequent, understanding is impaired), 3 = severe (most of the answer, hard to follow).
Also give overall_disorganization (0-3): how disorganized the speech is overall as evidence of formal thought disorder. Circumstantiality alone or normal topic changes should not raise this much.

Return JSON only: {"derailment":n,"tangentiality":n,"incoherence":n,"perseveration":n,"clanging":n,"circumstantiality":n,"overall_disorganization":n}"""


def load_items():
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, 'gen_*.json'))):
        d = json.load(open(f, encoding='utf-8'))
        for it in d['items']:
            rows.append(dict(item_id=f"{it['q']}_{it['cond']}", q=it['q'], cond=it['cond'],
                             question=d['questions'][it['q']], text=it['text']))
    return pd.DataFrame(rows)


def rate(client, model, question, text, rubric=RUBRIC, budget=None):
    r = client.chat.completions.create(
        model=model, temperature=0, response_format={'type': 'json_object'},
        messages=[{'role': 'system', 'content': rubric},
                  {'role': 'user', 'content': f'Interviewer question: {question}\n\nAnswer:\n{text}'}])
    if budget:
        budget.add(r.usage)
    return {k: int(v) for k, v in json.loads(r.choices[0].message.content).items()}


def rate_items(it, out, key='item_id', rubric=RUBRIC, budget=None):
    """it(question, text, key 열)를 GPT-4.1로 평가해 out(csv)에 캐시. 섞은 순서로 평가하고, 이미 평가한 것은 건너뛴다."""
    done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(columns=[key])
    todo = it[~it[key].isin(done[key])].sample(frac=1, random_state=42)
    rows = []
    if len(todo):
        load_dotenv(os.path.join(ROOT, '..', '.env'))
        client = OpenAI()
        try:
            for r in todo.itertuples():
                rows.append({key: getattr(r, key), **rate(client, 'gpt-4.1', r.question, r.text, rubric, budget)})
        finally:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_csv(out, index=False)
    return done


def main():
    model = sys.argv[sys.argv.index('--model') + 1] if '--model' in sys.argv else 'gpt-4.1'
    load_dotenv(os.path.join(ROOT, '..', '.env'))
    client = OpenAI()
    items = load_items()
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=['item_id', 'model'])
    seen = set(done.loc[done.model == model, 'item_id'])
    # 섞은 순서로 평가 (조건 순서가 평가에 영향을 주지 않도록)
    todo = items[~items.item_id.isin(seen)].sample(frac=1, random_state=42)
    new = []
    for _, it in todo.iterrows():
        new.append(dict(item_id=it.item_id, model=model, **rate(client, model, it.question, it.text)))
        print(it.item_id, new[-1]['overall_disorganization'], flush=True)
    pd.concat([done, pd.DataFrame(new)], ignore_index=True).to_csv(OUT, index=False)
    print(f'{len(new)} rated, total {len(done) + len(new)}')


if __name__ == '__main__':
    main()
