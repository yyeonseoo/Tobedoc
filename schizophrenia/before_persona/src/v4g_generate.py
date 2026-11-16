"""v4g: v4(Claude)와 같은 인물, 같은 질문으로 GPT-4.1이 19개 조건을 다시 작성 -> data/v4g/gen_gpt.json.

계획: reports/v4/00_plan.md 'v4g 계획'. 지시문은 v4b와 같고 인물과 질문만 v4로 바꾼다.
"""
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from v4b_generate import SYSTEM, conditions

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'v4g')
PERSONA = ('a 34-year-old man who lives alone in a flat in an English town, works part-time shifts in a warehouse, '
           'and likes football and cooking')


def questions():
    q = {}
    for f in ['gen_q01_q02.json', 'gen_q03_q04.json', 'gen_q05_q06.json']:
        q.update(json.load(open(os.path.join(ROOT, 'data', 'v4', f), encoding='utf-8'))['questions'])
    return q


def main():
    load_dotenv(os.path.join(ROOT, '..', '.env'))
    client = OpenAI()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'gen_gpt.json')
    qs = questions()
    data = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {'questions': qs, 'items': []}
    have = {(i['q'], i['cond']) for i in data['items']}
    for q, qt in qs.items():
        for c, instr in conditions().items():
            if (q, c) in have:
                continue
            r = client.chat.completions.create(model='gpt-4.1', temperature=0.9, messages=[
                {'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': f'Speaker: {PERSONA}.\nInterviewer question: {qt}\nInstruction: {instr}'}])
            data['items'].append({'q': q, 'cond': c, 'text': r.choices[0].message.content.strip()})
            print(q, c, len(data['items'][-1]['text'].split()), flush=True)
            json.dump(data, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('items', len(data['items']))


if __name__ == '__main__':
    assert len(questions()) == 6
    main()
