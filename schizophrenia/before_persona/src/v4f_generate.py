"""v4f: 미묘한 국소 사고장애 발화 생성 (GPT-4.1) -> data/v4f/gen_local.json. 계획: reports/v4/00_plan.md 'v4f 계획'.

국소 조건은 답변 대부분이 정상이고 한두 문장만 사고장애이며, 그 구간을 [[ ]]로 표시한다.
"""
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from v4b_generate import DEF, PERSONA, QUESTIONS, SYSTEM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'v4f')
LOCAL = ('Answer the question normally and coherently for most of the answer. In exactly one place, for one or two '
         'sentences only (about 15 to 30 words), show mild {d}. It should be subtle, the kind of slip a listener might '
         'notice but that does not ruin the whole answer. Put exactly that part inside double square brackets [[ like this ]]. '
         'Everything outside the brackets must be normal, on-topic speech.')
CONDS = {
    'N0': 'Answer the question normally and coherently. No thought disorder. Do not use any brackets.',
    'CIRC': ('Show mild circumstantiality: add some unnecessary, tedious detail before getting to the point, but clearly '
             'answer the question by the end. No other thought disorder. Do not use any brackets.'),
    **{t: LOCAL.format(d=d) for t, d in DEF.items()},
}


def main():
    load_dotenv(os.path.join(ROOT, '..', '.env'))
    client = OpenAI()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'gen_local.json')
    data = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {'questions': QUESTIONS, 'items': []}
    have = {(i['q'], i['cond'], i['rep']) for i in data['items']}
    for rep in range(2):
        for q, qt in QUESTIONS.items():
            for c, instr in CONDS.items():
                if (q, c, rep) in have:
                    continue
                r = client.chat.completions.create(model='gpt-4.1', temperature=0.9, seed=rep, messages=[
                    {'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': f'Speaker: {PERSONA}.\nInterviewer question: {qt}\nInstruction: {instr}'}])
                text = r.choices[0].message.content.strip()
                data['items'].append({'q': q, 'cond': c, 'rep': rep, 'text': text})
                print(q, c, rep, len(text.split()), '[[' in text, flush=True)
                json.dump(data, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('items', len(data['items']))


if __name__ == '__main__':
    assert len(CONDS) == 7
    main()
