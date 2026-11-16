"""v4j: 말 빈곤 발화 생성(GPT-4.1, v4g와 같은 인물·질문) + 블라인드 빈곤 평가 -> data/v4j/. 계획: 00_plan.md 'v4j 계획'.

사용법: python src/v4j_poverty.py   (이미 만든 발화와 평가는 다시 부르지 않는다)
"""
import json
import os

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from v4_judge import load_items, rate_items
from v4b_generate import LEVEL, SYSTEM
from v4g_compare import load_v4g
from v4g_generate import PERSONA, questions

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'v4j')
DEF = {
    'POV': ('poverty of speech: restriction in the amount of spontaneous speech, so replies are brief, '
            'unelaborated and add nothing beyond what is strictly asked'),
    'POVC': ('poverty of content of speech: the reply is long enough but conveys little information, because it is '
             'vague, over-abstract or over-concrete, stereotyped, or repetitive'),
}
RUBRIC = """You are an experienced psychiatrist rating negative formal thought disorder in a transcribed interview answer, using the definitions of Andreasen's Scale for the Assessment of Thought, Language and Communication (TLC). Rate only what is in the text. Most answers from ordinary people score 0.

Items:
- poverty_of_speech: restriction in the amount of spontaneous speech; replies are brief and unelaborated.
- poverty_of_content: the reply is adequate in amount but conveys little information because it is vague, over-abstract or over-concrete, stereotyped or repetitive.

Scale for each item: 0 = absent, 1 = mild, 2 = moderate, 3 = severe.
Return JSON only: {"poverty_of_speech":n,"poverty_of_content":n}"""


def instruction(t, lv):
    s = f'Show {DEF[t]}. Severity {lv} ({LEVEL[lv]}). Apart from this, keep the speaker and facts plausible.'
    return s + (' Ignore the 150 to 170 word length rule: say only as much as this speaker would.' if t == 'POV' else '')


def main():
    load_dotenv(os.path.join(ROOT, '..', '.env'))
    client = OpenAI()
    os.makedirs(OUT, exist_ok=True)
    gp = os.path.join(OUT, 'gen_gpt.json')
    qs = questions()
    data = json.load(open(gp, encoding='utf-8')) if os.path.exists(gp) else {'questions': qs, 'items': []}
    have = {(i['q'], i['cond']) for i in data['items']}
    for q, qt in qs.items():
        for t in DEF:
            for lv in (1, 2, 3):
                if (q, f'{t}{lv}') in have:
                    continue
                r = client.chat.completions.create(model='gpt-4.1', temperature=0.9, messages=[
                    {'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': f'Speaker: {PERSONA}.\nInterviewer question: {qt}\nInstruction: {instruction(t, lv)}'}])
                data['items'].append({'q': q, 'cond': f'{t}{lv}', 'text': r.choices[0].message.content.strip()})
                print(q, t, lv, len(data['items'][-1]['text'].split()), flush=True)
                json.dump(data, open(gp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    new = pd.DataFrame(data['items']).assign(question=lambda x: x.q.map(qs))
    it = pd.concat([load_v4g()[['q', 'cond', 'question', 'text']], new], ignore_index=True)
    it['item_id'] = it.q + '_' + it.cond
    rate_items(it, os.path.join(OUT, 'judge_poverty.csv'), rubric=RUBRIC)   # GPT-4.1 자기 평가(참고)
    # 교차 평가: Claude가 쓴 세트(말 빈곤 36 + v4 114)를 GPT-4.1이 평가. GPT-4.1이 쓴 세트는 Claude 에이전트가 평가(judge_x*.csv)
    d = json.load(open(os.path.join(OUT, 'gen_claude.json'), encoding='utf-8'))
    new = pd.DataFrame(d['items']).assign(question=lambda x: x.q.map(d['questions']))
    it = pd.concat([load_items()[['q', 'cond', 'question', 'text']], new], ignore_index=True)
    rate_items(it.assign(item_id=it.q + '_' + it.cond), os.path.join(OUT, 'judge_x_gpt.csv'), rubric=RUBRIC)


if __name__ == '__main__':
    assert 'Ignore' in instruction('POV', 2) and 'Ignore' not in instruction('POVC', 2)
    main()
