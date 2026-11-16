"""v4b: GPT-4.1이 단계별 사고장애 발화를 새로 작성 -> data/v4b/gen_gpt.json, 평가용 블라인드 파일 blind.jsonl.

계획: reports/v4/00_plan.md의 'v4b 검증 계획'. 이미 만든 발화는 다시 부르지 않는다.
"""
import json
import os
import random

from dotenv import load_dotenv
from openai import OpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'v4b')
QUESTIONS = {
    'Q11': 'How do you usually spend your mornings?',
    'Q12': 'What was your working life like?',
    'Q13': 'Tell me about your grandchildren.',
    'Q14': 'What do you grow in your garden?',
    'Q15': 'What was the last holiday you went on?',
    'Q16': 'Who do you talk to most during the week?',
    'Q17': 'What did you have for dinner yesterday?',
    'Q18': 'What would you change about where you live?',
}
PERSONA = ('a 52-year-old woman, a retired nurse who worked on hospital wards for 25 years, widowed, lives in a '
           'semi-detached house in a Midlands town, has two grown-up children and three grandchildren, and loves gardening')
LEVEL = {1: 'mild: it happens occasionally, and the answer is still mostly understandable',
         2: 'moderate: it happens frequently, and understanding is clearly impaired',
         3: 'severe: it dominates most of the answer, and it is hard to follow'}
DEF = {
    'DER': 'derailment: ideas slip off track onto obliquely related or unrelated ideas within the answer',
    'TAN': 'tangentiality: the reply is oblique or irrelevant to the question, not addressing it from the start or soon leaving it',
    'INC': 'incoherence (word salad): words and phrases are joined without meaningful grammatical or semantic connection within sentences',
    'PER': 'perseveration: persistent repetition of the same words, phrases or ideas',
    'CLANG': 'clanging: word choice is driven by sound (rhyme, alliteration) rather than meaning',
}
SPECIAL = {
    'N0': 'Answer the question normally and coherently. No thought disorder.',
    'NMULTI': 'Answer normally and coherently, but cover several different relevant sub-topics one after another, with clear transitions. No thought disorder.',
    'CIRC1': ('Show mild circumstantiality: add some unnecessary, tedious detail before getting to the point, '
              'but clearly answer the question by the end. No other thought disorder.'),
    'CIRC2': ('Show marked circumstantiality: go through long chains of irrelevant detail for most of the answer, '
              'but finally return to the question and answer it at the end. No other thought disorder.'),
}
SYSTEM = ('You write realistic transcribed spoken answers for a research study that tests automatic speech measures '
          'against the definitions of the Scale for the Assessment of Thought, Language and Communication (TLC). '
          'Write plain spoken British English as the speaker would say it in an interview, first person, with no '
          'stage directions, no quotation marks, no labels, and no psychiatric or diagnostic words. '
          'Length: 150 to 170 words. Output only the answer text.')


def conditions():
    out = dict(SPECIAL)
    for t, d in DEF.items():
        for lv, ld in LEVEL.items():
            out[f'{t}{lv}'] = f'Show {d}. Severity {lv} ({ld}). Apart from this, keep the speaker and facts plausible.'
    return out


def main():
    load_dotenv(os.path.join(ROOT, '..', '.env'))
    client = OpenAI()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'gen_gpt.json')
    data = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {'questions': QUESTIONS, 'items': []}
    have = {(i['q'], i['cond']) for i in data['items']}
    for q, qt in QUESTIONS.items():
        for c, instr in conditions().items():
            if (q, c) in have:
                continue
            r = client.chat.completions.create(model='gpt-4.1', temperature=0.9, messages=[
                {'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': f'Speaker: {PERSONA}.\nInterviewer question: {qt}\nInstruction: {instr}'}])
            data['items'].append({'q': q, 'cond': c, 'text': r.choices[0].message.content.strip()})
            print(q, c, len(data['items'][-1]['text'].split()), flush=True)
            json.dump(data, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    # 평가용: 조건을 지우고 섞은 블라인드 파일. 대응표는 따로 둔다.
    items = data['items'][:]
    random.Random(7).shuffle(items)
    with open(os.path.join(OUT, 'blind.jsonl'), 'w', encoding='utf-8') as f:
        for k, it in enumerate(items):
            f.write(json.dumps({'id': f'B{k:03d}', 'question': QUESTIONS[it['q']], 'answer': it['text']}, ensure_ascii=False) + '\n')
    json.dump({f'B{k:03d}': f"{it['q']}_{it['cond']}" for k, it in enumerate(items)},
              open(os.path.join(OUT, 'blind_key.json'), 'w'), indent=1)
    print('items', len(items))


if __name__ == '__main__':
    assert len(conditions()) == 19
    main()
