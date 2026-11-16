"""페르소나 모델 v1: 인물 + TLC 증상 설정 -> 여러 턴 면담. 계획: reports/00_plan.md.

python src/persona_model.py                 GPT-4.1로 면담 일괄 생성 -> data/gen_gpt.json (이미 만든 면담은 건너뜀)
python src/persona_model.py --claude-tasks  Claude 에이전트가 쓸 지시문 -> data/tasks/
python src/persona_model.py --chat A DER=2  직접 질문하는 대화 모드 (--chat B MIX 처럼 설정 이름도 됨, 빈 줄 = 종료)
"""
import json
import os
import sys

from tlc import DEF, LEVEL, Budget, client

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data')
PERSONAS = {
    'A': 'a 34-year-old man who lives alone in a flat in an English town, works part-time shifts in a warehouse, and likes football and cooking',
    'B': ('a 52-year-old woman, a retired nurse who worked on hospital wards for 25 years, widowed, lives in a semi-detached '
          'house in a Midlands town, has two grown-up children and three grandchildren, and loves gardening'),
    'C': ('a 27-year-old woman who used to work in a call centre but left last year, lives with her mother in a terraced house '
          'in a northern English city, and likes drawing'),
}
PROFILES = {'NORM': {}, 'DER': {'DER': 2}, 'INC': {'INC': 2}, 'POV': {'POV': 2}, 'MIX': {'DER': 1, 'PER': 1, 'POV': 1}}
QUESTIONS = {
    'Q01': 'What did you do last weekend?',
    'Q02': 'Can you tell me a bit about your job?',
    'Q03': "What's your family like?",
    'Q04': "What's a film or TV show you've enjoyed recently?",
    'Q05': 'What was it like where you grew up?',
    'Q06': 'What do you like to cook?',
    'Q07': 'How have you been sleeping lately?',
    'Q08': 'What are your plans for the next few weeks?',
}
REPS = 2


def system_prompt(persona, profile):
    p = [f'You are role-playing {PERSONAS[persona]}, answering questions in a recorded research interview. '
         'Stay the same person with the same life facts for the whole interview. Speak plain spoken British English, '
         'first person, as transcribed speech: no stage directions, no quotation marks, no labels, and no psychiatric or '
         'diagnostic words. Never mention or explain how you speak.']
    if not profile:
        p.append('Answer each question normally and coherently, at a natural length. No thought disorder.')
    else:
        p.append('Your speech shows the following, consistently in every answer for the whole interview '
                 '(defined as in the Scale for the Assessment of Thought, Language and Communication):')
        p += [f'- {DEF[t]}. Severity {lv} ({LEVEL[lv]}).' for t, lv in profile.items()]
        if 'POV' not in profile:
            p.append('Apart from this, answer at a natural spoken length.')
    return '\n'.join(p)


def interview(client, persona, profile, qs=None, budget=None, model='gpt-4.1'):
    """한 턴씩 실제로 대화한다. -> [{'q', 'question', 'text'}]"""
    qs = qs or QUESTIONS
    msgs = [{'role': 'system', 'content': system_prompt(persona, profile)}]
    turns = []
    for q, qt in qs.items():
        turns.append({'q': q, 'question': qt, 'text': reply(client, msgs, qt, budget, model)})
    return turns


def reply(client, msgs, question, budget=None, model='gpt-4.1'):
    """한 턴: 질문을 대화에 붙이고 답을 받아 대화에 붙인다."""
    msgs.append({'role': 'user', 'content': question})
    r = client.chat.completions.create(model=model, temperature=0.9, messages=msgs)
    if budget:
        budget.add(r.usage)
    a = r.choices[0].message.content.strip()
    msgs.append({'role': 'assistant', 'content': a})
    return a


def parse_profile(args):
    """['DER=2', 'PER=1'] 또는 ['MIX'] -> {'DER': 2, 'PER': 1}"""
    if len(args) == 1 and args[0] in PROFILES:
        return PROFILES[args[0]]
    return {k: int(v) for k, v in (a.split('=') for a in args)}


def chat(persona, profile):
    sys.stdout.reconfigure(errors='replace')
    c, msgs = client(), [{'role': 'system', 'content': system_prompt(persona, profile)}]
    budget = Budget(os.path.join(OUT, 'spent_chat.json'))
    print(f'인물 {persona}, 증상 {profile or "없음"}. 영어로 질문하세요. 빈 줄 = 종료.')
    while q := input('\n질문> ').strip():
        print('\n' + reply(c, msgs, q, budget), f'\n(누적 ${budget.spent:.3f})')


def claude_tasks():
    """Claude 에이전트용: 인물마다 지시문 10개(설정 5 × 반복 2)와 질문을 담은 작업 파일."""
    os.makedirs(os.path.join(OUT, 'tasks'), exist_ok=True)
    qs = QUESTIONS
    for p in PERSONAS:
        jobs = [{'persona': p, 'profile': name, 'rep': r, 'system': system_prompt(p, prof)}
                for name, prof in PROFILES.items() for r in range(REPS)]
        json.dump({'questions': qs, 'jobs': jobs}, open(os.path.join(OUT, 'tasks', f'persona_{p}.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)


def main():
    c = client()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'gen_gpt.json')
    data = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {'interviews': []}
    have = {(i['persona'], i['profile'], i['rep']) for i in data['interviews']}
    budget = Budget(os.path.join(OUT, 'spent.json'))
    qs = QUESTIONS
    for p in PERSONAS:
        for name, prof in PROFILES.items():
            for r in range(REPS):
                if (p, name, r) in have:
                    continue
                turns = interview(c, p, prof, qs, budget)
                data['interviews'].append({'persona': p, 'profile': name, 'rep': r, 'turns': turns})
                json.dump(data, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
                print(p, name, r, [len(t['text'].split()) for t in turns], f'${budget.spent:.2f}', flush=True)
    print('interviews', len(data['interviews']))


if __name__ == '__main__':
    s = system_prompt('A', {'DER': 1, 'POV': 1})
    assert 'derailment' in s and 'poverty of speech' in s and 'natural spoken length' not in s
    assert 'No thought disorder' in system_prompt('B', {})
    assert parse_profile(['DER=2', 'PER=1']) == {'DER': 2, 'PER': 1} and parse_profile(['MIX']) == PROFILES['MIX']
    if '--chat' in sys.argv:
        a = sys.argv[sys.argv.index('--chat') + 1:]
        chat(a[0], parse_profile(a[1:]))
    else:
        claude_tasks() if '--claude-tasks' in sys.argv else main()
