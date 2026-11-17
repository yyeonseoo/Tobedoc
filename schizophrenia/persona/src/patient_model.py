"""조현병 가상 환자 모델: PSYCHE의 MFC(증상 내용) + TLC 말하기 방식(말의 형식). 계획과 검증: reports/04_patient_model.md.

지시문 순서: 말하기 방식(TLC 정의와 단계)을 맨 앞에, MFC 요약, 최소 규칙. 약 700토큰.
MFC는 PSYCHE 원본 프롬프트로 만든다(src/psyche_tlc.py make_mfc). 인생 이야기와 말투 칸(말의 양, 사고 과정)은 지시문에 넣지 않는다.
04의 2 × 2에서 증상 내용은 넣고, 면담 질문은 병원 질문보다 일상 질문(persona_model.QUESTIONS)을 쓸 때 탈선과 지리멸렬이 잘 나왔다.

python src/patient_model.py --new P4 30 Female           새 환자 MFC 생성 (GPT-4.1) -> data/patients/mfc.json
python src/patient_model.py --chat P4 DER=2 POV=1        직접 질문하는 대화 모드 (빈 줄 = 종료)
"""
import json
import os
import sys

from tlc import Budget, client, symptom_lines

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'patients')
MFC_PATH = os.path.join(OUT, 'mfc.json')


def summary(mfc, symptoms=True):
    """MFC에서 증상 내용과 배경만 추린다. 모델로 요약하지 않고 칸을 그대로 옮긴다."""
    p, b = mfc['MFC-Profile'], mfc['MFC-Behavior']
    i, s, d, m = p['Identifying data'], p['Present illness']['Symptom'], p['Developmental/social history'], p['Current medication']
    lines = [
        f"- {i['Age']}-year-old {i['Sex'].lower()}, {i['Marital status'].lower()}, {i['Occupation'].lower()}.",
        f"- Main problem: {p['Chief complaint']} For {s['Length']}. Worse with: {s['Exacerbating factor']} Started after: {s['Triggering factor']}",
        f"- What you hear and see: {b['Perception']}",
        f"- What you believe: {b['Thought content']}",
        f"- Mood: {b['Mood']} Affect: {b['Affect']}",
        f"- How you see your illness: {b['Insight']}",
        f"- Medication: {m['Medication name']}, {m['Compliance']}",
        f"- Family: {d['Childhood history']['Members of family']} {d['Childhood history']['Home environment']} {p['Family history']['Substance use']}",
        f"- School and work: {d['School history']} {d['Work history']}"]
    if not symptoms:  # 증상 내용(주 증상, 환청, 망상, 병식, 약)을 빼고 생활 배경만
        lines = [l for l in lines if not l.startswith(('- Main problem', '- What you', '- How you see', '- Medication'))]
    return '\n'.join(lines)


def system_prompt(mfc, profile, symptoms=True, max_words=None):
    """profile: {'DER': 2} 같은 TLC 설정. 빈 dict면 말하기 지시 없음. max_words: 답 길이 상한(없으면 자연스러운 길이)."""
    out = ['You are role-playing a psychiatric patient answering questions in a clinical interview.', '', 'HOW YOU SPEAK']
    if profile:
        out += ['Your speech shows the following, consistently in every answer for the whole interview '
                '(defined as in the Scale for the Assessment of Thought, Language and Communication):', *symptom_lines(profile)]
    if 'POV' not in profile:
        out.append(f'Keep each answer under about {max_words} words.' if max_words else
                   'Apart from this, answer at a natural spoken length.' if profile else 'Answer at a natural spoken length.')
    out += ['', 'WHO YOU ARE', summary(mfc, symptoms), '',
            'Speak in the first person as transcribed speech: no stage directions, no labels. Use everyday words, not '
            'psychiatric terms or diagnosis names. Never mention or explain how you speak. Stay the same person for the whole interview.']
    return '\n'.join(out)


def load_mfc():
    return json.load(open(MFC_PATH, encoding='utf-8')) if os.path.exists(MFC_PATH) else {}


def new_patient(pid, age, sex, diagnosis='Schizophrenia'):
    from psyche_tlc import make_mfc
    os.makedirs(OUT, exist_ok=True)
    mfcs = load_mfc()
    if pid not in mfcs:
        mfcs[pid] = make_mfc(client(), Budget(os.path.join(OUT, 'spent.json')), age, sex, diagnosis)
        json.dump(mfcs, open(MFC_PATH, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return mfcs[pid]


def chat(pid, profile):
    from persona_model import reply
    sys.stdout.reconfigure(errors='replace')
    c, msgs = client(), [{'role': 'system', 'content': system_prompt(load_mfc()[pid], profile)}]
    budget = Budget(os.path.join(OUT, 'spent_chat.json'))
    print(f'환자 {pid}, 말하기 {profile or "지시 없음"}. 영어로 질문하세요. 빈 줄 = 종료.')
    while q := input('\n질문> ').strip():
        print('\n' + reply(c, msgs, q, budget), f'\n(누적 ${budget.spent:.3f})')


if __name__ == '__main__':
    m = json.load(open(os.path.join(ROOT, 'data', 'psyche', 'mfc.json'), encoding='utf-8'))['P1']
    s = system_prompt(m, {'DER': 2})
    assert s.index('derailment') < s.index('WHO YOU ARE') and 'Decreased' not in s and 'natural spoken length' in s
    assert 'natural spoken length' not in system_prompt(m, {'POV': 2}) and 'What you believe' not in summary(m, False)
    a = sys.argv[1:]
    if a[:1] == ['--new']:
        print(json.dumps(new_patient(a[1], int(a[2]), a[3])['MFC-Behavior'], ensure_ascii=False, indent=1))
    elif a[:1] == ['--chat']:
        from persona_model import parse_profile
        chat(a[1], parse_profile(a[2:]) if a[2:] else {})
