"""조현병 가상 환자 모델: PSYCHE의 MFC(증상 내용) + TLC 말하기 방식(말의 형식). 계획과 검증: reports/04_patient_model.md.

지시문 순서: 말하기 방식(TLC 정의와 단계)을 맨 앞에, MFC 요약, 최소 규칙. 약 700토큰.
MFC는 PSYCHE 원본 프롬프트로 만든다(src/psyche_tlc.py make_mfc). 인생 이야기와 말투 칸(말의 양, 사고 과정)은 지시문에 넣지 않는다.
04의 2 × 2에서 증상 내용은 넣고, 면담 질문은 병원 질문보다 일상 질문(persona_model.QUESTIONS)을 쓸 때 탈선과 지리멸렬이 잘 나왔다.

python src/patient_model.py --new P4 30 Female           새 환자 MFC 생성 (GPT-4.1) -> data/patients/mfc.json
python src/patient_model.py --new C05 31 Male "사례 설명"  사례 설명(팀 프로필 등)을 따르는 MFC
python src/patient_model.py --chat P4 DER=2 POV=1        직접 질문하는 대화 모드 (빈 줄 = 종료)
"""
import json
import os
import re
import sys

from tlc import Budget, client, symptom_lines

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'patients')
MFC_PATH = os.path.join(OUT, 'mfc.json')
# 탈선은 TLC 원본 단계(면담 중 횟수)에 맞춘 구체적인 문구를 쓴다. 1, 3단계는 04 7단계에서 세기가 조절됨을 확인
DER_LEVELS = {
    1: ('- derailment: in some answers (not all), your ideas slip off track once onto an obliquely related idea and you '
        'carry on from there; otherwise you are easy to follow.'),
    # 2단계는 TLC 원문 탈선 설명의 특징을 옮긴 문구(04 9단계, GPT 채점자 0.4 -> 0.83). 앞 문구는 psyche_ablation DER_W에 남김
    2: ('- derailment: your ideas slip off the track onto other ideas that are only obliquely related or completely '
        'unrelated. You put things side by side that have no meaningful connection, and you shift from one frame of '
        'reference to another without warning. Your sentences are disjointed: they do not hang together, and you use '
        '"it", "they" or "that" without making clear who or what you mean. With each slip you get farther from the '
        'question, and you do not notice that your answer no longer has anything to do with it. This happens whenever '
        'you talk for more than a sentence or two.'),
    3: ('- derailment: in every answer, your ideas keep slipping off track, five or more times, onto obliquely related '
        'or unrelated ideas; you never get back to the question and the answer as a whole is hard to follow, although '
        'each sentence makes sense on its own.')}
# 말의 빈곤과 보속증도 원본 단계 문구로 (04 8단계에서 확인). 원본: 말의 빈곤은 짧은 답의 비율, 보속증은 반복되는 묶음 수
POV_LEVELS = {
    1: '- poverty of speech: now and then a reply is short and you do not add detail even where it would be natural; most replies are ordinary.',
    2: ('- poverty of speech: many of your replies are only one to a few words ("Yeah." "Not really." "Last week."), and '
        'you rarely add any detail unless asked again.'),
    3: '- poverty of speech: you answer in one or two words at most, and sometimes you give no answer at all ("..." or "Don\'t know.").'}
PER_LEVELS = {
    1: '- perseveration: there is one phrase or idea that you keep coming back to in some of your answers.',
    2: '- perseveration: there are two or three phrases or ideas that you keep coming back to across your answers.',
    3: '- perseveration: there are four or more phrases or ideas that you keep coming back to, in almost every answer.'}
LEVEL_TEXT = {'DER': DER_LEVELS, 'POV': POV_LEVELS, 'PER': PER_LEVELS}
LINE_START = {'DER': '- derailment:', 'POV': '- poverty of speech:', 'PER': '- perseveration:'}
# 병원 질문에서 탈선 0.88 -> 1.38, 지리멸렬 0.96 -> 1.38(04 7단계). 기본으로 넣는다
# 10단계: 말 길이와 구어체. 질문에 맞춰 길이를 바꾸고 글처럼 쓰지 않게 한다
NATURAL_LENGTH = ('Match the length to the question, as real patients do: a greeting or a yes/no or simple factual question gets '
                  'a short reply (a few words or one sentence); an open question gets a few sentences at most. Never give a '
                  'long speech.')
SPOKEN = ('Talk like a real person in a clinic, not like a writer: plain everyday words, short or unfinished sentences, '
          'false starts and fillers now and then ("I dunno", "kind of", "like"). No vivid images, metaphors or literary '
          'descriptions, and do not narrate scenes in detail.')
# 11단계: 와해형은 열린 질문에서 말할 여지를 조금 더 주고, 답마다 정도를 다르게 하고, 보속증은 같은 표현의 반복으로 쓴다
DIS_LENGTH = ('Match the length to the question: a greeting or a yes/no or simple factual question gets one or two sentences; '
              'an open question gets up to about five sentences, enough for your way of speaking to show. Never give a long speech.')
VARY = ('How strongly this shows varies from answer to answer, as in real patients: some answers are fairly clear, others '
        'are badly affected. Do not make every answer the same.')
PER_LEVELS_11 = {
    1: ('- perseveration: one exact phrase (the same words, not just the same topic) keeps coming back in some of your '
        'answers, even where it does not fit.'),
    2: ('- perseveration: two or three exact phrases or words (the same wording, not just the same worry) keep coming back '
        'across your answers, even when they do not fit the question.'),
    3: ('- perseveration: four or more exact phrases or words keep coming back in almost every answer, whatever the '
        'question is.')}
# 12단계: 탈선은 우리가 TLC 특징대로 쓴 짧은 예시를 붙인다(원문 환자 예시는 베낄까 봐 쓰지 않음). 보속증은 스스로 만든 표현
DER_EXAMPLES = ('For example (the kind of thing, not the content; never reuse these words): asked about weekends: "Weekends, '
                'yeah. I go to the shop on the corner, they changed the sign. My uncle had a boat. It doesn\'t sit right, so '
                'they keep moving it. Anyway, the bus." Asked about sleep: "Not bad. The ceiling\'s white, they did it in '
                'spring. That\'s when the letters came. You can\'t keep it all in the drawer, she said, but it was the blue one."')
# 13단계(탈선 다듬기): 05에서 탈선이 "매 답 끝에 엉뚱한 꼬리를 같은 공식으로 붙인다"는 평을 받았다. 예시 두 개가 같은 모양이었다
DER2_13 = ('- derailment: your ideas slip off the track onto other ideas that are only obliquely related or completely '
           'unrelated. You put things side by side that have no meaningful connection, sentences do not hang together, and '
           'you sometimes use "it", "they" or "that" without making clear who or what you mean. You do not notice when your '
           'answer has drifted away from the question. This shows in about half of your longer answers, not in every one.')
DER2_13B = DER2_13.replace('This shows in about half of your longer answers, not in every one.',
                           'This shows in most of your longer answers, though not in every one.')  # 13b: dx2의 빈도만 올림
DER_VARY_13 = ('Do not use one pattern. In particular, do not answer the question and then tack an unrelated remark onto the '
               'end. Vary where and how you slip: sometimes you lose the thread right at the start, sometimes in the middle, '
               'sometimes you never get to the point, sometimes you half come back, and some answers are fairly clear.')
DER_EXAMPLES_13 = ('For example (the kind of thing, not the content; never reuse these words): asked about work: "Work, I was '
                   'on the, they had the forms, the green ones. My cousin does that too, in the summer, with the boats." Asked '
                   'about sleep: "Sleep is okay. Mostly okay. The thing about the neighbours, it is the same as the radio, you '
                   'turn it and it is still there." Asked about family: "My mum calls on Sundays. She is fine, she worries. '
                   'My brother moved, so."')
OWN_WORDS = 'WHO YOU ARE below is background, not lines to recite: say things in your own words and do not repeat its sentences.'
PER_LEVELS_12 = {
    1: ('- perseveration: you have one pet phrase of your own (make it up; do not take it from WHO YOU ARE) that keeps '
        'coming back in some answers, even where it does not fit.'),
    2: ('- perseveration: you have two or three pet phrases of your own (make them up; do not take them from WHO YOU ARE) '
        'that keep coming back across your answers, even when they do not fit the question.'),
    3: ('- perseveration: you have four or more pet phrases of your own (make them up; do not take them from WHO YOU ARE) '
        'that come back in almost every answer, whatever the question is.')}
KEEP_FORM = ('Keep speaking this way even when you are asked about your symptoms, feelings or treatment; do not switch to '
             'a clear, organized account.')


def summary(mfc, symptoms=True, quotes=True):
    """MFC에서 증상 내용과 배경만 추린다. 모델로 요약하지 않고 칸을 그대로 옮긴다.
    quotes=False면 MFC 예시 대사의 따옴표를 뺀다(내용은 남김). 12단계: 모델이 따옴표 대사를 매 답 그대로 베꼈다."""
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
    if not quotes:
        lines = [re.sub(r'["“”]', '', l) for l in lines]  # 따옴표만 빼고 내용은 남긴다
    if not symptoms:  # 증상 내용(주 증상, 환청, 망상, 병식, 약)을 빼고 생활 배경만
        lines = [l for l in lines if not l.startswith(('- Main problem', '- What you', '- How you see', '- Medication'))]
    return '\n'.join(lines)


def system_prompt(mfc, profile, symptoms=True, max_words=None, der_original=True, keep_form=True, short=False,
                  levels=('DER', 'POV', 'PER'), natural=True, step=12, der_examples=False, der_most=False):
    """profile: {'DER': 2} 같은 TLC 설정. 빈 dict면 말하기 지시 없음. max_words: 답 길이 상한(없으면 자연스러운 길이).
    keep_form: 증상을 물어도 말투를 유지하라는 줄(병원 질문용). short: 대화처럼 두세 문장 길이로.
    natural: 질문에 맞춘 길이와 구어체(10단계). False면 9단계까지의 지시문.
    step: natural일 때 10이면 10단계 지시문, 11이면 와해형 길이(DIS_LENGTH), 답마다 변화(VARY), 보속증 문구(PER_LEVELS_11).
    12면 배경의 예시 대사를 빼고, 탈선 2단계에 예시(DER_EXAMPLES), 보속증은 스스로 만든 표현(PER_LEVELS_12).
    13이면 탈선 2단계를 DER2_13(긴 답의 절반 정도)과 패턴을 바꾸라는 줄(DER_VARY_13)로, der_examples=True면 모양이 다른 예시 셋."""
    out = ['You are role-playing a psychiatric patient answering questions in a clinical interview.', '', 'HOW YOU SPEAK']
    if profile:
        out += [('Your speech shows the following throughout the interview, whenever you say more than a few words '
                 if natural else 'Your speech shows the following, consistently in every answer for the whole interview ')
                + '(defined as in the Scale for the Assessment of Thought, Language and Communication):', *symptom_lines(profile)]
    if keep_form and profile:
        out.append(KEEP_FORM)
    if natural and step >= 11 and any(t in profile for t in ('DER', 'INC', 'TAN', 'CLANG')):  # 말의 빈곤은 VARY를 넣으면 길게 답해 약해짐(11단계)
        out.append(VARY)
    if natural and 'POV' not in profile and not max_words and not short:
        dis = step >= 11 and any(t in profile for t in ('DER', 'INC', 'TAN', 'CLANG', 'PER'))
        out.append(DIS_LENGTH if dis else NATURAL_LENGTH)
    elif 'POV' not in profile:
        out.append(f'Keep each answer under about {max_words} words.' if max_words else
                   'Answer in about two to four spoken sentences, as people do in conversation.' if short else
                   'Apart from this, answer at a natural spoken length.' if profile else 'Answer at a natural spoken length.')
    if natural:
        out.append(SPOKEN)
    if natural and step == 12 and profile.get('DER') == 2:
        out.append(DER_EXAMPLES)
    if natural and step >= 13 and profile.get('DER') == 2:
        out.append(DER_VARY_13)
        if der_examples:
            out.append(DER_EXAMPLES_13)
    if natural and step >= 12:
        out.append(OWN_WORDS)
    out += ['', 'WHO YOU ARE', summary(mfc, symptoms, quotes=not (natural and step >= 12)), '',
            'Speak in the first person as transcribed speech: no stage directions, no labels. Use everyday words, not '
            'psychiatric terms or diagnosis names. Never mention or explain how you speak. Stay the same person for the whole interview.']
    s = '\n'.join(out)
    for t in (levels if der_original else ()):  # 단계별 원본 기준 문구로 바꿔 넣는다
        if profile.get(t):
            text = (PER_LEVELS_12 if step >= 12 else PER_LEVELS_11) if t == 'PER' and natural and step >= 11 else LEVEL_TEXT[t]
            if t == 'DER' and natural and step >= 13 and profile[t] == 2:
                text = {2: DER2_13B if der_most else DER2_13}
            i = s.index(LINE_START[t])
            s = s[:i] + text[profile[t]] + s[s.index('\n', i):]
    return s


def load_mfc():
    return json.load(open(MFC_PATH, encoding='utf-8')) if os.path.exists(MFC_PATH) else {}


def new_patient(pid, age, sex, diagnosis='Schizophrenia', case=None):
    """case: 팀 환자 프로필 같은 사례 설명(선택). 주면 MFC가 그 설정을 따른다."""
    from psyche_tlc import make_mfc
    os.makedirs(OUT, exist_ok=True)
    mfcs = load_mfc()
    if pid not in mfcs:
        mfcs[pid] = make_mfc(client(), Budget(os.path.join(OUT, 'spent.json')), age, sex, diagnosis, case)
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
    assert 'five or more times' in system_prompt(m, {'DER': 3}) and KEEP_FORM in system_prompt(m, {'DER': 2}) and KEEP_FORM not in system_prompt(m, {})
    assert 'two to four spoken sentences' in system_prompt(m, {}, short=True)
    assert 'disjointed' in s and 'disjointed' not in system_prompt(m, {'DER': 2}, der_original=False)
    s = system_prompt(m, {'DER': 2}, step=11)
    assert DIS_LENGTH in s and VARY in s and NATURAL_LENGTH in system_prompt(m, {}) and VARY not in system_prompt(m, {})
    assert VARY not in system_prompt(m, {'POV': 2}) and system_prompt(m, {'POV': 2}, step=11) == system_prompt(m, {'POV': 2}, step=10)
    assert 'exact phrases' in system_prompt(m, {'PER': 2}, step=11) and 'pet phrases' in system_prompt(m, {'PER': 2})
    s13 = system_prompt(m, {'DER': 2}, step=13)
    assert DER2_13 in s13 and DER_VARY_13 in s13 and DER_EXAMPLES not in s13 and DER_EXAMPLES_13 not in s13
    assert DER_EXAMPLES_13 in system_prompt(m, {'DER': 2}, step=13, der_examples=True) and DER_VARY_13 not in system_prompt(m, {}, step=13)
    assert DER2_13B in system_prompt(m, {'DER': 2}, step=13, der_examples=True, der_most=True)
    assert DER_EXAMPLES in system_prompt(m, {'DER': 2}) and DER_EXAMPLES not in system_prompt(m, {'DER': 2}, step=11)
    assert '"' not in summary(m, quotes=False) and '"' in summary(m) and OWN_WORDS in system_prompt(m, {}) and OWN_WORDS not in system_prompt(m, {}, step=11)
    s = system_prompt(m, {'DER': 2}, step=10)
    assert NATURAL_LENGTH in s and SPOKEN in s and VARY not in s
    assert 'natural spoken length' in system_prompt(m, {'DER': 2}, natural=False) and SPOKEN not in system_prompt(m, {}, natural=False)
    assert NATURAL_LENGTH not in system_prompt(m, {'POV': 2}) and SPOKEN in system_prompt(m, {'POV': 2})
    assert s.index('derailment') < s.index('WHO YOU ARE') and 'Decreased' not in s and 'What you believe' not in summary(m, False)
    a = sys.argv[1:]
    if a[:1] == ['--new']:  # --new C05 31 Male "사례 설명" (사례 설명은 선택)
        print(json.dumps(new_patient(a[1], int(a[2]), a[3], case=a[4] if len(a) > 4 else None)['MFC-Behavior'], ensure_ascii=False, indent=1))
    elif a[:1] == ['--chat']:
        from persona_model import parse_profile
        chat(a[1], parse_profile(a[2:]) if a[2:] else {})
