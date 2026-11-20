"""PSYCHE 원본 구조(MFC + 환자 지시문)에 TLC 말하기 지시를 더한 실험. 계획과 결과: reports/02_psyche_tlc.md.

python src/psyche_tlc.py            MFC 생성 + 면담 생성 (GPT-4.1)        -> data/psyche/
python src/psyche_tlc.py --blind    Claude 채점용 블라인드 파일             -> data/psyche/blind_*.jsonl
python src/psyche_tlc.py --eval     지표, 판정, 보고서
"""
import json
import os
import random
import re
import sys
from typing import Literal

import pandas as pd
import yaml
from jinja2 import Template
from pydantic import BaseModel, Field
from scipy import stats

from tlc import ITEMS, Budget, client, symptom_lines

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'psyche')
REPORT = os.path.join(ROOT, 'reports', '02_psyche_tlc.md')
MODEL, TEMP = 'gpt-4.1', 0.7  # PatientHub 기본 온도
SEEDS = {'P1': (25, 'Male'), 'P2': (34, 'Female'), 'P3': (48, 'Male')}
CONDS = {'BASE': {}, 'DER': {'DER': 2}, 'INC': {'INC': 2}, 'POV': {'POV': 2}}
RATE = dict(DER='derailment', INC='incoherence', POV='poverty_of_speech')
QUESTIONS = [
    'What brings you to the hospital today?',  # 1~5: 팀 노트북과 같음
    'How have you been feeling recently?',
    'Have you noticed any changes in your sleep?',
    'Have you experienced anything unusual lately?',
    'How have these experiences affected your daily life?',
    'Can you tell me about your family?',
    'What do you usually do during the day?',
    'What are your plans for the next few weeks?',
]
NCLAUDE = 2
INTERP = """### 5. 해석

- **사전 기준대로면 통과한 조건은 없었다.** 처음 계산에서 지리멸렬은 채점은 통과했지만(1.00점, 지속 79%) 놀람도가 오르지 않았다(z 0.18, p = .15). 결과를 본 뒤 놀람도를 머뭇거림을 뺀 텍스트로 계산하도록 지표를 고쳤고(아래), 고친 지표로는 지리멸렬도 통과한다(z 2.35, p < .001). 이 통과는 사후 수정에 따른 것이다. 말의 빈곤은 거의 통과했다(평균 0.96점, 지표 p < .001, 지속 92%, 답변 중앙값 15단어). 탈선은 약했다(0.50점, 지속 50%).
- **MFC의 사고 과정 값은 말투로 옮겨지지 않았다.** GPT-4.1이 만든 MFC 셋 모두 사고 과정에 주제 이탈이나 탈선을 적었지만, BASE 면담은 어느 항목도 거의 채점되지 않았다(1점 이상 턴 8%). 팀 노트북(Qwen 1.5B)에서 와해가 안 보인 것은 모델 크기만의 문제가 아니다. PSYCHE 환자 지시문은 사고 과정을 "일관되게 유지하라"고만 하고 어떻게 말하는지는 정하지 않는다.
- **PSYCHE 지시문이 TLC 지시를 약하게 만든다.** 같은 GPT-4.1과 같은 TLC 문구로 v1(짧은 인물 지시문)에서는 탈선 1.44점, 지리멸렬 1.96점이 나왔다. PSYCHE 지시문의 "묻는 것만 답하라", MFC의 말의 양 "감소", 머뭇거림 넣기 규칙이 함께 들어가면서 탈선은 사고의 흐름이 끊기기보다 증상 설명이 길어지는 쪽으로 나왔다(중앙값 126단어, 우회증 0.25점). 탈선 조건에서 환자가 "말이 뒤섞인다"고 자기 말투를 설명하는 경우도 있었다.
- **지리멸렬은 단어 결합이 아니라 내용의 기괴함으로 나왔다.** "문이 숨 쉰다", "TV가 윙크한다"처럼 문장은 대체로 문법에 맞고, 단어의 25%가 "uh, um" 같은 머뭇거림이었다. 머뭇거림은 예측하기 쉬워 놀람도를 끌어내렸다. 그래서 `src/metrics.py`의 놀람도를 머뭇거림을 뺀 텍스트로 계산하도록 바꿨다. 지표 세트 검증 자료에는 머뭇거림이 거의 없어(0~0.5%) 지리멸렬 단계와의 ρ가 그대로였다(GPT-4.1 두 세트 0.851 -> 0.856, 0.851 -> 0.845, Claude 세트는 머뭇거림 0개라 같음). v1 결과도 판정은 같고 놀람도 z만 소수점 둘째 자리에서 바뀌었다.
- **진단명을 직접 말한 턴은 0개였다.** 팀 노트북 Qwen 1.5B에서는 "my schizophrenia" 같은 표현이 나왔다. 큰 모델은 PSYCHE의 "진단명을 말하지 마라" 규칙을 지켰다.
- **한계**: 환자 3명, 반복 1회, 채점은 AI(Claude)가 했다."""


# PatientHub patienthub/schemas/psyche.py와 같은 필드. 구조화 출력이 기본값을 받지 않아 Description만 필수로 바꿈
class IdentifyingData(BaseModel):
    age: str = Field(alias='Age')
    sex: str = Field(alias='Sex')
    marital_status: Literal['Single', 'Married', 'Divorced', 'Widowed'] = Field(alias='Marital status')
    occupation: str = Field(alias='Occupation')


class Symptom(BaseModel):
    name: str = Field(alias='Name')
    length: str = Field(alias='Length')
    alleviating_factor: str = Field(alias='Alleviating factor')
    exacerbating_factor: str = Field(alias='Exacerbating factor')
    triggering_factor: str = Field(alias='Triggering factor')
    stressor: Literal['work', 'home', 'school', 'legal issue', 'medical comorbidity', 'interpersonal difficulty', 'none'] = Field(alias='Stressor')


class PresentIllness(BaseModel):
    symptom: Symptom = Field(alias='Symptom')


class PastPsychiatric(BaseModel):
    presence: str = Field(alias='Presence')
    description: str | None = Field(alias='Description')


class PastMedical(BaseModel):
    presence: str = Field(alias='Presence')
    history: str = Field(alias='History')


class Medication(BaseModel):
    medication_name: str = Field(alias='Medication name')
    duration: str = Field(alias='Duration')
    compliance: str = Field(alias='Compliance')
    effect: str = Field(alias='Effect')
    side_effect: str = Field(alias='Side effect')


class Family(BaseModel):
    diagnosis: str = Field(alias='Diagnosis')
    substance_use: str = Field(alias='Substance use')


class Childhood(BaseModel):
    home_environment: str = Field(alias='Home environment')
    members_of_family: str = Field(alias='Members of family')
    social_environment: str = Field(alias='Social environment')


class Social(BaseModel):
    childhood_history: Childhood = Field(alias='Childhood history')
    school_history: str = Field(alias='School history')
    work_history: str = Field(alias='Work history')


class Impulsivity(BaseModel):
    suicidal_ideation: Literal['High', 'Moderate', 'Low'] = Field(alias='Suicidal ideation')
    suicidal_plan: Literal['Presence', 'Absence'] = Field(alias='Suicidal plan')
    suicidal_attempt: Literal['Presence', 'Absence'] = Field(alias='Suicidal attempt')
    self_mutilating: Literal['High', 'Moderate', 'Low'] = Field(alias='Self-mutilating behavior risk')
    homicide_risk: Literal['High', 'Moderate', 'Low'] = Field(alias='Homicide risk')


class MFCProfile(BaseModel):
    identifying_data: IdentifyingData = Field(alias='Identifying data')
    chief_complaint: str = Field(alias='Chief complaint')
    present_illness: PresentIllness = Field(alias='Present illness')
    past_psychiatric_history: PastPsychiatric = Field(alias='Past psychiatric history')
    past_medical_history: PastMedical = Field(alias='Past medical history')
    current_medication: Medication = Field(alias='Current medication')
    family_history: Family = Field(alias='Family history')
    developmental_social_history: Social = Field(alias='Developmental/social history')
    impulsivity: Impulsivity = Field(alias='Impulsivity')


class MFCBehavior(BaseModel):
    general: str = Field(alias='General appearance/attitude/behavior')
    mood: str = Field(alias='Mood')
    affect: str = Field(alias='Affect')
    spontaneity: str = Field(alias='Spontaneity')
    verbal_productivity: Literal['Decreased', 'Normal', 'Increased'] = Field(alias='Verbal productivity')
    tone_of_voice: str = Field(alias='Tone of voice')
    social_judgement: str = Field(alias='Social judgement')
    insight: str = Field(alias='Insight')
    reliability: str = Field(alias='Reliability')
    perception: str = Field(alias='Perception')
    thought_process: str = Field(alias='Thought process')
    thought_content: str = Field(alias='Thought content')


def prompts(name):
    d = yaml.safe_load(open(os.path.join(ROOT, 'psyche', f'{name}.yaml'), encoding='utf-8'))['en']
    return {k: Template(v) for k, v in d.items()}


def ask(c, budget, system, schema=None):
    kw = dict(model=MODEL, temperature=TEMP, messages=[{'role': 'system', 'content': system}])
    if schema:
        r = c.chat.completions.parse(response_format=schema, **kw)
        budget.add(r.usage)
        return r.choices[0].message.parsed.model_dump(by_alias=True)
    r = c.chat.completions.create(**kw)
    budget.add(r.usage)
    return r.choices[0].message.content.strip()


def make_mfc(c, budget, age, sex, diagnosis='Schizophrenia', case=None):
    """PatientHub PsycheGenerator와 같은 순서: Profile -> History -> Behavior.
    case: 팀 환자 프로필 같은 사례 설명. 있으면 세 단계 프롬프트 끝에 붙여 MFC가 이를 따르게 한다(원본 프롬프트는 그대로)."""
    g, seed = prompts('generator'), dict(diagnosis=diagnosis, age=age, sex=sex)
    add = ('' if not case else '\n\nAdditional case information (the MFC must be consistent with it and must not contradict it):\n'
           + case)
    profile = ask(c, budget, g['MFC_Profile'].render(**seed) + add, MFCProfile)
    pj = json.dumps(profile, ensure_ascii=False)
    history = ask(c, budget, g['MFC_History'].render(**seed, mfc_profile_json=pj) + add)
    behavior = ask(c, budget, g['MFC_Behavior'].render(**seed, mfc_profile_json=pj, mfc_history_json=history) + add, MFCBehavior)
    return {'MFC-Profile': profile, 'MFC-History': history, 'MFC-Behavior': behavior}


def patient_prompt(mfc, cond):
    s = prompts('client')['system_prompt'].render(data={'mfc': json.dumps(mfc, ensure_ascii=False, indent=2)})
    if not CONDS[cond]:
        return s
    return s + '\n'.join([
        '', '============================================================',
        'SECTION 4 — SPEECH FORM', '====================================================',
        'Your speech shows the following, consistently in every answer for the whole interview '
        '(defined as in the Scale for the Assessment of Thought, Language and Communication):',
        *symptom_lines(CONDS[cond]),
        'Where the thought process or verbal productivity in MFC-Behavior differ from this, follow this section. '
        'Never mention or explain how you speak.'])


def generate():
    os.makedirs(OUT, exist_ok=True)
    c, budget = client(), Budget(os.path.join(OUT, 'spent.json'))
    mp, ip = os.path.join(OUT, 'mfc.json'), os.path.join(OUT, 'interviews.json')
    mfcs = json.load(open(mp, encoding='utf-8')) if os.path.exists(mp) else {}
    ivs = json.load(open(ip, encoding='utf-8')) if os.path.exists(ip) else []
    for pid, (age, sex) in SEEDS.items():
        if pid not in mfcs:
            mfcs[pid] = make_mfc(c, budget, age, sex)
            json.dump(mfcs, open(mp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            print(pid, 'MFC', mfcs[pid]['MFC-Behavior']['Verbal productivity'], f'${budget.spent:.2f}', flush=True)
        for cond in CONDS:
            if any(i['patient'] == pid and i['cond'] == cond for i in ivs):
                continue
            msgs, turns = [{'role': 'system', 'content': patient_prompt(mfcs[pid], cond)}], []
            for k, q in enumerate(QUESTIONS):
                msgs.append({'role': 'user', 'content': q})
                r = c.chat.completions.create(model=MODEL, temperature=TEMP, messages=msgs)
                budget.add(r.usage)
                a = r.choices[0].message.content.strip()
                msgs.append({'role': 'assistant', 'content': a})
                turns.append({'turn': k + 1, 'question': q, 'text': a})
            ivs.append({'patient': pid, 'cond': cond, 'turns': turns})
            json.dump(ivs, open(ip, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            print(pid, cond, [len(t['text'].split()) for t in turns], f'${budget.spent:.2f}', flush=True)


def load():
    rows = [dict(patient=i['patient'], cond=i['cond'], **t) for i in json.load(open(os.path.join(OUT, 'interviews.json'), encoding='utf-8'))
            for t in i['turns']]
    d = pd.DataFrame(rows)
    d['tid'] = d.patient + '_' + d.cond + '_' + d.turn.astype(str)
    return d


def blind():
    """설정을 가리고 섞은 턴을 Claude 에이전트 수만큼 나눈다."""
    from tlc import RUBRIC
    g = load().to_dict('records')
    random.Random(11).shuffle(g)
    key = {}
    for k in range(NCLAUDE):
        with open(os.path.join(OUT, f'blind_{k + 1}.jsonl'), 'w', encoding='utf-8') as f:
            for j, r in list(enumerate(g))[k::NCLAUDE]:
                key[f'S{j:03d}'] = r['tid']
                f.write(json.dumps(dict(id=f'S{j:03d}', question=r['question'], text=r['text']), ensure_ascii=False) + '\n')
    json.dump(key, open(os.path.join(OUT, 'blind_key.json'), 'w'))
    open(os.path.join(OUT, 'rubric.txt'), 'w', encoding='utf-8').write(RUBRIC)


def evaluate():
    from metrics import PROFILE, Cached, flags
    key = json.load(open(os.path.join(OUT, 'blind_key.json')))
    j = pd.concat([pd.read_csv(os.path.join(OUT, f'judge_claude_{k + 1}.csv')) for k in range(NCLAUDE)])
    j['tid'] = j.id.map(key)
    d = load().merge(j.drop(columns='id'), on='tid', validate='1:1')
    sc = Cached()
    d = pd.concat([d, pd.DataFrame([sc(t, q) for t, q in zip(d.text, d.question)])], axis=1)
    sc.save()
    z = pd.concat([flags(g, g.cond == 'BASE') for _, g in d.groupby('patient')]).loc[d.index]
    d = pd.concat([d, z.add_prefix('z_')], axis=1)
    base = d[d.cond == 'BASE']
    res = []
    for cond, item in RATE.items():
        t = d[d.cond == cond]
        r = dict(조건=cond, 턴수=len(t), 평가_평균=t[item].mean(), 평가_BASE=base[item].mean(),
                 발현_p=stats.mannwhitneyu(t[item], base[item], alternative='greater').pvalue,
                 지표_z=t[f'z_{cond}'].mean(),
                 지표_p=stats.mannwhitneyu(t[f'z_{cond}'].dropna(), base[f'z_{cond}'].dropna(), alternative='greater').pvalue,
                 지속_1점이상=(t[item] >= 1).mean())
        r['작동'] = bool(r['발현_p'] < .05 and r['평가_평균'] >= 1 and r['지표_p'] < .05 and r['지속_1점이상'] >= .75)
        res.append(r)
    res = pd.DataFrame(res)
    means = d.groupby('cond')[ITEMS].mean().loc[list(CONDS)]
    zs = d.groupby('cond')[[f'z_{t}' for t in ['DER', 'INC', 'POV']]].mean().loc[list(CONDS)]
    label = d.text.str.contains(r'schizo|psychos|psychotic', flags=re.I)
    extra = pd.DataFrame(dict(단어수_중앙값=d.groupby('cond').n_words.median(), 진단명_직접언급_턴=label.groupby(d.cond).sum())).loc[list(CONDS)]
    mfc = json.load(open(os.path.join(OUT, 'mfc.json'), encoding='utf-8'))
    mt = pd.DataFrame({p: {'나이/성별': f"{m['MFC-Profile']['Identifying data']['Age']} {m['MFC-Profile']['Identifying data']['Sex']}",
                           '말의 양': m['MFC-Behavior']['Verbal productivity'],
                           '사고 과정': m['MFC-Behavior']['Thought process'][:120]} for p, m in mfc.items()}).T
    base_fp = (base[ITEMS].max(axis=1) >= 1).mean()
    spent = json.load(open(os.path.join(OUT, 'spent.json')))['spent']
    T = lambda x, i=False: x.to_markdown(index=i, floatfmt='.2f')
    plan = open(REPORT, encoding='utf-8').read().split('\n## 결과')[0].rstrip()
    md = f"""{plan}

## 결과

재현: `python src/psyche_tlc.py`, `--blind`, (Claude 에이전트 채점 -> `data/psyche/judge_claude_*.csv`), `--eval`. GPT 비용 {spent:.2f}달러.

### 1. 판정 (턴 단위, 조건마다 24턴 대 BASE 24턴)

{T(res)}

### 2. 조건별 평균 채점 (0~3)

{T(means, True)}

### 3. 조건별 지표 표시 (같은 환자 BASE 기준 z, 양수 = 그 유형 쪽)

{T(zs, True)}

{T(extra, True)}

BASE에서 어느 항목이든 1점 이상인 턴: {base_fp:.0%}.

### 4. GPT-4.1이 만든 MFC

{T(mt, True)}

{INTERP}
"""
    open(REPORT, 'w', encoding='utf-8').write(md)
    d.drop(columns=['text', 'question']).to_csv(os.path.join(OUT, 'turn_scores.csv'), index=False)
    print(md[len(plan):])


if __name__ == '__main__':
    s = patient_prompt({'x': 1}, 'DER')
    assert 'SECTION 4' in s and 'derailment' in s and 'SECTION 4' not in patient_prompt({'x': 1}, 'BASE')
    {'--blind': blind, '--eval': evaluate}.get(next((a for a in sys.argv[1:] if a.startswith('--')), ''), generate)()
