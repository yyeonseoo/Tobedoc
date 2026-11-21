"""3조건 비교: A 원본 PSYCHE, B PSYCHE와 프로필, C 우리 모델. 계획과 결과: reports/05_compare_arms.md.

python src/compare_arms.py gen MODEL C01 C05 ...   면담 생성 (프로필을 주지 않으면 12개 전부) -> data/compare/MODEL/
python src/compare_arms.py translate MODEL         한국어 번역 (GPT-4.1)
python src/compare_arms.py blind MODEL             Claude 블라인드 채점 파일 (TLC, 현실성과 답 순서를 섞은 변형)
"""
import json
import os
import sys

from patient_model import system_prompt
from psyche_tlc import MFCBehavior, MFCProfile, patient_prompt, prompts
from tlc import Budget, client

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'compare')
CAP = 6.8  # 이번 작업 GPT 상한(사용자 지정 7달러 안. 처음 5달러에서 미완성 생성을 피하려고 올림)

# 현서님 설계안 v0.2의 프로필 원문. (나이, 성별, 유형, 설명)
CASES = {
    'C01': (22, 'Male', 1, '22세 남성 대학생. 기숙사 생활. 지난 학기부터 혼자 있을 때 자신을 비난하는 목소리를 경험하며 수업 집중이 어렵다. 수면이 줄어 상담에 왔다.'),
    'C02': (34, 'Female', 1, '34세 여성 사무직. 혼자 살며 직장 동료들이 자신을 관찰한다고 믿는다. 업무 실수가 늘어 가족 권유로 왔다. 최근 새로운 치료를 시작하지 않았다.'),
    'C03': (45, 'Male', 1, '45세 남성 소매점 운영. 가족과 살고 가게를 운영한다. 최근 혼자 있을 때 들리는 목소리 때문에 손님 응대와 잠에 어려움이 생겼다. 도움을 받으러 자발적으로 왔다.'),
    'C04': (26, 'Female', 2, '26세 여성 대학원 휴학. 가족과 지낸다. 휴학 후 일상을 정리하기 어렵고 대화가 길어졌다는 말을 들었다. 수면이 불규칙하다.'),
    'C05': (31, 'Male', 2, '31세 남성 물류 업무 휴직. 동생과 산다. 최근 일을 쉬며 대화 중 생각이 다른 방향으로 이어져 동생이 내용을 따라가기 어렵다고 한다.'),
    'C06': (39, 'Female', 2, '39세 여성 프리랜서. 배우자와 산다. 최근 작업 진행과 집안일을 설명할 때 이야기 흐름을 유지하기 어렵다고 느낀다. 면담에는 참여하려 한다.'),
    'C07': (28, 'Male', 3, '28세 남성 구직 중. 부모와 산다. 최근 활동과 친구 연락이 줄었다. 부모 권유로 왔으며 일상에 대한 기억은 유지된다.'),
    'C08': (51, 'Female', 3, '51세 여성 세탁 업무 시간제. 가족과 살고 시간제로 일한다. 치료를 받고 있으며 최근 여가 활동과 대화량이 줄었다. 수면과 식사는 대체로 일정하다.'),
    'C09': (36, 'Male', 3, '36세 남성 음식점 휴직. 가족과 살며 일을 쉬고 있다. 하루 활동이 적고 본인이 먼저 말할 일이 줄었다. 정기 진료에 참여 중이다.'),
    'C10': (24, 'Female', 4, '24세 여성 복학한 대학생. 치료 후 복학했고 가족과 산다. 과거의 목소리 경험을 증상으로 이해한다. 현재는 학업과 규칙적인 수면을 유지하며 정기 진료 중이다.'),
    'C11': (47, 'Male', 4, '47세 남성 의료 지식이 있는 약국 업무 경험자. 배우자와 산다. 장기간 치료 중이며 최근 생활이 안정적이다. 과거 증상과 치료 경과를 비교적 구체적으로 기억한다.'),
    'C12': (58, 'Female', 4, '58세 여성 은퇴한 도서관 직원. 가족과 살며 치료를 지속한다. 가끔 과거와 비슷한 목소리 경험을 느끼지만 대처 방법을 알고 일상을 유지한다.'),
}
TYPE_PROFILE = {1: {}, 2: {'DER': 2}, 3: {'POV': 2}, 4: {}}  # C 조건에만. 생성 전에 정함
# 팀장 제안 12개를 영어로. Q1~6 임상, Q7~10 자유 서술, Q11~12 일관성. Q12는 캡처가 잘려 임시 문장
QUESTIONS = [
    'Could you tell me what brought you to the hospital today?',
    'What has been the hardest thing for you recently?',
    'When did that start, and how has it changed?',
    "Recently, have you had any special or unusual experiences that other people don't seem to know about?",
    'Have you noticed any changes lately in your motivation or in how you get along with people?',
    'How are these experiences affecting your daily life?',
    'Could you tell me about your day yesterday, from morning to evening?',
    'Could you describe in detail one thing that happened recently that stands out in your memory?',
    'Could you tell me about activities or hobbies you used to enjoy?',
    'Is there a conversation you had with someone recently that you remember? What was it about?',
    'Could you explain again when the hardest experience you mentioned earlier first started?',
    "Of all the difficulties you've mentioned so far, which one feels most important to you, and why?",  # 임시
]
FREE_Q = {7, 8, 9, 10}


def chat(c, budget, model, messages, schema=None):
    kw = dict(model=model, messages=messages)
    if 'sol' in model or 'terra' in model or 'astra' in model or 'luna' in model:
        kw['reasoning_effort'] = 'low'  # 온도는 기본값만 지원
    else:
        kw['temperature'] = 0.7
    if schema:
        r = c.chat.completions.parse(response_format=schema, **kw)
        budget.add(r.usage)
        return r.choices[0].message.parsed.model_dump(by_alias=True)
    r = c.chat.completions.create(**kw)
    budget.add(r.usage)
    return r.choices[0].message.content.strip()


def make_mfc(c, budget, model, age, sex, case=None):
    """psyche_tlc.make_mfc와 같은 순서와 프롬프트, 모델만 인자로."""
    g, seed = prompts('generator'), dict(diagnosis='Schizophrenia', age=age, sex=sex)
    add = '' if not case else ('\n\nAdditional case information (the MFC must be consistent with it and must not contradict it):\n' + case)
    sysm = lambda s: [{'role': 'system', 'content': s}]
    profile = chat(c, budget, model, sysm(g['MFC_Profile'].render(**seed) + add), MFCProfile)
    pj = json.dumps(profile, ensure_ascii=False)
    history = chat(c, budget, model, sysm(g['MFC_History'].render(**seed, mfc_profile_json=pj) + add))
    behavior = chat(c, budget, model, sysm(g['MFC_Behavior'].render(**seed, mfc_profile_json=pj, mfc_history_json=history) + add), MFCBehavior)
    return {'MFC-Profile': profile, 'MFC-History': history, 'MFC-Behavior': behavior}


def arm_prompt(arm, mfc_a, mfc_b, pid):
    if arm == 'A':
        return patient_prompt(mfc_a, 'BASE')
    if arm == 'B':
        return patient_prompt(mfc_b, 'BASE')
    return system_prompt(mfc_b, TYPE_PROFILE[CASES[pid][2]])


def load(path, default):
    return json.load(open(path, encoding='utf-8')) if os.path.exists(path) else default


def gen(model, pids):
    d = os.path.join(OUT, model)
    os.makedirs(d, exist_ok=True)
    c, budget = client(), Budget(os.path.join(OUT, 'spent.json'), cap=CAP)
    mp, ip = os.path.join(d, 'mfc.json'), os.path.join(d, 'interviews.json')
    mfcs, ivs = load(mp, {}), load(ip, [])
    for pid in pids:
        age, sex, _, case = CASES[pid]
        for k, cs in (('A', None), ('B', case)):
            if f'{pid}_{k}' not in mfcs:
                mfcs[f'{pid}_{k}'] = make_mfc(c, budget, model, age, sex, cs)
                json.dump(mfcs, open(mp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        for arm in 'ABC':
            if any(i['pid'] == pid and i['arm'] == arm for i in ivs):
                continue
            msgs, turns = [{'role': 'system', 'content': arm_prompt(arm, mfcs[f'{pid}_A'], mfcs[f'{pid}_B'], pid)}], []
            for k, q in enumerate(QUESTIONS):
                msgs.append({'role': 'user', 'content': q})
                a = chat(c, budget, model, msgs)
                msgs.append({'role': 'assistant', 'content': a})
                turns.append({'turn': k + 1, 'question': q, 'text': a})
            ivs.append(dict(pid=pid, arm=arm, type=CASES[pid][2], model=model, turns=turns))
            json.dump(ivs, open(ip, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            print(pid, arm, [len(t['text'].split()) for t in turns], f'${budget.spent:.2f}', flush=True)


TR = ('Translate this psychiatric patient interview from English into natural spoken Korean (존댓말, the way a Korean patient '
      "would actually talk in a clinic). Keep the patient's way of speaking exactly: if an answer is short, keep it short; "
      'if ideas jump or drift, sentences break off, or references are unclear, keep that exactly as it is. Do not tidy, '
      'fix, connect, explain or shorten anything, and do not add anything. Return JSON: {"turns": [{"question": "...", '
      '"answer": "..."}, ...]} with the same number of turns.')


def translate(model):
    d = os.path.join(OUT, model)
    ivs, tp = load(os.path.join(d, 'interviews.json'), []), os.path.join(d, 'interviews_ko.json')
    tr = load(tp, {})
    c, budget = client(), Budget(os.path.join(OUT, 'spent.json'), cap=CAP)
    for i in ivs:
        k = f"{i['pid']}_{i['arm']}"
        if k in tr:
            continue
        src = json.dumps({'turns': [{'question': t['question'], 'answer': t['text']} for t in i['turns']]}, ensure_ascii=False)
        r = c.chat.completions.create(model='gpt-4.1', temperature=0, response_format={'type': 'json_object'},
                                      messages=[{'role': 'system', 'content': TR}, {'role': 'user', 'content': src}])
        budget.add(r.usage)
        tr[k] = json.loads(r.choices[0].message.content)['turns']
        json.dump(tr, open(tp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print('translated', k, f'${budget.spent:.2f}', flush=True)


SHUFFLE_N = 6  # 채점자 기본 점검: 답 순서를 섞은 변형 면담 수


def blind(model):
    """TLC 블라인드(답변 단위, 4개로 나눔)와 현실성 블라인드(면담 단위, 답 순서를 섞은 변형 6개 포함)."""
    import random
    from patient_model import summary
    from tlc import RUBRIC
    d = os.path.join(OUT, model)
    ivs = load(os.path.join(d, 'interviews.json'), [])
    mfcs = load(os.path.join(d, 'mfc.json'), {})
    rows = [dict(tid=f"{i['pid']}_{i['arm']}_{t['turn']}", question=t['question'], text=t['text']) for i in ivs for t in i['turns']]
    random.Random(51).shuffle(rows)
    key = {}
    for k in range(4):
        with open(os.path.join(d, f'tlc_blind_{k + 1}.jsonl'), 'w', encoding='utf-8') as f:
            for j, r in list(enumerate(rows))[k::4]:
                key[f'T{j:03d}'] = r['tid']
                f.write(json.dumps(dict(id=f'T{j:03d}', question=r['question'], text=r['text']), ensure_ascii=False) + '\n')
    json.dump(key, open(os.path.join(d, 'tlc_key.json'), 'w'))
    open(os.path.join(d, 'rubric.txt'), 'w', encoding='utf-8').write(RUBRIC)
    # 현실성: 배경은 A는 A의 MFC, B와 C는 프로필을 넣은 MFC 요약(조건을 드러내는 말투 지시는 넣지 않음)
    rng = random.Random(52)
    items = []
    for i in ivs:
        bg = summary(mfcs[f"{i['pid']}_{'A' if i['arm'] == 'A' else 'B'}"])
        items.append((f"{i['pid']}_{i['arm']}", bg, [(t['question'], t['text']) for t in i['turns']]))
    for i in rng.sample(ivs, SHUFFLE_N):  # 답을 한 칸씩 밀어 질문과 답이 어긋나게
        ans = [t['text'] for t in i['turns']]
        ans = ans[1:] + ans[:1]
        bg = summary(mfcs[f"{i['pid']}_{'A' if i['arm'] == 'A' else 'B'}"])
        items.append((f"{i['pid']}_{i['arm']}_SHUF", bg, list(zip([t['question'] for t in i['turns']], ans))))
    rng.shuffle(items)
    rkey = {}
    with open(os.path.join(d, 'real_blind.jsonl'), 'w', encoding='utf-8') as f:
        for j, (k, bg, qa) in enumerate(items):
            rkey[f'R{j:02d}'] = k
            f.write(json.dumps(dict(id=f'R{j:02d}', background=bg, turns=[dict(question=q, answer=a) for q, a in qa]), ensure_ascii=False) + '\n')
    json.dump(rkey, open(os.path.join(d, 'real_key.json'), 'w'))
    print('tlc turns', len(rows), 'realism interviews', len(items))



def blind_real_profile(model):
    """사후 민감도 분석: 현실성 배경에 프로필 원문을 더한다(B, C는 그 프로필로 만들었으므로). A는 프로필을 받지 않아 기존 배경."""
    import random
    from patient_model import summary
    d = os.path.join(OUT, model)
    ivs, mfcs = load(os.path.join(d, 'interviews.json'), []), load(os.path.join(d, 'mfc.json'), {})
    rkey_old = json.load(open(os.path.join(d, 'real_key.json')))
    shuf = {k.replace('_SHUF', '') for k in rkey_old.values() if k.endswith('_SHUF')}  # 같은 변형을 다시 쓴다
    items = []
    for i in ivs:
        k = f"{i['pid']}_{i['arm']}"
        bg = summary(mfcs[f"{i['pid']}_A"]) if i['arm'] == 'A' else 'Case profile: ' + CASES[i['pid']][3] + '\n' + summary(mfcs[f"{i['pid']}_B"])
        qa = [(t['question'], t['text']) for t in i['turns']]
        items.append((k, bg, qa))
        if k in shuf:
            ans = [a for _, a in qa]
            items.append((k + '_SHUF', bg, list(zip([q for q, _ in qa], ans[1:] + ans[:1]))))
    random.Random(53).shuffle(items)
    rkey = {}
    with open(os.path.join(d, 'realp_blind.jsonl'), 'w', encoding='utf-8') as f:
        for j, (k, bg, qa) in enumerate(items):
            rkey[f'P{j:02d}'] = k
            f.write(json.dumps(dict(id=f'P{j:02d}', background=bg, turns=[dict(question=q, answer=a) for q, a in qa]), ensure_ascii=False) + '\n')
    json.dump(rkey, open(os.path.join(d, 'realp_key.json'), 'w'))
    print('realism (profile) interviews', len(items))


def evaluate(model, sub=''):
    """사전 계획대로 계산. sub: 결과 파일이 있는 하위 폴더 이름(기본은 model 폴더)."""
    import pandas as pd
    d = os.path.join(OUT, sub or model)
    I = ['derailment', 'tangentiality', 'incoherence', 'perseveration', 'clanging', 'circumstantiality', 'poverty_of_speech']
    ivs = load(os.path.join(d, 'interviews.json'), [])
    t = pd.DataFrame([dict(tid=f"{i['pid']}_{i['arm']}_{x['turn']}", pid=i['pid'], arm=i['arm'], type=i['type'], turn=x['turn'],
                           nw=len(x['text'].split())) for i in ivs for x in i['turns']])
    key = json.load(open(os.path.join(d, 'tlc_key.json')))
    j = pd.concat([pd.read_csv(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.startswith('tlc_judge_')])
    j['tid'] = j.id.map(key)
    t = t.merge(j.drop(columns='id'), on='tid', validate='1:1')
    out = []
    # 현실성과 채점자 점검
    rk = json.load(open(os.path.join(d, 'real_key.json')))
    R = pd.concat([pd.read_csv(os.path.join(d, f)).assign(rater=f) for f in sorted(os.listdir(d)) if f.startswith('real_judge_')])
    R['k'] = R.id.map(rk)
    R['shuf'] = R.k.str.endswith('_SHUF')
    R['base'] = R.k.str.replace('_SHUF', '', regex=False)
    ok_raters = []
    for rt, g in R.groupby('rater'):
        orig = g[~g.shuf].set_index('base').score
        sh = g[g.shuf].set_index('base').score
        low = int((sh < orig[sh.index]).sum())
        passed = low >= -(-2 * len(sh) // 3)  # 3분의 2 이상(올림)
        out.append(f'채점자 점검 {rt}: 답을 섞은 변형 {len(sh)}개 중 원본보다 낮게 본 것 {low}개 -> {"통과" if passed else "탈락"}')
        if passed:
            ok_raters.append(rt)
    Rn = R[~R.shuf].copy()
    Rn[['pid', 'arm']] = Rn.k.str.split('_', expand=True)
    Rn['type'] = Rn.pid.map(lambda p: CASES[p][2])
    use = Rn[Rn.rater.isin(ok_raters)]
    if len(use):
        out.append('현실성 (점검을 통과한 채점자 평균), 조건별: ' + str(use.groupby('arm').score.mean().round(2).to_dict()))
        out.append('현실성, 조건 × 유형:\n' + use.pivot_table(index='type', columns='arm', values='score', aggfunc='mean').round(2).to_markdown())
    else:
        out.append('현실성: 점검을 통과한 채점자가 없어 결과로 쓰지 않는다.')
    # 증상 충실도
    rows = []
    for arm in 'ABC':
        a = t[t.arm == arm]
        fp = a[a.type.isin([1, 4])]
        der = a[(a.type == 2) & a.turn.isin(FREE_Q)]
        pov = a[a.type == 3]
        ref = a[a.type.isin([1, 4])]
        ratio = pov.groupby('turn').nw.median().sum() / max(ref.groupby('turn').nw.median().sum(), 1)
        rows.append(dict(조건=arm, 오탐_1_4유형=f"{(fp[I].max(axis=1) >= 1).mean():.0%}",
                         탈선_2유형_자유서술=round(der.derailment.mean(), 2), 탈선_1점이상=f"{(der.derailment >= 1).mean():.0%}",
                         말의빈곤_3유형=round(pov.poverty_of_speech.mean(), 2), 말의빈곤_길이비율=round(ratio, 2),
                         답단어중앙값=a.nw.median()))
    out.append('증상 충실도 (Claude, 답변 단위 0~3점 변형 척도):\n' + pd.DataFrame(rows).to_markdown(index=False))
    t.to_csv(os.path.join(d, 'turn_scores.csv'), index=False)
    txt = '\n\n'.join(out)
    print(txt)
    return txt


DEV_VARIANTS = {'cur': dict(), 'dx1': dict(step=13), 'dx2': dict(step=13, der_examples=True),
                'dx3': dict(step=13, der_examples=True, der_most=True)}  # 13단계 탈선 다듬기 (dx3은 13b)


def dev_der(model):
    """개발: 기존 환자 P4~P6(팀 프로필 아님), 탈선 2단계, 질문 12개로 지시문 세 가지."""
    from patient_model import load_mfc
    d = os.path.join(OUT, model, 'dev_der')
    os.makedirs(d, exist_ok=True)
    c, budget = client(), Budget(os.path.join(OUT, 'spent.json'), cap=CAP)
    ip = os.path.join(d, 'interviews.json')
    ivs, mfcs = load(ip, []), load_mfc()
    for v, kw in DEV_VARIANTS.items():
        for pid in ['P4', 'P5', 'P6']:
            if any(i['pid'] == pid and i['arm'] == v for i in ivs):
                continue
            msgs, turns = [{'role': 'system', 'content': system_prompt(mfcs[pid], {'DER': 2}, **kw)}], []
            for k, q in enumerate(QUESTIONS):
                msgs.append({'role': 'user', 'content': q})
                a = chat(c, budget, model, msgs)
                msgs.append({'role': 'assistant', 'content': a})
                turns.append({'turn': k + 1, 'question': q, 'text': a})
            ivs.append(dict(pid=pid, arm=v, type=2, model=model, turns=turns))
            json.dump(ivs, open(ip, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            print(pid, v, [len(t['text'].split()) for t in turns], f'${budget.spent:.2f}', flush=True)

if __name__ == '__main__':
    m = {'Profile': {}, 'x': 1}
    assert len(QUESTIONS) == 12 and len(CASES) == 12 and all(CASES[p][2] in TYPE_PROFILE for p in CASES)
    a = sys.argv[1:]
    if a[:1] == ['gen']:
        gen(a[1], a[2:] or list(CASES))
    elif a[:1] == ['translate']:
        translate(a[1])
    elif a[:1] == ['blind']:
        blind(a[1])
    elif a[:1] == ['dev-der']:
        dev_der(a[1])
    elif a[:1] == ['blind-profile']:
        blind_real_profile(a[1])
    elif a[:1] == ['eval']:
        evaluate(a[1], a[2] if len(a) > 2 else '')
