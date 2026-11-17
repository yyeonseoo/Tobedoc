"""PSYCHE 환자 지시문에서 규칙을 하나씩 빼며 어떤 규칙이 말 장애를 막는지 본다. 계획과 결과: reports/03_psyche_ablation.md.

python src/psyche_ablation.py qwen          Qwen 2.5 7B(ollama)로 전체 변형 생성       -> data/ablation/qwen.json
python src/psyche_ablation.py gpt VARIANT   고른 변형 하나를 GPT-4.1로 반복 2회 생성   -> data/ablation/gpt.json
python src/psyche_ablation.py --blind       Claude 채점용 블라인드 파일
python src/psyche_ablation.py --eval        지표, 표, 보고서 결과 절
MFC는 02 실험에서 GPT-4.1이 만든 것(data/psyche/mfc.json)을 그대로 쓴다. 규칙만 바뀌게 하려는 것이다.
"""
import json
import os
import random
import sys
import time

import requests

from patient_model import load_mfc, system_prompt
from persona_model import QUESTIONS as V1_QUESTIONS
from psyche_tlc import QUESTIONS, TEMP, patient_prompt
from tlc import Budget, client

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'ablation')
REPORT = os.path.join(ROOT, 'reports', '03_psyche_ablation.md')
MFC = json.load(open(os.path.join(ROOT, 'data', 'psyche', 'mfc.json'), encoding='utf-8'))
NEW = load_mfc()  # 04 검증용 새 환자. 앞 실험의 jobs는 P1~P3만 돈다

# 환자 지시문에서 빼는 부분: (시작 문구, 다음 부분의 시작 문구)
CUT = {'brief': ('\n1. Avoid being overly cooperative', '\n2. Avoid copying technical'),
       'para': ('\n3. Use paralanguage', '\n4. Match verbal productivity'),
       'vp': ('\n4. Match verbal productivity', '\nFollow all the instructions above')}
VARIANTS = {'FULL': [], 'no_brief': ['brief'], 'no_para': ['para'], 'no_vp': ['vp'], 'no_all': ['brief', 'para', 'vp']}
CONDS = ['BASE', 'DER', 'INC']
NCLAUDE = 6
DAY_QUESTIONS = list(V1_QUESTIONS.values())  # 일상 질문(v1과 같음). 변형 이름이 _day로 끝나면 이 질문을 쓴다
GRID = ['tf_life', 'tlc_first_day', 'tf_life_day']  # tlc_first(증상 내용 + 병원 질문)와 함께 2 × 2


INTERP = """### 해석

- **Qwen 7B는 PSYCHE 안에서 TLC 지시를 전혀 따르지 않았다.** 규칙을 어떻게 빼도 탈선과 지리멸렬은 0~0.06점이었다. 같은 Qwen에 PSYCHE 없이 v1의 짧은 인물 지시문을 주면 탈선 0.38점, 지리멸렬 0.62점이 나왔다(탈선 조건은 우회증 1.04점, 지리멸렬 조건은 보속증 0.96점이 함께 나옴). 미리 정한 기준(1점 이상)에는 못 미치지만 0과는 분명히 다르다. Qwen이 TLC 지시를 못 따르는 것만이 아니라, 긴 PSYCHE 지시문(약 4,500토큰, 대부분 MFC) 속에서 TLC 절이 묻힌다.
- **GPT-4.1에서 규칙 셋을 다 빼면 탈선과 지리멸렬이 조금 늘었다.** 탈선 0.50 -> 0.69점(1점 이상 50% -> 67%), 지리멸렬 1.00 -> 1.31점(79% -> 96%). 02와 채점 묶음이 달라 직접 비교에는 한계가 있다. v1 짧은 지시문의 탈선 1.44점에는 여전히 못 미친다. 규칙 셋은 원인의 일부이고, PSYCHE 구조 자체(MFC 중심의 긴 지시문)가 더 크게 작용한다.
- **말을 짧게 하라는 규칙을 빼면 말의 빈곤이 약해진다.** GPT 말의 빈곤 0.96 -> 0.65점. 이 규칙은 말의 빈곤에는 도움이 되고 탈선에는 방해가 된다. 조건에 따라 규칙을 켜고 끄는 편이 맞다.
- **규칙을 다 빼도 지시 없는 BASE에서는 와해가 0%였다.** MFC의 사고 과정 칸에 탈선이 적혀 있어도 말로 옮겨지지 않는다는 02 결과를 다시 확인했다.
- **머뭇거림 규칙을 뺀 Qwen 지리멸렬 조건은 전보식 말투로 나왔다.** "TV says stuff. People watching me." 같은 답이 말의 빈곤 1.33점, 놀람도 z 3.25로 잡혔다. 의도한 지리멸렬이 아니라 짧고 끊긴 문장이다.
- **tlc_first(말하기 방식을 맨 앞에, MFC 요약, 약 700토큰)에서 GPT 말의 빈곤이 처음으로 판정을 통과했다.** 1.15점, 1점 이상 98%, p < .001. 탈선은 0.77점(1점 이상 75%)으로 지금까지 PSYCHE 계열 중 가장 높지만 평균 기준 1.0에는 못 미친다. 지리멸렬은 0.83점으로 오히려 낮아졌고, 지리멸렬 조건 답이 탈선으로 더 많이 채점됐다(1.27점). 문장 안이 무너지기보다 문장끼리 이어지지 않는 쪽으로 나왔다. 답은 길어졌다(지시 없음 90단어, 탈선 220단어).
- **Qwen 7B는 tlc_first에서도 약했다.** 탈선 0.12, 지리멸렬 0, 말의 빈곤 0.31점으로 v1 짧은 지시문(탈선 0.38, 지리멸렬 0.62)보다 낮다. 지시문 길이를 줄여도 환자 설정과 증상 내용이 들어가면 Qwen은 증상을 조리 있게 설명하는 쪽으로 간다.
- **정리하면** 지시문 길이와 순서가 중요하다(PSYCHE 원본 < 규칙 제거 < tlc_first). 그래도 GPT에서 v1 수준의 탈선과 지리멸렬이 나오지 않아, 환자 설정 자체(증상 설명을 유도하는 면담 질문과 MFC 내용)가 형식 장애를 누르는 것으로 보인다.
- **한계**: 채점은 AI(Claude). v1 확인은 인물 3명 반복 1회. Qwen 인물 B의 지리멸렬 면담 두 답이 600토큰에서 잘렸다(끝없이 이어 쓰는 문제)."""


# 9단계: TLC 원문(Andreasen 1986) 탈선 설명의 특징을 그대로 옮긴 2단계 문구. 원문 예시는 베낄까 봐 넣지 않는다
DER_X = ('- derailment: your ideas slip off the track onto other ideas that are only obliquely related or completely '
         'unrelated. You put things side by side that have no meaningful connection, and you shift from one frame of '
         'reference to another without warning. Your sentences are disjointed: they do not hang together, and you use '
         '"it", "they" or "that" without making clear who or what you mean. With each slip you get farther from the '
         'question, and you do not notice that your answer no longer has anything to do with it. This happens several '
         'times in every answer.')


# 5~8단계의 2단계 탈선 문구. patient_model 기본값이 9단계 문구(DER_X)로 바뀌어, 앞 단계 조건을 재현하려고 남긴다
DER_W = ('- derailment: within each answer, your ideas slip off track two to four times onto obliquely related or '
         'unrelated ideas, and you do not return to the question; each sentence makes sense on its own, but the '
         'links between them are lost.')


def old_der(s, prof):
    if prof.get('DER') == 2:
        i = s.index('- derailment:')
        s = s[:i] + DER_W + s[s.index('\n', i):]
    return s


def parse_cond(cond):
    """'BASE' -> {}, 'DER3' -> {'DER': 3}, 'DER2INC1' -> {'DER': 2, 'INC': 1}"""
    import re
    return {} if cond == 'BASE' else {t: int(l) for t, l in re.findall(r'([A-Z]+)(\d)', cond)}


def variant_prompt(mfc, cond, variant):
    if variant == 'final_x':  # 9단계: 기본값에서 탈선 문구만 DER_X로
        s = system_prompt(mfc, parse_cond(cond))
        if 'DER' in parse_cond(cond):
            i = s.index('- derailment:')
            s = s[:i] + DER_X + s[s.index('\n', i):]
        return s
    if variant.startswith('final'):  # 8단계: 그때의 우리 모델 기본값(2단계 탈선은 DER_W)
        return old_der(system_prompt(mfc, parse_cond(cond)), parse_cond(cond))
    if variant.startswith('model'):  # 우리 모델(patient_model): model 일상 질문, model_short 짧게, model_clin 병원 질문, model_clinfix 병원 질문 + 말투 유지
        return old_der(system_prompt(mfc, parse_cond(cond), keep_form=variant == 'model_clinfix', short=variant == 'model_short',
                                     levels=('DER',)), parse_cond(cond))  # 7단계 조건 그대로 재현
    if variant.startswith(('tlc_first', 'tf_life')):
        prof = {'BASE': {}, 'DER': {'DER': 2}, 'DERS': {'DER': 2}, 'INC': {'INC': 2}, 'POV': {'POV': 2},
                'MIX': {'DER': 1, 'PER': 1, 'POV': 1}, 'DERW': {'DER': 2},
                'TAN': {'TAN': 2}, 'PER': {'PER': 2}, 'CLANG': {'CLANG': 2}}[cond]  # DERS: 탈선 + 120단어 상한, MIX: v1과 같은 섞음 설정
        return system_prompt(mfc, prof, symptoms=not variant.startswith('tf_life'), max_words=120 if cond == 'DERS' else None,
                             der_original=False, keep_form=False) if cond != 'DERW' else             old_der(system_prompt(mfc, prof, keep_form=False), prof)  # 앞 단계 조건 그대로 재현
    s = patient_prompt(mfc, cond)
    for k in VARIANTS[variant]:
        a, b = CUT[k]
        i, j = s.index(a), s.index(b)
        s = s[:i] + s[j:]
    return s


def qwen(msgs):
    r = requests.post('http://localhost:11434/api/chat', timeout=1800, json=dict(
        model='qwen2.5:7b', messages=msgs, stream=False, options=dict(temperature=TEMP, num_ctx=8192, num_predict=600)))  # 끝없이 이어 쓰는 답을 막는다
    r.raise_for_status()
    return r.json()['message']['content'].strip()


def gpt(c, budget):
    def f(msgs):
        r = c.chat.completions.create(model='gpt-4.1', temperature=TEMP, messages=msgs)
        budget.add(r.usage)
        return r.choices[0].message.content.strip()
    return f


def run(name, jobs, chat, only=None):
    """jobs: [(variant, cond, rep)]. 이미 만든 면담은 건너뛴다(중간에 끊겨도 이어서 돈다)."""
    from metrics import temp
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'{name}.json')
    ivs = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else []
    have = {(i['patient'], i['variant'], i['cond'], i['rep']) for i in ivs}
    for v, cond, rep in jobs:
        for pid in (MFC if not (v == 'tlc_first_day' or v.startswith(('model', 'final'))) else {**MFC, **NEW}):
            if only and pid not in only:
                continue
            if (pid, v, cond, rep) in have:
                continue
            if temp() >= 95:
                print('95°C 이상, 쉬는 중', flush=True)
                while temp() > 85:
                    time.sleep(30)
            t0, msgs, turns = time.time(), [{'role': 'system', 'content': variant_prompt({**MFC, **NEW}[pid], cond, v)}], []
            for k, q in enumerate(DAY_QUESTIONS if v.endswith('_day') or v in ('model', 'model_short', 'final', 'final_x') else QUESTIONS):
                msgs.append({'role': 'user', 'content': q})
                a = chat(msgs)
                msgs.append({'role': 'assistant', 'content': a})
                turns.append({'turn': k + 1, 'question': q, 'text': a})
            ivs.append(dict(patient=pid, variant=v, cond=cond, rep=rep, model=name, turns=turns))
            json.dump(ivs, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            print(name, pid, v, cond, rep, [len(t['text'].split()) for t in turns], f'{time.time() - t0:.0f}s', flush=True)


def qwen_v1():
    """추가 확인: PSYCHE 없이 v1 인물 지시문으로 Qwen 7B. patient 칸에 인물, variant 칸에 'v1'."""
    import persona_model as pm
    path = os.path.join(OUT, 'qwen_v1.json')
    ivs = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else []
    have = {(i['patient'], i['variant'], i['cond']) for i in ivs}
    for cond, prof in [('BASE', {}), ('DER', {'DER': 2}), ('INC', {'INC': 2})]:
        for p in pm.PERSONAS:
            if (p, 'v1', cond) in have:
                continue
            msgs, turns = [{'role': 'system', 'content': pm.system_prompt(p, prof)}], []
            for k, q in enumerate(pm.QUESTIONS.values()):
                msgs.append({'role': 'user', 'content': q})
                a = qwen(msgs)
                msgs.append({'role': 'assistant', 'content': a})
                turns.append({'turn': k + 1, 'question': q, 'text': a})
            ivs.append(dict(patient=p, variant='v1', cond=cond, rep=0, model='qwen', turns=turns))
            json.dump(ivs, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            print('qwen_v1', p, cond, [len(t['text'].split()) for t in turns], flush=True)


def load():
    import pandas as pd
    rows = []
    for name in ['qwen', 'qwen_v1', 'gpt']:
        p = os.path.join(OUT, f'{name}.json')
        if os.path.exists(p):
            rows += [dict({k: i[k] for k in ['patient', 'variant', 'cond', 'rep', 'model']}, **t)
                     for i in json.load(open(p, encoding='utf-8')) for t in i['turns']]
    d = pd.DataFrame(rows)
    d['tid'] = d.model + '_' + d.patient + '_' + d.variant + '_' + d.cond + '_' + d.rep.astype(str) + '_' + d.turn.astype(str)
    return d


def blind():
    """아직 채점 안 된 턴만 섞어서 NCLAUDE개로 나눈다. 번호는 이어서 붙인다."""
    from tlc import RUBRIC
    kp = os.path.join(OUT, 'blind_key.json')
    key = json.load(open(kp)) if os.path.exists(kp) else {}
    done = set(key.values())
    g = [r for r in load().to_dict('records') if r['tid'] not in done]
    random.Random(len(key)).shuffle(g)
    batch = len([f for f in os.listdir(OUT) if f.startswith('blind_') and f.endswith('.jsonl')]) // NCLAUDE + 1
    start = len(key)
    for k in range(NCLAUDE):
        with open(os.path.join(OUT, f'blind_{batch}_{k + 1}.jsonl'), 'w', encoding='utf-8') as f:
            for j, r in list(enumerate(g))[k::NCLAUDE]:
                sid = f'A{start + j:04d}'
                key[sid] = r['tid']
                f.write(json.dumps(dict(id=sid, question=r['question'], text=r['text']), ensure_ascii=False) + '\n')
    json.dump(key, open(kp, 'w'))
    open(os.path.join(OUT, 'rubric.txt'), 'w', encoding='utf-8').write(RUBRIC)
    print('batch', batch, 'turns', len(g))


def evaluate():
    import pandas as pd
    from scipy import stats
    from metrics import Cached, flags
    from tlc import ITEMS
    key = json.load(open(os.path.join(OUT, 'blind_key.json')))
    j = pd.concat([pd.read_csv(os.path.join(OUT, f)) for f in sorted(os.listdir(OUT)) if f.startswith('judge_')])
    j['tid'] = j.id.map(key)
    d = load().merge(j.drop(columns='id'), on='tid', validate='1:1')
    sc = Cached()
    d = pd.concat([d, pd.DataFrame([sc(t, q) for t, q in zip(d.text, d.question)])], axis=1)
    sc.save()
    # 지표 기준: 같은 모델, 같은 환자의 FULL BASE 턴(v1 인물은 v1 BASE 턴)
    # 질문 묶음이 다르면 기준도 따로: 병원 질문은 FULL BASE, 일상 질문은 tlc_first_day BASE(v1 인물은 v1 BASE)
    d['qset'] = (d.variant.str.endswith('_day') | d.variant.isin(['v1', 'model', 'model_short', 'final', 'final_x'])).map({True: 'day', False: 'clinic'})
    normal = d.variant.isin(['FULL', 'v1', 'tlc_first_day', 'model_clin']) & (d.cond == 'BASE')  # model_short BASE는 비교 대상이라 기준에서 뺀다
    z = pd.concat([flags(g, normal[g.index]) for _, g in d.groupby(['model', 'patient', 'qset'])]).loc[d.index]
    d = pd.concat([d, z.add_prefix('z_')], axis=1)
    item = dict(BASE=None, DER='derailment', DERS='derailment', DERW='derailment', INC='incoherence', POV='poverty_of_speech', MIX=None,
                TAN='tangentiality', PER='perseveration', CLANG='clanging')
    rows = []
    for (m, v, c), g in d.groupby(['model', 'variant', 'cond']):
        pc = parse_cond(c) if c not in item else {}
        it = item[c] if c in item else item[max(pc, key=pc.get)] if len(pc) == 1 else None  # 'DER3'은 유형 이름으로, 섞음은 따로 본다
        b = d[(d.model == m) & normal & (d.qset == g.qset.iloc[0]) & (d.patient.isin(g.patient))]
        p_item = stats.mannwhitneyu(g[it], b[it], alternative='greater').pvalue if it else None
        zc = 'z_DER' if c in ('DERS', 'DERW') else f'z_{c}' if c in item else f'z_{next(iter(pc), "DER")}'
        p_z = stats.mannwhitneyu(g[zc].dropna(), b[zc].dropna(), alternative='greater').pvalue if it else None
        rows.append(dict(모델=m, 변형=v, 조건=c, 턴수=len(g), 발현_p=p_item and round(p_item, 3), 지표_p=p_z and round(p_z, 3),
                         의도항목=round(g[it].mean(), 2) if it else None,
                         지속_1점이상=round((g[it] >= 1).mean(), 2) if it else None,
                         지표_z=round(g[zc].mean(), 2) if it else None,
                         아무항목_1점이상=round((g[ITEMS].max(axis=1) >= 1).mean(), 2),
                         단어수_중앙값=g.n_words.median()))
    res = pd.DataFrame(rows)
    means = d.groupby(['model', 'variant', 'cond'])[ITEMS].mean().round(2)
    d.drop(columns=['text', 'question']).to_csv(os.path.join(OUT, 'turn_scores.csv'), index=False)
    plan = open(REPORT, encoding='utf-8').read().split('\n## 결과')[0].rstrip()
    md = f"{plan}\n\n## 결과\n\n### 조건별 요약\n\n{res.to_markdown(index=False)}\n\n### 항목별 평균 (0~3)\n\n{means.to_markdown()}\n\n{INTERP}\n"
    open(REPORT, 'w', encoding='utf-8').write(md)
    print(md[len(plan):])


# 6단계: TLC 원본(Andreasen 1986) 방식, 면담 전체를 보고 0~4점. 단계 문구는 reports/tlc_original.md
IV_ITEMS = ['poverty_of_speech', 'poverty_of_content', 'tangentiality', 'derailment', 'incoherence', 'clanging',
            'perseveration', 'circumstantiality']
IV_RUBRIC = """You are an experienced psychiatrist rating formal thought disorder with Andreasen's Scale for the Assessment of Thought, Language and Communication (TLC, 1986). You will read one whole transcribed interview (several questions and answers). Rate the interview as a whole, using only what is in the text. Ordinary speakers score 0 on every item.

Definitions:
- poverty_of_speech: restriction in the amount of spontaneous speech; replies are brief, concrete and unelaborated.
- poverty_of_content: replies are long enough but convey little information; vague, overabstract or overconcrete, repetitive or stereotyped.
- tangentiality: replying to a question in an oblique or irrelevant manner; the reply is related to the question in a distant way or not at all.
- derailment: ideas slip off the track onto another one that is obliquely related or completely unrelated.
- incoherence: speech that is essentially incomprehensible at times; words or phrases joined without meaningful connection, within the sentence.
- clanging: sounds rather than meaningful relationships govern word choice (rhyme, puns).
- perseveration: persistent repetition of words, ideas or subjects.
- circumstantiality: speech that is very indirect and delayed in reaching its goal because of tedious detail; the goal is eventually reached.

Scales:
- poverty_of_speech 0-4: 0 none (replies include additional information); 1 slight (occasional replies not elaborated when appropriate); 2 moderate (some replies not elaborated, many monosyllabic or very brief); 3 severe (answers rarely more than a few words); 4 extreme (essentially mute).
- poverty_of_content 0-4: 0 none; 1 mild (occasional replies too vague or can be markedly condensed); 2 moderate (such replies make up at least a quarter of the interview); 3 severe (at least half); 4 extreme (most of the interview).
- tangentiality, derailment, incoherence, clanging 0-4: 0 none; 1 mild (occurs once in the interview); 2 moderate (two to four times); 3 severe (5 to 10 times); 4 extreme (more than 10 times, or so often the interview is incomprehensible).
- perseveration 0-3: 0 none; 1 mild (persistent repetition of one set of words or ideas); 2 moderate (two or three sets); 3 severe (four or more sets).
- circumstantiality 0-3: 0 none; 1 mild (occasional circumstantial reply); 2 moderate (several circumstantial replies); 3 severe (many circumstantial replies).

Return one CSV row per interview: id,poverty_of_speech,poverty_of_content,tangentiality,derailment,incoherence,clanging,perseveration,circumstantiality"""


def blind_iv():
    """우리 모델(tlc_first_day) GPT 면담 전체를 섞어 면담 단위 블라인드 파일로."""
    d = load()
    d = d[(d.model == 'gpt') & (d.variant == 'tlc_first_day')]
    ivs = [(k, g.sort_values('turn')) for k, g in d.groupby(['patient', 'cond', 'rep'])]
    random.Random(17).shuffle(ivs)
    key, n = {}, 3
    for b in range(n):
        with open(os.path.join(OUT, f'iv_blind_{b + 1}.jsonl'), 'w', encoding='utf-8') as f:
            for j, (k, g) in list(enumerate(ivs))[b::n]:
                key[f'I{j:03d}'] = '_'.join(map(str, k))
                f.write(json.dumps(dict(id=f'I{j:03d}', turns=[dict(question=q, answer=t) for q, t in zip(g.question, g.text)]),
                                   ensure_ascii=False) + '\n')
    json.dump(key, open(os.path.join(OUT, 'iv_key.json'), 'w'))
    open(os.path.join(OUT, 'iv_rubric.txt'), 'w', encoding='utf-8').write(IV_RUBRIC)
    print('interviews', len(ivs))


def eval_iv():
    import pandas as pd
    key = json.load(open(os.path.join(OUT, 'iv_key.json')))
    j = pd.concat([pd.read_csv(os.path.join(OUT, f)) for f in sorted(os.listdir(OUT)) if f.startswith('iv_judge_')])
    j[['patient', 'cond', 'rep']] = j.id.map(key).str.split('_', expand=True)
    m = j.groupby('cond')[IV_ITEMS].mean().round(2)
    m.insert(0, '면담수', j.groupby('cond').size())
    target = dict(DER='derailment', DERS='derailment', DERW='derailment', INC='incoherence', POV='poverty_of_speech',
                  TAN='tangentiality', PER='perseveration', CLANG='clanging')
    hit = {c: f"{(g[target[c]] >= 2).mean():.0%}" for c, g in j.groupby('cond') if c in target}
    base_ok = (j[j.cond == 'BASE'][IV_ITEMS].max(axis=1) <= 1).mean()
    out = f"{m.to_markdown()}\n\n의도한 항목 2점 이상 면담 비율: {hit}\n\n지시 없는 면담 중 모든 항목 1점 이하: {base_ok:.0%}\n"
    j.to_csv(os.path.join(OUT, 'iv_scores.csv'), index=False)
    print(out)
    return out



def rate_gpt():
    """Claude와 같은 채점표로 GPT-4.1이 다시 채점한다(조건은 모름). 대상: P4~P6의 tlc_first_day, model, final 면담."""
    from tlc import RUBRIC, rate_items
    d = load()
    d = d[(d.model == 'gpt') & d.patient.isin(['P4', 'P5', 'P6']) & d.variant.isin(['tlc_first_day', 'model', 'final', 'final_x'])]
    d = d.groupby(['variant', 'cond'], group_keys=False).apply(lambda g: g.sample(len(g) if g.name[0] == 'final_x' else min(len(g), 10), random_state=3))  # 조건마다 10턴(9단계는 전부), 비용을 줄이려고
    rate_items(d[['tid', 'question', 'text']], os.path.join(OUT, 'gpt_rater.csv'), 'tid', RUBRIC,
               Budget(os.path.join(OUT, 'spent_08.json'), cap=2.0))
    print('rated', len(d))


if __name__ == '__main__':
    m = MFC['P1']
    assert 'Answer **only what is asked**' in variant_prompt(m, 'DER', 'FULL')
    assert 'Answer **only what is asked**' not in variant_prompt(m, 'DER', 'no_brief')
    s = variant_prompt(m, 'DER', 'no_all')
    assert 'paralanguage' not in s and 'Verbal productivity”** controls' not in s and 'SECTION 4' in s and 'Avoid copying' in s
    a = sys.argv[1:]
    assert 'disjointed' in variant_prompt(m, 'DER2', 'final_x') and 'two to four times' not in variant_prompt(m, 'DER2', 'final_x')
    assert parse_cond('DER2INC1') == {'DER': 2, 'INC': 1} and parse_cond('BASE') == {}
    assert 'one or two words at most' in variant_prompt(m, 'POV3', 'final') and 'one or two words at most' not in variant_prompt(m, 'POV3', 'model')
    assert 'two to four times' in variant_prompt(m, 'DER2', 'final') and 'two to four times' in variant_prompt(m, 'DER2', 'model_clin')
    s = variant_prompt(m, 'DERW', 'tlc_first_day')
    assert 'two to four times' in s and 'obliquely related or unrelated ideas within the answer' not in s and 'WHO YOU ARE' in s
    s = variant_prompt(m, 'DER', 'tf_life_day')
    assert 'What you believe' not in s and 'School and work' in s and 'derailment' in s
    assert 'What you believe' in variant_prompt(m, 'DER', 'tlc_first_day')
    s = variant_prompt(m, 'DER', 'tlc_first')
    assert s.index('derailment') < s.index('WHO YOU ARE') and 'Decreased' not in s and 'natural spoken length' in s
    assert 'natural spoken length' not in variant_prompt(m, 'POV', 'tlc_first')
    if a and a[0] == 'qwen' and len(a) > 1:  # 변형 하나만
        run('qwen', [(a[1], c, r) for r in range(2) for c in ['BASE', 'DER', 'INC', 'POV']], qwen)
    elif a and a[0] == 'qwen':  # 반복 0을 전부 먼저, 시간이 남으면 반복 1
        run('qwen', [(v, c, r) for r in range(2) for v in VARIANTS for c in CONDS], qwen)
    elif a and a[0] == 'qwen_v1':
        qwen_v1()
    elif a and a[0] == 'gpt_step8':  # 04 8단계: 단계 문구(말의 빈곤, 보속증), 섞음, 기본값 반복 (새 환자 P4~P6), 상한 2달러
        jobs = [('final', c, 0) for c in ['POV1', 'POV3', 'PER1', 'PER2', 'DER2INC1', 'POV2PER1']]
        jobs += [('final', c, 1) for c in ['BASE', 'DER2', 'INC2', 'POV2']]
        run('gpt', jobs, gpt(client(), Budget(os.path.join(OUT, 'spent_08.json'), cap=2.0)), only=['P4', 'P5', 'P6'])
    elif a and a[0] == 'gpt_step9':  # 04 9단계: 원문 특징 탈선 문구, 새 환자 P4~P6, 반복 2회 (8단계 상한 파일)
        run('gpt', [('final_x', 'DER2', r) for r in range(2)], gpt(client(), Budget(os.path.join(OUT, 'spent_08.json'), cap=2.0)),
            only=['P4', 'P5', 'P6'])
    elif a and a[0] == 'gpt_rate':  # 04 8단계: 두 번째 채점자 GPT-4.1, 새 환자 P4~P6의 우리 모델 면담 (같은 상한 파일)
        rate_gpt()
    elif a and a[0] == 'gpt_step7':  # 04 7단계: 세기, 병원 질문, 길이 (새 환자 P4~P6)
        jobs = [('model', c, 0) for c in ['DER1', 'DER3', 'INC1', 'INC3', 'POV1', 'POV3', 'PER1']]
        jobs += [('model_clin', c, 0) for c in ['BASE', 'DER2', 'INC2']] + [('model_clinfix', c, 0) for c in ['DER2', 'INC2']]
        jobs += [('model_short', c, 0) for c in ['BASE', 'DER2', 'INC2']]
        run('gpt', jobs, gpt(client(), Budget(os.path.join(OUT, 'spent_04.json'))), only=['P4', 'P5', 'P6'])
    elif a and a[0] == 'gpt_step5':  # 04 5단계: 새 환자 P4~P6만
        run('gpt', [('tlc_first_day', c, 0) for c in ['TAN', 'PER', 'CLANG', 'DERW']] + [('tlc_first_day', c, 1) for c in ['BASE', 'DER']],
            gpt(client(), Budget(os.path.join(OUT, 'spent_04.json'))), only=['P4', 'P5', 'P6'])
    elif a and a[0] == 'gpt_step4':  # 04 4단계: 탈선 + 길이 상한, 섞음 설정 (6명)
        run('gpt', [('tlc_first_day', c, 0) for c in ['DERS', 'MIX']], gpt(client(), Budget(os.path.join(OUT, 'spent_04.json'))))
    elif a and a[0] == 'gpt_check':  # 04 검증: 고른 구조로 새 환자 포함 BASE, DER, INC, POV
        run('gpt', [('tlc_first_day', c, 0) for c in ['BASE', 'DER', 'INC', 'POV']], gpt(client(), Budget(os.path.join(OUT, 'spent_04.json'))))
    elif a and a[0] == 'gpt_grid':  # 2 × 2 나머지 세 칸, 반복 0
        run('gpt', [(v, c, 0) for v in GRID for c in CONDS], gpt(client(), Budget(os.path.join(OUT, 'spent_04.json'))))
    elif a and a[0] == 'gpt':
        # FULL은 지표 기준용 BASE만. FULL의 TLC 조건은 02 실험 결과와 비교한다(비용 약 2.3달러)
        run('gpt', [('FULL', 'BASE', r) for r in range(2)] + [(a[1], c, r) for r in range(2) for c in ['BASE', 'DER', 'INC', 'POV']],
            gpt(client(), Budget(os.path.join(OUT, 'spent.json'))))
    elif a:
        {'--blind': blind, '--eval': evaluate, '--blind-iv': blind_iv, '--eval-iv': eval_iv}[a[0]]()
