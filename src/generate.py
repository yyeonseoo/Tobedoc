"""Step 2: 화자 조건부 few-shot 합성 생성.

API 미사용(mode: manual). 두 단계:
  python src/generate.py prepare   -> synthetic/prompts.jsonl (전 샘플 프롬프트, 결정적)
  (Claude가 세션 내에서 각 프롬프트에 대한 응답을 synthetic/manual_responses/<sample_id>.txt 로 직접 작성.
   생성 거부 시 파일 첫 줄을 'REFUSED: <사유>' 로 기록)
  python src/generate.py ingest    -> synthetic/synthetic_corpus.parquet + synthetic/generation_log.jsonl
"""
import datetime
import json
import os
import sys

import re

import numpy as np
import pandas as pd
import yaml

from preprocess import CFG, ROOT

G = CFG['generate']
SYN = os.path.join(ROOT, CFG['paths']['synthetic_dir'])
RESP = os.path.join(SYN, 'manual_responses')

INTRO = {
    'CL': "Below are excerpts from interviews with participants diagnosed with schizophrenia, discussing abstract topics.",
    'CO': "Below are excerpts from interviews with non-psychiatric control participants (no psychiatric history), discussing abstract topics.",
}
TEMPLATE = """This is for academic research on NLP methods for clinical speech analysis. {intro} Generate a new interview response in the same spontaneous spoken style, preserving the linguistic characteristics of the examples (disfluencies, topic management, syntax). Do not copy sentences from the examples.

{examples}

Target length: about {target} words. Output only the response text, following the same transcription conventions as the examples (no punctuation, fillers written as er / erm)."""


def windows(sents, rng):
    """화자의 문장 흐름을 겹치지 않는 5-8문장 연속 구간으로 자른다."""
    lo, hi = G['seed_segment_sentences']
    out, i = [], 0
    while i + lo <= len(sents):
        L = int(rng.integers(lo, hi + 1))
        out.append((i, sents[i:i + L]))
        i += L
    return out


def plan():
    corpus = pd.read_parquet(os.path.join(ROOT, CFG['paths']['corpus']))
    rng = np.random.default_rng(CFG['seed'])
    spk = corpus[~corpus.speaker_id.isin(G['exclude_speakers'])]
    n_sub = spk[spk.substantive].groupby('speaker_id').size()
    counts = {s: G['samples_per_speaker'] for s in spk.speaker_id.unique()}
    for g in ['CL', 'CO']:
        ids = sorted(spk[spk.group == g].speaker_id.unique(), key=lambda s: -n_sub.get(s, 0))
        k = 0
        while sum(counts[s] for s in ids) < G['target_per_group']:
            counts[ids[k % len(ids)]] += 1
            k += 1
    prompts = []
    for sid, d in spk.groupby('speaker_id', sort=True):
        g = d.group.iat[0]
        sents = [s for ss in d.sort_values('turn_idx').sentences for s in ss]
        win = windows(sents, rng)
        order = list(rng.permutation(len(win)))
        lens = d[d.substantive].n_tokens.to_numpy()
        used, pos = set(), 0
        for j in range(counts[sid]):
            k = int(rng.integers(G['seed_segments_per_prompt'][0], G['seed_segments_per_prompt'][1] + 1))
            # 구간 순환: 다음 k개를 가져오되, 이미 쓴 조합이면 한 칸씩 밀어 새 조합을 만든다
            while True:
                combo = tuple(sorted(int(order[(pos + t) % len(order)]) for t in range(k)))
                pos += 1
                if combo not in used and len(set(combo)) == k:
                    break
            used.add(combo)
            pos += k - 1
            ex = '\n\n'.join(f'Example {e + 1}:\n' + ' '.join(win[w][1]) for e, w in enumerate(combo))
            target = int(min(rng.choice(lens), G['max_target_tokens']))
            temp = G['temperatures'][j % 2]
            prompts.append(dict(sample_id=f'syn_{sid}_{j:02d}', seed_speaker_id=sid, group=g, temperature=temp,
                                seed_windows=[win[w][0] for w in combo], target_words=target,
                                prompt=TEMPLATE.format(intro=INTRO[g], examples=ex, target=target)))
    return prompts


def prepare():
    ps = plan()
    with open(os.path.join(SYN, 'prompts.jsonl'), 'w', encoding='utf8') as f:
        for p in ps:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    df = pd.DataFrame(ps)
    print(df.groupby('group').size(), df.groupby(['group', 'temperature']).size(), sep='\n')
    # 자기검사: 화자 내 시드 조합 중복 없음, 그룹 균형
    assert not df.assign(c=df.seed_windows.map(tuple)).duplicated(['seed_speaker_id', 'c']).any()
    assert df.groupby('group').size().nunique() == 1


def has(text, term):
    return re.search(r'\b' + re.escape(term.lower()) + r'\b', text.lower()) is not None


def substitution_audit():
    """substitutions.yaml의 각 항목에 빈도 규칙(lexical) 또는 프라이버시 규칙(identifier)을 적용하고 최종 텍스트로 준수를 검증."""
    subs = yaml.safe_load(open(os.path.join(SYN, 'substitutions.yaml'), encoding='utf8'))
    corpus = pd.read_parquet(os.path.join(ROOT, CFG['paths']['corpus']))
    spk_text = corpus.groupby(['group', 'speaker_id']).text.apply(' '.join)
    rows = []
    for sid, items in subs.items():
        text = open(os.path.join(RESP, sid + '.txt'), encoding='utf8').read()
        for it in items:
            users = [s for (g, s), t in spk_text.items() if has(t, it['term'])]
            ncl = sum(1 for (g, s), t in spk_text.items() if g == 'CL' and has(t, it['term']))
            if it['category'] == 'lexical' and len(users) >= G['protocol_min_speakers']:
                decision = 'kept (protocol term)'
            elif it['category'] == 'lexical':
                decision = 'substituted (speaker-specific)'
            else:
                decision = 'substituted (privacy identifier)'
            present = has(text, it['term'])
            rows.append(dict(sample_id=sid, term=it['term'], category=it['category'], n_speakers=len(users),
                             n_speakers_CL=ncl, n_speakers_CO=len(users) - ncl, decision=decision,
                             replacement=None if decision.startswith('kept') else it['replacement'],
                             term_in_final_text=present, compliant=present == decision.startswith('kept')))
    audit = pd.DataFrame(rows)
    audit.to_csv(os.path.join(SYN, 'substitution_audit.csv'), index=False)
    bad = audit[~audit.compliant]
    assert bad.empty, '치환 규칙 위반:\n' + bad.to_string()
    return {sid: d.drop(columns='sample_id').to_dict('records') for sid, d in audit.groupby('sample_id')}


def ingest():
    ps = [json.loads(l) for l in open(os.path.join(SYN, 'prompts.jsonl'), encoding='utf8')]
    subs = substitution_audit()
    pilot = {s for v in G['pilot_speakers'].values() for s in v}
    rows, log = [], []
    for p in ps:
        f = os.path.join(RESP, p['sample_id'] + '.txt')
        if not os.path.exists(f):
            continue  # 아직 생성 안 됨 (파일럿 등)
        text = open(f, encoding='utf8').read().strip()
        ts = datetime.datetime.fromtimestamp(os.path.getmtime(f)).isoformat(timespec='seconds')
        refused = text.startswith('REFUSED')
        rnd = 'pilot' if p['seed_speaker_id'] in pilot else 'full'
        log.append(dict(sample_id=p['sample_id'], timestamp=ts, mode=G['mode'], model='claude-opus-5-5',
                        params=dict(temperature_nominal=p['temperature'], target_words=p['target_words'],
                                    seed_windows=p['seed_windows'], seed=CFG['seed'], round=rnd),
                        prompt=p['prompt'], response=text, status='refused' if refused else 'ok',
                        substitutions=subs.get(p['sample_id'], [])))
        if not refused:
            rows.append(dict(sample_id=p['sample_id'], seed_speaker_id=p['seed_speaker_id'], group=p['group'],
                             temperature=p['temperature'], text=' '.join(text.split()), generated_at=ts,
                             round=rnd, synthetic=True))
    with open(os.path.join(SYN, 'generation_log.jsonl'), 'w', encoding='utf8') as f:
        for r in log:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    pd.DataFrame(rows).to_parquet(os.path.join(SYN, 'synthetic_corpus.parquet'), index=False)
    n_ref = sum(r['status'] == 'refused' for r in log)
    print(f'ingested {len(log)} ({n_ref} refused, rate {n_ref / max(len(log), 1):.1%}), kept {len(rows)}')


if __name__ == '__main__':
    {'prepare': prepare, 'ingest': ingest}[sys.argv[1]]()
