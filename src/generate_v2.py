"""Step B (2차): 프로파일 조건화 생성.

1차 plan()에서 sample_id, 목표 길이, 명목 temperature는 그대로 가져오고, 시드 few-shot 구간만 1차와 다른 윈도우로 교체한다.
(별도 RNG seed+1, 시작 오프셋 2-4문장 → 구간 경계가 1차와 어긋남. 1차와 완전히 같은 (시작, 길이) 구간은 제외.)
프롬프트 = 1차 템플릿 + 시드 화자의 측정 프로파일 문단.

  python src/generate_v2.py prepare   -> synthetic/v2/prompts.jsonl (+ 1차 대비 시드 문장 중복률 출력)
  (Claude가 synthetic/v2/manual_responses/<sample_id>.txt 직접 작성)
  python src/post_filter.py           -> 범위·시드복사·1차복사 필터, 재생성 대상 출력
  python src/generate_v2.py ingest    -> synthetic/v2/synthetic_corpus.parquet (필터 통과분만) + generation_log.jsonl
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import generate
from preprocess import CFG, ROOT

V2 = os.path.join(ROOT, 'synthetic', 'v2')
G = CFG['generate']

PROFILE = ("Match these measured speech statistics of the speaker: 'er' ≈ {er_per100:.1f} per 100 words, "
           "'erm' ≈ {erm_per100:.1f} per 100 words, other fillers (you know / I mean / like) ≈ {other_filler_per100:.1f} per 100 words, "
           "immediate word repetitions ≈ {rep_per100:.1f} per 100 words, truncated word fragments ≈ {frag_per100:.1f} per 100 words, "
           "mean sentence length ≈ {sent_len_mean:.0f} words (sd {sent_len_sd:.0f}), moving-average type-token ratio ≈ {mattr:.2f}. "
           "Spontaneous spoken register: include false starts, self-corrections, and incomplete sentences at a similar rate "
           "to the examples. Do not write polished prose. {filler_rule} "
           "Do not reuse word sequences from the examples; retell content in different wording.")
# 파일럿 r1 이후 추가 (사용자 승인 옵션 2). 저채움말 화자 = 총 filler_per100 < LOW_FILLER_FACTOR x 그룹 응답 5백분위
# (26CT11 1.70x, 02AR17 1.89x, 28CT11 2.13x; 다음 화자 23CT18 2.93x와 간격이 커서 2.5로 둠. CL은 5백분위가 0이라 해당 없음)
LOW_FILLER_FACTOR = 2.5
FILLER_FLOOR = ("Match the speaker's measured filler rates ('er', 'erm', other fillers) as a floor, not a ceiling: "
                "do not go below them.")
FILLER_EXACT = ("This speaker's measured filler rates are already near the low end for the group: aim to match them "
                "exactly, and do not go below them.")


def v1_windows(sents, prompt):
    """1차 프롬프트의 예시 텍스트로부터 1차 구간 (시작, 길이)를 복원."""
    out = set()
    for ex in prompt.split('\n\nExample ')[1:]:
        text = ex.split(':\n', 1)[1].split('\n\nTarget length:')[0].strip()
        hits = [(i, L) for i in range(len(sents)) for L in range(5, 9) if ' '.join(sents[i:i + L]) == text]
        assert hits, 'v1 window not recovered'
        out.add(hits[0])
    return out


def plan_v2():
    corpus = pd.read_parquet(os.path.join(ROOT, CFG['paths']['corpus']))
    prof = pd.read_parquet(os.path.join(ROOT, 'data/processed/speaker_profiles.parquet')).set_index('speaker_id')
    p5 = pd.read_parquet(os.path.join(ROOT, 'data/processed/response_profiles.parquet')).groupby('group').filler_per100.quantile(.05)
    prof['filler_rule'] = [FILLER_EXACT if p5[g] > 0 and f < LOW_FILLER_FACTOR * p5[g] else FILLER_FLOOR
                           for g, f in zip(prof.group, prof.filler_per100)]
    v1 = generate.plan()
    rng = np.random.default_rng(CFG['seed'] + 1)
    lo, hi = G['seed_segment_sentences']
    out, overlap = [], []
    for sid, ps in pd.DataFrame(v1).groupby('seed_speaker_id', sort=True):
        d = corpus[corpus.speaker_id == sid].sort_values('turn_idx')
        sents = [s for ss in d.sentences for s in ss]
        used_v1 = set().union(*(v1_windows(sents, p) for p in ps.prompt))
        v1_sent_idx = {i for s, L in used_v1 for i in range(s, s + L)}
        # 1순위: 1차 시드에 한 번도 안 쓰인 문장 구간에서만 5-8문장 윈도우를 자른다
        free, i = [], 0
        while i < len(sents):
            if i in v1_sent_idx:
                i += 1
                continue
            j = i
            while j < len(sents) and j not in v1_sent_idx:
                j += 1
            s = i
            while s + lo <= j:
                L = min(int(rng.integers(lo, hi + 1)), j - s)
                free.append((s, sents[s:s + L]))
                s += L
            i = j
        # 2순위(부족할 때만): 오프셋을 준 재분할 윈도우 중 1차와 동일하지 않은 것, 1차 문장 중복이 적은 순
        need = len(ps) * G['seed_segments_per_prompt'][1]
        rest = []
        if len(free) < need:
            i = int(rng.integers(2, 5))
            while i + lo <= len(sents):
                L = int(rng.integers(lo, hi + 1))
                if (i, L) not in used_v1:
                    rest.append((i, sents[i:i + L]))
                i += L
            rest.sort(key=lambda w: len(set(range(w[0], w[0] + len(w[1]))) & v1_sent_idx))
            rest = rest[:need - len(free)]
        win = free + rest
        order = list(rng.permutation(len(free))) + list(range(len(free), len(win)))
        used, pos = set(), 0
        for p in ps.to_dict('records'):
            k = int(rng.integers(G['seed_segments_per_prompt'][0], G['seed_segments_per_prompt'][1] + 1))
            k = min(k, len(win))
            while True:
                combo = tuple(sorted(int(order[(pos + t) % len(order)]) for t in range(k)))
                pos += 1
                # 버그 수정: free와 rest에 같은 (시작, 길이) 윈도우가 함께 있을 수 있어, 한 프롬프트에 같은 예시가 두 번 들어가는 조합은 건너뛴다 (23CT18_00)
                if combo not in used and len(set(combo)) == k and len({(win[w][0], len(win[w][1])) for w in combo}) == k:
                    break
            used.add(combo)
            pos += k - 1
            ex = '\n\n'.join(f'Example {e + 1}:\n' + ' '.join(win[w][1]) for e, w in enumerate(combo))
            pr = prof.loc[sid].to_dict()
            prompt = generate.TEMPLATE.format(intro=generate.INTRO[p['group']], examples=ex, target=p['target_words'])
            head, tail = prompt.rsplit('\n\nTarget length:', 1)
            idx = {j for w in combo for j in range(win[w][0], win[w][0] + len(win[w][1]))}
            overlap.append(dict(sample_id=p['sample_id'], seed_sent_overlap_with_v1=len(idx & v1_sent_idx) / len(idx)))
            out.append(dict(sample_id=p['sample_id'].replace('syn_', 'syn2_'), seed_speaker_id=sid, group=p['group'],
                            temperature=p['temperature'], target_words=p['target_words'],
                            seed_windows=[win[w][0] for w in combo],
                            profile={k2: round(float(v), 3) for k2, v in pr.items() if k2 not in ('group', 'n_responses', 'n_words', 'filler_rule')},
                            filler_rule='exact' if pr['filler_rule'] == FILLER_EXACT else 'floor',
                            prompt=head + '\n\n' + PROFILE.format(**pr) + '\n\nTarget length:' + tail))
    return out, pd.DataFrame(overlap)


def prepare():
    os.makedirs(os.path.join(V2, 'manual_responses'), exist_ok=True)
    assert not os.listdir(os.path.join(V2, 'manual_responses')), '이미 응답이 있음: prepare는 생성 전에만 실행'
    out, ov = plan_v2()
    with open(os.path.join(V2, 'prompts.jsonl'), 'w', encoding='utf8') as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    ov.to_csv(os.path.join(V2, 'seed_overlap_with_v1.csv'), index=False)
    print(len(out), 'prompts; seed sentences shared with v1 used windows (speaker level): '
          f"mean {ov.seed_sent_overlap_with_v1.mean():.1%}, median {ov.seed_sent_overlap_with_v1.median():.1%}, "
          f"zero-overlap samples {(ov.seed_sent_overlap_with_v1 == 0).mean():.0%}")


def ingest():
    import post_filter
    st = post_filter.status_map()
    generate.ingest(syn=V2, status=lambda sid: st.get(sid, 'unfiltered'), extra_log=post_filter.rejected_log())


if __name__ == '__main__':
    {'prepare': prepare, 'ingest': ingest}[sys.argv[1]]()
