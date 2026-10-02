"""Step C (2차): 사후 범위 필터 + 시드 복사 필터.

  python src/post_filter.py     -> 새로 작성·수정된 응답만 평가해 synthetic/v2/filter_log.jsonl에 추가,
                                   reports/04_post_filter.md 갱신, 재생성 대상(retry) 목록 출력

규칙:
- 범위: filler_per100, rep_per100, sent_len_mean(profile_speakers 정의), adj_sim, first_dist(validate 정의)가
  해당 그룹 실제 응답 분포의 5-95 백분위 밖이면 탈락 (의미 지표가 NaN = 1문장 응답이면 그 지표는 검사 생략)
- 복사: 그 샘플 프롬프트의 시드 예시와 6-gram 이상 일치하면 탈락
- 1차 복사: 1차 합성 코퍼스(synthetic/synthetic_corpus.parquet, 112개 전체)와 6-gram 이상 일치하면 탈락 (앵커링 방지)
- 탈락 시 같은 시드로 재생성. 최초 + 재시도 2회 = 최대 3회 시도, 3회째도 탈락하면 결번(dropped)
"""
import datetime
import hashlib
import json
import os
import shutil

import pandas as pd

from preprocess import CFG, ROOT, clean, process
from profile_speakers import response_profile

V2 = os.path.join(ROOT, 'synthetic', 'v2')
LOG = os.path.join(V2, 'filter_log.jsonl')
RANGE_METRICS = ['filler_per100', 'rep_per100', 'sent_len_mean', 'adj_sim', 'first_dist']
MAX_ATTEMPTS = 3
COPY_N = 6


def words(t):
    return clean(t).lower().split()


def ngrams(w, n):
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def seed_text(prompt):
    return prompt.split('\n\n', 1)[1].rsplit('\n\nMatch these measured', 1)[0]


def semantic(texts, model):
    import validate   # build()의 adj_sim/first_dist 정의를 그대로 쓴다
    f, _ = validate.build(pd.DataFrame({'text': texts}), model)
    return f[['adj_sim', 'first_dist']]


def real_ranges(model):
    path = os.path.join(ROOT, 'data/processed/real_filter_metrics.parquet')
    if not os.path.exists(path):
        r = pd.read_parquet(os.path.join(ROOT, 'data/processed/response_profiles.parquet'))
        c = pd.read_parquet(os.path.join(ROOT, CFG['paths']['corpus']))
        c = c[c.substantive].reset_index(drop=True)
        r[['adj_sim', 'first_dist']] = semantic(c.text.tolist(), model).to_numpy()
        r.to_parquet(path, index=False)
    r = pd.read_parquet(path)
    return r, {g: d[RANGE_METRICS].quantile([.05, .95]) for g, d in r.groupby('group')}


def load_log():
    return [json.loads(l) for l in open(LOG, encoding='utf8')] if os.path.exists(LOG) else []


def status_map():
    st = {}
    for e in load_log():
        st[e['sample_id']] = 'pass' if e['passed'] else ('dropped' if e['attempt'] >= MAX_ATTEMPTS else 'retry')
    return st


def rejected_log():
    """generation_log용: 탈락한 시도들 (아카이브된 텍스트 포함)."""
    out = []
    for e in load_log():
        if not e['passed']:
            out.append(dict(sample_id=e['sample_id'], timestamp=e['timestamp'], status='filter_rejected_attempt',
                            attempt=e['attempt'], reasons=e['reasons'],
                            response=open(os.path.join(V2, 'rejected', f"{e['sample_id']}__a{e['attempt']}.txt"), encoding='utf8').read()))
    return out


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(CFG['validate']['sbert_model'], device='cpu')
    real, rng = real_ranges(model)
    v1 = pd.read_parquet(os.path.join(ROOT, CFG['paths']['synthetic_dir'], 'synthetic_corpus.parquet'))
    v1_6 = set().union(*(ngrams(words(t), COPY_N) for t in v1.text))
    prompts = {json.loads(l)['sample_id']: json.loads(l) for l in open(os.path.join(V2, 'prompts.jsonl'), encoding='utf8')}
    log = load_log()
    last = {e['sample_id']: e for e in log}
    todo = []
    for sid, p in prompts.items():
        f = os.path.join(V2, 'manual_responses', sid + '.txt')
        if not os.path.exists(f):
            continue
        text = ' '.join(open(f, encoding='utf8').read().split())
        h = hashlib.sha1(text.encode()).hexdigest()
        prev = last.get(sid)
        if prev and (prev['sha1'] == h or prev['passed'] or prev['attempt'] >= MAX_ATTEMPTS):
            continue   # 이미 평가됨 / 통과 / 결번 확정
        todo.append((sid, p, text, h, (prev['attempt'] if prev else 0) + 1))
    if todo:
        cleaned = [clean(t) for _, _, t, _, _ in todo]
        sem = semantic(cleaned, model)
        sents = [s for s, _ in process(cleaned)]
        os.makedirs(os.path.join(V2, 'rejected'), exist_ok=True)
        with open(LOG, 'a', encoding='utf8') as fo:
            for (sid, p, text, h, att), ct, ss, (_, sm) in zip(todo, cleaned, sents, sem.iterrows()):
                m = {k: v for k, v in response_profile(ct, ss).items() if k in RANGE_METRICS}
                m.update(adj_sim=sm.adj_sim, first_dist=sm.first_dist)
                q = rng[p['group']]
                reasons = [f'{k}={m[k]:.3f} 범위밖[{q.at[.05, k]:.3f},{q.at[.95, k]:.3f}]' for k in RANGE_METRICS
                           if pd.notna(m[k]) and not (q.at[.05, k] <= m[k] <= q.at[.95, k])]
                shared = ngrams(words(text), COPY_N) & ngrams(words(seed_text(p['prompt'])), COPY_N)
                if shared:
                    reasons.append(f'seed_copy_{COPY_N}gram x{len(shared)}: ' + ' | '.join(' '.join(g) for g in list(shared)[:3]))
                v1hit = ngrams(words(text), COPY_N) & v1_6
                if v1hit:
                    reasons.append(f'v1_copy_{COPY_N}gram x{len(v1hit)}: ' + ' | '.join(' '.join(g) for g in list(v1hit)[:3]))
                e = dict(sample_id=sid, group=p['group'], seed_speaker_id=p['seed_speaker_id'], attempt=att, sha1=h,
                         passed=not reasons, reasons=reasons, metrics={k: (None if pd.isna(v) else round(float(v), 4)) for k, v in m.items()},
                         timestamp=datetime.datetime.now().isoformat(timespec='seconds'))
                fo.write(json.dumps(e, ensure_ascii=False) + '\n')
                if reasons:
                    shutil.copy(os.path.join(V2, 'manual_responses', sid + '.txt'), os.path.join(V2, 'rejected', f'{sid}__a{att}.txt'))
    report(real, rng)


def report(real, rng):
    log = pd.DataFrame(load_log())
    st = pd.Series(status_map())
    first = log[log.attempt == 1]
    cat = lambda r: r.split('=')[0] if '=' in r.split(' ')[0] else r.split(':')[0].split(' ')[0]
    reasons = pd.Series([cat(r) for rs in log.reasons for r in rs]).value_counts()
    # 보정: 실제 응답이 같은 범위 필터(복사 필터 제외)를 통과하는 비율
    calib = {g: float(d[RANGE_METRICS].apply(lambda col: col.isna() | col.between(rng[g].at[.05, col.name], rng[g].at[.95, col.name])).all(axis=1).mean())
             for g, d in real.groupby('group')}
    by_g = log.assign(final=log.sample_id.map(st)).drop_duplicates('sample_id', keep='last').groupby('group').final.value_counts().unstack(fill_value=0)
    md = f"""# 04. 사후 범위 필터 (2차 Step C)

재현: `python src/post_filter.py` (상태 기록: `synthetic/v2/filter_log.jsonl`, 탈락 원문: `synthetic/v2/rejected/`)

규칙:
- 그룹별 실제 응답 분포의 5–95 백분위 밖이면 탈락. 대상 지표: {', '.join(RANGE_METRICS)}.
- 시드 예시와 {COPY_N}-gram 이상 일치하면 탈락.
- 1차 합성 코퍼스(112개)와 {COPY_N}-gram 이상 일치하면 탈락 (1차 출력 앵커링 방지).
- 탈락하면 같은 시드로 다시 생성한다. 최대 {MAX_ATTEMPTS}회까지 시도하고, 그래도 실패하면 결번이다.

## 범위 (실제 응답 5 / 95 백분위)
{pd.concat(rng, axis=1).round(3).to_markdown()}

**보정 기준선**: 같은 범위 필터(복사 필터 제외)를 실제 응답에 적용했을 때 통과율은 {', '.join(f'{g} {v:.1%}' for g, v in calib.items())}이다. 지표 5개를 각각 90% 구간으로 자르므로 실제 응답도 상당수가 탈락한다. 합성의 첫 시도 통과율은 이 값과 비교해서 읽어야 한다.

## 통과율
- 평가한 시도: {len(log)}회 / 샘플 {log.sample_id.nunique()}개
- **첫 시도 통과율: {first.passed.mean():.1%}** ({int(first.passed.sum())}/{len(first)}), 첫 시도 탈락률 {1 - first.passed.mean():.1%}
- 시도 차수별 통과율: {', '.join(f'{a}차 {d.passed.mean():.0%} (n={len(d)})' for a, d in log.groupby('attempt'))}
- 최종 상태: {st.value_counts().to_dict()}

그룹별 최종 상태:
{by_g.to_markdown()}

## 탈락 사유 분포 (모든 탈락 시도, 사유 중복 가능)
{reasons.to_markdown()}

## 결번·재시도 대기
{log[log.sample_id.map(st).isin(['dropped', 'retry'])].drop_duplicates('sample_id', keep='last')[['sample_id', 'attempt', 'reasons']].to_markdown(index=False) if st.isin(['dropped', 'retry']).any() else '없음'}
"""
    open(os.path.join(ROOT, CFG['paths']['reports'], '04_post_filter.md'), 'w', encoding='utf8').write(md)
    retry = st[st == 'retry'].index.tolist()
    print(f'first-attempt pass {first.passed.mean():.1%}; status {st.value_counts().to_dict()}; real calib {calib}')
    for sid in retry:
        e = log[log.sample_id == sid].iloc[-1]
        print('RETRY', sid, 'attempt', e.attempt, '|', '; '.join(e.reasons))


if __name__ == '__main__':
    assert ngrams('a b c d e f g'.split(), 6) == {tuple('abcdef'), tuple('bcdefg')}
    main()
