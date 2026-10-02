"""3차 시도: OpenAI Chat Completions로 실제 샘플링 생성 (temperature 1.0, 호출마다 랜덤 seed).

프롬프트 = 2차 프롬프트(synthetic/v2/prompts.jsonl, 시드·프로파일·템플릿 그대로) + 비유창성 위치 지시 한 문단.
필터·치환 감사·검증·판정은 2차 코드를 수정 없이 재사용한다 (post_filter/generate_v2의 디렉터리 상수만 실행 중 v3로 지정).

  python src/generate_v3.py prepare            -> synthetic/v3/prompts.jsonl (프롬프트 전문) + 비용 추정 출력
  python src/generate_v3.py run <speakers|all> -> 아직 응답이 없는 슬롯 생성 (API)
  python src/generate_v3.py filter             -> post_filter 실행 (v3 디렉터리), 재샘플링 대상 출력
  python src/generate_v3.py resample           -> 필터 'retry' 슬롯을 같은 프롬프트로 재샘플링 (새 seed)
  python src/generate_v3.py ingest             -> 치환 감사 + synthetic/v3/synthetic_corpus.parquet

키: 프로젝트 루트 .env의 OPENAI_API_KEY (python-dotenv). 키 값은 어디에도 기록하지 않는다.
"""
import datetime
import hashlib
import json
import os
import secrets
import shutil
import sys

import tiktoken

import generate
import generate_v2
from preprocess import CFG, ROOT

C3 = CFG['generate_v3']
V2 = os.path.join(ROOT, 'synthetic', 'v2')
V3 = os.path.join(ROOT, 'synthetic', 'v3')
RESP = os.path.join(V3, 'manual_responses')          # 디렉터리명은 2차 코드(ingest/post_filter) 규약을 그대로 따른다
CALL_LOG = os.path.join(V3, 'generation_log.jsonl')  # API 호출 로그 (ingest 기록은 ingest_log.jsonl로 분리)
COST = os.path.join(V3, 'cost_state.json')

ADD = ("Disfluencies must appear mid-clause, not only at clause boundaries: include immediate word repeats "
       "(e.g. 'it it', 'the the'), false starts that restart mid-sentence, and hedges ('sort of', 'kind of') woven "
       "into the flow. Do not place fillers only between complete sentences.")
REFUSAL_MARKERS = ("i'm sorry", "i am sorry", "i can't", "i cannot", "i can’t", "as an ai", "i'm unable", "i am unable")


class HardLimit(Exception):
    pass


def sha(s):
    return hashlib.sha1(s.encode()).hexdigest()


def load_prompts():
    return [json.loads(l) for l in open(os.path.join(V3, 'prompts.jsonl'), encoding='utf8')]


def price(model, pin, pout):
    p = C3['pricing'][model]
    return pin / 1e6 * p['input'] + pout / 1e6 * p['output']


def prepare():
    os.makedirs(RESP, exist_ok=True)
    out = []
    for l in open(os.path.join(V2, 'prompts.jsonl'), encoding='utf8'):
        p = json.loads(l)
        head, tail = p['prompt'].rsplit('\n\nTarget length:', 1)
        p['prompt'] = head + '\n\n' + ADD + '\n\nTarget length:' + tail
        p['sample_id'] = p['sample_id'].replace('syn2_', 'syn3_')
        out.append(p)
    with open(os.path.join(V3, 'prompts.jsonl'), 'w', encoding='utf8') as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    estimate(out)


def estimate(ps):
    enc = tiktoken.encoding_for_model('gpt-4o')
    tin = [len(enc.encode(p['prompt'])) + 8 for p in ps]            # + 메시지 오버헤드
    tout = [int(p['target_words'] * 1.35) + 10 for p in ps]          # 구어 영어 ~1.35 토큰/단어
    pilot = {s for v in CFG['generate']['pilot_speakers'].values() for s in v}
    pil = [i for i, p in enumerate(ps) if p['seed_speaker_id'] in pilot]
    print(f'prompts {len(ps)} (pilot {len(pil)}); input tokens/call mean {sum(tin) / len(tin):.0f}, output est mean {sum(tout) / len(tout):.0f}')
    for m in (C3['model'], C3['fallback_model']):
        one = price(m, sum(tin), sum(tout))
        exp = one * (1 + 0.70 + 0.28)       # 2차 실측 재시도 비율(2차 시도 78/112, 3차 시도 31/112) 가정
        worst = one * 3 * (1 + C3['api_retry'])  # 전부 3회 시도 + 매번 API 재호출
        print(f'{m}: 1 pass ${one:.3f} | pilot 1 pass ${price(m, sum(tin[i] for i in pil), sum(tout[i] for i in pil)):.3f} | '
              f'expected with retries ${exp:.3f} | worst case ${worst:.3f}')


def cost_state():
    if os.path.exists(COST):
        return json.load(open(COST, encoding='utf8'))
    return dict(usd=0.0, prompt_tokens=0, completion_tokens=0, calls=0)


def client():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(os.path.join(ROOT, '.env'))
    if not os.environ.get('OPENAI_API_KEY'):
        sys.exit('OPENAI_API_KEY가 .env에 없습니다 (.env 저장 여부 확인).')
    return OpenAI()


def call(cli, p, attempt):
    """한 시도 = API 호출 1회 + 오류·거부 시 재호출 1회. 성공 시 텍스트, 실패 시 None."""
    model = C3['model']
    for k in range(1 + C3['api_retry']):
        st = cost_state()
        if st['usd'] >= C3['hard_limit_usd']:
            raise HardLimit(f"누적 추정 지출 ${st['usd']:.4f} >= ${C3['hard_limit_usd']}")
        seed = secrets.randbelow(2 ** 31)
        # store=False: 응답을 OpenAI distillation/evals 제품용으로 저장하지 않음 (학습 미사용은 API 기본 정책, ATTEMPT_3_SUMMARY §1)
        params = dict(model=model, temperature=C3['temperature'], seed=seed, store=False,
                      max_tokens=int(p['target_words'] * C3['max_tokens_factor']) + 50)
        rec = dict(sample_id=p['sample_id'], attempt=attempt, api_try=k + 1, timestamp=datetime.datetime.now().isoformat(timespec='seconds'),
                   model=model, params=params, prompt_sha1=sha(p['prompt']))
        try:
            r = cli.chat.completions.create(messages=[{'role': 'user', 'content': p['prompt']}], **params)
            text = (r.choices[0].message.content or '').strip()
            u = r.usage
            st.update(usd=st['usd'] + price(model, u.prompt_tokens, u.completion_tokens), calls=st['calls'] + 1,
                      prompt_tokens=st['prompt_tokens'] + u.prompt_tokens, completion_tokens=st['completion_tokens'] + u.completion_tokens)
            json.dump(st, open(COST, 'w', encoding='utf8'))
            refused = not text or text.lower().startswith(REFUSAL_MARKERS)
            rec.update(model_returned=r.model, system_fingerprint=r.system_fingerprint, finish_reason=r.choices[0].finish_reason,
                       response=text, prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                       cumulative_usd=round(st['usd'], 6), status='refused' if refused else 'ok')
        except Exception as e:  # API 오류: 메시지만 기록 (키 등 민감정보는 SDK 메시지에 포함되지 않음)
            rec.update(status='api_error', error=f'{type(e).__name__}: {str(e)[:300]}', cumulative_usd=round(cost_state()['usd'], 6))
            text, refused = None, True
        with open(CALL_LOG, 'a', encoding='utf8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        if not refused:
            return text
    return None


def write_or_drop(p, text, attempt):
    if text is None:   # 오류·거부 2회 → 결번 (응답 파일 없음 → ingest·필터 대상 아님)
        with open(os.path.join(V3, 'api_dropped.jsonl'), 'a', encoding='utf8') as f:
            f.write(json.dumps(dict(sample_id=p['sample_id'], attempt=attempt), ensure_ascii=False) + '\n')
        return False
    open(os.path.join(RESP, p['sample_id'] + '.txt'), 'w', encoding='utf8').write(text + '\n')
    return True


def run(which):
    cli = client()
    pilot = {s for v in CFG['generate']['pilot_speakers'].values() for s in v}
    dropped = {json.loads(l)['sample_id'] for l in open(os.path.join(V3, 'api_dropped.jsonl'), encoding='utf8')} \
        if os.path.exists(os.path.join(V3, 'api_dropped.jsonl')) else set()
    todo = [p for p in load_prompts() if not os.path.exists(os.path.join(RESP, p['sample_id'] + '.txt'))
            and p['sample_id'] not in dropped and (which == 'all' or (which == 'pilot' and p['seed_speaker_id'] in pilot))]
    try:
        for p in todo:
            write_or_drop(p, call(cli, p, 1), 1)
    finally:
        print(f"done {len(todo)} slots; cumulative ${cost_state()['usd']:.4f} over {cost_state()['calls']} calls")


def _point_filter_to_v3():
    import post_filter
    post_filter.V2 = V3                                   # 2차 코드 수정 없이 디렉터리만 v3로 지정
    post_filter.LOG = os.path.join(V3, 'filter_log.jsonl')
    return post_filter


def run_filter():
    pf = _point_filter_to_v3()
    rep = os.path.join(ROOT, CFG['paths']['reports'], '04_post_filter.md')   # post_filter.report()의 고정 출력 경로
    v2copy = os.path.join(V2, '04_post_filter_v2.md')
    if os.path.exists(rep) and not os.path.exists(v2copy):
        shutil.copy(rep, v2copy)                                  # 2차 필터 리포트 보존
    pf.main()
    shutil.copy(rep, os.path.join(ROOT, CFG['paths']['reports'], '04_post_filter_v3.md'))
    shutil.copy(v2copy, rep)                                      # 원래 자리는 2차 것으로 되돌림


def resample():
    pf = _point_filter_to_v3()
    st = pf.status_map()
    last = {}
    for e in pf.load_log():
        last[e['sample_id']] = e
    cli = client()
    ps = {p['sample_id']: p for p in load_prompts()}
    retry = [s for s, v in st.items() if v == 'retry']
    try:
        for s in retry:
            write_or_drop(ps[s], call(cli, ps[s], last[s]['attempt'] + 1), last[s]['attempt'] + 1)
    finally:
        print(f"resampled {len(retry)}; cumulative ${cost_state()['usd']:.4f}")


def ingest():
    pf = _point_filter_to_v3()
    generate_v2.V2 = V3                                   # build_substitutions()가 v3 프롬프트·응답을 보도록
    generate_v2.build_substitutions()
    calls = open(CALL_LOG, encoding='utf8').read()       # generate.ingest가 generation_log.jsonl을 덮어쓰므로 보존 후 분리
    st = pf.status_map()
    generate.ingest(syn=V3, status=lambda sid: st.get(sid, 'unfiltered'), extra_log=pf.rejected_log())
    os.replace(CALL_LOG, os.path.join(V3, 'ingest_log.jsonl'))
    open(CALL_LOG, 'w', encoding='utf8').write(calls)


if __name__ == '__main__':
    cmd = sys.argv[1]
    try:
        {'prepare': prepare, 'filter': run_filter, 'resample': resample, 'ingest': ingest,
         'run': lambda: run(sys.argv[2] if len(sys.argv) > 2 else 'pilot'),
         'estimate': lambda: estimate(load_prompts())}[cmd]()
    except HardLimit as e:
        sys.exit(f'HARD LIMIT: {e} — 중단')
