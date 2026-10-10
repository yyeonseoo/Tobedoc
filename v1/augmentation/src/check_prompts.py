"""v2 프롬프트 전수 재검사 -> reports/04b_prompt_check.md (원문 미포함: ID·해시·개수만 기록)

검사: (1) 프롬프트 안 시드 예시 중복 0건 (2) plan_v2() 재계산 결과와 저장본 일치(결정성)
      (3) 이미 생성·ingest된 샘플(generation_log의 프롬프트)과 현재 프롬프트 일치 (4) 수정으로 바뀐 프롬프트 목록(인자로 받은 기록)
"""
import datetime
import hashlib
import json
import os
import sys

from preprocess import CFG, ROOT
import generate_v2

V2 = generate_v2.V2
h = lambda s: hashlib.sha1(s.encode()).hexdigest()[:12]
cur = [json.loads(l) for l in open(os.path.join(V2, 'prompts.jsonl'), encoding='utf8')]
dups = []
for p in cur:
    ex = [e.split(':\n', 1)[1].split('\n\nMatch')[0].strip() for e in p['prompt'].split('\n\nExample ')[1:]]
    if len(set(ex)) < len(ex):
        dups.append(p['sample_id'])
new, _ = generate_v2.plan_v2()
det = [p['sample_id'] for p, q in zip(new, cur) if p != q]
glog = os.path.join(V2, 'generation_log.jsonl')
gen = {}
if os.path.exists(glog):
    for l in open(glog, encoding='utf8'):
        e = json.loads(l)
        if 'prompt' in e:
            gen[e['sample_id']] = e['prompt']
curd = {p['sample_id']: p['prompt'] for p in cur}
mismatch = [s for s, pr in gen.items() if curd.get(s) != pr]
changed = sys.argv[1].split(',') if len(sys.argv) > 1 else []
md = f"""# 04b. v2 프롬프트 전수 재검사

실행: {datetime.datetime.now().isoformat(timespec='seconds')} · `python src/check_prompts.py <변경 목록>`

배경: 23CT18_00 프롬프트에 같은 시드 예시가 두 번 들어가 있었다. 원인은 1순위(free)와 2순위(rest) 윈도우 목록에 같은 (시작, 길이) 구간이 함께 있었던 것이다.
수정: 조합을 고를 때 같은 윈도우가 두 번 들어가는 조합은 건너뛴다 (`generate_v2.py`). 이 단계는 난수를 소비하지 않으므로 다른 화자의 프롬프트는 바뀌지 않는다.
(처음 시도한 수정 — rest 목록에서 중복 윈도우를 빼는 방식 — 은 10개 프롬프트를 바꿨고 이미 생성된 파일럿 28CT11_04도 포함돼 있어서 적용 전에 폐기했다.)

| 검사 | 결과 |
|---|---|
| 전체 프롬프트 수 | {len(cur)} |
| 시드 예시가 중복된 프롬프트 | **{len(dups)}건** {dups if dups else ''} |
| plan_v2() 재계산과 저장본 불일치 (결정성) | {len(det)}건 {det if det else ''} |
| 이미 ingest된 샘플의 생성 당시 프롬프트와 현재 프롬프트 불일치 | {len(mismatch)}건 / 대조 {len(gen)}개 (파일럿) {mismatch if mismatch else ''} |
| 수정으로 바뀐 프롬프트 | **{len(changed)}건**: {', '.join(changed)} |

바뀐 프롬프트가 1건이 아니라 4건인 이유: 23CT18_00에서 중복 조합을 건너뛰면 그 화자의 순환 위치(pos)가 한 칸 밀린다. 그래서 같은 화자의 뒤 샘플(_01–_03)의 시드 조합도 따라 바뀐다. 네 샘플 모두 수정 시점에 아직 생성 전이었다.

바뀐 프롬프트 해시(현재): {', '.join(f'{s}={h(curd[s])}' for s in changed if s in curd)}
"""
open(os.path.join(ROOT, CFG['paths']['reports'], '04b_prompt_check.md'), 'w', encoding='utf8').write(md)
print(md)
assert not dups and not det and not mismatch
