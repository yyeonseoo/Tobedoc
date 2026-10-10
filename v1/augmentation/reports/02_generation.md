# 02. 합성 생성 (Step 2)

재현: `python src/generate.py prepare` → (응답 작성) → `python src/generate.py ingest`

## 생성 방식
- **API 미사용** (사용자 지시). 모델(Claude, `claude-opus-5-5`)이 세션 안에서 `synthetic/prompts.jsonl`의 프롬프트마다 응답을 직접 작성해 `synthetic/manual_responses/<sample_id>.txt`에 저장했다. 프롬프트는 모든 샘플에 같은 템플릿(`src/generate.py` `TEMPLATE`)을 쓴다.
- 템플릿은 지정 문구를 그대로 쓰고 두 가지만 덧붙였다.
  - CO용 소개 문장: "non-psychiatric control participants"
  - 마지막 줄: 목표 길이와 전사 규약 준수 지시
- **temperature는 명목상 라벨이다.** 직접 생성이라 실제 샘플링 파라미터가 적용되지 않았으므로 0.8과 1.0 조건 간 비교는 의미가 없다. 배정 결과는 CL 28/28, CO 30/26이다 (5개 생성 화자는 화자 내에서 번갈아 배정되므로 홀수가 된다).
- 로그: `synthetic/generation_log.jsonl`. 샘플마다 프롬프트 전문, 파라미터(명목 temperature, 목표 길이, 시드 구간, seed, round), 응답, 상태, 생성 시각, 치환 기록이 들어 있다.

## 시드 구성
- 각 화자의 문장 흐름(턴 순서)을 겹치지 않는 5–8문장 연속 구간으로 나눴다. 샘플마다 2–3개 구간을 섞인 순서대로 순환하며 가져오되, 이미 쓴 조합이면 건너뛴다. 화자 안에서 같은 구간 조합은 한 번도 반복되지 않았다 (`prepare`의 assert로 확인).
- 구간이 3–4개뿐인 **02AR17, 16OV11**은 조합은 모두 달라도 개별 구간이 평균 2회씩 재사용됐다.
- 목표 길이는 시드 화자의 실질 응답(≥30토큰) 길이 분포에서 뽑았다. 상한 350에 걸린 샘플이 2개 있다 (실제 실질 응답 중 350토큰 초과는 4.4%).

## 생성량

| | CL | CO |
|---|---|---|
| 생성 화자 | 14 (18UG10 제외: 시드 불가) | 13 |
| 샘플 | 56 (화자당 4) | 56 (화자당 4, 실질 응답이 많은 4명만 5) |
| 공백 단어 수 평균 / 중앙값 / 최대 | 95 / 58 / 443 | 117 / 103 / 322 |
| 거부 | 0 | 0 |
| 필터링 제외 | 0 | 0 |

- **거부율 0%** (0/112). 직접 생성 모드라서 API 생성에서 말하는 거부율과 같은 의미는 아니다.
- **라운드**: pilot 17개(시드 화자 4명) → 파일럿 검증 → full 95개. 파일럿에서 비유창성이 부족하고 MATTR이 높다는 진단이 나와, full 라운드에서는 생성자가 비유창성 밀도를 재현하는 데 더 신경 썼다. 프롬프트 템플릿은 바꾸지 않았다. 효과는 03_validation의 라운드별 표 참고 (작음).
- 작성 중 자체 수정 1건: 10EB15 샘플 4개의 er 밀도가 실제 화자의 약 2배(캐리커처)여서 전체 검증 전에 다시 썼다.

## 시드 표현 치환 규칙과 감사
`synthetic/substitutions.yaml`에 치환·회피한 시드 표현을 모두 기록했다. `ingest`가 규칙을 적용하고 최종 텍스트로 준수 여부를 검증한다. 위반이 있으면 ingest가 중단되며, 결과는 `synthetic/substitution_audit.csv`와 generation_log의 `substitutions` 필드에 남는다. **CL과 CO에 같은 규칙을 적용한다.**

| 범주 | 규칙 | 판정 | 항목 수 | 어휘 |
|---|---|---|---|---|
| lexical | 원본에서 ≥2명 화자에 등장 → 프로토콜 주제어 → **원어 유지** | 유지 | 48 | boomerang(8명), edge(7), consideration(6), salty(6), opinion(5), attitude(4), smelly(4), sour(4), sharpener(3), spicy(3), aromatic(3), fragrant(3), smoothed(2). 모두 심리언어 과제 자극어 |
| lexical | 1명에게만 등장 → 화자 특이 어휘 → **치환** | 치환 | 14 | accidie, apotropaic (23EB14 1명만), stroked, deaf, perfumed, lifescape, spoonerism, Paintshop/JQuery/Netbeans/Notepad/RWD |
| identifier | 인물·작품·기관·지명·금액·병력 → 빈도와 무관하게 **치환** (프라이버시) | 치환 | 36 | 예: 가수명, 게임명, 대학·기관명, 금액, 가족 병력 |

- 처음 생성할 때는 과제 자극어(boomerang 등)를 화자 특이 어휘로 잘못 보고 회피했다. 사용자 지시에 따라 원본의 화자별 등장 수를 확인한 뒤, 프로토콜 주제어로 판정된 항목을 해당 샘플 27개에 복원했다. 반대로 1명에게만 등장하는 "perfumed"는 그대로 옮겨 적었던 것을 치환했다. 복원은 해당 단어만 자연스럽게 다시 넣는 최소 수정이다.
- identifier 범주에는 빈도 규칙을 적용하지 않았다. 예를 들어 "Oxford"는 2명에게 등장하지만 치환을 유지했다. 이 범주에도 빈도 규칙을 적용하려면 `substitutions.yaml`의 category만 바꾸면 된다.
