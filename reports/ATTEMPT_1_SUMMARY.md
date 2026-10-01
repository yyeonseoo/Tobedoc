# 1차 시도 요약: 화자 조건부 few-shot 합성 + 4축 검증

> ⚠ `reports/03_validation.md`와 `reports/02_pilot_preview.md`에는 원본 전사 발췌가 들어 있어 프로젝트 밖으로 공유하면 안 된다 (git에서도 제외). 이 문서에는 원본 발췌가 없다.

## 1. 생성 개요
- **방식**: Claude가 세션 안에서 직접 작성했다 (외부 API 호출 없음). 화자 조건부 few-shot 템플릿을 쓰고, 시드는 화자당 5–8문장 연속 구간 2–3개다.
- **규모**: 112개 (CL 56 / CO 56). 파일럿 17개(시드 화자 4명) + 본 생성 95개.
- **생성 화자**: 27명. 18UG10은 실질 응답(≥30토큰)이 0개라 제외했다.
- **거부·필터링**: 0 / 0. 직접 생성이라 거부율 0%에는 정보가 없다.
- 상세: [02_generation.md](02_generation.md), [FINAL_AUGMENTATION_REPORT.md](FINAL_AUGMENTATION_REPORT.md)

## 2. 4축 검증 결과

| 축 | 핵심 수치 | 판정 |
|---|---|---|
| Fidelity | 실제에서 유의한 CL/CO 차이 8개 중 **4개만 보존** (방향 일치 5/8, n_sentences는 유의하게 뒤집힘). 합성 CL에 실제로 없는 topic drift 과장(adj_sim d=−0.34, first_dist d=+0.48, 실제 ≈ 0)과 MATTR 차이가 가짜 효과로 생김 | **불합격** |
| Diversity | 중복·준중복 0, distinct-1/2는 실제 95% 구간 안. 합성 간 평균 pairwise cos가 실제보다 높음 (CL 0.39 vs 0.31, CO 0.55 vs 0.41) → 의미 다양성 다소 부족 | **주의** |
| Memorization | pooled 5-gram overlap 2.7%로 실제 화자 간 기준선 0.8%의 **3.5배**. 최장 공유 15단어(기준선 최대 16). 시드의 특징적 구절을 복사한 사례 있음 (syn_18UG11_01: 11단어, syn_18UG14_02) | **주의** |
| Discriminability | TF-IDF+로지스틱 5-fold **AUC 0.94** (CL 0.89 / CO 0.94). 길이만 쓴 기준선은 0.48. 원인: `erm` 과용, quite/actually/really 같은 정제된 표현, `er`와 단어 반복(`it it`, `that that`)·절단어 부족 | **불합격** |

## 3. 방법론적 한계
1. **합격 임계값을 결과를 본 뒤 정했다.** 판정이 사후적이다.
2. **독립성이 없다.** 생성자, 검증 코드, 임계값이 모두 같은 세션의 같은 모델에서 나왔고, 파일럿 검증을 본 뒤 본 생성을 했다.
3. **temperature 라벨(0.8/1.0)은 실제 샘플링이 아니므로 무의미하다.** 조건 간 비교가 불가능하다.
4. 파일럿 뒤 생성자가 비유창성을 교정하려 했지만 효과는 미미했다 (MATTR 편차 0.070 → 0.057).

## 4. 데이터 관련 발견
- **그룹과 주제가 교란돼 있다.** CO 13명 중 10명은 "How was the experiment?"(실험 소감)로, CL 15명 중 14명은 창의성 질문으로 면담을 시작했다. → **하류 분석에서는 주제 층화(또는 주제 공변량 통제)가 필수다.** 합성 데이터도 이 교란을 그대로 물려받는다.
- **원천 파일**: 21AN11과 21UN11의 `_Raw.txt`가 FULL 대화 파일과 맞지 않는다 (21AN11은 약 절반 누락, 21UN11은 다른 텍스트가 섞인 것으로 의심). → 태그 포함 speaker-only 파일(`Speaker_only_for_analysis/*_speaker.txt`)을 원천으로 쓴다.
- **실제 데이터 경로**: `./data/`가 아니라 `DAIS-C-Annotated-Upload/DAIS-C-Annotated - Upload/` (config.yaml `paths.raw_root`).
- 원전사에 구두점이 없어서 문장은 spaCy가 추정한 분절이다. 실제 턴의 65%가 30토큰 미만이다.

## 5. 치환 정책
- **lexical (화자 수 규칙, CL/CO 동일 적용)**
  - 원본에서 1명만 쓰는 희귀어 → 치환: accidie, apotropaic, perfumed 등 14건.
  - 2명 이상이 쓰는 프로토콜 자극어 → 원어 유지·복원: boomerang, edge, consideration, salty, opinion, attitude, smelly, sour, sharpener, spicy, aromatic, fragrant, smoothed (48건).
- **identifier**: 고유명사와 개인정보(인물, 작품, 기관, 지명, 금액, 병력)는 화자 수와 무관하게 항상 치환한다 (36건, 2명에게 나오는 Oxford 포함).
- `ingest`가 최종 텍스트로 규칙 준수를 자동 검증한다. 위반은 0건. 상세는 `synthetic/substitution_audit.csv` (git 제외; 원본 유래 고유명사가 들어 있음).

## 6. 2차 시도 계획
1. **임계값 선고정**: 생성하기 전에 4축 합격/주의/불합격 수치 기준을 `config.yaml`에 확정하고 커밋한다. 검증 코드(`validate.py`)는 평가 로직을 바꾸지 않고 재사용한다 (입출력 경로 인자만 추가, 수정 내역 기록).
2. **Step A**: 실제 데이터로 화자별 비유창성 프로파일을 측정한다 (er/erm/기타 filler 빈도, 즉시 반복률, 문장 길이 평균·SD, MATTR, 절단어 비율) → `data/processed/speaker_profiles.parquet`.
3. **Step B**: 1차와 같은 few-shot 구조(같은 시드 구간과 목표 길이)에 시드 화자의 측정 프로파일 수치를 명시해 다시 생성한다. 특정 단어를 금지하지 않고 분포를 맞추라는 지시만 쓴다. 치환 규칙과 감사는 그대로 적용한다. 규모도 1차와 같다.
4. **Step C**: 사후 범위 필터. 그룹 실제 분포의 5–95 백분위를 벗어나거나 시드와 6-gram 이상 일치하면 탈락시키고 재생성한다 (최대 2회, 그래도 실패하면 결번).
5. **Step D**: 같은 검증 코드로 재검증하고 1차 대비 비교표를 낸다. **Step E**: 요약하고 커밋한다.
6. **중단 조건**: 파일럿 AUC가 1차보다 나쁘거나 필터 탈락률이 50%를 넘으면 전체 생성 전에 멈춘다. 2차 최종 AUC가 0.85를 넘으면 멈추고 사람의 결정을 기다린다.
