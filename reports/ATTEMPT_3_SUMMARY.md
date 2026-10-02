# 3차 시도 요약: OpenAI API 실제 샘플링 (최종 시도)

> 진행 중 문서. 결과 절은 각 단계가 끝날 때 채운다.

## 1. 외부 전송 결정과 근거 (2026-10-02, 사용자 결정: "확인함, 전송 진행")

3차 시도는 프롬프트 안의 시드 few-shot 예시(실제 참여자 전사 발췌, CL 포함)를 OpenAI API로 전송한다. 1–2차에서는 "프로젝트 밖으로 내보내지 않는다"를 원칙으로 삼았으므로, 이 변경의 근거를 기록한다.

**사용자 판단 근거**
1. **라이선스와 공개 범위**: 코퍼스는 UK Data Service ReShare 기록 855021(DOI 10.5255/UKDA-SN-855021)로 공개돼 있다.
   - 라이선스: "Creative Commons: Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)"
   - 파일 접근: "Accessible to: Anyone (open access data)"
   - 다운로드 출처: 원본 7z의 Zone.Identifier `HostUrl=https://reshare.ukdataservice.ac.uk/855021/1/DAIS-C-Annotated-Upload.7z`로 확인.
2. **참여자 동의 범위**: 동의서(SA5_ICForm v.2.2)에 선택 동의 항목이 있다. "(Optional) I consent to the (permanent) archiving of my (anonymised) data, via the UK Data Service, so that it may be used in other studies and/or by other researchers."
3. **전송 대상**: 이미 공개된 익명화 전사 발췌만 보낸다 (전사 단계에서 `<An>` 태그로 익명화됨). 메타데이터·동의서·화자 신원 정보는 보내지 않는다.

**함께 기록하는 반대 방향 문구**: 참여자 설명서(SA5_PIS v.3.2)에는 "Your data will be anonymous to everyone outside of the research team."라는 문구가 있다. 익명성은 위 3번으로 유지된다. 상업 AI 제공자에게 전송하는 것이 동의의 취지에 맞는지는 라이선스로 정해지지 않으며, 사용자가 판단해 진행을 결정했다.

**OpenAI API 데이터 정책 (호출 전 확인, 2026-10-02, developers.openai.com/api/docs/guides/your-data)**
- 학습 사용: "data sent to the OpenAI API is not used to train or improve OpenAI models (unless you explicitly opt in to share data with us)." → 기본적으로 학습에 쓰이지 않는다. 이 프로젝트는 opt-in하지 않는다.
- 남용 모니터링 보관: /v1/chat/completions 기본 30일.
- Zero Data Retention: "subject to prior approval by OpenAI and acceptance of additional requirements" → 호출 설정만으로는 켤 수 없다 (미적용).
- `store` 파라미터: Chat Completions 출력을 distillation/evals 제품용으로 저장할지 정한다. 계정에 따라 기본 저장될 수 있다.
  → **모든 호출에 `store=False`를 명시**한다 (`src/generate_v3.py`, 호출 로그 params에 기록).

**결론**: 학습 미사용은 API 기본 정책으로 보장되고, 출력 저장은 `store=False`로 끈다. 30일 남용 모니터링 보관은 ZDR 승인 없이는 피할 수 없다. 이 잔여 위험은 사용자 판단에 포함된 것으로 기록한다.

## 2. 설정
- 모델: gpt-4o (예상 비용 $0.66, 최악 $1.99 < $5 → 하향 불필요).
  - 단가: input $2.50 / output $10.00 per 1M tokens (2026-10-01 확인).
- temperature 1.0, 호출마다 랜덤 seed, `store=False`. 누적 추정 지출이 $5에 도달하면 즉시 중단.
- 프롬프트: 2차 프롬프트 + 비유창성 위치 지시 한 문단.
- 그대로 재사용: 필터, 치환 감사, 검증, 판정 코드 (수정 없음).
- 바뀐 것은 입출력 경로뿐이다: `validate.py`에 v3 실행 항목을 추가했다. `post_filter`·`generate_v2`는 실행 중에 디렉터리 상수만 v3로 지정한다.

## 3. 결과
(진행 후 기록)
