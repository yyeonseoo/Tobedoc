# Clinical feature audit of patient persona systems

회의 자료: [2쪽 중간보고서 PDF](share/meeting-20261108/중간회의_연구보고서.pdf) · [편집용 Word](share/meeting-20261108/중간회의_연구보고서.docx) · [Git 이력과 이전 산출물](archive/README.md).

2026-11-08 · 로드맵 3/6: [기존 자료 강도 비교 마감](share/team-strength-v01/강도비교_마감결과.md). GPT-4.1 중간 불확실 4개 → 강한 지시에서 있음 1·불확실 2·없음 1. Claude는 두 지시 모두 5/5 있음(천장). 기존 팀 자료 탐색을 닫고 본 실험 실행 결정은 별도로 둔다.

## 2026-11-04 최신 결과

로드맵 1단계 내 두 번째 탐색: [INC3 어휘 구성 분해 보고서](share/inc3-mechanism-v01/INC3_어휘구성분해보고서.md). 기능어 목록의 반복 손실 변화가 Claude +0.2391, GPT-4.1 −0.0699로 전체 MATTR의 반대 방향을 설명했다. 목록 외 단어의 MATTR25 차이는 각각 +0.0093/+0.0092로 작았다. 고정 목록의 산술 분해이며 품사·임상 타당도 검증이나 인과 기전은 아니다. Ollama 6회로 24개 원문을 보조 검토했고 인용 불일치 2개와 해석 한계를 기록했다. 최신 체크포인트는 config/research-checkpoint-v14.json이다. 현재는 초기 탐색이며, 앞으로 약 한 달간 본 실험·필요시 수정·미사용 조건 검증을 진행한 뒤 논문을 작성한다.

Claude·GPT-4.1 각 114개, 총 228개 생성 원문과 설명서를 확보하여 첫 특징 비교를 완료했다. [특징 비교 보고서](share/team-corpus-v01/특징비교보고서.md), [전체 조건표](share/team-corpus-v01/전체조건표.md), [동일 질문 원문 예시](share/team-corpus-v01/원문예시.md)를 먼저 읽는다. 원문 미확보 상태는 해소됐으며, 원평가표·국소 참조 구간은 별도 미확보다.

PER3의 반복률 증가는 두 생성 자료에 공통으로 나타났으나 INC3 어휘 다양성 변화는 반대 방향이었다. Claude의 대화·평가 피드백과 GPT의 API 생성 차이를 고려해 모델 우열로 해석하지 않는다. 새 발화 생성은 없었다. 입력·설정·코드·결과는 새 로컬 Git 이력에 기록한다. 기존 연구 전체의 소급 Git 백업이나 팀 원격 저장소 push는 하지 않았다.

이 폴더는 **기존 환자 페르소나의 임상 특징 표현, 구현, 생성 행동을 분석하고 한계를 개선하는 연구**의 작업공간이다. 영어를 사용하며 실제 환자 모집·제한된 임상자료 확보를 선행조건으로 삼지 않는다. 교수님 자문은 가능하지만 체계적인 전문가 채점은 확보되지 않았다.

## 먼저 읽을 문서

- [네 특징 통합과 수치 보존 후속 실험](share/overnight-completion-v01/페르소나%20특징%20비교와%20수치%20보존%20후속%20실험%20보고서.docx): pleasing 40응답과 미사용 사례의 단일 문장 제약 비교 32응답을 연속 실행해 완료했다. 사후 수치 표현 분석의 단일값 일치는 발견 단계에서 무지시·간결함 24/24, PSI 13/24, PatientSim 5/24였다. 후속은 PSI 12→18/24, PatientSim 7→13/24였지만 세 항목을 모두 보존한 제약 답변은 3/8·1/8뿐이었다. 두 소스 모두 안정적인 기본 수치 보존 모드로 채택하지 않았다. 원래 고정 규칙의 점수와 시간 환산·구간·문맥을 구분한 사후 보조 분석을 별도 보존했다. 독립 임상 평가가 아니다. 기존 발화량·정보 공개·주제 유지와 합쳐 주요 과제의 응답 기록은 240개이며, 고유 텍스트·독립 환자 수가 아니다. [실행 인계와 재현 안내](share/overnight-completion-v01/README.ko.md), [통합 수치](analysis/integrated-checkpoint-v05/results.json), [발견 원문](analysis/pleasing-discovery-v01/transcripts.md), [후속 원문](analysis/pleasing-factlock-v01/transcripts.md), [다음 의사결정](config/research-checkpoint-v05.json). 이번 로컬 생성 8.1분, 입력 17,076·출력 5,308토큰, 유료 API·자동 judge·다운로드·원격 작업 0회. 전체 구현 테스트 134개 통과. 재검증 `python3 scripts/verify_pleasing_study.py --report`. 추가 생성이나 자동화는 남겨두지 않았다.
- [질문 대응과 문장 유사도 분리 분석](share/topic-local-separation-v01/페르소나%20질문%20대응과%20문장%20유사도%20분리%20분석.docx): 기존 48응답·411문장을 저장 벡터로 재분석했다. plain/verbose/tangent 모두 요청 정보 48/48개를 언급했으나, tangent−verbose의 인접 유사도는 −0.03856(14/16쌍), 동일 문장 집합의 무작위 배열 기대값을 뺀 차이는 −0.02172(13/16쌍)였다. 같은 단어·문장 수의 요청/다른 사건 문장 삽입 대조에서 다른 사건 삽입의 인접 유사도가 오히려 높아진 경우는 20/48쌍이었다. 질문 대응·주제 관련도·국소 유사도는 별개로 유지하며 coherence 등급으로 통합하지 않는다. plain의 인접 유사도 분모는 1문장 답변을 제외한 15개다. 새 생성·임베딩·LLM judge·유료 API·다운로드 0회. [집계](analysis/topic-local-separation-v01/results.json), [원문과 대조](analysis/topic-local-separation-v01/response-details.json), [해석](analysis/topic-local-separation-v01/interpretation.ko.md), [현재 체크포인트와 다음 후보](config/research-checkpoint-v04.json). 재계산 `python3 scripts/topic_local_separation.py`, 검증 `python3 scripts/verify_topic_local_separation.py --report`. 단위 테스트 126개 통과. 첫 세 특징 분석을 마무리하고 다음은 pleasing의 표현 완화와 사실값 변경을 분리하는 과제 설계다. 아직 새 pleasing 발화는 생성하지 않았다.
- [정보 공개와 거절 표현 비교 보고서](share/disclosure-comparison-v02/페르소나%20정보%20공개와%20거절%20표현%20비교%20보고서.docx): 새 허구 사례 2개 × 질문 순서 2종 × seed 2개 × 지시 조건 5개의 첫 응답 40개를 로컬에서 생성했다. 단순 간결함과 PATIENT-Ψ reserved는 평균 32.3/29.3단어, 개인 정보 표식은 24/24와 13/24였다. PatientSim distrust는 73.3단어·19/24표식이었으며, 요청 목적을 따지는 8응답 중 6개는 개인 표식 3개를 모두 언급했다. 정보 생략을 의도적 거절로 통합하지 않았다. 사전 비공개 문구 규칙의 누락을 발견했지만 원래 점수는 보존하고, 40개 원문의 어시스턴트 근거 구간 대조를 사후 보조 분석으로 분리했다. 독립 사람·전문가 평가가 아니다. 동일 Qwen 위의 추출 지시 비교이며 원형 시스템 순위가 아니다. [측정값](analysis/disclosure-comparison-v02/results.json), [원문 대조](analysis/disclosure-comparison-v02/reference-audit.json), [전체 원문](analysis/disclosure-comparison-v02/transcripts.md). 로컬 생성 3.3분, 유료 API·새 다운로드 0회, 구현 테스트 117개 통과. 재집계 `python3 scripts/disclosure_comparison.py analyze`, 원문 감사 재검산 `python3 scripts/audit_disclosure_comparison.py`, 묶음 검증 `python3 scripts/verify_disclosure_comparison.py --report`. 재생성할 필요가 없다.
- [추가 서술 분석과 후속 비교 설계](share/role-enrichment-v01/페르소나%20추가%20서술%20분석과%20후속%20비교%20설계.docx): 관련 1차 자료 6편을 검토하고 기존 80응답을 724문장으로 분리했다. 이름·역할 단어 동시 출현 후보 27개를 전수 점검해 입력에 명시되지 않은 역할 구체화 6개를 연결했다. 무지시/단순 verbose/PSI verbose/PatientSim verbose/concise는 각각 0/12, 1/12, 3/12, 2/12, 0/12였다. 전체 환각률이나 원형 시스템 순위가 아니라 어시스턴트의 근거 구간 대조를 포함한 좁은 탐색 결과다. 자동 규칙의 누락과 최종 후보 감사 결과를 구분했다. [판정 지침](catalog/additional-narrative-codebook-v01.ko.md), [7개 비교 축과 실행 순서](config/feature-comparison-roadmap-v03.json), [문헌별 적용 결정](catalog/literature-decisions-v03.json), [역할 집계](analysis/role-enrichment-v01/audited-results.json). 기존 144응답의 계측 대조도 저장 벡터로 재확인했으며 새 실험으로 세지 않는다. 추가 모델 추론·유료 API·다운로드 0회. 재집계 `python3 scripts/role_enrichment.py analyze` 다음 `python3 scripts/finalize_role_audit.py`, 기존 계측 재확인 `python3 scripts/recheck_saved_metric_controls.py`, 검증 `python3 scripts/verify_role_audit.py --report`. 사람 평가 자료는 [25문장 미평가 묶음](analysis/role-enrichment-v01/team-review-packet.json)으로 남겼으며 독립 사람 평가를 수행했다고 보고하지 않았다.
- [공통 대조군 비교 보고서](share/shared-verbosity-v02/환자%20페르소나%20공통%20대조군%20비교%20보고서.docx): 새 허구 사례 4개·질문 대상 2개·seed 2개·지시 조건 5개의 첫 응답 80개. 공통 무지시 대조군을 두고 문장 상한을 제거했다. 평균 단어 수는 무지시 28.25 / 단순 verbose 127.19 / PATIENT-Ψ verbose 238.56 / PatientSim verbose 309.69 / PatientSim concise 18.56이었다. 요청 정보 표식은 순서대로 59/64, 64/64, 62/64, 64/64, 60/64였다. 두 소스의 verbose는 단순 지시보다 더 길지만 제공 정보 언급은 늘지 않았다. 21개 원문 점검에서 입력 밖 역할·환경 구체화 사례도 확인했으며 전체 오류율로 집계하지 않았다. [측정값](analysis/shared-verbosity-v02/response-metrics.json), [전체 원문](analysis/shared-verbosity-v02/transcripts.md), [해석](analysis/shared-verbosity-v02/interpretation.ko.md). 오프라인 재집계 `python3 scripts/shared_verbosity.py analyze`, 완성 묶음 검증 `python3 scripts/verify_shared_verbosity.py --report`. 재생성할 필요가 없다. 동일 Qwen 위의 추출 지시 비교이며 원형 앱 순위가 아니다.
- [복귀 금지 절 수정 비교 보고서](share/return-control-ablation-v01/복귀%20금지%20지시%20수정%20비교%20보고서.docx): 새 자유응답 32개와 별도 고정 전환 이어쓰기 32개를 생성 전에 고정하고 원형·단일 절 수정을 비교했다. 자유응답의 다른 사건 표식은 5/16→1/16, 복귀 표식은 1/16→0/16이었다. 이어쓰기의 원래 사건 표식은 두 조건 모두 16/16이었다. 접두문 복사를 제외한 공통 10대응쌍도 양쪽 모두 재언급했다. 수정안을 개선으로 채택하지 않았다. [결론](analysis/return-control-ablation-v01/interpretation.ko.md), [결과 그림](share/return-control-ablation-v01/return-control-results.png). 재집계는 `python3 scripts/return_control_ablation.py --analyze-only`, 완성 묶음 검증은 `python3 scripts/verify_return_control_ablation.py`.
- [문장별 주제 흐름 분석 보고서](share/topic-trajectory-v01/페르소나%20주제%20흐름%20분석%20보고서.docx): 기존 48답변을 411문장으로 분해했다. 주제 이탈 지시에서 다른 제공 사건 언급 2/16, 그 뒤 원래 사건 재언급 1/16을 원문과 연결했다. [결과 그림](share/topic-trajectory-v01/topic-trajectory-results.png), [보조 분류와 원문 오류 점검](analysis/context-topic-labeler-v01/report.ko.md). 로컬 14B 보조 분류는 작성 대조 기준을 통과했지만 실제 답변 3개·4문장에서 참조 경고가 확인되어 그 의미 분류율은 주 결과로 채택하지 않았다. 새 환자 발화 생성 없이 로컬 분류 60회를 수행했다. 오프라인 검증은 `python3 scripts/verify_topic_trajectory.py`.
- [두 특징 통합 보고서 DOCX](share/two-feature-audit-v01/페르소나%20특징%20구현%20분석%20통합%20보고서.docx): 경계적 말투와 주제 이탈 지시의 결과·한계·남은 검증을 3쪽으로 정리했다. 기존 보고서는 덮어쓰지 않았다. [이번 주제 선택 실험의 상세 결과](analysis/topic-selection-probe-v01/report.ko.md): 기존 144개 원문 재검증 후 48개 첫 답변을 새로 생성했다. 세 말투 모두 질문한 사건의 표식은 48/48, 다른 사건은 plain/verbose 0/48, tangent 5/48(2/16응답)이었다. 사전 기술적 결정선은 충족하지 못했다. 프로필 밖 이야기와 표식으로 포착되지 않는 복귀 표현이 있어 표식 0을 임상적 주제 이탈 부재로 해석하지 않는다. 재집계 `python3 scripts/topic_selection_probe.py --analyze-only`, 보고 검증 `python3 scripts/verify_two_feature_checkpoint.py`.
- [새 사례·질문 표현에서 순서 효과 전이 검증](analysis/disclosure-transfer-v01/report.ko.md): 미사용 허구 기록 2개×표현 2종×새 seed 2개×순서 2종×말투 2종, 32개 첫 답변. 처음부터 완전 교차해 간결 대조군과 비교했다. 경계적 말투의 개인 정보 표현 식별은 행사 우선 22/24, 개인 우선 9/24였고 사전 기술적 결정선을 충족했다. 단위 환산 표현을 인정한 사후 점검에서도 방향은 유지됐다. 일부 답변은 정보 회피보다 근사화·불확실한 표현에 해당하므로 누락률을 경계심 점수로 해석하지 않는다. 동일 기반 모델·템플릿 내 제한적 검증이다. 재현: `python3 scripts/disclosure_transfer.py --analyze-only`.
- [정보 공개와 질문 순서: 이번 단계의 통합 결과](analysis/fact-disclosure-order-followup-v01/report.ko.md): 초기 18개+탐색 후속 6개, 실제 새 답변 24개. 간결 조건은 정보를 모두 언급했지만, reserved에서는 두 질문의 순서에 따라 행사·개인 정보가 함께 언급되거나 생략됐다. 순서와 seed를 교차한 reserved 12응답에서 행사 우선은 개인 17/18·행사 18/18, 개인 우선은 두 영역 모두 0/18 식별. 금액 오류 한 건은 정보 회피와 구분했다. 임상적 경계심·원형 시스템 실패의 판정이 아니다. 초기 실험과 사후 후속 설계를 구분해 보존했다. 재현: `python3 scripts/disclosure_order_followup.py --analyze-only`.
- [경계적 말투와 문장 제한 분리 실험](analysis/reserved-length-factorial-v01/report.ko.md): 실제 첫 답변 24개, 같은 인물의 3상황×2seed×2말투×2상한. 공통 최대 5문장 아래에서 reserved 지시 묶음은 평균 −24.17단어, 말투 설정 없이 상한을 3문장으로 강화하면 −19.83단어였다. 경계심의 임상적 구현을 입증하지 않는다. 별도 질문 문단 1건을 표시하고 전체 블록 제외의 사후 민감도 결과도 보존했다. 재집계: `python3 scripts/reserved_length_factorial.py --analyze-only`.
- [설정–발화 연결 감사](analysis/feature-audit-v01/report.ko.md): 두 시스템의 설계 특징 18항목, PSI 6스타일의 원문 지시·관찰 수치·고정 규칙으로 고른 실제 답변을 연결했다. 기존 144응답과 교차 소스 72응답을 분리해 재검산했으며 새 생성은 없다. 발화량 변화와 불신·동의·주제 이탈의 구현 여부를 구분한다. 재현: `python3 scripts/build_feature_audit.py`.
- [교차 소스 말투 모듈 파일럿 실제 결과](analysis/framework-pilot-v01/framework-pilot-20261030/report.ko.md): 개발 3인물·18대화·72응답 완료. verbose−plain 단어 수 차이: 단순 지시 +25.5, PATIENT-Ψ 모듈 +17.0, PatientSim 모듈 +51.0. 문장 수 상한 집중 때문에 평가셋 자동 확대는 보류했다. 전체 원형 시스템의 임상 비교가 아니다.
- [공통 feature·교차 소스 비교 준비](analysis/comparison-preparation-v01/report.ko.md): 기존 17개 정의 고정, 단순 대조군/PATIENT-Ψ/PatientSim의 plain–verbose 지시 모듈 비교 조건 준비. 첫 파일럿 18대화·72응답 예정이며 아직 생성하지 않았다. 전체 원형 프레임워크 비교가 아니다.
- [10월 30일 단일 필드 연결 실험](analysis/thought-alias-v01/thought-alias-20261030/report.ko.md): 자동사고 연결만 변경하고 원형/수정 조건을 새로 생성한 24대화·96응답. 첫 턴 문구 유사도 변화는 plain +0.051, tangent -0.022로 일괄적인 증가가 아니었다.
- [10월 30일 강건성·프롬프트 대응 후속 분석](analysis/feature-robustness-v01/native-features-20261030/report.ko.md): 기존 144응답을 길이·상황·턴·분절 방식별로 재분석하고 원문 행동 요구 19개와 측정 범위를 대조했다. 새 생성·수동 채점 없음.
- [10월 30일 자동 feature 정량화 결과](analysis/automatic-features-v01/native-features-20261030/report.ko.md): 원형 6스타일·36대화·144응답에서 17개 수치 추출. 수동 S/L/G·LLM judge 없음.
- [스타일별 17개 feature 평균 CSV](analysis/automatic-features-v01/native-features-20261030/style-means.csv), [응답별 feature 행렬 CSV](analysis/automatic-features-v01/native-features-20261030/features-flat.csv).
- [10월 29일 팀 공유 프로토타입](share/team-prototype-20261029/team-report.ko.md): 실제 6대화·24응답의 한국어 결과와 후속 연구 인사이트.
- [Plus·로컬 우선 운영](docs/local-workflow.ko.md): 어시스턴트/로컬 역할 분담, 현재 기기와 설치 모델, 과금 경계.
- [연구 프로토콜](docs/protocol-v0.1.md): 연구 질문, 비교 방식, 평가와 중단 기준.
- [첫 조사 결과](docs/first-findings.ko.md): 현재 확인한 사실과 그에 따른 판단.
- [교수님 자문 자료](docs/advisor-brief.ko.md): 임상 개념과 해석에 대한 구체적인 자문 항목.
- [특징 대응표](catalog/feature-map.csv): 설계 특징과 임상 개념 사이의 잠정 대응. 모든 임상 대응은 자문 전이다.
- `evidence/source-lock.json`: 출처의 고정 버전과 로컬 사본의 해시·접근 상태.
- `evidence/static-findings.json`: 실제 소스에서 재계산한 정적 검사 결과.
- `evidence/feature-evidence.json`: 특징 27항목의 출처 위치 검사. 임상 대응의 타당도 검증은 아님.
- `pilot/manifest.jsonl`: 실행 준비된 면담 조건. **환자 발화 또는 실험 결과가 아니다.**
- [관찰 코드북](catalog/clinical-codebook.md): 영어 연구용 정의와 관찰 불가·불확실 항목 처리.
- [검증 기록](docs/verification.json): 실행한 명령과 실제 오프라인 검사 결과.

## 현재 완료와 미완료

단일 필드 실험: `auto_thought → auto_thoughts` 연결 하나만 바꾸고 두 스타일·세 상황·두 반복의 두 조건을 같은 로컬 모델에서 새로 실행했다. 96/96응답이 완료됐고 기존 17개+사전 지정 보조 3개 feature를 추출했다. 첫 턴과 전체 대화 효과를 구분했다. 스타일 간 인접 유사도 격차가 줄어든 것은 tangent의 상승보다 plain의 하락에서 비롯됐으므로 개선으로 해석하지 않았다. 재계산은 `python3 scripts/extract_automatic_features.py --run-id thought-alias-20261030` 다음 `python3 scripts/analyze_thought_alias_ablation.py --cached-only`이다. 결과·원문·검증 묶음은 `analysis/thought-alias-v01/thought-alias-20261030/`에 있다. 로컬 전체 48개 테스트, 묶음 내 선택 18개 테스트와 오프라인 수치 재현을 확인했다.

후속 분석: tangent의 낮은 인접 문장 유사도는 세 상황과 대체 분절에서도 유지됐지만, 질문 유사도의 문장별 기울기 차이는 거의 없었다. 단어 수·문장 수 동시 일치는 24대응쌍 중 4개뿐이어서 길이와 독립적인 효과를 확정하지 않았다. 원문 행동 요구 19개 중 7개는 부분 대리값만 있고 12개는 직접 측정하지 못한다. 이는 모델 실패율이 아니라 측정 범위 감사다. `python3 scripts/analyze_feature_robustness.py --cached-only`로 새 생성·Swift·네트워크 없이 재계산할 수 있다. 후속 통합 묶음과 보고서는 `analysis/feature-robustness-v01/native-features-20261030/`에 있다. 40개 테스트와 추출 묶음의 오프라인 수치 재현을 확인했다.

2026-10-30: 기존 모델 발화의 feature 정량화를 우선하도록 경로를 수정했다. 원형 demo mapping을 유지한 6스타일×3상황×2반복의 새 자료 144응답을 생성했고, 17개 자동 feature와 조건별 프로파일을 계산했다. 한 예제 인물이며 여러 환자 또는 여러 프레임워크의 임상 비교가 아니다. 수동 채점은 보류·제외했다. 현재 자동 정의는 `config/automatic-features-v01.json`에 있다.

재집계는 `python3 scripts/extract_automatic_features.py --run-id native-features-20261030`, 프로파일 보조 분석은 `python3 scripts/analyze_feature_profiles.py --run-id native-features-20261030`이다. 캐시된 벡터를 쓰므로 새 생성이나 유료 API가 필요 없다. 결과·코드·원문 묶음은 분석 폴더의 `automatic-feature-results-20261030.zip`이다.

소스 수집, 정적 구현 검사, 임상 개념 대응 초안, PATIENT-Ψ 프롬프트 이식 및 수정 조건 준비를 수행한다. 원본 첨부 문서는 변경하지 않는다. 저장소의 외부 코드를 실행하지 않고, 허용 목록에 있는 공개 파일만 읽는다.

**2026-10-29: 로컬 Qwen3-14B에서 실제 6대화·24응답을 생성했다. 임상가 평가는 수행하지 않았다.** `pilot/manifest.jsonl`은 여전히 36개 준비 조건이며 생성 결과 자체가 아니다. 실제 관측은 `pilot/runs/team-prototype-20261029/`에 있다. API 키는 코드·로그·이 대화에 넣지 않는다.

추가 API 구매는 기본 경로에서 제외했다. 판단은 어시스턴트가, 반복 처리는 로컬 코드가 수행한다. Qwen3-14B는 이번 개발용 프로토타입의 기반 모델이며, 최종 연구 전체의 기반 모델로 확정한 것은 아니다. `python3 scripts/local_preflight.py`는 기기·모델 메타데이터만 확인하며 추론이나 다운로드를 하지 않는다.

실제 결과의 로컬 재집계: `python3 scripts/summarize_local_pilot.py --run-id team-prototype-20261029`. 팀 공유 묶음 검증: `python3 scripts/package_prototype.py --run-id team-prototype-20261029`. 두 명령 모두 새 추론을 하지 않는다.

## 재현 명령

아래 명령은 이 폴더에서 실행한다. `python3`는 Python 3.10 이상을 뜻한다. 초기 수집에는 네트워크가 필요하다. PDF 추출에는 `pdftotext`가 있으면 이를 사용하고, 없으면 `pypdf`를 사용한다. 나머지는 표준 라이브러리로 동작한다.

```bash
python3 scripts/fetch_sources.py
python3 scripts/audit_sources.py
python3 scripts/prepare_pilot.py
python3 -m unittest discover -s tests -v
```

초기 수집 이후에는 `python3 scripts/verify_offline.py` 하나로 소스 감사·조건 재생성·테스트를 네트워크와 LLM 호출 없이 실행하고 검증 기록을 갱신할 수 있다. 조건 파일을 다시 생성하므로 연구자가 수동으로 바꾼 실험 조건은 별도 버전에 보존한 뒤 실행한다. 실제 응답 파일을 만들거나 덮어쓰지는 않는다.

이 작업공간에서 확인한 bundled Python:

```text
/Users/ohs/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
```

초기 수집 뒤 `source-lock.json`을 보존한다. 이후 수집은 기존 해시와 달라진 본문을 조용히 대체하지 않는다. 출처 실패·버전 변화가 있으면 기록하고 해당 출처를 다시 검토한다.

## 실험의 두 경계

1. 원형 분석과 조현병 관련 확장 실험을 구분한다. PATIENT-Ψ는 CBT, PSYCHE는 정신과 면담, PatientSim은 응급실 상담을 목적으로 하므로 전체 순위를 매기지 않는다.
2. 기준·문헌과의 정합성과 실제 환자자료에 대한 외적 타당도는 다르다. 본 연구는 후자를 입증하지 않는다.

## 출처와 권리

PATIENT-Ψ 코드의 MIT 라이선스는 `evidence/snapshots/psi_license.txt`에 보존한다. 다른 자료의 접근 가능성을 재배포 권한으로 해석하지 않는다. `evidence/snapshots/`는 로컬 분석용 출처 캐시이며 공개 저장소에 자동 포함하지 않는다. PSYCHE는 현재 고정한 논문 v1을 분석한다. 저자 공식 구현은 확인하지 못했으며 제3자 구현을 저자 원형으로 표시하지 않는다.
