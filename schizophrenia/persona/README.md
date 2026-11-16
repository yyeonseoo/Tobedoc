# 조현병 페르소나 모델

실제 환자 데이터 없이 TLC(Andreasen의 사고·언어·의사소통 척도) 정의만으로, 증상과 정도를 조절할 수 있는 조현병 가상 환자 면담을 만들고 검증한다.

- 입력: 인물(A, B, C) + TLC 증상과 단계(예: 탈선 2) + 면담 질문
- 출력: 그 증상을 가진 인물의 여러 턴 면담 답변
- v1은 TLC 정의로 지시문을 조립해 GPT-4.1에 넣는 방식이다. 다음 단계(v2)는 검증된 면담으로 Qwen을 파인튜닝한다.

이전 작업(실제 데이터 분석, 지표 세트를 만든 과정, v1~v4)은 [`pre-persona` 브랜치](https://github.com/yyeonseoo/Tobedoc/tree/pre-persona/schizophrenia/before_persona)에 있다.

## v1 결과 (2026-11-08)

인물 3명 × 설정 5개(정상, 탈선 2, 지리멸렬 2, 말의 빈곤 2, 혼합) × 반복 2 = 면담 60개, 480턴(GPT-4.1, Claude). 채점은 쓴 모델과 다른 모델이 설정을 모른 채 했다.

- 단일 증상 설정 6개 중 4개가 사전 기준을 모두 통과했다. 나머지 2개(Claude 탈선, 말의 빈곤)는 평균 점수만 1.0에 조금 못 미쳤다(0.96, 0.98).
- 증상은 8턴 내내 유지됐고(1점 이상 턴 92~100%), 정상 면담의 오탐은 0%였다.
- 고칠 점: 탈선이 우회증과 섞여 나온다. 놀람도는 아주 짧은 답에서 잘못 오른다. 약한 증상을 섞은 혼합 설정은 거의 드러나지 않는다. 답이 실제 사람보다 길다.

자세히: [reports/01_persona_v1.md](reports/01_persona_v1.md), 계획: [reports/00_plan.md](reports/00_plan.md)

## 지표 세트

이전 단계에서 TLC 단계별 합성 발화로 검증하고 고정했다. 단계 ρ ≥ 0.4, 블라인드 채점과 ρ ≥ 0.3이 기준(두 생성 모델 모두).

| 유형 | 지표 | 단계 ρ (Claude / GPT) | 채점 ρ (Claude / GPT) |
| --- | --- | --- | --- |
| 탈선 | 질문 유사도 | 0.81 / 0.64 | 0.54 / 0.54 |
| 주제 이탈 | 질문 유사도 | 0.87 / 0.73 | 0.62 / 0.51 |
| 지리멸렬 | 놀람도 (GPT-2) | 0.91 / 0.85 | 0.54 / 0.74 |
| 보속 | 어휘 다양도 | 0.93 / 0.80 | 0.38 / 0.61 |
| 음향 연상 | 운율 | 0.92 / 0.83 | 0.63 / 0.67 |
| 말의 빈곤 | 단어 수 | 0.97 / 0.72 | 0.59 / 0.56 |
| 내용 빈곤 | 정보 밀도 | 0.87 / 0.62 | 0.27 / 0.32 (실패) |

## 파일

| 경로 | 내용 |
| --- | --- |
| `src/tlc.py` | TLC 정의와 단계 문구, 채점표, GPT 채점, 비용 상한 |
| `src/metrics.py` | 지표 세트 계산, 계산값 저장, 정상 기준 z 점수 |
| `src/persona_model.py` | 페르소나 모델 (면담 생성, 대화 모드) |
| `src/evaluate.py` | 교차 채점 준비, 지표, 판정, 보고서 |
| `reports/` | 계획(00)과 결과(01) |
| `data/` | 생성 면담, 채점, 계산값 (git 제외) |

## 실행

`schizophrenia/persona`에서 실행한다. GPT를 쓰는 명령은 저장소 루트 `.env`의 `OPENAI_API_KEY`가 필요하고, 한 번에 2.5달러를 넘으면 멈춘다.

```
python src/persona_model.py --chat A DER=2  # 직접 질문 (설정 이름도 됨: --chat B MIX, 빈 줄 = 종료)
python src/persona_model.py                 # GPT-4.1 면담 생성                -> data/gen_gpt.json
python src/persona_model.py --claude-tasks  # Claude 에이전트가 쓸 지시문      -> data/tasks/
#   (Claude 에이전트가 같은 지시문으로 면담 작성                           -> data/gen_claude_*.json)
python src/evaluate.py --rate               # GPT-4.1이 Claude 면담 채점 + 블라인드 파일
#   (data/blind_*.jsonl을 설정 모르는 Claude 에이전트가 채점              -> data/judge_claude_on_gpt_*.csv)
python src/evaluate.py                      # 지표, 판정                       -> reports/01_persona_v1.md
```

증상 코드: DER 탈선, TAN 주제 이탈, INC 지리멸렬, PER 보속, CLANG 음향 연상, POV 말의 빈곤. 단계 1~3.
