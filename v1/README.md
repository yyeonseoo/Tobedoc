# v1 (보관)

2026-10-01~02 작업. 새 분석(v2)은 저장소 루트에서 진행한다.

| 경로 | 내용 |
|---|---|
| `reports/SUMMARY_2026-10-02.md` | v1 전체 정리: LLM 증강 중단, 질문-답변 단위, SBERT 지표와 검정 전 확인 |
| `src/`, `config.yaml`, `reports/` | DAIS-C 단독 분석 코드와 리포트 |
| `augmentation/` | LLM 합성 증강 1~3차 (종료). 결론은 `augmentation/REPORT.md` |
| `colab_v5_1.ipynb` | 팀 Colab 노트북 (Reddit 발견 + DAIS-C 확인). 조회 전용 |
| `data/` | v1 산출물 (git 제외) |

v1 코드는 경로 기준이 바뀌어 이 위치에서는 실행되지 않는다. 다시 돌리려면 커밋 `ac4ccf3`을 체크아웃한다.

v2로 넘어가며 반영한 점: Reddit 본문은 데이터셋 제작자가 NLTK 불용어를 지운 상태라 두 코퍼스에 같은 가공을 적용한다. 길이는 내용어 수로 맞춘다. 노트북의 DAIS-C 원천(`Speaker Only_Raw`)은 2명이 원본과 맞지 않아 태그 포함 파일을 쓴다.
