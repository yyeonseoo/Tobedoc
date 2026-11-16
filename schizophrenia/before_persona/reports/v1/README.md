# v1 (2026-10-01~02, 종료)

DAIS-C만으로 한 첫 분석과 LLM 합성 증강 시도. 결론은 [SUMMARY_2026-10-02.md](SUMMARY_2026-10-02.md), 증강 경과는 [augmentation/REPORT.md](augmentation/REPORT.md).

- 증강은 세 번 모두 판별 기준(AUC ≤ 0.85)에 걸려 쓰지 않았다. 이후 분석은 증강 없이 질문-답변 단위로 했다.
- v1 코드는 경로 기준이 바뀌어 지웠다. 다시 돌려야 하면 커밋 `ac4ccf3`을 체크아웃한다.
- 원문 발췌가 들어간 증강 리포트와 합성 텍스트는 git에서 뺐다(로컬 `data/v1/`, `reports/v1/augmentation/` 일부).
- 팀 Colab 노트북은 [notebooks/](../../notebooks/)로 옮겼다.

v2에서 바꾼 점: Reddit 본문은 데이터셋 제작자가 NLTK 불용어를 지운 상태라 두 코퍼스에 같은 가공을 한다. 길이는 내용어 수로 맞춘다. 노트북의 DAIS-C 원천(`Speaker Only_Raw`)은 2명이 원본과 맞지 않아 태그 포함 파일을 쓴다.
