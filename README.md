# MindCraft: 조현병 발화의 의미 이탈 정량화 (DAIS-C 파일럿)

영국 영어 면담 코퍼스 DAIS-C(조현병군 15명, 대조군 13명)에서 면담자 질문으로부터 답변이 얼마나 멀어지는지를 SBERT 거리로 측정한다. 전체 경과와 결과는 [reports/SUMMARY_2026-10-02.md](reports/SUMMARY_2026-10-02.md)에 있다.

## 데이터

원본은 저장소에 포함하지 않는다. UK Data Service ReShare 855021(DOI 10.5255/UKDA-SN-855021, CC BY-SA 4.0)에서 받아 `DAIS-C-Annotated-Upload/` 아래에 둔다. 전처리 산출물(`data/`)도 원문을 담고 있어 git에서 제외한다.

## 파이프라인

```
python src/explore.py            # 원본 구조 탐색            -> reports/00_exploration.md
python src/preprocess.py         # speaker-only 턴 코퍼스     -> data/processed/corpus.parquet, reports/01_preprocess.md
python src/build_qa_units.py     # 질문-답변 단위 구축        -> data/processed/qa_units.parquet, reports/02_qa_units.md
python src/position_features.py  # 10단어 조각 지표, 확인     -> qa_chunks/qa_features.parquet, reports/03_features.md, fig5-8
python src/check_aug_impact.py   # 증강 방식별 영향 실측      -> data/processed/aug_impact.parquet
python src/summary_figures.py    # 증강 결과·데이터 개요 그림 -> fig1-4
```

설정은 `config.yaml` 하나로 관리한다.

## 현재 상태 (2026-10-02)

- 데이터 증강은 하지 않는다. LLM 합성은 3회 모두 실제와 쉽게 구분됐고(AUC 0.93 이상), 문장 섞기와 EDA는 이탈 지표를 지우거나 가짜 차이를 만들었다. 종료한 증강 작업은 `archive/augmentation/`에 있다.
- 분석 단위는 면담자 질문 하나에 대한 답변 하나이고, 답변 전체를 10단어 조각으로 나눠 질문과의 거리를 위치별로 구한다. 30단어 이상 답변 520개, 화자 26명.
- 기술통계상 그룹 차이는 대부분 답변 첫 10~20단어에서 생기고 그 뒤로는 뚜렷하게 벌어지지 않는다. 출발점 효과를 뺀 이후 구간의 기울기 차이는 d 0.00이다. 시작 질문 형식, 면담 방식(CL 전화·대면, CO 화상), 성별이 그룹과 겹쳐 있다.
- 그룹 검정과 분류는 아직 하지 않았다. 분석 계획을 고정한 뒤 진행한다.
