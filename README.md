# MindCraft: 조현병 발화의 의미 이탈 정량화

Reddit 글에서 의미 이탈 지표를 찾고(발견), 같은 지표를 DAIS-C 임상 면담 전사에 그대로 적용해 확인한다(검증). 분석 계획은 결과를 보기 전에 [reports/00_analysis_plan.md](reports/00_analysis_plan.md)로 고정했고, 결과는 [reports/01_results.md](reports/01_results.md)에 있다.

## 폴더

| 경로 | 내용 |
|---|---|
| `config.yaml` | 설정 하나로 관리 |
| `src/` | v2 코드 |
| `reports/` | 분석 계획, 결과, 그림 |
| `data/raw/`, `data/processed/` | 원본과 산출물 (git 제외) |
| `v1/` | 이전 작업 보관 (DAIS-C 단독 분석, LLM 증강, 팀 Colab 노트북) |

## 데이터

- DAIS-C: UK Data Service ReShare 855021 (CC BY-SA 4.0). `data/raw/dais-c/`에 둔다.
- Reddit: Kaggle "Reddit-Based Schizophrenia Detection Dataset" (S. S. Sachidhanand, CC BY-NC 4.0). `data/raw/reddit/`에 둔다. 본문은 제작자가 소문자 변환, 문장부호 제거, NLTK 불용어 제거를 해 둔 상태다.

## 실행 순서

```
python src/prepare_daisc.py    # DAIS-C 질문-답변 단위, 판단 태그 위치  -> data/processed/daisc_units.parquet
python src/prepare_reddit.py   # Reddit 정리, 작성 주체 1차 분류         -> data/processed/reddit_units.parquet
python src/features.py         # 앞 40 내용어 SBERT 지표 (30·60 민감도)  -> features.parquet, chunks.parquet
python src/analyze.py          # 계획서의 분석 1-3, 민감도               -> reports/01_results.md, fig1-3
python src/explore.py          # 탐색 분석 E1-E3 (계획서 변경 기록)     -> reports/02_exploratory.md, fig4
```

## 처리 원칙

- 두 코퍼스에 같은 가공을 한다: 소문자, 문장부호 제거, NLTK 불용어 제거, 더듬기 제거, 진단 직접 단어 제거.
- 길이를 맞춘다: 내용어 40개 이상 단위만 쓰고 앞 40개를 5개씩 8조각으로 나눠 지표를 구한다.
- 누수를 막는다: Reddit은 작성자 단위, DAIS-C는 화자 단위로 집계·검증한다.

## 현재 결과 (2026-10-11)

- Reddit: 길이를 맞추자 주 지표(이탈 기울기)의 그룹 차이가 거의 없다(g −0.05, 95% CI −0.10~0.00). 6개 지표 분류 AUC는 0.54로 우연보다는 높지만 작다. 본문 길이 하나만으로 AUC 0.69가 나와서, 팀 노트북의 AUC 0.78에는 길이 차이가 섞였을 가능성이 크다.
- DAIS-C: 주 지표 g +0.35(95% CI −0.42~1.30)로 Reddit과 방향이 반대다. 공변량을 넣은 혼합효과모형에서는 CL이 더 크다(p = 0.024). Reddit 모델을 그대로 적용한 AUC는 0.39다.
- 연구자 판단 태그(DT, TC) 구간에서 같은 사람 안의 인접 거리 증가는 뚜렷하지 않다(p 0.34, 0.26).
- 판정(계획서 기준): Reddit 발견 미지지, DAIS-C 재현 안 됨.
- 탐색 분석 ([reports/02_exploratory.md](reports/02_exploratory.md)): 주제 군집을 보정해도 Reddit 이탈 기울기 차이는 없다(표준화 계수 0.01). 인접 조각 거리는 주제와 무관하게 조현병 게시판 쪽이 크다(0.23). 그런데 같은 게시판 안에서 본인 글과 가족·보호자 글을 비교하면 본인 글의 인접 거리가 오히려 작아서(g −0.21), 이 차이는 환자의 언어보다 게시판 성격에서 왔을 가능성이 크다.
