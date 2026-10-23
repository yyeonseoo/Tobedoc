# 05b. 판별기 피처 조회: 화상·전화 통화 어휘 (진단용)

재현: `python src/feature_probe.py v2`. 3d와 같은 설정(TF-IDF 1–2gram min_df=2 + 로지스틱 balanced)으로 전체 데이터에 학습. 어휘 크기 10471.
순위 1 = 그 방향(합성 또는 실제)으로 계수가 가장 큰 피처.

| term         | in_vocab   |    coef |   rank_toward_syn |   rank_toward_real |   real_per10k |   syn_per10k |
|:-------------|:-----------|--------:|------------------:|-------------------:|--------------:|-------------:|
| signal       | False      | nan     |               nan |                nan |         0.13  |        0     |
| network      | False      | nan     |               nan |                nan |         0     |        1.216 |
| connection   | True       |  -0.152 |             10006 |                466 |         1.565 |        0     |
| internet     | True       |  -0.08  |              8979 |               1493 |         1.565 |        0     |
| wifi         | False      | nan     |               nan |                nan |         0.13  |        0     |
| line         | True       |   0.221 |               618 |               9854 |         5.999 |        6.08  |
| phone        | True       |   0.104 |              1761 |               8711 |         2.087 |        3.648 |
| call         | True       |   0.149 |              1181 |               9291 |         1.695 |        3.648 |
| video        | True       |   0.154 |              1119 |               9353 |         0.913 |        2.432 |
| zoom         | True       |  -0.084 |              9103 |               1369 |         0.782 |        0     |
| screen       | True       |   0.218 |               635 |               9837 |         0.913 |        2.432 |
| camera       | True       |   0.099 |              1822 |               8650 |         3.521 |        2.432 |
| hear         | True       |   0.039 |              2740 |               7732 |         2.347 |        4.864 |
| crackly      | False      | nan     |               nan |                nan |         0     |        0     |
| patchy       | False      | nan     |               nan |                nan |         0     |        1.216 |
| broke up     | False      | nan     |               nan |                nan |         0     |        1.216 |
| cut out      | False      | nan     |               nan |                nan |         0     |        0     |
| lost you     | False      | nan     |               nan |                nan |         0.13  |        0     |
| can you hear | False      | nan     |               nan |                nan |         0     |        0     |

**상위 20 피처에 든 통화 어휘: 없음.**
