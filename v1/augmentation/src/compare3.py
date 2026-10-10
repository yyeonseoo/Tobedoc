"""1차/2차/3차 4축 비교표 (선고정 임계값 판정은 judge.py 함수 그대로) -> stdout (06_validation_v3.md, ATTEMPT_3_SUMMARY에 첨부)."""
import json
import os

import pandas as pd

import judge
from preprocess import ROOT

RUNS = [('1차', 'reports/validation_metrics.json'), ('2차', 'reports/v2_validation_metrics.json'),
        ('3차', 'reports/v3_validation_metrics.json')]
M = {k: json.load(open(os.path.join(ROOT, p), encoding='utf8')) for k, p in RUNS}
V = {k: judge.axis_values(m) for k, m in M.items()}
J = {k: judge.judge(v) for k, v in V.items()}

print('## 1차 / 2차 / 3차 비교\n')
print('### 핵심 지표\n')
eff = {k: pd.DataFrame(m['effect']).set_index('feature') for k, m in M.items()}
real_fd = eff['1차'].at['first_dist', 'd_real']
key = pd.DataFrame({k: [M[k]['discrim']['auc_mean'], eff[k].at['first_dist', 'd_syn'], eff[k].at['first_dist', 'q_syn']] for k in M},
                   index=['Discriminability AUC (기준 ≤0.75 합격, ≤0.85 주의)',
                          f'first_dist d, 합성 CL−CO (실제 {real_fd:.3f}) — topic drift 과장',
                          'first_dist q (BH)'])
print(key.round(3).to_markdown())
print('\n### 4축 선고정 판정\n')
rows = {}
for k in M:
    d = judge.describe(V[k])
    rows[f'{k} 수치'] = d
    rows[f'{k} 판정'] = J[k]
print(pd.DataFrame(rows).to_markdown())
print('\n### 세부\n')
det = {}
for k, m in M.items():
    div = pd.DataFrame(m['diversity']).set_index('group')
    det[k] = {'규모 (CL/CO)': None,
              '방향 일치 (실제 유의 8개 중)': f"{V[k]['fid_match']}/{V[k]['fid_sig']}",
              '가짜 효과': ', '.join(V[k]['fid_spurious']) or '없음',
              'adj_sim d (실제 0.015)': round(eff[k].at['adj_sim', 'd_syn'], 3),
              'pairwise cos CL (실제 0.306)': round(div.at['CL', 'pair_cos_syn_mean'], 3),
              'pairwise cos CO (실제 0.414)': round(div.at['CO', 'pair_cos_syn_mean'], 3),
              'distinct-1 CL (실제 구간)': f"{div.at['CL', 'distinct1_syn']:.3f} ({div.at['CL', 'distinct1_real_95']})",
              'distinct-1 CO (실제 구간)': f"{div.at['CO', 'distinct1_syn']:.3f} ({div.at['CO', 'distinct1_real_95']})",
              '5-gram overlap 비율 (기준선 대비)': round(V[k]['mem_ratio'], 2),
              '최장 공유 구간 (단어)': m['memorization']['synthetic → all real']['longest shared span (words, max)'],
              'AUC CL / CO': f"{m['discrim']['auc_CL']:.3f} / {m['discrim']['auc_CO']:.3f}",
              'AUC 길이만': round(m['discrim']['auc_length_only'], 3)}
sizes = {'1차': '56/56', '2차': '50/50', '3차': '43/53'}
for k in det:
    det[k]['규모 (CL/CO)'] = sizes[k]
print(pd.DataFrame(det).to_markdown())
