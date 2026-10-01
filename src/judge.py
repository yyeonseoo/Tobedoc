"""선고정 임계값(config.yaml thresholds)으로 validation_metrics.json을 판정하고 1차 대비 비교표를 낸다.

  python src/judge.py reports/validation_metrics.json                       # 단일 판정
  python src/judge.py reports/v2_validation_metrics.json reports/validation_metrics.json   # 2차 판정 + 1차 비교
"""
import json
import sys

import pandas as pd

from preprocess import CFG

T = CFG['thresholds']


def axis_values(m):
    eff = pd.DataFrame(m['effect'])
    a = T['fidelity']['alpha']
    sig = eff[eff.q_real < a]
    spurious = eff[(eff.q_real >= a) & (eff.q_syn < a)]
    div = pd.DataFrame(m['diversity'])
    mem = m['memorization']
    return dict(
        fid_sig=len(sig), fid_match=int(sig.direction_match.sum()), fid_spurious=spurious.feature.tolist(),
        div_absdiff=float((div.pair_cos_syn_mean - div.pair_cos_real_mean).abs().max()),
        div_neardup=int(m['near_dup']['syn_pairs']),
        mem_ratio=mem['synthetic → all real']['pooled 5-gram overlap'] / mem['baseline: real → other speakers']['pooled 5-gram overlap'],
        auc=m['discrim']['auc_mean'])


def judge(v):
    f, d, mm, dc = T['fidelity'], T['diversity'], T['memorization'], T['discriminability']
    if f['spurious_effect_fails'] and v['fid_spurious']:
        fid = '불합격'
    else:
        fid = '합격' if v['fid_match'] >= f['pass_min_direction_match'] else '주의' if v['fid_match'] >= f['caution_min_direction_match'] else '불합격'
    if v['div_neardup'] > 0:
        div = '불합격'
    else:
        div = '합격' if v['div_absdiff'] <= d['pass_max_abs_diff'] else '주의' if v['div_absdiff'] <= d['caution_max_abs_diff'] else '불합격'
    mem = '합격' if v['mem_ratio'] <= mm['pass_max_ratio'] else '주의' if v['mem_ratio'] <= mm['caution_max_ratio'] else '불합격'
    dis = '합격' if v['auc'] <= dc['pass_max_auc'] else '주의' if v['auc'] <= dc['caution_max_auc'] else '불합격'
    return dict(Fidelity=fid, Diversity=div, Memorization=mem, Discriminability=dis)


def describe(v):
    return dict(Fidelity=f"방향 일치 {v['fid_match']}/{v['fid_sig']}, 가짜 효과 {len(v['fid_spurious'])}개 {v['fid_spurious']}",
                Diversity=f"|Δ pairwise cos| 최대 {v['div_absdiff']:.3f}, 준중복 {v['div_neardup']}쌍",
                Memorization=f"overlap 비율 {v['mem_ratio']:.2f}×",
                Discriminability=f"AUC {v['auc']:.3f}")


def table(paths):
    cols = {}
    for label, p in zip(['2차', '1차'] if len(paths) == 2 else ['판정 대상'], paths):
        v = axis_values(json.load(open(p, encoding='utf8')))
        j, ds = judge(v), describe(v)
        cols[f'{label} 수치'] = ds
        cols[f'{label} 판정'] = j
    return pd.DataFrame(cols)


if __name__ == '__main__':
    print(table(sys.argv[1:]).to_markdown())
