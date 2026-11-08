"""Validate decomposition against v01, exact class counts, quotes and offline replay."""
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from analyze_team_corpus import tokens

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/inc3-mechanism-v01'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    cfg=json.loads((ROOT/'config/inc3-mechanism-v01.json').read_text())
    prov=json.loads((OUT/'run-provenance.json').read_text())
    source=ROOT/cfg['input']
    assert digest(source)==prov['input_sha256']
    original=list(csv.DictReader(source.open(encoding='utf-8-sig')))
    key=lambda r:(r['generator'],r['q_id'],r['condition'])
    raw={key(r):r['text'] for r in original}
    old={key(r):r for r in csv.DictReader((ROOT/'analysis/team-corpus-v01/features.csv').open())}
    rows=list(csv.DictReader((OUT/'features.csv').open()))
    assert len(rows)==384 and len({key(r) for r in rows})==96
    lexicons={'base':set(cfg['function_word_base'].split()),
        'extended':set((cfg['function_word_base']+' '+cfg['extended_additions']).split())}
    error=0
    for r in rows:
        metric='mattr50' if r['scope']=='full' else 'mattr50_100'
        assert abs(float(r['mattr50'])-float(old[key(r)][metric]))<1e-12
        ts=tokens(raw[key(r)])
        if r['scope']=='first100':ts=ts[:100]
        f=c=0
        lex=lexicons[r['lexicon']]
        for start in range(len(ts)-49):
            counts=Counter(ts[start:start+50])
            f+=sum(n-1 for word,n in counts.items() if word in lex)
            c+=sum(n-1 for word,n in counts.items() if word not in lex)
        den=50*(len(ts)-49)
        assert abs(f/den-float(r['function_repeat_loss']))<1e-12
        assert abs(c/den-float(r['nonlist_repeat_loss']))<1e-12
        error=max(error,abs(1-f/den-c/den-float(r['mattr50'])))
    assert error<1e-12
    contrasts=json.loads((OUT/'contrasts.json').read_text())
    index={tuple(r[k] for k in ['generator','condition','lexicon','scope','metric']):r for r in contrasts}
    for k,r in index.items():
        if k[-1]!='mattr50':continue
        f=index[k[:-1]+('function_repeat_loss',)];c=index[k[:-1]+('nonlist_repeat_loss',)]
        for t,fl,cl in zip(r['question_deltas'],f['question_deltas'],c['question_deltas']):
            assert abs(t+fl+cl)<1e-12
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_inc3_mechanism.py','-v'],cwd=ROOT,capture_output=True,text=True)
    assert tests.returncode==0,tests.stderr
    (OUT/'unit-test-log.txt').write_text(tests.stdout+tests.stderr)
    paths=[p for p in OUT.glob('*.json') if p.name!='verification.json']+list(OUT.glob('*.csv'))+list((ROOT/'share/inc3-mechanism-v01').glob('*.md'))
    before={str(p.relative_to(ROOT)):digest(p) for p in paths}
    for script in ['analyze_inc3_mechanism.py','report_inc3_mechanism.py']:
        subprocess.run([sys.executable,'scripts/'+script],cwd=ROOT,check=True,capture_output=True)
    after={str(p.relative_to(ROOT)):digest(p) for p in paths}
    assert before==after
    audit=json.loads((OUT/'local-review/audit.json').read_text())
    mapping=json.loads((OUT/'local-review/mapping.json').read_text())
    expected={r['row_id'] for r in mapping.values()}
    items=[r for r in audit['items'] if 'row_id' in r]
    assert len(items)==24 and {r['row_id'] for r in items}==expected
    assert all(r['quote_count_valid'] for r in items)
    quotes=[q for r in items for q in r['quotes']]
    for r in items:
        text=raw[tuple(r['row_id'].split(':'))]
        for q in r['quotes']:
            assert q['exact_substring']==(bool(q['text']) and q['text'] in text)
    report={'source_rows':96,'feature_rows':len(rows),'v01_mattr_matches':len(rows),
        'independently_checked_repeat_components':2*len(rows),'maximum_identity_error':error,
        'unit_tests_passed':4,'offline_replay_identical':True,
        'qwen_items':len(items),'quotes':len(quotes),'invalid_quotes':sum(not q['exact_substring'] for q in quotes),
        'clinical_ratings':0,'artifact_sha256':after}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='artifact_sha256'}))
if __name__=='__main__':main()
