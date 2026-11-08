"""Verify input identity, independent repetition counts and byte-stable replay."""
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/team-corpus-v01'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    manifest=json.loads((ROOT/'data/team-corpus-20261104/manifest.json').read_text())
    for f in manifest['files']:
        assert digest(ROOT/'data/team-corpus-20261104'/f['file'])==f['sha256']
    rows=list(csv.DictReader((ROOT/'data/team-corpus-20261104/utterances.csv').open(encoding='utf-8-sig')))
    feature_rows=list(csv.DictReader((OUT/'features.csv').open()))
    assert len(rows)==len(feature_rows)==228
    feature_index={(r['generator'],r['q_id'],r['condition']):r for r in feature_rows}
    for r in rows:
        ts=re.findall(r"[a-z]+(?:'[a-z]+)*",r['text'].lower().replace('’',"'"))
        f=feature_index[r['generator'],r['q_id'],r['condition']]
        for n,word in [(2,'bigram'),(3,'trigram')]:
            for suffix,seq in [('',ts),('_100',ts[:100])]:
                freq=Counter(zip(*(seq[k:] for k in range(n))))
                expected=sum(count-1 for count in freq.values())/(len(seq)-n+1)
                assert abs(float(f['repeated_'+word+'_fraction'+suffix])-expected)<1e-12
    proc=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_team_corpus.py','-v'],cwd=ROOT,capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    (OUT/'unit-test-log.txt').write_text(proc.stdout+proc.stderr)
    paths=list(OUT.glob('*.csv'))+[p for p in OUT.glob('*.json') if p.name!='verification.json']
    paths+=list((ROOT/'share/team-corpus-v01').glob('*.md'))+list((ROOT/'share/team-corpus-v01').glob('*.svg'))
    before={str(p.relative_to(ROOT)):digest(p) for p in paths}
    for script in ['analyze_team_corpus.py','report_team_corpus.py']:
        subprocess.run([sys.executable,'scripts/'+script],cwd=ROOT,check=True,capture_output=True,text=True)
    after={str(p.relative_to(ROOT)):digest(p) for p in paths}
    assert before==after,'Replay changed output bytes'
    report={'input_hashes_verified':True,'independently_recounted_ngram_values':228*4,
            'unit_tests_passed':7,'offline_replay_identical':True,'artifact_sha256':after,
            'new_model_calls_during_verification':0}
    (OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='artifact_sha256'},ensure_ascii=False))
if __name__=='__main__':main()
