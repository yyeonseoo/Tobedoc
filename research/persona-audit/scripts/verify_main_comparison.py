"""Offline packet/source replay checks. Never calls a model or fills annotations."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from validate_main_annotations import validate
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/main-comparison-v01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    spec=json.loads((ROOT/'config/main-comparison-v01.json').read_text())
    assert len(spec['conditions'])==6
    assert spec['development']['generation_authorized_now'] is False
    assert spec['confirmation_plan']['generation_authorized_now'] is False
    assert spec['confirmation_plan']['outputs_if_later_released']==4*2*2*6
    rows=json.loads((OUT/'blinded-inputs.json').read_text())
    key=json.loads((OUT/'blinding-key.json').read_text())
    assert len(rows)==16 and len({r['blind_id'] for r in rows})==16
    lookup={r['blind_id']:r for r in key}
    for r in rows:
        assert set(r)=={'blind_id','question','text','text_sha256'}
        assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
        assert r['text']==lookup[r['blind_id']]['text']
    annotation=validate(ROOT/'share/main-comparison-v01/annotations.json')
    assert not annotation['errors'],annotation
    files=[OUT/n for n in ['evidence-audit.json','blinded-inputs.json','blinding-key.json']]
    files += [ROOT/'share/main-comparison-v01'/n for n in ['블라인드16개_검토.md','annotations.json','본실험_한장설계.md','설계근거.md']]
    before={str(p.relative_to(ROOT)):sha(p) for p in files}
    subprocess.run([sys.executable,'scripts/prepare_main_comparison.py'],cwd=ROOT,check=True,capture_output=True)
    after={str(p.relative_to(ROOT)):sha(p) for p in files}
    assert before==after,'Unexpected replay change, including potentially filled user annotation'
    result={'packet_rows':16,'public_fields_hide_generator_and_condition':True,'all_original_text_hashes_match':True,
        'offline_replay_identical':True,'annotation_rows_completed':annotation['completed'],
        'new_generation':0,'new_embeddings':0,'old_holdout_outputs':0,
        'script_sha256':sha(Path(__file__)),'artifact_sha256':after}
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['artifact_sha256','script_sha256']}))
if __name__=='__main__':main()
