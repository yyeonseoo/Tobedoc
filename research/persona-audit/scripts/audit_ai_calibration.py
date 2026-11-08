"""Audit stored ratings locally; never load the condition key or call a model."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis/main-comparison-v01"
PILOT = BASE / "ai-pilot-v01"
def read(p): return json.loads(p.read_text())

def main():
    source = read(BASE / "blinded-inputs.json")
    assert [r["blind_id"] for r in source] == [f"B{i:02}" for i in range(1,17)]
    rows = sum((read(PILOT / name)["rows"] for name in ("ratings.json", "ratings-remaining.json")), [])
    fields = {"opening_answer": {"direct","partial","absent","uncertain"},
              "unclear_transition": {"yes","no","uncertain","unassessable"},
              "within_clause_difficulty": {"yes","no","uncertain"}}
    for r, s in zip(rows, source, strict=True):
        assert r["blind_id"] == s["blind_id"]
        assert r["text_sha256"] == s["text_sha256"] == sha256(s["text"].encode()).hexdigest()
        assert all(r[k] in v for k,v in fields.items()) and r["rationale"].strip()
        a,b = r["excerpt"],r["next_excerpt"]
        assert a and a in s["text"]
        if r["unclear_transition"] in {"yes","uncertain"}: assert b
        if b: assert a+' '+b in s["text"] or a+b in s["text"]
    human = read(ROOT/"share/main-comparison-v01/annotations.json")
    assert len(human)==16 and all(not v for r in human for k,v in r.items() if k!="blind_id")
    counts = {k:dict(Counter(r[k] for r in rows)) for k in fields}
    print(json.dumps(counts, ensure_ascii=False, indent=2))

if __name__=="__main__": main()
