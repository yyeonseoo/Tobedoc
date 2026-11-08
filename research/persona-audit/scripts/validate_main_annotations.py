"""Validate IDs and literal evidence without filling blanks or inventing ratings."""
import argparse
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def validate(path):
    source=json.loads((ROOT/'analysis/main-comparison-v01/blinded-inputs.json').read_text())
    texts={r['blind_id']:r['text'] for r in source}
    rows=json.loads(path.read_text())
    errors=[];completed=0
    ids=[r.get('blind_id') for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(texts):errors.append('Missing, duplicate or unknown blind IDs')
    for r in rows:
        ident=r.get('blind_id');text=texts.get(ident,'')
        for field,allowed in [('opening_answer',['direct','partial','absent','uncertain']),
                              ('transition',['yes','no','uncertain','unassessable']),
                              ('within_clause_unintelligible',['yes','no','uncertain'])]:
            if r.get(field,'') not in ['']+allowed:errors.append(ident+': invalid '+field)
        filled=all(r.get(k) for k in ['opening_answer','transition','within_clause_unintelligible'])
        if filled:completed+=1
        if r.get('transition')=='yes':
            a,b=r.get('before_quote',''),r.get('after_quote','')
            if not a or not b or not r.get('transition_reason','').strip():errors.append(ident+': transition evidence/reason required')
            elif a not in text or b not in text:errors.append(ident+': quote is not a literal substring')
            else:
                # Accept at least one ordered nonoverlapping pair if a phrase occurs repeatedly.
                starts=[i for i in range(len(text)) if text.startswith(a,i)]
                if not any(text.find(b,i+len(a))>=0 for i in starts):errors.append(ident+': quote order/overlap invalid')
        if r.get('within_clause_unintelligible')=='yes':
            q=r.get('within_clause_quote','')
            if not q or q not in text:errors.append(ident+': within-clause evidence required')
    return {'rows':len(rows),'completed':completed,'pending':len(rows)-completed,'errors':errors,
            'clinical_gold':False,'interrater_reliability_available':False}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,default=ROOT/'share/main-comparison-v01/annotations.json')
    args=ap.parse_args();result=validate(args.input)
    print(json.dumps(result,ensure_ascii=False))
    if result['errors']:raise SystemExit(1)
if __name__=='__main__':main()
