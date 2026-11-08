"""Import supplied corpus without modifying originals; standard-library only."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/team-corpus-20261104'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    entries = []
    for extension, name in [('.csv', 'utterances.csv'), ('.docx', 'generation-description.docx')]:
        candidates = list(args.source.glob('*' + extension))
        assert len(candidates) == 1, candidates
        source = candidates[0]
        target = DEST / name
        raw = source.read_bytes()
        if target.exists():
            assert target.read_bytes() == raw, 'Refusing to replace different input'
        else:
            shutil.copyfile(source, target)
        entries.append({'original_path': str(source), 'file': name,
                        'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with zipfile.ZipFile(DEST / 'generation-description.docx') as z:
        root = ET.fromstring(z.read('word/document.xml'))
    paragraphs = [''.join(t.text or '' for t in p.findall('.//w:t', ns))
                  for p in root.findall('.//w:p', ns)]
    (DEST / 'generation-description.txt').write_text('\n'.join(paragraphs) + '\n')
    manifest = {'acquired_date': '2026-11-04', 'files': entries,
                'source': 'User-supplied local files; not downloaded from teammate GitHub',
                'github_reference': {'repository': 'https://github.com/yyeonseoo/Tobedoc',
                    'tree_sha': '58477f301a5b850b7fe0f8056f18bf9f01a10ea7'},
                'claude_pilot_question_ids': None,
                'pilot_id_status': 'User cannot currently confirm; do not assume Q01/Q02',
                'new_utterance_generation': 0}
    (DEST / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(entries, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
