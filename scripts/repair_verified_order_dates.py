"""Repair two PDF-verified bundled Orders, retaining existing IDs and cached vectors.

Only changed original records are journalled; no database/model/index backup.
Default prepares a reviewable migration. --apply installs that prepared migration.
"""
import argparse
import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from build_orders import english_passages, parse_date_line, PART_YEARS
from corpus import INDEX, PROC
from embed import Embedder

OUT = ROOT / 'eval/date_parser_20260927'
FIXES = [('rob9_0224', '2.July 1867', 'rob9_0224_18670702', [137, 138, 139, 140], [140, 141]),
         ('rob9_0520', '25 July 1898', 'rob9_0520_18980725', [323], [323, 324])]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    paths = [PROC / 'orders.jsonl', INDEX / 'order_en.npy', INDEX / 'manifest.json', PROC / 'orders_stats.json']
    if args.apply:
        report = json.loads((OUT / 'migration.json').read_text())
        for path in paths:
            assert digest(path) == report['before_sha256'][str(path.relative_to(ROOT))], f'Changed since preparation: {path}'
        for path in paths:
            staged = path.with_name(path.name + '.date_repair')
            assert digest(staged) == report['after_sha256'][str(path.relative_to(ROOT))]
        for path in paths:
            os.replace(path.with_name(path.name + '.date_repair'), path)
        (OUT / 'applied.json').write_text(json.dumps(report['after_sha256'], indent=2))
        print('Installed verified date repairs and corresponding English vectors.')
        return

    records = [json.loads(line) for line in paths[0].read_text(encoding='utf-8').splitlines()]
    assert not any(r['id'] == FIXES[0][2] for r in records), 'Already repaired'
    before = deepcopy(records)
    changed, mapping = [], {}
    for oid, heading, child_id, parent_pages, child_pages in FIXES:
        original = next(r for r in records if r['id'] == oid)
        changed.append(deepcopy(original))
        assert original['en'].count(heading) == 1
        left, right = original['en'].split(heading, 1)
        child = deepcopy(original)
        date = parse_date_line(heading, original['year'], PART_YEARS[original['part']])
        assert date is not None
        original.update(en=left.strip(), en_passages=english_passages(left.strip()), pdf_pages=parent_pages)
        child.update(date, id=child_id, en=right.strip(), en_passages=english_passages(right.strip()),
                     pdf_pages=child_pages, topics=[], my='', my_passages=[], my_alignment=None,
                     split_from=oid)
        assert original['en'] + ' ' + heading + ' ' + child['en'] == changed[-1]['en']
        records.append(child)
        mapping[oid] = [oid, child_id]
    assert [r['id'] for r in records[:len(before)]] == [r['id'] for r in before]
    unaffected = {r['id']: r for r in before if r['id'] not in mapping}
    assert all(r == unaffected[r['id']] for r in records[:len(before)] if r['id'] in unaffected)
    # All untouched embeddings are copied exactly; only genuinely changed passage
    # texts are embedded locally. Burmese/Chronicle indexes and models are untouched.
    old_texts = [p for r in before for p in r['en_passages']]
    old_vec = np.load(paths[1])
    assert len(old_texts) == len(old_vec)
    reuse = dict(zip(old_texts, old_vec))
    texts = [p for r in records for p in r['en_passages']]
    missing = list(dict.fromkeys(p for p in texts if p not in reuse))
    if missing:
        reuse.update(zip(missing, Embedder().passages(missing).astype(old_vec.dtype)))
    vec = np.asarray([reuse[p] for p in texts], dtype=old_vec.dtype)
    (OUT / 'changed_records_before.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in changed), encoding='utf-8')
    staged = [p.with_name(p.name + '.date_repair') for p in paths]
    staged[0].write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    with staged[1].open('wb') as f:
        np.save(f, vec)
    manifest = json.loads(paths[2].read_text())
    manifest['groups']['order_en'].update(rows=len(texts), text_sha256_16=hashlib.sha256('\n'.join(texts).encode()).hexdigest()[:16])
    staged[2].write_text(json.dumps(manifest, indent=1))
    stats = json.loads(paths[3].read_text())
    stats['parts']['9']['orders'] += len(FIXES)
    stats['date_repairs'] = mapping
    staged[3].write_text(json.dumps(stats, indent=1))
    report = dict(old_orders=len(before), new_orders=len(records), changed_existing_records=len(changed),
                  old_to_new=mapping, vectors_reused=len(texts)-sum(p in missing for p in texts),
                  unique_passages_embedded=len(missing), total_passages=len(texts),
                  before_sha256={str(p.relative_to(ROOT)): digest(p) for p in paths},
                  after_sha256={str(p.relative_to(ROOT)): digest(s) for p, s in zip(paths, staged)},
                  previous_manifest=json.loads(paths[2].read_text()), previous_stats=json.loads(paths[3].read_text()))
    (OUT / 'migration.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
