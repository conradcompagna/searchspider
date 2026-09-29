"""Evaluate original-question ranking against full planned-query consensus."""
import json
from pathlib import Path

from dossier import Dossier
from search import source_normalized
from search_settings import SETTINGS

PLANS = Path(__file__).resolve().parents[1] / 'eval' / 'thematic' / 'plans.jsonl'


def run(engine, spider, tid, k=SETTINGS.gemini_shortlist):
    plan = next(json.loads(x) for x in PLANS.read_text(encoding='utf-8').splitlines() if json.loads(x)['tid'] == tid)
    d = Dossier(engine, spider, {'q': plan['q'], 'sources': ['order', 'chronicle'], 'words': 0})
    d.queries = plan.get('queries', [])
    d.gather()
    k = max(1, min(SETTINGS.gemini_shortlist, k))
    original = [d.base['all'][key] for key in d.pool]
    variants = {'original_question': [it['key'] for it in source_normalized(original)[:k]]}
    d.score()
    variants['hybrid_consensus'] = [it['key'] for it in source_normalized(d.items())[:k]]
    union = list(dict.fromkeys(key for keys in variants.values() for key in keys))
    selected = [d.pool[key] for key in union]
    d._aliases(d.items())
    for it in selected:
        it['alias'] = d.alias_of[it['key']]
    d.assess(selected, 'Evaluation: original question / query consensus union')
    result = {'tid': tid, 'q': plan['q'], 'union': len(union), 'variants': {},
              'usage_in': sum(u['in'] for u in d.usage), 'usage_out': sum(u['out'] for u in d.usage)}
    for name, keys in variants.items():
        verdicts = [d.verdict.get(key, {}).get('v') for key in keys]
        result['variants'][name] = {v: verdicts.count(v) for v in ('r', 'p', 'n')}
        result['variants'][name]['missing'] = verdicts.count(None)
    return result
