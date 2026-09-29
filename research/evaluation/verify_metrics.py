"""Recompute public headline metrics from saved ranks and judgments; standard library only.

This verifies result arithmetic, not semantic judgment quality or a fresh retrieval run.
"""
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def csv_rows(name):
    with (ROOT / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def json_file(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def jsonl(name):
    return [json.loads(line) for line in (ROOT / name).read_text(encoding='utf-8').splitlines() if line.strip()]


def close(actual, expected, label):
    if not math.isclose(actual, expected, abs_tol=1e-10):
        raise AssertionError(f'{label}: calculated {actual}, expected {expected}')


def rank_metrics(ranks):
    ranks = [int(rank) if rank else None for rank in ranks]
    return {**{f'hit@{k}': sum(rank is not None and rank <= k for rank in ranks) / len(ranks)
               for k in (1, 5, 10)},
            'mrr': sum(1 / rank for rank in ranks if rank) / len(ranks), 'n': len(ranks)}


def dcg(grades):
    return sum(grade / math.log2(rank + 2) for rank, grade in enumerate(grades))


def main():
    targeted = [row for row in csv_rows('targeted/ranks.csv')
                if row['config'] == 'app' and row['split'] == 'test' and row['q_lang'] == 'en']
    assert len(targeted) == 691
    target = rank_metrics([row['rank'] for row in targeted])
    for key, value in json_file('targeted/summary.json')['test']['app|all'].items():
        close(target[key], value, f'targeted {key}')
    pages = rank_metrics([row['page_rank'] if row['source'] == 'chronicle' else row['rank'] for row in targeted])
    assert round(pages['hit@10'] * 100, 1) == 95.7 and round(pages['mrr'], 3) == .839
    print(f"Targeted: n=691, Hit@10={target['hit@10']:.3%}, MRR={target['mrr']:.6f}")
    print(f"Document target: Hit@10={pages['hit@10']:.3%}, MRR={pages['mrr']:.6f}")

    labels = {}
    for name in ('thematic/grades_existing.jsonl', 'thematic/grades_added.jsonl'):
        for row in jsonl(name):
            assert row['pid'] not in labels
            labels[row['pid']] = int(row['grade'])
    assert len(labels) == 2834
    top = csv_rows('thematic/top10_all_grades.csv')
    assert len(top) == 960
    assert all(int(row['grade']) == labels[row['pid']] for row in top)
    relevant = defaultdict(list)
    for pid, grade in labels.items():
        tid, unit = pid.split('|', 1)
        relevant[tid, 'order' if unit.startswith('rob') else 'chronicle'].append(grade)
    groups = defaultdict(list)
    for row in top:
        if row['in_original_question_set'] == 'True':
            groups[row['tid'], row['source']].append(row)
    metrics = defaultdict(list)
    for (tid, source), rows in groups.items():
        grades = [int(row['grade']) for row in sorted(rows, key=lambda row: int(row['rank']))]
        assert len(grades) == 10
        ideal = sorted(relevant[tid, source], reverse=True)
        n_rel = ideal.count(2)
        result = {}
        for k in (5, 10):
            result[f'p@{k}'] = sum(grades[:k]) / (2 * k)
            result[f'ndcg@{k}'] = dcg(grades[:k]) / dcg(ideal[:k])
            result[f'recall@{k}'] = grades[:k].count(2) / n_rel
            result[f'judged@{k}'] = 1.0
        result['mrr'] = next((1 / i for i, grade in enumerate(grades, 1) if grade == 2), 0)
        metrics[source].append(result)
    summary = json_file('thematic/summary.json')['fixed_question_set']
    for source, rows in metrics.items():
        assert len(rows) == summary[source]['n']
        for key in rows[0]:
            close(sum(row[key] for row in rows) / len(rows), summary[source][key], f'{source} {key}')
        print(f"Thematic {source}: n={len(rows)}, P@5={summary[source]['p@5']:.3%}, nDCG@5={summary[source]['ndcg@5']:.6f}")

    refs = csv_rows('treasure_hunt/reference_ranks.csv')
    assert len(refs) == 2067 and len({(row['question'], row['document_id']) for row in refs}) == 2067
    recovered = [row for row in refs if row['rank']]
    assert len(recovered) == 1995
    baseline = json_file('treasure_hunt/summary.json')
    for cutoff, expected in baseline['totals'].items():
        count = sum(int(row['rank']) <= int(cutoff) for row in recovered)
        assert count == expected, (cutoff, count, expected)
    for question, saved in baseline['questions'].items():
        rows = [row for row in refs if row['question'] == question]
        assert len(rows) == saved['reference_total']
        assert sum(bool(row['rank']) for row in rows) == saved['pool_known']
    print('Six-topic recovery: 1,995/2,067; top50=234, top100=408, top250=781')

    configs = {row['config']: row for row in csv_rows('reranker/configurations.csv')}
    assert int(configs['baseline']['top50']) == 234
    assert max(int(row['top50']) for row in configs.values()) == 235
    graph = csv_rows('graph/runs.csv')
    assert sum(int(row['graph_documents']) for row in graph) == 328
    assert sum(int(row['graph_already_in_text']) for row in graph) == 323
    assert all(int(row['graph_only_actual_gemini50']) == 0 for row in graph)
    print('Reranker and graph audit totals verified. All checks passed.')


if __name__ == '__main__':
    main()
