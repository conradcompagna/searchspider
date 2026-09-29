# Architecture and retrieval decisions

SearchSpider separates local retrieval from model-assisted research. Both modes use the same `Engine.retrieval` and `Engine.consensus` methods; the agent does not maintain a second search implementation.

## Local candidate retrieval

The data adapters convert Royal Orders into dated documents with English/Burmese passages and the Chronicle into sentences linked to owner pages. Runtime retrieval indexes English passages. The ONNX embedder uses multilingual-e5-small with `query:` / `passage:` prefixes, mean pooling and L2 normalization; stored passage vectors avoid re-embedding the collection on requests.

For each original or planned query, the engine gathers up to **250 documents per source per channel** from dense similarity, unexpanded BM25 and alias-expanded BM25 where distinct. Chronicle sentences collapse to pages before applying the depth. It retains the union across channels and queries. Spelling aliases widen gathering but do not add ranking votes.

## Document ranking and excerpt selection

Ranking uses whole-document BM25 with `k1=0.6`, `b=0.75`; semantic scoring uses the strongest passage in the document. Each channel is normalized separately by source and query over the allowed corpus:

```
channel strength = 0.5 * minmax(raw score) + 0.5 * 61 / (60 + rank)
query score = 0.6 * keyword strength + 0.4 * dense strength
document score = mean of the strongest ceil(0.6 * unique-query-count) query scores
```

Missing keyword evidence contributes zero. Duplicate queries and multiple discovery routes do not add votes. The original question remains in the query set. The engine returns every gathered document, including candidates without keyword evidence, in separate Order and Chronicle-page rankings.

Representative passage selection is intentionally independent: the tested selector averages passage-level 75% keyword / 25% dense reciprocal-rank strengths over all queries. That passage anchors excerpts, highlights and the model's reading context even when a whole document supplied its ranking score.

## Agent orchestration

`backend/dossier.py` owns the research sequence:

1. Ask Gemini for distinct English query aspects with historical vocabulary and balanced topic coverage.
2. Run the original question and planned queries through the local engine.
3. Normalize the two source rankings for a shared reading budget and assess the first 50 documents in batches of 25, using a bounded context around each representative.
4. For additional reading, assess the next unseen documents in the **same fixed initial ranking**. Normalize the complete pool before removing previously seen items. Retry missing verdicts once.
5. Synthesize from accumulated relevant/partial assessment summaries, with document citations and an output-token budget.

The regular setting allows 0–4 additional reading rounds, initially 50 plus up to 50 documents per round. With complete batches and no retries, there are 4 Gemini calls with no extra round, 6 with one and 12 with four. The server derives the answer budget as `500 * (1 + reading_rounds)` tokens, up to 2,500; clients cannot set a separate response-length slider. Free hosted initial searches are fixed at 50 assessed documents and a 500-token answer. Follow-up questions and user-requested deeper reading have their own calls and access checks. [Hosted access and budgeting](ACCESS.md) describes the shared sponsored allowance and persistent Language Engine account connections.

The answer model synthesizes the assessment summaries. Each citation links the answer to a specific source document, preserving the reader's route from a claim to its supporting evidence.

## Features removed after measurement

| Experiment | Observation | Runtime decision |
|---|---|---|
| Recursive query generation after first assessment | 153 new known positives versus 173 from reading the existing unread ranking across six 50-document budgets | Continue reading the initial ranking |
| Graph candidate gathering | 323/328 graph matches already appeared in text retrieval; none of five unique additions reached top 50 across nine cached runs | Keep spelling aliases; do not load/traverse graph edges in application search |
| Final Ettin cross-encoder trials | Poor standalone ranking, marginal blended gains, extra CPU cost and failed INT8 numerical checks | Keep hybrid ranking without cross-encoder |

The [evaluation record](../research/EVALUATION.md) compares single-query and agentic workloads, including the older mMiniLM results and final Ettin trials.

## Porting to another document collection

The reusable parts are sparse/dense gathering, document aggregation, normalization, structured model calls, evidence budgets, citation identifiers, API/UI flow and ranking evaluation. The collection-specific parts are the corpus adapters, source identifiers, spelling tables, document filters and question/reference sets.

For a new domain, start with stable document IDs and traceable passages, regenerate embeddings, adapt planning vocabulary, and evaluate against the collection's own questions and relevance judgments. The retrieval engine, agent orchestration, cost controls and evaluation workflow carry over to the new collection.

## Shared site navigation

The live Neural Reader, Chronicle Reader, Knowledge Graph, Graph API and SearchSpider use the same navigation styles and dropdown interactions in [`site-navigation.css`](../frontend/src/site-navigation.css) and [`site-navigation.js`](../frontend/src/site-navigation.js). SearchSpider bundles these files; the existing reader templates load copies of the same files from their shared static directory. A stable scrollbar gutter keeps link positions consistent between scrolling documents and fixed reader panels. The dropdown supports keyboard activation and Escape, and fits narrow mobile viewports.
