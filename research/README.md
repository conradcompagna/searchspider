# Evaluation

This package reports the last completed local evaluation rounds dated **27 September 2026**. It preserves their questions, counts and ranking evidence while separating runs that used different plans, selectors or scoring rules. Source artifacts and SHA-256 hashes are listed in [evaluation/provenance.json](evaluation/provenance.json).

The scores report retrieval quality and evidence selection. The production release adds the public interface, authentication and server-side usage controls to this evaluated retrieval pipeline.

## 1. Targeted retrieval: can the system find the designated source?

The benchmark contains **691 English test questions**: 354 for Royal Orders and 337 for Chronicle sentences. Questions were generated from individual source passages and tend to include names. Only the designated target receives credit, so another relevant passage can still count as a miss. The 79 Burmese test questions are excluded because the current application deliberately disables Burmese query retrieval.

The current run uses the original question directly with the shared hybrid engine. It does not call Gemini or use planned query expansions.

| Scoring unit | Questions | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|
| Orders + exact Chronicle sentences | 691 | 74.8% | 89.7% | **92.6%** | **0.814** |
| Orders | 354 | 74.9% | 91.8% | 95.5% | 0.827 |
| Exact Chronicle sentence | 337 | 74.8% | 87.5% | 89.6% | 0.800 |
| Target Chronicle page | 337 | 79.2% | 93.5% | 95.8% | 0.852 |
| Orders + target Chronicle pages | 691 | 77.0% | 92.6% | **95.7%** | **0.839** |

Hit@k measures whether the target appears in the first k results; MRR averages its reciprocal rank. The application ranks Chronicle pages and selects a representative sentence. Page-level scoring therefore captures successful page retrieval even when a different sentence is shown. Both units are reported.

The older hybrid+mMiniLM run achieved Hit@1 **86.4%**, Hit@5 **96.7%**, Hit@10 **98.1%**, MRR **0.909** on the same 691 questions. That configuration led the single-query benchmark. The final architecture decision also incorporates the multi-query evidence-recovery and CPU reranker experiments below. The comparison uses the preserved per-question rank file.

Evidence: [current ranks](evaluation/targeted/ranks.csv), [old/new per-question ranks](evaluation/targeted/rank_comparison.csv), [summary](evaluation/targeted/summary.json), [questions](evaluation/targeted/questions.jsonl).

## 2. Thematic retrieval: are early results useful for broader questions?

The final completed thematic evaluation contains **2,834 graded question-passage pairs**: 2,390 original labels plus 444 returned annotations that fill gaps in the current top ten. All **960 top-ten slots** across 48 questions and two sources have a grade. For the headline comparison, the original eligibility rule preserves **46 Order and 47 Chronicle question-source pairs**.

The relevance labels were produced through model-assisted annotation. The final pass included structural validation and manual review of 48 judgments; one borderline full-versus-partial label was retained unchanged. The published label files record the judgments used for every reported score.

| Metric | Orders (46) | Chronicle representative sentences (47) |
|---|---:|---:|
| **nDCG@5** | **0.781** | **0.758** |
| nDCG@10 | 0.770 | 0.726 |
| **Weighted P@5** | **75.9%** | **74.5%** |
| Weighted P@10 | 72.1% | 66.7% |
| Recall@10 of pooled fully relevant items | 52.9% | 44.9% |
| MRR within top ten | 0.833 | 0.786 |
| Judged@10 | 100% | 100% |

Weighted precision awards 1 to fully relevant, 0.5 to partial and 0 to nonrelevant. Counting full and partial equally instead gives **194/230 (84.3%)** positive top-five Order slots and **197/235 (83.8%)** positive Chronicle slots. These are distinct metrics.

The older best-nDCG mMiniLM configuration had fully judged P@5 of **80.4% / 76.6%**; its best-P@5 variant reached **81.7% / 77.4%**. These P@5 comparisons use the same question set. Older nDCG and recall values are reported separately because their smaller judgment pools define different ideal rankings and recall denominators.

Evidence: [completed summary](evaluation/thematic/summary.json), [per-question metrics](evaluation/thematic/per_question.csv), [all top-ten grades](evaluation/thematic/top10_all_grades.csv), [questions](evaluation/thematic/questions.jsonl), [original labels](evaluation/thematic/grades_existing.jsonl), [added labels](evaluation/thematic/grades_added.jsonl).

## 3. Six-topic treasure hunt: recover known evidence and place it high

The second evaluation starts with independently assembled positive document lists and asks the application to recover them from six broad questions. Five topics cover labour, water management, money and credit, captivity, and diplomacy; the sixth covers royal consecration. There are **2,067 positive question-document pairs**. The same document can count once for each question for which it has a positive label.

The latest six full searches supplied fixed initial candidate pools for the final CPU reranker experiment. The table below is that experiment's **unchanged hybrid baseline**, ordered exactly once over the full initial pool. No cross-encoder, graph retrieval or new Gemini call is included in this cached rank analysis.

| Topic | Known positives | Recovered in pool | First 50 | First 100 | First 250 |
|---|---:|---:|---:|---:|---:|
| Labour | 515 | 491 | 44 | 76 | 149 |
| Water management | 149 | 148 | 42 | 66 | 103 |
| Money and credit | 556 | 530 | 43 | 83 | 181 |
| Captives and displacement | 213 | 202 | 26 | 41 | 75 |
| Diplomacy | 402 | 398 | 41 | 78 | 157 |
| Coronation | 232 | 226 | 38 | 64 | 116 |
| **Total** | **2,067** | **1,995 (96.5%)** | **234/300 (78.0%)** | **408/600 (68.0%)** | **781/1,500 (52.1%)** |

The first 50 matter because they are the initial model reading budget. The full pool retains substantially more known evidence than can be assessed in one pass. Later reading rounds expose more of that evidence at additional model cost.

**Reference construction and scoring:** the five-topic reference package was assembled independently of system rankings through lexical and conceptual discovery, passage inspection and full-document review of **15 documents**. The evaluation counts recovery of the listed positive question-document pairs. Rank-band yield counts reference matches within each reading budget; documents outside the reference lists remain ungraded.

The scoring configuration was selected from **9,527 configurations on the five-topic development set**. The six-topic evaluation adds royal consecration. Each ranking configuration was compared against the same saved query plans and candidate pools.

Evidence: [baseline summary](evaluation/treasure_hunt/summary.json), [all 2,067 reference ranks](evaluation/treasure_hunt/reference_ranks.csv), [rank bands](evaluation/treasure_hunt/layers.csv), [five exact questions](evaluation/treasure_hunt/five_questions.jsonl). The sixth exact question was: "how were burmese kings consecrated?/what procedures and ceremonies did their coronation rituals consist of?"

### Why continued reading replaced recursive query generation

One extra query-planning/retrieval round delivered **153** additional known positives across six 50-document assessment budgets. Reading the existing unread ranking instead would have delivered **173**, for **407 versus 387** known positives after 100 documents per topic. Recursive query generation cost six additional planning calls without winning the aggregate selection comparison. Production therefore reads further down the initial ranking.

The final fixed-order analysis above gives **408**, whereas that historical control gives **407**: the earlier control normalized the unread pool; the final selector normalizes the complete initial pool before excluding read documents. These are different selectors and the recorded values are kept separate. [Original aggregate comparison](evaluation/treasure_hunt/reading_comparison.json).

## 4. Why the application does not use a cross-encoder

The final experiment evaluated **Ettin 32M** on the same six cached pools using CPU ONNX inference, representative-centered windows capped at 512 tokens, several reranking depths and reciprocal-rank blends. Reference labels never entered the model input.

| Configuration | Known positives in first 50/topic | In next 100/topic | First 250/topic |
|---|---:|---:|---:|
| **Hybrid baseline** | **234** | **319** | **781** |
| Best top-50 blend, diagnostic INT8 | 235 | 311 | 781 |
| Best top-50 FP32 blend | 235 | 304 | 779 |
| Pure INT8 reranking of first 250 | 187 | 320 | 781 |
| Pure FP32 reranking of first 500 | 152 | 267 | 644 |

The best blended top-50 result improves by **one document across 300 slots**, while losing matches in the next band. Pure reranking has large losses. FP32 controls show that quantization alone does not explain the result.

On the local four-thread CPU experiment, scoring 250 documents with the tested INT8 export averaged **14.34 seconds per question**; FP32 scoring of 500 averaged **43.99 seconds**. These are local experiment timings, not production latency. Publisher INT8 exports also failed numerical checks against FP32 and varied with batch composition; their rankings are diagnostic, not evidence of a validated deployable model.

The final **agentic multi-query pipeline** uses hybrid ranking. Model selection followed the workload: mMiniLM led the single-query tests, while hybrid ranking retained more evidence across the agentic reading sequence with lower CPU cost than the tested Ettin rerankers.

Evidence: [configuration totals](evaluation/reranker/configurations.csv), [per-question counts](evaluation/reranker/per_question.csv), [timings](evaluation/reranker/timings.csv), [numerical checks](evaluation/reranker/numerical_validation.json).

## Evaluation records

The linked result files contain test questions, document ranks, relevance labels, configuration comparisons and timing measurements. The provenance manifest records the source artifacts for each evaluation round, and the corpus is described in the [root README](../README.md#data-and-models).
