"""Shared, visible retrieval settings. Regular search and Spider use the same engine."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SearchSettings:
    channel_depth: int = 250
    rrf_k: int = 60
    bm25_k1: float = 0.6
    bm25_b: float = 0.75
    dense_weight: float = 0.4
    rank_weight: float = 0.5
    query_top_fraction: float = 0.6
    candidate_bm25_b: float = 0.3
    representative_dense_weight: float = 0.25
    short_chunk_words: int = 30
    aliases_per_entity: int = 10
    gemini_shortlist: int = 50
    gemini_batch: int = 25
    reading_rounds: int = 1
    max_reading_rounds: int = 4
    response_tokens: int = 1000
    min_response_tokens: int = 500
    max_response_tokens: int = 2500
    response_min_fraction: float = 0.5
    followup_tokens: int = 500
    deeper_tranche: int = 50
    deeper_summary_tokens: int = 500
    cached_runs: int = 20
    excerpt_chars: int = 1000
    results_per_page: int = 20

    def public(self):
        return {**asdict(self), "search_language": "English", "stemmer": "Snowball English (Porter2)",
                "stopwords": "Apache Lucene EnglishAnalyzer",
                "candidate_channels": "English BM25 + dense, with recorded name aliases",
                "reading_scope": "next 50 unread documents per round in the fixed initial source-normalized ranking",
                "push_deeper": "next 50 unread documents in the same ranking per click; earlier verdicts and the answer are unchanged",
                "follow_up": "answered from the cached run's relevant/partial summaries and answer; no new search",
                "ranking": "mean of each document's strongest 60% of unique query scores; 60% keyword + 40% dense",
                "keyword_scoring_unit": "whole Order or Chronicle page; dense uses strongest passage",
                "aliases": "candidate gathering only; scoring uses the unchanged query",
                "representative": "unchanged passage selector: mean 75% keyword / 25% dense reciprocal ranks across all queries",
                "ranking_normalization": "50% min-max raw score + 50% (k + 1) / (k + rank), separately per source and query",
                "spider_source_normalization": "min-max separately per source before combined shortlists"}


SETTINGS = SearchSettings()


def response_budget(reading_rounds):
    return 500 * (1 + reading_rounds)
