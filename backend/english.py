"""One English analyzer for documents, queries, aliases and highlighted word forms.

Stop list: Apache Lucene EnglishAnalyzer.ENGLISH_STOP_WORDS_SET, Apache-2.0.
https://github.com/apache/lucene/blob/main/lucene/analysis/common/src/java/org/apache/lucene/analysis/en/EnglishAnalyzer.java
Stemming: https://snowballstem.org/algorithms/english/stemmer.html
No minimum word length, home-made plural folding, or title-word exclusions.
"""
import re
import threading
from functools import lru_cache

import snowballstemmer

EN_STOP = frozenset("""a an and are as at be but by for if in into is it no not of on or such
that the their then there these they this to was will with""".split())
_local = threading.local()
WORD = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)*")


@lru_cache(maxsize=65536)
def stem(word):
    if not hasattr(_local, "stemmer"):
        _local.stemmer = snowballstemmer.stemmer("english")
    return _local.stemmer.stemWord(word.replace("’", "'"))


def words(text):
    return WORD.findall(text.lower())


def en_tokens(text):
    return [stem(w) for w in words(text) if w not in EN_STOP]
