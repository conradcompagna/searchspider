# Burmese search is disabled. Kept as commented source for a future restoration.
# The original Burmese text, embeddings and corpus-build support remain on disk.

# MY_WEIGHTS = {"bm25_my": 1.0, "dense_my": 0.001}  # dense only orders what keywords did not match

# def is_burmese(text):
#     """True when most letters in the query are Myanmar script."""
#     my = len(re.findall(r"[\u1000-\u109F]", text))
#     latin = len(re.findall(r"[A-Za-z]", text))
#     return my > latin
#

# # Burmese syllable break: before a consonant/independent vowel that is not stacked (preceded by
# # virama) and not killed (followed by asat or virama).
# _MY_BREAK = re.compile(r"(?<!္)([က-ဪဿ၌-၏])(?![်္])")
#
#
# def my_tokens(text):
#     text = re.sub(r"[၊။()\[\],.;:!?\"'\-]", " ", text)
#     sylls = []
#     for chunk in text.split():
#         if re.search(r"[က-႟]", chunk):
#             sylls.extend(s for s in _MY_BREAK.sub(r" \1", chunk).split() if s)
#         else:
#             sylls.append(chunk.lower())
#     # syllable unigrams + bigrams (bigrams approximate words)
#     return sylls + [a + b for a, b in zip(sylls, sylls[1:])]
#
#

#     def _channels(self, qvec, q_en, q_my, prefix, n_units, allowed, use=("dense", "bm25")):
#         out = {}
#         for lang in ("en", "my"):
#             g = self.groups[f"{prefix}_{lang}"]
#             ok = allowed[g["unit"]]
#             if "dense" in use:
#                 dense = g["vec"] @ qvec
#                 out[f"dense_{lang}"] = self._best(g["unit"], np.where(ok, dense, -np.inf), n_units)
#             toks = q_en if lang == "en" else q_my
#             bm = g["bm25"].scores(toks) if (toks and "bm25" in use) else None
#             if bm is not None:
#                 out[f"bm25_{lang}"] = self._best(g["unit"], np.where(ok & (bm > 0), bm, -np.inf), n_units)
#         if prefix == "order" and "topic" in use and q_en:
#             ts = self.topic_bm25.scores(q_en)
#             if ts is not None:
#                 # many orders share a heading: break ties by English dense similarity
#                 tie = out.get("dense_en", np.zeros(n_units, np.float32))
#                 tie = np.where(np.isfinite(tie), tie, 0)
#                 out["topic"] = np.where(allowed & (ts > 0), ts + 0.01 * tie, -np.inf).astype(np.float32)
#         return out
#

#         # Burmese-language queries: the reranker reads English only, and English channels get little from
#         # a Burmese query, so skip reranking and down-weight the English channels in fusion.
#         burmese = route_my and is_burmese(query)
#         if burmese:  # channels not listed get weight 0
#             w = my_weights if my_weights is not None else MY_WEIGHTS
#             weights = {c: w.get(c, 0.0) for c in ("bm25_my", "dense_my", "bm25_en", "dense_en")}
#         rerank = rerank and self.reranker is not None and not burmese
