"""Cross-encoder reranker: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (int8 ONNX, Apache-2.0).

Retained for explicit offline experiments only. The application does not load or call it.
Reads an English query and passage together and returns one relevance logit.
"""
import os
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / os.getenv("RERANK_MODEL", "mmarco-mMiniLMv2-L12")


class Reranker:
    def __init__(self, model_dir: Path = MODEL_DIR, max_len: int = 200):
        self.lock = threading.Lock()
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.max_len = max_len
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding()
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(next(model_dir.glob("*.onnx"))), opts,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def scores(self, query, texts, batch=16, max_len=None):
        """Exactly one English representative per document, same token budget for both sources."""
        with self.lock:
            return self._scores(query, texts, batch, max_len)

    def _scores(self, query, texts, batch, max_len):
        self.tok.enable_truncation(max_len or self.max_len)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths per batch: less padding
        res = np.zeros(len(texts), np.float32)
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            enc = self.tok.encode_batch([(query, texts[i]) for i in idx])
            feed = {"input_ids": np.array([e.ids for e in enc], np.int64),
                    "attention_mask": np.array([e.attention_mask for e in enc], np.int64)}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.array([e.type_ids for e in enc], np.int64)
            res[idx] = self.sess.run(None, feed)[0].ravel()
        return res
