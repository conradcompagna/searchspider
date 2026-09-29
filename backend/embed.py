"""multilingual-e5-small (int8 ONNX) sentence embedder. No PyTorch needed.

e5 convention: queries are prefixed "query: ", documents "passage: "; mean pooling; L2 norm.
"""
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / os.getenv("EMBED_MODEL", "multilingual-e5-small")


class Embedder:
    def __init__(self, model_dir: Path = MODEL_DIR, max_len: int = 512):
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding()
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(model_dir / "model_quantized.onnx"), opts,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def _run(self, texts):
        enc = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        h = self.sess.run(None, feed)[0]
        v = (h * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def passages(self, texts, token_budget=8192, progress=None):
        out = []
        # sort by length so batches pad less; batch size shrinks as texts get longer (memory)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        lens = [len(e.ids) for e in self.tok.encode_batch(["passage: " + texts[i] for i in order])]
        b = 0
        while b < len(order):
            size = max(1, min(64, token_budget // max(lens[b], 1)))
            size = max(1, min(size, token_budget // max(lens[min(b + size, len(order)) - 1], 1)))
            idx = order[b:b + size]
            out.append((idx, self._run(["passage: " + texts[i] for i in idx])))
            b += size
            if progress:
                progress(b, len(order))
        res = np.zeros((len(texts), out[0][1].shape[1]), dtype=np.float32)
        for idx, v in out:
            res[idx] = v
        return res

    def query(self, text):
        return self._run(["query: " + text])[0]
