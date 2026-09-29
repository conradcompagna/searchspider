"""Embed every passage once (multilingual-e5-small, int8 ONNX) and store float16 matrices in index/.

Run after build_orders.py and build_chronicle.py:  python scripts/build_index.py
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from corpus import GROUPS, INDEX, load_corpus, passage_table  # noqa: E402
from embed import Embedder  # noqa: E402


def main():
    INDEX.mkdir(exist_ok=True)
    orders, chron = load_corpus()
    table = passage_table(orders, chron)
    emb = Embedder()
    mpath = INDEX / "manifest.json"
    manifest = json.load(open(mpath)) if mpath.exists() else \
        {"model": "Xenova/" + os.getenv("EMBED_MODEL", "multilingual-e5-small") + " (model_quantized.onnx)", "groups": {}}
    only = sys.argv[1:] or GROUPS  # e.g. python scripts/build_index.py chron_my
    for g in only:
        texts = [p[3] for p in table[g]]
        t0 = time.time()
        last = [0]

        def prog(done, total):
            if done - last[0] >= 1000 or done == total:
                last[0] = done
                print(f"  {g}: {done}/{total}  {time.time() - t0:.0f}s", flush=True)

        vecs = emb.passages(texts, progress=prog).astype(np.float16)
        np.save(INDEX / f"{g}.npy", vecs)
        digest = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]
        manifest["groups"][g] = {"rows": len(texts), "text_sha256_16": digest}
    json.dump(manifest, open(INDEX / "manifest.json", "w"), indent=1)
    print("index ready:", {g: v["rows"] for g, v in manifest["groups"].items()})


if __name__ == "__main__":
    main()
