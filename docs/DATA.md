# Data, models and provenance

The live collection joins Than Tun's *Royal Orders of Burma*, Parts 3–9, and three volumes of the *Konbaungset Yazawin*. The application searches English text and retains Burmese source text for reading. Extraction and translation quality constrain the evidence available to retrieval.

## What this repository includes

- Application, corpus-preparation, replay and evaluation source code.
- Test fixtures, evaluation question wording, grade IDs, per-question scores and reference ranks.
- A compact provenance manifest identifying original experiment artifacts and their SHA-256 hashes.

Full book PDFs/transcriptions, model weights, production databases, request logs, payment data, API keys and machine-specific configuration are omitted. Original source works, translations and upstream models retain their own rights and terms; this repository does not assign a new license to those materials.

## Chronicle data

The related [Konbaung Chronicle Knowledge Graph dataset](https://zenodo.org/records/22949204) publishes source sentences, claims, embedding matrices and entity-resolution tables. Its [data guide](https://github.com/conradcompagna/konbaung-knowledge-graph/tree/main/research/data-release) describes the released schema and provenance.

The graph release is a separate research dataset. SearchSpider's deployed corpus uses its own document records and e5 passage indexes.

## Embedding model

The deployed application uses the quantized [Xenova multilingual-e5-small ONNX export](https://huggingface.co/Xenova/multilingual-e5-small) for CPU embeddings. The original embedding model is [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small).

The model is an upstream dependency, not a model trained by this project. Offline reranker code is retained for experiments, but no reranker runs in application search.

## Language processing

English stemming uses `snowballstemmer==3.1.1`. The English stopword list derives from Apache Lucene's `EnglishAnalyzer`. The corresponding upstream software retains its BSD and Apache 2.0 notices/terms. See [THIRD_PARTY.md](../THIRD_PARTY.md).
