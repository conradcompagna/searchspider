# Third-party components and source material

searchspider uses upstream libraries and models; their authors retain their copyrights and license terms.

| Component | Role | Upstream |
|---|---|---|
| Snowball / `snowballstemmer` | English stemming | [Snowball](https://snowballstem.org/), BSD-licensed distribution |
| Apache Lucene EnglishAnalyzer stopword list | English lexical normalization | [EnglishAnalyzer.java](https://github.com/apache/lucene/blob/main/lucene/analysis/common/src/java/org/apache/lucene/analysis/en/EnglishAnalyzer.java), Apache 2.0 |
| multilingual-e5-small | Dense passage and query embeddings | [intfloat model](https://huggingface.co/intfloat/multilingual-e5-small), [Xenova ONNX export](https://huggingface.co/Xenova/multilingual-e5-small) |
| ONNX Runtime and Hugging Face Tokenizers | CPU model execution and tokenization | [ONNX Runtime](https://github.com/microsoft/onnxruntime), [Tokenizers](https://github.com/huggingface/tokenizers) |
| FastAPI, React, TypeScript and Vite | HTTP service and browser interface | Dependencies and versions in `requirements.txt` and `frontend/package-lock.json` |
| Google Gen AI SDK | Optional Gemini calls | [python-genai](https://github.com/googleapis/python-genai) |

Model weights and original book files are not bundled. The source corpus has its own bibliographic provenance and rights, described in [docs/DATA.md](docs/DATA.md). No repository-wide license is assigned to the source books, translations or upstream models by this publication.
