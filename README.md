# FYP Local RAG

Local retrieval-augmented generation over LlamaParse markdown and extracted figures for the option-pricing papers in this project.

## Layout

- `data/parsed/` downloaded markdown + per-page JSON
- `data/parsed/images/` chart/table/layout images from LlamaParse
- `data/chroma/` local Chroma vector store
- `scripts/download_parsed.py` pull completed LlamaParse jobs
- `scripts/download_images.py` download layout figures
- `scripts/ingest_rag.py` chunk text + figure captions into Chroma
- `scripts/query_rag.py` retrieve (and optionally generate an answer)

## Setup

```powershell
cd G:\FYP
uv sync --python 3.12
$env:LLAMA_CLOUD_API_KEY = [Environment]::GetEnvironmentVariable('LLAMA_CLOUD_API_KEY','User')
```

Embedding model: `BAAI/bge-small-en-v1.5` (local). Generation uses the OpenAI-compatible relay at `http://127.0.0.1:10100/v1`.

Figure chunks store caption/alt-text plus a local `image_path`. Filter with `--kind chart`, `--kind table`, or `--kind figure`.

## Run

```powershell
uv run python scripts/download_parsed.py
uv run python scripts/download_images.py
uv run python scripts/ingest_rag.py
uv run python scripts/query_rag.py "How do neural networks compare with Black-Scholes for option pricing?"
uv run python scripts/query_rag.py --retrieve-only --kind chart "implied volatility surface Heston"
uv run python scripts/query_rag.py --retrieve-only --kind table "LSTM GRU option pricing results"
```
