"""Query the local Chroma RAG index."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHROMA_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma"
COLLECTION = "fyp-option-pricing"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_LLM = "xai/grok-4.6"
RELAY_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:10100/v1")


def retrieve(query: str, k: int = 6, kinds: list[str] | None = None) -> list[dict]:
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION, embedding_function=embedding_fn)
    where = None
    if kinds:
        if len(kinds) == 1:
            where = {"kind": kinds[0]}
        else:
            where = {"kind": {"$in": kinds}}
    result = collection.query(
        query_texts=[query],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        hits.append({"text": doc, "metadata": meta, "distance": dist})
    return hits


def answer(query: str, k: int, retrieve_only: bool, kinds: list[str] | None) -> None:
    hits = retrieve(query, k=k, kinds=kinds)
    print("=== Retrieved chunks ===")
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        kind = meta.get("kind") or "text"
        image_path = meta.get("image_path") or ""
        extra = f"  image={image_path}" if image_path else ""
        print(f"[{i}] ({kind}) {meta.get('source')} p.{meta.get('page')}  distance={hit['distance']:.3f}{extra}")
        print(hit["text"][:500].replace("\n", " "))
        print()
    if retrieve_only:
        return

    context_parts = []
    for i, h in enumerate(hits, 1):
        meta = h["metadata"]
        kind = meta.get("kind") or "text"
        image_path = meta.get("image_path") or ""
        header = f"[{i}] ({kind}) {meta.get('source')} p.{meta.get('page')}"
        if image_path:
            header += f" file={image_path}"
        context_parts.append(header + "\n" + h["text"])
    context = "\n\n".join(context_parts)
    client = OpenAI(base_url=RELAY_URL, api_key=os.environ.get("OPENAI_API_KEY", "not-needed"))
    completion = client.chat.completions.create(
        model=os.environ.get("RAG_MODEL", DEFAULT_LLM),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research assistant for an option-pricing FYP. "
                    "Answer only from the provided context. Cite sources as [n] with paper name and page. "
                    "If a retrieved item is a chart or table, mention the figure/table and its local file path. "
                    "If the context is insufficient, say so."
                ),
            },
            {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"},
        ],
        temperature=0.2,
    )
    print("=== Answer ===")
    print(completion.choices[0].message.content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query local FYP RAG")
    parser.add_argument("query", help="Natural-language question")
    parser.add_argument("-k", type=int, default=6, help="Number of chunks to retrieve")
    parser.add_argument("--retrieve-only", action="store_true", help="Skip LLM generation")
    parser.add_argument(
        "--kind",
        action="append",
        choices=["text", "chart", "table", "figure"],
        help="Filter by chunk kind; repeat to include several. Default: all.",
    )
    args = parser.parse_args()
    answer(args.query, k=args.k, retrieve_only=args.retrieve_only, kinds=args.kind)


if __name__ == "__main__":
    main()
