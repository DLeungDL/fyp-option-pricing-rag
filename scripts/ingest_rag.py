"""Chunk LlamaParse markdown plus local chart/table figures into Chroma."""
from __future__ import annotations

import json
import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

PARSED_DIR = Path(__file__).resolve().parents[1] / "data" / "parsed"
IMAGES_DIR = PARSED_DIR / "images"
CHROMA_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma"
COLLECTION = "fyp-option-pricing"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

PAGE_RE = re.compile(r"page_(\d+)", re.I)
KIND_RE = re.compile(r"_(chart|table|image|figure)", re.I)
MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
CAPTION_RE = re.compile(
    r"(?:\*\*)?(Figure|Table|Fig\.?)\s*([0-9]+[A-Za-z]?)\.?\s*[:.\-]?\s*(.*)",
    re.I,
)
HTML_CAPTION_RE = re.compile(r"<caption>(.*?)</caption>", re.I | re.S)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        if end < n:
            window = text[start:end]
            split_at = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if split_at >= chunk_size // 3:
                end = start + split_at + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_paper_pages() -> dict[str, dict]:
    papers: dict[str, dict] = {}
    for path in sorted(PARSED_DIR.glob("*.json")):
        if path.name == "manifest.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        pages = {
            int(page.get("page_number") or 0): (page.get("markdown") or "")
            for page in (payload.get("pages") or [])
        }
        papers[payload["slug"]] = {
            "job_id": payload.get("job_id"),
            "name": payload.get("name"),
            "slug": payload.get("slug"),
            "pages": pages,
        }
    return papers


def figure_kind(filename: str) -> str:
    match = KIND_RE.search(filename)
    if match:
        token = match.group(1).lower()
        return "figure" if token == "image" else token
    return "figure"


def page_from_filename(filename: str) -> int:
    match = PAGE_RE.search(filename)
    return int(match.group(1)) if match else 0


def nearby_context(markdown: str, filename: str, window: int = 700) -> str:
    if not markdown:
        return ""
    alts = []
    for alt, target in MD_IMAGE_RE.findall(markdown):
        if Path(target).name == filename or filename in target:
            alts.append(alt.strip())
    captions = []
    for line in markdown.splitlines():
        stripped = re.sub(r"[*_`#]+", "", line).strip()
        if CAPTION_RE.match(stripped):
            captions.append(stripped)
    html_captions = [re.sub(r"<[^>]+>", " ", c).strip() for c in HTML_CAPTION_RE.findall(markdown)]
    idx = markdown.find(filename)
    if idx >= 0:
        snippet = markdown[max(0, idx - window) : idx + window]
    else:
        snippet = markdown[:1600]
    snippet = re.sub(r"<[^>]+>", " ", snippet)
    snippet = re.sub(r"\s+", " ", snippet).strip()
    parts = []
    if alts:
        parts.append("Alt text: " + " | ".join(alts))
    if captions:
        parts.append("Captions: " + " | ".join(captions[:4]))
    if html_captions:
        parts.append("Table captions: " + " | ".join(html_captions[:4]))
    if snippet:
        parts.append("Page context: " + snippet[:1400])
    return "\n".join(parts)


def load_figures(papers: dict[str, dict]) -> list[dict]:
    figures: list[dict] = []
    for paper in papers.values():
        manifest_path = IMAGES_DIR / paper["slug"] / "manifest.json"
        if not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for image in payload.get("images") or []:
            filename = image.get("filename") or ""
            page = page_from_filename(filename)
            kind = figure_kind(filename)
            rel_path = f"data/parsed/images/{paper['slug']}/{filename}"
            abs_path = IMAGES_DIR / paper["slug"] / filename
            if not abs_path.exists():
                continue
            markdown = paper["pages"].get(page, "")
            context = nearby_context(markdown, filename)
            text = (
                f"[{kind.upper()}] {paper['name']} page {page}\n"
                f"File: {rel_path}\n"
                f"Filename: {filename}\n"
                f"{context}"
            ).strip()
            figures.append(
                {
                    "id": f"{paper['slug']}-p{page}-{kind}-{Path(filename).stem}",
                    "text": text,
                    "metadata": {
                        "source": paper["name"],
                        "slug": paper["slug"],
                        "job_id": paper["job_id"],
                        "page": int(page),
                        "chunk": 0,
                        "kind": kind,
                        "image_path": rel_path,
                        "filename": filename,
                    },
                }
            )
    return figures


def main() -> None:
    papers = load_paper_pages()
    if not papers:
        raise SystemExit(f"No parsed JSON found in {PARSED_DIR}. Run download_parsed.py first.")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    for paper in papers.values():
        for page, markdown in paper["pages"].items():
            for i, chunk in enumerate(chunk_text(markdown)):
                ids.append(f"{paper['slug']}-p{page}-c{i}")
                texts.append(chunk)
                metadatas.append(
                    {
                        "source": paper["name"],
                        "slug": paper["slug"],
                        "job_id": paper["job_id"],
                        "page": int(page),
                        "chunk": i,
                        "kind": "text",
                        "image_path": "",
                        "filename": "",
                    }
                )

    figures = load_figures(papers)
    for fig in figures:
        ids.append(fig["id"])
        texts.append(fig["text"])
        metadatas.append(fig["metadata"])

    batch = 64
    for i in range(0, len(texts), batch):
        collection.add(
            ids=ids[i : i + batch],
            documents=texts[i : i + batch],
            metadatas=metadatas[i : i + batch],
        )
        print(f"Indexed {min(i + batch, len(texts))}/{len(texts)} chunks")

    n_text = sum(1 for m in metadatas if m["kind"] == "text")
    n_fig = len(metadatas) - n_text
    print(
        f"Done. Collection '{COLLECTION}' has {collection.count()} chunks "
        f"({n_text} text, {n_fig} figures) from {len(papers)} papers."
    )


if __name__ == "__main__":
    main()

