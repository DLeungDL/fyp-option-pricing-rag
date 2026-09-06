"""Translate parsed English markdown papers into Traditional Chinese (zh-TW)."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PARSED_DIR = Path(__file__).resolve().parents[1] / "data" / "parsed"
OUT_DIR = PARSED_DIR / "zh-TW"
STATE_PATH = OUT_DIR / "_translate_state.json"
RELAY_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:10100/v1")
MODEL = os.environ.get("TRANSLATE_MODEL", "xai/grok-4.20-0309-non-reasoning")
MAX_CHARS = int(os.environ.get("TRANSLATE_MAX_CHARS", "3500"))
RETRIES = 4

SYSTEM_PROMPT = (
    "You are a professional academic translator. Translate English scholarly papers "
    "into Traditional Chinese used in Taiwan (zh-TW).\n\n"
    "Rules:\n"
    "- Output ONLY the translated Markdown. No preface or task restatement.\n"
    "- Use Taiwan Traditional Chinese wording and punctuation.\n"
    "- Keep Markdown structure: headings, lists, tables, HTML, image links.\n"
    "- Do not translate or alter LaTeX/math, citation keys, DOIs, URLs, emails, "
    "file names, or image paths.\n"
    "- For key technical terms, give Chinese first then English in parentheses on "
    "first appearance in a section, e.g. yinhan bodonglu (implied volatility) in zh-TW.\n"
    "- Preserve numbers, symbols, and table numeric values exactly.\n"
    "- Do not omit paragraphs. Translate the full chunk."
)


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"papers": {}}


def save_state(state: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def split_pages(slug: str, markdown: str) -> list[str]:
    json_path = PARSED_DIR / f"{slug}.json"
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        pages = [(p.get("markdown") or "").strip() for p in (payload.get("pages") or [])]
        pages = [p for p in pages if p]
        if pages:
            return refine_chunks(pages)
    return refine_chunks([markdown])


def refine_chunks(parts: list[str]) -> list[str]:
    out: list[str] = []
    for part in parts:
        if len(part) <= MAX_CHARS:
            if part.strip():
                out.append(part.strip())
            continue
        out.extend(split_long(part))
    return out


def split_long(text: str) -> list[str]:
    blocks = re.split(r"(\n\s*\n)", text)
    chunks: list[str] = []
    buf = ""
    for block in blocks:
        if not block:
            continue
        if len(buf) + len(block) <= MAX_CHARS or not buf:
            buf += block
            continue
        chunks.append(buf.strip())
        buf = block.lstrip("\n")
    if buf.strip():
        chunks.append(buf.strip())
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= int(MAX_CHARS * 1.4):
            final.append(chunk)
        else:
            for i in range(0, len(chunk), MAX_CHARS):
                piece = chunk[i : i + MAX_CHARS].strip()
                if piece:
                    final.append(piece)
    return final


def translate_chunk(client: OpenAI, paper_name: str, index: int, total: int, chunk: str) -> str:
    user = (
        f"Paper: {paper_name}\n"
        f"Chunk {index}/{total} of this paper.\n"
        "Translate the following Markdown into zh-TW Traditional Chinese.\n\n"
        "----- SOURCE -----\n"
        f"{chunk}\n"
        "----- END SOURCE -----"
    )
    last_err: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                temperature=0.1,
                timeout=120,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
            )
            text = (completion.choices[0].message.content or "").strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:markdown|md)?\n", "", text)
                text = re.sub(r"\n```$", "", text).strip()
            if not text:
                raise RuntimeError("empty translation")
            return text
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            wait = min(20, 2 ** attempt)
            print(f"    retry {attempt}/{RETRIES} after error: {exc}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed to translate chunk {index}: {last_err}")


def write_paper(out_path: Path, title: str, source_name: str, translations: list[str]) -> None:
    header = (
        f"# {title}\n\n"
        f"> 原文檔案：`{source_name}`  \n"
        "> 語言：繁體中文（臺灣，zh-TW）  \n"
        "> 說明：由 LlamaParse Markdown 分段機器翻譯；公式、表格數字與檔名未改寫。\n\n"
        "---\n\n"
    )
    body = "\n\n".join(translations).strip() + "\n"
    out_path.write_text(header + body, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PARSED_DIR / "manifest.json").read_text(encoding="utf-8"))
    client = OpenAI(base_url=RELAY_URL, api_key=os.environ.get("OPENAI_API_KEY", "not-needed"))
    state = load_state()

    for item in manifest:
        slug = item["slug"]
        name = item["name"]
        src = PARSED_DIR / f"{slug}.md"
        out_path = OUT_DIR / f"{slug}.zh-TW.md"
        if not src.exists():
            print(f"skip missing {src.name}")
            continue
        markdown = src.read_text(encoding="utf-8")
        chunks = split_pages(slug, markdown)
        paper_state = state["papers"].setdefault(slug, {"done": 0, "parts": []})
        done = int(paper_state.get("done") or 0)
        parts: list[str] = list(paper_state.get("parts") or [])
        print(f"=== {name} ({len(chunks)} chunks, resume at {done}) ===", flush=True)
        for i, chunk in enumerate(chunks):
            if i < done:
                continue
            print(f"  translating {i + 1}/{len(chunks)} ({len(chunk)} chars)", flush=True)
            translated = translate_chunk(client, name, i + 1, len(chunks), chunk)
            if i < len(parts):
                parts[i] = translated
            else:
                parts.append(translated)
            paper_state["done"] = i + 1
            paper_state["parts"] = parts
            state["papers"][slug] = paper_state
            save_state(state)
            write_paper(out_path, Path(name).stem, src.name, parts)
        write_paper(out_path, Path(name).stem, src.name, parts)
        print(f"  wrote {out_path} ({out_path.stat().st_size} bytes)")

    index_lines = ["# 論文繁體中文譯本（zh-TW）", "", "由英文 LlamaParse Markdown 機器翻譯。", ""]
    for item in manifest:
        out_name = f"{item["slug"]}.zh-TW.md"
        index_lines.append(f"- [{item["name"]}]({out_name})")
    (OUT_DIR / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print("All papers processed.")


if __name__ == "__main__":
    main()
