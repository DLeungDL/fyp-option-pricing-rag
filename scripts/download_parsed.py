"""Download completed LlamaParse jobs as local markdown."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx

API_BASE = "https://api.cloud.llamaindex.ai"
PROJECT_ID = "092d105c-86d3-4b3c-9bb8-ebe4212224ec"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "parsed"


def slugify(name: str) -> str:
    stem = Path(name).stem
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return slug[:80] or "document"


def main() -> None:
    api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise SystemExit("LLAMA_CLOUD_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "X-Project-Id": PROJECT_ID,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=120.0) as client:
        listed = client.get(f"{API_BASE}/api/v2/parse", params={"limit": 50}, headers=headers)
        listed.raise_for_status()
        jobs = listed.json().get("items", [])
        completed = [j for j in jobs if j.get("status") == "COMPLETED"]
        print(f"Found {len(completed)} completed parse jobs")

        manifest = []
        for job in completed:
            job_id = job["id"]
            name = job.get("name") or job_id
            slug = slugify(name)
            print(f"Downloading {name} ({job_id})")
            res = client.get(
                f"{API_BASE}/api/v2/parse/{job_id}",
                params={"expand": ["markdown", "markdown_full"]},
                headers=headers,
            )
            res.raise_for_status()
            payload = res.json()
            pages = ((payload.get("markdown") or {}).get("pages") or [])
            markdown_full = payload.get("markdown_full") or "\n\n".join(
                (p.get("markdown") or "") for p in pages
            )
            md_path = OUT_DIR / f"{slug}.md"
            json_path = OUT_DIR / f"{slug}.json"
            md_path.write_text(markdown_full, encoding="utf-8")
            record = {
                "job_id": job_id,
                "name": name,
                "slug": slug,
                "tier": job.get("tier"),
                "status": job.get("status"),
                "n_pages": len(pages),
                "markdown_chars": len(markdown_full),
                "pages": [
                    {
                        "page_number": p.get("page_number"),
                        "markdown": p.get("markdown") or "",
                    }
                    for p in pages
                ],
            }
            json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest.append({k: record[k] for k in ("job_id", "name", "slug", "n_pages", "markdown_chars")})
            print(f"  saved {md_path.name} ({len(markdown_full)} chars, {len(pages)} pages)")

        (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {len(manifest)} documents to {OUT_DIR}")


if __name__ == "__main__":
    main()
