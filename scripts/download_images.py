"""Download LlamaParse layout images (charts, tables, figures) locally."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx

API_BASE = "https://api.cloud.llamaindex.ai"
PROJECT_ID = "092d105c-86d3-4b3c-9bb8-ebe4212224ec"
PARSED_DIR = Path(__file__).resolve().parents[1] / "data" / "parsed"
IMAGES_DIR = PARSED_DIR / "images"


def slugify(name: str) -> str:
    stem = Path(name).stem
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return slug[:80] or "document"


def is_figure(image: dict) -> bool:
    category = (image.get("category") or "").lower()
    filename = (image.get("filename") or "").lower()
    if category == "screenshot":
        return False
    if category == "layout":
        return True
    return any(token in filename for token in ("chart", "figure", "table", "image"))


def rewrite_markdown(md_path: Path, slug: str, filenames: set[str]) -> int:
    if not md_path.exists():
        return 0
    text = md_path.read_text(encoding="utf-8")
    replaced = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal replaced
        alt, target = match.group(1), match.group(2)
        name = Path(target).name
        if name in filenames:
            replaced += 1
            return f"![{alt}](images/{slug}/{name})"
        return match.group(0)

    new_text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, text)
    if replaced:
        md_path.write_text(new_text, encoding="utf-8")
    return replaced


def main() -> None:
    api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise SystemExit("LLAMA_CLOUD_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "X-Project-Id": PROJECT_ID,
    }
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        listed = client.get(f"{API_BASE}/api/v2/parse", params={"limit": 50}, headers=headers)
        listed.raise_for_status()
        jobs = [j for j in listed.json().get("items", []) if j.get("status") == "COMPLETED"]
        print(f"Found {len(jobs)} completed jobs")

        for job in jobs:
            job_id = job["id"]
            name = job.get("name") or job_id
            slug = slugify(name)
            out_dir = IMAGES_DIR / slug
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"Images for {name}")
            res = client.get(
                f"{API_BASE}/api/v2/parse/{job_id}",
                params=[("expand", "images_content_metadata")],
                headers=headers,
            )
            res.raise_for_status()
            images = ((res.json().get("images_content_metadata") or {}).get("images") or [])
            saved = []
            for image in images:
                if not is_figure(image):
                    continue
                filename = image.get("filename") or f"image_{image.get('index', 0)}.jpg"
                url = image.get("presigned_url")
                if not url:
                    continue
                dest = out_dir / filename
                if not dest.exists() or dest.stat().st_size == 0:
                    img_res = client.get(url)
                    img_res.raise_for_status()
                    dest.write_bytes(img_res.content)
                saved.append(
                    {
                        "filename": filename,
                        "category": image.get("category"),
                        "content_type": image.get("content_type"),
                        "bytes": dest.stat().st_size,
                        "bbox": image.get("bbox"),
                    }
                )
                print(f"  saved {filename} ({dest.stat().st_size} bytes)")

            rewritten = rewrite_markdown(PARSED_DIR / f"{slug}.md", slug, {item["filename"] for item in saved})
            record = {
                "job_id": job_id,
                "name": name,
                "slug": slug,
                "n_remote_images": len(images),
                "n_saved_figures": len(saved),
                "n_markdown_links_updated": rewritten,
                "images": saved,
            }
            (out_dir / "manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest.append({k: record[k] for k in ("job_id", "name", "slug", "n_remote_images", "n_saved_figures", "n_markdown_links_updated")})
            print(f"  figures={len(saved)} markdown_links_updated={rewritten}")

    (IMAGES_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote figure manifest to {IMAGES_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
