from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import requests
from dotenv import load_dotenv


SYSTEM_PROMPT = (
    "You are a scientific PDF OCR engine. "
    "Read the page image carefully and preserve reading order. "
    "Do not hallucinate missing text. "
    "If a token is unreadable, use [UNCLEAR]."
)


def build_user_prompt(output_format: str) -> str:
    if output_format == "latex":
        return (
            "Convert this single scientific PDF page into LaTeX body content only. "
            "Do not include \\documentclass, packages, or \\begin{document}. "
            "Preserve section headings, paragraphs, lists, tables when legible, and all mathematics. "
            "Use valid LaTeX math syntax. "
            "For display equations, use equation or align only when clearly appropriate. "
            "For figures or diagrams, do not invent TikZ; instead insert a LaTeX comment line like "
            "`% FIGURE: concise description`. "
            "Return LaTeX only."
        )
    return (
        "Convert this single scientific PDF page into Markdown while preserving reading order. "
        "Render all mathematics in LaTeX using $...$ or $$...$$. "
        "Keep section hierarchy, lists, and tables when legible. "
        "For figures or diagrams, insert a line like `[FIGURE: concise description]`. "
        "Return Markdown only."
    )


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def load_api_config(args: argparse.Namespace, require_model: bool) -> tuple[str, str, str]:
    load_dotenv()
    api_key = args.api_key or os.getenv("OCR_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = args.base_url or os.getenv("OCR_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    model = args.model or os.getenv("OCR_MODEL")
    if args.render_only:
        return api_key or "", normalize_base_url(base_url) if base_url else "", model or ""
    if not api_key:
        raise ValueError("Missing API key. Set OCR_API_KEY or OPENAI_API_KEY.")
    if not base_url:
        raise ValueError("Missing base URL. Set OCR_BASE_URL or OPENAI_BASE_URL.")
    if require_model and not model:
        raise ValueError("Missing OCR model name. Set OCR_MODEL or pass --model.")
    return api_key, normalize_base_url(base_url), model or ""


def parse_page_spec(page_spec: str | None, max_pages: int | None = None) -> list[int] | None:
    if not page_spec:
        return None
    pages: set[int] = set()
    for chunk in page_spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"Invalid page range: {chunk}")
            pages.update(range(start, end + 1))
        else:
            pages.add(int(chunk))
    ordered = sorted(page for page in pages if page > 0)
    if max_pages is not None:
        ordered = [page for page in ordered if page <= max_pages]
    return ordered


def get_pdf_page_count(pdf_path: Path) -> int:
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo is None:
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pdfinfo or pypdf is required to count PDF pages.") from exc
        return len(PdfReader(str(pdf_path)).pages)

    proc = subprocess.run(
        [pdfinfo, str(pdf_path)],
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
        errors="replace",
    )
    match = re.search(r"^Pages:\s+(\d+)\s*$", proc.stdout, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not parse page count from pdfinfo output for {pdf_path}.")
    return int(match.group(1))


def ensure_pdftoppm() -> str:
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        raise RuntimeError("pdftoppm was not found in PATH.")
    return pdftoppm


def render_pdf_pages(
    pdf_path: Path,
    output_dir: Path,
    dpi: int,
    pages: Iterable[int],
) -> list[Path]:
    pdftoppm = ensure_pdftoppm()
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    for page in pages:
        prefix = output_dir / f"page-{page:04d}"
        subprocess.run(
            [
                pdftoppm,
                "-png",
                "-r",
                str(dpi),
                "-f",
                str(page),
                "-l",
                str(page),
                str(pdf_path),
                str(prefix),
            ],
            check=True,
            capture_output=True,
        )
        image_path = output_dir / f"{prefix.name}-{page}.png"
        if not image_path.exists():
            candidates = sorted(output_dir.glob(f"{prefix.name}*.png"))
            if not candidates:
                raise RuntimeError(f"Failed to render page {page} for {pdf_path}.")
            image_path = candidates[0]
        rendered.append(image_path)
    return rendered


def encode_image_as_data_url(image_path: Path) -> str:
    mime = "image/png"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def call_chat_completions(
    *,
    api_key: str,
    base_url: str,
    model: str,
    image_path: Path,
    output_format: str,
    temperature: float,
    timeout: float,
) -> dict:
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_user_prompt(output_format)},
                    {"type": "image_url", "image_url": {"url": encode_image_as_data_url(image_path)}},
                ],
            },
        ],
    }
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def extract_text_from_response(response_json: dict) -> str:
    choices = response_json.get("choices", [])
    if not choices:
        raise RuntimeError("OCR response did not contain any choices.")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts).strip()
    raise RuntimeError("Unsupported response content shape.")


def list_models(api_key: str, base_url: str, timeout: float) -> dict:
    response = requests.get(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def merge_outputs(page_texts: list[tuple[int, str]], output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    chunks: list[str] = []
    for page, text in page_texts:
        if suffix == ".tex":
            chunks.append(f"% ===== Page {page} =====\n{text.strip()}\n")
        else:
            chunks.append(f"<!-- ===== Page {page} ===== -->\n\n{text.strip()}\n")
    output_path.write_text("\n".join(chunks).strip() + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a PDF and send page images to an OpenAI-compatible OCR/vision API."
    )
    parser.add_argument("--pdf", type=Path, required=False, help="Path to the PDF file.")
    parser.add_argument("--pages", type=str, default=None, help="Pages like 1-3,5.")
    parser.add_argument("--max-pages", type=int, default=None, help="Limit page count after parsing --pages.")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--model", type=str, default=None, help="OCR or vision model name.")
    parser.add_argument("--api-key", type=str, default=None, help="Override OCR_API_KEY / OPENAI_API_KEY.")
    parser.add_argument("--base-url", type=str, default=None, help="Override OCR_BASE_URL / OPENAI_BASE_URL.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-format", choices=["markdown", "latex"], default="markdown")
    parser.add_argument("--output-dir", type=Path, default=Path("ocr/output"))
    parser.add_argument("--render-dir", type=Path, default=Path("ocr/output/rendered"))
    parser.add_argument("--render-only", action="store_true", help="Only render PDF pages, do not call the API.")
    parser.add_argument("--list-models", action="store_true", help="List models exposed by the configured API.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing page outputs.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    api_key, base_url, model = load_api_config(
        args,
        require_model=not args.list_models and not args.render_only,
    )

    if args.list_models:
        models_json = list_models(api_key=api_key, base_url=base_url, timeout=args.timeout)
        print(json.dumps(models_json, ensure_ascii=False, indent=2))
        return

    if args.pdf is None:
        parser.error("--pdf is required unless --list-models is used.")

    pdf_path = args.pdf.resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    page_count = get_pdf_page_count(pdf_path)
    selected_pages = parse_page_spec(args.pages, max_pages=page_count)
    if selected_pages is None:
        selected_pages = list(range(1, page_count + 1))
    if args.max_pages is not None:
        selected_pages = selected_pages[: args.max_pages]
    if not selected_pages:
        raise ValueError("No pages selected.")

    doc_stem = pdf_path.stem.replace(" ", "_")
    render_dir = args.render_dir / doc_stem
    result_dir = args.output_dir / doc_stem
    render_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    rendered_images = render_pdf_pages(
        pdf_path=pdf_path,
        output_dir=render_dir,
        dpi=args.dpi,
        pages=selected_pages,
    )

    manifest = {
        "pdf": str(pdf_path),
        "pages": selected_pages,
        "dpi": args.dpi,
        "output_format": args.output_format,
        "base_url": base_url,
        "model": model,
        "render_dir": str(render_dir),
        "result_dir": str(result_dir),
    }
    (result_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.render_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    page_outputs: list[tuple[int, str]] = []
    extension = ".tex" if args.output_format == "latex" else ".md"
    for page, image_path in zip(selected_pages, rendered_images):
        page_output_path = result_dir / f"page-{page:04d}{extension}"
        raw_response_path = result_dir / f"page-{page:04d}.response.json"
        if page_output_path.exists() and not args.overwrite:
            text = page_output_path.read_text(encoding="utf-8")
            page_outputs.append((page, text))
            continue

        response_json = call_chat_completions(
            api_key=api_key,
            base_url=base_url,
            model=model,
            image_path=image_path,
            output_format=args.output_format,
            temperature=args.temperature,
            timeout=args.timeout,
        )
        raw_response_path.write_text(
            json.dumps(response_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        text = extract_text_from_response(response_json)
        page_output_path.write_text(text + "\n", encoding="utf-8")
        page_outputs.append((page, text))
        print(f"Finished page {page}: {page_output_path}")
        sys.stdout.flush()

    merged_path = result_dir / ("document.tex" if args.output_format == "latex" else "document.md")
    merge_outputs(page_outputs, merged_path)
    print(f"Wrote merged output to {merged_path}")


if __name__ == "__main__":
    main()
