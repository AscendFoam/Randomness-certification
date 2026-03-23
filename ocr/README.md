# OCR Workflow

## Purpose

This directory is for higher-fidelity paper extraction than plain `pdftotext`.
The main script renders PDF pages as images, then calls an OpenAI-compatible OCR or vision endpoint to produce Markdown or LaTeX-oriented text.

This is especially useful for:

- formulas that `pdftotext` loses or mangles
- figure captions and page layout cues
- scientific pages where later code analysis depends on the exact math

## Main Script

Use [pdf_ocr_to_latex.py](/d:/Codes/Quantum/Randomness-certification/ocr/pdf_ocr_to_latex.py).

It supports:

- page rendering via `pdftoppm`
- model discovery via `GET /models`
- page-by-page OCR calls through an OpenAI-compatible `/chat/completions` endpoint
- Markdown output with LaTeX math
- LaTeX body output for later manual cleanup

## Environment

The script reads these environment variables from `.env` or the shell:

- `OCR_API_KEY` or `OPENAI_API_KEY`
- `OCR_BASE_URL` or `OPENAI_BASE_URL`
- `OCR_MODEL`

Important:

- the current repo already has DeepSeek-compatible API settings in `.env`
- but the public DeepSeek API docs do not clearly document a dedicated OCR model in the same way they document `deepseek-chat` and `deepseek-reasoner`
- so keep the model name configurable instead of hardcoding a provider-specific OCR name

## Examples

List available models from the configured endpoint:

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' ocr/pdf_ocr_to_latex.py --list-models
```

Render only, without calling the API:

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' ocr/pdf_ocr_to_latex.py `
  --pdf docs/PhysRevA.106.042414.pdf `
  --pages 1-2 `
  --render-only
```

Generate Markdown with LaTeX math:

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' ocr/pdf_ocr_to_latex.py `
  --pdf docs/PhysRevA.106.042414.pdf `
  --pages 1-3 `
  --model your-ocr-model `
  --output-format markdown
```

Generate LaTeX body text:

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' ocr/pdf_ocr_to_latex.py `
  --pdf docs/PhysRevA.95.042340.pdf `
  --pages 1-3 `
  --model your-ocr-model `
  --output-format latex
```

## Outputs

Results are stored under `ocr/output/<document-stem>/`:

- `manifest.json`
- `page-0001.md` or `page-0001.tex`
- `page-0001.response.json`
- merged `document.md` or `document.tex`

Rendered images are stored under `ocr/output/rendered/<document-stem>/`.
