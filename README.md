# Scanned PDF → audiobook (OCR + local LLM correction)

Turns a scanned book or document (an image-only PDF) into an MP3 audiobook. It runs entirely on your
own machine:

```
scanned PDF ──► OpenCV clean-up ──► Tesseract OCR ──► local LLM fixes OCR errors ──► text clean-up ──► edge-tts ──► MP3
              ocr_llm_postprocessor.py                                               cleanup.py      generate.py   merge.py
```

## Why each stage exists

| Stage | Problem it solves | How |
|---|---|---|
| **Pre-processing** (`preprocess_image`) | Grey, noisy scans make Tesseract misread letters | grayscale → 3×3 Gaussian blur → **Otsu** binarisation; crops the top 7% to drop running headers |
| **OCR** (`perform_ocr`) | Image → text | Tesseract `--oem 3 --psm 6` (LSTM engine, treat the page as one uniform text block) |
| **LLM correction** (`create_correction_prompt`, `get_llm_response`) | OCR leaves substitutions ("Th quick brwn fx") that a dictionary can't fix in context | Few-shot prompt to a **local** model (default: `mistral` on Ollama; `transformers` and `llama.cpp` also supported). Temperature 0.1, with stop tokens so it can't ramble. |
| **Output guard** (`post_process_llm_output`) | LLMs add chatter ("Here is the corrected text:") or echo the few-shot example | Regex strips those phrases, page-number-only lines and ISBN/copyright lines. If the LLM call fails, the raw OCR text is kept. |
| **Page joining** | Sentences and hyphenated words split across pages | Glue without a space after a trailing `-`, glue with a space if the page ended mid-sentence, otherwise start a new paragraph |
| **Throughput** | Large PDFs exhaust memory | Pages are rasterised in batches of 10 and processed by a thread pool (8 workers) |
| **Text clean-up** (`cleanup.py`) | Leftover forms, artifacts and repeated paragraphs | Drops lines of underscores (form fields), stray `\| ~ ° ¢ ©`, duplicate paragraphs, chatter; rejoins hyphenated line breaks |
| **TTS** (`generate.py`, `merge.py`) | TTS services time out on long inputs | Split into ≤ 4 000-character chunks on paragraph boundaries, synthesise each with edge-tts (`en-US-GuyNeural`), then concatenate the MP3 parts |

## Run it

Requirements: Python 3.10+, [Tesseract](https://github.com/tesseract-ocr/tesseract),
[Poppler](https://poppler.freedesktop.org/) (for PDF rasterising; set `POPPLER_PATH` if it isn't on
`PATH`), and [Ollama](https://ollama.com) with a model pulled (`ollama pull mistral`).

```bash
pip install -r requirements.txt

python ocr_llm_postprocessor.py book.pdf -o final_audio.txt          # OCR + LLM correction
python ocr_llm_postprocessor.py book.pdf --start_page 1 --end_page 5  # quick test on 5 pages
python cleanup.py --input final_audio.txt --output final_audio_clean.txt
python generate.py                                                    # → audio_parts/part_001.mp3 …
python merge.py                                                       # → final_audio_complete.mp3
```

Useful flags: `--model_name`, `--llm_framework {ollama,transformers,llama_cpp}`, `--no_preprocess`,
`--dpi`, `--top_crop`, `--workers`.

## Quick check without any OCR tools

`cleanup.py` uses only the standard library:

```bash
printf 'Here is the corrected text:\nThe meeting start-\ned at nine.\n\n12\n\nName: ______\n' > sample.txt
python cleanup.py --input sample.txt --output out.txt && cat out.txt
# -> The meeting started at nine.
```

## Notes and limitations

- Built for one specific book first. `cleanup.py` contains two document-specific rules: an end-of-text
  marker ("Phoenix Consulting Group Contact Data File") and a small typo map. Edit or remove them for
  other inputs.
- The LLM can still "fix" correct but unusual words such as names and jargon. Temperature 0.1 and the
  strict prompt reduce this but don't eliminate it.
- No sample PDF is included. Use any scanned PDF; `--start_page 1 --end_page 5` keeps a first run short.
