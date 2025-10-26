# local-llm

A tiny PyQt6 desktop chat UI that runs an open‑source LLM locally via Hugging Face Transformers. CUDA GPUs (e.g., RTX 4090) are supported for fast inference. Built to be simple, hackable, and easy to embed into other projects.

---

## Features

* Local inference with `transformers` + `accelerate`
* Streaming token output into the UI
* Settings dialog for model & precision (UI: `auto`/`fp16`; `int4` via `.env`)
* `.env` overrides for generation params (temperature, top‑p, max tokens)
* Cross‑platform (Linux/macOS/Windows)†

> † Note: 4‑bit (`bitsandbytes`) support is most reliable on Linux. On Windows/macOS, prefer `auto` or `fp16`.

---

## Requirements

* **Python**: 3.12+
* **GPU (optional)**: NVIDIA CUDA‑capable GPU for acceleration (e.g., RTX 4090)
* **OS**: Linux/macOS/Windows

---

## Quick Start

### 1) Create a virtual environment & install

```bash
# from repo root
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Configure (optional)

Create a `.env` file at the repo root. Sensible defaults exist; use this to pin your model/precision and generation params.

```env
# Model & precision
MODEL_ID=Qwen/Qwen2.5-3B-Instruct
PRECISION=auto          # auto | fp16 | int4 (int4 best on Linux)

# Generation
MAX_NEW_TOKENS=256
TEMPERATURE=0.7
TOP_P=0.9

# Device/placement
DEVICE_MAP=auto         # e.g., auto, cuda, cpu

# System prompt
SYSTEM_PROMPT=You are a helpful assistant.
```

You can also change model & precision at runtime via **Settings → Settings…**. The UI exposes common models and `auto`/`fp16`; set `PRECISION=int4` in `.env` if you want 4‑bit.

### 3) Run

```bash
# run as a module so package‑relative imports work
python -m src.app
```

The window title shows the active model & precision.

---

## What’s inside

```
├── .env                  # not checked in; local overrides
├── README.md             # this file
├── requirements.txt      # Python deps
├── scripts/
│   └── tree_script.py
└── src/
    ├── app.py            # boots the Qt app, wires Config + MainWindow
    ├── config.py         # loads env, exposes Config, save_env()
    ├── llm/
    │   ├── engine.py     # HF model/tokenizer load, streaming, precisions
    │   └── worker.py     # background generation, Qt signals
    ├── ui/
    │   └── main_window.py# chat UI + settings dialog
    └── utils/
        └── logging.py    # basic logging setup
```

**Flow:** `src/app.py` → read `.env` into `Config` → show `MainWindow` → user selects/uses model → `llm/engine.py` loads model/tokenizer (precision: auto/fp16/int4) → `llm/worker.py` streams tokens back to UI.

---

## Models & Precision

* Works with many decoder‑style HF models; good starters:

  * `Qwen/Qwen2.5-3B-Instruct` (fast, capable)
  * `microsoft/Phi-3.5-mini-instruct` (tiny, very fast)
  * `mistralai/Mistral-7B-Instruct-v0.3` (larger, stronger)
* Precisions:

  * **auto**: lets Accelerate/Transformers choose (recommended)
  * **fp16**: half precision on GPU
  * **int4**: 4‑bit (via `bitsandbytes`), best on Linux; use when VRAM is tight

> Tip (RTX 4090): `PRECISION=fp16` usually yields great throughput. For bigger models, try `int4`.

---

## CUDA & Performance Notes

* Install NVIDIA drivers/CUDA runtime appropriate for your OS.
* If `bitsandbytes` fails to load on Windows/macOS, fall back to `PRECISION=fp16`.
* OOM? Try a smaller model, `MAX_NEW_TOKENS` ↓, or `PRECISION=int4`.

---

## Troubleshooting

* **Module import errors**: always run from repo root with `python -m src.app`.
* **4‑bit load errors**: ensure `bitsandbytes` supports your environment; prefer Linux.
* **CUDA not detected**: set `DEVICE_MAP=auto` (or `cuda`) and update drivers; otherwise it will use CPU.
* **Tokenizer/model mismatch**: ensure `MODEL_ID` points to a valid chat‑tuned model.

---

## Development

* Keep business logic in `llm/` and UI concerns in `ui/`.
* Avoid mixing Qt signals with heavy model ops; use the worker for background generation.
* Consider adding unit tests (see below) and a `Makefile` for common tasks.

### Suggested Tests (optional)

```
tests/
  test_config.py     # env parsing, defaults, save_env()
  test_engine.py     # model load stub/mocks; streamer plumbing
```

Run with:

```bash
pip install pytest
pytest -q
```

---

## Roadmap

* History persistence
* Stop‑generation button
* Model download/provisioning helper

---

## License

MIT (or your preference)
