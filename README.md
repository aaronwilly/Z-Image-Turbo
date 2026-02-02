---
title: Z Image Turbo
emoji: 🖼️
colorFrom: yellow
colorTo: yellow
sdk: gradio
sdk_version: 6.0.1
app_file: app.py
pinned: true
---

# Z-Image-Turbo

Ultra-fast AI image generation (8 steps). Uses [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) with SDPA/Flash Attention. **GPU (CUDA) required.**

---

## Install locally

**Requirements:** Python 3.10+, CUDA-capable GPU, pip

1. Clone or download this repo, then create and activate a venv:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Install **PyTorch 2.10.0 with CUDA** (required for GPU). This project uses torch 2.10.0; use the cu126 index:

   ```powershell
   pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
   ```

   If you already installed and see *"Torch not compiled with CUDA enabled"*, run the line above again (optionally add `--force-reinstall`).

3. Install the rest (online):

   ```powershell
   pip install -r requirements.txt
   ```

---

## Offline install (wheels)

Use this when you have no network on the target machine. Do the following **while online**, then copy the project (including the `wheels` folder) to the offline machine. The download script fetches **PyTorch 2.10.0 with CUDA 12.6** into `wheels` so the app can use the GPU.

1. From the **project root** (same folder as `app.py`), in PowerShell:

   ```powershell
   .\download_wheels.ps1
   ```

   If execution is blocked, run once:

   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
   ```

   This creates a `wheels` folder with all dependencies. Copy the whole project (including `wheels` and `requirements-offline.txt`) to the offline machine.

2. **On the offline machine**, from the project root:

   ```powershell
   .\install_offline.ps1
   ```

   This installs from `wheels` only (no network). Then run the app as in [Run locally](#run-locally); for offline model loading, set `Z_IMAGE_MODEL_PATH` and use a pre-downloaded model (see [OFFLINE.md](OFFLINE.md)).

---

## Run locally

1. With the venv activated:

   ```powershell
   python app.py
   ```

2. Open the URL shown in the terminal (e.g. `http://127.0.0.1:7860`).

**Model:** If the model is already in a `models_cache` folder at the project root, the app loads from there and does **not** download. Put the Z-Image-Turbo snapshot there (e.g. `huggingface-cli download Tongyi-MAI/Z-Image-Turbo --local-dir models_cache`). To use a different path, set `Z_IMAGE_MODEL_PATH` (e.g. `$env:Z_IMAGE_MODEL_PATH = ".\my_model"`).

---

Configuration reference: https://huggingface.co/docs/hub/spaces-config-reference
