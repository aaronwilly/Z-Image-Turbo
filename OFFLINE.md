# Running Z-Image-Turbo fully offline

## 1. While online (one-time)

### Download Python wheels

From the project root, with your venv activated:

```powershell
.\download_wheels.ps1
```

This creates a `wheels` folder with all dependencies. Copy the whole project (including `wheels` and `requirements-offline.txt`) to the offline machine.

### Download the model (one-time)

On a machine with internet:

```bash
pip install huggingface_hub
huggingface-cli download Tongyi-MAI/Z-Image-Turbo --local-dir ./model/Z-Image-Turbo
```

Copy the `model` folder to the offline machine (e.g. next to `app.py`).

## 2. On the offline machine

### Install from wheels (no network)

```powershell
.\install_offline.ps1
```

### Run the app with local model

Set the model path to your local copy, then run:

```powershell
$env:Z_IMAGE_MODEL_PATH = ".\model\Z-Image-Turbo"   # or full path
python app.py
```

The app uses `Z_IMAGE_MODEL_PATH` when set and loads with `local_files_only=True`, so no network is used.

## Files used for offline

| File / folder | Purpose |
|---------------|--------|
| `requirements-offline.txt` | PyPI-only deps (no git URLs) for wheel download |
| `wheels/` | Downloaded wheels (from `download_wheels.ps1`) |
| `download_wheels.ps1` | Run once online to fill `wheels/` |
| `install_offline.ps1` | Install from `wheels/` without network |
| `Z_IMAGE_MODEL_PATH` | Env var pointing to local model dir for offline run |

## Note

- Use the **same OS and Python version** (and CUDA if needed) when running `download_wheels.ps1` and on the offline machine, so the wheels are compatible.
- For PyTorch with CUDA, run the download on a machine that matches the offline one (e.g. same Windows + CUDA version) so the right `torch` wheel is pulled.
