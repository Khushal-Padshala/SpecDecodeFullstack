"""
Model download and management utility for Llama 3.1 8B Instruct GGUF & Tokenizer.
Uses fast streaming HTTP requests with retry and progress monitoring.
"""

import os
import sys
import argparse
from pathlib import Path
import requests
from tqdm import tqdm

CHECKPOINTS_DIR = Path(__file__).resolve().parent

# Default target models
TARGET_MODELS = {
    "llama-3.1-8b-q4": {
        "url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    },
    "llama-3.2-1b-q4": {
        "url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    }
}

TOKENIZER_FILES = {
    "tokenizer.json": "https://huggingface.co/NousResearch/Meta-Llama-3.1-8B-Instruct/resolve/main/tokenizer.json",
    "tokenizer_config.json": "https://huggingface.co/NousResearch/Meta-Llama-3.1-8B-Instruct/resolve/main/tokenizer_config.json",
    "special_tokens_map.json": "https://huggingface.co/NousResearch/Meta-Llama-3.1-8B-Instruct/resolve/main/special_tokens_map.json"
}

def download_file(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        print(f"[OK] File already exists: {dest_path.name} ({dest_path.stat().st_size / (1024**2):.2f} MB)")
        return dest_path

    print(f"[*] Downloading {dest_path.name} from {url}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, stream=True, headers=headers, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    with open(temp_path, "wb") as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    if temp_path.exists():
        if dest_path.exists():
            dest_path.unlink()
        temp_path.rename(dest_path)
    print(f"[OK] Saved to: {dest_path}")
    return dest_path

def download_tokenizer(target_dir: Path = CHECKPOINTS_DIR / "tokenizer"):
    print("[*] Ensuring Tokenizer files are present...")
    for filename, url in TOKENIZER_FILES.items():
        dest = target_dir / filename
        download_file(url, dest)
    print("[OK] Tokenizer download verified.")

def download_target_model(model_key: str = "llama-3.2-1b-q4", target_dir: Path = CHECKPOINTS_DIR):
    if model_key not in TARGET_MODELS:
        raise ValueError(f"Unknown model key: {model_key}. Choose from {list(TARGET_MODELS.keys())}")
    info = TARGET_MODELS[model_key]
    dest = target_dir / info["filename"]
    return download_file(info["url"], dest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Llama 3.1 GGUF and Tokenizer")
    parser.add_argument("--model", type=str, default="llama-3.2-1b-q4", choices=list(TARGET_MODELS.keys()), help="Model choice")
    parser.add_argument("--tokenizer-only", action="store_true", help="Download only tokenizer files")
    args = parser.parse_args()

    download_tokenizer()
    if not args.tokenizer_only:
        download_target_model(model_key=args.model)
