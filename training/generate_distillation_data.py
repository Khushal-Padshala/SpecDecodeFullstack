"""
Distillation Data Generation Pipeline.
Prompts the Target Model across diverse domains and records prompts, completions,
and token sequences into data/distillation_corpus.jsonl.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import argparse
from typing import Optional
from tqdm import tqdm

from training.prompt_templates import get_training_prompts
from backend.models.target_loader import TargetModelLoader

def generate_corpus(
    output_path: Path,
    model_path: Optional[str] = None,
    max_tokens: int = 96,
    temperature: float = 0.2,
    limit: Optional[int] = None
):
    """Runs the target teacher model to generate rich distillation data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompts = get_training_prompts()
    if limit:
        prompts = prompts[:limit]

    print(f"[*] Initializing Target Teacher Model...")
    target_loader = TargetModelLoader(model_path=model_path, n_ctx=2048, n_threads=8)
    target_loader.load()

    print(f"[*] Generating distillation corpus for {len(prompts)} prompts...")
    records = []

    with open(output_path, "w", encoding="utf-8") as f:
        for item in tqdm(prompts, desc="Generating Distillation Pairs"):
            prompt_text = item["prompt"]
            domain = item["domain"]
            
            start_time = time.time()
            completion, token_ids, elapsed, tok_per_sec = target_loader.generate_baseline(
                prompt=prompt_text,
                max_tokens=max_tokens,
                temperature=temperature
            )

            record = {
                "domain": domain,
                "prompt": prompt_text,
                "completion": completion,
                "token_ids": token_ids,
                "num_tokens": len(token_ids),
                "generation_time_s": round(elapsed, 3),
                "timestamp": time.time()
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            records.append(record)

    print(f"[OK] Distillation corpus generated at: {output_path}")
    print(f"     Total records: {len(records)}")
    total_toks = sum(r["num_tokens"] for r in records)
    print(f"     Total tokens captured: {total_toks:,}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Distillation Corpus")
    parser.add_argument("--output", type=str, default="data/distillation_corpus.jsonl", help="Output JSONL path")
    parser.add_argument("--model-path", type=str, default=None, help="Target GGUF model path")
    parser.add_argument("--max-tokens", type=int, default=96, help="Max tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts")
    args = parser.parse_args()

    generate_corpus(
        output_path=Path(args.output),
        model_path=args.model_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        limit=args.limit
    )
