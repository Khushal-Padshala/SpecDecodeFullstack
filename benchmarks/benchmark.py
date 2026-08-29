"""
Speculative Decoding Benchmark Suite.
Runs a side-by-side performance evaluation between Standard Target Autoregressive Baseline
and Speculative Decoding across varied test prompts.
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
from typing import List, Dict, Any

from backend.models.target_loader import TargetModelLoader
from backend.models.draft_model import DraftModelManager
from backend.engine import SpeculativeEngine
from backend.schemas import GenerateRequest

BENCHMARK_PROMPTS = [
    {
        "category": "Code Generation",
        "prompt": "Write a Python function to compute the Fibonacci sequence using dynamic programming with memoization."
    },
    {
        "category": "Algorithmic Reasoning",
        "prompt": "Explain step-by-step why binary search has O(log n) time complexity compared to O(n) linear search."
    },
    {
        "category": "Summarization",
        "prompt": "Summarize the three core advantages of using speculative decoding for LLM inference on edge hardware."
    },
    {
        "category": "Conversational Q&A",
        "prompt": "Explain the concept of key-value caching in transformer models to a high school computer science student."
    },
    {
        "category": "Creative Writing",
        "prompt": "Write a short paragraph describing a futuristic city powered by quantum computing nodes."
    }
]

def run_benchmark(
    target_model_path: str = None,
    draft_model_path: str = None,
    max_tokens: int = 128,
    num_draft_tokens: int = 4,
    temperature: float = 0.0
):
    print("=" * 80)
    print("      SPECULATIVE DECODING BENCHMARK: BASELINE vs SPECULATIVE DECODING")
    print("=" * 80)

    # 1. Initialize models
    print("\n[*] Initializing Target Model...")
    target_loader = TargetModelLoader(model_path=target_model_path)
    target_loader.load()

    print("\n[*] Initializing Draft Model...")
    draft_loader = DraftModelManager(checkpoint_path=draft_model_path)
    draft_loader.load()

    engine = SpeculativeEngine(target_model=target_loader, draft_model=draft_loader)
    results = []

    print("\n[*] Starting benchmark evaluation runs...\n")
    print(f"{'Category':<22} | {'Mode':<11} | {'Tokens':<6} | {'Time (s)':<8} | {'Tok/s':<7} | {'Acc. Rate':<9} | {'Speedup':<7}")
    print("-" * 80)

    for item in BENCHMARK_PROMPTS:
        category = item["category"]
        prompt = item["prompt"]

        # Run Baseline
        base_req = GenerateRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            use_speculative=False
        )
        base_resp = engine.generate(base_req)

        # Run Speculative
        spec_req = GenerateRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            use_speculative=True,
            num_draft_tokens=num_draft_tokens
        )
        spec_resp = engine.generate(spec_req)

        speedup = spec_resp.tokens_per_sec / base_resp.tokens_per_sec if base_resp.tokens_per_sec > 0 else 1.0
        acc_rate_str = f"{spec_resp.speculative_stats.acceptance_rate * 100:.1f}%"

        print(f"{category:<22} | {'Baseline':<11} | {base_resp.tokens_generated:<6} | {base_resp.wall_clock_time_s:<8.3f} | {base_resp.tokens_per_sec:<7.2f} | {'N/A':<9} | {'1.00x':<7}")
        print(f"{category:<22} | {'Speculative':<11} | {spec_resp.tokens_generated:<6} | {spec_resp.wall_clock_time_s:<8.3f} | {spec_resp.tokens_per_sec:<7.2f} | {acc_rate_str:<9} | {speedup:<6.2f}x")
        print("-" * 80)

        results.append({
            "category": category,
            "prompt": prompt,
            "baseline": {
                "tokens": base_resp.tokens_generated,
                "time_s": base_resp.wall_clock_time_s,
                "tok_per_sec": base_resp.tokens_per_sec,
                "text": base_resp.text
            },
            "speculative": {
                "tokens": spec_resp.tokens_generated,
                "time_s": spec_resp.wall_clock_time_s,
                "tok_per_sec": spec_resp.tokens_per_sec,
                "acceptance_rate": spec_resp.speculative_stats.acceptance_rate,
                "draft_proposed": spec_resp.speculative_stats.draft_tokens_proposed,
                "draft_accepted": spec_resp.speculative_stats.draft_tokens_accepted,
                "draft_rejected": spec_resp.speculative_stats.draft_tokens_rejected,
                "target_forward_passes": spec_resp.speculative_stats.target_forward_passes,
                "speedup": round(speedup, 2),
                "text": spec_resp.text
            }
        })

    # Save summary
    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    avg_speedup = sum(r["speculative"]["speedup"] for r in results) / len(results)
    avg_acc = sum(r["speculative"]["acceptance_rate"] for r in results) / len(results)
    
    print("\n" + "=" * 80)
    print(f"BENCHMARK SUMMARY:")
    print(f"  Average Empirical Speedup : {avg_speedup:.2f}x")
    print(f"  Average Draft Accept Rate : {avg_acc * 100:.1f}%")
    print(f"  Detailed log saved to     : benchmarks/benchmark_results.json")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Speculative Decoding Benchmark Suite")
    parser.add_argument("--target-model", type=str, default=None, help="Target GGUF model path")
    parser.add_argument("--draft-model", type=str, default=None, help="Draft model checkpoint path")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens per generation")
    parser.add_argument("--draft-k", type=int, default=4, help="Draft tokens per cycle (K)")
    args = parser.parse_args()

    run_benchmark(
        target_model_path=args.target_model,
        draft_model_path=args.draft_model,
        max_tokens=args.max_tokens,
        num_draft_tokens=args.draft_k
    )
