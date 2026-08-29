"""
Interactive CLI Demo for Speculative Decoding Backend.
Allows typing any arbitrary prompt and viewing real-time generation with speculative metrics.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from backend.models.target_loader import TargetModelLoader
from backend.models.draft_model import DraftModelManager
from backend.engine import SpeculativeEngine
from backend.schemas import GenerateRequest

def main():
    parser = argparse.ArgumentParser(description="Interactive Speculative Decoding Demo")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt to run")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens to generate")
    parser.add_argument("--draft-k", type=int, default=2, help="Draft tokens per cycle (K)")
    parser.add_argument("--stream", action="store_true", help="Stream tokens in real-time with source tags")
    args = parser.parse_args()

    print("=" * 80)
    print("       SPECULATIVE DECODING INTERACTIVE DEMO (CPU / Intel Iris Xe)")
    print("=" * 80)

    # Initialize models
    print("\n[*] Loading models into RAM...")
    target_loader = TargetModelLoader(n_ctx=2048, n_threads=8)
    target_loader.load()

    draft_loader = DraftModelManager(checkpoint_path="checkpoints/draft_model.pt")
    draft_loader.load()

    engine = SpeculativeEngine(target_model=target_loader, draft_model=draft_loader)
    print("[OK] Ready!\n")

    def run_prompt(prompt_text: str):
        print("-" * 80)
        print(f"PROMPT: {prompt_text}")
        print("-" * 80)

        # 1. Run Baseline
        print("\n[1] Running Standard Target Baseline (No Speculative)...")
        base_req = GenerateRequest(prompt=prompt_text, max_tokens=args.max_tokens, temperature=0.0, use_speculative=False)
        base_resp = engine.generate(base_req)
        print(f"Output:\n{base_resp.text}\n")
        print(f"--> Baseline Speed: {base_resp.tokens_per_sec:.2f} tok/s ({base_resp.tokens_generated} tokens in {base_resp.wall_clock_time_s:.2f}s)")

        # 2. Run Speculative Decoding
        print("\n[2] Running Speculative Decoding (Draft Model K=" + str(args.draft_k) + ")...")
        spec_req = GenerateRequest(prompt=prompt_text, max_tokens=args.max_tokens, temperature=0.0, use_speculative=True, num_draft_tokens=args.draft_k)
        
        if args.stream:
            print("Streaming Output [tags: [D]=Draft Accepted, [T]=Target]:")
            for chunk in engine.generate_stream(spec_req):
                if chunk.source == "draft_accepted":
                    sys.stdout.write(f"\033[92m{chunk.token}\033[0m") # Green
                elif chunk.source == "target_correction":
                    sys.stdout.write(f"\033[94m{chunk.token}\033[0m") # Blue
                elif chunk.source == "done":
                    print(f"\n\nStats: {chunk.stats}")
                sys.stdout.flush()
        else:
            spec_resp = engine.generate(spec_req)
            print(f"Output:\n{spec_resp.text}\n")
            stats = spec_resp.speculative_stats
            print(f"--> Speculative Speed:     {spec_resp.tokens_per_sec:.2f} tok/s ({spec_resp.tokens_generated} tokens in {spec_resp.wall_clock_time_s:.2f}s)")
            print(f"--> Draft Acceptance Rate: {stats.acceptance_rate * 100:.1f}% ({stats.draft_tokens_accepted} accepted / {stats.draft_tokens_proposed} proposed)")
            print(f"--> Target Passes:         {stats.target_forward_passes} forward passes (vs {spec_resp.tokens_generated} baseline passes)")
            print(f"--> Theoretical Speedup:   {stats.theoretical_speedup:.2f}x")
        print("=" * 80)

    if args.prompt:
        run_prompt(args.prompt)
    else:
        print("Tip: You are in Interactive Mode. Type any prompt and press Enter (or type 'exit' to quit).\n")
        while True:
            try:
                user_p = input("\nEnter your prompt >> ").strip()
                if not user_p:
                    continue
                if user_p.lower() in ["exit", "quit", "q"]:
                    print("Exiting demo. Goodbye!")
                    break
                run_prompt(user_p)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting demo. Goodbye!")
                break

if __name__ == "__main__":
    main()
