"""
Target Model Loader and Verification Wrapper for Llama-3.1-8B-Instruct (GGUF).
Supports high-throughput single-pass candidate verification and baseline generation on CPU.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
from typing import List, Tuple, Optional, Dict, Any
from llama_cpp import Llama

class TargetModelLoader:
    """Manages Llama 3.1 8B Instruct GGUF model loading, tokenization, and verification."""
    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_threads: int = 8,
        verbose: bool = False
    ):
        self.model_path = model_path or os.getenv("TARGET_MODEL_PATH", "checkpoints/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
        if not Path(self.model_path).exists():
            fallback = Path("checkpoints/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
            if fallback.exists():
                self.model_path = str(fallback)

        self.n_ctx = int(os.getenv("TARGET_N_CTX", n_ctx))
        self.n_threads = int(os.getenv("TARGET_N_THREADS", n_threads))
        self.verbose = verbose
        self.llm: Optional[Llama] = None
        self.is_loaded: bool = False

    def load(self):
        """Loads the quantized GGUF checkpoint into memory."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Target GGUF model not found at '{self.model_path}'. "
                f"Please download it via 'python checkpoints/download_target.py' or provide a valid path."
            )

        print(f"[*] Loading Target Model (GGUF) from: {self.model_path}")
        print(f"    Context Window: {self.n_ctx}, CPU Threads: {self.n_threads}")

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_batch=512,
            logits_all=True, # Required for multi-token candidate verification in a single forward pass
            verbose=self.verbose
        )
        self.is_loaded = True
        print(f"[OK] Target Model loaded successfully into memory.")

    def tokenize(self, text: str, add_bos: bool = True) -> List[int]:
        """Encodes text to token IDs."""
        if not self.is_loaded or self.llm is None:
            raise RuntimeError("Target model not loaded.")
        return self.llm.tokenize(text.encode("utf-8"), add_bos=add_bos)

    def detokenize(self, tokens: List[int]) -> str:
        """Decodes token IDs back to string."""
        if not self.is_loaded or self.llm is None:
            raise RuntimeError("Target model not loaded.")
        return self.llm.detokenize(tokens).decode("utf-8", errors="replace")

    def generate_baseline(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0
    ) -> Tuple[str, List[int], float, float]:
        """
        Standard autoregressive baseline generation using only the target model.
        Returns (generated_text, token_ids, elapsed_seconds, tokens_per_second).
        """
        if not self.is_loaded or self.llm is None:
            raise RuntimeError("Target model not loaded.")

        t0 = time.perf_counter()
        output = self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )
        elapsed = time.perf_counter() - t0

        text = output["choices"][0]["text"]
        tokens = self.tokenize(text, add_bos=False)
        tok_per_sec = len(tokens) / elapsed if elapsed > 0 else 0.0

        return text, tokens, elapsed, tok_per_sec

    def verify_draft_candidates(
        self,
        prefix_tokens: List[int],
        draft_candidates: List[int],
        temperature: float = 0.0
    ) -> Tuple[List[int], Optional[int], List[int], float]:
        """
        Verifies K proposed draft candidates against the target model in a SINGLE forward evaluation.
        
        Returns:
            - accepted_draft_tokens: List of draft tokens accepted by the target model
            - correction_token: The target model's corrected token (or bonus token if all draft tokens accepted)
            - rejected_draft_tokens: List of draft tokens that were rejected
            - target_time_s: Time taken for the single target model evaluation
        """
        if not self.is_loaded or self.llm is None:
            raise RuntimeError("Target model not loaded.")

        t0 = time.perf_counter()
        k = len(draft_candidates)
        if k == 0:
            return [], None, [], 0.0

        full_seq = prefix_tokens + draft_candidates
        self.llm.reset()
        self.llm.eval(full_seq)

        accepted: List[int] = []
        rejected: List[int] = []
        correction: Optional[int] = None

        prefix_len = len(prefix_tokens)
        
        for i in range(k):
            logit_idx = prefix_len - 1 + i
            target_token = self.llm.sample(idx=logit_idx, temp=temperature)
            candidate = draft_candidates[i]

            if target_token == candidate:
                accepted.append(candidate)
            else:
                correction = target_token
                rejected = draft_candidates[i:]
                break

        # If all k candidate tokens were accepted, sample the bonus token
        if len(accepted) == k:
            bonus_logit_idx = prefix_len + k - 1
            correction = self.llm.sample(idx=bonus_logit_idx, temp=temperature)

        elapsed = time.perf_counter() - t0
        return accepted, correction, rejected, elapsed
