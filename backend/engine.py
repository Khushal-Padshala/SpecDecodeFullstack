"""
Speculative Decoding Engine & Autoregressive Baseline Engine.
Implements the multi-token propose-and-verify speculative loop with metric collection.
"""

import time
from typing import List, Generator, Tuple, Optional, Dict, Any
from backend.models.target_loader import TargetModelLoader
from backend.models.draft_model import DraftModelManager
from backend.schemas import (
    GenerateRequest,
    GenerateResponse,
    SpeculativeStats,
    StreamChunk
)
from backend.metrics import global_metrics

class SpeculativeEngine:
    """Core Speculative Decoding Inference Engine."""
    def __init__(
        self,
        target_model: TargetModelLoader,
        draft_model: DraftModelManager,
        eos_token_ids: Optional[List[int]] = None
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.eos_token_ids = eos_token_ids or [128000, 128001, 128009]

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        """Executes full speculative or baseline generation for arbitrary user prompts."""
        t_start = time.perf_counter()
        
        prompt_tokens = self.target_model.tokenize(req.prompt, add_bos=True)
        if not prompt_tokens:
            prompt_tokens = [128000]

        if not req.use_speculative:
            # Baseline target-only generation
            text, tokens, elapsed, tok_per_sec = self.target_model.generate_baseline(
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature
            )
            response = GenerateResponse(
                text=text,
                tokens_generated=len(tokens),
                prompt_tokens=len(prompt_tokens),
                wall_clock_time_s=round(elapsed, 4),
                tokens_per_sec=round(tok_per_sec, 2),
                speculative_stats=SpeculativeStats(enabled=False)
            )
            global_metrics.record_request(len(tokens), elapsed, response.speculative_stats)
            return response

        # Speculative Decoding Execution Loop
        prefix = list(prompt_tokens)
        generated_tokens: List[int] = []
        
        total_proposed = 0
        total_accepted = 0
        total_rejected = 0
        target_passes = 0
        total_draft_time = 0.0
        total_target_time = 0.0

        while len(generated_tokens) < req.max_tokens:
            k = min(req.num_draft_tokens, req.max_tokens - len(generated_tokens))
            draft_candidates, d_time = self.draft_model.propose_tokens(
                prefix_token_ids=prefix,
                num_draft_tokens=k,
                temperature=req.temperature
            )
            total_proposed += len(draft_candidates)
            total_draft_time += d_time

            accepted, correction, rejected, t_time = self.target_model.verify_draft_candidates(
                prefix_tokens=prefix,
                draft_candidates=draft_candidates,
                temperature=req.temperature
            )
            target_passes += 1
            total_target_time += t_time
            total_accepted += len(accepted)
            total_rejected += len(rejected)

            should_stop = False
            for tok in accepted:
                generated_tokens.append(tok)
                prefix.append(tok)
                if tok in self.eos_token_ids or len(generated_tokens) >= req.max_tokens:
                    should_stop = True
                    break

            if should_stop:
                break

            if correction is not None and len(generated_tokens) < req.max_tokens:
                generated_tokens.append(correction)
                prefix.append(correction)
                if correction in self.eos_token_ids:
                    break

        total_wall_clock = time.perf_counter() - t_start
        tok_per_sec = len(generated_tokens) / total_wall_clock if total_wall_clock > 0 else 0.0
        acceptance_rate = total_accepted / total_proposed if total_proposed > 0 else 0.0
        theoretical_speedup = (len(generated_tokens) / target_passes) if target_passes > 0 else 1.0

        stats = SpeculativeStats(
            enabled=True,
            num_draft_tokens_k=req.num_draft_tokens,
            draft_tokens_proposed=total_proposed,
            draft_tokens_accepted=total_accepted,
            draft_tokens_rejected=total_rejected,
            acceptance_rate=round(acceptance_rate, 4),
            target_forward_passes=target_passes,
            draft_time_s=round(total_draft_time, 4),
            target_time_s=round(total_target_time, 4),
            theoretical_speedup=round(theoretical_speedup, 2)
        )

        output_text = self.target_model.detokenize(generated_tokens)
        response = GenerateResponse(
            text=output_text,
            tokens_generated=len(generated_tokens),
            prompt_tokens=len(prompt_tokens),
            wall_clock_time_s=round(total_wall_clock, 4),
            tokens_per_sec=round(tok_per_sec, 2),
            speculative_stats=stats
        )

        global_metrics.record_request(len(generated_tokens), total_wall_clock, stats)
        return response

    def generate_stream(self, req: GenerateRequest) -> Generator[StreamChunk, None, None]:
        """Yields token chunks in real-time with source tags for frontend rendering."""
        t_start = time.perf_counter()
        prompt_tokens = self.target_model.tokenize(req.prompt, add_bos=True)
        if not prompt_tokens:
            prompt_tokens = [128000]

        if not req.use_speculative:
            text, tokens, elapsed, tok_per_sec = self.target_model.generate_baseline(
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature
            )
            for tok in tokens:
                yield StreamChunk(
                    token=self.target_model.detokenize([tok]),
                    token_id=tok,
                    source="target_baseline"
                )
            yield StreamChunk(
                token="",
                token_id=-1,
                source="done",
                is_final=True,
                stats={"tokens_generated": len(tokens), "tokens_per_sec": round(tok_per_sec, 2)}
            )
            return

        prefix = list(prompt_tokens)
        generated_tokens: List[int] = []
        
        total_proposed = 0
        total_accepted = 0
        total_rejected = 0
        target_passes = 0

        while len(generated_tokens) < req.max_tokens:
            k = min(req.num_draft_tokens, req.max_tokens - len(generated_tokens))
            draft_candidates, _ = self.draft_model.propose_tokens(
                prefix_token_ids=prefix,
                num_draft_tokens=k,
                temperature=req.temperature
            )
            total_proposed += len(draft_candidates)

            accepted, correction, rejected, _ = self.target_model.verify_draft_candidates(
                prefix_tokens=prefix,
                draft_candidates=draft_candidates,
                temperature=req.temperature
            )
            target_passes += 1
            total_accepted += len(accepted)
            total_rejected += len(rejected)

            should_stop = False
            for tok in accepted:
                generated_tokens.append(tok)
                prefix.append(tok)
                yield StreamChunk(
                    token=self.target_model.detokenize([tok]),
                    token_id=tok,
                    source="draft_accepted"
                )
                if tok in self.eos_token_ids or len(generated_tokens) >= req.max_tokens:
                    should_stop = True
                    break

            if should_stop:
                break

            if correction is not None and len(generated_tokens) < req.max_tokens:
                generated_tokens.append(correction)
                prefix.append(correction)
                yield StreamChunk(
                    token=self.target_model.detokenize([correction]),
                    token_id=correction,
                    source="target_correction"
                )
                if correction in self.eos_token_ids:
                    break

        total_wall_clock = time.perf_counter() - t_start
        tok_per_sec = len(generated_tokens) / total_wall_clock if total_wall_clock > 0 else 0.0
        acc_rate = total_accepted / total_proposed if total_proposed > 0 else 0.0

        final_stats = {
            "tokens_generated": len(generated_tokens),
            "wall_clock_time_s": round(total_wall_clock, 4),
            "tokens_per_sec": round(tok_per_sec, 2),
            "acceptance_rate": round(acc_rate, 4),
            "target_forward_passes": target_passes
        }
        
        yield StreamChunk(
            token="",
            token_id=-1,
            source="done",
            is_final=True,
            stats=final_stats
        )
