"""
Pydantic data schemas for Speculative Decoding Backend API.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Arbitrary prompt text from user")
    max_tokens: int = Field(default=128, ge=1, le=2048, description="Maximum new tokens to generate")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature (0.0 for greedy)")
    use_speculative: bool = Field(default=True, description="Enable speculative decoding with draft model")
    num_draft_tokens: int = Field(default=2, ge=1, le=16, description="Number of tokens proposed by draft model per cycle (K)")

class SpeculativeStats(BaseModel):
    enabled: bool = Field(default=True)
    num_draft_tokens_k: int = Field(default=2)
    draft_tokens_proposed: int = Field(default=0)
    draft_tokens_accepted: int = Field(default=0)
    draft_tokens_rejected: int = Field(default=0)
    acceptance_rate: float = Field(default=0.0)
    target_forward_passes: int = Field(default=0)
    draft_time_s: float = Field(default=0.0)
    target_time_s: float = Field(default=0.0)
    theoretical_speedup: float = Field(default=1.0)

class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    prompt_tokens: int
    wall_clock_time_s: float
    tokens_per_sec: float
    speculative_stats: Optional[SpeculativeStats] = None

class StreamChunk(BaseModel):
    token: str
    token_id: int
    source: str # "draft_accepted", "target_correction", "target_baseline"
    is_final: bool = False
    stats: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    target_model_loaded: bool
    draft_model_loaded: bool
    target_model_path: Optional[str] = None
    draft_params: Optional[int] = None
    context_window: int
    memory_used_mb: float
    memory_total_mb: float
    cpu_percent: float

class MetricsResponse(BaseModel):
    total_requests: int
    total_tokens_generated: int
    baseline_requests: int
    speculative_requests: int
    avg_baseline_tok_per_sec: float
    avg_speculative_tok_per_sec: float
    overall_speedup: float
    overall_acceptance_rate: float
    recent_latencies: List[float] = []
