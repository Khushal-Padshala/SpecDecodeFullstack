"""
Performance and speculative metric accounting engine.
"""

import time
import threading
from typing import Dict, List, Any
from backend.schemas import MetricsResponse, SpeculativeStats

class MetricsTracker:
    """Thread-safe cumulative metrics tracker."""
    def __init__(self):
        self._lock = threading.Lock()
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.baseline_requests = 0
        self.speculative_requests = 0
        
        self.total_baseline_tokens = 0
        self.total_baseline_time = 0.0
        
        self.total_speculative_tokens = 0
        self.total_speculative_time = 0.0
        
        self.total_draft_proposed = 0
        self.total_draft_accepted = 0
        self.total_draft_rejected = 0
        
        self.recent_latencies: List[float] = []

    def record_request(
        self,
        tokens_generated: int,
        wall_clock_time: float,
        speculative_stats: SpeculativeStats = None
    ):
        with self._lock:
            self.total_requests += 1
            self.total_tokens_generated += tokens_generated
            self.recent_latencies.append(wall_clock_time)
            if len(self.recent_latencies) > 50:
                self.recent_latencies.pop(0)

            if speculative_stats and speculative_stats.enabled:
                self.speculative_requests += 1
                self.total_speculative_tokens += tokens_generated
                self.total_speculative_time += wall_clock_time
                self.total_draft_proposed += speculative_stats.draft_tokens_proposed
                self.total_draft_accepted += speculative_stats.draft_tokens_accepted
                self.total_draft_rejected += speculative_stats.draft_tokens_rejected
            else:
                self.baseline_requests += 1
                self.total_baseline_tokens += tokens_generated
                self.total_baseline_time += wall_clock_time

    def get_summary(self) -> MetricsResponse:
        with self._lock:
            avg_base_toks = (
                self.total_baseline_tokens / self.total_baseline_time
                if self.total_baseline_time > 0 else 0.0
            )
            avg_spec_toks = (
                self.total_speculative_tokens / self.total_speculative_time
                if self.total_speculative_time > 0 else 0.0
            )
            speedup = (avg_spec_toks / avg_base_toks) if (avg_base_toks > 0 and avg_spec_toks > 0) else 1.0
            acceptance_rate = (
                self.total_draft_accepted / self.total_draft_proposed
                if self.total_draft_proposed > 0 else 0.0
            )

            return MetricsResponse(
                total_requests=self.total_requests,
                total_tokens_generated=self.total_tokens_generated,
                baseline_requests=self.baseline_requests,
                speculative_requests=self.speculative_requests,
                avg_baseline_tok_per_sec=round(avg_base_toks, 2),
                avg_speculative_tok_per_sec=round(avg_spec_toks, 2),
                overall_speedup=round(speedup, 2),
                overall_acceptance_rate=round(acceptance_rate, 4),
                recent_latencies=[round(l, 3) for l in self.recent_latencies[-10:]]
            )

# Global tracker instance
global_metrics = MetricsTracker()
