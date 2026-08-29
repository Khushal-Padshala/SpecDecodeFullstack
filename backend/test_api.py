"""
Comprehensive FastAPI Backend Endpoint Test Suite.
Tests /health, /metrics, /generate (arbitrary prompt), and /generate/stream (SSE).
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from fastapi.testclient import TestClient
from backend.app import app

def test_api_endpoints():
    print("=" * 80)
    print("           FASTAPI BACKEND ENDPOINTS VALIDATION SUITE")
    print("=" * 80)

    with TestClient(app) as client:
        # 1. Test /health
        print("\n[1] Testing GET /health ...")
        resp = client.get("/health")
        print(f"Status Code: {resp.status_code}")
        print(f"Payload: {json.dumps(resp.json(), indent=2)}")
        assert resp.status_code == 200
        health = resp.json()
        assert health["target_model_loaded"] is True
        assert health["draft_model_loaded"] is True

        # 2. Test /generate with arbitrary prompt (Baseline mode)
        print("\n[2] Testing POST /generate (Baseline Mode) ...")
        payload = {
            "prompt": "What is the capital of France?",
            "max_tokens": 16,
            "temperature": 0.0,
            "use_speculative": False
        }
        resp = client.post("/generate", json=payload)
        print(f"Status Code: {resp.status_code}")
        data = resp.json()
        print(f"Generated text: {data['text']}")
        print(f"Tokens: {data['tokens_generated']}, Tok/s: {data['tokens_per_sec']}")
        assert resp.status_code == 200
        assert data["tokens_generated"] > 0

        # 3. Test /generate with arbitrary prompt (Speculative mode)
        print("\n[3] Testing POST /generate (Speculative Mode) ...")
        payload = {
            "prompt": "Write a one-sentence python function to add two numbers.",
            "max_tokens": 24,
            "temperature": 0.0,
            "use_speculative": True,
            "num_draft_tokens": 4
        }
        resp = client.post("/generate", json=payload)
        print(f"Status Code: {resp.status_code}")
        data = resp.json()
        print(f"Generated text: {data['text']}")
        print(f"Speculative stats: {json.dumps(data['speculative_stats'], indent=2)}")
        assert resp.status_code == 200
        assert data["speculative_stats"]["enabled"] is True

        # 4. Test /generate/stream (SSE streaming)
        print("\n[4] Testing POST /generate/stream (SSE Streaming) ...")
        payload = {
            "prompt": "Say hello in 3 languages:",
            "max_tokens": 16,
            "temperature": 0.0,
            "use_speculative": True,
            "num_draft_tokens": 3
        }
        resp = client.post("/generate/stream", json=payload)
        print(f"Status Code: {resp.status_code}")
        print(f"SSE Content-Type: {resp.headers.get('content-type')}")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type")

        # 5. Test /metrics
        print("\n[5] Testing GET /metrics ...")
        resp = client.get("/metrics")
        print(f"Status Code: {resp.status_code}")
        print(f"Metrics: {json.dumps(resp.json(), indent=2)}")
        assert resp.status_code == 200
        metrics = resp.json()
        assert metrics["total_requests"] >= 2

        # 6. Test Error Handling (Empty prompt)
        print("\n[6] Testing Error Handling (Empty Prompt) ...")
        resp = client.post("/generate", json={"prompt": ""})
        print(f"Status Code: {resp.status_code} (Expected 400)")
        assert resp.status_code == 400

    print("\n" + "=" * 80)
    print("[PHASE D SUCCESS] All FastAPI Endpoints, SSE Streaming, Health, & Metrics fully verified!")
    print("=" * 80)

if __name__ == "__main__":
    test_api_endpoints()
