# Speculative Decoding Backend (Llama 3.1 8B Instruct + Custom Draft Model)

A production-grade, end-to-end speculative decoding inference engine and FastAPI service engineered specifically for edge/CPU hardware constraints (**12th Gen Intel Core i7 / Iris Xe GPU, 16GB RAM**).

---

## 🚀 Key Features

- **Generic Prompt Inference**: Accepts arbitrary prompts via REST API and SSE streaming without hardcoded templates or prompt allow-lists.
- **Quantized Target Model**: Runs `Llama-3.1-8B-Instruct` quantized to GGUF (`Q4_K_M`, ~4.9GB RAM) on CPU via `llama-cpp-python` with AVX2 acceleration and batch verification.
- **Custom Distilled Draft Model**: A custom-designed ~75M parameter causal transformer decoder built from scratch with exact Meta Llama 3.1 tokenization (128,256 vocabulary) and tied embedding layers for ultra-low memory (~150MB) and near-instant CPU execution (~1ms/token).
- **Knowledge Distillation Pipeline**: Dedicated training suite (`generate_distillation_data.py`, `train_draft_model.py`, `export_draft_model.py`) generating diverse datasets across 5 domains (Code, Reasoning, Chat, Summarization, Creative).
- **Real-Time Token Streaming**: Server-Sent Events (`POST /generate/stream`) tagging each token as `draft_accepted` or `target_correction` for visual frontend verification.
- **Granular Metrics & Benchmarks**: Real-time tracking of tokens/sec, draft acceptance rate ($\alpha$), target forward passes, and theoretical & empirical speedup multipliers.

---

## 📐 Draft Model Architecture Specification

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Vocabulary Size** | `128,256` | Identical token IDs to Meta Llama 3.1 8B Instruct |
| **Decoder Layers** | `5` | Low CPU latency (~1-2 ms per token proposal) |
| **Hidden Dimension ($d_{model}$)** | `512` | Balanced expressiveness and memory bandwidth |
| **Attention Heads** | `8` ($d_{head} = 64$) | Standard causal multi-head self-attention |
| **Feed-Forward ($d_{ff}$)** | `1,536` | SwiGLU non-linear projection |
| **Positional Encoding** | RoPE ($\theta = 500,000$) | Matches Llama 3.1 rotary base |
| **Weight Tying** | Yes (`lm_head` $\equiv$ `embed_tokens`) | Saves 65M duplicate parameters and reduces RAM |
| **Total Parameters** | **74,964,480 (~75M)** | Ideal size for edge CPU speculative decoding |

---

## 🛠️ Hardware Requirements & Assumptions

- **Operating System**: Windows 10/11 x64, Linux, or macOS.
- **CPU**: Intel Core i5/i7/i9 or AMD Ryzen with AVX2 support.
- **Memory**: 16 GB RAM (allocates ~5.0 GB for target model + KV cache + draft model).
- **Batch Size**: 1 (optimized for single-request interactive low-latency generation).

---

## 📦 Setup & Installation

### 1. Create Virtual Environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Obtain Target Model & Tokenizer
Download the default `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` checkpoint and tokenizer files:
```bash
python checkpoints/download_target.py
```

---

## 🧠 Distillation & Training Pipeline (Training-Time)

The draft model is trained to mimic the output distribution of the Llama 3.1 8B target model across diverse domains.

### Step 1: Generate Distillation Corpus
Prompts the local 8B teacher model across 5 varied domains:
```bash
python training/generate_distillation_data.py --output data/distillation_corpus.jsonl --max-tokens 256
```

### Step 2: Train Student Draft Model
Trains the custom transformer on the distillation corpus:
```bash
python training/train_draft_model.py --data data/distillation_corpus.jsonl --output checkpoints/draft_model.pt --epochs 5 --lr 3e-4
```

### Step 3: Package & Export Model
```bash
python training/export_draft_model.py --output checkpoints/draft_model.pt
```

---

## 🌐 Running the FastAPI Backend (Inference-Time)

Start the production server on port 8000:
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### 1. `POST /generate`
Accepts arbitrary user prompts and returns the generated text alongside speculative decoding metrics.

**Request Body:**
```json
{
  "prompt": "Write a Python function to check if a number is prime.",
  "max_tokens": 128,
  "temperature": 0.0,
  "use_speculative": true,
  "num_draft_tokens": 4
}
```

**Response:**
```json
{
  "text": "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
  "tokens_generated": 48,
  "prompt_tokens": 18,
  "wall_clock_time_s": 1.25,
  "tokens_per_sec": 38.4,
  "speculative_stats": {
    "enabled": true,
    "num_draft_tokens_k": 4,
    "draft_tokens_proposed": 48,
    "draft_tokens_accepted": 36,
    "draft_tokens_rejected": 12,
    "acceptance_rate": 0.75,
    "target_forward_passes": 15,
    "theoretical_speedup": 3.2
  }
}
```

#### 2. `POST /generate/stream`
Streams Server-Sent Events (SSE) token by token with visual source tags (`draft_accepted` vs `target_correction`).

#### 3. `GET /health`
Returns system memory usage, CPU load, and model readiness.

#### 4. `GET /metrics`
Returns aggregate statistics across all requests (lifetime tokens, average speedup, aggregate draft acceptance rate).

---

## 📊 Benchmarking Suite

Run the side-by-side comparison between baseline autoregressive generation and speculative decoding:
```bash
python benchmarks/benchmark.py --max-tokens 128 --draft-k 4
```
