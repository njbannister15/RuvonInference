# Project RUVON INFERENCE

We are building an educational inference engine from scratch — a miniature but real system that can load pretrained transformer models (starting with GPT-2 124M → scaling to GPT-2 XL 1.5B), tokenize user prompts, run efficient prefill and incremental decode passes, and serve generations over an HTTP API with streaming, batching, and telemetry. The core focus is not training new models, but serving existing ones efficiently: implementing a paged KV-cache allocator for memory reuse, a continuous batching scheduler so new requests can join mid-flight, and a Triton-based fused decode kernel for speed.

Around this runtime we’ll build a FastAPI service layer with OpenAI-like endpoints, sampling strategies (greedy, top-k, nucleus, temperature), speculative decoding for acceleration, and quantization so larger models can fit into limited GPU memory.

Each part we’ll cut a vertical slice — from tokenizer → kernel → scheduler → streaming client → cost telemetry — and publish a demo plus a short article explaining the “what, why, and how.” By Part 20, the engine will be able to serve GPT-2 XL interactively with multi-user batching, real-time metrics, and cost dashboards: a functioning, investor-ready showcase of how to build an inference-optimized LLM runtime in just four weeks.


# 🚀 20-Part Educational Inference Engine Plan

## Phase 1 — Build a Vertical Slice

**Part 1 — Tokenizer + Single Forward Pass** ✅ **COMPLETE**
* **What:** Load GPT-2 124M weights in PyTorch, tokenize a string, run through the model, print logits. ✅

**Part 2 — Greedy Decode Loop** ✅ **COMPLETE**
* **What:** Implement looped decoding (argmax each step). ✅

**Part 3 — KV-Cache (Single Request)** ✅ **COMPLETE**
* **What:** Add caching of K/V states; no recompute each step. ✅

**Part 4 — HTTP Server (FastAPI)** ✅ **COMPLETE**
* **What:** Wrap decoder in `/completions` endpoint with streaming. ✅

**Part 5 — Sampling (temp, top-k, top-p)** ✅ **COMPLETE**
* **What:** Add non-greedy sampling. ✅

---

## Phase 2 — Multi-Request + Scheduling

**Part 6 — Multiple Sequential Requests** ✅ **COMPLETE**
* **What:** Handle multiple queued requests. ✅

**Part 7 — Prefill Batching** ✅ **COMPLETE**
* **What:** Batch multiple prompts in one forward pass. ✅

**Part 8 — Continuous Batching (Prefill + Decode Waves)** ✅ **COMPLETE**
* **What:** Scheduler: new prompts join in flight. ✅

**Part 9 — FlashAttention Integration** ✅ **COMPLETE**
* **What:** Integrate FlashAttention for memory-efficient attention. ✅

**Part 10 — Logprobs API**
* **What:** Return top-n logprobs per token.


**Part 11 — Telemetry (Tokens/s, Latency, GPU Mem)**
* **What:** Add Prometheus counters; log TTFT + TPOT.

---

## Phase 3 — Efficiency + Scaling

**Part 12 — Triton Decode Kernel (Paged KV)**
* **What:** Fused Triton kernel for attention decode.

**Part 13 — Paged KV-Cache Allocator**
* **What:** Alloc/free pages, per-seq indirection.

**Part 14 — Streaming Client**

* **What:** Simple React/CLI client for live streaming.

**Part 15 — Stress Test (10000 Requests)**
* **What:** Load test + backpressure logic.

---

## Phase 4 —  Showcase

**Part 16 — Speculative Decoding**
* **What:** Draft model + target model accept/reject.

**Part 17 — Quantization (Int8/Int4)**
* **What:** Load GPT-2 with weight quantization.

**Part 18 — LoRA Hot-Swap**
* **What:** Merge LoRA adapters at load.

**Part 19 — Cost Dashboard**
* **What:** $/req and efficiency metrics.

**Part 20 — Full Showcase**
* **What:** Run GPT-2 XL (1.5B) with batching, streaming, telemetry, speculative decoding.
