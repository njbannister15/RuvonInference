# Project RUVON VLLM

We are building an educational inference engine from scratch — a miniature but real system that can load pretrained transformer models (starting with GPT-2 124M → scaling to GPT-2 XL 1.5B), tokenize user prompts, run efficient prefill and incremental decode passes, and serve generations over an HTTP API with streaming, batching, and telemetry. The core focus is not training new models, but serving existing ones efficiently: implementing a paged KV-cache allocator for memory reuse, a continuous batching scheduler so new requests can join mid-flight, and a Triton-based fused decode kernel for speed.

Around this runtime we’ll build a FastAPI service layer with OpenAI-like endpoints, sampling strategies (greedy, top-k, nucleus, temperature), speculative decoding for acceleration, and lightweight quantization so larger models can fit into limited GPU memory.

Each part we’ll cut a vertical slice — from tokenizer → kernel → scheduler → streaming client → cost telemetry — and publish a demo plus a short article explaining the “what, why, and how.” By Part 20, the engine will be able to serve GPT-2 XL interactively with multi-user batching, real-time metrics, and cost dashboards: a functioning, investor-ready showcase of how to build an inference-optimized LLM runtime in just four weeks.


# 🚀 20-Part Educational Inference Engine Plan

## Phase 1 — Build a Vertical Slice

**Part 1 — Tokenizer + Single Forward Pass** ✅ **COMPLETE**

* **What:** Load GPT-2 124M weights in PyTorch, tokenize a string, run through the model, print logits. ✅
* **Why:** Establishes the foundation: "We can already turn text → numbers → predictions." ✅
* **Demo:** Input `"Hello world"` → see logits → decode `"Hello world!"`. ✅
* **Article:** *"How I ran a transformer from scratch: turning text into tokens, then back into words."* ✅
* **BONUS:** Beautiful CLI with Rich + Typer, educational documentation, clean architecture ✅

**Part 2 — Greedy Decode Loop** ✅ **COMPLETE**

* **What:** Implement looped decoding (argmax each step). ✅
* **Why:** Proves end-to-end generation is working. ✅
* **Demo:** Prompt `"Once upon a time"` → stream 20 tokens. ✅
* **Article:** *"My first tiny LLM: teaching a transformer to tell a story one token at a time."* ✅
* **BONUS:** Step-by-step visualization, generation statistics, beautiful CLI interface ✅

**Part 3 — KV-Cache (Single Request)** ✅ **COMPLETE**

* **What:** Add caching of K/V states; no recompute each step. ✅
* **Why:** Cuts latency, shows we understand inference efficiency. ✅
* **Demo:** Compare speed with/without cache. ✅ (5.1x speedup demonstrated!)
* **Article:** *"How caching slashes LLM latency: building the heart of a modern inference engine."* ✅
* **BONUS:** Performance benchmarking CLI, detailed metrics, educational insights ✅

**Part 4 — HTTP Server (FastAPI)** ✅ **COMPLETE**

* **What:** Wrap decoder in `/completions` endpoint with streaming. ✅
* **Why:** Investors see product surface, not just code. ✅
* **Demo:** `curl` → live streaming response. ✅
* **Article:** *"From script to service: standing up a GPT-2 API in 24 hours."* ✅
* **BONUS:** OpenAI-compatible API, beautiful CLI server command, health checks ✅

**Part 5 — Sampling (temp, top-k, top-p)** ✅ **COMPLETE**

* **What:** Add non-greedy sampling. ✅
* **Why:** Unlocks creativity; shows product realism. ✅
* **Demo:** Same prompt, compare greedy vs creative outputs. ✅
* **Article:** *"Making my tiny LLM creative: top-k, top-p, and temperature explained simply."* ✅
* **BONUS:** Beautiful CLI commands for sampling comparison, API integration with OpenAI-compatible parameters ✅

---

## Phase 2 — Multi-Request + Scheduling

**Part 6 — Multiple Sequential Requests** ✅ **COMPLETE**

* **What:** Handle multiple queued requests. ✅
* **Why:** Basic multi-user support. ✅
* **Demo:** Fire 3 curl requests, see them complete in order. ✅
* **Article:** *"Teaching my tiny inference engine to juggle multiple requests."*

**Part 7 — Prefill Batching** ✅ **COMPLETE**

* **What:** Batch multiple prompts in one forward pass. ✅
* **Why:** Demonstrates efficiency for real workloads. ✅
* **Demo:** Compare throughput with 5 prompts batched vs serial. ✅
* **Article:** *"Batching: why LLMs prefer a crowd over a queue."* ✅

**Part 8 — Continuous Batching (Prefill + Decode Waves)** ✅ **COMPLETE**

* **What:** Scheduler: new prompts join in flight. ✅
* **Why:** Core vLLM idea — reduces wait times. ✅
* **Demo:** 2 long + 3 short requests, all served together. ✅
* **Article:** *"Continuous batching: the secret to high-throughput inference."* ✅

**Part 9 — FlashAttention Integration** ✅ **COMPLETE**

* **What:** Integrate FlashAttention for memory-efficient attention. ✅
* **Why:** Demonstrates O(n²) → O(n) memory breakthrough, modern optimization techniques. ✅
* **Demo:** Compare attention implementations (eager, sdpa, flash_attention_2). ✅
* **Article:** *"FlashAttention Integration: Using memory-efficient attention for longer sequences."* ✅
* **BONUS:** Flexible attention implementation system, comprehensive benchmarking, AWS production testing planned ✅

**Part 10 — Logprobs API**

* **What:** Return top-n logprobs per token.
* **Why:** Needed for fine-grained control + eval.
* **Demo:** Show prompt with per-token logprobs.
* **Article:** *"Peeking into the mind of an LLM: logprobs in action."*

**Part 11 — Telemetry (Tokens/s, Latency, GPU Mem)**

* **What:** Add Prometheus counters; log TTFT + TPOT.
* **Why:** Investors love metrics.
* **Demo:** Live dashboard of tokens/s during load test.
* **Article:** *"Measuring what matters: telemetry in a tiny LLM engine."*

---

## Phase 3 — Efficiency + Scaling

**Part 12 — Triton Decode Kernel (Paged KV)**

* **What:** Fused Triton kernel for attention decode.
* **Why:** Real systems need custom kernels.
* **Demo:** Benchmark token latency drop.
* **Article:** *“Writing my first GPU kernel in Triton for LLMs.”*

**Part 13 — Paged KV-Cache Allocator**

* **What:** Alloc/free pages, per-seq indirection.
* **Why:** Enables long prompts & high concurrency.
* **Demo:** Handle 20 requests without OOM.
* **Article:** *“Paged memory for transformers: bringing OS ideas to AI.”*

**Part 14 — Streaming Client**

* **What:** Simple React/CLI client for live streaming.
* **Why:** Polished user experience.
* **Demo:** Type prompt → see words stream out.
* **Article:** *“Streaming inference: why typing indicators matter for LLM UX.”*

**Part 15 — Stress Test (100 Requests)**

* **What:** Load test + backpressure logic.
* **Why:** Shows engine survives real-world load.
* **Demo:** Graph of latency vs concurrency.
* **Article:** *“What happens when 100 people hit my tiny LLM at once.”*

---

## Phase 4 —  Showcase

**Part 16 — Speculative Decoding**

* **What:** Draft model + target model accept/reject.
* **Why:** Cutting-edge technique; ties to your earlier struggles.
* **Demo:** Compare speed with/without speculative decode.
* **Article:** *“How two models can finish each other’s sentences faster.”*

**Part 17 — Quantization (Int8/Int4)**

* **What:** Load GPT-2 with weight quantization.
* **Why:** Fit bigger models in same GPU.
* **Demo:** Serve GPT-2 345M on modest GPU.
* **Article:** *“Shrinking models to fit: quantization made simple.”*

**Part 18 — LoRA Hot-Swap**

* **What:** Merge LoRA adapters at load.
* **Why:** Customization story for enterprises.
* **Demo:** Swap between “business” and “poetry” adapters.
* **Article:** *“Teaching my tiny LLM new tricks instantly with LoRA.”*

**Part 19 — Cost Dashboard**

* **What:** $/req and efficiency metrics.
* **Why:** Show investors unit economics.
* **Demo:** Live dashboard: “This run costs $0.0001/token.”
* **Article:** *“Turning inference into dollars: making LLM costs visible.”*

**Part 20 — Full Showcase**

* **What:** Run GPT-2 XL (1.5B) with batching, streaming, telemetry, speculative decoding.
* **Why:** Prove educational inference engine can handle real models.
* **Demo:** Live interactive chat with GPT-2 XL in your engine.
* **Article:** *"From zero to inference engine in 20 parts: my journey building an educational inference engine."*
