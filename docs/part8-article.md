# Part 8: Continuous Batching - Dynamic Request Management

*Implementing continuous batching with independent request lifecycles and greedy cache invalidation*

## Overview

In Part 8 of our 20-part inference engine series, we implement **continuous batching** - an advanced optimization that allows requests to join and leave batches dynamically during generation. This builds upon Part 7's prefill batching by removing fixed batch boundaries, enabling requests to have independent lifecycles within a continuously evolving batch.

## From Prefill Batching to Continuous Batching

### Part 7 Recap: Static Prefill Batching

Part 7 implemented **prefill batching** using `BatchedRequestQueue` and `generate_batch_with_sampling()`. The key characteristics from the actual code:

```python
# Part 7: BatchedRequestQueue behavior (from batched_queue.py)
def collect_next_batch(self) -> Optional[RequestBatch]:
    """Collect up to max_batch_size requests, wait max_wait_time for more"""
    batch_requests = []
    # Collect up to 32 requests immediately
    while len(batch_requests) < self.max_batch_size:
        try:
            request = self._queue.get_nowait()
            batch_requests.append(request)
        except Exception:
            break
    # Wait up to 100ms for more requests if batch not full
    # Process batch when full or timeout reached
```

```python
# Part 7: Batch generation behavior (from batch_generator.py)
def generate_batch_with_sampling(self, batch_input_ids, max_length=20, ...):
    """Process multiple requests simultaneously with dynamic sequence management"""
    active_sequences = list(range(batch_size))
    for step in range(max_length):
        # Generate next token for all active sequences
        # Remove finished sequences dynamically: active_sequences.remove(seq_idx)
        # Continue until all sequences finish
```

**Key Characteristics of Part 7:**
- **Batch Formation**: Collect up to 32 requests, wait max 100ms
- **Dynamic Sequence Management**: Sequences can finish at different times within a batch
- **KV-Cache Reuse**: Maintained throughout batch processing
- **Fixed Batch Membership**: Once formed, no new requests can join

### Part 8: Continuous Batching Implementation

Part 8 introduces `ContinuousRequest` and `DynamicBatch` for truly dynamic batch management:

```python
# Part 8: ContinuousRequest (from continuous_queue.py)
@dataclass
class ContinuousRequest:
    """A request with independent lifecycle in continuous batching."""
    id: str
    state: RequestState  # WAITING → ACTIVE → COMPLETED/FAILED
    input_tokens: List[int]
    generated_tokens: List[int]
    joined_batch_at: Optional[float] = None
    generation_step: int = 0
    # Individual sampling parameters per request
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
```

**Key Differences from Part 7:**
- **Independent Lifecycles**: Each request tracks its own state and timing
- **Dynamic Membership**: Requests can join batches mid-generation
- **Per-Request Parameters**: Individual sampling settings
- **State Tracking**: Detailed lifecycle management

## What is Continuous Batching?

**Continuous batching** maintains a dynamic batch where requests can join and leave at any point during generation. Instead of processing fixed groups like Part 7, the system maintains a continuously flowing stream of requests.

### Core Components

#### 1. Request States and Lifecycle

The system tracks requests through distinct states:

```python
class RequestState(Enum):
    """State of a request in continuous batching."""
    WAITING = "waiting"      # In queue, not yet in a batch
    ACTIVE = "active"        # Currently generating in batch
    COMPLETED = "completed"  # Generation finished
    FAILED = "failed"        # Request failed
```

#### 2. Dynamic Batch Management

```python
@dataclass
class DynamicBatch:
    """A batch that can dynamically change composition during generation."""
    id: str
    requests: Dict[str, ContinuousRequest]  # Dynamic membership
    generation_step: int = 0
    past_key_values: Optional[Any] = None   # Shared KV-cache

    def add_request(self, request: ContinuousRequest) -> None:
        """Add a new request to the batch mid-generation."""
        request.state = RequestState.ACTIVE
        request.joined_batch_at = time.time()
        request.generation_step = self.generation_step
        self.requests[request.id] = request

    def remove_request(self, request_id: str) -> Optional[ContinuousRequest]:
        """Remove a completed request from the batch."""
        if request_id in self.requests:
            request = self.requests.pop(request_id)
            request.completed_at = time.time()
            return request
        return None
```

## The Continuous Generation Loop

The heart of continuous batching is the `continuous_generation_loop()` method, which runs continuously to manage dynamic batches:

### Phase-by-Phase Breakdown

```python
async def continuous_generation_loop(self):
    """Main continuous generation loop with 8 phases."""

    # Load model and tokenizer once
    model = GPT2Model("gpt2", device="cpu")
    model.load_model()
    tokenizer = GPT2TokenizerWrapper("gpt2")

    past_key_values = None
    generation_step_counter = 0

    while not self._shutdown:
        try:
            # PHASE 1: DYNAMIC BATCH EXPANSION
            added_count = self.add_waiting_requests_to_batch()
            if added_count > 0:
                logger.info(f"Added {added_count} new requests to batch")
                # CACHE INVALIDATION: New batch composition breaks KV-cache
                past_key_values = None
```

### Critical Implementation Detail: Greedy Cache Invalidation

**The Key Limitation**: Our implementation uses greedy cache invalidation whenever batch composition changes:

```python
# When requests JOIN the batch (Phase 1)
if added_count > 0:
    past_key_values = None  # Reset cache

# When requests LEAVE the batch (Phase 7)
if completed_requests:
    past_key_values = None  # Reset cache
```

**Why Cache Invalidation is Necessary:**

The KV-cache stores attention patterns with fixed tensor dimensions:
```python
past_key_values = [
    (key_tensor, value_tensor),  # [batch_size, num_heads, seq_len, head_dim]
    # ... for each transformer layer
]
```

**When batch composition changes:**
- **Requests join**: Cache `[3, 12, 50, 64]` → need `[4, 12, 50, 64]` (different batch size)
- **Requests leave**: Cache `[4, 12, 50, 64]` → need `[2, 12, 50, 64]` (indices no longer align)

**Our solution**: Discard entire cache and restart with prefill for all remaining requests.

### Complete Generation Loop Flow

```python
# PHASE 2: BATCH EXISTENCE CHECK
if not self.current_batch or self.current_batch.size == 0:
    await asyncio.sleep(self.generation_interval)  # 10ms
    continue

# PHASE 3: ACTIVE REQUEST FILTERING
active_requests = self.current_batch.get_active_requests()
if not active_requests:
    await asyncio.sleep(self.generation_interval)
    continue

# PHASE 4: TOKENIZATION OF NEW ARRIVALS
for request in active_requests:
    if not request.input_tokens and hasattr(request.request_data, "prompt"):
        input_ids = tokenizer.encode(request.request_data.prompt, return_tensors=True)
        request.input_tokens = input_ids.squeeze().tolist()

# PHASE 5: PARALLEL TOKEN GENERATION
try:
    next_tokens, past_key_values, finished_flags = model.generate_continuous_step(
        active_requests=active_requests,
        past_key_values=past_key_values,
        show_progress=False,
    )
    generation_step_counter += 1
    self.total_generation_steps += 1
    if self.current_batch:
        self.current_batch.generation_step += 1

# PHASE 6: HANDLE COMPLETED REQUESTS
completed_requests = self.remove_completed_requests()
for request in completed_requests:
    if request.state == RequestState.COMPLETED:
        # Convert tokens back to text and create API response
        full_tokens = request.input_tokens + request.generated_tokens
        full_text = tokenizer.decode(full_tokens)
        generated_text = full_text[len(request.request_data.prompt):]
        # Create CompletionResponse...

# PHASE 7: CACHE MANAGEMENT
if completed_requests:
    # CACHE INVALIDATION: Batch composition changed
    past_key_values = None

# PHASE 8: TIMING CONTROL
await asyncio.sleep(self.generation_interval)  # 10ms heart beat
```

## Continuous Step Generation

The `generate_continuous_step()` method (from `gpt2.py`) handles the actual token generation for dynamic batches:

```python
def generate_continuous_step(
    self,
    active_requests: List,
    past_key_values: Optional[Any] = None,
    show_progress: bool = False,
) -> tuple:
    """Generate next token for all active requests in continuous batching."""

    if past_key_values is None:
        # First step: process full input sequences (prefill)
        max_input_len = max(len(req.input_tokens) for req in active_requests)
        # Pad all sequences and create attention masks
        # Forward pass with attention_mask
    else:
        # Subsequent steps: process only last generated token
        last_tokens = []
        for req in active_requests:
            if req.generated_tokens:
                last_tokens.append([req.generated_tokens[-1]])
        # Forward pass with cached key-values

    # Sample next tokens using per-request parameters
    for i, req in enumerate(active_requests):
        next_token_id = sample_token(
            logits[i, -1, :],
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )
        req.generated_tokens.append(next_token_id)
        req.generation_step += 1

    return next_tokens, new_past_key_values, finished_flags
```

## Real-World Continuous Batching Flow

Let's trace through how requests flow through the system:

### Example Timeline: 5 Requests Over 10 Steps

```python
Step 1:  Batch=[Req1, Req2]                    # 2 requests start
         Cache: None → prefill both requests
Step 2:  Batch=[Req1, Req2]                    # Generation continues
         Cache: Reused from step 1
Step 3:  Batch=[Req1, Req2, Req3]              # Req3 joins
         Cache: None → prefill all 3 (cache invalidated)
Step 4:  Batch=[Req1, Req2, Req3]              # Generation continues
         Cache: Reused from step 3
Step 5:  Batch=[Req2, Req3]                    # Req1 finishes
         Cache: None → prefill remaining 2 (cache invalidated)
Step 6:  Batch=[Req2, Req3, Req4, Req5]        # Req4, Req5 join
         Cache: None → prefill all 4 (cache invalidated)
```

### Key Observations

1. **Dynamic batch size**: 2→3→2→4 (continuously changing)
2. **Independent completion**: Req1 finishes when ready, others continue
3. **Frequent cache invalidation**: Steps 1, 3, 5, 6 require prefill
4. **Cache efficiency**: Only 40% cache reuse in this example

## Performance Analysis

### Configuration

Our implementation uses these settings:

```python
# From continuous_queue.py
continuous_scheduler = ContinuousBatchScheduler(
    max_batch_size=8,        # Up to 8 requests in dynamic batch
    max_sequence_length=512, # Handle longer sequences
    generation_interval=0.01, # 10ms between generation steps
)
```

### The Cost of Greedy Cache Invalidation

**Cache Invalidation Events:**
- **Request joins batch**: `past_key_values = None`
- **Request leaves batch**: `past_key_values = None`

**Performance Impact:**
- **Lost computation**: All previous attention calculations discarded
- **Prefill penalty**: Must restart from scratch for all remaining requests
- **Memory churn**: Constant cache allocation/deallocation

**Realistic Scenario Analysis:**
```python
# 10 requests with varying arrival/completion patterns
Cache invalidations: ~60% of steps (high churn)
Cache reuse: ~40% of steps (suboptimal)
Prefill overhead: Significant performance penalty
```

## Testing Continuous Batching

Our test suite validates continuous batching functionality:

### Key Test Cases

```python
def test_single_request_prefill_step(model, tokenizer):
    """Test first step with past_key_values=None"""
    request = create_continuous_request("test_001", "Once upon a time", tokenizer)
    next_tokens, past_key_values, finished_flags = model.generate_continuous_step(
        active_requests=[request],
        past_key_values=None,
    )
    assert len(next_tokens) == 1
    assert past_key_values is not None

def test_dynamic_batch_composition(model, tokenizer):
    """Test adding requests mid-generation"""
    # Start with one request
    request1 = create_continuous_request("dynamic_001", "The future of AI", tokenizer)
    # First step
    next_tokens, past_key_values, _ = model.generate_continuous_step([request1], None)

    # Add second request (simulating dynamic addition)
    request2 = create_continuous_request("dynamic_002", "Machine learning is", tokenizer)
    # Reset cache due to batch composition change
    next_tokens, past_key_values, _ = model.generate_continuous_step(
        [request1, request2], None  # Cache reset required
    )
    assert len(next_tokens) == 2
```

**Test Insight**: The `test_dynamic_batch_composition` test explicitly shows cache invalidation:
```python
# For demonstration, we restart KV cache when batch composition changes
# In production, this would be handled more elegantly
past_key_values = None  # Reset cache for new batch composition
```

## Architectural Integration

### Strategy Pattern Implementation

Continuous batching integrates with the existing strategy pattern:

```python
class ContinuousQueueStrategy(QueueStrategy):
    """Continuous processing strategy with dynamic batch scheduling."""

    async def process_request(self, request: "CompletionRequest") -> "CompletionResponse":
        # Add request to continuous scheduler
        request_id = self._scheduler.add_request(request)

        # Wait for completion with polling
        while time.time() - start_time < max_wait_time:
            continuous_request = self._scheduler.get_request_status(request_id)
            if continuous_request["status"] == "completed":
                return continuous_request["result"]
            await asyncio.sleep(0.1)
```

### Request Flow Architecture

```python
# Complete request flow
1. API Request → ContinuousQueueStrategy.process_request()
2. Convert to ContinuousRequest → continuous_scheduler.add_request()
3. Add to waiting_queue → continuous_generation_loop picks up
4. Join active batch → model.generate_continuous_step()
5. Generate tokens → remove_completed_requests()
6. Return CompletionResponse
```

## Monitoring and Statistics

The continuous scheduler provides comprehensive metrics:

```python
@property
def stats(self) -> Dict[str, Any]:
    """Get continuous batching statistics."""
    return {
        # Queue metrics
        "waiting_requests": self.waiting_queue.qsize(),
        "active_requests": len(self.active_requests),
        "current_batch_size": self.current_batch.size if self.current_batch else 0,

        # Performance metrics
        "total_requests": self.total_requests,
        "total_completed": self.total_completed,
        "total_generation_steps": self.total_generation_steps,
        "current_generation_step": current_generation_step,

        # Timing metrics
        "average_wait_time": avg_wait_time,
        "average_generation_time": avg_generation_time,

        # System identification
        "mode": "continuous",
        "part": 8,
    }
```

## Educational Simplifications and Limitations

### What We Implemented

✅ **Dynamic batch composition**: Requests can join/leave mid-generation
✅ **Independent lifecycles**: Each request completes when ready
✅ **Continuous processing**: No idle time between batches
✅ **Per-request parameters**: Individual sampling settings
✅ **State management**: Comprehensive request tracking

### What We Simplified

❌ **Greedy cache invalidation**: Discard entire cache on any composition change
❌ **No cache slicing**: Don't preserve cache for continuing requests
❌ **Basic memory management**: No optimization for memory efficiency
❌ **Simple scheduling**: No sophisticated batch composition optimization

### The Educational Trade-off

Our implementation prioritizes **clarity and understanding** over **production efficiency**. The greedy cache invalidation approach demonstrates:

1. **The core concepts** of continuous batching
2. **The challenges** of dynamic batch management
3. **The importance** of sophisticated cache management
4. **A foundation** for understanding advanced optimizations

## Future Work: Production-Grade Optimizations

Real production systems require more sophisticated approaches:

### 1. Intelligent Cache Management

```python
# Production concept (not implemented)
def preserve_cache_for_continuing_requests(completed_indices):
    """Slice cache to remove only completed requests."""
    new_cache = []
    for layer_cache in past_key_values:
        keys, values = layer_cache
        # Remove rows for completed requests
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        active_mask[completed_indices] = False
        new_keys = keys[active_mask]
        new_values = values[active_mask]
        new_cache.append((new_keys, new_values))
    return new_cache
```

### 2. Predictive Batch Composition

```python
# Production concept (not implemented)
def optimize_batch_composition(waiting_requests):
    """Group requests to minimize cache invalidation."""
    # Predict completion times
    # Group by similar expected length
    # Balance throughput vs cache efficiency
    pass
```

### 3. Memory-Efficient Architectures

```python
# Production concept (not implemented)
class PaddedCacheManager:
    """Pre-allocate maximum cache, use masking for dynamic sizes."""
    def __init__(self, max_batch_size=32):
        self.max_cache = self.allocate_max_cache()
        self.active_mask = torch.zeros(max_batch_size, dtype=torch.bool)
```

## Comparison with Part 7

| Aspect | Part 7 (Prefill Batching) | Part 8 (Continuous Batching) |
|--------|---------------------------|------------------------------|
| **Batch Formation** | Wait up to 100ms for 32 requests | Dynamic, immediate processing |
| **Request Lifecycle** | Fixed batch membership | Independent, can join/leave |
| **Cache Management** | Maintained within batch | Invalidated on composition change |
| **Scheduling** | Simple queue collection | Sophisticated dynamic management |
| **Completion** | All finish, then new batch | Independent completion timing |
| **Complexity** | Moderate | High |
| **Cache Efficiency** | High (within batch) | Low (frequent invalidation) |

## Key Insights

### 1. **Dynamic vs Static Trade-offs**
Continuous batching provides better utilization and lower latency at the cost of cache efficiency and complexity.

### 2. **Cache Management is Critical**
The effectiveness of continuous batching depends heavily on cache management strategy. Our greedy approach demonstrates the concept but limits performance.

### 3. **Independent Lifecycles Enable Optimization**
Allowing requests to complete independently eliminates synchronization overhead and enables optimal resource utilization.

### 4. **Educational vs Production**
Our implementation excellently demonstrates concepts while acknowledging the need for sophisticated optimizations in production systems.

## Conclusion

Part 8's continuous batching implementation successfully demonstrates dynamic batch management with independent request lifecycles. While our greedy cache invalidation approach limits performance, it provides an excellent foundation for understanding the core concepts and challenges of continuous batching.

The implementation shows:
- **How** requests can join and leave batches dynamically
- **Why** cache management is crucial for efficiency
- **What** trade-offs exist between simplicity and performance
- **Where** future optimizations can be applied

This foundation prepares us for more advanced optimizations in future parts, such as intelligent cache slicing, predictive batch composition, and hardware-specific optimizations that power production LLM serving systems.

---

## Navigation

← **Previous**: [Part 7: Prefill Batching](part7-article.md) | **Next**: Part 9: FlashAttention Integration (Coming Soon) →

---

*Next up: Advanced cache management techniques to achieve production-grade efficiency in continuous batching systems.*
