# Part 7: Prefill Batching - Static Batch Processing

*Implementing batched request processing with dynamic sequence management and KV-cache optimization*

## Overview

In Part 7 of our 20-part inference engine series, we implement **prefill batching** - an optimization that allows LLM inference engines to process multiple requests simultaneously in static batches. This builds upon Part 6's sequential processing by collecting requests into batches and processing them together, achieving throughput improvements through parallel computation.

## What is Prefill Batching?

**Prefill batching** is the technique of collecting multiple requests into fixed-composition batches and processing them together in a single model forward pass. The system waits to collect multiple requests, then processes the entire batch together before moving to the next batch.

### Implementation Architecture

Our prefill batching system consists of three main components:

1. **BatchedRequestQueue**: Collects requests into batches
2. **BatchedQueueStrategy**: Manages request lifecycle through batching
3. **GPT2Model.generate_batch_with_sampling()**: Performs actual batched generation

## BatchedRequestQueue: Static Batch Collection

The `BatchedRequestQueue` handles request collection with these actual configuration values:

```python
# From batched_queue.py - actual configuration
batched_request_queue = BatchedRequestQueue(
    max_batch_size=32,  # Process up to 32 requests together
    max_wait_time=0.1,  # Wait max 100ms to form larger batches
    min_batch_size=1,   # Process single requests immediately if queue is empty
)
```

### Batch Formation Logic

The core batch collection happens in `collect_next_batch()`:

```python
# From batched_queue.py - actual implementation
def collect_next_batch(self) -> Optional[RequestBatch]:
    """
    Collect the next batch of requests for processing.

    This implements the core batching logic:
    1. Try to collect up to max_batch_size requests
    2. Wait up to max_wait_time for more requests
    3. Process when batch is full or timeout reached
    """
    batch_requests = []
    start_time = time.time()

    # Collect initial requests (non-blocking)
    while len(batch_requests) < self.max_batch_size:
        try:
            request = self._queue.get_nowait()
            batch_requests.append(request)
        except Exception:
            break  # No more requests available immediately

    # If we have some requests but batch isn't full, wait a bit for more
    if batch_requests and len(batch_requests) < self.max_batch_size:
        remaining_wait = self.max_wait_time - (time.time() - start_time)

        while remaining_wait > 0 and len(batch_requests) < self.max_batch_size:
            try:
                # Wait briefly for more requests
                request = self._queue.get(timeout=min(0.01, remaining_wait))
                batch_requests.append(request)
                remaining_wait = self.max_wait_time - (time.time() - start_time)
            except Exception:
                break

    # Create batch if we have enough requests
    if len(batch_requests) >= self.min_batch_size:
        batch_id = f"batch-{int(time.time())}-{len(batch_requests)}"
        return RequestBatch(
            id=batch_id,
            requests=batch_requests,
            created_at=time.time(),
        )

    return None
```

**Key Characteristics:**
- **Greedy Collection**: Immediately grab all available requests
- **Timed Waiting**: Wait up to 100ms for additional requests to fill batch
- **Flexible Processing**: Will process partial batches rather than wait indefinitely
- **Static Composition**: Once formed, batch membership is fixed

## BatchedQueueStrategy: Request Lifecycle Management

The strategy pattern integration handles individual request processing:

```python
# From strategies/batched.py - actual implementation
class BatchedQueueStrategy(QueueStrategy):
    """Batched processing strategy that collects requests into static batches."""

    async def process_request(self, request: "CompletionRequest") -> "CompletionResponse":
        """Process a single request through the batched queue."""

        # Add request to batched queue
        request_id = self._queue.add_request(request)

        # Wait for request to complete with timeout
        max_wait_time = 300  # 5 minutes timeout
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            batched_request = self._queue.get_request_status(request_id)

            if batched_request is None:
                raise Exception("Request not found in batched queue")

            if batched_request.status == "completed":
                return batched_request.result

            elif batched_request.status == "failed":
                raise Exception(f"Request failed: {batched_request.error}")

            # Still processing, wait a bit
            await asyncio.sleep(0.1)

        # Request timed out
        raise Exception("Request timeout after 5 minutes")
```

**Request Flow:**
1. Request added to queue with `add_request()`
2. Polling loop checks status every 100ms
3. Returns result when batch processing completes
4. 5-minute timeout for safety

## Batched Generation Implementation

The core generation logic is in `GPT2Model.generate_batch_with_sampling()`:

```python
# From batch_generator.py - actual method signature
def generate_batch_with_sampling(
    self,
    batch_input_ids: List[torch.Tensor],  # List of input tensors, one per request
    max_length: int = 20,                 # Maximum number of NEW tokens to generate
    temperature: float = 1.0,             # Controls randomness
    top_k: Optional[int] = None,          # Limit sampling to top-k tokens
    top_p: Optional[float] = None,        # Nucleus sampling threshold
    use_cache: bool = True,               # Enable KV-cache for speedup
    show_progress: bool = False,          # Print detailed progress
) -> List[List[int]]:
    """
    CORE BATCHED GENERATION: Process multiple requests simultaneously.

    This is the heart of prefill batching (Part 7) - instead of processing
    requests one-by-one, we batch them together to maximize GPU utilization.
    """
```

### Actual Generation Loop Structure

The real implementation has sophisticated logic:

```python
# From batch_generator.py - actual generation loop
# Initialize sequence tracking and cache
active_sequences = list(range(batch_size))
past_key_values = None

# Main generation loop
for step in range(max_length):
    # Early termination if all sequences finished
    if not active_sequences:
        break

    # Prepare batch for current step
    if step == 0:
        # Prefill phase: process full input sequences
        batch_tensor, attention_mask = self._create_prefill_tensors(
            batch_sequences, active_sequences
        )
        current_indices = active_sequences.copy()
    else:
        # Incremental decode: process only new tokens
        # Handle KV-cache slicing for finished sequences
        past_key_values, active_sequences = (
            self._slice_kv_cache_for_active_sequences(
                past_key_values,
                active_sequences,
                batch_size,
                step,
                show_progress,
            )
        )

        # Prepare batch with only active sequences
        current_batch_ids = []
        current_indices = []
        for seq_idx in active_sequences:
            current_batch_ids.append([batch_sequences[seq_idx][-1]])
            current_indices.append(seq_idx)

        batch_tensor = torch.tensor(
            current_batch_ids, dtype=torch.long, device=self.model.device
        )
        attention_mask = None

    # Model forward pass
    with torch.no_grad():
        if step == 0:
            # Prefill forward pass
            outputs = self.model.model(
                batch_tensor,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        else:
            # Incremental forward pass
            outputs = self.model.model(
                batch_tensor,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

        logits = outputs.logits
        if use_cache:
            past_key_values = outputs.past_key_values

    # Sample next tokens and update sequences
    sequences_to_remove = self._sample_next_tokens(
        logits,
        current_indices,
        batch_sequences,
        temperature,
        top_k,
        top_p,
        show_progress,
    )

    # Remove finished sequences
    for seq_idx in sorted(sequences_to_remove, reverse=True):
        if seq_idx in active_sequences:
            active_sequences.remove(seq_idx)
```

### Key Implementation Features

#### 1. **Dynamic Sequence Management**
Unlike what you might expect from "static batching," sequences can finish at different times:
```python
# Remove finished sequences dynamically during generation
for seq_idx in sorted(sequences_to_remove, reverse=True):
    if seq_idx in active_sequences:
        active_sequences.remove(seq_idx)
```

#### 2. **Sophisticated KV-Cache Management**
The implementation includes real KV-cache slicing for finished sequences:
```python
# Actual method call for cache management
past_key_values, active_sequences = (
    self._slice_kv_cache_for_active_sequences(
        past_key_values, active_sequences, batch_size, step, show_progress,
    )
)
```

#### 3. **Two-Phase Processing**
- **Step 0 (Prefill)**: Process full input sequences with attention masks
- **Step 1+ (Decode)**: Process only new tokens with KV-cache reuse

#### 4. **Progressive Batch Shrinkage**
As sequences finish, the effective batch size shrinks, but processing continues for remaining sequences.

## Real-World Example: Processing 100 Requests

With `max_batch_size=32`, here's how 100 requests would actually be processed:

```python
# Actual batch formation for 100 requests
Batch 1: Requests 1-32   (full batch, collected immediately)
Batch 2: Requests 33-64  (full batch, collected immediately)
Batch 3: Requests 65-96  (full batch, collected immediately)
Batch 4: Requests 97-100 (partial batch, 4 requests)

# Within each batch, sequences finish independently
# Batch 1 processing:
Step 0: Process all 32 sequences (prefill)
Step 1: Process remaining 32 sequences (decode)
Step 2: Process remaining 28 sequences (4 finished)
Step 3: Process remaining 20 sequences (8 more finished)
...continues until all finish
```

**Key Insight**: Instead of 100 sequential model calls, we execute **4 batch calls** (one per batch), with dynamic sequence management within each batch.

## Performance Characteristics

### Configuration Analysis

```python
# Memory scaling with batch size
max_batch_size=32 → up to 32x parallelization
max_wait_time=0.1 → 100ms maximum wait for batch formation
min_batch_size=1 → no requests left waiting
```

### Throughput Calculation

**Theoretical Maximum:**
- Sequential: 100 requests = 100 model calls
- Batched: 100 requests = 4 batch calls (significant throughput improvement)

**Practical Considerations:**
- **Batch formation time**: 0-100ms depending on request arrival
- **Memory usage**: Scales linearly with batch size
- **KV-cache efficiency**: High within each batch due to sophisticated cache management

## Testing and Validation

Our test suite demonstrates the actual functionality:

```python
# From test_batch_generation_integration.py - actual test
def test_specific_prompts_batch_generation(self, model, tokenizer):
    """Test batch generation with specific sentence prompts."""
    prompts = ["to be or not to", "its great to be", "toPart is a good Part to"]

    batch_input_ids = []
    original_lengths = []

    for prompt in prompts:
        input_ids_tensor = tokenizer.encode(prompt)
        input_ids = input_ids_tensor.squeeze().tolist()
        batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
        original_lengths.append(len(input_ids))

    # Generate tokens with moderate creativity
    results = model.generate_batch_with_sampling(
        batch_input_ids=batch_input_ids,
        max_length=10,      # Generate up to 10 new tokens per sequence
        temperature=0.8,    # Moderately creative
        top_k=50,          # Consider top 50 tokens
        top_p=0.9,         # Nucleus sampling
        use_cache=True,
        show_progress=True,  # Show progress for educational purposes
    )

    # Verify results
    assert len(results) == len(prompts)

    for i, (result_sequence, original_length, prompt) in enumerate(
        zip(results, original_lengths, prompts)
    ):
        # Decode and display result
        full_text = tokenizer.decode(torch.tensor(result_sequence))
        generated_text = tokenizer.decode(
            torch.tensor(result_sequence[original_length:])
        )
        print(f"Prompt {i+1}: '{prompt}'")
        print(f"Full result: '{full_text}'")
        print(f"Generated: '{generated_text}'")

        # Validate sequence properties
        assert len(result_sequence) > original_length
        assert len(result_sequence) <= original_length + 10
```

## Static vs Dynamic: The Terminology Clarification

The term "static batching" refers to **batch formation**, not **sequence processing**:

- **Static Batch Formation**: Once a batch is formed, no new requests can join
- **Dynamic Sequence Management**: Within a batch, sequences can finish at different times

This is different from Part 8's "continuous batching" where requests can join/leave batches during generation.

## Architecture Integration

### Strategy Pattern

Prefill batching integrates with the strategy pattern:

```python
# From strategies/batched.py - actual integration
class BatchedQueueStrategy(QueueStrategy):
    """
    Batched processing strategy that collects requests into static batches.

    Key characteristics:
    - Collects up to max_batch_size requests
    - Waits up to max_wait_time for batch to fill
    - Processes entire batch in single model call
    - All requests in batch start together (but can finish independently)
    """
```

### Request Flow

```python
# Complete request processing flow
1. API Request → BatchedQueueStrategy.process_request()
2. Add to queue → BatchedRequestQueue.add_request()
3. Batch formation → collect_next_batch()
4. Batch processing → generate_batch_with_sampling()
5. Result creation → Individual CompletionResponse objects
6. Response delivery → Back to original API caller
```

## Monitoring and Statistics

The batched queue provides performance metrics:

```python
# From strategies/batched.py - actual stats method
def get_stats(self) -> Dict[str, Any]:
    """Get current statistics for the batched queue."""
    stats = self._queue.stats
    stats["mode"] = "batched"
    stats["part"] = self.part_number
    return stats
```

**Key Metrics:**
- Queue depth and batch formation rates
- Average batch sizes and wait times
- Request processing times and success rates
- Memory usage and cache efficiency

## Comparison with Part 6 (Sequential)

| Aspect | Part 6 (Sequential) | Part 7 (Prefill Batching) |
|--------|-------------------|---------------------------|
| **Processing Model** | One request at a time | Batches of up to 32 requests |
| **Throughput** | ~1x baseline | Significant improvement |
| **Latency** | Immediate processing | 0-100ms batch formation delay |
| **Memory Usage** | Per-request allocation | Shared batch allocation |
| **Complexity** | Simple queue | Batch collection + sophisticated generation |
| **GPU Utilization** | Low (sequential) | High (parallel batch processing) |

## Educational Implementation Choices

### What We Implemented

✅ **Static batch formation** with configurable timing and sizes
✅ **Dynamic sequence management** within batches
✅ **Sophisticated KV-cache slicing** for finished sequences
✅ **Two-phase processing** (prefill + decode)
✅ **Real tensor operations** and attention mechanisms
✅ **Production-ready request handling** with timeouts and error handling

### Implementation Sophistication

Unlike typical educational examples, our implementation includes:

- **Real KV-cache management** with `_slice_kv_cache_for_active_sequences()`
- **Proper tensor batching** with `_create_prefill_tensors()`
- **Sophisticated sampling** with temperature, top-k, and top-p
- **Progressive batch shrinkage** as sequences finish
- **Memory-efficient processing** with proper device management

## Key Insights

### 1. **"Static" Refers to Batch Membership**
The batch composition is fixed at formation time, but sequence processing within the batch is dynamic.

### 2. **Sophisticated Cache Management**
Unlike simple batching implementations, ours includes real KV-cache slicing to maintain efficiency as sequences finish.

### 3. **Production-Ready Architecture**
The strategy pattern integration and error handling make this suitable for real serving scenarios.

### 4. **Foundation for Continuous Batching**
The dynamic sequence management within batches provides the foundation for Part 8's fully dynamic batching.

## Looking Ahead to Part 8

Part 7's limitations that Part 8 addresses:

- **Fixed batch membership**: Can't add new requests once batch starts processing
- **Batch formation delays**: Must wait for requests to accumulate
- **Underutilization**: Partial batches waste capacity

Part 8's continuous batching removes these limitations by allowing requests to join and leave batches during generation, though at the cost of more complex cache management.

## Conclusion

Part 7's prefill batching successfully demonstrates static batch processing with sophisticated sequence management. The implementation provides significant throughput improvements while maintaining the simplicity of fixed batch composition.

Key achievements:
- **Significant throughput improvement** through batch parallelization
- **Dynamic sequence management** within static batches
- **Sophisticated KV-cache optimization** with real cache slicing
- **Production-ready request handling** with proper error management
- **Foundation for advanced batching** techniques in Part 8

The implementation strikes an excellent balance between educational clarity and production sophistication, providing real optimizations while remaining understandable.

---

## Navigation

← **Previous**: [Part 6: Sequential Request Handling](part6-article.md) | **Next**: [Part 8: Continuous Batching](part8-article.md) →

---

*Next up: Part 8 - Continuous Batching, where we remove the static batch limitations and allow truly dynamic request management during generation.*
