# Serial greedy continuation

Model-neutral continuation after a real final-prefill completion. It feeds
each sampled token back into the next canonical decode frame, preserves request
and model generations, increments sequence/position/sample counters, and copies
the existing reserved page table and capability order unchanged. It does not
invent a page, recipe, device approval or model identity.

All requested input plus decode positions must fit the declared context before
construction. Generated tokens and alternating frame storage are fixed-capacity.
Length/EOS stop before another submission. Completion identity/kind/count and
greedy mode are checked; a failed or foreign completion poisons continuation.
Cancellation suppresses further output but the caller must still drain inflight
execution and release request/KV ownership before reusing resources.

The caller owns the page reservation for the complete request and the execution
pipeline. It must retire a step before `accept`, and consume/encode its next plan
before overwriting the borrowed completion buffer. This is not continuous
batching, text tokenization or a model-specific decoder implementation.

```mbt nocheck
let generation = @serial_generation.SerialGreedyGeneration::new(
  final_prefill, actual_completion, limits, inference,
  maximum_new_tokens=8, maximum_context=64, stop_tokens=[eos_id],
)
while generation.finish_reason() is Running {
  let next = generation.next_decode()
  // Submit through the same loaded execution pipeline and await retirement.
  generation.accept(actual_decode_completion)
}
// Drain/release the pipeline; then expose generation.token_at(i).
```
