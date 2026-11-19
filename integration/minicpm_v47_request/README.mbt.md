# MiniCPM-V 4.7 request execution

This adapter stages the initial single-flight greedy text profile over the full
flagship decoder. It is not a scheduling-policy owner. Tokenization/admission,
HTTP streaming, deadlines and string-stop matching remain outer responsibilities.
The host controller owns preallocated metadata packets and request progress; the
native driver uploads those packets and submits existing prepared queues.

Prefill uses bounded chunks and emits logits only for the final chunk. Decode
feeds the previous generated token at the next causal/cache address. T/H/W
coordinates come from the prepared position owner. Both recurrent kernels share
the first-frame reset flag; full-attention KV reads see only the current prefix.
One static identity page table is reused, with no cross-request prefix sharing.

Only completed frames commit progress. Token stops, requested token limits and
the effective context ceiling terminate generation. Stop tokens count as sampled
outputs but are excluded from the emitted token span. Cancellation requires a
device quiescence barrier before metadata or recurrent state can be reused.

Buffers, packet encoding and queue descriptions are allocated before stepping.
Input uploads and four-byte sampled-token readback are required execution
transfers; they are distinct from diagnostic readbacks. CPU/software-driver
checks do not establish model logits, real CUDA execution or performance.

`Profile::from_program` accepts one sequence and one selected output row, using
the exact declared context ceiling rather than the rounded page capacity.
`Host` can stage an independent transport; `Runner` binds the same metadata to
the existing device executor. Both reuse prepared buffers. The model remains
MiniCPM-V 4.7 35B-A3B with all 40 layers; this profile does not replace it with a
smaller model. Native token ingress preserves already-templated IDs exactly.

`prepare_launches` accepts explicit prepared prefill/decode spans, allowing an
image bridge to insert a stage after token embedding. It shares the metadata
and progress law with `prepare`; no image branch, callback or per-step feature
transfer enters the host controller. All owners must use the same text program.

Create the runner before preparation and retain it through partial failure so
`close` can release acquired queues. It borrows the prepared frame, storage,
stream and context; all must use the same text program and outlive the runner.
Close runner first. A submitted frame must finish or be cancelled before a new
request begins. Cancellation synchronizes the retained queue, then a new
request's first frame resets recurrent history/state. Prior KV entries are
hidden by the new sequence length and replaced as the new request advances.

For an offline blocking caller, after startup has prepared the runner:

```mbt check
///|
fn _generate(
  runner : @minicpm_v47_request.Runner,
  prompt : @prompt.PreparedTextPrompt,
) -> Unit raise {
  runner.begin(
    prompt,
    maximum_new_tokens=8,
    stop_token_ids=ReadOnlyArray::from_array(@prompt.stop_token_ids()),
  )
  while runner.submit_next() {
    runner.wait()
  }
}
```

An event-loop caller instead calls `poll` until completion, publishes the newly
emitted token span, and submits the next frame. `wait`/`poll` retire one frame,
not an entire request. Read emitted tokens before starting another request,
because the retained output span is reused. Only final-prefill/decode frames
read the four-byte sampled ID; intermediate prefill has no result readback.
Stop tokens count toward sampled usage and do not enter the output span.
Unsupported stochastic sampling, string stops and batching require an outer
adapter or later profile; they are not silently interpreted as greedy options.
