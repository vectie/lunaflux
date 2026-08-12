# Scheduler core registry

This package begins LunaFlux's deterministic single-owner scheduler with the
request boundary and fixed owner storage that later batch planning will
consume.
It accepts only immutable `TokenizedRequest` values, never HTTP or text input,
and validates selected model identity, duplicate live request IDs, prompt and
context bounds, page-envelope capacity, and checked absolute monotonic
deadlines before owning a request slot.

The owner allocates fixed request-slot arrays and one intrusive FIFO waiting
queue at startup. A scheduler-global `RequestGeneration` sequence advances on
every admission, cancellation, and deadline transition, so re-admitting the
same external request ID cannot make an old worker completion current even if
the new request occupies a different slot. Slot generations independently
reject stale local handles.

Waiting cancellation and deadline expiry publish slot-free terminal notices
through a fixed-capacity ring before recycling request storage. Publication
backpressure and generation exhaustion are preflighted before deadline sweeps,
so a rejected sweep does not partially mutate requests. Cancellation and
deadline generations authenticate later completion retirement even after a
request becomes active or in flight.

Startup authenticates separate immutable intermediate-prefill, final-prefill,
and decode capability recipes against the selected model identity and loaded
model-plan generation, proves the worst-case per-step capability-cell envelope,
and allocates two physically distinct plan owners plus fixed selection and
ownership journals.

`build_next` activates FIFO waiting requests only after a complete submitted
plan exists. It reserves decode work before prefill policy, preserves the
physical emergency page reserve for prefill, and serializes prefill rows before
decode rows as required by the worker protocol. Page, block-table, and plan
checkpoints make a failed build restore exact owner identities and FIFO state.
At most one row per request is emitted, and exact token, page, row, completion
slot, and capability budgets are enforced without growing a collection.

This slice deliberately stops at authenticated plan submission. It does not
retire completions, publish generated tokens, or integrate prefix reuse. Each
submitted plan keeps its A/B owner unavailable; a second plan may use the
other owner for different non-in-flight requests, and a third build is
backpressured until future completion authentication retires and resets the
exact owner. The scheduler claims aging only among eligible prefills, not
global bounded waiting or recomputation-based preemption.
When no request is eligible, `build_next` returns the typed `NoRunnableWork`
issue before opening any ownership checkpoint.
Cancellation or deadline expiry of submitted work advances the authenticated
request generation immediately, but intentionally leaves page/table ownership
attached until the deferred completion-retirement slice proves device work is
retired; stale work cannot advance request state or publish a token.
The reusable worker protocol now distinguishes intermediate prefill from a
final prompt chunk that samples the first output token; this scheduler uses
that distinction when building rows. Completion retirement, stop-token and
maximum-output enforcement remain scheduler-owned follow-up work. Until a bounded
incremental matcher exists, nonempty stop strings are rejected explicitly.
