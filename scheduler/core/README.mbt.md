# Scheduler core registry

This package begins LunaFlux's deterministic single-owner scheduler with the
request boundary and lifecycle state that later batch planning will consume.
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
so a rejected sweep does not partially mutate requests. Active or in-flight
requests retain their slot and resource ownership after their generation is
advanced; resource release belongs to authenticated completion retirement.

This slice deliberately does not claim a production batch scheduler. It does
not yet activate waiting requests, allocate request block tables/pages, build
plans, retire completions, publish generated tokens, or integrate prefix reuse.
The reusable worker protocol now distinguishes intermediate prefill from a
final prompt chunk that samples the first output token; the next scheduler
slice must use that contract and prove page rollback, decode reserve, chunking,
aging, completion retirement, and hot-path allocation gates. Stop-token and
maximum-output enforcement will be scheduler-owned; stop-string decoding needs
a separate bounded incremental matcher and is not implemented here.
