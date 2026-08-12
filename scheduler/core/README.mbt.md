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
deadline generations are ready for later authenticated completion retirement,
but this slice creates no active or in-flight request resources.

Startup authenticates immutable prefill/decode capability recipes against the
selected model identity and loaded model-plan generation, and proves the
worst-case per-step capability-cell envelope. Distinct A/B plan and completion
owners, FIFO activation, and page transactions remain part of the next real
`build_next` operation rather than a public staging API.

This slice deliberately does not claim a production batch scheduler. It does
not yet allocate pages for prompt chunks, build or submit plans, retire
completions, publish generated tokens, or integrate prefix reuse.
The reusable worker protocol now distinguishes intermediate prefill from a
final prompt chunk that samples the first output token; the next scheduler
slice must use that contract and prove page rollback, decode reserve, chunking,
aging, completion retirement, and hot-path allocation gates. Stop-token and
maximum-output enforcement will be scheduler-owned. Until a bounded
incremental matcher exists, nonempty stop strings are rejected explicitly.
