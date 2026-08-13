# Scheduler core registry

This package implements LunaFlux's deterministic single-owner scheduler
foundation: bounded request ownership, transactional batch planning,
authenticated completion retirement, and fixed publication rings.
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
Before copying any block-table identity, the scheduler authenticates its exact
generation and positive active-request reference against the canonical page
allocator; the exact submitted plan epoch then carries that authorization to
the in-process worker without exposing allocator mutation authority.
At most one row per request is emitted, and exact token, page, row, completion
slot, and capability budgets are enforced without growing a collection.

Two paired A/B completion owners issue exclusive worker leases only for the
exact submitted plan owner and epoch. Retirement authenticates a complete
batch, preflights all generated-token, terminal, and KV-release obligations,
then commits rows in plan order. Intermediate prefill advances without output;
final prefill publishes the first sampled token; decode publishes subsequent
tokens. Stop-token, maximum-output, cancellation, deadline, and worker-failure
paths release request page/table ownership exactly once. Stale cancelled or
expired completions retire device work without advancing state or publishing a
token. Plans retire strictly in sequence, after which their paired completion
and plan buffers reset for reuse.

Normal scheduling states do not allocate error payloads: `build_next` returns
the flat value-type `BuildNextOutcome` for idle, two-owner backpressure, or a
submitted identity. `submitted_plan` resolves only the exact current owner and
epoch. Completion slots use a fixed O(1) index, and token/terminal dequeue uses
direct value results after checking the corresponding count. A second plan may
use the other owner while the first is in flight; a third attempt reports
value-type backpressure until ordered completion retirement frees a side.

The scheduler claims aging only among eligible prefills, not global bounded
waiting or recomputation-based preemption.
Cancellation or deadline expiry of submitted work advances the authenticated
request generation immediately, but intentionally leaves page/table ownership
attached until completion proves device work is retired; stale work cannot
advance request state or publish a token.
The reusable worker protocol now distinguishes intermediate prefill from a
final prompt chunk that samples the first output token; this scheduler uses
that distinction when building rows. Until a bounded incremental matcher
exists, nonempty stop strings are rejected explicitly. Prefix reuse, generated
text decoding, transport integration, recomputation-based preemption, runtime
allocation instrumentation, and device KV execution remain outside this
package's current evidence.
