# Scheduler core registry

This package implements LunaFlux's deterministic single-owner scheduler
foundation: bounded request ownership, transactional batch planning,
authenticated completion retirement, and fixed publication rings.
It accepts only immutable `TokenizedRequest` values, never HTTP or text input,
and validates selected model identity, duplicate live request IDs, prompt and
context bounds, page-envelope capacity, and checked absolute monotonic
deadlines before owning a request slot.

Prompt token-range preflight authenticates the opaque `TokenBuffer` backing
generation and compares its exact cached maximum against the loaded worker
vocabulary in constant scalar work. It neither exposes nor rescans prompt
tokens. Later plan construction still reads each authenticated token needed to
populate a bounded prefill row.

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

Generated-token and terminal rings remain physically separate and fixed, but
every committed notice receives one scheduler-global monotonic publication
sequence. `take_next_publication` atomically selects and consumes the globally
oldest ring, allowing an online adapter to preserve exact token/terminal
order—including the cancellation cut—without merging storage or allocating an
event queue. The lower-level per-ring dequeues remain fail-closed against a
wrong-ring choice.

Exact cancellation also supports an owner-resident reservation transaction.
It preflights the live handle, terminal credit, fresh request/publication
generations, and deterministic KV release before a session consumes output.
While reserved, every ordinary scheduler mutation is rejected; only the next
globally ordered generated notice for that exact request generation may be
dequeued, at most once. Commit is non-raising and always publishes the physical
`CancelledByCaller` reason. A higher session may bind its private string-stop
or output-failure cut only to that exact next-generation notice. A queued
natural terminal is classified without reserving or inventing a cancellation.

Owned online startup uses a separate boundary-restricted exclusive admission
shell. Timeless validation and exact handle storage complete before rooted
worker activation. While that shell is reserved, the same global mutation gate
rejects ordinary admission, planning, cancellation, expiry, and instance-loss
drain. Deadline is sampled only after startup; commit either installs the
preallocated handle, consumes the shell as expired, or returns Invalidated if
an outer retained alias released or reused the prompt-token backing after
prepare. Backing reauthentication precedes the deadline check, and every
noncommit result clears the reservation. Abort, invalidation, and expiry are
one-shot, so an aliased stale shell cannot later admit.

After an unrecoverable worker-instance loss, `drain_instance_loss` retires at
most one live request per call, publishes `WorkerFailed`, and releases active
page/table authority or removes one waiting request transactionally. Terminal
backpressure mutates nothing; `Complete` means no live request remains.

Startup authenticates separate immutable intermediate-prefill, final-prefill,
and decode capability recipes against the selected model identity and loaded
model-plan generation, proves the worst-case per-step capability-cell envelope,
retains the exact resolved worker-protocol limits for downstream owner binding,
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
and plan buffers normally reset for reuse. A semantically invalid submitted
completion is replaced by an exact `WorkerFailed` outcome only after terminal,
publication, release, and identity preflight; publication backpressure consumes
nothing. If either paired owner can no longer survive replacement, the pair is
retired terminally as completion `Consumed` and plan `Retired`. The other pair
remains reusable, and planning reports `Plan`/`Exhausted` rather than transient
backpressure once no reusable or retireable pair remains.

The scheduler has no serialized worker-frame dependency. Its caller receives
an exact protocol completion writer, stages authenticated scalar outcomes, and
submits that paired epoch. `complete` returns the allocation-free
`CompletionBackpressured` value when output or terminal publication lacks
capacity; semantic and ownership failures remain typed errors. The caller may
then use `complete_submitted` to reauthenticate and retry the exact current
completion without retaining or reconstructing a mutable owner.

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
text decoding, process I/O and supervision, recomputation-based preemption, runtime
allocation instrumentation, and device KV execution remain outside this
package's current evidence.
