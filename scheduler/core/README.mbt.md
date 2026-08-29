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

Stop-token semantics cross the boundary only through an opaque token-only Luna
projection. Cache-aware admission adds a second narrow opaque projection that
exposes only validated permission and security-scope bytes, bound to the exact
tokenizer digest; stop strings, inference limits, raw token arrays, and full
semantic views are not scheduler inputs. Legacy `TokenizedRequest::new` remains
explicitly cache-disabled. Admission authenticates the projections, then retains one exact semantic authority after every request
preflight and deadline check and immediately before the first scheduler
mutation. Prepared admission lets the online worker acquire its independent
retention before sampling the clock and committing. The semantic lease cannot
release while either retention is live; every slot recycle releases the
scheduler retention exactly once. Completion uses bounded allocation-free
membership and maps stale authority to `Request`/`Stale` before mutation.

The owner allocates fixed request-slot arrays and one intrusive waiting queue
at startup. Unaged admission candidates prefer the largest reusable full-page
prefix, with FIFO as the exact tie-break. Once the oldest eligible request
reaches the configured age bound it has absolute priority, so cache affinity
cannot starve ordinary work. A scheduler-global `RequestGeneration` sequence advances on
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
an outer retained alias released or reused the prompt-token or semantic
backing after prepare. Both backings are reauthenticated before the deadline;
the scheduler retention is then acquired before installation. Saturated
retention invalidates without scheduler mutation. Every noncommit result
clears the reservation. Abort, invalidation, and expiry are one-shot, so an
aliased stale shell cannot later admit.

After an unrecoverable worker-instance loss, `drain_instance_loss` retires at
most one live request per call, publishes `WorkerFailed`, and releases active
page/table authority or removes one waiting request transactionally. Terminal
backpressure mutates nothing; `Complete` means no live request remains and any
device-derived cached prefix anchors have been released.
Recoverable device-state invalidation also clears every cached physical prefix
anchor. Waiting requests retain only logical match evidence, never a table,
page, or radix-entry reference, so invalidation clears that evidence before the
replacement worker becomes usable; no old `PageId` can cross the device
generation boundary.

Startup authenticates separate immutable intermediate-prefill, final-prefill,
and decode capability recipes against the selected model identity and loaded
model-plan generation, proves the worst-case per-step capability-cell envelope,
retains the exact resolved worker-protocol limits for downstream owner binding,
and allocates two physically distinct plan owners plus fixed selection and
ownership journals.

`build_next` keeps waiting requests logically queued while selecting work.
Only a selected row generation-checks its exact salted prefix evidence and
transactionally retains the matching entry/pages into fixed plan scratch. A
stale candidate is cleared immediately and falls back to uncached admission or
remains waiting without physical authority; all pre-submit checkpoint and
build failures release the exact selected references. Hits and reused-token
telemetry commit only with a successfully submitted plan. The scheduler
reserves decode work before prefill policy, preserves the
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
the flat value-type `BuildNextOutcome` for idle, two-owner backpressure, one
completed recompute-preemption transition, or a submitted identity.
`submitted_plan` resolves only an exact submitted owner and rejects the
preemption outcome. Completion slots use a fixed O(1) index, and token/terminal
dequeue uses direct value results after checking the corresponding count. A
second plan may use the other owner while the first is in flight; a third
attempt reports value-type backpressure before any preemption policy mutation.

An exactly eligible aged prefill is selected before decode rows. An ineligible
aged row reserves no token or page budget. When active-room or page pressure is
the blocker, deterministic zero-active-reference prefix eviction is attempted
before the scheduler may preempt one non-inflight active prefill or
decode in persistent slot-cursor order. It releases only device-derived KV
state and places the request back in the FIFO. Generated-token history remains
in dense startup storage. Replay uses ordinary non-sampling prefill over the
prompt plus all generated tokens except the retained latest token, then resumes
decode; the public processed-input count remains unchanged. Preemption can
temporarily raise the internal waiting count above the configured admission
waiting bound, but never above the fixed total-slot capacity, and new admission
remains blocked by its normal capacity checks. Decode row selection uses a
separate commit-only round-robin cursor so every ready slot is eventually
selected when ready decodes exceed the row limit.

`LunaSchedulerTelemetry` is an allocation-free scalar snapshot of waiting
depth, active requests, exact KV pages used/free, prefix lookups/hits/misses,
evictions, reused/computed tokens, publications, and live prefix entries/pages.
The separate value-type
`LunaSchedulerPlanTelemetry` retains the last successfully submitted plan's
sequence, row count, and token budget. Those plan scalars change only with the
same transaction that commits submission; failed, idle, backpressured, and
preemption-only builds cannot advance them. Neither snapshot exposes a plan,
page, table, request, mutable owner, or generation capability.
Cancellation or deadline expiry of submitted work advances the authenticated
request generation immediately, but intentionally leaves page/table ownership
attached until completion proves device work is retired; stale work cannot
advance request state or publish a token.
The reusable worker protocol now distinguishes intermediate prefill from a
final prompt chunk that samples the first output token; this scheduler uses
that distinction when building rows. Stop strings remain an outer incremental
output concern. Prefix lookup reuses only complete pages strictly before the
last prompt token, so every request preserves its caller-visible processed
input count, generation, deadline, generated-token history, and final-logit
semantics. Final prefill may publish full prompt pages under `ReadWrite`; the
scheduler preflights and retains physical cached references before committing
radix metadata, and releases only page IDs returned by deterministic eviction.
Duplicate publication, full prefix capacity, and an empty eviction set remain
opaque scalar start outcomes on this warmed path rather than typed exceptions.
The radix package never mutates page ownership. Physical CUDA numerical reuse,
hardware benchmarks, generated text decoding, and process I/O remain separate
evidence gates.
