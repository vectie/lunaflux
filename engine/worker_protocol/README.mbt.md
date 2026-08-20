# Worker scheduling protocol

This package owns LunaFlux's backend- and model-family-neutral scheduler/worker
message vocabulary. A `SubmittedSchedulePlan` carries monotonic plan and
loaded-model identities, request cancellation generations, flattened token and
generational page tables, exact model-plan capability IDs, row sampling
parameters, and bounded completion slots. A `SubmittedCompletion` is accepted
only when every plan, model, request-generation, row-kind, token-count, and
slot identity agrees with the submitted plan.

Prefill completion semantics are explicit. An intermediate prompt chunk only
reports its exact processed-token count and never samples. The final prompt
chunk carries the request sampling parameters and returns both its processed
count and exactly one sampled token. That sampled token is the first generated
output and becomes the input of the first decode row at sequence position
`prompt_length`; each later decode row consumes the sampled token returned by
the preceding decode row. Neither row kind nor output behavior is inferred
from chunk length.

Plan validation authenticates completed work for resource retirement. Because
cancellation may advance after submission, publication and KV reuse require a
second `SubmittedCompletionEntry::is_current` check against scheduler-owned
current request state.

The authoritative plan representation is a startup-allocated, fixed-capacity
`SchedulePlanBuffer`: the scheduler appends prefill rows before decode rows,
submits one authenticated epoch, retires it only after separately
authenticating worker completion, then resets it for reuse. Opaque submitted
and row handles recheck both lifecycle and epoch on every access, so an old
handle cannot silently read a reused buffer. Exactly two buffers is a
scheduler overlap invariant, not a protocol global.

The mutable owner is single-writer and thread-confined. Append copies each
caller `ArrayView` synchronously; callers must not mutate a source concurrently
during that call. Tables and row columns never grow after construction, and
canonical `PageIdStorage` avoids optional boxed page cells. Native object-code
inspection covers successful append, indexed recipe application, submit,
handle-return, accessor, retire, and reset paths. The Phase 3 runtime
allocation-instrumentation gate is still required before promoting the whole
scheduling path.

Schedulers may instead build a row through an authenticated `PlanRowDraft`:
begin, push scalar token/page/capability cells from existing immutable storage,
then commit the exact suffix or roll it back. Until commit, no row descriptor
is visible. This closes the temporary-array bridge for token buffers and block
tables without importing either implementation package.

`RowCapabilityRecipe` is the startup-owned bridge from a model plan to those
drafts. It is pinned to an exact model identity and loaded-plan generation,
defensively owns one nonempty, bounded capability list, and preserves its exact
order, including repeated semantic operation IDs. Scheduler policy can receive
separate prefill and decode recipes without importing model-plan types or
inventing model-family branches. `required_cells` checks the flattened table
capacity for a bounded row count. `PlanRowDraft::push_recipe` proves the whole
remaining capability capacity before copying directly into the preallocated
draft, so a failed application writes no partial suffix and a successful
steady-state application allocates nothing. Recipe construction is the only
allocating step.

A `PlanBuildCheckpoint` provides latest-checkpoint-only transactionality across
several committed rows. It is bound to one writable owner and plan epoch,
stores only scalar logical ends, and rejects open row drafts. Rollback clears
the discarded page-table suffix and restores token, page, capability, prefill,
and decode counts without allocation. It does not release physical KV pages or
mutate request block tables; the scheduler must roll back those owners in its
same failure path.

Worker completions have a matching fixed-capacity `CompletionBuffer`.
Successful intermediate-prefill, final-prefill, decode, and failure appends
write scalar columns. The scheduler retains each mutable completion owner and
issues the worker an exclusive `CompletionWriter` bound to one exact completion
and submitted-plan epoch. Submit authenticates every result against that plan;
abort discards a partial logical suffix and advances the epoch so all aliased
writers become stale before a clean retry. Scheduler accessors remain
epoch-checked through consume and reset. Retirement order is completion
preflight and scheduler commit, completion consume, plan retire, then completion
and plan reset. Plan authentication remains separate from `is_current`:
old-generation work can retire resources but may not publish output.

Payload-bearing plans, submitted handles, and completions deliberately do not
implement `Debug`, so prompt and generated token IDs cannot enter ordinary
diagnostic formatting. Errors contain only bounded categories and indices.

This does not itself make a production scheduler. Convenience `ArrayView`
append callers may allocate their source arrays; allocation-sensitive
integration uses scalar drafts and reusable completion buffers. A transport
ring and live worker overlap remain separate phase gates. `retire` is an
ownership assertion after external completion has been authenticated—the
protocol cannot observe device progress itself.

The package imports no API, tokenizer, scheduler-policy, model-family, device,
kernel implementation, or CUDA package.
