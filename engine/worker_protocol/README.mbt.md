# Worker scheduling protocol

This package owns LunaFlux's backend- and model-family-neutral immutable
scheduler/worker message vocabulary. A `SchedulePlan` carries monotonic plan
and loaded-model identities, request cancellation generations, flattened token
and generational page tables, exact model-plan capability IDs, row sampling
parameters, and bounded completion slots. A `CompletionRecord` is accepted only
when every plan, model, request-generation, row-kind, token-count, and slot
identity agrees with the submitted plan.

Plan validation authenticates completed work for resource retirement. Because
cancellation may advance after submission, publication and KV reuse require a
second `CompletionEntry::is_current` check against scheduler-owned current
request state.

The immutable `SchedulePlan` constructor defensively copies caller arrays and
remains the convenient fixture API. The production foundation is a
startup-allocated, fixed-capacity `SchedulePlanBuffer`: the scheduler appends
prefill rows before decode rows, submits one authenticated epoch, retires it
only after separately authenticating worker completion, then resets it for
reuse. Opaque submitted and row handles recheck both lifecycle and epoch on
every access, so an old handle cannot silently read a reused buffer. Exactly
two buffers is a scheduler overlap invariant, not a protocol global.

The mutable owner is single-writer and thread-confined. Append copies each
caller `ArrayView` synchronously; callers must not mutate a source concurrently
during that call. Tables and row columns never grow after construction, and
canonical `PageIdStorage` avoids optional boxed page cells. Native object-code
inspection covers successful append, submit, handle-return, accessor, retire,
and reset paths; the Phase 3 runtime allocation-instrumentation gate is still
required before promoting the whole scheduling path.

Payload-bearing plans, submitted handles, and completions deliberately do not
implement `Debug`, so prompt and generated token IDs cannot enter ordinary
diagnostic formatting. Errors contain only bounded categories and indices.

This does not yet make a production scheduler. Append page inputs are ordinary
views and do not directly borrow request block-table storage; integration needs
a reusable staging/source bridge or scalar append path without per-step array
creation. `CompletionRecord` construction still allocates an immutable copy;
the bounded completion ingress/ring and live worker overlap remain separate
phase gates. `retire` is an ownership assertion after external completion has
been authenticated—the protocol cannot observe device progress itself.

The package imports no API, tokenizer, scheduler-policy, model-family, device,
kernel implementation, or CUDA package.
