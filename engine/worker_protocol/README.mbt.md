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

Construction defensively copies caller arrays. Payload-bearing plans and
completions deliberately do not implement `Debug`, so prompt and generated
token IDs cannot enter ordinary diagnostic formatting. Errors contain only
bounded categories and indices.

This is the semantic protocol foundation, not the Phase 3 allocation proof.
Its constructors allocate immutable copies. The scheduler integration must own
and instrument a reusable double-buffered builder before claiming no general
heap allocation after warm-up; this package makes no such claim.

The package imports no API, tokenizer, scheduler-policy, model-family, device,
kernel implementation, or CUDA package.
