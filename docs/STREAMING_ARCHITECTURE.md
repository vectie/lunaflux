# Streaming residency decision

The `stream` workstream uses `engine/streaming_ir` as the backend-neutral
transaction vocabulary shared by scheduler, transport and device interpreter.
Its `Control::decide` function is pure: it maps immutable protocol state plus
a bounded command to an immutable next state and an effect tag. The worker
interpreter commits the decision and issues the corresponding cache operation.
The existing pure KV residency state machine separately owns DMA cancellation,
quarantine, drain and host-slot generation transitions.

## One logical prefix index

The existing radix owns model/tokenizer/layout/security-scope matching for both
tiers. A page anchor records resident or host-backed payload placement. Shared
ancestors remain one anchor and one cached reference. Only complete immutable
pages with zero active physical references may spill. The scheduler retains a
source reference while DMA is live and releases device capacity only after the
child confirms completion. Copying one page never changes its logical key.

A multi-page restore reserves every destination before submission and pins the
protecting radix entry. All destinations remain private until the entire copy
transaction completes. Cancellation suppresses publication and retains the
reservations until the submitted copy has completed or the child is reaped.
The resulting resident radix anchors are acquired by ordinary request planning;
the transfer path cannot directly activate a request or publish a token.

## Process and startup boundary

The private 64-byte transfer frame contains version, child epoch, transaction
sequence, opcode, page index/generation, and host slot/generation. It contains
no pointer, owner reference, model text or filesystem locator. The parent uses
the same cooperative framed transport as graph plans. Child polls return
immediately; a ready graph cannot stage while DMA is pending. Spilling mode
disables the speculative second graph-plan buffer until the first finishes.

Startup protocol v5 adds a bounded immutable streaming capability and a child
epoch to v4; unconfigured v4 frames retain their exact encoding. Replacement
increments the epoch without changing model identity or graph-plan sequence.
Device loss invalidates every old host identity after child cleanup. The
runtime descriptor's explicit `lunaflux.runtime.v6` schema admits the host
budget and measured per-page cost profile. No token-path environment switch,
cryptographic verification, qualification scan or new backend import enters
the scheduler.

This decision owns the experimental boundary until the streaming workstream's
physical parity/performance gate. At that boundary the v6 capability is either
promoted with evidence or removed; no second scheduler or prefix index is kept.
