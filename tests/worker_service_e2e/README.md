# Worker-service end-to-end gate

This executable joins the real scheduler, worker service, process supervisor,
and separately linked deterministic worker child. It proves two outstanding
plans, output-publication backpressure with exact frame retry, orderly child
close, scheduler worker-failure retirement, non-reusing replacement startup,
and a successful post-restart completion with balanced KV ownership.

The child remains a deterministic protocol implementation rather than a CUDA
worker. Device-backed readiness is a separate physical release gate.

The persistent-online slice uses a startup-sized `LunaRequestPreparationPool`:
external submission yields opaque work, one central owner advances bounded
steps, and Ready work publishes a destructively transferable prepared claim.
Its evidence covers single-lane storage-generation reuse, Saturated/Draining
receipt non-consumption, online Busy/Draining prepared-shell non-consumption,
lower-admission rollback, healthy retirement, recovery close, and stale
Work/Prepared/Claim rejection after reuse. Sequential requests in one child,
foreign preparation binding, queued expiry, stale tickets, and post-admission
failure reuse remain covered. The event evidence additionally
proves single-issued opaque credits, delayed stale-alias rejection after the
next event is live, Usage-to-terminal epoch separation, abort invalidation, and
framed-adapter composition. The harness releases the transport frame before it
ACKs the semantic credit; the adapter never receives ACK authority.

After warm pool startup, proportional tokenization/copy/output-setup progress
is allocation-free. Ready assembly may allocate only constant-size
`TokenizedRequest` and claim shells, and `take_prepared` may allocate one
constant prepared shell; the generated-C gate filters exactly those owners and
rejects payload arrays, byte/text construction, or growing collections.
