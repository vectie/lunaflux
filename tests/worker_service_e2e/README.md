# Worker-service end-to-end gate

This executable joins the real scheduler, worker service, process supervisor,
and separately linked deterministic worker child. It proves two outstanding
plans, output-publication backpressure with exact frame retry, orderly child
close, scheduler worker-failure retirement, non-reusing replacement startup,
and a successful post-restart completion with balanced KV ownership.

The child remains a deterministic protocol implementation rather than a CUDA
worker. Device-backed readiness is a separate physical release gate.

The persistent-online slice prepares requests outside the online instance and
then destructively transfers one preallocated `LunaPreparedRequest` claim. Its
evidence covers sequential requests in the same child, Busy/Draining
non-consumption and retry, foreign preparation binding, queued expiry without
scheduler mutation, retained-alias replay rejection, stale tickets, and
instance reuse after post-admission failure. The event evidence additionally
proves single-issued opaque credits, delayed stale-alias rejection after the
next event is live, Usage-to-terminal epoch separation, abort invalidation, and
framed-adapter composition. The harness releases the transport frame before it
ACKs the semantic credit; the adapter never receives ACK authority.
