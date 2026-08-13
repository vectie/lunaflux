# Worker-service end-to-end gate

This executable joins the real scheduler, worker service, process supervisor,
and separately linked deterministic worker child. It proves two outstanding
plans, output-publication backpressure with exact frame retry, orderly child
close, scheduler worker-failure retirement, non-reusing replacement startup,
and a successful post-restart completion with balanced KV ownership.

The child remains a deterministic protocol implementation rather than a CUDA
worker. Device-backed readiness is a separate physical release gate.
