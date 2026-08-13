# A/B isolated-worker supervisor

This package owns one exact child channel plus physically distinct A/B plan and
completion frame owners. Submission is strictly monotonic and allocation-free
after startup storage. At most two plans may be in flight; the oldest response
is received first.

A response side remains pinned after wire validation. Callers may inspect its
exact frame repeatedly while scheduler publication is backpressured, and only
`retire_received` permits that side to be reused. Foreign, stale, out-of-order,
malformed, partial, timed-out, or closed-channel traffic fails closed. Native
process handles and transport buffers never escape.

The production worker executable and restart/readiness policy remain separate.
This supervisor intentionally does not infer recovery after process failure:
the service must close it, retire submitted work through scheduler failure
semantics, construct a replacement, and seed the last accepted predecessor.
