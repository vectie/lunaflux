# Executable benchmark lifecycle collection

This package is the bounded runner-side bridge into `benchmarks/evidence`.
One `BenchmarkTrialCollector` owns a startup-fixed number of request lanes and
the exact `Submitted -> Admitted -> FirstToken -> Terminal` observations for
one engine/profile trial. Every public observation samples LunaFlux's existing
process-monotonic clock. The source clock is milliseconds; timestamps are
encoded in nanosecond units as exact multiples of 1,000,000 and the evidence
binds the SHA-256 identity of
`lunaflux.benchmark-runner.system-monotonic-milliseconds.v1`.

Request ordinals must be submitted once in canonical order. State changes are
transactional across clock rollback and checked token-total overflow. Finish
fails until every startup lane has exactly one terminal outcome, then passes
the terminal records directly to `BenchmarkSummaryEvidence::admit`. The
evidence package remains the sole owner of summary derivation and canonical
raw/summary digests.

This package deliberately does not launch LunaFlux, vLLM, or SGLang, open a
socket or file, own a device, accept labels, declare correctness, or claim a
performance result. A later comparison campaign adapter should own its pinned
process and protocol authority and call this collector at the four lifecycle
boundaries. Correctness records and `BenchmarkTrial::from_summary` remain
separate promotion obligations.
