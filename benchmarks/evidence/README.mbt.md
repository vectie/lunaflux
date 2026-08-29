# Benchmark evidence admission

This package owns the inert, canonical comparison declaration required by
`docs/BENCHMARKING.md`. It has no filesystem, process, network, device, CUDA,
runner, or release-promotion authority.

Admission requires exact LunaFlux, vLLM, and SGLang identities; the complete
nine-profile workload envelope; three correctness-passed trials for every
engine/profile pair; complete request outcome accounting; independently
digest-bound raw events, correctness records, and summaries; and a
counterbalanced engine order. Inputs are supplied in canonical order so one
logical declaration has one byte representation and one SHA-256 identity.

A `BenchmarkComparison` proves only that evidence has the required shape and
bindings. It does not prove that a runner executed, that reported numbers are
accurate, or that LunaFlux won. Release promotion still requires independent
artifact verification and named reviewer approval.

## Raw events and summaries

`BenchmarkRequestEvent` admits exactly one terminal outcome plus caller-supplied
monotonic-clock nanosecond timestamps. `BenchmarkSummaryEvidence::admit`
accepts at most 8,192 events in exact request-ordinal order with nondecreasing
submission timestamps. It rejects missing outcomes, permutations, malformed
timestamp sequences, and integer overflow rather than repairing the input.

The derived `BenchmarkTrialSummary` contains only integer counts, token totals,
elapsed time, and deterministic nearest-rank p50/p95/p99 distributions for
queue, TTFT, service, decode, and end-to-end time. Empty distributions are all
zero. Raw events and their summary have separate domain-separated canonical
bytes and SHA-256 identities; the summary binds the raw-event digest.
`BenchmarkTrial::from_summary` carries those bound identities and exact outcome
counts into comparison admission.

This layer does not sample wall time, open files, start processes, contact a
network or device, interpret unbounded labels, serialize floating-point values,
or assert that the supplied events came from an executed benchmark.
