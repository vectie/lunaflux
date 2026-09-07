# Per-row-bucket AOT projection artifacts

## Implementation

Dense and QKV projection export now retains measured small-row distribution
winners alongside the maximum-profile kernel, in the same AOT module. Each
optional variant declares a token-row bound, symbol, and launch dimensions.
The original entry remains the fallback. Without applicable tuning records,
the exporter produces no additional symbols or metadata.

The existing backend-neutral projection strategy and tile compilation pipeline
select and schedule each variant. CUDA lowering emits its device implementation.
Artifact and full-graph manifest serialization carry the typed variants through
startup loading. Startup creates smaller bounded queues using the smallest
covering variant and its bucket-specific geometry. No tuning, symbol lookup,
or filesystem access is added to the token-step path.

Variants retain the parent's operand layout and numerical contract. They are
currently consumed only by BF16 QKV/output projection execution; I8 manifests
reject this extension. Fused ingress does not automatically acquire these
projection variants. MLP, language-model head, attention, and alternative
residency schedules are not covered by this first extension.

## Validation and remaining work

Targeted native tests cover bounded selection, fallback, launch geometry,
metadata serialization, and invalid bounds/symbol aliases. An opt-in physical
test compares two dense schedules at token counts 1, 2, 7, 8 and the primary
fallback at 9, 16, 32, 256; it checks captured execution and untouched output
tails. Its synthetic tuning record is a test fixture, not a performance record.

The exact `f69e798` source passed warning-denied Linux native checking and the
physical dense fixture on the RTX 5060 Ti. Captured queues used the actual
`paged_ordered_launches` variant consumer. All four sanitizer runs passed:
memcheck, synccheck, and initcheck reported zero errors; racecheck reported
zero hazards. The first `1bd8112` physical test failed because its fixture
assumed candidate-array index 1 meant two workgroups, while the candidate
actually used one. The correction selects the candidate by explicit width;
the first failure logs remain archived.

Local affected-package tests passed: artifact 22, artifact-file 23, full-graph
manifest 4, execution-manifest-file 20, projection AOT 16, kernel bundle 16,
device-step 160 (261 total; opt-in GPU test returns early locally). Native
warning-denied checking passed. An earlier overly broad package-name filter
also selected the pre-existing failing FP8 projection test; this is not a
claim that the repository-wide suite passes.

## Kernel-only timing

BF16 dense 1024-by-1024, all-one input and 1/1024 weights, seven CUDA-event
samples per configuration, 64 repeated captured launches per sample; medians
below are microseconds per kernel. The smaller variant uses two workgroups
instead of eight. Configurations were measured sequentially, not interleaved;
these are diagnostic timings, not production tuning records.

| Token rows | Primary | Row variant | Time reduction |
|---:|---:|---:|---:|
| 1 | 2.684 | 2.553 | 4.9% |
| 2 | 13.233 | 10.389 | 21.5% |
| 7 | 13.257 | 10.406 | 21.5% |
| 8 | 13.250 | 10.425 | 21.3% |

Primary fallback medians at rows 9, 16, 32, 256 were respectively 13.260,
13.260, 13.272, 51.516 microseconds. These rows have no variant comparison.
The result supports retaining bucket-specific alternatives; it does not
establish a Qwen serving speedup or justify installing this synthetic-fixture
record in a production tuning database.

Run directory:
`/dev/shm/lunaflux-row-variants-f69e798-20260907-r2`.
Source archive SHA-256:
`ad9b81fa345cb22d7f05bd6a4937fae488946cbf5d9e62b2b60043e3a2c9d814`.
Results archive:
`/private/tmp/lunaflux-row-variants-results-f69e798-20260907.tar.gz`.
Archive SHA-256:
`d2cbd2f198be8f0297012a8768ac45eba273b6e78991ab61c69d286d910f8f10`.

## Still remaining

QKV physical differential coverage and real per-bucket Qwen timing are still
needed before installing tuning winners. This change does not resolve the C8
token variation investigation or implement larger matrix/attention schedules.
No new Qwen end-to-end benchmark result is claimed. The GPU was idle after
the run; production deployment was not changed.
