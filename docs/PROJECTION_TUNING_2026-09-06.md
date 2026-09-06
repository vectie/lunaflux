# Persisted offline projection tuning

Implementation scope: a pure, versioned tuning snapshot plus one offline file
adapter shared by candidate export and release binding. A snapshot is scoped
to backend, architecture, exact device identity, and compiler/toolchain digest.
It contains full typed shape/strategy keys, median nanoseconds, and sample
counts. Selecting a snapshot is an explicit artifact-build input, never a
request-time filesystem lookup or benchmark.

The Qwen exporter and release binder accept an optional trailing
`--projection-tuning ROOT RELATIVE_FILE SHA256 DEVICE_ID`. Both reconstruct
the same candidates from the same snapshot. Omission preserves static policy.
The independently supplied device identity is a build target, not a live-device
attestation; deployment must still assign that target. Source and launch
geometry are derived together from selected schedules.

The current artifact producer freezes one schedule at the admitted maximum
token bucket. This is not dynamic per-row kernel replacement: measurements
for an artifact selection must cover its intended row vector, including
single-row decode. A measured win at one row count alone is insufficient to
install a whole-profile default. Runtime graph buckets do not yet imply that
multiple projection artifacts have been compiled and bound.

Implemented and checked: canonical round trips, duplicate/conflicting records,
device/toolchain mismatch, real file loading, exporter/binder argument parity,
and identical producer source/recipes after persistence. Focused tests pass:
snapshot 4, file adapter 2, producer 12, exporter 9, binder 5. The full native
warning-denied compile passes. Physical measurements and end-to-end results
remain separate from serialization and plumbing tests. The checked-in fixture
is explicitly synthetic and is never a serving default.
