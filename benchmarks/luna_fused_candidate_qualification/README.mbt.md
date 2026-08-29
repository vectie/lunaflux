# Fused candidate benchmark qualification

This native-only package admits inert microbenchmark observations for the two
Phase-5 fused candidates. It owns no clock, runner, process, CUDA context,
device allocation, manifest, runtime, deployment approval, or promotion path.
All measurements are caller-supplied raw claims. The result is useful only as
one input to later independent physical reproduction and release review.

The fixed shape identity uses the exact canonical four-case boundary matrix
shared with the independent scalar referee and physical-evidence admission:
origin, page tail, cross-page pair, and the exact eight-token page-boundary
envelope. Qualification requires 16 warmup
iterations, eight measured pairs per shape, and 64 identical repetitions per
arm. Baseline-first and candidate-first execution alternate by trial ordinal.
Missing, duplicate, reordered, or widened coverage is rejected.

The win policy deliberately makes the Phase-5 plan's “wins every declared
microbenchmark shape” gate conservative and integer-only:

- the candidate must be strictly faster in every paired observation; and
- its aggregate duration must be at least five percent lower on every shape.

This avoids promoting a noisy aggregate that hides a per-pair regression. It
is intentionally stronger than a bare total-duration win. It does not replace
the separately required sanitizer/race evidence or the end-to-end mixed
workload non-regression gate.

Evidence binds the exact candidate, source, recipe, sm120 target, profile,
launch dimensions, compiler identity and flags, numerical policy, standalone
fallback source/recipe identities, clock and resource-collector identities,
fixed shape-set identity, counts, raw paired durations and work counters,
dispatch canaries, resource balances, and deterministic integer summaries.
The canonical record always says `manifest_bindable=false` and
`promotion_authority=false`.

The generic `benchmarks/evidence` types are not reused here because they own a
three-engine, nine-profile, 81-request-trial service comparison. Recasting a
kernel pair as a service engine/profile would weaken both type boundaries. The
older candidate-package raw evidence is also not extended: benchmark
qualification belongs here and must not enlarge a CUDA candidate package into
a generic benchmark or promotion authority.
