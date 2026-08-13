# Bounded stochastic sampling

The stochastic path consumes the canonical, already validated
`contracts/inference.SamplingParameters`. It does not reinterpret zero
temperature as greedy selection: callers use the existing `greedy` APIs for
greedy requests, while `stochastic_sample` rejects greedy parameters.

For each call, LunaFlux validates every logit before changing RNG state, orders
candidates by descending logit with the lowest token ID first on ties, applies
top-k, computes a maximum-shifted temperature softmax, and finally applies
top-p to that top-k distribution. Top-p keeps the smallest non-empty leading
nucleus whose cumulative mass reaches the requested threshold. Sampling then
renormalizes over the retained nucleus.

Greedy and stochastic selection share the same strict finite-logit policy: a
NaN or either infinity anywhere in the input row rejects the whole row, even
when another token would otherwise win. This keeps corrupt kernel output from
being silently masked by a filter or maximum operation.

`SamplerScratch` owns fixed token-ID and weight arrays allocated at
construction. Preparation and selection overwrite those arrays without
growing a collection, and observation exposes only scalar values. The RNG is a
specified xorshift64 stream with an explicit non-zero canonical request seed;
one successful stochastic selection consumes exactly one value. The
counter-addressed `stochastic_sample_at` path derives an independent draw from
`(seed, sample_index)` so isolated-worker retry does not depend on shared
mutable RNG state.

The source structure and fixed storage establish the intended allocation-free
steady-state algorithm, but native allocation instrumentation and production
latency/throughput evidence have not yet been captured. Those performance
claims remain open until the release gates run on production hardware.
