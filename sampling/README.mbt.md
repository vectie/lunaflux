# Bounded stochastic sampling

The stochastic path consumes the canonical, already validated
`contracts/inference.SamplingParameters`. It does not reinterpret zero
temperature as greedy selection: callers use the existing `greedy` APIs for
greedy requests, while `stochastic_sample` rejects greedy parameters.

For each call, LunaFlux revokes the previous scratch snapshot, validates every
logit before writing a new candidate or weight or changing RNG state, orders
candidates by descending logit with the lowest token ID first on ties, applies
top-k, computes a maximum-shifted temperature softmax, and finally applies
top-p to that top-k distribution. Top-p keeps the smallest non-empty leading
nucleus whose cumulative mass reaches the requested threshold. Sampling then
renormalizes over the retained nucleus.

An explicit restrictive top-k retains the exact best `K` candidates in a
worst-root heap and sorts only those candidates. Its worst-case ordering work
is `O(V log K)` with `O(V)` preallocated scratch. Candidate identity, tie
order, exponent evaluation, and every floating-point accumulation remain
identical to sorting the full vocabulary first.

Without a restrictive top-k, the canonical path truthfully remains
`O(V log V)`. In particular, top-p requires the complete softmax total before
the nucleus threshold is known. Computing that total in token-ID order or
sampling unfiltered token-ID intervals directly would change IEEE-754 addition
order and the fixed-draw token mapping. LunaFlux therefore preserves the
descending canonical order rather than introducing a hidden truncation or a
numerically different shortcut.

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

`stochastic_sample_at_scalars` applies the same distribution and counter draw
to already validated plan-frame scalars. This avoids reconstructing a
heap-backed request parameter object in the isolated worker; replay tests pin
its result to the canonical parameter path across a bounded sequence.

The source structure, fixed storage, release-C allocation probe, and structural
work counters establish the intended allocation-free steady-state algorithm
and its top-k bound. Production latency/throughput evidence has not yet been
captured; those performance claims remain open until the release gates run on
production hardware.
