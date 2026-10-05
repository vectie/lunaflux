# AKO mixed decode companion and 64K-history probes

## Hypothesis and corrected priority

Correction after the fresh kernel-symbol timeline collected later on October 5:
pure C1 selects partitioned decode, but pure C2 still selects ordinary decode.
The earlier interpretation of numeric owner 77 as partitioned was incorrect;
owner IDs alone are not kernel identity. See the
[subsequent long-stage report](BENCHMARK_AKO_LONG_STAGE_AND_C2_DECODE_2026-10-05.md).
This experiment targeted the **mixed** graph's ordinary c468
decode companion. Route 7 already represents unsplit prefill followed by
partitioned decode; its executable preparation exists, but the completed
mixed-route table selected route 1 and contained no route-7 measurements.

This is not a missing compiler IR layer or a new attention algorithm. The
experiment preserves c322 prefill, immutable KV, ordered effects, numerical
contracts and existing AOT modules. Only the decode companion's complete
ordinary versus partial+merge chain differs. Missing measured coverage is
established by source/dispatch, not guessed from occupancy.

Budget: two mixed shapes on .179, independent repetitions of those two on .178,
and two 64K-history decode shapes on .178; then one four-start serving ABBA and
a separate actual dispatch check. GPU work is serialized per machine. Probe
units have MemoryMax=8 GiB, zero swap, finite runtime and a 32 GiB available
memory reserve. Serving uses the existing bounded launcher plus an outer
48 GiB no-swap/1200-second unit. No production service was changed.

## Complete mixed attention chain

Both sides use the same c322 prefill recipe/cubin and query metadata, excluding
the one decode row. Ordinary decode consumes CSR offsets. The partitioned
alternative consumes the same counts, positions, CSR and immutable KV and
includes its merge; no partial-only timing is used. Runtime capacity geometry
is retained, including inactive rows. Five alternating paired trials average
30 complete chains after warmup. Uploads, allocation and checks are untimed.

| GPU / prefill queries / prior histories (prefill, decode) | Ordinary companion chain, ms | Partitioned companion chain, ms | Median paired gain | Worst paired gain |
| --- | ---: | ---: | ---: | ---: |
| .179 / 1536 / 30976,32512 | 7.367 | 6.557 | 11.24% | 9.06% |
| .179 / 512 / 32000,32512 | 4.471 | 3.570 | 20.27% | 18.99% |
| .178 / 1536 / 30976,32512 | 7.134 | 6.381 | 10.55% | 9.16% |
| .178 / 512 / 32000,32512 | 4.424 | 3.482 | 21.24% | 20.39% |

Each cell clears the predeclared 3% gain in every pair. These are warmed-buffer
whole **attention-chain** observations, not whole-model gains or cross-engine
comparisons. Only .179's own observations enter .179's table.

Full differential output/KV checks and the sampled independent FP64 oracle pass
under the 0.003 BF16 tolerance. The traced shape reports maximum difference
0.000488281 and oracle error 0.000373563; it is **not bitwise equivalent**.
The .179 first mixed shape passes memcheck, racecheck and synccheck with zero
errors/hazards. This validates the changed probe composition, not every serving
shape or model-quality equivalence. Synthetic operands reproduce the traced
shape, not that layer's exact values. No new hardware counters were collected.

Exact .179 artifacts: c322 prefill cubin
`9aedf7709b833ad4ba05bf017d54199b33908e89d63c7fae51e73f94cf0a655d`;
c468 ordinary and partial/merge cubin
`b4e24a7bd1014106c55f0eb215a4dff2cfa7c29e4c33ee2edb60b97fc4a5dd7a`.
GPU UUIDs are `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6` (.179) and
`GPU-9c3d3cf0-439a-5da2-67e9-20414255879f` (.178), both GB10/sm121.
nvcc 13.0.88 executable SHA-256 is
`fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.

The existing code decomposes the decode KV domain into eight partitions, giving
more concurrent work than the ordinary rows×KV-head grid. That is a plausible
latency-hiding explanation, not a new instruction-level counter attribution.
The direct paired result justifies testing route selection; it does not justify
an extrapolation to larger batches, which lost in the preceding frontier test.

## 64K-history decode on .178

Both recipes resolve the ordinary and partial/merge symbols from the same c468
cubin. Five paired trials, full differential/KV checks and sampled FP64 oracle:

| Rows / final context | Ordinary, µs | Partitioned chain, µs | Median paired gain | Correctness |
| --- | ---: | ---: | ---: | --- |
| 1 / 65536 | 2731.181 | 1157.500 | 57.42% | bitwise equal, maximum difference 0 |
| 2 / 65536 | 2718.363 | 2316.682 | 14.74% | maximum difference 0.000488281 |

Oracle maximum errors are 0.000242835/0.000243832, within the existing 0.003
tolerance. Both retain immutable KV and release the GPU. Minimum sampled
MemAvailable remains above 116 GiB. These are **kernel-only** synthetic probes:
Qwen3-0.6B's admitted serving context is still 32768; no 64K/1M model-serving,
RoPE extrapolation, quality or memory-capacity claim is made.

## Startup selection and functional compiler boundary

The offline adapter now recognizes `--mixed-decode-chain`, requires identical
prefill artifact paths, and emits candidate IDs 1/7 with actual unequal row
histories. When refreshing a measured bucket, it replaces that bucket's old
alternatives as a group; it never combines a new candidate with a historical
baseline or creates duplicate bucket records. Unaffected pure/mixed buckets
remain intact, including the separate 4096-query mixed bucket.

`engine/device_step/mixed_decode_split_prepare.mbt` consumes route 7 at startup,
reuses the legal read-only partial/merge functions and scratch, and charges its
capture to the existing graph-memory budget. Dispatch remains an immutable
owner lookup. No request-path timing, filesystem admission, JIT, model-name
branch or new mutable global was introduced. Production compiler/runtime source
and kernel bytes are unchanged; this is a newly measured benchmark bundle.

Affected `.mbtx` checks/tests pass with warnings denied; tests cover unchanged
prefill identity, unequal histories, 64K aggregate page bounds and replacing
different representatives of the same runtime bucket. Script-mode `moon info`
is unsupported because this directory is not a package; no public API changed
and no full-repository release claim is made.

## Serving boundary

The route-7 bundle was compared against the preceding mixed-tuned bundle at
input 32512/output 64, C1 control and C2. Four fresh starts in ABBA order,
one excluded warmup plus three measured trials per cell/start, exact input
bodies and all output/token-arrival vectors are saved. Normal serving stderr
is empty; all requests complete with 64 output tokens. No other GPU workload
runs during either serving or the separate diagnostic replay.

| Cell | Prior completion, ms | Route-7 completion, ms | Completion reduction | Prior / route-7 TTFT, ms | Prior / route-7 output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 control | 4607.5 | 4612.5 | -0.11% | 3137.5 / 3139.0 | 13.891 / 13.875 |
| C2 | 9489.0 | 9473.5 | 0.16% | 4753.5 / 4743.0 | 13.489 / 13.511 |

These are median observed rates including prefill, not decode-only rates.
The C2 reduction is **not a robust whole-engine win**. Only a small number of
mixed tail steps changes: a 0.8-ms layer-chain saving is approximately 23 ms
across 28 layers, not 10–21% of the entire request. C1's control movement and
fresh-start variation must not be treated as an optimization effect.

More importantly, the baseline repeats identical token vectors on both cells,
whereas route 7's C2 repeated identical-input waves differ by up to 32/64
positions in row 0 and 9/64 in row 1. C1 remains repeatable. Sampled BF16 probe
tolerance and sanitizers do **not** establish repeatable greedy model output.
This does not isolate a race: different mixed-step timing/ownership and rounding
can also change a near-tied greedy decision. That cause remains unproven.
Full vectors, arrival times, input bodies and cross-route differences are in
`serving-comparison.json`; no accuracy or quality equivalence is claimed.

The separate instrumented C2 replay confirms propagation: at sequence 32,
actual queries are 1536 prefill plus one decode with histories 30976/32512.
`kind=17` selects owner **96** for mixed bucket 4068 (2 rows, 2048 queries,
32768 context), and `kind=3/4/5` submits/completes that same owner **96**.
Route 7 maps to this prepared mixed partial+merge owner; the preceding bundle
used owner 27 followed by its ordinary mixed companion 28. Pure decode still
executes its existing owner 77. This trace is diagnostic, not a timing sample.
The runtime digest is
`e3ce996539ff152ff575dcb3ae53b99d107cd398e7a347dc6df0f0ef1dca46bd`.

**Decision:** retain route 7 as a benchmark-only candidate, not the new serving
default. Keep the preceding mixed-tuned bundle. The new probe/adapter coverage
is useful, but neither a tiny serving delta nor numerical instability should
be promoted as a production improvement. Before another route change, isolate
the first divergent logit/argmax and the full-request cost of mixed versus pure
steps. Repeat the existing comparison with that diagnosis; do not generalize
the 64K kernel result into a model-serving claim.

## Saved experiments

- .179 mixed/serving: `/home/wlc004s/lunaflux-ako-mixed-decode-20261005.1zxOQJJf`.
- .178 mixed: `/home/wlc003s/lunaflux-ako-mixed-decode-20261005.20eTSIHB`.
- .178 long decode: `/home/wlc003s/lunaflux-ako-decode-long-20261005.2PkIEEW8`.
- Downloads: `/tmp/lunaflux-ako-next-20261005.AA1MVFnr`.

Downloaded archive hashes and all extracted manifest entries verify:

| Archive | SHA-256 |
| --- | --- |
| mixed179.tar.gz | `912f2ef8b5ccea3d64b30898c7060e2fb7182fa0d33d8567e49cad31e9e8a796` |
| mixed178.tar.gz | `1524afaeb0944aaa521301d257c2f9eae5645ebc93183463478dc89aa0f4d89b` |
| long178.tar.gz | `c2bbe3cf1b9dd205bc2325e9c7dfc0d78e2e3196b37f5f2e2f3a2689da4b822c` |
