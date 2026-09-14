# Actual-row attention replay

## Result

The earlier roughly 14% c324 microbenchmark advantage is not representative of
the serving row distribution. Replaying 34 distinct prefill/mixed row vectors
(100 observed steps, weighted by their recorded frequencies) gives **1.112%
less isolated attention time**, close to the earlier serving trace's **1.03%**.
The serving trace's total kernel-time reduction remains **0.20%**; this replay
is not a new end-to-end throughput result.

The replay preserves each row's query count and past length, prefill/decode
classification, page capacity, metadata ABI, and 128-thread launch. It uses
synthetic tensor values and warm repeated launches, not captured model
activations or a natural-language quality corpus. Both kernels pass the
sampled independent scalar oracle, retained-KV comparison, and the variant's
immediate repeat comparison. This does not close model-token divergences.

Runtime metadata emits tiles only for prefill rows. Decode rows in the same
batch are handled by the separate decode kernel. The corrected replay follows
that rule. Earlier replay attempts are retained but excluded: r1 lacked a
header, r2 exceeded the old 8192-page fixture capacity, and r3 incorrectly
included decode rows in prefill work. The final r4 supports the production
9216-page envelope and preserves the phase partition.

## Uninstrumented paired medians

Five alternating-order timing trials per shape; microseconds per launch.

| Actual work | c322 | c324 | Interpretation |
| --- | ---: | ---: | --- |
| One row: 2048 queries, no history | 535.65 | 532.49 | 0.59% reduction |
| One row: 2048 queries, 2048 history | 1446.59 | 1411.95 | 2.39% reduction |
| C8: 28 prefill queries at history 4068, seven decode rows | 249.75 | 259.36 | 3.85% regression |
| C16: 98 queries at history 3998, 1936 fresh queries, fourteen decode rows | 674.25 | 673.71 | Essentially unchanged |
| C16 final tail: 112 prefill queries, fifteen decode rows | 271.28 | 323.03 | 19.08% regression |

Weighted sum over the observed 100 steps: 99638.79 versus 98530.76 us.
These sums represent one attention layer's repeated workload geometry, not
complete-model execution.

## Hardware counters

Separate Nsight Compute captures, one selected launch after three warmups.
Caches and clocks are not controlled; use the uninstrumented repeated medians
above for timing, not these single profiled launches. L2 bytes are not DRAM
bytes; the requested DRAM metric was unavailable and is not treated as zero.

| Shape | Instructions c322 / c324 | L2 bytes c322 / c324 | Long-scoreboard % c322 / c324 | Barrier % c322 / c324 |
| --- | ---: | ---: | ---: | ---: |
| 2048 fresh queries | 49,780,216 / 63,997,354 | 320,734,624 / 320,955,520 | 17.88 / 11.48 | 1.62 / 3.93 |
| 2048 queries + 2048 history | 133,096,192 / 173,904,640 | 858,781,664 / 857,973,888 | 11.60 / 7.35 | 2.13 / 4.53 |
| C8 short prefill tail | 3,875,888 / 4,905,136 | 34,712,768 / 34,486,240 | 14.58 / 13.21 | 25.21 / 23.33 |
| C16 mixed step | 55,255,816 / 71,069,906 | 370,103,328 / 370,660,096 | 15.26 / 8.64 | 7.77 / 12.88 |

The smaller KV tile reduces registers (228 to 130) and loading dependencies,
but adds roughly 29–31% instructions on the large shapes. With nearly unchanged
L2 traffic and increased synchronization, the benefit is largely offset.
Source inspection is consistent with more KV-loop iterations when changing
the tile from 64 to 32. These are KV-tile alternatives, **not** the prior
corrected async-versus-sync comparison; do not revive the obsolete general
claim that async always adds 33% instructions.

## Runtime integration correction

Physical V6 startup exposed a missing bootstrap classifier branch: bundle
admission understood V6, but the worker classified only V3–V5 as reusable
bundles. The fix also includes the hex-encoded observation payload in the
bounded startup read envelope. Regression tests cover a real exported V6
bundle and integer-limit boundaries; bootstrap tests pass 25/25.

After rebuilding both parent and child, V6 completed six workload families,
C8/C16, one warmup and three measured repetitions: 48 trials, 576 requests,
with successful terminal counts and shutdown. The table contains one genuine
baseline observation (four complete-graph samples, median 67062920 ns for
1/2048/2048), not invented winning c324 timings. Unmeasured buckets retain the
existing policy. This validates V6 startup/serving, not a measured c324 win or
bitwise model equivalence.

Remote sources/results:

- `/tmp/lunaflux-replay-attention-rows-20260914-r4`
- `/tmp/lunaflux-replay-attention-rows-20260914-r4/counters-privileged`
- `/tmp/lfroutes-v6-20260914-r3/wide`

Remaining numerical work: capture matching request/sample identities and
logit margins for the previously observed full/partial and c322/c324 token
differences. Candidate micro-level tolerance alone does not resolve that work.
