# Query ownership and reduction width are independent

The compiler search space now includes Q64/K32 with four subgroups and one
split-readiness stage (324). This uses the existing portable tile, ownership,
read-view and transfer-readiness semantics; it introduces no separate CUDA
implementation and no model-specific selector. It retains the explicit BF16
alternative-softmax-reduction requirement and async-copy capability filter.

## Experiments and measurements

All timings below are standalone attention on the RTX 5060 Ti, not end-to-end
serving. Q64/K64 and Q64/K32 production-generated sources each used six
processes, five alternating reference/new trials per workload. Q128 and K128
were isolated source experiments, not compiler-selected production paths.

| Query tokens / rows / history | Q64/K64 exact, µs | Q64/K32, µs | Reduction |
| --- | ---: | ---: | ---: |
| 512 / 8 / 0 | 33.01 | 27.91 | 15.4% |
| 520 / 8 / 0 | 47.06 | 36.33 | 22.8% |
| 1528 / 1 / 0 | 325.59 | 320.59 | 1.5% |
| 1528 / 8 / 0 | 90.62 | 77.36 | 14.6% |
| 2048 / 8 / 2048 | 1019.17 | 972.13 | 4.6% |
| 1528 / 8 / 4096 | 1429.76 | 1360.60 | 4.8% |

The two improved schedules were separate campaigns, not interleaved together;
each campaign alternated against the same original c318 artifact. The full
16-case vector passed the probe's numerical referee and unchanged-KV checks.
Changing reduction width is **not bitwise equivalent**: maximum absolute BF16
difference versus c318 was 0.000976562 with no history and 0.000488281 with
history, below the existing probe's 0.003 threshold. This is not a replacement
for full-model quality testing. Neither approximate exp nor global fast math
is enabled.

Larger tiles did not win: Q64/K128 took roughly 527/1480/2042 µs on
1528/1/0, 2048/8/2048, 1528/8/4096. Q128/K64 was also slower at approximately
395/1172/1699 µs. Q128/K32 did not outperform Q64/K32. These rejected variants
remain experiments only.

## Hardware explanation

For 1528/8/4096, c324 reports 130 registers, a three-block/SM resource ceiling,
zero local-memory sectors, and 79.16% tensor activity. It executes 174,754,576
warp instructions. The preceding exact c322 campaign had 147,429,824
instructions and 74.40% tensor activity: narrower reduction increases fold
count but improves residency and latency hiding. Counting fewer instructions
alone is therefore not the objective.

Previously measured vLLM/SGLang paged timings were 1349.84/1271.65 µs for that
case; the new LF measurement is still approximately 0.8%/7.0% slower. For
1528/1/0, LF remains approximately 8.4%/13.1% slower than 295.62/283.52 µs.
These baseline values were **not rerun** in this campaign and retain the
launch/layout caveats in ATTENTION_THREEWAY_METRICS_2026-09-14.md. No overall
framework-parity claim follows from these isolated comparisons.

Remote results:
`/tmp/lunaflux-instruction-compiler324-timing-20260914-r1` and
`/tmp/lunaflux-instruction-compiler324-counters-20260914-r1`.
Runtime tuning, selected-kernel tracing and a fresh full-model comparison are
separate outstanding steps, not established by adding a search-space entry.

Validation: full native suite 3739/3739, strategy 18/18, source 32/32, IR 3/3,
schedule 18/18 and CUDA lowering 6/6 passed. All nine representative
memcheck/racecheck/synccheck runs passed. The historical-interval change also
passed six additional 63/64/65-position boundary cases.

## Full-runtime integration attempt

Current-worktree source snapshot SHA-256:
`959032e9312508b7e2afc4bb78460ccfa866853ee05ee755488e621853cb7ef2`.
This includes existing worktree changes; it is not a clean commit snapshot.
Linux release builds of the worker, runtime, bridge, supervisor, candidate
exporter, release binder and fused-bundle exporter passed in
`/run/lunaflux-toolchain-4896771-20260913/attention-current.vX2pXE`.

Fresh measurements of the actual exported 318/322/324 artifacts over
`query={129,1528} × rows={1,8} × history={0,4096}` produced aggregate median
latencies of 6,106,456 / 4,329,342 / 4,347,735 ns. The resulting tuning table
selected c322, not c324. A narrower reduction is not universally faster, and
these nearly tied aggregate values must not be presented as a large win for
either schedule. Both remain available through the common compiler path.

End-to-end measurements are blocked, not passed: the root filesystem has zero
unprivileged available blocks; `/dev/shm` is also almost full. Moving this
attempt to `/run` allowed the Linux build, but the existing AOT producer uses
the hard-coded `/tmp/lunaflux-bf16-producer-input.XXXXXX` scratch path and failed
twice before kernel compilation. `runtime-tuned/kernel-build*.stderr` preserves
those failures. The earlier incomplete extraction was moved, without deletion,
to this campaign's `interrupted-disk-full` directory. Production was untouched.

Downloaded GPU-result archive SHA-256:
`3b216f0f07366bf7d7093e70d282261fca95cb922661d1044efe2c78c51050fc`.
Local copy: `/tmp/lunaflux-instruction-fix.9n6ITe/results324.tar.gz`.
