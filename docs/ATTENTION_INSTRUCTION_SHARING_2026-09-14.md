# Attention instruction sharing: first implemented correction

This is a measured reduction, **not closure of the instruction-count gap**.
The subsequent [exact numeric if-conversion](ATTENTION_IF_CONVERSION_2026-09-14.md)
reduces the gap further; the tables below preserve this first correction's results.
The generated split-readiness attention kernel previously recomputed the same
K/V read-view address map at both transfer points. Its CUDA lowering now
materializes the value-address product at the K transfer, consumes it at the V
transfer, and scalarizes the finite product with static unrolling. The row's
first position is loop invariant. No model-family branch, approximate math,
floating-point reassociation, or production selection override was introduced.

The retained product is valid only for the existing single-slot split-readiness
schedule: each K producer is followed by its V consumer before the next K
producer. Other schedules retain their existing lowering. Tail elements retain
a null sentinel and issue zero-filled transfers. Register usage rises from 173
to 239, but still permits two blocks/SM; both ptxas and measured local-memory
sectors report zero spills.

## Physical measurements

RTX 5060 Ti, sm120, CUDA 13.1, same metadata ABI and inputs as
[the three-way comparison](ATTENTION_THREEWAY_METRICS_2026-09-14.md).
Query count is total across rows, not per row. History is the fixture's base
history argument. Each timing is the median of 30 observations from six fresh
processes, alternating old/new measurements. All 480 comparisons were bitwise
equal, with maximum absolute difference zero; KV mutation checks passed.

| Total queries | Rows | History | Old synchronous c318 µs | New shared-map c322 µs |
|---:|---:|---:|---:|---:|
| 63 | 1 | 0 | 19.35 | 18.48 |
| 64 | 1 | 0 | 18.87 | 18.48 |
| 65 | 1 | 0 | 22.62 | 21.48 |
| 512 | 8 | 0 | 38.08 | 35.37 |
| 520 | 8 | 0 | 55.45 | 51.09 |
| 1528 | 1 | 0 | 459.83 | 393.25 |
| 1528 | 8 | 0 | 112.41 | 102.00 |
| 2048 | 1 | 0 | 762.95 | 651.92 |
| 2048 | 8 | 0 | 163.66 | 145.56 |
| 2048 | 16 | 0 | 119.10 | 106.85 |
| 2048 | 1 | 2048 | 2178.49 | 1740.45 |
| 2048 | 8 | 2048 | 1648.48 | 1253.58 |
| 1528 | 8 | 4096 | 2586.38 | 1790.15 |
| 2048 | 1 | 4096 | 3637.90 | 2849.12 |
| 2048 | 8 | 4096 | 3157.95 | 2379.24 |
| 2048 | 16 | 2048 | 1608.35 | 1223.13 |

For 1528/8/4096, the previously measured async c322 was 2366.24 µs;
the new c322 is 24.3% lower latency. Versus the contemporaneously rerun
synchronous reference, it is 30.8% lower latency.

| 1528/8/4096 | Old async c322 | Shared-map c322 | vLLM previous run | SGLang paged previous run |
|---|---:|---:|---:|---:|
| Warp instructions | 222,423,936 | 166,814,816 | 50,401,280 | 64,378,912 |
| Tensor active % | 44.79 | 58.87 | 77.28 | 85.24 |
| Timing µs, unprofiled | 2366.24 | 1790.15 | 1349.84 | 1271.65 |

The instruction reduction is 25.0%. The remaining instruction ratio is
3.31× vLLM / 2.59× SGLang. Timing remains 1.33× / 1.41× respectively.
These are standalone warm-cache attention comparisons, **not new end-to-end
serving measurements**. Baseline timings are reused, not rerun in this change.
Their cache-control and API/layout differences remain documented in the
three-way report; don't present these as perfectly identical execution routes.

## Remaining instruction work, before other tuning

The imported SASS counters show essentially equal MMA work: new LunaFlux
13.099M HMMA instructions, vLLM 13.189M, SGLang 13.445M. The excess is not
an extra attention matrix multiplication. Representative opcode counts:

| Opcode | LunaFlux new | vLLM | SGLang |
|---|---:|---:|---:|
| MOV | 26,182,592 | 522,752 | 2,635,584 |
| BRA (excluding BRA.U) | 8,277,376 | 212,224 | 211,392 |
| BSSY | 5,369,664 | 107,648 | 6,144 |
| IADD.64 | 4,759,296 | 824,320 | 438,912 |
| MUFU.EX2 | 3,429,504 | 3,397,248 | 3,666,816 |

Follow-up must isolate remaining move/control sequences and numeric lowering,
then repeated address arithmetic. Opcode totals alone do not attribute every
move or branch to a source expression. Baselines use packed BF16 conversion
and FTZ arithmetic; LunaFlux currently preserves its existing precise numeric
contract. Do not silently replace that contract merely to match their counts.

Two isolated experiments were deliberately not integrated:

- Retaining Q fragments in addition to addresses was not consistently better.
- Replacing expf with __expf improved time, but changed output (maximum absolute
  BF16 difference 0.000488281). It needs an explicit numeric policy and broader
  model validation, not an unannounced fast-math substitution.

## Reproduction and validation

Generate through the real compiler, not a patched CUDA fixture:
`moon run tests/attention_tile_cuda_source_probe --target native -- 322 instruction-map`.

Generated source SHA-256:
`be95c8cc6a475c703791418c5c2412fc3c5d0bab5f53fe403ffafe697a23cf5a`.
Cubin SHA-256:
`921af5bc2ac63c1b9d454819e1d853b6017f67b9433535215a4f31f33049e3f5`.
Flags: `-O3 -arch=sm_120 --fmad=false -lineinfo -Xptxas=-v -cubin`.

Remote results are under `/tmp/lunaflux-instruction-compiler-{timing,counters,sanitizer}-20260914-r1`.
Memcheck, racecheck, and synccheck passed for 1528/1/0, 2048/8/2048,
and 1528/8/4096. Affected-package tests: 32/32. Warning-denied native
check and scoped formatting passed; moon info completed.
Full native suite: 3738/3739; the unrelated zero-wait TCP poll test returned
None instead of Some(8). Its package rerun passed 54/54. This is not recorded
as a clean full-suite pass. Production deployment is unchanged.
