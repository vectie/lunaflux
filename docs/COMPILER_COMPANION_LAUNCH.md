# Independent compound-kernel launch geometry

MLP-up and MLP-down are independent ordered maps separated by an explicit
workspace dependency. Their workgroup geometries need not be equal. The
compiler retains a companion launch alongside the primary launch, including
row-bounded variants. The CUDA backend chooses its own subgroup width; model
plans and the scheduler remain unchanged.

Artifacts optionally bind `companion_dimensions` to the existing `_down`
entry-point convention. Absence means the historical primary geometry exactly.
The full-graph serializer, manifest reader, artifact bundle, bootstrap identity,
and prepared eager/bucketed executor retain the declaration. New metadata
is rejected by older strict readers rather than silently misexecuted.

For the BF16 matrix MLP, down uses up to four subgroups with its own grid and
shared scratch; its single-token GEMV follows the same declared block. No dot
product is reassociated and the existing intermediate BF16 rounding is retained.
The generic schedule records an independent output-map tile count in its
canonical identity; CUDA alone maps that count to 32-lane subgroups and scratch
bytes. This is an initial consumer-distribution policy, not a complete
per-operation autotuner. Scalar/single-kernel MLPs have no companion metadata.

## Measured on 2026-09-08

RTX 5060 Ti, CUDA 13.1.115, sm120. BF16 MLP widths 1024/3072/1024;
three paired trials, each timing a graph containing 20 up+down executions.
These are **combined MLP kernel times**, not whole-model latency. Median
microseconds across the three trials:

| Token rows | Previous shared launch | Independent down launch | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 17.483 | 17.147 | 1.020× |
| 2 | 56.310 | 56.309 | 1.000× |
| 7 | 56.454 | 56.530 | 0.999× |
| 8 | 56.443 | 56.675 | 0.996× |
| 17 | 57.150 | 57.234 | 0.999× |
| 128 | 187.467 | 181.234 | 1.034× |
| 256 | 371.645 | 359.230 | 1.035× |
| 512 | 686.779 | 664.211 | 1.034× |
| 1024 | 1357.178 | 1307.722 | 1.038× |

All nine shapes produced bitwise-identical outputs and intermediate workspace,
with untouched output tails, in every timing trial. Memcheck, racecheck,
initcheck, and synccheck passed for 1/7/17/128 tokens. Both modules and all probe
resources were explicitly released. The new down kernel uses 104 registers,
zero spills, 128 threads and 4096 shared bytes, versus the old launch's 512
threads and 16384 shared bytes. Short-input differences are too small to claim
a useful speedup. The long-input improvement does not resolve the remaining
prefill-attention gap, and no new vLLM/SGLang or full-Qwen comparison was run.

Raw results: `/private/tmp/lunaflux-companion-results-20260908-r1`; remote
original: `/dev/shm/lunaflux-companion-20260908-r1`. Both include source, probe,
MoonBit runner, timing logs and sanitizer outputs. Source/cubin SHA-256:

- Old source: `009bb7b1fb7525f0e9a8c4fc8f083d0288c77c2dca8374d7644ae74b4f8b29a0`
- New source: `9e4edfaf7438fbfb2d6d8592c64fbd2bac13efbe096b79db1c41d5802a7283ee`
- Old cubin: `cf36ba8a042f9e85119ef15c3b6f2fd2deeddf42c9f3ebed85b01ca5dff29a54`
- New cubin: `64ffe0260c62d82f3c256b397d1240fecdbd9dccfb7d1309bff9ab7915b67c3e`

Software regressions cover optional/complete metadata, strict malformed-field
rejection, primary/row-variant serialization, scalar exclusion, eager preparation,
legacy bucket behavior and explicit down-bucket geometry. Existing absent-field
bootstrap digests remain unchanged.

Native warning-denied check and scoped formatting/interface checks passed.
The full working-tree suite passed 3611/3614 tests inside the sandbox; the
three failures were denied local TCP listener creation. Rerunning their two
packages outside the sandbox passed 140/140, including all three failures.
