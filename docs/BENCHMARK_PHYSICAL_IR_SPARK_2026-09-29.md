# Physical-IR migration: bounded two-Spark regression

## Scope and conclusion

This campaign tests the physical-IR migration on two DGX Spark GB10 systems.
It measures generated projection/MLP and query-owned attention kernels, not
full-model serving throughput. The comparison is committed `aad4e7e578b4e2d7f984ce3c9f448f4e90543121`
against the working tree. The working tree contains earlier uncommitted work;
changes against HEAD must not all be attributed to this IR extraction.

Projection and attention timing is effectively preserved in this sample. This
is expected: QKV and query-owned attention source remains byte-identical; MLP
source differences are formatting-only. A new IR layer makes scheduling and
storage explicit, but does not itself select a faster schedule. Individual
median fluctuations with overlapping ranges are not demonstrated speedups.

The integrated architecture is documented in
[Physical tile IR](PHYSICAL_TILE_IR_2026-09-29.md). It replaces opaque fold bodies
and independently reconstructed producer/consumer storage with immutable typed
plans, and adds explicit online-softmax state transitions. It does not yet
migrate every attention ownership family or replace offline target selection.

## Hardware and memory controls

| Item | Projection/MLP | Attention |
| --- | --- | --- |
| Host | `spark-57f5` / `192.168.2.178` | `spark-368c` / `192.168.2.179` |
| GPU | GB10, sm121 | GB10, sm121 |
| CUDA compiler | 13.0.88 | 13.0.88 |
| Driver | 580.178.04 | 580.178.04 |
| Unified system RAM | approximately 121 GiB | approximately 121 GiB |
| Representative largest process RSS | 1,624,224 KiB (head) | 254,592 KiB (long attention) |
| SwapFree before/after | 16,443,268 / 16,443,268 KiB | 16,367,272 / 16,367,272 KiB |

One GPU workload runs per host. Before each test, require no compute process and
at least 32 GiB `MemAvailable`. Jobs use an 8 GiB user-systemd memory limit,
zero cgroup swap, bounded task count and timeout. GPU-driver allocations need
not be fully charged to RSS/cgroups: fixed buffer bounds and system memory
readings are also checked. The largest head harness allocates about 0.94 GB
of device arrays; the long-attention host/device array envelope is below
384 MiB. No model server or full-model profiler replay is launched.

Projection `MemAvailable` was 123,448,428 → 123,449,900 KiB; long attention was
123,435,188 → 123,120,324 KiB. No swap growth or OOM was observed. This does not
establish a full-model serving memory limit.

## Projection and MLP timings

Token vector: **1, 8, 32, 128, 129, 1024, 2048**. Five counterbalanced CUDA-event
trials per point. Values below are medians in microseconds; brackets are the
minimum and maximum of the five trials. The head selects up to 32 output rows,
so its 2048-token point does not mean computing 2048 vocabulary rows.

| Kernel, 2048 tokens | HEAD median [min, max] µs | Current median [min, max] µs |
| --- | ---: | ---: |
| QKV | 410.69 [394.76, 434.21] | 398.13 [393.94, 436.56] |
| Output | 208.75 [208.30, 226.74] | 209.85 [208.63, 210.21] |
| Selected-row head | 2161.65 [2137.39, 2166.14] | 2139.52 [2136.33, 2163.38] |
| Gate/up | 509.03 [506.93, 558.45] | 534.67 [506.17, 559.85] |
| Down | 317.05 [315.54, 335.44] | 318.20 [315.83, 352.26] |
| MLP chain | 864.44 [857.98, 905.28] | 865.08 [858.82, 904.37] |

MLP chain has its own launch/timing sequence, and need not equal the sum of
separately timed gate/up and down. All seven token points are retained in the
raw logs and reproduced by `benchmarks/physical_ir_spark/summarize.mbtx`.

The paired campaign passed 28 standard projection/MLP shape points and eight
additional QKV/MLP intermediate-down points with stage-3/stage-4 pipelines and
32-wide transfers. Configurations exceeding the existing 48 KiB static shared
memory ceiling were rejected before physical execution; rejection is not a
hardware pass. Fused ingress additionally passed head dimensions 16/32/64/128
at tails 1/2/7/8/15/16/17/31/32 for both revisions.

QKV, output and MLP passed bounded memcheck. Fused ingress passed memcheck,
racecheck and synccheck. Vocabulary-head memcheck is not covered here.

### Changed fused-ingress path

Unlike the unchanged projection kernels above, fused ingress changes from the
HEAD serial WMMA path to a shared asynchronous-copy pipeline. A separate paired
harness checks exact output/KV equality and scalar reference results for all
36 head-dimension/token combinations. All pass. Three counterbalanced timing
trials per point show a substantial improvement in the head-128 fixture:

| Head dimension | Tokens | HEAD µs | Current µs | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 26.631 | 14.346 | 1.86× |
| 128 | 8 | 38.912 | 14.336 | 2.71× |
| 128 | 16 | 27.670 | 16.424 | 1.68× |
| 128 | 31 | 49.124 | 16.385 | 3.00× |
| 128 | 32 | 24.582 | 16.384 | 1.50× |
| 16 | 32 | 6.164 | 6.146 | 1.00× |
| 32 | 32 | 8.200 | 7.051 | 1.16× |
| 64 | 32 | 10.284 | 10.241 | 1.00× |

These are tiny synthetic full-ingress fixtures with input width 128, not Qwen serving speedups or
long-prefill throughput. The changed pipeline includes work predating this
physical-IR extraction, so the improvement cannot be credited to IR layering
alone. Full trial ranges are retained by `ingress_summary.mbtx`. Largest
observed process RSS was 103,664 KiB; available memory remained above 117 GiB
and swap-free was unchanged.

## Long query-owned attention

Candidate 322, three counterbalanced old/new process pairs; each process uses
nine timing samples of 40 launches. These are independent kernel measurements,
not complete Qwen layers. Numerical checks include short exhaustive and long
sampled references, plus ragged/tail geometry. All 26 long-suite cases passed.

| Queries | Context | HEAD µs | Current µs |
| ---: | ---: | ---: | ---: |
| 512 | 2048 | 177.309 | 177.824 |
| 512 | 4096 | 355.398 | 353.548 |
| 512 | 8192 | 702.114 | 704.578 |
| 1024 | 2048 | 249.838 | 248.611 |
| 1024 | 4096 | 514.584 | 514.451 |
| 1024 | 8192 | 1037.90 | 1039.52 |
| 2048 | 2048 | 342.373 | 342.024 |
| 2048 | 4096 | 848.942 | 845.090 |
| 2048 | 8192 | 1870.86 | 1875.93 |
| 513 / 127 / 256 | 4096 / 2048 / 8192 | 706.894 | 718.546 |

The last row is one ragged batch. The normal suite additionally passed
memcheck, racecheck and synccheck. Its 128-query/4096-context point was
149.489 → 149.809 µs. Source SHA-256 is identical between revisions:

- Normal: `ca35676c9feacfb3d480a7666ebf8dcba286ba630734d66b3e4b0e0b6dcf57fc`.
- Long: `d9db2e9be5202e3ccd47bbd7ae8d88c8cb7e28f7d2b05f3a831a9fa43269e8e0`.

## Reproduction and artifacts

### Local validation

Toolchain: moon `0.1.20260920` (`914d7da`), moonc
`v0.10.14+7d59c7ec9-dev`. The final full native run passed **4128/4128**:

```sh
moon test --target native --warn-list '-79-25' --diagnostic-limit 5
moon check compiler/physical_tile_ir compiler/attention_physical_ir --target native --deny-warn
```

The new IR packages pass warning-denied checking. The full-suite command
explicitly suppresses existing migration warnings 79/25 and does not deny other
warnings; it is not a warning-clean whole-repository claim. Scoped formatting,
generated interfaces and `git diff --check` also passed. An earlier test run
overlapped an in-progress edit with a reserved `loop` identifier and failed;
that edit was corrected before the final successful run.

### Retained files

Automation is MoonBit `.mbtx` under `benchmarks/physical_ir_spark`. It uploads
generated CUDA and bounded harnesses only, uses non-overwriting output paths,
and retains commands, compiler output, exit codes, hashes and memory readings.

Local root: `/tmp/lunaflux-physical-ir-spark.21eDJ0`:

- `projection-campaign`: final paired projection/MLP results and downloaded
  source/CUBIN artifacts with local/remote hashes.
- `ingress-timing-campaign`: all 36 paired ingress fixture timings and
  downloaded, hash-verified source/CUBIN/executable artifacts.
- `attention-logs`: normal attention and sanitizers.
- `attention-long-logs-r2`: final long attention results.
- `attention-long-binaries`: downloaded old/new sources and executables.

Normal attention executables were downloaded separately to
`/tmp/lunaflux-attention-normal-artifacts.kdQekv`. Downloaded normal/long
attention executable hashes match their corresponding remote files.

Remote long attention: `/home/wlc004s/lunaflux-physical-ir-long.D7WmsH`.
Remote projection path is recorded in `projection-campaign/remote-path.txt`.
The earlier long-attention attempt stopped at a missing local export before
running its GPU comparison; its logs are preserved and excluded from results.

The projection fixture declares an sm120/CUDA-13.1 specialization policy; here
its generated source is compiled for sm121/CUDA-13.0. This is a fixed-schedule
portability/regression diagnostic, not production AOT target admission. No
vLLM/SGLang comparison, full-model speedup, multi-node inference, deployment,
or fresh Nsight counter attribution is claimed by this report.
