# Decode K/V pipeline: compact buffers preserve the useful overlap

Implementation: `c04b9de`. This follows
[vector transfer measurements](DECODE_VECTOR_TRANSFER_2026-09-06.md).

The compact two-stage implementation improves the isolated decode attention
operator over the preceding vectorized compiler kernel. It is not yet selected
by the production AOT adapters; no new Qwen serving tokens/s result or new
vLLM/SGLang comparison is claimed here. Production was not changed.

## Compiler changes

- Portable strategy exposes two-stage async K/V schedules with 64-token and
  32-token staging tiles (440 and 441), subject to device resource capabilities.
- Existing immutable schedule fields carry asynchronous memory, stage count,
  vector width and resource use through CUDA lowering. There is no runtime JIT,
  mutable tuning database, new per-token allocation or model-name branch.
- Logical softmax partition grain is now distinct from staging tile size.
  Grouped split decode retains its 64-token partition decomposition while
  consuming it through one 64-token or two ordered 32-token transfers.
  This field belongs to the generic schedule, not CUDA source generation.
  Canonical schedule/lowering versions are now v3/v4.
- CUDA realizes the schedule with alternating shared buffers, async issue and
  commit, completion wait, consumer synchronization and terminal storage reuse.
  `cp.async`, shared-address conversion and subgroup instructions remain in
  the CUDA backend. SASS contains `LDGSTS`, not a synchronous emulation.

This is a functional compiler scheduling optimization: preserve the pure
map/fold result while changing the realization of storage and effects. It is
not a claim that every functional program can be pipelined automatically; this
realization currently covers grouped split-subgroup decode attention.

## Experiment that did not work

Naively doubling the 64-token buffer increased dynamic shared memory from
34,076 to 66,844 bytes. On this 100-KiB-shared-memory-per-SM device that permits
only one resident block instead of two (a resource bound, not an occupancy
counter measurement). B1 direct attention became faster, but B8 direct and
many split8 shapes regressed substantially. More stages alone are not a win.

The first 32-token experiment recovered the storage footprint, but also moved
split boundaries. The bitwise check failed at context 59 in split8. This run
was retained as failed; it was not benchmarked or relabeled successful. The
generic partition-grain separation fixed the numerical grouping rather than
loosening the comparison. The corrected run passes bitwise comparisons.

| Resource | Vector K64 | Async K64 x2 | Async K32 x2 |
| --- | ---: | ---: | ---: |
| Dynamic shared bytes | 34,076 | 66,844 | 33,820 |
| Direct registers/thread | 48 | 59 | 43 |
| Partial registers/thread | 52 | 61 | 44 |
| Stack/local bytes | 0 | 0 | 0 |

The modules additionally report 1,024 static shared bytes per entry point.
Actual occupancy and stall-counter attribution were not recollected this run.

## Measured operator latency

RTX 5060 Ti, BF16, Q16/KV8/head128/page8; deterministic scattered-page fixture.
Three independent trials per shape/version, 200 launches captured in a CUDA
graph, three warm replays and one timed replay, reversed version order in the
middle trial. The GPU had no competing compute process. Times are means in
microseconds, and speedup is old latency divided by new latency.

The table uses the corrected K32 campaign's interleaved vector control. B1
split8 includes its partial and merge kernels; B8 direct is a single kernel.
These routes match the previous serving experiment, not a newly changed
dispatch policy. Full direct/split8/partial/merge matrices are in `summary.csv`.

| Batch | Context tokens | Route | Vector | Async K32 x2 | Speedup |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 59 | split8 | 9.880 | 9.320 | 1.060x |
| 1 | 128 | split8 | 11.088 | 10.569 | 1.049x |
| 1 | 512 | split8 | 11.512 | 11.068 | 1.040x |
| 1 | 1528 | split8 | 27.312 | 26.150 | 1.044x |
| 1 | 4096 | split8 | 66.963 | 64.703 | 1.035x |
| 8 | 59 | direct | 9.788 | 9.231 | 1.060x |
| 8 | 128 | direct | 18.447 | 17.599 | 1.048x |
| 8 | 512 | direct | 66.729 | 63.201 | 1.056x |
| 8 | 1528 | direct | 239.479 | 201.214 | 1.190x |
| 8 | 4096 | direct | 624.959 | 531.473 | 1.176x |

For context, the four implementation variants on B8 direct are below. Scalar,
vector and K64 pipeline were interleaved in campaign A; vector and K32 pipeline
were interleaved in campaign B. The two vector columns expose cross-campaign
variation rather than pretending all four were one interleaved experiment.

| Context | Scalar A | Vector A | Async K64 A | Vector B | Async K32 B |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1528 | 429.634 | 238.430 | 311.554 | 239.479 | 201.214 |
| 4096 | 1143.909 | 621.041 | 823.018 | 624.959 | 531.473 |

## Correctness and scope

- 69 affected-package tests passed: strategy 13, schedule 8, compiler 7,
  CUDA lowering 5, source 17, AOT 4, paged-attention AOT 15.
- `moon info`, `moon fmt`, warning-denied native check and `git diff --check`
  passed. This is targeted validation, not a claim of a fresh whole-repo suite.
- Both successful variants match the preceding vector module bit-for-bit for
  direct and split8 output at contexts 1, 7, 8, 9, 59, 63, 64, 65, 128, 256,
  512, 1528 and 4096, plus a mixed prefill/decode row case and B8/context1528.
- The CPU double oracle, unchanged persistent KV, untouched prefill rows and
  explicit resource teardown pass. These deterministic fixtures do not prove
  bitwise equivalence for every possible input.
- memcheck, racecheck, synccheck and initcheck pass at B8/context128, covering
  buffer reuse; racecheck reports zero hazards. No sanitizer timings enter the
  performance table. No new soak or production campaign was run.

## Reproduction and remaining integration

Generated CUDA comes from `tests/attention_tile_cuda_source_probe` modes
`decode-pipeline` / `decode-pipeline32`, through the full compiler. Their
test-only selection records are explicitly not measured autotune data.

Remote campaigns:

- `/dev/shm/lunaflux-pipeline-decode-20260906-r1`: successful K64 experiment.
- `/dev/shm/lunaflux-pipeline32-decode-20260906-r1`: failed repartitioning test.
- `/dev/shm/lunaflux-pipeline32-decode-20260906-r2`: corrected K32 experiment.

Archive, including raw results, scripts, generated sources and machine code:
`/private/tmp/lunaflux-decode-pipeline-20260906-r1.tar.gz` locally and the same
basename under `/dev/shm` remotely. SHA-256:
`75d945adb33c119f428c6cfe6bcad544857b6f0e1be6887ee84c2548b55295be`.

Corrected K32 CUBIN:
`45876fc1df55e76dbcc36d92a8ac5a28a948b2541252be92a99985f8480f7592`.
Generated source:
`7574832be46f6b1df307b652c2e819c2ff0cc219920fcf7046df039716bae692`.
GPU UUID: `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`, PCI `00000000:17:00.0`.
CUDA 13.1.115 nvcc hash:
`6ce80365abc2f14ff5f9e069fe687a93d4eaacc555dbf1d1696ee899d39df52a`.

Next: wire measured, device/shape-scoped selection into the serving AOT
adapters and rerun Qwen end-to-end with unchanged token vectors. Production
adapters still explicitly request synchronous, one-stage capabilities; this
commit must not be reported as an already deployed serving improvement.
