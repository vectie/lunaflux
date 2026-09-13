# MoonBit toolchain retest — 2026-09-13

The clean committed source `4896771` was rebuilt on Linux with
`moon 0.1.20260904` and `moonc v0.10.12+1634b282e`. Native release build,
warning-denied native check, and 28 focused migration tests passed. All serving
executables and AOT artifacts were rebuilt. Nine fused CUBINs matched the prior
runtime byte-for-byte. CUDA remained 13.1.115.

Qwen3-0.6B BF16 ran on RTX 5060 Ti, UUID
`GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`. Each cell used one warmup and five
measured trials, fixed input token IDs, greedy output, and the same measurement
client. New runtime ran first, then the preserved old `f9df976` runtime, without
overlapping GPU workloads. This is a sequential same-day comparison, not a
randomized/interleaved statistical experiment. No profiler was active.

## Output throughput (tokens/s, arithmetic mean of five trials)

| Input / output | C | Old runtime | New toolchain |
|---|---:|---:|---:|
| 59 / 256 | 1 | 246.11 | 246.87 |
| 59 / 256 | 2 | 464.53 | 462.43 |
| 59 / 256 | 4 | 872.53 | 871.35 |
| 59 / 256 | 8 | 1642.37 | 1646.58 |
| 59 / 256 | 16 | 2706.86 | 2702.58 |
| 128 / 128 | 1 | 239.34 | 239.70 |
| 128 / 128 | 2 | 448.02 | 446.93 |
| 128 / 128 | 4 | 834.15 | 832.53 |
| 128 / 128 | 8 | 1561.94 | 1563.36 |
| 128 / 128 | 16 | 2509.81 | 2503.13 |
| 512 / 64 | 1 | 210.40 | 211.08 |
| 512 / 64 | 2 | 367.82 | 369.95 |
| 512 / 64 | 4 | 616.87 | 619.26 |
| 512 / 64 | 8 | 1004.74 | 1014.27 |
| 512 / 64 | 16 | 1336.82 | 1339.62 |
| 1528 / 32 | 1 | 144.81 | 145.21 |
| 1528 / 32 | 2 | 210.54 | 209.58 |
| 1528 / 32 | 4 | 271.30 | 269.94 |
| 1528 / 32 | 8 | 339.52 | 339.61 |
| 1528 / 32 | 16 | 371.18 | 371.12 |

Observed throughput changes range approximately from -0.50% to +0.95%.
The upgrade does not show a material performance improvement. Long C8 mean
TTFT is 305.025 ms old versus 305.700 ms new. Long C16 is 561.125 versus
561.900 ms. Previous vLLM/SGLang results were not rerun here and are not
same-day controls for this experiment.

## Output caveat

All requests completed with the requested token counts. Across the 620 measured
request pairs (same cell/trial/ordinal), 617 token sequences matched exactly.
Two mismatches occurred at 59/256 C2 and one at 128/128 C2. At 59/256 C2 both
old and new runs independently contained two distinct output sequences; at
128/128 C2 only the new run did. All other cells matched. Thus this is not a
blanket deterministic-output qualification. Batch-arrival/schedule sensitivity
is a hypothesis to investigate, not an established cause or an excuse to waive
the mismatches. A controlled batch-membership replay is the appropriate next
diagnostic before attributing the differences to the compiler.

## Reproduction and environment

Remote working directory: `/run/lunaflux-toolchain-4896771-20260913`.
`current/` and `control/` retain raw SSE, per-request IDs/timestamps, trial JSON,
launch identities, GPU snapshots, exit status, and shutdown confirmation.
Both servers stopped and the GPU was idle afterwards.

The host disk and `/dev/shm` were almost full, so an isolated directory on
`/run` was used. Its bind mount alone was remounted executable; the rest of
`/run` remains noexec. The initial noexec export attempt is preserved under
`runtime-noexec-attempt`; the subsequent export/build/setup succeeded.

Local archive: `/private/tmp/lunaflux-toolchain-retest-results-20260913/results.tar.gz`.
Local and remote SHA-256 both equal
`101f94d9ef79b14400ba7ee233f26e867f04bfc21c7a3a72e4ebbaa9cb26c8fe`.
It excludes reproducible script build caches. Raw results are preserved; the
initial summary script in the archive rejects cross-request output differences.
The revised analysis explicitly counts those differences rather than treating
them as a pass.
