# Matrix distribution and selected-row storage

These are isolated compiler-generated kernel measurements, not a new serving
benchmark or a comparison with vLLM/SGLang. Static serving defaults are unchanged.

## Implementation

`luna_projection_strategy` now represents independent output-map distribution
(1/2/4/8/16 tiles per workgroup) and selected-input residency as immutable
schedule choices. They preserve the ordered reduction. The projection compiler
passes these choices into source generation, local-storage accounting and graph
launch geometry. The BF16 artifact producer consumes optional typed offline
records; no request-time benchmarking or new runtime JIT is introduced.

Selected-row residency gathers an input row tile once outside the reduction
fold. This is a storage-placement decision, not an unconditional consequence of
CSE. Inactive output groups still participate in shared-memory gathering and
barriers; masked stores handle the partial vocabulary tile.

The matrix-tiled MLP now describes its actual 16x16x16 primitive separately from
its output distribution. Its maximum launch envelope also covers the decode
grid when the declared prefill envelope is small. Previously a four-token
envelope could leave intermediate decode columns uncomputed.

## Paired results

RTX 5060 Ti, BF16, input width 1024. Dense output width 1024, standalone QKV
output width 4096, MLP intermediate width 3072, vocabulary 151936. Means of
three paired trials; CUDA Graph replay, 50 iterations for distribution and 10
for resident-head tests. Both variants use identical data and output checks.

| Kernel / schedule | Rows | Previous µs | New µs | Speedup |
|---|---:|---:|---:|---:|
| Dense, 2 output tiles | 8 | 12.91 | 4.57 | 2.83x |
| Dense, 2 output tiles | 256 | 48.16 | 48.17 | 1.00x |
| Standalone QKV, 2 output tiles | 8 | 13.08 | 13.03 | 1.00x |
| Standalone QKV, 2 output tiles | 256 | 177.53 | 171.72 | 1.03x |
| MLP, 2 output tiles | 8 | 56.68 | 30.75 | 1.84x |
| MLP, 8 output tiles | 59 | 113.17 | 91.56 | 1.24x |
| MLP, 8 output tiles | 256 | 372.74 | 337.85 | 1.10x |
| Resident head, 8 output tiles | 1 | 737.25 | 736.97 | 1.00x |
| Resident head, 8 output tiles | 8 | 2487.11 | 1523.04 | 1.63x |
| Resident head, 8 output tiles | 59 | 9798.22 | 2478.74 | 3.95x |
| Resident head, 8 output tiles | 256 | 39058.92 | 11386.05 | 3.43x |

For the head, **rows means requested output rows**, not the length of a single
prompt. The harness uses one-token row offsets. Long-prompt ragged gathering
and end-to-end serving still need separate testing. Standalone QKV results do
not measure the production fused QKV/QKNorm/RoPE/KV-write kernel.

Losing alternatives matter: resident-head group 1 takes 6414.08 µs at eight
rows versus 2487.14 µs before. Group 4 wins at eight rows (1331.81 µs) but loses
at one row (858.06 versus 737.17 µs). These must not become blanket defaults.

All tested row counts `[1,2,8,15,16,17,32,59,128,256]` matched the baseline
bitwise, including initialized output/workspace sentinels. Memcheck, racecheck,
synccheck and initcheck passed for group-2 dense/QKV/MLP and group-8/group-16
resident heads at rows 8 and 17. This is deterministic baseline equivalence,
not an independent model-quality reference test.

## Reproduction and remaining work

GPU UUID `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`, PCI
`00000000:17:00.0`; CUDA 13.1.115, `--fmad=false --maxrregcount=128`.
The archive contains generated source, cubins, probes, MoonBit automation,
per-trial logs, sanitizer logs and full CSV vectors, including losing choices:

- Local: `/private/tmp/lunaflux-matrix-schedules-20260906-r1.tar.gz`
- SHA-256: `def213191da1580dbb1ec80a870641c857ed9d78ad7169480dfa154eb273a6bf`
- Remote results: `/dev/shm/lunaflux-distribution-20260906-r1`,
  `/dev/shm/lunaflux-residency-20260906-r1`,
  `/dev/shm/lunaflux-matrix-sanitize-20260906-r1`.

Remaining: physical-device-scoped record persistence and CLI consumption,
production selection of profitable variants, ragged/independent-reference
checks, and fresh exact-source serving measurements. The current adapter's
architecture-only target key is not enough to generalize these timings to all
GPUs sharing an instruction-set target.

Affected package tests and warning-denied native compilation pass. The full
suite aborts in the FP8 projection identity test; the same abort reproduces on
the untouched starting commit `5713b2c`. No full-suite pass is claimed.
