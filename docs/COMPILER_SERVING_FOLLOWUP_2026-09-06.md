# Compiler serving follow-up

## Implemented

- `5dbcecf`: distinguish semantic online-fold updates from transfer tiles in
  the generic attention cost model. Enable the CUDA backend's asynchronous
  decode capability under a 48-KiB occupancy budget; the serving exporter now
  selects schedule 441 through the compiler rather than pinning its ID.
- `cf44a8e`: predicate captured pure-suffix arithmetic from model output demand.
  Retain the prefix through the last persistent-state operation. Baseline and
  residual-fused steps share a stable zero-work descriptor view for entirely
  output-free frames. Mixed frames retain full upstream row identity. This
  adds 20 bytes to the existing counts allocation, no extra transfer call or
  graph, and does not eliminate captured node-launch overhead.
- `86a66f1`: explicit offline full/partial/unfused ingress evaluation, without
  inventing timing records before alternatives can be measured.
- `9be6973`: localize CUDA attention macros and helper symbols. Combining the
  32-position asynchronous kernel with the 64-position split kernel had exposed
  a macro-redefinition failure. The corrected combined module builds.
- `dc2368c`: align worker admission with the executor's residual-only/unfused
  ingress choice. A lone valid residual module is accepted; incomplete ingress
  pairs and invalid residual operation seeds remain rejected.

Planning stays pure and model-neutral where applicable; asynchronous-copy and
CUDA source composition details remain in the backend. None of these changes
adds token-step filesystem access or identity verification.

## Same-workload Qwen measurement

Qwen3-0.6B BF16, RTX 5060 Ti, same token-ID requests and fixed output lengths.
One warmup and two measured repetitions per cell; rates below are arithmetic
means of the two measured throughput values. No concurrent GPU workload.
Old is `9402e1b`; new full and partial are `9be6973`. This is a small paired
comparison, not a confidence-interval study or a rerun of other engines.

| Input → output tokens | C | Old full tok/s | New full tok/s | Full gain | Partial evaluation tok/s | Unfused fallback tok/s |
|---|---:|---:|---:|---:|---:|---:|
| 59 → 256 | 1 | 224.07 | 225.85 | +0.8% | 231.78 | 55.94 |
| 59 → 256 | 8 | 833.20 | 849.79 | +2.0% | 975.01 | 367.95 |
| 128 → 128 | 1 | 216.95 | 217.87 | +0.4% | 221.27 | 54.27 |
| 128 → 128 | 8 | 774.01 | 791.96 | +2.3% | 894.72 | 336.29 |
| 512 → 64 | 1 | 176.56 | 176.07 | −0.3% | 179.02 | 20.78 |
| 512 → 64 | 8 | 457.96 | 474.51 | +3.6% | 506.68 | 107.69 |
| 1528 → 32 | 1 | 86.61 | 87.19 | +0.7% | 87.55 | 5.89 |
| 1528 → 32 | 8 | 123.05 | 126.33 | +2.7% | 127.40 | 14.65 |

The two integrated optimizations were measured together, not separately ablated.
Seven full-path cells have identical token arrays. The 59→256 C8 cell still
contains three sequences; every new sequence already occurs in the old run,
but request-ordinal assignment varies. Batch-invariant generation is not proved.

Partial fusion is **not selected as the serving default**. Six cells contain
new sequences relative to the full path. Code inspection identifies different
QKNorm reduction trees and different RoPE arithmetic: partial uses block-128
reduction and per-token division/pow; full uses subgroup-32 reduction and
compiler-hoisted inverse-frequency multiplication. These are concrete semantic
differences to align or numerically qualify, not proof of the sole cause of
every token divergence. A faster timing alone is insufficient for selecting an
equivalent schedule. No measured-winner record was manufactured from these runs.

The first unfused evaluation aborted before readiness. The worker still
required an ingress/read-only pair even though the exporter and executor
supported residual-only bundles. `dc2368c` fixes that inconsistency; the physical
retest completes all eight cells. Unfused uses `dc2368c` (only worker admission
and its regression differ from `9be6973`; the attention module hashes match).
Five unfused cells also contain new token sequences relative to full fusion.
The original failed evaluation
has no throughput result. The unfused fallback also removes compiler read-only
attention, so its throughput must not be interpreted as an ingress-only ablation.

## Validation and records

Warning-denied native check passed; focused tests: 223/223. The full repository
suite is not claimed here (the previously recorded unrelated FP8 regression is
outside this change).

The exact `9be6973` combined serving attention module passed the existing
deterministic boundary probe (including mixed rows and page/partition tails),
with bitwise comparison passing, KV unchanged and resources released. Memcheck,
racecheck, synccheck and initcheck passed on its direct/split probe. All four
sanitizers also passed the captured descriptor mixed/empty-frame regression.

- Source archive SHA-256:
  `16c8ac381be1a19b4cfd508a9bfb23c8748d21968c39e5d11f50119ee99d89ab`.
- Downloaded benchmark/descriptor/alternative-diagnostic archive:
  `/private/tmp/lunaflux-remaining-9be6973-20260906-results.tar.gz`.
  Local SHA-256 matches remote:
  `4b6b11e075fe8c6e87cfc6a5b5c7b415755ff2c1c107ab7404d6ce9418372180`.
- Additional exact-attention probe logs:
  `/dev/shm/lunaflux-attention-serving-9be6973-20260906-r1` on the test host.
- Those logs and the fixed unfused run are downloaded in
  `/private/tmp/lunaflux-unfused-dc2368c-20260906-results-r2.tar.gz`, SHA-256
  `3b5a2ddaab7fbc7f2e86866e53095f33d3df062912978b30253ae37355657440`.
  The final `FILES.v2.sha256` verifies after shutdown. The first archive was
  captured before the shutdown log flush and is superseded, not overwritten.

All isolated benchmark servers were stopped; the GPU has no compute processes.
Production deployments were not changed.

## Still open

Per-bucket distinct projection/ingress artifacts and their startup dispatch
remain unimplemented. Partial/full numerical alignment and measured equivalent
ingress selection remain open. The existing C8 sequence variation is not fixed.
Captured suffix arithmetic is predicated, not a separately pruned graph.
No production deployment, other-engine speedup, or all-workstreams completion
is claimed.
