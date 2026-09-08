# Compiler map and fold realization

This increment implements resident attention folds and evaluates a rejected
MLP map-packing alternative without changing the model graph, numeric contract,
or existing launch ABI.

## Attention output fold

The generic attention schedule chooses subgroup-resident output storage for
matrix QK/PV tiles with at most two 16-row fragments and sufficient subgroups to
own every head-dimension tile. Wider tiles retain the shared-memory realization.
This is a storage/lifetime transformation of the same ordered online-softmax
fold, not a reassociation of its reduction.

CUDA lowering retains the output accumulator fragments across KV tiles, applies
each row's previous-scale factor in registers, and stores only at termination.
The opaque WMMA element-to-row mapping is obtained once using the documented
fragment load operation on row tags; no architecture-specific lane map is
assumed. Shared scratch remains necessary for initialization and terminal output.
Longer register lifetimes must be measured; lower shared traffic alone does not
prove lower latency.

## Rejected independent projection map packing

A prototype factored independent output groups over existing workgroup
subgroups, filling subgroups that previously returned immediately. Each output
retained one owner and its complete ordered dot product. All token-vector output
and workspace comparisons were bit-exact; all four sanitizers passed.

However, three paired trials on the Qwen-shaped 1024/3072 MLP showed nearly
2x worse short-prefill latency: 8-token combined gate/up+down took about 56.5 us
before versus 106.8 us after. At 256 tokens it improved only from 371.6 to
359.2 us; at 1024 tokens it slightly regressed from 1357.2 to 1363.0 us.
Packing reduces useful CTA count and does not solve the block-occupancy problem.
The prototype and its unused planning API were removed. Independent stage launch
metadata remains necessary; it cannot be silently applied to older artifacts
whose down entry point explicitly requires the original block size.

The rejected experiment is preserved at
`/private/tmp/lunaflux-map-packing-20260908-results-r1` locally and
`/dev/shm/lunaflux-map-packing-20260908-r1` on the test host, including both
generated modules, probe, three timing trials, and sanitizer output.

## Validation and remaining scope

Retained focused tests cover attention storage selection and emitted CUDA
realization. Physical timing and sanitizer results are recorded below. These
changes do not complete GEMM pipeline
selection, direct ragged/new-KV attention views, or full compound-stage launch
planning.

### Measured attention result

RTX 5060 Ti, CUDA 13.1.115, Qwen geometry (16 query heads, 8 KV heads,
head dimension 128), schedule 312, eight requests. Query counts below are
per request, not flattened batch counts. Identical harness and compiler flags
compare the prior shared fold with the new resident fold.

| Query tokens | Context tokens | Shared fold, ms | Resident fold, ms | Speedup |
| --- | --- | --- | --- | --- |
| 16 | 4096 | 1.3641 | 1.0479 | 1.30x |
| 64 | 4096 | 3.1820 | 2.4921 | 1.28x |
| 128 | 4096 | 5.9800 | 4.6567 | 1.28x |
| 256 | 4096 | 11.2691 | 8.7038 | 1.29x |
| 512 | 512 | 1.6653 | 1.3364 | 1.25x |
| 512 | 4096 | 21.4604 | 16.5128 | 1.30x |

All 24 output captures match bit-for-bit, including single and ragged cases.
Memcheck, racecheck, initcheck, and synccheck pass. Registers increase from 126
to 128; both builds report zero spills and the same 32-byte stack frame. Shared
allocation remains 45,056 bytes. This is one paired microbenchmark campaign,
not a repeated end-to-end serving comparison against vLLM or SGLang.

Raw source, probes and results are preserved at
`/dev/shm/lunaflux-resident-fold-20260908-r1` on the test host and downloaded to
`/private/tmp/lunaflux-resident-fold-20260908-results-r1` locally. Directory
`c316` contains the **new realization of schedule 312**, not async schedule 316;
the numeric directory name is only a paired-run harness slot.
