# Sparse projection and HC ownership

Scope: latest row-reuse/scales/bulk/local-MoE/exact-window candidate,
fixed first 64-token decoder chunk, both ranks, cold and warm. No overall
benchmark or payload hashing. Comparison stays outside the compiler.
Stop only the owned GLM service; restore its prior configuration afterward.
Limits: 100 GiB/device, 16 GiB host reserve, no owned-process swap.

The preceding warm rank-1 trace attributes 18.110240 ms to sparse projection
and 9.813023 ms to HC. Within those families, sparse output alone is
6.097984 ms and the 89 deferred post/pre calls sum 8.278496 ms.
These are selected kernel durations, not an additive whole-request estimate.

Finite campaign: at most two alternatives per target before reconsidering
the hypothesis. Preserve negative trials. Validate generated source, actual
launch ownership, numerical frontiers and sanitizers before both-rank replay.

Initial hypotheses to test:

1. Sparse output/query-B still use column-heavy 16x64 ownership despite
   multiple independent token rows. Reuse rows only where it does not undo
   the already-qualified wider-K staging selection. Keep the same ascending
   matrix accumulation and BF16 publication.
2. Deferred HC still publishes every level of the same reduction tree
   through workgroup shared memory. A subgroup-local suffix may preserve
   exactly the same addition tree while reducing barriers and shared traffic.
   Test the complete row-dispatched entry, not merely a standalone bucket.

The installed vLLM source and retained selected trace are performance
references, not correctness oracles. Its HC uses a matrix prenorm path;
changing our scalar arithmetic into that path is not assumed equivalent.
The local `../vlm` tree may be newer than the installed reference.

## Isolated evidence

HC source references: installed model snapshot
`/tmp/lunaflux-glm-selected-vllm-source-20261019-v3`, model.py
`hc_fused_post_pre`; local `../vlm/vllm/model_executor/kernels/mhc/tilelang.py`
433–634 distinguishes the small FMA route from larger-row post + matrix
prenorm. `tilelang_kernels.py` 525–640 uses subgroup reductions in the
small route. Our retained 64-row vLLM trace actually selects
`sm120_tf32_hc_prenorm_gemm_impl`; do not mislabel the small FMA source as
the executed prefill kernel. Neither its reassociation nor TF32 arithmetic
is silently substituted for our declared normalized scalar folds.

HC root `/tmp/lunaflux-glm-hc-vocab-reduction-local-20261020-v1` preserves
the first subgroup-only trial. At 64 rows it saves only a few percent; the
22/36-row trials are noisy. Counters still show dominant global-load waits.
Second root `/tmp/lunaflux-glm-hc-vocab-reduction-wide-20261020-v2` retains
the full dispatched baseline (widths1/2/4) and subgroup+reuse alternative
(widths1/2/8). All seven 64-row pairs improve: baseline56.20–56.77 us,
candidate36.64–38.98 us. Both preserve every tested publication, inverse
and control projection bit, including inactive bytes.
The failed v1 preparation is retained: SSH control masters expired; no
candidate GPU execution occurred there. Both masters were re-established.

Selected 64-row counters, same complete dispatch (not isolated width arms):

| Metric | Shared-tree / width4 | Subgroup suffix / width8 |
| --- | ---: | ---: |
| Warp instructions | 9,042,304 | 6,096,832 |
| L2 read sectors | 1,591,894 | 1,477,788 |
| Registers/thread | 96 | 104 |
| Register-limited resident blocks | 2 | 2 |
| Active warps | 32.08% | 32.75% |
| Long-scoreboard share | 69.87% | 68.73% |
| Barrier share | 4.94% | 5.02% |

Fewer instructions and repeated producers matter, not a lower stall
percentage. NCU replay durations are not substituted for warm graph trials.

Sparse output root
`/tmp/lunaflux-glm-projection-staging-sparse-reuse-20261020-v1` tests the
actual dense lowering, 64 rows, K8192/N4096, reserved1024-CTA grid.
M32/N32/K32 is inconclusive (one pair loses); M32/N32/K64 wins all five
pairs, median paired gain15.408%, minimum3.233%. Median time483.443→417.394 us.
NCU instructions24,591,360→16,872,448; L2 sectors8,212,913→6,527,076;
active warps44.41→44.15%; tensor activity4.99→7.20%; registers40→40.
The prior KDA fused-K64 loss is not contradicted: it has a different
resource envelope. This alternative is selected only by an explicit
independent sparse projection ownership envelope.

The pure `MatrixRowReuse` pass preserves previously selected K128 or buffered
staging. The supplied test envelope uses minimum capacity32, minimum K1536,
and K64 only from K8192; other qualifying reductions use K32. It reaches
sparse output, Q-B and index-Q through actual source and launch construction.
Q-A, KV-A, KV-B, index-K/weights and pooling, recurrent kernels, numerical
laws and device allocations remain unchanged. These bounds are explicit
offline configuration, not model-name branching or runtime tuning.

Physical `ReductionTreePartition` preserves every descending addition level,
moving only the suffix into one subgroup. The eight-control candidate changes
producer reuse, not control ordering. Comparison remains external.

Both-rank traced results, validation gates and restoration are pending.

The width8 trial loses at some36-row pairs. Final catalog selection restricts
width8 to live64, retaining widths1/2/4 elsewhere. Final complete-wrapper
root `/tmp/lunaflux-glm-hc-vocab-reduction-bucketed-20261020-v1` retains a
small36-row regression even with four-control ownership (typically
27.8→28.6 us): adding a bucket changes the shared compiled resource envelope.
This is visible, not claimed solved by the branch alone. All seven64-row
pairs still improve (53.84–58.30→36.48–40.01 us). No production promotion
is inferred from this long-prefill qualification.

The intermediate whole-AOT v1 compilation was explicitly stopped after
refining this row domain; no GPU model run used it. Its source/logs remain.
