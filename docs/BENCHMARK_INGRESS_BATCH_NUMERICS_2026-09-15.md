# Ingress numerical replay and batch invariance

This is a numerical diagnostic, not a performance measurement or a production
qualification. It follows the unresolved actual-model differences in
`BENCHMARK_LOGIT_MARGINS_2026-09-14.md`.

## Historical binary replay

The original full runtime bundle in
`/tmp/lfmarginfull-20260914-r1/full/runtime.v3` contains ingress module SHA-256
`d15996ed53f4eb3314da2348aca5bdda00e7d69aaeb892319ba98a5eefbe33f6`.
That exact module was extracted and loaded through the CUDA driver; it was
not recompiled. The partial/full bundles have identical attention and residual
module hashes. This does not prove their scheduler trajectories were identical.

The first replay incorrectly constructed 16-token pages for an 8-token-page
historical module. It consequently skipped positions 8–15 in each 16-token
group. Those runs are invalid numerical comparisons, not a production defect.
Their logs remain in `/tmp/lunaflux-historical-ingress-physical` and its `-r2`
directory. The corrected fixture now takes page geometry explicitly and derives
page-table capacity from it. Regression tests cover both 8- and 16-token pages.

With matching 8-token page geometry, historical full ingress and current
ordinary projection plus partial ingress produce bit-identical rotated output
and KV arenas for token counts 1/2/4/8/16/17/31/32/127/1024/2048. These are
synthetic signed BF16 operands with 17 exponent bins, one query row, input width
1024, output width 4096 and head dimension 128. This does not cover actual
model activations, all projection variants, or mixed request execution.

## Identical input row under different batch sizes

A separate comparison holds row zero, weights and position fixed, changing
only the number of input rows. All multi-token cases differ from the single
token case at one of the 4096 projection components:

| Component | Single-row output | Matrix output | CPU Float64 dot |
| --- | ---: | ---: | ---: |
| 3864 | -1.15625 | -1.140625 | -1.1554306486668793 |

The difference is 0.015625. Component 3864 belongs to the value projection,
so it is not modified by QKNorm or RoPE; the same difference survives historical
full ingress. In this example the single-row result is closer to the independent
Float64 dot. This is one cancellation-sensitive synthetic example, not a bound
on whole-model error or proof that one schedule is universally more accurate.

Thus same-batch full/partial agreement does **not** establish batch-invariant
arithmetic. Single-row subgroup reduction and matrix accumulation can round
differently even with identical logical operands. Compiler numerical guarantees
must distinguish fixed-schedule repeatability from cross-schedule bit identity.

The actual-model first-divergence cause remains unproven. The next comparison
must match real activation inputs and actual batch/selected-kernel trajectories;
it must not infer causation merely because divergences happened near request
completion. Do not change tolerances, force all work onto a slower scalar path,
or promote full ingress based on this diagnostic.

## Validation

The expanded fused fixture passes 24/24 package tests; native warning-denied
check and interface generation pass. Diagnostic GPU runs used the RTX 5060 Ti
UUID `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6` with no concurrent GPU workload.
No serving implementation or model-specific production rule changed.

Downloaded sources, original module, probes and successful replay logs:
`/tmp/lunaflux-historical-ingress-retest.tar.gz`, matching local/remote SHA-256
`435ef52075b65478bf58aa460192b99faff8f6aa4c4ab5675cb9f92cd65cd568`.

## Actual trace join and within-configuration repeatability

Reprocessing the existing 3072-output traces on each side now attaches the
recorded batch rows, batch tokens, prefill rows and decode rows to every logit
record. No new GPU run was performed for this join. The original 96 requests
still have four cross-configuration first divergences at sample 31.

The single-row transition does not explain every observation: for
`uniform-distinct-c16-t1-r0`, every matching partial trace is multi-token
(2/8/13/16), and its full trace is a four-token decode. Matching remains
ambiguous when repeated requests produce identical output vectors; all matches
are retained rather than picking a convenient one.

More importantly, each runtime has one unstable request body among 16 distinct
byte-identical bodies that were repeated. This occurs even at fixed concurrency:

| Runtime | Fixed concurrency | Last-token observations across four trials |
| --- | --- | --- |
| Partial | C8 | 22, 22, 16, 16 |
| Full | C16 | 16, 22, 16, 16 |

Trial zero is warmup; this table includes it, consistently with the earlier
counts. The same request body's other tokens agree. Thus the pairwise 4/96
count alone cannot isolate a fusion effect: the unchanged runtime also varies
between repetitions. This is not proof of a race, nor proof that batch size
alone causes it. Compare fixed/replayed schedules and actual intermediate
activations before attributing the difference to an individual compiler pass.

`summarize_logit_margins.mbtx` now preserves batch context and rejects a margin
without a preceding batch record. `compare_logit_margins.mbtx` additionally
reports identical-body repeatability for each configuration. Regression tool
`test_margin_analysis.mbtx` exercises batch changes, within-configuration
instability, a stable control and missing-batch rejection. It passes locally.

The joined summaries, comparison, repeatability checks and scripts were
downloaded as `/tmp/lunaflux-margin-batches-20260915.tar.gz`; local/remote SHA-256
is `1656df2f46fba7d2ca3ffc5794057e184b49375419fd1b4f00e863837446848b`.
The original raw trace archive remains the one linked in the logit-margin
report; no original trace or response was overwritten.
