# GLM dummy-profile allocation and fragment traffic campaign

Continuation of `GLM_GRAPH_WARMUP_AND_TRANSFER_WAITS_2026-10-22.md`.
Initial budget: one observed reference profile run, one evidence-led allocation
retest, two wider fragment ownership alternatives. After proving removal of the
unreachable workspace, one additional final initialization-cache retest was
announced, based on its new snapshot. No overall benchmark, payload hashes,
tolerance changes or unqualified serving promotion.100GiB/Spark,16GiB host reserve.
Restore the accepted ed6be466/b1788d1d GLM service after owned trials retire.

The installed V2 runner calls `profile_run` before graph capture, even with
explicit KV bytes. The new external observer wraps that actual method, keeps
its arithmetic/capacities unchanged, records per-layer boundaries and bounded
allocator history, and samples process memory during long compilation calls.
The graph-only hooks from the previous campaign cannot identify this earlier
peak. No allocation reduction is assumed before the new records exist.

Projection hypothesis: retain the existing swizzle, two operand epochs,
fragment reuse, register lookahead and ascending K16 fold, but increase the
independent output-column ownership from64 to128. Test M32/N128 then M64/N128.
For the M64/N4096/K8192 operand set, the logical no-cache reuse counts predict:

| Ownership | Weight bytes | Input bytes | Total bytes |
| --- | ---: | ---: | ---: |
| M32/N64 accepted | 134217728 | 67108864 | 201326592 |
| M32/N128 candidate | 134217728 | 33554432 | 167772160 |
| M64/N128 candidate | 67108864 | 33554432 | 100663296 |

These are planned transfer bytes, not DRAM estimates. The accepted6.294M and
reference3.150M L2 read sectors are consistent with the endpoints, but cache
traffic must be measured on the candidates. Installed nvjet source is closed;
its actual SASS shows128/64 coordinates and selected symbol
`nvjet_sm121_tst_mma_128x64x64_4_32x64x64_tmaAB_bz_TNNN`.
The measured launch is256 blocks,128 threads,255 registers/thread and99,328
dynamic shared bytes. Grid size alone does not prove useful tile count or split-K.
The proposed native change does not introduce split-K or change precision.

The wider ownership is an immutable `MatrixFragmentPipeline` choice; only
private lowering changes pointer expressions, fragment count and local storage.
The accepted default remains64 columns. Higher register pressure and fewer
independent tiles are explicit falsification risks, not presumed benefits.

## Measured dummy-profile cause and correction

Root `/tmp/lunaflux-reference-bench-glm-dummy-phases-20261022-v1` reached
the actual V2 profile entry and layers0–3. The observer's original forward
is unchanged; module hooks are removed before any serving trace.
Rank1's layer3 sparse-attention call changed:

| Torch counter | Before | After |
| --- | ---: | ---: |
| Live allocated bytes | 95074390528 | 95119741440 |
| Reserved bytes | 95093260288 | 97855209472 |
| Peak allocated bytes | 95079109120 | 97302876672 |

The retained allocator snapshot `dummy-rank1-headroom.pickle` contains an
inactive2,181,038,080-byte segment, with allocation stack at installed
`model_executor/layers/attention/mla_attention.py:759`, `forward_impl`.
That is the **2.03125GiB synthetic dense-MHA chunk workspace**, allocated
when `attn_metadata is None`; it is not a model tensor or graph allocation.
The installed SM120 implementation explicitly sets
`supports_dense_mha_prefill = False`. Both logs select this implementation and
say it only uses top-k MQA. The dummy profile nevertheless allocated the
dense-only workspace unconditionally. This is a demonstrated capability/profile
mismatch, not an inference from the last compilation warning.

`scripts/prepare-glm-dummy-mla-workspace.mbtx` guards only this synthetic
allocation with the existing sparse/dense capability flag. Dense-capable
backends still allocate their forecast; zero-filled dummy outputs, real
attention execution, KV budget and graph policy remain unchanged. The fix is
an explicit installed-reference source overlay, outside LunaFlux's compiler.
It neither disables profiling nor suppresses memory accounting for a supported
execution path. Installed sources are preserved under
`/tmp/lunaflux-glm-installed-dummy-20261022-v1` and original/corrected sources
under `/tmp/lunaflux-mla-profile-fix-capability-20261022-v1`.

The first trial retired both exact containers at the earlier reserve guard,
not an OOM. Combined owner/window observations: host minima18.919/17.366GiB,
GPU maxima89.708/89.962GiB; captured model-worker smaps report zero swap.
These are sampled envelopes,
not continuous physical maxima. The corrective retest root is
`/tmp/lunaflux-reference-bench-glm-dummy-fixed-20261022-v1` proves that at the
same after-layer3-attention boundary live bytes are exactly unchanged,
95,119,741,440, while reserved bytes fall from97,855,209,472 to95,674,171,392:
**exactly2,181,038,080 bytes removed**. Peak allocated bytes fall to95,658,516,992.
This establishes the targeted memory fix, not complete startup success.

The second trial also retired conservatively: rank0 after attention had
19,207,216KiB available (18.317GiB); rank1 reached after layer3 MoE with
19,856,708KiB (18.937GiB). Both are above the user's16GiB floor but below the
observer's19GiB early boundary. The snapshot now shows a different, inactive
512MiB segment. Peak GPU owner-watch values were89.708/90.283GiB. The observer
records lower host availability than the less-frequent owner watch; its lower
values must not be omitted from the envelope.

The final bounded trial, root
`/tmp/lunaflux-reference-bench-glm-dummy-release-20261022-v1`, retains the
capability fix and releases unused caches only after completed dummy module
calls: synchronize, collect unreachable Python objects, release inactive CUDA
cache, and trim the calling process's free libc heap. Before/after records
quantify the effect. It does not clear live tensors, reset peak statistics,
alter graph sizes, or add a token-step operation; all temporary hooks retire.
Its terminal result is pending, not a claimed fix for every initialization peak.

## Traffic reduction succeeded, performance qualification did not

Actual compiled candidates in
`/tmp/lunaflux-glm-projection-staging-wide-fragment-20261022-v1`, separate export
root `wide-fragment-export-20261022-v1`, ran the accepted1024-block/128-thread
capacity launch. Inactive CTAs exit according to the unchanged live-row plan.
Rows0/1/3/22/36/64/65 pass byte comparisons and the unchanged sampled F64 bound.
No reduction splitting or arithmetic reassociation was introduced.

|64-row counter|M32/N128|M64/N128|
|---|---:|---:|
|Warp instructions|11,172,352|8,271,232|
|L2 read sectors|5,246,468|3,149,690|
|Registers/thread|80|124|
|Static shared bytes|40,960|49,152|
|Active warps,% peak|11.271|8.600|
|Tensor activity,% peak|9.671|14.548|
|Long-scoreboard share,%|62.510|52.824|
|Barrier share,%|18.225|16.843|
|Profile duration,us|379.776|372.000|

M64/N128 reaches the reference's approximately3.150M L2 read sectors and
halves accepted traffic; instructions fall from retained17.119M to8.271M.
It still executes3.61times the reference's2.293M instructions. Matching traffic
does not match the pipeline: substantial load waiting remains, while register
ownership and fewer independent useful tiles change latency-hiding resources.
These observations do not isolate a causal millisecond cost of registers or
prove bandwidth saturation; stall percentages are not wall-time allocations.

Five alternating pairs of20 graph replays, unchanged seed713. Values below
are median paired candidate/baseline ratios; greater than1 is slower:

|Rows|M32/N128|M64/N128|
|---:|---:|---:|
|1|1.03720|1.02448|
|3|0.99248|1.01089|
|22|1.00813|1.00963|
|36|1.00476|1.02883|
|64|1.00279|1.00504|

Row1 uses the unchanged decode path and demonstrates trial variability, not
an optimization effect. Neither candidate meets the paired minimum-gain gate.
The AKO loop therefore withdraws the experimental public IR/lowering changes
and preserves them in
`benchmarks/experiments/glm_wide_fragment_traffic_20261022.patch`. The patch
excludes concurrent Int32 migration changes. No sanitizer pass or serving
speedup is claimed for these rejected candidates. Defaults and the accepted
serving artifact are unchanged.

With candidates, affected tests passed38/38; after withdrawal37/37. Observer,
profile-correction, reference-runner and window tests pass1+1+10+1.
The plain full warning-denied check/test commands stop on pre-existing
deprecated/derived-method migration diagnostics. Native check and affected
tests pass with the existing `-20-25-29-35-79-92` migration exclusions; no
configuration or exclusion is changed. Targeted `moon info` completes, and
the generated public interface is restored to its prior shape.

Fresh reference/native trace results and service restoration are pending.
