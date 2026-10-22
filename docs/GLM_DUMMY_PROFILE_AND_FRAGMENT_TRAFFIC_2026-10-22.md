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
The final trial reached layer7 attention, instead of stopping at layer3, but
again retired at the conservative reserve guard **before graph capture**.
At rank0's layer7 attention boundary, reserved bytes fell from95,709,822,976
to95,172,952,064 after cache release: another512MiB, with live allocated bytes
unchanged at95,153,820,160. Available host memory rose from19,138,172KiB to
19,692,820KiB (18.252/18.780GiB). The independent guard also observed
19,169,808KiB and stopped the owned pair. Owner-watch minima were18.282/19.483GiB;
the lower boundary-hook value is included above rather than omitted. GPU
owner-watch maxima were89.783/90.271GiB.

These are successful, directly measured reductions of two temporary allocations,
not successful full initialization. The19GiB observer/18.5GiB external early
guards deliberately precede the user's16GiB floor: they do **not** establish
that this engine cannot fit within16GiB, nor constitute an OOM. Remaining
pressure is observed in layer7 attention; its inner allocation has not been
causally isolated. The finite reference-start budget is exhausted. No fresh
vLLM graph/request execution trace was produced by these three attempts.

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

One additional bounded, serial counter-only capture used the accepted, wide
and installed-reference leaves on identical saved operands:
`/tmp/lunaflux-projection-memory-counters-20261022-v1`. It confirms L2 read
sectors6,294,081/3,149,648/3,149,632 respectively. Requested DRAM byte/throughput
metrics are absent from the exported raw reports, so no DRAM bandwidth claim
can be made. The respective profile durations321.824/371.872/339.296us are not
alternating unprofiled trials and must not be used to claim an engine speedup.
All three bounded4GiB leaves retired before any full-model trace began.

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

## Completed latest native two-rank retrace

Root `/tmp/lunaflux-glm-http-dummy-traffic-20261022-v2` uses the qualified
ed6be466 executable/b1788d1d AOT path: scales+bulk experts, local MoE combine,
BF16 projection TP and16-site graph windows. No rejected wide-fragment
candidate is selected. The earlier `v1` launch failed before either GPU owner
started because the runner inherited obsolete SSH sockets; explicit existing
context sockets corrected the harness in the distinct `v2` evidence root.

Both literal requests finish with7973 prompt tokens,0 cached tokens,1 requested
output token and `[DONE]`. They are cold/warm diagnostic trace requests, not an
overall performance sweep or answer-quality evaluation. A separate complete
arithmetic request returns exactly`323`, finish reason`stop`. Both collectors
seal two windows with dropped/invalid/errors/refused all0. Capture is bounded
to1000ms per window; the table below allocates **one complete64-token chunk,
all45 layers**, not the whole7973-token request. The old graph topology still
applies: HC -> attention projections/core -> TP -> HC/FFN input -> parallel
routed/shared branches -> local combine -> TP -> next HC. First three FFNs
are dense instead of MoE.

Non-overlapping wall allocation, milliseconds. The vLLM column is explicitly
the **retained October19 rank1 trace**, not a new successful reference run:

| Stage | Native cold R0 | Native cold R1 | Retained vLLM R1 |
| --- | ---: | ---: | ---: |
| KDA projections | 34.218433 | 34.762221 | 26.704800 |
| Sparse projections | 15.860067 | 15.681542 | 9.372864 |
| HC | 6.726022 | 7.054959 | 3.950080 |
| KDA other | 8.850429 | 8.941144 | 4.422656 |
| Dense FFNs | 5.861578 | 5.854931 | 2.281952 |
| TP, including waits | 18.203554 | 16.758112 | 17.468992 |
| MoE union, overlap counted once | 183.166032 | 183.149620 | 184.459176 |
| Sparse core/index/metadata | 1.816410 | 1.950577 | 2.116095 |
| No recorded kernel | 0.651455 | 0.856122 | 5.255609 |
| Small cross-family overlaps | 0.007520 | 0.007520 | 0.038464 |
| **Decoder wall** | **275.361500** | **275.016748** | **256.070688** |

Cold R1's historical excess is18.946060ms; projections contribute14.366099ms,
75.83% of that difference. This is attribution, not a guaranteed recoverable
gain. The native path is unchanged from its earlier qualified retrace, so
variation from the previous273.584ms capture is **not** a compiler improvement
or regression established by paired trials. Historical vLLM clocks/load state
are not contemporaneous controls.

| Cold/warm first-chunk measurement | Cold R0 | Cold R1 | Warm R0 | Warm R1 |
| --- | ---: | ---: | ---: | ---: |
| Decoder wall,ms | 275.361500 | 275.016748 | 274.223265 | 274.119178 |
| TP union,ms | 18.203554 | 16.758112 | 17.484461 | 15.543488 |
| Shared overlapping other work,ms | 107.910566 | 107.240019 | 108.216471 | 109.298029 |
| Shared uncovered,ms | 0.139808 | 0.141410 | 0.098079 | 0.096929 |
| Compute/TP overlap,ms | 0 | 0 | 0 | 0 |
| Decoder end -> next chunk,ms | 2.203513 | 2.564608 | 2.463544 | 2.858373 |
| Handoff without recorded device activity,ms | 1.861595 | 2.230807 | 2.127929 | 2.523773 |

Shared work remains almost entirely overlapped; that does not prove resource
contention is absent. Handoff includes head/metadata/status operations outside
the decoder table. Its uncovered interval is not identified as NCCL wire time
or attributed to a specific CPU function.

All90 collectives per chunk match by order/kind across ranks. Causal offset
interval widths are0.224261ms cold/0.238429ms warm; host clocks are not assumed
equal. Cold first collective R0/R1 durations are1.247420/0.630896ms; R0 early
arrival is bounded0.492560–0.716821ms. Warm first durations fall to
0.314303/0.205797ms. The first two dense-FFN reductions on cold R1 are
0.181509/0.163364ms, not the earlier3ms stall. Across all90 sites, the largest
R0 remainder after both ranks start is bounded0.530599–0.754860ms cold and
0.173024–0.340767ms warm; R1 maxima are0.630896ms cold and
0.223135–0.281175ms warm. These include device/protocol progress, **not pure
network transfer time**.

The same-stream producer analysis correctly rejected graph-created streams
whose first visible event is a collective. Reports therefore use the existing
`--collectives-only` mode: no guessed producer timestamp or launch latency.
Graph edges, not adjacency on these virtual streams, would be needed for that
additional attribution. Per-site early-arrival bounds remain available.

Raw evidence: `rank{0,1}-activity.tsv`, `rank{0,1}-{cold,warm}.sqlite`,
`rank{0,1}-{first,warm-first}/summary.json`, `{cold,warm}-paired-tp.json`,
`{cold,warm}-allocation.json` and `rank{0,1}-{cold,warm}-handoff.json`.
The retained reference database remains
`/tmp/lunaflux-reference-bench-glm-long-egress-trace-20261019-v2/rank1.sqlite`.

Sampled native retrace GPU maxima94.241/94.270GiB and host minima
17.601/18.647GiB satisfy the requested100/16GiB bounds; owned cgroups report
zero swap. These are sampled physical envelopes, not a continuous-peak proof.
Both diagnostic model owners retired before restoration began.

## Decision and restoration

The unreachable vLLM dummy workspace correction is demonstrated. Complete
reference initialization remains unresolved. Wider native projection ownership
reduces traffic/instructions but has **no qualified timing benefit** and is not
enabled. The remaining projection gap is real; matching L2 traffic alone did
not close it. Preserve the load-dependency/instruction evidence and do not
replace it with a claim that all remaining time is DRAM or NCCL.

The accepted GLM service is restored. Both exact USER units are active,
`/health` reports healthy, and a fresh complete answer returns exactly`323`
with finish reason`stop`. Separate restoration root
`/tmp/lunaflux-glm-dummy-traffic-restoration-20261022-v1` records host minima
17.563/19.099GiB and GPU maxima94.239/94.268GiB; both owned units report zero
swap. The observer retired after health/answer verification and leaves the
production service running. No overall benchmark, payload hash, numerical
tolerance change or unqualified production selection occurred.
