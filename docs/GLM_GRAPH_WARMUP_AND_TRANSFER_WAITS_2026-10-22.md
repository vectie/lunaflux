# GLM graph warmup and projection waits

Continuation of `GLM_LOADER_ALLOCATIONS_AND_INSTRUCTIONS_2026-10-21.md`.
Comparison remains external to the compiler. No payload hashing, overall
benchmark, relaxed numerical bound, or unqualified serving selection.
Limits: 100 GiB per Spark, 16 GiB host reserve, zero owned-process swap.
Budget: one warmup-observation launch, one lifetime-release retest, two
projection alternatives. Preserve every result and restore the accepted GLM.

## Actual source and scope

The installed reference is vLLM `0.1.dev20051+g487ecf187`, cached image
`ghcr.nju.edu.cn/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-e3-20260907`.
Its logs select the **V2** runner, not `v1/worker/gpu_model_runner.py`.
CPU-only installed snapshots are retained under
`/tmp/lunaflux-glm-installed-warmup-20261022-v1`.

`v1/worker/gpu/model_runner.py::capture_model` clears caches before entering
the graph manager. `gpu/cudagraph_utils.py::CudaGraphManager.capture` then
creates an eager-warmup closure, executes it, creates fresh attention state,
and captures graphs. The eager closure still exists while replacement state
is constructed, and eager allocator caches are not explicitly released at
that boundary. This establishes lifetimes, not their measured byte cost.

The external diagnostic records phase, allocated/reserved/active CUDA bytes,
process anonymous/file/shared PSS, host availability and bounded allocator
history. It does not instrument production token execution. The release
alternative synchronizes completed warmup work, drops the obsolete closure
where a fresh one is required, collects unreachable objects, releases unused
CUDA cache, and returns free libc heap pages. Capture sizes `[1,2,4,8]`, graph
modes, model tensors, KV256MiB, context8192, sequence capacity8 and batch64
remain unchanged. The alternative is not a claim that a particular allocation
caused the old peak until its before/after records establish that.

The first launch stopped during weight loading, before reaching the observer:
rank0 host availability20,458,936KiB (19.511GiB), against a20GiB early guard.
It cannot diagnose graph capture. Exact root:
`/tmp/lunaflux-reference-bench-glm-warmup-phases-20261022-v1`.
Unlike the prior five-second graceful-stop tail, the diagnostic owner now
immediately kills only its two named containers when retiring. The subsequent
trial uses an18.5GiB early guard, not a lower user floor, with the same immediate
retirement. Phase observations also refuse further work below that headroom.
Prepared release-v1 was never launched; corrected immutable release-v2 avoids
overwriting snapshots at repeated graph sizes.

### The retest stopped before graph capture, not inside it

The second launch, `/tmp/lunaflux-reference-bench-glm-warmup-release-20261022-v2`,
completed all150226 weight entries on both ranks. Installed logs report
88.46GiB model loading,201.236s/204.773s. The last model logs are HC/TileLang
dummy-warmup compilation warnings; no graph-observer event was reached.
Consequently the proposed graph-lifetime release never executed and is
**unvalidated**, not a demonstrated memory fix.

Installed `v1/worker/gpu_worker.py:478` calls `profile_run()` even with an
explicit KV-byte budget. V2 `gpu/model_runner.py:782` runs the max-token dummy
forward before graph capture. The terminal sequence is consistent with this
earlier profile warmup; the particular allocating operator is not identified
by the current graph-only hooks. A future allocation capture must start at
this earlier entry, not repeat the graph-only experiment. An HC compilation
warning alone does not prove HC owns the memory peak.

The external headroom guard retired both exact owned containers immediately.
At the trigger rank0 had17,395,996KiB available; the independent process sample
minimum was17,395,492KiB (16.590GiB). Rank1's independent minimum was
18,387,144KiB (17.535GiB), lower than its slower owner-watch observation.
Owner-watch GPU maxima were89.708/90.271GiB. These are sampled bounds, not a
continuous physical peak. Exit137 was the requested retirement, not evidence
of an OOM killer. No fresh reference decoder trace was produced. Retained
reader/process/terminal records are in the same root.

## Projection experiments: no qualified improvement

The accepted sparse-output leaf is M64/N4096/K8192, runtime grid1024/block128,
M32/N64/K64 ownership, ascending K16 F32 accumulation and BF16 publication.
Installed vLLM dispatch selects
`nvjet_sm121_tst_mma_128x64x64_4_32x64x64_tmaAB_bz_TNNN`; its SASS, rather than
an invented public kernel implementation, demonstrates tensor-map transfers,
swizzled matrix loads, fragment reuse and phase-driven sleeping waits.

Both new alternatives preserve the accepted arithmetic and fragment lookahead.
They combine rectangular tensor-map transfers with the existing128-byte XOR
swizzle and independent operand epochs. Descriptors are encoded once outside
timed/graph work. Candidate1 lets consumers wait using phase checks plus sleep;
candidate2 uses one elected waiter followed by workgroup publication. These
are different from the earlier unswizzled small-tile WMMA/TMA experiment.
The temporary pure `FragmentOperandTransfer` plan describes transfer/waiter
ownership; CUDA descriptors and instruction selection remain private lowering.

All active/inactive bytes match at rows0/1/3/22/36/64/65, with the unchanged
sampled F64 bound. Five alternating pairs,20 graph replays each, seed713:

| Rows | Candidate1 baseline→candidate median, us | Candidate2 baseline→candidate median, us |
| ---: | ---: | ---: |
| 1 | 290.602→292.517 | 318.907→311.733 |
| 3 | 307.776→299.806 | 298.954→301.309 |
| 22 | 317.798→307.162 | 307.565→313.675 |
| 36 | 316.062→307.067 | 304.974→307.363 |
| 64 | 309.118→323.862 | 319.099→311.210 |

Every cell fails the minimum-pair gain gate. Candidate2's64-row ratio of
medians looks positive, but its median paired gain is−0.153% and its worst
pair is−3.559%. Row1 uses the unchanged decode branch, not tensor-map transfer.
These are not serving speedups. No sanitizer pass is claimed for rejected
performance candidates.

| Counter at64 rows | Consumer phase wait | Elected waiter |
| --- | ---: | ---: |
| Executed warp instructions | 14,816,116 | 12,140,833 |
| L2 read sectors | 6,294,408 | 6,299,062 |
| Registers/thread | 51 | 51 |
| Active warps,% peak | 21.991 | 22.006 |
| Long-scoreboard share,% | 16.365 | 2.815 |
| Barrier share,% | 0.856 | 67.783 |
| Tensor activity,% peak | 10.137 | 9.446 |
| Profile duration,us | 370.560 | 377.568 |

Compared with retained accepted-source counters, the first candidate reduces
scoreboard/barrier shares substantially without a reliable latency win. The
elected waiter moves most sampled waiting into workgroup publication. Traffic
remains near6.29M sectors rather than the reference's3.15M. Counter categories
are not additive wall-time allocations, and these observations do not prove
DRAM bandwidth saturation or an isolated number of milliseconds recoverable.
Do not repeat buffer-depth or wait-style changes as if their benefit were
already established.

The unsuccessful compiler/lowering/harness edits are withdrawn and preserved
in `benchmarks/experiments/glm_swizzled_rectangular_waits_20261022.patch`.
The patch applies to accepted main. Generated sources, paired samples and
counters remain in `/tmp/lunaflux-glm-projection-staging-swizzled-transfer-20261022-v1`
and its separate `swizzled-transfer-export` root. No new comparison dependency,
experimental public IR, descriptor ABI or serving policy remains active.
Affected tests passed39/39 with the experiment and37/37 after withdrawal;
warning-denied native check passes with existing migration exclusions unchanged.

## Two-rank trace: complete first chunk, incomplete campaign

Root `/tmp/lunaflux-glm-http-warmup-waits-followup-20261022-v1` runs the accepted
ed6be466 service executable/b1788d1d artifacts, BF16 projection TP, local MoE
combine,16-site replay windows,scales+bulk experts. Neither rejected transfer
candidate is selected. The literal7973-token request uses64-token chunks.

The campaign reached its17GiB diagnostic headroom stop on rank0, before the
16GiB user floor. It therefore did not finish the long request or warm replay.
Nevertheless both retained cold traces contain the complete **first64-token
chunk through all45 layers**, including90 paired collectives. The analyzer
checked that extent before allocation; partial later chunks are excluded.
There is no new long-request correctness or warm-replay completion claim.

Non-overlapping first-chunk wall allocations,ms:

| Stage | LunaFlux rank0,Oct22 | LunaFlux rank1,Oct22 | Retained vLLM rank1,Oct19 |
| --- | ---: | ---: | ---: |
| KDA projections | 34.367 | 34.120 | 26.705 |
| Sparse projections | 15.413 | 15.797 | 9.373 |
| HC | 6.642 | 6.671 | 3.950 |
| KDA other | 8.715 | 9.036 | 4.423 |
| Dense FFN | 5.820 | 5.852 | 2.282 |
| TP including waits | 16.979 | 15.527 | 17.469 |
| MoE union including shared overlap | 183.714 | 183.368 | 184.459 |
| Sparse core/index/metadata | 1.817 | 1.836 | 2.116 |
| Kernel-free/cross-family overlap | 0.639 | 1.376 | 5.294 |
| Decoder wall | 274.105 | 273.584 | 256.071 |

The reference column is explicitly historical, not a fresh paired result.
No speedup is attributed to this turn: its serving implementation is unchanged.
The two projection categories still contribute13.84ms of the historical
rank1 difference; changing wait counters alone did not remove their work.

Shared work overlaps other compute for108.345/107.343ms, leaving only
0.137/0.145ms uncovered on rank0/1. No compute/TP overlap is recorded on either
rank. It would double-count most of the shared branch to add its full duration
to decoder wall time; its resource contention remains unisolated.

The90 collective pairs constrain clock offset to a0.244ms interval. Rank0's
first collective is1.790ms, with1.103–1.347ms bounded as early arrival; rank1's
is0.559ms. The first two dense-FFN collective durations are0.183/0.169ms on
rank0 and0.162/0.165ms on rank1. The maximum per-site interval after both ranks
start is bounded0.443–0.687ms on rank0,0.559ms on rank1. These are causal
intervals, **not network wire times**; producer-CPU launch attribution is not
present in the filtered trace.

First-chunk completion to next-chunk start is2.346/2.469ms; respectively
2.004/2.134ms has no recorded device work. This is not a diagnosed CPU cause.
Reports: `rank{0,1}-first/summary.json`, `rank{0,1}-handoff.json`,
`cold-paired-tp.json`, `cold-allocation.json`. Observed GPU maxima94.241/94.270GiB,
host minima16.879/19.254GiB,owned swap0. The failed campaign and later partial
activity are retained; neither a full capture pass nor a fresh vLLM comparison
is claimed.

## Restoration and decision

Qualified GLM is restored, both user units active, private health healthy.
A fresh greedy26-token prompt asking17×19 completed with`323` and`stop`.
Records under `/tmp/lunaflux-glm-warmup-waits-restoration-20261022-v1` include
before/after unit states, the fresh answer, health and the bounded observer's
terminal completion. GPU maxima94.239/94.268GiB, host minima17.422/19.193GiB,
owned swap0. The accepted ed6be466/b1788d1d service remains selected.
The initial test incorrectly used the public-route model alias on the private
endpoint and received400; the corrected v2 request uses the installed
`glm-5.3` model identifier. This was a test-request mistake, not a model fix.

The three affected diagnostic scripts format/check successfully and their
12 tests pass. The archived experiment applies cleanly to accepted main.
No new full-suite, sanitizer, long-answer correctness or fresh-reference pass
is claimed. No payload hashing or overall benchmark was run.

The AKO finite budget is exhausted: two bounded loading/warmup launches and
two rejected projection alternatives. Both requested performance fixes remain
unresolved. The evidence rules out treating a different wait counter as a
latency improvement and moves the reference allocation investigation before
graph capture. It does not justify lowering the memory reserve, shrinking the
reference workload, changing numerical tolerances, or promoting the patch.
