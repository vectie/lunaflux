# Bounded vLLM loading and projection follow-up

## Outcome and scope

**The fresh paired vLLM trace remains blocked by loading memory; the projection
candidate was rejected for a measured regression. Neither gap is claimed fixed.**
The accepted GLM compiler/runtime and serving projection selection are unchanged.
This campaign contains no overall benchmark, payload hashing, lower-reserve
retry, tolerance change, or substitution of a smaller model/workload.

The finite budget was two loading configurations and at most two projection
candidates. Both loading configurations were tried. One coherent projection
candidate was tested; no second candidate was invented simply to consume the
budget. The previously tested M64 and failed TMA/split-K alternatives were not
repeated. Numerical comparison and measurement remain outside the compiler.

## Actual reference loader and memory failure

The cached reference image remains
`ghcr.nju.edu.cn/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-e3-20260907`,
installed vLLM`0.1.dev20051+g487ecf187`. TP2, NVFP4 checkpoint, context8192,
eight sequences, batch64, KV256MiB, graph maximum8, and language-model-only
are unchanged. Both ranks have96GiB container memory and equal memory+swap
caps. The requested device ceiling remains100GiB/Spark and host floor16GiB.
The existing diagnostic guard remains18GiB to allow retirement headroom.
Activity collection is not enabled until readiness; neither attempt reached it.

Inspection of **installed** source, rather than a guessed loader option, found:

- Default safetensors loading is lazy; eager whole-file loading is not a
  justified memory reduction. Checkpoint181.30GiB exceeds available host RAM.
- The installed Runai loader accepts `distributed`, `concurrency` and
  `memory_limit`. Its library default is40,000,000,000bytes; `-1` is unbounded,
  while `0` uses a buffer large enough for the largest tensor.
- The vLLM iterator yields a tensor clone. Consequently the stream buffer limit
  is **not** a limit on all transient tensors or the whole worker process.

Installed source snapshots, with no new execution dependency, are retained in
`/tmp/lunaflux-reference-bench-glm-stream-load-20261021-v1/installed-*.txt`.

Both trials set `--load-format runai_streamer` and extra config
`{"distributed":false,"concurrency":1,"memory_limit":0}`. The second also
sets libc `MALLOC_ARENA_MAX=2` and `MALLOC_TRIM_THRESHOLD_=131072`, testing
allocator retention without changing inference arithmetic or graph mode.

| Loading trial | Sampled GPU max R0/R1, GiB | Host minimum R0/R1, GiB | Result |
| --- | ---: | ---: | --- |
| Minimum-buffer stream | 89.628 /89.628 | 17.999851 /19.344669 | Early guard, both retired |
| Same + compact libc allocation | 89.628 /89.628 | 17.014591 /19.319046 | Early guard, both retired |

Roots are `/tmp/lunaflux-reference-bench-glm-stream-load-20261021-v1` and
`/tmp/lunaflux-reference-bench-glm-stream-load-compact-20261021-v1`.
No observed sample breaches16GiB; these are sampled envelopes, **not continuous
physical minima**. The second run's drop between observations shows why the
18GiB early guard must not be lowered. No global cache drop, swap clear, or
unrelated process termination was used. Existing system swap is not attributed
to these no-additional-swap owners.

At the stopping observation rank0 AnonPages was6,565,040KiB in trial1 and
7,553,936KiB in trial2, while GPU usage remained89.628GiB. These observations
show that constraining the reader buffer did not bound total host anonymous
memory. They do **not** prove which tensor/process retained the extra pages or
prove that the libc change increased peak usage: the trials reached different
loading frontiers. No specific Python leak or CUPTI leak is inferred from them.

The external guard now records bounded meminfo fields and saves
`RESERVE_STOP.txt` **before** paired retirement. Previously the owner watcher
could replace that cause with generic `CommandFailed`. Both stops are now
reported with their actual reserve cause. Exit137 after requested Docker
retirement is not by itself an OOM diagnosis. Parallel five-second retirement
replaces sequential thirty-second stops for the two task-owned containers.

There is no fresh vLLM rank0/rank1 execution graph, overlap allocation or TP wait
bound from these failed loads. The October19 rank1 reference remains historical;
the valid latest native cold/warm paired trace is described in
[the filtered trace report](GLM_FILTERED_TWO_RANK_TRACE_2026-10-21.md).
Its approximately13.45ms projection excess versus that retained reference is
not re-labelled as a fresh matched result.

## Projection experiment: fewer epochs, unchanged arithmetic

Evidence preceding the experiment was the actual accepted
`glm_sparse_3_output` BF16 leaf and matched M64/N4096/K8192 counters. It has
much more per-thread transport/addressing work than the installed reference
`nvjet_sm121_tst_mma_128x64x64_4_32x64x64_tmaAB_bz_TNNN`.
The reference uses TMA; its implementation is a library kernel, not available
as readable implementation in `../vlm`. Installed dispatch/SASS is the evidence.
Reference counters are retained historical measurements, not collected anew.

Falsifiable hypothesis: doubling the operand epoch K64→K128 halves publication
and reader-retirement rounds while preserving M32/N64 ownership, two shared
slots, two register fragment slots, and the ascending K16 F32 MMA fold.
Expected cost: static shared storage doubles from24 to48KiB. This experiment
does not reduce repeated weight bytes, use TMA, or introduce split-K reduction.

Implementation was expressed through immutable `MatrixFragmentPipeline` and
`MatrixRowReuse`, propagated to actual source metadata and the explicit service
policy, with CUDA details private to lowering. Defaults retained K64; unsupported
tile/launch combinations were rejected. The generated baseline leaf was directly
compared with the qualified serving BF16 export and was unchanged. No digest.

Export: `/tmp/lunaflux-glm-projection-staging-wide-epochs-export-20261021-v1`.
Physical trial: `/tmp/lunaflux-glm-projection-staging-wide-epochs-20261021-v1`.
Variant0 is the accepted M32/K64 leaf, variant1 M32/K128, on Spark.179.
Runtime grid1024/block128 is preserved, including inactive capacity/decode CTAs.
Seed713, five alternating graph-timing pairs,20 replays per pair. No HTTP timing.

All active/inactive output bytes match at live rows0,1,3,22,36,64,65; the existing
sampled F64 oracle threshold is unchanged. Invalid row counts perform no stores.
This is bounded fixture correctness, not all-input proof or model-quality parity.

| Live rows | K64 median, us | K128 median, us | Paired decision |
| --- | ---: | ---: | --- |
| 1 | 310.832 | 286.522 | Inconclusive; same arithmetic path, one pair loses |
| 3 | 299.304 | 289.350 | Component gain, not selected serving envelope |
| 22 | 306.976 | 295.110 | Inconclusive; one pair loses |
| 36 | 305.042 | 427.390 | Regression |
| 64 | 314.517 | 444.914 | Regression |

The target64-row median duration increases41.46%; median paired increase41.34%.
These are different statistics. The candidate is not selected for long prefill.

### Same-campaign hardware counters

| Metric | Accepted K64 | Candidate K128 |
| --- | ---: | ---: |
| Profiled duration, us | 374.080 | 500.544 |
| Warp instructions | 17,119,232 | 13,490,688 |
| L2 read sectors | 6,294,258 | 6,294,942 |
| Registers/thread | 46 | 56 |
| Static shared, KiB | 24 | 48 |
| Shared-limited resident CTAs/SM | 4 | 2 |
| Active warps, % peak | 22.356 | 14.996 |
| Tensor active, % peak | 9.708 | 8.509 |
| Long-scoreboard contribution, % | 63.839 | 64.526 |
| Barrier contribution, % | 21.995 | 20.758 |
| Short-scoreboard contribution, % | 1.148 | 3.244 |

Profiled times are not unprofiled graph times. Stall percentages use active-cycle
denominators; they are not additive wall-time allocations. The candidate removes
about21.2% of instructions but leaves L2 traffic essentially unchanged. The
shared-memory residency limit halves, active warps fall, and load stalls persist.
This supports a resource/latency-hiding explanation for the regression; there is
no separate residency-only intervention proving an exclusive causal percentage.
Lowering the count of barriers alone did not solve the projection gap.

Following the AKO gate, the candidate was **removed from compiled production
packages**, not merely left disabled in a new runtime switch. Its complete
14-file immutable-plan/lowering/test patch is retained as
`benchmarks/experiments/glm_wide_operand_epochs_20261021.patch` against
`ad4cc85b`, for disposable offline reproduction. No new projection artifact was
deployed, no full-model candidate retrace was justified, and no sanitizer pass
is claimed for this rejected candidate. Its device allocations were released.

## Remaining work, without a guessed speedup

1. Reference loading: isolate actual host ownership at the next loading frontier
   before another configuration. The reader buffer and libc arena settings have
   failed to establish safe readiness. A GPU memory-utilization setting alone
   does not remove the fixed checkpoint plus host process footprint.
2. Projection: reduce redundant data movement/addressing without doubling the
   resident tile footprint. Actual reference L2 reads are about half native's;
   native K128 did not reduce them. Existing M64 and earlier TMA failures must
   inform the next design, not be silently repeated or promoted.
3. Only a component winner with correctness/resource qualification can justify
   new AOT deployment and both-rank trace attribution. Fresh vLLM readiness is a
   separate prerequisite. No new aggregate throughput claim follows this work.

## Validation and restoration

Only the external launcher/observer changes are retained executable changes.
Launcher tests10/10 and memory-parser test1/1 pass; native check passes with the
repository's existing migration-warning exclusions. Script formatting is clean.
The experimental plan/source tests passed during development, but that code is
now archived rather than active. Full strict repository validation remains
distinct: both warning-denied full check and test were attempted and fail on
pre-existing migration warnings, starting with deprecated `try?` in
`compiler/parallel_ir/collective_windows.mbt:85`. Logs are retained in
`/tmp/lunaflux-streamload-phase-checks-20261021-v1`. No new warning suppression
or unrelated migration edit was made. The archived patch also passes
`git apply --check` against the restored production files.

GLM restoration is monitored at
`/tmp/lunaflux-glm-streamload-restoration-20261021-v1` using the previously
accepted `ed6be466` units and qualified `b1788d1d` AOT. Final health, complete
answer and memory observations are complete. Both units are active/successful;
private `/health` is healthy. The complete arithmetic response is`323`, with
`finish_reason=stop`. This is restoration verification, not broader model-quality
requalification or a new customer-key public-route test.

Sampled restoration GPU maxima are94.239/94.268GiB; host availability minima
are17.105/19.211GiB for ranks0/1. Owned swap remains zero. The observer has
retired after recording terminal unit states, health and answer. Both failed
reference owners and the component probe have retired; GLM is left serving.
