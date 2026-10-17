# GLM expert pipeline experiment

The user authorized temporarily stopping `glm53-exl3-head` on Spark .178 and
`glm53-exl3-worker` on .179, then restoring those same containers. No production
engine cutover is authorized. Preserve both containers and the separate NFS
container. Benchmark limits remain 100 GiB per Spark, zero task swap and an
independent 16 GiB host available-memory reserve. No payload hashing.

The retained clean native baseline is coalesced-20261014-v1. Its selected
prefill trace attributes 74.721536 ms to routed gate/up and 36.826272 ms to down.
Prior same-source synthetic counters show substantial load-dependency and
barrier waiting, but are not counters from the full checkpoint. The previous
shared-input-staging optimization did not demonstrate a full-model benefit.

Current lowering issues `cp.async`, immediately waits for all groups, computes,
then retires shared readers. Thus it has asynchronous instructions but no
inter-chunk copy/compute overlap. The local vLLM Marlin implementation separates
shared-memory fetch, completion, register fetch and matrix work, maintaining
multiple live stages. This motivates a measured experiment, not a claim that
Marlin is the exact selected vLLM kernel for this checkpoint.

At most two candidate schedules: two-slot independent staging, and two-slot
shared-left staging. An immutable grouped-matrix plan selects stage count;
physical storage/lifetime planning defines disjoint slots; private CUDA lowering
publishes the next chunk before consuming the current chunk. Ascending K order,
MMA arithmetic, scales, activation quantization and output publication remain
unchanged. Test boundary chunks, stage retirement, exact outputs, sanitizers,
hardware counters and clean full-model timing. Keep comparison outside the
compiler and preserve losing results. Do not select an unmeasured default.

## Loading schedules rejected

Full-width synthetic probes now allocate 288 distinct weight banks, rather
than mapping 64 logical experts onto eight repeated banks. They still are not
captured checkpoint operands. The two-stage independent candidate preserved
all output/workspace bits, but R22 chain median only fell from 7.791 to 7.611 ms.
Its NCU gate/up capture shows 96.53M to 125.83M warp instructions, shared memory
14,848 to 28,672 bytes, active warps 45.66% to 23.94%, load-wait percentage
75.46% to 64.67%, and barrier percentage 19.52% to 25.00%. Counter percentages
are not additive wall-time contributions. Requested DRAM-byte metrics were
not emitted by this device/report; no bandwidth claim is inferred.

Reusing the sequential gate/up scratch restored the intended footprint, but
R22 chain median was 7.749 versus 7.783 ms (less than 0.5%). Two-stage paired
staging likewise had no useful benefit. All retained probe roots start with
`/tmp/lunaflux-glm-fp4-performance-two-stage-20261017-` or
`/tmp/lunaflux-glm-fp4-performance-two-stage-paired-20261017-v1`.
These experimental source changes were removed from the repository; their
generated sources, negative measurements and counters remain in those roots.

## Stable grouping fix

The same expert chain's grouping kernel used one thread to perform repeated
route searches (15.896 ms summed over the retained profiled prefill). The new
pure `StableAssignmentPlan` factors first-occurrence order into independent
leader/rank calculations, one ordered integer prefix, and uniquely owned
publication. CUDA lowering alone chooses bounded shared storage and thread
mapping. Arithmetic, expert order, within-expert order and padded metadata are
unchanged; large capacities retain the previous implementation.

`stable-ranks-20261017-v1` R22 chain median is 6.771 versus 7.793 ms (13.1%
less component time), with bitwise outputs/workspaces. The v2 harness also
compares the entire padded metadata area. Its separate integer oracle covers
35 combinations of empty, invalid, singleton, boundary and capacity rows,
with identical, distinct and mixed expert patterns. All 35 pass plain,
memcheck/full leak, racecheck and synccheck runs under 2 GiB/no swap.
Full-model benefit is still to be measured; this is not a 13.1% engine claim.

The fresh retained-engine cohort `/tmp/lunaflux-model-bench-glm-baseline-20261017-v1`
completes the six measured prompts in 4.685 s, all prior native output IDs
unchanged. Its selected source and memory observations are retained. The
temporary original GLM containers remain stopped during the bounded retest.
