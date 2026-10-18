# GLM prefill projections, distributed activation scales, and serving

Status: implementation and independent measurements in progress. No performance
or serving-cutover claim yet. Baseline is `a8d8fb87`; the prior deferred-HC
candidate did not demonstrate an end-to-end win and is not the new baseline.

## Experiment contract

Keep numerical programs unchanged. Finite nonnegative maxima admit exact
partitioning; floating-point sums and matrix publication retain their declared
laws. Physical IR owns partition coverage, scratch placement and producer/read
lifetimes. CUDA lowering alone owns workgroup and instruction details. No
comparison, profiler or payload hashing enters the compiler/request path.

The retained selected trace has one-workgroup input/activation scale scans and
slow BF16 prefill projections. These are hypotheses for the first-token gap,
not an additive attribution of the entire observed 11.2% completion deficit.

Budget: at most three distributed-maximum geometries and three projection
schedules in the first campaign. Measure complete producer/consumer chains,
decode as well as prefill rows, then separately benchmark each winning change
against the frozen native baseline and matched vLLM prompts. Retain losing
shapes and failures. Actual exported source, launch geometry, workspace owner
and executed symbols must agree before a path is considered selected.

## Maximum-scan design

Keep the two existing ordered launches. The first partitions the finite scan
and publishes private partial maxima beyond the numeric packed-buffer extent.
The pack launch reads all partials and derives the same scalar independently;
only its first lane publishes the scalar for subsequent matrix consumers.
No pack workgroup reads another pack workgroup's write. Scratch remains live
until every reader finishes, and is disjoint from packed payload and scales.
One-row work retains the single-partition arithmetic path. All allocation and
launch decisions are startup plans, with no per-token allocation or host scan.

## Serving boundary

The user now requests LunaFlux as the GLM serving engine, superseding the prior
instruction to restore vLLM afterward. Preserve the existing endpoint/model
contract and original container configuration as a recoverable fallback.
The original EXL3/DFlash service reports an 850000 context limit and exceeds
both the 100 GiB device budget and 16 GiB host reserve; a 64-context NVFP4 CLI
benchmark is not a drop-in replacement or proof of production readiness.
Inspect and test the actual resident HTTP/tokenization/cancellation path before
cutover, and disclose any model-format, context or API difference explicitly.

All GPU work is serialized per Spark, with 100 GiB/process and device ceilings,
zero experiment swap and at least 16 GiB host MemAvailable. Comparison stays
outside the compiler; no tolerance, reference or checkpoint changes are allowed.

## Isolated physical results (not end-to-end results)

Actual generated FP4 quantization at 64-row capacity was tested with input width
4096 and eight routed activation rows of width1024 per token. Complete scan+pack
chains use seven alternating CUDA-graph trials, not scan-only timing. Candidate
32 partitions retains two launches and adds128 scratch bytes per operand.

| Live tokens | Input chain old/new µs | Routed activation chain old/new µs |
|---|---:|---:|
|1|5.024 / 5.165|8.564 / 8.708|
|3|9.955 / 2.981|18.740 / 4.399|
|22|57.746 / 5.664|113.704 / 9.218|
|36|92.522 / 7.367|183.135 / 12.579|
|64|163.361 / 11.146|324.755 / 19.976|

Root: `/tmp/lunaflux-glm-distributed-scales-selected-20261018-v1`.
Numeric packed bytes/error results match, including zero/inactive/nonfinite and
capacity-boundary cases. Full leak/memory and race/synchronization checks pass
for32 partitions at rows1/3. This does not claim all64 rows under racecheck.
Sixteen partitions were slower in prefill;64 improve some prefill cases but
increase decode cost. Decode overhead remains visible, not waived.

Sparse projections had not forwarded matrix staging, so they always selected
synchronous copies even when recurrent projections were buffered. Independent
staging now reaches the sparse owner, all eight leaf projections and their
actual launch metadata. Same F32 outputs were checked across four projection
shapes and rows0/1/3/22/36/64/65.

| Projection N×K,22 rows | Sync µs | Buffered K32 µs | K64 µs | K128 µs |
|---|---:|---:|---:|---:|
|4096×4096|214.564|186.719|139.232|164.778|
|4096×8192|440.167|417.017|308.252|437.279|
|1536×4096|94.989|40.427|40.562|36.522|
|128×4096|93.776|40.430|41.010|36.549|

Root: `/tmp/lunaflux-glm-prefill-pipelines-20261018-v1`. Larger K is not a
universal improvement:4096×8192 at64 rows rises from431.951 µs(K32) to632.288
(K64) and884.551(K128). Resource/occupancy attribution requires counters;
timings alone do not prove the cause. First full-model ablation therefore uses
K32 for sparse projections and leaves the already-buffered recurrent schedule
unchanged. K128 memory/leak/race/sync checks pass; K32 is separately gated.

The user explicitly approved a limited initial LunaFlux service (NVFP4,C1,
greedy, tested bounded context). The original EXL3 containers are stopped and
retained. HTTP implementation and full-model ablations remain pending; this is
not a serving-success claim.
