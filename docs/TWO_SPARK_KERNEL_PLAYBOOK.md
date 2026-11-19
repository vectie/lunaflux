# Two-Spark kernel, collective and loading playbook

Written 2026-10-24; evidence through 2026-10-22. This is the implementation
companion to the [performance engineering guide](TWO_SPARK_PERFORMANCE_GUIDE.md)
and [history / latest measurements](TWO_SPARK_PERFORMANCE_HISTORY.md), not a new
hardware campaign. Read the guide's numerical, ownership and memory contracts
before applying these recipes. No candidate below is universally optimal.

## 1. Learn from vLLM/SGLang with source, selected kernels and counters

Follow a concrete chain: installed model adapter → selected backend/dispatcher
→ numerical/layout contract → launch/workspace → actual symbol and SASS
→ measured counters → controlled source change → unprofiled timing.

Use `../vlm` to understand implementation and locate candidate differences,
then check whether the installed build executes them. Trace the entire relevant
chain, including normalization, rotary, cache writes, packing, reductions and
postprocessing. Do not compare our fused chain with only their GEMM and call
the instruction ratio redundant work. For proprietary leaves such as selected
nvjet kernels, identify the closed-source boundary: SASS, launch metadata,
operands and counters are evidence; unobserved source internals are not.

The source/counter checklist is:

| Dimension | What to compare |
| --- | --- |
| Useful work | Actual rows, selected keys, experts/tokens, tails, repeated scoring, padding |
| Decomposition | CTA tile versus MMA microtile, grids, persistent workers, row/expert packing |
| Data movement | Requested copies, cache sectors, shared transactions, scales, layout conversion |
| Instructions | MMA, LSU, integer/address, MOV/shuffle, scalar math, control/wait polling |
| Pipeline | Copy lead distance, stages, fragment reuse, load/MMA overlap, publication/retirement |
| Resources | Registers, shared bytes, spills, resident blocks, useful grid waves |
| Progress | Eligible/issued warps, tensor activity, long/short scoreboard, barrier waits |
| System path | Launch cadence, graph cuts, host gaps, TP dtype/volume, rank arrival, proxy queues |

Record metric names, units, aggregation and replay passes. Warp instructions
are not thread instructions. A stall percentage is a share of observed warp
latency, not a wall-time fraction to multiply by decoder duration. A high cache
hit rate does not erase addressing/copy instructions. L2 sectors are not DRAM
bytes; missing DRAM counters mean no supported DRAM-saturation claim.

Build a causal test: change one mechanism while retaining work and numerical
contract; predict which counters should change, what could regress, and the
acceptance boundary. If counters improve but time does not, stop calling that
mechanism the explanation of the whole gap. Revisit occupancy, independent
work, dependency length, issue pressure and interactions using new evidence.

## 2. Kernel and execution recipes from the history

### 2.1 Projections: reuse without starving the machine

Separate instruction microtile, CTA tile, epilogue ownership and full fusion
region. Full QKV fusion once forced 16 token rows × one head per CTA: 4,096
CTAs versus 256 for the selected reference projection. More stages could not
escape that decomposition. Compare full fusion against a larger GEMM plus
partial epilogue, including all normalization/RoPE/cache-write work.

Hoist genuinely invariant addresses and metadata at their correct lifetime;
cache/reuse positional data only under its numeric contract. Stage coalesced
operands, choose a bank-compatible swizzle, reuse loaded fragments, and overlap
future operand loads with independent MMA work. Retire readers before storage
reuse. A barrier may implement a real ownership transfer; delete redundant
scope, not a required happens-before edge.

Increasing row/column reuse reduces traffic but can raise registers/shared
memory, reduce useful CTAs and lengthen dependencies. Evaluate all affected
shapes, especially decode and partial prefill. Producer/consumer specialization,
bulk transfer, TMA, async copy, or elected waits are mechanisms—not guarantees.
The current evidence includes rejected invariant-address, wait-policy and wide
fragment variants. Do not reintroduce them as default fixes without a changed
mechanism and new paired evidence.

A useful projection worksheet, before writing another candidate:

- Record live `M,N,K`, storage/accumulator/output types and the exact reduction
  order. Useful dense arithmetic is `2*M*N*K` under the usual FLOP convention;
  separately count padded/masked MMA work and repeated epilogue work.
- For BF16 operands and complete unfused tiles, an illustrative no-cache copy
  model is `2*K*(M*ceil(N/tile_N) + N*ceil(M/tile_M))` bytes. Add tails/scales and
  actual reloads where applicable. This is requested payload, not DRAM traffic.
- Compute useful tile count separately from capacity-grid blocks. Record
  inactive CTAs and scheduling/swizzle behavior; a 256-block launch alone does
  not reveal matrix orientation or split-K.
- Bound residency by registers, shared memory, threads and device block limits;
  then check whether the useful grid supplies enough independent work. Raising
  reuse can reduce both resident blocks and the number of useful tiles.
- Count global-to-shared epochs, shared-to-register loads, MMA instructions,
  layout moves, publications, retirement waits and epilogue stores. An invariant
  pointer expression only helps if compiled SASS actually removes repeated work.

The logical pipeline must distinguish two independent events: **data ready for
consumers** and **all readers finished before overwrite**. Double buffering
permits loading a future tile while consuming another, but only after those
slot-specific dependencies are satisfied. Fragment lookahead is a second level
of overlap between register loads and MMA; it is not implied by having two shared
buffers. Use full/tail/wraparound sanitizer fixtures to test both levels. The
query-source history found a real write-after-read hazard from reused scratch;
its reader-completion barrier was necessary correctness, not avoidable overhead.

### 2.2 Attention and indexing: remove repeated work before tuning stalls

For dense attention, examine blockwise softmax/QK/PV, GQA reuse, history-aware
tiling, query/fragment ownership, async staging and split decode where legal.
FlashAttention is a family of algorithms and schedules, not a checkbox or one
vendor library. Prefill and decode have different parallelism/resource limits.
KV128 can amortize loop overhead yet lose to KV64 through resource pressure.

For GLM sparse attention, separately measure pooling, index scores, selection,
membership/metadata, selected-key attention and projections. The history found
single-thread repeated top-k scanning and quadratic duplicate checks. Reuse
completed pooled keys with page-generation ownership; bypass score computation
when all candidates are selected; preserve stable ties and causal validity.
Choose bounded parallel selection rather than replacing one unbounded scan
with a quadratic method for all domains.

Cooperative contiguous key copies and matrix-based scoring can reduce support
work. Quantized E4M3 index scoring is an explicit numerical alternative, not a
bitwise rewrite. Compressed/latent cache storage must reach writers and readers
and preserve normalization/scale semantics; its extra query absorption/value
expansion may outweigh memory savings. The tested latent-cache service lost
performance and was not the later selected cache path.

### 2.3 KDA and recurrent models: optimize the actual dominant phase

Distinguish KDA projections, convolution/gating, recurrent updates and output
publication. An early GLM capture spent about 40% in KDA projections and only
0.105% in recurrence: replacing recurrence with chunk KDA was then the wrong
priority. Chunk algebra also has distinct numerical and persistent-state rules.

Later chunk KDA remained excluded from the 8K/eight-slot HTTP envelope because
its plan reserved about 2.71 GB extra across 34 layers. The current formulation
reserved full row workspace per slot although only 64 total rows were live.
Packed per-sequence allocation and cross-layer transient reuse are physical
plan opportunities, not already-qualified fixes. Do not enable the candidate
merely because its individual GPU kernel works.

### 2.4 MoE: packed useful rows, native low-bit work and one TP boundary

Measure expert occupancy by live rows, not maximum batch capacity. Pack token
assignments by expert; preserve deterministic assignment, route weights,
inverse mapping, scales and inactive-row behavior. Choose small-row and prefill
paths separately. A singleton register-only path removed barriers yet became
slower from additional instructions/load dependencies.

Keep low-bit payload/scales native where qualified; avoid re-decoding a whole
expert bank repeatedly. Grouped/persistent execution and separate producers
and consumers amortize weight/metadata work but require correct readiness and
reuse epochs. Gate/up and down can have different optimal schedules.

“Supports FP8/FP4” is not an execution description. Record weight storage,
activation representation, scale granularity, MMA operand format, accumulation
and output publication separately. A low-bit checkpoint decoded to BF16 for
each dot is not the same path as native low-bit matrix execution. INT4, FP4 and
their block-scaled variants are not interchangeable by renaming a dtype.

Run shared and routed branches concurrently only with disjoint writes and a
join before consumption. Measure union time **and contention**: an overlapped
shared branch can lengthen routed work. Compose the actual graph rather than
adding isolated gains. Combine rank-local routed/shared contributions before
one MoE reduction where the declared numerical program permits it. This removed
redundant collective boundaries; it did not eliminate all TP or all MoE cost.

### 2.5 HC, small kernels, scale scans and vocabulary

Parallelize reductions and row/stream ownership; avoid one-thread HC and
single-workgroup scans over all activation-scale work. Fuse compatible HC
post/pre boundaries only when rounding, state/effects and live-row semantics
allow it. Fusion may help one row and hurt 22 rows. Distributed scales need
rank-local pre-TP captures and full-state checks if instability appears.

Shard vocabulary and non-expert projections across TP where the plan owns the
layout and communication. Count casts, scratch and gathers, not only GEMM.
Do not move a GPU computation to CPU as an accidental correctness workaround
or call a leaf-level speedup a full-service result.

### 2.6 TP submission and CUDA graphs

Eliminate whole-stream waits when a narrower event/ownership dependency is
sufficient. Preserve cancellation, buffer leases, KV generation and final
completion of every lane. Graph replay reduces submission overhead, not math,
and can move a bottleneck into NCCL proxy progress or capture memory.

The demonstrated cold NCCL stall was synchronous ingestion of many graph-posted
operations on the progress thread. Bounded collective graph windows reduced
that burst without changing the reduction or adding intermediate host waits.
The selected horizon16 is a measured resource choice, not an NCCL constant in
model semantics. Decode, exact prefill and unmatched tails need distinct legal
replay behavior; padded communication must not quietly include inactive rows.

For BF16 reduction-after-local-rounding versus F32-reduce-then-round, explicitly
declare the numerical difference. Recheck answers, state and memory. Neither
BF16 transport nor two GPUs inherently gives 2× speed: replicated work,
arrival skew, serialization and communication must be measured.

## 3. Diagnose loading and warmup without changing the benchmark's meaning

Keep loading observers and reference overlays outside the compiler. Record
phase entry/exit, per-rank process trees, tensor/allocator ownership and actual
allocation stacks. Distinguish live model bytes, inactive caching allocations,
file mappings, CPU anonymous copies, pinned buffers, graph pools and profile
workspaces. “The last log says warmup” is not an allocation diagnosis.

The October21 owned-reader fix roughly halved each worker's anonymous PSS while
leaving GPU model allocation unchanged. It did not establish readiness. The
October22 dummy-profile fix removed exactly 2,181,038,080 bytes: a synthetic
dense-MHA workspace allocated by a selected sparse backend that explicitly did
not support dense-MHA prefill. Guarding by that actual capability preserved the
supported execution and memory forecast. A subsequent initialization-only cache
release reclaimed another inactive 512 MiB, not live model tensors.

Both are real memory fixes; complete reference startup still stopped at the
conservative guard before graph capture. Do not disable profiling, hide peaks,
reduce capacities silently, use swap, or report a successful load as a completed
request. Initialization-only cache release is not a new token-step operation.
Use a finite retry budget; after a new failure, isolate the new allocation
before repeating another full model load.

Label a reference run with any installed-source overlay explicitly: it is not
an unmodified framework run. Preserve original source and changed lines,
capabilities, capacities and outputs; confirm the overlay affects only the
intended startup behavior before using it for a performance comparison.

## 4. Define the workload before comparing frameworks

Maintain two evaluation tracks: **fixed-work performance** and **complete task
answers**. Fixed output lengths isolate execution but may intentionally truncate
meaning; natural stops assess answers but generate unequal work. Neither track
can substitute for the other.

Every comparison records:

- Checkpoint, actual quantization/scales, tokenizer/template and model revision.
- Prompt token IDs/counts, output budget and actual counts, EOS/stop behavior.
- TP/PP placement, chunk rows, batch/concurrency, live/history lengths, page
  layout, KV capacity, prefix-cache hits and speculative decoding settings.
- Selected kernels/numeric laws, executable/AOT labels, installed framework
  source/build, image tag and complete effective arguments.
- Machine/driver/toolchain, clocks/power/thermal state, memory envelope and
  other owners; loading, cold, warm and cached phases explicitly labelled.

Image names are not checkpoint-format evidence. A local `../vlm` checkout is
not necessarily the installed vLLM build. A Spark template for one model or
architecture is not evidence that another image implements it. Missing SGLang
support, a failed pull, loading abort or SSH loss is a blocker/result—not a
missing timing cell to fill with an estimate.

Cover short and long prompts, prefill and decode, C1 and admitted concurrency,
mixed lengths, fragmented/recycled pages, tails and cancellation. Useful
projection live-row controls in this history include 0/1/3/22/36/64/65; long
service controls used actual counts 165/2085/6181/7973. These are examples,
not universal benchmark definitions. Vary content, selected expert occupancy,
route skew, history, generated length and request scheduling independently.

For fair implementation attribution, match capacities and scheduling where
possible. For native-engine strategy comparison, allow native policies but
report their differences. The October4 SGLang 8192-token chunk policy differed
from LunaFlux/vLLM 2048 chunks; that was not an identical-scheduler comparison.
Do not silently lower context, slots or rows to make a reference load, then
call the result matched.

Report TTFT, completion, output throughput, inter-token distributions, queueing,
failure counts and memory—not one aggregate tok/s. Token-arrival intervals
include transport/scheduling and are not isolated GPU decode times. Freeze
candidate selection before holdout validation; use alternating/counterbalanced
unprofiled trials, preserve individual pairs and variability, and distinguish
warmup from repetitions. A gain smaller than run variation remains inconclusive.

When the requested scope is tracing, **do not replace it with an overall
benchmark**. Collect the exact diagnostic work and graph needed to answer the
question, retaining existing answer qualification and separate restoration tests.

## 5. Practical runbook and existing tools

Do not copy historical run roots, credentials or retired unit names into a new
deployment. Inspect each `.mbtx` entry's current arguments first. These are
reusable tool locations, not a command to launch every campaign:

| Task | Entry points |
| --- | --- |
| Capacity and service preparation | [`plan-glm-service-capacity.mbtx`](../scripts/plan-glm-service-capacity.mbtx), [`prepare-glm-http-spark.mbtx`](../scripts/prepare-glm-http-spark.mbtx) |
| Installed reference provenance | [`collect-glm-selected-vllm-source.mbtx`](../scripts/collect-glm-selected-vllm-source.mbtx), [`capture-glm-template-run.mbtx`](../scripts/capture-glm-template-run.mbtx) |
| Bounded native/reference windows | [`trace-glm-http-window.mbtx`](../scripts/trace-glm-http-window.mbtx), [`trace-glm-reference-window.mbtx`](../scripts/trace-glm-reference-window.mbtx), [`trace-glm-activity.mbtx`](../scripts/trace-glm-activity.mbtx) |
| Wall graph and paired TP | [`analyze-glm-execution-graph.mbtx`](../scripts/analyze-glm-execution-graph.mbtx), [`analyze-glm-paired-tp.mbtx`](../scripts/analyze-glm-paired-tp.mbtx), [`analyze-glm-graph-cadence.mbtx`](../scripts/analyze-glm-graph-cadence.mbtx) |
| Projection and expert leaves | [`probe-glm-installed-projection.mbtx`](../scripts/probe-glm-installed-projection.mbtx), [`probe-glm-selected-projection.mbtx`](../scripts/probe-glm-selected-projection.mbtx), [`probe-glm-grouped-pair.mbtx`](../scripts/probe-glm-grouped-pair.mbtx) |
| Counters and memory guard | [`summarize-glm-projection-counters.mbtx`](../scripts/summarize-glm-projection-counters.mbtx), [`guard-glm-trace-memory.mbtx`](../scripts/guard-glm-trace-memory.mbtx) |
| Loading/profile diagnosis | [`prepare-glm-loader-observer.mbtx`](../scripts/prepare-glm-loader-observer.mbtx), [`prepare-glm-warmup-observer.mbtx`](../scripts/prepare-glm-warmup-observer.mbtx), [`prepare-glm-dummy-profile-observer.mbtx`](../scripts/prepare-glm-dummy-profile-observer.mbtx) |
| Exact dummy workspace correction | [`prepare-glm-dummy-mla-workspace.mbtx`](../scripts/prepare-glm-dummy-mla-workspace.mbtx) |
| External task and distribution checks | [`benchmark-glm-broad-http.mbtx`](../scripts/benchmark-glm-broad-http.mbtx), [`assess-glm-complete-answers.mbtx`](../scripts/assess-glm-complete-answers.mbtx), [`compare-teacher-forced-logits.mbtx`](../scripts/compare-teacher-forced-logits.mbtx) |
| Lifecycle qualification | [`validate-glm-serving-candidate.mbtx`](../scripts/validate-glm-serving-candidate.mbtx), [`validate-glm-http-startup-stop.mbtx`](../scripts/validate-glm-http-startup-stop.mbtx) |

Preflight: inspect git/dirty state; read the latest report; identify both GPU and
controller owners; establish available memory/no swap; inspect disk capacity;
resolve exact service stop/restore targets; confirm toolchain and installed
reference capability; create a fresh bounded run plan. Do not infer idle from
one empty `nvidia-smi` sample while a chain controller still owns the pair.

During execution: stop the owned pair on rank failure or reserve breach; retain
logs and explicit requested-stop intent so intentional drain is not classified
as a crash. Diagnose a failed SSH connection before launching more work. Keep
actual request completion and observer completion separate. Never erase a failed
result because a later runner correction explains it.

Classify failures from evidence: a requested container retirement can produce
exit137 without OOM; SIGABRT can be a runtime invariant failure rather than
memory pressure. The idle socket-coroutine churn and active-GPU event polling
were different paths in this history. A bounded reproducer or short wakeup test
does not justify claiming a natural-abort stack or a multi-hour stability soak.

At a completed code phase, the repository requires `moon info`, `moon fmt`,
`moon check --target native --deny-warn`, and `moon test --target native --deny-warn`,
plus boundary-appropriate GPU gates. Focused edits use affected checks first.
Report pre-existing migration failures and any existing warning exclusions
literally; an excluded check is not an unexcluded clean-suite pass. Preserve
unrelated dirty toolchain/TLS/Int32 work. Author automation in MoonBit `.mbtx`;
production remains MoonBit-native with private CUDA/NCCL ABI and AOT artifacts.
