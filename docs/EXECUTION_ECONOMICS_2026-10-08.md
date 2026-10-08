# Optimize the legal execution, not just the current kernel

This is the design policy for the next compiler workstream. It extends the
[multilayer compiler](COMPILER_ARCHITECTURE_COMPLETION_2026-09-29.md), rather
than replacing it with another compiler or a request-time tuning system.

## Objective and evidence

The objective is the cheapest **legal complete execution** of an admitted
workload. A faster isolated instruction, fewer bank conflicts, more occupancy,
more fusion, or more IR layers is not that objective.

The [2026-10-08 comparison](BENCHMARK_GAP_REPAIR_2026-10-08.md) measured
8,192-input/64-output/C8 completion at 4,692.5 ms for LunaFlux, 4,496 ms for
vLLM, and 4,551 ms for SGLang. Closing the roughly 200 ms gap establishes
parity, not a 20% lead. At fixed output work, 20% higher throughput than vLLM
requires at most 3,746.7 ms: approximately 945.8 ms less than that LunaFlux
run. A 20% reduction in completion time is a different target: 3,596.8 ms.

A nearby, separately profiled control attributed about 62% of GPU-plus-gap
time to attention. Its roughly 129 ms of GPU gaps cannot explain a potential
946 ms improvement. These observations prioritize experiments; they are not
independent causal savings that can be added. The measured runtime includes
opt-in reference attention and vendor projection routes, not an all-default,
all-compiler-kernel claim. The comparison fixes token work; it does not prove
cross-framework output-quality parity.

## Three kinds of constraint

1. **Semantic obligations:** observable values, numerical policy, causal
   visibility, token dependencies, cancellation, KV ownership and retirement.
   An optimization must preserve them.
2. **Hardware constraints:** instruction shapes, memory capacity, bandwidth,
   register/shared-memory limits and supported synchronization. A backend
   describes these; model builders and the scheduler do not.
3. **Implementation choices:** fusion cuts, tiles, staging buffers, ownership
   changes, launch boundaries, layout conversions and pipeline depth. These
   are alternatives to search, not requirements to preserve indefinitely.

An existing implementation's reduction order is binding only under its exact
numerical contract. Reassociation or reduced precision requires a separately
admitted policy and accuracy tests; it is not a layout-only transformation.

## Ask these questions in order

| Question | Compiler responsibility | Required evidence |
| --- | --- | --- |
| Must this work happen? | Demand propagation, live physical domains, invariant reuse | Same required outputs and effects; no omitted live rows |
| Must this data move? | Producer/consumer ownership, retained values, common layouts | Actual traffic and conversion instructions, not source-level counts |
| Must this wait happen? | Explicit data/effect dependencies and lifetimes | Execution timeline, safe cancellation and ordered KV retirement |
| Is this decomposition appropriate? | Fusion cuts, history partitions, matrix/SIMT alternatives | Complete-chain latency, resource feasibility, merge/padding cost |
| Can the remaining instructions run better? | Backend pipeline and instruction selection | Selected-kernel counters plus unprofiled paired timing |

These are priorities, not permission to delete state or overlap dependent
work. Recomputing a cheap pure value can be preferable to retaining it; fusion
can lose by restricting GEMM geometry; asynchronous copy can lose by increasing
address work. The compiler must be able to represent both choices.

## Architecture: preserve alternatives until their context is known

```text
Semantic graph + explicit numerical policy
  -> demand/effect analysis
  -> legal algorithm and fusion alternatives
  -> retained physical ownership/storage/effect plans
  -> supported backend realizations and resource facts
  -> complete-region observations / explicitly labelled estimates
  -> continuation-aware nondominated frontier
  -> workload + memory + consumer-compatible selection
  -> existing AOT artifact and prepared-executor binding
```

The arrows describe information flow, not a requirement that each box become
a new package. Existing domain dialects remain distinct: projection is not
online softmax, and token publication is not a matrix epilogue. Feedback from
compilation and measurement supplies new immutable inputs to another selection;
it does not mutate semantic meaning or introduce ambient global state.

A locally slower producer must survive if it exposes a better consumer layout,
retains fewer bytes, or enables a cheaper legal continuation. Compare complete
chains whenever cache state, launch composition or overlap changes. Do not
sum isolated medians and label the result a whole-chain measurement.

Only alternatives with the same continuation contract may dominate each other.
That contract includes the complete live-value layout and ownership handoff,
numerical policy, observable/effect obligations and cost-relevant continuation
context, including cache/residency and launch assumptions. Layout equality
alone does not prove that two producers leave an equally cheap continuation.
When that context is unknown, keep separate classes or measure the whole chain;
do not prune using an unjustified equivalence. It is not merely a dtype
or a tensor shape. A caller-owned class ID is scoped to one comparison, never
a globally meaningful ABI identifier or a source of execution authority.

Within a class, retain latency/workspace/local-storage tradeoffs until the
consumer and resource envelope are known. An alternative can be discarded
only when another is no worse in every retained dimension, with deterministic
ID tie-breaking. Unknown facts must not be represented as zero cost. Peak
reusable scratch is distinct from persistent activation storage, which remains
the memory planner's responsibility.

## Functional programming boundary

Planning is a pure transformation of immutable values. Mutation confined to a
local construction buffer does not create observable shared compiler state.
Candidate enumeration order must not affect the result. Effect plans retain
KV writes, publication, submission, completion and release explicitly. Device
lowering may use mutable storage and synchronization to realize those plans.

Purity does not require serial execution, and more compiler levels do not
automatically discover latency hiding. The compiler must have executable
alternatives, consume its own plans during lowering, and select the right
artifact. NVIDIA instructions and library calls remain backend details.

Production stays AOT and content-addressed. Offline exploration may be broad;
startup selects admitted implementations; token steps use preallocated state
and prepared dispatch. No JIT, profiling, tuning search, filesystem validation,
or benchmark-record parsing enters the request path.

## Implementation ledger

This increment separates whole-partition frontier construction from final
selection in `compiler/fusion_regions`. The existing ingress selection path
uses the same implementation through `select_partition`; it is not a second,
unused selector. Existing materialized ingress boundaries occupy continuation
class `Materialized`. New callers can retain different `Scoped` classes and
resolve them only after the consumer contract is known. Resource limits remain
a final query, not an early reason to permanently discard a useful alternative.

The connected path is the existing runtime-bundle exporter's
`prepare_ingress_choice` → `select_attention_ingress_cut` → `select_partition`
→ frontier construction/selection → `bind_ingress_choice` → emitted module
selection. No token-step code or kernel renderer changes in this increment.

This is selection architecture, **not a new kernel or a measured speedup**.
Existing scalar-cost callers keep their selected IDs. It does not invent new
layouts, relax accuracy, enable overlap, or automatically connect all compiler
families to joint selection.

Next executable work, in priority order:

1. Expose complete decode alternatives jointly over history partition, grouped
   query reuse, accumulator ownership, matrix/SIMT realization and merge cost.
   These features already exist individually; their joint selection is the task.
2. Compare ingress plus attention plus output-projection continuations, retaining
   compatible ownership/layout alternatives rather than independent winners.
3. Represent necessary submission/publication dependencies separately from
   incidental host sequencing. Evaluate real batch overlap, not merely moving
   completion metadata work earlier.
4. Feed exact workload vectors and whole-region measurements into those
   choices. Include row/history diversity, cache state, padding, mixed steps,
   concurrency and output length; do not optimize one uniform benchmark only.

Each step is complete only when its prepared artifact is actually selected and
its complete chain passes correctness and physical measurement. A search
frontier is not proof that its alternatives have been implemented or measured.

## Joint decode selection upgrade

The next increment connects a joint search to the existing offline candidate
exporter. The portable compiler enumerates the explicit candidate/workgroup
target cross product instead of retaining only one independently ranked kernel
per target. The retained candidates carry grouped-query sharing, score/accumulator
ownership, SIMT or matrix realization, KV tile and pipeline depth. The optimizer
derives legal partition counts; the CUDA adapter emits both partial and merge
entry points and their workspace. Existing numerical permissions remain binding.

Whole-chain selection uses `compiler/fusion_regions`, not another cost model.
It compares complete partial-plus-merge measurements for the same compilation,
device, toolchain/flags and workload; scratch and local-storage budgets remain
final selection constraints. A measured winner returns its original AOT value,
including source bytes, symbols and launch geometry. Ordinary single-kernel
autotune records cannot discard alternatives in this search.

The Qwen candidate exporter accepts an optional search suffix:

```text
--decode-chain-search ROWS HISTORY CANDIDATE_IDS WORKGROUP_TARGETS
--decode-chain-tuning ABSOLUTE_FILE FILE_SHA256 DEVICE_ID
```

The lists are comma-separated. For a 16-query-head/4-KV-head shape at C8,
`8 4096 452,480,482 64,128,256` explores three existing SIMT/matrix schedules
at workgroup targets yielding 2, 4 and 8 partitions. This is an example search
domain, not a recommended performance policy. It opts into the compiler's
existing alternative-softmax numerical domain for offline evaluation; it does
not prove that those laws meet a model's production accuracy requirements.

Each `decode-chain-vN` directory contains a real `kernel.cu` and `kernel.recipe`.
The recipe records the exact compiler flags, source/compilation identity,
workspace and distinct partial/merge grids. Grouped partial work launches over
KV heads; merge work launches over query heads. A timing input additionally
publishes the selected source pair to `decode-chain-selected`, without replacing
`reusable-qwen-decode-attention`. Outputs use the existing non-overwriting writer.

The timing file has tab-separated fields and a final newline:

```text
luna-decode-chain-timing-v1
scope FRONTIER_SHA256 DEVICE_ID TOOLCHAIN_SHA256 COMPILER_FLAGS standalone-partial-merge-v1
workload decode ROWS HISTORY
chain COMPILATION_SHA256 WHOLE_CHAIN_NS SAMPLES
```

Use literal tabs, not the spaces shown above. `COMPILER_FLAGS` is the exact
`chain_compiler_flags` recipe value. All observations in one comparison must
use the same inputs, cache/warmup protocol and unprofiled timing boundary;
the schema checks identity, not the honesty of a supplied measurement. Scope
does not transfer these standalone timings to captured mixed-serving graphs.
Unknown compilations, duplicate records, mismatched workloads/flags and fewer
than three samples are rejected. Long-chain latency uses checked Int64 values.

This increment completes offline enumeration, whole-chain selection and artifact
export. It does **not** change a serving default, add new GPU instructions or
claim a speedup. Single-launch versus partitioned execution, ragged/history
vectors, broader explicit partition counts beyond the existing optimizer's
legal envelope, and automatic binding of measured winners to serving buckets
remain separate work. Joint selection is available for physical experiments;
priority 1 is not yet physically complete under the criterion above.

## Experiment discipline

- Name one falsifiable hypothesis and estimate its maximum end-to-end impact
  before changing the compiler. Stop when that ceiling cannot meet the goal.
- Hold model, numerical policy, token work, GPU, memory reserve, toolchain,
  selected artifacts and timing boundaries fixed. Keep frozen baselines.
- Use counters to distinguish redundant work, latency, bandwidth, conversion,
  synchronization and host gaps. Counters explain candidates; completion time
  chooses the winner.
- Alternate paired unprofiled trials, report variability and losing shapes,
  and verify selected symbols/owners. A profiler replay is not serving time.
- Check accuracy, sanitizer and deterministic release at the changed boundary.
  Pure selection tests need not rerun an unrelated 24-hour lifecycle soak.
- Preserve rejected experiments. Repeated non-wins require a new hypothesis,
  not a longer sequence of unmeasured compiler changes.

Competitors are controls and sources of executable alternatives, not proof that
their decomposition is optimal for every target. A 20% lead remains a measured
workload-specific goal, not an architectural entitlement.

## Validation of the continuation frontier

- The pure fusion-region package passes 13/13 native tests with warnings denied
  and no warning exclusions. Coverage includes an exhaustive oracle over 60
  candidates and 60 continuation/resource-envelope combinations, stable ties,
  permutation invariance, immutable reuse, foreign graphs, mixed cost bases,
  duplicate identities and unavailable consumer contracts.
- The affected fusion strategy, semantic cut and real runtime-bundle exporter
  integration passes 40/40 native tests. It retains the existing full/partial/
  unfused selection and emitted module contracts, with existing migration
  warning exclusions `-79-20-29-25-92-14`.
- A clean snapshot of `24ef8d5f` plus only this increment passes native checking
  and the full 3,460/3,460 native suite with the same exclusions. Full formatting
  checking and interface generation also complete. The unrestricted repository-wide warning-denied
  check still fails on pre-existing migration warnings; it is not reported as
  warning-clean.
- Formatting and generated interfaces are reviewed. No kernel/native ABI or
  token-step behavior changes; no fresh GPU benchmark, sanitizer result, or
  performance improvement is claimed here.

## Validation of joint decode selection

- The common fusion selector, portable attention compiler, CUDA AOT adapter
  and candidate exporter pass 77/77 focused native tests. Coverage includes
  the joint candidate/partition domain, permutation invariance, resource
  pruning, whole-chain workspace tradeoffs, exact returned artifacts,
  partial/merge launch domains and rejection of incomparable timing records.
- A clean snapshot of `4f00312a` plus only this upgrade passes formatting,
  native checking and the full 3,470/3,470 native suite, with the existing
  migration-warning exclusions `-79-20-29-25-92-14`. Generated interfaces and
  the isolated diff are reviewed; unrelated worktree changes are excluded.
- No GPU campaign ran for this offline selection upgrade. Existing kernel
  renderers, native ABI and serving defaults are unchanged. Physical
  correctness, whole-chain timing and serving-bucket binding remain necessary
  before a measured alternative can replace the serving selection.
