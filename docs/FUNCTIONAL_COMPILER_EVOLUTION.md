# Functional compiler evolution

Recorded direction: 2026-09-14. This document records the agreed architecture
and implementation sequence; it does not claim these capabilities already
exist or authorize a production JIT. Implementation status remains in
[FUNCTIONAL_COMPILER_COMPLETION.md](FUNCTIONAL_COMPILER_COMPLETION.md).

## Objective

Build a sufficiently layered, composable, incrementally recompilable functional
compiler. Compiler size and pass count are not goals. Each pass must have a
defined input/output contract, an identifiable benefit, and a correctness
argument. More passes do not automatically produce faster kernels.

The immediate direction is **AOT + real-time plan selection + offline/background
incremental tuning**. Optional JIT is a later, explicit product-boundary decision.

## Layers and responsibilities

| Layer | Responsibility |
| --- | --- |
| Model semantic IR | Mathematical operations independent of devices and scheduling |
| Numerical and effect IR | Precision, reduction order, KV reads/writes, aliasing and lifetimes |
| Graph optimization | Fusion, common-subexpression elimination, layout propagation and redundant-work removal |
| Tile algorithm IR | Tiling, online softmax, split-K and data reuse |
| Schedule IR | Tile choices, work distribution, pipeline stages and register ownership |
| Storage and synchronization optimization | Address hoisting, buffer reuse and proven barrier merging/elimination |
| Device lowering | CUDA/PTX or other backend instructions and device resource constraints |
| Measurement feedback and selection | Calibrated cost models and workload-specific plan selection |

These are logical responsibilities, not a requirement for eight new packages
or one rigid pass order. Passes can repeat where dependencies require it, with
bounded iteration and observable convergence.

## Functional compilation contract

- Each transformation consumes immutable IR and returns new IR. Stable inputs
  must produce reproducible plans; measurement records are explicit inputs.
- Represent effects, memory ownership, cross-thread dependencies and numerical
  constraints explicitly. Generated mutable memory and asynchronous operations
  do not violate this functional compiler contract.
- Cache intermediate results by all relevant semantic, numerical, target and
  compiler inputs. Incremental compilation invalidates affected dependents,
  not unrelated plans.
- Retain explanations of selection and rejected alternatives outside the hot
  path. Do not collect or render compiler diagnostics per token.
- Generic CSE cannot prove a CTA barrier redundant by itself. Barrier removal
  requires producer/consumer readiness and buffer-reuse dependencies to remain
  satisfied, including tails and inactive workers.
- NVIDIA-specific instruction and resource details remain in backend capability
  descriptions and lowering, not model builders, scheduler or generic semantics.

## Replace magic-number policy, not merely rename constants

Distinguish three classes of numbers:

1. Hardware constraints: instruction shapes, alignment and resource limits.
   Obtain these from explicit backend capabilities with documented units.
2. Search parameters: tile extents, stages, transfer windows and work partitions.
   Generate legal combinations from constraints; do not permanently equate a
   hand-maintained candidate ID with an optimization policy.
3. Cost estimates: launch, transfer, synchronization and arithmetic weights.
   Decompose and calibrate them against device measurements. Clearly label
   uncalibrated estimates rather than presenting them as measured cycles.

The intended selection chain is:

**constraints → legal schedules → resource filtering → shape-specific
measurements → lowest-cost numerically acceptable plan**.

Missing observations must have an explicit fallback. Occupancy, register count,
bank-conflict count and bytes moved are explanatory metrics, not independent
optimization objectives. A smaller KV tile can reduce register pressure while
increasing fold iterations, address work and barriers; select it by complete
workload cost, not by one improved counter.

## Meaning of real-time

Real-time **selection** chooses among prepared, immutable executable plans from
actual batch and query/history geometry. Dispatch must be bounded and must not
compile, profile, read tuning files or allocate steady-state scheduling memory.

Background **compilation/tuning** is separate from request execution. It reuses
cached IR, measures alternatives under controlled resource ownership, and
produces new AOT artifacts. Profiling must not silently contend with serving or
invalidate benchmark measurements. Initial publication can remain at startup;
live plan replacement, if added later, needs explicit generation/lifetime
handoff so in-flight work retains its original plan and buffers.

Real-time **machine-code generation** is JIT. It is not part of this initial
direction and does not change the existing compiler-free production request
path or AOT product contract. Any later JIT proposal must separately address
latency, resource contention, artifact publication and predictable fallback.

## Implementation sequence and acceptance

1. Inventory hardcoded choices and distinguish constraints, search domains and
   cost estimates. Preserve current supported shapes and numerical behavior.
2. Establish immutable effect/lifetime and schedule contracts, then move policy
   out of source generators into reusable compiler transformations.
3. Generate the legal schedule space and integrate resource-aware pruning.
4. Calibrate and select using actual query/history/batch distributions, including
   mixed prefill/decode, tails, staggered arrivals and unequal request lengths.
5. Add incremental compilation/cache reuse and bounded runtime plan selection.
6. Evaluate optional background publication and, separately, optional JIT only
   after the preceding architecture is validated.

For each optimization, check numerical behavior, ownership/synchronization,
selected generated instructions and matched timing. Preserve arithmetic order
where the contract requires it; do not relax token agreement simply to admit a
faster candidate. Report kernel and end-to-end improvements separately, along
with compilation cost and tuning coverage. A losing schedule is a useful search
result, not a reason to force it into production.

Current motivation and measurements:

- [Actual-row attention replay](BENCHMARK_ATTENTION_ROW_REPLAY_2026-09-14.md)
- [Logit-margin diagnosis](BENCHMARK_LOGIT_MARGINS_2026-09-14.md)

This record alone changes no runtime implementation, kernel selection or
deployment behavior.

## Implementation follow-up

The initial refactoring now exposes a dimensionless `AttentionCostBreakdown`
from the compiled plan. Its arithmetic, memory, scheduling, fold and transfer
terms sum to the unchanged fallback estimate. A regression checks that the
64-to-32 KV retile doubles fold/transfer terms without changing the modeled
arithmetic or bytes. This makes the policy inspectable; it does **not** claim
that the existing coefficients are calibrated or that candidate generation has
already replaced the hand-maintained frontier.

`AttentionFoldFences` moves query-owned single-slot publication/release fusion
into a generic schedule description. A diagnostic CUDA realization preserved
bitwise outputs and passed bounded sanitizer checks, but did not consistently
improve matched timings. Production lowering therefore retains its prior fence
placement; measured selection between placements remains unimplemented. This
description is not an activated optimization.

The query-owned attention frontier now expands the Cartesian product of the
currently implemented query/KV ownership domain and three transfer schedules,
then applies the existing numeric and resource constraints. This generates 12
combinations rather than seven manually selected combinations. Existing IDs
318–324 retain their exact meanings; five new combinations use disjoint IDs.
The numerical opt-in remains required. This removes the hand-picked subset for
this family, not all compiler-family search tables or backend geometry limits.
Physical compilation and numerical coverage of the expanded frontier must be
verified before treating its additional choices as validated serving paths.

Validation follow-up: all 12 combinations compiled on sm120; the five new
combinations passed independent scalar-reference, deterministic replay and
read-only KV checks for query=2048 with history=0 and history=2048. This is
synthetic-activation kernel coverage, not model token agreement or end-to-end
validation. The KV32 choices differ by up to 0.000488281 from the baseline;
the KV64 choices were bitwise equal in these samples. All five were slower on
these two shapes, so frontier expansion is not a performance improvement.

The first physical replay exposed a diagnostic envelope bug: instruction-map
export still bounded physical pages at 8192 while replay used the corrected
9216-page pool. Fresh inputs passed, but historical pages above the stale
bound were rejected. The diagnostic export now matches 9216; old counter and
pipeline fixtures retain their original envelopes. Both failed and corrected
results remain under `/tmp/lunaflux-frontier-20260914` on the test host.
Local warning-denied native check and all 3762 tests passed before the final
diagnostic capacity correction; its focused native check also passed afterward.

Still outstanding: remaining families' constraint generation, calibrated
measured selection, incremental cache/publication, and broader numerical and
performance coverage. The expanded opt-in search can change selection when
no exact measurement is supplied; it must not be confused with approval of
every new schedule for serving.

The functional compiler no longer discards legal schedules using an estimated
work/shared-storage Pareto test. Those estimates do not prove latency dominance
and could erase transfer/geometry alternatives before tuning. Resource-feedback
and no-feedback frontiers now preserve the same legal candidates; exact offline
measurements still collapse selection to their winner. The tuning parser also
rejects mixed sample counts within one workload-vector comparison, matching its
documented equal-repetition contract.

Incremental attention compilation now accepts an earlier immutable in-process
frontier. Identical semantic problem, candidate, backend, target and capabilities
reuse elaboration/optimization/scheduling; resource feedback is rebound and
selection is rerun. Changed shapes or targets miss conservatively. The CUDA AOT
wrapper reruns device lowering and source emission, so ABI/symbol changes never
reuse stale generated code. The Qwen resource-feedback export pass uses this
path. This is not a persistent cache, live publication, or runtime JIT; those
remain separate work. Regression compares complete incremental and fresh values,
including generated source and digests, rather than only selected IDs.

Both prefill and decode resource-feedback exports now reuse the earlier
frontier. Decode also skips its formerly unconditional second compilation when
there is no resource input. Invalidation regressions cover causal/read-view
effects, numerical permission, parallel capabilities, shape and target; adding
and withdrawing measured observations must select exactly as a fresh compile.
