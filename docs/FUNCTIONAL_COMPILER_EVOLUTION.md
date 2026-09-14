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
