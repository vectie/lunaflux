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

Block-owned matrix QK/PV schedules 310–317 are now generated from the supported
query/KV domain and the narrower realized async domain. Compatibility tests
preserve every legacy field, including the small-query profitability threshold.
This removes repeated schedule records without changing the legal set or
claiming new kernel speed. Scalar/subgroup and matrix-QK-only domains now use
the same generation pattern; 13 additional compatibility cases preserve their
complete geometry, phase, arithmetic family, transfer mode and thresholds.
These realized-domain limits remain support constraints, not measured tuning
policy. Grouped decode generation and other kernel families are separate from
this base table. Strategy (22), compiler (19), and CUDA source (32) tests pass.

The CUDA AOT frontier now exposes `rejected_candidates()` with candidate ID
and compiler-stage reason. Non-selected backend failures previously disappeared
silently. Every functional variant must now be accounted for by exactly one
emitted or rejected entry; failure of the selected variant still fails the
compilation. Diagnostics remain outside executable canonical bytes and outside
token execution. This does not make a rejected lowering supported.

Whole-frontier consumers now generate validated static plans in one pass rather
than regenerating the entire candidate set for each member. Both ordinary and
partitioned compilation consume those plans. The public single-candidate entry
point still validates external candidates. A regression compares every batch
plan to its individually validated equivalent. Resource-budget preparation also
uses the previous AOT frontier. These reduce compiler work, not GPU latency;
no compile-time speedup percentage has yet been measured.

### Linux cross-check of fda84583

Committed strategy/compiler/AOT/probe packages were overlaid onto the existing
isolated Linux diagnostic source tree (not a clean whole-repository release).
Native warning-denied tests passed: strategy 23/23, compiler 19/19, AOT 5/5.
The release diagnostic exporter rebuilt successfully. Generated instruction-map
sources for 1003/1004/1005/1006/1008 match the capacity-corrected physical replay
sources exactly. No new GPU timing was run: these unchanged sources provide no
basis for claiming inference improvement from the compiler refactoring.

Downloaded results: `/tmp/lunaflux-compiler-linux-fda84583.tar.gz`, SHA-256
`e7b772a0679b8e42c91a6fa4a565d87de02a08d9c01af8b190afa437323b93f5`.
This checks these packages and sources only; full exporter integration,
end-to-end serving, other kernel families and calibrated selection remain open.

Projection follow-up found inconsistent capability checks: static gated-MLP
decode required an eligible matrix companion, but offline subgroup-GEMV
selection checked only input divisibility. Both now use one pure eligibility
predicate. Regression rejects missing and dimension-incompatible companions
while retaining the supported path. Projection strategy 9/9 and tile compiler
53/53 tests pass. This closes an invalid-selection case, not a GEMM speed gap.

Projection fold observations now retain sample counts and require the same
count as their matched baseline. Counts may differ between independent baseline
groups. Previously counts were discarded after checking only a minimum of
three, allowing unequal-sample comparisons. Tests cover rejected mismatches,
valid comparisons and independent groups; no production measurements were
rewritten to satisfy the new rule.

Measured attention selection now reuses an already compiled identical candidate
as well as resource-only retuning. Selection provenance and request observations
are rebound explicitly; semantic/capability identity remains required. Both
prefill and decode exporters pass their prepared frontier into this final
selection stage. Tests require one reuse when applying and withdrawing a
measurement and exact equality with fresh compilation, including provenance.
This closes the measured-selection reuse gap, not persistent caching or JIT.

### Functional pass reuse CPU measurement

`tests/attention_compiler_bench` on Apple M4, native release, measures the same
12-candidate request with complete output equality in both arms. One warmup
per arm, five alternating trials, 20 compilations each:

| Trial | Fresh microseconds | Reused microseconds |
| --- | ---: | ---: |
| 0 | 201.3958 | 16.2875 |
| 1 | 178.58335 | 14.34375 |
| 2 | 174.4333 | 15.4 |
| 3 | 184.3042 | 13.93545 |
| 4 | 179.2375 | 14.01665 |

Medians: 179.2375 versus 14.34375 us, approximately 12.5x for this pure-pass
workload. Equality-check cost is included. Source emission, nvcc, file caching,
changed shapes and GPU inference are excluded. This demonstrates reduced CPU
compiler work, not a kernel or serving speedup. The executable remains a
reproducible benchmark rather than a timing assertion in normal tests.

### Projection bucket geometry

Replaced the two handwritten row/slot tables with inverse geometric rules:
decode owns slot zero (one row); prefill slot `s` has capacity `2^(s+2)`.
The existing minimum prefill capacity of eight and maximum of 1,048,576
remain unchanged. This removes duplicated policy, not the compatibility
domain. An exhaustive native test covers every supported row count, minimal
capacity, round-trip slot identity and rejected out-of-range inputs.
Projection strategy tests pass 10/10 and compiler tests pass 53/53.
No kernel schedule or inference speedup is claimed for this refactor.

The CPU benchmark's prose-only README uses `.md`, avoiding the toolchain's
deprecated blackbox-test input convention for executable packages.

### Partitioned measured-selection resource fix

Full native regression after the bucket refactor passed 3,770/3,770.
Further audit reproduced a split-K compiler bug: an exact autotune winner
returned before checking the supplied resource budget. Ordinary compilation
already rejected zero residency. Partitioned compilation now applies the
same feasibility rule; measured latency still outranks static ranking but
cannot override shared-memory, thread or register capacity. Regression covers
all three impossible budgets and proves a feasible measurement preserves its
candidate, semantic program, schedule and digest. This changes invalid-plan
handling, not generated kernels or inference latency.

### Preserve distinct split-K schedules

Resource-conditioned regression exposed another frontier bug: targets 32 and
64 can select different query tiles (c314 and c312) while both use two KV
partitions. Deduplicating only by partition count discarded the second legal
plan. The frontier now collapses only identical complete compilation digests,
preserving the first target for genuine duplicates. CUDA source naming retains
existing unique-count `_pN` names and adds a deterministic `_vI` suffix for
repeated counts, preventing symbol collisions in the expanded family.
Exporter artifact paths use the same repeated-count distinction so different
plans cannot overwrite or collide with an earlier `-pN` artifact pair.

Tests cover the reproduced two-plan case, genuine duplicate collapse, 33
independent target/resource comparisons, and repeated-count symbol uniqueness.
The resource observations in the reproduction are synthetic compiler test
inputs, not GPU measurements or new production tuning records.
Native regressions pass: attention compiler 23/23, CUDA AOT 6/6 (including
full lowering/emission of both equal-count plans), Qwen exporter 11/11.

### Wide attention geometry accounting

A plan-only regression reproduced silent 32-bit overflow in fallback resource
scoring: a legal large batch was charged 76,737,115,525,742,592 cost units
instead of 38,368,557,762,871,296 because wrapped workgroup counts lost the
second resident slot. These are dimensionless model scores, not nanoseconds.
Workgroup products now widen before multiplication in both compiler resource
scoring and the optimizer's partition decision. The tests cover large legal
batch sizes, exact expected scoring and retention of the unsplit fold when
query/head parallelism is already sufficient. No tensor memory is allocated
by these compiler tests; they do not claim that such a batch fits a GPU.

### Linux regression and compiler timing at cc215c71

The seven committed compiler-related packages were overlaid on the existing
isolated Linux diagnostic tree. This is a package-level cross-check, not a
clean whole-source release or physical serving qualification. Native tests
passed: attention strategy 23, optimizer 8, compiler 24, CUDA AOT 6, projection
strategy 10, tuning 5, projection compiler 53 (129 total).

On the host's Intel Core i9-7900X, the same native-release CPU benchmark used
one warmup per arm, five alternating trials and 20 compilations per trial.
Every result equals fresh compilation and all twelve variants are reused.

| Trial | Fresh microseconds | Reused microseconds |
| --- | ---: | ---: |
| 0 | 468.6694 | 44.8034 |
| 1 | 456.92505 | 44.58775 |
| 2 | 434.3678 | 39.4921 |
| 3 | 437.5933 | 39.2686 |
| 4 | 432.7358 | 39.5098 |

Medians are 437.5933 and 39.5098 us (approximately 11.1x). This measures only
pure compiler passes and equality checks, excluding CUDA source emission,
nvcc, persistence and GPU execution. It must not be reported as inference
speedup or as a controlled CPU comparison with the earlier Apple M4 run.

Uploaded package archive SHA-256:
`f0169e9d8c0125589f0a3627d4722d9e3ddca8b4f3a9441b2983d4fc6170b80d`.
Downloaded results: `/tmp/lunaflux-compiler-linux-cc215c71.tar.gz`, SHA-256
`6262bb4dc30117772db11d47088688fade74e2720f1ca354db66aa34bbec0ea2`.
The test filesystem had only 519 MiB free before this run; larger new-source
builds and end-to-end campaigns require space planning rather than overwriting
or deleting existing model/evidence directories.

### Resource-only incremental transitions

Added regression for a cached winner becoming infeasible after new register
observations, complete resource exhaustion, and subsequent restoration of the
original request. Reused results must equal fresh compilation and the old
immutable frontier must remain recoverable. All 25 compiler tests pass.
The selection fold now carries the incumbent score explicitly, evaluating
each variant once instead of rescoring the incumbent and final winner.
This removes redundant pure compiler work without changing scoring policy,
candidate tie-breaking, canonical output or runtime kernel execution.
