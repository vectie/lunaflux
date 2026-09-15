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

### Demand-driven fallback ranking

Attention strategy selection now evaluates the full static ranking only when
there is no exact applicable measurement. Previously it ranked every generated
alternative before discarding that result in favor of the measured winner.
The selected measured plan still carries its own static score, and record
validation and candidate compatibility remain mandatory. Tests cover each
generated candidate as the measured winner, score identity, order-independent
ties and the existing foreign-target fallback. No GPU performance claim is
made for removing this unused compiler calculation.

### Bounded segmentation arithmetic

The operand segmentation unroll guard added round count to operand count in
32-bit arithmetic. A valid total of INT_MAX vectors split across two operands
could overflow that guard and enter an enormous expansion loop. The guard now
compares round count with the remaining budget using subtraction; operand
count is already bounded to 128. Tests exercise extreme valid totals without
performing the unsafe expansion, the exact existing budget boundary, and a
compact two-segment map over the same large extent. Supported ordinary maps
and the existing 4,096 expansion budget are unchanged.
Focused native tests pass 54/54 on macOS and Linux. The local aggregate native
run passes 3,778/3,778; because the segmentation regression was added while
that run was compiling, the separately completed 54-test run is the explicit
verification of the new case. Existing C allocation-probe attribute warnings
remain unrelated to this compiler arithmetic change.

### Fresh expanded-attention GPU replay at b4aa5b73

The committed attention packages and diagnostic exporter were overlaid on the
existing isolated Linux source tree. Exporter rebuild and five fresh cubin
compilations used CUDA 13.1.115; UUID/PCI matched the RTX 5060 Ti and it was idle
at admission. All five emitted sources equal the earlier capacity-corrected
sources byte-for-byte. This is not a clean whole-repository serving rebuild.

Five interleaved trials per candidate compare 2,048 query tokens with either
zero or 2,048 history tokens against the existing c322 baseline. Median us:

| Candidate | History 0 baseline / new | History 2048 baseline / new |
| --- | ---: | ---: |
| 1003 | 533.48 / 1659.61 | 1426.97 / 4966.78 |
| 1004 | 534.27 / 1030.71 | 1436.06 / 2639.09 |
| 1005 | 533.87 / 2134.77 | 1432.05 / 6145.28 |
| 1006 | 538.31 / 590.18 | 1443.65 / 1584.32 |
| 1008 | 536.86 / 626.29 | 1440.34 / 1647.07 |

All ten probe runs exit zero with empty stderr and pass the independent scalar
referee. Candidates 1003/1004/1005 equal the baseline bitwise; 1006/1008 differ
by at most 0.000488281. Thus these latter cases are not bitwise equivalence or
model-token validation. All remain slower, so no new schedule is promoted.
No fresh sanitizer, serving or competitor campaign was run by this replay.

Downloaded archive (driver, package archive, sources, cubins and raw results):
`/tmp/lunaflux-compiler-gpu-b4aa5b73.tar.gz`, SHA-256
`628e135e9e3768a2806be0abe16108c7c2177f73c0d47d0ac5bcfe7c14602820`.

### QKV epilogue extent identity

Regression reproduced acceptance of enormous positive head counts whose
32-bit sum/product wrapped to a small projection output width. The ingress
constructor now requires exact head-dimension divisibility and compares the
quotient with a 64-bit head-count sum, avoiding both narrow multiplication and
an unnecessarily large wide product. Existing valid ingress plans retain
their semantic representation; two wrapped-extent regressions are rejected.
This is a compiler input-identity correction, not the completion of QKV operand
segmentation or a change to kernel arithmetic.

### Shared concatenated-row planning

Projection compilation now owns a pure prefix-sum plan for concatenated
immutable weight operands: ordinal operand identity and half-open row ranges,
with nonpositive and overflowing extents rejected. Scalar and matrix QKV
lowering consume this same plan through one device-side branch renderer.
The renderer preserves the existing nested selection and generated spelling;
this change does not enable the previously slower address-hoisting schedule.

Tests cover one, three and four operands, unequal widths, complete row coverage,
the Int extent boundary and exact QKV branch spelling. Native projection
compiler tests pass 56/56 and CUDA projection AOT tests pass 71/71; the full
warning-denied native check passes. This is shared row-view infrastructure,
not complete transfer-worker segmentation, persistent tuning, or a GPU speedup.

QKV operand binding also consumes the same row plan rather than independently
recomputing its total. Two binder regressions reject overflowing concatenations
before source generation. The pre-binder-change full native suite completed
3782/3782; it emitted C warnings from allocation-probe `malloc` macros expanding
inside the new toolchain's allocator attribute. This is not a zero-warning C
build and needs a separate probe-header correction. The subsequent binder
change is covered by the focused projection AOT suite (72/72).

### Native allocator probe migration

Running the release allocation executable exposed a real coverage failure:
the record positive control observed zero allocations with the new mimalloc
backend. Five probe headers now redirect malloc calls without rewriting bare
attribute tokens, and separately intercept mi_malloc. The mimalloc wrapper
calls mi_malloc, not libc_malloc, preserving the runtime's matching free path.
Production runtime files and allocation behavior are unchanged.

Native release hot_path_alloc, device_step_alloc, rank_group_wire_alloc and
tensor_parallel_device_worker_alloc run successfully, including their positive
controls. device_worker_alloc still exits with DeviceWorkerError.Executor;
that executable is not a passing allocation gate and requires diagnosis.

The ordered-executor failure was isolated to the test double rejecting policy
2 (explicit eager fallback), despite valid 12-kernel/59-argument geometry.
The double now accepts policies 0 and 2 in eager mode, still rejecting required
capture. With startup repaired, the release campaign reaches its measurement
and detects 195 direct allocations across 65 cycles (three per cycle), with
zero array/string allocations. This is an unresolved allocation regression;
the zero-allocation assertion is retained and the campaign correctly fails.

### Allocation-free functional graph selection

Temporary native allocation backtraces localized all three per-cycle objects:
ExecutionGraphShape::new, ExecutionGraphBucketTable::select, and the bounds
tuple consumed by PagedOrderedExecutorDispatch::select. Shape and bucket are
now immutable value types; dispatch reads scalar bounds rather than allocating
a tuple. Selection semantics and captured-resource ownership remain unchanged;
no mutable cache or shared scratch object is introduced. Diagnostic backtraces
were removed before validation.

The release device_worker_alloc campaign now exits zero with empty output,
including positive allocation controls, the unchanged 65-cycle zero-allocation
assertion, launch/copy/readback counts, retirement and fault/cleanup checks.
Thus the previously observed 195 direct allocations are eliminated in this
campaign. This is host-side allocation validation, not a new GPU throughput
measurement or completion of the broader compiler roadmap.

Follow-up full native regression passes 3784/3784. The four other native release
allocation executables (hot path, device step, rank-group wire and tensor-parallel
worker) also pass with warning denial. Graph bucket exponent calculation now
uses ceil-log2 via count-leading-zeros rather than a repeated doubling loop.
An exhaustive test covers all 1,048,576 admitted positive inputs, proving both
coverage and minimal capacity, and rejects the lower/upper invalid boundaries.
The updated strategy suite passes 8/8. This arithmetic simplification retains
the existing shape limit and bucket policy; it is not a measured serving gain.

### Bucket-equivalence traversal during owner preparation

The neutral strategy exposes the first value of the next capacity bucket.
Runtime startup owner mapping now traverses these representatives rather than
every concrete rows/token pair. Prefill retains tokens >= rows and the mixed
phase's two-row minimum; existing owners retain precedence. Decode mapping and
decode-only clearing use the same traversal. This removes duplicate planning
work without changing a selected owner or adding hot-path state.

A test compares every slot against the original exhaustive algorithm for
7 row limits, 8 token limits and all 3 mapping operations (168 combinations),
including non-power-of-two limits and preexisting owner slots. Device-step
tests pass 196/196; strategy tests pass 8/8, including exhaustive successor
geometry over the admitted domain. For rows=32/tokens=4096, counting the loop
visits gives 5,398,176 old versus 2,373 new prefill/mixed slot visits across
21 context buckets. These are algorithmic work counts, not measured latency
or GPU throughput.

### Linux committed-package regression at 3c231dd6

On the NVIDIA host's MoonBit 0.1.20260904 toolchain, native warning-denied tests
pass for execution-graph strategy (8/8), projection tile compiler (56/56), and
CUDA projection AOT (72/72). These are the three packages archived from commit
3c231dd6, overlaid on the existing isolated diagnostic dependency tree; this
does not establish a clean whole-repository Linux build or Linux runtime
allocation-probe pass. No GPU workload was launched. Root disk availability
was 477 MiB before and 445 MiB after the run.

Source-package archive SHA-256:
`6569bf0d1c515a2f1c9aca22eba1780dcd2901eddfc7bff68cc3929a979d9ed9`.
Downloaded raw logs and driver:
`/tmp/lunaflux-compiler-linux-3c231dd6-results.tar.gz`, SHA-256
`b2f953d7e9007e6b9cf99e63b314149d0c0d0d85e6a5822a5e0270abcc2aac4d`.
The orchestration script reports an async dependency packaging deprecation;
that is separate from the package test results.

### Exact operand-segmentation expansion budget

Transfer interval planning now consumes the shared immutable concatenated-row
plan used by QKV lowering. It counts exact operand/worker-round intersections
before emission and checks them against a caller-controlled `max_segments`
budget (default 4096). The unrelated 128-operand ceiling is removed. The old
conservative estimate rejected [4095,1] with one worker even though it expands
to exactly 4096 segments; that boundary now succeeds, while 4097 requires an
explicit larger budget. Budget subtraction preserves overflow safety.

Tests cover exact/one-short budgets, aligned and unaligned boundaries, 129
operands, complete vector ownership, and extreme Int extents. Projection
compiler tests pass 57/57 and CUDA projection AOT tests 72/72. This unifies
semantic interval handling without forcing a new QKV transfer schedule or
claiming a physical speedup; full transfer-worker integration remains open.

### Matrix column-domain overflow correction

Source generation used `(output_width + columns - 1) / columns` in Int for
the matrix pipeline's column-group constant. A legal aligned output width of
2147483632 overflows the intermediate sum even though its tiled work count is
representable. The lowerer now uses `1 + (output_width - 1) / columns` for
positive validated extents. Tests compile real matrix plans for ordinary and
near-Int-limit widths and compare the generated constant with an independent
Int64 ceiling calculation. Projection AOT tests pass 73/73. This is a source
geometry bug fix; it does not allocate a giant tensor or claim a physical run
at that width, and does not complete QKV transfer scheduling.

### Concatenated weight views enter the semantic program

Weight segmentation is now immutable request/program data, rather than a row
plan reconstructed independently by CUDA source generators. The generic
`with_concatenated_weights` binding checks positive, overflow-safe row extents
and exact output coverage. It describes the logical dense-weight role and
rejects collapsing GatedMlp's multiple roles into one. Normalization preserves
the view; its intervals participate in semantic and compilation identities.
Two equal-total-width projections with different operand splits no longer
share a program identity. Abstract unbound requests retain their prior form.

QKV primary and row-variant AOT construction bind the validated ABI extents
before compilation. Scalar, single-token, matrix-map and matrix-pipeline
lowerings consume the program view, with no repeated Q/K/V extent arithmetic.
The attention-ingress constructor also binds the view and rejects later
rebinding that would disagree with its head partitions. CUDA pointer spelling
and branch lowering remain backend-owned; this does not force a new transfer
schedule or add work to token execution. Existing bound QKV recipe identities
change intentionally and require rebuilding, even where source is unchanged.

Validation: native warning-denied full suite 3,791/3,791; projection compiler
60/60; projection AOT 74/74; formatting, interface generation and native check
pass. New tests cover immutable rebinding, invalid extents, normalization,
same-total/different-partition identity and all four lowerers ignoring stale
ABI partition fields after compilation. The scalar fixture's source SHA is
unchanged; its recipe snapshot changes because semantics are now bound.

On the RTX 5060 Ti, CUDA 13.1.115 generates identical SASS for the matrix QKV
fixture and its prior decode-branch spelling. Fresh execution at 1/32/1024
tokens passes bitwise comparison and sampled independent scalar checks;
32-token memcheck, racecheck and synccheck pass. Five interleaved trials at
1024 tokens give medians 223.920 us (prior spelling) and 223.880 us (new),
effectively unchanged, not a speedup. The comparator restores only the prior
branch spelling in the same fixture: this is not an old full-runtime versus
new full-runtime benchmark. Other families, full fused ingress, Linux package
integration and end-to-end serving are not newly physically qualified here.

Raw source, cubins, SASS, scripts and results are downloaded in
`/tmp/lunaflux-weight-view-retest.tar.gz` (remote/local SHA-256
`f8929a6cad9b6198614d6fe081c13651bae13c7e138655c2e03b24fe6cee0acc`). Full transfer-worker integration,
other compiler work in the completion ledger and fresh serving validation
remain open.

### Matrix transfer packing and ownership plan, 2026-09-15

`ProjectionOperandTransferPlan` now jointly plans the two immutable operand
strips, their shared-slot offset/extent, and each striped vector-owner map.
Vector width and owner count are explicit device parameters, not NVIDIA
constants embedded in the generic planner. The matrix-pipeline lowerer consumes
these maps and storage bounds instead of separately multiplying and dividing
them while emitting CUDA. Ordered reductions and the existing producer/consumer
schedule are unchanged.

The planner rejects invalid dimensions, partial vectors and unrepresentable
combined storage using widened arithmetic. Tests invert the owner map for
every vector over row/column/vector-width/owner combinations, including partial
worker rounds, and cover exact Int-limit storage and overflowing products.
Projection compiler tests pass 62/62; projection AOT tests pass 74/74; native
warning-denied check, format and interface generation pass. The QKV/output/head
matrix fixture exports are byte-for-byte equal to the preceding weight-view
retest exports (`/tmp/lunaflux-transfer-plan-retest.mbtx` and its saved outputs).
No new timing improvement or extra physical coverage is claimed for this
source-preserving planning refactor. Alternative operand-specific worker
schedules and their measured selection are still separate work.

### Linux transfer-plan regression and fused-ingress barrier correction

The two packages archived from `2c6bfae5` pass Linux native warning-denied
tests: projection compiler 62/62 and projection AOT 74/74. They were overlaid
on the existing isolated diagnostic dependency tree, not built as a clean
whole-repository Linux release. Source archive SHA-256 is
`a2e3363d39550100ab3313e22387ddf959eb800607af62578ce645d086eb59eb`.
Downloaded logs: `/tmp/lunaflux-transfer-linux-2c6bfae5-results.tar.gz`,
SHA-256 `1ec34009fc058d9841fa48b3c53249465d478603a69eccf9a7e9409263ffaa25`.

Review then found a concrete effect-participation bug in full fused ingress:
its warp-dependent column loop contains block-wide barriers and cooperative
input staging for partial row tiles. At head dimensions 16/32 some warps skip
that loop, although they are needed for the block effects. The fix uses the
generic finite owner map's uniform round count and masks only the independent
matrix load/fold/store. Every warp still participates in staging and barriers.
Unsupported sub-matrix head dimensions now return an error before source
emission rather than reaching the lowerer's abort.

Fresh RTX 5060 Ti validation covers head dimensions 16/32/64/128 crossed with
1/2/7/8/15/16/17/31/32 tokens: all 36 cases pass structured independent numeric
checks and exact output-to-KV checks. All four executables pass memcheck,
racecheck and synccheck. Restoring the original column-loop structure in the
32-dimensional fixture reproduces a synccheck exit of 7, divergent-block-barrier
reports and an error summary of 480. This is an isolated old-loop reproduction,
not a complete previous runtime. The permanent diagnostic harness is
`tests/fused_ingress_barrier_cuda/probe.cu`; native source regressions cover all
four dimensions and unsupported-width rejection.

Local full native suite passes 3,795/3,795; fused AOT passes 22/22.
Downloaded sources, executables, scripts and old/new logs:
`/tmp/lunaflux-ingress-barrier-retest.tar.gz`, SHA-256
`c0ed688e5a98c83b88c8caf38a9939fee15cad7e5bb783d29c8a04532ff95f3b`.
No performance gain is inferred. These structured numerical cases do not
resolve the separate 128-dimensional full-chain model-logit divergence or
replace fresh end-to-end serving qualification.

### Full versus partial ingress: numerical cut-point retest

The fused fixture now accepts query/KV head counts, hidden width and token
capacity independently. Its opt-in `LUNA_TEST_EXPORT_INGRESS_CUT=1` export emits
ordinary projection, partial postprocessing and complete fused ingress from
the same model plan, profile and operands. The default test emits no sources.
The 1024-to-4096, 16-query/8-KV-head, head-128 fixture has a 2048-token envelope
and enough context/page-table capacity, rather than silently reusing the small
fixture's 256-position envelope.

Fresh RTX 5060 Ti synthetic tests compare token counts 1/17/32/1024/2048.
The diagnostic full kernel writes its BF16-rounded projection intermediate;
every element matches ordinary projection exactly. Applying partial
QKNorm/RoPE/KV-write to that ordinary projection also matches the complete
fused rotated output and both KV arenas bit-for-bit in all five cases.
Inputs and weights are deterministic pseudorandom BF16, not captured model
activations. These results therefore do not resolve the real-model logit
divergence, establish performance, or admit a new production runtime. The
next numerical test needs actual request activations and the selected runtime
variants. Fused package tests pass 23/23; native warning-denied check and
interface generation pass.

Downloaded cut-point sources, diagnostic executables and logs:
`/tmp/lunaflux-ingress-numeric-cuts.tar.gz`, SHA-256
`c23c00020058f56f8f7560e64dbb38fd8ad0a68b332a8a582aea91777bb79741`.

### Logical ingress plans and device limits

Removed the literal 1024-lane CUDA block ceiling from the backend-neutral
ingress numerical tree and head ownership map. Logical reduction width must
still be a positive power of two; head packing must contain complete subgroups
and remain within representable integer extents. Device executability is a
separate question: CUDA ingress still requires its 32-lane numerical tree,
and physical launch construction retains its device limits. This does not add
support for larger CUDA blocks or other device backends.

Regression cases cover logical widths up to 2^30, 4096-lane logical packing,
tails and Int-limit head indices. Projection compiler tests pass 63/63.
The three exported QKV/partial-ingress/full-ingress sources remain exactly
equal to the preceding cut-point test sources; no GPU speedup is claimed.

An additional physical numerical stress test uses a mixed integer hash and
17 exponent bins for deterministic signed BF16 inputs/weights, rather than
only the preceding bounded linear-range samples. For token counts
1/2/4/8/16/17/31/32/127/1024/2048, projection, rotated output and both KV arenas
match bit-for-bit between full and separated execution. This still uses
synthetic operands, one query row and one selected projection variant; it is
not the missing actual-model activation capture or a serving benchmark.
Downloaded archive `/tmp/lunaflux-ingress-dynamic-range-retest.tar.gz` has matching
local/remote SHA-256
`e86e4152d869e126bad5a9ffa3fb59935b3f3c737acbac947afd107097b7c51f`.
The full native suite passes 3798/3798; the CUDA-lowering regression explicitly
rejects the wider logical reduction plan. No public API signatures changed.

### Batch-dependent arithmetic diagnosis

`BENCHMARK_INGRESS_BATCH_NUMERICS_2026-09-15.md` replays the exact historical
full-ingress module and corrects an initial diagnostic page-geometry mismatch.
With matching geometry, same-batch full/partial outputs agree, but holding an
input row fixed while switching from single-row to matrix execution changes
one synthetic value-projection component. This separates fixed-schedule
repeatability from cross-schedule bit identity; it does not yet identify the
cause of the real-model divergence. The numerical fixture now derives its
capacity from 8/16-token page geometry and passes 24/24 package tests.

The subsequent actual trace join finds within-runtime variability as well:
partial C8 and full C16 each return both terminal tokens for an identical
request body across repeated trials. Not every divergence involves a single
token batch. The diagnostic comparator now reports within-configuration
repeatability and preserves observed batch context, with an executable
regression test. Across-configuration differences alone therefore cannot
qualify or reject a compiler pass without controlling execution schedules.

### Explicit projection transfer effects

The common projection compiler now owns `ProjectionFoldEffectPlan`, derived
immutably from a validated fold pipeline and explicit serial/overlapped
transfer execution. Ring geometry, priming count, future distance and allowed
pending transfer-group count are no longer recomputed by CUDA text generation.
CUDA retains instruction names, group waits, workgroup barriers and the compact
two-slot XOR realization. This is an AOT planning value, not runtime state or
request-path work.

The finite-ring tests cover 2/3/4 slots and 4/8/16/32/64 transfers under both
execution dispositions. The CUDA lifetime test now includes the default
two-slot path as well as wider rings. Both affected packages pass 137/137 tests
and the warning-denied native check passes. A temporary differential test
compared the preceding commit's renderer with the new renderer for all twelve
stage/mode/phase-argument combinations: generated text is identical. The old
renderer was removed after comparison rather than retained as a parallel
implementation. No GPU speedup is inferred from this behavior-preserving
refactor.

This closes the common projection ring's scalar effect planning gap, not all
family-specific effect/lifetime integration. Cross-batch model activation
diagnosis and fresh complete serving qualification remain open.

The ring plan now also supplies its ordered iteration actions: serial transfer
consumes, issues and publishes; overlapped transfer issues, consumes, waits and
publishes. CUDA renders those actions instead of choosing their order with
local conditionals. The finite-state tests execute the actual action sequence
for each ring/mode/extent, reject premature or repeated publication, check the
live slot before consumption and prohibit overwriting it with future data.
The twelve-case previous-renderer comparison still produces identical text.
This does not model arbitrary kernel effects or close the separate attention
and full-model numerical work.
