# Compiler architecture completion: scope and validation

This increment addresses the four source-architecture gaps recorded after the
physical attention merge. It does not introduce runtime JIT or claim that new
IR types alone make a kernel faster.

## Compilation boundaries

```text
Model operation graph
  -> legal fusion regions and observable/effect boundaries
  -> domain algorithms (projection, online attention, normalization,
                        rotary/KV writes, reduction/sampling)
  -> target-supported schedule candidates
  -> resource eligibility + scoped offline timing selection
  -> retained physical programs (ownership, layout, numerical order, effects)
  -> CUDA instruction/ABI binding
  -> content-addressed AOT artifacts
```

All planning and selection happen before serving. GPU submission, KV writes and
resource release remain explicit runtime effects; no compiler interpreter,
record parsing, or tuning search is added to the token-step path. CUDA lane
widths, instructions and launch limits belong to the CUDA implementation, not
model builders or the scheduler.

## Owned layers

| Responsibility | Owning package | Production connection |
| --- | --- | --- |
| Shared operand/fragment ownership and layout | `compiler/physical_tile_ir` | Projection and ingress physical refinement |
| Attention numerical state and fold laws | `compiler/attention_physical_ir` | Retained `AttentionPhysicalProgram` and single/partitioned emitters |
| Row-domain numerical and effect plans | `compiler/elementwise_physical_ir` | Pointwise, residual normalization, rotary/KV, sampling, post-projection ingress |
| Observable/effect-safe fusion partitions | `compiler/fusion_regions` | Semantic ingress cuts and bundle selection |
| Measured resource-eligible policy choice | `compiler/resource_policy` | Offline ingress variants; existing projection/attention tuning stays connected |
| Projection domain physical refinement | `kernels/luna_projection_tile_compiler` | Mandatory `PhysicalProjectionTileCompilation` consumed by CUDA emission |

The common abstraction is not “everything is a GEMM.” Attention keeps its
online state law; stochastic sampling keeps its ordered probability fold;
matrix operations retain fragment and operand lifetimes. Their common boundary
is an immutable physical program with explicit ownership and effects, followed
by device-specific realization. This avoids both disconnected metadata plans
and a universal instruction vocabulary too weak to express those algorithms.

## Numerical boundaries are part of the program

In particular, residual+RMSNorm keeps the BF16 rounding of the residual result
before its F32 sum of squares. Its shared-owner binary tree is distinct from a
subgroup reduction; replacing one with the other is a numerical policy change,
not an innocent storage rewrite. Both diagnostic and production emitters now
consume the retained normalization program. The production ABI still excludes
the diagnostic canary pointer, atomic increment and final diagnostic barrier.

Before this migration, qualification and production source digests were captured
from the committed compiler in an isolated tree. The regression tests require
those exact bytes after migration and reject an unsupported reduction topology.

## Selection is not a performance promise

Measured records must identify the device, toolchain, shape, program and generated
artifact they measured. Ineligible candidates cannot become eligible by falling
back to optimistic resource estimates. A deterministic unmeasured fallback is
explicitly labelled as such. A kernel's register count or occupancy alone is not
a latency objective.

Fusion selection preserves the supplied topological order and checks values used
outside a region, observable outputs and effect boundaries. It chooses among
actual supplied implementations; this does not synthesize an executable fused
kernel for every possible subgraph. Region costs and whole-partition measurements
are different inputs: summing isolated timings does not reproduce every cache
and launch interaction. Measured and estimated costs must not silently compete.

Without a comparable measurement or explicit evaluation choice, the runtime
bundle exporter selects the unfused ingress. The existing reusable full-ingress
build helper explicitly requests `--ingress-evaluate full`: it supplies that
artifact and remains an evaluation route, not an automatically measured winner.

Existing projection fold records and attention resource feedback remain in use.
The new resource-policy path fills the ingress gap and exposes the exact scope
and candidate source identities needed to collect offline observations. A
comparison needs a matching baseline; a stale source record cannot select a
different kernel. The CUDA ingress ABI still fixes its worker count, so the
candidate set varies supported accumulator windows rather than pretending any
arbitrary thread count is executable.

## Validation record

The four workstreams are implemented and connected for the existing dense
decoder production path. This does not extend the migration claim to the
separate advanced-model candidates or to new device backends.

- The three new generic IR packages pass 19/19 tests with warning-denied native
  checks and no warning exclusions.
- An isolated staged-source snapshot passes 373/373 affected native tests,
  including existing attention and the new projection, row-domain, fusion,
  tuning and exporter paths. This check excludes existing repository warning
  categories 79, 25, 29 and 20; it is not a repository-wide warning-clean claim.
- The complete isolated snapshot passes native type checking with warnings
  denied and the same four legacy-warning exclusions. The detokenizer's 7/7
  tests and formatting check also pass in this snapshot.
- The broad working-tree native suite passes 4,167/4,167 tests with warning
  categories 79 and 25 excluded (remaining warnings reported, not denied).
  This includes other in-progress work and is not the clean-commit test count;
  the isolated results above are the scoped integration checks. The small
  detokenizer compatibility correction is additionally covered by its isolated
  7-test run.
- All 42 committed-baseline projection/ingress source snapshots remain
  byte-identical. Residual normalization retains its two qualification and
  production digests; sampling and post-projection ingress snapshots also
  preserve their existing emitted source.
- Review regressions cover unsupported CUDA vector/subgroup geometry,
  consistent async copy/commit/wait planning, KV extent overflow, externally
  observed fusion values and stale-baseline resource-budget fallback.
- An existing MoonBit parser inconsistency in benchmark detokenizer control
  flow was reproduced against the previous source and removed with an
  equivalent expression. Its 7/7 tests include empty/incomplete streams and
  repeated finish rejection.
- Affected-package formatting and the existing build helper's shell syntax
  check pass. Generated package interfaces are updated.

No fresh GPU campaign or end-to-end benchmark was run for this increment.
Preserved default source bytes establish a refactoring regression boundary,
not better performance. Optional ingress policy variants require actual
device measurements before a speedup or a measured winner can be claimed.
