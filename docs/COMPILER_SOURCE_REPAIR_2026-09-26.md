# Compiler source repair — 2026-09-26

## Scope

Source work only at the user's request. GPU runs, test execution and benchmarking
are deferred. No runtime has been uploaded or deployed by this change. The
historical Spark comparison is not a measurement of the modified source.

## IR boundaries

More IR names alone do not make kernels faster. The existing projection compiler
already has typed Selected, Semantic, Optimized and Scheduled stages. The
identified defects were that full attention-ingress lowering invented part of
its projection loop organization after scheduling, reuse analysis did not
consistently respect observable-effect regions, and fusion cut consumers
duplicated materialization and KV-write ownership decisions.

The intended division remains:

| Boundary | Responsibility |
| --- | --- |
| Semantic/value IR | Observable results, demand, numerical rules, exact CSE/DCE |
| Fusion-region/cut plan | Which intermediates are materialized; preserve KV-write dependencies |
| Tile/schedule IR | Parallel maps, ordered folds, ownership and tile domains |
| Physical operand/effect plan | Bounded accumulator and operand lifetimes, layout, staging, synchronization |
| Device lowering | CUDA/WMMA/copy instructions and backend resource limits |

These are responsibilities, not five newly implemented IR packages. In
particular, an arbitrary-DAG fusion partitioner and measured resource-driven
selection are not completed here. The ingress cut plan covers the three
existing full, producer-separated and unfused schedules.
No model-specific conditions, NVIDIA types or runtime JIT are added to the
middle end. Necessary KV writes and GPU synchronization stay explicit effects.

## Implemented source repair

The full-ingress matrix path previously nested its complete K traversal inside
each independent column round. For a 128-component head, four consumers and
16-column tiles, this traversed the same input fragment twice per consumer;
partial token tiles also repeated cooperative padding and CTA barriers.

`ProjectionColumnFoldPlan` is an immutable, backend-neutral physical-loop plan:

- Partition independent output tiles across consumers and bounded windows.
- Keep each output's reduction in its original order.
- Reuse one input fragment across the independent columns in the same window.
- Keep inactive consumers participating in workgroup effects.
- Handle arbitrary positive domains and partial windows without integer
  overflow in the ownership mapping.

Scheduled dense projections can produce the same plan. Scheduled ingress uses
its explicit head-component domain. CUDA supplies four consumers and at most
two live accumulator fragments for its existing supported head envelope; it
does not decide the partition or reuse lifetime independently. The lowering
consumes `window -> reduction -> column`, rather than `column -> reduction`.
Its AOT recipe includes the physical plan's digest in addition to semantic and
schedule identity. This is startup/export work, not token-step hashing.

The single-token branch, per-output WMMA reduction order, BF16 rounding cuts,
normalization/rotary operations and KV-write semantics are retained. Increased
accumulator liveness may affect registers and occupancy. This source inspection
establishes eliminated duplicate operations, **not** an observed speedup or
proof of the historical performance gap's cause.

## Shared pipeline and explicit effects

Ingress now refines its head-component domain into the same
`ProjectionFoldPipeline` used by ordinary projection. Its CUDA renderer consumes
the common operand transfer plan and the common finite-ring effect renderer:
initial issue/completion/publication, future issue, ordered consumption, and
slot publication are no longer independently reconstructed in ingress.

Offline `MatrixFold` choices control transfer width, two-to-four operand stages
and one-to-two fragment stages. Fragment lookahead changes operand availability,
not the ordered accumulation. Masked row tails use zero-filled asynchronous
copies. All consumers participate in workgroup publication, including consumers
with no live output column. CUDA-specific WMMA fragments and copy instructions
remain in device lowering. This is shared planning and lifetime rendering, not
a claim that every projection instruction/layout renderer is now identical.

The CUDA lowering rejects choices exceeding its 48 KiB static shared-memory
envelope. This is a backend constraint, not a universal compiler limit or an
autotuned optimum. The default maximum-head configuration uses 45 KiB for the
operand ring and projected output. Register usage remains unmeasured.

`ProjectionEffectRegionEntry` partitions the value graph at observable stores
and KV commits. Reuse and ingress-elision analysis now requires matching effect
regions, consistent with the existing CSE barrier. The terminating effect is
part of its producer region; unrelated effects cannot be crossed speculatively.

## Semantic cuts and exporter integration

`AttentionIngressCutPlan` describes materialized intermediates, operation span
and permission to use read-only attention. Semantic matching records exclusive
use facts; selecting a cut checks only the values that cut eliminates:

- Full fusion requires exclusive projected and normalized intermediates.
- Producer-separated fusion preserves projected QKV and requires exclusive
  normalized QKV.
- Unfused execution preserves both and leaves KV writes with attention.

Final-output observations and other consumers prevent elimination. The partial
lowering recomputes these facts from the supplied model, rather than accepting
facts from a different semantic plan. Both fused lowerings validate their cut.
The exporter consumes the structural cut for module spans and KV ownership.
It reports whether selection came from a compatibility default, an explicit
evaluation request or a supplied measured comparison. Cost accumulation and
winner selection use functional folds with deterministic ties.

These are offline compiler/export changes. They add no request-path validation,
cryptography, filesystem work, JIT or per-token allocation.

## Still separate work

- A legal full fusion is not necessarily the fastest fusion cut. The exporter
  still defaults to full fusion without comparative measurements; this change
  explicitly labels its origin and does not relabel that default as a measured
  winner. It does not generate new measurements or an autotuning database.
- Shape-specific winners, actual register/residency feedback and measured
  fusion-chain costs still require the deferred validation campaign.
- CUDA instruction/layout realization remains specialized. Sharing lifetime
  planning does not establish additional device backends or prove optimality.
- No new hardware-counter diagnosis is claimed for attention or long sequences.

## Validation state

Regression cases were added for exhaustive small ownership domains, tail
consumers/windows, accumulator budgets, maximal integer extents, physical
identity, schedule refinement and generated ingress loop/barrier placement.
Additional cases cover effect fences, observable intermediate values, cut
ownership, exporter selection origins, multistage/fragment choices and the
static shared-memory boundary.
They are source-only cases until test execution is authorized again.

Targeted static checking includes their compilation. Strict checking initially
stopped on existing toolchain-migration warnings (79 and 25); checking with
those warnings excluded compiled the affected packages with zero errors, while
reporting existing dependency deprecation/unused-import warnings (20 and 29).
This is not a clean warning-denied whole-repository result. Formatting and
generated interfaces are checked separately. CUDA compilation, sanitizer,
numerical equivalence, resource usage and timing remain deferred.
