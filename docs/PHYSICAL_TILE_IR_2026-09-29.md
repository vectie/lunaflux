# Shared physical tile dialect and attention architecture merge

The architectural migration separates domain semantics, schedule choices,
physical dataflow/layout, and final device instructions. This increment replaces
the projection lifetime renderer's arbitrary source-string body with a closed
typed backend binding of an immutable physical fragment program.

```text
Projection semantic graph / fused ingress cut
  -> selected fold schedule
  -> FragmentProgram (shared operand IDs, ordered accumulators, lookahead)
     + OperandLayout (shared producer/consumer address permutation)
  -> closed CUDA fold binding
  -> packed MMA or WMMA instruction emission
  -> AOT CUDA source
```

Ordinary intermediate, matrix, sibling, and sibling-reuse pipelined folds use
the typed binding. Fused ingress uses the same fragment scheduling descriptor
and lifetime renderer, with WMMA instruction realization. No model-family
condition or NVIDIA instruction is introduced into the physical IR package.
Arithmetic ordering and selected scheduling policy are intentionally retained;
this is not a performance claim or a new autotuning result.

Regression coverage includes shared operand identities, independent accumulators,
register-slot liveness across one/two-slot schedules, tail reads, integer
overflow, and layout bijection/vector contiguity. Existing CUDA source tests
exercise lifetime publication and consumer fences with actual typed bodies.

## Integrated physical planning

The follow-up implementation adds disjoint `OperandStorage` regions and striped
`VectorOwnership`. Matrix, intermediate/down, gate/up, sibling-reuse phased copy,
and full-ingress pipelines derive producer addresses, shared allocation sizes,
and fragment consumer offsets from one plan. Flattening independent strips is
allowed only when their row permutation phases agree. Ring-slot publication and
reuse remain the responsibility of the existing explicit effect schedule.

Scalar projection now has an explicit product/reduction plan, including ordered
and strided pairwise folds, and a bounded selected-row lookahead plan. CUDA
validates its realizable worker widths separately from the generic plan. The
non-pipelined matrix path derives reduction geometry from its fragment plan.

Attention retains its own domain dialect: `compiler/attention_physical_ir`
describes raw and masked scores, running maxima, output rescaling, BF16-rounded
PV operands, and the unrounded denominator sum. The generic attention schedule
owns this plan, CUDA lowering retains it, and query-owned source emission walks
its operations between the existing transfer/publication effects. It does not
force online softmax into a GEMM abstraction or add runtime interpretation.

## Attention terminal migration completed

The physical program is now mandatory for every supported terminal family in
`luna_cuda_attention_tile_source`. A closed schedule-owned sum type composes direct
key folds, grouped/split-key folds, shared-score folds, and query-fragment
folds with their existing numeric and effect laws. CUDA lowering refines target
transport capability once; source emission consumes the selected program
instead of reconstructing generic plans from CUDA primitives. Unsupported
combinations remain explicit errors, never a fallback to a different family.

```text
Attention semantics -> selected schedule -> AttentionPhysicalProgram
  DirectKey       -> F32 per-key scale publication
  GroupedKey      -> same state law + grouped transfer effects
  SplitKey        -> same state law + split-key transfer effects
  SharedScore     -> storage/ownership + bootstrap/fold/terminal effects
  QueryFragment  -> SSA online fold + query transfer effects
                            |
                   target instruction lowering
                            |
                single / partitioned AOT emission
```

The shared-score program owns query offsets, shared-key factoring, numerical
probability representation, output/statistic residency, bootstrap and terminal
plans. Immediate-copy capability and matrix fragment geometry are explicit
backend inputs. Scalar programs preserve their F32 probability/denominator law;
matrix programs preserve BF16-rounded PV operands and unrounded sums. An
unsupported schedule can still be inspected as a lowering, but its retained
typed error prevents source emission. This preserves the prior rejection
boundary without an alternate legacy generator.

The redundant CUDA `direct_prefill_staging` and optional `physical_online_fold`
APIs have been removed. Single-launch and partitioned paths now use one closed
physical-program dispatch. All of this work runs during AOT compilation, not
token-step execution; no request-path interpreter or JIT has been added.

Regression coverage freezes 19 pre-merge source digests across all five terminal
families and partitioned decode, in addition to the existing query-owned source
snapshots. Capability refinement tests check that changing transport does not
change numerical/fold laws; geometry tests check factoring and invalid inputs.
These snapshots were captured before migration and pass without updates after
migration. The merge intentionally changes the compiler structure, not generated
kernel instructions, and has no new performance claim.

Validation of the merged working tree on 2026-09-29:

- Full native suite: **4,134/4,134 passed** with legacy warning categories 79
  and 25 disabled; remaining pre-existing unused-import warnings are reported,
  not treated as a warning-clean repository claim.
- `compiler/physical_tile_ir` and `compiler/attention_physical_ir`: native
  `moon check --deny-warn` passes without warning suppression.
- Affected-package `moon info`, formatting check, and `git diff --check` pass.
- No new GPU benchmark, deployment, or generated-kernel speedup is claimed for
  this byte-preserving architecture merge.

Before pushing compiler commit `685f55df`, its exact staged source (excluding
the unrelated dirty model/distributed-runtime work) was exported into a clean
directory. The full native type check passed with the same legacy-warning
exclusions; **307/307 affected tests** and affected-package formatting passed.
This clean-subset result is distinct from the 4,134-test working-tree result.

## Follow-up work identified after the attention merge

The four workstreams below are the audit that initiated the next increment;
they are retained as historical scope, not the current outstanding task list.
See [compiler architecture completion](COMPILER_ARCHITECTURE_COMPLETION_2026-09-29.md)
for their implementation and validation, including remaining hardware scope.

Source audit after the attention merge identifies four architectural workstreams:

1. **Make projection physical refinement mandatory before emission.** The
   descriptors are shared, but `source_dot_fold.mbt` still constructs a scalar
   fold and several projection/ingress renderers construct storage, layout and
   transfer plans. Move these into an explicit retained physical compilation
   stage; leave device buffer/instruction binding in the terminal backend.
2. **Extend coverage beyond projection and attention.** Pointwise and sampling
   source packages still use their own terminal generators. They need suitable
   domain semantics and common physical/effect interfaces, not forced conversion
   into a GEMM dialect. The older `luna_tile_ir` reference path is not the sole
   production compilation route.
3. **Generalize fusion-region selection.** `AttentionIngressCutPlan` correctly
   represents the three current cuts, but it is not an arbitrary-DAG partitioner.
   General cuts need observable-output/effect constraints and comparable whole-
   region costs. The exporter still chooses `FullyFused` as its explicitly
   labelled compatibility default when no comparison is supplied.
4. **Complete target/resource-driven policy selection.** For example, ingress
   binds four consumers and two live accumulators; pipeline defaults use two
   operand stages and one fragment stage. These are explicit static choices,
   not necessarily measured winners. CUDA's 48 KiB static-memory ceiling is a
   current backend/launch constraint, not a universal IR limit. Offline fold
   records and attention resource feedback already exist; in particular,
   `cmd/lunaflux_qwen3_bf16_candidate_export/attention_resources.mbt` parses and
   applies register observations for both ordinary and partitioned frontiers.
   Calling that exporter connection missing would be incorrect. Remaining work
   is consistent policy coverage and measured selection, not recreating it.

Inter-launch parallel/execution IR remains a separate concern: it cannot replace
intra-kernel register/shared-memory effects. Its current local TP overlap work
and other model additions are outside this scoped compiler commit. Additional
device backends and fresh end-to-end performance/overlap qualification are also
not established by this structural migration.

The new dialects close these opaque-body/storage and attention dispatch seams,
not every policy in the compiler or every other kernel family. The earlier Spark
campaign compares generated kernels against committed
`aad4e7e`; current source also contains earlier uncommitted fixes. Any difference
against that commit cannot be attributed solely to this architectural extraction.
The query-owned source snapshots deliberately remain byte-identical: a new IR
layer alone is not a new schedule or a speedup. Hardware tests here are bounded
standalone kernel measurements, not full serving or deployment qualification.
