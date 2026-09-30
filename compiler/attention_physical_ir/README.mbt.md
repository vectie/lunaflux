# Online attention physical state dialect

`OnlineFold` is an immutable, backend-neutral register-state program for the
ordered online-softmax recurrence. QK produces raw scores; the value phase
applies scale/visibility and computes a local maximum, then merges
the previous maximum, rescales the previous output, forms BF16 probabilities
and accumulates PV, then updates the F32 denominator. Each operation exposes
named read/write values, and the loop carries three explicit state edges.

The generic selected schedule constructs this program before CUDA lowering.
The selected query-owned CUDA emitter consumes these operations, rather than
owning their ordering as one opaque source block. Existing schedule-owned
transfer/publication/release effects surround the two physical phases. This
does not turn attention into GEMM or add runtime compiler work.

The current numerical contract preserves ascending-key folding, BF16-RNE PV
probabilities and an unrounded F32 denominator. Device instruction realization,
lane mapping, masked exponential, and reduction topology remain backend-owned.

`OnlineFragmentProgram` refines that same fold into QK/PV fragment products
using explicit target instruction dimensions. Query and rounded probability
fragments retain their reuse across right columns; K/V fragments have one
consumer and can use the shared physical IR's register-forwarding transform.
It does not reorder reduction steps, change probability precision, add
storage, or move reads across transfer publication/release effects.

`QueryOperandLifetime` jointly refines the invariant Q operands and epoch-local
score/PV fragment live ranges. A complete-fragment prefix may remain live across
ascending KV epochs; its explicit representation size estimates fragment
storage, not the hardware register count. The selected lowering retains this
plan and materializes separately named Q operands before the KV loop. No-retain,
half-prefix and full-prefix AOT alternatives preserve the same numerical fold.
Register lifetime and saved shared loads must be measured together; unchanged
static cost and stable-ID ties retain the prior default without fabricating a
performance win or growing KV width first.

`KeyFold` separately describes the F32 per-key statistics merge and scale
publication used by direct, grouped and split-key attention. It computes
component ownership from an explicit backend-supplied owner width, without
changing the scalar probability law into the BF16 matrix law. The schedule's
closed `AttentionPhysicalProgram` binds these distinct dialects and the
shared-score program to their transfer/storage effects. Full device instruction
and copy-address realization remain backend responsibilities.

`PagedRowOwnership` describes address-invariant sharing independently of a
vendor subgroup width. It elects one producer only when vector consumers of a
row fit wholly inside a subgroup; otherwise each consumer retains its address.
CUDA grouped synchronous and asynchronous transfers both consume this plan.
Collectives are outside the active-key predicate, including zero-fill lanes.
This is compile-time ownership, not an extra token-step validation pass.

`BlockwiseFold` is the distinct `blockwise-f32-probability-v1` decode law.
It computes a tile maximum and F32 probabilities, reduces one denominator,
accumulates an ordered tile PV numerator, then merges the running state once
per ascending tile. Its score/probability exchanges are subgroup effects;
shared-KV publication remains workgroup-scoped. It is not bit-equivalent to
`KeyFold` and does not round probabilities to BF16 like `OnlineFold`.

The separate `dual-score-blockwise-f32-probability-v2` law gives each subgroup
two disjoint score owners with smaller QK reduction trees. Cyclic component
group traversal is a bijection with a declared reassociation law, not a claim
of bitwise equality. CUDA lowering chooses the physical traversal phase from
the row stride and shared-word bank mapping; model and scheduler layers do not
contain warp or bank constants. Both key-tile widths remain measured AOT
alternatives, not an unmeasured replacement for the selected ordered kernel.

`PartitionMerge` gives invariant maximum/denominator/scale statistics one
owner and one publication. Disjoint component owners consume those statistics
using the unchanged ascending-partition numerator sum. Partial and merge
launches are measured together, including empty partitions and ragged masks;
parallelism alone does not establish that a split route is faster.

`RetainedCopyOwnership` is the finite product of vector slots and workgroup
owners. It gives every retained copy address and validity size a statically
named binding between the key and value transfer publications. Device lowering
consumes that ownership as scalar bindings, not a dynamically indexed array.
This preserves address sharing, zero-fill masks, stage order, numerical law
and CTA geometry. It does not assert that registers are free: the emitted
register count, local traffic and total time still require measurement.

`ValueComponentOwnership` is a bijection between independent output components
and consumer fragments. Supported contiguous fragments permit packed loads;
ragged or unsupported physical widths retain the masked interleaved mapping.
The same resolved ownership must govern accumulation and both ordinary and
partitioned output stores. It changes neither the key order nor the strict
probability law. The CUDA lowering independently chooses aligned copy width
and consumes retained addresses as named values rather than dynamic arrays.

`blockwise-fma-f32-probability-v3` is a separate, opt-in contraction law, not a
reinterpretation of V1. Only its explicitly contracted dot, PV and state-merge
operations use FMA. Exponential evaluation, probability precision and ascending
tile/partition ordering remain explicit. CUDA compilation still disables
implicit contraction globally. A distinct symbol, ABI and bundle schema carry
this law through export, packaging, startup admission and split-route binding;
strict bundles keep their existing identity. Numerical and whole-serving
comparisons are required before selecting this alternative.
