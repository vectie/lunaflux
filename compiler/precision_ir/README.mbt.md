# Precision / representation refinement

Pure, immutable plans separate storage precision from activation/accumulator/
output precision. Conversion placement and accuracy bounds are semantic inputs.
Startup writes, ordinary step writes and transactional cache commits are explicit
effects. Quantization publication stages are planned before device lowering.

The [numerical correctness roadmap](../../docs/NUMERICAL_CORRECTNESS_RETHINK_2026-10-12.md)
distinguishes exact rewrites, bounded rewrites with assumptions, and empirical
arithmetic alternatives. Existing laws do not imply that one vendor's reduction
order is universally correct, or that local bitwise agreement guarantees model
quality. Compositional expression/replay integration is planned, not claimed as
complete. Prediction comparison runs offline and never adds token-step effects.

`SquareReductionLaw` makes ordered versus compensated F32 square summation
explicit, independently of tensor storage and normalize-before-weight rounding.
It is a numerical accuracy choice, not permission to reassociate an ordered
reduction. MiniMax text encoding selects compensation in its immutable plan;
the CUDA renderer consumes that law without model-specific device branching.

`VectorizedTreeFloat32(partitions, vector_width)` specifies a different exact
F32 square-fold order: independent per-component sums, ordered component join,
then a descending partition tree, including explicit empty/tail identities.
`ReciprocalRootLaw` separately distinguishes reciprocal-sqrt from dividing by a
rounded sqrt. These are numerical contracts, not NVIDIA scheduling names.
Hyper control RMS and weighted branch RMS carry their own fold choices; their
default sources and six global frames remain unchanged. Lowerings must carry
the chosen workgroup requirement into execution, not reuse a fixed launch width.

`DotReductionLaw` similarly makes cancellation-sensitive dot summation an
explicit accuracy choice. BF16 products and final BF16 operator boundaries
remain unchanged. Compensated F32 retains contributions lost to a large running
sum; it is not bitwise equivalence to ordered F32. The reusable CUDA lowering
uses Neumaier compensation and preserves nonfinite/overflow propagation rather
than manufacturing finite outputs. Physical component probes check the actual
generated dense and staged-MLP source against an independent F64 oracle.

`GroupedVectorFloat32` retains an explicit two-level dot order: group/partition/
component ownership, separate or fused multiply-add rounding, descending local
trees, and ordered publication stripes. These are numerical semantics, not CUDA
launch dimensions. Parameter choices are explicit per numerical alternative;
there is no universal group-count constant or vendor reference in model plans.
An unsupported lowering rejects the law rather than silently changing it.

`StripedTreeFloat32(partitions)` adds an exact summation-order choice to that
same `DotReductionLaw`, not another numerical-policy layer. Partition p sums
p,p+P,... and a fixed upper-half/lower-half tree joins the partials. The grouped
BF16 output lowering implements it cooperatively; CUDA subgroup details stay
in its backend. Unsupported lowerings report the missing implementation rather
than substituting ordered or compensated sums. Ordinary callers retain ordered
accumulation. Matching one projection is not a whole-model quality guarantee.

`HyperConnectionPrecision` independently records control square-sum and
control-dot laws. Both learned attention and FFN envelopes consume those same
immutable laws. Compensated F32 and striped-tree F32 refine arithmetic without
changing stream ownership, parameter dtype, global workspace or queue effects.
Default control source remains byte-identical; an explicit diagnostic policy
is required to select the refined candidate. No reference tensors or vendor
library calls are embedded in the production lowering.

`Float32AffineLaw` and `SigmoidEvaluationLaw` independently specify vector
multiply-add rounding and sigmoid evaluation form. Hyper controls carry these
as immutable numerical inputs, separate from their square/dot laws and matrix
logits. Direct reciprocal sigmoid does not authorize an approximate exponential.
The CUDA backend spells fused F32 multiply-add locally; no NVIDIA instruction
name enters the semantic plan or model adapter. The selected choices reach both
attention and FFN envelopes without adding buffers or completion boundaries.
Ordinary callers retain separate affine products and stable sigmoid branches.

`PhysicalOperand` refines an existing physical operand region; it does not invent
an unrelated shared-memory map. Local/global F32 scaling and BF16 rounding are
retained boundaries. A caller-provided supported-placement set is a capability
input, not an assertion that arbitrary hardware implements low-bit MMA.

`WeightSchema` refines memory over the whole graph's ordered tensor bindings.
Alignment and sequential/concurrent scratch lifetimes are explicit; per-tensor
budgets cannot substitute for the aggregate device memory ceiling.

ExpertMlpPrecision additionally plans three compact expert projections,
noncontiguous rank placement, a shared packed bank, and bounded BF16
intermediate/weighted workspaces. PackedBufferLayout is the single source of
payload, block-scale and scalar-scale offsets for both upload and lowering.
Rank-local contributions are summed in F32 before any cross-device reduction.
The compute implementation uses pairwise F32 dot reductions and explicit BF16
stage rounding; it does not promise bitwise equivalence to a serial dot product.

MoeRoutingPrecision describes BF16 hidden rows, BF16/F32 router weights, F32
logits with an explicit dot-reduction law, an independent softplus evaluation
law, and selected-weight summation. The ordinary policies remain ordered dot,
stable softplus, and ordered selected sum. Thresholded strict softplus is an
explicit alternative, not a claim of stable-expression bitwise equivalence.
Selected butterfly sums pad to a power-of-two extent with zero identities;
no logical expert slot is introduced. All policies are immutable arithmetic
contracts; CUDA launch geometry belongs only to lowering.

The routing plan also specifies choice-only correction, group selection, stable
lower-ID ties, and unbiased selected-weight normalization/scaling. Sigmoid and
stable sqrt-softplus are explicit numeric policies. Five compact routing buffers
have an exact aggregate workspace size; no CUDA geometry or model name enters
this IR. The first lowering is an executable correctness implementation, not a
Tensor Core router or a demonstrated performance improvement.

See [implementation, tests and remaining serving work](../../docs/PRECISION_IR_2026-10-04.md).

RotaryHadamardFp4Precision specifies adjacent-pair suffix rotary followed by
normalized full-vector Sylvester Hadamard and blockwise E2M1 simulation, with
BF16 publication between operations. Two exact scratch tensors share one
aggregate budget. This StepWrite plan does not own retained cache history;
RetainedRowsPrecision separately specifies exact-word CacheCommit publication.

CausalConvolutionPrecision describes request-owned BF16 raw-input history,
ordered F32 causal convolution, SiLU and BF16 output rounding. Its exact frame
and persistent-state sizes are independent of batch sequence ordinals.
RecurrentDeltaPrecision describes normalized delta attention with F32 state.
Both expose CacheCommit effects; neither contains CUDA geometry or model names.

RotaryKvPrecision separates full-head adjacent-pair F32 rotary with BF16
publication from non-rotary E4M3 KV simulation. Base/YaRN frequency policy,
simulation block width, amax floor and scale law are immutable semantic inputs.
Simulation publishes BF16 operands (StepWrite), not packed persistent cache
storage or cache ownership (CacheCommit). Two exact output spans participate
in the surrounding aggregate workspace ceiling before device allocation.

WindowSharedKvPrecision owns a bounded shared-KV ring independently of maximum
context length. Its CacheCommit effects attend against old history and current
frame operands before retaining the latest window. Exact retained metadata,
append descriptor and BF16 output sizes join the aggregate state budget.
Single-request contiguous positions and explicit device reset define history;
CUDA block geometry and the correctness lowering's shared-score limit remain
outside the semantic plan.

WindowCompressedKvPrecision composes that same bounded ring with a separate
retained compressed cache. Its pure read-set policy is either learned selected
compressed row IDs or every causally completed compressed row. Compressed rows
remain separate from window/current rows; selected IDs are never expanded into
raw tokens. One sink-softmax normalization covers the complete read set. Cache
publication errors prevent the window transaction from committing, and current
frame operands remain live until the combined attention consumes them.
