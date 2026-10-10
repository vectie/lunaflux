# Precision / representation refinement

Pure, immutable plans separate storage precision from activation/accumulator/
output precision. Conversion placement and accuracy bounds are semantic inputs.
Startup writes, ordinary step writes and transactional cache commits are explicit
effects. Quantization publication stages are planned before device lowering.

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
projection/score arithmetic, choice-only correction, group selection, stable
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
