# Per-tensor and per-operation numeric contracts

`model/numeric_contract` owns reference-free, backend-free representation and
execution semantics. Precision is never selected once for an entire model.
Every stored tensor has an opaque `TensorStorageContract`; every semantic
operation has an opaque `OperationExecutionContract`. Graph-bound activation
representation is distinct from internal activation compute representation.
Finite E4M3 W8A8 binds BF16 graph input, finite E4M3 internal activation
compute, dynamic per-tensor F32 activation scaling, finite E4M3 weight input,
F32 accumulation, and BF16 graph output. The operation conversion is the
activation-input-to-compute conversion; weight conversion and stored scales
remain owned by each tensor storage contract.

Storage contracts distinguish dtype, layout, encoding, scale granularity,
zero-point mode, codebook layout, and conversion policy. They state whether a
scale, zero-point tensor, or codebook is required but own no model reference.
`ModelNumericSchema` composes those contracts into one complete immutable model
schema using validated stable tensor and operation ordinals. Tensor entries own
their exact role, shape, storage contract, and typed metadata ordinals;
operation entries own their exact execution contract. The generic model plan
maps those ordinals to its typed references without creating a second numeric
encoder or integer identity.

~~~mbt check
///|
test "construct one tensor-local representation" {
  let storage = @numeric_contract.TensorStorageContract::new(
    dtype=@numeric_contract.StorageDType::f8_e4m3_finite(),
    layout=@numeric_contract.StorageLayout::dense_row_major_v1(),
    encoding=@numeric_contract.StorageEncoding::scaled_v1(),
    scale_granularity=@numeric_contract.ScaleGranularity::per_tensor(),
    zero_point=@numeric_contract.ZeroPointMode::absent(),
    codebook_layout=@numeric_contract.CodebookLayout::absent(),
    conversion=@numeric_contract.NumericConversion::quantized_v1(),
  )
  assert_true(storage.requires_scale())
  assert_eq(storage.digest().as_hex().length(), 64)
}
~~~

Canonical storage, execution-v2, and complete-schema-v2 records use distinct
versioned domains. No v1 operation/schema loader fallback is accepted. The
schema SHA-256 identity covers tensor and operation
order, roles, shapes, exact nested contract bytes, and every metadata ordinal.
Storage and execution loaders require independent expected digests, exact
framing, checked reconstruction, and byte-identical re-encoding.

The package does not own semantic `TensorRef` values, parse model files, select
kernels, probe hardware, allocate memory, or grant runtime support.

## Dynamic finite-E4M3 activation policy v1

`dynamic_per_tensor_f32_v1` has one complete numerical meaning:

1. The tensor is the exact live logical activation input in canonical
   row-major element order. Every BF16 input must be finite. Convert each BF16
   value exactly to F32, canonicalizing both signed zeros to positive zero.
2. Reduce `max(abs(x))` by visiting that order once. Since inputs are finite
   and `max` never reassociates arithmetic, the result is deterministic.
3. For an all-zero tensor, the scale is exactly F32 `1.0`. Otherwise compute
   `scale = max_abs / 448.0` using IEEE-754 binary32 round-to-nearest,
   ties-to-even. The scale must be finite and strictly positive.
4. For each element, compute `x / scale` in binary32 with the same rounding,
   clamp to the finite E4M3FN interval `[-448, 448]`, and round to the nearest
   finite E4M3FN value with ties to an even significand. Subnormal E4M3FN
   values are retained; no NaN or infinity code may be emitted; zero is
   canonical positive zero.
5. The effective reconstructed activation is the exact F32 decode of that
   E4M3FN value multiplied by the F32 scale with binary32
   round-to-nearest/ties-to-even. Applying a separately bound tensor scale and
   the operation's declared accumulation/order policy remains part of the
   kernel numerical contract.

The F32 scale is runtime step data. It is neither model metadata nor a hidden
artifact constant. Any alternative reduction scope, zero scale, FP8 format,
finite maximum, clamp, rounding, subnormal, or reconstruction convention
requires a new `ActivationScalePolicy` identity.

`dynamic_per_tensor_f32_v1` owns exactly one ordered stage,
`external_operation_input_v1`. It is sufficient for a simple projection. It
does not authorize one scale to stand in for two distinct live tensors inside
a gated MLP.

## Compound gated-MLP activation policy v1

`gated_mlp_external_and_post_silu_f32_v1` owns exactly two ordered, independent
stages:

1. `external_operation_input_v1` is the live BF16 gated-MLP input `X`, visited
   in token-row-major then hidden-column order. Gate and up projections both
   consume the finite-E4M3 reconstruction of `X` produced by the dynamic
   policy above, using the first scale and their independently bound scalar
   F32 weight scales.
2. `post_silu_gate_up_product_v1` is the live F32 tensor `Z`, visited in
   token-row-major then intermediate-column order. For each element, gate `g`
   and up `u` are strict increasing-inner-index F32 dot products. Define the
   correctly rounded binary32 exponential `e = round_f32(exp(real(-g)))`, then
   `d = round_f32(1.0 + e)`, `a = round_f32(g / d)`, and
   `Z = round_f32(a * u)`, all round-to-nearest/ties-to-even. Every `Z` must be
   finite. Apply the same deterministic maximum, zero-tensor, division,
   clamp, finite-E4M3FN rounding, and reconstruction rules above to `Z`, but
   derive a new second scale. The down projection consumes only this second
   reconstruction and its independently bound scalar F32 weight scale.

The two reductions remain distinct even when their resulting F32 bit patterns
are equal. The down projection accumulates in increasing intermediate index
under the declared operation-order policy and rounds its output once to the
declared output type. Any non-finite live input, gate, up, `Z`, scale, or
accumulator is an execution failure and cannot publish a successful result.
Padding and any cached context excluded by the exact execution shape are not
live tensor elements. A kernel-specific exponential implementation, compiler,
and target must be bound separately and must prove that it implements this
numeric contract; recognizing this inert policy is not execution readiness.

`gated_mlp_external_and_post_silu_bound_expf_f32_v2` preserves the same two
ordered tensors, reductions, finite-E4M3 conversion, and failure rules, but it
does not claim the ideal correctly-rounded real exponential above. Instead,
`e` is the exact binary32 bit pattern returned by one separately authenticated
target/compiler-bound `expf(-g)` implementation. The kernel source digest,
compiler policy, target, and AOT recipe must all be joined before this policy
has executable meaning. Different `expf` implementations or compiler/target
identities require distinct joined kernel evidence. This policy therefore
cannot replay as the ideal v1 compound policy and remains inert by itself.

## Symmetric-I8 weight-only policy v1

`symmetric_i8_weight_only_v1` is one exact tensor-local representation and
execution profile. It is not a model-wide precision flag. A selected weight is
a dense row-major rank-two matrix `[out_channels, in_channels]` with signed
codes in the closed interval `[-127, 127]`. Code `-128` is reserved and must be
rejected; accepting it requires a future format identity. The zero-point is
exactly integer zero and is implicit only. An affine zero-point tensor and a
codebook are never equivalent to this profile.

Every selected weight owns one distinct typed scale-metadata reference. The
target is a dense, plain-F32, rank-one tensor with exact shape
`[out_channels]`. There is no scalar scale, alternate axis, rank-two scale, or
shared scale reference. For each output row, the complete deterministic
meaning is:

1. Visit finite source values in canonical row-major order and reduce
   `max(abs(w))` without reassociation, after exact conversion to F32 and
   canonicalization of signed zero to positive zero.
2. For an all-zero row, emit exactly F32 `1.0`. Otherwise compute
   `scale = max_abs / 127.0` in IEEE-754 binary32 round-to-nearest,
   ties-to-even. Every persisted scale must be finite and strictly positive.
3. For each element, divide by that row's F32 scale in binary32, clamp to
   `[-127, 127]`, then round to the nearest integer with ties-to-even. Code
   `-128` must never be emitted or admitted.
4. The effective reconstructed weight is `F32(code) * scale`, evaluated in
   binary32 round-to-nearest, ties-to-even. The operation keeps BF16 graph and
   internal activation inputs, has no activation scale, accumulates in F32,
   produces BF16, and preserves strict declared operation order.

Any other code interval, zero-row scale, reduction axis, F32 rounding,
integer rounding, clamp, reconstruction, zero-point, or metadata ownership
convention requires a new storage/execution identity. This package freezes
the semantics and typed schema only; it does not parse payloads, materialize
weights, select a kernel, or claim device readiness.
