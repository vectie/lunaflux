///|
/// Opaque stored representation. Recognition grants no kernel authority.
#valtype
pub struct StorageDType {
  priv value : Int
} derive(Debug, Eq, Hash)

///|
pub fn StorageDType::bfloat16() -> StorageDType {
  { value: 1 }
}

///|
pub fn StorageDType::f8_e4m3_finite() -> StorageDType {
  { value: 2 }
}

///|
pub fn StorageDType::i8() -> StorageDType {
  { value: 3 }
}

///|
pub fn StorageDType::f32() -> StorageDType {
  { value: 4 }
}

///|
pub fn StorageDType::byte_width(self : StorageDType) -> Int64 {
  match self.value {
    1 => 2L
    2 | 3 => 1L
    4 => 4L
    _ => 0L
  }
}

///|
/// Canonical physical ordering for one stored tensor.
#valtype
pub struct StorageLayout {
  priv value : Int
} derive(Debug, Eq, Hash)

///|
pub fn StorageLayout::dense_row_major_v1() -> StorageLayout {
  { value: 1 }
}

///|
/// Storage encoding is separate from dtype and applies to one tensor only.
#valtype
pub struct StorageEncoding {
  priv value : Int
} derive(Debug, Eq, Hash)

///|
pub fn StorageEncoding::plain_v1() -> StorageEncoding {
  { value: 1 }
}

///|
pub fn StorageEncoding::scaled_v1() -> StorageEncoding {
  { value: 2 }
}

///|
pub fn StorageEncoding::symmetric_v1() -> StorageEncoding {
  { value: 3 }
}

///|
pub fn StorageEncoding::affine_v1() -> StorageEncoding {
  { value: 4 }
}

///|
pub fn StorageEncoding::codebook_v1() -> StorageEncoding {
  { value: 5 }
}

///|
/// Compute representation for an operation input or output. `absent` is legal
/// only when the operation has no value in that role.
#valtype
pub struct ComputeDType {
  priv value : Int
} derive(Debug, Eq, Hash)

///|
pub fn ComputeDType::absent() -> ComputeDType {
  { value: 0 }
}

///|
pub fn ComputeDType::bfloat16() -> ComputeDType {
  { value: 1 }
}

///|
pub fn ComputeDType::f8_e4m3_finite() -> ComputeDType {
  { value: 2 }
}

///|
pub fn ComputeDType::i8() -> ComputeDType {
  { value: 3 }
}

///|
pub fn ComputeDType::f32() -> ComputeDType {
  { value: 4 }
}

///|
/// Activation scaling required before one operation's internal compute.
/// Weight scales remain tensor-local metadata in `TensorStorageContract`.
#valtype
pub struct ActivationScalePolicy {
  priv value : Int
} derive(Debug, Eq, Hash)

///|
/// Stable semantic identity for one dynamic activation-scale reduction stage.
///
/// A stage is numeric vocabulary only. It is not storage, a device address,
/// kernel authority, or evidence that an implementation exists.
#valtype
pub struct ActivationScaleStage {
  priv value : Int
} derive(Debug, Eq, Hash)

///|
/// The exact external operation input consumed by a simple projection, or by
/// both the gate and up projections of a gated MLP.
pub fn ActivationScaleStage::external_operation_input_v1() -> ActivationScaleStage {
  { value: 1 }
}

///|
/// The exact post-SiLU gate/up product consumed by the down projection of a
/// gated MLP.
pub fn ActivationScaleStage::post_silu_gate_up_product_v1() -> ActivationScaleStage {
  { value: 2 }
}

///|
pub fn ActivationScalePolicy::absent_exact_v1() -> ActivationScalePolicy {
  { value: 0 }
}

///|
/// One finite F32 scale is derived dynamically for each live activation
/// tensor before finite E4M3 conversion. The complete deterministic reduction,
/// rounding, zero-tensor, clamp, and reconstruction convention is specified in
/// this package's README and is part of this v1 policy identity.
pub fn ActivationScalePolicy::dynamic_per_tensor_f32_v1() -> ActivationScalePolicy {
  { value: 1 }
}

///|
/// Two independent finite F32 scales are derived for a gated MLP: first for
/// its external gate/up input, then for its post-SiLU gate/up product before
/// the down projection. The ordered stages and complete numerical convention
/// are part of this policy identity and are specified in this package's README.
pub fn ActivationScalePolicy::gated_mlp_external_and_post_silu_f32_v1() -> ActivationScalePolicy {
  { value: 2 }
}

///|
/// Two-stage gated-MLP scaling whose SiLU exponential is the exact output of
/// a separately authenticated, target/compiler-bound binary32 `expf`
/// implementation. This identity never aliases the ideal correctly-rounded
/// real exponential required by the v1 compound policy.
pub fn ActivationScalePolicy::gated_mlp_external_and_post_silu_bound_expf_f32_v2() -> ActivationScalePolicy {
  { value: 3 }
}

///|
/// Returns the exact number of ordered dynamic-scale stages owned by a known
/// policy. Exact/no-scale policies return zero.
pub fn ActivationScalePolicy::stage_count(self : ActivationScalePolicy) -> Int {
  match self.value {
    0 => 0
    1 => 1
    2 | 3 => 2
    _ => 0
  }
}

///|
/// Returns one typed stage in canonical order without allocating an array.
pub fn ActivationScalePolicy::stage_at(
  self : ActivationScalePolicy,
  index : Int,
) -> ActivationScaleStage? {
  match (self.value, index) {
    (1 | 2 | 3, 0) => Some(ActivationScaleStage::external_operation_input_v1())
    (2 | 3, 1) => Some(ActivationScaleStage::post_silu_gate_up_product_v1())
    _ => None
  }
}

///|
/// Accumulator representation for exactly one operation.
#valtype
pub struct AccumulatorDType {
  priv value : Int
} derive(Debug, Eq, Hash)

///|
pub fn AccumulatorDType::f32() -> AccumulatorDType {
  { value: 1 }
}

///|
pub fn AccumulatorDType::i32() -> AccumulatorDType {
  { value: 2 }
}

///|
/// Conversion rounding semantics.
#valtype
pub struct RoundingPolicy {
  priv value : Int
} derive(Debug, Eq, Hash)

///|
pub fn RoundingPolicy::exact_no_rounding_v1() -> RoundingPolicy {
  { value: 0 }
}

///|
pub fn RoundingPolicy::nearest_ties_to_even_v1() -> RoundingPolicy {
  { value: 1 }
}

///|
/// Conversion saturation semantics.
#valtype
pub struct SaturationPolicy {
  priv value : Int
} derive(Debug, Eq, Hash)

///|
pub fn SaturationPolicy::none_v1() -> SaturationPolicy {
  { value: 0 }
}

///|
pub fn SaturationPolicy::finite_storage_range_v1() -> SaturationPolicy {
  { value: 1 }
}

///|
/// NaN and infinity handling before conversion or persistence.
#valtype
pub struct FiniteValuePolicy {
  priv value : Int
} derive(Debug, Eq, Hash)

///|
pub fn FiniteValuePolicy::reject_non_finite_v1() -> FiniteValuePolicy {
  { value: 1 }
}

///|
/// Floating-point operation-order authority for one operation.
#valtype
pub struct OperationOrderPolicy {
  priv value : Int
} derive(Debug, Eq, Hash)

///|
pub fn OperationOrderPolicy::strict_declared_order_v1() -> OperationOrderPolicy {
  { value: 1 }
}
