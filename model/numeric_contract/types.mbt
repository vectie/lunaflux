///|
/// SHA-256 identity of one canonical tensor-storage contract.
#valtype
pub struct TensorStorageDigest {
  priv value : String
} derive(Debug, Eq, Hash)

///|
/// SHA-256 identity of one canonical operation-execution contract.
#valtype
pub struct OperationExecutionDigest {
  priv value : String
} derive(Debug, Eq, Hash)

///|
/// SHA-256 identity of one complete model numeric schema.
#valtype
pub struct ModelNumericSchemaDigest {
  priv value : String
} derive(Debug, Eq, Hash)

///|
/// Stable tensor position inside a reference-free numeric schema.
#valtype
pub struct NumericTensorOrdinal {
  priv value : Int
} derive(Debug, Eq, Hash)

///|
/// Stable operation position inside a reference-free numeric schema.
#valtype
pub struct NumericOperationOrdinal {
  priv value : Int
} derive(Debug, Eq, Hash)

///|
/// Typed scale-metadata ordinal in the same numeric schema.
#valtype
pub struct ScaleTensorOrdinal {
  priv value : Int
} derive(Debug, Eq, Hash)

///|
/// Typed zero-point-metadata ordinal in the same numeric schema.
#valtype
pub struct ZeroPointTensorOrdinal {
  priv value : Int
} derive(Debug, Eq, Hash)

///|
/// Typed codebook-metadata ordinal in the same numeric schema.
#valtype
pub struct CodebookTensorOrdinal {
  priv value : Int
} derive(Debug, Eq, Hash)

///|
/// Exact conversion semantics applied before values enter storage or compute.
pub struct NumericConversion {
  priv rounding : RoundingPolicy
  priv saturation : SaturationPolicy
  priv finite_values : FiniteValuePolicy
} derive(Debug, Eq)

///|
/// Immutable representation contract for exactly one stored tensor.
///
/// Metadata requirements are reference-free representation semantics. The
/// model plan binds them to exact typed tensor references.
pub struct TensorStorageContract {
  priv dtype : StorageDType
  priv layout : StorageLayout
  priv encoding : StorageEncoding
  priv scale_granularity : ScaleGranularity
  priv zero_point : ZeroPointMode
  priv codebook_layout : CodebookLayout
  priv conversion : NumericConversion
  priv canonical_length : Int
  priv digest : TensorStorageDigest
} derive(Debug, Eq)

///|
/// Immutable compute contract for exactly one semantic operation.
///
/// Activation, stored-tensor, accumulator, and output precision are distinct
/// fields. This value cannot describe a whole model.
pub struct OperationExecutionContract {
  priv activation_input : ComputeDType
  priv activation_compute : ComputeDType
  priv activation_scale : ActivationScalePolicy
  priv tensor_input : ComputeDType
  priv accumulator : AccumulatorDType
  priv output : ComputeDType
  priv conversion : NumericConversion
  priv order : OperationOrderPolicy
  priv canonical_length : Int
  priv digest : OperationExecutionDigest
} derive(Debug, Eq)

///|
/// Generic role of one numeric-schema tensor entry.
#valtype
pub struct NumericTensorRole {
  priv value : Int
} derive(Debug, Eq, Hash)

///|
/// One reference-free tensor schema entry with stable metadata ordinals.
pub struct NumericTensorSchema {
  priv ordinal : NumericTensorOrdinal
  priv role : NumericTensorRole
  priv shape : NumericShape
  priv storage : TensorStorageContract
  priv scale : ScaleTensorOrdinal?
  priv zero_point : ZeroPointTensorOrdinal?
  priv codebook : CodebookTensorOrdinal?
} derive(Debug, Eq)

///|
/// One reference-free operation schema entry.
pub struct NumericOperationSchema {
  priv ordinal : NumericOperationOrdinal
  priv execution : OperationExecutionContract
} derive(Debug, Eq)

///|
/// Complete canonical numeric identity for one model plan.
pub struct ModelNumericSchema {
  priv tensors : ReadOnlyArray[NumericTensorSchema]
  priv operations : ReadOnlyArray[NumericOperationSchema]
  priv canonical_length : Int
  priv digest : ModelNumericSchemaDigest
} derive(Debug, Eq)
