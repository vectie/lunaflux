///|
pub extend NumericContractRule with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend NumericContractRule with Eq::{not_equal, equal}

///|
pub extend NumericSchemaLimit with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend NumericSchemaLimit with Eq::{not_equal, equal}

///|
pub extend NumericContractError with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend NumericContractError with Eq::{not_equal, equal}

///|
pub extend ScaleGranularity with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ScaleGranularity with Eq::{not_equal, equal}

///|
pub extend ScaleGranularity with Hash::{hash, hash_combine}

///|
pub extend ZeroPointMode with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ZeroPointMode with Eq::{not_equal, equal}

///|
pub extend ZeroPointMode with Hash::{hash, hash_combine}

///|
pub extend CodebookLayout with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend CodebookLayout with Eq::{not_equal, equal}

///|
pub extend CodebookLayout with Hash::{hash, hash_combine}

///|
pub extend NumericShape with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend NumericShape with Eq::{not_equal, equal}

///|
pub extend TensorStorageDigest with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend TensorStorageDigest with Eq::{not_equal, equal}

///|
pub extend TensorStorageDigest with Hash::{hash, hash_combine}

///|
pub extend OperationExecutionDigest with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend OperationExecutionDigest with Eq::{not_equal, equal}

///|
pub extend OperationExecutionDigest with Hash::{hash, hash_combine}

///|
pub extend ModelNumericSchemaDigest with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend ModelNumericSchemaDigest with Eq::{not_equal, equal}

///|
pub extend ModelNumericSchemaDigest with Hash::{hash, hash_combine}

///|
pub extend NumericTensorOrdinal with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend NumericTensorOrdinal with Eq::{not_equal, equal}

///|
pub extend NumericTensorOrdinal with Hash::{hash, hash_combine}

///|
pub extend NumericOperationOrdinal with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend NumericOperationOrdinal with Eq::{not_equal, equal}

///|
pub extend NumericOperationOrdinal with Hash::{hash, hash_combine}

///|
pub extend ScaleTensorOrdinal with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ScaleTensorOrdinal with Eq::{not_equal, equal}

///|
pub extend ScaleTensorOrdinal with Hash::{hash, hash_combine}

///|
pub extend ZeroPointTensorOrdinal with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ZeroPointTensorOrdinal with Eq::{not_equal, equal}

///|
pub extend ZeroPointTensorOrdinal with Hash::{hash, hash_combine}

///|
pub extend CodebookTensorOrdinal with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend CodebookTensorOrdinal with Eq::{not_equal, equal}

///|
pub extend CodebookTensorOrdinal with Hash::{hash, hash_combine}

///|
pub extend NumericConversion with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend NumericConversion with Eq::{not_equal, equal}

///|
pub extend TensorStorageContract with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend TensorStorageContract with Eq::{not_equal, equal}

///|
pub extend OperationExecutionContract with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend OperationExecutionContract with Eq::{not_equal, equal}

///|
pub extend NumericTensorRole with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend NumericTensorRole with Eq::{not_equal, equal}

///|
pub extend NumericTensorRole with Hash::{hash, hash_combine}

///|
pub extend NumericTensorSchema with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend NumericTensorSchema with Eq::{not_equal, equal}

///|
pub extend NumericOperationSchema with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend NumericOperationSchema with Eq::{not_equal, equal}

///|
pub extend ModelNumericSchema with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ModelNumericSchema with Eq::{not_equal, equal}

///|
pub extend StorageDType with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend StorageDType with Eq::{not_equal, equal}

///|
pub extend StorageDType with Hash::{hash, hash_combine}

///|
pub extend StorageLayout with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend StorageLayout with Eq::{not_equal, equal}

///|
pub extend StorageLayout with Hash::{hash, hash_combine}

///|
pub extend StorageEncoding with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend StorageEncoding with Eq::{not_equal, equal}

///|
pub extend StorageEncoding with Hash::{hash, hash_combine}

///|
pub extend ComputeDType with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ComputeDType with Eq::{not_equal, equal}

///|
pub extend ComputeDType with Hash::{hash, hash_combine}

///|
pub extend ActivationScalePolicy with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ActivationScalePolicy with Eq::{not_equal, equal}

///|
pub extend ActivationScalePolicy with Hash::{hash, hash_combine}

///|
pub extend ActivationScaleStage with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ActivationScaleStage with Eq::{not_equal, equal}

///|
pub extend ActivationScaleStage with Hash::{hash, hash_combine}

///|
pub extend AccumulatorDType with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AccumulatorDType with Eq::{not_equal, equal}

///|
pub extend AccumulatorDType with Hash::{hash, hash_combine}

///|
pub extend RoundingPolicy with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend RoundingPolicy with Eq::{not_equal, equal}

///|
pub extend RoundingPolicy with Hash::{hash, hash_combine}

///|
pub extend SaturationPolicy with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend SaturationPolicy with Eq::{not_equal, equal}

///|
pub extend SaturationPolicy with Hash::{hash, hash_combine}

///|
pub extend FiniteValuePolicy with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend FiniteValuePolicy with Eq::{not_equal, equal}

///|
pub extend FiniteValuePolicy with Hash::{hash, hash_combine}

///|
pub extend OperationOrderPolicy with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend OperationOrderPolicy with Eq::{not_equal, equal}

///|
pub extend OperationOrderPolicy with Hash::{hash, hash_combine}
