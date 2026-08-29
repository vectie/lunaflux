///|
/// Bounded rule vocabulary for backend-free numeric admission failures.
pub(all) enum NumericContractRule {
  Identifier
  ShapeDimension
  ShapeArithmetic
  StorageArithmetic
  UnsupportedValue
  StorageCombination
  ExecutionCombination
  CanonicalBytes
  Digest
} derive(Debug, Eq)

///|
/// Bounded schema resource whose checked authority was exceeded.
pub(all) enum NumericSchemaLimit {
  SchemaTensorCount
  SchemaOperationCount
  SchemaRank
  SchemaCanonicalBytes
} derive(Debug, Eq)

///|
/// Payload-safe failures emitted by the numeric contract package.
pub(all) suberror NumericContractError {
  InvalidNumericContract(NumericContractRule)
  TensorStorageDigestMismatch
  OperationExecutionDigestMismatch
  NumericSchemaLimitExceeded(NumericSchemaLimit)
} derive(Debug, Eq)
