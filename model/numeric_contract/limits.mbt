///|
const MODEL_NUMERIC_SCHEMA_HARD_MAX_TENSORS : Int = 65536

///|
const MODEL_NUMERIC_SCHEMA_HARD_MAX_OPERATIONS : Int = 4096

///|
const MODEL_NUMERIC_SCHEMA_HARD_MAX_RANK : Int = 16

///|
const MODEL_NUMERIC_SCHEMA_HARD_MAX_CANONICAL_BYTES : Int = 16 * 1024 * 1024

///|
/// Checked, non-forgeable resource authority for numeric schema construction.
#valtype
pub struct ModelNumericSchemaLimits {
  priv max_tensors : Int
  priv max_operations : Int
  priv max_rank : Int
  priv max_canonical_bytes : Int
}

///|
pub fn ModelNumericSchemaLimits::new(
  max_tensors~ : Int,
  max_operations~ : Int,
  max_rank~ : Int,
  max_canonical_bytes~ : Int,
) -> ModelNumericSchemaLimits raise NumericContractError {
  if max_tensors < 0 || max_tensors > MODEL_NUMERIC_SCHEMA_HARD_MAX_TENSORS {
    raise NumericSchemaLimitExceeded(SchemaTensorCount)
  }
  if max_operations < 0 ||
    max_operations > MODEL_NUMERIC_SCHEMA_HARD_MAX_OPERATIONS {
    raise NumericSchemaLimitExceeded(SchemaOperationCount)
  }
  if max_rank < 0 || max_rank > MODEL_NUMERIC_SCHEMA_HARD_MAX_RANK {
    raise NumericSchemaLimitExceeded(SchemaRank)
  }
  if max_canonical_bytes < 49 ||
    max_canonical_bytes > MODEL_NUMERIC_SCHEMA_HARD_MAX_CANONICAL_BYTES {
    raise NumericSchemaLimitExceeded(SchemaCanonicalBytes)
  }
  { max_tensors, max_operations, max_rank, max_canonical_bytes }
}

///|
pub fn ModelNumericSchemaLimits::production() -> ModelNumericSchemaLimits {
  try! ModelNumericSchemaLimits::new(
    max_tensors=MODEL_NUMERIC_SCHEMA_HARD_MAX_TENSORS,
    max_operations=MODEL_NUMERIC_SCHEMA_HARD_MAX_OPERATIONS,
    max_rank=MODEL_NUMERIC_SCHEMA_HARD_MAX_RANK,
    max_canonical_bytes=MODEL_NUMERIC_SCHEMA_HARD_MAX_CANONICAL_BYTES,
  )
}
