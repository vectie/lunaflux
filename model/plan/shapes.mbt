///|
/// Maximum legal runtime shape and model dimensions for one immutable plan.
pub struct ShapeConstraints {
  priv max_batch_rows : Int32
  priv max_sequence_tokens : Int32
  priv vocabulary_size : Int32
  priv hidden_size : Int32
  priv intermediate_size : Int32
  priv attention_head_count : Int32
  priv kv_head_count : Int32
  priv head_dimension : Int32
} derive(Debug, Eq)

///|
/// Upper bound on temporary storage needed by any legal execution of a plan.
pub struct WorkspaceBounds {
  priv max_temporary_bytes : Int64
} derive(Debug, Eq)

///|
/// Semantic dimensions attached to an operation.
pub(all) enum OperationShape {
  EmbeddingShape(vocabulary_size~ : Int32, width~ : Int32)
  NormalizationShape(width~ : Int32, epsilon~ : Double)
  QkNormalizationShape(
    query_heads~ : Int32,
    kv_heads~ : Int32,
    head_dimension~ : Int32,
    epsilon~ : Double
  )
  ProjectionShape(input_width~ : Int32, output_width~ : Int32)
  RotaryShape(
    query_heads~ : Int32,
    kv_heads~ : Int32,
    head_dimension~ : Int32,
    rope_theta~ : Double
  )
  AttentionShape(
    query_heads~ : Int32,
    kv_heads~ : Int32,
    head_dimension~ : Int32
  )
  ElementwiseShape(width~ : Int32)
  GatedMlpShape(hidden_size~ : Int32, intermediate_size~ : Int32)
} derive(Debug, Eq)

///|
pub fn ShapeConstraints::new(
  max_batch_rows~ : Int32,
  max_sequence_tokens~ : Int32,
  vocabulary_size~ : Int32,
  hidden_size~ : Int32,
  intermediate_size~ : Int32,
  attention_head_count~ : Int32,
  kv_head_count~ : Int32,
  head_dimension~ : Int32,
) -> ShapeConstraints raise PlanValidationError {
  if max_batch_rows <= 0 ||
    max_sequence_tokens <= 0 ||
    vocabulary_size <= 0 ||
    hidden_size <= 0 ||
    intermediate_size <= 0 ||
    attention_head_count <= 0 ||
    kv_head_count <= 0 ||
    head_dimension <= 0 {
    raise InvalidShape(PositiveDimension)
  }
  if kv_head_count > attention_head_count ||
    attention_head_count % kv_head_count != 0 {
    raise InvalidShape(AttentionKvHeadRelationship)
  }
  if attention_head_count > 2147483647 / head_dimension ||
    kv_head_count > 2147483647 / head_dimension {
    raise InvalidShape(DimensionOverflow)
  }
  let query_width = attention_head_count * head_dimension
  let kv_width = kv_head_count * head_dimension
  if kv_width > (2147483647 - query_width) / 2 {
    raise InvalidShape(DimensionOverflow)
  }
  {
    max_batch_rows,
    max_sequence_tokens,
    vocabulary_size,
    hidden_size,
    intermediate_size,
    attention_head_count,
    kv_head_count,
    head_dimension,
  }
}

///|
pub fn WorkspaceBounds::new(
  max_temporary_bytes~ : Int64,
) -> WorkspaceBounds raise PlanValidationError {
  if max_temporary_bytes < 0L {
    raise InvalidWorkspaceBound
  }
  { max_temporary_bytes, }
}

///|
pub fn ShapeConstraints::max_batch_rows(self : ShapeConstraints) -> Int32 {
  self.max_batch_rows
}

///|
pub fn ShapeConstraints::max_sequence_tokens(self : ShapeConstraints) -> Int32 {
  self.max_sequence_tokens
}

///|
pub fn ShapeConstraints::vocabulary_size(self : ShapeConstraints) -> Int32 {
  self.vocabulary_size
}

///|
pub fn ShapeConstraints::hidden_size(self : ShapeConstraints) -> Int32 {
  self.hidden_size
}

///|
pub fn ShapeConstraints::intermediate_size(self : ShapeConstraints) -> Int32 {
  self.intermediate_size
}

///|
pub fn ShapeConstraints::attention_head_count(self : ShapeConstraints) -> Int32 {
  self.attention_head_count
}

///|
pub fn ShapeConstraints::kv_head_count(self : ShapeConstraints) -> Int32 {
  self.kv_head_count
}

///|
pub fn ShapeConstraints::head_dimension(self : ShapeConstraints) -> Int32 {
  self.head_dimension
}

///|
pub fn ShapeConstraints::query_width(self : ShapeConstraints) -> Int32 {
  self.attention_head_count * self.head_dimension
}

///|
pub fn ShapeConstraints::kv_width(self : ShapeConstraints) -> Int32 {
  self.kv_head_count * self.head_dimension
}

///|
pub fn WorkspaceBounds::max_temporary_bytes(self : WorkspaceBounds) -> Int64 {
  self.max_temporary_bytes
}

///|
pub extend ShapeConstraints with @debug.Debug::{to_repr}

///|
pub extend ShapeConstraints with Eq::{not_equal, equal}

///|
pub extend WorkspaceBounds with @debug.Debug::{to_repr}

///|
pub extend WorkspaceBounds with Eq::{not_equal, equal}

///|
pub extend OperationShape with @debug.Debug::{to_repr}

///|
pub extend OperationShape with Eq::{not_equal, equal}
