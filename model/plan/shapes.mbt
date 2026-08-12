///|
/// Maximum legal runtime shape and model dimensions for one immutable plan.
pub struct ShapeConstraints {
  max_batch_rows : Int
  max_sequence_tokens : Int
  vocabulary_size : Int
  hidden_size : Int
  intermediate_size : Int
  attention_head_count : Int
  kv_head_count : Int
  head_dimension : Int
} derive(Debug, Eq)

///|
/// Upper bound on temporary storage needed by any legal execution of a plan.
pub struct WorkspaceBounds {
  max_temporary_bytes : Int64
} derive(Debug, Eq)

///|
/// Semantic dimensions attached to an operation.
pub(all) enum OperationShape {
  EmbeddingShape(vocabulary_size~ : Int, width~ : Int)
  NormalizationShape(width~ : Int, epsilon~ : Double)
  ProjectionShape(input_width~ : Int, output_width~ : Int)
  RotaryShape(
    query_heads~ : Int,
    kv_heads~ : Int,
    head_dimension~ : Int,
    rope_theta~ : Double
  )
  AttentionShape(query_heads~ : Int, kv_heads~ : Int, head_dimension~ : Int)
  ElementwiseShape(width~ : Int)
  GatedMlpShape(hidden_size~ : Int, intermediate_size~ : Int)
} derive(Debug, Eq)

///|
pub fn ShapeConstraints::new(
  max_batch_rows~ : Int,
  max_sequence_tokens~ : Int,
  vocabulary_size~ : Int,
  hidden_size~ : Int,
  intermediate_size~ : Int,
  attention_head_count~ : Int,
  kv_head_count~ : Int,
  head_dimension~ : Int,
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
  if hidden_size % attention_head_count != 0 ||
    hidden_size / attention_head_count != head_dimension {
    raise InvalidShape(HiddenHeadRelationship)
  }
  if kv_head_count > attention_head_count ||
    attention_head_count % kv_head_count != 0 {
    raise InvalidShape(AttentionKvHeadRelationship)
  }
  if kv_head_count > 2147483647 / head_dimension {
    raise InvalidShape(DimensionOverflow)
  }
  let kv_width = kv_head_count * head_dimension
  if kv_width > (2147483647 - hidden_size) / 2 {
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
pub fn ShapeConstraints::max_batch_rows(self : ShapeConstraints) -> Int {
  self.max_batch_rows
}

///|
pub fn ShapeConstraints::max_sequence_tokens(self : ShapeConstraints) -> Int {
  self.max_sequence_tokens
}

///|
pub fn ShapeConstraints::vocabulary_size(self : ShapeConstraints) -> Int {
  self.vocabulary_size
}

///|
pub fn ShapeConstraints::hidden_size(self : ShapeConstraints) -> Int {
  self.hidden_size
}

///|
pub fn ShapeConstraints::intermediate_size(self : ShapeConstraints) -> Int {
  self.intermediate_size
}

///|
pub fn ShapeConstraints::attention_head_count(self : ShapeConstraints) -> Int {
  self.attention_head_count
}

///|
pub fn ShapeConstraints::kv_head_count(self : ShapeConstraints) -> Int {
  self.kv_head_count
}

///|
pub fn ShapeConstraints::head_dimension(self : ShapeConstraints) -> Int {
  self.head_dimension
}

///|
pub fn WorkspaceBounds::max_temporary_bytes(self : WorkspaceBounds) -> Int64 {
  self.max_temporary_bytes
}
