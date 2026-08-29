///|
/// Exact semantic batch-row ceiling for one dense-Llama paged plan.
///
/// The value is deliberately separate from scheduler capacity: a loaded plan
/// and every admitted execution artifact bind this ceiling into their model
/// identity before runtime scheduling begins.
pub struct LlamaPagedBatchEnvelope {
  priv max_batch_rows : Int
} derive(Debug, Eq)

///|
#warnings("-unnecessary_annotation")
pub fn LlamaPagedBatchEnvelope::new(
  max_batch_rows : Int,
) -> LlamaPagedBatchEnvelope raise @plan.PlanValidationError {
  if max_batch_rows <= 0 {
    raise @plan.InvalidShape(PositiveDimension)
  }
  { max_batch_rows, }
}

///|
pub fn LlamaPagedBatchEnvelope::single_row() -> LlamaPagedBatchEnvelope {
  { max_batch_rows: 1 }
}

///|
pub fn LlamaPagedBatchEnvelope::max_batch_rows(
  self : LlamaPagedBatchEnvelope,
) -> Int {
  self.max_batch_rows
}
