///|
/// Startup-only batch envelope for the bounded paged Mistral plan.
pub struct MistralPagedBatchEnvelope {
  priv max_batch_rows : Int
} derive(Debug, Eq)

///|
pub fn MistralPagedBatchEnvelope::new(
  max_batch_rows : Int,
) -> MistralPagedBatchEnvelope raise @plan.PlanValidationError {
  // The existing dense operation graph owns one canonical validation rule for
  // its batch envelope. This value exposes no Llama-family type to callers.
  ignore(@llama.LlamaPagedBatchEnvelope::new(max_batch_rows))
  { max_batch_rows, }
}

///|
pub fn MistralPagedBatchEnvelope::single_row() -> Self {
  { max_batch_rows: 1 }
}

///|
pub fn MistralPagedBatchEnvelope::max_batch_rows(self : Self) -> Int {
  self.max_batch_rows
}

///|
fn dense_core_metadata(
  metadata : @mistral_spec.MistralModelMetadata,
) -> @spec.LlamaModelMetadata {
  let source = metadata.spec()
  let dense = try! @spec.LlamaModelSpec::new(
    vocabulary_size=source.vocabulary_size(),
    hidden_size=source.hidden_size(),
    intermediate_size=source.intermediate_size(),
    layer_count=source.layer_count(),
    attention_head_count=source.attention_head_count(),
    kv_head_count=source.kv_head_count(),
    context_length=source.context_length(),
    rope_theta=source.rope_theta(),
    rms_norm_epsilon=source.rms_norm_epsilon(),
    weight_dtype=source.weight_dtype(),
    kv_dtype=source.kv_dtype(),
  )
  try! @spec.LlamaModelMetadata::from_verified_content(
    dense,
    metadata.content_digest(),
  )
}

///|
/// Builds the stateless full-context plan for the admitted Mistral profile.
///
/// Family selection ends here. The output is the same architecture-neutral
/// dense operation vocabulary used by the existing executor and contains no
/// Mistral, sliding-window, scheduler, KV-policy, device, or kernel branch.
pub fn build(
  metadata : @mistral_spec.MistralModelMetadata,
) -> @plan.ModelPlan raise @plan.PlanValidationError {
  @llama.build(dense_core_metadata(metadata))
}

///|
/// Builds a paged plan whose supported context never exceeds one declared
/// Mistral sliding window. Ordinary full causal attention is therefore exact.
pub fn build_paged(
  metadata : @mistral_spec.MistralModelMetadata,
  batch? : MistralPagedBatchEnvelope = MistralPagedBatchEnvelope::single_row(),
) -> @plan.ModelPlan raise @plan.PlanValidationError {
  let dense_batch = @llama.LlamaPagedBatchEnvelope::new(batch.max_batch_rows())
  @llama.build_paged(dense_core_metadata(metadata), batch=dense_batch)
}
