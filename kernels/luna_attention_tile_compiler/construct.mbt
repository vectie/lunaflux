///|
pub fn AttentionTileCompilerRequest::new(
  backend~ : @luna_attention_strategy.AttentionBackendId,
  target~ : @luna_attention_strategy.AttentionTargetId,
  problem~ : @luna_attention_strategy.AttentionCompileProblem,
  capabilities~ : @luna_attention_strategy.AttentionCompilerCapabilities,
  autotune_records : ArrayView[@luna_attention_strategy.AttentionAutotuneRecord],
) -> AttentionTileCompilerRequest {
  {
    backend,
    target,
    problem,
    capabilities,
    autotune_records: ReadOnlyArray::from_array(autotune_records),
  }
}
