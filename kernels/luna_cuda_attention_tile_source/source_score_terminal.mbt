///|
fn cuda_score_terminal_effects(
  plan : @luna_attention_tile_schedule.AttentionScoreTerminalPlan,
  point : @luna_attention_tile_schedule.AttentionScoreTerminalPoint,
  offsets : ReadOnlyArray[Int],
) -> String {
  plan
  .actions(point)
  .map(effect => {
    match effect {
      StoreResidentOutput => cuda_resident_output_fold_store()
      StoreResidentStatistics => cuda_query_fold_store(offsets)
      PublishTerminalWrites => "  __syncthreads();\n"
      ReleaseTerminalReaders => "  __syncthreads();"
    }
  })
  .join("")
}
