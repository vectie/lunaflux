///|
fn prediction_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  rows : Int,
  history : Int,
  ceiling : Int64,
) -> @prediction.DeepSeekPredictionExecution raise {
  @prediction.DeepSeekPredictionExecution::new(
    spec,
    maximum_main_rows=rows,
    maximum_positions=history,
    maximum_workspace_bytes=ceiling,
    maximum_expert_bytes=ceiling,
    temperature=0.0,
    accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
  )
}

///|
fn prediction_bytes(
  spec : @spec.DeepSeekV4ModelSpec,
  checkpoint : @materialize.PackedCheckpoint,
  rows : Int,
  history : Int,
  ceiling : Int64,
  enabled~ : Bool,
) -> Int64 raise {
  if !enabled {
    return 0L
  }
  let owner = prediction_plan(spec, rows, history, ceiling)
  defer owner.close()
  owner.required_device_bytes(checkpoint)
}
