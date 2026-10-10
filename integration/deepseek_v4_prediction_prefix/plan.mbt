///|
/// The model adapter supplies target count and checkpoint names. Row/precision
/// semantics and execution ownership remain shared across model families.
pub fn prediction_main_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  maximum_rows~ : Int,
  maximum_workspace_bytes~ : Int64,
) -> @precision.RowProjectionNormPrecision raise {
  guard spec.has_dspark() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  @precision.RowProjectionNormPrecision::new(
    input_width=spec.hidden_size() * spec.dspark_target_layer_ids().length(),
    output_width=spec.hidden_size(),
    maximum_rows~,
    normalization_epsilon=spec.rms_norm_epsilon(),
    activation_amax_floor=0.0001,
    maximum_workspace_bytes~,
  )
}

///|
pub fn bind_prediction_main_weights(
  checkpoint : @materialize.PackedCheckpoint,
  spec : @spec.DeepSeekV4ModelSpec,
) -> Array[@materialize.PackedCheckpointWeight] raise {
  let plan = prediction_main_plan(
    spec,
    maximum_rows=1,
    maximum_workspace_bytes=1048576L,
  )
  [
    checkpoint.bind_weight(
      representation=plan.weight_representation(),
      payload_name="mtp.0.main_proj.weight",
      scales_name="mtp.0.main_proj.scale",
    ),
    checkpoint.bind_vector(
      representation=@format.Representation::new(
        format=Bf16,
        rows=1,
        columns=spec.hidden_size(),
      ),
      payload_name="mtp.0.main_norm.weight",
    ),
  ]
}
