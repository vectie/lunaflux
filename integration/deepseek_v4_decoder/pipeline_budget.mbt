///|
/// Exact compact layer costs, including each hash layer's checked I32 table.
/// This startup operation binds metadata only; it never allocates GPU weights.
pub fn DeepSeekDecoderSlice::layer_device_bytes(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
) -> Array[Int64] raise {
  let tables = @expert.bind_token_hash_tables(
    checkpoint,
    self.spec,
    first_layer=self.first_layer,
    end_layer=self.end_layer,
  )
  self.layers.mapi((index, layer) => {
    let table_bytes = if index < tables.length() {
      tables[index].i32_bytes()
    } else {
      0L
    }
    layer.required_device_bytes(self.spec, checkpoint) + table_bytes
  })
}

///|
/// Use the shared pure feasibility planner, preserving every base layer once.
/// Four residual frames cover both boundaries plus at most two internal frames.
/// Text weights live only at their owning end. Caller reserve includes borrowed
/// request ports/module/transfer storage. Prediction/MTP layers are not base
/// decoder layers and must be placed separately.
pub fn DeepSeekDecoderSlice::partition(
  self : Self,
  checkpoint : @materialize.PackedCheckpoint,
  device_budgets : ArrayView[Int64],
  ingress_bytes~ : Int64,
  egress_bytes~ : Int64,
  additional_stage_bytes~ : Int64,
) -> @pipeline.DecoderPipelinePlan raise {
  guard self.first_layer == 0 && self.end_layer == self.spec.layer_count() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  guard additional_stage_bytes >= 0L &&
    additional_stage_bytes <= 9223372036854775807L - 4L * self.residual_bytes else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  @pipeline.DecoderPipelinePlan::partition(
    self.layer_device_bytes(checkpoint),
    device_budgets,
    stage_overhead_bytes=additional_stage_bytes + 4L * self.residual_bytes,
    ingress_bytes~,
    egress_bytes~,
  )
}
