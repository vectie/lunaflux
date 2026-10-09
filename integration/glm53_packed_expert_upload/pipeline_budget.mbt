///|
/// Bind exact checkpoint footprints without uploading or expanding weights.
/// State and private layer workspaces are included in each entry.
pub fn GlmDecoderSlice::layer_device_bytes(
  self : Self,
  checkpoint : @precision_materialize.PackedCheckpoint,
) -> Array[Int64] raise {
  self.layers.map(layer => layer.required_device_bytes(checkpoint))
}

///|
/// Partition a complete decoder plan using actual checkpoint byte costs.
/// Four residual frames per stage conservatively cover distinct boundary
/// input/output plus at most two internal alternates. Additional stage bytes
/// account for request ports, module storage and the caller's host/device reserve.
/// Returned placements are memory-feasible, not balanced performance promises.
pub fn GlmDecoderSlice::partition(
  self : Self,
  checkpoint : @precision_materialize.PackedCheckpoint,
  device_budgets : ArrayView[Int64],
  ingress_bytes~ : Int64,
  egress_bytes~ : Int64,
  additional_stage_bytes~ : Int64,
) -> @pipeline.DecoderPipelinePlan raise {
  guard self.first_layer == 0 && self.end_layer == self.model_layer_count else {
    raise @precision_ir.PrecisionPlanError::IllegalPlacement
  }
  guard additional_stage_bytes >= 0L &&
    additional_stage_bytes <= 9223372036854775807L - 4L * self.residual_bytes else {
    raise @precision_ir.PrecisionPlanError::MemoryBudgetExceeded
  }
  @pipeline.DecoderPipelinePlan::partition(
    self.layer_device_bytes(checkpoint),
    device_budgets,
    stage_overhead_bytes=additional_stage_bytes + 4L * self.residual_bytes,
    ingress_bytes~,
    egress_bytes~,
  )
}

///|
/// Global text costs for complete-model placement. No embedding/head duplication
/// across hosts: normalized/logit workspace is counted only at the last stage.
pub fn GlmTextIo::boundary_device_bytes(
  self : Self,
  checkpoint : @precision_materialize.PackedCheckpoint,
) -> (Int64, Int64) raise {
  guard self.boundary is Complete else {
    raise @precision_ir.PrecisionPlanError::IllegalPlacement
  }
  let weights = bind_text_io_weights(checkpoint, self.spec)
  (
    weights[0].buffer_bytes(),
    weights[1].buffer_bytes() +
    weights[2].buffer_bytes() +
    self.frame.workspace_bytes(),
  )
}
