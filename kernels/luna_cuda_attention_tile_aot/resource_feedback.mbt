///|
/// Carry offline resource facts through the terminal backend, without making
/// CUDA source generation responsible for portable schedule selection.
pub fn CudaAttentionTileAotRequest::with_resource_feedback(
  self : CudaAttentionTileAotRequest,
  budget : @luna_attention_tile_compiler.AttentionResourceBudget,
  observations : ArrayView[
    @luna_attention_tile_compiler.AttentionRegisterObservation,
  ],
) -> CudaAttentionTileAotRequest raise CudaAttentionTileAotError {
  let updated = {
    ..self,
    resource_budget: Some(budget),
    register_observations: ReadOnlyArray::from_array(observations),
  }
  ignore(updated.functional_request([]))
  updated
}

///|
pub fn CudaPartitionedAttentionTileAotRequest::with_resource_feedback(
  self : CudaPartitionedAttentionTileAotRequest,
  budget : @luna_attention_tile_compiler.AttentionResourceBudget,
  observations : ArrayView[
    @luna_attention_tile_compiler.AttentionRegisterObservation,
  ],
) -> CudaPartitionedAttentionTileAotRequest raise CudaAttentionTileAotError {
  let updated = {
    ..self,
    resource_budget: Some(budget),
    register_observations: ReadOnlyArray::from_array(observations),
  }
  ignore(updated.functional_request([]))
  updated
}

///|
fn CudaAttentionTileAotRequest::functional_request(
  self : CudaAttentionTileAotRequest,
  records : ArrayView[@luna_attention_strategy.AttentionAutotuneRecord],
) -> @luna_attention_tile_compiler.AttentionTileCompilerRequest raise CudaAttentionTileAotError {
  let request = @luna_attention_tile_compiler.AttentionTileCompilerRequest::new(
    backend=self.backend,
    target=self.target_id,
    problem=self.problem,
    capabilities=self.capabilities,
    records,
  )
  match self.resource_budget {
    None => request
    Some(budget) =>
      request.with_resource_feedback(budget, self.register_observations) catch {
        _ => raise InvalidCudaAttentionTileAot(InvalidRequest)
      }
  }
}

///|
fn CudaPartitionedAttentionTileAotRequest::functional_request(
  self : CudaPartitionedAttentionTileAotRequest,
  records : ArrayView[@luna_attention_strategy.AttentionAutotuneRecord],
) -> @luna_attention_tile_compiler.AttentionTileCompilerRequest raise CudaAttentionTileAotError {
  let request = @luna_attention_tile_compiler.AttentionTileCompilerRequest::new(
    backend=self.backend,
    target=self.target_id,
    problem=self.problem,
    capabilities=self.capabilities,
    records,
  )
  match self.resource_budget {
    None => request
    Some(budget) =>
      request.with_resource_feedback(budget, self.register_observations) catch {
        _ => raise InvalidCudaAttentionTileAot(InvalidRequest)
      }
  }
}
