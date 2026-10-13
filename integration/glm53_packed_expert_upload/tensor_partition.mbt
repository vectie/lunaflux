///|
/// GLM binds names only. The pure generic plan partitions each expert's
/// intermediate dimension and specifies the pre-round collective boundary.
pub fn tensor_expert_compute_plan(
  layer_index~ : Int,
  maximum_rows~ : Int,
  rank~ : Int,
  world_size~ : Int,
  accuracy~ : @precision_ir.Accuracy,
  matrix_law? : @precision_ir.DenseDotLaw = ScalarFloat32,
  memory_ceiling~ : Int64,
) -> @precision_ir.ExpertTensorPartition raise {
  // The full plan is metadata, not a full allocation. Its capacity ceiling
  // does not imply that the full bank is ever loaded on a rank.
  let full = expert_compute_plan(
    layer_index~,
    expert_indices=Array::makei(288, i => i),
    maximum_rows~,
    accuracy~,
    matrix_law~,
    memory_ceiling=9223372036854775807L,
  )
  @precision_ir.ExpertTensorPartition::new(
    full,
    rank~,
    world_size~,
    memory_ceiling~,
  )
}

///|
pub fn bind_tensor_experts(
  checkpoint : @precision_materialize.PackedCheckpoint,
  layer_index~ : Int,
  partition~ : @precision_ir.ExpertTensorPartition,
) -> Array[@precision_materialize.PackedCheckpointWeight] raise {
  let weights = bind_experts(
    checkpoint,
    layer_index~,
    expert_indices=partition.rank_plan().local_experts(),
  )
  weights.mapi((index, weight) => weight.slice(partition.slices()[index % 3]))
}

///|
pub fn upload_tensor_expert_bank(
  context : @device.Context,
  checkpoint : @precision_materialize.PackedCheckpoint,
  root : @approved_fs.ApprovedRoot,
  partition : @precision_ir.ExpertTensorPartition,
  bank : @device.Allocation,
  layer_index~ : Int,
  chunk_bytes~ : Int,
  limits~ : @streaming_safetensors.StreamingSafetensorsLimits,
) -> Unit raise {
  @packed_expert_mlp.upload_bank(
    context,
    checkpoint,
    root,
    partition.rank_plan(),
    bank,
    weights=bind_tensor_experts(checkpoint, layer_index~, partition~),
    chunk_bytes~,
    limits~,
  )
}
