///|
fn cuda_grouped_effect_plan(
  lowering : @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering,
) -> @luna_attention_tile_schedule.AttentionGroupedEffectPlan raise CudaAttentionTileSourceError {
  let memory = match lowering.copy() {
    CudaCooperativeSharedLoad => @luna_attention_strategy.AttentionStagedShared
    CudaAsynchronousSharedLoad => AttentionAsyncStagedShared
    _ => raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  @luna_attention_tile_schedule.AttentionGroupedEffectPlan::new(
    memory,
    stages=lowering.pipeline().0,
    readiness=lowering.transfer_readiness(),
  ) catch {
    _ => raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
}

///|
/// This renderer realizes actions in the common plan's order. In particular,
/// it cannot prefetch over a live slot or move the validity exit before V's wait.
fn cuda_grouped_effects(
  plan : @luna_attention_tile_schedule.AttentionGroupedEffectPlan,
  point : @luna_attention_tile_schedule.AttentionGroupedEffectPoint,
  transfer : String,
  indent : String,
) -> String {
  plan
  .actions(point)
  .map(action => {
    let text = match action {
      ResetStageValidity =>
        "if (thread == 0) { stage_valid[0] = 1; stage_valid[1] = 1; }"
      ResetTileValidity => "if (thread == 0) *tile_valid = 1;"
      PublishWorkgroup => "__syncthreads();"
      StageFirstPair => "if (key_begin < key_limit) stage_tile(key_begin, 0);"
      StageCurrentPair => return transfer
      AwaitPendingTransfers(groups) =>
        "asm volatile(\"cp.async.wait_group \{groups};\" ::: \"memory\");"
      RequireValidTile => "if (!*tile_valid) return;"
      RequireValidStage => "if (!stage_valid[slot]) return;"
      StageNextPairInOtherSlot =>
        "if (key_base + LF_KEY_VALUE_TILE < key_limit)\n" +
        indent +
        "  stage_tile(key_base + LF_KEY_VALUE_TILE, slot ^ 1);"
    }
    indent + text
  })
  .join("\n")
}
