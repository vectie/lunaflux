///|
fn cuda_grouped_effect_plan(
  lowering : @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering,
) -> @luna_attention_tile_schedule.AttentionGroupedEffectPlan raise CudaAttentionTileSourceError {
  match cuda_physical_program(lowering) {
    GroupedKeyProgram(_, effects) | SplitKeyProgram(_, effects) => effects
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
      ResetOperandValidity =>
        "if (thread == 0) { stage_valid[0] = stage_valid[1] = stage_valid[2] = stage_valid[3] = 1; }"
      PrimeOperandRings =>
        "stage_key(key_begin, 0); stage_value(key_begin, 0);\n" +
        indent +
        "stage_key(key_begin + LF_KEY_VALUE_TILE, 1); stage_value(key_begin + LF_KEY_VALUE_TILE, 1);"
      RequireValidKeyStage =>
        "if (!stage_valid[slot]) { asm volatile(\"cp.async.wait_group 0;\" ::: \"memory\"); __syncthreads(); return; }"
      RequireValidValueStage =>
        "if (!stage_valid[2 + slot]) { asm volatile(\"cp.async.wait_group 0;\" ::: \"memory\"); __syncthreads(); return; }"
      RefillKeyStage => "stage_key(key_base + 2 * LF_KEY_VALUE_TILE, slot);"
      RefillValueStage => "stage_value(key_base + 2 * LF_KEY_VALUE_TILE, slot);"
    }
    indent + text
  })
  .join("\n")
}
