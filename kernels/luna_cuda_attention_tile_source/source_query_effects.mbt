///|
/// Realize common ownership effects. Slot XOR and asynchronous-copy assembly
/// are CUDA details; the common plan owns their ordering and readiness points.
fn cuda_query_owned_effects(
  plan : @luna_attention_tile_schedule.AttentionQueryEffectPlan,
  point : @luna_attention_tile_schedule.AttentionQueryEffectPoint,
) -> String {
  plan
  .actions(point)
  .map(action => {
    match action {
      StageFirstKey => "stage_query_tile(key_begin,0,0);"
      StageFirstValue => "stage_query_tile(key_begin,0,1);"
      StageFirstPair => "stage_query_tile(key_begin,slot);"
      StageCurrentPair => "stage_query_tile(kb,0);"
      AwaitPendingTransfers(groups) =>
        "asm volatile(\"cp.async.wait_group \{groups};\" ::: \"memory\");"
      PublishWorkgroup => "__syncthreads();"
      StageNextPairInOtherSlot =>
        "if (kb+LF_KEY_VALUE_TILE<key_end) stage_query_tile(kb+LF_KEY_VALUE_TILE,slot^1);"
      StageNextKey =>
        "if (kb+LF_KEY_VALUE_TILE<key_end) stage_query_tile(kb+LF_KEY_VALUE_TILE,0,0);"
      StageNextValue =>
        "if (kb+LF_KEY_VALUE_TILE<key_end) stage_query_tile(kb+LF_KEY_VALUE_TILE,0,1);"
      AdvanceTransferSlot => "slot^=1;"
    }
  })
  .join(" ")
}
