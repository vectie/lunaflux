///|
fn cuda_score_transfer_mode(
  lowering : @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering,
) -> @luna_attention_tile_schedule.AttentionScoreTransferMode raise CudaAttentionTileSourceError {
  let memory = match lowering.copy() {
    CudaCooperativeSharedLoad => @luna_attention_strategy.AttentionStagedShared
    CudaAsynchronousSharedLoad => AttentionAsyncStagedShared
    _ => raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  @luna_attention_tile_schedule.AttentionScoreTransferMode::from_schedule(
    memory,
    stages=lowering.pipeline().0,
    immediate=lowering.direct_prefill_staging(),
  ) catch {
    _ => raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
}

///|
fn cuda_score_transfer_effects(
  mode : @luna_attention_tile_schedule.AttentionScoreTransferMode,
  point : @luna_attention_tile_schedule.AttentionScoreTransferPoint,
  cooperative_transfer : String,
  first_pair? : String = "  if (partition_begin < partition_end) stage_prefill_tile(partition_begin, 0);",
) -> String {
  mode
  .actions(point)
  .map(action => {
    match action {
      ResetValidity => "    if (thread == 0) tile_valid = 1;"
      PublishWorkgroup => "    __syncthreads();"
      StageCurrentPair =>
        if mode == CooperativeTransfer {
          cooperative_transfer
        } else {
          "    stage_prefill_tile(key_base, 0);"
        }
      AwaitPendingTransfers =>
        "    asm volatile(\"cp.async.wait_group 0;\" ::: \"memory\");"
      RequireValidTile => "    if (!tile_valid) return;"
      ReleaseValidationReaders =>
        match mode {
          LookaheadTransfer => "    __syncthreads();"
          ImmediateTransfer =>
            "    // The validation/key regions may be reused only after this join.\n    __syncthreads();"
          CooperativeTransfer =>
            "\n    // End validation lifetime before QK overwrites the aliased score slot.\n    __syncthreads();"
        }
      StageNextPairInOtherSlot =>
        "    const int next_key_base = key_base + LF_KEY_VALUE_TILE;\n" +
        "    if (next_key_base < partition_end)\n" +
        "      stage_prefill_tile(next_key_base, stage ^ 1);"
      StageFirstPair => first_pair
    }
  })
  .join("\n")
}

///|
fn cuda_score_cooperative_transfer(paged_load : String) -> String {
  (
    $|    for (int vector = thread; vector < key_value_vector_count; vector += LF_BLOCK_THREADS) {
    $|      const int first_value = vector * LF_VALUES_PER_VECTOR;
    $|      const int key = first_value / LF_HEAD_DIMENSION;
    $|      const int component = first_value % LF_HEAD_DIMENSION;
    $|      const int key_position = key_base + key;
    $|      uint4 packed_key = make_uint4(0U, 0U, 0U, 0U);
    $|      uint4 packed_value = make_uint4(0U, 0U, 0U, 0U);
    $|\{paged_load}
    $|      __nv_bfloat16 *stage_keys = key_tiles + stage * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
    $|      __nv_bfloat16 *stage_values = value_tiles + stage * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
    $|      *reinterpret_cast<uint4 *>(stage_keys + lf_operand_index(key, component, LF_HEAD_DIMENSION)) = packed_key;
    $|      *reinterpret_cast<uint4 *>(stage_values + lf_operand_index(key, component, LF_HEAD_DIMENSION)) = packed_value;
    $|    }
  )
}
