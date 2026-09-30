///|
/// Both single-launch and partitioned launches use this same closed dispatch.
fn cuda_emit_physical_attention(
  lowering : @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering,
  abi : CudaPagedAttentionTileAbi,
  symbol : String,
  maximum_source_bytes : Int,
) -> CudaAttentionTileSource raise CudaAttentionTileSourceError {
  match (lowering.phase(), cuda_physical_program(lowering)) {
    (Prefill, DirectKeyProgram(_)) if lowering.key_value_partitions() == 1 =>
      emit_cuda_direct_attention_tile_source(
        lowering, abi, symbol, maximum_source_bytes,
      )
    (Decode, DirectKeyProgram(_)) =>
      emit_cuda_direct_attention_tile_source(
        lowering, abi, symbol, maximum_source_bytes,
      )
    (Prefill, SharedScoreProgram(_)) =>
      emit_cuda_prefill_attention_tile_source(
        lowering, abi, symbol, maximum_source_bytes,
      )
    (Prefill, QueryFragmentProgram(_, _)) =>
      emit_cuda_query_owned_attention_source(
        lowering, abi, symbol, maximum_source_bytes,
      )
    (Decode, GroupedKeyProgram(_, _)) if lowering.key_value_partitions() == 1 =>
      emit_cuda_decode_attention_tile_source(
        lowering, abi, symbol, maximum_source_bytes,
      )
    (Decode, SplitKeyProgram(_, _)) =>
      emit_cuda_grouped_split_decode_attention_tile_source(
        lowering, abi, symbol, maximum_source_bytes,
      )
    (Decode, BlockwiseKeyProgram(fold, effects)) =>
      emit_cuda_blockwise_decode_source(
        lowering, abi, symbol, maximum_source_bytes, fold, effects,
      )
    _ => raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
}

///|
fn cuda_physical_program(
  lowering : @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering,
) -> @luna_attention_tile_schedule.AttentionPhysicalProgram raise CudaAttentionTileSourceError {
  lowering.physical_program() catch {
    _ => raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
}

///|
fn cuda_shared_score_program(
  lowering : @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering,
) -> @luna_attention_tile_schedule.AttentionSharedScoreProgram raise CudaAttentionTileSourceError {
  guard cuda_physical_program(lowering) is SharedScoreProgram(program) else {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  program
}

///|
fn cuda_key_fold(
  lowering : @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering,
) -> @attention_physical_ir.KeyFold raise CudaAttentionTileSourceError {
  match cuda_physical_program(lowering) {
    DirectKeyProgram(fold)
    | GroupedKeyProgram(fold, _)
    | SplitKeyProgram(fold, _) => fold
    _ => raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
}

///|
/// CUDA realizes publication as lane exchange. The common state program owns
/// the order and preserves the unrounded per-key probability/denominator law.
fn cuda_key_scale_source(fold : @attention_physical_ir.KeyFold) -> String {
  fold
  .scale_steps()
  .map(step => {
    match step {
      MergeKeyStatistics =>
        (
          #|      float old_scale = 0.0f;
          #|      float new_scale = 0.0f;
          #|      if (lane == 0) {
          #|        const float next_maximum = fmaxf(maximum, score);
          #|        old_scale = denominator == 0.0f
          #|            ? 0.0f : expf(maximum - next_maximum);
          #|        new_scale = expf(score - next_maximum);
          #|        maximum = next_maximum;
          #|        denominator = denominator * old_scale + new_scale;
          #|      }
        )
      PublishKeyScales =>
        (
          #|      old_scale = __shfl_sync(0xffffffffU, old_scale, 0);
          #|      new_scale = __shfl_sync(0xffffffffU, new_scale, 0);
        )
    }
  })
  .join("\n")
}
