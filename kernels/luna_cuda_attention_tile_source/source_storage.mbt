///|
/// The portable schedule owns storage offsets and peak capacity. CUDA only
/// chooses pointer types and the concrete shared-memory address space.
fn cuda_attention_storage_declarations(
  lowering : @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering,
) -> String raise CudaAttentionTileSourceError {
  guard lowering.matrix_storage_layout() is Some(layout) else {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let builder = StringBuilder()
  builder <+
    "  __nv_bfloat16 *query_tile = reinterpret_cast<__nv_bfloat16 *>(shared_arena + \{layout.offset(QueryStorage)});\n"
  builder <+
    "  __nv_bfloat16 *key_tiles = reinterpret_cast<__nv_bfloat16 *>(shared_arena + \{layout.offset(KeyStorage)});\n"
  builder <+
    "  __nv_bfloat16 *value_tiles = reinterpret_cast<__nv_bfloat16 *>(shared_arena + \{layout.offset(ValueStorage)});\n"
  builder <+
    "  float *scores = reinterpret_cast<float *>(shared_arena + \{layout.offset(ScoreStorage)});\n"
  builder <+
    "  __nv_bfloat16 *weights = reinterpret_cast<__nv_bfloat16 *>(shared_arena + \{layout.offset(ProbabilityStorage)});\n"
  builder <+
    "  float *accumulator = reinterpret_cast<float *>(shared_arena + \{layout.offset(AccumulatorStorage)});\n"
  builder <+
    "  float *running_maximum = reinterpret_cast<float *>(shared_arena + \{layout.offset(TerminalMaximumStorage)});\n"
  builder <+
    "  float *running_denominator = reinterpret_cast<float *>(shared_arena + \{layout.offset(TerminalDenominatorStorage)});\n"
  builder <+
    "  float *previous_scales = reinterpret_cast<float *>(shared_arena + \{layout.offset(PreviousScaleStorage)});\n"
  builder <+
    "  int &tile_valid = *reinterpret_cast<int *>(shared_arena + \{layout.offset(ValidationStorage)});\n"
  builder.to_string()
}
