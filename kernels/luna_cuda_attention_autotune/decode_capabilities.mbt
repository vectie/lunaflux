///|
/// Offline CUDA decode scheduling envelope. The 48 KiB block budget is an
/// occupancy policy, not the physical device's opt-in shared-memory maximum.
/// It allows two staged KV tiles without selecting the 64-position double
/// buffer that consumes most of one SM. Candidate choice remains in the pure
/// attention compiler; no candidate ID or model family is selected here.
/// The matrix shape advertises the backend's BF16 operation, independently
/// of whether a particular decode problem opts into its numerical law.
pub fn cuda_sm120_decode_compiler_capabilities() -> @luna_attention_strategy.AttentionCompilerCapabilities raise {
  @luna_attention_strategy.AttentionCompilerCapabilities::new(
    subgroup_width=32,
    max_threads_per_block=256,
    max_shared_memory_bytes=49152L,
    max_pipeline_stages=2,
    supports_async_copy=true,
    matrix_shape=Some(
      @luna_attention_strategy.AttentionMatrixShape::new(
        rows=16,
        columns=16,
        reduction=16,
      ),
    ),
  )
}
