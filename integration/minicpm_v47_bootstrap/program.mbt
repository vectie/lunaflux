///|
/// The same declared, unmeasured initial Spark profile used by offline AOT
/// preparation and resident startup. This does not observe/qualify a device.
pub fn compile_spark_attention(
  operators : @operators.DecoderOperators,
) -> @attention.Compilation raise {
  let capabilities = @strategy.AttentionCompilerCapabilities::new(
    subgroup_width=32,
    max_threads_per_block=256,
    max_shared_memory_bytes=49152L,
    max_pipeline_stages=1,
    supports_async_copy=false,
    parallel_workgroup_target=1,
    matrix_shape=Some(
      @strategy.AttentionMatrixShape::new(rows=16, columns=16, reduction=16),
    ),
  )
  @attention.compile(
    operators,
    device_target=@catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=1,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    target_id=@strategy.AttentionTargetId::new(
      "sm121-unqualified-software-profile",
    ),
    capabilities~,
  )
}

///|
pub fn text_program(
  operators : @operators.DecoderOperators,
  attention : @attention.Compilation,
) -> @text.Program raise {
  @text.Program::new(
    operators,
    @decoder.Program::new(operators, attention),
    maximum_workspace_bytes=1048576L,
  )
}
