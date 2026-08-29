///|
fn probe_launch_synchronous(
  function_ : @device.Function,
  dimensions : @device.KernelLaunchDimensions,
  arguments : @device.KernelArguments,
  stream : @device.Stream,
  balance : FusedProbeResourceBalance,
) -> Unit raise {
  balance.pending_work += 1
  try function_.launch_synchronous(dimensions, arguments, stream) catch {
    error => {
      balance.pending_work -= 1
      raise error
    }
  } noraise {
    _ => balance.pending_work -= 1
  }
}

///|
fn probe_region(
  arena : @device.Allocation,
  offset : Int64,
  byte_count : Int64,
  alignment : Int64,
) -> @device.KernelArgument raise {
  @device.KernelArgument::device_region(arena, offset~, byte_count~, alignment~)
}

///|
fn probe_qkv_arguments(
  arena : @device.Allocation,
  shape : @fused_parallel_qualification.FusedQualificationShape,
) -> @device.KernelArguments raise {
  let token_count = shape.token_count().to_int64()
  let page_count = shape.page_table().length().to_int64()
  @device.KernelArguments::new([
    probe_region(arena, PROBE_STEP_COUNTS, 20L, 4L),
    probe_region(arena, PROBE_QUERY_POSITIONS, token_count * 4L, 4L),
    probe_region(arena, PROBE_QUERY_ROW_OFFSETS, 8L, 4L),
    probe_region(arena, PROBE_SEQUENCE_LENGTHS, 4L, 4L),
    probe_region(arena, PROBE_PAGE_TABLE_OFFSETS, 8L, 4L),
    probe_region(arena, PROBE_PAGE_TABLE_INDICES, page_count * 4L, 4L),
    probe_region(arena, PROBE_INPUT, token_count * 256L, 16L),
    probe_region(arena, PROBE_Q_WEIGHT, 32768L, 16L),
    probe_region(arena, PROBE_K_WEIGHT, 16384L, 16L),
    probe_region(arena, PROBE_V_WEIGHT, 16384L, 16L),
    probe_region(arena, PROBE_ROTATED_QKV, token_count * 512L, 16L),
    probe_region(arena, PROBE_KEY_CACHE, 8192L, 16L),
    probe_region(arena, PROBE_VALUE_CACHE, 8192L, 16L),
    probe_region(arena, PROBE_QKV_CANARY, 4L, 4L),
  ])
}

///|
fn probe_production_qkv_arguments(
  arena : @device.Allocation,
  shape : @fused_parallel_qualification.FusedQualificationShape,
) -> @device.KernelArguments raise {
  let token_count = shape.token_count().to_int64()
  let page_count = shape.page_table().length().to_int64()
  @device.KernelArguments::new([
    probe_region(arena, PROBE_STEP_COUNTS, 20L, 4L),
    probe_region(arena, PROBE_QUERY_POSITIONS, token_count * 4L, 4L),
    probe_region(arena, PROBE_QUERY_ROW_OFFSETS, 8L, 4L),
    probe_region(arena, PROBE_SEQUENCE_LENGTHS, 4L, 4L),
    probe_region(arena, PROBE_PAGE_TABLE_OFFSETS, 8L, 4L),
    probe_region(arena, PROBE_PAGE_TABLE_INDICES, page_count * 4L, 4L),
    probe_region(arena, PROBE_INPUT, token_count * 256L, 16L),
    probe_region(arena, PROBE_Q_WEIGHT, 32768L, 16L),
    probe_region(arena, PROBE_K_WEIGHT, 16384L, 16L),
    probe_region(arena, PROBE_V_WEIGHT, 16384L, 16L),
    probe_region(arena, PROBE_ROTATED_QKV, token_count * 512L, 16L),
    probe_region(arena, PROBE_KEY_CACHE, 8192L, 16L),
    probe_region(arena, PROBE_VALUE_CACHE, 8192L, 16L),
  ])
}

///|
fn probe_residual_arguments(
  arena : @device.Allocation,
  shape : @fused_parallel_qualification.FusedQualificationShape,
) -> @device.KernelArguments raise {
  let bytes = shape.token_count().to_int64() * 256L
  @device.KernelArguments::new([
    probe_region(arena, PROBE_STEP_COUNTS, 20L, 4L),
    probe_region(arena, PROBE_RESIDUAL, bytes, 16L),
    probe_region(arena, PROBE_BRANCH, bytes, 16L),
    probe_region(arena, PROBE_NORM_WEIGHT, 256L, 16L),
    probe_region(arena, PROBE_RESIDUAL_OUTPUT, bytes, 16L),
    probe_region(arena, PROBE_NORM_OUTPUT, bytes, 16L),
    probe_region(arena, PROBE_RESIDUAL_CANARY, 4L, 4L),
  ])
}

///|
fn probe_production_residual_arguments(
  arena : @device.Allocation,
  shape : @fused_parallel_qualification.FusedQualificationShape,
) -> @device.KernelArguments raise {
  let bytes = shape.token_count().to_int64() * 256L
  @device.KernelArguments::new([
    probe_region(arena, PROBE_STEP_COUNTS, 20L, 4L),
    probe_region(arena, PROBE_RESIDUAL, bytes, 16L),
    probe_region(arena, PROBE_BRANCH, bytes, 16L),
    probe_region(arena, PROBE_NORM_WEIGHT, 256L, 16L),
    probe_region(arena, PROBE_RESIDUAL_OUTPUT, bytes, 16L),
    probe_region(arena, PROBE_NORM_OUTPUT, bytes, 16L),
  ])
}

///|
fn probe_attention_arguments(
  arena : @device.Allocation,
  shape : @fused_parallel_qualification.FusedQualificationShape,
) -> @device.KernelArguments raise {
  let token_count = shape.token_count().to_int64()
  let page_count = shape.page_table().length().to_int64()
  @device.KernelArguments::new([
    probe_region(arena, PROBE_STEP_COUNTS, 20L, 4L),
    probe_region(arena, PROBE_QUERY_POSITIONS, token_count * 4L, 4L),
    probe_region(arena, PROBE_QUERY_ROW_OFFSETS, 8L, 4L),
    probe_region(arena, PROBE_SEQUENCE_LENGTHS, 4L, 4L),
    probe_region(arena, PROBE_PAGE_TABLE_OFFSETS, 8L, 4L),
    probe_region(arena, PROBE_PAGE_TABLE_INDICES, page_count * 4L, 4L),
    probe_region(arena, PROBE_ROTATED_QKV, token_count * 512L, 16L),
    probe_region(arena, PROBE_ATTENTION_OUTPUT, token_count * 256L, 16L),
    probe_region(arena, PROBE_KEY_CACHE, 8192L, 16L),
    probe_region(arena, PROBE_VALUE_CACHE, 8192L, 16L),
  ])
}

///|
fn probe_standalone_projection_arguments(
  arena : @device.Allocation,
  shape : @fused_parallel_qualification.FusedQualificationShape,
) -> @device.KernelArguments raise {
  let token_count = shape.token_count().to_int64()
  @device.KernelArguments::new([
    probe_region(arena, PROBE_STEP_COUNTS, 20L, 4L),
    probe_region(arena, PROBE_INPUT, token_count * 256L, 16L),
    probe_region(arena, PROBE_Q_WEIGHT, 32768L, 16L),
    probe_region(arena, PROBE_K_WEIGHT, 16384L, 16L),
    probe_region(arena, PROBE_V_WEIGHT, 16384L, 16L),
    probe_region(arena, PROBE_STANDALONE_PROJECTED_QKV, token_count * 512L, 16L),
  ])
}

///|
fn probe_standalone_rope_arguments(
  arena : @device.Allocation,
  shape : @fused_parallel_qualification.FusedQualificationShape,
) -> @device.KernelArguments raise {
  let token_count = shape.token_count().to_int64()
  @device.KernelArguments::new([
    probe_region(arena, PROBE_STEP_COUNTS, 20L, 4L),
    probe_region(arena, PROBE_QUERY_POSITIONS, token_count * 4L, 4L),
    probe_region(arena, PROBE_STANDALONE_PROJECTED_QKV, token_count * 512L, 16L),
    probe_region(arena, PROBE_ROTATED_QKV, token_count * 512L, 16L),
  ])
}

///|
fn probe_standalone_kv_write_arguments(
  arena : @device.Allocation,
  shape : @fused_parallel_qualification.FusedQualificationShape,
) -> @device.KernelArguments raise {
  let token_count = shape.token_count().to_int64()
  let page_count = shape.page_table().length().to_int64()
  @device.KernelArguments::new([
    probe_region(arena, PROBE_STEP_COUNTS, 20L, 4L),
    probe_region(arena, PROBE_QUERY_POSITIONS, token_count * 4L, 4L),
    probe_region(arena, PROBE_QUERY_ROW_OFFSETS, 8L, 4L),
    probe_region(arena, PROBE_SEQUENCE_LENGTHS, 4L, 4L),
    probe_region(arena, PROBE_PAGE_TABLE_OFFSETS, 8L, 4L),
    probe_region(arena, PROBE_PAGE_TABLE_INDICES, page_count * 4L, 4L),
    probe_region(arena, PROBE_ROTATED_QKV, token_count * 512L, 16L),
    probe_region(arena, PROBE_KEY_CACHE, 8192L, 16L),
    probe_region(arena, PROBE_VALUE_CACHE, 8192L, 16L),
  ])
}
