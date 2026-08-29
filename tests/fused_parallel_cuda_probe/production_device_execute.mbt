///|
struct FusedProductionLoadedKernel {
  module_ : @device.Module
  function_ : @device.Function
}

///|
fn load_fused_production_kernel(
  context : @device.Context,
  cubin : Bytes,
  symbol : String,
  balance : FusedProbeResourceBalance,
) -> FusedProductionLoadedKernel raise {
  let module_ = context.load_module(cubin)
  balance.modules += 1
  let function_ = try module_.load_function(@utf8.encode(symbol)) catch {
    error => {
      module_.close() catch {
        _ => balance.cleanup_failures += 1
      }
      balance.modules -= 1
      raise error
    }
  } noraise {
    value => value
  }
  balance.functions += 1
  { module_, function_ }
}

///|
fn close_fused_production_kernel(
  kernel : FusedProductionLoadedKernel,
  balance : FusedProbeResourceBalance,
) -> Unit {
  kernel.function_.close() catch {
    _ => balance.cleanup_failures += 1
  }
  balance.functions -= 1
  kernel.module_.close() catch {
    _ => balance.cleanup_failures += 1
  }
  balance.modules -= 1
}

///|
fn run_fused_production_captured(
  context : @device.Context,
  stream : @device.Stream,
  launches : ArrayView[@device.OrderedKernelLaunch],
  balance : FusedProbeResourceBalance,
) -> Unit raise {
  let executor = context.create_ordered_kernel_executor(
    stream,
    launches,
    policy=OrderedKernelCaptureRequired,
  )
  balance.executors += 1
  defer {
    executor.close() catch {
      _ => balance.cleanup_failures += 1
    }
    balance.executors -= 1
  }
  require_fused_probe(
    executor.execution_mode() == OrderedKernelCaptured,
    "production capture-required executor did not capture",
  )
  balance.pending_work += 1
  try {
    executor.launch_captured()
    executor.record_completion()
    executor.wait_completion()
    executor.reset()
  } catch {
    error => {
      balance.pending_work -= 1
      raise error
    }
  }
  balance.pending_work -= 1
}

///|
fn production_probe_poison_output(
  arena : @device.Allocation,
  shape : @fused_parallel_qualification.FusedQualificationShape,
) -> Unit raise {
  probe_upload(
    arena,
    PROBE_ATTENTION_OUTPUT,
    Bytes::from_array(FixedArray::make(shape.token_count() * 256, b'\xa5')[:]),
  )
}

///|
fn run_production_qkv_readonly_shape(
  context : @device.Context,
  stream : @device.Stream,
  arena : @device.Allocation,
  production_qkv : FusedProductionLoadedKernel,
  production_readonly : FusedProductionLoadedKernel,
  oracle_qkv : FusedProductionLoadedKernel,
  oracle_rope : FusedProductionLoadedKernel,
  oracle_attention : FusedProductionLoadedKernel,
  candidate_context : FusedProbeCandidateContext,
  candidates : FusedProductionProbeCandidates,
  shape : @fused_parallel_qualification.FusedQualificationShape,
  balance : FusedProbeResourceBalance,
) -> (Int64, Int64) raise {
  let fixture = @fused_parallel_qualification.qkv_qualification_fixture(shape)
  upload_qkv_fixture(arena, fixture)
  production_probe_poison_output(arena, shape)
  let launches = [
    @device.OrderedKernelLaunch::new(
      production_qkv.function_,
      probe_dimensions(
        candidate_context.qualification.qkv_candidate().launch_dimensions(),
      ),
      probe_production_qkv_arguments(arena, shape),
    ),
    @device.OrderedKernelLaunch::new(
      production_readonly.function_,
      probe_dimensions(candidates.oracle_attention.launch_dimensions()),
      probe_attention_arguments(arena, shape),
    ),
  ]
  run_fused_production_captured(context, stream, launches[:], balance)
  let production_output = probe_read_bf16(
    arena,
    PROBE_ATTENTION_OUTPUT,
    shape.token_count() *
    @fused_parallel_qualification.qualification_input_width(),
  )
  let rotated = probe_qkv_comparison(
    probe_read_bf16(
      arena,
      PROBE_ROTATED_QKV,
      shape.token_count() *
      @fused_parallel_qualification.qualification_output_width(),
    ),
    fixture.expected_rotated_qkv(),
  )
  let key = probe_qkv_comparison(
    probe_read_bf16(
      arena,
      PROBE_KEY_CACHE,
      @fused_parallel_qualification.qualification_cache_elements(),
    ),
    fixture.expected_key_cache(),
  )
  let value = probe_qkv_comparison(
    probe_read_bf16(
      arena,
      PROBE_VALUE_CACHE,
      @fused_parallel_qualification.qualification_cache_elements(),
    ),
    fixture.expected_value_cache(),
  )
  upload_qkv_fixture(arena, fixture)
  production_probe_poison_output(arena, shape)
  probe_launch_synchronous(
    oracle_qkv.function_,
    probe_dimensions(
      candidate_context.qualification.qkv_fallback().launch_dimensions(),
    ),
    probe_standalone_projection_arguments(arena, shape),
    stream,
    balance,
  )
  probe_launch_synchronous(
    oracle_rope.function_,
    probe_dimensions(
      candidate_context.qualification.rope_fallback().launch_dimensions(),
    ),
    probe_standalone_rope_arguments(arena, shape),
    stream,
    balance,
  )
  probe_launch_synchronous(
    oracle_attention.function_,
    probe_dimensions(candidates.oracle_attention.launch_dimensions()),
    probe_attention_arguments(arena, shape),
    stream,
    balance,
  )
  let attention = probe_qkv_comparison(
    production_output,
    probe_read_bf16(
      arena,
      PROBE_ATTENTION_OUTPUT,
      shape.token_count() *
      @fused_parallel_qualification.qualification_input_width(),
    ),
  )
  (
    rotated.0.max(key.0).max(value.0).max(attention.0),
    rotated.1.max(key.1).max(value.1).max(attention.1),
  )
}

///|
fn run_production_residual_shape(
  context : @device.Context,
  stream : @device.Stream,
  arena : @device.Allocation,
  kernel : FusedProductionLoadedKernel,
  candidate : @luna_cuda_fused_parallel_aot.FusedResidualRmsNormProductionCudaCandidate,
  shape : @fused_parallel_qualification.FusedQualificationShape,
  balance : FusedProbeResourceBalance,
) -> (Int64, Int64) raise {
  let fixture = @fused_parallel_qualification.residual_norm_qualification_fixture(
    shape,
  )
  upload_residual_fixture(arena, fixture)
  let launches = [
    @device.OrderedKernelLaunch::new(
      kernel.function_,
      probe_dimensions(candidate.launch_dimensions()),
      probe_production_residual_arguments(arena, shape),
    ),
  ]
  run_fused_production_captured(context, stream, launches[:], balance)
  let elements = shape.token_count() *
    @fused_parallel_qualification.qualification_input_width()
  let residual = probe_residual_comparison(
    probe_read_bf16(arena, PROBE_RESIDUAL_OUTPUT, elements),
    fixture.expected_residual_output(),
  )
  let norm = probe_residual_comparison(
    probe_read_bf16(arena, PROBE_NORM_OUTPUT, elements),
    fixture.expected_norm_output(),
  )
  (residual.0.max(norm.0), residual.1.max(norm.1))
}
