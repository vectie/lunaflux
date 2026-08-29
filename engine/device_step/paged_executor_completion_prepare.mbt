///|
fn prepare_paged_completion_scratch(
  model_plan : @plan.ModelPlan,
  memory_plan : @device_memory.DeviceMemoryPlan,
  limits : DeviceStepLimits,
) -> PagedCompletionScratch raise DeviceStepError {
  let vocabulary_size = model_plan.shape_constraints().vocabulary_size()
  let row_bytes = vocabulary_size.to_int64() * 2L
  let token_rows = memory_plan.token_rows()
  guard token_rows > 0L &&
    memory_plan.region(model_plan.final_output()) is Some(output) &&
    output.is_retained_output() &&
    output.logical_bytes() % token_rows == 0L &&
    output.logical_bytes() / token_rows == row_bytes else {
    raise InvalidExecutor(SamplingOutput)
  }
  let sampling_limits = @sampling.SamplingLimits::new(
    limits.max_rows(),
    vocabulary_size,
  ) catch {
    error => raise ExecutorSampling(error)
  }
  let sampler = @sampling.SamplerScratch::new(vocabulary_size, sampling_limits) catch {
    error => raise ExecutorSampling(error)
  }
  {
    final_output_offset: output.offset_bytes(),
    row_bytes,
    vocabulary_size,
    limits: sampling_limits,
    sampler,
    encoded_logits: FixedArray::make(row_bytes.to_int(), b'\x00'),
    logits: FixedArray::make(vocabulary_size, 0.0),
    selected_tokens: FixedArray::make(limits.max_rows(), 0),
    mode: HostLogitsCompletion,
  }
}
