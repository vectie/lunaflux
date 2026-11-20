///|
fn copy_paged_cuda_greedy_results(
  executor : PagedGraphExecutor,
  descriptor : DeviceStepOwner,
  row_count : Int32,
) -> Unit raise DeviceStepError {
  let row_count = projected_result_count(descriptor, row_count)
  guard executor.completion.mode is CudaGreedyCompletion(host) else { return }
  guard executor.resources.greedy_sampling_results is Some(owner) &&
    owner.open &&
    row_count >= 0 &&
    row_count <= executor.completion.selected_tokens.length() else {
    poison_paged_completion(executor, descriptor)
    raise InvalidExecutor(SamplingOutput)
  }
  descriptor.context.copy_to_fixed_host(
    owner.resource,
    host,
    0L,
    0L,
    row_count.to_int64() * PAGED_GREEDY_RESULT_ROW_BYTES,
  ) catch {
    error =>
      poison_paged_completion_and_raise(
        executor,
        descriptor,
        ExecutorDevice(ActivationReadback, error),
      )
  }
}

///|
fn select_paged_cuda_greedy_result(
  executor : PagedGraphExecutor,
  descriptor : DeviceStepOwner,
  row : Int32,
) -> Int32 raise DeviceStepError {
  guard executor.completion.mode is CudaGreedyCompletion(host) &&
    row >= 0 &&
    row < executor.completion.selected_tokens.length() else {
    return poison_paged_completion_and_raise(
      executor,
      descriptor,
      InvalidExecutor(SamplingOutput),
    )
  }
  let output_row = projected_result_row(descriptor, row)
  if output_row < 0 ||
    output_row >= executor.completion.selected_tokens.length() {
    return poison_paged_completion_and_raise(
      executor,
      descriptor,
      InvalidExecutor(SamplingOutput),
    )
  }
  let token = read_paged_i32(host, output_row * 2)
  let status = read_paged_i32(host, output_row * 2 + 1)
  if status >= 0 {
    return poison_paged_completion_and_raise(
      executor,
      descriptor,
      ExecutorSampling(NonFiniteLogit(row~, token_id=status)),
    )
  }
  if status != -1 || token < 0 || token >= executor.completion.vocabulary_size {
    return poison_paged_completion_and_raise(
      executor,
      descriptor,
      InvalidExecutor(SamplingOutput),
    )
  }
  token
}
