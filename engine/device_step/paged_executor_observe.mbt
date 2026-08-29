///|
fn validate_offline_logit_selection(
  vocabulary_size : Int,
  token_ids : ArrayView[Int],
  destination_length : Int,
) -> Unit raise DeviceStepError {
  if token_ids.length() == 0 ||
    token_ids.length() > 32 ||
    destination_length != token_ids.length() {
    raise InvalidExecutor(SamplingOutput)
  }
  for token_id in token_ids {
    if token_id < 0 || token_id >= vocabulary_size {
      raise InvalidExecutor(SamplingOutput)
    }
  }
}

///|
/// Copies only predeclared terminal-logit indices after one authenticated
/// execution into caller-owned fixed storage.
///
/// This bounded correctness observer is unavailable before graph completion or
/// after `finish`. It reuses the executor's startup-owned full-row scratch and
/// does not expose an allocation, device address, mutable row, or serving-path
/// sampling authority.
pub fn PagedGraphExecutor::observe_offline_selected_logits(
  self : PagedGraphExecutor,
  executed : ExecutedPagedGraphExecution,
  batch_index : Int,
  token_ids : ArrayView[Int],
  destination : FixedArray[Double],
) -> Unit raise DeviceStepError {
  let descriptor = authenticate_executed_paged(executed, self)
  validate_offline_logit_selection(
    self.completion.vocabulary_size,
    token_ids,
    destination.length(),
  )
  let row_count = read_paged_i32(descriptor.host.counts, 2)
  let token_count = read_paged_i32(descriptor.host.counts, 3)
  if batch_index < 0 || batch_index >= row_count {
    raise InvalidExecutor(SamplingOutput)
  }
  let token_end = read_paged_i32(
    descriptor.host.query_row_offsets,
    batch_index + 1,
  )
  let token_index = token_end - 1
  if token_end <= 0 || token_index >= token_count {
    raise InvalidExecutor(SamplingOutput)
  }
  let scratch = self.completion
  let activation = open_activation_workspace(self.resources)
  try
    descriptor.context.copy_to_fixed_host(
      activation,
      scratch.encoded_logits,
      scratch.final_output_offset + token_index.to_int64() * scratch.row_bytes,
      0L,
      scratch.row_bytes,
    )
  catch {
    error => {
      poison_paged_completion(self, descriptor)
      raise ExecutorDevice(ActivationReadback, error)
    }
  } noraise {
    _ => ()
  }
  decode_paged_bf16_logits(scratch)
  for index, token_id in token_ids {
    let value = scratch.logits[token_id]
    if value.is_nan() || value.is_inf() {
      poison_paged_completion(self, descriptor)
      raise InvalidExecutor(SamplingOutput)
    }
    destination[index] = value
  }
}
