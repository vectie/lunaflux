///|
fn read_paged_i32(buffer : FixedArray[Byte], index : Int) -> Int {
  let start = index * INT32_BYTES
  buffer[start].to_int() |
  (buffer[start + 1].to_int() << 8) |
  (buffer[start + 2].to_int() << 16) |
  (buffer[start + 3].to_int() << 24)
}

///|
fn decode_paged_bf16_logits(scratch : PagedCompletionScratch) -> Unit {
  for token_id in 0..<scratch.vocabulary_size {
    let offset = token_id * 2
    let low = scratch.encoded_logits[offset].to_uint()
    let high = scratch.encoded_logits[offset + 1].to_uint()
    let bits = ((high << 8) | low) << 16
    scratch.logits[token_id] = Float::reinterpret_from_uint(bits).to_double()
  }
}

///|
fn select_paged_logit(
  scratch : PagedCompletionScratch,
  parameters : @inference.SamplingParameters,
  seed_value : UInt64,
  sample_index : Int,
) -> Int raise DeviceStepError {
  let logits = scratch.logits[:]
  let selection = try {
    match parameters.mode() {
      Greedy => @sampling.greedy(logits, scratch.limits)
      Sample =>
        @sampling.stochastic_sample_at(
          logits,
          parameters,
          scratch.limits,
          try! @inference.SamplingSeed::new(seed_value),
          sample_index,
          scratch.sampler,
        )
    }
  } catch {
    error => raise ExecutorSampling(error)
  } noraise {
    value => value
  }
  selection.token_id()
}

///|
fn open_activation_workspace(
  resources : PagedGraphResources,
) -> @device.Allocation raise DeviceStepError {
  guard resources.activation_workspace is Some(owner) && owner.open else {
    raise InvalidExecutor(ResourceState)
  }
  owner.resource
}

///|
fn poison_paged_completion(
  executor : PagedGraphExecutor,
  descriptor : DeviceStepOwner,
) -> Unit {
  executor.lifecycle = GraphPoisoned
  descriptor.lifecycle = Poisoned
}

///|
fn[T] poison_paged_completion_and_raise(
  executor : PagedGraphExecutor,
  descriptor : DeviceStepOwner,
  error : DeviceStepError,
) -> T raise DeviceStepError {
  poison_paged_completion(executor, descriptor)
  raise error
}

///|
fn read_and_select_paged_row(
  executor : PagedGraphExecutor,
  descriptor : DeviceStepOwner,
  row_index : Int,
  token_count : Int,
  parameters : @inference.SamplingParameters,
  seed_value : UInt64,
  sample_index : Int,
) -> Int raise DeviceStepError {
  let token_end = read_paged_i32(
    descriptor.host.query_row_offsets,
    row_index + 1,
  )
  let token_index = token_end - 1
  if token_end <= 0 || token_index >= token_count {
    poison_paged_completion(executor, descriptor)
    raise InvalidExecutor(SamplingOutput)
  }
  let scratch = executor.completion
  let activation = open_activation_workspace(executor.resources)
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
      poison_paged_completion(executor, descriptor)
      raise ExecutorDevice(ActivationReadback, error)
    }
  } noraise {
    _ => ()
  }
  decode_paged_bf16_logits(scratch)
  select_paged_logit(scratch, parameters, seed_value, sample_index) catch {
    error => poison_paged_completion_and_raise(executor, descriptor, error)
  }
}

///|
fn validate_completion_plan(
  descriptor : DeviceStepOwner,
  plan : @worker_protocol.SubmittedSchedulePlan,
  writer : @worker_protocol.CompletionWriter,
) -> DeviceStepSummary raise DeviceStepError {
  guard descriptor.staged_plan_owner is Some(owner) else {
    raise InvalidExecutor(ResourceState)
  }
  let actual_owner = plan.buffer_identity() catch {
    error => raise Protocol(error)
  }
  let sequence_value = plan.sequence_wire() catch {
    error => raise Protocol(error)
  }
  if !owner.same_owner(actual_owner) ||
    sequence_value != descriptor.staged_summary.sequence_value {
    raise ExecutorOwnerMismatch
  }
  writer.validate_plan(plan) catch {
    error => raise Protocol(error)
  }
  descriptor.staged_summary
}

///|
fn append_paged_completion(
  plan : @worker_protocol.SubmittedSchedulePlan,
  writer : @worker_protocol.CompletionWriter,
  summary : DeviceStepSummary,
  selected_tokens : FixedArray[Int],
) -> @worker_protocol.SubmittedCompletion raise DeviceStepError {
  for prefill_index in 0..<summary.prefill_count {
    let row = try plan.prefill_row(prefill_index) catch {
      error => raise Protocol(error)
    } noraise {
      value => value
    }
    let produces_token = try row.produces_token() catch {
      error => raise Protocol(error)
    } noraise {
      value => value
    }
    let slot = try row.completion_slot() catch {
      error => raise Protocol(error)
    } noraise {
      value => value
    }
    let request_id = try row.request_id() catch {
      error => raise Protocol(error)
    } noraise {
      value => value
    }
    let request_generation = try row.request_generation() catch {
      error => raise Protocol(error)
    } noraise {
      value => value
    }
    let tokens = try row.tokens() catch {
      error => raise Protocol(error)
    } noraise {
      value => value
    }
    let processed_tokens = tokens.length()
    if produces_token {
      writer.append_final_prefill_completed(
        slot~,
        request_id~,
        request_generation~,
        processed_tokens~,
        token_id=selected_tokens[prefill_index],
      ) catch {
        error => raise Protocol(error)
      }
    } else {
      writer.append_prefill_completed(
        slot~,
        request_id~,
        request_generation~,
        processed_tokens~,
      ) catch {
        error => raise Protocol(error)
      }
    }
  }
  for decode_index in 0..<summary.decode_count {
    let row_index = summary.prefill_count + decode_index
    let row = try plan.decode_row(decode_index) catch {
      error => raise Protocol(error)
    } noraise {
      value => value
    }
    let slot = try row.completion_slot() catch {
      error => raise Protocol(error)
    } noraise {
      value => value
    }
    let request_id = try row.request_id() catch {
      error => raise Protocol(error)
    } noraise {
      value => value
    }
    let request_generation = try row.request_generation() catch {
      error => raise Protocol(error)
    } noraise {
      value => value
    }
    writer.append_decode_completed(
      slot~,
      request_id~,
      request_generation~,
      token_id=selected_tokens[row_index],
    ) catch {
      error => raise Protocol(error)
    }
  }
  writer.submit() catch {
    error => raise Protocol(error)
  }
}

///|
/// Reads the retained BF16 terminal logits, deterministically selects every
/// producing row by its request seed and output index, and freezes the exact
/// completion writer bound to `plan`.
///
/// All device reads and sampling complete before the first completion entry is
/// appended. Protocol failures during the final append/submit phase leave the
/// caller-owned writer open so it can be explicitly aborted. Device readback
/// or invalid kernel output permanently poisons the executor.
pub fn PagedGraphExecutor::sample_completion(
  self : PagedGraphExecutor,
  executed : ExecutedPagedGraphExecution,
  plan : @worker_protocol.SubmittedSchedulePlan,
  writer : @worker_protocol.CompletionWriter,
) -> @worker_protocol.SubmittedCompletion raise DeviceStepError {
  let descriptor = authenticate_executed_paged(executed, self)
  let summary = validate_completion_plan(descriptor, plan, writer)
  copy_paged_cuda_greedy_results(self, descriptor, summary.row_count)
  for prefill_index in 0..<summary.prefill_count {
    let row = try plan.prefill_row(prefill_index) catch {
      error => raise Protocol(error)
    } noraise {
      value => value
    }
    let produces_token = try row.produces_token() catch {
      error => raise Protocol(error)
    } noraise {
      value => value
    }
    if produces_token {
      let parameters = try row.sampling() catch {
        error => raise Protocol(error)
      } noraise {
        value => value
      }
      let seed_value = try row.sampling_seed_value() catch {
        error => raise Protocol(error)
      } noraise {
        value => value
      }
      let sample_index = try row.sample_index() catch {
        error => raise Protocol(error)
      } noraise {
        value => value
      }
      match self.completion.mode {
        HostLogitsCompletion =>
          self.completion.selected_tokens[prefill_index] = read_and_select_paged_row(
            self,
            descriptor,
            prefill_index,
            summary.token_count,
            parameters,
            seed_value,
            sample_index,
          )
        CudaGreedyCompletion(_) =>
          self.completion.selected_tokens[prefill_index] = select_paged_cuda_greedy_result(
            self, descriptor, prefill_index,
          )
      }
    }
  }
  for decode_index in 0..<summary.decode_count {
    let row_index = summary.prefill_count + decode_index
    let row = try plan.decode_row(decode_index) catch {
      error => raise Protocol(error)
    } noraise {
      value => value
    }
    let parameters = try row.sampling() catch {
      error => raise Protocol(error)
    } noraise {
      value => value
    }
    let seed_value = try row.sampling_seed_value() catch {
      error => raise Protocol(error)
    } noraise {
      value => value
    }
    let sample_index = try row.sample_index() catch {
      error => raise Protocol(error)
    } noraise {
      value => value
    }
    match self.completion.mode {
      HostLogitsCompletion =>
        self.completion.selected_tokens[row_index] = read_and_select_paged_row(
          self,
          descriptor,
          row_index,
          summary.token_count,
          parameters,
          seed_value,
          sample_index,
        )
      CudaGreedyCompletion(_) =>
        self.completion.selected_tokens[row_index] = select_paged_cuda_greedy_result(
          self, descriptor, row_index,
        )
    }
  }
  append_paged_completion(
    plan,
    writer,
    summary,
    self.completion.selected_tokens,
  )
}
