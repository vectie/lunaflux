///|
fn validate_wire_completion_plan(
  descriptor : DeviceStepOwner,
  frame : @worker_wire.ValidatedPlanFrame,
  writer : @worker_wire.CompletionFrameWriter,
) -> DeviceStepSummary raise DeviceStepError {
  guard descriptor.staged_wire_owner is Some(owner) else {
    raise InvalidExecutor(ResourceState)
  }
  let matches = frame.matches_owner_epoch(owner, descriptor.staged_wire_epoch) catch {
    error => raise Wire(error)
  }
  if !matches {
    raise ExecutorOwnerMismatch
  }
  let sequence_value = frame.sequence_value() catch {
    error => raise Wire(error)
  }
  if sequence_value != descriptor.staged_summary.sequence_value {
    raise ExecutorOwnerMismatch
  }
  writer.validate_plan(frame) catch {
    error => raise Wire(error)
  }
  descriptor.staged_summary
}

///|
fn select_paged_wire_logit(
  executor : PagedGraphExecutor,
  descriptor : DeviceStepOwner,
  row_index : Int,
  token_count : Int,
  sampling_mode : Int,
  temperature : Double,
  has_top_k : Bool,
  top_k : Int,
  has_top_p : Bool,
  top_p : Double,
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
  descriptor.context.copy_to_fixed_host(
    activation,
    scratch.encoded_logits,
    scratch.final_output_offset + token_index.to_int64() * scratch.row_bytes,
    0L,
    scratch.row_bytes,
  ) catch {
    error => {
      poison_paged_completion(executor, descriptor)
      raise ExecutorDevice(ActivationReadback, error)
    }
  }
  decode_paged_bf16_logits(scratch)
  select_paged_scalar_logit(
    scratch, sampling_mode, temperature, has_top_k, top_k, has_top_p, top_p, seed_value,
    sample_index,
  ) catch {
    error => poison_paged_completion_and_raise(executor, descriptor, error)
  }
}

///|
fn select_paged_scalar_logit(
  scratch : PagedCompletionScratch,
  sampling_mode : Int,
  temperature : Double,
  has_top_k : Bool,
  top_k : Int,
  has_top_p : Bool,
  top_p : Double,
  seed_value : UInt64,
  sample_index : Int,
) -> Int raise DeviceStepError {
  let logits = scratch.logits[:]
  if sampling_mode != 0 && sampling_mode != 1 {
    raise InvalidExecutor(SamplingOutput)
  }
  let selection = try {
    if sampling_mode == 0 {
      @sampling.greedy(logits, scratch.limits)
    } else {
      @sampling.stochastic_sample_at_scalars(
        logits,
        temperature~,
        has_top_k~,
        top_k~,
        has_top_p~,
        top_p~,
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
fn sample_wire_prefill(
  executor : PagedGraphExecutor,
  descriptor : DeviceStepOwner,
  frame : @worker_wire.ValidatedPlanFrame,
  index : Int,
  token_count : Int,
) -> Unit raise DeviceStepError {
  let row = frame.prefill_row(index) catch { error => raise Wire(error) }
  let produces = row.produces_token() catch { error => raise Wire(error) }
  if produces {
    executor.completion.selected_tokens[index] = match
      executor.completion.mode {
      CudaGreedyCompletion(_) =>
        select_paged_cuda_greedy_result(executor, descriptor, index)
      CudaRestrictiveTopKCompletion(scratch) => {
        let mode = row.sampling_mode_value() catch {
          error => raise Wire(error)
        }
        let temperature = row.sampling_temperature() catch {
          error => raise Wire(error)
        }
        let has_top_k = row.sampling_has_top_k() catch {
          error => raise Wire(error)
        }
        let top_k = row.sampling_top_k_value() catch {
          error => raise Wire(error)
        }
        let has_top_p = row.sampling_has_top_p() catch {
          error => raise Wire(error)
        }
        let top_p = row.sampling_top_p_value() catch {
          error => raise Wire(error)
        }
        if paged_device_sampling_route(
            scratch.plan,
            mode,
            temperature,
            has_top_k,
            top_k,
            has_top_p,
            top_p,
          ) ==
          DeviceRestrictiveTopK {
          select_paged_cuda_device_sampling_result(executor, descriptor, index)
        } else {
          select_paged_wire_logit(
            executor,
            descriptor,
            index,
            token_count,
            mode,
            temperature,
            has_top_k,
            top_k,
            has_top_p,
            top_p,
            row.sampling_seed_value() catch {
              error => raise Wire(error)
            },
            row.sample_index() catch {
              error => raise Wire(error)
            },
          )
        }
      }
      HostLogitsCompletion =>
        select_paged_wire_logit(
          executor,
          descriptor,
          index,
          token_count,
          row.sampling_mode_value() catch {
            error => raise Wire(error)
          },
          row.sampling_temperature() catch {
            error => raise Wire(error)
          },
          row.sampling_has_top_k() catch {
            error => raise Wire(error)
          },
          row.sampling_top_k_value() catch {
            error => raise Wire(error)
          },
          row.sampling_has_top_p() catch {
            error => raise Wire(error)
          },
          row.sampling_top_p_value() catch {
            error => raise Wire(error)
          },
          row.sampling_seed_value() catch {
            error => raise Wire(error)
          },
          row.sample_index() catch {
            error => raise Wire(error)
          },
        )
    }
  }
}

///|
fn sample_wire_decode(
  executor : PagedGraphExecutor,
  descriptor : DeviceStepOwner,
  frame : @worker_wire.ValidatedPlanFrame,
  decode_index : Int,
  row_index : Int,
  token_count : Int,
) -> Unit raise DeviceStepError {
  let row = frame.decode_row(decode_index) catch { error => raise Wire(error) }
  executor.completion.selected_tokens[row_index] = match
    executor.completion.mode {
    CudaGreedyCompletion(_) =>
      select_paged_cuda_greedy_result(executor, descriptor, row_index)
    CudaRestrictiveTopKCompletion(scratch) => {
      let mode = row.sampling_mode_value() catch { error => raise Wire(error) }
      let temperature = row.sampling_temperature() catch {
        error => raise Wire(error)
      }
      let has_top_k = row.sampling_has_top_k() catch {
        error => raise Wire(error)
      }
      let top_k = row.sampling_top_k_value() catch {
        error => raise Wire(error)
      }
      let has_top_p = row.sampling_has_top_p() catch {
        error => raise Wire(error)
      }
      let top_p = row.sampling_top_p_value() catch {
        error => raise Wire(error)
      }
      if paged_device_sampling_route(
          scratch.plan,
          mode,
          temperature,
          has_top_k,
          top_k,
          has_top_p,
          top_p,
        ) ==
        DeviceRestrictiveTopK {
        select_paged_cuda_device_sampling_result(
          executor, descriptor, row_index,
        )
      } else {
        select_paged_wire_logit(
          executor,
          descriptor,
          row_index,
          token_count,
          mode,
          temperature,
          has_top_k,
          top_k,
          has_top_p,
          top_p,
          row.sampling_seed_value() catch {
            error => raise Wire(error)
          },
          row.sample_index() catch {
            error => raise Wire(error)
          },
        )
      }
    }
    HostLogitsCompletion =>
      select_paged_wire_logit(
        executor,
        descriptor,
        row_index,
        token_count,
        row.sampling_mode_value() catch {
          error => raise Wire(error)
        },
        row.sampling_temperature() catch {
          error => raise Wire(error)
        },
        row.sampling_has_top_k() catch {
          error => raise Wire(error)
        },
        row.sampling_top_k_value() catch {
          error => raise Wire(error)
        },
        row.sampling_has_top_p() catch {
          error => raise Wire(error)
        },
        row.sampling_top_p_value() catch {
          error => raise Wire(error)
        },
        row.sampling_seed_value() catch {
          error => raise Wire(error)
        },
        row.sample_index() catch {
          error => raise Wire(error)
        },
      )
  }
}

///|
fn append_wire_completion_frame(
  frame : @worker_wire.ValidatedPlanFrame,
  writer : @worker_wire.CompletionFrameWriter,
  summary : DeviceStepSummary,
  selected : FixedArray[Int],
) -> Unit raise DeviceStepError {
  for index in 0..<summary.prefill_count {
    let row = frame.prefill_row(index) catch { error => raise Wire(error) }
    let produces = row.produces_token() catch { error => raise Wire(error) }
    if produces {
      writer.append_final_prefill_completed(row, selected[index]) catch {
        error => raise Wire(error)
      }
    } else {
      writer.append_prefill_completed(row) catch {
        error => raise Wire(error)
      }
    }
  }
  for index in 0..<summary.decode_count {
    let row = frame.decode_row(index) catch { error => raise Wire(error) }
    writer.append_decode_completed(row, selected[summary.prefill_count + index]) catch {
      error => raise Wire(error)
    }
  }
}

///|
/// Reads terminal BF16 logits and appends a canonical completion frame using
/// the exact plan-frame owner and epoch previously staged by this executor.
/// The writer deliberately remains open: its aggregate owner must retire the
/// executed graph successfully before publishing with `writer.submit()`.
/// Sampling/readback failure poisons execution. Any sampling or write failure
/// leaves the exclusive writer open so its owner can abort explicitly.
pub fn PagedGraphExecutor::sample_completion_frame(
  self : PagedGraphExecutor,
  executed : ExecutedPagedGraphExecution,
  frame : @worker_wire.ValidatedPlanFrame,
  writer : @worker_wire.CompletionFrameWriter,
) -> Unit raise DeviceStepError {
  let descriptor = authenticate_executed_paged(executed, self)
  let summary = validate_wire_completion_plan(descriptor, frame, writer)
  copy_paged_cuda_greedy_results(self, descriptor, summary.row_count)
  copy_paged_cuda_device_sampling_results(self, descriptor, summary.row_count)
  for index in 0..<summary.prefill_count {
    sample_wire_prefill(self, descriptor, frame, index, summary.token_count)
  }
  for index in 0..<summary.decode_count {
    sample_wire_decode(
      self,
      descriptor,
      frame,
      index,
      summary.prefill_count + index,
      summary.token_count,
    )
  }
  append_wire_completion_frame(
    frame,
    writer,
    summary,
    self.completion.selected_tokens,
  )
}
