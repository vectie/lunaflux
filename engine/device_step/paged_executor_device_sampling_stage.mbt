///|
fn stage_submitted_device_sampling_parameters(
  scratch : PagedDeviceSamplingScratch,
  plan : @worker_protocol.SubmittedSchedulePlan,
  summary : DeviceStepSummary,
  compact? : Bool = false,
) -> Unit raise DeviceStepError {
  let mut output_index = 0
  for index in 0..<summary.prefill_count {
    let row = plan.prefill_row(index) catch { error => raise Protocol(error) }
    let produces = row.produces_token() catch { error => raise Protocol(error) }
    if !produces {
      if !compact {
        write_paged_device_sampling_fallback(scratch.parameters, index)
      }
      continue
    }
    let parameters = row.sampling() catch { error => raise Protocol(error) }
    let mode = match parameters.mode() {
      Greedy => 0
      Sample => 1
    }
    let (has_top_k, top_k) = match parameters.top_k() {
      Some(value) => (true, value)
      None => (false, 0)
    }
    let (has_top_p, top_p) = match parameters.top_p() {
      Some(value) => (true, value)
      None => (false, 0.0)
    }
    stage_paged_device_sampling_row(
      scratch,
      if compact {
        output_index
      } else {
        index
      },
      true,
      mode,
      parameters.temperature(),
      has_top_k,
      top_k,
      has_top_p,
      top_p,
      row.sampling_seed_value() catch {
        error => raise Protocol(error)
      },
      row.sample_index() catch {
        error => raise Protocol(error)
      },
    )
    output_index += 1
  }
  for index in 0..<summary.decode_count {
    let row_index = if compact {
      output_index
    } else {
      summary.prefill_count + index
    }
    let row = plan.decode_row(index) catch { error => raise Protocol(error) }
    let parameters = row.sampling() catch { error => raise Protocol(error) }
    let mode = match parameters.mode() {
      Greedy => 0
      Sample => 1
    }
    let (has_top_k, top_k) = match parameters.top_k() {
      Some(value) => (true, value)
      None => (false, 0)
    }
    let (has_top_p, top_p) = match parameters.top_p() {
      Some(value) => (true, value)
      None => (false, 0.0)
    }
    stage_paged_device_sampling_row(
      scratch,
      row_index,
      true,
      mode,
      parameters.temperature(),
      has_top_k,
      top_k,
      has_top_p,
      top_p,
      row.sampling_seed_value() catch {
        error => raise Protocol(error)
      },
      row.sample_index() catch {
        error => raise Protocol(error)
      },
    )
    output_index += 1
  }
  if compact && output_index == 1 && summary.row_count > 1 {
    write_paged_device_sampling_fallback(scratch.parameters, 1)
  }
}

///|
fn stage_wire_device_sampling_parameters(
  scratch : PagedDeviceSamplingScratch,
  frame : @worker_wire.ValidatedPlanFrame,
  summary : DeviceStepSummary,
  compact? : Bool = false,
) -> Unit raise DeviceStepError {
  let mut output_index = 0
  for index in 0..<summary.prefill_count {
    let row = frame.prefill_row(index) catch { error => raise Wire(error) }
    let produces = row.produces_token() catch { error => raise Wire(error) }
    if compact && !produces {
      continue
    }
    stage_paged_device_sampling_row(
      scratch,
      if compact {
        output_index
      } else {
        index
      },
      produces,
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
    output_index += 1
  }
  for index in 0..<summary.decode_count {
    let row_index = if compact {
      output_index
    } else {
      summary.prefill_count + index
    }
    let row = frame.decode_row(index) catch { error => raise Wire(error) }
    stage_paged_device_sampling_row(
      scratch,
      row_index,
      true,
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
    output_index += 1
  }
  if compact && output_index == 1 && summary.row_count > 1 {
    write_paged_device_sampling_fallback(scratch.parameters, 1)
  }
}

///|
fn upload_paged_device_sampling_parameters(
  executor : PagedGraphExecutor,
  descriptor : DeviceStepOwner,
  row_count : Int,
) -> Unit raise DeviceStepError {
  guard executor.completion.mode is CudaRestrictiveTopKCompletion(scratch) &&
    executor.resources.device_sampling_parameters is Some(owner) &&
    owner.open &&
    row_count > 0 &&
    row_count <= scratch.plan.maximum_rows() else {
    raise InvalidExecutor(ResourceState)
  }
  descriptor.context.copy_from_fixed_host(
    owner.resource,
    scratch.parameters,
    0L,
    0L,
    row_count.to_int64() * scratch.plan.parameter_row_bytes(),
  ) catch {
    error => raise ExecutorDevice(SamplingParameters, error)
  }
}

///|
fn stage_paged_device_sampling_plan(
  executor : PagedGraphExecutor,
  descriptor : DeviceStepOwner,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> Unit raise DeviceStepError {
  guard executor.completion.mode is CudaRestrictiveTopKCompletion(scratch) else {
    return
  }
  stage_submitted_device_sampling_parameters(
    scratch,
    plan,
    descriptor.staged_summary,
    compact=descriptor.host.project_output_rows,
  )
  if descriptor.host.project_output_rows && descriptor.host.output_rows == 0 {
    return
  }
  upload_paged_device_sampling_parameters(
    executor,
    descriptor,
    if descriptor.host.project_output_rows {
      output_physical_rows(descriptor.host, descriptor.staged_summary)
    } else {
      descriptor.staged_summary.row_count
    },
  )
}

///|
fn stage_paged_device_sampling_frame(
  executor : PagedGraphExecutor,
  descriptor : DeviceStepOwner,
  frame : @worker_wire.ValidatedPlanFrame,
) -> Unit raise DeviceStepError {
  guard executor.completion.mode is CudaRestrictiveTopKCompletion(scratch) else {
    return
  }
  stage_wire_device_sampling_parameters(
    scratch,
    frame,
    descriptor.staged_summary,
    compact=descriptor.host.project_output_rows,
  )
  if descriptor.host.project_output_rows && descriptor.host.output_rows == 0 {
    return
  }
  upload_paged_device_sampling_parameters(
    executor,
    descriptor,
    if descriptor.host.project_output_rows {
      output_physical_rows(descriptor.host, descriptor.staged_summary)
    } else {
      descriptor.staged_summary.row_count
    },
  )
}

///|
fn copy_paged_cuda_device_sampling_results(
  executor : PagedGraphExecutor,
  descriptor : DeviceStepOwner,
  row_count : Int,
) -> Unit raise DeviceStepError {
  let row_count = projected_result_count(descriptor, row_count)
  guard executor.completion.mode is CudaRestrictiveTopKCompletion(scratch) else {
    return
  }
  guard executor.resources.device_sampling_results is Some(owner) &&
    owner.open &&
    row_count >= 0 &&
    row_count <= scratch.plan.maximum_rows() else {
    poison_paged_completion(executor, descriptor)
    raise InvalidExecutor(SamplingOutput)
  }
  descriptor.context.copy_to_fixed_host(
    owner.resource,
    scratch.results,
    0L,
    0L,
    row_count.to_int64() * scratch.plan.result_row_bytes(),
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
fn select_paged_cuda_device_sampling_result(
  executor : PagedGraphExecutor,
  descriptor : DeviceStepOwner,
  row : Int,
) -> Int raise DeviceStepError {
  guard executor.completion.mode is CudaRestrictiveTopKCompletion(scratch) &&
    row >= 0 &&
    row < scratch.plan.maximum_rows() else {
    return poison_paged_completion_and_raise(
      executor,
      descriptor,
      InvalidExecutor(SamplingOutput),
    )
  }
  let output_row = projected_result_row(descriptor, row)
  if output_row < 0 || output_row >= scratch.plan.maximum_rows() {
    return poison_paged_completion_and_raise(
      executor,
      descriptor,
      InvalidExecutor(SamplingOutput),
    )
  }
  let token = read_paged_i32(scratch.results, output_row * 2)
  let status = read_paged_i32(scratch.results, output_row * 2 + 1)
  if status >= 0 {
    return poison_paged_completion_and_raise(
      executor,
      descriptor,
      ExecutorSampling(NonFiniteLogit(row~, token_id=status)),
    )
  }
  if status != -1 || token < 0 || token >= scratch.plan.vocabulary_size() {
    return poison_paged_completion_and_raise(
      executor,
      descriptor,
      InvalidExecutor(SamplingOutput),
    )
  }
  token
}
