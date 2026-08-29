///|
priv enum TensorParallelCompletionRole {
  LeaderRole
  FollowerRole
} derive(Eq)

///|
fn completion_role(rank : Int) -> TensorParallelCompletionRole {
  if rank == 0 {
    LeaderRole
  } else {
    FollowerRole
  }
}

///|
fn decode_bf16_logits(scratch : TensorParallelLeaderScratch) -> Unit {
  for token_id in 0..<scratch.vocabulary_size {
    let offset = token_id * 2
    let low = scratch.encoded_logits[offset].to_uint()
    let high = scratch.encoded_logits[offset + 1].to_uint()
    let bits = ((high << 8) | low) << 16
    scratch.logits[token_id] = Float::reinterpret_from_uint(bits).to_double()
  }
}

///|
fn select_wire_logit(
  scratch : TensorParallelLeaderScratch,
  sampling_mode : Int,
  temperature : Double,
  has_top_k : Bool,
  top_k : Int,
  has_top_p : Bool,
  top_p : Double,
  seed_value : UInt64,
  sample_index : Int,
) -> Int raise TensorParallelDeviceWorkerError {
  if sampling_mode != 0 && sampling_mode != 1 {
    raise Invalid(FinalOutput)
  }
  let selection = try {
    if sampling_mode == 0 {
      @sampling.greedy(scratch.logits[:], scratch.sampling_limits)
    } else {
      @sampling.stochastic_sample_at_scalars(
        scratch.logits[:],
        temperature~,
        has_top_k~,
        top_k~,
        has_top_p~,
        top_p~,
        scratch.sampling_limits,
        try! @inference.SamplingSeed::new(seed_value),
        sample_index,
        scratch.sampler,
      )
    }
  } catch {
    error => raise Sampling(error)
  } noraise {
    value => value
  }
  selection.token_id()
}

///|
fn read_and_select_row(
  owner : TensorParallelDeviceWorkerOwner,
  exact : @device_step.StagedDeviceStep,
  scratch : TensorParallelLeaderScratch,
  row : Int,
  sampling_mode : Int,
  temperature : Double,
  has_top_k : Bool,
  top_k : Int,
  has_top_p : Bool,
  top_p : Double,
  seed_value : UInt64,
  sample_index : Int,
) -> Int raise TensorParallelDeviceWorkerError {
  let token_index = exact.last_token_index(row) catch {
    error => raise Descriptor(error)
  }
  owner.resources.context.copy_to_fixed_host(
    owner.resources.activation_workspace.resource,
    scratch.encoded_logits,
    scratch.final_output_offset + token_index.to_int64() * scratch.row_bytes,
    0L,
    scratch.row_bytes,
  ) catch {
    error => raise Device(ActivationWorkspace, error)
  }
  decode_bf16_logits(scratch)
  select_wire_logit(
    scratch, sampling_mode, temperature, has_top_k, top_k, has_top_p, top_p, seed_value,
    sample_index,
  )
}

///|
fn sample_leader_rows(
  owner : TensorParallelDeviceWorkerOwner,
  exact : @device_step.StagedDeviceStep,
  frame : @worker_wire.ValidatedPlanFrame,
  scratch : TensorParallelLeaderScratch,
) -> Int raise TensorParallelDeviceWorkerError {
  let prefill_count = frame.prefill_count() catch { error => raise Wire(error) }
  let decode_count = frame.decode_count() catch { error => raise Wire(error) }
  let admitted_prefill_count = require_leader_row_counts(
    owner.active_rows,
    prefill_count,
    decode_count,
  )
  for index in 0..<prefill_count {
    let row = frame.prefill_row(index) catch { error => raise Wire(error) }
    let produces = row.produces_token() catch { error => raise Wire(error) }
    if produces {
      scratch.selected_tokens[index] = read_and_select_row(
        owner,
        exact,
        scratch,
        index,
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
  for index in 0..<decode_count {
    let row = frame.decode_row(index) catch { error => raise Wire(error) }
    let output_row = prefill_count + index
    scratch.selected_tokens[output_row] = read_and_select_row(
      owner,
      exact,
      scratch,
      output_row,
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
  admitted_prefill_count
}

///|
fn require_leader_row_counts(
  active_rows : Int,
  prefill_count : Int,
  decode_count : Int,
) -> Int raise TensorParallelDeviceWorkerError {
  if active_rows < 0 ||
    prefill_count < 0 ||
    decode_count < 0 ||
    prefill_count > active_rows ||
    decode_count != active_rows - prefill_count {
    raise Invalid(PlanFrame)
  }
  prefill_count
}

///|
fn append_leader_rows(
  frame : @worker_wire.ValidatedPlanFrame,
  writer : @worker_wire.CompletionFrameWriter,
  scratch : TensorParallelLeaderScratch,
  prefill_count : Int,
  decode_count : Int,
) -> Unit raise TensorParallelDeviceWorkerError {
  for index in 0..<prefill_count {
    let row = frame.prefill_row(index) catch { error => raise Wire(error) }
    let produces = row.produces_token() catch { error => raise Wire(error) }
    if produces {
      writer.append_final_prefill_completed(row, scratch.selected_tokens[index]) catch {
        error => raise Wire(error)
      }
    } else {
      writer.append_prefill_completed(row) catch {
        error => raise Wire(error)
      }
    }
  }
  for index in 0..<decode_count {
    let row = frame.decode_row(index) catch { error => raise Wire(error) }
    writer.append_decode_completed(
      row,
      scratch.selected_tokens[prefill_count + index],
    ) catch {
      error => raise Wire(error)
    }
  }
}

///|
fn fail_open_writer(
  owner : TensorParallelDeviceWorkerOwner,
  writer : @worker_wire.CompletionFrameWriter,
  primary : TensorParallelDeviceWorkerError,
) -> TensorParallelDeviceWorkerError {
  writer.abort() catch {
    error =>
      if owner.cleanup_failure is None {
        owner.cleanup_failure = Some(Wire(error))
      }
  }
  fail_rank_execution(owner, primary)
}

///|
fn retire_success(
  owner : TensorParallelDeviceWorkerOwner,
  exact : @device_step.StagedDeviceStep,
) -> Unit raise TensorParallelDeviceWorkerError {
  let sequence = owner.active_sequence
  owner.resources.executor.resource.reset() catch {
    error => raise Device(KernelExecutor, error)
  }
  owner.resources.descriptor.finish(exact) catch {
    error => raise Descriptor(error)
  }
  owner.lifecycle = WorkerReady
  owner.active_epoch = 0UL
  owner.active_sequence = 0UL
  owner.active_rows = 0
  owner.active_tokens = 0
  owner.operation_index = 0
  owner.collective_index = 0
  owner.resources.collective.abort_required = false
  owner.resources.executor.abort_required = false
  match owner.resources.executor.resource.execution_mode() {
    OrderedKernelCaptured =>
      if owner.graph_hits != 0xffffffffffffffffUL {
        owner.graph_hits += 1UL
      }
    OrderedKernelEager =>
      if owner.graph_misses != 0xffffffffffffffffUL {
        owner.graph_misses += 1UL
      }
  }
  owner.last_graph_sequence = sequence
}

///|
fn finish_leader(
  owner : TensorParallelDeviceWorkerOwner,
  exact : @device_step.StagedDeviceStep,
  frame : @worker_wire.ValidatedPlanFrame,
  scratch : TensorParallelLeaderScratch,
) -> @worker_wire.ValidatedCompletionFrame raise TensorParallelDeviceWorkerError {
  let prefill_count = sample_leader_rows(owner, exact, frame, scratch) catch {
    error => raise fail_rank_execution(owner, error)
  }
  let decode_count = owner.active_rows - prefill_count
  let writer = scratch.completion.writer_for(frame) catch {
    error => {
      let translated = Wire(error)
      raise fail_rank_execution(owner, translated)
    }
  }
  append_leader_rows(frame, writer, scratch, prefill_count, decode_count) catch {
    error => raise fail_open_writer(owner, writer, error)
  }
  retire_success(owner, exact) catch {
    error => raise fail_open_writer(owner, writer, error)
  }
  let completion = writer.submit() catch {
    error => {
      let translated = Wire(error)
      raise fail_rank_execution(owner, translated)
    }
  }
  completion
}

///|
fn finish_follower(
  owner : TensorParallelDeviceWorkerOwner,
  exact : @device_step.StagedDeviceStep,
) -> TensorParallelFollowerExecutionAck raise TensorParallelDeviceWorkerError {
  let sequence = owner.active_sequence
  retire_success(owner, exact) catch {
    error => raise fail_rank_execution(owner, error)
  }
  {
    group_generation_value: owner.plan.group_generation.value(),
    model_generation_value: owner.plan.model_generation.value(),
    plan_sequence_value: sequence,
    rank: owner.plan.execution.rank(),
    world_size: owner.plan.execution.world_size(),
  }
}

///|
/// Retires one exact executed plan. Rank zero alone owns readback, sampling,
/// and completion-frame publication; every follower can return only its
/// generation/sequence/rank acknowledgement.
pub fn TensorParallelDeviceWorkerOwner::finish_leader_frame(
  self : TensorParallelDeviceWorkerOwner,
  executed : ExecutedTensorParallelDeviceExecution,
  frame : @worker_wire.ValidatedPlanFrame,
) -> @worker_wire.ValidatedCompletionFrame raise TensorParallelDeviceWorkerError {
  let exact = authenticate_executed_execution(executed, self, frame)
  guard completion_role(self.plan.execution.rank()) == LeaderRole &&
    self.completion is LeaderAuthority(scratch) else {
    let error = Invalid(CompletionRole)
    raise fail_rank_execution(self, error)
  }
  finish_leader(self, exact, frame, scratch)
}

///|
/// Retires one exact follower execution and returns only its scalar ack.
pub fn TensorParallelDeviceWorkerOwner::finish_follower(
  self : TensorParallelDeviceWorkerOwner,
  executed : ExecutedTensorParallelDeviceExecution,
  frame : @worker_wire.ValidatedPlanFrame,
) -> TensorParallelFollowerExecutionAck raise TensorParallelDeviceWorkerError {
  let exact = authenticate_executed_execution(executed, self, frame)
  guard completion_role(self.plan.execution.rank()) == FollowerRole &&
    self.completion is FollowerAuthority else {
    let error = Invalid(CompletionRole)
    raise fail_rank_execution(self, error)
  }
  finish_follower(self, exact)
}
