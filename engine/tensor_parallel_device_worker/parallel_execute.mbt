///|
// The same readiness rule is used by the executor and queue replay tests.
// CUDA waits on unrecorded events are no-ops; never enqueue such a wait.
fn parallel_command_ready(
  command : ParallelCommand,
  recorded : FixedArray[Bool],
  communication : Bool,
  collective_pending : Bool,
  submission_pending : Bool,
) -> Bool {
  if communication && submission_pending {
    return false
  }
  match command {
    Wait(event) => recorded[event] && (!communication || !collective_pending)
    Collective(_) => !collective_pending
    _ => true
  }
}

///|
fn advance_parallel_lane(
  owner : TensorParallelDeviceWorkerOwner,
  overlap : TensorParallelOverlap,
  communication : Bool,
) -> Bool raise TensorParallelDeviceWorkerError {
  let commands = if communication {
    overlap.commands.communication
  } else {
    overlap.commands.compute
  }
  let cursor = if communication {
    overlap.communication_cursor
  } else {
    overlap.compute_cursor
  }
  if cursor == commands.length() {
    return false
  }
  let command = commands[cursor]
  if !parallel_command_ready(
      command,
      overlap.recorded,
      communication,
      overlap.collective_pending,
      overlap.submission_pending,
    ) {
    return false
  }
  let stream = if communication {
    overlap.communication_stream.resource
  } else {
    owner.resources.stream.resource
  }
  match command {
    Kernel(index) => {
      owner.resources.executor.abort_required = true
      owner.resources.executor.resource.enqueue(index) catch {
        error => raise Device(KernelExecutor, error)
      }
      owner.operation_index += 1
    }
    Collective(binding) => {
      submit_current_collective(owner, binding, stream~)
      overlap.collective_pending = true
      overlap.submission_pending = true
      owner.lifecycle = WorkerExecuting
    }
    Record(event) => {
      overlap.events[event].resource.record(stream) catch {
        error => raise Device(ParallelEvent, error)
      }
      overlap.recorded[event] = true
    }
    Wait(event) =>
      stream.wait_event(overlap.events[event].resource) catch {
        error => raise Device(ParallelEvent, error)
      }
  }
  if communication {
    overlap.communication_cursor += 1
  } else {
    overlap.compute_cursor += 1
  }
  true
}

///|
fn progress_overlap(
  owner : TensorParallelDeviceWorkerOwner,
  overlap : TensorParallelOverlap,
) -> Bool raise TensorParallelDeviceWorkerError {
  owner.lifecycle = WorkerExecuting
  if overlap.submission_pending {
    let ready = owner.resources.collective.resource.poll_submitted() catch {
      error => raise fail_collective(owner, Device(Collective, error))
    }
    if ready {
      overlap.submission_pending = false
    }
  }
  if overlap.collective_pending && !overlap.submission_pending {
    if poll_current_collective(owner) {
      overlap.collective_pending = false
    }
  }
  // Alternate ready commands in source FIFO order. Both cursors and all event
  // state are startup-owned; no closures, graph walks, or allocations here.
  for ;; {
    let compute = advance_parallel_lane(owner, overlap, false)
    let communication = advance_parallel_lane(owner, overlap, true)
    if !compute && !communication {
      break
    }
  }
  if overlap.compute_cursor == overlap.commands.compute.length() &&
    !overlap.completion_recorded {
    owner.resources.executor.resource.record_completion() catch {
      error => raise Device(KernelExecutor, error)
    }
    overlap.completion_recorded = true
  }
  // NCCL poll covers its stream tail; executor completion covers compute.
  // Do not retire either owner, descriptor or events until both have finished.
  if overlap.communication_cursor != overlap.commands.communication.length() ||
    overlap.collective_pending ||
    !overlap.completion_recorded {
    return false
  }
  poll_device_completion(owner)
}
