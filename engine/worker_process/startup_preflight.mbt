///|
fn perform_startup_handshake_status(
  child : @process.ChildProcess,
  source : @worker_wire.EncodedBootstrapSource,
  startup_io_timeout_millis : Int,
  prepared : PreparedWorkerProcessStartup,
) -> Int {
  let configure_status = child.startup_write_frame_status(
    prepared.startup_outgoing,
    prepared.startup_outgoing_length,
    startup_io_timeout_millis,
  )
  if configure_status != 0 {
    return 100 + configure_status
  }
  let source_status = child.startup_write_frame_status(
    prepared.source_outgoing,
    source.length(),
    startup_io_timeout_millis,
  )
  if source_status != 0 {
    return 100 + source_status
  }
  let response_length = child.startup_read_frame_status(
    prepared.startup_incoming,
    startup_io_timeout_millis,
  )
  if response_length <= 0 {
    return 200 - response_length
  }
  if response_length != prepared.expected_ready_length {
    return 300
  }
  for index in 0..<response_length {
    if prepared.startup_incoming[index] != prepared.expected_ready[index] {
      return 300
    }
  }
  0
}

///|
fn preflight_prepare(
  executable : Bytes,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  limits : WorkerProcessLimits,
) -> PreparedWorkerProcessStartup raise WorkerProcessError {
  if source.digest() != startup.bootstrap_source_digest() {
    raise Invalid(SourceMismatch)
  }
  if executable.length() == 0 {
    raise Invalid(Submission)
  }
  for index = 0; index < executable.length(); index = index + 1 {
    if executable[index] == b'\x00' {
      raise Invalid(Submission)
    }
  }
  let worker_limits = startup.worker_limits()
  let inference_limits = startup.inference_limits()
  let plan_a = @worker_wire.PlanFrameBuffer::new(
    worker_limits, inference_limits,
  ) catch {
    error => raise Wire(error)
  }
  let plan_b = @worker_wire.PlanFrameBuffer::new(
    worker_limits, inference_limits,
  ) catch {
    error => raise Wire(error)
  }
  let completion_a = @worker_wire.CompletionFrameBuffer::new(
    worker_limits, inference_limits,
  ) catch {
    error => raise Wire(error)
  }
  let completion_b = @worker_wire.CompletionFrameBuffer::new(
    worker_limits, inference_limits,
  ) catch {
    error => raise Wire(error)
  }
  let startup_frame = @worker_wire.StartupFrameBuffer::new()
  let required = plan_a
    .capacity()
    .max(completion_a.capacity())
    .max(startup_frame.capacity())
    .max(source.capacity())
  if required > limits.max_frame_bytes {
    raise Invalid(Capacity)
  }
  let process_limits = @process.ProcessLimits::new(
    max_frame_bytes=limits.max_frame_bytes,
    io_timeout_millis=limits.io_timeout_millis,
    shutdown_timeout_millis=limits.shutdown_timeout_millis,
  ) catch {
    error => raise transport_error(error)
  }
  let startup_sender = @worker_wire.StartupFrameBuffer::new()
  let executable_snapshot = Bytes::makei(executable.length(), index => {
    executable[index]
  })
  let child_preparation = @process.prepare_child(
    executable_snapshot, process_limits,
  ) catch {
    error => raise transport_error(error)
  }
  let child = child_preparation.child_owner()
  let startup_outgoing = FixedArray::make(startup_sender.capacity(), b'\x00')
  let source_outgoing = FixedArray::make(source.length(), b'\x00')
  let startup_incoming = FixedArray::make(startup_sender.capacity(), b'\x00')
  let expected_ready = FixedArray::make(startup_sender.capacity(), b'\x00')
  let configure = startup_sender.encode(Configure, startup) catch {
    error => raise Wire(error)
  }
  let startup_outgoing_length = configure.length() catch {
    error => raise Wire(error)
  }
  configure.copy_to(startup_outgoing, destination_offset=0) catch {
    error => raise Wire(error)
  }
  source.copy_to(source_outgoing, destination_offset=0) catch {
    error => raise Wire(error)
  }
  let ready = startup_sender.encode(Ready, startup) catch {
    error => raise Wire(error)
  }
  let expected_ready_length = ready.length() catch {
    error => raise Wire(error)
  }
  ready.copy_to(expected_ready, destination_offset=0) catch {
    error => raise Wire(error)
  }
  let tx_a = FixedArray::make(plan_a.capacity(), b'\x00')
  let tx_b = FixedArray::make(plan_b.capacity(), b'\x00')
  let rx = FixedArray::make(completion_a.capacity(), b'\x00')
  let rx_staging = FixedArray::make(completion_a.capacity(), b'\x00')
  let exchange_request_ids = FixedArray::make(
    worker_limits.max_completion_slots(),
    0UL,
  )
  let exchange_request_generations = FixedArray::make(
    worker_limits.max_completion_slots(),
    0UL,
  )
  let exchange_expected_kinds = FixedArray::make(
    worker_limits.max_completion_slots(),
    ExpectedEmpty,
  )
  let exchange_processed_tokens = FixedArray::make(
    worker_limits.max_completion_slots(),
    0,
  )
  let exchange_slots = FixedArray::make(
    worker_limits.max_completion_slots(),
    -1,
  )
  let supervisor : WorkerProcessSupervisor = {
    child,
    child_selected: false,
    startup,
    source,
    plan_a,
    plan_b,
    completion_a,
    completion_b,
    tx_a,
    tx_b,
    rx,
    rx_staging,
    exchange_request_ids,
    exchange_request_generations,
    exchange_expected_kinds,
    exchange_processed_tokens,
    exchange_slots,
    exchange_entry_count: 0,
    state_a: 0,
    state_b: 0,
    sequence_a: 0UL,
    sequence_b: 0UL,
    last_submitted: if startup.has_predecessor() {
      startup.predecessor_value()
    } else {
      0UL
    },
    last_retired: if startup.has_predecessor() {
      startup.predecessor_value()
    } else {
      0UL
    },
    lifecycle: Ready,
  }
  let failed : FailedWorkerProcessStartup = {
    child,
    child_selected: false,
    cause_kind: 0,
    cause_status: 0,
    cleanup_status: 0,
    closed: false,
  }
  {
    executable: executable_snapshot,
    child: child_preparation,
    startup_outgoing,
    startup_outgoing_length,
    source_outgoing,
    startup_incoming,
    expected_ready,
    expected_ready_length,
    supervisor,
    failed,
  }
}
