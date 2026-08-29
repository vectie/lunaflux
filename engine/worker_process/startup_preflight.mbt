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
  let attestation_status = child.startup_write_frame_status(
    prepared.attestation_outgoing,
    prepared.attestation_outgoing_length,
    startup_io_timeout_millis,
  )
  if attestation_status != 0 {
    return 100 + attestation_status
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
/// Validates the immutable startup envelope without borrowing executable or
/// root authority. Root-bound preparation uses this before role validation so
/// source/capacity errors remain platform-independent.
fn validate_preflight_envelope(
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  limits : WorkerProcessLimits,
) -> Unit raise WorkerProcessError {
  if source.digest() != startup.bootstrap_source_digest() {
    raise Invalid(SourceMismatch)
  }
  let worker_limits = startup.worker_limits()
  let plan_capacity = @worker_wire.plan_frame_byte_capacity(worker_limits) catch {
    error => raise Wire(error)
  }
  let completion_capacity = @worker_wire.completion_frame_byte_capacity(
    worker_limits,
  ) catch {
    error => raise Wire(error)
  }
  let required = plan_capacity
    .max(completion_capacity)
    .max(@worker_wire.StartupFrameBuffer::new().capacity())
    .max(source.capacity())
  if required > limits.max_frame_bytes {
    raise Invalid(Capacity)
  }
  ignore(
    @process.ProcessLimits::new(
      max_frame_bytes=limits.max_frame_bytes,
      io_timeout_millis=limits.io_timeout_millis,
      shutdown_timeout_millis=limits.shutdown_timeout_millis,
    ) catch {
      error => raise transport_error(error)
    },
  )
}

///|
fn preflight_prepare_with_authority(
  executable : @worker_executable_file.WorkerExecutableAdmission,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  limits : WorkerProcessLimits,
  parent_approval : @worker_wire.WorkerParentApprovalSubjects?,
) -> PreparedWorkerProcessStartup raise WorkerProcessError {
  validate_preflight_envelope(startup, source, limits)
  let worker_limits = startup.worker_limits()
  let inference_limits = startup.inference_limits()
  let plan_capacity = @worker_wire.plan_frame_byte_capacity(worker_limits) catch {
    error => raise Wire(error)
  }
  let tx_a = FixedArray::make(plan_capacity, b'\x00')
  let tx_b = FixedArray::make(plan_capacity, b'\x00')
  let plan_a = @worker_wire.PlanFrameBuffer::new_transport_owned(
    worker_limits,
    inference_limits,
    tx_a,
    frame_offset=0,
  ) catch {
    error => raise Wire(error)
  }
  let plan_b = @worker_wire.PlanFrameBuffer::new_transport_owned(
    worker_limits,
    inference_limits,
    tx_b,
    frame_offset=0,
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
  let child_preparation = executable.prepare_child(process_limits) catch {
    _ => raise Transport(SpawnFailed)
  }
  let child = child_preparation.child_owner()
  let startup_outgoing = FixedArray::make(startup_sender.capacity(), b'\x00')
  let source_outgoing = FixedArray::make(source.length(), b'\x00')
  let attestation_outgoing = FixedArray::make(
    @worker_wire.parent_approval_attestation_capacity(),
    b'\x00',
  )
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
  let attestation_outgoing_length = match parent_approval {
    None =>
      @worker_wire.encode_absent_parent_approval_attestation_v1(
        attestation_outgoing,
      ) catch {
        error => raise Wire(error)
      }
    Some(subjects) =>
      @worker_wire.encode_parent_approval_subjects_attestation_v1(
        attestation_outgoing,
        subjects,
        bootstrap_digest_sha256=startup.bootstrap_digest().as_hex(),
        source_digest_sha256=startup.bootstrap_source_digest().as_hex(),
        model_generation=startup.model_generation_value(),
        device_ordinal=startup.device_ordinal(),
      ) catch {
        error => raise Wire(error)
      }
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
  let rx = FixedArray::make(completion_a.capacity(), b'\x00')
  let rx_staging = FixedArray::make(completion_a.capacity(), b'\x00')
  let graph_rx = FixedArray::make(
    @worker_wire.worker_graph_telemetry_frame_capacity(),
    b'\x00',
  )
  let graph_rx_staging = FixedArray::make(
    @worker_wire.worker_graph_telemetry_frame_capacity(),
    b'\x00',
  )
  let supervisor : WorkerProcessSupervisor = {
    child,
    child_selected: false,
    startup,
    source,
    parent_approval,
    plan_a,
    plan_b,
    completion_a,
    completion_b,
    tx_a,
    tx_b,
    rx,
    rx_staging,
    graph_rx,
    graph_rx_staging,
    state_a: 0,
    state_b: 0,
    sequence_a: 0UL,
    sequence_b: 0UL,
    graph_telemetry_a: @worker_wire.WorkerGraphTelemetrySlot::new(),
    graph_telemetry_b: @worker_wire.WorkerGraphTelemetrySlot::new(),
    last_graph_telemetry: @worker_wire.WorkerGraphTelemetrySlot::new(),
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
    maintenance_phase: 0,
    maintenance_failure: None,
    lifecycle: Ready,
  }
  let failed : FailedWorkerProcessStartup = {
    child,
    child_selected: false,
    cause_kind: 0,
    cause_status: 0,
    cleanup_status: 0,
    cleanup_active: false,
    closed: false,
  }
  {
    approved_executable: executable,
    child: child_preparation,
    startup_outgoing,
    startup_outgoing_length,
    source_outgoing,
    attestation_outgoing,
    attestation_outgoing_length,
    startup_incoming,
    expected_ready,
    expected_ready_length,
    supervisor,
    failed,
  }
}

///|
fn preflight_prepare_approved(
  executable : @worker_executable_file.WorkerExecutableAdmission,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  limits : WorkerProcessLimits,
  parent_approval : @worker_wire.WorkerParentApprovalSubjects?,
) -> PreparedWorkerProcessStartup raise WorkerProcessError {
  preflight_prepare_with_authority(
    executable, startup, source, limits, parent_approval,
  )
}
