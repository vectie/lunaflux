///|
pub fn RankGroupProcessLimits::new(
  max_payload_bytes~ : Int,
  startup_io_timeout_millis~ : Int,
  io_timeout_millis~ : Int,
  shutdown_timeout_millis~ : Int,
  collective_exchange_timeout_millis~ : Int,
) -> RankGroupProcessLimits raise RankGroupProcessError {
  if max_payload_bytes <= 0 ||
    startup_io_timeout_millis <= 0 ||
    startup_io_timeout_millis > 600000 ||
    io_timeout_millis <= 0 ||
    io_timeout_millis > 600000 ||
    shutdown_timeout_millis <= 0 ||
    shutdown_timeout_millis > 600000 ||
    collective_exchange_timeout_millis <= 0 ||
    collective_exchange_timeout_millis > 3600000 {
    raise Invalid(Limit)
  }
  ignore(
    @rank_group_wire.RankGroupWireLimits::new(max_payload_bytes~) catch {
      _ => raise Invalid(Limit)
    },
  )
  {
    max_payload_bytes,
    startup_io_timeout_millis,
    io_timeout_millis,
    shutdown_timeout_millis,
    collective_exchange_timeout_millis,
  }
}

///|
pub fn RankGroupProcessLimits::max_payload_bytes(
  self : RankGroupProcessLimits,
) -> Int {
  self.max_payload_bytes
}

///|
pub fn RankGroupProcessLimits::collective_exchange_timeout_millis(
  self : RankGroupProcessLimits,
) -> Int {
  self.collective_exchange_timeout_millis
}

///|
pub fn RankGroupProcessRankStartup::new(
  configure : @tensor_parallel_rank_configure.TensorParallelRankConfigureContract,
) -> RankGroupProcessRankStartup {
  { configure, }
}

///|
pub fn RankGroupProcessRankStartup::binding(
  self : RankGroupProcessRankStartup,
) -> @rank_group_wire.RankGroupWireBinding {
  self.configure.binding()
}

///|
pub fn RankGroupProcessRankStartup::configure(
  self : RankGroupProcessRankStartup,
) -> @tensor_parallel_rank_configure.TensorParallelRankConfigureContract {
  self.configure
}

///|
fn copy_rank_startups(
  source : ArrayView[RankGroupProcessRankStartup],
) -> FixedArray[RankGroupProcessRankStartup] raise RankGroupProcessError {
  if source.length() < 2 || source.length() > 16 {
    raise Invalid(Binding)
  }
  let copied = FixedArray::make(source.length(), source[0])
  for rank in 0..<source.length() {
    let startup = source[rank]
    if startup.binding().rank() != rank ||
      startup.binding().world_size() != source.length() {
      raise Invalid(Binding)
    }
    copied[rank] = startup
  }
  copied
}

///|
fn copy_plan_identities(
  source : ArrayView[@worker_protocol.PlanBufferIdentity],
) -> FixedArray[@worker_protocol.PlanBufferIdentity] raise RankGroupProcessError {
  if source.length() != 2 || source[0].same_owner(source[1]) {
    raise Invalid(Binding)
  }
  let copied = FixedArray::make(source.length(), source[0])
  for index in 0..<source.length() {
    copied[index] = source[index]
  }
  copied
}

///|
fn prepared_children(
  executable : @worker_executable_file.WorkerExecutableAdmission,
  process_limits : @process.ProcessLimits,
  world_size : Int,
) -> FixedArray[RankGroupChildOwner] raise RankGroupProcessError {
  let first = executable.prepare_child(process_limits) catch {
    _ => raise Invalid(Limit)
  }
  let children = FixedArray::make(world_size, PreparedChild(first))
  for rank in 1..<world_size {
    let prepared = executable.prepare_child(process_limits) catch {
      _ => raise Invalid(Limit)
    }
    children[rank] = PreparedChild(prepared)
  }
  children
}

///|
/// Preallocates all rank owners, protocol owners, codecs, and shared transport
/// storage. It borrows no root and spawns no child.
fn prepare_rank_group_process_internal(
  executable : @worker_executable_file.WorkerExecutableAdmission,
  rank_startups : ArrayView[RankGroupProcessRankStartup],
  plan_buffer_identities : ArrayView[@worker_protocol.PlanBufferIdentity],
  previous_plan_sequence_value~ : UInt64,
  limits : RankGroupProcessLimits,
) -> PreparedRankGroupProcess raise RankGroupProcessError {
  let startups = copy_rank_startups(rank_startups)
  let world_size = startups.length()
  let identities = copy_plan_identities(plan_buffer_identities)
  let first_binding = startups[0].binding()
  let worker_startup = startups[0].configure.startup()
  if first_binding.identity() != worker_startup.identity() ||
    first_binding.model_generation_value() !=
    worker_startup.model_generation_value() {
    raise Invalid(Binding)
  }
  let wire_limits = @rank_group_wire.RankGroupWireLimits::new(
    max_payload_bytes=limits.max_payload_bytes,
  ) catch {
    _ => raise Invalid(Limit)
  }
  for startup in startups {
    let contract = startup.configure
    let rank_startup = contract.startup()
    if contract.length() > wire_limits.max_payload_bytes() {
      raise Invalid(Limit)
    }
    if rank_startup.identity() != worker_startup.identity() ||
      rank_startup.model_generation_value() !=
      worker_startup.model_generation_value() ||
      rank_startup.worker_limits() != worker_startup.worker_limits() ||
      rank_startup.inference_limits() != worker_startup.inference_limits() ||
      rank_startup.has_predecessor() != (previous_plan_sequence_value != 0UL) ||
      rank_startup.predecessor_value() != previous_plan_sequence_value {
      raise Invalid(Binding)
    }
  }
  let process_limits = @process.ProcessLimits::new(
    max_frame_bytes=wire_limits.max_frame_bytes(),
    io_timeout_millis=limits.io_timeout_millis,
    shutdown_timeout_millis=limits.shutdown_timeout_millis,
  ) catch {
    _ => raise Invalid(Limit)
  }
  let bindings = FixedArray::make(world_size, first_binding)
  for rank in 0..<world_size {
    bindings[rank] = startups[rank].binding()
  }
  let wire_transcript = @rank_group_wire.RankGroupWireTranscript::new(
    bindings,
    previous_sequence_value=previous_plan_sequence_value,
  ) catch {
    _ => raise Invalid(Binding)
  }
  let model_generation = @worker_protocol.ModelPlanGeneration::from_wire(
    worker_startup.model_generation_value(),
  ) catch {
    _ => raise Invalid(Binding)
  }
  let group_generation = @rank_group_protocol.RankGroupGeneration::from_wire(
    first_binding.group_generation_value(),
  ) catch {
    _ => raise Invalid(Binding)
  }
  let protocol = @rank_group_protocol.RankGroupOwner::new(
    identity=worker_startup.identity(),
    model_plan_generation=model_generation,
    generation=group_generation,
    world_size~,
    previous_plan_sequence_value~,
    plan_buffer_identities=identities,
  ) catch {
    _ => raise Invalid(Binding)
  }
  let tx = FixedArray::make(wire_limits.max_frame_bytes(), b'\x00')
  let plan_frame = @worker_wire.PlanFrameBuffer::new_transport_owned(
    worker_startup.worker_limits(),
    worker_startup.inference_limits(),
    tx,
    frame_offset=@rank_group_wire.rank_group_wire_header_bytes(),
  ) catch {
    _ => raise Invalid(Limit)
  }
  let completion_frame = @worker_wire.CompletionFrameBuffer::new(
    worker_startup.worker_limits(),
    worker_startup.inference_limits(),
  ) catch {
    _ => raise Invalid(Limit)
  }
  if plan_frame.capacity() > limits.max_payload_bytes ||
    completion_frame.capacity() > limits.max_payload_bytes {
    raise Invalid(Limit)
  }
  let supervisor : RankGroupProcessSupervisor = {
    limits,
    wire_limits,
    children: prepared_children(executable, process_limits, world_size),
    startups,
    cleanup_started: FixedArray::make(world_size, false),
    plan_identities: identities,
    protocol,
    wire_transcript,
    plan_frame,
    completion_frame,
    tx,
    rx: FixedArray::make(wire_limits.max_frame_bytes(), b'\x00'),
    rx_staging: FixedArray::make(wire_limits.max_frame_bytes(), b'\x00'),
    clock: @monotonic_clock.MonotonicClock::system(),
    activation_consumed: false,
    lifecycle_value: Prepared,
    failure_value: None,
    cleanup_failure_value: None,
    group_failure: None,
    group_failure_taken: false,
    io_rank: -1,
    io_phase: 0,
    next_rank: 0,
    rank_done: FixedArray::make(world_size, false),
    active_slot_code: -1,
    active_sequence_value: 0UL,
    encoded_plan_length: 0,
    active_frame_deadline_millis: 0L,
    leader_received: false,
    graph_telemetry: @worker_wire.WorkerGraphTelemetrySlot::new(),
    last_graph_telemetry: @worker_wire.WorkerGraphTelemetrySlot::new(),
    rank_sidecar_pending: FixedArray::make(world_size, false),
    exchange_epoch: 0UL,
    accepted_epoch: 0UL,
    active_deadline_millis: 0UL,
    cleanup_cursor: 0,
    cleanup_lost: FixedArray::make(world_size, false),
    healthy_terminal: false,
    survivor_terminal: false,
    survivor_phase: 0,
    survivor_rank: -1,
  }
  PreparedRankGroupProcess(supervisor)
}

///|
/// Prepares every rank child from one reusable pinned executable capability.
pub fn prepare_rank_group_process_approved(
  executable : @worker_executable_file.WorkerExecutableAdmission,
  rank_startups : ArrayView[RankGroupProcessRankStartup],
  plan_buffer_identities : ArrayView[@worker_protocol.PlanBufferIdentity],
  previous_plan_sequence_value~ : UInt64,
  limits : RankGroupProcessLimits,
) -> PreparedRankGroupProcess raise RankGroupProcessError {
  prepare_rank_group_process_internal(
    executable,
    rank_startups,
    plan_buffer_identities,
    previous_plan_sequence_value~,
    limits,
  )
}
