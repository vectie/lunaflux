///|
fn receive_inbound(
  owner : @rank_child_control.RankChildControl,
) -> @rank_child_control.RankChildInbound {
  try! owner.begin_receive()
  for ;; {
    let progress = try! owner.progress_receive()
    if progress.has_inbound() {
      return try! owner.inbound()
    }
    if progress.is_clean_eof() {
      alloc_probe_abort()
    }
  }
}

///|
fn finish_send(owner : @rank_child_control.RankChildControl) -> Unit {
  for ;; {
    let progress = try! owner.progress_send()
    if progress.is_complete() {
      return
    }
  }
}

///|
fn encode_configure(
  wire : FixedArray[Byte],
  limits : @rank_group_wire.RankGroupWireLimits,
  binding : @rank_group_wire.RankGroupWireBinding,
  payload : FixedArray[Byte],
) -> Int {
  try! @rank_group_wire.encode_rank_group_wire_payload_into(
    wire,
    limits,
    binding,
    Configure,
    ParentToChild,
    None,
    0UL,
    payload,
    payload.length(),
  )
}

///|
fn encode_submit(
  wire : FixedArray[Byte],
  limits : @rank_group_wire.RankGroupWireLimits,
  binding : @rank_group_wire.RankGroupWireBinding,
  payload : FixedArray[Byte],
  sequence : UInt64,
) -> Int {
  let slot = if sequence % 2UL == 0UL { @rank_group_wire.B } else { A }
  try! @rank_group_wire.encode_rank_group_wire_payload_into(
    wire,
    limits,
    binding,
    Submit,
    ParentToChild,
    Some(slot),
    sequence,
    payload,
    payload.length(),
  )
}

///|
fn encode_poll(
  wire : FixedArray[Byte],
  limits : @rank_group_wire.RankGroupWireLimits,
  binding : @rank_group_wire.RankGroupWireBinding,
  sequence : UInt64,
) -> Int {
  let slot = if sequence % 2UL == 0UL { @rank_group_wire.B } else { A }
  try! @rank_group_wire.encode_rank_group_wire_control_into(
    wire,
    limits,
    binding,
    PollComplete,
    ParentToChild,
    Some(slot),
    sequence,
  )
}

///|
fn encode_parent_control(
  wire : FixedArray[Byte],
  limits : @rank_group_wire.RankGroupWireLimits,
  binding : @rank_group_wire.RankGroupWireBinding,
  kind : @rank_group_wire.RankGroupWireKind,
) -> Int {
  try! @rank_group_wire.encode_rank_group_wire_control_into(
    wire,
    limits,
    binding,
    kind,
    ParentToChild,
    None,
    0UL,
  )
}

///|
fn send_parent_frame(wire : FixedArray[Byte], length : Int) -> Unit {
  if fixture_send(wire, length) != 0 {
    alloc_probe_abort()
  }
}

///|
fn receive_child_frame(
  wire : FixedArray[Byte],
  limits : @rank_group_wire.RankGroupWireLimits,
  binding : @rank_group_wire.RankGroupWireBinding,
  expected_kind : @rank_group_wire.RankGroupWireKind,
  expected_slot : @rank_group_wire.RankGroupWireSlot?,
  expected_sequence : UInt64,
  expected_payload : FixedArray[Byte],
  expected_payload_length : Int,
) -> Unit {
  let length = fixture_receive(wire)
  if length <= 0 {
    alloc_probe_abort()
  }
  let header = try! @rank_group_wire.validate_rank_group_wire_binding(
    wire, length, limits, binding,
  )
  if header.kind() != expected_kind ||
    header.direction() != ChildToParent ||
    header.slot() != expected_slot ||
    header.sequence_value() != expected_sequence ||
    header.payload_length() != expected_payload_length {
    alloc_probe_abort()
  }
  let payload_offset = @rank_group_wire.rank_group_wire_header_bytes()
  for index in 0..<expected_payload_length {
    if wire[payload_offset + index] != expected_payload[index] {
      alloc_probe_abort()
    }
  }
}

///|
fn publish_graph_sidecar(
  owner : @rank_child_control.RankChildControl,
  child_wire : FixedArray[Byte],
  sequence : UInt64,
) -> Unit {
  let report = try! @worker_wire.WorkerGraphTelemetry::without_shape(
    sequence~,
    path=WorkerGraphOrderedEager,
    graph_hits=0UL,
    graph_misses=sequence,
  )
  try! owner.begin_graph_telemetry_sidecar(report)
  finish_send(owner)
  let length = fixture_receive(child_wire)
  if length != @worker_wire.worker_graph_telemetry_frame_capacity() {
    alloc_probe_abort()
  }
  let decoded = try! @worker_wire.decode_worker_graph_telemetry(
    child_wire, length,
  )
  if decoded.sequence_value() != sequence ||
    decoded.selected_path() != WorkerGraphOrderedEager ||
    decoded.has_selected_shape_class() ||
    decoded.graph_hits() != 0UL ||
    decoded.graph_misses() != sequence {
    alloc_probe_abort()
  }
}

///|
fn send_submit(
  owner : @rank_child_control.RankChildControl,
  limits : @rank_group_wire.RankGroupWireLimits,
  binding : @rank_group_wire.RankGroupWireBinding,
  parent_wire : FixedArray[Byte],
  plan_payload : FixedArray[Byte],
  sequence : UInt64,
) -> Unit {
  send_parent_frame(
    parent_wire,
    encode_submit(parent_wire, limits, binding, plan_payload, sequence),
  )
  try! owner.accept_submit(receive_inbound(owner))
}

///|
fn finish_plan(
  owner : @rank_child_control.RankChildControl,
  limits : @rank_group_wire.RankGroupWireLimits,
  binding : @rank_group_wire.RankGroupWireBinding,
  parent_wire : FixedArray[Byte],
  child_wire : FixedArray[Byte],
  completion : FixedArray[Byte],
  sequence : UInt64,
) -> Unit {
  send_parent_frame(
    parent_wire,
    encode_poll(parent_wire, limits, binding, sequence),
  )
  let poll = receive_inbound(owner)
  try! owner.begin_leader_complete(
    poll,
    completion,
    payload_length=completion.length(),
  )
  finish_send(owner)
  let slot = if sequence % 2UL == 0UL { @rank_group_wire.B } else { A }
  receive_child_frame(
    child_wire,
    limits,
    binding,
    Complete,
    Some(slot),
    sequence,
    completion,
    completion.length(),
  )
  publish_graph_sidecar(owner, child_wire, sequence)
}

///|
fn finish_plan_and_prefetch(
  owner : @rank_child_control.RankChildControl,
  limits : @rank_group_wire.RankGroupWireLimits,
  binding : @rank_group_wire.RankGroupWireBinding,
  parent_wire : FixedArray[Byte],
  child_wire : FixedArray[Byte],
  plan_payload : FixedArray[Byte],
  completion : FixedArray[Byte],
  sequence : UInt64,
) -> Unit {
  send_parent_frame(
    parent_wire,
    encode_poll(parent_wire, limits, binding, sequence),
  )
  let poll = receive_inbound(owner)
  try! owner.begin_leader_complete(
    poll,
    completion,
    payload_length=completion.length(),
  )
  let next_sequence = sequence + 1UL
  send_parent_frame(
    parent_wire,
    encode_submit(parent_wire, limits, binding, plan_payload, next_sequence),
  )
  try! owner.begin_receive()
  for ;; {
    let progress = try! owner.progress_receive()
    if progress.is_held() {
      break
    }
    if progress.has_inbound() || progress.is_clean_eof() {
      alloc_probe_abort()
    }
  }
  finish_send(owner)
  let slot = if sequence % 2UL == 0UL { @rank_group_wire.B } else { A }
  receive_child_frame(
    child_wire,
    limits,
    binding,
    Complete,
    Some(slot),
    sequence,
    completion,
    completion.length(),
  )
  publish_graph_sidecar(owner, child_wire, sequence)
  let published = try! owner.progress_receive()
  if !published.has_inbound() {
    alloc_probe_abort()
  }
  try! owner.accept_submit(try! owner.inbound())
}

///|
fn finish_terminal_exchange(
  owner : @rank_child_control.RankChildControl,
  limits : @rank_group_wire.RankGroupWireLimits,
  binding : @rank_group_wire.RankGroupWireBinding,
  parent_wire : FixedArray[Byte],
  child_wire : FixedArray[Byte],
  empty_payload : FixedArray[Byte],
  command : @rank_group_wire.RankGroupWireKind,
  response : @rank_group_wire.RankGroupWireKind,
) -> Unit {
  send_parent_frame(
    parent_wire,
    encode_parent_control(parent_wire, limits, binding, command),
  )
  let inbound = receive_inbound(owner)
  if command == Drain {
    try! owner.begin_drained(inbound)
  } else if command == Close {
    try! owner.begin_closed(inbound)
  } else {
    alloc_probe_abort()
  }
  finish_send(owner)
  receive_child_frame(
    child_wire,
    limits,
    binding,
    response,
    None,
    0UL,
    empty_payload,
    0,
  )
}

///|
fn main {
  verify_positive_control()
  let limits = try! @rank_group_wire.RankGroupWireLimits::new(
    max_payload_bytes=64,
  )
  let binding = fixture_binding()
  let parent_wire = FixedArray::make(limits.max_frame_bytes(), b'\x00')
  let child_wire = FixedArray::make(limits.max_frame_bytes(), b'\x00')
  let configure_payload = FixedArray::make(8, b'\x31')
  let plan_payload = FixedArray::make(16, b'\x41')
  let completion = FixedArray::make(8, b'\x51')
  if fixture_begin() != 0 {
    alloc_probe_abort()
  }
  let owner = try! @rank_child_control.open_rank_child_control(
    limits,
    io_timeout_millis=1000,
    shutdown_timeout_millis=1000,
  )
  send_parent_frame(
    parent_wire,
    encode_configure(parent_wire, limits, binding, configure_payload),
  )
  let configure = receive_inbound(owner)
  try! owner.begin_ready(configure, binding, previous_sequence_value=0UL)
  finish_send(owner)
  let empty_payload = FixedArray::make(0, b'\x00')
  receive_child_frame(
    child_wire,
    limits,
    binding,
    Ready,
    None,
    0UL,
    empty_payload,
    0,
  )
  send_submit(owner, limits, binding, parent_wire, plan_payload, 1UL)
  finish_plan_and_prefetch(
    owner, limits, binding, parent_wire, child_wire, plan_payload, completion, 1UL,
  )

  fixture_reset_evidence()
  alloc_probe_begin()
  for sequence in 2UL..<130UL {
    finish_plan_and_prefetch(
      owner, limits, binding, parent_wire, child_wire, plan_payload, completion,
      sequence,
    )
  }
  let observed = alloc_probe_end()
  finish_plan(
    owner, limits, binding, parent_wire, child_wire, completion, 130UL,
  )
  finish_terminal_exchange(
    owner,
    limits,
    binding,
    parent_wire,
    child_wire,
    empty_payload,
    Drain,
    Drained,
  )
  finish_terminal_exchange(
    owner,
    limits,
    binding,
    parent_wire,
    child_wire,
    empty_payload,
    Close,
    Closed,
  )
  if fixture_shutdown_write() != 0 {
    alloc_probe_abort()
  }
  try! owner.begin_receive()
  for ;; {
    let eof = try! owner.progress_receive()
    if eof.is_clean_eof() {
      break
    }
    if eof.has_inbound() {
      alloc_probe_abort()
    }
  }
  if alloc_probe_check(observed, 1) != 1 ||
    fixture_check_evidence(3U, 128UL * 3UL) != 1 ||
    owner.lifecycle() != ClosedGroup ||
    fixture_finish() != 0 {
    alloc_probe_abort()
  }
}
