///|
extern "C" fn alloc_probe_begin() = "lunaflux_alloc_probe_begin"

///|
extern "C" fn alloc_probe_end() -> UInt64 = "lunaflux_alloc_probe_end"

///|
extern "C" fn alloc_probe_check(observed : UInt64, expect_zero : Int) -> Int = "lunaflux_alloc_probe_check"

///|
extern "C" fn alloc_probe_abort() = "lunaflux_alloc_probe_abort"

///|
extern "C" fn alloc_probe_seed() -> Int = "lunaflux_alloc_probe_seed"

///|
#borrow(value)
extern "C" fn keep_array_alive(value : FixedArray[Int]) = "lunaflux_alloc_probe_keep_alive"

///|
extern "C" fn execution_reset() = "lunaflux_rank_wire_execution_reset"

///|
extern "C" fn execution_mark(bit : Int) = "lunaflux_rank_wire_execution_mark"

///|
extern "C" fn execution_check(
  expected_mask : UInt,
  minimum_count : UInt64,
) -> Int = "lunaflux_rank_wire_execution_check"

///|
fn verify_positive_control() -> Unit {
  alloc_probe_begin()
  let allocated = FixedArray::make(8, alloc_probe_seed())
  keep_array_alive(allocated)
  let observed = alloc_probe_end()
  if alloc_probe_check(observed, 0) != 1 {
    alloc_probe_abort()
  }
}

///|
fn identity() -> @spec.ModelIdentity {
  let content = try! @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let plan = try! @spec.PlanDigest::from_sha256(
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
  )
  try! @spec.ModelIdentity::new(content, plan)
}

///|
fn binding(rank : Int) -> @rank_group_wire.RankGroupWireBinding {
  try! @rank_group_wire.RankGroupWireBinding::new(
    identity=identity(),
    model_generation_value=1UL,
    group_generation_value=1UL,
    rank~,
    world_size=2,
    group_digest="1111111111111111111111111111111111111111111111111111111111111111",
    topology_digest="2222222222222222222222222222222222222222222222222222222222222222",
    worker_digest=if rank == 0 {
      "3333333333333333333333333333333333333333333333333333333333333333"
    } else {
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    },
    device_plan_digest=if rank == 0 {
      "4444444444444444444444444444444444444444444444444444444444444444"
    } else {
      "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    },
    execution_plan_digest="9999999999999999999999999999999999999999999999999999999999999999",
    collective_digest=if rank == 0 {
      "5555555555555555555555555555555555555555555555555555555555555555"
    } else {
      "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    },
    kv_digest=if rank == 0 {
      "6666666666666666666666666666666666666666666666666666666666666666"
    } else {
      "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
    },
  )
}

///|
fn accept_control(
  transcript : @rank_group_wire.RankGroupWireTranscript,
  sender : @rank_group_wire.RankGroupWireFrameBuffer,
  rank_binding : @rank_group_wire.RankGroupWireBinding,
  kind : @rank_group_wire.RankGroupWireKind,
  direction : @rank_group_wire.RankGroupWireDirection,
  slot : @rank_group_wire.RankGroupWireSlot?,
  sequence : UInt64,
) -> Unit {
  let frame = if kind == Configure {
    let payload = FixedArray::make(1, b'\x01')
    try! sender.encode_configure_payload(rank_binding, payload, 1)
  } else {
    try! sender.encode_control(rank_binding, kind, direction, slot, sequence)
  }
  try! transcript.accept(frame)
}

///|
fn start_ready(
  transcript : @rank_group_wire.RankGroupWireTranscript,
  sender : @rank_group_wire.RankGroupWireFrameBuffer,
  bindings : FixedArray[@rank_group_wire.RankGroupWireBinding],
) -> Unit {
  for rank in 0..<2 {
    accept_control(
      transcript,
      sender,
      bindings[rank],
      Configure,
      ParentToChild,
      None,
      0UL,
    )
  }
  for rank in 0..<2 {
    accept_control(
      transcript,
      sender,
      bindings[rank],
      Ready,
      ChildToParent,
      None,
      0UL,
    )
  }
}

///|
fn run_cycle(
  transcript : @rank_group_wire.RankGroupWireTranscript,
  sender : @rank_group_wire.RankGroupWireFrameBuffer,
  receiver : @rank_group_wire.RankGroupWireFrameBuffer,
  bindings : FixedArray[@rank_group_wire.RankGroupWireBinding],
  payload : FixedArray[Byte],
  completion : FixedArray[Byte],
  wire : FixedArray[Byte],
  sequence : UInt64,
) -> Unit {
  let slot = if sequence % 2UL == 0UL { @rank_group_wire.B } else { A }
  let first = try! sender.encode_payload(
    bindings[0],
    Submit,
    ParentToChild,
    slot,
    sequence,
    payload,
    payload.length(),
  )
  execution_mark(1)
  let first_length = try! first.length()
  try! first.copy_to(wire, destination_offset=0)
  let loaded = try! receiver.load(wire, first_length)
  execution_mark(2)
  try! transcript.accept(loaded)
  execution_mark(4)

  let second = try! sender.encode_payload(
    bindings[1],
    Submit,
    ParentToChild,
    slot,
    sequence,
    payload,
    payload.length(),
  )
  execution_mark(1)
  try! transcript.accept(second)
  execution_mark(4)

  let poll_leader = try! sender.encode_control(
    bindings[0],
    PollComplete,
    ParentToChild,
    Some(slot),
    sequence,
  )
  execution_mark(1)
  let poll_leader_length = try! poll_leader.length()
  try! poll_leader.copy_to(wire, destination_offset=0)
  try! transcript.load_and_accept(receiver, wire, poll_leader_length)
  execution_mark(8)

  let leader = try! sender.encode_payload(
    bindings[0],
    Complete,
    ChildToParent,
    slot,
    sequence,
    completion,
    completion.length(),
  )
  execution_mark(1)
  try! transcript.accept(leader)
  execution_mark(4)

  let poll_follower = try! sender.encode_control(
    bindings[1],
    PollComplete,
    ParentToChild,
    Some(slot),
    sequence,
  )
  execution_mark(1)
  try! transcript.accept(poll_follower)
  execution_mark(4)

  let follower = try! sender.encode_control(
    bindings[1],
    Complete,
    ChildToParent,
    Some(slot),
    sequence,
  )
  execution_mark(1)
  let follower_length = try! follower.length()
  try! follower.copy_to(wire, destination_offset=0)
  try! transcript.load_and_accept(receiver, wire, follower_length)
  execution_mark(8)
}

///|
fn main {
  verify_positive_control()
  let limits = try! @rank_group_wire.RankGroupWireLimits::new(
    max_payload_bytes=16,
  )
  let bindings = FixedArray::makei(2, rank => binding(rank))
  let transcript = try! @rank_group_wire.RankGroupWireTranscript::new(
    [bindings[0], bindings[1]],
    previous_sequence_value=0UL,
  )
  let sender = @rank_group_wire.RankGroupWireFrameBuffer::new(limits)
  let receiver = @rank_group_wire.RankGroupWireFrameBuffer::new(limits)
  let payload = FixedArray::make(8, b'\x2a')
  let completion = FixedArray::make(4, b'\x17')
  let wire = FixedArray::make(sender.capacity(), b'\x00')
  start_ready(transcript, sender, bindings)
  run_cycle(
    transcript, sender, receiver, bindings, payload, completion, wire, 1UL,
  )

  execution_reset()
  alloc_probe_begin()
  for sequence in 2UL..<130UL {
    run_cycle(
      transcript, sender, receiver, bindings, payload, completion, wire, sequence,
    )
  }
  let observed = alloc_probe_end()
  if alloc_probe_check(observed, 1) != 1 ||
    execution_check(15U, 128UL * 10UL) != 1 ||
    transcript.previous_sequence_value() != 129UL {
    alloc_probe_abort()
  }
}
