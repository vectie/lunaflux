///|
fn control_test_identity() -> @spec.ModelIdentity {
  let content = try! @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let plan = try! @spec.PlanDigest::from_sha256(
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
  )
  try! @spec.ModelIdentity::new(content, plan)
}

///|
fn control_test_binding(
  group_generation_value? : UInt64 = 1UL,
  rank? : Int = 0,
) -> @rank_group_wire.RankGroupWireBinding {
  try! @rank_group_wire.RankGroupWireBinding::new(
    identity=control_test_identity(),
    model_generation_value=1UL,
    group_generation_value~,
    rank~,
    world_size=2,
    group_digest="1111111111111111111111111111111111111111111111111111111111111111",
    topology_digest="2222222222222222222222222222222222222222222222222222222222222222",
    worker_digest="3333333333333333333333333333333333333333333333333333333333333333",
    device_plan_digest="4444444444444444444444444444444444444444444444444444444444444444",
    execution_plan_digest="9999999999999999999999999999999999999999999999999999999999999999",
    collective_digest="5555555555555555555555555555555555555555555555555555555555555555",
    kv_digest="6666666666666666666666666666666666666666666666666666666666666666",
  )
}

///|
fn control_test_limits() -> @rank_group_wire.RankGroupWireLimits {
  try! @rank_group_wire.RankGroupWireLimits::new(max_payload_bytes=64)
}

///|
fn control_test_owner() -> RankChildControl {
  try! open_rank_child_control(
    control_test_limits(),
    io_timeout_millis=1000,
    shutdown_timeout_millis=1000,
  )
}

///|
fn publish_test_configure(
  owner : RankChildControl,
  binding : @rank_group_wire.RankGroupWireBinding,
) -> RankChildInbound raise {
  let payload = FixedArray::make(3, b'\x2a')
  let length = try! @rank_group_wire.encode_rank_group_wire_payload_into(
    owner.rx_destination,
    owner.wire_limits,
    binding,
    Configure,
    ParentToChild,
    None,
    0UL,
    payload,
    payload.length(),
  )
  let progress = try! owner.publish_received(length)
  assert_true(progress.has_inbound())
  try! owner.inbound()
}

///|
test "rank child locks Configure and publishes only epoch-pinned payload copies" {
  let owner = control_test_owner()
  let binding = control_test_binding()
  let configure = publish_test_configure(owner, binding)
  assert_eq(try! configure.kind(), Configure)
  assert_eq(try! configure.payload_length(), 3)
  assert_eq(try! configure.binding().group_generation_value(), 1UL)
  let copied = FixedArray::make(3, b'\x00')
  try! configure.copy_payload_to(copied, destination_offset=0)
  assert_eq(copied, FixedArray::make(3, b'\x2a'))
  try! owner.begin_ready(configure, binding, previous_sequence_value=0UL)
  assert_true(owner.send_active)
  assert_eq(owner.lifecycle(), ReadyGroup)
  try configure.kind() catch {
    Invalid(Epoch) => ()
    _ => fail("unexpected stale inbound error")
  } noraise {
    _ => fail("consumed Configure lease remained valid")
  }
}

///|
test "rank child rejects substituted local admission without arming TX" {
  let owner = control_test_owner()
  let configure = publish_test_configure(owner, control_test_binding())
  try
    owner.begin_ready(
      configure,
      control_test_binding(group_generation_value=2UL),
      previous_sequence_value=0UL,
    )
  catch {
    Failed(Wire) => ()
    _ => fail("unexpected substituted binding error")
  } noraise {
    _ => fail("substituted local binding was admitted")
  }
  assert_true(owner.is_failed())
  assert_false(owner.send_active)
}

///|
test "rank child keeps completed RX unpublished while TX owns storage" {
  let owner = control_test_owner()
  owner.send_active = true
  owner.receive_held = true
  owner.receive_held_eof = false
  owner.receive_held_length = @rank_group_wire.rank_group_wire_header_bytes()
  let held = try! owner.progress_receive()
  assert_true(held.is_held())
  assert_false(owner.inbound_active)
}

///|
test "rank child local failure is sticky and emits no response" {
  let owner = control_test_owner()
  try owner.fail_local_admission() catch {
    Failed(LocalAdmission) => ()
    _ => fail("unexpected local-admission failure")
  } noraise {
    _ => fail("local admission failure was ignored")
  }
  assert_true(owner.is_failed())
  assert_false(owner.send_active)
  try owner.begin_receive() catch {
    Failed(LocalAdmission) => ()
    _ => fail("sticky local-admission rule changed")
  } noraise {
    _ => fail("faulted child control remained usable")
  }
}
