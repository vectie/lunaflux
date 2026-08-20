///|
fn foreign_supervisor_limits() -> @worker_protocol.WorkerProtocolLimits raise {
  @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=3,
    max_decode_rows=3,
    max_plan_rows=3,
    max_plan_tokens=8,
    max_plan_pages=8,
    max_capabilities=8,
    max_completion_slots=3,
    max_sequence_tokens=16,
    max_token_id=31,
  )
}

///|
fn foreign_supervisor_plan(
  completion_slots : Array[Int],
) -> @worker_protocol.SubmittedSchedulePlan raise {
  let limits = foreign_supervisor_limits()
  let owner = @worker_protocol.SchedulePlanBuffer::new(limits)
  for index, completion_slot in completion_slots {
    owner.append_prefill(
      request_id=@inference.RequestId::new((index + 1).to_uint64()),
      request_generation=@inference.RequestGeneration::initial(),
      sequence_token_start=0,
      tokens=[1],
      pages=[supervisor_page()],
      capabilities=[@plan.KernelCapabilityId::embedding_lookup()],
      completion_slot=@worker_protocol.CompletionSlot::new(
        completion_slot, limits,
      ),
    )
  }
  let generation = @worker_protocol.ModelPlanGeneration::from_wire(1UL)
  owner.submit(
    sequence=@worker_protocol.PlanSequence::from_wire(1UL),
    model_plan_generation=generation,
    token_budget=completion_slots.length(),
    None,
    generation,
  )
}

///|
fn inject_exchange_completion(
  owner : RootBoundWorkerProcessSupervisor,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> Unit raise {
  let length = encoded_completion(plan, owner.process.rx)
  owner.process.accept_exchange_completion_bytes(owner.exchange_side, length)
}

///|
priv enum RootBoundSnapshotAttack {
  Sequence
  ModelGeneration
  CompletionSlot
  RequestId
  RequestGeneration
  CompletionKind
  ProcessedCount
}

///|
fn hostile_write_u32(
  bytes : FixedArray[Byte],
  offset : Int,
  value : UInt,
) -> Unit {
  bytes.unsafe_write_uint32_le(offset, value)
}

///|
fn hostile_write_u64(
  bytes : FixedArray[Byte],
  offset : Int,
  value : UInt64,
) -> Unit {
  bytes.unsafe_write_uint64_le(offset, value)
}

///|
fn hostile_refresh_checksum(bytes : FixedArray[Byte], length : Int) -> Unit {
  let mut checksum = 2166136261U
  for index in 0..<length {
    if index < 40 || index >= 44 {
      checksum = (checksum ^ bytes[index].to_uint()) * 16777619U
    }
  }
  hostile_write_u32(bytes, 40, checksum)
}

///|
fn apply_root_bound_snapshot_attack(
  bytes : FixedArray[Byte],
  length : Int,
  attack : RootBoundSnapshotAttack,
) -> Unit {
  let entry = 48
  match attack {
    Sequence => hostile_write_u64(bytes, 16, 2UL)
    ModelGeneration => hostile_write_u64(bytes, 24, 2UL)
    CompletionSlot => hostile_write_u32(bytes, entry, 0U)
    RequestId => hostile_write_u64(bytes, entry + 8, 9UL)
    RequestGeneration => hostile_write_u64(bytes, entry + 16, 1UL)
    CompletionKind => {
      hostile_write_u32(bytes, entry + 4, 2U)
      hostile_write_u32(bytes, entry + 24, 0U)
    }
    ProcessedCount => hostile_write_u32(bytes, entry + 24, 1U)
  }
  hostile_refresh_checksum(bytes, length)
}

///|
fn expect_root_bound_snapshot_reject(
  attack : RootBoundSnapshotAttack,
  expected_rule : @worker_wire.WorkerWireFrameRule,
) -> Unit raise {
  let owner = test_root_bound_supervisor()
  let plan = supervisor_plan(1UL, completion_slot=1)
  assert_eq(owner.begin_exchange(plan), 1UL)
  let length = encoded_completion(plan, owner.process.rx)
  apply_root_bound_snapshot_attack(owner.process.rx, length, attack)
  try
    owner.process.accept_exchange_completion_bytes(owner.exchange_side, length)
  catch {
    Wire(InvalidFrame(rule)) => assert_eq(rule, expected_rule)
    _ => fail("unexpected root-bound snapshot rejection")
  } noraise {
    _ => fail("hostile completion snapshot was accepted")
  }
  assert_eq(owner.process.outstanding_count(), 1)
  assert_eq(owner.submission_for_sequence(1UL).sequence_value(), 1UL)
  owner.close()
}

///|
test "exchange accepts an exact completion at a noncontiguous slot" {
  let owner = test_root_bound_supervisor()
  let plan = supervisor_plan(1UL, completion_slot=1)
  assert_eq(owner.begin_exchange(plan), 1UL)
  inject_exchange_completion(owner, plan)
  let received = owner.received_for_sequence(1UL)
  assert_eq(received.frame(owner).entry_for_slot(1).slot(), 1)
  owner.clear_exchange()
  owner.retire_received(received)
  owner.close()
}

///|
test "exchange snapshot rejects every exact-plan authority mismatch" {
  expect_root_bound_snapshot_reject(Sequence, CompletionEntry)
  expect_root_bound_snapshot_reject(ModelGeneration, CompletionEntry)
  expect_root_bound_snapshot_reject(CompletionSlot, CompletionSlot)
  expect_root_bound_snapshot_reject(RequestId, CompletionEntry)
  expect_root_bound_snapshot_reject(RequestGeneration, CompletionEntry)
  expect_root_bound_snapshot_reject(CompletionKind, CompletionEntry)
  expect_root_bound_snapshot_reject(ProcessedCount, CompletionEntry)
}

///|
test "exchange snapshot rejects a stale slot from the preceding plan" {
  let owner = test_root_bound_supervisor()
  let first = supervisor_plan(1UL)
  ignore(owner.begin_exchange(first))
  while !owner.process.require_child().progress_write_frame() {

  }
  inject_exchange_completion(owner, first)
  let received = owner.received_for_sequence(1UL)
  owner.clear_exchange()
  owner.retire_received(received)

  let second = supervisor_plan(2UL, completion_slot=1)
  ignore(owner.begin_exchange(second))
  try inject_exchange_completion(owner, supervisor_plan(2UL)) catch {
    Wire(InvalidFrame(_)) => ()
    _ => fail("unexpected stale-slot validation failure")
  } noraise {
    _ => fail("stale completion slot was accepted for the next plan")
  }
  owner.close()
}

///|
test "foreign wider plans typed-fail only after reserving the obligation" {
  let too_many_rows = test_root_bound_supervisor()
  try too_many_rows.begin_exchange(foreign_supervisor_plan([0, 1, 2])) catch {
    RootBoundFailure(Process(Invalid(Submission))) => ()
    _ => fail("unexpected foreign row-capacity failure")
  } noraise {
    _ => fail("foreign plan exceeded the snapshot row capacity")
  }
  assert_eq(too_many_rows.lifecycle(), RootBoundRecoveryRequired)
  assert_eq(too_many_rows.submission_for_sequence(1UL).sequence_value(), 1UL)
  too_many_rows.close()

  let out_of_range_slot = test_root_bound_supervisor()
  try out_of_range_slot.begin_exchange(foreign_supervisor_plan([2])) catch {
    RootBoundFailure(Process(Invalid(Submission))) => ()
    _ => fail("unexpected foreign slot-capacity failure")
  } noraise {
    _ => fail("foreign completion slot exceeded the snapshot capacity")
  }
  assert_eq(out_of_range_slot.lifecycle(), RootBoundRecoveryRequired)
  assert_eq(
    out_of_range_slot.submission_for_sequence(1UL).sequence_value(),
    1UL,
  )
  out_of_range_slot.close()
}
