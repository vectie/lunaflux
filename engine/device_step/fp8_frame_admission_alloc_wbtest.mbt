///|
fn fp8_admission_test_state() -> Fp8FrameAdmissionStateV3 {
  {
    armed: false,
    descriptor_epoch: 0UL,
    wire_owner: None,
    wire_epoch: 0UL,
    sequence_value: 0UL,
    row_count: 0,
    token_count: 0,
    page_count: 0,
  }
}

///|
fn fp8_admission_test_summary(sequence : UInt64) -> DeviceStepSummary {
  {
    sequence_value: sequence,
    prefill_count: 1,
    decode_count: 0,
    row_count: 1,
    token_count: 1,
    page_count: 1,
  }
}

///|
fn fp8_admission_allocation_positive_control(
  seed : Int,
) -> Fp8FrameAdmissionStateV3 {
  {
    armed: seed < 0,
    descriptor_epoch: seed.to_uint64(),
    wire_owner: None,
    wire_epoch: 0UL,
    sequence_value: 0UL,
    row_count: 0,
    token_count: 0,
    page_count: 0,
  }
}

///|
fn fp8_admission_warmed_cycles(
  executor : PagedGraphExecutor,
  wire_owner : @worker_wire.PlanFrameOwnerIdentity,
  cycle_count : Int,
) -> Unit raise DeviceStepError {
  for cycle in 1..<=cycle_count {
    let epoch = cycle.to_uint64()
    let summary = fp8_admission_test_summary(epoch)
    publish_fp8_frame_token_v3(
      executor,
      wire_owner,
      descriptor_epoch=epoch,
      wire_epoch=epoch,
      summary,
    )
    consume_fp8_frame_token_v3(
      executor,
      staged_epoch=epoch,
      staged_owner=wire_owner,
      wire_epoch=epoch,
      sequence_value=epoch,
    )
  }
}

///|
test "FP8 v3 warmed admission success, poison clear, and rearm allocate zero" {
  let admission = fp8_admission_test_state()
  let frame = encoded_step_frame(executor_test_submitted_plan())
  let wire_owner = frame.owner_identity()
  let executor = dummy_paged_executor(
    GraphReady,
    fp8_frame_admission=Some(admission),
  )
  let positive = fp8_admission_allocation_positive_control(7)
  assert_eq(positive.descriptor_epoch, 7UL)
  fp8_admission_warmed_cycles(executor, wire_owner, 128)
  assert_false(admission.armed)
  assert_eq(admission.descriptor_epoch, 0UL)
  assert_true(admission.wire_owner is None)
  assert_eq(admission.wire_epoch, 0UL)
  assert_eq(admission.sequence_value, 0UL)
  assert_eq(admission.row_count, 0)
  assert_eq(admission.token_count, 0)
  assert_eq(admission.page_count, 0)
}

///|
test "FP8 v3 admission rejects a second publication until explicit retirement" {
  let admission = fp8_admission_test_state()
  let summary = fp8_admission_test_summary(7UL)
  let wire_owner = encoded_step_frame(executor_test_submitted_plan()).owner_identity()
  let executor = dummy_paged_executor(
    GraphReady,
    fp8_frame_admission=Some(admission),
  )
  publish_fp8_frame_token_v3(
    executor,
    wire_owner,
    descriptor_epoch=3UL,
    wire_epoch=5UL,
    summary,
  )
  try
    publish_fp8_frame_token_v3(
      executor,
      wire_owner,
      descriptor_epoch=4UL,
      wire_epoch=6UL,
      summary,
    )
  catch {
    InvalidExecutor(Fp8FrameAdmission) => ()
    _ => fail("unexpected duplicate FP8 admission failure")
  } noraise {
    _ => fail("duplicate FP8 admission was accepted")
  }
  consume_fp8_frame_token_v3(
    executor,
    staged_epoch=3UL,
    staged_owner=wire_owner,
    wire_epoch=5UL,
    sequence_value=7UL,
  )
  publish_fp8_frame_token_v3(
    executor,
    wire_owner,
    descriptor_epoch=4UL,
    wire_epoch=6UL,
    summary,
  )
  assert_true(admission.armed)
  assert_eq(admission.descriptor_epoch, 4UL)
  assert_eq(admission.wire_epoch, 6UL)
  clear_fp8_frame_admission_v3_state(admission)
}

///|
test "FP8 v3 consume rejects foreign owner and retains exact retirement authority" {
  let admission = fp8_admission_test_state()
  let owner = encoded_step_frame(executor_test_submitted_plan()).owner_identity()
  let foreign = encoded_step_frame(executor_test_submitted_plan()).owner_identity()
  let executor = dummy_paged_executor(
    GraphReady,
    fp8_frame_admission=Some(admission),
  )
  publish_fp8_frame_token_v3(
    executor,
    owner,
    descriptor_epoch=9UL,
    wire_epoch=11UL,
    fp8_admission_test_summary(13UL),
  )
  try
    consume_fp8_frame_token_v3(
      executor,
      staged_epoch=9UL,
      staged_owner=foreign,
      wire_epoch=11UL,
      sequence_value=13UL,
    )
  catch {
    InvalidExecutor(Fp8FrameAdmission) => ()
    _ => fail("unexpected foreign FP8 consume failure")
  } noraise {
    _ => fail("foreign FP8 frame owner consumed admission")
  }
  assert_true(admission.armed)
  consume_fp8_frame_token_v3(
    executor,
    staged_epoch=9UL,
    staged_owner=owner,
    wire_epoch=11UL,
    sequence_value=13UL,
  )
  assert_false(admission.armed)
}

///|
test "FP8 v3 poison clears admission and recovery requires a fresh executor" {
  let poisoned_admission = fp8_admission_test_state()
  let wire_owner = encoded_step_frame(executor_test_submitted_plan()).owner_identity()
  let poisoned = dummy_paged_executor(
    GraphStaged,
    fp8_frame_admission=Some(poisoned_admission),
  )
  publish_fp8_frame_token_v3(
    poisoned,
    wire_owner,
    descriptor_epoch=3UL,
    wire_epoch=5UL,
    fp8_admission_test_summary(7UL),
  )
  poison_fp8_graph_lifecycle(poisoned)
  assert_eq(poisoned.lifecycle, GraphPoisoned)
  assert_false(poisoned_admission.armed)
  publish_fp8_frame_token_v3(
    poisoned,
    wire_owner,
    descriptor_epoch=4UL,
    wire_epoch=6UL,
    fp8_admission_test_summary(8UL),
  )
  // Scalar recovery never revives the poisoned resource owner.
  assert_eq(poisoned.lifecycle, GraphPoisoned)
  let replacement_admission = fp8_admission_test_state()
  let replacement = dummy_paged_executor(
    GraphReady,
    fp8_frame_admission=Some(replacement_admission),
  )
  publish_fp8_frame_token_v3(
    replacement,
    wire_owner,
    descriptor_epoch=1UL,
    wire_epoch=1UL,
    fp8_admission_test_summary(1UL),
  )
  assert_eq(replacement.lifecycle, GraphReady)
  assert_true(replacement_admission.armed)
}
