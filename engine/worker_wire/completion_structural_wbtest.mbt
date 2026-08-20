///|
fn copy_completion_fixture(source : FixedArray[Byte]) -> FixedArray[Byte] {
  FixedArray::makei(source.length(), index => source[index])
}

///|
fn assert_prior_completion_is_live(
  frame : ValidatedCompletionFrame,
) -> Unit raise {
  assert_eq(frame.sequence_value(), 1UL)
  assert_eq(frame.model_generation_value(), 2UL)
  assert_eq(frame.entry_for_slot(0).kind(), PrefillCompletion)
  assert_eq(frame.entry_for_slot(1).failure(), ResourceUnavailable)
}

///|
fn expect_completion_rule(
  owner : CompletionFrameBuffer,
  bytes : FixedArray[Byte],
  rule : WorkerWireFrameRule,
  prior : ValidatedCompletionFrame,
) -> Unit raise {
  try owner.load(bytes, bytes.length()) catch {
    InvalidFrame(actual) => assert_eq(actual, rule)
    _ => fail("unexpected structural completion rejection")
  } noraise {
    _ => fail("structurally hostile completion was admitted")
  }
  assert_prior_completion_is_live(prior)
}

///|
test "completion header rejects preserve prior authenticated authority" {
  let owner = completion_test_owner()
  let valid = valid_completion_frame()
  let prior = owner.load(valid, valid.length())

  let version = copy_completion_fixture(valid)
  write_u32(version, 4, 2)
  refresh_completion_checksum(version)
  expect_completion_rule(owner, version, Version, prior)

  let kind = copy_completion_fixture(valid)
  write_u32(kind, 8, 1)
  refresh_completion_checksum(kind)
  expect_completion_rule(owner, kind, Kind, prior)

  let declared_length = copy_completion_fixture(valid)
  write_i32(declared_length, 12, valid.length() - 1)
  refresh_completion_checksum(declared_length)
  expect_completion_rule(owner, declared_length, Length, prior)

  let reserved = copy_completion_fixture(valid)
  write_u32(reserved, 36, 1)
  refresh_completion_checksum(reserved)
  expect_completion_rule(owner, reserved, Reserved, prior)

  let checksum = copy_completion_fixture(valid)
  write_u32(checksum, completion_checksum_offset, 0)
  expect_completion_rule(owner, checksum, Checksum, prior)

  let sequence = copy_completion_fixture(valid)
  write_u64(sequence, 16, 0UL)
  refresh_completion_checksum(sequence)
  expect_completion_rule(owner, sequence, RequestIdentity, prior)

  let generation = copy_completion_fixture(valid)
  write_u64(generation, 24, 0UL)
  refresh_completion_checksum(generation)
  expect_completion_rule(owner, generation, RequestIdentity, prior)

  let count = copy_completion_fixture(valid)
  write_i32(count, 32, 0)
  refresh_completion_checksum(count)
  expect_completion_rule(owner, count, Counts, prior)
}

///|
test "completion entry rejects preserve prior authenticated authority" {
  let owner = completion_test_owner()
  let valid = valid_completion_frame()
  let prior = owner.load(valid, valid.length())
  let first = completion_entry_offset(0)

  let slot = copy_completion_fixture(valid)
  write_i32(slot, first, -1)
  refresh_completion_checksum(slot)
  expect_completion_rule(owner, slot, CompletionSlot, prior)

  let request = copy_completion_fixture(valid)
  write_u64(request, first + 8, 0UL)
  refresh_completion_checksum(request)
  expect_completion_rule(owner, request, RequestIdentity, prior)

  let reserved = copy_completion_fixture(valid)
  write_u32(reserved, first + 36, 1)
  refresh_completion_checksum(reserved)
  expect_completion_rule(owner, reserved, Reserved, prior)

  let processed = copy_completion_fixture(valid)
  write_i32(processed, first + 24, 0)
  refresh_completion_checksum(processed)
  expect_completion_rule(owner, processed, CompletionEntry, prior)

  let token = copy_completion_fixture(valid)
  write_u32(token, first + 4, 1)
  write_i32(token, first + 28, 33)
  refresh_completion_checksum(token)
  expect_completion_rule(owner, token, CompletionEntry, prior)
}

///|
test "completion source truncation preserves prior authenticated authority" {
  let owner = completion_test_owner()
  let valid = valid_completion_frame()
  let prior = owner.load(valid, valid.length())
  try owner.load(valid, completion_header_bytes - 1) catch {
    SourceTooSmall => ()
    _ => fail("unexpected truncated completion rejection")
  } noraise {
    _ => fail("truncated completion was admitted")
  }
  assert_prior_completion_is_live(prior)
}
