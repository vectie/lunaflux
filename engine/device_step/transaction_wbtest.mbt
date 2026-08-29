///|
fn transaction_summary() -> DeviceStepSummary {
  {
    sequence_value: 2UL,
    prefill_count: 1,
    decode_count: 1,
    row_count: 2,
    token_count: 4,
    page_count: 3,
  }
}

///|
fn fake_upload_transaction(
  host : DeviceStepHostBuffers,
  summary : DeviceStepSummary,
  upload : (DeviceStepStage, FixedArray[Byte], Int) -> Unit raise DeviceStepError,
) -> Unit raise DeviceStepError {
  upload(TokenIds, host.token_ids, summary.token_count)
  upload(TokenPositions, host.token_positions, summary.token_count)
  upload(QueryRowOffsets, host.query_row_offsets, summary.row_count + 1)
  upload(SequenceLengths, host.sequence_lengths, summary.row_count)
  upload(PageOffsets, host.page_offsets, summary.row_count + 1)
  upload(PhysicalPageIndices, host.physical_page_indices, summary.page_count)
  publish_counts(host, summary)
  upload(Counts, host.counts, COUNTS_LENGTH)
}

///|
fn fake_publish_or_poison(
  lifecycle : DeviceStepLifecycle,
  set_lifecycle : (DeviceStepLifecycle) -> Unit,
  host : DeviceStepHostBuffers,
  summary : DeviceStepSummary,
  upload : (DeviceStepStage, FixedArray[Byte], Int) -> Unit raise DeviceStepError,
) -> Unit raise DeviceStepError {
  require_lifecycle(lifecycle, Ready)
  let mut upload_complete = false
  defer (if !upload_complete { set_lifecycle(Poisoned) })
  fake_upload_transaction(host, summary, upload)
  upload_complete = true
}

///|
test "upload publishes payloads before one real count header" {
  let host = make_host_buffers(step_device_limits())
  publish_counts(host, transaction_summary())
  let calls = Ref(0)
  let count_value = Ref(-1)
  let count_call = Ref(-1)
  fake_upload_transaction(host, transaction_summary(), (stage, buffer, _cells) => {
    if stage == Counts {
      count_value.val = read_host_i32(buffer, 0)
      count_call.val = calls.val
    }
    calls.val += 1
  })
  assert_eq(calls.val, 7)
  assert_eq(count_call.val, 6)
  assert_eq(count_value.val, 1)
}

///|
test "payload failure poisons owner without a diagnostic count upload" {
  let host = make_host_buffers(step_device_limits())
  let lifecycle : Ref[DeviceStepLifecycle] = Ref(Ready)
  fake_publish_or_poison(
    lifecycle.val,
    value => lifecycle.val = value,
    host,
    transaction_summary(),
    (_stage, _buffer, _cells) => (),
  )
  assert_true(lifecycle.val == Ready)
  assert_eq(read_host_i32(host.counts, 0), 1)
  let device_count_header = Ref(1)
  let calls = Ref(0)
  try
    fake_publish_or_poison(
      lifecycle.val,
      value => lifecycle.val = value,
      host,
      transaction_summary(),
      (stage, buffer, _cells) => {
        if stage == Counts {
          device_count_header.val = read_host_i32(buffer, 0)
        }
        calls.val += 1
        if stage == TokenPositions {
          raise Device(TokenPositions, Busy(CopyToDevice))
        }
      },
    )
  catch {
    Device(TokenPositions, Busy(CopyToDevice)) => ()
    _ => fail("unexpected injected upload failure")
  } noraise {
    _ => fail("expected upload failure")
  }
  assert_true(lifecycle.val == Poisoned)
  assert_eq(device_count_header.val, 1)
  try
    fake_publish_or_poison(
      lifecycle.val,
      value => lifecycle.val = value,
      host,
      transaction_summary(),
      (_stage, _buffer, _cells) => (),
    )
  catch {
    InvalidLifecycle(expected=Ready, actual=Poisoned) => ()
    _ => fail("unexpected poisoned lifecycle failure")
  } noraise {
    _ => fail("expected poisoned owner rejection")
  }
}

///|
test "cleanup failure retains retry authority and success is idempotent" {
  let lifecycle = Ref(Poisoned)
  let calls = Ref(0)
  try
    close_with_lifecycle(lifecycle.val, value => lifecycle.val = value, () => {
      calls.val += 1
      raise Device(Counts, Busy(AllocationClose))
    })
  catch {
    Device(Counts, Busy(AllocationClose)) => ()
    _ => fail("unexpected cleanup failure")
  } noraise {
    _ => fail("expected cleanup failure")
  }
  assert_true(lifecycle.val == ClosingFailed)
  close_with_lifecycle(lifecycle.val, value => lifecycle.val = value, () => {
    calls.val += 1
  })
  close_with_lifecycle(lifecycle.val, value => lifecycle.val = value, () => {
    calls.val += 1
  })
  assert_true(lifecycle.val == Closed)
  assert_eq(calls.val, 2)
}
