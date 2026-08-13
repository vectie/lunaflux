///|
fn wb_ready_resource(resource : Int) -> ResourcePreparation[Int] raise {
  let inspection = wb_inspect_file(wb_stream_file())
  ResourceReady({ resource, layout: inspection.layout })
}

///|
test "file close failure preserves an earlier inspection failure" {
  let closed = Ref(false)
  let outcome : Result[Int, DeviceWeightError] = Err(FileTruncated)
  try
    finish_weight_file_body(outcome, () => {
      closed.val = true
      raise FileCloseFailed
    })
  catch {
    FileTruncated => ()
    _ => fail("file close replaced the primary inspection failure")
  } noraise {
    _ => fail("failed inspection unexpectedly succeeded")
  }
  assert_true(closed.val)
}

///|
test "successful inspection reports deterministic file close failure" {
  let outcome : Result[Int, DeviceWeightError] = Ok(7)
  try finish_weight_file_body(outcome, () => raise FileCloseFailed) catch {
    FileCloseFailed => ()
    _ => fail("unexpected successful-inspection close error")
  } noraise {
    _ => fail("file close failure was swallowed")
  }
}

///|
test "load close failure closes a ready allocation before reporting" {
  let allocation_closes = Ref(0)
  try
    finish_weight_file_load(
      Ok(wb_ready_resource(7)),
      () => raise FileCloseFailed,
      _resource => allocation_closes.val += 1,
    )
  catch {
    FileCloseFailed => ()
    _ => fail("unexpected ready-allocation close error")
  } noraise {
    _ => fail("file close failure published a ready allocation")
  }
  assert_eq(allocation_closes.val, 1)
}

///|
test "dual file and allocation close failure retains cleanup authority" {
  let allocation_closes = Ref(0)
  let result = finish_weight_file_load(
    Ok(wb_ready_resource(11)),
    () => raise FileCloseFailed,
    _resource => {
      allocation_closes.val += 1
      raise DriverFailure(AllocationClose)
    },
  )
  guard result is ResourceCleanupRequired(failed) else {
    fail("dual close failure did not retain allocation authority")
  }
  assert_eq(failed.resource, 11)
  assert_eq(failed.cause, FileCloseFailed)
  assert_eq(failed.stage, SourceClose)
  assert_eq(failed.cleanup_failure, DriverFailure(AllocationClose))
  assert_eq(allocation_closes.val, 1)
}

///|
test "file close failure preserves existing cleanup authority" {
  let retained : ResourcePreparation[Int] = ResourceCleanupRequired({
    resource: 13,
    cause: ContentDigestMismatch,
    stage: Transfer,
    cleanup_failure: Busy(AllocationClose),
  })
  let allocation_closes = Ref(0)
  let result = finish_weight_file_load(
    Ok(retained),
    () => raise FileCloseFailed,
    _resource => allocation_closes.val += 1,
  )
  guard result is ResourceCleanupRequired(failed) else {
    fail("existing cleanup authority was discarded")
  }
  assert_eq(failed.resource, 13)
  assert_eq(failed.cause, ContentDigestMismatch)
  assert_eq(failed.stage, Transfer)
  assert_eq(failed.cleanup_failure, Busy(AllocationClose))
  assert_eq(allocation_closes.val, 0)
}
