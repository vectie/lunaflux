///|
test "same-size mutation of copied bytes rejects and closes exactly once" {
  let file = wb_stream_file()
  let changed_array = file.to_array()
  changed_array[changed_array.length() - 1] = (changed_array[changed_array.length() -
  1].to_int() ^
  1).to_byte()
  let changed = Bytes::from_array(changed_array)
  let opens = Ref(0)
  let closes = Ref(0)
  let copies = Ref(0)
  try
    load_streaming_with_resource(
      wb_stream_model(file),
      wb_stream_plan(file),
      wb_stamp(file),
      file.length().to_int64(),
      wb_stream_limits(transfer_chunk_bytes=19),
      64L,
      4096L,
      (destination, destination_offset, position, count) => {
        let source = if opens.val == 0 { file } else { changed }
        wb_fill(source, destination, destination_offset, position, count)
      },
      () => wb_stamp(file),
      _arena_bytes => {
        opens.val = opens.val + 1
        3
      },
      (_resource, _source, _source_offset, _destination, _count) => {
        copies.val = copies.val + 1
      },
      _resource => closes.val = closes.val + 1,
    )
  catch {
    FileChangedDuringLoad => ()
    _ => fail("unexpected mutation error")
  } noraise {
    _ => fail("expected mutation rejection")
  }
  assert_eq(opens.val, 1)
  assert_true(copies.val > 0)
  assert_eq(closes.val, 1)
}

///|
test "source stamp changes reject before open and clean up after open" {
  let file = wb_stream_file()
  let opens = Ref(0)
  let stamp_reads = Ref(0)
  try
    load_streaming_with_resource(
      wb_stream_model(file),
      wb_stream_plan(file),
      wb_stamp(file),
      file.length().to_int64(),
      wb_stream_limits(),
      64L,
      4096L,
      (destination, destination_offset, position, count) => {
        wb_fill(file, destination, destination_offset, position, count)
      },
      () => {
        stamp_reads.val = stamp_reads.val + 1
        wb_stamp(file, modified=2)
      },
      _arena_bytes => {
        opens.val = opens.val + 1
        1
      },
      (_resource, _source, _source_offset, _destination, _count) => (),
      _resource => (),
    )
  catch {
    FileChangedDuringLoad => ()
    _ => fail("unexpected pre-open stamp error")
  } noraise {
    _ => fail("expected pre-open stamp change")
  }
  assert_eq(stamp_reads.val, 1)
  assert_eq(opens.val, 0)

  let closes = Ref(0)
  stamp_reads.val = 0
  try
    load_streaming_with_resource(
      wb_stream_model(file),
      wb_stream_plan(file),
      wb_stamp(file),
      file.length().to_int64(),
      wb_stream_limits(),
      64L,
      4096L,
      (destination, destination_offset, position, count) => {
        wb_fill(file, destination, destination_offset, position, count)
      },
      () => {
        stamp_reads.val = stamp_reads.val + 1
        if stamp_reads.val == 1 {
          wb_stamp(file)
        } else {
          wb_stamp(file, modified=2)
        }
      },
      _arena_bytes => {
        opens.val = opens.val + 1
        2
      },
      (_resource, _source, _source_offset, _destination, _count) => (),
      _resource => closes.val = closes.val + 1,
    )
  catch {
    FileChangedDuringLoad => ()
    _ => fail("unexpected post-open stamp error")
  } noraise {
    _ => fail("expected post-open stamp change")
  }
  assert_eq(stamp_reads.val, 2)
  assert_eq(opens.val, 1)
  assert_eq(closes.val, 1)
}

///|
test "streaming transfer failure closes the destination exactly once" {
  let file = wb_stream_file()
  let copies = Ref(0)
  let closes = Ref(0)
  try
    load_streaming_with_resource(
      wb_stream_model(file),
      wb_stream_plan(file),
      wb_stamp(file),
      file.length().to_int64(),
      wb_stream_limits(),
      64L,
      4096L,
      (destination, destination_offset, position, count) => {
        wb_fill(file, destination, destination_offset, position, count)
      },
      () => wb_stamp(file),
      _arena_bytes => 5,
      (_resource, _source, _source_offset, _destination, _count) => {
        copies.val = copies.val + 1
        if copies.val == 4 {
          raise DriverFailure(CopyToDevice)
        }
      },
      _resource => closes.val = closes.val + 1,
    )
  catch {
    DeviceFailed(DriverFailure(CopyToDevice)) => ()
    _ => fail("unexpected transfer failure")
  } noraise {
    _ => fail("expected transfer failure")
  }
  assert_eq(copies.val, 4)
  assert_eq(closes.val, 1)
}

///|
test "streaming double failure retains retryable cleanup authority" {
  let file = wb_stream_file()
  let copies = Ref(0)
  let closes = Ref(0)
  let outcome = load_streaming_with_resource(
    wb_stream_model(file),
    wb_stream_plan(file),
    wb_stamp(file),
    file.length().to_int64(),
    wb_stream_limits(),
    64L,
    4096L,
    (destination, destination_offset, position, count) => {
      wb_fill(file, destination, destination_offset, position, count)
    },
    () => wb_stamp(file),
    _arena_bytes => 5,
    (_resource, _source, _source_offset, _destination, _count) => {
      copies.val += 1
      if copies.val == 4 {
        raise DriverFailure(CopyToDevice)
      }
    },
    _resource => {
      closes.val += 1
      if closes.val < 3 {
        raise DriverFailure(AllocationClose)
      }
    },
  )
  guard outcome is ResourceCleanupRequired(failed) else {
    fail("double failure discarded cleanup authority")
  }
  assert_eq(failed.cause, DeviceFailed(DriverFailure(CopyToDevice)))
  assert_eq(failed.cleanup_failure, DriverFailure(AllocationClose))
  let open = Ref(true)
  let cleanup_failure = Ref(failed.cleanup_failure)
  let retry_close = (close : (Int) -> Unit raise @device.DeviceError) => {
    if !open.val {
      return
    }
    close(failed.resource) catch {
      error => {
        cleanup_failure.val = error
        raise CleanupFailed(stage=Transfer, error)
      }
    }
    open.val = false
  }
  try
    retry_close(_resource => {
      closes.val += 1
      if closes.val < 3 {
        raise DriverFailure(AllocationClose)
      }
    })
  catch {
    CleanupFailed(stage=Transfer, DriverFailure(AllocationClose)) => ()
    _ => fail("unexpected retry cleanup failure")
  } noraise {
    _ => fail("first retry unexpectedly closed")
  }
  assert_true(open.val)
  retry_close(_resource => closes.val += 1)
  assert_false(open.val)
  retry_close(_resource => closes.val += 1)
  assert_eq(closes.val, 3)
}
