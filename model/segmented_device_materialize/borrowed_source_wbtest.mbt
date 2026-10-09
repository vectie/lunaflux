///|
test "opaque borrowed source validates exact one-shot arena provision" {
  let layout = segmented_test_layout()
  let arenas = [FixedArray::make(8, b'a'), FixedArray::make(8, b'b')]
  let mut calls = 0
  validate_borrowed_source(layout, provide=(arena_ordinal, consumer) => {
    calls += 1
    consumer.accept(arenas[arena_ordinal])
  })
  assert_eq(calls, 3)
  try
    validate_borrowed_source(layout, provide=(_, consumer) => {
      consumer.accept(FixedArray::make(7, b'x'))
    })
  catch {
    HostArenaLengthMismatch(0) => ()
    _ => fail("unexpected borrowed length rejection")
  } noraise {
    _ => fail("wrong borrowed arena length was admitted")
  }
  try validate_borrowed_source(layout, provide=(_, _) => ()) catch {
    HostArenaNotProvided(0) => ()
    _ => fail("unexpected missing provision rejection")
  } noraise {
    _ => fail("provider returned without supplying its arena")
  }
  try
    validate_borrowed_source(layout, provide=(arena_ordinal, consumer) => {
      consumer.accept(arenas[arena_ordinal])
      consumer.accept(arenas[arena_ordinal])
    })
  catch {
    HostArenaAlreadyProvided(0) => ()
    _ => fail("unexpected double provision rejection")
  } noraise {
    _ => fail("one consumer accepted an arena twice")
  }
}

///|
test "borrow consumer is revoked when its provider callback returns" {
  let layout = segmented_test_layout()
  let retained : Array[@segmented_host_borrow.SegmentedHostArenaConsumer] = []
  try
    validate_borrowed_source(layout, provide=(_, consumer) => {
      retained.push(consumer)
    })
  catch {
    HostArenaNotProvided(0) => ()
    _ => fail("unexpected retained-consumer rejection")
  } noraise {
    _ => fail("provider retained a live consumer")
  }
  assert_eq(retained.length(), 1)
  try retained[0].accept(FixedArray::make(8, b'x')) catch {
    BorrowClosed(0) => ()
    _ => fail("unexpected late-consumer rejection")
  } noraise {
    _ => fail("retained consumer remained live after callback")
  }
}

///|
test "borrowed fake core copies without exposing host ownership" {
  let layout = segmented_test_layout()
  let arenas = [
    FixedArray::from_array([b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h']),
    FixedArray::from_array([b'W', b'X', b'Y', b'Z', b'1', b'2', b'3', b'4']),
  ]
  let prepared = prepare_with_borrowed_resource(
    layout,
    layout.source_identity(),
    layout.source_digest(),
    layout.digest(),
    (arena_ordinal, consumer) => consumer.accept(arenas[arena_ordinal]),
    bytes => {
      fake_segmented_allocation(
        bytes,
        fail_validate_at=-1,
        fail_copy_at=-1,
        close_failures=0,
      )
    },
    fake_validate_segmented,
    fake_copy_segmented,
    fake_close_segmented,
  )
  match prepared {
    SegmentedResourceReady(ready) => {
      assert_eq(Bytes::from_array(ready.resource.bytes[0:3]), b"bcd")
      assert_eq(Bytes::from_array(ready.resource.bytes[64:68]), b"WXYZ")
      assert_eq(Bytes::from_array(ready.resource.bytes[128:130]), b"fg")
      assert_eq(Bytes::from_array(arenas[0][:]), b"abcdefgh")
    }
    SegmentedResourceCleanupRequired(_) => fail("borrowed copy needs cleanup")
  }
}

///|
test "borrowed provider failure retains cleanup when close fails" {
  let layout = segmented_test_layout()
  match
    prepare_with_borrowed_resource(
      layout,
      layout.source_identity(),
      layout.source_digest(),
      layout.digest(),
      (arena_ordinal, _) => raise SourceUnavailable(arena_ordinal),
      bytes => {
        fake_segmented_allocation(
          bytes,
          fail_validate_at=-1,
          fail_copy_at=-1,
          close_failures=1,
        )
      },
      fake_validate_segmented,
      fake_copy_segmented,
      fake_close_segmented,
    ) {
    SegmentedResourceCleanupRequired(failed) => {
      assert_eq(failed.cause, HostSourceUnavailable(0))
      assert_eq(failed.stage, Transfer)
      assert_eq(failed.cleanup_failure, Busy(AllocationClose))
      assert_false(failed.resource.closed)
      fake_close_segmented(failed.resource)
      assert_true(failed.resource.closed)
    }
    SegmentedResourceReady(_) => fail("unavailable host source became ready")
  }
}
