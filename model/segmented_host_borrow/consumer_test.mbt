///|
test "consumer admits one exact synchronous arena then revokes" {
  let arena = FixedArray::make(4, b'x')
  let mut calls = 0
  let consumer = SegmentedHostArenaConsumer::new(
    arena_ordinal=2,
    expected_length=4,
    consume=borrowed => {
      calls += 1
      borrowed[0] == b'x'
    },
  )
  consumer.accept(arena)
  assert_true(consumer.was_accepted())
  assert_eq(calls, 1)
  try consumer.accept(arena) catch {
    ArenaAlreadyProvided(2) => ()
    _ => fail("unexpected duplicate-arena rejection")
  } noraise {
    _ => fail("consumer accepted one arena twice")
  }
  consumer.revoke()
  try consumer.accept(arena) catch {
    BorrowClosed(2) => ()
    _ => fail("unexpected revoked-borrow rejection")
  } noraise {
    _ => fail("revoked consumer accepted an arena")
  }
}

///|
test "consumer rejects the wrong arena length and failed callback" {
  let consumer = SegmentedHostArenaConsumer::new(
    arena_ordinal=1,
    expected_length=4,
    consume=_ => false,
  )
  try consumer.accept(FixedArray::make(3, b'x')) catch {
    ArenaLengthMismatch(1) => ()
    _ => fail("unexpected length rejection")
  } noraise {
    _ => fail("wrong arena length was admitted")
  }
  try consumer.accept(FixedArray::make(4, b'x')) catch {
    ConsumerRejected(1) => ()
    _ => fail("unexpected consumer rejection")
  } noraise {
    _ => fail("failed consume callback was admitted")
  }
  assert_false(consumer.was_accepted())
}

///|
test "consumer constructor rejects unbounded identities" {
  try
    SegmentedHostArenaConsumer::new(arena_ordinal=-1, expected_length=1, consume=_ => {
      true
    })
  catch {
    InvalidConsumer => ()
    _ => fail("unexpected invalid-consumer rejection")
  } noraise {
    _ => fail("negative arena ordinal was admitted")
  }
}
