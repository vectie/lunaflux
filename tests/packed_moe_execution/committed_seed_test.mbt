///|
test "committed seed upload is bounded allocation-free and retires before reuse" {
  let baseline = live_resources()
  let context = @device.open_context(0)
  let seed = @text_frame.CommittedTokenSeed::new(context, vocabulary_size=1024)
  let output = FixedArray::make(4, b'\x00')
  let allocation = seed.allocation()
  let waits = blocking_syncs()
  begin_allocations()
  for token in 0..<256 {
    seed.begin(token)
    while !seed.poll() {
      ()
    }
  }
  assert_eq(end_allocations(), 0UL)
  assert_eq(blocking_syncs(), waits)
  context.copy_to_fixed_host(allocation, output, 0L, 0L, 4L)
  assert_eq(output[0], b'\xff')
  assert_eq(output[1], b'\x00')
  for invalid in [-1, 1024] {
    try seed.begin(invalid) catch {
      _ => ()
    } noraise {
      _ => fail("invalid seed uploaded")
    }
  }
  seed.begin(513)
  try seed.begin(2) catch {
    _ => ()
  } noraise {
    _ => fail("pending seed replaced")
  }
  // Cancellation drains the only pending upload and its allocation lease.
  seed.close()
  seed.close()
  context.close()
  assert_eq(live_resources(), baseline)
}
