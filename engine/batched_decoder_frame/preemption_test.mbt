///|
test "acknowledged eviction admits a waiter and replays the same generation" {
  let host = host()
  defer host.close()
  let (_, full, writer) = plan([
    (10UL, 0UL, 0, [1], false, true, false),
    (20UL, 1UL, 0, [2], false, true, false),
    (30UL, 2UL, 0, [3], false, true, false),
  ])
  host.stage(full, writer)
  ignore(host.complete([2, 3, 4], execution_failed=false))
  let (_, waiting, writer) = plan([(40UL, 3UL, 0, [4], false, true, false)])
  try host.stage(waiting, writer) catch {
    @batched_decoder_frame.CapacityExceeded => ()
    e => raise e
  } noraise {
    _ => fail("unacknowledged eviction reused resident slot")
  }
  host.retire_request(10UL, 0UL)
  host.stage(waiting, writer)
  assert_eq(port(host, RecurrentResets), [1])
  ignore(host.complete([5], execution_failed=false))
  assert_eq(host.processed_tokens(20UL, 1UL), 1)
  assert_eq(host.processed_tokens(30UL, 2UL), 1)
  host.retire_request(40UL, 3UL)
  let (_, replay, writer) = plan([(10UL, 0UL, 0, [1, 2], false, true, false)])
  host.stage(replay, writer)
  assert_eq(port(host, RecurrentResets), [1])
  ignore(host.complete([6], execution_failed=false))
  assert_eq(host.processed_tokens(10UL, 0UL), 2)
}
