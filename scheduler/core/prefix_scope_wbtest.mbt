///|
test "oversized cache scope is a miss and never evicts valid entries" {
  let owner = wb_prefix_scheduler()
  let (seed, seed_lease) = wb_prefix_request(
    961UL,
    [1, 2, 3, 4, 5],
    ReadWrite,
    b"scope",
  )
  ignore(try! owner.admit(seed, 0UL))
  wb_complete_one_token_request(owner, 7)
  ignore(try! owner.take_generated_token())
  ignore(try! owner.take_terminal_notice())
  try! seed_lease.release()

  let (oversized, oversized_lease) = wb_prefix_request(
    962UL,
    [8, 9, 10, 11, 12],
    ReadWrite,
    b"abcdefghij",
  )
  ignore(try! owner.admit(oversized, 0UL))
  wb_complete_one_token_request(owner, 7)
  ignore(try! owner.take_generated_token())
  ignore(try! owner.take_terminal_notice())
  try! oversized_lease.release()
  let telemetry = owner.luna_telemetry()
  assert_eq(telemetry.prefix_entries(), 1)
  assert_eq(telemetry.prefix_evictions(), 0UL)
  assert_eq(telemetry.prefix_publications(), 1UL)
}

///|
test "cache-disabled request neither looks up nor publishes" {
  let owner = wb_prefix_scheduler()
  let (request, lease) = wb_prefix_request(
    963UL,
    [1, 2, 3, 4, 5],
    Disabled,
    b"scope",
  )
  ignore(try! owner.admit(request, 0UL))
  wb_complete_one_token_request(owner, 7)
  assert_eq((try! owner.take_generated_token()).token_id(), 7)
  ignore(try! owner.take_terminal_notice())
  try! lease.release()
  let telemetry = owner.luna_telemetry()
  assert_eq(telemetry.prefix_lookups(), 0UL)
  assert_eq(telemetry.prefix_publications(), 0UL)
  assert_eq(telemetry.prefix_entries(), 0)
  assert_eq(telemetry.prefix_pages(), 0)
}
