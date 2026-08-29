///|
test "duplicate prompt publication retains one physical cached run" {
  let owner = wb_prefix_scheduler()
  let (first, first_lease) = wb_prefix_request(
    971UL,
    [1, 2, 3, 4, 5],
    ReadWrite,
    b"scope",
  )
  ignore(try! owner.admit(first, 0UL))
  wb_complete_one_token_request(owner, 7)
  ignore(try! owner.take_generated_token())
  ignore(try! owner.take_terminal_notice())
  try! first_lease.release()

  let before = owner.luna_telemetry()
  assert_eq(before.prefix_publications(), 1UL)
  assert_eq(before.prefix_entries(), 1)
  assert_eq(before.prefix_pages(), 2)
  assert_eq(before.kv_pages_used(), 2)

  let (duplicate, duplicate_lease) = wb_prefix_request(
    972UL,
    [1, 2, 3, 4, 5],
    ReadWrite,
    b"scope",
  )
  ignore(try! owner.admit(duplicate, 0UL))
  wb_complete_one_token_request(owner, 7)
  ignore(try! owner.take_generated_token())
  ignore(try! owner.take_terminal_notice())
  try! duplicate_lease.release()

  let after = owner.luna_telemetry()
  assert_eq(after.prefix_publications(), 1UL)
  assert_eq(after.prefix_entries(), 1)
  assert_eq(after.prefix_pages(), 2)
  assert_eq(after.kv_pages_used(), 2)
  try! owner.prefix_index.unwrap().debug_check_invariants()
}
