///|
test "paged profile digest changes with trace admitted profile and page trace" {
  let base = admit_paged_profile([paged_profile_observation(4, 600L)], 600L)
  let changed_trace = admit_paged_profile_with(
    [paged_profile_observation(4, 600L)],
    600L,
    paged_profile_launches(paged_profile_test_plan()),
    7,
    '0',
  )
  let plan = paged_profile_test_plan()
  let changed_contract = admit_paged_profile_with(
    [paged_profile_observation(4, 600L)],
    600L,
    paged_profile_launches_with(
      plan,
      @launch_contract.PagedKernelProfile::new(
        id=@launch_contract.KernelProfileId::from_stable_int(7),
        max_query_rows=4,
        max_query_tokens=16,
        max_page_table_entries=32,
      ),
      4,
    ),
    7,
    'f',
  )
  let changed_table = admit_paged_profile(
    [
      paged_profile_observation(
        4,
        600L,
        shape=paged_profile_shape(page_table_trace='0'),
      ),
    ],
    600L,
  )
  assert_true(base.digest() != changed_trace.digest())
  assert_true(base.digest() != changed_contract.digest())
  assert_true(base.digest() != changed_table.digest())
  assert_true(base.contract().launches().length() > 0)
  assert_eq(
    base.contract().launch_identity(),
    admit_paged_profile([paged_profile_observation(4, 600L)], 600L)
    .contract()
    .launch_identity(),
  )
}
