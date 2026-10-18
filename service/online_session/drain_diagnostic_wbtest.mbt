///|
test "cold drain diagnostics observe without changing ownership" {
  let service = coordinator_fixture(multi_lane_count=2).service
  let first = service.drain_diagnostic()
  assert_true(first.contains("tickets=0"))
  assert_true(first.contains("recovery=false"))
  assert_eq(service.drain_diagnostic(), first)
}
