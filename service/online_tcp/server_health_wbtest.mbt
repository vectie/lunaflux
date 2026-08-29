///|
test "server health is sticky-failed for every latched rule code" {
  assert_eq(luna_online_tcp_server_health(0, false), LunaOnlineTcpServerHealthy)
  assert_eq(luna_online_tcp_server_health(0, true), LunaOnlineTcpServerFailed)
  for code in [1, 2, 17, 2147483647] {
    assert_eq(
      luna_online_tcp_server_health(code, false),
      LunaOnlineTcpServerFailed,
    )
  }
}

///|
test "pipeline boundary rejects a partial next frame" {
  assert_false(
    luna_online_tcp_pipeline_boundary_ready(true, true, 0, false, false),
  )
}

///|
test "pipeline boundary rejects a backpressured coalesced tail" {
  assert_false(
    luna_online_tcp_pipeline_boundary_ready(true, true, 17, true, true),
  )
}

///|
test "pipeline boundary requires all authenticated scalar evidence" {
  assert_true(
    luna_online_tcp_pipeline_boundary_ready(true, true, 0, false, true),
  )
  assert_false(
    luna_online_tcp_pipeline_boundary_ready(false, true, 0, false, true),
  )
  assert_false(
    luna_online_tcp_pipeline_boundary_ready(true, false, 0, false, true),
  )
}
