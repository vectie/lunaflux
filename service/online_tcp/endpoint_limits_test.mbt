///|
test "Luna online TCP limits reject every unbounded envelope" {
  let valid = LunaOnlineTcpLimits::new(
    read_chunk_bytes=4096,
    write_chunk_bytes=4096,
    accept_timeout_millis=1000,
    input_idle_timeout_millis=1000,
    write_timeout_millis=1000,
    reactor_transition_budget=17,
  )
  assert_eq(valid.read_chunk_bytes(), 4096)
  assert_eq(valid.write_chunk_bytes(), 4096)
  assert_eq(valid.reactor_transition_budget(), 17)
  for capacity in [0, 16777217] {
    try
      LunaOnlineTcpLimits::new(
        read_chunk_bytes=capacity,
        write_chunk_bytes=4096,
        accept_timeout_millis=1000,
        input_idle_timeout_millis=1000,
        write_timeout_millis=1000,
        reactor_transition_budget=17,
      )
    catch {
      LunaOnlineTcpFailed(LunaOnlineTcpLimits) => ()
      _ => fail("unexpected Luna TCP capacity error")
    } noraise {
      _ => fail("invalid Luna TCP capacity passed")
    }
  }
  for timeout in [0, 86400001] {
    try
      LunaOnlineTcpLimits::new(
        read_chunk_bytes=4096,
        write_chunk_bytes=4096,
        accept_timeout_millis=timeout,
        input_idle_timeout_millis=1000,
        write_timeout_millis=1000,
        reactor_transition_budget=17,
      )
    catch {
      LunaOnlineTcpFailed(LunaOnlineTcpLimits) => ()
      _ => fail("unexpected Luna TCP timeout error")
    } noraise {
      _ => fail("invalid Luna TCP timeout passed")
    }
  }
}
