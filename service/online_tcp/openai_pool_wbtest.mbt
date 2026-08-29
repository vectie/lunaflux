///|
test "OpenAI connection pool limits are startup bounded" {
  let limits = LunaOnlineOpenAIConnectionPoolLimits::new(
    connection_capacity=2,
    output_buffer_bytes=4096,
    io_poll_timeout_millis=1,
  )
  assert_eq(limits.connection_capacity(), 2)
  assert_eq(limits.output_buffer_bytes(), 4096)
  for capacity in [0, 1, LUNA_ONLINE_TCP_MAX_TRANSITIONS + 1] {
    try
      LunaOnlineOpenAIConnectionPoolLimits::new(
        connection_capacity=capacity,
        output_buffer_bytes=4096,
        io_poll_timeout_millis=1,
      )
    catch {
      LunaOnlineTcpFailed(LunaOnlineTcpLimits) => ()
      _ => fail("unexpected connection pool capacity error")
    } noraise {
      _ => fail("invalid connection pool capacity was admitted")
    }
  }
}

///|
test "OpenAI connection pool output and poll bounds fail closed" {
  for output in [0, -1, LUNA_ONLINE_TCP_MAX_SCRATCH_BYTES + 1] {
    try
      LunaOnlineOpenAIConnectionPoolLimits::new(
        connection_capacity=2,
        output_buffer_bytes=output,
        io_poll_timeout_millis=1,
      )
    catch {
      LunaOnlineTcpFailed(LunaOnlineTcpLimits) => ()
      _ => fail("unexpected connection pool output error")
    } noraise {
      _ => fail("invalid connection pool output capacity was admitted")
    }
  }
  for timeout in [0, -1, LUNA_ONLINE_TCP_MAX_TIMEOUT_MILLIS + 1] {
    try
      LunaOnlineOpenAIConnectionPoolLimits::new(
        connection_capacity=2,
        output_buffer_bytes=4096,
        io_poll_timeout_millis=timeout,
      )
    catch {
      LunaOnlineTcpFailed(LunaOnlineTcpLimits) => ()
      _ => fail("unexpected connection pool poll error")
    } noraise {
      _ => fail("invalid connection pool poll timeout was admitted")
    }
  }
}

///|
test "OpenAI connection pool requires one write to hold a complete output cell" {
  let fitting = LunaOnlineTcpLimits::new(
    read_chunk_bytes=64,
    write_chunk_bytes=4096,
    accept_timeout_millis=1,
    input_idle_timeout_millis=1,
    write_timeout_millis=1,
    reactor_transition_budget=1,
  )
  let too_short = LunaOnlineTcpLimits::new(
    read_chunk_bytes=64,
    write_chunk_bytes=4095,
    accept_timeout_millis=1,
    input_idle_timeout_millis=1,
    write_timeout_millis=1,
    reactor_transition_budget=1,
  )
  let pool = LunaOnlineOpenAIConnectionPoolLimits::new(
    connection_capacity=2,
    output_buffer_bytes=4096,
    io_poll_timeout_millis=1,
  )
  assert_true(pool.fits_write_chunk(fitting))
  assert_false(pool.fits_write_chunk(too_short))
}

///|
test "OpenAI pool recognizes the final write for immediate flight release" {
  assert_true(luna_openai_pool_write_completes(0, 17, 17))
  assert_true(luna_openai_pool_write_completes(9, 17, 8))
  assert_false(luna_openai_pool_write_completes(0, 17, 16))
  assert_false(luna_openai_pool_write_completes(9, 17, 0))
  assert_false(luna_openai_pool_write_completes(9, 17, 9))
}
