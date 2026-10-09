///|
test "OpenAI runnable handoff cannot acquire an idle listener timer" {
  for
    progress in [
      LunaOnlineOpenAIConnectionPoolAdvanced,
      LunaOnlineOpenAIConnectionAccepted,
      LunaOnlineOpenAIConnectionRetired,
      LunaOnlineOpenAIConnectionPoolDrainingProgress,
      LunaOnlineOpenAIConnectionPoolClosedProgress,
    ] {
    for waiting in [false, true] {
      guard luna_openai_listener_poll(progress, waiting, 1) is SkipListener else {
        fail("runnable or terminal effect acquired an idle listener wait")
      }
    }
  }
}

///|
test "OpenAI pending worker admits only a nonblocking listener probe" {
  guard luna_openai_listener_poll(
      LunaOnlineOpenAIConnectionPoolBackpressured,
      true,
      1000,
    )
    is ProbeListener else {
    fail("worker feedback can be delayed by an idle admission timer")
  }
}

///|
test "OpenAI idle listener retains its configured bounded wait" {
  for timeout in [1, 17, LUNA_ONLINE_TCP_MAX_TIMEOUT_MILLIS] {
    guard luna_openai_listener_poll(
        LunaOnlineOpenAIConnectionPoolBackpressured,
        false,
        timeout,
      )
      is WaitListener(actual) else {
      fail("idle reactor lost its bounded wait")
    }
    assert_eq(actual, timeout)
  }
}
