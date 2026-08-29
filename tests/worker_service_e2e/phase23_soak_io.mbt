///|
async fn phase23_soak_write_fragmented(
  connection : @socket.Tcp,
  source : FixedArray[Byte],
  cycle : Int,
) -> Unit {
  let bytes = Bytes::makei(source.length(), index => source[index])
  let mut cursor = 0
  while cursor < bytes.length() {
    let quantum = 1 + (cursor + cycle) % 17
    let remaining = bytes.length() - cursor
    let offered = if quantum < remaining { quantum } else { remaining }
    let written = connection.write_once(bytes, offset=cursor, len=offered)
    if written <= 0 || written > offered {
      raise Failure("phase23 soak client write violated its bounded range")
    }
    cursor += written
  }
}

///|
fn phase23_soak_assert_usage(
  usage : @inference.Usage,
  output_tokens : Int,
) -> Unit raise {
  assert_eq(usage.input_tokens(), 2)
  assert_eq(usage.cached_input_tokens(), 0)
  assert_eq(usage.output_tokens(), output_tokens)
  assert_eq(usage.total_tokens(), 2 + output_tokens)
}

///|
async fn phase23_soak_read_wave(
  connection : @socket.Tcp,
  cancelled : Bool,
  first_id : UInt64,
  second_id : UInt64,
  ledger : Phase23SoakLedger,
) -> Unit {
  guard tcp_read_event(connection) is Accepted(first_accepted) else {
    raise Failure("phase23 soak lost the first ordered Accepted event")
  }
  assert_eq(first_accepted.model_identity(), identity())
  guard tcp_read_event(connection) is Accepted(second_accepted) else {
    raise Failure("phase23 soak lost the second ordered Accepted event")
  }
  assert_eq(second_accepted.model_identity(), identity())
  if cancelled {
    guard tcp_read_event(connection) is Usage(first_usage) else {
      raise Failure("phase23 soak cancelled request lost Usage")
    }
    phase23_soak_assert_usage(first_usage, 0)
    guard tcp_read_event(connection) is Completed(first_completed) else {
      raise Failure("phase23 soak cancelled request lost Completed")
    }
    assert_eq(first_completed.finish_reason(), Cancelled)
    phase23_soak_assert_usage(first_completed.final_usage(), 0)
  } else {
    guard tcp_read_event(connection) is Token(first_token) else {
      raise Failure("phase23 soak first request lost Token")
    }
    assert_eq(first_token.position(), 0)
    guard tcp_read_event(connection) is Usage(first_usage) else {
      raise Failure("phase23 soak first request lost Usage")
    }
    phase23_soak_assert_usage(first_usage, 1)
    guard tcp_read_event(connection) is Completed(first_completed) else {
      raise Failure("phase23 soak first request lost Completed")
    }
    assert_eq(first_completed.finish_reason(), TokenLimit)
    phase23_soak_assert_usage(first_completed.final_usage(), 1)
  }
  guard tcp_read_event(connection) is Token(second_token) else {
    raise Failure("phase23 soak surviving request lost Token")
  }
  assert_eq(second_token.position(), 0)
  guard tcp_read_event(connection) is Usage(second_usage) else {
    raise Failure("phase23 soak surviving request lost Usage")
  }
  phase23_soak_assert_usage(second_usage, 1)
  guard tcp_read_event(connection) is Completed(second_completed) else {
    raise Failure("phase23 soak surviving request lost Completed")
  }
  assert_eq(second_completed.finish_reason(), TokenLimit)
  phase23_soak_assert_usage(second_completed.final_usage(), 1)
  ledger.terminal_id_sum += first_id + second_id
}
