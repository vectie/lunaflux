///|
fn assert_string_stop_session(
  executable : Bytes,
  tokenizer : @tokenizer.TokenizerSpec,
  received : @request_admission.ReceivedRequest,
  expected_tokens : Array[String?],
  expected_output_tokens : Int,
) -> Unit raise {
  let instance = prepare_foundation_instance(executable, tokenizer.digest())
  let session = begin_foundation_session(
    instance,
    prepare_foundation_request(tokenizer, received),
  )
  session.ack_event()
  let mut token_index = 0
  let mut terminal = false
  while !terminal {
    progress_foundation_to_event(session)
    match foundation_event(session) {
      Token(value) => {
        assert_true(token_index < expected_tokens.length())
        match (value.decoded_delta(), expected_tokens[token_index]) {
          (Some(delta), Some(expected)) => assert_eq(delta.text(), expected)
          (None, None) => ()
          _ => raise Failure("online string-stop token prefix mismatch")
        }
        token_index += 1
        session.ack_event()
      }
      Usage(value) => {
        assert_eq(token_index, expected_tokens.length())
        assert_eq(value.output_tokens(), expected_output_tokens)
        session.ack_event()
        match foundation_event(session) {
          Completed(completed) => {
            assert_eq(completed.finish_reason(), StopSequence)
            assert_eq(
              completed.final_usage().output_tokens(),
              expected_output_tokens,
            )
            assert_true(completed.final_delta() is None)
          }
          _ => raise Failure("online string stop did not publish Completed")
        }
        session.ack_event()
        terminal = true
      }
      _ => raise Failure("online string stop leaked a physical terminal")
    }
  }
  progress_foundation_to_idle(session)
  drain_foundation_instance(instance)
}

///|
fn assert_string_cut_abort(
  executable : Bytes,
  request_id : UInt64,
  ack_prefix : Bool,
) -> Unit raise {
  let tokenizer = string_session_tokenizer(b"ax", "5".repeat(64))
  let instance = prepare_foundation_instance(executable, tokenizer.digest())
  let session = begin_foundation_session(
    instance,
    prepare_foundation_request(
      tokenizer,
      session_received(
        Streaming,
        request_id~,
        stop_string=b"x",
        max_new_tokens=2,
      ),
    ),
  )
  session.ack_event()
  progress_foundation_to_event(session)
  match foundation_event(session) {
    Token(value) => assert_eq(value.decoded_delta().unwrap().text(), "a")
    _ => raise Failure("online string cut did not pin its safe prefix")
  }
  if ack_prefix {
    session.ack_event()
  }
  session.begin_abort()
  progress_foundation_to_idle(session)
  drain_foundation_instance(instance)
}
