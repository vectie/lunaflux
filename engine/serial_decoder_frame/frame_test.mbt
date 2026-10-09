///|
test "chunked prefill and decode share exact committed history and selected rows" {
  let host = host_frame()
  let (submitted, frame, writer) = request_plan()
  host.stage(frame, writer)
  assert_eq(host.processed_tokens(), 0)
  assert_eq(host.live_rows(), 2)
  assert_false(host.produces_token())
  assert_eq(port_value(host, 0), 1)
  assert_eq(port_value(host, 0, index=1), 0)
  assert_eq(port_value(host, 0, index=2), 1)
  assert_eq(port_value(host, 0, index=3), 2)
  assert_eq(port_value(host, 1), 2)
  assert_eq(port_value(host, 1, index=1), 3)
  assert_eq(port_value(host, 2, index=1), 2)
  assert_eq(port_value(host, 4), 1)
  assert_eq(port_value(host, 5), 1)
  assert_eq(port_value(host, 6), 0)
  let completion = host.complete(sampled_token=-1, execution_failed=false)
  completion.validate_for(submitted)
  host.completion().validate_for(submitted)
  assert_eq(host.processed_tokens(), 2)
  let (submitted, frame, writer) = request_plan(
    position=2,
    tokens=[4, 5, 6],
    produces=true,
  )
  host.stage(frame, writer)
  assert_eq(port_value(host, 4), 0)
  assert_eq(port_value(host, 5), 0)
  assert_eq(port_value(host, 6), 1)
  assert_eq(port_value(host, 7), 2)
  let completion = host.complete(sampled_token=11, execution_failed=false)
  completion.validate_for(submitted)
  assert_eq(completion.entry(0).token_id(), 11)
  assert_eq(host.processed_tokens(), 5)
  let (submitted, frame, writer) = request_plan(
    position=5,
    tokens=[11],
    decode=true,
  )
  host.stage(frame, writer)
  assert_eq(port_value(host, 0), 0)
  assert_eq(port_value(host, 0, index=1), 1)
  assert_eq(port_value(host, 7), 0)
  let completion = host.complete(sampled_token=12, execution_failed=false)
  completion.validate_for(submitted)
  assert_eq(host.processed_tokens(), 6)
  host.release_request()
  let (_, frame, writer) = request_plan(id=8UL, generation=1UL)
  host.stage(frame, writer)
  assert_eq(port_value(host, 4), 1)
  assert_eq(port_value(host, 5), 1)
  host.close()
  host.close()
}

///|
test "exact context capacity accepts its last row but rejects the next decode" {
  let host = host_frame()
  for position in [0, 4, 8] {
    let (submitted, frame, writer) = request_plan(
      position~,
      tokens=[1, 2, 3, 4],
      produces=position == 8,
    )
    host.stage(frame, writer)
    host
    .complete(sampled_token=7, execution_failed=false)
    .validate_for(submitted)
  }
  assert_eq(host.processed_tokens(), 12)
  let (_, frame, writer) = request_plan(position=12, tokens=[7], decode=true)
  try host.stage(frame, writer) catch {
    @serial_decoder_frame.SerialDecoderError::InvalidPosition => ()
    error => raise error
  } noraise {
    _ => fail("advanced beyond retained history capacity")
  }
  writer.abort()
  assert_eq(host.processed_tokens(), 12)
  host.close()
  try host.completion() catch {
    @serial_decoder_frame.SerialDecoderError::Closed => ()
    error => raise error
  } noraise {
    _ => fail("closed frame exposed a completion")
  }
}

///|
test "unsupported sampling and wrong position reject before state changes" {
  let host = host_frame()
  let (_, frame, writer) = request_plan(produces=true, stochastic=true)
  try host.stage(frame, writer) catch {
    @serial_decoder_frame.SerialDecoderError::UnsupportedSampling => ()
    error => raise error
  } noraise {
    _ => fail("silently changed sampling")
  }
  writer.abort()
  let (_, frame, writer) = request_plan(position=2)
  try host.stage(frame, writer) catch {
    @serial_decoder_frame.SerialDecoderError::InvalidPosition => ()
    error => raise error
  } noraise {
    _ => fail("invented prefix state")
  }
  writer.abort()
  let (_, frame, writer) = request_plan()
  host.stage(frame, writer)
  ignore(host.complete(sampled_token=0, execution_failed=false))
  let (_, frame, writer) = request_plan(position=1)
  try host.stage(frame, writer) catch {
    @serial_decoder_frame.SerialDecoderError::InvalidPosition => ()
    error => raise error
  } noraise {
    _ => fail("reprocessed history")
  }
  writer.abort()
  let (_, frame, writer) = request_plan(id=8UL, position=0)
  try host.stage(frame, writer) catch {
    @serial_decoder_frame.SerialDecoderError::RequestStillLive => ()
    error => raise error
  } noraise {
    _ => fail("overwrote another request")
  }
  writer.abort()
  assert_eq(host.processed_tokens(), 2)
  host.close()
}

///|
test "samples outside vocabulary and sparse errors publish failure and poison history" {
  for token in [-1, 16] {
    let host = host_frame()
    let (submitted, frame, writer) = request_plan(produces=true)
    host.stage(frame, writer)
    let result = host.complete(sampled_token=token, execution_failed=false)
    result.validate_for(submitted)
    assert_true(host.is_faulted())
    assert_eq(host.processed_tokens(), 0)
    try host.release_request() catch {
      @serial_decoder_frame.SerialDecoderError::Faulted => ()
      error => raise error
    } noraise {
      _ => fail("reused failed GPU history")
    }
    host.close()
  }
  let host = host_frame()
  let (submitted, frame, writer) = request_plan()
  host.stage(frame, writer)
  let result = host.complete(sampled_token=1, execution_failed=true)
  result.validate_for(submitted)
  assert_true(host.is_faulted())
  host.close()
}

///|
test "token and context limits reject before preparing GPU payloads" {
  let host = host_frame()
  let (_, frame, writer) = request_plan(tokens=[16])
  try host.stage(frame, writer) catch {
    @serial_decoder_frame.SerialDecoderError::InvalidToken => ()
    error => raise error
  } noraise {
    _ => fail("out of vocabulary embedding")
  }
  writer.abort()
  let (_, frame, writer) = request_plan(model_generation=10UL)
  try host.stage(frame, writer) catch {
    @serial_decoder_frame.SerialDecoderError::ModelGenerationMismatch => ()
    error => raise error
  } noraise {
    _ => fail("wrong loaded generation")
  }
  writer.abort()
  let (_, frame, writer) = request_plan(tokens=[1, 2, 3, 4, 5])
  try host.stage(frame, writer) catch {
    @serial_decoder_frame.SerialDecoderError::InvalidPosition => ()
    error => raise error
  } noraise {
    _ => fail("row ceiling ignored")
  }
  writer.abort()
  host.close()
}
