///|
fn replay_fixture() -> (
  @protocol.WorkerProtocolLimits,
  @inference.InferenceLimits,
  Array[Bytes],
) raise {
  let (limits, inference) = fixture_limits()
  (
    limits,
    inference,
    @serial_prompt_frames.encode_prompt(
      [64, 65, 66, 67, 68],
      limits,
      inference,
      maximum_rows=2,
      maximum_context=16,
      model_generation=1UL,
    ),
  )
}

///|
test "prepared replay admits exact prompt plus generation capacity and retains immutable frames" {
  let (limits, inference, frames) = replay_fixture()
  let bytes = frames.fold(init=0L, (used, frame) => {
    used + frame.length().to_int64()
  })
  let expected = frames[0]
  let replay = @serial_prompt_frames.PreparedSerialPrompt::new(
    frames,
    limits,
    inference,
    maximum_new_tokens=11,
    maximum_context=16,
    maximum_encoded_bytes=bytes,
    model_generation=1UL,
  )
  assert_eq(replay.input_token_count(), 5)
  assert_eq(replay.maximum_new_tokens(), 11)
  assert_eq(replay.frame_count(), 3)
  for index, frame in frames {
    assert_eq(replay.frame_bytes(index), frame)
  }
  frames[0] = b"source list replaced after preparation"
  assert_eq(replay.frame_bytes(0), expected)
  for index in [-1, 3] {
    try replay.frame_bytes(index) catch {
      _ => ()
    } noraise {
      _ => fail("out-of-range replay frame accepted")
    }
  }
}

///|
test "prepared replay rejects one-token capacity overflow before execution" {
  let (limits, inference, frames) = replay_fixture()
  for maximum in [0, 12, 16, 17] {
    try
      @serial_prompt_frames.PreparedSerialPrompt::new(
        frames,
        limits,
        inference,
        maximum_new_tokens=maximum,
        maximum_context=16,
        maximum_encoded_bytes=1048576L,
        model_generation=1UL,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("generation capacity overflow accepted")
    }
  }
}

///|
test "prepared replay enforces aggregate retained-byte and model-context limits" {
  let (limits, inference, frames) = replay_fixture()
  let bytes = frames.fold(init=0L, (used, frame) => {
    used + frame.length().to_int64()
  })
  for
    (budget, context, generation) in [
      (0L, 16, 1UL),
      (bytes - 1L, 16, 1UL),
      (bytes, 0, 1UL),
      (bytes, 17, 1UL),
      (bytes, 16, 0UL),
      (bytes, 16, 2UL),
    ] {
    try
      @serial_prompt_frames.PreparedSerialPrompt::new(
        frames,
        limits,
        inference,
        maximum_new_tokens=1,
        maximum_context=context,
        maximum_encoded_bytes=budget,
        model_generation=generation,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid replay envelope accepted")
    }
  }
}

///|
test "prepared replay rejects empty malformed reordered and incomplete prompt chains" {
  let (limits, inference, frames) = replay_fixture()
  for
    invalid in [
      [],
      [b"truncated"],
      [frames[1], frames[0], frames[2]],
      [frames[0], frames[2]],
      [frames[0], frames[1]],
      [frames[0], frames[1], frames[2], frames[2]],
    ] {
    try
      @serial_prompt_frames.PreparedSerialPrompt::new(
        invalid,
        limits,
        inference,
        maximum_new_tokens=1,
        maximum_context=16,
        maximum_encoded_bytes=1048576L,
        model_generation=1UL,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid prompt chain accepted")
    }
  }
}
