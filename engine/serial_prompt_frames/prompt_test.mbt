///|
fn fixture_limits() -> (
  @protocol.WorkerProtocolLimits,
  @inference.InferenceLimits,
) raise {
  (
    @protocol.WorkerProtocolLimits::new(
      max_prefill_rows=1,
      max_decode_rows=1,
      max_plan_rows=2,
      max_plan_tokens=4,
      max_plan_pages=1,
      max_capabilities=1,
      max_completion_slots=2,
      max_sequence_tokens=16,
      max_token_id=127,
    ),
    @inference.InferenceLimits::new(
      max_text_bytes=64,
      max_input_tokens=16,
      max_new_tokens=16,
      max_context_tokens=16,
      max_token_id=127,
      max_stop_token_ids=1,
      max_stop_strings=1,
      max_stop_string_bytes=1,
      max_trace_bytes=1,
      max_cache_scope_bytes=1,
      max_decoded_delta_bytes=1,
      max_deadline_millis=1000UL,
      max_top_k=1,
      max_temperature=1.0,
    ),
  )
}

///|
test "offline prompt chunks preserve sequence positions and only sample the last" {
  let (limits, inference) = fixture_limits()
  let ids = [64, 65, 66, 67, 68]
  let bytes = @serial_prompt_frames.encode_prompt(
    ids,
    limits,
    inference,
    maximum_rows=2,
    maximum_context=16,
    model_generation=1UL,
  )
  assert_eq(bytes.length(), 3)
  let owner = @wire.PlanFrameBuffer::new(limits, inference)
  for index, input in bytes {
    let source = FixedArray::make(owner.capacity(), b'\x00')
    for offset, byte in input {
      source[offset] = byte
    }
    let frame = owner.load(source, input.length())
    assert_eq(frame.sequence_value(), index.to_uint64() + 1UL)
    assert_eq(frame.model_generation_value(), 1UL)
    assert_eq(frame.decode_count(), 0)
    assert_eq(frame.prefill_count(), 1)
    let row = frame.prefill_row(0)
    assert_eq(row.sequence_token_start(), index * 2)
    assert_eq(row.produces_token(), index == 2)
    assert_eq(frame.token_count(), if index == 2 { 1 } else { 2 })
    for token in 0..<frame.token_count() {
      assert_eq(frame.token_at(token), ids[index * 2 + token])
    }
  }
}

///|
test "offline prompt refuses empty over-context and foreign tokenizer IDs" {
  let (limits, inference) = fixture_limits()
  for ids in [[], Array::make(16, 64), [128]] {
    try
      @serial_prompt_frames.encode_prompt(
        ids,
        limits,
        inference,
        maximum_rows=2,
        maximum_context=16,
        model_generation=1UL,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid prompt accepted")
    }
  }
}
