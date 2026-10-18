///|
test "invalid token and model generation reject before leasing any history" {
  let host = host()
  let (_, frame, writer) = plan([(10UL, 0UL, 0, [16], false, false, false)])
  try host.stage(frame, writer) catch {
    @batched_decoder_frame.BatchFrameError::InvalidToken => ()
    e => raise e
  } noraise {
    _ => fail("invalid embedding index")
  }
  writer.abort()
  let (_, frame, writer) = plan(
    [(10UL, 0UL, 0, [1], false, false, false)],
    model_generation=10UL,
  )
  try host.stage(frame, writer) catch {
    @batched_decoder_frame.BatchFrameError::ModelGenerationMismatch => ()
    e => raise e
  } noraise {
    _ => fail("wrong model generation")
  }
  writer.abort()
  let (_, frame, writer) = plan([(10UL, 0UL, 0, [1], false, true, false)])
  host.stage(frame, writer)
  assert_eq(port(host, SequenceSlots), [0])
  ignore(host.complete([2], execution_failed=false))
  for generation in [0UL, 1UL] {
    let (_, frame, writer) = plan([
      (10UL, generation, 0, [1], false, false, false),
    ])
    try host.stage(frame, writer) catch {
      @batched_decoder_frame.BatchFrameError::InvalidPosition =>
        assert_eq(generation, 0UL)
      @batched_decoder_frame.BatchFrameError::GenerationMismatch =>
        assert_eq(generation, 1UL)
      e => raise e
    } noraise {
      _ => fail("silently reset retained request")
    }
    writer.abort()
  }
  assert_eq(host.processed_tokens(10UL, 0UL), 1)
  host.close()
}

///|
test "invalid completion shape can be retried without publishing partial results" {
  let host = host()
  let (submitted, frame, writer) = plan([
    (10UL, 0UL, 0, [1], false, true, true),
    (20UL, 0UL, 0, [1], false, true, true),
  ])
  host.stage(frame, writer)
  try host.complete([3], execution_failed=false) catch {
    @batched_decoder_frame.BatchFrameError::InvalidCapacity => ()
    e => raise e
  } noraise {
    _ => fail("truncated completion")
  }
  assert_false(host.is_faulted())
  assert_eq(host.output_rows(), 2)
  host.complete([3, 4], execution_failed=false).validate_for(submitted)
  host.completion().validate_for(submitted)
  host.close()
}

///|
test "transport copies only live prefixes and capacity is not tied to context" {
  // Batch token capacity can exceed an individual sequence's context capacity.
  let host = @batched_decoder_frame.BatchedDecoderFrame::new(
    maximum_rows=8,
    maximum_sequences=4,
    cache_slots=8,
    maximum_context=2,
    vocabulary_size=16,
    model_generation=9UL,
  )
  let (_, frame, writer) = plan([
    (10UL, 0UL, 0, [1, 2], false, true, false),
    (20UL, 0UL, 0, [3, 4], false, false, false),
  ])
  host.stage(frame, writer)
  assert_eq(host.live_rows(), 4)
  let copied = FixedArray::make(host.input_capacity(TokenIds), b'\xff')
  assert_eq(host.copy_input(TokenIds, copied), 16)
  for index in 16..<copied.length() {
    assert_eq(copied[index], b'\xff')
  }
  try host.copy_input(TokenIds, FixedArray::make(15, b'\x00')) catch {
    @batched_decoder_frame.BatchFrameError::InvalidCapacity => ()
    e => raise e
  } noraise {
    _ => fail("short DMA payload")
  }
  host.retire_stage(execution_failed=false)
  assert_eq(host.processed_tokens(10UL, 0UL), 2)
  assert_eq(host.processed_tokens(20UL, 0UL), 2)
  host.close()
}
