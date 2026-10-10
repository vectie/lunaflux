///|
async test "prompt sequence shares exact storage budget and survives closed file authority" {
  let directory = @fs.tmpdir(prefix="lunaflux-prepared-sequence")
  defer @fs.rmdir(directory, recursive=true)
  let path = @fs.realpath(directory)
  let limits = @protocol.WorkerProtocolLimits::new(
    max_prefill_rows=1,
    max_decode_rows=1,
    max_plan_rows=2,
    max_plan_tokens=2,
    max_plan_pages=1,
    max_capabilities=1,
    max_completion_slots=2,
    max_sequence_tokens=4,
    max_token_id=127,
  )
  let inference = @inference.InferenceLimits::new(
    max_text_bytes=32,
    max_input_tokens=4,
    max_new_tokens=4,
    max_context_tokens=4,
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
  )
  let frames = [
    @prompt.encode_prompt(
      [64],
      limits,
      inference,
      maximum_rows=2,
      maximum_context=4,
      model_generation=1UL,
    ),
    @prompt.encode_prompt(
      [65, 66, 67],
      limits,
      inference,
      maximum_rows=2,
      maximum_context=4,
      model_generation=1UL,
    ),
  ]
  let mut total = 0L
  for request, prompt in frames {
    @fs.mkdir(path + "/request-\{request}")
    for index, bytes in prompt {
      @fs.write_file(
        path + "/request-\{request}/plan-\{index}.bin",
        bytes,
        create_mode=CreateNew,
      )
      total += bytes.length().to_int64()
    }
  }
  let root = @approved_fs.ApprovedRoot::open_absolute(path)
  defer root.close()
  for (budget, maximum) in [(total - 1L, 1), (total, 2)] {
    try
      @serial_prompt_file.load_generation_sequence(
        root,
        [1, 2],
        limits,
        inference,
        maximum_new_tokens=maximum,
        maximum_context=4,
        maximum_bytes=budget,
        model_generation=1UL,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("late request capacity or aggregate storage overflow accepted")
    }
  }
  let prepared = @serial_prompt_file.load_generation_sequence(
    root,
    [1, 2],
    limits,
    inference,
    maximum_new_tokens=1,
    maximum_context=4,
    maximum_bytes=total,
    model_generation=1UL,
  )
  root.close()
  assert_eq(prepared.length(), 2)
  assert_eq(prepared[0].input_token_count(), 1)
  assert_eq(prepared[1].input_token_count(), 3)
  assert_eq(prepared[0].encoded_bytes() + prepared[1].encoded_bytes(), total)
  assert_eq(prepared[1].frame_bytes(1), frames[1][1])
}
