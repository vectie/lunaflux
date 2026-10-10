///|
async test "serial prompt file snapshots survive root closure and reject missing frames" {
  let directory = @fs.tmpdir(prefix="lunaflux-prepared-prompt-files")
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
  let frames = @prompt.encode_prompt(
    [64, 65, 66],
    limits,
    inference,
    maximum_rows=2,
    maximum_context=4,
    model_generation=1UL,
  )
  @fs.write_file(path + "/plan-0.bin", frames[0], create_mode=CreateNew)
  let root = @approved_fs.ApprovedRoot::open_absolute(path)
  defer root.close()
  try
    @serial_prompt_file.load_generation_prompt(
      root,
      2,
      limits,
      inference,
      maximum_new_tokens=1,
      maximum_context=4,
      maximum_bytes=1048576L,
      model_generation=1UL,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("missing final prompt frame accepted")
  }
  @fs.write_file(path + "/plan-1.bin", frames[1], create_mode=CreateNew)
  let replay = @serial_prompt_file.load_generation_prompt(
    root,
    2,
    limits,
    inference,
    maximum_new_tokens=1,
    maximum_context=4,
    maximum_bytes=1048576L,
    model_generation=1UL,
  )
  root.close()
  assert_eq(replay.input_token_count(), 3)
  assert_eq(replay.frame_bytes(1), frames[1])
}
