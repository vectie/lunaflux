///|
fn prompt_limits(
  rows : Int,
  history : Int,
  vocabulary : Int,
) -> (@protocol.WorkerProtocolLimits, @inference.InferenceLimits) raise {
  (
    @protocol.WorkerProtocolLimits::new(
      max_prefill_rows=1,
      max_decode_rows=1,
      max_plan_rows=2,
      max_plan_tokens=rows,
      max_plan_pages=1,
      max_capabilities=1,
      max_completion_slots=2,
      max_sequence_tokens=history,
      max_token_id=vocabulary - 1,
    ),
    @inference.InferenceLimits::new(
      max_text_bytes=1048576,
      max_input_tokens=history,
      max_new_tokens=history,
      max_context_tokens=history,
      max_token_id=vocabulary - 1,
      max_stop_token_ids=1,
      max_stop_strings=1,
      max_stop_string_bytes=1,
      max_trace_bytes=1,
      max_cache_scope_bytes=1,
      max_decoded_delta_bytes=1,
      max_deadline_millis=300000UL,
      max_top_k=1,
      max_temperature=1.0,
    ),
  )
}

///|
fn preflight_generation(
  model_root : String,
  rows : String,
  history : String,
  maximum : String,
  stop : String,
  frames : String,
  count : String,
  byte_budget : String,
) -> Unit raise {
  let root = @approved_fs.ApprovedRoot::open_absolute(model_root)
  defer root.close()
  let spec = inspect_config(root)
  let rows = @string.parse_int(rows)
  let history = @string.parse_int(history)
  let maximum = @string.parse_int(maximum)
  let stop = @string.parse_int(stop)
  guard rows > 0 &&
    rows <= history &&
    history <= spec.max_position_embeddings() &&
    stop >= -1 &&
    stop < spec.vocabulary_size() else {
    fail("invalid generation geometry or stop token")
  }
  let (limits, inference) = prompt_limits(rows, history, spec.vocabulary_size())
  let frame_root = @approved_fs.ApprovedRoot::open_absolute(frames)
  defer frame_root.close()
  let prompt = @prompt_file.load_generation_prompt(
    frame_root,
    @string.parse_int(count),
    limits,
    inference,
    maximum_new_tokens=maximum,
    maximum_context=history,
    maximum_bytes=@string.parse_int64(byte_budget),
    model_generation=1UL,
  )
  println(
    "request_prepared: input_tokens=\{prompt.input_token_count()} maximum_new_tokens=\{prompt.maximum_new_tokens()} context=\{history}",
  )
}
