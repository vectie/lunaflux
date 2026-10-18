///|
test "generation admits all declared EOS ids without changing the single-id CLI" {
  assert_eq(parse_generation_stops("154820,154827,154829", 154880), [
    154820, 154827, 154829,
  ])
  assert_eq(parse_generation_stops("154827", 154880), [154827])
  assert_eq(parse_generation_stops("-1", 154880), [])
  for invalid in ["", "154820,154820", "-1,154827", "154880", "1,"] {
    try parse_generation_stops(invalid, 154880) catch {
      _ => ()
    } noraise {
      _ => fail("invalid stop accepted")
    }
  }
}

///|
test "every declared EOS retires on prefill and decode without another step" {
  let (limits, inference) = prompt_limits(2, 16, 154880)
  let encoded = @prompt.encode_prompt(
    [64, 65],
    limits,
    inference,
    maximum_rows=2,
    maximum_context=16,
    model_generation=1UL,
  )
  let plans = @wire.PlanFrameBuffer::new(limits, inference)
  let scratch = FixedArray::make(plans.capacity(), b'\x00')
  for i, byte in encoded[0] {
    scratch[i] = byte
  }
  let plan = plans.load(scratch, encoded[0].length())
  let completions = @wire.CompletionFrameBuffer::new(limits, inference)
  let stops = parse_generation_stops("154820,154827,154829", 154880)
  for stop in stops {
    for initial_stop in [true, false] {
      let writer = completions.writer_for(plan)
      writer.append_final_prefill_completed(
        plan.prefill_row(0),
        if initial_stop {
          stop
        } else {
          66
        },
      )
      let owner = @generation.SerialGreedyGeneration::new(
        plan,
        writer.submit(),
        limits,
        inference,
        maximum_new_tokens=4,
        maximum_context=16,
        stop_tokens=stops,
      )
      if !initial_stop {
        let next = owner.next_decode()
        let writer = completions.writer_for(next)
        writer.append_decode_completed(next.decode_row(0), stop)
        owner.accept(writer.submit())
      }
      assert_eq(owner.finish_reason(), @generation.GenerationFinish::StopToken)
      assert_eq(owner.token_count(), if initial_stop { 1 } else { 2 })
      try owner.next_decode() catch {
        @generation.SerialGenerationError::Finished => ()
        error => raise error
      } noraise {
        _ => fail("EOS submitted another step")
      }
    }
  }
}

///|
async test "GLM prompt replay resolves capacity before file authority is released" {
  let directory = @fs.tmpdir(prefix="lunaflux-glm-prompt-replay")
  defer @fs.rmdir(directory, recursive=true)
  let path = @fs.realpath(directory)
  let (limits, inference) = prompt_limits(2, 16, 128)
  let frames = @prompt.encode_prompt(
    [64, 65, 66, 67, 68],
    limits,
    inference,
    maximum_rows=2,
    maximum_context=16,
    model_generation=1UL,
  )
  for index, bytes in frames {
    @fs.write_file(path + "/plan-\{index}.bin", bytes, create_mode=CreateNew)
  }
  let root = @approved_fs.ApprovedRoot::open_absolute(path)
  defer root.close()
  for maximum in [12, 0, -2147483648] {
    try
      @prompt_file.load_generation_prompt(
        root,
        3,
        limits,
        inference,
        maximum_new_tokens=maximum,
        maximum_context=16,
        maximum_bytes=1048576L,
        model_generation=1UL,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid generation capacity accepted")
    }
  }
  let prompt = @prompt_file.load_generation_prompt(
    root,
    3,
    limits,
    inference,
    maximum_new_tokens=11,
    maximum_context=16,
    maximum_bytes=1048576L,
    model_generation=1UL,
  )
  root.close()
  assert_eq(prompt.input_token_count(), 5)
  assert_eq(prompt.frame_count(), 3)
  for index, bytes in frames {
    assert_eq(prompt.frame_bytes(index), bytes)
  }
  try
    @prompt_file.load_generation_prompt(
      root,
      3,
      limits,
      inference,
      maximum_new_tokens=11,
      maximum_context=16,
      maximum_bytes=1048576L,
      model_generation=1UL,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("closed file authority unexpectedly reusable")
  }
}
