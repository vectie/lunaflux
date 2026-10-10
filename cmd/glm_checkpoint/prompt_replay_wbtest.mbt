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
