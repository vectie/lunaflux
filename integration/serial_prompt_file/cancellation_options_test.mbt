///|
test "request cancellation suffix is parsed before loading and has no hidden defaults" {
  let (plain, absent) = @serial_prompt_file.split_request_cancellation_options([
    "x",
  ])
  assert_eq(plain.to_owned(), ["x"])
  assert_true(absent is None)
  let (args, option) = @serial_prompt_file.split_request_cancellation_options([
    "frames", "2", "--cancel-request", "0", "--after-steps", "3",
  ])
  assert_eq(args.to_owned(), ["frames", "2"])
  guard option is Some(plan) else { fail("missing cancellation plan") }
  plan.validate_request_count(2)
  assert_eq(plan.after_steps_for(0), 3)
  assert_eq(plan.after_steps_for(1), 0)
  for
    malformed in [
      ["--cancel-request", "0"],
      ["--after-steps", "1"],
      ["--cancel-request", "-1", "--after-steps", "1"],
      ["--cancel-request", "0", "--after-steps", "0"],
      ["--cancel-request", "x", "--after-steps", "1"],
      ["--cancel-request", "0", "--after-steps", "1", "extra"],
      ["--cancel-request", "--cancel-request", "0", "--after-steps", "1"],
    ] {
    try
      @serial_prompt_file.split_request_cancellation_options(malformed)
    catch {
      _ => ()
    } noraise {
      _ => fail("malformed cancellation suffix accepted")
    }
  }
}
