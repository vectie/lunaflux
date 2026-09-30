///|
test "explicit contraction is mutually exclusive with strict blockwise admission" {
  let raw = runtime_bundle_test_arguments("5")
  raw.append(["--ingress-evaluate", "full", "--decode-blockwise-fma-f32-v3"])
  let (args, _) = prepare_ingress_choice(raw)
  assert_true(args.blockwise_decode && args.contracting_decode)
  raw.push("--decode-blockwise-f32-v1")
  try prepare_ingress_choice(raw) catch {
    InvalidArguments => ()
    _ => fail("unexpected duplicate numerical law error")
  } noraise {
    _ => fail("two numerical laws accepted")
  }
}
