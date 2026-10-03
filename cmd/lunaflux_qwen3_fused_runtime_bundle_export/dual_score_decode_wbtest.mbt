///|
test "dual score export declares one numerical law and rejects mixed flags" {
  let flags = [
    "--decode-dual-score-blockwise-f32-v2", "--decode-blockwise-f32-v1", "--decode-blockwise-fma-f32-v3",
  ]
  let raw = runtime_bundle_test_arguments("5")
  raw.append(["--ingress-evaluate", "full", flags[0]])
  let (args, _) = prepare_ingress_choice(raw)
  assert_true(args.blockwise_decode && args.dual_score_decode)
  assert_false(args.contracting_decode)
  for first in flags {
    for second in flags {
      let conflicting = runtime_bundle_test_arguments("5")
      conflicting.append(["--ingress-evaluate", "full", first, second])
      try prepare_ingress_choice(conflicting) catch {
        InvalidArguments => ()
        _ => fail("unexpected conflicting law error")
      } noraise {
        _ => fail("multiple decode laws accepted")
      }
    }
  }
}
