///|
test "small row route composes with cached ingress and rejects duplicate or unsafe options" {
  let valid = runtime_bundle_test_arguments("5")
  valid.append([
    "--ingress-evaluate", "full", "--step-rotary-cache", "--ingress-row-variant",
    "64", "16", "0", "/tmp/small.cubin",
  ])
  let (args, choice) = prepare_ingress_choice(valid)
  assert_eq(choice, @luna_fusion_strategy.FullyFused)
  guard args.ingress_row_variant is Some(variant) else {
    fail("row alternative missing")
  }
  assert_eq(variant.maximum_token_rows, 64)
  assert_eq(variant.query_tile_rows, 16)
  assert_eq(variant.shared_bytes, 0)
  assert_eq(variant.module_path, "/tmp/small.cubin")
  for
    (full, cached, path, repeat_option) in [
      (true, false, "/tmp/small.cubin", false),
      (false, false, "/tmp/small.cubin", false),
      (true, true, "relative.cubin", false),
      (true, true, "/tmp/small.cubin", true),
    ] {
    let raw = runtime_bundle_test_arguments("5")
    raw.append(["--ingress-evaluate", if full { "full" } else { "unfused" }])
    if cached {
      raw.push("--step-rotary-cache")
    }
    let option = ["--ingress-row-variant", "64", "16", "0", path]
    raw.append(option)
    if repeat_option {
      raw.append(option)
    }
    try prepare_ingress_choice(raw) catch {
      InvalidArguments => ()
      _ => fail("unexpected row route error")
    } noraise {
      _ => fail("invalid row route accepted")
    }
  }
}
