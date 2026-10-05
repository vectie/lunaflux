///|
test "release target reaches recipe validation and bootstrap without a host patch" {
  let digest = "a".repeat(64)
  let base = [
    "binder", "/model", "config.json", digest, digest, "weights.numeric", digest,
    digest, "/compiled", "/toolchain", digest, "13", "0", "88", "8", "16384", "32",
    "8192", "16384", "/new-output",
  ]
  let legacy = parse_qwen3_release_bind_arguments(base)
  assert_eq(release_target_name(legacy), "sm_120")
  assert_eq(legacy.max_activation_arena_bytes, 2147483648L)
  for (major, minor) in [(8, 0), (8, 6), (9, 0), (12, 0), (12, 1)] {
    let args = parse_qwen3_release_bind_arguments(
      base + ["--device-target", major.to_string(), minor.to_string()],
    )
    let target = release_target(args)
    assert_eq(target.compute_major(), major)
    assert_eq(target.compute_minor(), minor)
    assert_eq(release_target_name(args), "sm_\{major * 10 + minor}")
    assert_eq(release_profile(args).max_query_tokens(), 8192)
    assert_eq(load_release_projection_records(args).length(), 0)
  }
  let combined = parse_qwen3_release_bind_arguments(
    base +
    [
      "--device-target", "12", "1", "--compiler-register-limit", "255", "--projection-folds",
      "/folds", digest, "device-a",
    ],
  )
  assert_eq(release_target_name(combined), "sm_121")
  assert_eq(release_compiler(combined).max_registers(), 255)
  for
    suffix in [
      ["--device-target"],
      ["--device-target", "12"],
      ["--device-target", "7", "5"],
      ["--device-target", "12", "10"],
      ["--device-target", "12", "-1"],
      ["--device-target", "12", "1", "--device-target", "12", "1"],
    ] {
    try parse_qwen3_release_bind_arguments(base + suffix) catch {
      InvalidArguments => ()
      _ => fail("target rejection")
    } noraise {
      _ => fail("malformed target accepted")
    }
  }
}

///|
test "activation budget is explicit bounded and reaches memory planning" {
  let digest = "a".repeat(64)
  let base = [
    "binder", "/model", "config.json", digest, digest, "weights.numeric", digest,
    digest, "/compiled", "/toolchain", digest, "13", "0", "88", "8", "16384", "32",
    "8192", "16384", "/new-output",
  ]
  for gib in [1, 2, 4, 64] {
    let args = parse_qwen3_release_bind_arguments(
      base +
      [
        "--device-target",
        "12",
        "1",
        "--activation-arena-gib",
        gib.to_string(),
        "--compiler-register-limit",
        "255",
      ],
    )
    assert_eq(args.max_activation_arena_bytes, gib.to_int64() * 1073741824L)
    assert_eq(
      execution_limits(283, args).device_memory().max_arena_bytes(),
      args.max_activation_arena_bytes,
    )
    assert_eq(release_target_name(args), "sm_121")
    assert_eq(release_compiler(args).max_registers(), 255)
  }
  for
    suffix in [
      ["--activation-arena-gib"],
      ["--activation-arena-gib", "0"],
      ["--activation-arena-gib", "65"],
      ["--activation-arena-gib", "-1"],
      ["--activation-arena-gib", "4", "--activation-arena-gib", "4"],
    ] {
    try parse_qwen3_release_bind_arguments(base + suffix) catch {
      InvalidArguments => ()
      _ => fail("budget rejection")
    } noraise {
      _ => fail("malformed budget accepted")
    }
  }
}
