///|
test "binder consumes the same explicit saved projection tuning suffix" {
  let digest = "a".repeat(64)
  let arguments = [
    "binder", "/model", "config.json", digest, digest, "weights.numeric", digest,
    digest, "/compiled", "/toolchain", digest, "13", "1", "115", "8", "8192", "32",
    "256", "5120", "/new-output",
  ]
  let legacy = parse_qwen3_release_bind_arguments(arguments)
  assert_true(legacy.projection_tuning is None)
  assert_eq(load_release_projection_records(legacy).length(), 0)
  assert_true(legacy.projection_folds is None)
  assert_eq(legacy.compiler_register_limit, 128)
  arguments.append([
    "--projection-tuning", "/tuning", "projection.v1", digest, "device-a",
  ])
  assert_true(
    parse_qwen3_release_bind_arguments(arguments).projection_tuning is Some(_),
  )
  arguments.push("extra")
  try parse_qwen3_release_bind_arguments(arguments) catch {
    InvalidArguments => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("extra argument accepted")
  }
}

///|
test "binder regenerates measured projection envelope and rejects malformed fold suffix" {
  let digest = "a".repeat(64)
  let base = [
    "binder", "/model", "config.json", digest, digest, "weights.numeric", digest,
    digest, "/compiled", "/toolchain", digest, "13", "1", "115", "8", "8192", "32",
    "256", "5120", "/new-output",
  ]
  let parsed = parse_qwen3_release_bind_arguments(
    base +
    [
      "--compiler-register-limit", "255", "--projection-folds", "/folds", digest,
      "device-a",
    ],
  )
  assert_eq(parsed.projection_folds, Some(("/folds", digest, "device-a")))
  assert_eq(release_compiler(parsed).max_registers(), 255)
  for
    suffix in [
      ["--compiler-register-limit", "256"],
      ["--compiler-register-limit", "0"],
      ["--projection-folds", "relative", digest, "device-a"],
      ["--projection-folds", "/folds", "A".repeat(64), "device-a"],
      ["--projection-folds", "/folds", digest, ""],
      ["--projection-folds", "/folds", digest],
      ["--compiler-register-limit", "255", "--compiler-register-limit", "255"],
    ] {
    try parse_qwen3_release_bind_arguments(base + suffix) catch {
      InvalidArguments => ()
      _ => fail("wrong error")
    } noraise {
      _ => fail("malformed fold option accepted")
    }
  }
}

///|
test "release binder accepts only the exact bounded CLI shape" {
  let digest = "a".repeat(64)
  let parsed = parse_qwen3_release_bind_arguments([
    "binder", "/model", "config.json", digest, digest, "weights.numeric", digest,
    digest, "/compiled", "/toolchain", digest, "13", "1", "115", "8", "8192", "1",
    "1", "5120", "/new-output",
  ])
  assert_eq(parsed.compiler_major, 13U)
  assert_eq(parsed.compiler_minor, 1U)
  assert_eq(parsed.compiler_patch, 115U)
  try
    parse_qwen3_release_bind_arguments([
      "binder", "/model", "config.json", digest, digest, "weights.numeric", digest,
      digest, "/compiled", "/toolchain", digest, "0", "1", "115", "8", "8192", "1",
      "1", "5120", "/new-output",
    ])
  catch {
    InvalidArguments => ()
    error => {
      ignore(error)
      fail("unexpected release binder error")
    }
  } noraise {
    _ => fail("zero compiler major must be rejected")
  }
}

///|
test "release binder rejects malformed independent digests" {
  try
    parse_qwen3_release_bind_arguments([
      "binder",
      "/model",
      "config.json",
      "a".repeat(64),
      "a".repeat(64),
      "weights.numeric",
      "a".repeat(64),
      "a".repeat(64),
      "/compiled",
      "/toolchain",
      "A".repeat(64),
      "13",
      "1",
      "115",
      "8",
      "8192",
      "1",
      "1",
      "5120",
      "/new-output",
    ])
  catch {
    InvalidArguments => ()
    error => {
      ignore(error)
      fail("unexpected release binder error")
    }
  } noraise {
    _ => fail("uppercase digest must be rejected")
  }
}

///|
test "release binder output path admission is canonical and no-alias" {
  let (parent, basename) = validate_new_output_path("/tmp/new-release")
  assert_eq(parent, "/tmp")
  assert_eq(basename, "new-release")
  for hostile in ["relative", "/", "/tmp//x", "/tmp/../x", "/tmp/x/"] {
    try validate_new_output_path(hostile) catch {
      Publication => ()
      error => {
        ignore(error)
        fail("unexpected release binder error")
      }
    } noraise {
      _ => fail("hostile output path must be rejected")
    }
  }
}

///|
test "Qwen release geometry and producer ceilings admit the full graph" {
  let digest = "a".repeat(64)
  let arguments = parse_qwen3_release_bind_arguments([
    "binder", "/model", "config.json", digest, digest, "weights.numeric", digest,
    digest, "/compiled", "/toolchain", digest, "13", "1", "115", "8", "8192", "32",
    "32", "5120", "/new-output",
  ])
  let profile = release_profile(arguments)
  assert_eq(profile.max_query_rows(), 32)
  assert_eq(profile.max_query_tokens(), 32)
  assert_eq(profile.max_page_table_entries(), 5120)
  ignore(release_target(arguments))
  ignore(release_compiler(arguments))
  ignore(producer_limits())
}
