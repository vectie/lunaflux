///|
test "release binder accepts only the exact bounded CLI shape" {
  let digest = "a".repeat(64)
  let parsed = parse_qwen3_release_bind_arguments([
    "binder", "/model", "config.json", digest, digest, "weights.numeric", digest,
    digest, "/compiled", "/toolchain", digest, "13", "1", "115", "8", "8192", "1",
    "5120", "/new-output",
  ])
  assert_eq(parsed.compiler_major, 13U)
  assert_eq(parsed.compiler_minor, 1U)
  assert_eq(parsed.compiler_patch, 115U)
  try
    parse_qwen3_release_bind_arguments([
      "binder", "/model", "config.json", digest, digest, "weights.numeric", digest,
      digest, "/compiled", "/toolchain", digest, "0", "1", "115", "8", "8192", "1",
      "5120", "/new-output",
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
    digest, "/compiled", "/toolchain", digest, "13", "1", "115", "8", "8192", "1",
    "5120", "/new-output",
  ])
  let profile = release_profile(arguments)
  assert_eq(profile.max_query_tokens(), 1)
  assert_eq(profile.max_page_table_entries(), 5120)
  ignore(release_target())
  ignore(release_compiler(arguments))
  ignore(producer_limits())
}
