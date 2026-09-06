///|
test "offline file loader closes roots and consumes only the exact snapshot scope" {
  let root = @env.current_dir().unwrap() +
    "/release/projection_tuning_file/fixtures"
  let digest = "ebf9b3323c748c34ee83d38bd2b170334e733f7194c251ebed110e3b5dfef78b"
  let input = parse_projection_tuning_suffix([
    "--projection-tuning", root, "synthetic.v1", digest, "device-a",
  ])
  let backend = @strategy.ProjectionBackendId::new("test-backend")
  let architecture = @strategy.ProjectionTargetId::new("test-architecture")
  let records = load_projection_tuning_records(
    input,
    backend~,
    architecture~,
    toolchain_sha256="a".repeat(64),
  )
  assert_eq(records.length(), 1)
  assert_eq(records[0].median_latency_ns(), 12000L)
  for
    bad in [
      parse_projection_tuning_suffix([
        "--projection-tuning", root, "synthetic.v1", digest, "device-b",
      ]),
      parse_projection_tuning_suffix([
        "--projection-tuning",
        root,
        "synthetic.v1",
        "0".repeat(64),
        "device-a",
      ]),
      parse_projection_tuning_suffix([
        "--projection-tuning", root, "absent.v1", digest, "device-a",
      ]),
    ] {
    try
      load_projection_tuning_records(
        bad,
        backend~,
        architecture~,
        toolchain_sha256="a".repeat(64),
      )
    catch {
      TuningReadFailure => ()
      _ => fail("unexpected error")
    } noraise {
      _ => fail("mismatched file accepted")
    }
  }
  try
    load_projection_tuning_records(
      input,
      backend~,
      architecture~,
      toolchain_sha256="b".repeat(64),
    )
  catch {
    TuningReadFailure => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("foreign compiler accepted")
  }
}
