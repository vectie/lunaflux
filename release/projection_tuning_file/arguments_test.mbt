///|
test "optional tuning input is explicit and bounded" {
  assert_true(
    @projection_tuning_file.parse_projection_tuning_suffix([]) is None,
  )
  assert_true(
    @projection_tuning_file.parse_projection_tuning_suffix([
      "--projection-tuning",
      "/tmp/tuning",
      "projection.v1",
      "a".repeat(64),
      "device-a",
    ])
    is Some(_),
  )
  for
    suffix in [
      ["--projection-tuning"],
      ["--other", "/tmp/tuning", "projection.v1", "a".repeat(64), "device-a"],
      [
        "--projection-tuning",
        "/tmp/tuning",
        "../projection.v1",
        "a".repeat(64),
        "device-a",
      ],
      [
        "--projection-tuning",
        "/tmp/tuning",
        "projection.v1",
        "A".repeat(64),
        "device-a",
      ],
      [
        "--projection-tuning",
        "/tmp/tuning",
        "projection.v1",
        "a".repeat(64),
        "device\na",
      ],
    ] {
    try @projection_tuning_file.parse_projection_tuning_suffix(suffix) catch {
      InvalidTuningInput => ()
      _ => fail("unexpected error")
    } noraise {
      _ => fail("malformed suffix accepted")
    }
  }
}
