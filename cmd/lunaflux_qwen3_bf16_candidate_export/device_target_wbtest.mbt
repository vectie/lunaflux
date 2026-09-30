///|
test "candidate target is explicit and independent of compiler version" {
  let digest = "a".repeat(64)
  let base = [
    "export", "/model", "config.json", digest, digest, "weights.numeric", digest,
    digest, digest, "13", "1", "115", "8", "8192", "32", "256", "5120", "/new-output",
  ]
  for (major, minor) in [(8, 0), (12, 1)] {
    let args = base + ["--device-target", major.to_string(), minor.to_string()]
    let parsed = parse_qwen3_candidate_export_arguments(args)
    assert_eq(parsed.compute_major, major)
    assert_eq(parsed.compute_minor, minor)
    args.append(["--device-target", "12", "0"])
    try parse_qwen3_candidate_export_arguments(args) catch {
      InvalidArguments => ()
      _ => fail("unexpected target error")
    } noraise {
      _ => fail("duplicate device target accepted")
    }
  }
  for target in [["7", "5"], ["12", "10"]] {
    try
      parse_qwen3_candidate_export_arguments(
        base + ["--device-target"] + target,
      )
    catch {
      InvalidArguments => ()
      _ => fail("unexpected target error")
    } noraise {
      _ => fail("unsupported BF16 target accepted")
    }
  }
}
