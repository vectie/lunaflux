///|
test "candidate export explicitly selects a step-prepared rotary module" {
  let digest = "a".repeat(64)
  let raw = [
    "export", "/model", "config.json", digest, digest, "weights.numeric", digest,
    digest, digest, "13", "1", "115", "8", "8192", "32", "256", "5120", "/new-output",
  ]
  assert_false(parse_qwen3_candidate_export_arguments(raw).step_rotary_cache)
  raw.push("--step-rotary-cache")
  assert_true(parse_qwen3_candidate_export_arguments(raw).step_rotary_cache)
  raw.push("--step-rotary-cache")
  try parse_qwen3_candidate_export_arguments(raw) catch {
    InvalidArguments => ()
    _ => fail("unexpected duplicate cache option error")
  } noraise {
    _ => fail("duplicate cache selector accepted")
  }
}
