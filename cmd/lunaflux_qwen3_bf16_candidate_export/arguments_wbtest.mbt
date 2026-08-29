///|
test "offline export arguments are exact and positive" {
  let parsed = parse_qwen3_candidate_export_arguments([
    "lunaflux-qwen3-bf16-candidate-export",
    "/approved/model",
    "config.json",
    "1".repeat(64),
    "2".repeat(64),
    "weights.numeric",
    "4".repeat(64),
    "5".repeat(64),
    "3".repeat(64),
    "13",
    "1",
    "115",
    "8",
    "1",
    "1",
    "1",
    "1",
    "/new/candidates",
  ])
  assert_eq(parsed.model_root, "/approved/model")
  assert_eq(parsed.compiler_major, 13U)
  assert_eq(parsed.compiler_minor, 1U)
  assert_eq(parsed.compiler_patch, 115U)
  assert_eq(parsed.tokens_per_page, 8)
  assert_eq(parsed.total_page_count, 1)
  assert_eq(parsed.max_query_rows, 1)
  assert_eq(parsed.max_query_tokens, 1)
  assert_eq(parsed.max_page_table_entries, 1)
  assert_eq(parsed.absolute_new_output, "/new/candidates")
}

///|
test "offline export arguments retain c1 and admit explicit c32 geometry" {
  let parsed = parse_qwen3_candidate_export_arguments([
    "lunaflux-qwen3-bf16-candidate-export",
    "/approved/model",
    "config.json",
    "1".repeat(64),
    "2".repeat(64),
    "weights.numeric",
    "4".repeat(64),
    "5".repeat(64),
    "3".repeat(64),
    "13",
    "1",
    "115",
    "8",
    "8192",
    "32",
    "256",
    "5120",
    "/new/candidates",
  ])
  assert_eq(parsed.max_query_rows, 32)
  assert_eq(parsed.max_query_tokens, 256)
  assert_eq(parsed.max_page_table_entries, 5120)
}

///|
test "offline export arguments reject omission text zero and overflow" {
  let invalid = [
    ["lunaflux-qwen3-bf16-candidate-export"],
    [
      "lunaflux-qwen3-bf16-candidate-export",
      "/approved/model",
      "config.json",
      "1".repeat(64),
      "2".repeat(64),
      "weights.numeric",
      "4".repeat(64),
      "5".repeat(64),
      "3".repeat(64),
      "13",
      "1",
      "115",
      "x",
      "1",
      "1",
      "1",
      "1",
      "/new/candidates",
    ],
    [
      "lunaflux-qwen3-bf16-candidate-export",
      "/approved/model",
      "config.json",
      "1".repeat(64),
      "2".repeat(64),
      "weights.numeric",
      "4".repeat(64),
      "5".repeat(64),
      "3".repeat(64),
      "13",
      "1",
      "115",
      "8",
      "0",
      "1",
      "1",
      "1",
      "/new/candidates",
    ],
    [
      "lunaflux-qwen3-bf16-candidate-export",
      "/approved/model",
      "config.json",
      "1".repeat(64),
      "2".repeat(64),
      "weights.numeric",
      "4".repeat(64),
      "5".repeat(64),
      "3".repeat(64),
      "13",
      "1",
      "115",
      "8",
      "9999999999",
      "1",
      "1",
      "1",
      "/new/candidates",
    ],
    [
      "lunaflux-qwen3-bf16-candidate-export",
      "/approved/model",
      "config.json",
      "1".repeat(64),
      "2".repeat(64),
      "weights.numeric",
      "4".repeat(64),
      "5".repeat(64),
      "3".repeat(64),
      "13",
      "1",
      "1000000",
      "8",
      "1",
      "1",
      "1",
      "1",
      "/new/candidates",
    ],
    [
      "lunaflux-qwen3-bf16-candidate-export",
      "/approved/model",
      "config.json",
      "A".repeat(64),
      "2".repeat(64),
      "weights.numeric",
      "4".repeat(64),
      "5".repeat(64),
      "3".repeat(64),
      "13",
      "1",
      "115",
      "8",
      "1",
      "1",
      "1",
      "1",
      "/new/candidates",
    ],
  ]
  for arguments in invalid {
    try parse_qwen3_candidate_export_arguments(arguments) catch {
      InvalidArguments => ()
      _ => fail("unexpected invalid-argument rejection")
    } noraise {
      _ => fail("expected invalid-argument rejection")
    }
  }
}
