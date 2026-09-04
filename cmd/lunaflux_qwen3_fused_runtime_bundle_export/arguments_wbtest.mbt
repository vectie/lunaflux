///|
fn runtime_bundle_test_arguments(attention_operation : String) -> Array[String] {
  [
    "bundle-export",
    "1".repeat(64),
    "2".repeat(64),
    "7",
    "256",
    "1",
    "1",
    "128",
    "512",
    "/tmp/residual.cubin",
    "2",
    "256",
    "56",
    "1",
    "128",
    "1024",
    "/tmp/ingress.cubin",
    "5",
    "32",
    "16",
    "1",
    "256",
    "45056",
    "/tmp/prefill.cubin",
    "5",
    "16",
    "16",
    "1",
    "256",
    "70032",
    "lunaflux_attention_prefill_tile_compiler_v1_c314",
    "/tmp/prefill-wide.cubin",
    "5",
    "32",
    "16",
    "8",
    "256",
    "31952",
    "/tmp/prefill-partial.cubin",
    "5",
    "32",
    "16",
    "1",
    "256",
    "0",
    "/tmp/prefill-merge.cubin",
    "5",
    "32",
    "16",
    "8",
    "256",
    "31952",
    "/tmp/prefill-partitioned8-partial.cubin",
    "5",
    "32",
    "16",
    "1",
    "256",
    "0",
    "/tmp/prefill-partitioned8-merge.cubin",
    attention_operation,
    "32",
    "8",
    "1",
    "64",
    "34076",
    "/tmp/attention.cubin",
    "32",
    "64",
    "32",
    "64",
    "/tmp/runtime.v3",
  ]
}

///|
test "runtime bundle arguments preserve frontier and phase launches" {
  let parsed = parse_arguments(runtime_bundle_test_arguments("5"))
  assert_eq(parsed.residual.first_operation, 7)
  assert_eq(parsed.ingress.grid_y, 56)
  assert_eq(parsed.attention_prefill_wide.shared_memory_bytes, 70032)
  assert_eq(
    parsed.attention_prefill_wide_symbol,
    "lunaflux_attention_prefill_tile_compiler_v1_c314",
  )
  assert_eq(parsed.attention_read.first_operation, 5)
  assert_eq(parsed.attention_read.shared_memory_bytes, 34076)
  assert_eq(parsed.attention_prefill_partitioned8_partial.grid_z, 8)
  assert_eq(parsed.attention_prefill_partitioned8_merge.grid_z, 1)
  assert_eq(parsed.prefill_query_tile_rows, 32)
  assert_eq(parsed.wide_query_tile_rows, 64)
  assert_eq(parsed.partitioned_query_tile_rows, 32)
  assert_eq(parsed.deep_partitioned_query_tile_rows, 64)
}

///|
test "runtime bundle arguments reject a detached attention operation" {
  try parse_arguments(runtime_bundle_test_arguments("6")) catch {
    InvalidArguments => ()
    _ => fail("unexpected invalid-argument rejection")
  } noraise {
    _ => fail("detached attention operation was accepted")
  }
}
