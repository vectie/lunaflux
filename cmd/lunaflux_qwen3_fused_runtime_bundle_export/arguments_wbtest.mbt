///|
test "runtime bundle arguments preserve three exact operation launches" {
  let parsed = parse_arguments([
    "bundle-export",
    "1".repeat(64),
    "2".repeat(64),
    "7",
    "256",
    "1",
    "128",
    "512",
    "/tmp/residual.cubin",
    "2",
    "256",
    "56",
    "128",
    "1024",
    "/tmp/ingress.cubin",
    "5",
    "256",
    "1",
    "1",
    "0",
    "/tmp/attention.cubin",
    "/tmp/runtime.v2",
  ])
  assert_eq(parsed.residual.first_operation, 7)
  assert_eq(parsed.ingress.grid_y, 56)
  assert_eq(parsed.attention_read.first_operation, 5)
  assert_eq(parsed.attention_read.shared_memory_bytes, 0)
}

///|
test "runtime bundle arguments reject a detached attention operation" {
  try
    parse_arguments([
      "bundle-export",
      "1".repeat(64),
      "2".repeat(64),
      "7",
      "256",
      "1",
      "128",
      "512",
      "/tmp/residual.cubin",
      "2",
      "256",
      "56",
      "128",
      "1024",
      "/tmp/ingress.cubin",
      "6",
      "256",
      "1",
      "1",
      "0",
      "/tmp/attention.cubin",
      "/tmp/runtime.v2",
    ])
  catch {
    InvalidArguments => ()
    _ => fail("unexpected invalid-argument rejection")
  } noraise {
    _ => fail("detached attention operation was accepted")
  }
}
