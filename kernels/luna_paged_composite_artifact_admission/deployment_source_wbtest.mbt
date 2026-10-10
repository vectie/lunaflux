///|
test "incomplete fused production source fails structurally with an unrelated label" {
  let source = @utf8.encode(
    "schema=lunaflux-fused-production-runtime-source.v1\nfused_source_hex=00\n",
  )
  try
    ignore(admit_digest_pinned_fused_production_runtime(source, "0".repeat(64)))
  catch {
    InvalidPagedCompositeArtifact(Module) => ()
    _ => fail("fused production source reported an unrelated parse failure")
  } noraise {
    _ => fail("incomplete fused production source was admitted")
  }
}

///|
test "self-pinned incomplete fused production source grants no aggregate" {
  let source = @utf8.encode(
    "schema=lunaflux-fused-production-runtime-source.v1\nfused_source_hex=00\n",
  )
  let digest = @crypto.bytes_to_hex_string(@crypto.sha256(source))
  try
    ignore(admit_digest_pinned_fused_production_runtime(source, digest))
  catch {
    InvalidPagedCompositeArtifact(Module) => ()
    _ =>
      fail("incomplete fused production source reported an unrelated failure")
  } noraise {
    _ => fail("incomplete fused production source granted aggregate authority")
  }
}
