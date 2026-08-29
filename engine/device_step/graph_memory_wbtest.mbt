///|
fn graph_memory_metadata(
  capture_safe : Bool,
  bytes : Int64,
  alignment? : Int64 = 256L,
) -> @luna_capability_manifest.LunaGraphCapabilityMetadata raise {
  @luna_capability_manifest.LunaGraphCapabilityMetadata::new(
    shape_class=@luna_capability_manifest.LunaGraphShapeClass::from_execution_shape(
      @device_profile.ExactExecutionShape::full_recompute(
        batch_rows=1,
        sequence_tokens=1,
      ),
    ),
    capture_safe~,
    graph_memory_bytes=bytes,
    graph_memory_alignment=alignment,
    eager_fallback=@luna_capability_manifest.LunaValidatedEagerFallbackDigest::from_sha256(
      "1111111111111111111111111111111111111111111111111111111111111111",
    ),
  )
}

///|
test "blueprint graph memory evidence distinguishes absence unsupported and bound" {
  let absent = derive_graph_memory_evidence(None)
  assert_eq(absent.availability(), GraphMemoryAbsent)
  assert_eq(absent.declared_upper_bound_bytes(), None)
  assert_eq(absent.alignment_bytes(), None)

  let unsupported = derive_graph_memory_evidence(
    Some(graph_memory_metadata(false, 0L)),
  )
  assert_eq(unsupported.availability(), GraphMemoryUnsupported)
  assert_eq(unsupported.declared_upper_bound_bytes(), None)
  assert_eq(unsupported.alignment_bytes(), None)

  let bounded = derive_graph_memory_evidence(
    Some(graph_memory_metadata(true, 4096L)),
  )
  assert_eq(bounded.availability(), GraphMemoryBounded)
  assert_eq(bounded.declared_upper_bound_bytes(), Some(4096L))
  assert_eq(bounded.alignment_bytes(), Some(256L))
}

///|
test "capture-safe blueprint metadata cannot use zero as missing accounting" {
  try graph_memory_metadata(true, 0L) catch {
    @luna_capability_manifest.Invalid(GraphMemory) => ()
    _ => fail("unexpected graph-memory metadata failure")
  } noraise {
    _ => fail("zero capture bound was admitted")
  }
}
