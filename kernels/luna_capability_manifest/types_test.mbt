///|
test "capability architecture ranges reject overflow-prone ordinals" {
  try
    @luna_capability_manifest.LunaGpuArchitectureRange::new(
      minimum_major=100,
      minimum_minor=0,
      maximum_major=100,
      maximum_minor=0,
    )
  catch {
    Invalid(Architecture) => ()
    _ => fail("unexpected architecture-range failure")
  } noraise {
    _ => fail("overflow-prone architecture range was admitted")
  }
}

///|
test "non-capture graph metadata cannot reserve capture memory" {
  let shape = @luna_capability_manifest.LunaGraphShapeClass::from_execution_shape(
    @device_profile.ExactExecutionShape::full_recompute(
      batch_rows=1,
      sequence_tokens=1,
    ),
  )
  try
    @luna_capability_manifest.LunaGraphCapabilityMetadata::new(
      shape_class=shape,
      capture_safe=false,
      graph_memory_bytes=256L,
      graph_memory_alignment=256L,
      eager_fallback=@luna_capability_manifest.LunaValidatedEagerFallbackDigest::from_sha256(
        "1111111111111111111111111111111111111111111111111111111111111111",
      ),
    )
  catch {
    Invalid(GraphMemory) => ()
    _ => fail("unexpected graph-memory failure")
  } noraise {
    _ => fail("non-capture graph memory was admitted")
  }
}

///|
test "capture-safe graph metadata requires a positive declared bound" {
  let shape = @luna_capability_manifest.LunaGraphShapeClass::from_execution_shape(
    @device_profile.ExactExecutionShape::full_recompute(
      batch_rows=1,
      sequence_tokens=1,
    ),
  )
  try
    @luna_capability_manifest.LunaGraphCapabilityMetadata::new(
      shape_class=shape,
      capture_safe=true,
      graph_memory_bytes=0L,
      graph_memory_alignment=256L,
      eager_fallback=@luna_capability_manifest.LunaValidatedEagerFallbackDigest::from_sha256(
        "1111111111111111111111111111111111111111111111111111111111111111",
      ),
    )
  catch {
    Invalid(GraphMemory) => ()
    _ => fail("unexpected zero capture-bound failure")
  } noraise {
    _ => fail("capture-safe metadata fabricated a zero-byte bound")
  }
}
