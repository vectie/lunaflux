///|
test "valid document accepts fields in any order" {
  let schema = @schema.validate_document("lunaflux.config.v1alpha1", [
    "device", "schema_version", "model", "service",
  ])
  assert_eq(schema.version(), V1Alpha1)
  inspect(schema.version().name(), content="lunaflux.config.v1alpha1")
}

///|
test "unknown and duplicate fields fail with bounded positions" {
  try
    @schema.validate_document("lunaflux.config.v1alpha1", [
      "schema_version", "service", "model", "surprise",
    ])
  catch {
    error => debug_inspect(error, content="UnknownField(index=3)")
  } noraise {
    _ => fail("expected unknown field rejection")
  }
  try
    @schema.validate_document("lunaflux.config.v1alpha1", [
      "schema_version", "service", "model", "model",
    ])
  catch {
    error =>
      debug_inspect(error, content="DuplicateField(index=3, first_index=2)")
  } noraise {
    _ => fail("expected duplicate field rejection")
  }
}

///|
test "unsupported version and missing boundary section fail closed" {
  try
    @schema.validate_document("lunaflux.config.v2", [
      "schema_version", "service", "model", "device",
    ])
  catch {
    error => debug_inspect(error, content="UnsupportedVersion")
  } noraise {
    _ => fail("expected unsupported version rejection")
  }
  try
    @schema.validate_document("lunaflux.config.v1alpha1", [
      "schema_version", "service", "model",
    ])
  catch {
    error => debug_inspect(error, content="MissingSection(Device)")
  } noraise {
    _ => fail("expected missing section rejection")
  }
}
