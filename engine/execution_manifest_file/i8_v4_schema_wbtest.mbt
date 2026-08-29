///|
fn i8_v4_fixture_device() -> @device.DeviceNumericCapability {
  let observed = try! @device.DeviceCapability::from_observed_facts(
    compute_major=8,
    compute_minor=9,
    total_memory_bytes=8L * 1024L * 1024L * 1024L,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  try! @device.admit_symmetric_i8_weight_only_per_output_channel_v1(observed)
}

///|
fn i8_v4_fixture_manifest(
  plan : @plan.ModelPlan,
  target : @catalog.DeviceTarget,
  device_numeric : @device.DeviceNumericCapability,
) -> String {
  fixture_manifest(plan, target)
  .replace(old="\"schema_version\":2", new="\"schema_version\":4")
  .replace(old="\"catalog_version\":3", new="\"catalog_version\":4")
  .replace(
    old="\"device_target\":",
    new="\"numeric_authorization\":{" +
      "\"numeric_schema_sha256\":\"" +
      plan.numeric_binding().digest().as_hex() +
      "\",\"i8_admission_sha256\":\"" +
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" +
      "\",\"device_numeric_sha256\":\"" +
      device_numeric.digest().as_hex() +
      "\",\"weight_artifact_sha256\":\"" +
      "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" +
      "\"},\"device_target\":",
  )
}

///|
test "schema-v4 parser binds exact numeric authorities and excludes legacy" {
  let plan = try! @llama.build_paged(fixture_metadata('0'))
  let target = fixture_target()
  let device_numeric = i8_v4_fixture_device()
  let limits = fixture_limits(plan.operations().length())
  let valid = i8_v4_fixture_manifest(plan, target, device_numeric)
  let claims = parse_manifest_v4(
    @utf8.encode(valid),
    plan,
    device_numeric,
    target,
    limits,
  )
  assert_eq(
    claims.numeric.weight_artifact_digest.as_hex(),
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  )
  try parse_manifest(@utf8.encode(valid), plan, target, limits) catch {
    InvalidField(area=Root, issue=UnsupportedValue, index=_) => ()
    _ => fail("unexpected legacy parser rejection")
  } noraise {
    _ => fail("schema v4 entered the legacy parser")
  }
  let legacy = valid.replace(
    old="\"schema_version\":4",
    new="\"schema_version\":3",
  )
  try
    parse_manifest_v4(
      @utf8.encode(legacy),
      plan,
      device_numeric,
      target,
      limits,
    )
  catch {
    InvalidField(area=Root, issue=UnsupportedValue, index=_) => ()
    _ => fail("unexpected schema-v3 rejection")
  } noraise {
    _ => fail("schema v3 entered the v4 parser")
  }
}

///|
test "schema-v4 parser rejects numeric substitution duplicate and op reorder" {
  let plan = try! @llama.build_paged(fixture_metadata('0'))
  let target = fixture_target()
  let device_numeric = i8_v4_fixture_device()
  let limits = fixture_limits(plan.operations().length())
  let valid = i8_v4_fixture_manifest(plan, target, device_numeric)
  let numeric_substitution = valid.replace(
    old=plan.numeric_binding().digest().as_hex(),
    new="cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  )
  try
    parse_manifest_v4(
      @utf8.encode(numeric_substitution),
      plan,
      device_numeric,
      target,
      limits,
    )
  catch {
    IdentityMismatch(NumericSchemaDigest) => ()
    _ => fail("unexpected numeric-schema substitution rejection")
  } noraise {
    _ => fail("foreign numeric schema was admitted")
  }
  let duplicate = valid.replace(
    old="\"weight_artifact_sha256\":",
    new="\"weight_artifact_sha256\":" +
      "\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\"," +
      "\"weight_artifact_sha256\":",
  )
  try
    parse_manifest_v4(
      @utf8.encode(duplicate),
      plan,
      device_numeric,
      target,
      limits,
    )
  catch {
    DuplicateField(_) => ()
    _ => fail("unexpected duplicate numeric-authority rejection")
  } noraise {
    _ => fail("duplicate numeric authority was admitted")
  }
  let reordered = valid.replace(
    old="\"operation_id\":0",
    new="\"operation_id\":1",
  )
  try
    parse_manifest_v4(
      @utf8.encode(reordered),
      plan,
      device_numeric,
      target,
      limits,
    )
  catch {
    InvalidField(area=Operation, issue=InvalidOrder, index=0) => ()
    _ => fail("unexpected v4 operation-order rejection")
  } noraise {
    _ => fail("reordered v4 operation was admitted")
  }
}
