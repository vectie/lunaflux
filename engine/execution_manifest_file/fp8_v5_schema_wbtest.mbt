///|
fn fp8_v5_numeric() -> @device.DeviceNumericCapability raise {
  @device.admit_finite_fp8_e4m3_w8a8(
    @device.DeviceCapability::from_observed_facts(
      compute_major=8,
      compute_minor=9,
      total_memory_bytes=8589934592L,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
  )
}

///|
test "schema v5 target gate admits exact sm120" {
  let blackwell = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  let unsupported = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=1,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  assert_true(fp8_v5_target_supported(blackwell))
  assert_false(fp8_v5_target_supported(unsupported))
}

///|
fn fp8_v5_parser_fixture() -> (
  String,
  @plan.ModelPlan,
  @device.DeviceNumericCapability,
  @catalog.DeviceTarget,
) raise {
  let plan = @llama.build(fixture_metadata('a'))
  let numeric = fp8_v5_numeric()
  let target = @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=9,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let authorization = "\"numeric_authorization\":{" +
    "\"numeric_schema_sha256\":\"" +
    plan.numeric_binding().digest().as_hex() +
    "\",\"device_numeric_sha256\":\"" +
    numeric.digest().as_hex() +
    "\",\"weight_artifact_sha256\":\"" +
    "1111111111111111111111111111111111111111111111111111111111111111" +
    "\",\"fp8_kernel_manifest_sha256\":\"" +
    "2222222222222222222222222222222222222222222222222222222222222222" +
    "\",\"fp8_runtime_recipe_v2_sha256\":\"" +
    "3333333333333333333333333333333333333333333333333333333333333333" +
    "\",\"fp8_reusable_recipe_v3_sha256\":\"" +
    "4444444444444444444444444444444444444444444444444444444444444444" +
    "\",\"workspace_storage_sha256\":\"" +
    "5555555555555555555555555555555555555555555555555555555555555555" +
    "\"},\"fp8_release\":{" +
    "\"manifest_locator\":\"fp8-release.bin\",\"operations\":[{" +
    "\"operation_id\":0,\"module_locator\":\"kernel.cubin\"," +
    "\"source_digest_sha256\":\"6666666666666666666666666666666666666666666666666666666666666666\"," +
    "\"recipe_digest_sha256\":\"7777777777777777777777777777777777777777777777777777777777777777\"}]},"
  let base = fixture_manifest(plan, target)
  let text = base
    .replace(
      old="\"schema_version\":2,",
      new="\"schema_version\":5," + authorization,
    )
    .replace(old="\"catalog_version\":3", new="\"catalog_version\":4")
  (text, plan, numeric, target)
}

///|
test "schema v5 parser binds every FP8 reconstruction digest" {
  let (text, plan, numeric, target) = fp8_v5_parser_fixture()
  let claims = parse_manifest_v5(
    @utf8.encode(text),
    plan,
    numeric,
    target,
    fixture_limits(plan.operations().length()),
  )
  assert_eq(
    claims.numeric.reusable_recipe_v3_digest,
    "4444444444444444444444444444444444444444444444444444444444444444",
  )
  assert_eq(claims.operations.length(), plan.operations().length())
}

///|
test "schema v5 rejects digest substitution duplicate and foreign version" {
  let (text, plan, numeric, target) = fp8_v5_parser_fixture()
  for
    hostile in [
      text.replace(
        old=plan.numeric_binding().digest().as_hex(),
        new="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      ),
      text.replace(old="\"schema_version\":5", new="\"schema_version\":6"),
      text.replace(
        old="\"workspace_storage_sha256\":",
        new="\"workspace_storage_sha256\":\"bad\",\"workspace_storage_sha256\":",
      ),
    ] {
    try
      parse_manifest_v5(
        @utf8.encode(hostile),
        plan,
        numeric,
        target,
        fixture_limits(plan.operations().length()),
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("hostile FP8 schema-v5 manifest was admitted")
    }
  }
}
