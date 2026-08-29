///|
fn fixture_content() -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
}

///|
fn fixture_plan() -> @plan.ModelPlan {
  let operation = try! @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(0),
    kind=Embedding,
    inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
    outputs=[@plan.ActivationRef::new(0)],
    shape=EmbeddingShape(vocabulary_size=128, width=16),
    required_capability=@plan.KernelCapabilityId::embedding_lookup(),
    workspace_bytes=0L,
  )
  try! @plan.ModelPlan::new_stateless(
    content_digest=fixture_content(),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=@plan.ShapeConstraints::new(
      max_batch_rows=1,
      max_sequence_tokens=4,
      vocabulary_size=128,
      hidden_size=16,
      intermediate_size=32,
      attention_head_count=4,
      kv_head_count=2,
      head_dimension=4,
    ),
    kv_cache_geometry=@plan.KvCacheGeometry::new(
      layer_count=1,
      kv_head_count=2,
      head_dimension=4,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    ),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations=[operation],
    final_output=@plan.ActivationRef::new(0),
    numeric_schema=@plan_test_fixture.synthetic_bf16_numeric_schema([operation]),
  )
}

///|
fn fixture_module_bytes() -> Bytes {
  b"multi-function-cubin"
}

///|
fn digest_bytes(bytes : BytesView) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn fixture_digest() -> @catalog.AotArtifactDigest {
  let result = try! @catalog.AotArtifactDigest::from_sha256(
    digest_bytes(fixture_module_bytes()),
  )
  result
}

///|
fn fixture_family(
  digest : @catalog.AotArtifactDigest,
) -> @catalog.AotKernelFamily {
  let family_id = try! @catalog.AotKernelFamilyId::from_stable_int(1)
  @catalog.AotKernelFamily::new(module_digest=digest, family_id~)
}

///|
fn fixture_target() -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
}

///|
fn fixture_catalog(
  digest : @catalog.AotArtifactDigest,
) -> @catalog.ResolvedKernelCatalog {
  let entry = try! @catalog.CatalogEntry::new(
    semantic_version=@catalog.KernelSemanticVersion::v1(),
    operation_kind=Embedding,
    operation_shape=EmbeddingShape(vocabulary_size=128, width=16),
    capability=@plan.KernelCapabilityId::embedding_lookup(),
    compute_major=8,
    compute_minor=0,
    implementation=ContentAddressedAot(fixture_family(digest)),
    workspace_bytes=0L,
    workspace_alignment=1L,
  )
  try! @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v1(), entries=[
    entry,
  ]).resolve(fixture_plan(), fixture_target())
}

///|
fn fixture_profile(
  stable_id : Int,
  rows : Int,
) -> @launch_contract.KernelProfile {
  try! @launch_contract.KernelProfile::full_prefill(
    id=@launch_contract.KernelProfileId::from_stable_int(stable_id),
    batch_rows=1,
    sequence_tokens=rows,
    token_rows=rows.to_int64(),
  )
}

///|
fn fixture_entry(
  digest : @catalog.AotArtifactDigest,
  stable_id : Int,
) -> @catalog.AotKernelEntryPoint {
  let entry_point_id = try! @catalog.AotKernelEntryPointId::from_stable_int(
    stable_id,
  )
  @catalog.AotKernelEntryPoint::new(
    family=fixture_family(digest),
    entry_point_id~,
  )
}

///|
fn fixture_contract_input(
  digest : @catalog.AotArtifactDigest,
  profile_id : Int,
  rows : Int,
  entry_id : Int,
) -> @launch_contract.AotLaunchContractInput {
  let resolved_profile = try! @launch_contract.KernelProfileId::from_stable_int(
    profile_id,
  )
  let operation_id = try! @plan.OperationId::new(0)
  let dimensions = try! @launch_contract.AotLaunchDimensions::new(
    grid_x=rows,
    grid_y=1,
    grid_z=1,
    block_x=32,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  let tensor = try! @plan.TensorRef::new(0)
  let activation = try! @plan.ActivationRef::new(0)
  let external_operand = try! @launch_contract.AotOperand::new(
    role=ExternalInput(TokenIds),
    byte_count=rows.to_int64() * 4L,
    alignment=4L,
  )
  let weight_operand = try! @launch_contract.AotOperand::new(
    role=WeightInput(tensor),
    byte_count=4096L,
    alignment=16L,
  )
  let output_operand = try! @launch_contract.AotOperand::new(
    role=ActivationOutput(activation),
    byte_count=rows.to_int64() * 32L,
    alignment=16L,
  )
  try! @launch_contract.AotLaunchContractInput::new(
    profile_id=resolved_profile,
    operation_id~,
    entry_point=fixture_entry(digest, entry_id),
    dimensions~,
    operands=[external_operand, weight_operand, output_operand],
  )
}

///|
fn fixture_contracts(
  digest? : @catalog.AotArtifactDigest = fixture_digest(),
) -> @launch_contract.LaunchContractSet {
  try! @launch_contract.admit(
    fixture_plan(),
    fixture_catalog(digest),
    [fixture_profile(1, 2), fixture_profile(2, 4)],
    [
      fixture_contract_input(digest, 1, 2, 11),
      fixture_contract_input(digest, 2, 4, 12),
    ],
    @launch_contract.LaunchContractLimits::new(
      max_profiles=2,
      max_contracts=2,
      max_operands_per_contract=4,
    ),
  )
}

///|
fn fixture_file_limits(
  max_manifest_bytes? : Int64 = 16384L,
  max_path_bytes? : Int = 128,
  max_module_bytes? : Int64 = 1024L,
  max_total_module_bytes? : Int64 = 2048L,
  max_modules? : Int = 4,
  max_entry_points? : Int = 8,
) -> @artifact_file.KernelArtifactFileLimits {
  try! @artifact_file.KernelArtifactFileLimits::new(
    max_manifest_bytes~,
    max_json_depth=8,
    max_path_bytes~,
    artifact=@artifact.KernelArtifactLimits::new(
      max_modules~,
      max_entry_points~,
      max_module_bytes~,
      max_total_module_bytes~,
      max_function_symbol_bytes=64,
    ),
  )
}

///|
fn entry_json(family : Int, entry : Int, symbol : String) -> String {
  "{\"family_id\":\{family},\"entry_point_id\":\{entry},\"symbol\":\"\{symbol}\"}"
}

///|
fn module_json(
  digest : String,
  path : String,
  entries : ArrayView[String],
) -> String {
  "{\"sha256\":\"\{digest}\",\"path\":\"\{path}\",\"entry_points\":[\{entries.join(",")}]}"
}

///|
fn manifest_json(
  modules : ArrayView[String],
  extra_root? : String = "",
  catalog_version? : Int = 1,
  plan? : @plan.ModelPlan = fixture_plan(),
) -> Bytes {
  let identity = plan.identity()
  @utf8.encode(
    "{\"schema_version\":1,\"model_identity\":{\"content_sha256\":\"\{identity.content().as_hex()}\",\"plan_sha256\":\"\{identity.plan().as_hex()}\"},\"device_target\":{\"compute_major\":8,\"compute_minor\":0,\"supports_bf16\":true,\"supports_cublas_lt\":false},\"catalog_version\":\{catalog_version},\"modules\":[\{modules.join(",")}]\{extra_root}}",
  )
}

///|
fn valid_manifest(
  path? : String = "kernel.cubin",
  entry_ids? : Array[Int] = [11, 12],
) -> Bytes {
  manifest_json([
    module_json(fixture_digest().sha256(), path, [
      entry_json(1, entry_ids[0], "embedding_prefill_2"),
      entry_json(1, entry_ids[1], "embedding_prefill_4"),
    ]),
  ])
}

///|
fn expected_manifest(bytes : BytesView) -> @artifact_file.KernelManifestDigest {
  let result = try! @artifact_file.KernelManifestDigest::from_sha256(
    digest_bytes(bytes),
  )
  result
}

///|
async fn write_file(root : String, name : String, bytes : Bytes) -> Unit {
  @fs.write_file("\{root}/\{name}", bytes, create_mode=CreateNew)
}

///|
async fn replace_file(root : String, name : String, bytes : Bytes) -> Unit {
  @fs.write_file("\{root}/\{name}", bytes, create_mode=CreateOrTruncate)
}
