///|
fn fused_qkv_readonly_bundle_identity(
  bundle : @luna_kernel_bundle.LunaKernelBundleInput,
  operation : @plan.OperationId,
) -> (String, String) raise FusedGroupedOperationError {
  let mut found : (String, String)? = None
  for compiled in bundle.operations() {
    if compiled.operation_id() == operation {
      if found is Some(_) {
        raise MismatchedFusedGroupedOperation(StandaloneFallback)
      }
      found = Some(
        (compiled.source_digest_sha256(), compiled.recipe_digest_sha256()),
      )
    }
  }
  guard found is Some(identity) else {
    raise MismatchedFusedGroupedOperation(StandaloneFallback)
  }
  identity
}

///|
fn fused_qkv_readonly_operation_id(
  index : Int,
) -> @plan.OperationId raise FusedGroupedOperationError {
  @plan.OperationId::new(index) catch {
    _ => raise MismatchedFusedGroupedOperation(OperationOrder)
  }
}

///|
fn require_fused_qkv_readonly_attention_fallback_join(
  standalone : (String, String),
  readonly_fallback : (String, String),
  fused_kv_write : (String, String),
  qualified_kv_write : (String, String),
) -> Unit raise FusedGroupedOperationError {
  if standalone != readonly_fallback ||
    standalone != fused_kv_write ||
    standalone != qualified_kv_write {
    raise MismatchedFusedGroupedOperation(StandaloneFallback)
  }
}

///|
fn require_fused_qkv_readonly_plan_span(
  plan : @plan.ModelPlan,
  qkv_index : Int,
  rope_index : Int,
  attention_index : Int,
) -> (@plan.ActivationRef, Int) raise FusedGroupedOperationError {
  guard qkv_index + 1 == rope_index &&
    rope_index + 1 == attention_index &&
    plan.operations().get(qkv_index) is Some(qkv) &&
    plan.operations().get(rope_index) is Some(rope) &&
    plan.operations().get(attention_index) is Some(attention) &&
    qkv.kind() == QkvProjection &&
    rope.kind() == RotaryEmbedding &&
    attention.kind() == CausalAttention &&
    qkv.outputs().get(0) is Some(qkv_output) &&
    rope.inputs().get(0) is Some(Activation(rope_input)) &&
    rope.outputs().get(0) is Some(positioned_qkv) &&
    attention.inputs().get(0) is Some(Activation(attention_input)) &&
    rope_input == qkv_output &&
    attention_input == positioned_qkv &&
    qkv.shape() is ProjectionShape(output_width~, ..) &&
    rope.shape() is RotaryShape(query_heads~, kv_heads~, head_dimension~, ..) &&
    attention.shape()
    is AttentionShape(
      query_heads=attention_query_heads,
      kv_heads=attention_kv_heads,
      head_dimension=attention_head_dimension
    ) &&
    query_heads == attention_query_heads &&
    kv_heads == attention_kv_heads &&
    head_dimension == attention_head_dimension &&
    output_width == (query_heads + 2 * kv_heads) * head_dimension else {
    raise MismatchedFusedGroupedOperation(OperationOrder)
  }
  (positioned_qkv, output_width)
}

///|
fn require_fused_qkv_readonly_fallbacks(
  standalone : @luna_kernel_bundle.LunaKernelBundleInput,
  qkv_module : @luna_fused_artifact_admission.ApprovedFusedKernelModule,
  readonly_identity : @luna_paged_composite_artifact_admission.ApprovedPagedAttentionReadOnlyRuntimeIdentity,
  kv_write : @luna_paged_composite_artifact_admission.QualifiedPagedKvWriteFallbackIdentity,
  qkv_index : Int,
  rope_index : Int,
  attention_index : Int,
) -> Unit raise FusedGroupedOperationError {
  let fallbacks = qkv_module.fallbacks()
  if fallbacks.length() != 3 ||
    fallbacks[0].role() != QkvProjectionFallback ||
    fallbacks[1].role() != PositionedRopeFallback ||
    fallbacks[2].role() != PagedKvWriteFallback ||
    fallbacks[0].operation_index() != qkv_index ||
    fallbacks[1].operation_index() != rope_index ||
    fallbacks[2].operation_index() != attention_index {
    raise MismatchedFusedGroupedOperation(StandaloneFallback)
  }
  let qkv = fused_qkv_readonly_bundle_identity(
    standalone,
    fused_qkv_readonly_operation_id(qkv_index),
  )
  let rope = fused_qkv_readonly_bundle_identity(
    standalone,
    fused_qkv_readonly_operation_id(rope_index),
  )
  let attention = fused_qkv_readonly_bundle_identity(
    standalone,
    fused_qkv_readonly_operation_id(attention_index),
  )
  if qkv.0 != fallbacks[0].source_digest().as_hex() ||
    qkv.1 != fallbacks[0].recipe_digest().as_hex() ||
    rope.0 != fallbacks[1].source_digest().as_hex() ||
    rope.1 != fallbacks[1].recipe_digest().as_hex() {
    raise MismatchedFusedGroupedOperation(StandaloneFallback)
  }
  require_fused_qkv_readonly_attention_fallback_join(
    attention,
    (
      readonly_identity.fallback_source_digest_sha256(),
      readonly_identity.fallback_recipe_digest_sha256(),
    ),
    (
      fallbacks[2].source_digest().as_hex(),
      fallbacks[2].recipe_digest().as_hex(),
    ),
    (
      kv_write.fallback_source_digest_sha256(),
      kv_write.fallback_recipe_digest_sha256(),
    ),
  )
}

///|
fn fused_qkv_readonly_digest(
  canonical : Bytes,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest raise FusedGroupedOperationError {
  @luna_cuda_fused_parallel_aot.FusedParallelDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  ) catch {
    _ => raise InvalidFusedGroupedOperation(Resource)
  }
}

///|
fn fused_qkv_readonly_canonical(
  generation : @worker_protocol.ModelPlanGeneration,
  identity : @spec.ModelIdentity,
  blueprint : PagedKvExecutionBlueprint,
  fused_catalog : @luna_fused_artifact_admission.ApprovedFusedKernelCatalog,
  fused_qkv : @luna_fused_artifact_admission.ApprovedFusedKernelModule,
  readonly_identity : @luna_paged_composite_artifact_admission.ApprovedPagedAttentionReadOnlyRuntimeIdentity,
  live_device : @device.ContextDeviceIdentity,
  kv_write : @luna_paged_composite_artifact_admission.QualifiedPagedKvWriteFallbackIdentity,
  qkv_index : Int,
  rope_index : Int,
  attention_index : Int,
  positioned_qkv : @plan.ActivationRef,
  full_qkv_row_width : Int,
) -> Bytes {
  let profile = blueprint.profile()
  let layout = blueprint.layout()
  let production_abi_status = if fused_qkv_readonly_production_abis_match(
      fused_qkv.kernel_abi(),
      readonly_identity.kernel_abi().canonical_name(),
    ) {
    "matched"
  } else {
    "missing"
  }
  @utf8.encode(
    "schema=lunaflux-fused-qkv-readonly-attention-composite.v1\n" +
    "model_content_sha256=" +
    identity.content().as_hex() +
    "\nmodel_plan_sha256=" +
    identity.plan().as_hex() +
    "\nmodel_generation=" +
    generation.value().to_string() +
    "\ntarget=sm_" +
    blueprint.target().compute_major().to_string() +
    blueprint.target().compute_minor().to_string() +
    "\nprofile_id=" +
    profile.id().as_int().to_string() +
    "\nprofile=" +
    profile.max_query_rows().to_string() +
    "," +
    profile.max_query_tokens().to_string() +
    "," +
    profile.max_page_table_entries().to_string() +
    "\nkv_layout_stable_version=" +
    layout.stable_layout_version().to_string() +
    "\ntokens_per_page=" +
    layout.tokens_per_page().to_string() +
    "\ntotal_page_count=" +
    layout.total_page_count().to_string() +
    "\npage_stride_bytes=" +
    layout.page_stride_bytes().to_string() +
    "\ncomponent_stride_bytes=" +
    layout.component_stride_bytes().to_string() +
    "\nqkv_operation_id=" +
    qkv_index.to_string() +
    "\nrope_operation_id=" +
    rope_index.to_string() +
    "\nattention_operation_id=" +
    attention_index.to_string() +
    "\npositioned_qkv_activation_ref=" +
    positioned_qkv.as_int().to_string() +
    "\nfull_qkv_row_width=" +
    full_qkv_row_width.to_string() +
    "\nfused_manifest_sha256=" +
    fused_catalog.manifest_digest().as_hex() +
    "\nfused_catalog_sha256=" +
    fused_catalog.digest().as_hex() +
    "\nfused_candidate_sha256=" +
    fused_qkv.candidate_digest().as_hex() +
    "\nfused_source_sha256=" +
    fused_qkv.source_digest().as_hex() +
    "\nfused_recipe_sha256=" +
    fused_qkv.recipe_digest().as_hex() +
    "\nfused_compiled_binding_sha256=" +
    fused_qkv.compiled_binding_digest().as_hex() +
    "\nfused_cubin_sha256=" +
    fused_qkv.module_digest().as_hex() +
    "\nfused_canary_per_token=" +
    fused_qkv.dispatch_canary_per_token().to_string() +
    "\nreadonly_manifest_sha256=" +
    readonly_identity.manifest_digest_sha256() +
    "\nreadonly_catalog_sha256=" +
    readonly_identity.catalog_digest_sha256() +
    "\nreadonly_candidate_sha256=" +
    readonly_identity.candidate_digest_sha256() +
    "\nreadonly_compiled_binding_sha256=" +
    readonly_identity.compiled_binding_digest_sha256() +
    "\nreadonly_cubin_sha256=" +
    readonly_identity.module_digest_sha256() +
    "\nreadonly_physical_evidence_sha256=" +
    readonly_identity.physical_evidence_digest_sha256() +
    "\nreadonly_policy_sha256=" +
    readonly_identity.approved_policy_digest_sha256() +
    "\nreadonly_runtime_device_sha256=" +
    readonly_identity.runtime_device_digest_sha256() +
    "\nlive_device_ordinal=" +
    live_device.ordinal().to_string() +
    "\nlive_device_uuid=" +
    live_device.uuid() +
    "\nlive_device_pci_bus_id=" +
    live_device.pci() +
    "\nlive_device_name_hex=" +
    @crypto.bytes_to_hex_string(@utf8.encode(live_device.name())) +
    "\nlive_device_total_memory_bytes=" +
    live_device.total_memory_bytes().to_string() +
    "\nreadonly_device_uuid=" +
    readonly_identity.approved_device_uuid() +
    "\nreadonly_device_pci_bus_id=" +
    readonly_identity.approved_device_pci_bus_id() +
    "\nreadonly_device_name_hex=" +
    readonly_identity.approved_device_name_hex() +
    "\nreadonly_device_total_memory_bytes=" +
    readonly_identity.approved_device_total_memory_bytes().to_string() +
    "\nreadonly_canary_per_token=" +
    readonly_identity.dispatch_canary_per_token().to_string() +
    "\nreadonly_canary_cell_count=" +
    readonly_identity.dispatch_canary_cell_count().to_string() +
    "\nkv_write_candidate_sha256=" +
    kv_write.candidate_digest_sha256() +
    "\nkv_write_source_sha256=" +
    kv_write.source_digest_sha256() +
    "\nkv_write_recipe_sha256=" +
    kv_write.recipe_digest_sha256() +
    "\nkv_write_compiled_binding_sha256=" +
    kv_write.compiled_binding_digest_sha256() +
    "\nkv_write_cubin_sha256=" +
    kv_write.module_digest_sha256() +
    "\nkv_write_physical_evidence_sha256=" +
    kv_write.physical_evidence_digest_sha256() +
    "\nkv_write_policy_sha256=" +
    kv_write.approved_policy_digest_sha256() +
    "\ntoolchain_sha256=" +
    fused_qkv.toolchain_digest_sha256() +
    "\ndriver_identity_sha256=" +
    fused_qkv.driver_identity_sha256() +
    "\nlaunch_order=fused-qkv-rope-kvwrite,readonly-attention" +
    "\nstream_order=single-stream-writer-before-reader" +
    "\nexecution_policy=eager-only\nspan_replacement=atomic-three-to-two" +
    "\nproduction_fused_abi=" +
    @luna_cuda_fused_parallel_aot.QkvPositionedRopePagedKvWriteProductionAbiV2.canonical_name() +
    "\nproduction_readonly_abi=" +
    @luna_cuda_paged_attention_readonly_aot.RotatedQueryPagedKvReadOnlyProductionRawPointerV2.canonical_name() +
    "\nproduction_abi_status=" +
    production_abi_status +
    "\nproduction_artifacts=absent\ndispatch_authority=false" +
    "\ntrusted-runtime-bootstrap=absent\n",
  )
}

///|
pub fn admit_fused_qkv_readonly_attention_composite_requirement(
  plan : @plan.ModelPlan,
  generation : @worker_protocol.ModelPlanGeneration,
  blueprint : PagedKvExecutionBlueprint,
  standalone : @luna_kernel_bundle.LunaKernelBundleInput,
  fused_catalog : @luna_fused_artifact_admission.ApprovedFusedKernelCatalog,
  fused_qkv : @luna_fused_artifact_admission.ApprovedFusedKernelModule,
  readonly_identity : @luna_paged_composite_artifact_admission.ApprovedPagedAttentionReadOnlyRuntimeIdentity,
  live_device : @device.ContextDeviceIdentity,
  kv_write : @luna_paged_composite_artifact_admission.QualifiedPagedKvWriteFallbackIdentity,
) -> FusedQkvReadOnlyAttentionCompositeRequirement raise FusedGroupedOperationError {
  require_exact_fused_catalog_entry(fused_catalog, fused_qkv)
  if fused_qkv.family() != QkvPositionedRopePagedKvWrite ||
    fused_qkv.kernel_abi() != QkvPositionedRopePagedKvWriteAbiV1 {
    raise MismatchedFusedGroupedOperation(Family)
  }
  if blueprint.identity() != plan.identity() ||
    blueprint.layout().identity() != plan.identity() ||
    kv_write.model_content_digest_sha256() != plan.identity().content().as_hex() ||
    kv_write.model_plan_digest_sha256() != plan.identity().plan().as_hex() {
    raise MismatchedFusedGroupedOperation(ModelIdentity)
  }
  let fallbacks = fused_qkv.fallbacks()
  if fallbacks.length() != 3 {
    raise MismatchedFusedGroupedOperation(StandaloneFallback)
  }
  let qkv_index = fallbacks[0].operation_index()
  let rope_index = fallbacks[1].operation_index()
  let attention_index = fallbacks[2].operation_index()
  let (positioned_qkv, full_qkv_row_width) = require_fused_qkv_readonly_plan_span(
    plan, qkv_index, rope_index, attention_index,
  )
  require_fused_qkv_readonly_fallbacks(
    standalone, fused_qkv, readonly_identity, kv_write, qkv_index, rope_index, attention_index,
  )
  for index in [qkv_index, rope_index, attention_index] {
    require_blueprint_fallback_module(standalone, blueprint, index)
  }
  if readonly_identity.operation_id() != attention_index ||
    readonly_identity.activation_ref() != positioned_qkv.as_int() ||
    readonly_identity.input_row_width() != full_qkv_row_width ||
    kv_write.operation_id().as_int() != attention_index ||
    kv_write.positioned_qkv_activation() != positioned_qkv ||
    kv_write.positioned_qkv_row_width() != full_qkv_row_width {
    raise MismatchedFusedGroupedOperation(Activation)
  }
  let profile = blueprint.profile()
  let readonly_shape = readonly_identity.profile_shape()
  if readonly_identity.profile_id() != profile.id().as_int() ||
    readonly_shape.0 != profile.max_query_rows() ||
    readonly_shape.1 != profile.max_query_tokens() ||
    readonly_shape.2 != profile.max_page_table_entries() ||
    kv_write.profile() != profile {
    raise MismatchedFusedGroupedOperation(Profile)
  }
  let layout = blueprint.layout()
  if layout.stable_layout_version() != 1 ||
    readonly_identity.tokens_per_page() != layout.tokens_per_page() ||
    readonly_identity.page_stride_bytes() != layout.page_stride_bytes() ||
    readonly_identity.physical_page_capacity() !=
    layout.total_page_count().to_int64() ||
    kv_write.tokens_per_page() != layout.tokens_per_page() ||
    kv_write.total_page_count() != layout.total_page_count() ||
    kv_write.page_stride_bytes() != layout.page_stride_bytes() {
    raise MismatchedFusedGroupedOperation(KvLayout)
  }
  if fused_qkv.target() != blueprint.target() ||
    readonly_identity.target() != blueprint.target() ||
    kv_write.target() != blueprint.target() {
    raise MismatchedFusedGroupedOperation(Target)
  }
  if fused_qkv.toolchain_digest_sha256() !=
    readonly_identity.toolchain_digest_sha256() ||
    fused_qkv.toolchain_digest_sha256() != kv_write.toolchain_digest_sha256() {
    raise MismatchedFusedGroupedOperation(Toolchain)
  }
  if fused_qkv.driver_identity_sha256() !=
    readonly_identity.driver_identity_sha256() ||
    fused_qkv.driver_identity_sha256() != kv_write.driver_identity_sha256() {
    raise MismatchedFusedGroupedOperation(Driver)
  }
  if readonly_identity.approved_device_uuid() != live_device.uuid() ||
    readonly_identity.approved_device_pci_bus_id() != live_device.pci() {
    raise MismatchedFusedGroupedOperation(RuntimeDevice)
  }
  let target = readonly_identity.target()
  let observed = live_device.capability()
  let live_device_join = fused_qkv_readonly_live_device_join(
    expected_uuid=readonly_identity.approved_device_uuid(),
    expected_pci=readonly_identity.approved_device_pci_bus_id(),
    expected_name_hex=readonly_identity.approved_device_name_hex(),
    expected_memory_bytes=readonly_identity.approved_device_total_memory_bytes(),
    expected_compute_major=target.compute_major(),
    expected_compute_minor=target.compute_minor(),
    expected_bf16=target.supports_bf16(),
    expected_cublas_lt=target.supports_cublas_lt(),
    observed_uuid=live_device.uuid(),
    observed_pci=live_device.pci(),
    observed_name_hex=@crypto.bytes_to_hex_string(
      @utf8.encode(live_device.name()),
    ),
    observed_memory_bytes=live_device.total_memory_bytes(),
    observed_compute_major=observed.compute_major(),
    observed_compute_minor=observed.compute_minor(),
    observed_bf16=observed.supports_bf16(),
    observed_cublas_lt=observed.supports_cublas_lt(),
  )
  let fused_canary = fused_qkv.dispatch_canary_per_token()
  let readonly_canary = readonly_identity.dispatch_canary_per_token()
  if fused_canary == 0U ||
    readonly_canary <= 0 ||
    fused_canary == readonly_canary.reinterpret_as_uint() ||
    profile.max_query_tokens().reinterpret_as_uint() >
    0xffffffffU / fused_canary ||
    readonly_identity.dispatch_canary_cell_count() != profile.max_query_tokens() {
    raise InvalidFusedGroupedOperation(KernelAbi)
  }
  let canonical = fused_qkv_readonly_canonical(
    generation,
    plan.identity(),
    blueprint,
    fused_catalog,
    fused_qkv,
    readonly_identity,
    live_device,
    kv_write,
    qkv_index,
    rope_index,
    attention_index,
    positioned_qkv,
    full_qkv_row_width,
  )
  let requirement : FusedQkvReadOnlyAttentionCompositeRequirement = {
    generation,
    identity: plan.identity(),
    fused_catalog: fused_catalog.digest(),
    fused_qkv,
    readonly_identity,
    live_device,
    live_device_join,
    kv_write,
    qkv_index,
    rope_index,
    attention_index,
    positioned_qkv,
    full_qkv_row_width,
    canonical,
    digest: fused_qkv_readonly_digest(canonical),
  }
  requirement.require_fused_qkv_readonly_retained_authority()
  requirement
}
