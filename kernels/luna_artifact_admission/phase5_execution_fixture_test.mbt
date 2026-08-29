///|
fn phase5_module_bytes() -> Bytes {
  Bytes::from_array([b'L', b'U', b'N', b'A', b'T', b'I', b'L', b'E'])
}

///|
fn phase5_module_digest() -> @catalog.AotArtifactDigest raise {
  @catalog.AotArtifactDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(phase5_module_bytes())),
  )
}

///|
fn phase5_target() -> @catalog.DeviceTarget raise {
  @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
}

///|
fn phase5_semantics(
  operation : @plan.PlanOperation,
) -> @catalog.KernelExecutionSemantics raise {
  let kv_layout = if operation.persistent_state() is KvCacheReadWrite(_) {
    Some(
      @catalog.PagedKvLayoutContract::new(
        layout=LayerMajorSplitKeyValueV1,
        stable_version=1,
        tokens_per_page=4,
      ),
    )
  } else {
    None
  }
  @catalog.KernelExecutionSemantics::paged_v3(
    kv_layer=operation.kv_layer(),
    runtime_inputs=operation.runtime_inputs(),
    persistent_state=operation.persistent_state(),
    kv_layout~,
  )
}

///|
fn phase5_catalog() -> @catalog.ResolvedKernelCatalog raise {
  let entries : Array[@catalog.CatalogEntry] = []
  for operation in phase5_model_plan().operations() {
    let family = @catalog.AotKernelFamily::new(
      module_digest=phase5_module_digest(),
      family_id=@catalog.AotKernelFamilyId::from_stable_int(
        operation.id().as_int() + 1,
      ),
    )
    entries.push(
      @catalog.CatalogEntry::new_paged_v3(
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        execution_semantics=phase5_semantics(operation),
        compute_major=8,
        compute_minor=0,
        implementation=ContentAddressedAot(family),
        workspace_bytes=0L,
        workspace_alignment=1L,
      ),
    )
  }
  @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v3(), entries~).resolve_paged(
    phase5_model_plan(),
    phase5_kv_layout(),
    phase5_target(),
  )
}

///|
fn phase5_static_plan() -> @device_plan.StaticDevicePlan raise {
  @device_plan.build(
    phase5_model_plan(),
    phase5_weight_layout(),
    phase5_catalog(),
  )
}

///|
fn phase5_memory_plan() -> @device_memory.DeviceMemoryPlan raise {
  @device_memory.plan(
    phase5_static_plan(),
    @device_memory.DeviceMemoryLimits::new(
      activation_alignment=64L,
      max_arena_bytes=65536L,
    ),
  )
}

///|
fn phase5_profile() -> @device_profile.DeviceExecutionProfile raise {
  @device_profile.build(
    phase5_static_plan(),
    phase5_memory_plan(),
    @device_profile.ExactExecutionShape::full_recompute(
      batch_rows=2,
      sequence_tokens=2,
    ),
  )
}
