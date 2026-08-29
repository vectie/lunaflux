///|
fn probe_module_digest(bytes : Bytes) -> @catalog.AotArtifactDigest raise {
  @catalog.AotArtifactDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
}

///|
fn probe_family(
  module_digest : @catalog.AotArtifactDigest,
  index : Int,
) -> @catalog.AotKernelFamily raise {
  @catalog.AotKernelFamily::new(
    module_digest~,
    family_id=@catalog.AotKernelFamilyId::from_stable_int(index + 1),
  )
}

///|
fn probe_entry_point(
  module_digest : @catalog.AotArtifactDigest,
  index : Int,
) -> @catalog.AotKernelEntryPoint raise {
  @catalog.AotKernelEntryPoint::new(
    family=probe_family(module_digest, index),
    entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(index + 101),
  )
}

///|
fn probe_representative_operation(plan : @plan.ModelPlan, index : Int) -> Int {
  let operation = plan.operations()[index]
  for earlier in 0..<index {
    let candidate = plan.operations()[earlier]
    if candidate.kind() == operation.kind() &&
      candidate.shape() == operation.shape() &&
      candidate.required_capability() == operation.required_capability() &&
      candidate.kv_layer() == operation.kv_layer() &&
      candidate.runtime_inputs() == operation.runtime_inputs() &&
      candidate.persistent_state() == operation.persistent_state() &&
      candidate.workspace_bytes() == operation.workspace_bytes() &&
      plan.numeric_binding().operations()[earlier].execution().digest() ==
      plan.numeric_binding().operations()[index].execution().digest() {
      return earlier
    }
  }
  index
}

///|
fn probe_execution_semantics(
  operation : @plan.PlanOperation,
  layout : @device_layout.DeviceKvLayout,
) -> @catalog.KernelExecutionSemantics raise {
  let kv_layout = if operation.kind() == CausalAttention {
    Some(
      @catalog.PagedKvLayoutContract::new(
        layout=layout.geometry().layout(),
        stable_version=layout.stable_layout_version(),
        tokens_per_page=layout.tokens_per_page(),
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
fn probe_resolved_catalog(
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  target : @catalog.DeviceTarget,
  module_digest : @catalog.AotArtifactDigest,
) -> @catalog.ResolvedKernelCatalog raise {
  let entries : Array[@catalog.CatalogEntry] = []
  for index, operation in plan.operations() {
    let representative = probe_representative_operation(plan, index)
    let family = probe_family(module_digest, representative)
    let entry = @catalog.CatalogEntry::new_paged_v4(
      operation_kind=operation.kind(),
      operation_shape=operation.shape(),
      capability=operation.required_capability(),
      execution_semantics=probe_execution_semantics(operation, layout),
      operation_execution_digest=plan.numeric_binding().operations()[index]
        .execution()
        .digest(),
      aot_entry_point=probe_entry_point(module_digest, representative),
      compute_major=target.compute_major(),
      compute_minor=target.compute_minor(),
      implementation=ContentAddressedAot(family),
      workspace_bytes=0L,
      workspace_alignment=1L,
    )
    if !entries.contains(entry) {
      entries.push(entry)
    }
  }
  @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v4(), entries~).resolve_paged_v4(
    plan, layout, target,
  )
}
