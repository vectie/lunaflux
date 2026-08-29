///|
fn paged_artifact_v4_entry(
  digest : @catalog.AotArtifactDigest,
  operation : Int,
) -> @catalog.AotKernelEntryPoint {
  if operation == 0 {
    entry(digest, 10, 3)
  } else if operation % 2 == 1 {
    entry(digest, 11, 1)
  } else {
    entry(digest, 12, 2)
  }
}

///|
fn paged_artifact_v4_catalog(
  digest : @catalog.AotArtifactDigest,
) -> @catalog.ResolvedKernelCatalog raise {
  let plan = paged_artifact_plan()
  let entries : Array[@catalog.CatalogEntry] = []
  for index, operation in plan.operations() {
    let family_id = if index == 0 {
      10
    } else if index % 2 == 1 {
      11
    } else {
      12
    }
    entries.push(
      @catalog.CatalogEntry::new_paged_v4(
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        execution_semantics=paged_artifact_semantics(
          operation.kind(),
          operation.kv_layer(),
        ),
        operation_execution_digest=plan.numeric_binding().operations()[index]
          .execution()
          .digest(),
        aot_entry_point=paged_artifact_v4_entry(digest, index),
        compute_major=8,
        compute_minor=0,
        implementation=ContentAddressedAot(family(digest, family_id)),
        workspace_bytes=0L,
        workspace_alignment=1L,
      ),
    )
  }
  @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v4(), entries~).resolve_paged_v4(
    plan,
    paged_artifact_layout(),
    @catalog.DeviceTarget::new(
      compute_major=8,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
  )
}

///|
fn paged_artifact_v4_contracts(
  digest : @catalog.AotArtifactDigest,
) -> @launch_contract.PagedV4LaunchContractSet raise {
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=2,
    max_query_tokens=4,
    max_page_table_entries=4,
  )
  let inputs : Array[@launch_contract.PagedV4AotLaunchContractInput] = []
  for operation in 0..<5 {
    inputs.push(
      @launch_contract.PagedV4AotLaunchContractInput::new(
        profile_id=profile.id(),
        operation_id=@plan.OperationId::new(operation),
        entry_point=paged_artifact_v4_entry(digest, operation),
        dimensions=dimensions(4),
        operands=paged_artifact_v4_operands(operation),
      ),
    )
  }
  @launch_contract.admit_paged_graph_v4(
    paged_artifact_plan(),
    paged_artifact_v4_catalog(digest),
    paged_artifact_layout(),
    [profile],
    inputs,
    @launch_contract.LaunchContractLimits::new(
      max_profiles=2,
      max_contracts=8,
      max_operands_per_contract=16,
    ),
  )
}

///|
fn paged_artifact_v4_operands(
  operation : Int,
) -> Array[@launch_contract.AotOperand] raise {
  if operation == 0 {
    return paged_artifact_operands(operation)
  }
  let layer = @plan.KvLayerId::new((operation - 1) / 2)
  let input = @plan.ActivationRef::new(operation - 1)
  let output = @plan.ActivationRef::new(operation)
  let prefix = [
    paged_artifact_operand(RuntimeInput(StepCounts), 20L, 4L),
    paged_artifact_operand(RuntimeInput(QueryPositions), 16L, 4L),
  ]
  if operation % 2 == 1 {
    prefix.push(paged_artifact_operand(ActivationInput(input), 256L, 16L))
    prefix.push(paged_artifact_operand(ActivationOutput(output), 256L, 16L))
    return prefix
  }
  prefix.push(paged_artifact_operand(RuntimeInput(QueryRowOffsets), 12L, 4L))
  prefix.push(paged_artifact_operand(RuntimeInput(SequenceLengths), 8L, 4L))
  prefix.push(paged_artifact_operand(RuntimeInput(PageTableOffsets), 12L, 4L))
  prefix.push(
    paged_artifact_operand(RuntimeInput(PageTablePageIndices), 16L, 4L),
  )
  prefix.push(paged_artifact_operand(ActivationInput(input), 256L, 16L))
  prefix.push(paged_artifact_operand(ActivationOutput(output), 128L, 16L))
  prefix.push(
    paged_artifact_operand(PersistentKvComponent(layer, Key), 2048L, 256L),
  )
  prefix.push(
    paged_artifact_operand(PersistentKvComponent(layer, Value), 2048L, 256L),
  )
  prefix
}

///|
fn paged_artifact_v4_exports(
  digest : @catalog.AotArtifactDigest,
) -> Array[@artifact.KernelEntryPointInput] {
  [
    export_input(digest, 10, 3, "embedding_v4"),
    export_input(digest, 11, 1, "rotary_v4"),
    export_input(digest, 12, 2, "attention_v4"),
  ]
}
