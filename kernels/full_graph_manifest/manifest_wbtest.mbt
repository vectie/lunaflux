///|
fn full_manifest_dimensions(
  operation : Int,
) -> @launch_contract.AotLaunchDimensions raise {
  @launch_contract.AotLaunchDimensions::new(
    grid_x=operation + 1,
    grid_y=1,
    grid_z=1,
    block_x=32,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
}

///|
fn full_manifest_contracts() -> @launch_contract.PagedKvLaunchContractSet raise {
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(7),
    max_query_rows=2,
    max_query_tokens=4,
    max_page_table_entries=4,
  )
  let inputs : Array[@launch_contract.PagedKvAotLaunchContractInput] = []
  for operation in 0..<3 {
    inputs.push(
      @launch_contract.PagedKvAotLaunchContractInput::new(
        profile_id=profile.id(),
        operation_id=@plan.OperationId::new(operation),
        entry_point=full_manifest_entry(operation),
        dimensions=full_manifest_dimensions(operation),
        operands=full_manifest_operands(operation),
      ),
    )
  }
  @launch_contract.admit_paged_graph(
    full_manifest_plan(),
    full_manifest_resolved(),
    full_manifest_layout(),
    [profile],
    inputs,
    @launch_contract.LaunchContractLimits::new(
      max_profiles=1,
      max_contracts=3,
      max_operands_per_contract=16,
    ),
  )
}

///|
fn full_manifest_artifacts(
  bad_operand_at? : Int = -1,
  bad_dimensions_at? : Int = -1,
) -> Array[FullGraphAotOperationArtifact] raise {
  let contracts = full_manifest_contracts().contracts()
  let artifacts : Array[FullGraphAotOperationArtifact] = []
  for index, contract in contracts {
    let operands = if index == bad_operand_at {
      let changed = full_manifest_operands(index)
      changed[0] = full_manifest_operand(RuntimeInput(StepCounts), 24L, 4L)
      changed
    } else {
      full_manifest_operands(index)
    }
    let dimensions = if index == bad_dimensions_at {
      @launch_contract.AotLaunchDimensions::new(
        grid_x=9,
        grid_y=1,
        grid_z=1,
        block_x=32,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      )
    } else {
      full_manifest_dimensions(index)
    }
    artifacts.push(
      FullGraphAotOperationArtifact::new(
        operation_id=contract.operation_id(),
        entry_point=contract.entry_point(),
        module_relative_path="sha256/" +
          contract.entry_point().family().module_digest().sha256() +
          ".cubin",
        function_symbol="kernel_" + (index + 1).to_string(),
        dimensions~,
        operands~,
        workspace_bytes=contract.workspace_bytes(),
        workspace_alignment=contract.workspace_alignment(),
      ),
    )
  }
  artifacts
}

///|
fn full_manifest_limits(
  maximum? : Int64 = 65536L,
) -> FullGraphManifestLimits raise {
  FullGraphManifestLimits::new(
    max_manifest_bytes=maximum,
    artifact_file=@artifact_file.KernelArtifactFileLimits::new(
      max_manifest_bytes=65536L,
      max_json_depth=16,
      max_path_bytes=96,
      artifact=@artifact.KernelArtifactLimits::new(
        max_modules=4,
        max_entry_points=8,
        max_module_bytes=4096L,
        max_total_module_bytes=16384L,
        max_function_symbol_bytes=32,
      ),
    ),
  )
}

///|
test "producer emits deterministic schema-v2 full-graph input" {
  let first = produce_execution_manifest_v2(
    full_manifest_plan(),
    full_manifest_target(),
    full_manifest_contracts(),
    full_manifest_artifacts(),
    full_manifest_limits(),
  )
  let second = produce_execution_manifest_v2(
    full_manifest_plan(),
    full_manifest_target(),
    full_manifest_contracts(),
    full_manifest_artifacts(),
    full_manifest_limits(),
  )
  assert_eq(first.bytes(), second.bytes())
  assert_eq(first.digest_sha256(), second.digest_sha256())
  assert_eq(first.module_count(), 2)
  assert_eq(first.entry_point_count(), 3)
  assert_eq(first.operation_count(), 3)
  let text = @utf8.decode(first.bytes())
  guard @json.parse(text) is Object(root) else {
    fail("expected canonical manifest object")
  }
  guard root["schema_version"] is Number(schema, repr=_) && schema.to_int() == 2 else {
    fail("expected schema version two")
  }
  guard root["catalog_version"] is Number(catalog, repr=_) &&
    catalog.to_int() == 3 else {
    fail("expected catalog version three")
  }
  guard root["modules"] is Array(modules) else { fail("expected module array") }
  guard modules[0] is Object(first_module) else {
    fail("expected first module")
  }
  guard first_module["sha256"] is String(digest) else {
    fail("expected module digest")
  }
  assert_eq(
    digest, "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
}

///|
test "producer rejects artifact launch and operand drift" {
  for
    pair in [
      (OperandContract, full_manifest_artifacts(bad_operand_at=1)),
      (LaunchDimensions, full_manifest_artifacts(bad_dimensions_at=2)),
    ] {
    let (rule, artifacts) = pair
    try
      produce_execution_manifest_v2(
        full_manifest_plan(),
        full_manifest_target(),
        full_manifest_contracts(),
        artifacts,
        full_manifest_limits(),
      )
    catch {
      Mismatch(rule=actual, operation_index=_) => assert_eq(actual, rule)
      _ => fail("unexpected artifact contract error")
    } noraise {
      _ => fail("expected artifact contract rejection")
    }
  }
}

///|
test "producer rejects invalid path and bounded output" {
  let contracts = full_manifest_contracts().contracts()
  let artifacts = full_manifest_artifacts()
  let first = contracts[1]
  artifacts[1] = FullGraphAotOperationArtifact::new(
    operation_id=first.operation_id(),
    entry_point=first.entry_point(),
    module_relative_path="../escape.cubin",
    function_symbol="kernel_2",
    dimensions=first.dimensions(),
    operands=first.operands(),
    workspace_bytes=first.workspace_bytes(),
    workspace_alignment=first.workspace_alignment(),
  )
  try
    produce_execution_manifest_v2(
      full_manifest_plan(),
      full_manifest_target(),
      full_manifest_contracts(),
      artifacts,
      full_manifest_limits(),
    )
  catch {
    Mismatch(rule=ModulePath, operation_index=1) => ()
    _ => fail("unexpected path error")
  } noraise {
    _ => fail("expected invalid path rejection")
  }
  try
    produce_execution_manifest_v2(
      full_manifest_plan(),
      full_manifest_target(),
      full_manifest_contracts(),
      full_manifest_artifacts(),
      full_manifest_limits(maximum=64L),
    )
  catch {
    LimitExceeded(rule=ManifestBytes, maximum=64L) => ()
    _ => fail("unexpected bounded output error")
  } noraise {
    _ => fail("expected manifest size rejection")
  }
}
