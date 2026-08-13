///|
test "canonical v1 operation ceiling is exact" {
  let artifact = @artifact.KernelArtifactLimits::new(
    max_modules=1,
    max_entry_points=4096,
    max_module_bytes=1L,
    max_total_module_bytes=1L,
    max_function_symbol_bytes=1,
  )
  let artifact_file = @artifact_file.KernelArtifactFileLimits::new(
    max_manifest_bytes=1L,
    max_json_depth=1,
    max_path_bytes=1,
    artifact~,
  )
  let memory = @device_memory.DeviceMemoryLimits::new(
    activation_alignment=16L,
    max_arena_bytes=1L,
  )
  let contracts = @launch_contract.LaunchContractLimits::new(
    max_profiles=1,
    max_contracts=4096,
    max_operands_per_contract=1,
  )
  ignore(
    ExecutionManifestFileLimits::new(
      max_manifest_bytes=1L,
      max_json_depth=1,
      max_modules=1,
      max_operations=4096,
      device_memory=memory,
      launch_contract=contracts,
      artifact_file~,
    ),
  )
  try
    ExecutionManifestFileLimits::new(
      max_manifest_bytes=1L,
      max_json_depth=1,
      max_modules=1,
      max_operations=4097,
      device_memory=memory,
      launch_contract=contracts,
      artifact_file~,
    )
  catch {
    InvalidLimit(OperationCount) => ()
    _ => fail("unexpected operation ceiling failure")
  } noraise {
    _ => fail("4097 operations were admitted")
  }
}

///|
test "nested ceilings are canonically reconstructed" {
  let memory = @device_memory.DeviceMemoryLimits::new(
    activation_alignment=16L,
    max_arena_bytes=1024L,
  )
  let contracts = @launch_contract.LaunchContractLimits::new(
    max_profiles=1,
    max_contracts=1,
    max_operands_per_contract=1,
  )
  let artifact = @artifact.KernelArtifactLimits::new(
    max_modules=1,
    max_entry_points=1,
    max_module_bytes=1L,
    max_total_module_bytes=1L,
    max_function_symbol_bytes=1,
  )
  let artifact_file = @artifact_file.KernelArtifactFileLimits::new(
    max_manifest_bytes=1L,
    max_json_depth=1,
    max_path_bytes=1,
    artifact~,
  )
  let admitted = ExecutionManifestFileLimits::new(
    max_manifest_bytes=1L,
    max_json_depth=1,
    max_modules=1,
    max_operations=1,
    device_memory=memory,
    launch_contract=contracts,
    artifact_file~,
  )
  assert_eq(admitted.device_memory(), memory)
  assert_eq(admitted.launch_contract(), contracts)
  assert_eq(admitted.artifact_file().max_manifest_bytes(), 1L)
  assert_eq(admitted.artifact_file().artifact().max_modules(), 1)
}
