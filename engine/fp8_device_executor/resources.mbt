///|
fn empty_fp8_resources(
  context : @device.Context,
  weights : @numeric_weight_file.AuthenticatedNumericWeightAuthority,
) -> Fp8DeviceResources {
  {
    context,
    weights: Some(weights),
    weight_lease: None,
    stream: None,
    workspace: None,
    ephemeral: [],
    modules: [],
    functions: [],
  }
}

///|
fn fp8_module_index(
  modules : Array[Fp8OwnedModule],
  digest : @catalog.AotArtifactDigest,
) -> Int? {
  for index, module_ in modules {
    if module_.digest == digest {
      return Some(index)
    }
  }
  None
}

///|
fn fp8_ephemeral_allocation(
  resources : Fp8DeviceResources,
  key : Fp8EphemeralStorageKey,
) -> Fp8OwnedAllocation raise Fp8DeviceExecutorError {
  for allocation in resources.ephemeral {
    if allocation.key == Some(key) && allocation.open {
      return allocation
    }
  }
  raise InvalidFp8DeviceExecutor(ResourceState)
}

///|
fn fp8_weight_allocation(
  resources : Fp8DeviceResources,
) -> @device.Allocation raise Fp8DeviceExecutorError {
  guard resources.weights is Some(weights) else {
    raise InvalidFp8DeviceExecutor(ResourceState)
  }
  weights.allocation() catch {
    error => raise Fp8WeightFailure(error)
  }
}

///|
fn fp8_workspace_allocation(
  resources : Fp8DeviceResources,
) -> @device.Allocation raise Fp8DeviceExecutorError {
  guard resources.workspace is Some(workspace) && workspace.open else {
    raise InvalidFp8DeviceExecutor(ResourceState)
  }
  workspace.resource
}

///|
fn fp8_stream(
  resources : Fp8DeviceResources,
) -> @device.Stream raise Fp8DeviceExecutorError {
  guard resources.stream is Some(stream) && stream.open else {
    raise InvalidFp8DeviceExecutor(ResourceState)
  }
  stream.resource
}

///|
fn fp8_function(
  resources : Fp8DeviceResources,
  index : Int,
) -> @device.Function raise Fp8DeviceExecutorError {
  guard resources.functions.get(index) is Some(function) && function.open else {
    raise InvalidFp8DeviceExecutor(ResourceState)
  }
  function.resource
}
