///|
fn validate_identity_surface(
  model : @plan.ModelPlan,
  launch_set : @tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet,
  device_plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
  artifacts : @artifact.KernelArtifactBundle,
  limits : TensorParallelExecutionPlanLimits,
) -> Unit raise TensorParallelExecutionPlanError {
  let identity = model.identity()
  if launch_set.model_identity() != identity ||
    device_plan.identity() != identity ||
    artifacts.model_identity() != identity {
    raise rejected(ModelIdentity)
  }
  if launch_set.rank() != device_plan.rank() {
    raise rejected(Rank)
  }
  if launch_set.world_size() <= 1 ||
    launch_set.world_size() != device_plan.world_size() {
    raise rejected(WorldSize)
  }
  if launch_set.device_target() != device_plan.target() ||
    launch_set.device_target() != artifacts.device_target() {
    raise rejected(DeviceTarget)
  }
  if launch_set.catalog_version() != artifacts.catalog_version() {
    raise rejected(CatalogVersion)
  }
  if !artifacts.authenticates_tensor_parallel(
      launch_set,
      limits.artifact_limits(),
    ) {
    raise rejected(ArtifactBundle)
  }
}

///|
fn select_contracts(
  model : @plan.ModelPlan,
  launch_set : @tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet,
  device_plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
  profile : @launch_contract.PagedKernelProfile,
  limits : TensorParallelExecutionPlanLimits,
) -> Array[@tensor_parallel_launch_contract.TensorParallelAotLaunchContract] raise TensorParallelExecutionPlanError {
  let operations = model.operations()
  if operations.is_empty() ||
    operations.length() > limits.max_operations ||
    device_plan.operation_count() != operations.length() {
    raise rejected(OperationCount)
  }
  let profiles = launch_set.profiles()
  let mut profile_index = -1
  for index, candidate in profiles {
    if candidate.id() == profile.id() {
      if candidate != profile || profile_index >= 0 {
        raise rejected(Profile)
      }
      profile_index = index
    }
  }
  if profile_index < 0 || profiles.is_empty() {
    raise rejected(Profile)
  }
  let expected_contract_count = checked_multiply(
    Int64::from_int(profiles.length()),
    Int64::from_int(operations.length()),
  )
  let all = launch_set.contracts()
  if expected_contract_count != Int64::from_int(all.length()) {
    raise rejected(OperationCount)
  }
  let start = profile_index * operations.length()
  let selected : Array[
    @tensor_parallel_launch_contract.TensorParallelAotLaunchContract,
  ] = []
  let mut operand_count = 0
  for operation_index, operation in operations {
    let contract = all[start + operation_index]
    guard device_plan.operation(operation.id()) is Some(geometry) else {
      raise rejected(OperationOrder, operation=operation_index)
    }
    if contract.profile() != profile ||
      contract.operation_id().as_int() != operation_index {
      raise rejected(OperationOrder, operation=operation_index)
    }
    if contract.operation_kind() != operation.kind() {
      raise rejected(OperationKind, operation=operation_index)
    }
    if contract.capability() != operation.required_capability() {
      raise rejected(OperationCapability, operation=operation_index)
    }
    if geometry.operation().as_int() != operation_index ||
      geometry.kind() != contract.operation_kind() ||
      geometry.capability() != contract.capability() ||
      geometry.input_width() != contract.input_width() ||
      geometry.output_width() != contract.output_width() ||
      geometry.inner_width() != contract.inner_width() ||
      geometry.query_head_count() != contract.query_head_count() ||
      geometry.kv_head_count() != contract.kv_head_count() ||
      geometry.head_dimension() != contract.head_dimension() {
      raise rejected(OperationCapability, operation=operation_index)
    }
    operand_count += contract.operands().length()
    if operand_count > limits.max_operands {
      raise rejected(Operand, operation=operation_index)
    }
    selected.push(contract)
  }
  selected
}

///|
fn validate_artifact_binding(
  artifacts : @artifact.KernelArtifactBundle,
  contract : @tensor_parallel_launch_contract.TensorParallelAotLaunchContract,
  operation : Int,
) -> (@catalog.AotArtifactDigest, String) raise TensorParallelExecutionPlanError {
  let identity = contract.entry_point()
  guard artifacts.entry_point(identity) is Some(entry) else {
    raise rejected(ArtifactEntry, operation~)
  }
  if entry.identity() != identity ||
    entry.module_index() < 0 ||
    entry.module_index() >= artifacts.module_count() {
    raise rejected(ArtifactEntry, operation~)
  }
  let artifact_module = artifacts.modules()[entry.module_index()]
  let expected_digest = identity.family().module_digest()
  if artifact_module.digest() != expected_digest ||
    artifacts.find_module(expected_digest) is None {
    raise rejected(ArtifactModule, operation~)
  }
  (artifact_module.digest(), entry.function_symbol())
}

///|
fn validate_weight_region(
  device_plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
  reference : @plan.TensorRef,
  byte_count : Int64,
  alignment : Int64,
  operation : Int,
  operand : Int,
) -> TensorParallelScalarRegion raise TensorParallelExecutionPlanError {
  guard device_plan.tensor(reference) is Some(view) else {
    raise rejected(WeightRegion, operation~, operand~)
  }
  if view.reference() != reference ||
    view.region_byte_length() != byte_count ||
    view.region_offset_bytes() < 0L ||
    !valid_alignment(alignment) ||
    view.region_offset_bytes() % alignment != 0L {
    raise rejected(WeightRegion, operation~, operand~)
  }
  let end = checked_add(
    view.region_offset_bytes(),
    view.region_byte_length(),
    operation~,
    operand~,
  )
  if end > device_plan.arena_bytes() {
    raise rejected(WeightRegion, operation~, operand~)
  }
  if view.placement() != Replicated {
    let local_elements = checked_multiply(
      view.local_rows(),
      view.local_columns(),
      operation~,
      operand~,
    )
    let full_elements = checked_multiply(
      view.full_rows(),
      view.full_columns(),
      operation~,
      operand~,
    )
    if local_elements >= full_elements {
      raise rejected(FullShardedWeight, operation~, operand~)
    }
  }
  {
    space: WeightSpace,
    offset_bytes: view.region_offset_bytes(),
    byte_count: view.region_byte_length(),
    reserved_bytes: view.region_byte_length(),
    alignment,
  }
}

///|
fn kv_component_offset(
  layout : @device_layout.DeviceKvLayout,
  layer : @plan.KvLayerId,
  component : @device_layout.DeviceKvComponent,
  operation : Int,
  operand : Int,
) -> Int64 raise TensorParallelExecutionPlanError {
  let component_offset = match component {
    Key => 0L
    Value => layout.component_stride_bytes()
  }
  checked_add(
    checked_multiply(
      Int64::from_int(layer.as_int()),
      layout.layer_stride_bytes(),
      operation~,
      operand~,
    ),
    component_offset,
    operation~,
    operand~,
  )
}

///|
fn validate_kv_region(
  layout : @device_layout.DeviceKvLayout,
  layer : @plan.KvLayerId,
  component : @device_layout.DeviceKvComponent,
  byte_count : Int64,
  alignment : Int64,
  operation : Int,
  operand : Int,
) -> TensorParallelScalarRegion raise TensorParallelExecutionPlanError {
  if byte_count != layout.component_stride_bytes() ||
    alignment != layout.segment_alignment_bytes() ||
    layer.as_int() < 0 ||
    layer.as_int() >= layout.geometry().layer_count() {
    raise rejected(KvRegion, operation~, operand~)
  }
  let region = scalar_region(
    PersistentKvSpace,
    kv_component_offset(layout, layer, component, operation, operand),
    byte_count,
    alignment,
    operation~,
    operand~,
  )
  if region.end_offset_bytes() > layout.arena_bytes() {
    raise rejected(KvRegion, operation~, operand~)
  }
  region
}

///|
fn validate_device_tensors(
  device_plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
) -> Unit raise TensorParallelExecutionPlanError {
  if device_plan.tensor_count() <= 0 {
    raise rejected(WeightRegion)
  }
  let mut previous_end = 0L
  for index in 0..<device_plan.tensor_count() {
    let reference = @plan.TensorRef::new(index) catch {
      _ => raise rejected(WeightRegion, operand=index)
    }
    guard device_plan.tensor(reference) is Some(view) else {
      raise rejected(WeightRegion, operand=index)
    }
    if view.reference().as_int() != index ||
      view.region_offset_bytes() < previous_end {
      raise rejected(WeightRegion, operand=index)
    }
    previous_end = checked_add(
      view.region_offset_bytes(),
      view.region_byte_length(),
      operand=index,
    )
  }
  if previous_end > device_plan.arena_bytes() {
    raise rejected(WeightRegion)
  }
}
