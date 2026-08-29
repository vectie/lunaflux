///|
fn descriptor_role(
  role : @tensor_parallel_launch_contract.TensorParallelOperandRole,
) -> TensorParallelDescriptorRole? {
  match role {
    RuntimeInput(value) => Some(RuntimeDescriptor(value))
    ExternalInput(value) => Some(ExternalDescriptor(value))
    _ => None
  }
}

///|
fn find_descriptor(
  values : ArrayView[TensorParallelDescriptorBinding],
  role : TensorParallelDescriptorRole,
) -> Int {
  for index, value in values {
    if value.role == role {
      return index
    }
  }
  -1
}

///|
fn plan_descriptors(
  contracts : ArrayView[
    @tensor_parallel_launch_contract.TensorParallelAotLaunchContract,
  ],
  limits : TensorParallelExecutionPlanLimits,
) -> (ReadOnlyArray[TensorParallelDescriptorBinding], Int64) raise TensorParallelExecutionPlanError {
  let values : Array[TensorParallelDescriptorBinding] = []
  let mut cursor = 0L
  for operation, contract in contracts {
    for operand_index, operand in contract.operands() {
      guard descriptor_role(operand.role()) is Some(role) else { continue }
      let existing = find_descriptor(values, role)
      if existing >= 0 {
        let region = values[existing].region
        if region.byte_count != operand.byte_count() ||
          region.alignment != operand.alignment() {
          raise rejected(Descriptor, operation~, operand=operand_index)
        }
        continue
      }
      if values.length() >= limits.max_descriptors {
        raise rejected(Descriptor, operation~, operand=operand_index)
      }
      cursor = checked_align(
        cursor,
        operand.alignment(),
        operation~,
        operand=operand_index,
      )
      let region = scalar_region(
        DescriptorSpace,
        cursor,
        operand.byte_count(),
        operand.alignment(),
        operation~,
        operand=operand_index,
      )
      cursor = region.end_offset_bytes()
      if cursor > limits.max_descriptor_bytes {
        raise rejected(Descriptor, operation~, operand=operand_index)
      }
      values.push({ role, region })
    }
  }
  (ReadOnlyArray::from_array(values), cursor)
}

///|
fn descriptor_region(
  descriptors : ReadOnlyArray[TensorParallelDescriptorBinding],
  role : TensorParallelDescriptorRole,
  operation : Int,
  operand : Int,
) -> TensorParallelScalarRegion raise TensorParallelExecutionPlanError {
  for descriptor in descriptors {
    if descriptor.role == role {
      return descriptor.region
    }
  }
  raise rejected(Descriptor, operation~, operand~)
}
