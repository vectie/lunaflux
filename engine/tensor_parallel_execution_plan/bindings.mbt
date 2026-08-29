///|
fn output_slice_region(
  memory : PlanningMemory,
  reference : @plan.ActivationRef,
  operand_byte_count : Int64,
  operand_alignment : Int64,
  collective : @tensor_parallel_launch_contract.TensorParallelCollectiveAfter,
  rank : Int,
  operation : Int,
  operand : Int,
) -> TensorParallelScalarRegion raise TensorParallelExecutionPlanError {
  let whole = activation_region(memory, reference, operation, operand)
  match collective {
    CollectiveLastAxisAllGather(..) => {
      let rank_offset = checked_multiply(
        Int64::from_int(rank),
        operand_byte_count,
        operation~,
        operand~,
      )
      let offset = checked_add(
        whole.offset_bytes(),
        rank_offset,
        operation~,
        operand~,
      )
      let slice = scalar_region(
        ActivationWorkspaceSpace,
        offset,
        operand_byte_count,
        operand_alignment,
        operation~,
        operand~,
      )
      if slice.end_offset_bytes() > whole.end_offset_bytes() {
        raise rejected(CollectiveGeometry, operation~, operand~)
      }
      slice
    }
    _ => {
      if whole.byte_count() != operand_byte_count ||
        whole.offset_bytes() % operand_alignment != 0L {
        raise rejected(ActivationRegion, operation~, operand~)
      }
      scalar_region(
        ActivationWorkspaceSpace,
        whole.offset_bytes(),
        operand_byte_count,
        operand_alignment,
        operation~,
        operand~,
      )
    }
  }
}

///|
fn bind_operand(
  role : @tensor_parallel_launch_contract.TensorParallelOperandRole,
  byte_count : Int64,
  alignment : Int64,
  contract : @tensor_parallel_launch_contract.TensorParallelAotLaunchContract,
  memory : PlanningMemory,
  launch_set : @tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet,
  device_plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
  operation : Int,
  operand : Int,
) -> TensorParallelScalarRegion raise TensorParallelExecutionPlanError {
  if byte_count <= 0L || !valid_alignment(alignment) {
    raise rejected(Operand, operation~, operand~)
  }
  match role {
    RuntimeInput(value) =>
      descriptor_region(
        memory.descriptors,
        RuntimeDescriptor(value),
        operation,
        operand,
      )
    ExternalInput(value) =>
      descriptor_region(
        memory.descriptors,
        ExternalDescriptor(value),
        operation,
        operand,
      )
    RankConstants(version~) => {
      if version != launch_set.rank_constants().version() ||
        memory.rank_constants.byte_count() != byte_count ||
        memory.rank_constants.alignment() != alignment {
        raise rejected(RankConstants, operation~, operand~)
      }
      memory.rank_constants
    }
    WeightInput(reference) =>
      validate_weight_region(
        device_plan, reference, byte_count, alignment, operation, operand,
      )
    ActivationInput(reference) => {
      let region = activation_region(memory, reference, operation, operand)
      if region.byte_count() != byte_count ||
        region.offset_bytes() % alignment != 0L {
        raise rejected(ActivationRegion, operation~, operand~)
      }
      scalar_region(
        ActivationWorkspaceSpace,
        region.offset_bytes(),
        byte_count,
        alignment,
        operation~,
        operand~,
      )
    }
    ActivationOutput(reference) =>
      output_slice_region(
        memory,
        reference,
        byte_count,
        alignment,
        contract.collective_after(),
        launch_set.rank(),
        operation,
        operand,
      )
    PersistentKvComponent(layer, component) =>
      validate_kv_region(
        launch_set.device_kv_layout(),
        layer,
        component,
        byte_count,
        alignment,
        operation,
        operand,
      )
    Workspace => {
      if byte_count > memory.workspace.byte_count() ||
        memory.workspace.offset_bytes() % alignment != 0L {
        raise rejected(WorkspaceRegion, operation~, operand~)
      }
      scalar_region(
        ActivationWorkspaceSpace,
        memory.workspace.offset_bytes(),
        byte_count,
        alignment,
        operation~,
        operand~,
      )
    }
  }
}

///|
fn build_operation_bindings(
  contracts : ArrayView[
    @tensor_parallel_launch_contract.TensorParallelAotLaunchContract,
  ],
  launch_set : @tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet,
  device_plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
  artifacts : @artifact.KernelArtifactBundle,
  memory : PlanningMemory,
) -> ReadOnlyArray[TensorParallelOperationBinding] raise TensorParallelExecutionPlanError {
  let output : Array[TensorParallelOperationBinding] = []
  for operation, contract in contracts {
    let (module_digest, function_symbol) = validate_artifact_binding(
      artifacts, contract, operation,
    )
    let operands : Array[TensorParallelPhysicalOperand] = []
    for operand_index, source in contract.operands() {
      operands.push({
        role: source.role(),
        region: bind_operand(
          source.role(),
          source.byte_count(),
          source.alignment(),
          contract,
          memory,
          launch_set,
          device_plan,
          operation,
          operand_index,
        ),
      })
    }
    output.push({
      operation_id: contract.operation_id(),
      operation_kind: contract.operation_kind(),
      capability: contract.capability(),
      entry_point: contract.entry_point(),
      module_digest,
      function_symbol,
      dimensions: contract.dimensions(),
      operands: ReadOnlyArray::from_array(operands),
    })
  }
  ReadOnlyArray::from_array(output)
}

///|
fn operation_output_region(
  model : @plan.ModelPlan,
  memory : PlanningMemory,
  operation : Int,
) -> TensorParallelScalarRegion raise TensorParallelExecutionPlanError {
  let outputs = model.operations()[operation].outputs()
  if outputs.length() != 1 {
    raise rejected(ActivationRegion, operation~)
  }
  activation_region(memory, outputs[0], operation, -1)
}

///|
fn validate_collective_site(
  device_plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
  index : Int,
  operation : Int,
  expected_kind : @tensor_parallel_plan.TensorCollectiveKind,
  full_width : Int,
  local_width : Int,
) -> Unit raise TensorParallelExecutionPlanError {
  guard device_plan.collective(index) is Some(site) else {
    raise rejected(CollectiveCount, operation~)
  }
  if site.sequence().as_int() != index + 1 ||
    site.operation().as_int() != operation {
    raise rejected(CollectiveSequence, operation~)
  }
  if site.kind() != expected_kind ||
    site.full_width() != full_width ||
    site.local_width() != local_width {
    raise rejected(CollectiveGeometry, operation~)
  }
}

///|
fn build_collective_bindings(
  model : @plan.ModelPlan,
  contracts : ArrayView[
    @tensor_parallel_launch_contract.TensorParallelAotLaunchContract,
  ],
  launch_set : @tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet,
  device_plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
  memory : PlanningMemory,
  limits : TensorParallelExecutionPlanLimits,
) -> ReadOnlyArray[TensorParallelCollectiveBinding] raise TensorParallelExecutionPlanError {
  let output : Array[TensorParallelCollectiveBinding] = []
  for operation, contract in contracts {
    match contract.collective_after() {
      NoCollective => ()
      CollectiveSumAllReduce(sequence~, width~) => {
        if sequence != output.length() + 1 ||
          output.length() >= limits.max_collectives {
          raise rejected(CollectiveSequence, operation~)
        }
        validate_collective_site(
          device_plan,
          output.length(),
          operation,
          SumAllReduce,
          width,
          width,
        )
        let region = operation_output_region(model, memory, operation)
        let expected = checked_multiply(
          checked_multiply(
            Int64::from_int(contract.profile().max_query_tokens()),
            Int64::from_int(width),
            operation~,
          ),
          2L,
          operation~,
        )
        if region.byte_count() != expected {
          raise rejected(CollectiveGeometry, operation~)
        }
        output.push({
          sequence,
          operation_id: contract.operation_id(),
          kind: InPlaceSumAllReduce,
          region,
          local_send_offset_bytes: region.offset_bytes(),
          local_send_bytes: region.byte_count(),
          receive_bytes: region.byte_count(),
        })
      }
      CollectiveLastAxisAllGather(sequence~, full_width~, local_width~) => {
        if sequence != output.length() + 1 ||
          output.length() >= limits.max_collectives ||
          full_width != local_width * launch_set.world_size() {
          raise rejected(CollectiveSequence, operation~)
        }
        validate_collective_site(
          device_plan,
          output.length(),
          operation,
          LastAxisAllGather,
          full_width,
          local_width,
        )
        let region = operation_output_region(model, memory, operation)
        let local_bytes = checked_multiply(
          checked_multiply(
            Int64::from_int(contract.profile().max_query_tokens()),
            Int64::from_int(local_width),
            operation~,
          ),
          2L,
          operation~,
        )
        let full_bytes = checked_multiply(
          local_bytes,
          Int64::from_int(launch_set.world_size()),
          operation~,
        )
        if region.byte_count() != full_bytes {
          raise rejected(CollectiveGeometry, operation~)
        }
        let send_offset = checked_add(
          region.offset_bytes(),
          checked_multiply(
            Int64::from_int(launch_set.rank()),
            local_bytes,
            operation~,
          ),
          operation~,
        )
        if checked_add(send_offset, local_bytes, operation~) >
          region.end_offset_bytes() {
          raise rejected(CollectiveGeometry, operation~)
        }
        output.push({
          sequence,
          operation_id: contract.operation_id(),
          kind: InPlaceLastAxisAllGather,
          region,
          local_send_offset_bytes: send_offset,
          local_send_bytes: local_bytes,
          receive_bytes: full_bytes,
        })
      }
    }
  }
  if output.length() != device_plan.collective_count() {
    raise rejected(CollectiveCount)
  }
  ReadOnlyArray::from_array(output)
}
