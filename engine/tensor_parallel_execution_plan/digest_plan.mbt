///|
fn append_physical_plan(
  encoder : ExecutionDigestEncoder,
  plan : TensorParallelExecutionPlan,
) -> Unit {
  encoder.append_i64(plan.descriptor_bytes)
  encoder.append_i64(plan.weight_arena_bytes)
  encoder.append_i64(plan.kv_arena_bytes)
  encoder.append_i64(plan.activation_workspace_bytes)
  encoder.append_region(plan.rank_constants)
  encoder.append_region(plan.workspace)
  encoder.append_int(plan.descriptors.length())
  for descriptor in plan.descriptors {
    match descriptor.role {
      RuntimeDescriptor(value) => {
        encoder.append_int(1)
        encoder.append_int(runtime_input_code(value))
      }
      ExternalDescriptor(TokenIds) => encoder.append_int(2)
    }
    encoder.append_region(descriptor.region)
  }
  encoder.append_int(plan.activations.length())
  for activation in plan.activations {
    encoder.append_int(activation.reference.as_int())
    encoder.append_int(activation.slot)
    encoder.append_int(activation.produced_by.as_int())
    encoder.append_int(activation.live_through_step)
    encoder.append_region(activation.region)
  }
  encoder.append_int(plan.operations.length())
  for operation in plan.operations {
    encoder.append_int(operation.operation_id.as_int())
    encoder.append_int(operation_kind_code(operation.operation_kind))
    encoder.append_int(operation.capability.as_int())
    encoder.append_entry_point(operation.entry_point)
    encoder.append_string(operation.module_digest.sha256())
    encoder.append_string(operation.function_symbol)
    encoder.append_dimensions(operation.dimensions)
    encoder.append_int(operation.operands.length())
    for operand in operation.operands {
      encoder.append_operand_role(operand.role)
      encoder.append_region(operand.region)
    }
  }
  encoder.append_int(plan.collectives.length())
  for collective in plan.collectives {
    encoder.append_int(collective.sequence)
    encoder.append_int(collective.operation_id.as_int())
    encoder.append_int(
      match collective.kind {
        InPlaceSumAllReduce => 1
        InPlaceLastAxisAllGather => 2
      },
    )
    encoder.append_region(collective.region)
    encoder.append_i64(collective.local_send_offset_bytes)
    encoder.append_i64(collective.local_send_bytes)
    encoder.append_i64(collective.receive_bytes)
  }
}

///|
fn derive_digest(
  plan : TensorParallelExecutionPlan,
  model : @plan.ModelPlan,
  launch_set : @tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet,
  contracts : ArrayView[
    @tensor_parallel_launch_contract.TensorParallelAotLaunchContract,
  ],
  device_plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
  artifacts : @artifact.KernelArtifactBundle,
) -> TensorParallelExecutionPlanDigest raise TensorParallelExecutionPlanError {
  let encoder : ExecutionDigestEncoder = { bytes: [] }
  encoder.append_string("lunaflux.tensor-parallel-execution-plan.v1")
  encoder.append_identity(plan.identity)
  encoder.append_int(plan.rank)
  encoder.append_int(plan.world_size)
  encoder.append_int(plan.device_ordinal)
  encoder.append_target(plan.target)
  encoder.append_int(plan.catalog_version.as_int())
  encoder.append_profile(plan.profile)
  append_model_graph(encoder, model)
  append_launch_sources(encoder, launch_set, contracts)
  append_kv_layout(encoder, launch_set.device_kv_layout())
  append_device_plan(encoder, device_plan)
  append_artifacts(encoder, artifacts)
  append_physical_plan(encoder, plan)
  encoder.finish()
}
