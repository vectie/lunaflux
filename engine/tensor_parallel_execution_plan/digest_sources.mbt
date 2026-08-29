///|
fn append_operation_shape(
  encoder : ExecutionDigestEncoder,
  shape : @plan.OperationShape,
) -> Unit {
  match shape {
    EmbeddingShape(vocabulary_size~, width~) => {
      encoder.append_int(1)
      encoder.append_int(vocabulary_size)
      encoder.append_int(width)
    }
    NormalizationShape(width~, epsilon~) => {
      encoder.append_int(2)
      encoder.append_int(width)
      encoder.append_u64(epsilon.reinterpret_as_uint64())
    }
    ProjectionShape(input_width~, output_width~) => {
      encoder.append_int(3)
      encoder.append_int(input_width)
      encoder.append_int(output_width)
    }
    RotaryShape(query_heads~, kv_heads~, head_dimension~, rope_theta~) => {
      encoder.append_int(4)
      encoder.append_int(query_heads)
      encoder.append_int(kv_heads)
      encoder.append_int(head_dimension)
      encoder.append_u64(rope_theta.reinterpret_as_uint64())
    }
    AttentionShape(query_heads~, kv_heads~, head_dimension~) => {
      encoder.append_int(5)
      encoder.append_int(query_heads)
      encoder.append_int(kv_heads)
      encoder.append_int(head_dimension)
    }
    ElementwiseShape(width~) => {
      encoder.append_int(6)
      encoder.append_int(width)
    }
    GatedMlpShape(hidden_size~, intermediate_size~) => {
      encoder.append_int(7)
      encoder.append_int(hidden_size)
      encoder.append_int(intermediate_size)
    }
    QkNormalizationShape(query_heads~, kv_heads~, head_dimension~, epsilon~) => {
      encoder.append_int(8)
      encoder.append_int(query_heads)
      encoder.append_int(kv_heads)
      encoder.append_int(head_dimension)
      encoder.append_u64(epsilon.reinterpret_as_uint64())
    }
  }
}

///|
fn append_model_graph(
  encoder : ExecutionDigestEncoder,
  model : @plan.ModelPlan,
) -> Unit {
  let operations = model.operations()
  encoder.append_int(operations.length())
  for operation in operations {
    encoder.append_int(operation.id().as_int())
    encoder.append_int(operation_kind_code(operation.kind()))
    encoder.append_int(operation.required_capability().as_int())
    encoder.append_i64(operation.workspace_bytes())
    append_operation_shape(encoder, operation.shape())
    encoder.append_int(operation.inputs().length())
    for input in operation.inputs() {
      match input {
        ModelInput(TokenIds) => encoder.append_int(1)
        Tensor(reference) => {
          encoder.append_int(2)
          encoder.append_int(reference.as_int())
        }
        Activation(reference) => {
          encoder.append_int(3)
          encoder.append_int(reference.as_int())
        }
      }
    }
    encoder.append_int(operation.outputs().length())
    for output in operation.outputs() {
      encoder.append_int(output.as_int())
    }
    match operation.kv_layer() {
      None => encoder.append_int(-1)
      Some(layer) => encoder.append_int(layer.as_int())
    }
    encoder.append_int(operation.runtime_inputs().length())
    for input in operation.runtime_inputs() {
      encoder.append_int(runtime_input_code(input))
    }
    match operation.persistent_state() {
      NoPersistentState => encoder.append_int(0)
      KvCacheReadWrite(layer) => {
        encoder.append_int(1)
        encoder.append_int(layer.as_int())
      }
    }
  }
  encoder.append_int(model.final_output().as_int())
}

///|
fn append_launch_sources(
  encoder : ExecutionDigestEncoder,
  launch_set : @tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet,
  contracts : ArrayView[
    @tensor_parallel_launch_contract.TensorParallelAotLaunchContract,
  ],
) -> Unit {
  let constants = launch_set.rank_constants()
  for
    value in [
      constants.version(),
      constants.rank(),
      constants.world_size(),
      constants.compute_major(),
      constants.compute_minor(),
      constants.kv_layout_version(),
      constants.tokens_per_page(),
    ] {
    encoder.append_int(value)
  }
  encoder.append_i64(constants.byte_count())
  encoder.append_i64(constants.alignment())
  encoder.append_int(contracts.length())
  for contract in contracts {
    encoder.append_profile(contract.profile())
    encoder.append_int(contract.operation_id().as_int())
    encoder.append_int(operation_kind_code(contract.operation_kind()))
    encoder.append_int(contract.capability().as_int())
    encoder.append_entry_point(contract.entry_point())
    encoder.append_dimensions(contract.dimensions())
    encoder.append_int(contract.input_width())
    encoder.append_int(contract.output_width())
    encoder.append_int(contract.inner_width())
    encoder.append_int(contract.query_head_count())
    encoder.append_int(contract.kv_head_count())
    encoder.append_int(contract.head_dimension())
    encoder.append_int(contract.operands().length())
    for operand in contract.operands() {
      encoder.append_operand_role(operand.role())
      encoder.append_i64(operand.byte_count())
      encoder.append_i64(operand.alignment())
    }
    match contract.collective_after() {
      NoCollective => encoder.append_int(0)
      CollectiveSumAllReduce(sequence~, width~) => {
        encoder.append_int(1)
        encoder.append_int(sequence)
        encoder.append_int(width)
      }
      CollectiveLastAxisAllGather(sequence~, full_width~, local_width~) => {
        encoder.append_int(2)
        encoder.append_int(sequence)
        encoder.append_int(full_width)
        encoder.append_int(local_width)
      }
    }
  }
}

///|
fn append_kv_layout(
  encoder : ExecutionDigestEncoder,
  layout : @device_layout.DeviceKvLayout,
) -> Unit {
  encoder.append_int(layout.geometry().layer_count())
  encoder.append_int(layout.geometry().kv_head_count())
  encoder.append_int(layout.geometry().head_dimension())
  encoder.append_int(layout.stable_layout_version())
  encoder.append_int(layout.tokens_per_page())
  encoder.append_int(layout.total_page_count())
  encoder.append_i64(layout.segment_alignment_bytes())
  encoder.append_i64(layout.page_logical_bytes())
  encoder.append_i64(layout.page_stride_bytes())
  encoder.append_i64(layout.component_stride_bytes())
  encoder.append_i64(layout.layer_stride_bytes())
  encoder.append_i64(layout.arena_bytes())
}

///|
fn append_device_plan(
  encoder : ExecutionDigestEncoder,
  plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
) -> Unit raise TensorParallelExecutionPlanError {
  encoder.append_int(plan.tensor_count())
  encoder.append_i64(plan.arena_bytes())
  encoder.append_i64(plan.materialized_bytes())
  for index in 0..<plan.tensor_count() {
    let reference = @plan.TensorRef::new(index) catch {
      _ => raise rejected(WeightRegion, operand=index)
    }
    guard plan.tensor(reference) is Some(view) else {
      raise rejected(WeightRegion, operand=index)
    }
    encoder.append_int(view.reference().as_int())
    encoder.append_int(placement_code(view.placement()))
    encoder.append_i64(view.full_rows())
    encoder.append_i64(view.full_columns())
    encoder.append_i64(view.local_row_start())
    encoder.append_i64(view.local_rows())
    encoder.append_i64(view.local_column_start())
    encoder.append_i64(view.local_columns())
    encoder.append_i64(view.region_offset_bytes())
    encoder.append_i64(view.region_byte_length())
  }
  encoder.append_int(plan.operation_count())
  for index in 0..<plan.operation_count() {
    let operation = @plan.OperationId::new(index) catch {
      _ => raise rejected(OperationOrder, operation=index)
    }
    guard plan.operation(operation) is Some(geometry) else {
      raise rejected(OperationOrder, operation=index)
    }
    encoder.append_int(geometry.operation().as_int())
    encoder.append_int(operation_kind_code(geometry.kind()))
    encoder.append_int(geometry.capability().as_int())
    encoder.append_int(geometry.input_width())
    encoder.append_int(geometry.output_width())
    encoder.append_int(geometry.inner_width())
    encoder.append_int(geometry.query_head_count())
    encoder.append_int(geometry.kv_head_count())
    encoder.append_int(geometry.head_dimension())
  }
  encoder.append_int(plan.collective_count())
  for index in 0..<plan.collective_count() {
    guard plan.collective(index) is Some(site) else {
      raise rejected(CollectiveCount, operand=index)
    }
    encoder.append_int(site.sequence().as_int())
    encoder.append_int(site.operation().as_int())
    encoder.append_int(
      match site.kind() {
        SumAllReduce => 1
        LastAxisAllGather => 2
      },
    )
    encoder.append_int(site.full_width())
    encoder.append_int(site.local_width())
  }
}

///|
fn append_artifacts(
  encoder : ExecutionDigestEncoder,
  artifacts : @artifact.KernelArtifactBundle,
) -> Unit {
  encoder.append_i64(artifacts.total_module_bytes())
  encoder.append_int(artifacts.module_count())
  for artifact_module in artifacts.modules() {
    encoder.append_string(artifact_module.digest().sha256())
    encoder.append_int(artifact_module.module_bytes().length())
  }
  encoder.append_int(artifacts.entry_point_count())
  for entry in artifacts.entry_points() {
    encoder.append_entry_point(entry.identity())
    encoder.append_int(entry.module_index())
    encoder.append_string(entry.function_symbol())
  }
}
