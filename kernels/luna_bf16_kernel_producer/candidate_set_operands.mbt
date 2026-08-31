///|
fn candidate_checked_mul(
  left : Int64,
  right : Int64,
) -> Int64 raise LunaBf16KernelProducerError {
  if left <= 0L || right <= 0L || left > 9223372036854775807L / right {
    raise Invalid(CandidateSet)
  }
  left * right
}

///|
fn candidate_matrix_bytes(
  rows : Int,
  columns : Int,
) -> Int64 raise LunaBf16KernelProducerError {
  candidate_checked_mul(
    candidate_checked_mul(rows.to_int64(), columns.to_int64()),
    2L,
  )
}

///|
/// Qwen-sized tile MLPs use one BF16 intermediate row as decode scratch. The
/// semantic plan remains unchanged; the selected kernel binding owns this
/// backend-specific temporary region inside the plan's existing workspace
/// envelope.
fn candidate_workspace_bytes(
  operation : @plan.PlanOperation,
  profile : @launch_contract.PagedKernelProfile,
) -> Int64 {
  match (operation.kind(), operation.shape()) {
    (GatedMlp, GatedMlpShape(hidden_size~, intermediate_size~)) if hidden_size >
      0 &&
      hidden_size % 256 == 0 &&
      intermediate_size > 0 &&
      intermediate_size % 256 == 0 &&
      intermediate_size <= 3840 =>
      intermediate_size.to_int64() * profile.max_query_tokens().to_int64() * 2L
    _ => operation.workspace_bytes()
  }
}

///|
fn candidate_runtime_bytes(
  input : @plan.OperationRuntimeInput,
  profile : @launch_contract.PagedKernelProfile,
) -> Int64 {
  match input {
    StepCounts => 20L
    QueryPositions => profile.max_query_tokens().to_int64() * 4L
    QueryRowOffsets | PageTableOffsets =>
      (profile.max_query_rows().to_int64() + 1L) * 4L
    SequenceLengths => profile.max_query_rows().to_int64() * 4L
    PageTablePageIndices => profile.max_page_table_entries().to_int64() * 4L
  }
}

///|
fn candidate_activation_width(
  model_plan : @plan.ModelPlan,
  reference : @plan.ActivationRef,
) -> Int raise LunaBf16KernelProducerError {
  let mut width : Int? = None
  for operation in model_plan.operations() {
    for output in operation.outputs() {
      if output == reference {
        if width is Some(_) {
          raise Invalid(CandidateSet)
        }
        width = Some(
          match operation.shape() {
            EmbeddingShape(width~, ..) => width
            NormalizationShape(width~, ..) => width
            QkNormalizationShape(query_heads~, kv_heads~, head_dimension~, ..) =>
              (query_heads + 2 * kv_heads) * head_dimension
            ProjectionShape(output_width~, ..) => output_width
            RotaryShape(query_heads~, kv_heads~, head_dimension~, ..) =>
              (query_heads + 2 * kv_heads) * head_dimension
            AttentionShape(query_heads~, head_dimension~, ..) =>
              query_heads * head_dimension
            ElementwiseShape(width~) => width
            GatedMlpShape(hidden_size~, ..) => hidden_size
          },
        )
      }
    }
  }
  guard width is Some(value) else { raise Invalid(CandidateSet) }
  value
}

///|
fn candidate_weight_bytes(
  model_plan : @plan.ModelPlan,
  operation : @plan.PlanOperation,
  weight_index : Int,
) -> Int64 raise LunaBf16KernelProducerError {
  match operation.shape() {
    EmbeddingShape(vocabulary_size~, width~) =>
      if weight_index == 0 {
        candidate_matrix_bytes(vocabulary_size, width)
      } else {
        raise Invalid(CandidateSet)
      }
    NormalizationShape(width~, ..) =>
      if weight_index == 0 {
        candidate_matrix_bytes(1, width)
      } else {
        raise Invalid(CandidateSet)
      }
    QkNormalizationShape(head_dimension~, ..) =>
      if weight_index < 2 {
        candidate_matrix_bytes(1, head_dimension)
      } else {
        raise Invalid(CandidateSet)
      }
    ProjectionShape(input_width~, output_width~) =>
      if operation.kind() == QkvProjection {
        if weight_index >= 3 {
          raise Invalid(CandidateSet)
        }
        guard operation.inputs()[weight_index + 1] is Tensor(reference) else {
          raise Invalid(CandidateSet)
        }
        guard model_plan.numeric_binding().tensors().get(reference.as_int())
          is Some(tensor) else {
          raise Invalid(CandidateSet)
        }
        let shape = tensor.shape()
        if tensor.reference().as_int() != reference.as_int() ||
          tensor.storage().dtype() != @numeric_contract.StorageDType::bfloat16() ||
          shape.rank() != 2 ||
          shape.dimension(1) != Some(input_width.to_int64()) {
          raise Invalid(CandidateSet)
        }
        candidate_checked_mul(shape.element_count(), 2L)
      } else if weight_index == 0 {
        candidate_matrix_bytes(output_width, input_width)
      } else {
        raise Invalid(CandidateSet)
      }
    GatedMlpShape(hidden_size~, intermediate_size~) =>
      match weight_index {
        0 | 1 => candidate_matrix_bytes(intermediate_size, hidden_size)
        2 => candidate_matrix_bytes(hidden_size, intermediate_size)
        _ => raise Invalid(CandidateSet)
      }
    _ => raise Invalid(CandidateSet)
  }
}

///|
fn candidate_operand(
  role : @launch_contract.AotOperandRole,
  byte_count : Int64,
  alignment : Int64,
) -> @launch_contract.AotOperand raise LunaBf16KernelProducerError {
  @launch_contract.AotOperand::new(role~, byte_count~, alignment~) catch {
    _ => raise Invalid(CandidateSet)
  }
}

///|
fn derive_candidate_operands(
  model_plan : @plan.ModelPlan,
  operation : @plan.PlanOperation,
  profile : @launch_contract.PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
) -> Array[@launch_contract.AotOperand] raise LunaBf16KernelProducerError {
  let result : Array[@launch_contract.AotOperand] = []
  for runtime in operation.runtime_inputs() {
    result.push(
      candidate_operand(
        RuntimeInput(runtime),
        candidate_runtime_bytes(runtime, profile),
        4L,
      ),
    )
  }
  let mut weight_index = 0
  for input in operation.inputs() {
    match input {
      ModelInput(reference) =>
        result.push(
          candidate_operand(
            ExternalInput(reference),
            profile.max_query_tokens().to_int64() * 4L,
            4L,
          ),
        )
      Tensor(reference) => {
        result.push(
          candidate_operand(
            WeightInput(reference),
            candidate_weight_bytes(model_plan, operation, weight_index),
            16L,
          ),
        )
        weight_index += 1
      }
      Activation(reference) =>
        result.push(
          candidate_operand(
            ActivationInput(reference),
            candidate_matrix_bytes(
              profile.max_query_tokens(),
              candidate_activation_width(model_plan, reference),
            ),
            16L,
          ),
        )
    }
  }
  for output in operation.outputs() {
    result.push(
      candidate_operand(
        ActivationOutput(output),
        candidate_matrix_bytes(
          profile.max_query_tokens(),
          candidate_activation_width(model_plan, output),
        ),
        16L,
      ),
    )
  }
  match operation.persistent_state() {
    NoPersistentState => ()
    KvCacheReadWrite(layer) => {
      result.push(
        candidate_operand(
          PersistentKvComponent(layer, Key),
          layout.component_stride_bytes(),
          layout.segment_alignment_bytes(),
        ),
      )
      result.push(
        candidate_operand(
          PersistentKvComponent(layer, Value),
          layout.component_stride_bytes(),
          layout.segment_alignment_bytes(),
        ),
      )
    }
  }
  let workspace_bytes = candidate_workspace_bytes(operation, profile)
  if workspace_bytes > 0L {
    result.push(candidate_operand(Workspace, workspace_bytes, 16L))
  }
  result
}
