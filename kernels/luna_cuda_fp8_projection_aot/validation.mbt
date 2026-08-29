///|
priv struct Fp8ProjectionParameters {
  family : Fp8ProjectionFamily
  input_width : Int
  output_width : Int
  intermediate_width : Int
}

///|
fn fp8_projection_parameters(
  operation : @plan.PlanOperation,
) -> Fp8ProjectionParameters raise Fp8ProjectionAotError {
  if operation.outputs().length() != 1 ||
    (operation.workspace_bytes() != 0L && operation.workspace_bytes() != 4L) ||
    operation.persistent_state() != NoPersistentState ||
    operation.runtime_inputs().length() != 1 ||
    operation.runtime_inputs()[0] != StepCounts {
    raise Mismatch(Operation)
  }
  match (operation.kind(), operation.shape()) {
    (QkvProjection, ProjectionShape(input_width~, output_width~)) => {
      let kv_width = (output_width - input_width) / 2
      if operation.inputs().length() != 4 ||
        input_width <= 0 ||
        output_width <= input_width ||
        (output_width - input_width) % 2 != 0 ||
        kv_width <= 0 {
        raise Mismatch(Operation)
      }
      {
        family: QkvProjection,
        input_width,
        output_width,
        intermediate_width: kv_width,
      }
    }
    (
      OutputProjection
      | LanguageModelHead,
      ProjectionShape(input_width~, output_width~),
    ) => {
      if operation.inputs().length() != 2 ||
        input_width <= 0 ||
        output_width <= 0 {
        raise Mismatch(Operation)
      }
      {
        family: if operation.kind() == OutputProjection {
          DenseProjection
        } else {
          LanguageModelHead
        },
        input_width,
        output_width,
        intermediate_width: 0,
      }
    }
    (GatedMlp, GatedMlpShape(hidden_size~, intermediate_size~)) => {
      if operation.inputs().length() != 4 ||
        hidden_size <= 0 ||
        intermediate_size <= 0 {
        raise Mismatch(Operation)
      }
      // One operation-level dynamic scale cannot honestly cover both the
      // gate/up input and the post-SiLU activation consumed by down_proj.
      // ABI v1 exposes one scalar cell, so W8A8 gated MLP fails closed.
      raise Unsupported(CompoundGatedActivationScale)
    }
    _ => raise Unsupported(OperationKind)
  }
}

///|
fn fp8_exact_execution(
  execution : @numeric_contract.OperationExecutionContract,
) -> Bool {
  execution.activation_input() == @numeric_contract.ComputeDType::bfloat16() &&
  execution.activation_compute() ==
  @numeric_contract.ComputeDType::f8_e4m3_finite() &&
  execution.activation_scale() ==
  @numeric_contract.ActivationScalePolicy::dynamic_per_tensor_f32_v1() &&
  execution.tensor_input() == @numeric_contract.ComputeDType::f8_e4m3_finite() &&
  execution.accumulator() == @numeric_contract.AccumulatorDType::f32() &&
  execution.output() == @numeric_contract.ComputeDType::bfloat16() &&
  execution.conversion() == @numeric_contract.NumericConversion::quantized_v1() &&
  execution.order() ==
  @numeric_contract.OperationOrderPolicy::strict_declared_order_v1()
}

///|
fn fp8_validate_schema(
  model_plan : @plan.ModelPlan,
  operation : @plan.PlanOperation,
) -> @numeric_contract.OperationExecutionDigest raise Fp8ProjectionAotError {
  let index = operation.id().as_int()
  guard model_plan.operations().get(index) is Some(exact) &&
    exact == operation &&
    model_plan.numeric_binding().operations().get(index) is Some(binding) &&
    binding.operation() == operation.id() else {
    raise Mismatch(ModelIdentity)
  }
  if !fp8_exact_execution(binding.execution()) {
    raise Unsupported(NumericExecution)
  }
  let mut weights = 0
  for input in operation.inputs() {
    match input {
      Activation(_) => ()
      ModelInput(_) => raise Mismatch(Operation)
      Tensor(reference) => {
        guard model_plan.tensor(reference) is Some(weight) &&
          weight.role() == ParameterTensor &&
          weight.storage().dtype() ==
          @numeric_contract.StorageDType::f8_e4m3_finite() &&
          weight.storage().layout() ==
          @numeric_contract.StorageLayout::dense_row_major_v1() &&
          weight.storage().encoding() ==
          @numeric_contract.StorageEncoding::scaled_v1() &&
          weight.storage().scale_granularity() ==
          @numeric_contract.ScaleGranularity::per_tensor() &&
          weight.scale() is Some(scale) &&
          model_plan.tensor(scale.tensor()) is Some(scale_tensor) &&
          scale_tensor.role() == ScaleMetadataTensor &&
          scale_tensor.storage().dtype() ==
          @numeric_contract.StorageDType::f32() &&
          scale_tensor.shape().rank() == 0 else {
          raise Mismatch(NumericSchema)
        }
        weights += 1
      }
    }
  }
  let expected = match operation.kind() {
    QkvProjection | GatedMlp => 3
    OutputProjection | LanguageModelHead => 1
    _ => 0
  }
  if weights != expected {
    raise Mismatch(NumericSchema)
  }
  binding.execution().digest()
}

///|
fn fp8_tensor_bytes(
  tensor : @plan.PlanTensor,
) -> Int64 raise Fp8ProjectionAotError {
  @numeric_contract.storage_byte_length(
    tensor.storage().dtype(),
    tensor.shape(),
  ) catch {
    _ => raise Invalid(OperandSize)
  }
}

///|
fn fp8_value_bytes(
  rows : Int,
  width : Int,
) -> Int64 raise Fp8ProjectionAotError {
  let value = rows.to_int64() * width.to_int64()
  if rows <= 0 || width <= 0 || value > 9223372036854775807L / 2L {
    raise Invalid(OperandSize)
  }
  value * 2L
}

///|
pub fn derive_fp8_projection_operands(
  model_plan : @plan.ModelPlan,
  operation~ : @plan.OperationId,
  profile : @launch_contract.PagedKernelProfile,
) -> ReadOnlyArray[@launch_contract.AotOperand] raise Fp8ProjectionAotError {
  guard model_plan.operations().get(operation.as_int()) is Some(op) &&
    op.id() == operation else {
    raise Mismatch(Operation)
  }
  let parameters = fp8_projection_parameters(op)
  ignore(fp8_validate_schema(model_plan, op))
  let result : Array[@launch_contract.AotOperand] = [
    @launch_contract.AotOperand::new(
      role=RuntimeInput(StepCounts),
      byte_count=20L,
      alignment=4L,
    ) catch {
      _ => raise Invalid(OperandSize)
    },
  ]
  for input in op.inputs() {
    match input {
      Activation(reference) =>
        result.push(
          @launch_contract.AotOperand::new(
            role=ActivationInput(reference),
            byte_count=fp8_value_bytes(
              profile.max_query_tokens(),
              parameters.input_width,
            ),
            alignment=16L,
          ) catch {
            _ => raise Invalid(OperandSize)
          },
        )
      Tensor(reference) => {
        guard model_plan.tensor(reference) is Some(weight) &&
          weight.scale() is Some(scale) &&
          model_plan.tensor(scale.tensor()) is Some(scale_tensor) else {
          raise Mismatch(NumericSchema)
        }
        result.push(
          @launch_contract.AotOperand::new(
            role=WeightInput(reference),
            byte_count=fp8_tensor_bytes(weight),
            alignment=16L,
          ) catch {
            _ => raise Invalid(OperandSize)
          },
        )
        result.push(
          @launch_contract.AotOperand::new(
            role=WeightScaleInput(reference, scale),
            byte_count=fp8_tensor_bytes(scale_tensor),
            alignment=16L,
          ) catch {
            _ => raise Invalid(OperandSize)
          },
        )
      }
      ModelInput(_) => raise Mismatch(Operation)
    }
  }
  for output in op.outputs() {
    result.push(
      @launch_contract.AotOperand::new(
        role=ActivationOutput(output),
        byte_count=fp8_value_bytes(
          profile.max_query_tokens(),
          parameters.output_width,
        ),
        alignment=16L,
      ) catch {
        _ => raise Invalid(OperandSize)
      },
    )
  }
  result.push(
    @launch_contract.AotOperand::new(
      role=Workspace,
      byte_count=4L,
      alignment=4L,
    ) catch {
      _ => raise Invalid(OperandSize)
    },
  )
  ReadOnlyArray::from_array(result)
}
