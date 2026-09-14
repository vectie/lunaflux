///|
fn probe_matrix_family(
  family : String,
  options : ArrayView[String],
) -> Unit raise {
  let choices : Array[@luna_projection_tile_compiler.ProjectionFoldChoice] = []
  match options {
    [] => ()
    [stages, window, fragments] =>
      choices.push(
        @luna_projection_tile_compiler.ProjectionFoldChoice::new(
          MatrixFold,
          stages=probe_choice_int(stages),
          transfer_tiles=probe_choice_int(window),
          fragment_stages=probe_choice_int(fragments),
        ),
      )
    _ => fail("FAMILY [STAGES TRANSFER_TILES FRAGMENT_STAGES]")
  }
  let (kind, input_width, widths) : (@plan.OperationKind, Int, Array[Int]) = match
    family {
    "qkv" => (QkvProjection, 1024, [2048, 1024, 1024])
    "output" => (OutputProjection, 2048, [1024])
    "head" => (LanguageModelHead, 1024, [151936])
    _ => fail("unknown family")
  }
  let output_width = widths.fold(init=0, (a, b) => a + b)
  let activation = @plan.ActivationRef::new(0)
  let output = @plan.ActivationRef::new(1)
  let inputs : Array[@plan.ValueRef] = [Activation(activation)]
  let runtime : Array[@plan.OperationRuntimeInput] = [StepCounts]
  let operands = [
    @launch_contract.AotOperand::new(
      role=RuntimeInput(StepCounts),
      byte_count=20L,
      alignment=4L,
    ),
  ]
  if family == "head" {
    runtime.push(QueryRowOffsets)
    operands.push(
      @launch_contract.AotOperand::new(
        role=RuntimeInput(QueryRowOffsets),
        byte_count=132L,
        alignment=4L,
      ),
    )
  }
  operands.push(
    @launch_contract.AotOperand::new(
      role=ActivationInput(activation),
      byte_count=2048L * input_width.to_int64() * 2L,
      alignment=16L,
    ),
  )
  for index, width in widths {
    let tensor = @plan.TensorRef::new(index)
    inputs.push(Tensor(tensor))
    operands.push(
      @launch_contract.AotOperand::new(
        role=WeightInput(tensor),
        byte_count=width.to_int64() * input_width.to_int64() * 2L,
        alignment=16L,
      ),
    )
  }
  operands.push(
    @launch_contract.AotOperand::new(
      role=ActivationOutput(output),
      byte_count=2048L * output_width.to_int64() * 2L,
      alignment=16L,
    ),
  )
  let operation = @plan.PlanOperation::new_with_runtime_inputs(
    id=@plan.OperationId::new(1),
    kind~,
    inputs~,
    outputs=[output],
    shape=ProjectionShape(input_width~, output_width~),
    required_capability=if family == "qkv" {
      @plan.KernelCapabilityId::qkv_projection()
    } else {
      @plan.KernelCapabilityId::dense_projection()
    },
    workspace_bytes=0L,
    runtime_inputs=runtime,
  )
  let compiled = @luna_cuda_projection_aot.lower_reusable_projection_cuda_aot_candidate(
    operation,
    @launch_contract.PagedKernelProfile::new(
      id=@launch_contract.KernelProfileId::from_stable_int(1),
      max_query_rows=32,
      max_query_tokens=2048,
      max_page_table_entries=8192,
    ),
    @catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    @luna_specialization.LunaCompilerPolicy::new(
      identity=@luna_specialization.LunaCompilerIdentity::new(
        toolchain_digest_sha256="0".repeat(64),
        major=13,
        minor=1,
        patch=115,
      ),
      optimization_level=3,
      fused_multiply_add=false,
      reassociate=false,
      max_registers=128,
    ),
    operands,
    @luna_cuda_projection_aot.ProjectionAotLimits::new(
      max_source_bytes=1048576,
      max_recipe_bytes=65536,
    ),
    fold_choices=choices,
  )
  println(@utf8.decode(compiled.source_bytes()))
  println("// RECIPE-BEGIN")
  for line in @utf8.decode(compiled.recipe_bytes()).split("\n") {
    println("// \{line}")
  }
}
