///|
fn probe_matrix_family(
  family : String,
  options : ArrayView[String],
) -> Unit raise {
  let choices : Array[@luna_projection_tile_compiler.ProjectionFoldChoice] = []
  let mut max_registers = 128
  match options {
    ["small-batch-output"] => {
      max_registers = 255
      choices.push(
        @luna_projection_tile_compiler.ProjectionFoldChoice::new(
          MatrixFold,
          stages=2,
          transfer_tiles=4,
          fragment_stages=1,
          row_tiles=4,
          output_tiles=4,
          hoist_addresses=true,
        ),
      )
      choices.push(
        @luna_projection_tile_compiler.ProjectionFoldChoice::new(
          MatrixFold,
          stages=2,
          transfer_tiles=4,
          fragment_stages=1,
          row_tiles=1,
          output_tiles=2,
          hoist_addresses=true,
          consumer_groups=2,
          maximum_token_rows=16,
        ),
      )
    }
    [
      stages,
      window,
      fragments,
      rows,
      columns,
      addresses,
      registers,
      consumers,
      bound,
    ] => {
      max_registers = probe_register_choice(registers)
      choices.push(
        @luna_projection_tile_compiler.ProjectionFoldChoice::new(
          MatrixFold,
          stages=probe_choice_int(stages),
          transfer_tiles=probe_choice_int(window),
          fragment_stages=probe_choice_int(fragments),
          row_tiles=probe_choice_int(rows),
          output_tiles=probe_choice_int(columns),
          hoist_addresses=addresses == "1",
          consumer_groups=probe_choice_int(consumers),
          maximum_token_rows=probe_choice_int(bound),
        ),
      )
    }
    [] => ()
    ["128" | "192" | "255" as registers] =>
      max_registers = probe_register_choice(registers)
    [stages, window, fragments] =>
      choices.push(
        @luna_projection_tile_compiler.ProjectionFoldChoice::new(
          MatrixFold,
          stages=probe_choice_int(stages),
          transfer_tiles=probe_choice_int(window),
          fragment_stages=probe_choice_int(fragments),
        ),
      )
    [stages, window, fragments, rows, columns, addresses, registers] => {
      max_registers = match registers {
        "128" => 128
        "192" => 192
        "255" => 255
        _ => fail("register envelope must be 128, 192 or 255")
      }
      guard addresses == "0" || addresses == "1" else {
        fail("address choice must be 0 or 1")
      }
      choices.push(
        @luna_projection_tile_compiler.ProjectionFoldChoice::new(
          MatrixFold,
          stages=probe_choice_int(stages),
          transfer_tiles=probe_choice_int(window),
          fragment_stages=probe_choice_int(fragments),
          row_tiles=probe_choice_int(rows),
          output_tiles=probe_choice_int(columns),
          hoist_addresses=addresses == "1",
        ),
      )
    }
    _ =>
      fail(
        "FAMILY [STAGES TRANSFER_TILES FRAGMENT_STAGES [ROW_TILES OUTPUT_TILES HOIST MAX_REGISTERS]]",
      )
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
      max_registers~,
    ),
    operands,
    @luna_cuda_projection_aot.ProjectionAotLimits::new(
      max_source_bytes=1048576,
      max_recipe_bytes=65536,
    ),
    fold_choices=choices,
  )
  println(@utf8.decode(compiled.source_bytes()))
  probe_row_variants(compiled)
  let dimensions = compiled.launch_dimensions()
  println("// PROBE_PRIMARY_GRID=\{dimensions.grid_x()}")
  println("// PROBE_PRIMARY_BLOCK=\{dimensions.block_x()}")
  println("// PROBE_PRIMARY_SHARED=\{dimensions.shared_memory_bytes()}")
  let schedule = compiled.tile_compilation().schedule()
  if schedule.matrix_pipeline() is Some(pipeline) {
    let work_grid = pipeline
      .workgroup_count(
        maximum_rows=if family == "head" { 32 } else { 2048 },
        output_columns=output_width,
        matrix_columns=16,
      )
      .unwrap()
    println("// PROBE_PRIMARY_ACTIVE_GRID=\{work_grid}")
  }
  println("// RECIPE-BEGIN")
  for line in @utf8.decode(compiled.recipe_bytes()).split("\n") {
    println("// \{line}")
  }
}
