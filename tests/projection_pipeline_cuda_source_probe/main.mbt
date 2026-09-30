///|
/// Source-only diagnostic through the production compiler; no artifact
/// admission or deployment receipt is created by this fixture.
fn main raise {
  if @env.args() is [_, "qkv" | "output" | "head" as family, .. options] {
    probe_matrix_family(family, options)
    return
  }
  let (sibling_ownership, options) : (
    @luna_projection_tile_compiler.ProjectionSiblingOwnership,
    ArrayView[String],
  ) = match @env.args() {
    [_, "gate-retained", .. options] => (RetainSiblingProducer, options)
    [_, "gate-coowned", .. options] => (CoownedSiblingValues, options)
    [_, "gate-shared", .. options] => (SplitSiblingPlanes, options)
    [_, .. options] => (SplitSiblingPlanes, options)
    _ => fail("missing diagnostic executable name")
  }
  let choices : Array[@luna_projection_tile_compiler.ProjectionFoldChoice] = []
  let (sibling_transport, options) : (
    @luna_projection_tile_compiler.ProjectionSiblingTransport,
    ArrayView[String],
  ) = match options {
    ["transport-segmented", .. rest] => (SegmentedSiblingTransport, rest)
    ["transport-independent", .. rest] => (IndependentSiblingTransport, rest)
    _ => (BaselineSiblingTransport, options)
  }
  let mut max_registers = 128
  match options {
    ["small-batch-gate"] => {
      max_registers = 255
      choices.push(
        @luna_projection_tile_compiler.ProjectionFoldChoice::new(
          IntermediateFold,
          stages=2,
          transfer_tiles=4,
          fragment_stages=1,
          row_tiles=8,
          hoist_addresses=true,
        ),
      )
      choices.push(
        @luna_projection_tile_compiler.ProjectionFoldChoice::new(
          SiblingFold,
          stages=2,
          transfer_tiles=4,
          fragment_stages=1,
          row_tiles=1,
          output_tiles=2,
          consumer_groups=2,
          maximum_token_rows=16,
        ),
      )
    }
    [
      role,
      stages,
      window,
      fragments,
      rows,
      columns,
      hoist,
      consumers,
      registers,
      bound,
    ] => {
      max_registers = probe_register_choice(registers)
      let role : @luna_projection_tile_compiler.ProjectionFoldRole = match
        role {
        "sibling" => SiblingFold
        "intermediate" => IntermediateFold
        _ => fail("fold role")
      }
      choices.push(
        @luna_projection_tile_compiler.ProjectionFoldChoice::new(
          role,
          stages=probe_choice_int(stages),
          transfer_tiles=probe_choice_int(window),
          fragment_stages=probe_choice_int(fragments),
          row_tiles=probe_choice_int(rows),
          output_tiles=probe_choice_int(columns),
          hoist_addresses=hoist == "1",
          consumer_groups=probe_choice_int(consumers),
          maximum_token_rows=probe_choice_int(bound),
        ),
      )
    }
    [] => ()
    ["128" | "192" | "255" as registers] =>
      max_registers = probe_register_choice(registers)
    [
      "sibling",
      stages,
      window,
      fragments,
      rows,
      columns,
      hoist,
      consumers,
      registers,
    ] => {
      max_registers = probe_register_choice(registers)
      choices.push(
        @luna_projection_tile_compiler.ProjectionFoldChoice::new(
          SiblingFold,
          stages=probe_choice_int(stages),
          transfer_tiles=probe_choice_int(window),
          fragment_stages=probe_choice_int(fragments),
          row_tiles=probe_choice_int(rows),
          output_tiles=probe_choice_int(columns),
          hoist_addresses=hoist == "1",
          consumer_groups=probe_choice_int(consumers),
        ),
      )
    }
    ["sibling", stages, window, fragments, rows, columns, hoist, registers] => {
      max_registers = probe_register_choice(registers)
      choices.push(
        @luna_projection_tile_compiler.ProjectionFoldChoice::new(
          SiblingFold,
          stages=probe_choice_int(stages),
          transfer_tiles=probe_choice_int(window),
          fragment_stages=probe_choice_int(fragments),
          row_tiles=probe_choice_int(rows),
          output_tiles=probe_choice_int(columns),
          hoist_addresses=hoist == "1",
        ),
      )
    }
    ["intermediate", stages, window, fragments, rows, hoist, registers] => {
      max_registers = probe_register_choice(registers)
      choices.push(
        @luna_projection_tile_compiler.ProjectionFoldChoice::new(
          IntermediateFold,
          stages=probe_choice_int(stages),
          transfer_tiles=probe_choice_int(window),
          fragment_stages=probe_choice_int(fragments),
          row_tiles=probe_choice_int(rows),
          hoist_addresses=hoist == "1",
        ),
      )
    }
    [role, stages, window, fragments] => {
      let role : @luna_projection_tile_compiler.ProjectionFoldRole = match
        role {
        "sibling" => SiblingFold
        "intermediate" => IntermediateFold
        _ => fail("expected sibling or intermediate")
      }
      choices.push(
        @luna_projection_tile_compiler.ProjectionFoldChoice::new(
          role,
          stages=probe_choice_int(stages),
          transfer_tiles=probe_choice_int(window),
          fragment_stages=probe_choice_int(fragments),
        ),
      )
    }
    [stages, window, fragments] =>
      for role in [@luna_projection_tile_compiler.SiblingFold, IntermediateFold] {
        choices.push(
          @luna_projection_tile_compiler.ProjectionFoldChoice::new(
            role,
            stages=probe_choice_int(stages),
            transfer_tiles=probe_choice_int(window),
            fragment_stages=probe_choice_int(fragments),
          ),
        )
      }
    _ => fail("expected optional STAGES TRANSFER_TILES FRAGMENT_STAGES")
  }
  let activation = @plan.ActivationRef::new(0)
  let output = @plan.ActivationRef::new(1)
  let tensors = [0, 1, 2].map(i => @plan.TensorRef::new(i))
  let operation = @plan.PlanOperation::new_with_runtime_inputs(
    id=@plan.OperationId::new(1),
    kind=GatedMlp,
    inputs=[
      Activation(activation),
      Tensor(tensors[0]),
      Tensor(tensors[1]),
      Tensor(tensors[2]),
    ],
    outputs=[output],
    shape=GatedMlpShape(hidden_size=1024, intermediate_size=3072),
    required_capability=@plan.KernelCapabilityId::gated_mlp(),
    workspace_bytes=0L,
    runtime_inputs=[StepCounts],
  )
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=32,
    max_query_tokens=2048,
    max_page_table_entries=8192,
  )
  let operands = [
    @launch_contract.AotOperand::new(
      role=RuntimeInput(StepCounts),
      byte_count=20L,
      alignment=4L,
    ),
    @launch_contract.AotOperand::new(
      role=ActivationInput(activation),
      byte_count=2048L * 1024L * 2L,
      alignment=16L,
    ),
    @launch_contract.AotOperand::new(
      role=WeightInput(tensors[0]),
      byte_count=3072L * 1024L * 2L,
      alignment=16L,
    ),
    @launch_contract.AotOperand::new(
      role=WeightInput(tensors[1]),
      byte_count=3072L * 1024L * 2L,
      alignment=16L,
    ),
    @launch_contract.AotOperand::new(
      role=WeightInput(tensors[2]),
      byte_count=3072L * 1024L * 2L,
      alignment=16L,
    ),
    @launch_contract.AotOperand::new(
      role=ActivationOutput(output),
      byte_count=2048L * 1024L * 2L,
      alignment=16L,
    ),
    @launch_contract.AotOperand::new(
      role=Workspace,
      byte_count=2048L * 3072L * 2L,
      alignment=16L,
    ),
  ]
  let compiled = @luna_cuda_projection_aot.lower_reusable_projection_cuda_aot_candidate(
    operation,
    profile,
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
    sibling_ownership~,
    sibling_transport~,
  )
  println(@utf8.decode(compiled.source_bytes()))
  probe_row_variants(compiled)
  println("// RECIPE-BEGIN")
  for line in @utf8.decode(compiled.recipe_bytes()).split("\n") {
    println("// \{line}")
  }
  let launch = compiled.launch_dimensions()
  println(
    "// probe_primary_launch=\{launch.grid_x()},\{launch.block_x()},\{launch.shared_memory_bytes()}",
  )
  println("// PROBE_PRIMARY_GRID=\{launch.grid_x()}")
  println("// PROBE_PRIMARY_BLOCK=\{launch.block_x()}")
  println("// PROBE_PRIMARY_SHARED=\{launch.shared_memory_bytes()}")
  if compiled.companion_dimensions() is Some(companion) {
    println(
      "// probe_companion_launch=\{companion.grid_x()},\{companion.block_x()},\{companion.shared_memory_bytes()}",
    )
    println("// PROBE_DOWN_GRID=\{companion.grid_x()}")
    println("// PROBE_DOWN_BLOCK=\{companion.block_x()}")
    println("// PROBE_DOWN_SHARED=\{companion.shared_memory_bytes()}")
  }
  let physical = compiled.physical_compilation()
  for
    (role, pipeline) in [
      ("sibling", physical.sibling),
      ("intermediate", physical.intermediate),
    ] {
    if pipeline is Some(pipeline) {
      let columns = pipeline.schedule.output_tiles() *
        pipeline.schedule.tile_width()
      let width = if role == "sibling" { 3072 } else { 1024 }
      let active_grid = (2048 + pipeline.schedule.row_width() - 1) /
        pipeline.schedule.row_width() *
        (width / columns)
      println("// probe_\{role}_active_grid=\{active_grid}")
    }
  }
}

///|
fn probe_register_choice(value : String) -> Int raise {
  match value {
    "128" => 128
    "192" => 192
    "255" => 255
    _ => fail("expected register envelope 128, 192, or 255")
  }
}

///|
fn probe_choice_int(value : String) -> Int raise {
  match value {
    "1" => 1
    "2" => 2
    "3" => 3
    "4" => 4
    "8" => 8
    "16" => 16
    _ => fail("unsupported finite scheduling choice")
  }
}

///|
fn probe_row_variants(
  compiled : @luna_cuda_projection_aot.ProjectionCudaLowering,
) -> Unit {
  for variant in compiled.row_variants() {
    let rows = variant.maximum_token_rows()
    let geometry = variant.dimensions()
    println(
      "// ROWS\{rows}_PRIMARY_GRID=\{variant.bounded_work_grid_x().unwrap_or(geometry.grid_x())}",
    )
    println("// ROWS\{rows}_PRIMARY_BLOCK=\{geometry.block_x()}")
    println("// ROWS\{rows}_PRIMARY_SHARED=\{geometry.shared_memory_bytes()}")
    if variant.companion_dimensions() is Some(geometry) {
      println("// ROWS\{rows}_DOWN_GRID=\{geometry.grid_x()}")
      println("// ROWS\{rows}_DOWN_BLOCK=\{geometry.block_x()}")
      println("// ROWS\{rows}_DOWN_SHARED=\{geometry.shared_memory_bytes()}")
    }
  }
}
