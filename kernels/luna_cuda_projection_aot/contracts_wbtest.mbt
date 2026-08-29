///|
fn projection_fixture_operands(
  operation : Int,
) -> Array[@launch_contract.AotOperand] raise {
  let counts = projection_fixture_operand(RuntimeInput(StepCounts), 20L, 4L)
  match operation {
    0 =>
      [
        counts,
        projection_fixture_operand(ExternalInput(TokenIds), 16L, 4L),
        projection_fixture_operand(
          WeightInput(@plan.TensorRef::new(0)),
          128L,
          16L,
        ),
        projection_fixture_operand(
          ActivationOutput(@plan.ActivationRef::new(0)),
          32L,
          16L,
        ),
      ]
    1 =>
      [
        counts,
        projection_fixture_operand(
          ActivationInput(@plan.ActivationRef::new(0)),
          32L,
          16L,
        ),
        projection_fixture_operand(
          WeightInput(@plan.TensorRef::new(1)),
          32L,
          16L,
        ),
        projection_fixture_operand(
          WeightInput(@plan.TensorRef::new(2)),
          16L,
          16L,
        ),
        projection_fixture_operand(
          WeightInput(@plan.TensorRef::new(3)),
          16L,
          16L,
        ),
        projection_fixture_operand(
          ActivationOutput(@plan.ActivationRef::new(1)),
          64L,
          16L,
        ),
      ]
    2 =>
      [
        counts,
        projection_fixture_operand(RuntimeInput(QueryPositions), 16L, 4L),
        projection_fixture_operand(
          ActivationInput(@plan.ActivationRef::new(1)),
          64L,
          16L,
        ),
        projection_fixture_operand(
          ActivationOutput(@plan.ActivationRef::new(2)),
          64L,
          16L,
        ),
      ]
    3 => {
      let layout = projection_fixture_layout()
      let layer = @plan.KvLayerId::new(0)
      [
        counts,
        projection_fixture_operand(RuntimeInput(QueryPositions), 16L, 4L),
        projection_fixture_operand(RuntimeInput(QueryRowOffsets), 12L, 4L),
        projection_fixture_operand(RuntimeInput(SequenceLengths), 8L, 4L),
        projection_fixture_operand(RuntimeInput(PageTableOffsets), 12L, 4L),
        projection_fixture_operand(RuntimeInput(PageTablePageIndices), 16L, 4L),
        projection_fixture_operand(
          ActivationInput(@plan.ActivationRef::new(2)),
          64L,
          16L,
        ),
        projection_fixture_operand(
          ActivationOutput(@plan.ActivationRef::new(3)),
          32L,
          16L,
        ),
        projection_fixture_operand(
          PersistentKvComponent(layer, Key),
          layout.component_stride_bytes(),
          layout.segment_alignment_bytes(),
        ),
        projection_fixture_operand(
          PersistentKvComponent(layer, Value),
          layout.component_stride_bytes(),
          layout.segment_alignment_bytes(),
        ),
      ]
    }
    4 =>
      [
        counts,
        projection_fixture_operand(
          ActivationInput(@plan.ActivationRef::new(3)),
          32L,
          16L,
        ),
        projection_fixture_operand(
          WeightInput(@plan.TensorRef::new(4)),
          32L,
          16L,
        ),
        projection_fixture_operand(
          ActivationOutput(@plan.ActivationRef::new(4)),
          32L,
          16L,
        ),
      ]
    5 =>
      [
        counts,
        projection_fixture_operand(
          ActivationInput(@plan.ActivationRef::new(4)),
          32L,
          16L,
        ),
        projection_fixture_operand(
          WeightInput(@plan.TensorRef::new(5)),
          64L,
          16L,
        ),
        projection_fixture_operand(
          WeightInput(@plan.TensorRef::new(6)),
          64L,
          16L,
        ),
        projection_fixture_operand(
          WeightInput(@plan.TensorRef::new(7)),
          64L,
          16L,
        ),
        projection_fixture_operand(
          ActivationOutput(@plan.ActivationRef::new(5)),
          32L,
          16L,
        ),
      ]
    6 =>
      [
        counts,
        projection_fixture_operand(
          ActivationInput(@plan.ActivationRef::new(5)),
          32L,
          16L,
        ),
        projection_fixture_operand(
          WeightInput(@plan.TensorRef::new(8)),
          128L,
          16L,
        ),
        projection_fixture_operand(
          ActivationOutput(@plan.ActivationRef::new(6)),
          128L,
          16L,
        ),
      ]
    _ => abort("unknown projection fixture operation")
  }
}

///|
fn projection_fixture_dimensions(
  operation : Int,
) -> @launch_contract.AotLaunchDimensions raise {
  let block = if operation == 3 { 1 } else { 256 }
  let grid_x = match operation {
    3 => 4
    _ => 1
  }
  @launch_contract.AotLaunchDimensions::new(
    grid_x~,
    grid_y=if operation == 3 { 2 } else { 1 },
    grid_z=1,
    block_x=block,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
}

///|
fn projection_fixture_contracts(
  qkv_query_weight_bytes? : Int64 = 32L,
  qkv_grid_x? : Int = 1,
) -> @launch_contract.PagedKvLaunchContractSet raise {
  let plan = projection_fixture_plan()
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=2,
    max_query_tokens=4,
    max_page_table_entries=4,
  )
  let inputs : Array[@launch_contract.PagedKvAotLaunchContractInput] = []
  for index, operation in plan.operations() {
    let operands = projection_fixture_operands(index)
    if index == 1 {
      operands[2] = projection_fixture_operand(
        operands[2].role(),
        qkv_query_weight_bytes,
        16L,
      )
    }
    let dimensions = if index == 1 && qkv_grid_x != 1 {
      @launch_contract.AotLaunchDimensions::new(
        grid_x=qkv_grid_x,
        grid_y=1,
        grid_z=1,
        block_x=256,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      )
    } else {
      projection_fixture_dimensions(index)
    }
    inputs.push(
      @launch_contract.PagedKvAotLaunchContractInput::new(
        profile_id=profile.id(),
        operation_id=operation.id(),
        entry_point=projection_fixture_entry(index),
        dimensions~,
        operands~,
      ),
    )
  }
  @launch_contract.admit_paged_graph(
    plan,
    projection_fixture_catalog(),
    projection_fixture_layout(),
    [profile],
    inputs,
    @launch_contract.LaunchContractLimits::new(
      max_profiles=1,
      max_contracts=8,
      max_operands_per_contract=12,
    ),
  )
}

///|
fn projection_fixture_compiler(
  reassociate? : Bool = false,
) -> @luna_specialization.LunaCompilerPolicy raise {
  @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256="2".repeat(64),
      major=13,
      minor=1,
      patch=0,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate~,
    max_registers=128,
  )
}

///|
fn projection_fixture_limits(
  max_source_bytes? : Int = 32768,
) -> ProjectionAotLimits raise {
  ProjectionAotLimits::new(max_source_bytes~, max_recipe_bytes=8192)
}

///|
fn projection_fixture_lowering(operation : Int) -> ProjectionCudaLowering raise {
  let plan = projection_fixture_plan()
  let contracts = projection_fixture_contracts()
  let contract = contracts.contracts()[operation]
  lower_projection_cuda_aot_candidate(
    plan.operations()[operation],
    contract.profile(),
    contract.entry_point().entry_point_id(),
    projection_fixture_target(),
    projection_fixture_compiler(),
    contract.operands(),
    projection_fixture_limits(),
  )
}
