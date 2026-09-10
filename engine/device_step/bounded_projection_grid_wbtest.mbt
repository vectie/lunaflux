///|
fn bounded_grid_dimensions(
  grid? : Int = 192,
) -> @launch_contract.AotLaunchDimensions raise {
  @launch_contract.AotLaunchDimensions::new(
    grid_x=grid,
    grid_y=1,
    grid_z=1,
    block_x=512,
    block_y=1,
    block_z=1,
    shared_memory_bytes=16384,
  )
}

///|
test "bounded compiler work grid is selected exactly without changing companions" {
  let base = qwen_fused_test_blueprint(qwen_fused_test_plan())
  let dimensions = bounded_grid_dimensions()
  let variants : Array[@artifact.KernelRowVariant] = []
  for bound in [8, 16, 32] {
    variants.push(
      @artifact.KernelRowVariant::new(
        maximum_token_rows=bound,
        function_symbol="matrix_rows\{bound}",
        dimensions~,
        bounded_work_grid_x=Some(if bound == 8 { 192 } else { 48 }),
      ),
    )
  }
  let step = {
    ..base.steps()[0],
    operation_kind: GatedMlp,
    function_symbol: "matrix",
    dimensions,
    operands: [
      {
        role: ActivationOutput(@plan.ActivationRef::new(0)),
        space: ActivationWorkspaceArena,
        offset_bytes: 0L,
        byte_count: 1024L * 1024L * 2L,
        alignment: 16L,
      },
      {
        role: Workspace,
        space: ActivationWorkspaceArena,
        offset_bytes: 0L,
        byte_count: 1024L * 3072L * 2L,
        alignment: 16L,
      },
    ],
    row_variants: ReadOnlyArray::from_array(variants),
  }
  let blueprint = {
    ..base,
    profile: @launch_contract.PagedKernelProfile::new(
      id=base.profile().id(),
      max_query_rows=32,
      max_query_tokens=1024,
      max_page_table_entries=base.profile().max_page_table_entries(),
    ),
    steps: [step],
  }
  let (functions, indices, companions) = collect_paged_functions(blueprint)
  let resources = empty_paged_resources()
  prepare_projection_variants(blueprint, functions, resources)
  for bound in [2, 7, 8, 9, 16, 17, 32] {
    let selected = select_projection_variant(
      resources.projection_variants,
      indices[0],
      bound,
    ).unwrap()
    let expected_bound = if bound <= 8 {
      8
    } else if bound <= 16 {
      16
    } else {
      32
    }
    assert_eq(
      functions[selected.function_index].symbol,
      "matrix_rows\{expected_bound}",
    )
    let launch = paged_bucket_launch(selected.launch, bound)
    assert_eq(launch.grid_x(), if expected_bound == 8 { 192 } else { 48 })
    assert_eq(launch.block_x(), dimensions.block_x())
    assert_eq(launch.shared_memory_bytes(), dimensions.shared_memory_bytes())
    let down = select_projection_variant(
      resources.projection_variants,
      companions[0],
      bound,
    ).unwrap()
    assert_eq(
      paged_bucket_launch(down.launch, bound).grid_x(),
      paged_projection_bucket_grid_x(
        ProjectionMlpDownGridX(dimensions, 1024),
        bound,
      ),
    )
  }
  assert_true(
    select_projection_variant(resources.projection_variants, indices[0], 1)
    is None,
  )
  assert_true(
    select_projection_variant(resources.projection_variants, indices[0], 33)
    is None,
  )
  for invalid in [0, -1, 193] {
    try
      paged_bucket_launch(ProjectionBoundedWorkGridX(dimensions, invalid), 16)
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid bounded launch accepted")
    }
  }
}

///|
test "bootstrap row encoding preserves legacy bytes and binds compiler grid" {
  let dimensions = bounded_grid_dimensions()
  let legacy : BootstrapEncoder = { bytes: [], maximum: 4096 }
  legacy.append_int(16)
  legacy.append_string("rows16")
  for value in [192, 1, 1, 512, 1, 1, 16384] {
    legacy.append_int(value)
  }
  legacy.append_string("companion-launch-v1")
  legacy.append_dimensions(dimensions)
  let encoded : BootstrapEncoder = { bytes: [], maximum: 4096 }
  encoded.append_projection_variant(
    @artifact.KernelRowVariant::new(
      maximum_token_rows=16,
      function_symbol="rows16",
      dimensions~,
      companion_dimensions=Some(dimensions),
    ),
  )
  assert_eq(encoded.bytes, legacy.bytes)
  let with_grid : BootstrapEncoder = { bytes: [], maximum: 4096 }
  with_grid.append_projection_variant(
    @artifact.KernelRowVariant::new(
      maximum_token_rows=16,
      function_symbol="rows16",
      dimensions~,
      companion_dimensions=Some(dimensions),
      bounded_work_grid_x=Some(48),
    ),
  )
  legacy.append_string("bounded-work-grid-x-v1")
  legacy.append_int(48)
  assert_eq(with_grid.bytes, legacy.bytes)
  assert_not_eq(with_grid.bytes, encoded.bytes)
}

///|
test "bootstrap refuses row launch policy not authenticated by artifacts" {
  let (base, artifacts) = bootstrap_fixture()
  let steps = base.steps()[:].to_owned()
  steps[0] = {
    ..steps[0],
    row_variants: [
      @artifact.KernelRowVariant::new(
        maximum_token_rows=1,
        function_symbol="unbound_rows1",
        dimensions=steps[0].dimensions(),
        bounded_work_grid_x=Some(1),
      ),
    ],
  }
  let changed = { ..base, steps: ReadOnlyArray::from_array(steps) }
  try
    admit_device_worker_bootstrap(
      changed,
      artifacts,
      device_ordinal=0,
      bootstrap_limits(),
    )
  catch {
    InvalidBootstrap(Artifact) => ()
    _ => fail("wrong bootstrap rejection")
  } noraise {
    _ => fail("unbound row launch accepted")
  }
}
