///|
test "independent reuse geometry preserves math and hoists only addresses" {
  let parameters : ProjectionParameters = {
    family: DenseProjection,
    input_width: 256,
    output_width: 512,
    query_width: 0,
    intermediate_width: 0,
  }
  let runtime = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  )
  for maximum_tokens in [8, 128, 1024] {
    let request = @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
      dispatch=runtime.dispatch_plan(1024),
      maximum_rows=8,
      maximum_tokens~,
      demand=AllTokenRows,
      single_row_numerics=AllowDeterministicDotTree,
    )
    let base = @luna_projection_tile_compiler.compile_projection_tile(request)
    let candidate = @luna_projection_tile_compiler.compile_projection_tile(
      request.with_fold_pipeline(
        MatrixFold,
        stages=2,
        transfer_tiles=2,
        fragment_stages=1,
        row_tiles=8,
        output_tiles=8,
        hoist_addresses=true,
      ),
    )
    let bounded = projection_row_variant_compilation(
      parameters,
      runtime.dispatch_plan(1024),
      8,
      maximum_tokens,
      fold_choices=[
        @luna_projection_tile_compiler.ProjectionFoldChoice::new(
          MatrixFold,
          stages=2,
          transfer_tiles=2,
          fragment_stages=1,
          row_tiles=8,
          output_tiles=8,
          hoist_addresses=true,
        ),
      ],
    )
    assert_eq(
      bounded.schedule().matrix_pipeline(),
      candidate.schedule().matrix_pipeline(),
    )
    assert_eq(base.program().digest(), candidate.program().digest())
    let physical = projection_refine_physical(candidate)
    let profile = @launch_contract.PagedKernelProfile::new(
      id=@launch_contract.KernelProfileId::from_stable_int(1),
      max_query_rows=8,
      max_query_tokens=maximum_tokens,
      max_page_table_entries=8192,
    )
    let refined = projection_refine_fold_dimensions(
      parameters,
      candidate,
      profile,
      projection_dimensions(profile, parameters, runtime.dispatch_plan(1024)),
    )
    let matrix_work = candidate
      .schedule()
      .matrix_pipeline()
      .unwrap()
      .workgroup_count(
        maximum_rows=maximum_tokens,
        output_columns=512,
        matrix_columns=16,
      )
      .unwrap()
    assert_eq(
      refined.grid_x(),
      Int::max(
        matrix_work,
        512 / candidate.schedule().output_tiles_per_workgroup(),
      ),
    )
    let pipeline = physical.matrix.unwrap()
    assert_eq(
      pipeline.schedule.row_width(),
      if maximum_tokens == 8 {
        16
      } else {
        128
      },
    )
    let source = projection_matrix_pipeline_source(parameters, physical, 8)
    let parts = source.split("auto stage_operands").to_array()
    assert_eq(parts.length(), 2)
    assert_true(parts[0].contains("input_address_base"))
    assert_true(parts[0].contains("__cvta_generic_to_shared"))
    assert_false(
      parts[1].split("float c[").to_array()[0].contains(
        "__cvta_generic_to_shared",
      ),
    )
    assert_true(parts[1].contains("input_address_base[address_slot] + k_base"))
    assert_true(parts[1].contains("__syncthreads"))
    // The pure affine address plan still covers each physical copy vector once.
    for operand in [0, 1] {
      let addresses = pipeline.producer_addresses(operand).unwrap()
      let (vectors, workers, slots) = addresses.geometry()
      let seen : Array[Int] = []
      for worker in 0..<workers {
        for slot in 0..<slots {
          if addresses.at(worker, slot) is Some((_, _, offset)) {
            assert_false(seen.contains(offset))
            seen.push(offset)
          }
        }
      }
      assert_eq(seen.length(), vectors)
    }
  }
}

///|
test "intermediate reuse lifts transport without moving loads before materialization" {
  let parameters = sibling_resource_test_parameters()
  let dispatch = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  ).dispatch_plan(1024)
  let compiled = @luna_projection_tile_compiler.compile_projection_tile(
    @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
      dispatch~,
      maximum_rows=32,
      maximum_tokens=1024,
      demand=AllTokenRows,
    ).with_fold_pipeline(
      IntermediateFold,
      stages=2,
      transfer_tiles=4,
      fragment_stages=1,
      row_tiles=8,
      hoist_addresses=true,
    ),
  )
  let physical = projection_refine_physical(compiled).intermediate.unwrap()
  let source = projection_fold_pipeline_source(
    physical, "workspace + row_start * 512", "down_weight", 512, 256,
  )
  assert_eq(physical.schedule.row_width(), 128)
  let pieces = source.split("auto stage_operands").to_array()
  assert_true(pieces[0].contains("pipeline_a"))
  assert_false(
    pieces[1].split("stage_operands(0, 0)").to_array()[0].contains(
      "__cvta_generic_to_shared",
    ),
  )
  assert_true(pieces[1].contains("cp.async"))
  assert_eq(source.split("__syncthreads").count(), 3)
}
