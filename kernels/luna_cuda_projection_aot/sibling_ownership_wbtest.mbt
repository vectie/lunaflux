///|
test "retained sibling producer imports only the peer result" {
  let parameters = sibling_resource_test_parameters()
  let runtime = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  )
  let compiled = sibling_resource_test_compilation(runtime.dispatch_plan(1024))
  let physical = projection_refine_physical(
    compiled,
    sibling_ownership=RetainSiblingProducer,
  )
  let epilogue = physical.sibling_epilogue.unwrap()
  assert_eq(epilogue.result_planes(), 1)
  let source = projection_sibling_fold_pipeline_source(
    physical.sibling.unwrap(),
    parameters,
    physical.materialization,
    epilogue~,
  )
  assert_true(
    source.contains(
      "float *peer_tiles = reinterpret_cast<float *>(sibling_storage);",
    ),
  )
  assert_true(source.contains("const float gate = c[row][value];"))
  assert_false(source.contains("gate_tiles"))
  assert_false(source.contains("sibling_output_bytes"))
  assert_false(source.contains("for (int element = threadIdx.x;"))
  assert_true(source.contains("(gate / (1.0f + expf(-gate))) * up"))
  let write = source.find("peer_tiles[((consumer_row + row)").unwrap()
  let publish = source[write:].find("__syncthreads();").unwrap() + write
  let read = source.find("const float up = peer_tiles").unwrap()
  let retire = source[read:].find("__syncthreads();").unwrap() + read
  assert_true(write < publish && publish < read && read < retire)
}

///|
test "coowned sibling fold retains both accumulators through pointwise map" {
  let parameters = sibling_resource_test_parameters()
  let runtime = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  )
  let compiled = sibling_resource_test_compilation(runtime.dispatch_plan(1024))
  let physical = projection_refine_physical(
    compiled,
    sibling_ownership=CoownedSiblingValues,
  )
  let sibling = physical.sibling.unwrap()
  assert_false(sibling.sibling_reuse)
  let epilogue = physical.sibling_epilogue.unwrap()
  assert_eq(epilogue.result_planes(), 0)
  assert_eq(epilogue.effects(), None)
  let source = projection_sibling_fold_pipeline_source(
    sibling,
    parameters,
    physical.materialization,
    epilogue~,
  )
  assert_true(source.contains("float cg[1][8] = {}, cu[1][8] = {};"))
  assert_true(
    source.contains("const float gate = cg[row][item], up = cu[row][item];"),
  )
  assert_false(source.contains("peer_tiles"))
  assert_false(source.contains("gate_tiles"))
  assert_false(source.contains("sibling_output_bytes"))
  let fold_end = source.find("const float gate = cg[row][item]").unwrap()
  assert_false(source[fold_end:].contains("__syncthreads();"))
  assert_true(source.contains("(gate / (1.0f + expf(-gate))) * up"))
}

///|
test "all sibling ownership maps cover active tails exactly once" {
  let parameters = sibling_resource_test_parameters()
  let runtime = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  )
  let compiled = sibling_resource_test_compilation(runtime.dispatch_plan(1024))
  for
    ownership in [
      @luna_projection_tile_compiler.SplitSiblingPlanes,
      RetainSiblingProducer,
      CoownedSiblingValues,
    ] {
    let physical = projection_refine_physical(
      compiled,
      sibling_ownership=ownership,
    )
    let epilogue = physical.sibling_epilogue.unwrap()
    let distribution = epilogue.distribution()
    let row_tiles = physical.sibling.unwrap().schedule.row_width() / 16
    let columns = physical.sibling.unwrap().schedule.output_tiles() * 16
    for active_rows in [1, 8, 16, 17, 31, 33, 63, 64] {
      let stores = Array::make(active_rows * columns, 0)
      for consumer in 0..<distribution.consumer_groups() {
        let row_group = consumer / distribution.output_groups()
        let column_group = consumer % distribution.output_groups()
        for row in 0..<distribution.rows_per_group() {
          let row_tile = row_group * distribution.rows_per_group() + row
          assert_true(row_tile < row_tiles)
          for lane in 0..<32 {
            for item in 0..<8 {
              let output_row = row_tile * 16 + lane / 4 + item % 4 / 2 * 8
              let output_column = column_group * 16 +
                lane % 4 * 2 +
                item % 2 +
                item / 4 * 8
              if output_row < active_rows {
                stores[output_row * columns + output_column] += 1
              }
            }
          }
        }
      }
      for count in stores {
        assert_eq(count, 1)
      }
    }
  }
}

///|
test "coowned sibling geometry shares larger independent row and output product" {
  let parameters = sibling_resource_test_parameters()
  let runtime = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  )
  let dispatch = runtime.dispatch_plan(1024)
  let choice = @luna_projection_tile_compiler.ProjectionFoldChoice::new(
    SiblingFold,
    stages=2,
    transfer_tiles=2,
    fragment_stages=1,
    row_tiles=4,
    output_tiles=8,
  )
  let compilation = projection_row_variant_compilation(
    parameters,
    dispatch,
    32,
    1024,
    fold_choices=[choice],
  )
  let physical = projection_refine_physical(
    compilation,
    sibling_ownership=CoownedSiblingValues,
  )
  let sibling = physical.sibling.unwrap()
  assert_eq(sibling.schedule.row_width(), 64)
  assert_eq(sibling.schedule.output_tiles(), 8)
  assert_eq(sibling.distribution.rows_per_group(), 2)
  assert_eq(sibling.distribution.consumer_groups(), 16)
  let source = projection_sibling_fold_pipeline_source(
    sibling,
    parameters,
    physical.materialization,
    epilogue=physical.sibling_epilogue.unwrap(),
  )
  assert_true(source.contains("const int groups = 24;"))
  assert_true(source.contains("column_begin = (work % groups) * 128;"))
  assert_true(source.contains("float cg[2][8] = {}, cu[2][8] = {};"))
  assert_false(source.contains("gate_tiles"))
  assert_false(source.contains("peer_tiles"))
  let profile = row_variant_test_profile()
  let dimensions = projection_refine_fold_dimensions(
    parameters,
    compilation,
    profile,
    projection_dimensions(
      profile,
      parameters,
      dispatch,
      sibling_ownership=CoownedSiblingValues,
    ),
  )
  assert_eq(dimensions.grid_x(), 384)
  assert_eq(dimensions.block_x(), 512)
  assert_eq(dimensions.shared_memory_bytes(), 0)
  // Every independent product coordinate has exactly one owner even at a tail.
  let distribution = sibling.distribution
  for active_rows in [1, 17, 33, 63, 64] {
    let stores = Array::make(active_rows * 128, 0)
    for consumer in 0..<distribution.consumer_groups() {
      for row in 0..<distribution.rows_per_group() {
        let row_tile = consumer /
          distribution.output_groups() *
          distribution.rows_per_group() +
          row
        for lane in 0..<32 {
          for value in 0..<8 {
            let output_row = row_tile * 16 + lane / 4 + value % 4 / 2 * 8
            let column = consumer % distribution.output_groups() * 16 +
              lane % 4 * 2 +
              value % 2 +
              value / 4 * 8
            if output_row < active_rows {
              stores[output_row * 128 + column] += 1
            }
          }
        }
      }
    }
    for count in stores {
      assert_eq(count, 1)
    }
  }
}

///|
test "coowned consumer geometry derives smaller workgroup without changing down" {
  let parameters = sibling_resource_test_parameters()
  let runtime = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  )
  let dispatch = runtime.dispatch_plan(1024)
  let baseline = projection_row_variant_compilation(
    parameters, dispatch, 32, 1024,
  )
  for row_tiles in [4, 8] {
    let choice = @luna_projection_tile_compiler.ProjectionFoldChoice::new(
      SiblingFold,
      stages=2,
      transfer_tiles=2,
      fragment_stages=1,
      row_tiles~,
      output_tiles=8,
      consumer_groups=8,
    )
    let compilation = projection_row_variant_compilation(
      parameters,
      dispatch,
      32,
      1024,
      fold_choices=[choice],
    )
    assert_eq(compilation.program(), baseline.program())
    assert_eq(
      compilation.schedule().single_row_reduction(),
      baseline.schedule().single_row_reduction(),
    )
    assert_eq(compilation.schedule().output_tiles_per_workgroup(), 8)
    assert_eq(
      compilation.schedule().companion_output_tiles_per_workgroup(),
      baseline.schedule().companion_output_tiles_per_workgroup(),
    )
    assert_eq(
      compilation.schedule().intermediate_pipeline(),
      baseline.schedule().intermediate_pipeline(),
    )
    let ordinary_groups = projection_row_variant_compilation(
      parameters,
      dispatch,
      32,
      1024,
      fold_choices=[
        @luna_projection_tile_compiler.ProjectionFoldChoice::new(
          SiblingFold,
          stages=2,
          transfer_tiles=2,
          fragment_stages=1,
          row_tiles~,
          output_tiles=8,
        ),
      ],
    )
    assert_false(
      compilation.schedule().digest() == ordinary_groups.schedule().digest(),
    )
    let physical = projection_refine_physical(
      compilation,
      sibling_ownership=CoownedSiblingValues,
    )
    let sibling = physical.sibling.unwrap()
    assert_eq(sibling.distribution.consumer_groups(), 8)
    assert_eq(sibling.distribution.rows_per_group(), row_tiles)
    assert_eq(physical.sibling_epilogue.unwrap().effects(), None)
    let profile = row_variant_test_profile()
    let dimensions = projection_refine_fold_dimensions(
      parameters,
      compilation,
      profile,
      projection_dimensions(
        profile,
        parameters,
        dispatch,
        sibling_ownership=CoownedSiblingValues,
      ),
    )
    assert_eq(dimensions.block_x(), 256)
    assert_eq(dimensions.shared_memory_bytes(), 0)
    assert_eq(dimensions.grid_x(), Int::max(384, 1024 / (row_tiles * 16) * 24))
    let source = @utf8.decode(
      projection_emit_source(
        "coowned_small_workgroup",
        parameters,
        1024,
        32,
        dispatch,
        physical,
        false,
        false,
        max_registers=192,
      ),
    )
    assert_true(source.contains("if (blockDim.x != 256) return;"))
    assert_true(
      source.contains(
        "float cg[\{row_tiles}][8] = {}, cu[\{row_tiles}][8] = {};",
      ),
    )
    assert_false(source.contains("gate_tiles"))
    // A complete independent product and masked tails have one unique owner.
    for active_rows in [1, 17, 63, row_tiles * 16 - 1, row_tiles * 16] {
      let stores = Array::make(active_rows * 128, 0)
      for consumer in 0..<8 {
        for row in 0..<row_tiles {
          for lane in 0..<32 {
            for value in 0..<8 {
              let output_row = row * 16 + lane / 4 + value % 4 / 2 * 8
              let column = consumer * 16 +
                lane % 4 * 2 +
                value % 2 +
                value / 4 * 8
              if output_row < active_rows {
                stores[output_row * 128 + column] += 1
              }
            }
          }
        }
      }
      for count in stores {
        assert_eq(count, 1)
      }
    }
  }
}
