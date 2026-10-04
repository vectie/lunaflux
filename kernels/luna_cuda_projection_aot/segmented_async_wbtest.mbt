///|
test "measured segmented effects reach primary and bounded AOT entries" {
  let input = @plan.ActivationRef::new(8)
  let weights = [9, 10, 11].map(i => @plan.TensorRef::new(i))
  let operation = projection_fixture_operation(
    9,
    GatedMlp,
    [Activation(input), ..weights.map(w => @plan.ValueRef::Tensor(w))],
    GatedMlpShape(hidden_size=1024, intermediate_size=3072),
    @plan.KernelCapabilityId::gated_mlp(),
  )
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=32,
    max_query_tokens=2048,
    max_page_table_entries=16384,
  )
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=1,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let compiler = projection_fixture_compiler(max_registers=255)
  let operands = [
    projection_fixture_operand(RuntimeInput(StepCounts), 20L, 4L),
    projection_fixture_operand(ActivationInput(input), 4194304L, 16L),
    ..weights.map(w => projection_fixture_operand(WeightInput(w), 6291456L, 16L)),
    projection_fixture_operand(
      ActivationOutput(@plan.ActivationRef::new(9)),
      4194304L,
      16L,
    ),
    projection_fixture_operand(Workspace, 12582912L, 16L),
  ]
  let dispatch = prepare_projection_cuda_runtime_dispatch(
    operation,
    profile,
    target,
    operands,
    [],
  )
  let limits = projection_fixture_limits(max_source_bytes=1048576)
  let baseline = lower_reusable_projection_cuda_aot_candidate_with_dispatch(
    operation, profile, target, compiler, operands, dispatch, limits,
  )
  let overlapped = lower_reusable_projection_cuda_aot_candidate_with_dispatch(
    operation,
    profile,
    target,
    compiler,
    operands,
    dispatch,
    limits,
    sibling_execution=OverlappedTransfer,
  )
  let base = baseline.source_digest().as_hex()
  let artifact = overlapped.source_digest().as_hex()
  assert_false(base == artifact)
  assert_eq(baseline.launch_dimensions(), overlapped.launch_dimensions())
  assert_eq(baseline.row_variants, overlapped.row_variants)
  let before = @utf8.decode(baseline.source_bytes())
  let after = @utf8.decode(overlapped.source_bytes())
  let down = "extern \"C\" __global__ void lunaflux_luna_gated_mlp_bf16_release_v1_down("
  assert_eq(
    before.split(down).to_array()[1].split("#include").to_array()[0],
    after.split(down).to_array()[1].split("#include").to_array()[0],
  )
  for variant in overlapped.row_variants {
    let body = after.split("void \{variant.function_symbol()}(").to_array()[1]
      .split("#include")
      .to_array()[0]
    assert_true(body.contains("cp.async.cg.shared.global"))
  }
  for latency in [50, 150] {
    // Synthetic observations exercise propagation only; never export them as measurements.
    let records = @luna_projection_tuning.parse_projection_fold_records(
      @utf8.encode(
        "luna-projection-fold-v4\nscope\ttest\nrecord\t\{base}\t\{base}\t100\t3\t-\nrecord\t\{base}\t\{artifact}\t\{latency}\t3\texecution:overlapped\n",
      ),
      scope="test",
    )
    let selected = lower_measured_projection_cuda_aot_candidate(
      operation,
      profile,
      target,
      compiler,
      operands,
      dispatch,
      limits,
      records[:],
    )
    assert_eq(
      selected.source_digest().as_hex(),
      if latency == 50 {
        artifact
      } else {
        base
      },
    )
  }
  if @env.get_env_var("LUNA_EXPORT_SEGMENTED_AOT") == Some("1") {
    let down_choice = @luna_projection_tile_compiler.ProjectionFoldChoice::new(
      IntermediateFold,
      stages=2,
      transfer_tiles=4,
      fragment_stages=1,
      row_tiles=8,
      hoist_addresses=true,
    )
    let small_choice = @luna_projection_tile_compiler.ProjectionFoldChoice::new(
      SiblingFold,
      stages=2,
      transfer_tiles=4,
      fragment_stages=1,
      row_tiles=1,
      output_tiles=2,
      consumer_groups=2,
      maximum_token_rows=16,
    )
    let serial = lower_reusable_projection_cuda_aot_candidate_with_dispatch(
      operation,
      profile,
      target,
      compiler,
      operands,
      dispatch,
      limits,
      fold_choices=[down_choice, small_choice],
    )
    let asynchronous = lower_reusable_projection_cuda_aot_candidate_with_dispatch(
      operation,
      profile,
      target,
      compiler,
      operands,
      dispatch,
      limits,
      fold_choices=[down_choice, small_choice],
      sibling_execution=OverlappedTransfer,
    )
    let serial = if @env.get_env_var("LUNA_EXPORT_DEFAULT_BASELINE") ==
      Some("1") {
      baseline
    } else {
      serial
    }
    for (label, lowered) in [("serial", serial), ("async", asynchronous)] {
      println("LUNA_SOURCE_BEGIN " + label)
      println(@utf8.decode(lowered.source_bytes()))
      println("// max_registers=255")
      let primary = lowered.launch_dimensions()
      let companion = lowered.companion_dimensions.unwrap()
      for (name, dimensions) in [("PRIMARY", primary), ("DOWN", companion)] {
        println("// PROBE_\{name}_GRID=\{dimensions.grid_x()}")
        println("// PROBE_\{name}_ACTIVE_GRID=\{dimensions.grid_x()}")
        println("// PROBE_\{name}_BLOCK=\{dimensions.block_x()}")
        println("// PROBE_\{name}_SHARED=\{dimensions.shared_memory_bytes()}")
      }
      println("LUNA_SOURCE_END")
    }
  }
}

///|
test "segmented lifetime refinement preserves sibling math layout and ownership" {
  let parameters = sibling_resource_test_parameters()
  let runtime = projection_build_runtime_dispatch(
    parameters,
    2048,
    projection_fixture_target(),
    [],
  )
  let dispatch = runtime.dispatch_plan(2048)
  let compilation = projection_row_variant_compilation(
    parameters, dispatch, 32, 2048,
  )
  let baseline = projection_refine_physical(compilation)
  let overlapped = projection_refine_physical(
    compilation,
    sibling_execution=OverlappedTransfer,
  )
  assert_eq(baseline.scheduled, overlapped.scheduled)
  assert_eq(baseline.sibling_epilogue, overlapped.sibling_epilogue)
  assert_eq(baseline.intermediate, overlapped.intermediate)
  let before = baseline.sibling.unwrap()
  let after = overlapped.sibling.unwrap()
  assert_eq(before.sibling_transport, SegmentedSiblingTransport)
  assert_eq(before.operands, after.operands)
  assert_eq(before.fragments, after.fragments)
  assert_eq(before.distribution, after.distribution)
  assert_eq(before.transfer_segments, after.transfer_segments)
  assert_false(before.effects.has_pending_transfers())
  assert_true(after.effects.has_pending_transfers())
  assert_eq(before.effects.geometry(), after.effects.geometry())
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=32,
    max_query_tokens=2048,
    max_page_table_entries=8192,
  )
  let primary = projection_dimensions(profile, parameters, dispatch)
  let down = projection_companion_dimensions(
    parameters, dispatch, profile, compilation,
  ).unwrap()
  for (label, physical) in [("serial", baseline), ("async", overlapped)] {
    let source = @utf8.decode(
      projection_emit_source(
        "lunaflux_luna_gated_mlp_bf16_release_v1",
        parameters,
        2048,
        32,
        dispatch,
        physical,
        false,
        true,
        max_registers=128,
      ),
    )
    let gate = source
      .split(
        "extern \"C\" __global__ void lunaflux_luna_gated_mlp_bf16_release_v1_down(",
      )
      .to_array()[0]
    if label == "async" {
      assert_true(gate.contains("cp.async.cg.shared.global"))
      assert_true(gate.contains("cp.async.commit_group"))
      assert_true(gate.contains("cp.async.wait_group 0"))
      assert_true(gate.contains("const bool valid = transfer_active[slot]"))
      assert_false(gate.contains("unsigned values["))
    } else {
      assert_false(gate.contains("cp.async"))
      assert_true(gate.contains("unsigned values["))
    }
    assert_true(gate.contains("source_row < token_count"))
    if @env.get_env_var("LUNA_EXPORT_SEGMENTED_ASYNC") == Some("1") {
      println("LUNA_SOURCE_BEGIN " + label)
      println(source)
      println("// max_registers=128")
      for (name, geometry) in [("PRIMARY", primary), ("DOWN", down)] {
        println("// PROBE_\{name}_GRID=\{geometry.grid_x()}")
        println("// PROBE_\{name}_ACTIVE_GRID=\{geometry.grid_x()}")
        println("// PROBE_\{name}_BLOCK=\{geometry.block_x()}")
        println("// PROBE_\{name}_SHARED=\{geometry.shared_memory_bytes()}")
      }
      println("LUNA_SOURCE_END")
    }
  }
}
