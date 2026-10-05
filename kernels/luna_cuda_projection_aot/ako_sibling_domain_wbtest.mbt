///|
// Bounded offline schedule experiment. These alternatives preserve the ordered
// fold and down program; none changes a serving default.
test "AKO sibling row ownership changes only the primary physical domain" {
  let large_chunks = @env.get_env_var("LUNA_EXPORT_AKO_LARGE_CHUNKS") ==
    Some("1")
  let capacity = if large_chunks { 8192 } else { 2048 }
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
    max_query_tokens=capacity,
    max_page_table_entries=16384,
  )
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=1,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let compiler = @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256="0".repeat(64),
      major=13,
      minor=0,
      patch=88,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=255,
  )
  let operands = [
    projection_fixture_operand(RuntimeInput(StepCounts), 20L, 4L),
    projection_fixture_operand(
      ActivationInput(input),
      capacity.to_int64() * 2048L,
      16L,
    ),
    ..weights.map(w => projection_fixture_operand(WeightInput(w), 6291456L, 16L)),
    projection_fixture_operand(
      ActivationOutput(@plan.ActivationRef::new(9)),
      capacity.to_int64() * 2048L,
      16L,
    ),
    projection_fixture_operand(Workspace, capacity.to_int64() * 6144L, 16L),
  ]
  let dispatch = prepare_projection_cuda_runtime_dispatch(
    operation,
    profile,
    target,
    operands,
    [],
  )
  let limits = projection_fixture_limits(max_source_bytes=1048576)
  let retained = [
    @luna_projection_tile_compiler.ProjectionFoldChoice::new(
      IntermediateFold,
      stages=2,
      transfer_tiles=4,
      fragment_stages=1,
      row_tiles=8,
      hoist_addresses=true,
    ),
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
    @luna_projection_tile_compiler.ProjectionFoldChoice::new(
      IntermediateFold,
      stages=2,
      transfer_tiles=8,
      fragment_stages=1,
      row_tiles=1,
      output_tiles=1,
      hoist_addresses=true,
      consumer_groups=1,
      maximum_token_rows=16,
    ),
  ]
  let baseline = lower_reusable_projection_cuda_aot_candidate_with_dispatch(
    operation,
    profile,
    target,
    compiler,
    operands,
    dispatch,
    limits,
    fold_choices=retained,
    sibling_execution=OverlappedTransfer,
  )
  let before = @utf8.decode(baseline.source_bytes())
  let down = "extern \"C\" __global__ void lunaflux_luna_gated_mlp_bf16_release_v1_down("
  for
    (label, row_tiles, consumers) in [
      ("baseline", 4, 16),
      ("r64-c8", 4, 8),
      ("r128-c16", 8, 16),
      ("r128-c8", 8, 8),
      ("r64-coowned-c16", 4, 16),
    ] {
    let candidate = if label == "baseline" {
      baseline
    } else {
      lower_reusable_projection_cuda_aot_candidate_with_dispatch(
        operation,
        profile,
        target,
        compiler,
        operands,
        dispatch,
        limits,
        fold_choices=[
          ..retained,
          @luna_projection_tile_compiler.ProjectionFoldChoice::new(
            SiblingFold,
            stages=2,
            transfer_tiles=2,
            fragment_stages=1,
            row_tiles~,
            output_tiles=4,
            consumer_groups=consumers,
          ),
        ],
        sibling_execution=OverlappedTransfer,
        sibling_transport=SegmentedSiblingTransport,
        sibling_ownership=if label == "r64-coowned-c16" {
          CoownedSiblingValues
        } else {
          SplitSiblingPlanes
        },
      )
    }
    let after = @utf8.decode(candidate.source_bytes())
    // Both primary and down carry the same exported ceiling. The probe must
    // never silently measure an AOT guard returning early at larger chunks.
    let capacity_guard = "if (token_count <= 0 || token_count > " +
      capacity.to_string() +
      ") return;"
    assert_eq(after.split(capacity_guard).count(), 3)
    assert_false(
      after.contains("token_count > " + (capacity + 1).to_string() + ")"),
    )
    for i in [0, 1] {
      if label == "r64-coowned-c16" {
        let a = candidate.row_variants[i]
        let b = baseline.row_variants[i]
        assert_eq(a.maximum_token_rows(), b.maximum_token_rows())
        assert_eq(a.function_symbol(), b.function_symbol())
        assert_eq(a.dimensions().grid_x(), b.dimensions().grid_x())
        assert_eq(a.dimensions().block_x(), b.dimensions().block_x())
        assert_eq(a.dimensions().shared_memory_bytes(), 0)
        assert_eq(a.companion_dimensions(), b.companion_dimensions())
        assert_eq(a.bounded_work_grid_x(), b.bounded_work_grid_x())
      } else {
        assert_eq(candidate.row_variants[i], baseline.row_variants[i])
      }
    }
    assert_eq(candidate.companion_dimensions, baseline.companion_dimensions)
    assert_eq(
      before.split(down).to_array()[1]
      .split("extern \"C\" __global__")
      .to_array()[0],
      after.split(down).to_array()[1]
      .split("extern \"C\" __global__")
      .to_array()[0],
    )
    if label != "baseline" {
      assert_false(candidate.source_digest() == baseline.source_digest())
    }
    if label == "r64-coowned-c16" {
      assert_eq(
        candidate.launch_dimensions().grid_x(),
        baseline.launch_dimensions().grid_x(),
      )
      assert_eq(
        candidate.launch_dimensions().block_x(),
        baseline.launch_dimensions().block_x(),
      )
      assert_eq(candidate.launch_dimensions().shared_memory_bytes(), 0)
      let primary = after.split(down).to_array()[0]
      assert_true(primary.contains("float cg[1][8]"))
      assert_true(primary.contains("cu[1][8]"))
      assert_false(primary.contains("sibling_output_bytes"))
    }
    let export_old = @env.get_env_var("LUNA_EXPORT_AKO_SIBLING_DOMAIN") ==
      Some("1") &&
      label != "r64-coowned-c16"
    let export_coowned = @env.get_env_var("LUNA_EXPORT_AKO_COOWNED_C16") ==
      Some("1") &&
      (label == "baseline" || label == "r64-coowned-c16")
    if export_old || export_coowned || large_chunks {
      let geometries : Array[Array[Int]] = []
      for
        (name, geometry) in [
          ("PRIMARY", candidate.launch_dimensions()),
          ("DOWN", candidate.companion_dimensions.unwrap()),
        ] {
        ignore(name)
        geometries.push([
          geometry.grid_x(),
          geometry.block_x(),
          geometry.shared_memory_bytes(),
        ])
      }
      let record : Json = {
        "label": label,
        "source": after,
        "source_sha256": candidate.source_digest().as_hex(),
        "geometry": geometries,
        "maximum_query_tokens": capacity,
      }
      println("LUNA_DOMAIN_RECORD " + record.stringify())
    }
  }
}
