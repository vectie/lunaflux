///|
test "offline fold winner is source bound and slower alternatives keep baseline" {
  let input = @plan.ActivationRef::new(0)
  let weight = @plan.TensorRef::new(0)
  let operation = projection_fixture_operation(
    1,
    OutputProjection,
    [Activation(input), Tensor(weight)],
    ProjectionShape(input_width=256, output_width=256),
    @plan.KernelCapabilityId::dense_projection(),
  )
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=32,
    max_query_tokens=256,
    max_page_table_entries=1024,
  )
  let operands = [
    @launch_contract.AotOperand::new(
      role=RuntimeInput(StepCounts),
      byte_count=20L,
      alignment=4L,
    ),
    @launch_contract.AotOperand::new(
      role=ActivationInput(input),
      byte_count=131072L,
      alignment=16L,
    ),
    @launch_contract.AotOperand::new(
      role=WeightInput(weight),
      byte_count=131072L,
      alignment=16L,
    ),
    @launch_contract.AotOperand::new(
      role=ActivationOutput(@plan.ActivationRef::new(1)),
      byte_count=131072L,
      alignment=16L,
    ),
  ]
  let target = projection_fixture_target()
  let compiler = projection_fixture_compiler()
  let limits = ProjectionAotLimits::new(
    max_source_bytes=1048576,
    max_recipe_bytes=65536,
  )
  let dispatch = prepare_projection_cuda_runtime_dispatch(
    operation,
    profile,
    target,
    operands,
    [],
  )
  let baseline = lower_reusable_projection_cuda_aot_candidate_with_dispatch(
    operation, profile, target, compiler, operands, dispatch, limits,
  )
  let choice = @luna_projection_tile_compiler.ProjectionFoldChoice::new(
    MatrixFold,
    stages=2,
    transfer_tiles=1,
    fragment_stages=1,
  )
  let variant = lower_reusable_projection_cuda_aot_candidate_with_dispatch(
    operation,
    profile,
    target,
    compiler,
    operands,
    dispatch,
    limits,
    fold_choices=[choice],
  )
  let base = baseline.source_digest().as_hex()
  let selected = variant.source_digest().as_hex()
  assert_false(base == selected)
  for (latency, expected) in [(50, selected), (150, base)] {
    let records = @luna_projection_tuning.parse_projection_fold_records(
      @utf8.encode(
        "luna-projection-fold-v1\nscope\ttest\nrecord\t\{base}\t\{base}\t100\t3\t-\nrecord\t\{base}\t\{selected}\t\{latency}\t3\tmatrix:2:1:1\n",
      ),
      scope="test",
    )
    let result = lower_measured_projection_cuda_aot_candidate(
      operation,
      profile,
      target,
      compiler,
      operands,
      dispatch,
      limits,
      records[:],
    )
    assert_eq(result.source_digest().as_hex(), expected)
  }
  let stale = @luna_projection_tuning.parse_projection_fold_records(
    @utf8.encode(
      "luna-projection-fold-v1\nscope\ttest\nrecord\t\{base}\t\{base}\t100\t3\t-\nrecord\t\{base}\t\{"0".repeat(64)}\t50\t3\tmatrix:2:1:1\n",
    ),
    scope="test",
  )
  try
    lower_measured_projection_cuda_aot_candidate(
      operation,
      profile,
      target,
      compiler,
      operands,
      dispatch,
      limits,
      stale[:],
    )
  catch {
    Invalid(CompilerPolicy) => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("stale source observation accepted")
  }
}
