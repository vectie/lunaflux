///|
fn plan(
  law : @precision.Float32AffineLaw,
) -> @precision.LearnedGatedPoolPrecision raise {
  @precision.LearnedGatedPoolPrecision::new(
    input_width=8,
    feature_width=16,
    ratio=4,
    preceding_window_overlap=true,
    maximum_rows=5,
    maximum_positions=1024,
    norm_epsilon=1.0e-6,
    maximum_state_bytes=1048576L,
    maximum_workspace_bytes=1048576L,
  ).with_projection_multiply_add(law)
}

///|
test "faster empirical arithmetic cannot authorize an exact replacement" {
  let ordinary = plan(SeparateMultiplyAdd)
  let fused = plan(FusedMultiplyAdd)
  let scalar = ReadOnlyArray::from_array([ordinary.expression(Normalize)])
  let folds = ReadOnlyArray::from_array([ordinary.fold(ProjectionUpdate)])
  let candidates : Array[@numeric_selection.MeasuredNumericalPlan] = [
    {
      label: "fast-but-different",
      expressions: scalar,
      folds: ReadOnlyArray::from_array([fused.fold(ProjectionUpdate)]),
      elapsed_ns: 1L,
      memory_bytes: 10L,
    },
    {
      label: "same-program-slow",
      expressions: scalar,
      folds,
      elapsed_ns: 20L,
      memory_bytes: 10L,
    },
    {
      label: "same-program-fast",
      expressions: scalar,
      folds,
      elapsed_ns: 10L,
      memory_bytes: 10L,
    },
    {
      label: "out-of-memory",
      expressions: scalar,
      folds,
      elapsed_ns: 2L,
      memory_bytes: 100L,
    },
  ]
  assert_eq(
    @numeric_selection.select_exact_program(
      scalar[:],
      folds[:],
      candidates,
      memory_ceiling=20L,
    ),
    Some(2),
  )
  assert_eq(
    @numeric_selection.select_exact_program(
      scalar[:],
      folds[:],
      candidates[:1],
      memory_ceiling=20L,
    ),
    None,
  )
  assert_eq(
    @numeric_selection.select_exact_program(
      scalar[:],
      folds[:],
      candidates,
      memory_ceiling=0L,
    ),
    None,
  )
}

///|
test "fold iteration and seed cannot disappear in selection" {
  let ordinary = plan(SeparateMultiplyAdd).fold(ProjectionUpdate)
  let changed = @precision.OrderedNumericFold::new(
    extent=ordinary.extent + 1,
    initial_bits=ordinary.initial_bits,
    update=ordinary.update,
  )
  let changed_seed = @precision.OrderedNumericFold::new(
    extent=ordinary.extent,
    initial_bits=0x80000000U,
    update=ordinary.update,
  )
  for fold in [changed, changed_seed] {
    let candidates : Array[@numeric_selection.MeasuredNumericalPlan] = [
      {
        label: "changed",
        expressions: [],
        folds: ReadOnlyArray::from_array([fold]),
        elapsed_ns: 1L,
        memory_bytes: 0L,
      },
    ]
    assert_eq(
      @numeric_selection.select_exact_program(
        [],
        [ordinary],
        candidates,
        memory_ceiling=0L,
      ),
      None,
    )
  }
}
