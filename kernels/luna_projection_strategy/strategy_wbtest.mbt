///|
fn strategy_test_shape() -> ProjectionShape raise ProjectionStrategyError {
  ProjectionShape::new(Dense, input_width=1024, output_width=1024)
}

///|
fn strategy_test_capabilities() -> ProjectionCapabilities raise ProjectionStrategyError {
  ProjectionCapabilities::new(
    subgroup_width=32,
    matrix_tile=Some(ProjectionTile::new(rows=16, columns=16, reduction=16)),
  )
}

///|
fn strategy_test_candidate(
  stable_id : Int,
  row_bucket : Int,
) -> ProjectionStrategy raise ProjectionStrategyError {
  ProjectionStrategy::new(
    stable_id~,
    phase=Prefill,
    strategy_class=Some(
      if row_bucket <= 8 {
        SmallRowTiledGemm
      } else {
        LargeRowTiledGemm
      },
    ),
    kernel_class=MatrixTile,
    row_bucket~,
    tile=ProjectionTile::new(rows=16, columns=16, reduction=16),
  )
}

///|
fn strategy_test_record(
  backend : ProjectionBackendId,
  target : ProjectionTargetId,
  shape : ProjectionShape,
  strategy : ProjectionStrategy,
  latency : Int64,
) -> ProjectionAutotuneRecord raise ProjectionStrategyError {
  ProjectionAutotuneRecord::new(
    key=ProjectionAutotuneKey::new(
      backend~,
      target~,
      numeric_type=ProjectionBFloat16,
      shape~,
      strategy~,
    ),
    median_latency_ns=latency,
    sample_count=9,
  )
}

///|
fn expect_projection_rule(
  body : () -> Unit raise,
  expected : ProjectionStrategyRule,
) -> Unit raise {
  try body() catch {
    ProjectionStrategyError::InvalidProjectionStrategy(rule) if rule == expected =>
      ()
    _ => fail("unexpected projection strategy rejection")
  } noraise {
    _ => fail("expected projection strategy rejection")
  }
}

///|
test "projection row buckets have explicit serving classes and extensible prefill" {
  let rows = [1, 2, 8, 9, 16, 17, 32, 33, 65]
  let expected = [1, 8, 8, 16, 16, 32, 32, 64, 128]
  for index, row_count in rows {
    assert_eq(projection_row_bucket(row_count), expected[index])
  }
  let shape = ProjectionShape::new(Qkv, input_width=1024, output_width=1536)
  let plan = plan_projection_dispatch(shape, 13, strategy_test_capabilities())
  assert_eq(plan.decode().strategy_class(), DecodeGemv)
  assert_eq(plan.decode().kernel_class(), SubgroupGemv)
  assert_eq(plan.prefill().strategy_class(), LargeRowTiledGemm)
  assert_eq(plan.prefill().kernel_class(), MatrixTile)
  assert_eq(plan.prefill().row_bucket(), 16)
  let small = build_projection_runtime_dispatch(
    [],
    backend=ProjectionBackendId::new("portable"),
    target=ProjectionTargetId::new("fixture"),
    numeric_type=ProjectionBFloat16,
    shape~,
    maximum_rows=32,
    capabilities=strategy_test_capabilities(),
  )
  assert_eq(small.select(7).strategy().strategy_class(), SmallRowTiledGemm)
  assert_eq(small.select(17).strategy().strategy_class(), LargeRowTiledGemm)
  let frozen = small.dispatch_plan(17)
  assert_eq(frozen.decode().strategy_class(), DecodeGemv)
  assert_eq(frozen.prefill().row_bucket(), 32)
  expect_projection_rule(
    () => {
      ignore(
        ProjectionStrategy::new(
          stable_id=999,
          phase=Prefill,
          kernel_class=ScalarTile,
          row_bucket=7,
          tile=ProjectionTile::new(rows=1, columns=32, reduction=1),
        ),
      )
    },
    Bucket,
  )
}

///|
test "gated MLP decode does not claim a missing tiled companion" {
  let shape = ProjectionShape::new(
    GatedMlp,
    input_width=128,
    output_width=128,
    intermediate_width=256,
  )
  let capabilities = ProjectionCapabilities::new(
    subgroup_width=32,
    matrix_tile=None,
  )
  let plan = plan_projection_dispatch(shape, 8, capabilities)
  assert_eq(plan.decode().kernel_class(), ScalarTile)
  assert_eq(plan.prefill().kernel_class(), ScalarTile)
  let tiled_shape = ProjectionShape::new(
    GatedMlp,
    input_width=1024,
    output_width=1024,
    intermediate_width=3072,
  )
  let tiled_capabilities = ProjectionCapabilities::new(
    subgroup_width=32,
    matrix_tile=Some(ProjectionTile::new(rows=16, columns=256, reduction=256)),
  )
  let tiled = plan_projection_dispatch(tiled_shape, 8, tiled_capabilities)
  assert_eq(tiled.decode().kernel_class(), SubgroupGemv)
  assert_eq(tiled.prefill().kernel_class(), MatrixTile)
}

///|
test "offline autotune selection is deterministic and record order independent" {
  let backend = ProjectionBackendId::new("portable")
  let target = ProjectionTargetId::new("fixture-a")
  let shape = strategy_test_shape()
  let stable_eight = strategy_test_record(
    backend,
    target,
    shape,
    strategy_test_candidate(8, 16),
    100L,
  )
  let stable_seven = strategy_test_record(
    backend,
    target,
    shape,
    strategy_test_candidate(7, 16),
    100L,
  )
  let first = build_projection_runtime_dispatch(
    [stable_eight, stable_seven],
    backend~,
    target~,
    numeric_type=ProjectionBFloat16,
    shape~,
    maximum_rows=16,
    capabilities=strategy_test_capabilities(),
  )
  let second = build_projection_runtime_dispatch(
    [stable_seven, stable_eight],
    backend~,
    target~,
    numeric_type=ProjectionBFloat16,
    shape~,
    maximum_rows=16,
    capabilities=strategy_test_capabilities(),
  )
  assert_eq(first.select(9).source(), OfflineAutotune)
  assert_eq(first.select(9).strategy().stable_id(), 7)
  assert_eq(second.select(9).strategy().stable_id(), 7)
}

///|
test "autotune records reject exact duplicates and conflicting measurements" {
  let backend = ProjectionBackendId::new("portable")
  let target = ProjectionTargetId::new("fixture-a")
  let shape = strategy_test_shape()
  let strategy = strategy_test_candidate(7, 16)
  let first = strategy_test_record(backend, target, shape, strategy, 100L)
  expect_projection_rule(
    () => {
      ignore(
        build_projection_runtime_dispatch(
          [first, first],
          backend~,
          target~,
          numeric_type=ProjectionBFloat16,
          shape~,
          maximum_rows=16,
          capabilities=strategy_test_capabilities(),
        ),
      )
    },
    Duplicate,
  )
  let conflict = strategy_test_record(backend, target, shape, strategy, 99L)
  expect_projection_rule(
    () => {
      ignore(
        build_projection_runtime_dispatch(
          [first, conflict],
          backend~,
          target~,
          numeric_type=ProjectionBFloat16,
          shape~,
          maximum_rows=16,
          capabilities=strategy_test_capabilities(),
        ),
      )
    },
    Conflict,
  )
  let aliased_strategy = ProjectionStrategy::new(
    stable_id=7,
    phase=Prefill,
    strategy_class=Some(LargeRowTiledGemm),
    kernel_class=ScalarTile,
    row_bucket=16,
    tile=ProjectionTile::new(rows=1, columns=32, reduction=1),
  )
  let aliased_record = strategy_test_record(
    backend, target, shape, aliased_strategy, 98L,
  )
  expect_projection_rule(
    () => {
      ignore(
        build_projection_runtime_dispatch(
          [first, aliased_record],
          backend~,
          target~,
          numeric_type=ProjectionBFloat16,
          shape~,
          maximum_rows=16,
          capabilities=strategy_test_capabilities(),
        ),
      )
    },
    Conflict,
  )
}

///|
test "runtime dispatch isolates device identity and uses honest static fallback" {
  let backend = ProjectionBackendId::new("portable")
  let target = ProjectionTargetId::new("fixture-a")
  let shape = strategy_test_shape()
  let foreign_backend = strategy_test_record(
    ProjectionBackendId::new("foreign"),
    target,
    shape,
    strategy_test_candidate(5, 16),
    1L,
  )
  let foreign_target = strategy_test_record(
    backend,
    ProjectionTargetId::new("fixture-b"),
    shape,
    strategy_test_candidate(6, 16),
    2L,
  )
  let matching = strategy_test_record(
    backend,
    target,
    shape,
    strategy_test_candidate(7, 16),
    90L,
  )
  let dispatch = build_projection_runtime_dispatch(
    [foreign_backend, foreign_target, matching],
    backend~,
    target~,
    numeric_type=ProjectionBFloat16,
    shape~,
    maximum_rows=32,
    capabilities=strategy_test_capabilities(),
  )
  assert_eq(dispatch.select(12).strategy().stable_id(), 7)
  assert_eq(dispatch.select(12).source(), OfflineAutotune)
  assert_eq(dispatch.select(20).source(), StaticFallback)
  assert_eq(dispatch.select(20).strategy().stable_id(), 201)
  expect_projection_rule(() => ignore(dispatch.select(33)), Bucket)
}
