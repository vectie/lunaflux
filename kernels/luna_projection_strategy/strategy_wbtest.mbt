///|
test "projection strategy separates decode GEMV and tiled prefill" {
  let shape = ProjectionShape::new(Qkv, input_width=1024, output_width=1536)
  let capabilities = ProjectionCapabilities::new(
    subgroup_width=32,
    matrix_tile=Some(ProjectionTile::new(rows=16, columns=16, reduction=16)),
  )
  let plan = plan_projection_dispatch(shape, 13, capabilities)
  assert_eq(plan.decode().kernel_class(), SubgroupGemv)
  assert_eq(plan.decode().row_bucket(), 1)
  assert_eq(plan.prefill().kernel_class(), MatrixTile)
  assert_eq(plan.prefill().row_bucket(), 16)
}

///|
test "autotune selection is device shape phase and bucket specific" {
  let shape = ProjectionShape::new(Dense, input_width=1024, output_width=1024)
  let tile = ProjectionTile::new(rows=16, columns=16, reduction=16)
  let slower = ProjectionAutotuneRecord::new(
    device_class="test-device",
    numeric_format="bf16",
    shape~,
    strategy=ProjectionStrategy::new(
      stable_id=9,
      phase=Prefill,
      kernel_class=MatrixTile,
      row_bucket=32,
      tile~,
    ),
    median_latency_ns=200L,
    sample_count=9,
  )
  let faster = ProjectionAutotuneRecord::new(
    device_class="test-device",
    numeric_format="bf16",
    shape~,
    strategy=ProjectionStrategy::new(
      stable_id=7,
      phase=Prefill,
      kernel_class=MatrixTile,
      row_bucket=16,
      tile~,
    ),
    median_latency_ns=100L,
    sample_count=9,
  )
  let records = [slower, faster]
  let selected = select_projection_autotune_record(
    records,
    "test-device",
    "bf16",
    shape,
    Prefill,
    8,
  )
  assert_eq(selected.strategy().stable_id(), 7)
}
