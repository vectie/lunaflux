///|
test "companion launch requires a split MLP and retains its own bucket geometry" {
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=1024,
    grid_y=1,
    grid_z=1,
    block_x=128,
    block_y=1,
    block_z=1,
    shared_memory_bytes=4096,
  )
  let operation = @plan.OperationId::new(0)
  let base = qwen_fused_test_blueprint(qwen_fused_test_plan())
  let legacy = paged_device_launch_dimensions(base.steps()[0].dimensions())
  for explicit in [false, true] {
    let blueprint = {
      ..base,
      steps: [
        {
          ..base.steps()[0],
          companion_dimensions: if explicit {
            Some(dimensions)
          } else {
            None
          },
        },
      ],
    }
    // Exercise eager preparation as well as bucket geometry below. Device
    // launch dimensions intentionally expose no host inspection interface.
    ignore(
      paged_companion_dimensions(
        blueprint,
        base.steps()[0].operation_id(),
        legacy,
      ),
    )
  }
  validate_projection_companion(operation, GatedMlp, true, Some(dimensions))
  validate_projection_companion(operation, LanguageModelHead, false, None)
  for
    pair in [(@plan.OperationKind::LanguageModelHead, true), (GatedMlp, false)] {
    let (kind, workspace) = pair
    let rejected = try {
      validate_projection_companion(
        operation,
        kind,
        workspace,
        Some(dimensions),
      )
      false
    } catch {
      _ => true
    }
    assert_true(rejected)
  }
  for pair in [(1, 256), (2, 16), (17, 32), (128, 128), (1024, 1024)] {
    let (rows, grid) = pair
    let launch = paged_bucket_launch(
      ProjectionMlpDownGridX(dimensions, 1024),
      rows,
    )
    assert_eq(launch.grid_x(), grid)
    assert_eq(launch.block_x(), 128)
    assert_eq(launch.shared_memory_bytes(), 4096)
  }
}
