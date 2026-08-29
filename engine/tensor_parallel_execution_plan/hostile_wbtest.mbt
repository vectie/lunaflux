///|
test "activation coloring never aliases overlapping liveness" {
  let facts : Array[MutableActivationFact] = [
    {
      reference: @plan.ActivationRef::new(0),
      byte_count: 64L,
      alignment: 64L,
      producer: @plan.OperationId::new(0),
      live_through: 3,
      slot: -1,
    },
    {
      reference: @plan.ActivationRef::new(1),
      byte_count: 128L,
      alignment: 128L,
      producer: @plan.OperationId::new(1),
      live_through: 2,
      slot: -1,
    },
    {
      reference: @plan.ActivationRef::new(2),
      byte_count: 192L,
      alignment: 64L,
      producer: @plan.OperationId::new(4),
      live_through: 5,
      slot: -1,
    },
  ]
  let slots = assign_activation_slots(facts)
  let (bindings, _) = place_activations(facts, slots)
  assert_true(bindings[0].slot() != bindings[1].slot())
  assert_eq(bindings[0].slot(), bindings[2].slot())
  assert_false(regions_overlap(bindings[0].region(), bindings[1].region()))
  assert_true(regions_overlap(bindings[0].region(), bindings[2].region()))
  assert_eq(bindings[2].region().byte_count(), 192L)
  assert_eq(bindings[2].region().offset_bytes() % 64L, 0L)
}

///|
test "checked region arithmetic and limits reject hostile scalars" {
  for alignment in [0L, 3L, 131072L] {
    expect_execution_rule(
      () => ignore(scalar_region(DescriptorSpace, 0L, 16L, alignment)),
      Alignment,
    )
  }
  let hostile_artifact_limits = @artifact.KernelArtifactLimits::new(
    max_modules=1,
    max_entry_points=1,
    max_module_bytes=1L,
    max_total_module_bytes=1L,
    max_function_symbol_bytes=1,
  )
  expect_execution_rule(
    () => ignore(checked_add(9223372036854775807L, 1L)),
    CapacityOverflow,
  )
  expect_execution_rule(
    () => ignore(checked_multiply(9223372036854775807L, 2L)),
    CapacityOverflow,
  )
  expect_execution_rule(
    () => {
      ignore(
        TensorParallelExecutionPlanLimits::new(
          max_operations=0,
          max_operands=1,
          max_descriptors=1,
          max_collectives=1,
          max_descriptor_bytes=1L,
          max_activation_workspace_bytes=1L,
          artifact_limits=hostile_artifact_limits,
        ),
      )
    },
    InvalidLimit,
  )
}

///|
test "activation consumer cannot substitute gathered output byte geometry" {
  let reference = @plan.ActivationRef::new(0)
  let facts : Array[MutableActivationFact] = [
    {
      reference,
      byte_count: 256L,
      alignment: 64L,
      producer: @plan.OperationId::new(0),
      live_through: 0,
      slot: -1,
    },
  ]
  expect_execution_rule(
    () => record_activation_input(facts, reference, 64L, 1, 0),
    ActivationRegion,
  )
  assert_eq(facts[0].live_through, 0)
}
