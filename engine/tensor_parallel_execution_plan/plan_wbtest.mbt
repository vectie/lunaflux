///|
fn expect_execution_rule(
  body : () -> Unit raise TensorParallelExecutionPlanError,
  expected : TensorParallelExecutionPlanRule,
) -> Unit raise {
  try body() catch {
    InvalidTensorParallelExecutionPlan(rule~, ..) => assert_eq(rule, expected)
  } noraise {
    _ => fail("invalid tensor-parallel execution evidence was admitted")
  }
}

///|
fn regions_overlap(
  left : TensorParallelScalarRegion,
  right : TensorParallelScalarRegion,
) -> Bool {
  left.offset_bytes() < right.end_offset_bytes() &&
  right.offset_bytes() < left.end_offset_bytes()
}

///|
fn digest_hex_difference(
  left : TensorParallelExecutionPlanDigest,
  right : TensorParallelExecutionPlanDigest,
) -> Int {
  let left_bytes = @utf8.encode(left.as_sha256())
  let right_bytes = @utf8.encode(right.as_sha256())
  if left_bytes.length() != right_bytes.length() {
    return left_bytes.length().max(right_bytes.length())
  }
  let mut changed = 0
  for index, byte in left_bytes {
    if byte != right_bytes[index] {
      changed += 1
    }
  }
  changed
}

///|
fn assert_plan_geometry(
  source : ExecutionFixture,
  result : TensorParallelExecutionPlan,
) -> Unit raise {
  assert_eq(result.identity(), source.model.identity())
  assert_eq(result.rank(), 1)
  assert_eq(result.world_size(), 2)
  assert_eq(result.device_ordinal(), 5)
  assert_eq(result.target(), source.launch_set.device_target())
  assert_eq(result.catalog_version(), @catalog.CatalogVersion::v3())
  assert_eq(result.operation_count(), source.model.operations().length())
  assert_eq(result.collective_count(), source.device_plan.collective_count())
  assert_true(result.descriptor_bytes() > 0L)
  assert_true(result.weight_arena_bytes() > 0L)
  assert_eq(
    result.kv_arena_bytes(),
    source.launch_set.device_kv_layout().arena_bytes(),
  )
  assert_true(result.activation_workspace_bytes() > 0L)
  assert_eq(result.rank_constants().space(), RankConstantsSpace)
  assert_eq(result.rank_constants().offset_bytes(), 0L)
  for descriptor in result.descriptors() {
    let region = descriptor.region()
    assert_eq(region.space(), DescriptorSpace)
    assert_eq(region.offset_bytes() % region.alignment(), 0L)
    assert_true(region.end_offset_bytes() <= result.descriptor_bytes())
  }
  for index in 0..<result.activation_count() {
    let reference = @plan.ActivationRef::new(index)
    guard result.activation(reference) is Some(binding) else {
      fail("activation binding is missing")
    }
    let region = binding.region()
    assert_eq(region.space(), ActivationWorkspaceSpace)
    assert_eq(region.offset_bytes() % region.alignment(), 0L)
    assert_true(region.end_offset_bytes() <= result.workspace().offset_bytes())
  }
  assert_eq(result.workspace().space(), ActivationWorkspaceSpace)
  assert_true(
    result.workspace().end_offset_bytes() <= result.activation_workspace_bytes(),
  )
}

///|
fn assert_activation_liveness(
  result : TensorParallelExecutionPlan,
) -> Unit raise {
  let mut reused = false
  for left_index in 0..<result.activation_count() {
    let left = result.activation(@plan.ActivationRef::new(left_index)).unwrap()
    for right_index in (left_index + 1)..<result.activation_count() {
      let right = result
        .activation(@plan.ActivationRef::new(right_index))
        .unwrap()
      let lifetimes_overlap = left.produced_by().as_int() <=
        right.live_through_step() &&
        right.produced_by().as_int() <= left.live_through_step()
      if lifetimes_overlap {
        assert_false(regions_overlap(left.region(), right.region()))
      } else if left.slot() == right.slot() {
        reused = true
        assert_true(regions_overlap(left.region(), right.region()))
      }
    }
  }
  assert_true(reused)
}

///|
fn assert_operation_artifacts_and_weights(
  source : ExecutionFixture,
  result : TensorParallelExecutionPlan,
) -> Unit raise {
  for operation_index, semantic in source.model.operations() {
    let binding = result.operation_exact(semantic.id())
    assert_eq(binding.operation_id().as_int(), operation_index)
    assert_eq(binding.operation_kind(), semantic.kind())
    assert_eq(binding.capability(), semantic.required_capability())
    assert_eq(
      binding.module_digest(),
      binding.entry_point().family().module_digest(),
    )
    guard source.artifacts.entry_point(binding.entry_point()) is Some(entry) else {
      fail("exact artifact lookup failed")
    }
    assert_eq(binding.function_symbol(), entry.function_symbol())
    for operand in binding.operands() {
      let region = operand.region()
      assert_eq(region.offset_bytes() % region.alignment(), 0L)
      match operand.role() {
        WeightInput(reference) => {
          guard source.device_plan.tensor(reference) is Some(view) else {
            fail("weight view is missing")
          }
          assert_eq(region.space(), WeightSpace)
          assert_eq(region.offset_bytes(), view.region_offset_bytes())
          assert_eq(region.byte_count(), view.region_byte_length())
          if view.placement() != Replicated {
            assert_true(
              view.local_rows() * view.local_columns() <
              view.full_rows() * view.full_columns(),
            )
          }
        }
        PersistentKvComponent(_, _) =>
          assert_eq(region.space(), PersistentKvSpace)
        _ => ()
      }
    }
  }
  let first = result.operation_exact(@plan.OperationId::new(0))
  assert_eq(first.operation_id().as_int(), 0)
  let collective = result.collective_exact(0)
  assert_eq(collective.sequence(), 1)
  try ignore(result.collective_exact(result.collective_count())) catch {
    InvalidTensorParallelExecutionPlan(rule=CollectiveSequence, ..) => ()
    _ => fail("unexpected exact collective rejection")
  } noraise {
    _ => fail("out-of-range exact collective was admitted")
  }
}

///|
fn assert_collective_geometry(
  result : TensorParallelExecutionPlan,
) -> Unit raise {
  for index in 0..<result.collective_count() {
    let collective = result.collective(index).unwrap()
    assert_eq(collective.sequence(), index + 1)
    assert_true(collective.local_send_bytes() > 0L)
    assert_true(collective.receive_bytes() >= collective.local_send_bytes())
    assert_true(
      collective.local_send_offset_bytes() >= collective.region().offset_bytes(),
    )
    assert_true(
      collective.local_send_offset_bytes() + collective.local_send_bytes() <=
      collective.region().end_offset_bytes(),
    )
    match collective.kind() {
      InPlaceSumAllReduce => {
        assert_eq(
          collective.local_send_offset_bytes(),
          collective.region().offset_bytes(),
        )
        assert_eq(collective.local_send_bytes(), collective.receive_bytes())
      }
      InPlaceLastAxisAllGather => {
        assert_eq(
          collective.receive_bytes(),
          collective.local_send_bytes() * 2L,
        )
        assert_eq(
          collective.local_send_offset_bytes(),
          collective.region().offset_bytes() + collective.local_send_bytes(),
        )
      }
    }
  }
  assert_eq(
    result.collective(result.collective_count() - 1).unwrap().kind(),
    InPlaceLastAxisAllGather,
  )
}

///|
async test "execution plan authenticates exact physical spaces liveness artifacts and collectives" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-tp-execution-plan")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let file = exec_weight_file()
    @fs.write_file(
      "\{directory}/weights.safetensors",
      file,
      create_mode=CreateNew,
    )
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => root.close())
    }
    let source = build_execution_fixture(root, file, 1, "module-primary")
    let result = admit(
      source.model,
      source.launch_set,
      source.device_plan,
      source.artifacts,
      source.profiles[0],
      exec_limits(),
    )
    let replay = admit(
      source.model,
      source.launch_set,
      source.device_plan,
      source.artifacts,
      source.profiles[0],
      exec_limits(),
    )
    assert_eq(result, replay)
    assert_plan_geometry(source, result)
    assert_activation_liveness(result)
    assert_operation_artifacts_and_weights(source, result)
    assert_collective_geometry(result)

    let second_profile = admit(
      source.model,
      source.launch_set,
      source.device_plan,
      source.artifacts,
      source.profiles[1],
      exec_limits(),
    )
    let rank_zero = build_execution_fixture(root, file, 0, "module-primary")
    let rank_zero_plan = admit(
      rank_zero.model,
      rank_zero.launch_set,
      rank_zero.device_plan,
      rank_zero.artifacts,
      rank_zero.profiles[0],
      exec_limits(),
    )
    let alternate = build_execution_fixture(root, file, 1, "module-alternate")
    let alternate_plan = admit(
      alternate.model,
      alternate.launch_set,
      alternate.device_plan,
      alternate.artifacts,
      alternate.profiles[0],
      exec_limits(),
    )
    assert_true(result.digest() != second_profile.digest())
    assert_true(result.digest() != rank_zero_plan.digest())
    assert_true(result.digest() != alternate_plan.digest())
    assert_true(
      digest_hex_difference(result.digest(), second_profile.digest()) >= 16,
    )
    assert_true(
      digest_hex_difference(result.digest(), rank_zero_plan.digest()) >= 16,
    )
    assert_true(
      digest_hex_difference(result.digest(), alternate_plan.digest()) >= 16,
    )

    let limits = exec_limits()
    expect_execution_rule(
      () => {
        ignore(
          admit(
            source.model,
            source.launch_set,
            rank_zero.device_plan,
            source.artifacts,
            source.profiles[0],
            limits,
          ),
        )
      },
      Rank,
    )
    expect_execution_rule(
      () => {
        ignore(
          admit(
            source.model,
            source.launch_set,
            source.device_plan,
            alternate.artifacts,
            source.profiles[0],
            limits,
          ),
        )
      },
      ArtifactBundle,
    )
    let substituted_profile = exec_profile(1, 3)
    expect_execution_rule(
      () => {
        ignore(
          admit(
            source.model,
            source.launch_set,
            source.device_plan,
            source.artifacts,
            substituted_profile,
            limits,
          ),
        )
      },
      Profile,
    )
  }
}
