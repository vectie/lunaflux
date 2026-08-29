///|
fn tpdp_expect_rule(
  body : () -> Unit raise TensorParallelDevicePlanError,
  expected : TensorParallelDevicePlanRule,
) -> Unit raise {
  try body() catch {
    InvalidTensorParallelDevicePlan(rule~, index=_) => assert_eq(rule, expected)
  } noraise {
    _ => fail("invalid tensor-parallel device evidence was admitted")
  }
}

///|
test "rank device rejects invalid process-visible ordinals" {
  let target = tpdp_target()
  for ordinal in [-1, 1024] {
    tpdp_expect_rule(
      () => {
        ignore(
          TensorParallelRankDevice::single(device_ordinal=ordinal, target~),
        )
      },
      DeviceOrdinal,
    )
  }
}

///|
test "device plan rejects model and materialized region identity drift" {
  let (model_plan, bindings) = tpdp_fixture()
  let (foreign_model, _foreign_bindings) = tpdp_fixture(
    content="abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
  )
  let rank_plan = tpdp_rank_plan(model_plan, bindings, 0, 2)
  let changed_regions = tpdp_rank_plan(
    model_plan,
    bindings,
    0,
    2,
    alignment=128L,
  )
  let target = tpdp_target()
  let device = TensorParallelRankDevice::from_local_topology(
    tpdp_two_rank_topology(target),
    rank=0,
  )
  tpdp_expect_rule(
    () => {
      ignore(
        build_authenticated(foreign_model, rank_plan, rank_plan, device, target),
      )
    },
    ModelIdentity,
  )
  tpdp_expect_rule(
    () => {
      ignore(
        build_authenticated(
          model_plan, rank_plan, changed_regions, device, target,
        ),
      )
    },
    MaterializedRankPlan,
  )
}

///|
test "device plan rejects rank world and target substitution" {
  let (model_plan, bindings) = tpdp_fixture()
  let rank_plan = tpdp_rank_plan(model_plan, bindings, 1, 2)
  let rank_zero_plan = tpdp_rank_plan(model_plan, bindings, 0, 2)
  let target = tpdp_target()
  let wrong_target = tpdp_target(minor=1)
  let topology = tpdp_two_rank_topology(target)
  tpdp_expect_rule(
    () => {
      ignore(
        build_authenticated(
          model_plan,
          rank_plan,
          rank_plan,
          TensorParallelRankDevice::from_local_topology(topology, rank=0),
          target,
        ),
      )
    },
    Rank,
  )
  tpdp_expect_rule(
    () => {
      ignore(
        build_authenticated(
          model_plan,
          rank_zero_plan,
          rank_zero_plan,
          TensorParallelRankDevice::single(device_ordinal=5, target~),
          target,
        ),
      )
    },
    WorldSize,
  )
  tpdp_expect_rule(
    () => {
      ignore(
        build_authenticated(
          model_plan,
          rank_plan,
          rank_plan,
          TensorParallelRankDevice::from_local_topology(topology, rank=1),
          wrong_target,
        ),
      )
    },
    DeviceTarget,
  )
}

///|
async test "public device plan binds authenticated file regions and topology" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-tp-device-plan")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let file = tpdp_weight_file()
    @fs.write_file(
      "\{directory}/weights.safetensors",
      file,
      create_mode=CreateNew,
    )
    let model = tpdp_file_model(file)
    let model_plan = @llama.build(model)
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    let inspection = @device_materialize.inspect_sharded_file(
      root,
      model,
      model_plan,
      @approved_fs.ApprovedRelativeLocator::new("weights.safetensors"),
      rank=1,
      world_size=2,
      alignment_bytes=64L,
      max_arena_bytes=4096L,
      limits=tpdp_file_limits(),
    )
    let target = tpdp_target()
    let result = build(
      model_plan,
      inspection.rank_plan(),
      inspection,
      TensorParallelRankDevice::from_local_topology(
        tpdp_two_rank_topology(target),
        rank=1,
      ),
      target,
    )
    assert_eq(result.identity(), model_plan.identity())
    assert_eq(result.rank(), 1)
    assert_eq(result.device_ordinal(), 5)
    guard result.tensor(@plan.TensorRef::new(result.tensor_count() - 1))
      is Some(logits) else {
      fail("final tensor view is missing")
    }
    assert_eq(logits.local_rows(), 4L)
    root.close()
  }
}
