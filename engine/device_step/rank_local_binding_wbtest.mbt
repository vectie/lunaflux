///|
fn expect_rank_local_error(
  body : () -> Unit raise DeviceStepError,
  expected : DeviceStepError,
) -> Unit raise {
  try body() catch {
    error => assert_eq(error, expected)
  } noraise {
    _ => fail("substituted rank-local descriptor evidence was admitted")
  }
}

///|
async test "rank-local descriptor binding accepts only exact KV-plan evidence" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-device-step-rank-local")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let file = rank_local_weight_file()
    @fs.write_file(
      "\{directory}/weights.safetensors",
      file,
      create_mode=CreateNew,
    )
    let metadata = rank_local_metadata(file)
    let model_plan = @llama.build_paged(metadata)
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    let kv_plan = rank_local_kv_plan(root, metadata, model_plan)
    let generation = @worker_protocol.ModelPlanGeneration::from_wire(7UL)
    let limits = DeviceStepLimits::new(
      max_rows=1,
      max_tokens=2,
      max_pages=4,
      max_sequence_tokens=16,
      max_token_id=7,
    )
    let (local_layout, required) = validate_rank_local_plan_binding(
      model_plan, generation, kv_plan, 0, limits,
    )
    assert_eq(local_layout.geometry().kv_head_count(), 1)
    assert_eq(local_layout, kv_plan.rank_layout(0))
    assert_true(
      required.contains(@plan.KernelCapabilityId::paged_causal_attention()),
    )

    let semantic_geometry = model_plan.kv_cache_geometry()
    let constructed_local_geometry = @plan.KvCacheGeometry::new(
      layer_count=semantic_geometry.layer_count(),
      kv_head_count=semantic_geometry.kv_head_count() / 2,
      head_dimension=semantic_geometry.head_dimension(),
      dtype=semantic_geometry.dtype(),
      layout=semantic_geometry.layout(),
    )
    let constructed_local_layout = @device_layout.plan(
      model_plan.identity(),
      constructed_local_geometry,
      @device_layout.DeviceKvLimits::new(
        tokens_per_page=128,
        total_page_count=4,
        max_arena_bytes=9223372036854775807L,
      ),
    )
    assert_eq(constructed_local_layout, local_layout)
    let full_layout = @device_layout.plan(
      model_plan.identity(),
      model_plan.kv_cache_geometry(),
      @device_layout.DeviceKvLimits::new(
        tokens_per_page=128,
        total_page_count=4,
        max_arena_bytes=1048576L,
      ),
    )
    ignore(validate_plan_binding(model_plan, full_layout, limits))
    expect_rank_local_error(
      () => ignore(validate_plan_binding(model_plan, local_layout, limits)),
      InvalidLayout,
    )
    expect_rank_local_error(
      () => {
        ignore(
          validate_plan_binding_for_geometry(
            model_plan, full_layout, constructed_local_geometry, limits,
          ),
        )
      },
      InvalidLayout,
    )

    let stale_generation = @worker_protocol.ModelPlanGeneration::from_wire(8UL)
    expect_rank_local_error(
      () => {
        ignore(
          validate_rank_local_plan_binding(
            model_plan, stale_generation, kv_plan, 0, limits,
          ),
        )
      },
      InvalidPlan(ModelGeneration),
    )
    for rank in [-1, 2] {
      expect_rank_local_error(
        () => {
          ignore(
            validate_rank_local_plan_binding(
              model_plan, generation, kv_plan, rank, limits,
            ),
          )
        },
        InvalidLayout,
      )
    }
    expect_rank_local_error(
      () => {
        ignore(
          validate_rank_local_plan_binding(
            model_plan,
            generation,
            kv_plan,
            0,
            DeviceStepLimits::new(
              max_rows=1,
              max_tokens=2,
              max_pages=4,
              max_sequence_tokens=17,
              max_token_id=7,
            ),
          ),
        )
      },
      InvalidLimit(SequenceTokens),
    )
    root.close()
  }
}
