///|
struct RuntimeRecipeV2Inputs {
  launches : Array[@fp8_launch_abi.Fp8StagedDynamicLaunchAbi]
  weights : Array[Fp8LaunchWeightRegionClaimV2]
}

///|
fn runtime_v2_inputs(
  fixture : RuntimeRecipeV2Fixture,
  inspection : @numeric_weight_file.NumericWeightFileInspection,
  arena? : String = "7777777777777777777777777777777777777777777777777777777777777777",
  second_arena? : String,
  second_capacity? : Int64 = 64L,
  weight_offset_delta? : Int64 = 0L,
  scale_offset_delta? : Int64 = 0L,
) -> RuntimeRecipeV2Inputs raise {
  let launches : Array[@fp8_launch_abi.Fp8StagedDynamicLaunchAbi] = []
  let claims : Array[Fp8LaunchWeightRegionClaimV2] = []
  let entries = fixture.startup.kernel_artifacts().manifest().entries()
  let profile = try! @launch_contract.KernelProfileId::from_stable_int(1)
  for index, entry in entries {
    let storage = if index == 1 { second_arena.unwrap_or(arena) } else { arena }
    let capacity = if index == 1 { second_capacity } else { 64L }
    let workspace = try! @fp8_launch_abi.Fp8LaunchStorageRegion::new(
      storage=@fp8_launch_abi.Fp8LaunchStorageIdentity::from_sha256(storage),
      storage_bytes=capacity,
      offset=0L,
      byte_length=fixture.plan.operations()[entry.operation().as_int()].workspace_bytes(),
      alignment=4L,
    )
    let scales : Array[@fp8_launch_abi.Fp8LaunchStorageRegion] = []
    let operands : Array[@fp8_launch_abi.Fp8LaunchStorageRegion] = []
    for binding_index, binding in entry.weights() {
      let mutate = index == 0 && binding_index == 0
      scales.push(
        layout_region(
          inspection,
          binding.scale().tensor(),
          offset_delta=if mutate { scale_offset_delta } else { 0L },
        ),
      )
      operands.push(
        layout_region(
          inspection,
          binding.tensor(),
          offset_delta=if mutate { weight_offset_delta } else { 0L },
        ),
      )
    }
    launches.push(
      @fp8_launch_abi.admit_staged_dynamic_activation_scale_launch_v2(
        fixture.plan,
        fixture.startup.kernel_artifacts(),
        PagedV4(fixture.contracts),
        operation=entry.operation(),
        shape=@fp8_launch_abi.Fp8ExactExecutionShape::paged_full_graph(
          profile~,
          rows=[
            @fp8_launch_abi.Fp8PagedRowShape::new(
              query_tokens=2,
              context_tokens=2,
              page_table_entries=1,
            ),
          ],
        ),
        weight_scale_regions=scales,
        workspace~,
      ),
    )
    claims.push(
      try! Fp8LaunchWeightRegionClaimV2::new(
        profile~,
        operation=entry.operation(),
        regions=operands,
      ),
    )
  }
  { launches, weights: claims }
}

///|
fn runtime_v2_limits(canonical? : Int = 65536) -> Fp8RuntimeRecipeV2Limits {
  try! Fp8RuntimeRecipeV2Limits::new(
    max_launches=8,
    max_workspace_arena_bytes=1024L,
    max_canonical_bytes=canonical,
  )
}

///|
fn expect_v2_rule(
  body : () -> Fp8RuntimeRecipeV2 raise Fp8RuntimeRecipeV2Error,
  expected : Fp8RuntimeRecipeV2Rule,
) -> Unit raise {
  try body() catch {
    InvalidFp8RuntimeRecipeV2(rule) => assert_eq(rule, expected)
  } noraise {
    _ => fail("hostile staged runtime recipe was admitted")
  }
}

///|
async test "runtime recipe v2 binds staged launches and one ordered arena" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-fp8-runtime-recipe-v2")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let fixture = runtime_recipe_v2_fixture()
    assert_eq(
      fixture.startup.version(),
      @fp8_startup_admission.Fp8StartupAdmissionV2Version::v2(),
    )
    assert_true(
      fixture.startup.digest().as_hex() !=
      fixture.legacy_startup.digest().as_hex(),
    )
    assert_eq(
      fixture.startup.model_plan().identity(),
      fixture.legacy_startup.model_plan().identity(),
    )
    assert_eq(
      fixture.startup.kernel_artifacts().manifest().digest(),
      fixture.legacy_startup.kernel_artifacts().manifest().digest(),
    )
    let file = recipe_numeric_file(fixture.plan)
    @fs.write_file("\{directory}/weights.bin", file, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    let inspection = inspect_recipe_file(
      root,
      fixture.plan,
      "weights.bin",
      file,
    )
    let source = runtime_v2_inputs(fixture, inspection)
    let admitted = admit_v2(
      fixture.startup,
      inspection,
      source.launches,
      source.weights,
      runtime_v2_limits(),
    )
    assert_eq(admitted.launches().length(), 3)
    assert_eq(admitted.workspaces().length(), 3)
    assert_eq(admitted.workspace_capacity_bytes(), 64L)
    assert_eq(
      admitted.workspaces()[0].workspace().storage(),
      admitted.workspaces()[1].workspace().storage(),
    )
    assert_eq(
      admitted.workspaces()[1].workspace().storage(),
      admitted.workspaces()[2].workspace().storage(),
    )
    for launch in admitted.launches() {
      assert_eq(launch.version(), @fp8_launch_abi.Fp8LaunchAbiVersion::v2())
      assert_eq(launch.launch_source_version(), PagedNumericV4)
      assert_eq(
        launch.stages()[0].stage(),
        @numeric_contract.ActivationScaleStage::external_operation_input_v1(),
      )
      assert_eq(
        launch.stages()[0].region().storage(),
        launch.workspace().storage(),
      )
      if launch.operation_kind() == GatedMlp {
        assert_eq(launch.stages().length(), 2)
        assert_eq(
          launch.stages()[1].stage(),
          @numeric_contract.ActivationScaleStage::post_silu_gate_up_product_v1(),
        )
      } else {
        assert_eq(launch.stages().length(), 1)
      }
    }
    let legacy = runtime_recipe_fixture()
    try
      @fp8_startup_admission.admit_v2(
        legacy.plan,
        recipe_device(),
        legacy.startup.kernel_artifacts(),
        @fp8_startup_admission.Fp8StartupAdmissionLimits::new(
          max_canonical_bytes=8192,
        ),
      )
    catch {
      Fp8StartupRejected(EntryBinding) => ()
      _ => fail("unexpected legacy startup replay failure")
    } noraise {
      _ => fail("legacy zero-workspace startup replayed as staged v2")
    }
    let repeated_inputs = runtime_v2_inputs(fixture, inspection)
    let repeated = admit_v2(
      fixture.startup,
      inspection,
      repeated_inputs.launches,
      repeated_inputs.weights,
      runtime_v2_limits(),
    )
    assert_eq(admitted.digest(), repeated.digest())
    let digest = admitted.digest()
    source.launches.push(source.launches[0])
    source.weights.push(source.weights[0])
    assert_eq(admitted.launches().length(), 3)
    assert_eq(admitted.digest(), digest)

    let foreign_arena = runtime_v2_inputs(
      fixture,
      inspection,
      second_arena="8888888888888888888888888888888888888888888888888888888888888888",
    )
    expect_v2_rule(
      () => {
        admit_v2(
          fixture.startup,
          inspection,
          foreign_arena.launches,
          foreign_arena.weights,
          runtime_v2_limits(),
        )
      },
      WorkspaceStorage,
    )
    let capacity_drift = runtime_v2_inputs(
      fixture,
      inspection,
      second_capacity=128L,
    )
    expect_v2_rule(
      () => {
        admit_v2(
          fixture.startup,
          inspection,
          capacity_drift.launches,
          capacity_drift.weights,
          runtime_v2_limits(),
        )
      },
      WorkspaceCapacity,
    )
    expect_v2_rule(
      () => {
        admit_v2(
          fixture.startup,
          inspection,
          [
            repeated_inputs.launches[1],
            repeated_inputs.launches[0],
            repeated_inputs.launches[2],
          ],
          repeated_inputs.weights,
          runtime_v2_limits(),
        )
      },
      LaunchOrder,
    )
    let wrong_weight = runtime_v2_inputs(
      fixture,
      inspection,
      weight_offset_delta=16L,
    )
    expect_v2_rule(
      () => {
        admit_v2(
          fixture.startup,
          inspection,
          wrong_weight.launches,
          wrong_weight.weights,
          runtime_v2_limits(),
        )
      },
      WeightRegion,
    )
    let wrong_scale = runtime_v2_inputs(
      fixture,
      inspection,
      scale_offset_delta=-16L,
    )
    expect_v2_rule(
      () => {
        admit_v2(
          fixture.startup,
          inspection,
          wrong_scale.launches,
          wrong_scale.weights,
          runtime_v2_limits(),
        )
      },
      WeightRegion,
    )
    let foreign_fixture = runtime_recipe_v2_fixture(
      content="9999999999999999999999999999999999999999999999999999999999999999",
    )
    let foreign_file = recipe_numeric_file(foreign_fixture.plan)
    @fs.write_file(
      "\{directory}/foreign.bin",
      foreign_file,
      create_mode=CreateNew,
    )
    let foreign_inspection = inspect_recipe_file(
      root,
      foreign_fixture.plan,
      "foreign.bin",
      foreign_file,
    )
    let foreign_launches = runtime_v2_inputs(
      foreign_fixture, foreign_inspection,
    )
    expect_v2_rule(
      () => {
        admit_v2(
          fixture.startup,
          inspection,
          foreign_launches.launches,
          repeated_inputs.weights,
          runtime_v2_limits(),
        )
      },
      LaunchBinding,
    )
    expect_v2_rule(
      () => {
        admit_v2(
          fixture.startup,
          inspection,
          repeated_inputs.launches,
          repeated_inputs.weights,
          runtime_v2_limits(canonical=1),
        )
      },
      CanonicalBytes,
    )
    root.close()
  }
}
