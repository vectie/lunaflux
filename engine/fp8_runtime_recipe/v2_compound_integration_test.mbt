///|
fn expect_missing_compound_stage_rejected(
  fixture : RuntimeRecipeV2Fixture,
  inspection : @numeric_weight_file.NumericWeightFileInspection,
) -> Unit raise {
  let entry = fixture.startup.kernel_artifacts().manifest().entries()[1]
  let scales : Array[@fp8_launch_abi.Fp8LaunchStorageRegion] = []
  for binding in entry.weights() {
    scales.push(layout_region(inspection, binding.scale().tensor()))
  }
  let missing_second_cell = try! @fp8_launch_abi.Fp8LaunchStorageRegion::new(
    storage=@fp8_launch_abi.Fp8LaunchStorageIdentity::from_sha256(
      "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    ),
    storage_bytes=64L,
    offset=0L,
    byte_length=4L,
    alignment=4L,
  )
  try
    @fp8_launch_abi.admit_staged_dynamic_activation_scale_launch_v2(
      fixture.plan,
      fixture.startup.kernel_artifacts(),
      PagedV4(fixture.contracts),
      operation=entry.operation(),
      shape=@fp8_launch_abi.Fp8ExactExecutionShape::paged_full_graph(
        profile=try! @launch_contract.KernelProfileId::from_stable_int(1),
        rows=[
          @fp8_launch_abi.Fp8PagedRowShape::new(
            query_tokens=2,
            context_tokens=2,
            page_table_entries=1,
          ),
        ],
      ),
      weight_scale_regions=scales,
      workspace=missing_second_cell,
    )
  catch {
    InvalidFp8LaunchAbi(WorkspaceContract) => ()
    _ => fail("unexpected compound Workspace rejection")
  } noraise {
    _ => fail("compound launch accepted a missing second stage cell")
  }
}

///|
async test "runtime recipe v2 authenticates compound GatedMlp stage cells" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-fp8-runtime-compound-v2")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let fixture = runtime_recipe_v2_fixture()
    let file = recipe_numeric_file(fixture.plan)
    @fs.write_file("\{directory}/weights.bin", file, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    let inspection = inspect_recipe_file(
      root,
      fixture.plan,
      "weights.bin",
      file,
    )
    let inputs = runtime_v2_inputs(fixture, inspection)
    let admitted = admit_v2(
      fixture.startup,
      inspection,
      inputs.launches,
      inputs.weights,
      runtime_v2_limits(),
    )
    let launch = admitted.launches()[1]
    assert_eq(launch.operation_kind(), GatedMlp)
    assert_eq(launch.workspace().offset(), 0L)
    assert_eq(launch.workspace().byte_length(), 8L)
    assert_eq(launch.stages().length(), 2)
    assert_eq(launch.stages()[0].ordinal(), 0)
    assert_eq(launch.stages()[1].ordinal(), 1)
    assert_eq(
      launch.stages()[0].stage(),
      @numeric_contract.ActivationScaleStage::external_operation_input_v1(),
    )
    assert_eq(
      launch.stages()[1].stage(),
      @numeric_contract.ActivationScaleStage::post_silu_gate_up_product_v1(),
    )
    assert_eq(launch.stages()[0].region().offset(), 0L)
    assert_eq(launch.stages()[1].region().offset(), 4L)
    for stage in launch.stages() {
      assert_eq(stage.region().storage(), launch.workspace().storage())
      assert_eq(stage.region().storage_bytes(), 64L)
      assert_eq(stage.region().byte_length(), 4L)
      assert_eq(stage.region().alignment(), 4L)
    }
    assert_eq(
      launch.stages()[0].region().offset() +
      launch.stages()[0].region().byte_length(),
      launch.stages()[1].region().offset(),
    )
    expect_missing_compound_stage_rejected(fixture, inspection)
    try
      runtime_v2_inputs(
        fixture,
        inspection,
        arena=inspection.artifact_digest().as_hex(),
      )
    catch {
      @fp8_launch_abi.InvalidFp8LaunchAbi(RegionStorageMismatch) => ()
      _ => fail("unexpected compound stage-arena alias rejection")
    } noraise {
      _ => fail("compound stage cells aliased authenticated weight storage")
    }
    root.close()
  }
}
