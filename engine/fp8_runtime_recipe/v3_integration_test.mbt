///|
async test "reusable v3 packs one disjoint evidence region per FP8 operation" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-fp8-runtime-v3")
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
    let v2 = admit_v2(
      fixture.startup,
      inspection,
      inputs.launches,
      inputs.weights,
      runtime_v2_limits(),
    )
    let profile = fixture.contracts.profiles()[0]
    let v3 = admit_reusable_paged_v3(
      v2,
      profile,
      recipe_v2_layout(fixture.plan),
      Fp8ReusablePagedRecipeV3Limits::new(
        max_launches=8,
        max_workspace_arena_bytes=64L,
        max_canonical_bytes=65536,
      ),
    )
    assert_eq(v3.launches().length(), 3)
    assert_eq(v3.workspace_capacity_bytes(), 16L)
    assert_eq(v3.launches()[0].workspace_offset(), 0L)
    assert_eq(v3.launches()[0].workspace_bytes(), 4L)
    assert_eq(v3.launches()[1].workspace_offset(), 4L)
    assert_eq(v3.launches()[1].workspace_bytes(), 8L)
    assert_eq(v3.launches()[2].workspace_offset(), 12L)
    assert_eq(v3.launches()[2].workspace_bytes(), 4L)
    // V2 remains the exact single-shot authority with its original shared
    // 64-byte storage contract; v3 is a separately domain-bound reusable
    // envelope and never reinterprets or mutates those v2 bytes.
    assert_eq(v2.workspace_capacity_bytes(), 64L)
    for launch in v2.launches() {
      assert_eq(launch.workspace().offset(), 0L)
    }
    assert_true(v2.canonical_bytes() != v3.canonical_bytes())
    assert_true(v2.digest().as_hex() != v3.digest().as_hex())
    for index in 1..<v3.launches().length() {
      let previous = v3.launches()[index - 1]
      let current = v3.launches()[index]
      assert_true(
        previous.workspace_offset() + previous.workspace_bytes() <=
        current.workspace_offset(),
      )
    }
    let repeated = admit_reusable_paged_v3(
      v2,
      profile,
      recipe_v2_layout(fixture.plan),
      Fp8ReusablePagedRecipeV3Limits::new(
        max_launches=8,
        max_workspace_arena_bytes=64L,
        max_canonical_bytes=65536,
      ),
    )
    assert_eq(v3.digest(), repeated.digest())
    assert_eq(v3.canonical_bytes(), repeated.canonical_bytes())
    root.close()
  }
}

///|
async test "reusable v3 rejects a packed evidence arena below the exact sum" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-fp8-runtime-v3-small")
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
    let v2 = admit_v2(
      fixture.startup,
      inspection,
      inputs.launches,
      inputs.weights,
      runtime_v2_limits(),
    )
    try
      admit_reusable_paged_v3(
        v2,
        fixture.contracts.profiles()[0],
        recipe_v2_layout(fixture.plan),
        Fp8ReusablePagedRecipeV3Limits::new(
          max_launches=8,
          max_workspace_arena_bytes=15L,
          max_canonical_bytes=65536,
        ),
      )
    catch {
      InvalidFp8ReusablePagedRecipeV3(Workspace) => ()
      _ => fail("unexpected undersized reusable evidence arena rejection")
    } noraise {
      _ => fail("undersized reusable evidence arena was admitted")
    }
    root.close()
  }
}
