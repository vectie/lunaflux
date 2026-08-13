///|
async test "step envelope rejects before manifest filesystem access" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-execution-preflight")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let metadata = fixture_metadata('0')
    let plan = @llama.build_paged(metadata)
    let layout = fixture_layout(plan)
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    let locator = @approved_fs.ApprovedRelativeLocator::new("missing.json")
    let digest = ExecutionManifestDigest::from_sha256(
      "0000000000000000000000000000000000000000000000000000000000000000",
    )
    try
      load_paged(
        root,
        locator,
        digest,
        plan,
        fixture_weight_layout(metadata),
        layout,
        fixture_target(),
        @device_step.DeviceStepLimits::new(
          max_rows=1,
          max_tokens=4,
          max_pages=2,
          max_sequence_tokens=9,
          max_token_id=7,
        ),
        fixture_limits(plan.operations().length()),
      )
    catch {
      UnsupportedModelEnvelope => ()
      FileUnavailable => fail("manifest opened before step validation")
      _ => fail("unexpected invalid-step failure")
    } noraise {
      _ => fail("invalid step envelope was admitted")
    }
    root.close()
  }
}
