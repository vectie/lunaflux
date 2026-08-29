///|
fn inspect_hostile_mistral_file(
  root : @approved_fs.ApprovedRoot,
  metadata : @mistral_spec.MistralModelMetadata,
  name : String,
  file : BytesView,
) -> @mistral_weights.MistralWeightFileInspection raise @mistral_weights.MistralWeightsError {
  @mistral_weights.inspect_paged_file(
    root,
    metadata,
    try! @approved_fs.ApprovedRelativeLocator::new(name),
    expected_artifact_digest=mistral_fixture_artifact_digest(file),
    limits=mistral_fixture_limits(),
    alignment_bytes=64L,
    max_arena_bytes=131072L,
  )
}

///|
async test "Mistral file facade authenticates the pinned numeric artifact and closes its root" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-mistral-weights")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let metadata = mistral_fixture_metadata()
    let plan = @mistral.build_paged(metadata)
    let file = mistral_fixture_numeric_file(plan)
    @fs.write_file(
      canonical + "/weights.numeric.safetensors",
      file,
      create_mode=CreateNew,
    )
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    let inspection = @mistral_weights.inspect_paged_file(
      root,
      metadata,
      @approved_fs.ApprovedRelativeLocator::new("weights.numeric.safetensors"),
      expected_artifact_digest=mistral_fixture_artifact_digest(file),
      limits=mistral_fixture_limits(),
      alignment_bytes=64L,
      max_arena_bytes=131072L,
    )
    assert_eq(inspection.identity(), plan.identity())
    assert_eq(inspection.layout().length(), 21)
    assert_eq(
      inspection.numeric_schema_digest(),
      plan.numeric_binding().digest(),
    )
    root.close()
  }
}

///|
async test "Mistral file facade rejects artifact and sliding-window replay before materialization" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-mistral-replay")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let metadata = mistral_fixture_metadata(window=16)
    let plan = @mistral.build_paged(metadata)
    let file = mistral_fixture_numeric_file(plan)
    @fs.write_file(canonical + "/weights", file, create_mode=CreateNew)
    let locator = @approved_fs.ApprovedRelativeLocator::new("weights")
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    let wrong = @numeric_weight_file.NumericWeightArtifactDigest::from_sha256(
      "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    )
    try
      @mistral_weights.inspect_paged_file(
        root,
        metadata,
        locator,
        expected_artifact_digest=wrong,
        limits=mistral_fixture_limits(),
        alignment_bytes=64L,
        max_arena_bytes=131072L,
      )
    catch {
      InvalidNumericWeightFile(ArtifactDigestMismatch) => ()
      _ => fail("unexpected artifact rejection")
    } noraise {
      _ => fail("foreign artifact digest was admitted")
    }
    try
      @mistral_weights.inspect_paged_file(
        root,
        mistral_fixture_metadata(window=32),
        locator,
        expected_artifact_digest=mistral_fixture_artifact_digest(file),
        limits=mistral_fixture_limits(),
        alignment_bytes=64L,
        max_arena_bytes=131072L,
      )
    catch {
      InvalidNumericWeightFile(ModelPlanMismatch | ModelContentMismatch) => ()
      _ => fail("unexpected sliding-window file replay rejection")
    } noraise {
      _ => fail("sliding-window file replay was admitted")
    }
    root.close()
  }
}

///|
async test "Mistral file facade rejects missing, extra, and mis-shaped numeric tensors" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-mistral-hostile")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let metadata = mistral_fixture_metadata()
    let plan = @mistral.build_paged(metadata)
    let cases : Array[(String, Bytes, @mistral_weights.MistralWeightsError)] = [
      (
        "missing",
        mistral_fixture_numeric_file(plan, omit=5),
        InvalidNumericWeightFile(TensorCountMismatch(expected=21, actual=20)),
      ),
      (
        "extra",
        mistral_fixture_numeric_file(plan, extra=true),
        InvalidNumericWeightFile(TensorCountMismatch(expected=21, actual=22)),
      ),
      (
        "shape",
        mistral_fixture_numeric_file(plan, shape_change=6),
        InvalidNumericWeightFile(InvalidTensorShape(ordinal=6, axis=0)),
      ),
    ]
    for item in cases {
      let (name, file, _) = item
      @fs.write_file(canonical + "/" + name, file, create_mode=CreateNew)
    }
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    for item in cases {
      let (name, file, expected) = item
      try inspect_hostile_mistral_file(root, metadata, name, file) catch {
        error => assert_eq(error, expected)
      } noraise {
        _ => fail("hostile Mistral numeric file was admitted")
      }
    }
    root.close()
  }
}
