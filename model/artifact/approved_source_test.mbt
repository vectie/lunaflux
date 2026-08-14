///|
test "artifact source validates fields before bounded duplicate admission" {
  for
    pair in [
      ("", "", "weights.bin", @artifact.ModelConfig),
      ("config.json", "../tokenizer.json", "weights.bin", Tokenizer),
      ("config.json", "tokenizer.json", "/weights.bin", Weights),
    ] {
    try
      @artifact.ArtifactSource::new(
        model_config=pair.0,
        tokenizer=pair.1,
        weights=pair.2,
      )
    catch {
      InvalidLocator(file) => assert_eq(file, pair.3)
      _ => fail("unexpected invalid artifact locator error")
    } noraise {
      _ => fail("invalid artifact locator was admitted")
    }
  }
  try
    @artifact.ArtifactSource::new(
      model_config="config.json",
      tokenizer="config.json",
      weights="weights.bin",
    )
  catch {
    DuplicateLocator(Tokenizer) => ()
    _ => fail("unexpected duplicate artifact locator error")
  } noraise {
    _ => fail("duplicate artifact locator was admitted")
  }
}

///|
async test "closed caller root maps to exact first-file authority failure" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-artifact-closed-root")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let config = selected_config()
    let tokenizer = selected_tokenizer()
    let weights = safetensors_file()
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    root.close()
    try
      @artifact.load_reference_bundle(
        root,
        test_source(),
        test_expected(config, tokenizer, weights),
        test_limits(),
      )
    catch {
      FileIo(file=ModelConfig, operation=OpenFile, issue=AuthorityClosed) => ()
      _ => fail("unexpected closed artifact-root failure")
    } noraise {
      _ => fail("closed artifact root published a bundle")
    }
    assert_true(root.is_closed())
  }
}

///|
async test "artifact source remains pinned across hostile root-path replacement" {
  @async.with_task_group() <| group => {
    let parent = @fs.realpath(
      @fs.tmpdir(prefix="lunaflux-artifact-pinned-root"),
    )
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(parent, recursive=true))
    }
    let root_path = "\{parent}/root"
    let moved_path = "\{parent}/moved"
    @fs.mkdir(root_path)
    let config = selected_config()
    let tokenizer = selected_tokenizer()
    let weights = safetensors_file()
    write_bundle(root_path, config, tokenizer, weights)
    let root = @approved_fs.ApprovedRoot::open_absolute(root_path)
    @fs.rename(root_path, moved_path)
    @fs.mkdir(root_path)
    write_bundle(
      root_path,
      selected_config(executable=true),
      b"hostile-tokenizer",
      b"hostile-weights",
    )
    let bundle = @artifact.load_reference_bundle(
      root,
      test_source(),
      test_expected(config, tokenizer, weights),
      test_limits(),
    )
    assert_eq(bundle.model_config_bytes(), config)
    assert_eq(bundle.tokenizer_bytes(), tokenizer)
    assert_eq(bundle.weights_bytes(), weights)
    assert_false(root.is_closed())
    root.close()
  }
}
