///|
async test "valid explicit bundle returns exact bytes without hashing any payload" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-artifact")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let config = selected_config()
    let tokenizer = selected_tokenizer()
    let weights = safetensors_file()
    let source = test_source()
    write_bundle(directory, config, tokenizer, weights)
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    let bundle = @artifact.load_reference_bundle(
      root,
      source,
      test_expected(config, tokenizer, weights),
      test_limits(),
    )
    assert_eq(bundle.model_config_bytes(), config)
    assert_eq(bundle.tokenizer_bytes(), tokenizer)
    assert_eq(bundle.weights_bytes(), weights)
    assert_eq(bundle.model().spec().hidden_size(), 2)
    assert_eq(bundle.tokenizer().vocab_size(), 256)
    assert_eq(bundle.safetensors().tensor_count(), 1)
    assert_eq(
      bundle.identity().model_content().as_hex(),
      artifact_digest(weights).as_hex(),
    )
    assert_eq(
      bundle.identity().tokenizer().as_string(),
      artifact_digest(tokenizer).as_hex(),
    )
    assert_false(root.is_closed())
    let labeled = @artifact.load_reference_bundle(
      root,
      source,
      @artifact.ExpectedDigests::new(
        model_config=@artifact.ArtifactDigest::parse("a".repeat(64)),
        tokenizer=@artifact.ArtifactDigest::parse("b".repeat(64)),
        weights=@artifact.ArtifactDigest::parse("f".repeat(64)),
      ),
      test_limits(),
    )
    assert_eq(labeled.weights_bytes(), weights)
    assert_eq(labeled.identity().model_config().as_hex(), "a".repeat(64))
    assert_eq(labeled.identity().tokenizer().as_string(), "b".repeat(64))
    assert_eq(labeled.identity().model_content().as_hex(), "f".repeat(64))
    root.close()
  }
}

///|
async test "tokenizer file loader retains labels without hashing and rejects malformed bytes" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-tokenizer-label")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let root_label = @fs.realpath(directory)
    let root = @approved_fs.ApprovedRoot::open_absolute(root_label)
    defer root.close()
    let path = "\{directory}/tokenizer.json"
    @fs.write_file(path, selected_tokenizer(), create_mode=CreateNew)
    let limits = test_limits()
    let file_limits = @json_file.TokenizerFileLimits::new(
      max_file_bytes=65536,
      max_json_depth=16,
    )
    for label in ["a", "b"] {
      let digest = @tokenizer.TokenizerDigest::parse(label.repeat(64))
      let admission = @json_file.load(
        root,
        root_label,
        "tokenizer.json",
        digest,
        file_limits,
        limits.tokenizer_limits,
      )
      assert_eq(admission.digest(), digest)
      assert_eq(admission.spec().digest(), digest)
      assert_eq(admission.spec().vocab_size(), 256)
      assert_true(
        admission.matches_source(
          root_label,
          "tokenizer.json",
          digest,
          file_limits,
          limits.tokenizer_limits,
        ),
      )
    }
    let digest = @tokenizer.TokenizerDigest::parse("c".repeat(64))
    try
      @json_file.load(
        root,
        root_label,
        "tokenizer.json",
        digest,
        @json_file.TokenizerFileLimits::new(
          max_file_bytes=16,
          max_json_depth=16,
        ),
        limits.tokenizer_limits,
      )
    catch {
      FileTooLarge(maximum=16L) => ()
      _ => fail("expected tokenizer allocation bound")
    } noraise {
      _ => fail("oversized tokenizer accepted")
    }
    @fs.write_file(path, b"{", create_mode=CreateOrTruncate)
    try
      @json_file.load(
        root,
        root_label,
        "tokenizer.json",
        digest,
        file_limits,
        limits.tokenizer_limits,
      )
    catch {
      InvalidTokenizer => ()
      _ =>
        fail("malformed tokenizer must fail parsing, not checksum validation")
    } noraise {
      _ => fail("malformed tokenizer accepted")
    }
  }
}

///|
async test "missing oversized malformed and path-type failures are payload safe" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-artifact-failures")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let source = test_source()
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    let config = selected_config()
    let tokenizer = selected_tokenizer()
    let weights = safetensors_file()
    let expected = test_expected(config, tokenizer, weights)
    try
      @artifact.load_reference_bundle(root, source, expected, test_limits())
    catch {
      FileUnavailable(ModelConfig) => ()
      _ => fail("expected missing config")
    } noraise {
      _ => fail("expected failure")
    }
    @fs.write_file("\{directory}/config.json", config, create_mode=CreateNew)
    try
      @artifact.load_reference_bundle(
        root,
        source,
        expected,
        test_limits(max_model_config_bytes=16L),
      )
    catch {
      FileTooLarge(file=ModelConfig, maximum=16L) => ()
      _ => fail("expected config size limit")
    } noraise {
      _ => fail("expected failure")
    }
    try
      @artifact.load_reference_bundle(
        root,
        source,
        expected,
        test_limits(max_model_config_bytes=16L, max_total_bytes=8L),
      )
    catch {
      FileTooLarge(file=ModelConfig, maximum=16L) => ()
      _ => fail("per-file limit did not precede aggregate limit")
    } noraise {
      _ => fail("file exceeding both limits was admitted")
    }
    let mutated_array = config.to_array()
    mutated_array[0] = b'['
    let mutated = Bytes::from_array(mutated_array)
    assert_eq(mutated.length(), config.length())
    @fs.write_file(
      "\{directory}/config.json",
      mutated,
      create_mode=CreateOrTruncate,
    )
    try
      @artifact.load_reference_bundle(root, source, expected, test_limits())
    catch {
      InvalidModelConfig(_) => ()
      _ => fail("expected malformed config rejection without hashing")
    } noraise {
      _ => fail("expected failure")
    }
    let symlink_path = "\{directory}/config.link"
    @fs.symlink(target="\{directory}/config.json", symlink_path)
    let symlink_source = test_source(model_config="config.link")
    try
      @artifact.load_reference_bundle(
        root,
        symlink_source,
        expected,
        test_limits(),
      )
    catch {
      UnsupportedPathType(ModelConfig) => ()
      _ => fail("expected symlink rejection")
    } noraise {
      _ => fail("expected failure")
    }
    @fs.mkdir("\{directory}/config-dir")
    let directory_source = test_source(model_config="config-dir")
    try
      @artifact.load_reference_bundle(
        root,
        directory_source,
        expected,
        test_limits(),
      )
    catch {
      UnsupportedPathType(ModelConfig) => ()
      _ => fail("expected path-type rejection")
    } noraise {
      _ => fail("expected failure")
    }
    root.close()
  }
}

///|
async test "truncated weights and executable metadata are rejected structurally" {
  @async.with_task_group() <| group => {
    let truncated_directory = @fs.tmpdir(prefix="lunaflux-artifact-truncated")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => {
        @fs.rmdir(truncated_directory, recursive=true)
      })
    }
    let config = selected_config()
    let tokenizer = selected_tokenizer()
    let truncated = safetensors_file(truncated=true)
    let truncated_source = test_source()
    write_bundle(truncated_directory, config, tokenizer, truncated)
    let truncated_root = @approved_fs.ApprovedRoot::open_absolute(
      @fs.realpath(truncated_directory),
    )
    try
      @artifact.load_reference_bundle(
        truncated_root,
        truncated_source,
        test_expected(config, tokenizer, truncated),
        test_limits(),
      )
    catch {
      InvalidSafetensors(TruncatedHeader) => ()
      _ => fail("expected truncated safetensors")
    } noraise {
      _ => fail("expected failure")
    }
    truncated_root.close()

    let unsafe_directory = @fs.tmpdir(prefix="lunaflux-artifact-unsafe")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => {
        @fs.rmdir(unsafe_directory, recursive=true)
      })
    }
    let unsafe_config = selected_config(executable=true)
    let weights = safetensors_file()
    let unsafe_source = test_source()
    write_bundle(unsafe_directory, unsafe_config, tokenizer, weights)
    let unsafe_root = @approved_fs.ApprovedRoot::open_absolute(
      @fs.realpath(unsafe_directory),
    )
    try
      @artifact.load_reference_bundle(
        unsafe_root,
        unsafe_source,
        test_expected(unsafe_config, tokenizer, weights),
        test_limits(),
      )
    catch {
      InvalidModelConfig(ExecutableMetadata(AutoMap)) => ()
      _ => fail("expected executable metadata rejection")
    } noraise {
      _ => fail("expected failure")
    }
    unsafe_root.close()
  }
}

///|
async test "model and tokenizer vocabularies must agree before weight loading" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-artifact-vocabulary")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let config = selected_config().to_array()
    let text = try! @utf8.decode(Bytes::from_array(config)).replace(
      old="\"vocab_size\":256",
      new="\"vocab_size\":257",
    )
    let incompatible_config = @utf8.encode(text)
    let tokenizer = selected_tokenizer()
    let weights = safetensors_file()
    let source = test_source()
    write_bundle(directory, incompatible_config, tokenizer, weights)
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    try
      @artifact.load_reference_bundle(
        root,
        source,
        test_expected(incompatible_config, tokenizer, weights),
        test_limits(),
      )
    catch {
      TokenizerVocabularyMismatch => ()
      _ => fail("expected tokenizer vocabulary mismatch")
    } noraise {
      _ => fail("expected incompatible tokenizer vocabulary")
    }
    root.close()
  }
}
