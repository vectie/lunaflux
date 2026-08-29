///|
fn bytelevel_scalar_for_byte(value : Int) -> Int {
  if (value >= 0x21 && value <= 0x7e) ||
    (value >= 0xa1 && value <= 0xac) ||
    (value >= 0xae && value <= 0xff) {
    value
  } else if value <= 0x20 {
    0x100 + value
  } else if value <= 0xa0 {
    0x121 + value - 0x7f
  } else {
    0x143
  }
}

///|
fn bytelevel_token(value : Int) -> String {
  bytelevel_scalar_for_byte(value).to_char().unwrap().to_string()
}

///|
fn selected_config(executable? : Bool = false) -> Bytes {
  let suffix = if executable {
    ",\"auto_map\":{\"AutoModel\":\"unsafe.py\"}"
  } else {
    ""
  }
  let base = "{\"architectures\":[\"LlamaForCausalLM\"],\"model_type\":\"llama\",\"vocab_size\":256,\"hidden_size\":2,\"intermediate_size\":2,\"num_hidden_layers\":1,\"num_attention_heads\":1,\"num_key_value_heads\":1,\"max_position_embeddings\":8,\"rope_theta\":10000.0,\"torch_dtype\":\"bfloat16\",\"tie_word_embeddings\":false,\"rms_norm_eps\":0.000001"
  @utf8.encode(base + suffix + "}")
}

///|
fn selected_tokenizer() -> Bytes {
  let vocab : Map[String, Json] = Map([])
  for value in 0..<256 {
    vocab[bytelevel_token(value)] = Json::number(value.to_double())
  }
  @utf8.encode(
    Json::object({
      "version": Json::string("1.0"),
      "truncation": Json::null(),
      "padding": Json::null(),
      "added_tokens": Json::array([]),
      "normalizer": Json::null(),
      "pre_tokenizer": Json::object({
        "type": Json::string("ByteLevel"),
        "add_prefix_space": Json::boolean(false),
        "trim_offsets": Json::boolean(true),
        "use_regex": Json::boolean(false),
      }),
      "post_processor": Json::null(),
      "decoder": Json::object({
        "type": Json::string("ByteLevel"),
        "add_prefix_space": Json::boolean(false),
        "trim_offsets": Json::boolean(true),
        "use_regex": Json::boolean(false),
      }),
      "model": Json::object({
        "type": Json::string("BPE"),
        "dropout": Json::null(),
        "unk_token": Json::null(),
        "continuing_subword_prefix": Json::null(),
        "end_of_word_suffix": Json::null(),
        "fuse_unk": Json::boolean(false),
        "byte_fallback": Json::boolean(false),
        "ignore_merges": Json::boolean(false),
        "vocab": Json::object(vocab),
        "merges": Json::array([]),
      }),
    }).stringify(),
  )
}

///|
fn safetensors_file(truncated? : Bool = false) -> Bytes {
  let header = if truncated {
    b"{"
  } else {
    @utf8.encode(
      "{\"w\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}",
    )
  }
  let declared = if truncated { 2 } else { header.length() }
  let output : Array[Byte] = []
  for shift in 0..<8 {
    output.push((declared.to_uint64() >> (shift * 8)).to_byte())
  }
  for byte in header {
    output.push(byte)
  }
  if !truncated {
    output.push(0)
    output.push(0)
  }
  Bytes::from_array(output)
}

///|
fn artifact_digest(bytes : BytesView) -> @artifact.ArtifactDigest {
  try! @artifact.ArtifactDigest::parse(
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
}

///|
fn test_source(
  model_config? : StringView = "config.json",
  tokenizer? : StringView = "tokenizer.json",
  weights? : StringView = "model.safetensors",
) -> @artifact.ArtifactSource {
  try! @artifact.ArtifactSource::new(model_config~, tokenizer~, weights~)
}

///|
fn test_expected(
  config : BytesView,
  tokenizer : BytesView,
  weights : BytesView,
) -> @artifact.ExpectedDigests {
  @artifact.ExpectedDigests::new(
    model_config=artifact_digest(config),
    tokenizer=artifact_digest(tokenizer),
    weights=artifact_digest(weights),
  )
}

///|
fn test_limits(
  max_model_config_bytes? : Int64 = 4096L,
  max_total_bytes? : Int64 = 131072L,
) -> @artifact.ArtifactLimits {
  let tokenizer_limits : @tokenizer.TokenizerLimits = {
    max_input_bytes: 1024,
    max_output_tokens: 1024,
    max_decoded_bytes: 4096,
    max_vocab_entries: 512,
    max_merge_rules: 128,
    max_token_bytes: 64,
    max_special_tokens: 16,
  }
  let safetensors_limits = try! @safetensors.HeaderLimits::new(
    max_header_bytes=4096,
    max_tensor_count=8,
    max_rank=8,
    max_name_length=128,
    max_payload_bytes=4096L,
    max_materialized_tensor_bytes=4096L,
  )
  try! @artifact.ArtifactLimits::new(
    max_model_config_bytes~,
    max_tokenizer_bytes=65536L,
    max_weights_bytes=65536L,
    max_total_bytes~,
    max_model_config_depth=8,
    max_tokenizer_depth=16,
    tokenizer_limits~,
    safetensors_limits~,
  )
}

///|
async fn write_bundle(
  directory : String,
  config : Bytes,
  tokenizer : Bytes,
  weights : Bytes,
) -> Unit {
  @fs.write_file("\{directory}/config.json", config, create_mode=CreateNew)
  @fs.write_file(
    "\{directory}/tokenizer.json",
    tokenizer,
    create_mode=CreateNew,
  )
  @fs.write_file(
    "\{directory}/model.safetensors",
    weights,
    create_mode=CreateNew,
  )
}

///|
async test "valid explicit bundle returns the hashed byte snapshot and metadata" {
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
    root.close()
  }
}

///|
async test "missing oversized digest and path-type failures are payload safe" {
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
      DigestMismatch(ModelConfig) => ()
      _ => fail("expected digest mismatch")
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
async test "truncated weights and executable metadata are rejected after hashing" {
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
