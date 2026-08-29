///|
fn fixture_config() -> Bytes {
  let document =
    #|{"architectures":["LlamaForCausalLM"],"model_type":"llama","vocab_size":32000,"hidden_size":16,"intermediate_size":32,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"max_position_embeddings":128,"rope_theta":10000.0,"torch_dtype":"bfloat16","tie_word_embeddings":false,"rope_scaling":null,"sliding_window":null,"hidden_act":"silu","rms_norm_eps":0.000001,"attention_bias":false,"attention_dropout":0.0,"mlp_bias":false,"head_dim":4,"pretraining_tp":1}
  @utf8.encode(document)
}

///|
fn fixture_digest(
  bytes : BytesView,
) -> @config_file.ModelConfigDigest raise @config_file.ModelConfigFileError {
  @config_file.ModelConfigDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
}

///|
fn fixture_content(
  value? : String = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
) -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(value)
}

///|
fn fixture_limits(
  maximum_bytes? : Int64 = 4096L,
) -> @config_file.ModelConfigFileLimits {
  try! @config_file.ModelConfigFileLimits::new(maximum_bytes~, maximum_depth=8)
}

///|
async test "approved config snapshot authenticates before parsing and binds independent model content" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-model-config-file")
    let root_path = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root_path, recursive=true))
    }
    let bytes = fixture_config()
    @fs.write_file(
      "\{root_path}/config.json",
      bytes,
      create_mode=CreateOrTruncate,
    )
    let root = @approved_fs.ApprovedRoot::open_absolute(root_path)
    let locator = @approved_fs.ApprovedRelativeLocator::new("config.json")
    let content = fixture_content()
    let metadata = @config_file.load(
      root,
      locator,
      fixture_digest(bytes),
      content,
      fixture_limits(),
    )
    assert_eq(metadata.content_digest(), content)
    assert_eq(metadata.spec().hidden_size(), 16)
    assert_false(root.is_closed())
    root.close()
  }
}

///|
async test "digest mismatch wins before malformed configuration parsing" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-model-config-digest")
    let root_path = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root_path, recursive=true))
    }
    @fs.write_file(
      "\{root_path}/config.json",
      b"not-json",
      create_mode=CreateOrTruncate,
    )
    let root = @approved_fs.ApprovedRoot::open_absolute(root_path)
    let locator = @approved_fs.ApprovedRelativeLocator::new("config.json")
    try
      @config_file.load(
        root,
        locator,
        fixture_digest(b"different"),
        fixture_content(),
        fixture_limits(),
      )
    catch {
      DigestMismatch => ()
      _ => fail("unexpected pre-parse digest failure")
    } noraise {
      _ => fail("digest-invalid malformed config was admitted")
    }
    try
      @config_file.load(
        root,
        locator,
        fixture_digest(b"not-json"),
        fixture_content(),
        fixture_limits(),
      )
    catch {
      InvalidConfiguration(MalformedJson) => ()
      _ => fail("unexpected authenticated parse failure")
    } noraise {
      _ => fail("malformed authenticated config was admitted")
    }
    root.close()
  }
}

///|
async test "snapshot and authority failures remain payload safe" {
  @async.with_task_group() <| group => {
    let temporary = @fs.tmpdir(prefix="lunaflux-model-config-errors")
    let root_path = @fs.realpath(temporary)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(root_path, recursive=true))
    }
    @fs.write_file(
      "\{root_path}/config.json",
      fixture_config(),
      create_mode=CreateOrTruncate,
    )
    let root = @approved_fs.ApprovedRoot::open_absolute(root_path)
    let locator = @approved_fs.ApprovedRelativeLocator::new("config.json")
    try
      @config_file.load(
        root,
        locator,
        fixture_digest(fixture_config()),
        fixture_content(),
        fixture_limits(maximum_bytes=16L),
      )
    catch {
      FileTooLarge(maximum=16L) => ()
      _ => fail("unexpected bounded snapshot error")
    } noraise {
      _ => fail("oversized config was admitted")
    }
    root.close()
    try
      @config_file.load(
        root,
        locator,
        fixture_digest(fixture_config()),
        fixture_content(),
        fixture_limits(),
      )
    catch {
      FileIo(operation=OpenFile, issue=AuthorityClosed) => ()
      _ => fail("unexpected closed-authority error")
    } noraise {
      _ => fail("closed root opened a config")
    }
  }
}

///|
test "expected digest and limits reject noncanonical inputs" {
  for
    value in [
      "", "ABC", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    ] {
    try @config_file.ModelConfigDigest::from_sha256(value) catch {
      InvalidExpectedDigest => ()
      _ => fail("unexpected invalid digest error")
    } noraise {
      _ => fail("invalid expected digest was accepted")
    }
  }
  for pair in [(0L, 8), (1L, 0), (67108865L, 8), (1L, 65)] {
    try
      @config_file.ModelConfigFileLimits::new(
        maximum_bytes=pair.0,
        maximum_depth=pair.1,
      )
    catch {
      InvalidLimit(_) => ()
      _ => fail("unexpected invalid config-file limit error")
    } noraise {
      _ => fail("invalid config-file limit was accepted")
    }
  }
}
