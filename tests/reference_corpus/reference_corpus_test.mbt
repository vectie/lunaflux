///|
let fixture_directory = "tests/reference_corpus"

///|
async fn read_fixture(name : String) -> Bytes {
  @fs.read_file("\{fixture_directory}/\{name}").binary()
}

///|
fn content_digest() -> @spec.ContentDigest raise {
  @spec.ContentDigest::from_sha256(
    "852db1b39acb2336abc997440c6f6d6e4ab640f91e5e2aa9e2488d5794159d30",
  )
}

///|
fn upstream_tokenizer_digest() -> @artifact.ArtifactDigest raise {
  @artifact.ArtifactDigest::parse(
    "fbcdbe15960e43ef351662e7b77a319ceb294b3c5dc2569c23b729fb87e13d7b",
  )
}

///|
fn header_limits() -> @safetensors.HeaderLimits raise {
  @safetensors.HeaderLimits::new(
    max_header_bytes=4096,
    max_tensor_count=21,
    max_rank=2,
    max_name_length=128,
    max_payload_bytes=208544L,
    max_materialized_tensor_bytes=208544L,
  )
}

///|
fn materialization_limits() -> @materialize.MaterializationLimits raise {
  @materialize.MaterializationLimits::new(
    max_header_bytes=4096,
    max_tensor_count=21,
    max_payload_bytes=208544L,
    max_host_bytes=208544L,
  )
}

///|
fn execution_limits() -> @reference.ExecutionLimits raise {
  @reference.ExecutionLimits::new(
    kernels=@refkernels.ReferenceLimits::new(
      max_rows=3000,
      max_columns=3000,
      max_elements=768000,
    ),
    sampling=@sampling.SamplingLimits::new(256, 3000),
    max_operations=64,
    max_tensors=32,
    max_input_tokens=256,
  )
}

///|
fn assert_logit(
  result : @reference.ExecutionResult,
  row : Int,
  token : Int,
  expected : Float,
) -> Unit raise {
  assert_true(
    result
    .logits()
    .at(row~, column=token)
    .is_close(
      expected,
      relative_tolerance=0.00001F,
      absolute_tolerance=0.00001F,
    ),
  )
}

///|
fn corpus_int(value : Json) -> Int raise {
  guard value is Number(number, repr=_) else {
    fail("corpus integer must be a number")
  }
  let integer = number.to_int()
  if integer.to_double() != number {
    fail("corpus integer must be exact")
  }
  integer
}

///|
fn corpus_float(value : Json) -> Float raise {
  guard value is Number(number, repr=_) else {
    fail("corpus logit must be a number")
  }
  Float::from_double(number)
}

///|
fn corpus_tokens(value : Json) -> Array[Int] raise {
  guard value is Array(values) else {
    fail("corpus token IDs must be an array")
  }
  values.map(corpus_int)
}

///|
fn parse_tokenizer_probe(
  document : Bytes,
  digest : @tokenizer.TokenizerDigest,
  limits : @tokenizer.TokenizerLimits,
) -> @tokenizer.TokenizerSpec raise @json_reader.TokenizerJsonError {
  @json_reader.parse_bytes(
    document,
    digest~,
    limits~,
    max_bytes=4096,
    max_depth=16,
  )
}

///|
fn tokenizer_probe_fails_closed(
  document : Bytes,
  digest : @tokenizer.TokenizerDigest,
  limits : @tokenizer.TokenizerLimits,
) -> Bool {
  try parse_tokenizer_probe(document, digest, limits) catch {
    UnsupportedSemantics(
      component=PostProcessor,
      feature=PostProcessingBehavior
    ) => true
    _ => false
  } noraise {
    _ => false
  }
}

///|
async test "exact pinned BF16 model matches independent logits and greedy tokens" {
  let config = read_fixture("config.json")
  let weights = read_fixture("model.safetensors")
  assert_eq(weights.length(), 210712)
  assert_eq(
    @crypto.bytes_to_hex_string(@crypto.sha256(weights)),
    content_digest().as_hex(),
  )
  let admitted = @llama_admission.admit(
    config,
    weights,
    content_digest(),
    config_max_bytes=2048,
    config_max_depth=8,
    header_limits=header_limits(),
  )
  assert_eq(admitted.safetensors().tensor_count(), 21)
  assert_eq(admitted.bindings().length(), 21)
  let host = @materialize.copy_to_host(
    admitted.bindings(),
    weights,
    limits=materialization_limits(),
  )
  let prepared = @reference.prepare(
    admitted.plan(),
    host,
    limits=execution_limits(),
  )
  guard @json.parse(@utf8.decode(read_fixture("corpus.json"))) is Object(root) &&
    root["model_cases"] is Array(cases) else {
    fail("reference model cases must be a JSON array")
  }
  for case in cases {
    guard case is Object(fields) &&
      fields["sampled_logits"] is Object(sampled_logits) else {
      fail("reference model case must be an object")
    }
    let input_ids = corpus_tokens(fields["input_ids"])
    let execution = @reference.execute(
      admitted.plan(),
      prepared,
      input_ids,
      limits=execution_limits(),
    )
    assert_eq(
      execution.selection().token_id(),
      corpus_int(fields["greedy_token"]),
    )
    for token_text, expected in sampled_logits {
      let token : Int = @strconv.from_str(token_text)
      assert_logit(
        execution,
        input_ids.length() - 1,
        token,
        corpus_float(expected),
      )
    }
    assert_eq(
      @reference.generate(
        admitted.plan(),
        prepared,
        input_ids,
        max_new_tokens=4,
        limits=execution_limits(),
      ).generated_tokens(),
      ReadOnlyArray::from_array(corpus_tokens(fields["greedy_continuation"])),
    )
  }
}

///|
async test "prefix campaign referee is digest pinned and independently replayable" {
  let referee_bytes = read_fixture("prefix_referee.json")
  assert_eq(
    @crypto.bytes_to_hex_string(@crypto.sha256(referee_bytes)),
    "765d4b80750e377fea131ce140a0a67931724ab06c24dc5e322c7e6b295ab8e3",
  )
  guard @json.parse(@utf8.decode(referee_bytes)) is Object(root) &&
    root["schema"] is String("lunaflux.approved-bf16-prefix-referee.v1") &&
    root["revision"] is String("3579d71fd57e04f5a364d824d3a2ec3e913dbb67") &&
    root["model_content_sha256"]
    is String(
      "852db1b39acb2336abc997440c6f6d6e4ab640f91e5e2aa9e2488d5794159d30"
    ) &&
    root["tokenizer_sha256"]
    is String(
      "fbcdbe15960e43ef351662e7b77a319ceb294b3c5dc2569c23b729fb87e13d7b"
    ) &&
    root["reference_corpus_sha256"]
    is String(
      "f4ade303b1b7b3264e86caa15b77cdb2805244e1142b9b74225ed3f36438f1d0"
    ) &&
    root["referee"] is String("approved_scalar_bf16_reference_executor") &&
    root["cases"] is Array([first, second]) else {
    fail("prefix referee identity changed")
  }
  let expected_inputs = [
    [1, 229, 153, 132, 75, 104, 111, 111],
    [1, 229, 153, 132, 75, 104, 111, 111, 114],
  ]
  let expected_tokens = [1355, 1240]
  let config = read_fixture("config.json")
  let weights = read_fixture("model.safetensors")
  let admitted = @llama_admission.admit(
    config,
    weights,
    content_digest(),
    config_max_bytes=2048,
    config_max_depth=8,
    header_limits=header_limits(),
  )
  let host = @materialize.copy_to_host(
    admitted.bindings(),
    weights,
    limits=materialization_limits(),
  )
  let prepared = @reference.prepare(
    admitted.plan(),
    host,
    limits=execution_limits(),
  )
  for index, case in [first, second] {
    guard case is Object(fields) else {
      fail("prefix referee case must be an object")
    }
    let input_ids = corpus_tokens(fields["input_ids"])
    assert_eq(input_ids, expected_inputs[index])
    assert_eq(corpus_int(fields["greedy_token"]), expected_tokens[index])
    let execution = @reference.execute(
      admitted.plan(),
      prepared,
      input_ids,
      limits=execution_limits(),
    )
    assert_eq(execution.selection().token_id(), expected_tokens[index])
  }
}

///|
async test "corpus provenance is pinned and tokenizer incompatibility fails closed" {
  guard @json.parse(@utf8.decode(read_fixture("corpus.json"))) is Object(root) else {
    fail("reference corpus must be a JSON object")
  }
  guard root["schema"] is String("lunaflux.reference-corpus.v1") &&
    root["revision"] is String("3579d71fd57e04f5a364d824d3a2ec3e913dbb67") &&
    root["tokenizer_cases"] is Array([_, _, _, _, _]) &&
    root["model_cases"] is Array([_, _, _]) else {
    fail("reference corpus identity or cases changed")
  }
  let digest = @tokenizer.TokenizerDigest::parse(
    "0afe36ee1358ce1fa277f4eac935250bb90253ed5c27867eb6ff376ded7d1980",
  )
  let tokenizer_limits : @tokenizer.TokenizerLimits = {
    max_input_bytes: 4096,
    max_output_tokens: 4096,
    max_decoded_bytes: 4096,
    max_vocab_entries: 4096,
    max_merge_rules: 4096,
    max_token_bytes: 128,
    max_special_tokens: 32,
  }
  assert_true(
    tokenizer_probe_fails_closed(
      read_fixture("tokenizer_semantics_probe.json"),
      digest,
      tokenizer_limits,
    ),
  )
}

///|
async test "approved upstream SentencePiece BPE tokenizer matches pinned IDs" {
  @async.with_task_group() <| group => {
    let root_label = @fs.realpath(fixture_directory)
    let root = @approved_fs.ApprovedRoot::open_absolute(root_label)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => root.close())
    }
    let tokenizer_limits : @tokenizer.TokenizerLimits = {
      max_input_bytes: 4096,
      max_output_tokens: 4096,
      max_decoded_bytes: 65536,
      max_vocab_entries: 3000,
      max_merge_rules: 64,
      max_token_bytes: 64,
      max_special_tokens: 3,
    }
    let file_digest = @tokenizer.TokenizerDigest::parse(
      upstream_tokenizer_digest().as_hex(),
    )
    let file_limits = @json_file.TokenizerFileLimits::new(
      max_file_bytes=65536,
      max_json_depth=16,
    )
    let file_admission = @json_file.load(
      root, root_label, "upstream_tokenizer.json", file_digest, file_limits, tokenizer_limits,
    )
    assert_true(
      file_admission.matches_source(
        root_label, "upstream_tokenizer.json", file_digest, file_limits, tokenizer_limits,
      ),
    )
    assert_eq(file_admission.spec().vocab_size(), 3000)
    let bundle = @artifact.load_reference_bundle(
      root,
      @artifact.ArtifactSource::new(
        model_config="config.json",
        tokenizer="upstream_tokenizer.json",
        weights="model.safetensors",
      ),
      @artifact.ExpectedDigests::new(
        model_config=@artifact.ArtifactDigest::parse(
          "9197475bfcc987a4f9361dbc22b33397b101372c137c228b6a6fd7e4adf21622",
        ),
        tokenizer=upstream_tokenizer_digest(),
        weights=@artifact.ArtifactDigest::parse(content_digest().as_hex()),
      ),
      @artifact.ArtifactLimits::new(
        max_model_config_bytes=2048L,
        max_tokenizer_bytes=65536L,
        max_weights_bytes=210712L,
        max_total_bytes=278296L,
        max_model_config_depth=8,
        max_tokenizer_depth=16,
        tokenizer_limits~,
        safetensors_limits=header_limits(),
      ),
    )
    assert_eq(bundle.tokenizer_bytes().length(), 64224)
    assert_eq(bundle.tokenizer().vocab_size(), 3000)
    assert_eq(bundle.tokenizer().merge_count(), 0)
    assert_eq(
      bundle.identity().tokenizer().as_string(),
      upstream_tokenizer_digest().as_hex(),
    )
    guard @json.parse(@utf8.decode(read_fixture("corpus.json")))
      is Object(corpus) &&
      corpus["tokenizer_cases"] is Array(cases) &&
      corpus["model_cases"] is Array([empty_model_case, ..]) else {
      fail("reference tokenizer cases must be present")
    }
    for case in cases {
      guard case is Object(fields) &&
        fields["text"] is String(text) &&
        fields["token_ids"] is Array(expected) else {
        fail("reference tokenizer case must be an object")
      }
      let encoded = bundle
        .tokenizer()
        .encode_text(
          text,
          special_policy=AsOrdinary,
          output_limit=128,
          truncation=RejectOverflow,
        )
      assert_eq(encoded.token_ids, expected.map(corpus_int))
    }
    guard empty_model_case is Object(empty_fields) &&
      empty_fields["sampled_logits"] is Object(empty_logits) else {
      fail("empty-text model case must be present")
    }
    let empty_tokens = bundle
      .tokenizer()
      .encode_text(
        "",
        special_policy=AsOrdinary,
        output_limit=16,
        truncation=RejectOverflow,
      ).token_ids
    assert_eq(empty_tokens, corpus_tokens(empty_fields["input_ids"]))
    let admitted = @llama_admission.complete(
      bundle.model(),
      bundle.safetensors(),
    )
    let host = @materialize.copy_to_host(
      admitted.bindings(),
      bundle.weights_bytes(),
      limits=materialization_limits(),
    )
    let prepared = @reference.prepare(
      admitted.plan(),
      host,
      limits=execution_limits(),
    )
    let execution = @reference.execute(
      admitted.plan(),
      prepared,
      empty_tokens,
      limits=execution_limits(),
    )
    assert_eq(
      execution.selection().token_id(),
      corpus_int(empty_fields["greedy_token"]),
    )
    for token_text, expected_logit in empty_logits {
      assert_logit(
        execution,
        0,
        @strconv.from_str(token_text),
        corpus_float(expected_logit),
      )
    }
    assert_eq(
      @reference.generate(
        admitted.plan(),
        prepared,
        empty_tokens,
        max_new_tokens=4,
        limits=execution_limits(),
      ).generated_tokens(),
      ReadOnlyArray::from_array(
        corpus_tokens(empty_fields["greedy_continuation"]),
      ),
    )
    assert_eq(bundle.tokenizer().decode_text([365], special_policy=Skip), "L")
    assert_eq(
      bundle.tokenizer().decode_text([229, 153, 132, 75], special_policy=Skip),
      "▁H",
    )
  }
}
