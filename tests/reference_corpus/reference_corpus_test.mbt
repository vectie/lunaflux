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
    UnsupportedSemantics(component=Normalizer, feature=NormalizationBehavior) =>
      true
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
