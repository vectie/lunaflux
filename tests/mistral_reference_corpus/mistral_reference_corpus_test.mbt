///|
let mistral_fixture_directory = "tests/mistral_reference_corpus"

///|
let approved_reference_directory = "tests/reference_corpus"

///|
async fn read_mistral_fixture(name : String) -> Bytes {
  @fs.read_file("\{mistral_fixture_directory}/\{name}").binary()
}

///|
async fn read_approved_reference(name : String) -> Bytes {
  @fs.read_file("\{approved_reference_directory}/\{name}").binary()
}

///|
fn sha256_hex(bytes : BytesView) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn mistral_content_digest() -> @spec.ContentDigest raise {
  @spec.ContentDigest::from_sha256(
    "852db1b39acb2336abc997440c6f6d6e4ab640f91e5e2aa9e2488d5794159d30",
  )
}

///|
fn mistral_header_limits() -> @safetensors.HeaderLimits raise {
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
fn mistral_materialization_limits() -> @materialize.MaterializationLimits raise {
  @materialize.MaterializationLimits::new(
    max_header_bytes=4096,
    max_tensor_count=21,
    max_payload_bytes=208544L,
    max_host_bytes=208544L,
  )
}

///|
fn mistral_execution_limits() -> @reference.ExecutionLimits raise {
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
fn corpus_integer(value : Json) -> Int raise {
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
  values.map(corpus_integer)
}

///|
fn assert_mistral_logit(
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
async test "Mistral semantic corpus authenticates every reused artifact" {
  let binding_bytes = read_mistral_fixture("binding.json")
  let config = read_mistral_fixture("mistral_config.json")
  let weights = read_approved_reference("model.safetensors")
  let corpus = read_approved_reference("corpus.json")
  guard @json.parse(@utf8.decode(binding_bytes)) is Object(binding) &&
    binding["schema"] is String("lunaflux.mistral.semantic-reference-corpus.v1") &&
    binding["source_model_family"] is String("LlamaForCausalLM") &&
    binding["admitted_model_family"] is String("MistralForCausalLM") &&
    binding["semantics"]
    is String("exact-dense-plan-equivalence-at-or-below-sliding-window") &&
    binding["fixture_only"] is True &&
    binding["production_mistral_artifact"] is False &&
    binding["physical_execution_evidence"] is False &&
    binding["mistral_config_sha256"] is String(config_digest) &&
    binding["model_content_sha256"] is String(weights_digest) &&
    binding["reference_corpus_sha256"] is String(corpus_digest) &&
    binding["sliding_window"] is Number(256.0, ..) &&
    binding["maximum_exercised_tokens"] is Number(7.0, ..) else {
    fail(
      "Mistral semantic corpus binding is incomplete or overclaims authority",
    )
  }
  assert_eq(sha256_hex(config), config_digest)
  assert_eq(sha256_hex(weights), weights_digest)
  assert_eq(sha256_hex(corpus), corpus_digest)
  assert_not_eq(sha256_hex(weights[1:]), weights_digest)
  let hostile_config = @utf8.encode(@utf8.decode(config) + " ")
  assert_not_eq(sha256_hex(hostile_config), config_digest)
  let hostile_corpus = @utf8.encode(@utf8.decode(corpus) + " ")
  assert_not_eq(sha256_hex(hostile_corpus), corpus_digest)
}

///|
async test "Mistral exact dense semantics pass the independent BF16 corpus" {
  let mistral_config = read_mistral_fixture("mistral_config.json")
  let llama_config = read_approved_reference("config.json")
  let weights = read_approved_reference("model.safetensors")
  let corpus = read_approved_reference("corpus.json")
  let content = mistral_content_digest()
  assert_eq(sha256_hex(weights), content.as_hex())
  let llama = @llama_admission.admit(
    llama_config,
    weights,
    content,
    config_max_bytes=2048,
    config_max_depth=8,
    header_limits=mistral_header_limits(),
  )
  let mistral_metadata = @config_reader.parse_mistral_bytes(
    mistral_config,
    content,
    max_bytes=2048,
    max_depth=8,
  )
  let mistral_plan = @mistral.build(mistral_metadata)
  assert_eq(mistral_metadata.spec().sliding_window(), Some(256))
  assert_eq(mistral_plan, llama.plan())
  let replay_config = @utf8.encode(
    @utf8.decode(mistral_config).replace(
      old="\"sliding_window\": 256",
      new="\"sliding_window\": 128",
    ),
  )
  let replay_metadata = @config_reader.parse_mistral_bytes(
    replay_config,
    content,
    max_bytes=2048,
    max_depth=8,
  )
  assert_not_eq(@mistral.build(replay_metadata), llama.plan())
  let mistral_bindings = @mistral_weights.bind(
    mistral_metadata,
    mistral_plan,
    llama.safetensors(),
  )
  assert_eq(mistral_bindings.length(), llama.bindings().length())
  assert_eq(
    mistral_bindings.materialized_bytes(),
    llama.bindings().materialized_bytes(),
  )
  for ordinal in 0..<mistral_bindings.length() {
    let reference = @plan.TensorRef::new(ordinal)
    assert_eq(
      mistral_bindings.binding(reference).unwrap().tensor(),
      llama.bindings().binding(reference).unwrap().tensor(),
    )
  }
  let host = @materialize.copy_to_host(
    llama.bindings(),
    weights,
    limits=mistral_materialization_limits(),
  )
  let prepared = @reference.prepare(
    mistral_plan,
    host,
    limits=mistral_execution_limits(),
  )
  guard @json.parse(@utf8.decode(corpus)) is Object(root) &&
    root["model_cases"] is Array(cases) else {
    fail("independent reference model cases must be an array")
  }
  assert_eq(cases.length(), 3)
  for case in cases {
    guard case is Object(fields) &&
      fields["sampled_logits"] is Object(sampled_logits) else {
      fail("independent reference model case must be complete")
    }
    let input_ids = corpus_tokens(fields["input_ids"])
    let execution = @reference.execute(
      mistral_plan,
      prepared,
      input_ids,
      limits=mistral_execution_limits(),
    )
    assert_eq(
      execution.selection().token_id(),
      corpus_integer(fields["greedy_token"]),
    )
    for token_text, expected in sampled_logits {
      let token : Int = @strconv.from_str(token_text)
      assert_mistral_logit(
        execution,
        input_ids.length() - 1,
        token,
        corpus_float(expected),
      )
    }
    assert_eq(
      @reference.generate(
        mistral_plan,
        prepared,
        input_ids,
        max_new_tokens=4,
        limits=mistral_execution_limits(),
      ).generated_tokens(),
      ReadOnlyArray::from_array(corpus_tokens(fields["greedy_continuation"])),
    )
  }
}
