///|
fn compatible_tokenizer_digest() -> @artifact.ArtifactDigest raise {
  @artifact.ArtifactDigest::parse(
    "82000898a396651e5d3b149c5dec02998427406c78ccdf14084cbf85c63d2686",
  )
}

///|
fn reference_bundle_limits() -> @artifact.ArtifactLimits raise {
  let tokenizer_limits : @tokenizer.TokenizerLimits = {
    max_input_bytes: 4096,
    max_output_tokens: 4096,
    max_decoded_bytes: 65536,
    max_vocab_entries: 3000,
    max_merge_rules: 64,
    max_token_bytes: 16,
    max_special_tokens: 1,
  }
  @artifact.ArtifactLimits::new(
    max_model_config_bytes=2048L,
    max_tokenizer_bytes=65536L,
    max_weights_bytes=210712L,
    max_total_bytes=278296L,
    max_model_config_depth=8,
    max_tokenizer_depth=16,
    tokenizer_limits~,
    safetensors_limits=header_limits(),
  )
}

///|
async test "compatible text reaches independent reference logits and tokens" {
  @async.with_task_group() <| group => {
    let root = @approved_fs.ApprovedRoot::open_absolute(
      @fs.realpath(fixture_directory),
    )
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => root.close())
    }
    let source = @artifact.ArtifactSource::new(
      model_config="config.json",
      tokenizer="compatible_bytelevel_tokenizer.json",
      weights="model.safetensors",
    )
    let expected = @artifact.ExpectedDigests::new(
      model_config=@artifact.ArtifactDigest::parse(
        "9197475bfcc987a4f9361dbc22b33397b101372c137c228b6a6fd7e4adf21622",
      ),
      tokenizer=compatible_tokenizer_digest(),
      weights=@artifact.ArtifactDigest::parse(content_digest().as_hex()),
    )
    let bundle = @artifact.load_reference_bundle(
      root,
      source,
      expected,
      reference_bundle_limits(),
    )
    assert_eq(bundle.tokenizer_bytes().length(), 48574)
    assert_eq(bundle.tokenizer().vocab_size(), 3000)
    assert_eq(
      bundle.identity().tokenizer().as_string(),
      compatible_tokenizer_digest().as_hex(),
    )
    let encoded = bundle
      .tokenizer()
      .encode_text(
        "Luna*c",
        special_policy=AsOrdinary,
        output_limit=16,
        truncation=RejectOverflow,
      )
    assert_eq(encoded.token_ids, [1, 42, 99])
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
    let corpus = read_fixture("corpus.json")
    assert_eq(
      @crypto.bytes_to_hex_string(@crypto.sha256(corpus)),
      "f4ade303b1b7b3264e86caa15b77cdb2805244e1142b9b74225ed3f36438f1d0",
    )
    guard @json.parse(@utf8.decode(corpus)) is Object(root_fields) &&
      root_fields["model_cases"] is Array([_, _, case]) &&
      case is Object(fields) &&
      fields["sampled_logits"] is Object(sampled_logits) else {
      fail("reference text-pipeline case must be present")
    }
    assert_eq(corpus_tokens(fields["input_ids"]), encoded.token_ids)
    let execution = @reference.execute(
      admitted.plan(),
      prepared,
      encoded.token_ids,
      limits=execution_limits(),
    )
    let greedy_token = corpus_int(fields["greedy_token"])
    assert_eq(greedy_token, 2185)
    assert_eq(execution.selection().token_id(), greedy_token)
    for token_text, expected_logit in sampled_logits {
      let token : Int = @strconv.from_str(token_text)
      assert_logit(
        execution,
        encoded.token_ids.length() - 1,
        token,
        corpus_float(expected_logit),
      )
    }
    let continuation = corpus_tokens(fields["greedy_continuation"])
    assert_eq(continuation, [2185, 688, 2844, 1377])
    assert_eq(
      @reference.generate(
        admitted.plan(),
        prepared,
        encoded.token_ids,
        max_new_tokens=4,
        limits=execution_limits(),
      ).generated_tokens(),
      ReadOnlyArray::from_array(continuation),
    )
  }
}
