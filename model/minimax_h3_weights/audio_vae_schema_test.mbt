///|
fn official_audio_vae_shards(
  metadata : @minimax_h3_spec.MiniMaxH3ModelMetadata,
) -> Array[@minimax_h3_weights.MiniMaxH3PartialShardBinding] {
  let plan = try! @minimax_h3.build(metadata)
  let schema = @minimax_h3_weights.MiniMaxH3AudioVaeSchema::official_v1()
  let shards : Array[@minimax_h3_weights.MiniMaxH3PartialShardBinding] = []
  for index in 0..<schema.tensor_count() {
    guard schema.tensor_name(index) is Some(name) else {
      abort("missing official AudioVae name")
    }
    guard schema.tensor_shape(index) is Some(shape) else {
      abort("missing official AudioVae shape")
    }
    shards.push(
      try! @minimax_h3_weights.bind(
        metadata,
        plan,
        AudioVae,
        try! one_tensor(name, shape~, dtype="F32"),
      ),
    )
  }
  shards
}

///|
test "official AudioVae single-file schema is complete F32 and byte exact" {
  let schema = @minimax_h3_weights.MiniMaxH3AudioVaeSchema::official_v1()
  assert_eq(schema.version(), 1)
  assert_eq(schema.tensor_count(), 1087)
  assert_eq(schema.tensor_bytes(), 605306340L)
  assert_eq(schema.file_bytes(), 605429340L)
  assert_eq(schema.header_bytes(), 122992)
  assert_eq(schema.filename(), "diffusion_pytorch_model.safetensors")
  assert_eq(
    schema.file_sha256(),
    "52c59e67ba8de5477c81bfbced0327aabf500f1bfdeefd5ee754529241cb26cb",
  )
  assert_eq(schema.tensor_name(-1), None)
  assert_eq(schema.tensor_name(1087), None)
  let names : Array[String] = []
  let mut total = 0L
  for index in 0..<schema.tensor_count() {
    guard schema.tensor_name(index) is Some(name) else { abort("missing name") }
    guard schema.tensor_shape(index) is Some(shape) else {
      abort("missing shape " + index.to_string() + ":" + name)
    }
    assert_eq(schema.tensor_dtype(index), Some(Float32))
    assert_false(names.contains(name))
    names.push(name)
    let mut elements = 1L
    for dimension in shape {
      elements *= dimension
    }
    total += elements * 4L
  }
  assert_eq(total, schema.tensor_bytes())
}

///|
test "official AudioVae vocabulary joins exactly once" {
  let metadata = h3_weight_metadata()
  let shards = official_audio_vae_shards(metadata)
  let complete = @minimax_h3_weights.join_complete_component(AudioVae, shards)
  assert_eq(complete.component(), AudioVae)
  assert_eq(complete.tensor_count(), 1087)
  assert_eq(complete.tensor(0).unwrap().role(), AudioDecoder)
  assert_eq(@minimax_h3_weights.AudioVae.required_tensor_count(), 1087)
}

///|
test "AudioVae binding rejects unknown shape dtype missing and duplicate" {
  let metadata = h3_weight_metadata()
  let plan = @minimax_h3.build(metadata)
  for
    tensor in [
      one_tensor(
        "decoder.resblocks.21.convs1.0.weight_v",
        shape=[8L, 8L, 3L],
        dtype="F32",
      ),
      one_tensor("encoder.block.0.weight_v", shape=[64L, 1L, 9L], dtype="F32"),
    ] {
    try @minimax_h3_weights.bind(metadata, plan, AudioVae, tensor) catch {
      UnknownSemanticTensor(index=0) | InvalidSemanticRank(index=0) => ()
      _ => fail("unexpected AudioVae schema rejection")
    } noraise {
      _ => fail("hostile AudioVae tensor admitted")
    }
  }
  let wrong_dtype = try! one_tensor("mean_proj.bias", shape=[32L])
  try @minimax_h3_weights.bind(metadata, plan, AudioVae, wrong_dtype) catch {
    InvalidSemanticDType(index=0) => ()
    _ => fail("unexpected AudioVae dtype rejection")
  } noraise {
    _ => fail("BF16 AudioVae tensor admitted")
  }
  let shards = official_audio_vae_shards(metadata)
  try
    @minimax_h3_weights.join_complete_component(
      AudioVae,
      shards[:shards.length() - 1],
    )
  catch {
    IncompleteComponent(expected=1087, actual=1086) => ()
    _ => fail("unexpected incomplete AudioVae rejection")
  } noraise {
    _ => fail("incomplete AudioVae admitted")
  }
  let duplicate : Array[@minimax_h3_weights.MiniMaxH3PartialShardBinding] = []
  duplicate.push(shards[0])
  duplicate.push(shards[0])
  for index in 1..<(shards.length() - 1) {
    duplicate.push(shards[index])
  }
  try @minimax_h3_weights.join_complete_component(AudioVae, duplicate) catch {
    DuplicateSemanticTensor(_) => ()
    _ => fail("unexpected duplicate AudioVae rejection")
  } noraise {
    _ => fail("duplicate AudioVae tensor admitted")
  }
}
