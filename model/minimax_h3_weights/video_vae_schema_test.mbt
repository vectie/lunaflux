///|
fn official_video_vae_shards(
  metadata : @minimax_h3_spec.MiniMaxH3ModelMetadata,
) -> Array[@minimax_h3_weights.MiniMaxH3PartialShardBinding] {
  let plan = try! @minimax_h3.build(metadata)
  let schema = @minimax_h3_weights.MiniMaxH3VideoVaeSchema::official_v1()
  let shards : Array[@minimax_h3_weights.MiniMaxH3PartialShardBinding] = []
  for index in 0..<schema.tensor_count() {
    guard schema.tensor_name(index) is Some(name) else {
      abort("missing official video-VAE name")
    }
    guard schema.tensor_shape(index) is Some(shape) else {
      abort("missing official video-VAE shape")
    }
    shards.push(
      try! @minimax_h3_weights.bind(
        metadata,
        plan,
        VideoVae,
        try! one_tensor(name, shape~, dtype="F32"),
      ),
    )
  }
  shards
}

///|
test "official video-VAE schema is complete F32 and byte exact" {
  let schema = @minimax_h3_weights.MiniMaxH3VideoVaeSchema::official_v1()
  assert_eq(schema.version(), 1)
  assert_eq(
    schema.source_revision(),
    "42ed227ee7df40d41602854ae760620d6eb651fe",
  )
  assert_eq(
    schema.index_sha256(),
    "15f6d44553c3c616b0dc999920aa784f92ecee7e4201f1f99ac405cfbf3061ca",
  )
  assert_eq(schema.tensor_count(), 703)
  assert_eq(schema.tensor_bytes(), 10415475936L)
  assert_eq(schema.shard_count(), 3)
  assert_eq(
    schema.shard_filename(2),
    Some("diffusion_pytorch_model-00003-of-00003.safetensors"),
  )
  assert_eq(schema.shard_tensor_count(0), Some(303))
  assert_eq(schema.shard_tensor_count(1), Some(372))
  assert_eq(schema.shard_tensor_count(2), Some(28))
  assert_eq(schema.shard_file_bytes(0), Some(5061033024L))
  assert_eq(schema.shard_file_bytes(1), Some(4955986528L))
  assert_eq(schema.shard_file_bytes(2), Some(398539336L))
  assert_eq(schema.shard_payload_bytes(0), Some(5060997120L))
  assert_eq(schema.shard_payload_bytes(1), Some(4955942592L))
  assert_eq(schema.shard_payload_bytes(2), Some(398536224L))
  assert_eq(schema.tensor_name(-1), None)
  assert_eq(schema.tensor_name(703), None)
  assert_eq(schema.shard_file_sha256(-1), None)
  assert_eq(schema.shard_filename(3), None)
  assert_eq(schema.shard_tensor_count(3), None)
  let names : Array[String] = []
  let mut total = 0L
  for index in 0..<schema.tensor_count() {
    guard schema.tensor_name(index) is Some(name) else {
      abort("missing tensor name")
    }
    guard schema.tensor_shape(index) is Some(shape) else {
      abort("missing tensor shape")
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
test "official video-VAE vocabulary joins exactly once" {
  let metadata = h3_weight_metadata()
  let shards = official_video_vae_shards(metadata)
  let complete = @minimax_h3_weights.join_complete_component(VideoVae, shards)
  assert_eq(complete.component(), VideoVae)
  assert_eq(complete.tensor_count(), 703)
  assert_eq(@minimax_h3_weights.VideoVae.required_tensor_count(), 703)
  assert_eq(complete.tensor(0).unwrap().role(), VideoDecoder)
  assert_eq(complete.tensor(702).unwrap().role(), VideoLatentProjection)
}

///|
test "video-VAE binding rejects unknown shape dtype missing and duplicate" {
  let metadata = h3_weight_metadata()
  let plan = @minimax_h3.build(metadata)
  for
    tensor in [
      one_tensor(
        "decoder.transformer_blocks.36.attn.to_q.weight",
        shape=[2048L, 2048L],
        dtype="F32",
      ),
      one_tensor("decoder.register_tokens", shape=[1L, 5L, 2048L], dtype="F32"),
    ] {
    try @minimax_h3_weights.bind(metadata, plan, VideoVae, tensor) catch {
      UnknownSemanticTensor(index=0) | InvalidSemanticRank(index=0) => ()
      _ => fail("unexpected video-VAE schema rejection")
    } noraise {
      _ => fail("hostile video-VAE tensor admitted")
    }
  }
  let wrong_dtype = try! one_tensor("decoder.norm_out.weight", shape=[2048L])
  try @minimax_h3_weights.bind(metadata, plan, VideoVae, wrong_dtype) catch {
    InvalidSemanticDType(index=0) => ()
    _ => fail("unexpected video-VAE dtype rejection")
  } noraise {
    _ => fail("BF16 video-VAE tensor admitted")
  }
  let shards = official_video_vae_shards(metadata)
  try
    @minimax_h3_weights.join_complete_component(
      VideoVae,
      shards[:shards.length() - 1],
    )
  catch {
    IncompleteComponent(expected=703, actual=702) => ()
    _ => fail("unexpected incomplete video-VAE rejection")
  } noraise {
    _ => fail("incomplete video-VAE admitted")
  }
  let duplicate : Array[@minimax_h3_weights.MiniMaxH3PartialShardBinding] = []
  duplicate.push(shards[0])
  duplicate.push(shards[0])
  for index in 1..<(shards.length() - 1) {
    duplicate.push(shards[index])
  }
  try @minimax_h3_weights.join_complete_component(VideoVae, duplicate) catch {
    DuplicateSemanticTensor(_) => ()
    _ => fail("unexpected duplicate video-VAE rejection")
  } noraise {
    _ => fail("duplicate video-VAE tensor admitted")
  }
}
