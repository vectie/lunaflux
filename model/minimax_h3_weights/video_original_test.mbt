///|
test "original VideoVAE admits all 560 exact source tensors without conversion" {
  let schema = @minimax_h3_weights.MiniMaxH3OriginalVideoVaeSchema::official_v1()
  let metadata = h3_weight_metadata()
  let plan = @minimax_h3.build(metadata)
  let names : Array[String] = []
  let shards = []
  let mut bytes = 0L
  for i in 0..<schema.tensor_count() {
    let name = schema.tensor_name(i).unwrap()
    let shape = schema.tensor_shape(i).unwrap()
    assert_false(names.contains(name))
    names.push(name)
    assert_true(schema.matches_tensor(name, Float32, shape))
    bytes += shape.fold(init=4L, (n, d) => n * d)
    shards.push(
      @minimax_h3_weights.bind(
        metadata,
        plan,
        VideoVae,
        one_tensor(name, shape~, dtype="F32"),
      ),
    )
  }
  assert_eq(bytes, 10415484128L)
  assert_eq(
    @minimax_h3_weights.join_complete_component(VideoVae, shards).tensor_count(),
    560,
  )
  assert_eq(schema.tensor_name(-1), None)
  assert_eq(schema.tensor_name(560), None)
  for broken in [shards[1:].to_owned(), shards[:559].to_owned()] {
    try @minimax_h3_weights.join_complete_component(VideoVae, broken) catch {
      IncompleteComponent(expected=560, actual=559) => ()
      _ => fail("unexpected incomplete original layout error")
    } noraise {
      _ => fail("incomplete original layout admitted")
    }
  }
  let mixed = shards[:].to_owned()
  mixed[2] = @minimax_h3_weights.bind(
    metadata,
    plan,
    VideoVae,
    one_tensor("decoder.proj_in.weight", shape=[2048L, 24L], dtype="F32"),
  )
  try @minimax_h3_weights.join_complete_component(VideoVae, mixed) catch {
    IncompleteComponent(expected=560, actual=560) => ()
    _ => fail("unexpected mixed source layout error")
  } noraise {
    _ => fail("mixed layouts admitted")
  }
}

///|
test "original head-interleaved QKV maps exact weight and bias slices" {
  for (suffix, head_bytes) in [("weight", 524288L), ("bias", 256L)] {
    for (projection, segment) in [("q", 0L), ("k", 1L), ("v", 2L)] {
      let copy = @minimax_h3_weights.BlockWeightCopy::new(
        tensor_name="decoder.transformer_blocks.35.attn.to_\{projection}.\{suffix}",
        source_offset=0L,
        destination_offset=1024L,
        byte_count=32L * head_bytes,
      )
      let slices = @minimax_h3_weights.original_video_vae_copies(copy)
      assert_eq(slices.length(), 32)
      for i, slice in slices {
        assert_eq(
          slice.tensor_name,
          "decoder.transformer_blocks.35.attn.to_qkv.\{suffix}",
        )
        assert_eq(
          slice.source_offset,
          (i.to_int64() * 3L + segment) * head_bytes,
        )
        assert_eq(slice.destination_offset, 1024L + i.to_int64() * head_bytes)
        assert_eq(slice.byte_count, head_bytes)
      }
      // Cross a head boundary; Q/K/V gaps in the source must not be copied.
      let partial = @minimax_h3_weights.original_video_vae_copies(
        @minimax_h3_weights.BlockWeightCopy::new(
          tensor_name=copy.tensor_name,
          source_offset=head_bytes - 4L,
          destination_offset=8L,
          byte_count=8L,
        ),
      )
      assert_eq(partial.length(), 2)
      assert_eq(partial[0].source_offset, (segment + 1L) * head_bytes - 4L)
      assert_eq(partial[1].source_offset, (3L + segment) * head_bytes)
      assert_eq(partial[0].byte_count, 4L)
      assert_eq(partial[1].destination_offset, 12L)
    }
  }
}

///|
test "original source aliases are exact and refuse unsliced QKV" {
  for
    (canonical, original) in [
      ("decoder.proj_in.weight", "decoder.x_embedder.weight"),
      (
        "decoder.transformer_blocks.0.ff.net.0.proj.weight", "decoder.transformer_blocks.0.ff.w1.weight",
      ),
      (
        "encoder.down_blocks.1.resnets.0.conv_shortcut.weight", "encoder.down.1.block.0.nin_shortcut.weight",
      ),
      (
        "encoder.down_blocks.3.downsamplers.0.conv.bias", "encoder.down.3.downsample.conv.bias",
      ),
    ] {
    assert_eq(
      @minimax_h3_weights.original_video_vae_source_name(canonical),
      Some(original),
    )
  }
  assert_eq(
    @minimax_h3_weights.original_video_vae_source_name(
      "decoder.transformer_blocks.0.attn.to_q.weight",
    ),
    None,
  )
  assert_eq(
    @minimax_h3_weights.original_video_vae_source_name(
      "encoder.down_blocks.6.resnets.0.conv1.weight",
    ),
    None,
  )
  let schema = @minimax_h3_weights.MiniMaxH3OriginalVideoVaeSchema::official_v1()
  assert_false(
    schema.matches_tensor(
      "decoder.transformer_blocks.0.attn.to_q.weight",
      Float32,
      [2048L, 2048L],
    ),
  )
  assert_false(
    schema.matches_tensor(
      "decoder.transformer_blocks.36.attn.to_qkv.weight",
      Float32,
      [6144L, 2048L],
    ),
  )
  assert_false(
    schema.matches_tensor("decoder.mask_token", BFloat16, [1L, 1L, 2048L]),
  )
}
